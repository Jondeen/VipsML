import numpy as np
import pyvips as Vips
from math import ceil 
import random
from functools import reduce
from keras.utils import Sequence
import threading
from tqdm import tqdm
from .tools import to_categorical, format_to_dtype, vips_to_np

USE_REGIONS = True

class PreCropModulationGenerator():
    """ Pipeline: Crop -> Rotation -> Flip -> To Memory """
    def __init__(self,im):
        self.im = im        
        rotations = [lambda i: i,lambda i: i.rot90(),lambda i: i.rot180(),lambda i: i.rot270()]
        self.rots_and_flips = rotations + \
                        [lambda i: rot(i).fliphor() for rot in rotations]
        self.dtype=format_to_dtype[im.format]
        w=im.width
        self.w=w
        h=im.height
        self.h=h
        self.bands=im.bands
        
    def fetch(self,x,y,s_x,s_y,n):
        fetched = self.rots_and_flips[n](self.im.crop(x,y,s_x,s_y))
        return vips_to_np(fetched)


class PostCropModulationGenerator():
    """ Pipeline: Rotation -> Flip -> Crop -> To Memory """
    def __init__(self,im):
        self.im = im
        
        self.dtype=format_to_dtype[im.format]
        
        rotations = [im,im.rot90(),im.rot180(),im.rot270()]
        self.rots_and_flips = rotations + \
                        [rot.fliphor() for rot in rotations]
        w=im.width
        self.w=w
        h=im.height
        self.h=h
        self.bands=im.bands
        self.position_correction = [lambda x,y,s: (x,y),
                              lambda x,y,s: (h-y-s,x),
                              lambda x,y,s: (w-x-s,h-y-s),
                              lambda x,y,s: (y,w-x-s),
                              lambda x,y,s: (w-x-s,y),
                              lambda x,y,s: (y,x),
                              lambda x,y,s: (x,h-y-s),
                              lambda x,y,s: (h-y-s,w-x-s)]
        
    def fetch(self,x,y,s,n):
        fetched = self.rots_and_flips[n].crop(*self.position_correction[n](x,y,s),s,s)
        return vips_to_np(fetched)

class RegionModulationGenerator():
    """ Rotate -> Flip -> Region() -> Fetch """
    def __init__(self,im):
        self.im = im
        
        self.dtype=format_to_dtype[im.format]
        
        self.rotations = [im,im.rot90(),im.rot180(),im.rot270()]
        self.rots_and_flips = [Vips.Region.new(rot) for rot in self.rotations] + \
                        [Vips.Region.new(rot.fliphor()) for rot in self.rotations]
        
        w=im.width
        self.w=w
        h=im.height
        self.h=h
        self.bands=im.bands
        self.position_correction = [lambda x,y,s_x,s_y: (x,y),
                              lambda x,y,s_x,s_y: (h-y-s_y,x),
                              lambda x,y,s_x,s_y: (w-x-s_x,h-y-s_y),
                              lambda x,y,s_x,s_y: (y,w-x-s_x),
                              lambda x,y,s_x,s_y: (w-x-s_x,y),
                              lambda x,y,s_x,s_y: (y,x),
                              lambda x,y,s_x,s_y: (x,h-y-s_y),
                              lambda x,y,s_x,s_y: (h-y-s_y,w-x-s_x)]
        
    def fetch(self,x,y,s_x,s_y,n):
        """ Returns a numpy region.
        Param x,y: x,y coordinate 
        Param s: size of frame (w==h==s)
        Param n: id of flip/rotate combination (0-7)"""
        x_, y_ = self.position_correction[n](int(x),int(y),s_x,s_y)
        fetched = self.rots_and_flips[n].fetch(x_, y_, s_x, s_y)
        return np.frombuffer(fetched, dtype=self.dtype).reshape(s_x,s_y,self.bands)    

class VipsScanImage():
    def __init__(self,vips_image,frame_size,
                 padding=0,outer_pad='white'):
        self.lock = threading.Lock()
        self.orig = vips_image
        h,w,d = vips_image.height, vips_image.width, vips_image.bands
        self.origShape= (h,w,d)
        
        if type(frame_size) != tuple:
            frame_size = (frame_size, frame_size)        
        if type(padding) != tuple:
            padding = (padding, padding)
            
        self.x_frame_size, self.y_frame_size = frame_size
        self.x_padding, self.y_padding = padding
        
        self.x_frames = w / self.x_frame_size
        self.y_frames = h / self.y_frame_size
        
        # Prone to rounding errors :(
        if self.x_frames % 1 < 0.0000000000001:
            self.x_frames = round(self.x_frames)
        else:
            self.x_frames = ceil(self.x_frames)
        
        if self.y_frames % 1 < 0.0000000000001:
            self.y_frames = round(self.y_frames)
        else:
            self.y_frames = ceil(self.y_frames)
        
        self.total_frames = self.x_frames * self.y_frames

        self.padded_x_frame_size = ceil(self.x_frame_size + (2*self.x_padding))
        self.padded_y_frame_size = ceil(self.y_frame_size + (2*self.y_padding))

        left_pad = self.x_padding
        top_pad = self.y_padding
        new_w = (self.x_frames*self.x_frame_size) + \
                        (2*self.x_padding)
        new_h = ((self.y_frames)*self.y_frame_size) + \
                        (2*self.y_padding)
        self.padded=vips_image.embed(left_pad,
                                     top_pad,
                                     new_w,
                                     new_h,
                                     extend=outer_pad)
        
        self.generate_modulation=RegionModulationGenerator(self.padded).fetch
    
    def __getitem__(self,index):
        return self.get(index)
    
    def __len__(self):
        return self.total_frames

    def get(self,index,mod_id=0):
        x_n = index % self.x_frames
        y_n = index // self.x_frames            
        x_px = x_n * self.x_frame_size
        y_px = y_n * self.y_frame_size
        s_x = self.x_frame_size+(2*self.x_padding)
        s_y = self.y_frame_size+(2*self.y_padding)
        try:
            with self.lock:
                frame = self.generate_modulation(int(x_px), int(y_px), int(s_x), int(s_y), mod_id)
        except:
            error_msg = "Requested area out of bounds: {}, {} (+{})"
            raise Exception(error_msg.format(x_px, y_px, s_x, s_y))
        return frame


class VipsML():
    def __init__(self,image,mask=None,frame_size=None,
                 padding=None,outer_pad=['white','black'],meta_ratio=1): 
        
        if type(image) == str:
            image = Vips.Image.new_from_file(image)
            
        if mask!= None and type(mask) == str:
            mask = Vips.Image.new_from_file(mask)
        
        self.frame_size = frame_size
        self.padding = padding
        self.outer_pad = outer_pad
        self.meta_ratio = meta_ratio
        
        self.image = VipsScanImage(image, frame_size, 
                                           padding, outer_pad[0])
        self.mask = self.prep_mask(mask,  frame_size, 
                                           padding, outer_pad[1])
        if meta_ratio != 1:
            assert type(meta_ratio) == int and meta_ratio % 2 != 0
            
            meta = image.colourspace(Vips.Interpretation.B_W).resize(1.0/meta_ratio)
            meta = meta.copy_memory()
            
            padded_frame = frame_size + (2*padding)
            meta_frame = ( (meta.width / self.image.x_frames), (meta.height / self.image.y_frames) ) 
            meta_padding = (padded_frame - meta_frame[0]) / 2.0, (padded_frame - meta_frame[1]) / 2.0
            self.meta = VipsScanImage(meta, meta_frame, meta_padding,outer_pad[1])
            
            assert self.meta.x_frames == self.image.x_frames
            assert self.meta.y_frames == self.image.y_frames
        
        self.total_frames = len(self.image)
        
        self.x_frames = self.image.x_frames        
        self.y_frames = self.image.y_frames  
        
        self.indices = list(range(self.total_frames))        
        
        s = frame_size+(2*padding)
        if self.meta_ratio == 1:
            self.shape = (s, s, self.image.orig.bands)
        else:
            self.shape = (s, s, self.image.orig.bands + 1)
            
        self.input_shape = self.shape
        
    def prep_mask(self,mask,frame_size,padding,outer_pad):
        if mask == None:
            return None
        else:
            if mask.bands == 1:
                cat_mask, self.classes = to_categorical(mask)
            else:
                self.classes = mask.bands
                cat_mask = mask
        self.output_shape = (frame_size + (2*padding), frame_size + (2*padding), self.classes)
        return VipsScanImage(cat_mask,frame_size,padding,outer_pad)
        
    def get_training_pair(self,index):
        mod_n = random.choice(range(8))      
        image = self.image.get(index,mod_n)
        mask = self.mask.get(index,mod_n)
        if self.meta_ratio != 1:
            meta = self.meta.get(index,mod_n)[:,:,0]
            image = np.dstack((image,meta))
        return image, mask 
        
    def get_single(self,index):
        image = self.image[index]
        if self.meta_ratio != 1:
            meta = self.meta.get(index)[:,:,0]
            image = np.dstack((image,meta))
        return image
    
    def predict_model(self, model, batch_size=25):
        start = 0
        stitches_x = 0
        stitches_y = 0
        line = None
        rows = None
        for start in tqdm(range(0,self.total_frames,batch_size)):
            end = min(start + batch_size, self.total_frames)
            batch = [self.get_single(i) for i in range(start, end)]
            predictions = model.predict_on_batch(np.asarray(batch))            
            grouped = self.adapt_data(predictions)
            for im in grouped:
                if stitches_x == 0:
                    line = im
                else:
                    line = np.concatenate((line,im), axis=-1)
                stitches_x += 1
                if stitches_x == self.x_frames:
                    if stitches_y == 0:
                        rows = line
                    else:
                        rows = np.concatenate((rows,line), axis=0)
                    line = None
                    stitches_x = 0
                    stitches_y += 1
        return rows
    
    def adapt_data(self,data):         
        return np.argmax(data, axis=-1)
        

class VipsSegmentationML(VipsML):
    def __init__(self,image,mask=None,frame_size=256,
                 padding=0,outer_pad=['white','black'], meta_ratio=1): 
        super().__init__(image,mask,frame_size,padding,outer_pad,meta_ratio)
    
class VipsClassificationML(VipsML):
    def __init__(self, image, mask=None, frame_size=3, 
                 padding=30,outer_pad=['white','black'], meta_ratio=1):        
        super().__init__(image,mask,frame_size,padding,outer_pad,meta_ratio)
        
    def prep_mask(self,mask,frame_size,padding,outer_pad):
        if (mask != None and frame_size != 1):   
            # We cannot deal with resizing a categorical mask
            assert mask.bands == 1
            # Make sure the mask is resized fit snugly with original img
            xscale=float(self.image.x_frames)/mask.width
            vscale=float(self.image.y_frames)/mask.height
            mask = mask.resize(xscale,vscale=vscale)
        return super().prep_mask(mask,1,0,outer_pad)
    
    def get_training_pair(self,index):
        image, mask = super().get_training_pair(index)
        return image, mask.reshape(self.classes)
    
    def adapt_data(self,data):
        return np.argmax(data, axis=-1)[:,np.newaxis,np.newaxis]

        
class VipsGroupML(Sequence):
    def __init__(self, images, batch_size=15):
        self.images = images
        self.batch_size = batch_size
        
        self.indices = reduce(lambda acc, val: acc+[(val,i) 
                          for i in range(self.images[val].total_frames)],
                              range(len(self.images)),[])   
        
        # Peek ahead:
        self.input_shape = self.images[0].input_shape
        self.classes = max([im.classes for im in self.images])
        self.output_shape = (self.images[0].output_shape[0],self.images[0].output_shape[1],self.classes)
        
        self.total_frames = len(self.indices)
        random.shuffle(self.indices)
        
    def split_generators(self,validation_ratio):        
        class SplitSet(Sequence):
            def __init__(self,image,start,end):
                self.image = image
                self.start = start
                self.end = end
            def __len__(self):
                return self.end-self.start
            def __getitem__(self,index):
                return self.image[self.start+index]
            def on_batch_end(self):
                random.shuffle(self.image.indices)
                
        split_point = int((1.0-validation_ratio)*len(self))
        return SplitSet(self,0,split_point), SplitSet(self,split_point,len(self))
    
    def __getitem__(self,index):
        start = index*self.batch_size
        end = min((index+1)*(self.batch_size),self.total_frames)        
        images = []
        masks = []
        for i in self.indices[start:end]:
            image, mask = self.get_training_pair(i)
            images += [image]
            #TODO: Check if we can reshape output shape from NN instead:
            masks += [mask]
        return np.asarray(images), np.asarray(masks)
        
    def __len__(self):
        return int(ceil(self.total_frames/self.batch_size))
    
    def get_training_pair(self,index):
        mod_n = random.choice(range(8))
        image_n, pair_n = index  
        pair = self.images[image_n].get_training_pair(pair_n)
        return pair
