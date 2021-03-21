import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import Callback
from functools import reduce
import pyvips as Vips
import random

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

def vips_to_np(im):
    return np.frombuffer(im.write_to_memory(), dtype=format_to_dtype[im.format]).reshape(im.height,im.width,im.bands)

class Previewer(Callback):
    def __init__(self, generator, number_of_images=2,save_path=None):
        super()
        self.generator = generator
        self.number_of_images = number_of_images
        self.save_path = save_path
        
    def on_epoch_end(self, epoch, logs={}):
        fig = preview_model(self.model, self.generator, self.number_of_images)
        if self.save_path:
            fig.savefig(self.save_path.format(epoch))
        
        
def preview_model(model, generator, images, weights_path=None, labels=False):
    if weights_path != None:
        model.load_weights(weights_path)
    
    n_images = len(images) if type(images) == list else images

    rs=list(range(generator.images[0].total_frames))
    fig, axarr = plt.subplots(n_images,3,figsize=(20,6*n_images))
    n=0
    
    test_ids=images if type(images) == list else []
    test_images=[generator.images[0].get_single(i) for i in test_ids]
    
    while len(test_images) < n_images:        
        i=random.choice(rs)
        
        orig=generator.images[0].get_single(i)
        
        if orig.mean() > 230:
            continue        
        test_ids += [i] 
        test_images += [orig]
        n+=1
    
    predictions=model.predict(np.asarray(test_images))
    
    for n,i in enumerate(test_ids):
        predicted=predictions[n].reshape((generator.output_shape[0],generator.output_shape[1],generator.classes))
        axarr[n][0].imshow(np.argmax(predicted,axis=2),vmax=4,vmin=0,cmap='hot')
        axarr[n][1].imshow(test_images[n])
        axarr[n][2].imshow(np.argmax(generator.images[0].mask.get(i),axis=2),vmax=4,vmin=0,cmap='hot')
        if labels:
            axarr[n][0].set_title(str(i))
            axarr[n][1].set_title(str(i))
            axarr[n][2].set_title(str(i))
    plt.show()
    return fig
    
def to_categorical(im, n_features=0):
    if (n_features == 0):
        hist = im.hist_find()
        if hist(0,0)+hist(hist.width-1,0) == im.width*im.height:
            n_features = 2
            return im, n_features 
        else:
            for n in range(hist.width):
                c = hist(n,0)
                if c != [0.0]:
                    n_features = n+1
    categorical = im == 0
    for klass in range(1,n_features):
        categorical = categorical.bandjoin(im == int(klass))
    return categorical, n_features

def flatten(image,n):
    return ((image/255.0)*n).cast('uchar')

def from_categorical(im):
    template = im.bandsplit()[1:im.bands]
    bands = im.bands-1
    for band in range(bands):
        template[band] = reduce(lambda a,b: a&~b,[template[i] for i in range(bands) if i!=band],template[band])
    result = Vips.Image.black(im.width,im.height)
    for band in range(0, bands):
        result |= flatten(template[band],band+1)
    return result

def resize_indexed(im,width,height):
    cat_image=to_categorical(im)[0]
    cat_image_resized = resize_categorical(cat_image,width,height)
    return from_categorical(cat_image_resized)

def resize_categorical(cat_image,width,height):
    reducer = lambda a,b: a.bandjoin(b.resize(width/cat_image.width,vscale=height/cat_image.height)>=63)
    return reduce(reducer,cat_image.bandsplit())

def show_vips_im(im):
    plt.figure(figsize=(40,40))
    plt.imshow(vips_to_np(im))
    
def show_np_im(im):
    plt.figure(figsize=(40,40))
    plt.imshow(im)
    