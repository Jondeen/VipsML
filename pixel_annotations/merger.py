#!/usr/bin/env python

import sys, pyvips

def loadimages(n):
    a=pyvips.Image.new_from_file("0"+str(n)+"-submucosa.tif")
    b=pyvips.Image.new_from_file("0"+str(n)+"-tissue.tif")
    c=pyvips.Image.new_from_file("0"+str(n)+"-vacuolised.tif")
    d=pyvips.Image.new_from_file("0"+str(n)+"-goblet.tif")
    return [a,b,c,d]

def flatten(image,n):
    return ((image/255)*n).cast('uchar')

def merge(images):
    villi=images[1]^images[0]
    vacuolised=villi&images[2]
    goblets=villi&images[3]
    vacuolised&=~goblets
    villi&=~(vacuolised|goblets)
    a=flatten(images[0],1)
    b=flatten(villi,2)
    c=flatten(vacuolised,3)
    d=flatten(goblets,4)
    return a|b|c|d

merge(loadimages(sys.argv[1])).write_to_file("0"+sys.argv[1]+"-mask.tif")

