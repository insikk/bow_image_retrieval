# Image related utils

from PIL import Image # Use PIL
import os
import numpy as np


def load_image(image_path):
    """
    return numpy array image with size of (h, w, 3) RGB
    """
    image = Image.open(image_path)
    if image.mode == "P": # PNG palette mode
        image = image.convert('RGBA')

    image = image.convert('RGB')
    
    return np.array(image)



def resize_image(path_list, max_longer_size=1200, prefix="resized_", output_dir=None):
    # print("resized image output dir:", output_dir)
    for image_path in path_list:
        image = Image.open(image_path)
        if image.mode == "P": # PNG palette mode
            image = image.convert('RGBA')
            
        image = image.convert('RGB')
            
        original_filename = os.path.basename(image_path)
        filename, ext = os.path.splitext(original_filename)
        
        original_size = max(image.size[0], image.size[1])    
        
        if (image.size[0] > image.size[1]):
            resized_width = max_longer_size
            resized_height = int(round((max_longer_size/float(image.size[0]))*image.size[1])) 
        else:
            resized_height = max_longer_size
            resized_width = int(round((max_longer_size/float(image.size[1]))*image.size[0]))

        image = image.resize((resized_width, resized_height), Image.ANTIALIAS)
        
        
        resized_filename = prefix + filename + ".jpg" # Always save as jpg
        if output_dir is not None:
            resized_filename = os.path.join(output_dir, resized_filename)
        image.save(resized_filename, 'JPEG') # Always save as jpg