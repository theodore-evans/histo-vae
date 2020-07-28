import random
import numpy as np
from numpy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from PIL import Image 
import os

class Augment:

    def __init__(self, folder_path):
        
        if not os.path.isdir(folder_path) : raise Exception(f"{folder_path} is not a valid directory")
        self.folder_path = folder_path
        self.export_path = os.path.join(self.folder_path, "augment/")
        print("Exporting transformed images to " + self.export_path)
        

    def TransformImage(self, image_filename, image_to_transform: ndarray, transformations = ['rotate, flip, noise']):

        if not os.path.isfile(os.path.join(self.folder_path,image_filename)) : raise Exception(f"{image_filename} is not a valid file")
        
        if not os.path.exists(self.export_path):
            os.mkdir(self.export_path)
        
        available_transformations = {
            'rotate' : self.RandomRotation,
            'flip' : self.RandomFlip,
            'noise' : self.RandomNoise
        }

        transformations = dict((key, available_transformations[key]) for key in transformations)

        num_transformations_to_apply = random.randint(0, len(transformations))
        num_transformations = 0

        transformed_image = None
        
        while num_transformations < num_transformations_to_apply:
            key = random.choice(list(transformations))
            transformed_image = transformations.pop(key)(image_to_transform)
    
            new_filename = f"{image_filename.rsplit('.', 1)[0]}_augment_{num_transformations}.png"
            image_path = os.path.join(self.export_path, new_filename)

            image = Image.fromarray((transformed_image * 255).astype(np.uint8))
            image.save(image_path, 'png')

            # print(f"Saving transformed image {new_filename}.png with transform {key}")
            num_transformations += 1

    def TransformAllImages(self, num_files_to_generate):

        images_to_transform = [ f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        num_generated_files = 0

        while num_generated_files <= num_files_to_generate:
            image_name = random.choice(images_to_transform)
            image_path = os.path.join(self.folder_path, image_name)

            image_to_transform = np.asarray(Image.open(image_path))

            self.TransformImage(image_name, image_to_transform)

            images_to_transform.remove(image_name) # don't transform the same image twice
            num_generated_files += 1


    def RandomRotation(self, image_array : ndarray):
        # rotates the image by 90, 180 or 270 degrees at random
        random_degree = 90 * random.randint(0, 3)
        return sk.transform.rotate(image_array, random_degree)
        
    def RandomNoise(self, image_array: ndarray):
        return sk.util.random_noise(image_array)

    def RandomFlip(self, image_array: ndarray):
        # flips image in either horizontal or vertical axis
        flip_axis = random.randint(0,1) 
        return np.flip(image_array, flip_axis)