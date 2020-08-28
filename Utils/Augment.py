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
    def TransformImage(self, image_to_transform: ndarray, transformations = ['rotate, flip, noise']):
        
        available_transformations = {
            'rotate' : self.RandomRotation,
            'flip' : self.RandomFlip,
            'noise' : self.RandomNoise
        }

        transformations = dict((key, available_transformations[key]) for key in transformations)

        num_transformations_to_apply = random.randint(0, len(transformations))
        num_transformations = 0

        transformed_images = []
        
        while num_transformations < num_transformations_to_apply:
            key = random.choice(list(transformations))
            transformed_image = transformations.pop(key)(image_to_transform)
            transformed_images.append(transformed_image)
            num_transformations += 1

        return np.array(transformed_images)

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