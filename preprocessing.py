import numpy as np
from skimage import transform as tm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

"""
this class preprocesses image data of flowers for classification tasks, including loading images,
resizing them, assigning labels, and converting labels to categorical form
"""

class Preprocessing:
    def __init__(self):
        pass
    
    def normalize_image(self, image):
        norm1 = image / 255
        norm2 = (image - np.min(image)) / (np.max(image) - np.min(image))
        norm3 = (image - np.percentile(image, 5)) / (np.percentile(image, 95) - np.percentile(image, 5))
        return norm1, norm2, norm3
    
    def apply_transformations(self, image):
        # Define transformations
        matrix_to_top_left = tm.SimilarityTransform(translation=[-image.shape[0] / 2, -image.shape[1] / 2])
        matrix_to_center = tm.SimilarityTransform(translation=[image.shape[0] / 2, image.shape[1] / 2])
        rot_transforms = tm.AffineTransform(rotation=np.deg2rad(45))
        scale_transforms_out = tm.AffineTransform(scale=(2, 2))
        scale_transforms_in = tm.AffineTransform(scale=(0.5, 0.5))
        shear_transforms = tm.AffineTransform(shear=np.deg2rad(45))
        
        # Apply transformations
        rot_image = tm.warp(image, matrix_to_top_left + rot_transforms + matrix_to_center)
        scale_image_zoom_out = tm.warp(image, matrix_to_top_left + scale_transforms_out + matrix_to_center)
        scale_image_zoom_in = tm.warp(image, matrix_to_top_left + scale_transforms_in + matrix_to_center)
        shear_image = tm.warp(image, matrix_to_top_left + shear_transforms + matrix_to_center)
        return rot_image, scale_image_zoom_out, scale_image_zoom_in, shear_image
    
    def create_image_generators(self, train_df, test_df):
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=30,
            height_shift_range=0.3,
            width_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            validation_split=0.2
        )

        test_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_images = train_gen.flow_from_dataframe(
            dataframe=train_df,
            x_col='FilePath',
            y_col='Labels',
            target_size=(224, 224),
            batch_size=32,
            shuffle=True,
            seed=42,
            color_mode='rgb',
            class_mode='categorical',
            subset='training'
        )

        validation_images = train_gen.flow_from_dataframe(
            dataframe=train_df,
            x_col='FilePath',
            y_col='Labels',
            target_size=(224, 224),
            batch_size=32,
            shuffle=False,
            color_mode='rgb',
            class_mode='categorical',
            subset='validation'
        )

        test_images = test_gen.flow_from_dataframe(
            dataframe=test_df,
            x_col='FilePath',
            y_col='Labels',
            target_size=(224, 224),
            batch_size=32,
            shuffle=False,
            color_mode='rgb',
            class_mode='categorical'
        )

        return train_images, validation_images, test_images
