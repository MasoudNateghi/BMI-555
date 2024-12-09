import os
import random
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self, dataset_path, directory_names, batch_size, image_size=(128, 128), shuffle=True, seed=None, preprocess_input=None):
        """
        Initializes the data generator.

        Parameters:
        - dataset_path (str): Base path to the directory containing subdirectories with .npy files.
        - directory_names (list of str): List of directory names containing .npy images for training.
        - batch_size (int): Number of images per batch.
        - image_size (tuple): Size of each image (width, height).
        - shuffle (bool): Whether to shuffle the order of images at the start and end of each epoch.
        - seed (int): Seed for reproducible shuffling.
        - preprocess_input (function): Preprocessing function to apply to images (e.g., from segmentation_models).
        """
        self.dataset_path = dataset_path
        self.directory_names = directory_names
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.seed = seed
        self.preprocess_input = preprocess_input

        # Collect all .npy files from specified directories
        self.image_files = []
        for dir_name in self.directory_names:
            dir_path = os.path.join(self.dataset_path, dir_name)
            self.image_files.extend(
                [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.npy')]
            )

        # Set up random generator
        self.rng = np.random.default_rng(self.seed)

        # Shuffle the file order initially if shuffle is True
        if self.shuffle:
            self.rng.shuffle(self.image_files)

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data.

        Parameters:
        - index (int): Index of the batch.

        Returns:
        - Tuple of (images, masks) where each is a numpy array.
        """
        batch_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        images, masks = self.__load_batch(batch_files)
        return images, masks

    def on_epoch_end(self):
        """
        Updates the order of files at the end of each epoch if shuffle is True.
        """
        if self.shuffle:
            self.rng.shuffle(self.image_files)

    def __load_batch(self, batch_files):
        """
        Loads and processes images and masks for a batch.

        Parameters:
        - batch_files (list of str): List of file paths for the current batch.

        Returns:
        - Tuple of (images, masks) as numpy arrays.
        """
        images = []
        masks = []
        for file in batch_files:
            data = np.load(file)
            image = data[..., 0]  # Assuming image data is in the 0th channel
            mask = data[..., 1]  # Assuming mask data is in the 1st channel

            # Convert grayscale to RGB by repeating the single channel 3 times
            image_rgb = np.repeat(image[..., np.newaxis], 3, axis=-1)

            # Apply preprocessing if provided
            if self.preprocess_input is not None:
                image_rgb = self.preprocess_input(image_rgb)

            images.append(image_rgb)
            masks.append(mask)

        # Reshape masks to match model output requirements
        images = np.array(images).reshape((-1, *self.image_size, 3)).astype('float32')  # Shape (batch_size * n_patch**2, 128, 128, 3)
        masks = np.array(masks).reshape((-1, *self.image_size, 1)).astype('float32')  # Shape (batch_size * n_patch**2, 128, 128, 1)
        return images, masks


def train_test_split(images, n_test=1, seed=None):
    np.random.seed(seed)
    test_images = random.sample(images, n_test)
    train_images = [img for img in images if img not in test_images]
    return train_images, test_images


def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2 * intersection / union
