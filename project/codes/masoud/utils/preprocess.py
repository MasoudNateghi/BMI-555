import os
import cv2
import numpy as np
from patchify import patchify
from joblib import Parallel, delayed


def load_tiff(tiff_file):
    frames = np.array(cv2.imreadmulti(tiff_file)[1])
    return frames


def preprocess(frames_img, frames_mask, frame_size=2048, patch_size=128, threshold=20, kernel_size=9, verbose=True):
    if verbose:
        print(f"frame_size: {frame_size}, threshold: {threshold}, kernel_size: {kernel_size}, patch_size: {patch_size}")

    # preprocessing function for each frame
    def preproc_mask(frame):
        frame_resized = cv2.resize(frame, (frame_size, frame_size))  # resize the frame to 2048 * 2048
        thresholded_image = np.where(frame_resized < threshold, 0, 1).astype(np.uint8)  # apply thresholding
        median = cv2.medianBlur(thresholded_image, kernel_size)  # apply median filtering

        # patchify the image with the given patch size with no overlap
        patches = patchify(median, (patch_size, patch_size), step=patch_size).reshape(-1, patch_size, patch_size)
        return patches

    def preproc_img(frame):
        frame_resized = cv2.resize(frame, (frame_size, frame_size))  # resize the frame to 2048 * 2048

        # patchify the image with the given patch size with no overlap
        patches = patchify(frame_resized, (patch_size, patch_size), step=patch_size).reshape(-1, patch_size, patch_size)
        return patches

    # apply the preprocessing function to each frame of img and mask in parallel
    N = len(frames_img)
    result = Parallel(n_jobs=-1)(delayed(preproc_mask)(frames_mask[i]) for i in range(N))
    patches_mask = np.array(result)
    result = Parallel(n_jobs=-1)(delayed(preproc_img)(frames_img[i]) for i in range(N))
    patches_img = np.array(result)

    # concat image and mask
    data = np.stack((patches_img, patches_mask), axis=-1)
    return data


def save_data(data, save_path):
    def save_frame(data_i, i):
        frame_data = data_i  # Extract frame Shape: (256, 128, 128, 2)
        file_path = os.path.join(save_path, f"f{i:04d}.npy")
        np.save(file_path, frame_data)

    # Parallel execution using joblib
    N = data.shape[0]  # number of frames
    Parallel(n_jobs=-1)(delayed(save_frame)(data[i], i) for i in range(N))