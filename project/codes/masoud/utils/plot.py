import cv2
import numpy as np
import matplotlib.pyplot as plt
from patchify import unpatchify


def add_edge(img, mask):
    edges = cv2.Canny((mask * 255).astype('uint8'), threshold1=50, threshold2=150)
    edged_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    edged_img[edges > 0] = [0, 0, 255]
    return edged_img


def plot_random_patch(img, mask):
    n = np.random.randint(0, 256)
    patch_img = img[n]
    patch_mask = mask[n]
    color_img = add_edge(patch_img, patch_mask)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis('off')
    plt.show()


def plot_whole_image(img, mask):
    if len(img.shape) == 3:
        img = unpatchify(img.reshape(16, 16, 128, 128), (2048, 2048))
    if len(mask.shape) == 3:
        mask = unpatchify(mask.reshape(16, 16, 128, 128), (2048, 2048))
    edged_image = add_edge(img, mask)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(edged_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis('off')
    plt.show()