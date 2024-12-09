from utils.plot import *
from utils.test import *
from patchify import patchify

BACKBONE = 'resnet34'
model_path = f'../../models/{BACKBONE}/best_model.keras'
image = 'ce01p02f0018.npy'
image_path = f'/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/codes/cell-tracking/codes/masoud/misc/files/{image}'

data = np.load(image_path)
img = data[..., 0]
mask = data[..., 1]
#%% A random patch
plot_random_patch(img, mask)

#%% Whole image
plot_whole_image(img, mask)

#%% Test U-Net model
mask_pred = predict(model_path, image_path)

#%% A random patch
pred_mask_patches = patchify(mask_pred, patch_size=(128, 128), step=128).reshape(-1, 128, 128)
plot_random_patch(img, pred_mask_patches)

#%% Whole image
plot_whole_image(img, mask_pred)
