import os
from utils.test import *
from utils.train import dice_metric
from tensorflow.keras.models import load_model
from segmentation_models import get_preprocessing

BACKBONE = 'resnet34'
model_path = f'../../models/{BACKBONE}/best_model.keras'
image_path = '/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/codes/cell-tracking/dataset/ce01p04/f0008.npy'
result_path = '../../results'

# Load patches and masks
data = np.load(image_path)
patches = data[..., 0]
masks = data[..., 1]

# Load model
custom_objects = {'dice_metric': dice_metric}
preprocess_input = get_preprocessing(BACKBONE)
model = load_model(model_path, custom_objects=custom_objects)

# Make predictions
print("Predicting masks...")
predictions = predict_patches(
    model,
    patches,
    preprocess_input=get_preprocessing('resnet34'),
    batch_size=32
)

# Reshape and unpatch the predictions
predicted_mask = reconstruct_from_patches(predictions, (2048, 2048))

# Plot and save results
fig_path = os.path.join(result_path, 'predicted_mask_patches.png')
visualize_results(patches, predicted_mask, save_path=fig_path)
fig_path = os.path.join(result_path, 'predicted_mask_mask.png')
visualize_results(masks, predicted_mask, save_path=fig_path)