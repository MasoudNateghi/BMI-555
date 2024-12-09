import numpy as np
import matplotlib.pyplot as plt
from patchify import unpatchify

def predict_patches(model, patches, preprocess_input=None, batch_size=32):
    """
    Predict masks for all patches with batch processing

    Args:
        model: Loaded keras model
        patches: Array of shape (256, 128, 128) or (256, 128, 128, 1)
        preprocess_input: Preprocessing function
        batch_size: Batch size for prediction

    Returns:
        Array of predicted masks
    """
    # Ensure patches are in the right shape (N, H, W, C)
    if len(patches.shape) == 3:
        patches = patches[..., np.newaxis]

    # Convert to RGB
    patches_rgb = np.repeat(patches, 3, axis=-1)

    # Preprocess
    if preprocess_input is not None:
        patches_rgb = preprocess_input(patches_rgb)

    # Predict in batches
    predictions = []
    for i in range(0, len(patches_rgb), batch_size):
        batch = patches_rgb[i:i + batch_size]
        pred = model.predict(batch, verbose=0)
        predictions.append(pred)

    # Combine all predictions
    predictions = np.concatenate(predictions, axis=0)

    # Apply threshold for binary prediction
    predictions = (predictions > 0.5).astype(np.float32)

    return predictions


def reconstruct_from_patches(patches, output_shape):
    # Reshape to (16, 16, 128, 128)
    patches_reshaped = patches.reshape(16, 16, 128, 128)

    # Unpatchify
    reconstructed = unpatchify(patches_reshaped, output_shape)

    return reconstructed


def visualize_results(original_patches, predicted_mask, save_path=None):
    # Reconstruct original image
    original_reshaped = original_patches.reshape(16, 16, 128, 128)
    original_image = unpatchify(original_reshaped, (2048, 2048)).astype('uint8')

    plt.figure(figsize=(30, 15))

    plt.subplot(121)
    plt.title('Original Image', fontsize=30)
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.title('Predicted Mask', fontsize=30)
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()