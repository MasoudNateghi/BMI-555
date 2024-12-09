import tensorflow as tf
from utils.train import *
import matplotlib.pyplot as plt
from segmentation_models import Unet
from tensorflow.keras.optimizers import Adam
from segmentation_models import get_preprocessing

# Set up paths
dataset_path = '../../dataset'
result_path = '../../results'
models_path = '../../models'
os.makedirs(result_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Model configuration
BACKBONE = 'resnet34'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

#%% Split images into train, test, and validation sets
images_path = sorted(os.listdir(dataset_path))
train_dirs, test_dirs = train_test_split(images_path, n_test=1, seed=SEED)
train_dirs, valid_dirs = train_test_split(train_dirs, n_test=1, seed=SEED)

# # test
# train_dirs = ['be01p01']
# valid_dirs = ['be01p01']

#%% Create data generators
preprocess_input = get_preprocessing(BACKBONE)

train_generator = DataLoader(
    dataset_path=dataset_path,
    directory_names=train_dirs,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,
    preprocess_input=preprocess_input
)

valid_generator = DataLoader(
    dataset_path=dataset_path,
    directory_names=valid_dirs,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=False,
    preprocess_input=preprocess_input
)

# Convert to tf.data.Dataset with prefetch
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *IMAGE_SIZE, 1), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_generator(
    lambda: valid_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *IMAGE_SIZE, 1), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

#%% Define the model
model = Unet(backbone_name=BACKBONE,
            encoder_weights='imagenet',
            input_shape=(*IMAGE_SIZE, 3),
            classes=1,
            activation='sigmoid')

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[dice_metric])
print(model.summary())

#%% Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(models_path, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]

#%% Train the model
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

#%% plot the training and validation accuracy and loss at each epoch
plt.figure(figsize=(10, 8))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
fig_name = os.path.join(result_path, 'loss.png')
plt.savefig(fig_name)
plt.close()

# Plot Dice Metric
plt.figure(figsize=(10, 8))
dice = history.history['dice_metric']
val_dice = history.history['val_dice_metric']

plt.plot(epochs, dice, 'y', label='Training Dice')
plt.plot(epochs, val_dice, 'r', label='Validation Dice')
plt.title('Training and Validation Dice Metric')
plt.xlabel('Epochs')
plt.ylabel('Dice Metric')
plt.legend()
fig_name = os.path.join(result_path, 'dice_metric.png')
plt.savefig(fig_name)
plt.close()

tf.keras.backend.clear_session()  # Clear the current session
