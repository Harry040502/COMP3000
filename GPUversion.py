import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers, models
import deeplake
import os
import cv2
from tqdm import tqdm

datasize = 50
ds = deeplake.load('hub://activeloop/coco-train')[:datasize]
IMG_SIZE = 640
batch_size = 16
epochs = 60
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')
def preprocess_data(image, caption):
    # Define your preprocessing logic here
    # return the preprocessed image and caption
    return image, caption

def resize_and_normalize(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

def load_data(item):
    image_path = item["images"]
    if isinstance(image_path, int):
        image_path = str(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = resize_and_normalize(image)
    caption = tf.strings.reduce_join(item["captions"], separator=' ', axis=-1)
    return image, caption

def create_dataset(ds):
    ds = tf.data.Dataset.from_generator(
        lambda: (load_data(item) for item in ds),
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )

    ds = ds.shuffle(buffer_size=datasize)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

ds = create_dataset(ds)

# Fit the tokenizer on the dataset
tokenizer = tf.keras.preprocessing.text.Tokenizer()
for _, captions in ds:
    tokenizer.fit_on_texts(captions.numpy())

def test_description(description, generator, tokenizer, output_dir):
    # Convert the description into a sequence of integers
    encoded_description = tokenizer.texts_to_sequences([description])

    # Pad the sequence to the maximum length
    max_length = 100
    padded_description = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_description, maxlen=max_length, padding='post')

    # Concatenate the random noise vector with the padded sequence
    noise = tf.random.normal([1, 100])
    generator_input = tf.concat([noise, padded_description], axis=-1)

    # Generate an image from the description
    generated_image = generator(generator_input, training=False)

    # Convert the image from [-1, 1] to [0, 1] range
    generated_image = (generated_image + 1) / 2.0

    # Save the generated image to the output directory
    image_filename = os.path.join(output_dir, f'{description}.jpg')
    bgr_image = cv2.cvtColor(np.float32(generated_image.numpy()[0] * 255), cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_filename, bgr_image)

    return generated_image

def build_generator():
    model = models.Sequential([
        layers.Dense(4 * 4 * 1024, activation='relu', input_shape=(210,)),
        layers.Reshape((4, 4, 1024)),
        layers.UpSampling2D(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.UpSampling2D(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.UpSampling2D(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.UpSampling2D(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(3, (3, 3), padding='same', activation='tanh')
    ])
    return model

with strategy.scope():
    generator = build_generator()

# Dummy discriminator, as we're not using it in this example
    discriminator = None

# Define loss function and optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(images, captions):
        noise = tf.random.normal([batch_size, 100])
        encoded_captions = tokenizer.texts_to_sequences(captions.numpy())
        max_length = 100
        padded_captions = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_captions, maxlen=max_length, padding='post')
        generator_input = tf.concat([noise, padded_captions], axis=-1)

        with tf.GradientTape() as gen_tape:
            generated_images = generator(generator_input, training=True)
            # You should calculate the loss using the discriminator, but we're skipping it for simplicity
            # This dummy loss will make the generator produce random images
            loss = loss_fn(tf.ones_like(generated_images), generated_images)

        gradients = gen_tape.gradient(loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    # Training loop
    num_steps = sum(1 for _ in ds)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        progress_bar = tqdm(total=num_steps, desc="Training")
        for step, (images, captions) in enumerate(ds):
            train_step(images, captions)
            progress_bar.update(1)  # Update the progress bar
            print('.', end='')
        progress_bar.close()
        print()

# Test the generator
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)
test_description("A beautiful beach with clear blue water", generator, tokenizer, output_dir)
