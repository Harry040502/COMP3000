import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import deeplake
import os
from tqdm import tqdm
import cv2
from pycocotools.coco import COCO

ds = deeplake.load('hub://activeloop/coco-train')[:100] # Returns a Deep Lake Dataset but does not download data locally
ds = ds.tensorflow()
IMG_SIZE = 640
batch_size = 32
epochs = 60

# Define the output directory for generated images
OUTPUT_DIR = "./generated_images"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def resize_and_normalize(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image


# Define the model architecture
dataset = []
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(dataset)

def load_data(item):
    image_path, captions = item['image'], item['captions']
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)
    return image, captions


for item in tqdm(ds, desc="loading dataset"):
    image, caption = load_data(item)
    image = tf.cast(image, tf.float32) / 255.0
    image_features = tf.keras.applications.inception_v3.preprocess_input(image)
    dataset.append((image_features, caption))
    tokenizer.fit_on_texts([caption])


def load_data_wrapper(item):
    progress_bar = tqdm(total=len(ds), desc="loading data")
    image = tf.io.read_file(item["images"])
    image = tf.image.decode_jpeg(image, channels=3)
    image = resize_and_normalize(image)
    caption = tf.reshape(tf.cast(item["categories"], tf.string), [1])
    progress_bar.update(1)
    return image, caption


def val_dataset(ds):
    # Define a function to load a single data item from the dataset
    def load_data(caption, image_path):
        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Convert the image to tf.uint8
        image = tf.cast(image, tf.uint8)

        # Preprocess the caption
        caption = preprocess_caption(caption)

        return image, caption

    # Load all the data items from the dataset
    val_dataset = tf.data.Dataset.from_generator(
        lambda: ((x, y) for x, y in zip(val_captions, val_images)),
        output_types=(tf.uint8, tf.string),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
    )
    val_dataset = val_dataset.map(
        lambda x, y: tuple(tf.py_function(load_data, [x, y], [tf.float32, tf.string])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Return the dataset
    return dataset


def generator(ds):
    for data in ds:
        # Extract the image tensor from the data dictionary
        image = tf.cast(data['images'], tf.float32)

        # Extract the caption tensor from the data dictionary and convert it to a string
        caption = tf.strings.reduce_join(data['captions'], separator=' ', axis=-1)

        # Yield a tuple of (image, caption) tensors
        yield image, caption


def create_dataset(ds):
    dataset = tf.data.Dataset.from_tensor_slices(ds)
    dataset = dataset.map(load_data)
    dataset = dataset.shuffle(buffer_size=len(ds))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



def test_description(description, generator, tokenizer):
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

    return generated_image



def make_generator_model():
    model = models.Sequential()
    model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(200,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 256, 256, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 512, 512, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 1024, 1024, 3)

    return model
def test_generator(generator):
    # Generate a random latent vector
    latent_vector = tf.ones([1, 200])

    # Generate an image from the latent vector
    generated_image = generator(latent_vector, training=False)

    # Convert the image from [-1, 1] to [0, 1] range
    generated_image = (generated_image + 1) / 2.0

    return generated_image


def test_description(description, generator, tokenizer):
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

    return generated_image

def generate_images(model, description, tokenizer, output_dir, epoch):
    # Convert the description into a sequence of integers
    encoded_description = tokenizer.texts_to_sequences([description])

    # Pad the sequence to the maximum length
    max_length = 100
    padded_description = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_description, maxlen=max_length, padding='post')

    # Generate an image from the description
    noise = tf.random.normal([1, 200])
    generator_input = tf.concat([noise, padded_description], axis=-1)
    generated_image = model(generator_input, training=False)

    # Convert the image from [-1, 1] to [0, 255] range
    generated_image = (generated_image + 1.0) * 127.5
    generated_image = tf.cast(generated_image, tf.uint8)

    # Save the generated image to a file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"generated_image_epoch_{epoch}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, generated_image.numpy()[0])

    return generated_image.numpy()[0]


def train(generator, discriminator, gan, dataset, epochs, checkpoint_prefix):
    # Define the loss functions
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(real_images, captions):
        # Generate random noise vectors
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images from the captions
            generated_images = generator([noise, captions], training=True)

            # Discriminate between real and fake images
            real_output = discriminator([real_images, captions], training=True)
            fake_output = discriminator([generated_images, captions], training=True)

            # Compute the loss for the generator
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            gen_mse_loss = mse(real_images, generated_images)
            gen_total_loss = gen_loss + (LAMBDA * gen_mse_loss)

            # Compute the loss for the discriminator
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = (real_loss + fake_loss) / 2.0

        # Compute the gradients for the generator and discriminator
        gradients_of_generator = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Apply the gradients to the generator and discriminator optimizers
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss, gen_mse_loss

    # Define a function to generate and save sample images
    def generate_images(model, epoch, test_input):
        # Generate images from the test input
        predictions = model(test_input, training=False)

        # Rescale the pixel values from [-1, 1] to [0, 255]
        predictions = (predictions + 1.0) / 2.0 * 255.0

        # Save the images to disk
        for i in range(predictions.shape[0]):
            image = tf.cast(predictions[i], tf.uint8)
            image = tf.image.encode_jpeg(image)
            filename = 'image_at_epoch_{:04d}_{}.jpg'.format(epoch + 1, i + 1)
            tf.io.write_file(os.path.join(checkpoint_prefix, filename), image)

    # Generate a fixed noise vector for testing the generator
    test_noise = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, LATENT_DIM])
    test_caption = tf.ones([NUM_EXAMPLES_TO_GENERATE, MAX_SEQUENCE_LENGTH])

    for epoch in range(epochs):
        # Train the generator and discriminator for one epoch
        gen_losses = []
        disc_losses = []
        gen_mse_losses = []
        for image_batch, caption_batch in dataset:
            gen_loss, disc_loss, gen_mse_loss = train_step(image_batch, caption_batch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)
            gen_mse_losses.append(gen_mse_loss)
            if (epoch + 1) % 10 == 0:
                # Generate images using the generator after every 10 epochs
                generate_images(generator, epoch + 1, test_description_text)

                # Save the model after every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

            # Save the final model
            checkpoint.save(file_prefix=checkpoint_prefix)

            # Generate a final set of images using the trained generator
            generate_images(generator, epochs, test_description_text)
