import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json
import cv2
from tensorflow.python.ops.ragged import ragged_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
inputimage = input("Enter image you want to create: ")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
sys.stderr = open("error_output.txt", "w")
datasize = 10000
batch_size = 3
epochs = 20
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 640
num_batches = 4  # Set the number of batches to process
image_folder = 'train2017/train2017/'
def is_utf8_encodable(s):
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False

image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if is_utf8_encodable(img)]
image_paths = image_paths[:datasize]
captions_dict = {}

print("Start")
def load_captions_json(json_path):
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        captions_data = json.load(f)

    for item in captions_data['annotations']:
        image_id = str(item['image_id']).zfill(12)
        caption = item['caption']

        if image_id not in captions_dict:
            captions_dict[image_id] = []
        captions_dict[image_id].append(caption)

    for image_id in captions_dict:
        captions_dict[image_id] = '<end>'.join(captions_dict[image_id])

    return captions_dict

captions_json_path = 'train2017/annotations_trainval2017/annotations/captions_train2017.json'

def create_lookup_table(captions_dict):
    keys = list(captions_dict.keys())
    values = list(captions_dict.values())
    keys_tensor = tf.constant(keys)
    values_tensor = tf.constant(values)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), "")
    return table


captions_dict = load_captions_json(captions_json_path)
captions_lookup_table = create_lookup_table(captions_dict)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def resize_and_normalize(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

def wrapper_function(image_path_tensor):
    def wrapped_load_data(image_path_tensor):
        image_path = image_path_tensor.numpy().decode('utf-8')
        image = load_image(image_path)
        caption = captions_lookup_table.lookup(image_path_tensor)
        return image, caption

    image, caption = tf.py_function(
        wrapped_load_data,
        inp=[image_path_tensor],
        Tout=[tf.float32, tf.string]
    )

    image.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return image, caption


def create_dataset(image_paths):
    # Load image paths
    image_paths = tf.data.Dataset.from_tensor_slices(image_paths)

    # Load images and captions
    ds = image_paths.map(wrapper_function, num_parallel_calls=AUTOTUNE)

    return ds

ds = create_dataset(image_paths)
for image, label in ds.take(1):
    print("Image:", image)
    print("Label:", label)
    print("Image shape:", image.shape)
# Fit the tokenizer on the dataset
tokenizer = tf.keras.preprocessing.text.Tokenizer()

captions_list = []
iter = 0
def resize_images(image, label):
    print(f"Image shape before resizing: {image.shape}")
    resized_image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return resized_image, label



ds = ds.map(resize_images)
ds = ds.batch(batch_size)
for images, captions in ds.take(num_batches):
    captions = [caption.numpy().decode('utf-8') for caption in captions]

    tokenizer.fit_on_texts(captions)

    iter += 1
    if iter >= datasize:
        break

print("reachmepls")

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
    generated_image_numpy = np.float32(generated_image.numpy()[0] * 255)
    bgr_image = cv2.cvtColor(generated_image_numpy, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_filename, bgr_image)

    plt.imshow(generated_image_numpy / 255.0)
    plt.show()

    return generated_image

def build_generator():
    model = models.Sequential([
        layers.Dense(20 * 20 * 1024, activation='relu', input_shape=(200,)),
        layers.Reshape((20, 20, 1024)),
        layers.UpSampling2D(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.UpSampling2D(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.UpSampling2D(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.UpSampling2D(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.UpSampling2D(),  # Add this layer to increase the spatial dimensions
        layers.Conv2D(3, (3, 3), padding='same', activation='tanh')
    ])
    return model




generator = build_generator()

# Dummy discriminator, as we're not using it in this example
discriminator = None

# Define loss function and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
def generator_loss(fake_output):
    return loss_fn(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_fn(tf.ones_like(real_output), real_output)
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def build_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


discriminator = build_discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images, captions):
    noise = tf.random.normal([images.shape[0], 100])  # Changed from 'batch_size' to 'images.shape[0]'
    captions = tf.cast(captions, tf.float32)  # Add this line to cast captions to float32
    generator_input = tf.concat([noise, captions], axis=-1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(generator_input, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    print(f"Generator loss: {gen_loss}, Discriminator loss: {disc_loss}")  # Remove .numpy() calls
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))




# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    for step, (images, captions) in enumerate(ds):
        encoded_captions = tokenizer.texts_to_sequences([caption.numpy().decode('utf-8') for caption in captions])
        max_length = 100
        padded_captions = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_captions, maxlen=max_length, padding='post')
        train_step(images, padded_captions)
        print('.', end='')
    print()

    #Tests generator on each epoch
    output_dir = f'generated_images/epoch_{epoch + 1}'
    os.makedirs(output_dir, exist_ok=True)
    #print(f"Output directory: {os.path.abspath(output_dir)}")
    test_description(inputimage, generator, tokenizer, output_dir)




#Test generator with example (Merge Descriptions to single text at top)
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)
test_description(inputimage, generator, tokenizer, output_dir)