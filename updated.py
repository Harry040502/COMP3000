import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer

import os
import json
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
import cv2
import pickle
from tensorflow.python.ops.ragged import ragged_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#inputimage = input("Enter image you want to create: ")
inputimage = "A blue sky"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
sys.stderr = open("error_output.txt", "w")
datasize = 100000
learning_rate = 1e-4
batch_size = 60
epochs = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 299
num_batches = 10
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
captions_list = []
for captions in captions_dict.values():
    captions_list.extend(captions)
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


def wrapper_function(img_path, cap):
    img_tensor = load_image(img_path)
    img_tensor = tf.image.resize(img_tensor, (299, 299))
    img_tensor = tf.keras.applications.inception_v3.preprocess_input(img_tensor)

    cap = tf.strings.as_string(cap) # convert to string type
    cap = captions_lookup_table.lookup(cap)

    return img_tensor, cap



new_captions_dict = {key: "<end>".join(value.split("<end>")[:3]) for key, value in captions_dict.items()}
print(new_captions_dict)
vocab_size = 10000  # Choose an appropriate vocab size based on your dataset
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(captions_list)
def create_dataset(image_paths, captions_dict, batch_size):
    # Tokenize and pad captions
    captions_list = list(captions_dict.values())
    encoded_captions = tokenizer.texts_to_sequences(captions_list)
    max_length = 100
    padded_captions = tf.keras.preprocessing.sequence.pad_sequences(encoded_captions, maxlen=max_length, padding='post')

    # Create the dataset
    image_paths = tf.data.Dataset.from_tensor_slices(image_paths)
    captions_tensor = tf.constant(padded_captions, dtype=tf.int32)
    captions = tf.data.Dataset.from_tensor_slices(captions_tensor)
    ds = tf.data.Dataset.zip((image_paths, captions))
    ds = ds.map(wrapper_function, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)  # Add this line to create batches
    return ds





ds = create_dataset(image_paths, captions_dict, batch_size)


for image, label in ds.take(1):
    print("Image:", image)
    print("Label:", label)
    print("Image shape:", image.shape)
#attempt to fit tokenizer (Review Tokenizer doesnt appear to work as intended)
tokenizer = tf.keras.preprocessing.text.Tokenizer()

captions_list = []
iter = 0

def resize_images(image, label):
    print(f"Image shape before resizing: {image.shape}")
    resized_image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return resized_image, label



captions_list = list(captions_dict.values())
tokenizer.fit_on_texts(captions_list)


def test_description(description, generator, tokenizer, output_dir):
    #description to images for use by Generator
    encoded_description = tokenizer.texts_to_sequences([description])

    #apply padding to sequence/description
    max_length = 100
    padded_description = tf.keras.preprocessing.sequence.pad_sequences(encoded_description, maxlen=100, padding='post')

    #apply noise
    noise = tf.random.normal([1, 100])
    generator_input = tf.concat([noise, tf.cast(padded_description, tf.float32)], axis=-1)
    #run generator
    generated_image = generator(generator_input, training=False)

    #change lighting of generated_image
    generated_image = (generated_image + 1) / 2.0

    #save image to output directly with description_name
    image_filename = os.path.join(output_dir, f'{description}.jpg')
    generated_image_numpy = np.float32(generated_image.numpy()[0] * 255)
    bgr_image = cv2.cvtColor(generated_image_numpy, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_filename, bgr_image)

    #plt.imshow(generated_image_numpy / 255.0)
    #plt.show()

    return generated_image


def build_generator(): #builds image generator
    input_layer = layers.Input(shape=(200,))
    dense1 = layers.Dense(20 * 20 * 1024, activation='relu')(input_layer)
    reshape1 = layers.Reshape((20, 20, 1024))(dense1)
    c1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(reshape1)
    p1 = layers.MaxPooling2D((2, 2), padding='same')(c1)
    c2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2), padding='same')(c2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D((2, 2), padding='same')(c3)

    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(p3)
    c3_resized = layers.Lambda(lambda x: tf.image.resize(x, (u1.shape[1], u1.shape[2])))(c3)
    u1 = layers.concatenate([u1, c3_resized])

    u2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u1)
    c2_resized = layers.Lambda(lambda x: tf.image.resize(x, (u2.shape[1], u2.shape[2])))(c2)
    u2 = layers.concatenate([u2, c2_resized])

    u3 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(u2)
    c1_resized = layers.Lambda(lambda x: tf.image.resize(x, (u3.shape[1], u3.shape[2])))(c1)
    u3 = layers.concatenate([u3, c1_resized])

    #More Upsampling Layers added after the fact
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(u3)
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u4)
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u5)
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)

    output_layer = layers.Conv2D(3, (3, 3), padding='same')(u7)
    output_layer = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(u7)
    #resize to IMG_SIZE
    resized_output = layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE)(output_layer)

    model = models.Model(inputs=input_layer, outputs=resized_output)
    return model

def gradient_penalty(real_images, fake_images, discriminator, caption):
    alpha = tf.random.normal([real_images.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = discriminator([interpolated_images, caption])
    gradients = tape.gradient(predictions, interpolated_images)
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=[1, 2, 3])
    gradient_penalty = tf.reduce_mean(gradients_sqr_sum - 1) * 10
    return gradient_penalty


generator = build_generator()

#init discriminator
discriminator = None
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=learning_rate, decay_steps=10000, alpha=0.1
)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate) #using rmsprop optimizer (might change to Adam depending on performance when results start coming through)

def build_discriminator():
    input_image = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    input_caption = layers.Input(shape=(100,))

    # Use an embedding layer to process captions
    x = layers.Embedding(vocab_size, 128)(input_caption)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(IMG_SIZE * IMG_SIZE)(x)
    x = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    # Combine image and caption
    x = layers.Concatenate(axis=-1)([input_image, x])

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)  # Fixed this line
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)  # Fixed this line
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    model = models.Model(inputs=[input_image, input_caption], outputs=output)
    return model




discriminator = build_discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(images, captions):

    for i in range(images.shape[0]):
        noise = tf.random.normal([1, 100])
        caption = tf.expand_dims(captions[i], axis=0)
        caption = tf.cast(caption, tf.float32)
        generator_input = tf.concat([noise, caption], axis=-1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = generator(generator_input, training=True)

            real_output = discriminator([images[i:i + 1], captions[i:i + 1]], training=True)
            fake_output = discriminator([generated_image, captions[i:i + 1]], training=True)

            #add a wasserstein loss function
            gen_loss = -tf.reduce_mean(fake_output)
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

            #add a gradient penalty
            gp = gradient_penalty(images[i:i+1], generated_image, discriminator, captions[i:i+1])
            disc_loss += gp

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        print(f"Generator loss: {gen_loss}, Discriminator loss: {disc_loss}")

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#Start the Training Loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print("Batch Size: " + str(batch_size))
    for step, (images, captions) in enumerate(ds.take(batch_size).unbatch().batch(batch_size)):
        captions_list = [caption.astype('U').tolist() for caption in captions.numpy()]
        encoded_captions = tokenizer.texts_to_sequences(captions_list)
        max_length = 100
        padded_captions = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_captions, maxlen=max_length, padding='post')
        train_step(images, padded_captions)
        print('.', end='')
    print()


    #Tests generator on each epoch
    #create all images in individual epoch folders starting from 1
    output_dir = f'generated_images/epoch_{epoch + 1}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    test_description(inputimage, generator, tokenizer, output_dir)

generator.save('generator_model')

with open('tokenizer.pk1', 'wb') as f:
    pickle.dump(tokenizer,f)