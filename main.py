import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3

import nvidia_smi
import os
import json
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
import cv2
import pickle
from tensorflow.python.ops.ragged import ragged_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from createnewdataset import filter_and_copy_images
#policy = mixed_precision.Policy('mixed_float16')
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) #experimental feature in this case required for consumer GPUs due to high GPU memory consumption by my program.
    except RuntimeError as e:
        print(e)
#mixed_precision.set_policy(policy)
inputtextimage = input("Enter the image you would like to generate: ")
if not inputtextimage:
    print("The input cannot be empty")
    raise SystemExit()
elif inputtextimage.isspace():
    print("The input cannot be empty")
    raise SystemExit()
source_images_folder = "train2017/train2017/"
source_annotations_file = "train2017/annotations_trainval2017/annotations/captions_train2017.json"
output_images_folder = "filtered_images/"
output_annotations_file = "filtered_captions.json"
filter_and_copy_images(inputtextimage, source_images_folder, source_annotations_file, output_images_folder, output_annotations_file)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if (len(tf.config.experimental.list_physical_devices('GPU')) > 0):
    print("I recommend you run this on anaconda with GPU support for faster run times (Note that this may affect max allowed batch size dependant on GPU memory size")
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

tf.keras.backend.clear_session() #needed to clear GPU memory from filter_and_copy images using it all up
sys.stderr = open("error_output.txt", "w")
datasize = 10000 #May need to reduce this if you have problems with certain inputs and memory problems on GPU
learning_rate = 1e-5
batch_size = 40
epochs = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 299
image_folder = 'filtered_images'
def evaluate_generator(generator, inception_model, image_paths, tokenizer, num_samples=1000):
    real_images = []
    generated_images = []
    num_samples = min(num_samples, len(image_paths))

    for i in range(num_samples):
        #load and process the real image
        real_image = load_image(image_paths[i])
        real_image = resize_and_normalize(real_image)
        real_image = np.expand_dims(real_image, axis=0)

        #Generate images based on the caption
        caption = captions_lookup_table.lookup(tf.constant(i, dtype=tf.int32))

        generated_image = test_description(caption.numpy().decode('utf-8'), generator, tokenizer, 'evaluation_output')

        real_images.append(real_image)
        generated_images.append(generated_image)

    real_images = np.concatenate(real_images, axis=0)
    generated_images = np.concatenate(generated_images, axis=0)
    #Note enabling the below line can cause longer train times. as it will generate a score per epoch.
    fid_score = calculate_fid(inception_model, real_images, generated_images)
    return fid_score

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
#load inception without top layers

def calculate_fid(model, images1, images2): #used for calculating accuracy of cGAN
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2)) #use scipy to use matrix square root for computation optimization

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    #Calculate FID Score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def load_captions_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    captions_dict = {}

    for item in data:  #Iterate over each dict in list
        image_id = int(item['image_id'])  #Convert the image_id to an integer so it can be processed
        caption = item['caption']
        captions_dict[image_id] = caption

    return captions_dict


captions_json_path = 'filtered_captions.json'
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
    img_tensor = tf.image.resize(img_tensor, (IMG_SIZE, IMG_SIZE))
    img_tensor = tf.keras.applications.inception_v3.preprocess_input(img_tensor)

    cap = captions_lookup_table.lookup(cap)

    return img_tensor, cap


new_captions_dict = {key: "<end>".join(value.split("<end>")[:3]) for key, value in captions_dict.items()}
print(new_captions_dict)
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(captions_list)
def create_dataset(image_paths, captions_dict, batch_size):
    captions_list = list(captions_dict.values())
    encoded_captions = tokenizer.texts_to_sequences(captions_list)
    max_length = 100
    padded_captions = tf.keras.preprocessing.sequence.pad_sequences(encoded_captions, maxlen=max_length, padding='post')

    #create dataset
    image_paths = tf.data.Dataset.from_tensor_slices(image_paths)
    captions_tensor = tf.constant(padded_captions, dtype=tf.int32)
    captions = tf.data.Dataset.from_tensor_slices(captions_tensor)
    ds = tf.data.Dataset.zip((image_paths, captions))
    ds = ds.map(wrapper_function, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
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

    #Use an embedding layer to process captions
    x = layers.Embedding(vocab_size, 128)(input_caption)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(IMG_SIZE * IMG_SIZE)(x)
    x = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    #Combines image and caption
    x = layers.Concatenate(axis=-1)([input_image, x])

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    model = models.Model(inputs=[input_image, input_caption], outputs=output)
    return model


discriminator = build_discriminator()
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
def plot_losses(gen_losses, disc_losses):
    epochs = range(1, len(gen_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, gen_losses, label='Generator Loss', linestyle='-', marker='o')
    plt.plot(epochs, disc_losses, label='Discriminator Loss', linestyle='-', marker='o')

    plt.title('Generator and Discriminator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.savefig('losses_plot.png')  # Save the plot to a file
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
        #print(f"Generator loss: {gen_loss}, Discriminator loss: {disc_loss}")

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

gen_losses = []
disc_losses = []
#Start the Training Loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print("Batch Size: " + str(batch_size))
    gen_loss_epoch = []
    disc_loss_epoch = []
    for step, (images, captions) in enumerate(ds.take(batch_size).unbatch().batch(batch_size)):
        captions_list = [caption.astype('U').tolist() for caption in captions.numpy()]
        encoded_captions = tokenizer.texts_to_sequences(captions_list)
        max_length = 100
        padded_captions = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_captions, maxlen=max_length, padding='post')
        gen_loss, disc_loss = train_step(images, padded_captions)
        gen_loss_epoch.append(gen_loss.numpy())
        disc_loss_epoch.append(disc_loss.numpy())

        print('.', end='')
    gen_losses.append(np.mean(gen_loss_epoch))
    disc_losses.append(np.mean(disc_loss_epoch))
    plot_losses(gen_losses, disc_losses)
    if (epoch + 1) % 50 == 0:  # Save the model every 50 epochs
        generator.save_weights(f'generator_weights_epoch_{epoch + 1}.h5')
        discriminator.save_weights(f'discriminator_weights_epoch_{epoch + 1}.h5')
    # May Cause Lots of lag (doesnt work on my GPU due to memory limits)

    try:
        if bool(gpus) == True:  # check if gpu is present
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # only works for 1 card as 0 is hardcoded
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)  # finds GPU card info (specifically free memory is what we are looking for)
            if (int(info.free) > (
                    12 / 1.074e+9)):  # determines how much free memory you have and whether you're capable of running this program
                inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
                fid_score = evaluate_generator(generator, inception_model, image_paths, tokenizer)
                print(f"FID score for epoch {epoch + 1}: {fid_score}")
    except:
        inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
        fid_score = evaluate_generator(generator, inception_model, image_paths, tokenizer)
        print(f"FID score for epoch {epoch + 1}: {fid_score}")
        print(info.free)



    print()


    #Tests generator on each epoch
    #create all images in individual epoch folders starting from 1
    output_dir = f'generated_images/epoch_{epoch + 1}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    test_description(inputtextimage, generator, tokenizer, output_dir)
try:
    generator.save('generator_model', save_format='tf') #saves generator as file so it doesnt require retraining.
except:
    print("Unable to save generator model.")
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
fid_score = evaluate_generator(generator, inception_model, image_paths, tokenizer)
print("FID score:", fid_score)
print("Reached me")
with open('tokenizer.pk1', 'wb') as f:
    pickle.dump(tokenizer,f)
sys.stdout = open('output.txt', 'wt')