import tensorflow as tf
import deeplake

ds = deeplake.load('hub://activeloop/coco-train')[:100]

def preprocess_data(image, caption):
    # define your preprocessing logic here
    # return the preprocessed image and caption
    return image, caption

ds_tf = tf.data.Dataset.from_generator(lambda: ((x['image'], x['caption']) for x in ds),
                                       output_types=(tf.float32, tf.string),
                                       output_shapes=((None, None, 3), ()))

ds_tf = ds_tf.map(preprocess_data)