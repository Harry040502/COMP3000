import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_weights_from_h5_file(h5_file_path):
    weights = []
    with h5py.File(h5_file_path, 'r') as f:
        for layer_name in f.keys():
            layer_weights = []
            for weight_name in f[layer_name].keys():
                weight_values = np.array(f[layer_name][weight_name])
                layer_weights.append(weight_values)
            weights.append(layer_weights)
    return weights

h5_file_path1 = 'discriminator_weights_epoch_100.h5'
h5_file_path2 = 'generator_weights_epoch_100.h5'

weights1 = read_weights_from_h5_file(h5_file_path1)
weights2 = read_weights_from_h5_file(h5_file_path2)

for i, (w1, w2) in enumerate(zip(weights1, weights2)):
    if not w1 or not w2:  # Check if either list is empty
        continue

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.hist(w1[0].flatten(), bins=50, alpha=0.5, label='Model 1')
    plt.xlabel(f'Layer {i+1} weights')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(w2[0].flatten(), bins=50, alpha=0.5, label='Model 2', color='orange')
    plt.xlabel(f'Layer {i+1} weights')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()

