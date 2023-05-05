import matplotlib.pyplot as plt
import numpy as np

epochs = []
fid_scores = []

with open("output.txt", "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if "FID score for epoch" in lines[i]:
            epoch = int(lines[i].split(":")[0].split()[-1])
            epochs.append(epoch)
            fid_score = float(lines[i].split(":")[1])
            fid_scores.append(fid_score)

plt.plot(epochs, fid_scores)
plt.xlabel("Epoch")
plt.ylabel("FID Score")
plt.title("FID Score per Epoch")

# Add trend line
z = np.polyfit(epochs, fid_scores, 1)
p = np.poly1d(z)
plt.plot(epochs, p(epochs), "--")

plt.show()
