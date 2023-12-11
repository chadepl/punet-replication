from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

ref = imread(Path("/Users/chadepl/Desktop/reference.png"), as_gray=True)
ref = resize(ref, (180, 180))
plt.imshow(ref)
plt.show()
fnames = []
dists = []
for p in Path("data/lidc_crops/train/images").rglob("*.png"):
    img = imread(p, as_gray=True)
    scores = []
    scores.append(np.linalg.norm(img - ref))
    scores.append(np.linalg.norm(img - ref[::-1, :]))
    scores.append(np.linalg.norm(img - ref[:, ::-1]))
    scores.append(np.linalg.norm(img - ref[::-1, ::-1]))
    fnames.append(p)
    dists.append(np.min(scores))

print(fnames[np.argmin(dists)])
print(np.min(dists))
