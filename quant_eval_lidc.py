from eval_util import get_energy_distance_components, calc_energy_distances

from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as FT
import matplotlib.pyplot as plt

from lidc_data import LIDCCrops
import pickle
import seaborn as sns
import pandas as pd

from punet import ProbabilisticUnet

PATH_TO_DATA = "data/lidc_crops"

BATCH_SIZE = 32
DEVICE = ["cpu", "mps", "cuda"][0]
EPOCHS = int(240000 * 32 * (1 / 8882))
NUM_CLASSES = 1
NUM_CHANNELS = [32, 64, 128, 256]  # original in paper   

LATENT_DIM=6  # # original in paper (3 for image2image VAE and 6 for PUnet)
NUM_CONVS_FCOMB=3 # original in paper   
BETA=1.0 # original in paper (10 for image2image VAE and 1 for PUnet)

VAL_FREQ = 10  # run validation every number of epochs
LEARNING_RATE = 1e-4
REG_WEIGHT = 1e-5

NUM_WORKERS = 32

# Load model
checkpoint_path = Path("training_logs/NicolasWork-lidc-patches-lv3/model_checkpoints//checkpoint_epoch-40.pth")
state_dict = torch.load(str(checkpoint_path), map_location=torch.device(DEVICE))

net = ProbabilisticUnet(num_input_channels=1,
                        num_classes=NUM_CLASSES,
                        num_filters=NUM_CHANNELS,
                        latent_dim=LATENT_DIM,
                        no_convs_fcomb=NUM_CONVS_FCOMB,
                        beta=BETA,
                        device=DEVICE)
net.to(DEVICE)

net.load_state_dict(state_dict=state_dict)

# load test dataset
test_dataset = LIDCCrops(data_home=PATH_TO_DATA, split="test", transform=dict(resize=dict(output_size=(128, 128))))

image_keys = test_dataset.get_patient_image_ids()

num_samples= 16

output_size=(128, 128)

eval_class_ids = [1]
num_modes = 4 # 4 gradiers for each image to generate 4 ground truth segmentation in dataset 

# iterate on test images to generate samples from trained model and calcuate energy distance. 
def eval_matrix(matrixfilename):
    d_matrices = {'YS': np.zeros(shape=(len(image_keys), num_modes, num_samples, len(eval_class_ids)),dtype=np.float32),
                  'YY': np.ones(shape=(len(image_keys), num_modes, num_modes, len(eval_class_ids)),dtype=np.float32),
                  'SS': np.ones(shape=(len(image_keys), num_samples, num_samples, len(eval_class_ids)),dtype=np.float32)}
    
    # iterate all images
    img_n = 0
    for (patient_id, image_id) in image_keys:
        img, segs = test_dataset.get_img_segs(patient_id, image_id)

        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(dim=0) # we add the channel dim

        # Resize to make compatible with network
        img = FT.resize(img, size=output_size, interpolation=v2.InterpolationMode.BILINEAR)
        segs = [torch.from_numpy(seg.astype(np.uint8)).long().unsqueeze(dim=0) for seg in segs] # we add the channel dim
        segs = [FT.resize(seg, size=output_size, interpolation=v2.InterpolationMode.NEAREST) for seg in segs]


        img = img.unsqueeze(dim=0)  # adds batch dim
        img = img.to(DEVICE)
        net(img, None, training=False)  # Run net (this initializes the unet features and the latent space)

        preds = []
        for j in range(num_samples):
            prob = net.sample(testing=True)  # samples a segmentation using the unet features + the latent space
            prob = torch.sigmoid(prob)
            pred = (prob > 0.5).float() # get segmentation result from probs
            preds.append(pred)

        gt_seg_modes = np.asarray([seg.unsqueeze(dim=0).numpy() for seg in segs])
        seg_samples = np.asarray([pred.detach().cpu().numpy() for pred in preds])

        energy_dist = get_energy_distance_components(gt_seg_modes=gt_seg_modes, seg_samples=seg_samples,eval_class_ids=eval_class_ids, ignore_mask=None)
        
        for k in d_matrices.keys():
            d_matrices[k][img_n] = energy_dist[k]

        img_n += 1
    
    with open(matrixfilename, 'wb') as fp:
        pickle.dump(d_matrices, fp)


# calculate energy distances for each sample size
def eval_energy(matrixfilename, energyfilename):
    with open(matrixfilename, 'rb') as fp:
        d_matrices = pickle.load(fp)

    e_distances = []
    e_means = []

    for s in [1,4,8,16]:
        e_dist = calc_energy_distances(d_matrices, num_samples=s) 
        e_dist = e_dist[~np.isnan(e_dist)]
        e_distances.append(e_dist)
        e_means.append(np.mean(e_dist))

    with open(energyfilename, 'wb') as fp:
        pickle.dump(e_distances, fp)
        pickle.dump(e_means, fp)

# draw stripplot
def draw_plot(energyfilename):
    with open(energyfilename, 'rb') as fp:
        e_distances= pickle.load(fp)
        e_means = pickle.load(fp)

    s = [1,4,8,16]
    samples_column = []
    for i in range(len(e_distances)):
        samples_column.extend([s[i]] * len(e_distances[i]))

    energy = pd.DataFrame(data={'energy':  np.concatenate(e_distances).ravel(), 'num_samples': samples_column})
    means = pd.DataFrame(data={'energy': e_means, 'num_samples': s})

    plt.figure(figsize=(5.5,3.5))
    ax = plt.gca()

    sns.stripplot(x="num_samples", y="energy", data=energy, color='limegreen', alpha=0.5, s=2, ax=ax, )
    sns.stripplot(x="num_samples", y="energy", data=means, s=16, marker='^', color='black', ax=ax, jitter=False)
    sns.stripplot(x="num_samples", y="energy", data=means, s=12, marker='^',color='limegreen', ax=ax, jitter=False)
    ax.set_title('LIDC (Probabilistic U-Net)', y=1)
    fs=11
    ax.set_ylabel(r'$D_{GED}^{2}$', fontsize=fs)
    ax.set_xlabel('# samples', fontsize=fs)

    plt.show()


if __name__ == "__main__":
    # save distance matrix in matrixfilename and final energy distance result in energyfilename
    matrixfilename = 'matrix.pkl'
    energyfilename = 'energy.pkl'
    eval_matrix(matrixfilename)
    eval_energy(matrixfilename,energyfilename)
    draw_plot(energyfilename)

