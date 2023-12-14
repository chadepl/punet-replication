from eval_util import get_energy_distance_components, calc_energy_distances

from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as FT
import matplotlib.pyplot as plt

from lidc_data import LIDCCrops
import pickle

from punet import ProbabilisticUnet

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

# Get random images
test_dataset = LIDCCrops(data_home="../data/lidc_crops", split="test", transform=dict(resize=dict(output_size=(128, 128))))

image_keys = test_dataset.get_patient_image_ids()

num_samples= 16

output_size=(128, 128)

eval_class_ids = [1]
num_modes = 4

def eval():
    d_matrices = {'YS': np.zeros(shape=(len(image_keys), num_modes, num_samples, len(eval_class_ids)),
                                    dtype=np.float32),
                    'YY': np.ones(shape=(len(image_keys), num_modes, num_modes, len(eval_class_ids)),
                                    dtype=np.float32),
                    'SS': np.ones(shape=(len(image_keys), num_samples, num_samples, len(eval_class_ids)),
                                    dtype=np.float32)}

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
            pred = (prob > 0).float()
            preds.append(pred)

        gt_seg_modes = np.asarray([seg.unsqueeze(dim=0).numpy() for seg in segs])
        seg_samples = np.asarray([pred.detach().cpu().numpy() for pred in preds])

        energy_dist = get_energy_distance_components(gt_seg_modes=gt_seg_modes, seg_samples=seg_samples,eval_class_ids=eval_class_ids, ignore_mask=None)
        
        for k in d_matrices.keys():
            d_matrices[k][img_n] = energy_dist[k]

        img_n += 1
    
    with open('matrices.pkl', 'wb') as fp:
        pickle.dump(d_matrices, fp)

eval()

with open('matrices.pkl', 'rb') as fp:
    d_matrices = pickle.load(fp)


e_distances = []
e_means = []
for s in [1,4,8,16]:
    e_dist = calc_energy_distances(d_matrices, num_samples=s) 
    e_dist = e_dist[~np.isnan(e_dist)]
    e_distances.append(e_dist)
    e_means.append(np.mean(e_dist))

print(e_means)

with open('energy.pkl', 'wb') as fp:
    pickle.dump(e_distances, fp)
    pickle.dump(e_means, fp)


