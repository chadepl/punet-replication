# from eval_util import get_energy_distance_components

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from lidc_data import LIDCCrops

from punet import ProbabilisticUnet

BATCH_SIZE = 32
DEVICE = ["cpu", "mps", "cuda"][2]
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
checkpoint_path = Path("training_logs/NicolasWork-lidc-patches-lv3/model_checkpoints//checkpoint_epoch-20.pth")
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

num_samples= [1,4,8,16][1]

for (patient_id, image_id) in image_keys:
    img, segs = test_dataset.get_img_segs(patient_id, image_id)

    num_mode = len(segs)
    print(num_mode)
    fig, axs = plt.subplots(nrows=2, ncols=num_mode+1, layout="tight")

    axs[0,0].imshow(img,cmap='gray')

    for j in range(num_mode):
        axs[0,j+1].imshow(segs[j],cmap='gray')

    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(dim=0)
    img = img.unsqueeze(dim=0)

    img = img.to(DEVICE)
    net(img, None, training=False)  # Run net (this initializes the unet features and the latent space)

    preds = []
    for j in range(5):
        prob = net.sample(testing=True)  # samples a segmentation using the unet features + the latent space
        # min=prob.min()
        # max=prob.max()
        # prob = (prob-min)/(max-min)
        pred = prob > 0
        
        preds.append(pred)
        axs[1,j].imshow(pred.detach().cpu().numpy()[0],cmap='gray')

    break

plt.show()

