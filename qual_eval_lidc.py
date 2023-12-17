#######################
# Qualitative Testing #
#######################

from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as FT
import matplotlib.pyplot as plt
from punet import ProbabilisticUnet
from lidc_data import LIDCCrops
from punet_config import NUM_CLASSES, NUM_CHANNELS, LATENT_DIM, NUM_CONVS_FCOMB, BETA, DEVICE

rng = np.random.default_rng(seed=42)

NUM_IMGS = 5
NUM_SAMPLES = 20
PROB_ISOVAL = 0.8

# Load model
checkpoint_path = Path("training_logs/NicolasWork-lidc-patches-lv3/model_checkpoints//checkpoint_epoch-100.pth")
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
test_dataset = LIDCCrops(data_home="data/lidc_crops", split="test", transform=dict(resize=dict(output_size=(128, 128))))
image_keys = test_dataset.get_patient_image_ids()
random_pos = rng.choice(np.arange(len(image_keys)), NUM_IMGS, replace=False)
random_ids = [image_keys[rp] for rp in random_pos]

imgs = []
segs_gts = []
segs_preds = []
segs_probs = []
for patient_id, image_id in random_ids:
    img, gts = test_dataset.get_img_segs(patient_id, image_id)        

    torch_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(dim=0)  # add channel dimension
    torch_img = FT.resize(torch_img, size=(128, 128), interpolation=v2.InterpolationMode.BILINEAR)
    torch_img = torch_img.unsqueeze(dim=0)  # add batch dimension

    # output_size=self.transform["resize"].get("output_size", input_size)
    
    # seg = FT.resize(seg, size=output_size, interpolation=v2.InterpolationMode.NEAREST)
    
    probs = []
    preds = []
    net(torch_img, None, training=False)  # Run net (this initializes the unet features and the latent space)
    for nsample in range(NUM_SAMPLES):        
        prob = net.sample(testing=True)  # samples a segmentation using the unet features + the latent space
        prob = torch.sigmoid(prob)
        pred = (prob > 0.5).float() # get segmentation result from probs

        prob = FT.resize(prob, size=(180, 180), interpolation=v2.InterpolationMode.BILINEAR)
        pred = FT.resize(pred, size=(180, 180), interpolation=v2.InterpolationMode.NEAREST)

        probs.append(prob.detach().cpu().numpy().squeeze())
        preds.append(pred.detach().cpu().numpy().squeeze())

    imgs.append(img)
    segs_gts.append(gts)
    segs_probs.append(probs)
    segs_preds.append(preds)




############
# PLOTTING #
############

if True:  # plot 1: 1 image, 20 predictions
    IMG_I = 4

    fig, axs = plt.subplots(ncols= 1 + 4, nrows=5, layout="tight", figsize=(8, 6))

    axs[0, 0].imshow(imgs[IMG_I], cmap="gray")
    axs[0, 0].set_ylabel("Image")

    for j in range(4):
        axs[0, j + 1].imshow(segs_gts[IMG_I][j], cmap="gray")
        axs[0, j + 1].set_ylabel(f"GT {j + 1}")

    for i in range(4):
        for j in range(5):
            axs[i + 1, j].imshow(segs_preds[IMG_I][i * 5 + j], cmap="gray")
            axs[i + 1, j].set_ylabel(f"Pred {i * 5 + j}")

    for ax in axs.flatten():
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.show()
    fig.savefig("qual_eval_lidc.png", dpi=300)


if False:  # plot 2: NUM_IMAGES x NUM_GTS + NUM-SAMPLES 
    NUM_SAMPLES = 4

    # plot
    fig, axs = plt.subplots(nrows=NUM_IMGS, ncols=1 + 4 + NUM_SAMPLES, layout="tight", figsize=(8, 4))  # 1 img, 4 gts, NUM_SAMPLES samples

    # - plot imgs
    for img_i in range(NUM_IMGS):
        axs[img_i, 0].imshow(imgs[img_i], cmap="gray")
        axs[img_i, 0].set_ylabel(f"Image {img_i + 1}")

    # - plot gt
    for img_i in range(NUM_IMGS):
        for j, seg_gt in enumerate(segs_gts[img_i]):
            if img_i == 0:
                axs[0, 1 + j].set_title(f"GT {j + 1}")        
            axs[img_i, 1 + j].imshow(seg_gt, cmap="gray")        

    # - plot samples
    for img_i in range(NUM_IMGS):
        for j, seg_pred in enumerate(segs_probs[img_i]):
            if img_i == 0:
                axs[0, 1 + 4 + j].set_title(f"Pred {j + 1}")
            axs[img_i, 1 + 4 + j].imshow(seg_pred, cmap="gray")

    for ax in axs.flatten():
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.show()
    fig.savefig("qual_eval_lidc_appendix.png", dpi=300)