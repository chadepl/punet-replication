
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from tqdm import tqdm

from lidc_data import LIDCCrops

from train import train_punet


############
# Training #
############

if True:  # we want to train a network or not
    BATCH_SIZE = 8
    DEVICE = ["cpu", "mps", "cuda"][1]
    EPOCHS = int(240000 * 32 * (1 / 8882))
    NUM_CLASSES = 1
    NUM_CHANNELS = [32, 64, 128, 256]  # original in paper   

    LATENT_DIM=2  # original in paper   
    NUM_CONVS_FCOMB=4 # original in paper   
    BETA=10.0 # original in paper   

    transforms = dict(
        rand_elastic=dict(
            alpha=(0., 800.),  # as in the PUnet repo for cityscapes
            sigma=(25., 35.)  # as in the PUnet repo for cityscapes
            ),
        rand_affine=dict(
            degrees=(np.rad2deg(-np.pi/8), np.rad2deg(np.pi/8)),  # as in the PUnet repo for cityscapes
            translate=None,  # as in the PUnet repo for cityscapes (not used)
            scale_ranges=(0.8, 1.2),  # as in the PUnet repo for cityscapes
            shears=None  # as in the PUnet repo for cityscapes (not used)
            ),
        rand_crop=dict(
            output_size=(128, 128)   # as in the PUnet repo for cityscapes
            )
    )
    train_dataset = LIDCCrops(data_home="../data/lidc_crops", split="train", transform=transforms)
    val_dataset = LIDCCrops(data_home="../data/lidc_crops", split="val", transform=None)
    
    train_punet(train_dataset=train_dataset,
                batch_size_train=BATCH_SIZE,
                val_dataset=val_dataset,
                batch_size_val=BATCH_SIZE,
                epochs=EPOCHS,
                num_classes=NUM_CLASSES,
                num_channels_unet=NUM_CHANNELS,
                latent_dim=LATENT_DIM,
                no_convs_fcomb=NUM_CONVS_FCOMB,
                beta=BETA,
                train_id=f"{DEVICE}-lidc-patches-lv3",
                device=DEVICE)


###########
# Testing #
###########

if False:  # we want to test the network

    # Load checkpoint and visualize results

    # visualization
    checkpoint_path = Path("../../models/2d/unet/model_checkpoints/unet_lidc-patches-lv3/checkpoint_epoch-05.pth")
    state_dict = torch.load(str(checkpoint_path))
    test_net = UNet(num_levels=3)
    test_net.load_state_dict(state_dict=state_dict)

    num_elements = 5
    test_dataset = LIDCCrops(data_home="../../data/real/lidc_crops", split="test")
    metadatas, imgs, segs = zip(*[test_dataset[i] for i in np.random.choice(np.arange(len(test_dataset)), num_elements, replace=False)])
    
    imgs = [img.unsqueeze(dim=0) for img in imgs]    
    imgs = torch.cat(imgs, dim=0)

    logits = test_net(imgs)
    probs = nn.Softmax(dim=1)(logits)
    pred = torch.argmax(probs, dim=1)

    vis_img = imgs[:, 0, :, :].numpy()
    vis_seg = np.concatenate([seg[np.newaxis] for seg in segs], axis=0)
    vis_probs0 = probs[:, 0, :, :].detach().numpy()
    vis_probs1 = probs[:, 1, :, :].detach().numpy()
    vis_pred = pred.detach().numpy()

    import napari

    viewer = napari.Viewer()
    viewer.add_image(vis_img)
    viewer.add_labels(vis_seg)
    # viewer.add_image(vis_probs0)
    # viewer.add_image(vis_probs1)
    viewer.add_labels(vis_pred)
    napari.run()
