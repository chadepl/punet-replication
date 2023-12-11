
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
    train_dataset = LIDCCrops(data_home="../data/lidc_crops", split="train")
    device = ["cpu", "mps", "cuda"][1]
    train_punet(train_dataset=train_dataset, epochs=5, num_classes=2, num_levels_unet=3, train_id=f"{device}-lidc-patches-lv3", device=device)


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
