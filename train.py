from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from punet import ProbabilisticUnet
from utils import l2_regularisation

import wandb

WANDB_LOGGING = False

def train_punet(train_dataset, batch_size_train=1,
                val_dataset=None, batch_size_val=1,
                num_classes=1, num_channels_unet=[32, 64, 128, 256],
                latent_dim=2, no_convs_fcomb=4, beta=10.0,
                val_freq=10, learning_rate=1e-4, reg_weight=1e-5,
                epochs=5, train_id=None, device="cpu", num_workers=1):
    
    saving_path = Path(f"training_logs/{train_id}")
    if saving_path.exists():
        raise Exception("Path already exists, change train_id")
    else:
        saving_path.mkdir(parents=True)
        saving_path.joinpath("model_checkpoints").mkdir()

        if WANDB_LOGGING:
            wandb.init(
            # set the wandb project where this run will be logged
                entity="punet-replication",
                project="pytorch-replication",
                name=f"{device}-{train_id}",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": learning_rate,
                "regularization-weight": reg_weight,
                "architecture": "PUNet",
                "dataset": "LIDC",
                "epochs": epochs,
                "device": device,
                "num_classes":num_classes,
                "num_channels_unet":num_channels_unet,
                "latent_dim":latent_dim,
                "no_convs_fcomb":no_convs_fcomb,
                "beta": beta
                }
            )

        device = torch.device(device)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, prefetch_factor=8, pin_memory=True, persistent_workers=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

        net = ProbabilisticUnet(num_input_channels=1,
                                num_classes=num_classes,
                                num_filters=num_channels_unet,
                                latent_dim=latent_dim,
                                no_convs_fcomb=no_convs_fcomb,
                                beta=beta,
                                device=device)
        net.to(device)

        # Los is defined in each iteration
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0)

        for epoch in range(1, epochs + 1):

            # Train
            epoch_losses = []    
            running_loss = 0
            net.train()

            with tqdm(total=int(np.ceil(len(train_dataset)/batch_size_train)), desc=f"Epoch {epoch}/{epochs} (loss: {running_loss})", unit="patch") as pbar:
                for batch_i, (metadata, img, seg) in enumerate(train_dataloader): 
                    img = img.to(device)
                    seg = seg.to(device)

                    seg = torch.unsqueeze(seg,1)
                    seg = seg.float()

                    net.forward(img, seg, training=True)

                    elbo = net.elbo(seg)
                    reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                    loss_val = -elbo + reg_weight * reg_loss

                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()

                    epoch_losses.append(loss_val.item())
                    running_loss = np.array(epoch_losses).mean()

                    if WANDB_LOGGING:
                        wandb.log({"epoch": epoch, "acc":net.elbo, "loss": net.reconstruction_loss})

                    pbar.update(1)
                    pbar.set_description(f"Epoch {epoch}/{epochs} (loss: {running_loss:.2f})")


            # Validate
            if epoch % val_freq == 0:   
                print(f"Saving model checkpoint of epoch {epoch:02d}")       
                checkpoint_path = saving_path.joinpath("model_checkpoints")
                state_dict = net.state_dict()
                torch.save(state_dict, str(checkpoint_path.joinpath(f"checkpoint_epoch-{epoch:02d}.pth")))