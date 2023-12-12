# Probabilistic UNet replication

This repository replicates some of the results of the Probabilistic UNet paper [1]. 
It also includes the code we used in the experiments. 

Link to google docs: https://docs.google.com/document/d/1P1Fcg6PXJnvt2uWD0YAyhzgBoMVvOCjDkT1a38h0EVc/edit 

## Setup

1. `conda create --name=punet-pytorch python=3.9`
2. `conda activate punet-pytorch`
3. `pip install torch torchvision torchaudio`
4. `pip install tqdm matplotlib scikit-image`

To train the network:
- `cd punet-pytorch`
- `python train_punet_lidc.py`

## References

[1] Kohl, S., Romera-Paredes, B., Meyer, C., De Fauw, J., Ledsam, J. R., Maier-Hein, K., ... & Ronneberger, O. (2018). A probabilistic u-net for segmentation of ambiguous images. Advances in neural information processing systems, 31.
