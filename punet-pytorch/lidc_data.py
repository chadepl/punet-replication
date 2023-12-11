from pathlib import Path
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as FT

# Dataset as described in HPUnet paper
class LIDCCrops(Dataset):
    def __init__(self, data_home=".", split="train", transform=None):
        super().__init__()
        self.data_home = Path(data_home)
        if not self.data_home.exists():
            raise FileNotFoundError(self.data_home)
        self.split_path = self.data_home.joinpath(split)
        if not self.split_path.exists():
            raise FileNotFoundError(self.split_path)
        
        self.idx = []
        self.patient_img_lab_dict = dict()
        for patient_path in self.split_path.joinpath("gt").glob("*"):
            patient_id = patient_path.stem
            for image_path in patient_path.glob("*.png"):
                image_lab_info = image_path.stem
                image_id = "_".join(image_lab_info.split("_")[:-1])
                lab_id = image_lab_info.split("_")[-1]

                if patient_id not in self.patient_img_lab_dict:
                    self.patient_img_lab_dict[patient_id] = dict()
                if image_id not in self.patient_img_lab_dict[patient_id]:
                    self.patient_img_lab_dict[patient_id][image_id] = []
                self.patient_img_lab_dict[patient_id][image_id].append(image_path)

                self.idx.append(dict(patient_id=patient_id, image_id=image_id, lab_id=lab_id))

        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        # Each image has multiple evaluations. Therefore, dataset is as big as num_images * num_evaluators
                
        metadata = self.idx[idx]

        patient_id = metadata["patient_id"]
        image_id = metadata["image_id"]
        lab_id = metadata["lab_id"]        
        
        img = imread(self.split_path.joinpath(f"images/{patient_id}/{image_id}.png"))
        img = (img - img.min())/(img.max() - img.min())
        seg = imread(self.split_path.joinpath(f"gt/{patient_id}/{image_id}_{lab_id}.png"))
        seg[seg>0] = 1
        
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(dim=0)  # we add the channel dim
        seg = torch.from_numpy(seg.astype(np.uint8)).long()

        if self.transform:
            input_size = list(img.size()[-2:])
            seg = seg.unsqueeze(dim=0)
            if "rand_elastic" in self.transform:
                alpha=self.transform["rand_elastic"].get("alpha", [10, 10])
                sigma=self.transform["rand_elastic"].get("sigma", [5, 5])
                elastic_params = v2.ElasticTransform.get_params(alpha=alpha, sigma=sigma, size=input_size)

                img = FT.elastic_transform(img, elastic_params, interpolation=v2.InterpolationMode.BILINEAR)
                seg = FT.elastic_transform(seg, elastic_params, interpolation=v2.InterpolationMode.NEAREST)

            if "rand_affine" in self.transform:
                degrees=self.transform["rand_affine"].get("degrees", [-10, 10])
                translate=self.transform["rand_affine"].get("translate", None)
                scale_ranges=self.transform["rand_affine"].get("scale_ranges", [0.8, 1.2])
                shears=self.transform["rand_affine"].get("shears", [0, 5, 0, 5])
                affine_params = v2.RandomAffine.get_params(degrees=degrees, 
                                                        translate=translate,
                                                        scale_ranges=scale_ranges,
                                                        shears=shears,
                                                        img_size=input_size)

                img = FT.affine(img, *affine_params, interpolation=v2.InterpolationMode.BILINEAR)
                seg = FT.affine(seg, *affine_params, interpolation=v2.InterpolationMode.NEAREST)

            if "rand_crop" in self.transform:
                output_size=self.transform["rand_crop"].get("output_size", input_size)
                crop_params = v2.RandomCrop.get_params(img, output_size=output_size)

                img = FT.crop(img, *crop_params)
                seg = FT.crop(seg, *crop_params)

            if "resize" in self.transform:
                output_size=self.transform["resize"].get("output_size", input_size)
                img = FT.resize(img, size=output_size, interpolation=v2.InterpolationMode.BILINEAR)
                seg = FT.resize(seg, size=output_size, interpolation=v2.InterpolationMode.NEAREST)

            seg = seg.squeeze(dim=0)

        return (metadata, img, seg)
    
    def get_patient_img_candidates(self, patient_id, image_id):
        return self.patient_img_lab_dict[patient_id][image_id]
    
    def read_img(self, img_path, is_mask=True):
        img = imread(img_path)
        img[img>0] = 1
        img = torch.from_numpy(img.astype(np.uint8)).long()
        return img
    
    def get_patient_image_ids(self):
        out = []
        for patient_id in self.patient_img_lab_dict.keys():
            for image_id in self.patient_img_lab_dict[patient_id].keys():
                out.append((patient_id, image_id))
        return out
    

if __name__ == "__main__":
    dataset = LIDCCrops(".", "train")
    print(len(dataset))

    metadata, img, seg = dataset[100]

    print(metadata)
    print(img.shape)
    print(seg.shape)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(img.numpy().squeeze())
    viewer.add_labels(seg.numpy())
    napari.run()