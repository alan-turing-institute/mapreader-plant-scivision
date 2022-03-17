#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image
import requests
import timm
from torchvision import transforms

from mapreader import classifier
from mapreader import loader
from mapreader import load_patches
from mapreader import patchTorchDataset

class MapReader_model:
    
    def __init__(self, 
                 model_path: str="https://github.com/alan-turing-institute/mapreader-plant-scivision/raw/main/mapreader-plant-scivision/model_checkpoint_7.pkl", 
                 checkpoint_path: str="https://github.com/alan-turing-institute/mapreader-plant-scivision/raw/main/mapreader-plant-scivision/checkpoint_7.pkl", 
                 device: str="default", 
                 tmp_model_dir: str="./mr_tmp",
                 tmp_slice_dir: str="./mr_tmp/slice",
                 batch_size: int=64,
                 resize2: int=224,
                 infer_name: str="infer_test"
                ):
        """
        Initialize the MapReader model.
        """

        # ---- SETUP
        self.tmp_model_dir = tmp_model_dir
        self.tmp_slice_dir = tmp_slice_dir 
        os.makedirs(self.tmp_model_dir, exist_ok=True)
        os.makedirs(self.tmp_slice_dir, exist_ok=True)

        self.batch_size = batch_size
        self._resize2 = resize2
        self.infer_name = infer_name

        # ---- DOWNLOAD MODEL
        self.download_file(checkpoint_path, path2save=os.path.join(self.tmp_model_dir, "checkpoint.pkl"))
        self.download_file(model_path, path2save=os.path.join(self.tmp_model_dir, "model_checkpoint.pkl"))
        
        # ---- CLASSIFIER
        myclassifier = classifier(device=device)
        myclassifier.load(os.path.join(self.tmp_model_dir, "checkpoint.pkl"))
        self.pretrained_model = myclassifier
        
        # ---- PREPROCESSOR
        self.data_transforms = self.preprocess()
    
    def download_file(self, 
                      url: str="https://github.com/alan-turing-institute/mapreader-plant-scivision/raw/main/mapreader-plant-scivision/checkpoint_7.pkl",
                      path2save: str="./mr_tmp/scivision_model.pkl",
                      chunk_size: int=1024
                      ):
        """Download a file from url to path2save."""

        os.makedirs(os.path.dirname(path2save), exist_ok=True)

        r = requests.get(url, stream=True)
        with open(path2save, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size): 
                if chunk:
                    f.write(chunk)
        f.close()

    def load_image(self, 
                   input_array: str,
                   slice_size: int=100,
                   save_image_path: str="./mr_tmp/orig_image.png",
                   **slice_kwds
                   ):
        
        # ---- Save image
        self.save_image_path = save_image_path
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)
        im = Image.fromarray(input_array)
        im.save(self.save_image_path)

        self.image_name = os.path.basename(os.path.abspath(self.save_image_path))
        
        myimgs = loader(self.save_image_path)
        
        # `method` can also be set to meters
        myimgs.sliceAll(path_save=self.tmp_slice_dir, 
                        slice_size=slice_size, # in pixels
                        **slice_kwds)
        
    def preprocess(self, 
                   normalize_mean: list=[0.485, 0.456, 0.406],
                   normalize_std: list=[0.229, 0.224, 0.225]
                   ):

        # ---- PREPROCESS
        data_transforms = transforms.Compose(
            [transforms.Resize((self._resize2, self._resize2)),
             transforms.ToTensor(),
             transforms.Normalize(normalize_mean, normalize_std)])
        
        return data_transforms
    
    def inference(self, plot_output: bool=False):

        # Load patches
        myimgs = load_patches(os.path.join(self.tmp_slice_dir, f"*{self.image_name}*PNG"), 
                              parent_paths=self.save_image_path)
        
        self.myimgs = myimgs
        
        # Convert to torch dataset
        imgs_pd, patches_pd = self.myimgs.convertImages(fmt="dataframe")
        patches_pd = patches_pd.reset_index()
        patches_pd.rename(columns={"index": "image_id"}, 
                          inplace=True)
        patches2infer = patches_pd[["image_path"]]

        patches2infer_dataset = patchTorchDataset(patches2infer, 
                                                  transform=self.data_transforms)


        # Create dataloader using batch_size, set num_workers to 0
        self.pretrained_model.add2dataloader(patches2infer_dataset, 
                                             set_name=self.infer_name, 
                                             batch_size=self.batch_size, 
                                             shuffle=False, 
                                             num_workers=0)

        # ---- INFERENCE
        self.pretrained_model.inference(set_name=self.infer_name)
        
        patches2infer['pred'] = self.pretrained_model.pred_label
        patches2infer['conf'] = np.max(np.array(self.pretrained_model.pred_conf), axis=1)
        
        patches2infer["name"] = patches2infer["image_path"].apply(lambda x: os.path.basename(x))
        self.myimgs.add_metadata(patches2infer, tree_level="child")
        
        if plot_output:
            self.show_output()
        
        return patches2infer
    
    def sample_results(self, 
                       num_samples: int=3, 
                       class_index: int=0,
                       min_conf: float=50,
                       max_conf: float=110
                      ):
    
        self.pretrained_model.inference_sample_results(num_samples=num_samples, 
                                                       class_index=class_index, 
                                                       set_name=self.infer_name,
                                                       min_conf=min_conf,
                                                       max_conf=max_conf)
    
    def show_output(self):
    
        self.myimgs.show_par(self.image_name, 
                    value="pred",
                    border=None,
                    plot_parent=True,
                    vmin=0, vmax=len(self.pretrained_model.class_names)-1,
                    figsize=(20, 20),
                    plot_histogram=False,
                    alpha=0.5, 
                    colorbar="inferno")
    
    def predict(self, 
                path2images: str,
                slice_size: int=100,
                plot_output: bool=True,
                **slice_kwds
                ):
        
        self.load_image(path2images, slice_size=slice_size, **slice_kwds)
        
        return self.inference(plot_output=plot_output)
