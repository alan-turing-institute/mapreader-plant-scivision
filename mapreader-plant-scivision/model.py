#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image
import requests
import timm
import time
from torchvision import transforms

from mapreader import classifier
from mapreader import loader
from mapreader import load_patches
from mapreader import patchTorchDataset

class MapReader_model:
    name: str
    
    def __init__(self, 
                 model_path: dict = None, 
                 checkpoint_path: dict = None,
                 device: str="default", 
                 tmp_model_dir: [str, None]=None,
                 tmp_slice_dir: [str, None]=None,
                 batch_size: int=64,
                 resize2: int=224,
                 infer_name: str="infer_test"
                ):
        """
        Initialize the MapReader model.
        """

        # ---- SETUP
        if tmp_model_dir is None:
            tmp_model_dir = f"./mr_tmp_{int(time.time())}"
            tmp_slice_dir = f"./{tmp_model_dir}/slice"
            
        self.tmp_model_dir = tmp_model_dir
        self.tmp_slice_dir = tmp_slice_dir 
        os.makedirs(self.tmp_model_dir, exist_ok=True)
        os.makedirs(self.tmp_slice_dir, exist_ok=True)

        self.batch_size = batch_size
        self._resize2 = resize2
        self.infer_name = infer_name

        if self.name = 'branch':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/branch_model_checkpoint_21.pkl", known_hash = 'md5:a0f5596beab2330ee54cd144f0c167cb'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/branch_checkpoint_21.pkl", known_hash = 'md5:4a05ff6a690bfd3472bfff6c98f85b1e')
        elif self.name = 'bud':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/bud_model_checkpoint_20.pkl", known_hash = 'md5:4cae0730ce3126fc2289489b5d6bd223'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/bud_checkpoint_20.pkl", known_hash = 'md5:7516dedcdf46451e9c77b01851d3115d')
        elif self.name = 'five_label':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/five_model_checkpoint_6.pkl", known_hash = 'md5:27d350ddb9606774531aa76ca2a7f71f'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/five_checkpoint_6.pkl", known_hash = 'md5:b3296b543b83e43d48ccb788725a3f49') 
        elif self.name = 'flower':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/flower_model_checkpoint_35.pkl", known_hash = 'md5:e24fdee9f98acf1db14cf6f8fd41dbd1'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/flower_checkpoint_35.pkl", known_hash = 'md5:3920af2cdbab45eedf40db91b8e401b2') 
        elif self.name = 'green_and_plant':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/green_model_checkpoint_10.pkl", known_hash = 'md5:822e7d1416200ffcd803aeb8e393f7aa'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/green_checkpoint_10.pkl", known_hash = 'md5:9a3638615ef8b907fc70c0919a926538')
        elif self.name = 'leaf':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/leaf_model_checkpoint_33.pkl", known_hash = 'md5:ceee6ce722b92a976f2077a5784f4811'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/leaf_checkpoint_33.pkl", known_hash = 'md5:ba05c0442a2fca6459970a79cc6a1898')
        elif self.name = 'plant_binary':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/plant_model_checkpoint_30.pkl", known_hash = 'md5:81528eae238d5fc984522a707c4fb83b'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/plant_checkpoint_30.pkl", known_hash = 'md5:c6aa53175fd1b21c1f55203649b7f53b')
        elif self.name = 'pod':
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/pod_model_checkpoint_29.pkl", known_hash = 'md5:88fd322da2270fe0d579ab1ad1df23e8'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/pod_checkpoint_29.pkl", known_hash = 'md5:9826caf4d13b6fd88643eb3e190728f6')
        else:
            model_path = dict(url="doi:10.5281/zenodo.12532188/files/six_model_checkpoint_10.pkl", known_hash = 'md5:99f9d0898a4dd5a1452fde615dccc361'), 
            checkpoint_path = dict(url="doi:10.5281/zenodo.12532188/files/six_checkpoint_10.pkl", known_hash = 'md5:4ad5676c2010af8e8721f89b38ae5047')

        # ---- DOWNLOAD MODEL
        #self.download_file(checkpoint_path, path2save=os.path.join(self.tmp_model_dir, "checkpoint.pkl"))
        #self.download_file(model_path, path2save=os.path.join(self.tmp_model_dir, "model_checkpoint.pkl"))
        self.model_path = pooch.retrieve(url=model_path['url'], known_hash=model_path['known_hash'])
        self.checkpoint_path = pooch.retrieve(url=checkpoint_path['url'], known_hash=checkpoint_path['known_hash'])

        # ---- CLASSIFIER
        myclassifier = classifier(device=device)
        myclassifier.load(os.path.join(self.tmp_model_dir, "checkpoint.pkl"))
        self.pretrained_model = myclassifier
        
        # ---- PREPROCESSOR
        self.data_transforms = self.preprocess()
    
    #def download_file(self, 
    #                  url: str="https://github.com/alan-turing-institute/mapreader-plant-scivision/raw/main/mapreader-plant-scivision/checkpoint_10.pkl",
    #                  path2save: str="./mr_tmp/scivision_model.pkl",
    #                  chunk_size: int=1024
    #                  ):
    #    """Download a file from url to path2save."""

    def download_file(self, 
                      url: str=checkpoint_path['url'],
                      path2save: str="./mr_tmp/scivision_model.pkl",
                      chunk_size: int=1024
                      ):
        """Download a file from url to path2save."""    
        print(f"[INFO] Download model from: {url}")

        os.makedirs(os.path.dirname(path2save), exist_ok=True)

        r = requests.get(url, stream=True)
        with open(path2save, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size): 
                if chunk:
                    f.write(chunk)
        f.close()
        
        print(f"[INFO] Save model         : {path2save}")

    def load_image(self, 
                   input_array: str,
                   slice_size: int=100,
                   save_image_path: [None, str]=None,
                   **slice_kwds
                   ):
        
        # ---- Save image
        if save_image_path is None:
            save_image_path = os.path.join(self.tmp_model_dir, "orig_image.png")
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
             transforms.ToTensor()
             #transforms.Normalize(normalize_mean, normalize_std)
            ])
        
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
