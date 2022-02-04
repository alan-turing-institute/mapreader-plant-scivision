from mapreader import classifier
from mapreader import loader
from torchvision import transforms
from mapreader import load_patches
from mapreader import patchTorchDataset
import numpy as np
import os

class MapReader_model:
    
    def __init__(self, 
                 model_path: str="./checkpoint_15.pkl", 
                 device: str="default", 
                 tmp_dir: str="./tmp_slice",
                 batch_size: int=64,
                ):
        
        self.tmp_dir = tmp_dir
        self.batch_size = batch_size
        self._resize2 = 224
        
        # ---- CLASSIFIER
        # e.g., model_path = "./models_tutorial/checkpoint_1.pkl"
        myclassifier = classifier(device=device)
        myclassifier.load(model_path)

        self.pretrained_model = myclassifier
        
        # ---- PREPROCESSOR
        self.data_transforms = self.preprocess()
        

    
    def load_images(self, 
                    path2images: str,
                    slice_size: int=100,
                    **slice_kwds
                   ):
        
        self.path2images = path2images
        self.image_name = os.path.basename(os.path.abspath(self.path2images))
        
        myimgs = loader(self.path2images)
        
        # `method` can also be set to meters
        myimgs.sliceAll(path_save=self.tmp_dir, 
                        slice_size=slice_size, # in pixels
                        **slice_kwds)
        
    def preprocess(self):
        
        # mean and standard deviations of pixel intensities in 
        # all the patches in 6", second edition maps
        normalize_mean = 1 - np.array([0.82860442, 0.82515008, 0.77019864])
        normalize_std = 1 - np.array([0.1025585, 0.10527616, 0.10039222])

        data_transforms = {
            'val': transforms.Compose(
                [transforms.Resize((self._resize2, self._resize2)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
                ]),
        }
        
        return data_transforms["val"]
    
    def inference(self):

        # ---- CREATE DATASET
        myimgs = load_patches(os.path.join(self.tmp_dir, f"*{self.image_name}*PNG"), 
                              parent_paths=self.path2images)
        
        self.myimgs = myimgs
        
        imgs_pd, patches_pd = self.myimgs.convertImages(fmt="dataframe")
        patches_pd = patches_pd.reset_index()
        patches_pd.rename(columns={"index": "image_id"}, 
                          inplace=True)
        patches2infer = patches_pd[["image_path"]]

        patches2infer_dataset = patchTorchDataset(patches2infer, 
                                                  transform=self.data_transforms)


        self.pretrained_model.add2dataloader(patches2infer_dataset, 
                                             set_name="infer_test", 
                                             batch_size=self.batch_size, 
                                             shuffle=False, 
                                             num_workers=0)

        # ---- INFERENCE
        self.pretrained_model.inference(set_name="infer_test")
        
        patches2infer['pred'] = self.pretrained_model.pred_label
        patches2infer['conf'] = np.max(np.array(self.pretrained_model.pred_conf), axis=1)
        
        patches2infer["name"] = patches2infer["image_path"].apply(lambda x: os.path.basename(x))
        self.myimgs.add_metadata(patches2infer, tree_level="child")
        
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
                                                       set_name="infer_test",
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
                **slice_kwds
                ):
        
        self.load_images(path2images, slice_size=slice_size, **slice_kwds)
        
        return self.inference()