# Classification of plant patches/images using MapReader

Example data for applying MapReader to classification of plant patches and running this model with the Scivision tool

## Installation

Make sure you have [git-lfs](https://git-lfs.com/) installed.

Create a conda environment:

```bash
conda create -n plant_py38 python=3.8
conda activate plant_py38
```

Install scivision:

```bash
pip install scivision
pip install jupyter
```
Windows users may need to run the following:

```bash
# install rasterio and fiona manually
conda install -c conda-forge rasterio=1.2.10
conda install -c conda-forge fiona=1.8.20

# install git
conda install git
```

## Running an example notebook

Follow the instructions at: https://github.com/scivision-gallery/plant-phenotyping-classification#how-to-run
