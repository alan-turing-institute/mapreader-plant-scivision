# Classification of plant patches/images using MapReader

Example data for applying MapReader to classification of plant patches and running this model with the Scivision tool

## Installation

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

Clone scivision repo as notebooks are stored on the repo:

```
git clone https://github.com/alan-turing-institute/scivision.git
```

Go to the scivision repo and open jupyter notebook:

```bash
cd scivision
jupyter notebook
```

Run `mapreader_plant_scivision.ipynb`.
