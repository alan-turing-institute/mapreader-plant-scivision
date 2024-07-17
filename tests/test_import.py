import pytest


def test_workflow():
    from scivision import default_catalog, load_pretrained_model, load_dataset

    models_catalog = default_catalog.models.to_dataframe()
    stp_repo = models_catalog[models_catalog.name == "mapreader-plant"].url.item()
    
    model = load_pretrained_model(stp_repo, allow_install=True, model_selection= "green_and_plant")
    
    data_config = load_dataset('https://github.com/alan-turing-institute/mapreader-plant-scivision')
    
    # --- plant_single dataset 
    plant_single = data_config.plant_single().to_dask()

    model.predict(plant_single[0], slice_size=100)
    
    model.predict(plant_single[1], slice_size=100)

    # --- plant_flower dataset 
    plant_flower = data_config.plant_flower().to_dask()

#     model.predict(plant_flower[0], slice_size=200)
    
#     model.predict(plant_flower[1], slice_size=200)
