# Soil Moisture Prediction

This system fetches, processes, and predicts soil moisture using real-time data.

## Description and background

The validators query the contributors during specific execution windows defined in the get_request_windows method. These windows are:
- 2:00 - 2:30
- 10:00 - 10:30
- 14:00 - 14:30
- 20:00 - 20:30
The contributors give prediction for the next window.

## Aim of the task

Maximize the final score, given any day
```python
final_score = await scoring_mechanism.score(pred_data)
```

#### How to start?

- Understand the flow, datasets, and what the base model does
- The data pulling/evaluation code may have bugs. Can you spot any?
- Data exception handling? For example, when receiving 404 from APIs.
- Look at these repos:
    - https://github.com/fkwai/geolearn
    - https://github.com/mhpi/hydroDL
    - https://github.com/leelew/CLSTM
    - https://github.com/ljz1228/CLM-LSTM-soil-moisture-prediction (repo removed, uploaded to folder `CLM-LSTM-soil-moisture-prediction`)
- Read this article on different models: https://nickel5.substack.com/p/global-soil-moisture-models

#### **Input Format**

The input to `run_inference` is a dictionary containing:

- `sentinel_ndvi`: A numpy array of shape [3, H, W] representing Sentinel-2 bands B8, B4, and NDVI.
- `elevation`: A numpy array of shape [1, H, W] representing elevation data.
- `era5`: A numpy array of shape [17, H, W] containing weather variables.

#### **Output Format**

The output must be a dictionary with:

- `surface`: A nested list (11x11) of floats between 0-1 representing surface soil moisture predictions.
- `rootzone`: A nested list (11x11) of floats between 0-1 representing root zone soil moisture predictions.

#### **Weather Data Details**

IFS weather variables (in order):
- t2m: Surface air temperature (2m height) (Kelvin)
- tp: Total precipitation (m/day)
- ssrd: Surface solar radiation downwards (Joules/m²)
- st: Soil temperature at surface (Kelvin)
- stl2: Soil temperature at 2m depth (Kelvin)
- stl3: Soil temperature at 3m depth (Kelvin)
- sp: Surface pressure (Pascals)
- d2m: Dewpoint temperature (Kelvin)
- u10: Wind components at 10m (m/s)
- v10: Wind components at 10m (m/s)
- ro: Total runoff (m/day)
- msl: Mean sea level pressure (Pascals)
- et0: Reference evapotranspiration (mm/day)
- bare_soil_evap: Bare soil evaporation (mm/day)
- svp: Saturated vapor pressure (kPa)
- avp: Actual vapor pressure (kPa)
- r_n: Net radiation (MJ/m²/day) 

**Note**:
Evapotranspiration are variables computed using the Penman-Monteith equation (FAO-56 compliant). 
see soil_apis.py for more information on the data processing, transformations, and scaling.

## NASA EarthData
1. Create an account at https://urs.earthdata.nasa.gov/
2. Accept the necessary EULAs for the following collections:
    - GESDISC Test Data Archive 
    - OB.DAAC Data Access 
    - Sentinel EULA

3. Generate an API token and save it in the .env file
```
EARTHDATA_USERNAME=<YOUR_EARTHDATA_USERNAME> 
EARTHDATA_PASSWORD=<YOUR_EARTHDATA_PASSWORD>
EARTHDATA_API_KEY=<YOUR_EARTHDATA_API_KEY> # earthdata api key for downloading data from NASA
```

## Installation

1. Clone the repository
2. Install gdal
Via `apt-get`
```bash
sudo apt-get install -y gdal-bin
sudo apt-get install -y libgdal-dev
sudo apt-get install -y python3-gdal
```
Via `brew`
```bash
brew install gdal
```
3. Install pip requirements
```bash
pip install -r requirements.txt
```

## Usage

Run the main prediction service:
```bash
python soil.py
```

## Components

- `soilmoisture.py`: Main script handling data fetching, processing, and comparison with base model
- `basemodel.py`: Base model implementation and model loading logic
- `custommodel.py`: Your custom model

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- Internet connection for data fetching

## Data Source

| Name        | Description                                           | Spatial Resolution | Temporal Resolution | Link                                                                                                     |
|-------------|-------------------------------------------------------|--------------------|---------------------|----------------------------------------------------------------------------------------------------------|
| SMAP L4     | Volumetric water content, soil moisture               | 9km               | 3 hrs               | [Link](https://nsidc.org/data/spl4smgp/versions/7)                                                      |
| Sentinel-2  | Red and near-infrared bands                           | 50m               | 5 days              | [Link](https://hls.gsfc.nasa.gov/)                                                                      |
| SRTM        | Elevation data                                        | 500m              | N/A                 | [Link](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm) |
| ERA5        | Surface temperature, wind speed, pressure, precipitation | 9km               | 1 hr                | [Link](https://cds.climate.copernicus.eu/how-to-api)                                                    |
| H3          | Grid Mapping                                          | N/A               | N/A                 | [Link](https://www.naturalearthdata.com/)                                                               |
