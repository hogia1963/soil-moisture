# Soil Moisture Prediction

This system fetches, processes, and predicts soil moisture using real-time data.

## Description and background

The validators queries the miners during specific execution windows defined in the get_request_windows method. These windows are:
- 2:00 - 2:30
- 10:00 - 10:30
- 14:00 - 14:30
- 20:00 - 20:30
The miners give prediction for the next window.

## Aim of the task

Maximize the final score
```python
final_score = await scoring_mechanism.score(pred_data)
```

# Steps

- Understand the flow, datasets, and what the base model does
- Look at these repos:
    - https://github.com/fkwai/geolearn
    - https://github.com/mhpi/hydroDL
    - https://github.com/leelew/CLSTM
    - https://github.com/ljz1228/CLM-LSTM-soil-moisture-prediction (This one seems to be the best but the authors have taken it down. I am requesting from the authors via email.)

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

- `soilmoisture.py`: Main script handling data fetching and processing
- `soilmoisture_basemodel.py`: Base model implementation and model loading logic
- `CUSTOMMODELS.md`: Guide for implementing custom prediction models

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
