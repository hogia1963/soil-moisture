# Soil Moisture Prediction System

This system fetches, processes, and predicts soil moisture using real-time data.

## Description and background

The validator queries the miners during specific execution windows defined in the get_validator_windows method. These windows are:

2:00 - 2:30
10:00 - 10:30
14:00 - 14:30
20:00 - 20:30
The validator checks if it is within an execution window and then queries the miners with the payload.

## Aim of the task

- Real-time soil data fetching and processing
- Historical data analysis
- Predictive modeling with:
  - Base model (Prophet-based fallback)
  - Support for custom model integration
- Generate predictions
- Ground truth validation

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

The system will:
- Fetch latest data
- Process historical values
- Make predictions for the next time frame
- Validate against ground truth when available

## Custom Model Integration

We encourage contributors to develop custom models that improve upon the base model's performance. To integrate your custom model:

1. Follow the implementation guide in [CUSTOMMODELS.md](CUSTOMMODELS.md)
2. Ensure your model follows the required naming conventions and interfaces
3. Test against the base model performance

See [CUSTOMMODELS.md](CUSTOMMODELS.md) for detailed specifications and examples.

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
