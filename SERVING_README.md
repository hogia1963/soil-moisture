# Soil Moisture Prediction Serving System

This document provides detailed instructions for running the soil moisture prediction serving system.

## Overview

The serving system provides an easy way to obtain soil moisture predictions from pre-trained models. It:

1. Downloads required Earth observation data for specified time periods
2. Processes the data into a format suitable for the prediction models
3. Runs predictions using one of four available model versions
4. Saves results and visualizations for analysis

## Quick Start

```bash
# Basic usage (will use a random region and current time)
python serving.py --model v4

# Specify a particular region with a bounding box
python serving.py --model v4 --bbox 148.54 -31.66 149.54 -30.66

# Use a specific date/time
python serving.py --model v3 --time 2023-09-15T12:00:00Z
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model version to use (v1, v2, v3, or v4) | v1 |
| `--time` | Target time in ISO format | Current time |
| `--bbox` | Bounding box coordinates as minlon minlat maxlon maxlat | Random selection |
| `--keep-data` | Keep downloaded data files | False |

## Model Versions

The system supports four different model versions with increasing complexity:

1. **v1 Model**: Basic CNN model that predicts soil moisture directly from satellite imagery
   - Input: Current satellite image
   - Output: Predicted soil moisture (surface and rootzone)

2. **v2 Model**: Incorporates historical SMAP labels to improve predictions
   - Input: Current satellite image + historical soil moisture labels
   - Output: Predicted soil moisture (surface and rootzone)

3. **v3 Model**: Temporal model that uses consecutive images
   - Input: Current image + previous image + historical soil moisture labels
   - Output: Predicted soil moisture (surface and rootzone)

4. **v4 Model**: Advanced model with a change prediction mechanism
   - Input: Previous image + current image + previous soil moisture labels
   - Output: Predicted soil moisture (surface and rootzone)

## Output Directory Structure

Results are stored in the `serving_data/` directory with the following organization:

