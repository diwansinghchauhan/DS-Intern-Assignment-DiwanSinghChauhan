# Smart Factory Sensor Data Description

This document provides detailed information about the features in the Smart Factory Energy Prediction dataset.

## Dataset Overview

The dataset contains sensor measurements collected from a manufacturing facility over time. It includes environmental readings from multiple zones within the factory, external weather conditions, and energy consumption metrics. The primary target variable is `equipment_energy_consumption`, which represents the energy used by manufacturing equipment in Wh (watt-hours).

## Feature Descriptions

### Time Information
- `timestamp`: Date and time of the measurement (format: YYYY-MM-DD HH:MM:SS)

### Energy Metrics
- `equipment_energy_consumption`: Energy consumption of manufacturing equipment in Wh (target variable)
- `lighting_energy`: Energy consumption of lighting systems in Wh

### Zone Measurements
The factory is divided into 9 distinct zones, each equipped with temperature and humidity sensors:

#### Zone 1 - Main Production Area
- `zone1_temperature`: Temperature in Zone 1 (°C)
- `zone1_humidity`: Relative humidity in Zone 1 (%)

#### Zone 2 - Assembly Line
- `zone2_temperature`: Temperature in Zone 2 (°C)
- `zone2_humidity`: Relative humidity in Zone 2 (%)

#### Zone 3 - Quality Control
- `zone3_temperature`: Temperature in Zone 3 (°C)
- `zone3_humidity`: Relative humidity in Zone 3 (%)

#### Zone 4 - Packaging Area
- `zone4_temperature`: Temperature in Zone 4 (°C)
- `zone4_humidity`: Relative humidity in Zone 4 (%)

#### Zone 5 - Raw Material Storage
- `zone5_temperature`: Temperature in Zone 5 (°C)
- `zone5_humidity`: Relative humidity in Zone 5 (%)

#### Zone 6 - Loading Bay
- `zone6_temperature`: Temperature in Zone 6 (°C)
- `zone6_humidity`: Relative humidity in Zone 6 (%)

#### Zone 7 - Office Space
- `zone7_temperature`: Temperature in Zone 7 (°C)
- `zone7_humidity`: Relative humidity in Zone 7 (%)

#### Zone 8 - Control Room
- `zone8_temperature`: Temperature in Zone 8 (°C)
- `zone8_humidity`: Relative humidity in Zone 8 (%)

#### Zone 9 - Staff Area
- `zone9_temperature`: Temperature in Zone 9 (°C)
- `zone9_humidity`: Relative humidity in Zone 9 (%)

### External Weather Conditions
- `outdoor_temperature`: Outside temperature (°C)
- `outdoor_humidity`: Outside relative humidity (%)
- `atmospheric_pressure`: Atmospheric pressure (mm Hg)
- `wind_speed`: Wind speed (m/s)
- `visibility_index`: Visibility measure (km)
- `dew_point`: Dew point temperature (°C)

### Additional Variables
- `random_variable1`: Random variable generated for the dataset - candidates should determine through analysis whether to include in modeling
- `random_variable2`: Random variable generated for the dataset - candidates should determine through analysis whether to include in modeling

## Target Variable Details

The main target variable is `equipment_energy_consumption`, which represents the energy consumption of manufacturing equipment measured in watt-hours (Wh). This is a continuous variable that typically ranges from 10 to 1080 Wh in the dataset. The distribution is right-skewed, with most values concentrated in the lower range.

The equipment energy consumption is influenced by various factors, including:
- Environmental conditions inside different factory zones
- External weather conditions
- Time of day and production cycles
- Other operational parameters

Accurately predicting this value would allow facility managers to optimize their operations for energy efficiency, potentially resulting in significant cost savings and reduced environmental impact.
