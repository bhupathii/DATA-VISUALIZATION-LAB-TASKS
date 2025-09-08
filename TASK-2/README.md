# Agriculture Crop Yield - Univariate Analysis

This project performs univariate analysis on agriculture crop yield data, creating visualizations for both categorical and continuous variables as specifically requested.

## Dataset Overview

The dataset contains **86 records** with **19 columns** covering agricultural data from various US states for the years 2020-2021.

### Dataset Structure
- **Shape**: 86 rows × 19 columns
- **Data Types**: 
  - 7 integer columns
  - 3 float columns  
  - 9 object (string) columns

## Variables Classification

### Categorical Variables (8)
1. **State** - US states (21 unique values)
2. **Crop_Type** - Types of crops (5 unique values: Corn, Soybeans, Wheat, Cotton, Rice)
3. **Season** - Growing season (2 unique values: Summer, Winter)
4. **Climate_Zone** - Climate classification (3 unique values: Continental, Subtropical, Mediterranean)
5. **Soil_Type** - Soil classification (5 unique values: Clay Loam, Clay, Loam, Sandy, Sandy Loam)
6. **Irrigation_Type** - Irrigation method (3 unique values: Flood Irrigation, Drip Irrigation, Sprinkler)
7. **Pest_Infestation_Level** - Pest levels (3 unique values: Low, Medium, High)
8. **Disease_Incidence** - Disease levels (3 unique values: Low, Medium, High)

### Continuous Variables (9)
1. **Area_Hectares** - Cultivated area in hectares
2. **Yield_Tonnes** - Total yield in tonnes
3. **Yield_per_Hectare** - Yield efficiency metric
4. **Fertilizer_Usage_kg** - Fertilizer application in kg
5. **Precipitation_mm** - Rainfall in millimeters
6. **Temperature_Celsius** - Temperature in Celsius
7. **Storage_Loss_Percentage** - Post-harvest loss percentage
8. **Market_Price_per_Tonne** - Market price per tonne
9. **Total_Revenue** - Total revenue generated

## Visualizations Created

### 1. Categorical Data Analysis (`categorical_analysis.png`)

#### Bar Charts:
- **Crop Type Distribution** - Shows frequency of each crop type
- **Season Distribution** - Shows frequency of Summer vs Winter crops

#### Pie Charts:
- **Crop Type Distribution** - Percentage breakdown of crop types
- **Season Distribution** - Percentage breakdown of seasons

### 2. Continuous Data Analysis (`continuous_analysis.png`)

#### Scatter Plots:
- **Area vs Yield** - Relationship between cultivated area and total yield
- **Temperature vs Yield per Hectare** - Relationship between temperature and yield efficiency

#### Line Plots:
- **Temperature by Crop Type** - Average temperature for each crop type
- **Market Price Trends by Year** - Average market price trends over time

#### Strip Plots:
- **Yield per Hectare by Crop Type** - Distribution of yield efficiency across crop types

#### Swarm Plots:
- **Market Price by Crop Type** - Distribution of market prices across crop types

#### Histograms:
- **Yield per Hectare Distribution** - Frequency distribution of yield efficiency
- **Precipitation Distribution with Rug Plot** - Frequency distribution of precipitation with rug plot overlay

#### Density Plots:
- **Temperature Distribution** - Probability density of temperature values

## Key Insights from the Analysis

### Categorical Data Insights:
1. **Crop Distribution**: Corn is the most common crop (34 records), followed by Soybeans (28 records)
2. **Seasonal Pattern**: Summer crops dominate (72 records vs 14 winter records)
3. **Geographic Distribution**: California has the most records (6), with most states having 4 records
4. **Climate Zones**: Continental climate is most common (52 records)
5. **Irrigation**: Flood irrigation is the primary method (80 records)

### Continuous Data Insights:
1. **Yield Efficiency**: Most crops show consistent yield per hectare patterns
2. **Area-Yield Relationship**: Strong positive correlation between area and total yield
3. **Market Dynamics**: Negative correlation between yield and market price
4. **Revenue Patterns**: Total revenue strongly correlates with area and yield

## Statistical Summary

### Continuous Variables Statistics:
- **Area_Hectares**: Mean ~148,802 ha, Range: 85,000 - 355,000 ha
- **Yield_Tonnes**: Mean ~753,314 tonnes, Range: 255,000 - 2,485,000 tonnes
- **Yield_per_Hectare**: Mean ~5.5 tonnes/ha, Range: 3.0 - 7.0 tonnes/ha
- **Market_Price_per_Tonne**: Mean ~$254, Range: $165 - $356

## Files Generated

1. `categorical_analysis.png` - Categorical data visualizations (Bar charts and Pie charts)
2. `continuous_analysis.png` - Continuous data visualizations (Scatter, Line, Strip, Swarm, Histogram, Density, Rug plots)
3. `univariate_analysis.py` - Python script for analysis
4. `requirements.txt` - Python dependencies

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- numpy >= 1.21.0

## Usage

Run the analysis script:
```bash
python3 univariate_analysis.py
```

This will generate the requested visualizations and print comprehensive statistical summaries to the console.

## Analysis Methodology

The analysis follows standard univariate analysis techniques focusing on the specifically requested chart types:
- **Categorical Data**: Frequency analysis with bar charts and pie charts
- **Continuous Data**: Distribution analysis with histograms, density plots, and various scatter/line visualizations including strip plots, swarm plots, and rug plots

This focused analysis provides clear insights into the agriculture crop yield dataset using the requested visualization techniques.
