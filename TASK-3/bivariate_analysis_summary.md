# Bivariate Analysis Summary - Agriculture Crop Yield Dataset

## Dataset Overview
- **Records**: 86 entries
- **Columns**: 19 variables
- **Years**: 2020-2021
- **States**: 20+ US states
- **Crops**: Corn, Wheat, Soybeans, Cotton, Rice

## Data Types

### Categorical (9):
State, Crop_Type, Season, Climate_Zone, Soil_Type, Irrigation_Type, Pest_Infestation_Level, Disease_Incidence, Harvest_Date

### Continuous (10):
Year, Area_Hectares, Yield_Tonnes, Yield_per_Hectare, Fertilizer_Usage_kg, Precipitation_mm, Temperature_Celsius, Storage_Loss_Percentage, Market_Price_per_Tonne, Total_Revenue

## Analysis Results

### 1. Categorical vs Categorical
- **Stacked Bar**: Crop Type vs Climate Zone
- **Grouped Bar**: Crop Type vs Season
- **Segmented Bar**: Pest Level vs Disease Incidence
- **Mosaic Plot**: Irrigation Type vs Soil Type

### 2. Continuous vs Continuous
- **Scatterplots with Fit Lines**:
  - Area vs Yield (r = 0.832)
  - Fertilizer vs Yield per Hectare (r = 0.776)
  - Precipitation vs Temperature (r = 0.571)
  - Market Price vs Total Revenue

### 3. Categorical vs Continuous
- **Bar Chart**: Average Yield by Crop Type
- **Box Plot**: Yield per Hectare by Crop Type
- **Violin Plot**: Total Revenue by Climate Zone
- **Beeswarm Plot**: Fertilizer Usage by Irrigation Type

## Statistical Tests

### Chi-Square Test: Crop Type vs Climate Zone
- **P-value**: 0.0006
- **Conclusion**: Significant association

### ANOVA Test: Yield per Hectare by Crop Type
- **P-value**: < 0.0001
- **Conclusion**: Highly significant differences

### Correlation Test: Area vs Yield
- **Correlation**: 0.832
- **P-value**: < 0.0001
- **Conclusion**: Strong positive correlation

## Key Correlations

### Strong (>0.7):
- Area vs Total Revenue (r = 0.943)
- Yield per Hectare vs Market Price (r = -0.872)
- Area vs Yield (r = 0.832)
- Yield vs Total Revenue (r = 0.897)

### Moderate (0.5-0.7):
- Temperature vs Storage Loss (r = 0.703)
- Yield vs Fertilizer Usage (r = 0.650)
- Precipitation vs Temperature (r = 0.571)

## Key Insights

1. **Scale Effect**: Larger areas = higher revenue
2. **Price Efficiency**: Higher yield = lower market price
3. **Climate Impact**: Temperature affects storage losses
4. **Resource Management**: Fertilizer increases yield
5. **Crop Specialization**: Different crops for different climates

## Files Generated
- `bivariate_analysis.png` - Main visualizations (12 plots)
- `correlation_matrix.png` - Correlation heatmap

## Technical Details
- **Language**: Python 3
- **Libraries**: pandas, matplotlib, seaborn, numpy, scipy
- **Output**: High-resolution PNG files (300 DPI) 