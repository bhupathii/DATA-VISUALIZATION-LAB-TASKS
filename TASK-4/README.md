# Multivariate Analysis of Agriculture Crop Yield Data

## Overview
This project performs comprehensive multivariate analysis on agriculture crop yield data using multiple visualization techniques. The analysis explores relationships between various agricultural variables and provides insights into crop production patterns across different states, climate zones, and time periods.

## Dataset Description
The dataset contains agriculture crop yield information with the following key variables:

### Numerical Variables (Measures):
- **Area_Hectares**: Cultivated area in hectares
- **Yield_Tonnes**: Total yield in tonnes
- **Yield_per_Hectare**: Yield efficiency per hectare
- **Fertilizer_Usage_kg**: Fertilizer consumption in kilograms
- **Precipitation_mm**: Rainfall in millimeters
- **Temperature_Celsius**: Average temperature in Celsius
- **Storage_Loss_Percentage**: Percentage of crop lost during storage
- **Market_Price_per_Tonne**: Market price per tonne
- **Total_Revenue**: Total revenue generated

### Categorical Variables:
- **Year**: Time period (2020-2021)
- **State**: US states
- **Crop_Type**: Type of crop (Corn, Wheat, Soybeans, Cotton, Rice)
- **Season**: Growing season (Summer/Winter)
- **Climate_Zone**: Climate classification
- **Soil_Type**: Type of soil
- **Irrigation_Type**: Irrigation method
- **Pest_Infestation_Level**: Pest infestation severity
- **Disease_Incidence**: Disease occurrence level

## Visualizations Created

### 1. Scatterplot Matrix
- **Purpose**: Explore pairwise relationships between numerical variables
- **Features**: 
  - Shows correlation coefficients (ρ) for each variable pair
  - Reveals linear and non-linear relationships
  - Identifies potential outliers and clusters
- **Variables**: 9 key numerical measures

### 2. Parallel Coordinates Plot
- **Purpose**: Visualize multivariate data across different dimensions
- **Features**:
  - Normalized variables for fair comparison
  - Color-coded by state
  - Shows patterns and trends across multiple variables simultaneously
- **Variables**: 6 key agricultural measures by state

### 3. Line Graphs (Time Series Analysis)
- **Purpose**: Analyze trends over time (2020-2021)
- **Subplots**:
  - Total yield by crop type over years
  - Average yield per hectare by top states
  - Total revenue by climate zone
  - Weather conditions (temperature and precipitation) trends

### 4. Stacked Bar Charts
- **Purpose**: Show composition and distribution across multiple categorical variables
- **Subplots**:
  - Yield by crop type and state
  - Revenue by climate zone and season
  - Area by soil type and irrigation type
  - Fertilizer usage by pest level and disease incidence

### 5. Correlation Heatmap (Bonus)
- **Purpose**: Visualize correlation matrix between numerical variables
- **Features**: Color-coded correlation strength with numerical values

## Key Insights from the Analysis

### Crop Performance:
- Corn shows the highest yield per hectare (7.0 tonnes)
- Soybeans have consistent yield across different regions
- Wheat performs better in winter seasons

### Regional Patterns:
- Iowa and Illinois lead in total agricultural revenue
- Mediterranean climate zones show different patterns than Continental zones
- Temperature and precipitation significantly impact yield efficiency

### Environmental Factors:
- Higher fertilizer usage correlates with increased yield
- Pest infestation and disease incidence affect storage losses
- Climate zones influence crop selection and performance

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python multivariate_analysis.py
```

### Output Files
The script generates the following visualization files:
- `scatterplot_matrix.png` - Scatterplot matrix of numerical variables
- `parallel_coordinates.png` - Parallel coordinates plot
- `line_graphs.png` - Time series line graphs
- `stacked_bar_charts.png` - Stacked bar charts
- `correlation_heatmap.png` - Correlation heatmap

## Technical Implementation

### Libraries Used:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Basic plotting and visualization
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations (imported for future use)

### Data Processing:
- Data normalization for parallel coordinates
- Aggregation by multiple categorical variables
- Time series analysis with year-based grouping
- Correlation analysis between numerical variables

### Visualization Features:
- High-resolution output (300 DPI)
- Professional styling with seaborn
- Clear titles and labels
- Color-coded legends
- Grid lines for better readability

## Analysis Methodology

1. **Exploratory Data Analysis**: Initial data exploration and summary statistics
2. **Correlation Analysis**: Understanding relationships between variables
3. **Temporal Analysis**: Trends and patterns over time
4. **Categorical Analysis**: Performance across different groups
5. **Multivariate Visualization**: Complex relationships across multiple dimensions

## Business Applications

This analysis can be used for:
- **Agricultural Planning**: Optimize crop selection based on regional conditions
- **Resource Allocation**: Efficient distribution of fertilizers and irrigation
- **Risk Management**: Understanding factors affecting crop losses
- **Market Analysis**: Revenue optimization strategies
- **Policy Making**: Evidence-based agricultural policies

## Future Enhancements

Potential improvements include:
- Interactive visualizations using Plotly
- Machine learning models for yield prediction
- Seasonal decomposition analysis
- Geographic mapping of agricultural data
- Real-time data integration capabilities
