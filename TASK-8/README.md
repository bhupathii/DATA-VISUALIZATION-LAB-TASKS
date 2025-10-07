# Spatial and Geospatial Analysis of Agriculture Crop Data

## Task 8: CO4, S3 - Geographical Map, Map Projections

Minimal spatial analysis of agriculture crop yield data with geographical mapping and insights.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python spatial_analysis.py
```

## 📋 5-Step Simple Algorithm

### Step 1: Load Data
- Read agriculture crop yield CSV file
- Create output folder for results

### Step 2: State Analysis  
- Group data by state
- Calculate yield metrics (total yield, yield per hectare, revenue)

### Step 3: Create Map
- Generate interactive choropleth map
- Color-code states by yield per hectare
- Save as HTML file

### Step 4: Create Chart
- Generate bar chart of top 10 states
- Save as PNG image

### Step 5: Generate Insights
- Display key statistics
- Identify best performing state and climate zone
- Save all outputs to 'output' folder

## 📁 File Structure

```
task 8/
├── agriculture_crop_yield.csv    # Dataset
├── spatial_analysis.py          # Single analysis script
├── requirements.txt             # Dependencies
├── output/                      # Generated outputs
│   ├── yield_map.html          # Interactive map
│   └── top_states.png          # Bar chart
└── README.md                   # This documentation
```

## 📊 Outputs

- **Interactive Map**: `output/yield_map.html` - Choropleth map of yield by state
- **Bar Chart**: `output/top_states.png` - Top 10 states by yield per hectare
- **Console Output**: Key insights and statistics

## 🎯 Key Features

- Minimal code (single Python file)
- Interactive geographical mapping
- State-wise performance analysis
- Climate zone insights
- Automated output generation
