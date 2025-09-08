# Agriculture Crop Yield - Bivariate Analysis

## Overview
Bivariate analysis on agriculture crop yield dataset with various visualization techniques.

## Files
- `agriculture_crop_yield.csv` - Dataset (86 records, 19 variables)
- `bivariate_analysis.py` - Analysis script
- `bivariate_analysis_summary.md` - Analysis summary
- `requirements.txt` - Dependencies

## Quick Start
```bash
pip3 install -r requirements.txt
python3 bivariate_analysis.py
```

## Analysis Types
### Categorical vs Categorical
- Stacked Bar Chart
- Grouped Bar Chart  
- Segmented Bar Chart
- Mosaic Plot

### Continuous vs Continuous
- Scatterplot with Fit Lines (4 pairs)

### Categorical vs Continuous
- Bar Chart with Summary Statistics
- Box Plots
- Violin Plots
- Beeswarm Plots

### Statistical Tests
- Chi-square Test
- ANOVA Test
- Pearson Correlation Test

## Key Findings
- **Strongest Correlation**: Area vs Total Revenue (r = 0.943)
- **Inverse Relationship**: Yield per Hectare vs Market Price (r = -0.872)
- **Climate Impact**: Temperature vs Storage Loss (r = 0.703)

## Dataset Info
- **Size**: 86 records × 19 columns
- **Period**: 2020-2021
- **Coverage**: 20+ US states
- **Crops**: Corn, Wheat, Soybeans, Cotton, Rice

## Output Files
- `bivariate_analysis.png` - Main visualizations
- `correlation_matrix.png` - Correlation heatmap 