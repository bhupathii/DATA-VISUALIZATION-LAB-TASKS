# Separated Bivariate Analysis - Agriculture Crop Yield Dataset

## Overview
The analysis has been separated into two distinct scripts based on data types for better organization and clarity.

## Scripts Created

### 1. `continuous_vs_continuous.py`
**Purpose**: Analyze relationships between continuous variables using scatterplots with fit lines.

**Features**:
- 4 scatterplots with red fit lines
- Correlation coefficients and p-values displayed on each plot
- Statistical significance testing
- Color-coded data points for better visualization

**Plots Generated**:
1. **Area vs Yield** (r = 0.832, p < 0.001) - Blue points
2. **Fertilizer vs Yield per Hectare** (r = 0.776, p < 0.001) - Green points
3. **Precipitation vs Temperature** (r = 0.571, p < 0.001) - Orange points
4. **Market Price vs Total Revenue** (r = -0.291, p = 0.007) - Purple points

**Output**: `continuous_vs_continuous.png`

### 2. `categorical_vs_continuous.py`
**Purpose**: Analyze relationships between categorical and continuous variables using various plot types.

**Features**:
- 6 different visualization types
- Statistical tests (ANOVA, Kruskal-Wallis)
- Summary statistics
- Enhanced visual elements

**Plots Generated**:
1. **Bar Chart** - Average Yield by Crop Type (with value labels)
2. **Grouped Kernel Density** - Yield per Hectare by Climate Zone
3. **Box Plot** - Yield per Hectare by Crop Type
4. **Violin Plot** - Total Revenue by Climate Zone
5. **Ridgeline Plot** - Yield per Hectare by Crop Type (overlapping histograms)
6. **Beeswarm Plot** - Fertilizer Usage by Irrigation Type (box + swarm overlay)

**Statistical Tests**:
- **ANOVA**: Yield per Hectare by Crop Type (F = 3016.96, p < 0.001)
- **Kruskal-Wallis**: Total Revenue by Climate Zone (H = 13.51, p = 0.001)

**Output**: `categorical_vs_continuous.png`

### 3. `run_analysis.py`
**Purpose**: Main script to run both analyses sequentially.

**Features**:
- Automated execution of both scripts
- Error handling
- Progress indicators
- Summary of generated files

## Key Findings

### Continuous vs Continuous:
- **Strongest correlation**: Area vs Yield (r = 0.832)
- **Resource efficiency**: Fertilizer vs Yield per Hectare (r = 0.776)
- **Climate relationship**: Precipitation vs Temperature (r = 0.571)
- **Market dynamics**: Market Price vs Total Revenue (r = -0.291)

### Categorical vs Continuous:
- **Crop performance**: Corn has highest average yield (6.94 tonnes/hectare)
- **Climate impact**: Continental zones generate highest revenue ($190M average)
- **Statistical significance**: All differences are highly significant (p < 0.001)

## Usage

### Individual Scripts:
```bash
python3 continuous_vs_continuous.py
python3 categorical_vs_continuous.py
```

### Combined Execution:
```bash
python3 run_analysis.py
```

## Advantages of Separation

1. **Modularity**: Each script focuses on specific analysis type
2. **Clarity**: Easier to understand and modify individual components
3. **Flexibility**: Can run specific analyses as needed
4. **Maintainability**: Simpler code structure
5. **Reusability**: Scripts can be used independently

## File Structure
```
├── continuous_vs_continuous.py      # Continuous vs continuous analysis
├── categorical_vs_continuous.py     # Categorical vs continuous analysis
├── run_analysis.py                  # Main execution script
├── continuous_vs_continuous.png     # Continuous analysis output
├── categorical_vs_continuous.png    # Categorical analysis output
└── agriculture_crop_yield.csv       # Dataset
```

## Technical Details
- **Language**: Python 3
- **Libraries**: pandas, matplotlib, seaborn, numpy, scipy
- **Output Format**: High-resolution PNG (300 DPI)
- **Statistical Tests**: Pearson correlation, ANOVA, Kruskal-Wallis 