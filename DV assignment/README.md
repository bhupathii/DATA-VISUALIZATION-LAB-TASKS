# Ride Sharing Dataset Analysis - Assignment

## Overview
This project performs comprehensive analysis on a ride-sharing dataset, covering all required tasks for the assignment.

## Dataset
The dataset includes synthetic ride-sharing data with the following fields:
- **driver_id**: Unique identifier for each driver
- **trip_distance**: Distance of the trip in kilometers
- **trip_cost**: Cost of the trip in dollars
- **rating**: Customer rating (1-5 stars)
- **rider_feedback**: Text feedback from riders
- **region**: City region where the ride occurred
- **latitude/longitude**: Geographic coordinates
- **timestamp**: Date and time of the ride

## Requirements

### Python Version
Python 3.7 or higher

### Installation
```bash
pip install -r requirements.txt
```

## Tasks Completed

### Task 1: Dataset Details (CO1, K2)
- Displays driver ID, trip distance, cost, and ratings
- Provides statistical summaries
- Output: `task1_dataset_details.txt`

### Task 2: Box Plots and Density Plots (CO2, K3)
- Box plots: Trip cost vs. distance categories
- Box plots: Trip cost vs. ratings
- Density plots: Trip frequency by distance and cost
- Output: `task2_boxplots_density.png`, `task2_cost_distance_scatter.png`

### Task 3: Word Cloud (CO3, K3)
- Generates word cloud from rider feedback
- Analyzes most frequent words
- Output: `task3_wordcloud.png`, `task3_word_frequency.txt`

### Task 4: Geo-spatial Heatmaps (CO4, K3)
- Interactive heatmap showing ride demand across city regions
- Static visualizations of regional demand
- Regional statistics analysis
- Output: `task4_geospatial_heatmap.html`, `task4_geospatial_analysis.png`, `task4_region_statistics.txt`

### Task 5: Time-Oriented Demand Trends (CO5, K3)
- Peak vs. non-peak hour analysis
- Daily and weekly demand patterns
- Hour vs. day of week heatmap
- Output: `task5_time_trends.png`, `task5_time_statistics.txt`

## Usage

### Quick Start (Recommended)
```bash
./run_analysis.sh
```
This will automatically generate the dataset and run all analyses.

### Manual Steps

#### Step 1: Generate Dataset
```bash
python3 generate_dataset.py
```
This will create `ride_sharing_dataset.csv` with 2000 records.

#### Step 2: Run Analysis
```bash
python3 ride_sharing_analysis.py
```
This will execute all 5 tasks and generate all output files.

**Note**: If `python3` is not available, use `python` instead.

## Output Files

All generated files will be saved in the `outputs/` folder:

1. **Dataset**: `ride_sharing_dataset.csv` (in project root)
2. **Task 1**: `outputs/task1_dataset_details.txt`
3. **Task 2**: `outputs/task2_boxplots_density.png`, `outputs/task2_cost_distance_scatter.png`
4. **Task 3**: `outputs/task3_wordcloud.png`, `outputs/task3_word_frequency.txt`
5. **Task 4**: `outputs/task4_geospatial_heatmap.html`, `outputs/task4_geospatial_analysis.png`, `outputs/task4_region_statistics.txt`
6. **Task 5**: `outputs/task5_time_trends.png`, `outputs/task5_time_statistics.txt`

**Note**: The `outputs/` folder is automatically created when you run the analysis script.

## Key Features

- **Comprehensive Analysis**: All 5 required tasks implemented
- **Clear Visualizations**: High-quality plots and charts
- **Interactive Maps**: HTML-based interactive heatmaps
- **Detailed Statistics**: Text files with numerical summaries
- **Well-Documented**: Code comments and clear structure

## Project Structure

```
.
├── generate_dataset.py          # Dataset generator
├── ride_sharing_analysis.py     # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── ride_sharing_dataset.csv     # Generated dataset (after running)
```

## Notes

- The dataset is randomly generated but follows realistic patterns
- Peak hours are defined as 7-9 AM and 5-7 PM
- All visualizations are saved in PNG format (300 DPI)
- Interactive heatmaps are saved as HTML files
- All statistics are exported to text files for reference

## Assignment Coverage

✅ **CO1, K2**: Dataset details display  
✅ **CO2, K3**: Box plots and density plots  
✅ **CO3, K3**: Word cloud generation  
✅ **CO4, K3**: Geo-spatial heatmaps  
✅ **CO5, K3**: Time-oriented demand trends  

