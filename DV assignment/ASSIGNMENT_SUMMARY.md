# Assignment Summary - Ride Sharing Dataset Analysis

## Assignment Overview
This project implements a complete data visualization and analysis solution for a ride-sharing dataset, covering all 5 required tasks.

## Task Coverage

### ✅ Task 1: Dataset Details (CO1, K2)
**Objective**: Show dataset details including driver ID, trip distance, cost, and ratings.

**Implementation**:
- Displays first 20 records with key columns
- Provides comprehensive statistical summary (mean, median, min, max)
- Shows rating distribution with percentages
- Exports detailed summary to text file

**Output Files**:
- `task1_dataset_details.txt` - Complete statistical summary

---

### ✅ Task 2: Box Plots and Density Plots (CO2, K3)
**Objective**: Construct box plots for trip cost vs. distance and density plots for trip frequency.

**Implementation**:
- **Box Plot 1**: Trip cost vs. distance categories (0-5km, 5-10km, etc.)
- **Box Plot 2**: Trip cost vs. rating
- **Density Plot 1**: Trip frequency by distance with KDE curve
- **Density Plot 2**: Trip frequency by cost with KDE curve
- **Bonus**: Scatter plot with regression line showing cost-distance relationship

**Output Files**:
- `task2_boxplots_density.png` - Main visualization with all plots
- `task2_cost_distance_scatter.png` - Additional scatter plot analysis

---

### ✅ Task 3: Word Cloud (CO3, K3)
**Objective**: Generate a word cloud from rider feedback.

**Implementation**:
- Combines all rider feedback text
- Creates visually appealing word cloud with color coding
- Analyzes word frequency
- Exports top 50 most frequent words

**Output Files**:
- `task3_wordcloud.png` - High-resolution word cloud visualization
- `task3_word_frequency.txt` - Word frequency analysis

---

### ✅ Task 4: Geo-spatial Heatmaps (CO4, K3)
**Objective**: Map ride demand across city regions using geo-spatial heatmaps.

**Implementation**:
- **Interactive Heatmap**: HTML-based interactive map with Folium
- **Regional Analysis**: Bar chart showing demand by region
- **Scatter Plot**: Geographic distribution colored by cost, sized by distance
- **Regional Statistics**: Average cost, distance, rating, and total rides per region

**Output Files**:
- `task4_geospatial_heatmap.html` - Interactive heatmap (open in browser)
- `task4_geospatial_analysis.png` - Static visualizations
- `task4_region_statistics.txt` - Regional statistics

---

### ✅ Task 5: Time-Oriented Demand Trends (CO5, K3)
**Objective**: Analyze time-oriented ride demand trends (peak vs. non-peak).

**Implementation**:
- **Peak Hours Definition**: 7-9 AM (morning) and 5-7 PM (evening)
- **Hourly Demand**: Line chart showing rides by hour of day
- **Peak vs Non-Peak**: Comparative bar charts for key metrics
- **Daily Patterns**: Demand by day of week
- **Time Series**: Daily demand trend over 3 months
- **Heatmap**: Hour vs. Day of week demand matrix
- **Statistical Comparison**: Peak vs. non-peak hour statistics

**Output Files**:
- `task5_time_trends.png` - Comprehensive time analysis (6 subplots)
- `task5_time_statistics.txt` - Detailed time-based statistics

---

## Dataset Information

**Dataset Size**: 2,000 ride records

**Fields Included**:
- `driver_id`: Unique driver identifier (1000-9999)
- `trip_distance`: Distance in kilometers (log-normal distribution)
- `trip_cost`: Cost in dollars (base fare + distance-based)
- `rating`: Customer rating (1-5 stars)
- `rider_feedback`: Text feedback from customers
- `region`: City region (10 different regions)
- `latitude/longitude`: Geographic coordinates
- `timestamp`: Date and time (3-month period)

**Data Characteristics**:
- Realistic trip distances (0-100+ km)
- Cost correlated with distance
- Ratings distribution: 40% 5-star, 30% 4-star, 15% 3-star, 10% 2-star, 5% 1-star
- 10 city regions with distinct geographic locations
- Time series spanning 90 days with peak/non-peak patterns

---

## Key Findings

1. **Distance vs. Cost**: Strong positive correlation - longer trips cost more
2. **Peak Hours**: 35-40% of rides occur during peak hours (7-9 AM, 5-7 PM)
3. **Regional Demand**: Downtown and Commercial District have highest demand
4. **Ratings**: Average rating is ~4.0/5.0, with most rides rated 4-5 stars
5. **Time Patterns**: Friday and Monday show highest demand; weekends show consistent demand

---

## Technical Implementation

**Languages & Libraries**:
- Python 3.7+
- Pandas: Data manipulation
- NumPy: Numerical operations
- Matplotlib: Basic plotting
- Seaborn: Statistical visualizations
- WordCloud: Word cloud generation
- Folium: Interactive maps

**Code Structure**:
- `generate_dataset.py`: Dataset generation module
- `ride_sharing_analysis.py`: Main analysis script with 5 task functions
- Modular design: Each task is a separate function
- Error handling: Graceful handling of missing files
- Documentation: Inline comments and docstrings

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Quick Run**:
   ```bash
   ./run_analysis.sh
   ```

3. **Manual Run**:
   ```bash
   python3 generate_dataset.py
   python3 ride_sharing_analysis.py
   ```

---

## Output Files Summary

All output files are saved in the `outputs/` folder (created automatically):

| Task | File | Description |
|------|------|-------------|
| Dataset | `ride_sharing_dataset.csv` | Complete dataset (in project root) |
| Task 1 | `outputs/task1_dataset_details.txt` | Statistical summary |
| Task 2 | `outputs/task2_boxplots_density.png` | Box & density plots |
| Task 2 | `outputs/task2_cost_distance_scatter.png` | Scatter plot |
| Task 3 | `outputs/task3_wordcloud.png` | Word cloud image |
| Task 3 | `outputs/task3_word_frequency.txt` | Word frequency |
| Task 4 | `outputs/task4_geospatial_heatmap.html` | Interactive map |
| Task 4 | `outputs/task4_geospatial_analysis.png` | Static maps |
| Task 4 | `outputs/task4_region_statistics.txt` | Regional stats |
| Task 5 | `outputs/task5_time_trends.png` | Time analysis |
| Task 5 | `outputs/task5_time_statistics.txt` | Time statistics |

---

## Assignment Requirements Checklist

- ✅ **CO1, K2**: Dataset details display - COMPLETE
- ✅ **CO2, K3**: Box plots and density plots - COMPLETE
- ✅ **CO3, K3**: Word cloud generation - COMPLETE
- ✅ **CO4, K3**: Geo-spatial heatmaps - COMPLETE
- ✅ **CO5, K3**: Time-oriented demand trends - COMPLETE

All tasks are fully implemented with clear visualizations and documentation.

