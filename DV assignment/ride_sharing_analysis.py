"""
Ride Sharing Dataset Analysis
Assignment Implementation - Complete Analysis Suite

This script performs all required analyses:
1. Dataset details display
2. Box plots and density plots
3. Word cloud generation
4. Geo-spatial heatmaps
5. Time-oriented demand trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create outputs directory if it doesn't exist
OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}/")

# Import dataset generator
from generate_dataset import generate_ride_sharing_dataset

def load_dataset():
    """Load or generate the dataset"""
    try:
        df = pd.read_csv('ride_sharing_dataset.csv')
        print("Dataset loaded from CSV file.")
    except FileNotFoundError:
        print("Dataset not found. Generating new dataset...")
        df = generate_ride_sharing_dataset(2000)
        df.to_csv('ride_sharing_dataset.csv', index=False)
        print("Dataset generated and saved.")
    return df

def task1_dataset_details(df):
    """
    Task 1: Show dataset details - driver ID, trip distance, cost, ratings
    CO1, K2
    """
    print("\n" + "="*80)
    print("TASK 1: DATASET DETAILS")
    print("="*80)
    
    # Display basic info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Records: {len(df)}")
    print(f"Total Drivers: {df['driver_id'].nunique()}")
    
    # Display key columns
    print("\n" + "-"*80)
    print("KEY COLUMNS SUMMARY:")
    print("-"*80)
    
    details_df = df[['driver_id', 'trip_distance', 'trip_cost', 'rating']].copy()
    
    print("\nFirst 20 Records:")
    print(details_df.head(20).to_string())
    
    print("\n" + "-"*80)
    print("STATISTICAL SUMMARY:")
    print("-"*80)
    print(details_df.describe())
    
    print("\n" + "-"*80)
    print("RATING DISTRIBUTION:")
    print("-"*80)
    rating_counts = df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        percentage = (count / len(df)) * 100
        print(f"Rating {rating}: {count} trips ({percentage:.2f}%)")
    
    # Save summary to file
    summary = f"""
DATASET DETAILS SUMMARY
=======================
Total Records: {len(df)}
Total Unique Drivers: {df['driver_id'].nunique()}
Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}

TRIP DISTANCE:
  Mean: {df['trip_distance'].mean():.2f} km
  Median: {df['trip_distance'].median():.2f} km
  Min: {df['trip_distance'].min():.2f} km
  Max: {df['trip_distance'].max():.2f} km

TRIP COST:
  Mean: ${df['trip_cost'].mean():.2f}
  Median: ${df['trip_cost'].median():.2f}
  Min: ${df['trip_cost'].min():.2f}
  Max: ${df['trip_cost'].max():.2f}

RATINGS:
  Mean Rating: {df['rating'].mean():.2f}/5.0
  Rating Distribution:
"""
    for rating, count in rating_counts.items():
        percentage = (count / len(df)) * 100
        summary += f"    {rating} stars: {count} trips ({percentage:.2f}%)\n"
    
    output_path = os.path.join(OUTPUT_DIR, 'task1_dataset_details.txt')
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"\nSummary saved to: {output_path}")
    return details_df

def task2_boxplots_density(df):
    """
    Task 2: Construct box plots for trip cost vs. distance and density plots for trip frequency
    CO2, K3
    """
    print("\n" + "="*80)
    print("TASK 2: BOX PLOTS AND DENSITY PLOTS")
    print("="*80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Task 2: Box Plots and Density Plots Analysis', fontsize=16, fontweight='bold')
    
    # 1. Box plot: Trip Cost vs Distance (categorized)
    ax1 = axes[0, 0]
    # Create distance categories
    df['distance_category'] = pd.cut(df['trip_distance'], 
                                     bins=[0, 5, 10, 15, 20, 100],
                                     labels=['0-5km', '5-10km', '10-15km', '15-20km', '20+km'])
    
    sns.boxplot(data=df, x='distance_category', y='trip_cost', ax=ax1, palette='viridis')
    ax1.set_title('Box Plot: Trip Cost vs Distance Category', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distance Category (km)', fontsize=10)
    ax1.set_ylabel('Trip Cost ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot: Trip Cost distribution by Rating
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='rating', y='trip_cost', ax=ax2, palette='coolwarm')
    ax2.set_title('Box Plot: Trip Cost vs Rating', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Rating', fontsize=10)
    ax2.set_ylabel('Trip Cost ($)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Density plot: Trip Frequency (Distance)
    ax3 = axes[1, 0]
    sns.histplot(df['trip_distance'], kde=True, ax=ax3, color='skyblue', bins=50)
    ax3.set_title('Density Plot: Trip Frequency by Distance', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trip Distance (km)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Density plot: Trip Frequency (Cost)
    ax4 = axes[1, 1]
    sns.histplot(df['trip_cost'], kde=True, ax=ax4, color='coral', bins=50)
    ax4.set_title('Density Plot: Trip Frequency by Cost', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trip Cost ($)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'task2_boxplots_density.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualizations saved to: {output_path}")
    plt.close()
    
    # Additional scatter plot with regression
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='trip_distance', y='trip_cost', 
                   hue='rating', size='rating', sizes=(50, 200), alpha=0.6, ax=ax)
    sns.regplot(data=df, x='trip_distance', y='trip_cost', 
               scatter=False, color='red', line_kws={'linewidth': 2}, ax=ax)
    ax.set_title('Trip Cost vs Distance (with Regression Line)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trip Distance (km)', fontsize=12)
    ax.set_ylabel('Trip Cost ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'task2_cost_distance_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Additional scatter plot saved to: {output_path}")
    plt.close()
    
    print("\nTask 2 completed successfully!")

def task3_wordcloud(df):
    """
    Task 3: Generate a word cloud from rider feedback
    CO3, K3
    """
    print("\n" + "="*80)
    print("TASK 3: WORD CLOUD FROM RIDER FEEDBACK")
    print("="*80)
    
    # Combine all feedback into single text
    feedback_text = ' '.join(df['rider_feedback'].astype(str))
    
    print(f"\nTotal feedback entries: {len(df)}")
    print(f"Total words in feedback: {len(feedback_text.split())}")
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate(feedback_text)
    
    # Create visualization
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud: Rider Feedback Analysis', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'task3_wordcloud.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nWord cloud saved to: {output_path}")
    plt.close()
    
    # Word frequency analysis
    from collections import Counter
    words = feedback_text.lower().split()
    word_freq = Counter(words)
    
    print("\nTop 20 Most Frequent Words in Feedback:")
    print("-" * 50)
    for word, freq in word_freq.most_common(20):
        print(f"{word:20s}: {freq:4d}")
    
    # Save word frequency to file
    output_path = os.path.join(OUTPUT_DIR, 'task3_word_frequency.txt')
    with open(output_path, 'w') as f:
        f.write("WORD FREQUENCY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        for word, freq in word_freq.most_common(50):
            f.write(f"{word:20s}: {freq:4d}\n")
    
    print(f"\nWord frequency analysis saved to: {output_path}")
    print("\nTask 3 completed successfully!")

def task4_geospatial_heatmap(df):
    """
    Task 4: Map ride demand across city regions using geo-spatial heatmaps
    CO4, K3
    """
    print("\n" + "="*80)
    print("TASK 4: GEO-SPATIAL HEATMAPS")
    print("="*80)
    
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("\nWarning: folium not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'folium'])
        import folium
        from folium.plugins import HeatMap
    
    # Calculate demand by region
    region_demand = df.groupby('region').size().reset_index(name='demand')
    region_demand = region_demand.sort_values('demand', ascending=False)
    
    print("\nRide Demand by Region:")
    print("-" * 50)
    for _, row in region_demand.iterrows():
        print(f"{row['region']:25s}: {row['demand']:4d} rides")
    
    # Create base map (centered on average coordinates)
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create heatmap data
    heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows()]
    
    # Create map with heatmap
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')
    
    # Add heatmap
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    # Add markers for top regions
    region_coords = df.groupby('region').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'driver_id': 'count'
    }).reset_index()
    region_coords.columns = ['region', 'lat', 'lon', 'count']
    
    for _, row in region_coords.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['count'] / 10,
            popup=f"{row['region']}: {row['count']} rides",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6
        ).add_to(m)
    
    # Save map
    output_path = os.path.join(OUTPUT_DIR, 'task4_geospatial_heatmap.html')
    m.save(output_path)
    print(f"\nInteractive heatmap saved to: {output_path}")
    
    # Create static visualization using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Bar chart of demand by region
    ax1 = axes[0]
    region_demand_sorted = region_demand.sort_values('demand')
    colors = plt.cm.viridis(np.linspace(0, 1, len(region_demand_sorted)))
    ax1.barh(region_demand_sorted['region'], region_demand_sorted['demand'], color=colors)
    ax1.set_xlabel('Number of Rides', fontsize=12)
    ax1.set_ylabel('Region', fontsize=12)
    ax1.set_title('Ride Demand by Region', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Scatter plot with density
    ax2 = axes[1]
    scatter = ax2.scatter(df['longitude'], df['latitude'], 
                         c=df['trip_cost'], s=df['trip_distance']*5,
                         alpha=0.5, cmap='YlOrRd', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title('Ride Distribution by Location\n(Color = Cost, Size = Distance)', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Trip Cost ($)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'task4_geospatial_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Static visualization saved to: {output_path}")
    plt.close()
    
    # Save region statistics
    region_stats = df.groupby('region').agg({
        'trip_distance': ['mean', 'sum'],
        'trip_cost': ['mean', 'sum'],
        'rating': 'mean',
        'driver_id': 'count'
    }).round(2)
    region_stats.columns = ['Avg Distance', 'Total Distance', 'Avg Cost', 'Total Revenue', 'Avg Rating', 'Total Rides']
    region_stats = region_stats.sort_values('Total Rides', ascending=False)
    
    output_path = os.path.join(OUTPUT_DIR, 'task4_region_statistics.txt')
    with open(output_path, 'w') as f:
        f.write("REGION STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(region_stats.to_string())
    
    print(f"Region statistics saved to: {output_path}")
    print("\nTask 4 completed successfully!")

def task5_time_trends(df):
    """
    Task 5: Analyze time-oriented ride demand trends (peak vs. non-peak)
    CO5, K3
    """
    print("\n" + "="*80)
    print("TASK 5: TIME-ORIENTED RIDE DEMAND TRENDS")
    print("="*80)
    
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['date'] = df['timestamp'].dt.date
    
    # Define peak hours (7-9 AM and 5-7 PM)
    df['is_peak'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    df['time_period'] = df['is_peak'].map({True: 'Peak Hours', False: 'Non-Peak Hours'})
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Task 5: Time-Oriented Ride Demand Trends Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Demand by hour of day
    ax1 = axes[0, 0]
    hourly_demand = df.groupby('hour').size()
    ax1.plot(hourly_demand.index, hourly_demand.values, marker='o', linewidth=2, markersize=8)
    ax1.axvspan(7, 9, alpha=0.3, color='red', label='Morning Peak')
    ax1.axvspan(17, 19, alpha=0.3, color='red', label='Evening Peak')
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Number of Rides', fontsize=11)
    ax1.set_title('Ride Demand by Hour of Day', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(range(0, 24, 2))
    
    # 2. Peak vs Non-Peak comparison
    ax2 = axes[0, 1]
    peak_comparison = df.groupby('time_period').agg({
        'driver_id': 'count',
        'trip_cost': 'mean',
        'trip_distance': 'mean',
        'rating': 'mean'
    })
    peak_comparison.columns = ['Ride Count', 'Avg Cost', 'Avg Distance', 'Avg Rating']
    
    x = np.arange(len(peak_comparison.index))
    width = 0.2
    ax2.bar(x - width, peak_comparison['Ride Count'] / peak_comparison['Ride Count'].max(), 
           width, label='Ride Count (normalized)', alpha=0.8)
    ax2.bar(x, peak_comparison['Avg Cost'] / peak_comparison['Avg Cost'].max(), 
           width, label='Avg Cost (normalized)', alpha=0.8)
    ax2.bar(x + width, peak_comparison['Avg Distance'] / peak_comparison['Avg Distance'].max(), 
           width, label='Avg Distance (normalized)', alpha=0.8)
    ax2.set_xlabel('Time Period', fontsize=11)
    ax2.set_ylabel('Normalized Value', fontsize=11)
    ax2.set_title('Peak vs Non-Peak Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(peak_comparison.index)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Demand by day of week
    ax3 = axes[0, 2]
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_demand = df.groupby('day_of_week').size().reindex(day_order)
    colors = ['red' if day in ['Monday', 'Friday'] else 'blue' for day in day_order]
    ax3.bar(weekday_demand.index, weekday_demand.values, color=colors, alpha=0.7)
    ax3.set_xlabel('Day of Week', fontsize=11)
    ax3.set_ylabel('Number of Rides', fontsize=11)
    ax3.set_title('Ride Demand by Day of Week', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Daily demand trend
    ax4 = axes[1, 0]
    daily_demand = df.groupby('date').size()
    ax4.plot(daily_demand.index, daily_demand.values, marker='o', linewidth=1.5, markersize=4)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Number of Rides', fontsize=11)
    ax4.set_title('Daily Ride Demand Trend', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Heatmap: Hour vs Day of Week
    ax5 = axes[1, 1]
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(day_order)
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Ride Count'})
    ax5.set_xlabel('Hour of Day', fontsize=11)
    ax5.set_ylabel('Day of Week', fontsize=11)
    ax5.set_title('Demand Heatmap: Hour vs Day of Week', fontsize=12, fontweight='bold')
    
    # 6. Peak vs Non-Peak statistics
    ax6 = axes[1, 2]
    peak_stats = df.groupby('time_period').agg({
        'trip_cost': 'mean',
        'trip_distance': 'mean',
        'rating': 'mean'
    })
    
    categories = ['Avg Cost ($)', 'Avg Distance (km)', 'Avg Rating']
    peak_values = [peak_stats.loc['Peak Hours', 'trip_cost'],
                   peak_stats.loc['Peak Hours', 'trip_distance'],
                   peak_stats.loc['Peak Hours', 'rating']]
    nonpeak_values = [peak_stats.loc['Non-Peak Hours', 'trip_cost'],
                     peak_stats.loc['Non-Peak Hours', 'trip_distance'],
                     peak_stats.loc['Non-Peak Hours', 'rating']]
    
    x = np.arange(len(categories))
    width = 0.35
    ax6.bar(x - width/2, peak_values, width, label='Peak Hours', alpha=0.8, color='red')
    ax6.bar(x + width/2, nonpeak_values, width, label='Non-Peak Hours', alpha=0.8, color='blue')
    ax6.set_ylabel('Value', fontsize=11)
    ax6.set_title('Peak vs Non-Peak: Key Metrics', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, rotation=15, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'task5_time_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTime trend analysis saved to: {output_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "-"*80)
    print("PEAK VS NON-PEAK STATISTICS:")
    print("-"*80)
    print(f"\nPeak Hours (7-9 AM, 5-7 PM):")
    peak_df = df[df['is_peak'] == True]
    print(f"  Total Rides: {len(peak_df)} ({len(peak_df)/len(df)*100:.2f}%)")
    print(f"  Average Cost: ${peak_df['trip_cost'].mean():.2f}")
    print(f"  Average Distance: {peak_df['trip_distance'].mean():.2f} km")
    print(f"  Average Rating: {peak_df['rating'].mean():.2f}/5.0")
    
    print(f"\nNon-Peak Hours:")
    nonpeak_df = df[df['is_peak'] == False]
    print(f"  Total Rides: {len(nonpeak_df)} ({len(nonpeak_df)/len(df)*100:.2f}%)")
    print(f"  Average Cost: ${nonpeak_df['trip_cost'].mean():.2f}")
    print(f"  Average Distance: {nonpeak_df['trip_distance'].mean():.2f} km")
    print(f"  Average Rating: {nonpeak_df['rating'].mean():.2f}/5.0")
    
    # Save detailed statistics
    time_stats = df.groupby(['hour', 'day_of_week']).agg({
        'driver_id': 'count',
        'trip_cost': 'mean',
        'trip_distance': 'mean'
    }).reset_index()
    time_stats.columns = ['Hour', 'Day', 'Ride Count', 'Avg Cost', 'Avg Distance']
    time_stats = time_stats.sort_values('Ride Count', ascending=False)
    
    output_path = os.path.join(OUTPUT_DIR, 'task5_time_statistics.txt')
    with open(output_path, 'w') as f:
        f.write("TIME-ORIENTED RIDE DEMAND STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write("PEAK VS NON-PEAK SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Peak Hours Rides: {len(peak_df)} ({len(peak_df)/len(df)*100:.2f}%)\n")
        f.write(f"Non-Peak Hours Rides: {len(nonpeak_df)} ({len(nonpeak_df)/len(df)*100:.2f}%)\n\n")
        f.write("TOP 20 BUSIEST HOUR-DAY COMBINATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(time_stats.head(20).to_string(index=False))
    
    print(f"\nDetailed statistics saved to: {output_path}")
    print("\nTask 5 completed successfully!")

def main():
    """Main function to run all analyses"""
    print("\n" + "="*80)
    print("RIDE SHARING DATASET ANALYSIS - COMPLETE ASSIGNMENT")
    print("="*80)
    
    # Load dataset
    df = load_dataset()
    
    # Run all tasks
    task1_dataset_details(df)
    task2_boxplots_density(df)
    task3_wordcloud(df)
    task4_geospatial_heatmap(df)
    task5_time_trends(df)
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll output files saved to: {OUTPUT_DIR}/")
    print("\nGenerated Files:")
    print("  - ride_sharing_dataset.csv (in project root)")
    print(f"  - {OUTPUT_DIR}/task1_dataset_details.txt")
    print(f"  - {OUTPUT_DIR}/task2_boxplots_density.png")
    print(f"  - {OUTPUT_DIR}/task2_cost_distance_scatter.png")
    print(f"  - {OUTPUT_DIR}/task3_wordcloud.png")
    print(f"  - {OUTPUT_DIR}/task3_word_frequency.txt")
    print(f"  - {OUTPUT_DIR}/task4_geospatial_heatmap.html")
    print(f"  - {OUTPUT_DIR}/task4_geospatial_analysis.png")
    print(f"  - {OUTPUT_DIR}/task4_region_statistics.txt")
    print(f"  - {OUTPUT_DIR}/task5_time_trends.png")
    print(f"  - {OUTPUT_DIR}/task5_time_statistics.txt")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

