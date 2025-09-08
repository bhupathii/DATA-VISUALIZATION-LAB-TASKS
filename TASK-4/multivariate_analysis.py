import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('agriculture_crop_yield.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nSummary Statistics:")
print(df.describe())

# 1. SCATTERPLOT MATRIX
print("\n" + "="*50)
print("1. SCATTERPLOT MATRIX")
print("="*50)

# Select numerical variables for scatterplot matrix
numerical_vars = ['Area_Hectares', 'Yield_Tonnes', 'Yield_per_Hectare', 
                  'Fertilizer_Usage_kg', 'Precipitation_mm', 'Temperature_Celsius',
                  'Storage_Loss_Percentage', 'Market_Price_per_Tonne', 'Total_Revenue']

# Create scatterplot matrix
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
fig.suptitle('Scatterplot Matrix: Multivariate Analysis of Agriculture Variables', 
             fontsize=16, fontweight='bold', y=0.95)

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Create scatterplots for each pair of variables
for i, var1 in enumerate(numerical_vars):
    for j, var2 in enumerate(numerical_vars):
        if i != j:
            idx = i * len(numerical_vars) + j
            if idx < len(axes_flat):
                ax = axes_flat[idx]
                ax.scatter(df[var1], df[var2], alpha=0.6, s=30)
                ax.set_xlabel(var1.replace('_', ' '))
                ax.set_ylabel(var2.replace('_', ' '))
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Add correlation coefficient
                corr = df[var1].corr(df[var2])
                ax.text(0.05, 0.95, f'ρ = {corr:.2f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Remove empty subplots
for i in range(len(numerical_vars)**2, len(axes_flat)):
    fig.delaxes(axes_flat[i])

plt.tight_layout()
plt.savefig('scatterplot_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. PARALLEL COORDINATES PLOT
print("\n" + "="*50)
print("2. PARALLEL COORDINATES PLOT")
print("="*50)

# Prepare data for parallel coordinates
# Normalize numerical variables for better visualization
df_normalized = df.copy()
for col in numerical_vars:
    df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())

# Select subset of data for clearer visualization (top states by total revenue)
top_states = df.groupby('State')['Total_Revenue'].sum().nlargest(8).index
df_parallel = df_normalized[df_normalized['State'].isin(top_states)]

# Create parallel coordinates plot
fig, ax = plt.subplots(figsize=(16, 10))

# Use a subset of variables for clarity
parallel_vars = ['Area_Hectares', 'Yield_Tonnes', 'Yield_per_Hectare', 
                'Fertilizer_Usage_kg', 'Precipitation_mm', 'Temperature_Celsius']

# Create parallel coordinates plot
parallel_coordinates(df_parallel[parallel_vars + ['State']], 'State', 
                    colormap=plt.cm.Set2, ax=ax)

ax.set_title('Parallel Coordinates Plot: Agriculture Variables by State', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('parallel_coordinates.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. LINE GRAPH - Time Series Analysis
print("\n" + "="*50)
print("3. LINE GRAPH - Time Series Analysis")
print("="*50)

# Create line graphs showing trends over years
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Line Graphs: Agricultural Trends Over Time (2020-2021)', 
             fontsize=16, fontweight='bold', y=0.95)

# 3.1 Total Yield by Crop Type over Years
yield_by_crop_year = df.groupby(['Year', 'Crop_Type'])['Yield_Tonnes'].sum().unstack()
yield_by_crop_year.plot(kind='line', marker='o', ax=axes[0,0], linewidth=3, markersize=8)
axes[0,0].set_title('Total Yield by Crop Type Over Years', fontweight='bold')
axes[0,0].set_ylabel('Yield (Tonnes)')
axes[0,0].legend(title='Crop Type')
axes[0,0].grid(True, alpha=0.3)

# 3.2 Average Yield per Hectare by State over Years
avg_yield_by_state = df.groupby(['Year', 'State'])['Yield_per_Hectare'].mean().unstack()
# Select top 5 states for clarity
top_states_yield = avg_yield_by_state.mean().nlargest(5).index
avg_yield_by_state[top_states_yield].plot(kind='line', marker='s', ax=axes[0,1], linewidth=3, markersize=8)
axes[0,1].set_title('Average Yield per Hectare by Top States', fontweight='bold')
axes[0,1].set_ylabel('Yield per Hectare')
axes[0,1].legend(title='State')
axes[0,1].grid(True, alpha=0.3)

# 3.3 Total Revenue by Climate Zone over Years
revenue_by_climate = df.groupby(['Year', 'Climate_Zone'])['Total_Revenue'].sum().unstack()
revenue_by_climate.plot(kind='line', marker='^', ax=axes[1,0], linewidth=3, markersize=8)
axes[1,0].set_title('Total Revenue by Climate Zone Over Years', fontweight='bold')
axes[1,0].set_ylabel('Total Revenue ($)')
axes[1,0].legend(title='Climate Zone')
axes[1,0].grid(True, alpha=0.3)

# 3.4 Average Temperature and Precipitation by Year
weather_by_year = df.groupby('Year')[['Temperature_Celsius', 'Precipitation_mm']].mean()
weather_by_year.plot(kind='line', marker='D', ax=axes[1,1], linewidth=3, markersize=8)
axes[1,1].set_title('Average Weather Conditions Over Years', fontweight='bold')
axes[1,1].set_ylabel('Temperature (°C) / Precipitation (mm)')
axes[1,1].legend(['Temperature (°C)', 'Precipitation (mm)'])
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('line_graphs.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. STACKED BAR CHART
print("\n" + "="*50)
print("4. STACKED BAR CHART")
print("="*50)

# Create multiple stacked bar charts
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Stacked Bar Charts: Multivariate Analysis of Agriculture Data', 
             fontsize=16, fontweight='bold', y=0.95)

# 4.1 Stacked Bar: Yield by Crop Type and State
yield_crop_state = df.groupby(['State', 'Crop_Type'])['Yield_Tonnes'].sum().unstack(fill_value=0)
# Select top 10 states by total yield
top_states_yield_total = yield_crop_state.sum(axis=1).nlargest(10).index
yield_crop_state.loc[top_states_yield_total].plot(kind='bar', stacked=True, ax=axes[0,0])
axes[0,0].set_title('Yield by Crop Type and State', fontweight='bold')
axes[0,0].set_ylabel('Yield (Tonnes)')
axes[0,0].legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,0].tick_params(axis='x', rotation=45)

# 4.2 Stacked Bar: Revenue by Climate Zone and Season
revenue_climate_season = df.groupby(['Climate_Zone', 'Season'])['Total_Revenue'].sum().unstack(fill_value=0)
revenue_climate_season.plot(kind='bar', stacked=True, ax=axes[0,1])
axes[0,1].set_title('Revenue by Climate Zone and Season', fontweight='bold')
axes[0,1].set_ylabel('Total Revenue ($)')
axes[0,1].legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].tick_params(axis='x', rotation=45)

# 4.3 Stacked Bar: Area by Soil Type and Irrigation Type
area_soil_irrigation = df.groupby(['Soil_Type', 'Irrigation_Type'])['Area_Hectares'].sum().unstack(fill_value=0)
area_soil_irrigation.plot(kind='bar', stacked=True, ax=axes[1,0])
axes[1,0].set_title('Area by Soil Type and Irrigation Type', fontweight='bold')
axes[1,0].set_ylabel('Area (Hectares)')
axes[1,0].legend(title='Irrigation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,0].tick_params(axis='x', rotation=45)

# 4.4 Stacked Bar: Fertilizer Usage by Pest Level and Disease Incidence
fertilizer_pest_disease = df.groupby(['Pest_Infestation_Level', 'Disease_Incidence'])['Fertilizer_Usage_kg'].sum().unstack(fill_value=0)
fertilizer_pest_disease.plot(kind='bar', stacked=True, ax=axes[1,1])
axes[1,1].set_title('Fertilizer Usage by Pest Level and Disease Incidence', fontweight='bold')
axes[1,1].set_ylabel('Fertilizer Usage (kg)')
axes[1,1].legend(title='Disease Incidence', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('stacked_bar_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. ADDITIONAL ANALYSIS: Correlation Heatmap
print("\n" + "="*50)
print("5. CORRELATION HEATMAP")
print("="*50)

# Create correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_vars].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap: Agriculture Variables', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. SUMMARY STATISTICS BY CATEGORIES
print("\n" + "="*50)
print("6. SUMMARY STATISTICS BY CATEGORIES")
print("="*50)

# Summary by Crop Type
print("\nSummary by Crop Type:")
crop_summary = df.groupby('Crop_Type').agg({
    'Area_Hectares': ['mean', 'sum'],
    'Yield_Tonnes': ['mean', 'sum'],
    'Yield_per_Hectare': 'mean',
    'Total_Revenue': ['mean', 'sum']
}).round(2)
print(crop_summary)

# Summary by Climate Zone
print("\nSummary by Climate Zone:")
climate_summary = df.groupby('Climate_Zone').agg({
    'Temperature_Celsius': 'mean',
    'Precipitation_mm': 'mean',
    'Yield_per_Hectare': 'mean',
    'Total_Revenue': 'sum'
}).round(2)
print(climate_summary)

# Summary by State
print("\nTop 10 States by Total Revenue:")
state_revenue = df.groupby('State')['Total_Revenue'].sum().sort_values(ascending=False).head(10)
print(state_revenue)

print("\nAnalysis Complete! All visualizations have been saved as PNG files.")
print("Files created:")
print("- scatterplot_matrix.png")
print("- parallel_coordinates.png") 
print("- line_graphs.png")
print("- stacked_bar_charts.png")
print("- correlation_heatmap.png")
