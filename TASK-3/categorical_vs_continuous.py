import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
df = pd.read_csv('agriculture_crop_yield.csv')

# Create visualizations for Task 3c: Categorical vs. Continuous analysis
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# 1. Bar Chart (Summary Statistics)
crop_yield_stats = df.groupby('Crop_Type')['Yield_per_Hectare'].mean().sort_values(ascending=False)
crop_yield_stats.plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Task 3c.1: Bar Chart - Average Yield by Crop Type')
axes[0,0].set_xlabel('Crop Type')
axes[0,0].set_ylabel('Average Yield per Hectare')
axes[0,0].tick_params(axis='x', rotation=45)

# Add value labels
for i, v in enumerate(crop_yield_stats.values):
    axes[0,0].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')

# 2. Grouped Kernel Density Plots
for climate in df['Climate_Zone'].unique():
    subset = df[df['Climate_Zone'] == climate]
    axes[0,1].hist(subset['Yield_per_Hectare'], alpha=0.6, label=climate, density=True, bins=15)
axes[0,1].set_title('Task 3c.2: Grouped Kernel Density - Yield per Hectare by Climate Zone')
axes[0,1].set_xlabel('Yield per Hectare')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()

# 3. Box Plot
df.boxplot(column='Yield_per_Hectare', by='Crop_Type', ax=axes[1,0])
axes[1,0].set_title('Task 3c.3: Box Plot - Yield per Hectare by Crop Type')
axes[1,0].set_xlabel('Crop Type')
axes[1,0].set_ylabel('Yield per Hectare')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Violin Plot
sns.violinplot(data=df, x='Climate_Zone', y='Total_Revenue', ax=axes[1,1])
axes[1,1].set_title('Task 3c.4: Violin Plot - Total Revenue by Climate Zone')
axes[1,1].set_xlabel('Climate Zone')
axes[1,1].set_ylabel('Total Revenue')
axes[1,1].tick_params(axis='x', rotation=45)

# 5. Ridgeline Plot (using overlapping histograms)
for crop in df['Crop_Type'].unique():
    subset = df[df['Crop_Type'] == crop]
    axes[2,0].hist(subset['Yield_per_Hectare'], alpha=0.7, label=crop, density=True, bins=10)
axes[2,0].set_title('Task 3c.5: Ridgeline Plot - Yield per Hectare by Crop Type')
axes[2,0].set_xlabel('Yield per Hectare')
axes[2,0].set_ylabel('Density')
axes[2,0].legend()

# 6. Beeswarm Plot
sns.boxplot(data=df, x='Irrigation_Type', y='Fertilizer_Usage_kg', ax=axes[2,1])
sns.swarmplot(data=df, x='Irrigation_Type', y='Fertilizer_Usage_kg', color='red', alpha=0.6, size=3, ax=axes[2,1])
axes[2,1].set_title('Task 3c.6: Beeswarm Plot - Fertilizer Usage by Irrigation Type')
axes[2,1].set_xlabel('Irrigation Type')
axes[2,1].set_ylabel('Fertilizer Usage (kg)')
axes[2,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('task_3c_categorical_vs_continuous.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical Tests for Task 3c
print("=== TASK 3c: CATEGORICAL vs CONTINUOUS ANALYSIS ===")

# ANOVA test
crop_groups = [group['Yield_per_Hectare'].values for name, group in df.groupby('Crop_Type')]
f_stat, p_value = stats.f_oneway(*crop_groups)
print(f"Task 3c.3 ANOVA (Yield by Crop Type): F = {f_stat:.4f}, p = {p_value:.4f}")

# Kruskal-Wallis test
climate_groups = [group['Total_Revenue'].values for name, group in df.groupby('Climate_Zone')]
h_stat, p_value = stats.kruskal(*climate_groups)
print(f"Task 3c.4 Kruskal-Wallis (Revenue by Climate): H = {h_stat:.4f}, p = {p_value:.4f}")

# Summary for Task 3c
print("\n=== TASK 3c SUMMARY ===")
print("Average Yield by Crop Type:")
for crop, mean_yield in crop_yield_stats.items():
    print(f"  {crop}: {mean_yield:.2f}")

print("\nGenerated: task_3c_categorical_vs_continuous.png") 