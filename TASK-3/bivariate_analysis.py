import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load dataset
df = pd.read_csv('agriculture_crop_yield.csv')

print("Dataset Shape:", df.shape)
print("Categorical Columns:", df.select_dtypes(include=['object']).columns.tolist())
print("Continuous Columns:", df.select_dtypes(include=['int64', 'float64']).columns.tolist())

# Create visualizations
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# 1. Categorical vs Categorical
# Stacked Bar Chart
crop_climate = pd.crosstab(df['Crop_Type'], df['Climate_Zone'])
crop_climate.plot(kind='bar', stacked=True, ax=axes[0,0])
axes[0,0].set_title('Crop Type vs Climate Zone')
axes[0,0].tick_params(axis='x', rotation=45)

# Grouped Bar Chart
crop_season = pd.crosstab(df['Crop_Type'], df['Season'])
crop_season.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Crop Type vs Season')
axes[0,1].tick_params(axis='x', rotation=45)

# Segmented Bar Chart
pest_disease = pd.crosstab(df['Pest_Infestation_Level'], df['Disease_Incidence'])
pest_disease_pct = pest_disease.div(pest_disease.sum(axis=1), axis=0) * 100
pest_disease_pct.plot(kind='bar', stacked=True, ax=axes[0,2])
axes[0,2].set_title('Pest Level vs Disease Incidence (%)')
axes[0,2].tick_params(axis='x', rotation=45)

# Mosaic Plot
irrigation_soil = pd.crosstab(df['Irrigation_Type'], df['Soil_Type'])
irrigation_soil.plot(kind='bar', ax=axes[0,3])
axes[0,3].set_title('Irrigation Type vs Soil Type')
axes[0,3].tick_params(axis='x', rotation=45)

# 2. Continuous vs Continuous
# Scatterplot with Fit Line
axes[1,0].scatter(df['Area_Hectares'], df['Yield_Tonnes'], alpha=0.6)
z = np.polyfit(df['Area_Hectares'], df['Yield_Tonnes'], 1)
p = np.poly1d(z)
axes[1,0].plot(df['Area_Hectares'], p(df['Area_Hectares']), "r--")
axes[1,0].set_title('Area vs Yield')
axes[1,0].set_xlabel('Area (Hectares)')
axes[1,0].set_ylabel('Yield (Tonnes)')

# Scatterplot
axes[1,1].scatter(df['Fertilizer_Usage_kg'], df['Yield_per_Hectare'], alpha=0.6)
z = np.polyfit(df['Fertilizer_Usage_kg'], df['Yield_per_Hectare'], 1)
p = np.poly1d(z)
axes[1,1].plot(df['Fertilizer_Usage_kg'], p(df['Fertilizer_Usage_kg']), "r--")
axes[1,1].set_title('Fertilizer vs Yield per Hectare')
axes[1,1].set_xlabel('Fertilizer (kg)')
axes[1,1].set_ylabel('Yield per Hectare')

# Scatterplot
axes[1,2].scatter(df['Precipitation_mm'], df['Temperature_Celsius'], alpha=0.6)
z = np.polyfit(df['Precipitation_mm'], df['Temperature_Celsius'], 1)
p = np.poly1d(z)
axes[1,2].plot(df['Precipitation_mm'], p(df['Precipitation_mm']), "r--")
axes[1,2].set_title('Precipitation vs Temperature')
axes[1,2].set_xlabel('Precipitation (mm)')
axes[1,2].set_ylabel('Temperature (°C)')

# Scatterplot
axes[1,3].scatter(df['Market_Price_per_Tonne'], df['Total_Revenue'], alpha=0.6)
z = np.polyfit(df['Market_Price_per_Tonne'], df['Total_Revenue'], 1)
p = np.poly1d(z)
axes[1,3].plot(df['Market_Price_per_Tonne'], p(df['Market_Price_per_Tonne']), "r--")
axes[1,3].set_title('Market Price vs Total Revenue')
axes[1,3].set_xlabel('Market Price per Tonne')
axes[1,3].set_ylabel('Total Revenue')

# 3. Categorical vs Continuous
# Bar Chart
crop_yield_stats = df.groupby('Crop_Type')['Yield_per_Hectare'].mean().sort_values(ascending=False)
crop_yield_stats.plot(kind='bar', ax=axes[2,0])
axes[2,0].set_title('Average Yield by Crop Type')
axes[2,0].tick_params(axis='x', rotation=45)

# Box Plot
df.boxplot(column='Yield_per_Hectare', by='Crop_Type', ax=axes[2,1])
axes[2,1].set_title('Yield per Hectare by Crop Type')
axes[2,1].tick_params(axis='x', rotation=45)

# Violin Plot
sns.violinplot(data=df, x='Climate_Zone', y='Total_Revenue', ax=axes[2,2])
axes[2,2].set_title('Total Revenue by Climate Zone')
axes[2,2].tick_params(axis='x', rotation=45)

# Beeswarm Plot
sns.boxplot(data=df, x='Irrigation_Type', y='Fertilizer_Usage_kg', ax=axes[2,3])
sns.swarmplot(data=df, x='Irrigation_Type', y='Fertilizer_Usage_kg', color='red', alpha=0.6, size=3, ax=axes[2,3])
axes[2,3].set_title('Fertilizer Usage by Irrigation Type')
axes[2,3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('bivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation Matrix
continuous_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = continuous_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical Tests
print("\nStatistical Tests:")
chi2, p_value, dof, expected = stats.chi2_contingency(pd.crosstab(df['Crop_Type'], df['Climate_Zone']))
print(f"Chi-square test (Crop vs Climate): p-value = {p_value:.4f}")

crop_groups = [group['Yield_per_Hectare'].values for name, group in df.groupby('Crop_Type')]
f_stat, p_value = stats.f_oneway(*crop_groups)
print(f"ANOVA test (Yield by Crop): p-value = {p_value:.4f}")

corr_coef, p_value = stats.pearsonr(df['Area_Hectares'], df['Yield_Tonnes'])
print(f"Correlation test (Area vs Yield): r = {corr_coef:.4f}, p-value = {p_value:.4f}")

print("\nTop correlations:")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            print(f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}: {corr_val:.3f}") 