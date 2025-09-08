import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

# Load data
df = pd.read_csv('agriculture_crop_yield.csv')

# 1. SCATTERPLOT MATRIX
numerical_vars = ['Area_Hectares', 'Yield_Tonnes', 'Yield_per_Hectare', 'Fertilizer_Usage_kg', 'Precipitation_mm', 'Temperature_Celsius']
sns.pairplot(df[numerical_vars])
plt.suptitle('Scatterplot Matrix: Agriculture Variables', y=1.02)
plt.savefig('scatterplot_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. PARALLEL COORDINATES
df_normalized = df.copy()
for col in numerical_vars:
    df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())

top_states = df.groupby('State')['Total_Revenue'].sum().nlargest(5).index
df_parallel = df_normalized[df_normalized['State'].isin(top_states)]

plt.figure(figsize=(12, 8))
parallel_coordinates(df_parallel[numerical_vars + ['State']], 'State', colormap=plt.cm.Set2)
plt.title('Parallel Coordinates: Agriculture Variables by State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('parallel_coordinates.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. LINE GRAPH
yield_by_crop_year = df.groupby(['Year', 'Crop_Type'])['Yield_Tonnes'].sum().unstack()
yield_by_crop_year.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Line Graph: Yield by Crop Type Over Years')
plt.ylabel('Yield (Tonnes)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_graph.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. STACKED BAR CHART
yield_crop_state = df.groupby(['State', 'Crop_Type'])['Yield_Tonnes'].sum().unstack(fill_value=0)
top_states_yield = yield_crop_state.sum(axis=1).nlargest(8).index
yield_crop_state.loc[top_states_yield].plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Stacked Bar Chart: Yield by Crop Type and State')
plt.ylabel('Yield (Tonnes)')
plt.xticks(rotation=45)
plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('stacked_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete! Files created:")
print("- scatterplot_matrix.png")
print("- parallel_coordinates.png")
print("- line_graph.png")
print("- stacked_bar_chart.png")
