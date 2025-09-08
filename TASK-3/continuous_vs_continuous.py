import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load dataset
df = pd.read_csv('agriculture_crop_yield.csv')

# Create scatterplots with fit lines
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Variable pairs for Task 3b: Continuous vs. Continuous analysis
pairs = [
    ('Area_Hectares', 'Yield_Tonnes', 'Task 3b.1: Area vs Yield', 'Area (Hectares)', 'Yield (Tonnes)'),
    ('Fertilizer_Usage_kg', 'Yield_per_Hectare', 'Task 3b.2: Fertilizer vs Yield per Hectare', 'Fertilizer Usage (kg)', 'Yield per Hectare'),
    ('Precipitation_mm', 'Temperature_Celsius', 'Task 3b.3: Precipitation vs Temperature', 'Precipitation (mm)', 'Temperature (°C)'),
    ('Market_Price_per_Tonne', 'Total_Revenue', 'Task 3b.4: Market Price vs Total Revenue', 'Market Price per Tonne', 'Total Revenue')
]

colors = ['blue', 'green', 'orange', 'purple']

# Create all plots for Task 3b
for i, ((x_col, y_col, title, x_label, y_label), color) in enumerate(zip(pairs, colors)):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Scatter plot
    ax.scatter(df[x_col], df[y_col], alpha=0.6, color=color)
    
    # Fit line
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "r--", linewidth=2)
    
    # Labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Correlation
    corr_coef, p_value = stats.pearsonr(df[x_col], df[y_col])
    ax.text(0.05, 0.95, f'r = {corr_coef:.3f}\np = {p_value:.4f}', 
           transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('task_3b_scatterplot_fit_lines.png', dpi=300, bbox_inches='tight')
plt.show()

# Print Task 3b summary
print("=== TASK 3b: CONTINUOUS vs CONTINUOUS ANALYSIS ===")
print("Scatterplot Fit Lines created for:")
for x_col, y_col, title, _, _ in pairs:
    corr_coef, p_value = stats.pearsonr(df[x_col], df[y_col])
    print(f"{title}: r = {corr_coef:.3f}, p = {p_value:.4f}")

print("\nGenerated: task_3b_scatterplot_fit_lines.png") 