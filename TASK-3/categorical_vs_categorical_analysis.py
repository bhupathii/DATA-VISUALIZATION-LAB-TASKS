import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and prepare data"""
    df = pd.read_csv('agriculture_crop_yield.csv')
    
    # Create categorical variables
    df['Yield_Category'] = pd.cut(df['Yield_per_Hectare'], bins=[0, 3, 5, 8], labels=['Low', 'Medium', 'High'])
    df['Revenue_Category'] = pd.cut(df['Total_Revenue'], bins=[0, 50000000, 150000000, 500000000], labels=['Low', 'Medium', 'High'])
    df['Area_Category'] = pd.cut(df['Area_Hectares'], bins=[0, 100000, 200000, 400000], labels=['Small', 'Medium', 'Large'])
    
    return df

def create_task_3a_charts(df):
    """Task 3a: Create Categorical vs. Categorical charts"""
    
    # 1. Stacked Bar Chart
    plt.figure(figsize=(12, 8))
    pivot1 = df.groupby(['Crop_Type', 'Season'])['Yield_Category'].value_counts().unstack(fill_value=0)
    pivot1.plot(kind='bar', stacked=True)
    plt.title('Task 3a.1: Stacked Bar Chart - Crop Type vs Season with Yield Categories')
    plt.xlabel('Crop Type and Season')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('task_3a_1_stacked_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Grouped Bar Chart
    plt.figure(figsize=(14, 8))
    pivot2 = df.groupby(['Climate_Zone', 'Crop_Type'])['Revenue_Category'].value_counts().unstack(fill_value=0)
    pivot2.plot(kind='bar')
    plt.title('Task 3a.2: Grouped Bar Chart - Climate Zone vs Crop Type with Revenue Categories')
    plt.xlabel('Climate Zone and Crop Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('task_3a_2_grouped_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Segmented Bar Chart
    plt.figure(figsize=(16, 10))
    top_states = df.groupby('State')['Area_Hectares'].sum().nlargest(15).index
    df_filtered = df[df['State'].isin(top_states)]
    pivot3 = df_filtered.groupby(['State', 'Crop_Type'])['Area_Category'].value_counts().unstack(fill_value=0)
    pivot3.plot(kind='bar', stacked=True)
    plt.title('Task 3a.3: Segmented Bar Chart - State vs Crop Type with Area Categories (Top 15 States)')
    plt.xlabel('State and Crop Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('task_3a_3_segmented_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Mosaic Plot (4 panels)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Panel 1: Crop Type vs Season
    pd.crosstab(df['Crop_Type'], df['Season']).plot(kind='bar', ax=ax1, stacked=True)
    ax1.set_title('Task 3a.4.1: Crop Type vs Season')
    ax1.tick_params(axis='x', rotation=45)
    
    # Panel 2: Climate Zone vs Soil Type
    pd.crosstab(df['Climate_Zone'], df['Soil_Type']).plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title('Task 3a.4.2: Climate Zone vs Soil Type')
    ax2.tick_params(axis='x', rotation=45)
    
    # Panel 3: Irrigation Type vs Pest Level
    pd.crosstab(df['Irrigation_Type'], df['Pest_Infestation_Level']).plot(kind='bar', ax=ax3, stacked=True)
    ax3.set_title('Task 3a.4.3: Irrigation Type vs Pest Level')
    ax3.tick_params(axis='x', rotation=45)
    
    # Panel 4: Disease vs Yield
    pd.crosstab(df['Disease_Incidence'], df['Yield_Category']).plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('Task 3a.4.4: Disease vs Yield Category')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('task_3a_4_mosaic_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for Task 3a: Categorical vs. Categorical Analysis"""
    print("=== TASK 3a: CATEGORICAL vs CATEGORICAL ANALYSIS ===")
    print("Creating required diagrams:")
    print("1. Stacked Bar Chart")
    print("2. Grouped Bar Chart")
    print("3. Segmented Bar Chart")
    print("4. Mosaic Plot (4 panels)")
    
    print("\nLoading data...")
    df = load_data()
    
    print("Creating Task 3a visualizations...")
    create_task_3a_charts(df)
    
    print("\n=== TASK 3a COMPLETE ===")
    print("Generated files:")
    print("- task_3a_1_stacked_bar_chart.png")
    print("- task_3a_2_grouped_bar_chart.png")
    print("- task_3a_3_segmented_bar_chart.png")
    print("- task_3a_4_mosaic_plot.png")

if __name__ == "__main__":
    main()
