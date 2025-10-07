#!/usr/bin/env python3
"""
Minimal Spatial Analysis of Agriculture Crop Data
Task 8: CO4, S3 - Geographical Map, Map Projections
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

def analyze_crop_data():
    """Minimal spatial analysis of agriculture crop data"""
    
    # Step 1: Load and prepare data
    print("📊 Loading agriculture crop data...")
    df = pd.read_csv('agriculture_crop_yield.csv')
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Step 2: State-wise analysis
    print("🗺️ Creating state-wise visualizations...")
    state_summary = df.groupby('State').agg({
        'Yield_Tonnes': 'sum',
        'Yield_per_Hectare': 'mean',
        'Total_Revenue': 'sum'
    }).reset_index()
    
    # Step 3: Create choropleth map
    fig = px.choropleth(
        state_summary,
        locations='State',
        locationmode='USA-states',
        color='Yield_per_Hectare',
        title='Crop Yield per Hectare by State',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(geo=dict(scope='usa'))
    fig.write_html('output/yield_map.html')
    print("✅ Map saved: output/yield_map.html")
    
    # Step 4: Create bar chart
    plt.figure(figsize=(12, 6))
    top_states = state_summary.nlargest(10, 'Yield_per_Hectare')
    plt.bar(top_states['State'], top_states['Yield_per_Hectare'])
    plt.title('Top 10 States by Yield per Hectare')
    plt.xlabel('State')
    plt.ylabel('Yield per Hectare (Tonnes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/top_states.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: output/top_states.png")
    
    # Step 5: Generate insights
    print("\n💡 Key Insights:")
    print(f"Total Records: {len(df)}")
    print(f"States: {df['State'].nunique()}")
    print(f"Crop Types: {df['Crop_Type'].nunique()}")
    
    top_state = state_summary.loc[state_summary['Yield_per_Hectare'].idxmax()]
    print(f"Best State: {top_state['State']} ({top_state['Yield_per_Hectare']:.2f} tonnes/hectare)")
    
    climate_perf = df.groupby('Climate_Zone')['Yield_per_Hectare'].mean()
    best_climate = climate_perf.idxmax()
    print(f"Best Climate: {best_climate} ({climate_perf[best_climate]:.2f} tonnes/hectare)")
    
    print("\n✅ Analysis complete! Check 'output' folder for results.")

if __name__ == "__main__":
    analyze_crop_data()

