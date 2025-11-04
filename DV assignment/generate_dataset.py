"""
Dataset Generator for Ride Sharing Analysis
Generates synthetic ride-sharing data with all required fields
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_ride_sharing_dataset(n_records=2000):
    """
    Generate synthetic ride-sharing dataset
    
    Parameters:
    -----------
    n_records : int
        Number of ride records to generate
    
    Returns:
    --------
    pd.DataFrame : Generated dataset
    """
    np.random.seed(42)
    random.seed(42)
    
    # Sample feedback phrases for word cloud
    feedback_phrases = [
        "excellent driver", "very smooth ride", "comfortable car", "professional service",
        "great experience", "punctual driver", "clean vehicle", "safe driving",
        "friendly driver", "quick response", "good music", "pleasant journey",
        "reasonable price", "on time", "highly recommend", "outstanding service",
        "bad experience", "late pickup", "rough ride", "uncomfortable seats",
        "rude driver", "dirty car", "long wait", "expensive fare",
        "traffic delay", "wrong route", "poor navigation", "noisy vehicle",
        "excellent service", "wonderful trip", "amazing driver", "perfect ride",
        "good value", "efficient service", "satisfied customer", "top quality"
    ]
    
    # City regions for geo-spatial analysis
    regions = ['Downtown', 'Midtown', 'Uptown', 'Airport', 'Suburb East', 
               'Suburb West', 'Commercial District', 'Residential North',
               'Residential South', 'University Area']
    
    # Generate data
    data = {
        'driver_id': np.random.randint(1000, 9999, n_records),
        'trip_distance': np.round(np.random.lognormal(2.5, 0.8, n_records), 2),
        'trip_cost': None,  # Will calculate based on distance
        'rating': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
        'rider_feedback': [random.choice(feedback_phrases) for _ in range(n_records)],
        'region': np.random.choice(regions, n_records),
        'latitude': None,  # Will generate based on region
        'longitude': None,  # Will generate based on region
        'timestamp': None,  # Will generate time series data
    }
    
    # Calculate trip cost based on distance (base fare + distance-based)
    base_fare = 2.5
    per_km = 1.2
    data['trip_cost'] = np.round(base_fare + data['trip_distance'] * per_km + 
                                 np.random.normal(0, 2, n_records), 2)
    data['trip_cost'] = np.maximum(data['trip_cost'], 3.0)  # Minimum fare
    
    # Generate geo-coordinates based on regions
    region_coords = {
        'Downtown': (40.7128, -74.0060),
        'Midtown': (40.7549, -73.9840),
        'Uptown': (40.7831, -73.9712),
        'Airport': (40.6413, -73.7781),
        'Suburb East': (40.6782, -73.9442),
        'Suburb West': (40.7282, -74.0776),
        'Commercial District': (40.7589, -73.9851),
        'Residential North': (40.8612, -73.9292),
        'Residential South': (40.6943, -73.9563),
        'University Area': (40.8075, -73.9626),
    }
    
    latitudes = []
    longitudes = []
    for region in data['region']:
        base_lat, base_lon = region_coords[region]
        # Add small random variations
        lat = base_lat + np.random.uniform(-0.05, 0.05)
        lon = base_lon + np.random.uniform(-0.05, 0.05)
        latitudes.append(lat)
        longitudes.append(lon)
    
    data['latitude'] = latitudes
    data['longitude'] = longitudes
    
    # Generate timestamps for time-series analysis
    start_date = datetime(2024, 1, 1)
    timestamps = []
    for i in range(n_records):
        # Generate dates over 3 months
        days_offset = np.random.randint(0, 90)
        hours_offset = np.random.randint(0, 24)
        minutes_offset = np.random.randint(0, 60)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
        timestamps.append(timestamp)
    
    data['timestamp'] = timestamps
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating ride-sharing dataset...")
    df = generate_ride_sharing_dataset(2000)
    
    # Save to CSV
    df.to_csv('ride_sharing_dataset.csv', index=False)
    print(f"Dataset generated successfully with {len(df)} records!")
    print(f"\nDataset saved to: ride_sharing_dataset.csv")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few records:")
    print(df.head())

