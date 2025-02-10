import pandas as pd
from geopy.distance import geodesic
from datetime import datetime, timedelta
import numpy as np

def create_location_distance_map(df, max_distance_km):
    # Get unique locations with their coordinates
    unique_locations = df[['location_id', 'latitude', 'longitude']].drop_duplicates()
    location_map = {}
    
    # Create a map of location_id -> list of nearby location_ids and distances
    for idx, row in unique_locations.iterrows():
        base_location = (row['latitude'], row['longitude'])
        base_id = row['location_id']
        nearby = []
        
        for idx2, row2 in unique_locations.iterrows():
            if row['location_id'] != row2['location_id']:
                compare_location = (row2['latitude'], row2['longitude'])
                distance = geodesic(base_location, compare_location).kilometers
                
                if distance <= max_distance_km:
                    nearby.append({
                        'location_id': row2['location_id'],
                        'distance_km': distance
                    })
        
        location_map[base_id] = nearby
    
    return location_map

def process_location_data(df, max_distance_km, max_hours_ago):

    print("Creating location distance map")

    # First create our location distance mapping
    location_distance_map = create_location_distance_map(df, max_distance_km)

    print("Location distance map created")
    
    # Create an empty list to store results
    nearby_locations = []

    print("Processing location data")
    
    # Now process each reading using our pre-computed distance map
    for idx, row in df.iterrows():
        base_id = row['location_id']
        nearby_sensors = location_distance_map[base_id]
        
        print(f"Processing {idx}/{len(df)} with {len(nearby_sensors)} nearby sensors")

        # For each nearby sensor location, find its readings
        for nearby in nearby_sensors:
            nearby_readings = df[df['location_id'] == nearby['location_id']]
            
            # Add each matching reading pair to our results
            for _, nearby_row in nearby_readings.iterrows():
                nearby_locations.append({
                    'base_location_id': base_id,
                    'nearby_location_id': nearby['location_id'],
                    'distance_km': nearby['distance_km'],
                    'base_reading': row['value'],
                    'nearby_reading': nearby_row['value'],
                    'base_time': row['epoch'],
                    'nearby_time': nearby_row['epoch']
                })

    # Convert results to dataframe
    nearby_df = pd.DataFrame(nearby_locations)
    return nearby_df
