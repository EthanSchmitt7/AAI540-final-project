from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openaq import OpenAQ

# Put your OpenAQ API key in a text file with the name "openaq_api_key.txt"
with open("api_keys/openaq_api_key.txt", "r") as file:
    API_KEY = file.read()

# Parameters
RADIUS = 10_000
LIMIT = 3
DATE_RANGE = [datetime.now() - timedelta(days=1), datetime.now()]

# Initialize the OpenAQ client
client = OpenAQ(api_key=API_KEY)


# Function to fetch sensor data near a given location, return a dataframe where each row is a sensor and has data about the location and one column contains the measurements
def fetch_nearby_sensors(latitude, longitude, radius=10_000, limit=3):
    # Get the locations near the given coordinates
    response = client.locations.list(coordinates=(latitude, longitude), radius=radius, limit=limit)

    format_string = "%Y-%m-%dT%H:%M:%SZ"
    data = {}
    # For each location in the response, fetch its sensors
    for location in response.results:
        for sensor in location.sensors:
            lat = location.coordinates.latitude
            long = location.coordinates.longitude
            loc_name = location.name
            location_id = location.id
            sensor = sensor.id

            # Fetch the recent measurements the sensor
            measurements = client.measurements.list(sensor)
            m_id = 0

            # For each measurement, record the relevant data
            for measurement in measurements.results:
                m_id += 1
                epoch = datetime.strptime(measurement.period.datetime_from.utc, format_string)
                duration = timedelta(seconds=pd.to_timedelta(measurement.period.interval).seconds)
                parameter = measurement.parameter.name
                value = measurement.value
                units = measurement.parameter.units

                data[m_id] = {
                    "measurement_id": m_id,
                    "sensor_id": sensor,
                    "location_id": location_id,
                    "location": loc_name,
                    "latitude": lat,
                    "longitude": long,
                    "epoch": epoch,
                    "duration": duration,
                    "parameter": parameter,
                    "value": value,
                    "units": units,
                }

    return pd.DataFrame.from_dict(data, orient="index")


# Example: Get sensors within 10km of Los Angeles (34.0549, -118.2426)
df = fetch_nearby_sensors(34.0549, -118.2426)

# Close the API client
client.close()

# Display results
df.head()
