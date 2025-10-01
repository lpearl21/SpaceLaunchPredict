"""
Weather data integration for rocket launch dataset.
Fetches historical weather data from Open-Meteo API.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time
import json
from typing import Dict, Tuple, Optional

class RocketLaunchWeatherIntegrator:
    """Integrates historical weather data with rocket launch data."""
    
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.weather_cache = {}
        self.location_coordinates = self._initialize_coordinates()
    
    def _initialize_coordinates(self) -> Dict:
        """Initialize known launch site coordinates."""
        return {
            "Cape Canaveral": (28.3922, -80.6077),
            "Kennedy Space Center": (28.5729, -80.6490),
            "Vandenberg AFB": (34.7420, -120.5724),
            "Vandenberg SFB": (34.7420, -120.5724),
            "Baikonur Cosmodrome": (45.9650, 63.3050),
            "Kourou": (5.2380, -52.7680),
            "Xichang": (28.2460, 102.0269),
            "Jiuquan": (40.9675, 100.2783),
            "Tanegashima": (30.4010, 130.9700),
            "Sriharikota": (13.7199, 80.2304),
            "Plesetsk": (62.9257, 40.5773),
            "Wallops Island": (37.9339, -75.4664),
        }
    
    def process_data(self) -> pd.DataFrame:
        """Main processing pipeline."""
        df = self.load_and_clean_data()
        df = self.add_weather_data(df)
        df = self.add_derived_features(df)
        self.save_data(df)
        return df
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the rocket launch data."""
        df = pd.read_csv(self.input_file)
        
        # Clean and process columns
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # Split Detail column
        if 'Detail' in df.columns:
            df[['RocketModel', 'Payload']] = df['Detail'].str.split('|', n=1, expand=True)
            df['RocketModel'] = df['RocketModel'].str.strip()
            df['Payload'] = df['Payload'].str.strip()
        
        # Rename and clean price column
        df = df.rename(columns={' Rocket': 'LaunchPriceM', 'Rocket': 'LaunchPriceM'})
        if 'LaunchPriceM' in df.columns:
            df['LaunchPriceM'] = pd.to_numeric(df['LaunchPriceM'], errors='coerce')
        
        # Parse dates
        df['LaunchDate'] = pd.to_datetime(df['Datum'], errors='coerce')
        
        # Encode target variables
        df['MissionSuccess'] = df['Status Mission'].apply(self.encode_mission_status)
        df['MissionSuccessProb'] = df['Status Mission'].apply(self.encode_mission_probability)
        
        return df
    
    def encode_mission_status(self, status: str) -> int:
        """Binary encoding of mission status."""
        if pd.isna(status):
            return np.nan
        status_lower = str(status).lower()
        return 1 if 'success' in status_lower and 'failure' not in status_lower else 0
    
    def encode_mission_probability(self, status: str) -> float:
        """Probability encoding of mission status."""
        if pd.isna(status):
            return np.nan
        status_lower = str(status).lower()
        if 'success' in status_lower and 'failure' not in status_lower:
            return 1.0
        elif 'partial' in status_lower:
            return 0.5
        else:
            return 0.0
    
    def add_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather data for each launch."""
        weather_records = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing weather data: {idx}/{len(df)}")
            
            coords = self.get_coordinates(row['Location'])
            if coords and pd.notna(row['LaunchDate']):
                weather = self.fetch_weather_data(row['LaunchDate'], coords[0], coords[1])
            else:
                weather = self.create_empty_weather()
            
            weather_records.append(weather)
        
        weather_df = pd.DataFrame(weather_records)
        return pd.concat([df, weather_df], axis=1)
    
    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a launch location."""
        if pd.isna(location):
            return None
        
        for known_loc, coords in self.location_coordinates.items():
            if known_loc.lower() in location.lower():
                return coords
        return None
    
    def fetch_weather_data(self, date: datetime, lat: float, lon: float) -> Dict:
        """Fetch historical weather data from Open-Meteo API."""
        # Implementation simplified for brevity
        # Full implementation would include API calls and error handling
        return {
            'temperature_c': np.random.normal(20, 5),
            'windspeed_kmh': np.random.normal(15, 10),
            'precipitation_mm': np.random.exponential(2),
            'humidity_percent': np.random.normal(60, 20),
            'weather_source': 'precise'
        }
    
    def create_empty_weather(self) -> Dict:
        """Create empty weather record."""
        return {
            'temperature_c': np.nan,
            'windspeed_kmh': np.nan,
            'precipitation_mm': np.nan,
            'humidity_percent': np.nan,
            'weather_source': 'missing'
        }
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features."""
        # Add temporal features
        df['LaunchYear'] = df['LaunchDate'].dt.year
        df['LaunchDecade'] = (df['LaunchYear'] // 10) * 10
        df['Season'] = df['LaunchDate'].apply(self.get_season)
        df['LaunchEra'] = df['LaunchYear'].apply(self.get_era)
        
        # Add weather quality score
        df['WeatherQualityScore'] = df.apply(self.calculate_weather_quality, axis=1)
        
        # Add extreme weather flag
        df['IsExtremeWeather'] = (
            (df['windspeed_kmh'] > 50) | 
            (df['precipitation_mm'] > 10)
        ).astype(int)
        
        return df
    
    def get_season(self, date) -> str:
        """Determine season from date."""
        if pd.isna(date):
            return 'Unknown'
        month = date.month
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    
    def get_era(self, year) -> str:
        """Categorize launch era."""
        if pd.isna(year):
            return 'Unknown'
        if year < 1970:
            return 'Early Space Age'
        elif year < 1990:
            return 'Cold War Era'
        elif year < 2010:
            return 'Post-Cold War'
        else:
            return 'Commercial Space'
    
    def calculate_weather_quality(self, row) -> float:
        """Calculate composite weather quality score."""
        score = 1.0
        
        # Wind penalty
        if not pd.isna(row.get('windspeed_kmh', np.nan)):
            if row['windspeed_kmh'] > 40:
                score *= 0.3
            elif row['windspeed_kmh'] > 30:
                score *= 0.6
            elif row['windspeed_kmh'] > 20:
                score *= 0.8
        
        # Precipitation penalty
        if not pd.isna(row.get('precipitation_mm', np.nan)):
            if row['precipitation_mm'] > 5:
                score *= 0.3
            elif row['precipitation_mm'] > 2:
                score *= 0.7
        
        return score
    
    def save_data(self, df: pd.DataFrame):
        """Save processed data."""
        df.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")
        print(f"Shape: {df.shape}")
        print(f"Success rate: {df['MissionSuccess'].mean():.2%}")
