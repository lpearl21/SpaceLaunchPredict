#!/usr/bin/env python3
"""
Main script for running the rocket launch ML pipeline.
Usage: python main.py [--mode train/predict/full]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.weather_integration import RocketLaunchWeatherIntegrator
from src.models.train_models import ModelTrainer
from src.models.predict import LaunchPredictor
from src.visualization.create_visualizations import create_all_visualizations

def run_data_processing():
    """Process raw data and add weather features."""
    print("=" * 60)
    print("STEP 1: DATA PROCESSING")
    print("=" * 60)
    
    integrator = RocketLaunchWeatherIntegrator(
        input_file='data/raw/Space_Corrected.csv',
        output_file='data/processed/rocket_launches_with_weather_ml_ready.csv'
    )
    
    df = integrator.process_data()
    print(f"✅ Data processing complete. Shape: {df.shape}")
    return df

def run_model_training():
    """Train all ML models."""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    
    trainer = ModelTrainer('data/processed/rocket_launches_with_weather_ml_ready.csv')
    results = trainer.run_pipeline()
    
    print("\n📊 Model Performance Summary:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    return results

def run_prediction_demo():
    """Demo prediction with sample conditions."""
    print("\n" + "=" * 60)
    print("STEP 3: PREDICTION DEMO")
    print("=" * 60)
    
    predictor = LaunchPredictor()
    
    # Sample conditions
    conditions = {
        'temperature_c': 22,
        'windspeed_kmh': 15,
        'precipitation_mm': 0,
        'humidity_percent': 65,
        'LaunchYear': 2024,
        'LaunchDecade': 2020,
        'WeatherQualityScore': 0.85,
        'IsExtremeWeather': 0,
        'LaunchPriceM': 62
    }
    
    print("\n🚀 Launch Conditions:")
    for key, value in conditions.items():
        print(f"  {key}: {value}")
    
    # Get predictions from all models
    results = predictor.predict_with_confidence(conditions)
    
    print("\n📈 Predictions:")
    for model, result in results.items():
        if model == 'Ensemble':
            print(f"\n{model} (Average):")
            print(f"  Probability: {result['probability']:.2%} ± {result.get('std', 0):.2%}")
        else:
            print(f"\n{model}:")
            print(f"  Probability: {result['probability']:.2%}")
        print(f"  Recommendation: {result['recommendation']}")

def run_visualizations():
    """Create all visualizations."""
    print("\n" + "=" * 60)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    create_all_visualizations('data/processed/rocket_launches_with_weather_ml_ready.csv')
    print("✅ Visualizations saved to reports/figures/")

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Rocket Launch ML Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['process', 'train', 'predict', 'viz', 'full'],
        default='full',
        help='Pipeline mode to run'
    )
    
    args = parser.parse_args()
    
    print("\n" + "🚀" * 20)
    print("ROCKET LAUNCH SUCCESS PREDICTION PIPELINE")
    print("🚀" * 20)
    
    if args.mode == 'process':
        run_data_processing()
    elif args.mode == 'train':
        run_model_training()
    elif args.mode == 'predict':
        run_prediction_demo()
    elif args.mode == 'viz':
        run_visualizations()
    else:  # full
        run_data_processing()
        run_model_training()
        run_prediction_demo()
        run_visualizations()
    
    print("\n✨ Pipeline complete!")

if __name__ == "__main__":
    main()
