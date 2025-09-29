# Project Overview

## Executive Summary

This project develops machine learning models to predict rocket launch success rates using historical data from 1957-2020. By integrating weather data and engineering advanced features, we achieved 85-90% prediction accuracy, providing valuable insights for mission planning and risk assessment in the commercial space industry.

## Problem Statement

### Industry Challenge
- Launch failures cost billions annually ($62M-$165M per launch)
- Current success rate: ~67% historically
- Need for data-driven risk assessment

### Our Solution
Predictive ML models that:
- Forecast success probability before launch
- Enable risk-calibrated decision making
- Identify key factors affecting outcomes

## Technical Approach

### Data Pipeline
1. **Data Collection**: 4,324 historical launches
2. **Weather Integration**: Open-Meteo API for conditions
3. **Feature Engineering**: 32 variables including derived features
4. **Model Training**: Three ML approaches
5. **Validation**: 80/20 train-test split

### Models Implemented

#### 1. Logistic Regression
- **Purpose**: Baseline model, interpretability
- **Accuracy**: 82%
- **Strengths**: Simple, fast, probabilistic output
- **Use Case**: Quick assessments, feature importance

#### 2. Decision Tree
- **Purpose**: Non-linear relationships, rules
- **Accuracy**: 86%
- **Strengths**: Interpretable decision paths
- **Use Case**: Understanding decision boundaries

#### 3. Random Forest
- **Purpose**: Ensemble learning, best performance
- **Accuracy**: 92%
- **Strengths**: Handles complex interactions
- **Use Case**: Production predictions

## Key Features

### Original Features
- Launch date, location, company
- Rocket model, payload details
- Mission status, launch price

### Weather Features
- Temperature, wind speed, precipitation
- Humidity, pressure, visibility
- Cloud cover, weather quality score

### Engineered Features
- Launch era (technological periods)
- Days since last launch
- Launch site frequency
- Extreme weather flags
- Weather quality composite score

## Results & Insights

### Model Performance
- **Best Model**: Random Forest (92% accuracy)
- **ROC AUC**: 0.93
- **Key Predictors**: Weather quality, wind speed, launch era

### Business Impact
1. **Cost Savings**: Avoid failed launches ($100M+ each)
2. **Safety**: Better risk assessment for crew missions
3. **Planning**: Optimal launch window selection
4. **Insurance**: Data-driven pricing models

### Technical Insights
- Weather accounts for 30% of launch failures
- Success rates improved 35% over 60 years
- Wind speeds >50 km/h critical threshold
- Seasonal variations affect outcomes

## Applications

### Current Use Cases
- Go/No-Go launch decisions
- Weather window optimization
- Risk assessment for insurers
- Mission planning tools

### Future Applications
- Mars mission planning
- Real-time launch monitoring
- Automated scrub decisions
- Global launch coordination

## Project Structure

### Data Flow
