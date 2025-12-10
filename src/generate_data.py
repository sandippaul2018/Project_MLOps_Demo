"""
Data Generation Script for Tobacco Curing Barn MLOps
Generates synthetic sensor data from tobacco curing barns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def generate_curing_data(n_samples=500, random_state=42):
    """
    Generate synthetic tobacco curing barn data
    
    Args:
        n_samples: Number of records to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic curing data
    """
    np.random.seed(random_state)
    
    # Generate sensor data with realistic distributions
    barn_temperature = np.random.normal(loc=70, scale=5, size=n_samples)  # Normal ~70°C, std=5
    barn_humidity = np.random.normal(loc=60, scale=8, size=n_samples)      # Normal ~60%, std=8
    leaf_moisture = np.random.normal(loc=18, scale=3, size=n_samples)      # Normal ~18%, std=3
    airflow_rate = np.random.uniform(low=5, high=25, size=n_samples)       # Uniform 5-25 CFM
    curing_time_hours = np.random.uniform(low=48, high=240, size=n_samples)  # 2-10 days
    
    # Create DataFrame
    df = pd.DataFrame({
        'barn_temperature': barn_temperature,
        'barn_humidity': barn_humidity,
        'leaf_moisture': leaf_moisture,
        'airflow_rate': airflow_rate,
        'curing_time_hours': curing_time_hours
    })
    
    # Generate quality grade based on logic:
    # If temp > 78 OR humidity < 45, grade is 'Standard', otherwise 'Premium'
    df['quality_grade'] = df.apply(
        lambda row: 'Standard' if (row['barn_temperature'] > 78 or row['barn_humidity'] < 45) else 'Premium',
        axis=1
    )
    
    # Round numeric columns for readability
    df['barn_temperature'] = df['barn_temperature'].round(2)
    df['barn_humidity'] = df['barn_humidity'].round(2)
    df['leaf_moisture'] = df['leaf_moisture'].round(2)
    df['airflow_rate'] = df['airflow_rate'].round(2)
    df['curing_time_hours'] = df['curing_time_hours'].round(2)
    
    return df

def main():
    """Main function to generate and save curing data"""
    print("=" * 60)
    print("Tobacco Curing Barn Data Generation")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating synthetic data for 500 curing cycles...")
    df = generate_curing_data(n_samples=500, random_state=42)
    
    # Save to root folder as curing_data.csv
    output_path = Path(__file__).parent.parent / 'curing_data.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Data saved to: {output_path}")
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    print("\nStatistical Summary:")
    print(df.describe().round(2))
    
    print("\nQuality Grade Distribution:")
    grade_counts = df['quality_grade'].value_counts()
    for grade, count in grade_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {grade}: {count} ({percentage:.1f}%)")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save metadata
    metadata = {
        'total_records': len(df),
        'columns': df.columns.tolist(),
        'quality_grades': df['quality_grade'].unique().tolist(),
        'grade_distribution': grade_counts.to_dict(),
        'random_seed': 42,
        'generation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = Path(__file__).parent.parent / 'curing_data_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
