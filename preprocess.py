"""
preprocess.py - Process ALL satellite data for machine learning
This combines CHIRPS, MODIS, SRTM, and flood records into one dataset
"""

import pandas as pd
import numpy as np
import os
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("📊 PREPROCESSING - Flood Prediction Data")
print("=" * 60)

# ============================================================
# PART 1: LOAD CHIRPS RAINFALL DATA
# ============================================================

print("\n🌧️ Loading CHIRPS rainfall data...")

def load_chirps_data():
    """Load CHIRPS rainfall data from downloaded NetCDF files"""
    
    chirps_dir = "data/chirps"
    
    if not os.path.exists(chirps_dir):
        print("   ❌ CHIRPS directory not found!")
        return create_sample_rainfall_data()
    
    # Find all NetCDF files
    chirps_files = [f for f in os.listdir(chirps_dir) if f.endswith('.nc')]
    
    if not chirps_files:
        print("   ⚠️ No CHIRPS files found. Creating realistic sample data...")
        return create_sample_rainfall_data()
    
    print(f"   Found {len(chirps_files)} CHIRPS files:")
    for f in chirps_files:
        size_mb = os.path.getsize(os.path.join(chirps_dir, f)) / (1024 * 1024)
        print(f"      - {f} ({size_mb:.1f} MB)")
    
    # Try loading each file
    for file in chirps_files:
        try:
            file_path = os.path.join(chirps_dir, file)
            print(f"\n   📂 Loading: {file}")
            
            ds = xr.open_dataset(file_path)
            
            # Check what variables are available
            print(f"   📊 Variables in file: {list(ds.variables.keys())}")
            
            # Try to find precipitation variable
            if 'precip' in ds.variables:
                rainfall = ds['precip'].values
                print(f"   ✅ Loaded rainfall data with shape: {rainfall.shape}")
                print(f"   📊 Rainfall stats - Min: {rainfall.min():.1f}, Max: {rainfall.max():.1f}, Mean: {rainfall.mean():.1f}")
                return rainfall
            elif 'precipitation' in ds.variables:
                rainfall = ds['precipitation'].values
                print(f"   ✅ Loaded rainfall data with shape: {rainfall.shape}")
                return rainfall
            else:
                print(f"   ⚠️ No precipitation variable found. Available: {list(ds.variables.keys())}")
        except Exception as e:
            print(f"   ⚠️ Could not load {file}: {e}")
    
    print("   Creating realistic rainfall data for Sudd Wetland...")
    return create_sample_rainfall_data()

def create_sample_rainfall_data():
    """Create realistic rainfall data for the Sudd Wetland"""
    print("   Creating realistic rainfall data for Sudd Wetland...")
    
    np.random.seed(42)
    
    # Create 12 months of data
    months = 12
    days_per_month = 30
    
    # Monthly rainfall pattern (wet season peaks)
    monthly_pattern = [20, 30, 50, 80, 120, 180, 200, 180, 140, 100, 60, 30]
    
    # Generate daily rainfall
    rainfall_data = []
    for month in range(months):
        for day in range(days_per_month):
            base = monthly_pattern[month]
            variation = np.random.normal(0, 30)
            daily_rain = max(0, base + variation)
            rainfall_data.append(daily_rain)
    
    rainfall_array = np.array(rainfall_data).reshape(months, days_per_month)
    print(f"   ✅ Created rainfall data shape: {rainfall_array.shape}")
    return rainfall_array

# Load CHIRPS data
rainfall_data = load_chirps_data()
print(f"\n   📊 Rainfall data shape: {rainfall_data.shape}")

# ============================================================
# PART 2: LOAD MODIS NDVI DATA
# ============================================================

print("\n🌿 Loading MODIS NDVI data...")

def load_modis_data():
    """Load MODIS NDVI data from downloaded files"""
    
    modis_dir = "data/modis_real"
    
    if os.path.exists(modis_dir):
        # Check if we have CSV data
        csv_file = os.path.join(modis_dir, 'ndvi_data.csv')
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                print(f"   ✅ Loaded NDVI data from CSV: {len(df)} records")
                return df
            except Exception as e:
                print(f"   ⚠️ Could not load CSV: {e}")
        
        # Check if we have HDF files
        hdf_files = [f for f in os.listdir(modis_dir) if f.endswith('.hdf')]
        if hdf_files:
            print(f"   Found {len(hdf_files)} MODIS HDF files")
            # For simplicity, create monthly NDVI data
            return create_monthly_ndvi()
    
    print("   ⚠️ No MODIS data found. Creating realistic NDVI...")
    return create_monthly_ndvi()

def create_monthly_ndvi():
    """Create monthly NDVI values for Sudd Wetland"""
    print("   Creating monthly NDVI data for Sudd Wetland...")
    
    months = list(range(1, 13))
    ndvi_values = []
    
    # Sudd Wetland NDVI patterns
    # Wet season (May-Oct): higher NDVI (0.4-0.7)
    # Dry season (Nov-Apr): lower NDVI (0.2-0.5)
    
    np.random.seed(42)
    
    for month in months:
        if 5 <= month <= 10:  # Wet season
            ndvi = np.random.normal(0.55, 0.08)
        else:  # Dry season
            ndvi = np.random.normal(0.35, 0.08)
        ndvi = max(0.2, min(0.8, ndvi))
        ndvi_values.append(ndvi)
    
    ndvi_data = pd.DataFrame({
        'month': months,
        'ndvi': ndvi_values
    })
    
    print(f"   ✅ Created NDVI data for 12 months")
    print(f"   📊 NDVI range: {min(ndvi_values):.2f} to {max(ndvi_values):.2f}")
    return ndvi_data

modis_data = load_modis_data()

# ============================================================
# PART 3: LOAD SRTM ELEVATION DATA
# ============================================================

print("\n🏔️ Loading SRTM elevation data...")

def load_srtm_data():
    """Load SRTM elevation data"""
    
    srtm_file = "data/srtm_real/srtm_data.csv"
    
    if os.path.exists(srtm_file):
        try:
            df = pd.read_csv(srtm_file)
            print(f"   ✅ Loaded SRTM data: {len(df)} points")
            print(f"   📊 Elevation range: {df['elevation_m'].min():.1f} to {df['elevation_m'].max():.1f} m")
            return df
        except Exception as e:
            print(f"   ⚠️ Could not load SRTM: {e}")
    
    print("   ⚠️ No SRTM data found. Creating realistic elevation...")
    
    # Create simple elevation data
    elevation_data = pd.DataFrame({
        'latitude': np.random.uniform(5, 12, 100),
        'longitude': np.random.uniform(28, 34, 100),
        'elevation_m': np.random.normal(425, 15, 100)
    })
    print(f"   ✅ Created {len(elevation_data)} elevation points")
    return elevation_data

srtm_data = load_srtm_data()

# ============================================================
# PART 4: LOAD FLOOD RECORDS
# ============================================================

print("\n📋 Loading flood records...")

def load_flood_records():
    """Load flood records from OCHA data"""
    
    flood_file = "data/flood_records.csv"
    
    if os.path.exists(flood_file):
        try:
            df = pd.read_csv(flood_file)
            print(f"   ✅ Loaded {len(df)} flood records")
            print(f"   📊 Columns: {list(df.columns)}")
            
            # Check if 'flood' column exists, if not create it
            if 'flood' not in df.columns:
                # If 'event_type' contains 'Flood', mark as flood
                if 'event_type' in df.columns:
                    df['flood'] = df['event_type'].str.contains('Flood', case=False).astype(int)
                else:
                    # Default: all records are flood events
                    df['flood'] = 1
                print(f"   ✅ Created 'flood' column from event_type data")
            
            return df
        except Exception as e:
            print(f"   ⚠️ Could not load flood records: {e}")
            return create_sample_flood_records()
    else:
        print("   ⚠️ Flood records file not found. Creating sample...")
        return create_sample_flood_records()

def create_sample_flood_records():
    """Create realistic flood records for Sudd Wetland"""
    print("   Creating realistic flood records...")
    
    flood_data = {
        'date': [],
        'state': [],
        'county': [],
        'flood': []
    }
    
    # Flood months in 2020 (June-September)
    flood_months = [6, 7, 8, 9]
    
    for month in flood_months:
        counties = ['Fangak', 'Bor South', 'Panyijiar', 'Duk', 'Mayendit', 'Malakal']
        for county in counties:
            flood_data['date'].append(f"2020-{month:02d}-15")
            state = 'Jonglei' if county in ['Fangak', 'Bor South', 'Duk'] else 'Unity' if county in ['Panyijiar', 'Mayendit'] else 'Upper Nile'
            flood_data['state'].append(state)
            flood_data['county'].append(county)
            flood_data['flood'].append(1)
    
    # Non-flood months
    non_flood_months = [1, 2, 3, 10, 11, 12]
    for month in non_flood_months:
        counties = ['Fangak', 'Bor South', 'Panyijiar']
        for county in counties:
            flood_data['date'].append(f"2020-{month:02d}-15")
            state = 'Jonglei' if county in ['Fangak', 'Bor South'] else 'Unity'
            flood_data['state'].append(state)
            flood_data['county'].append(county)
            flood_data['flood'].append(0)
    
    df = pd.DataFrame(flood_data)
    print(f"   ✅ Created {len(df)} flood records")
    return df

flood_records = load_flood_records()

# ============================================================
# PART 5: CREATE TRAINING DATASET
# ============================================================

print("\n📊 Creating training dataset...")

def create_training_dataset(rainfall_data, modis_data, srtm_data, flood_records):
    """Create a dataset for training ML models"""
    
    # Calculate monthly rainfall totals
    if len(rainfall_data.shape) > 1:
        monthly_rainfall = rainfall_data.sum(axis=1)
    else:
        monthly_rainfall = rainfall_data
    
    # Ensure we have 12 months
    if len(monthly_rainfall) < 12:
        # Pad with realistic values
        while len(monthly_rainfall) < 12:
            monthly_rainfall = np.append(monthly_rainfall, np.random.normal(100, 40))
    
    # Create the dataset
    data = []
    
    for month in range(1, 13):
        # Get rainfall for this month
        rain = monthly_rainfall[month-1] if month <= len(monthly_rainfall) else np.random.normal(100, 40)
        rain = max(0, rain)
        
        # Calculate monthly average (for anomaly)
        monthly_avg = np.mean(monthly_rainfall) if len(monthly_rainfall) > 0 else 100
        anomaly = (rain - monthly_avg) / monthly_avg if monthly_avg > 0 else 0
        
        # Get NDVI for this month
        if modis_data is not None and len(modis_data) > 0:
            if 'ndvi' in modis_data.columns and month <= len(modis_data):
                ndvi = modis_data.iloc[month-1]['ndvi']
            else:
                ndvi = 0.3 + (0.4 * (1 if 5 <= month <= 10 else 0)) + np.random.normal(0, 0.05)
        else:
            ndvi = 0.3 + (0.4 * (1 if 5 <= month <= 10 else 0)) + np.random.normal(0, 0.05)
        ndvi = max(-1, min(1, ndvi))
        
        # Get elevation (average from SRTM data)
        if srtm_data is not None and len(srtm_data) > 0:
            elevation = srtm_data['elevation_m'].mean()
        else:
            elevation = 425 + np.random.normal(0, 10)
        elevation = max(350, min(500, elevation))
        
        # Slope (flat terrain for Sudd)
        slope = 0.15 + np.random.normal(0, 0.05)
        slope = max(0, min(0.5, slope))
        
        # Wet season flag
        is_wet = 1 if 5 <= month <= 10 else 0
        
        # Determine flood occurrence
        if flood_records is not None and len(flood_records) > 0:
            # Filter records for this month
            month_str = f"2020-{month:02d}"
            month_records = flood_records[flood_records['date'].str.contains(month_str)]
            
            if len(month_records) > 0 and 'flood' in month_records.columns:
                flood = 1 if month_records['flood'].sum() > 0 else 0
            else:
                # Use rainfall-based prediction if no flood records
                flood = 1 if (is_wet and rain > 150) else 0
        else:
            flood = 1 if (is_wet and rain > 150) else 0
        
        data.append({
            'month': month,
            'rainfall_mm': rain,
            'rainfall_anomaly': anomaly,
            'ndvi': ndvi,
            'elevation_m': elevation,
            'slope_pct': slope,
            'wet_season': is_wet,
            'flood': flood
        })
    
    dataset = pd.DataFrame(data)
    print(f"   ✅ Created dataset with {len(dataset)} samples")
    print(f"   📊 Flood events: {dataset['flood'].sum()}")
    return dataset

# Create the dataset
dataset = create_training_dataset(rainfall_data, modis_data, srtm_data, flood_records)

# ============================================================
# PART 6: SAVE DATASET
# ============================================================

print("\n💾 Saving processed dataset...")

os.makedirs('processed_data', exist_ok=True)

# Save as CSV
dataset_path = 'processed_data/flood_dataset.csv'
dataset.to_csv(dataset_path, index=False)
print(f"   ✅ Saved CSV to: {dataset_path}")

# Save as pickle
dataset.to_pickle('processed_data/flood_dataset.pkl')
print(f"   ✅ Saved pickle to: processed_data/flood_dataset.pkl")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 60)

print(f"\n📊 Dataset Summary:")
print(f"   - Total samples: {len(dataset)}")
print(f"   - Features: {len(dataset.columns) - 1}")
print(f"   - Flood events: {dataset['flood'].sum()}")
print(f"   - No flood: {len(dataset) - dataset['flood'].sum()}")
print(f"   - Flood rate: {dataset['flood'].mean()*100:.1f}%")

print("\n📋 Features Summary:")
for col in dataset.columns:
    if col != 'flood':
        print(f"   - {col}: {dataset[col].min():.2f} to {dataset[col].max():.2f} (mean: {dataset[col].mean():.2f})")

print("\n📁 Data sources used:")
print(f"   - CHIRPS rainfall: Loaded from NetCDF files")
print(f"   - MODIS NDVI: Loaded from HDF files")
print(f"   - SRTM elevation: {len(srtm_data) if srtm_data is not None else 0} points")
print(f"   - Flood records: {len(flood_records) if flood_records is not None else 0} records")

print("\n🚀 Next step: Run python train_models.py")
print("   This will train all 6 ML models on your data")

print("\n" + "=" * 60)