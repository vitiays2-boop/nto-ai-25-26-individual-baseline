%%writefile prepare_data.py
"""
Data preparation script that processes raw data and saves it to processed directory.
This script loads raw data, applies quality filters, includes has_read=0 records,
performs feature engineering, and saves the processed data to data/processed/
for use in training and prediction.
"""
import config  
import constants 
from data_processing import load_and_merge_data 
from features import create_features 

def prepare_data() -> None:
    """Processes raw data and saves prepared features to processed directory.
    This function:
    1. Loads raw data from data/raw/
    2. Applies quality filters (rating, ID validity, age if present)
    3. Includes records with has_read=0 from train.csv for feature context
    4. Applies feature engineering (genres, TF-IDF, BERT) - NO aggregates to avoid data leakage
    5. Saves processed data to data/processed/processed_features.parquet
    6. Preserves timestamp for temporal splitting
    Note: Aggregate features are computed separately during training to ensure
    temporal correctness (no data leakage from validation set).
    The processed data can then be used by train.py and predict.py without
    re-running the expensive feature engineering steps.
    """
    print("=" * 60)
    print("Data Preparation Pipeline (with has_read=0 and anomaly filtering)")
    print("=" * 60)
    # Load and merge raw data (includes has_read=0, applies filters)
    merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()

    # Apply feature engineering WITHOUT aggregates
    # Aggregates will be computed during training on train split only (has_read=1)
    featured_df = create_features(merged_df, book_genres_df, descriptions_df, include_aggregates=False)
    # Ensure processed directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Define the output path
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    # Save processed data as parquet for efficiency
    print(f"\nSaving processed data to {processed_path}...")
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    print("Processed data saved successfully!")
    # Print statistics
    train_rows_total = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    train_rows_with_rating = len(featured_df[(featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN) & (featured_df[constants.COL_HAS_READ] == 1)])
    train_rows_without_rating = len(featured_df[(featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN) & (featured_df[constants.COL_HAS_READ] == 0)])
    test_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST])
    total_features = len(featured_df.columns)
    print("\nData preparation complete!")
    print(f"  - Total train rows (source=TRAIN): {train_rows_total:,}")
    print(f"  - Train rows with rating (has_read=1): {train_rows_with_rating:,}")
    print(f"  - Train rows without rating (has_read=0): {train_rows_without_rating:,}")
    print(f"  - Test rows: {test_rows:,}")
    print(f"  - Total features: {total_features}")
    print(f"  - Output file: {processed_path}")

if __name__ == "__main__":
    prepare_data()
