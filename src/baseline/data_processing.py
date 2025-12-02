%%writefile data_processing.py
"""
Data loading and merging script.
"""
from typing import Any
import pandas as pd
import config  
import constants 

def load_and_merge_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw data files and merges them into a single DataFrame.
    Combines train and test sets, then joins user and book metadata. The genre
    and description data are returned separately as they're needed for feature engineering.
    Filters out anomalies:
    - Ratings outside [0, 10] range (only applies to rows with has_read=1)
    - Age outside [10, 100] range (only applies if age is present in train/test and users.csv)
    - Missing user_id or book_id
    - has_read not in [0, 1]
    Includes records from train.csv with has_read=0 (without rating) for potential
    feature engineering context.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The merged DataFrame (train_has_read_1, train_has_read_0, test + metadata) with anomalies removed.
            - The book_genres DataFrame.
            - The genres DataFrame.
            - The book_descriptions DataFrame.
    """
    print("Loading data...")
    # Define dtypes for memory optimization
    dtype_spec: dict[str, Any] = {
        constants.COL_USER_ID: "int32",
        constants.COL_BOOK_ID: "int32",
        constants.COL_TARGET: "float32",
        constants.COL_GENDER: "category",
        constants.COL_AGE: "float32",
        constants.COL_AUTHOR_ID: "int32",
        constants.COL_PUBLICATION_YEAR: "float32",
        constants.COL_LANGUAGE: "category",
        constants.COL_PUBLISHER: "category",
        constants.COL_AVG_RATING: "float32",
        constants.COL_GENRE_ID: "int16",
        constants.COL_HAS_READ: "int8",
    }
    # Load datasets
    # CSV files use comma as separator (default pandas behavior)
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k in [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_TARGET, constants.COL_HAS_READ]
        },
        parse_dates=[constants.COL_TIMESTAMP],
    )

    # --- DATA QUALITY IMPROVEMENTS ---
    print("Applying data quality filters to train data...")
    initial_train_count = len(train_df)

    # 1. Filter out has_read not in [0, 1]
    has_read_filter = train_df[constants.COL_HAS_READ].isin([0, 1])
    train_df = train_df[has_read_filter].copy()
    print(f"Filtered by has_read [0, 1]: {initial_train_count} -> {len(train_df)} rows.")

    # 2. Filter out ratings outside [0, 10] range (only for rows where has_read=1)
    rating_filter = (train_df[constants.COL_HAS_READ] == 0) | \
                    (train_df[constants.COL_TARGET].isna()) | \
                    ((train_df[constants.COL_TARGET] >= 0) & (train_df[constants.COL_TARGET] <= 10))
    train_df = train_df[rating_filter].copy()
    print(f"Filtered by rating [0, 10] (has_read=1 only): {len(train_df)} -> {len(train_df)} rows.")

    # 3. Filter out age outside [10, 100] range (only applies if age column exists in train_df BEFORE merge)
    age_col_present_train = constants.COL_AGE in train_df.columns
    if age_col_present_train:
        print("Filtering by age (10-100) in train data (before merge)...")
        age_filter = (train_df[constants.COL_AGE].isna()) | \
                     ((train_df[constants.COL_AGE] >= 10) & (train_df[constants.COL_AGE] <= 100))
        train_df = train_df[age_filter].copy()
        print(f"Applied age filter (10-100) to train data (before merge): {len(train_df)} -> {len(train_df)} rows.")
    else:
        print(f"Warning: Column '{constants.COL_AGE}' not found in train data (before merge). Skipping age filter.")

    # 4. Filter out missing user_id or book_id (should ideally not be present in clean data, but good to check)
    id_filter = train_df[constants.COL_USER_ID].notna() & train_df[constants.COL_BOOK_ID].notna()
    train_df = train_df[id_filter].copy()
    print(f"Filtered by missing user_id/book_id in train: {len(train_df)} -> {len(train_df)} rows.")

    filtered_train_count = len(train_df)
    print(f"Applied filters to train  {initial_train_count} -> {filtered_train_count} rows.")

    # --- Separate train data based on has_read ---
    train_has_read_1 = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    train_has_read_0 = train_df[train_df[constants.COL_HAS_READ] == 0].copy()
    print(f"  - Train rows with rating (has_read=1): {len(train_has_read_1)}")
    print(f"  - Train rows without rating (has_read=0): {len(train_has_read_0)}")

    # Load test set
    test_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TEST_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_BOOK_ID]},
    )

    # Apply same filters to test set (where applicable)
    # 1. Filter out missing user_id or book_id in test set
    test_df = test_df[test_df[constants.COL_USER_ID].notna() & test_df[constants.COL_BOOK_ID].notna()].copy()
    print(f"Applied filters to test data. Shape: {test_df.shape}")

    # Load metadata
    # Adjust dtype for user_data loading if age is not defined or not present in users.csv
    user_data_dtype = {
        k: v for k, v in dtype_spec.items() if k in [constants.COL_USER_ID, constants.COL_GENDER]
    }
    # Only add age to dtype if COL_AGE is defined as "age" AND you expect it in users.csv
    # For safety, let's also exclude age from user_data_dtype unless explicitly needed later
    # if constants.COL_AGE == "age":
    #     user_data_dtype[constants.COL_AGE] = "float32"

    user_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.USER_DATA_FILENAME,
        dtype=user_data_dtype,
    )
    book_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k
            in [
                constants.COL_BOOK_ID,
                constants.COL_AUTHOR_ID,
                constants.COL_PUBLICATION_YEAR,
                constants.COL_LANGUAGE,
                constants.COL_AVG_RATING,
                constants.COL_PUBLISHER,
            ]
        },
    )
    book_genres_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME,
        dtype={k: v for k, v in dtype_spec.items() if k in [constants.COL_BOOK_ID, constants.COL_GENRE_ID]},
    )
    genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.GENRES_FILENAME)
    book_descriptions_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME,
        dtype={constants.COL_BOOK_ID: "int32"},
    )

    # Apply same filters to user metadata if needed (e.g., age) - Check if age exists in user_data_df
    age_col_present_user = constants.COL_AGE in user_data_df.columns
    if age_col_present_user:
        print("Filtering user data based on age (10-100)...")
        user_data_df = user_data_df[(user_data_df[constants.COL_AGE].isna()) | ((user_data_df[constants.COL_AGE] >= 10) & (user_data_df[constants.COL_AGE] <= 100))].copy()
        print(f"Filtered user data based on age. Shape: {user_data_df.shape}")
    else:
        print(f"Warning: '{constants.COL_AGE}' not found in user_data_df.")

    print("Data loaded. Merging datasets...")
    # Assign source labels
    train_has_read_1[constants.COL_SOURCE] = constants.VAL_SOURCE_TRAIN
    train_has_read_0[constants.COL_SOURCE] = constants.VAL_SOURCE_TRAIN # Mark as part of train context
    test_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TEST

    # Combine train (has_read=1 and has_read=0) and test
    # Only train_has_read_1 will have TARGET values for training
    combined_df = pd.concat([train_has_read_1, train_has_read_0, test_df], ignore_index=True, sort=False)

    print(f"Shape of combined data before merge: {combined_df.shape}")
    print(f"Columns in combined data before merge: {list(combined_df.columns)}")
    print(f"Shape of user  {user_data_df.shape}")
    print(f"Columns in user  {list(user_data_df.columns)}")

    # Join metadata
    # Check if age is present in user_data_df before merging to avoid KeyError if it's expected but missing
    if constants.COL_AGE in user_data_df.columns:
        print(f"Merging with user metadata including '{constants.COL_AGE}'.")
    else:
        print(f"Warning: '{constants.COL_AGE}' not found in user_data_df. Merging without it.")

    # Perform the merge
    combined_df = combined_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")
    print(f"Shape of combined data after user merge: {combined_df.shape}")
    print(f"Columns in combined data after user merge: {list(combined_df.columns)}")

    # Check explicitly for 'age' after merge
    age_col_after_merge = constants.COL_AGE in combined_df.columns
    if age_col_after_merge:
        print(f"Column '{constants.COL_AGE}' is present in combined data after merge.")
        print(f"  - Number of non-null values in '{constants.COL_AGE}': {combined_df[constants.COL_AGE].notna().sum()}")
        print(f"  - Number of null values in '{constants.COL_AGE}': {combined_df[constants.COL_AGE].isna().sum()}")
    else:
        print(f"Warning: Column '{constants.COL_AGE}' is NOT present in combined data after merge.")

    # Drop duplicates from book_data_df before merging
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    combined_df = combined_df.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")
    print(f"Merged data shape after quality filters and book merge: {combined_df.shape}")
    return combined_df, book_genres_df, genres_df, book_descriptions_df
