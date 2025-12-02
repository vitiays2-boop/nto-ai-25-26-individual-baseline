%%writefile features.py
"""
Feature engineering script.
"""
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import config  
import constants 

def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.
    Uses the training data (has_read=1) to compute mean ratings and interaction counts
    to prevent data leakage from the test set or from has_read=0 records if they were included.
    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data (has_read=1) for calculations.
    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")
    # User-based aggregates (mean, count) - calculated only on train_df (has_read=1)
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]
    # Book-based aggregates (mean, count) - calculated only on train_df (has_read=1)
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]
    # Author-based aggregates (mean) - calculated only on train_df (has_read=1)
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.
    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.
    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions.
    Trains a TF-IDF vectorizer only on training data descriptions (from train_df with has_read=1) to avoid
    data leakage. Applies the vectorizer to all books and merges the features.
    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting the vectorizer (has_read=1).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
    Returns:
        pd.DataFrame: The DataFrame with TF-IDF features added.
    """
    print("Adding text features (TF-IDF)...")
    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set (has_read=1) - this is the critical change
    train_books = train_df[constants.COL_BOOK_ID].unique()
    # Extract descriptions for training books only
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Check if vectorizer already exists (for prediction)
    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fit vectorizer on training descriptions only (has_read=1)
        print("Fitting TF-IDF vectorizer on training descriptions (has_read=1)...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        # Save vectorizer for use in prediction
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    # Transform all book descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
    # Get descriptions for books in df (in the same order)
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert sparse matrix to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    # Concatenate TF-IDF features with main DataFrame
    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf


def add_bert_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BERT embeddings from book descriptions.
    Extracts 768-dimensional embeddings using a pre-trained Russian BERT model.
    Embeddings are cached on disk to avoid recomputation on subsequent runs.
    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency, not used for BERT).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
    Returns:
        pd.DataFrame: The DataFrame with BERT embeddings added.
    """
    print("Adding text features (BERT embeddings)...")
    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings (this may take a while)...")
        print(f"Using device: {config.BERT_DEVICE}")
        # Limit GPU memory usage to prevent OOM errors
        if config.BERT_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.BERT_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.BERT_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(config.BERT_DEVICE)
        model.eval()

        # Prepare descriptions: get unique book_id -> description mapping
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
        # Get unique books and their descriptions
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        # Initialize embeddings dictionary
        embeddings_dict = {}

        # Process descriptions in batches
        num_batches = (len(descriptions) + config.BERT_BATCH_SIZE - 1) // config.BERT_BATCH_SIZE
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing BERT batches", unit="batch"):
                start_idx = batch_idx * config.BERT_BATCH_SIZE
                end_idx = min(start_idx + config.BERT_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                )
                # Move to device
                encoded = {k: v.to(config.BERT_DEVICE) for k, v in encoded.items()}

                # Get model outputs
                outputs = model(**encoded)

                # Mean pooling (original method)
                attention_mask = encoded["attention_mask"]
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                batch_embeddings = mean_pooled.cpu().numpy()

                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                # Small pause between batches to let GPU cool down and prevent overheating
                if config.BERT_DEVICE == "cuda":
                    time.sleep(0.2)  # 200ms pause between batches

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved BERT embeddings to {embeddings_path}")

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()
    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Create DataFrame with BERT features
    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)

    # Concatenate BERT features with main DataFrame
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
    print(f"Added {len(bert_feature_names)} BERT features.")
    return df_with_bert


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Fills missing values using a defined strategy.
    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.
    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data (has_read=1), used for calculating fill metrics.
    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")
    # Calculate global mean from training data (has_read=1) for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median from the current df (or train_df if preferred) - Only if age column exists in merged df
    if constants.COL_AGE in df.columns:
        age_median = df[constants.COL_AGE].median()
        df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)
    else:
        print(f"Warning: {constants.COL_AGE} not found in merged DataFrame. Skipping age fill.")

    # Fill aggregate features for "cold start" users/items (only if they exist)
    # Use global_mean or 0 as fallback if aggregates don't exist
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill TF-IDF features with 0 (for books without descriptions)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0 (for books without descriptions)
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df


def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.
    This function orchestrates the calls to add aggregate features (optional), genre
    features, text features (TF-IDF and BERT), and handle missing values.
    Args:
        df (pd.DataFrame): The merged DataFrame from `data_processing`.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
        include_aggregates (bool): If True, compute aggregate features. Defaults to False.
            Aggregates are typically computed separately during training to avoid data leakage.
    Returns:
        pd.DataFrame: The final DataFrame with all features engineered.
    """
    print("Starting feature engineering pipeline...")

    # Use only train rows with has_read=1 for calculations that depend on the target (e.g., aggregates)
    # This is the critical part: use *only* has_read=1 for target-dependent features like aggregates
    train_df_for_calcs = df[(df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN) & (df[constants.COL_HAS_READ] == 1)].copy()

    # Aggregate features are computed separately during training to ensure
    # no data leakage from validation set timestamps
    if include_aggregates:
        df = add_aggregate_features(df, train_df_for_calcs)

    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df_for_calcs, descriptions_df)
    df = add_bert_features(df, train_df_for_calcs, descriptions_df)
    df = handle_missing_values(df, train_df_for_calcs)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
