"""
This is a boilerplate pipeline 'preprocess_train_pipeline'
generated using Kedro 0.19.14
"""
from mlops_project.constants import ADMISSION_SOURCE_MAP, ADMISSION_TYPE_MAP, DISCHARGE_DISPOSITION_MAP, A1CRESULT_MAP,MAX_GLU_SERUM_MAP,AGE_MAP, MEDICATION_COLUMNS
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def process_diagnoses(df: pd.DataFrame, diag_categories: Dict[str, List[Any]] = None) -> pd.DataFrame:
    """Process diagnosis columns in a DataFrame by cleaning and grouping them.

    Args:
        df (pd.DataFrame): Input DataFrame with diagnosis columns.
        diag_categories (Dict[str, List[Any]], optional): Precomputed categories for encoding diag_1, diag_2, diag_3.

    Returns:
        pd.DataFrame: Processed DataFrame with grouped diagnosis columns.
    """
    diag_columns = ["diag_1", "diag_2", "diag_3"]
    present_columns = [col for col in diag_columns if col in df.columns]

    if not present_columns:
        for col in diag_columns:
            print(f"Column '{col}' was not found in the DataFrame.")
        return df

    df = df.copy()

    if "diag_1" in df.columns:
        df = df[df["diag_1"] != "?"]

    for diag in ["diag_2", "diag_3"]:
        if diag in df.columns:
            df.loc[:, diag] = df[diag].replace("?", "Unknown")

    def group_diag(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        df = df.copy()
        col_float = pd.to_numeric(df[col_name], errors='coerce')
        conditions = [
            df[col_name] == "Unknown",
            df[col_name].str.startswith("250", na=False),
            (col_float >= 390) & (col_float <= 459),
            (col_float >= 460) & (col_float <= 519),
            (col_float >= 520) & (col_float <= 579),
            (col_float >= 580) & (col_float <= 629),
            (col_float >= 680) & (col_float <= 709),
            (col_float >= 710) & (col_float <= 739),
            (col_float >= 800) & (col_float <= 999),
            (col_float >= 140) & (col_float <= 239),
            (col_float >= 240) & (col_float <= 279),
            (col_float >= 280) & (col_float <= 289),
            (col_float >= 290) & (col_float <= 319),
            (col_float >= 320) & (col_float <= 389),
            (col_float >= 780) & (col_float <= 799)
        ]
        choices = [
            "Unknown", "diabetes", "circulatory", "respiratory", "digestive",
            "genitourinary", "skin", "musculoskeletal", "injury", "neoplasms",
            "endocrine", "blood", "mental", "nervous", "symptoms"
        ]
        df[f"{col_name}_grouped"] = np.select(conditions, choices, default="other")
        df = df.drop(columns=[col_name])
        return df

    for diag in present_columns:
        df = group_diag(df, diag)
        
    if diag_categories:
        for diag in present_columns:
            col = f"{diag}_grouped"
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if x in diag_categories[col] else "other")
    
    return df

def categorical_encoder(df: pd.DataFrame, col_name: str, categories: List[Any] = None) -> pd.DataFrame:
    """
    Label encodes and one-hot encodes a column, using provided categories to prevent leakage.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column to encode.
        categories (List[Any], optional): Precomputed categories for encoding.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    if col_name not in df.columns:
        return df

    df = df.copy()
    
    if categories is not None:
        # Ensure unique categories by checking if "other" is already present
        unique_categories = categories.copy()
        if "other" not in unique_categories:
            unique_categories.append("other")
        # Map unseen categories to "other"
        df[col_name] = df[col_name].apply(lambda x: x if x in categories else "other")
        # Create a CategoricalDtype with unique categories
        cat_dtype = pd.CategoricalDtype(categories=unique_categories, ordered=False)
        df[col_name] = df[col_name].astype(cat_dtype).cat.codes
    else:
        # For training data, create categories
        df[col_name] = df[col_name].astype('category').cat.codes

    one_hot = pd.get_dummies(df[col_name], prefix=col_name)
    df = df.drop(columns=[col_name])
    df = pd.concat([df, one_hot], axis=1)
    return df

def winsorize_column(
    df: pd.DataFrame,
    col_name: str,
    lower: Optional[Union[float, int]] = None,
    upper: Optional[Union[float, int]] = None,
    quantiles: Optional[List[float]] = None,
    bounds: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Winsorize a column using provided bounds or quantiles.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column to winsorize.
        lower (float or int, optional): Lower bound.
        upper (float or int, optional): Upper bound.
        quantiles (List[float], optional): Quantiles for IQR calculation (training only).
        bounds (Tuple[float, float], optional): Precomputed (lower, upper) bounds.

    Returns:
        pd.DataFrame: DataFrame with winsorized column.
    """
    if col_name not in df.columns:
        return df
    
    df = df.copy()
    
    if bounds:
        lower, upper = bounds
    elif quantiles:
        q1, q3 = df[col_name].quantile(quantiles)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    
    df[col_name] = df[col_name].clip(lower=lower, upper=upper)
    return df

def clean_and_process_data(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process training and test DataFrames to prevent data leakage.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame, optional): Test DataFrame. If None, only process train_df.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed training and test DataFrames.
    """
    train_df = train_df.copy()
    if test_df is not None:
        test_df = test_df.copy()

    # Compute gender mode on training data
    gender_mode = train_df[train_df['gender'] != 'Unknown/Invalid']['gender'].mode()[0]

    # Apply to train and test
    train_df["race"] = train_df["race"].replace("?", "Unknown")
    train_df.loc[train_df['gender'] == 'Unknown/Invalid', 'gender'] = gender_mode
    if test_df is not None:
        test_df["race"] = test_df["race"].replace("?", "Unknown")
        test_df.loc[test_df['gender'] == 'Unknown/Invalid', 'gender'] = gender_mode

    # Replace categorical ID columns
    for df in [train_df, test_df] if test_df is not None else [train_df]:
        df['admission_type_id'] = df['admission_type_id'].replace(ADMISSION_TYPE_MAP)
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(DISCHARGE_DISPOSITION_MAP)
        df['admission_source_id'] = df['admission_source_id'].replace(ADMISSION_SOURCE_MAP)
        df['a1cresult'] = df['a1cresult'].replace(A1CRESULT_MAP).fillna(-99).astype(int)
        df['max_glu_serum'] = df['max_glu_serum'].replace(MAX_GLU_SERUM_MAP).fillna(-99).astype(int)
        df['age'] = pd.to_numeric(df['age'].replace(AGE_MAP), errors='coerce').astype(int)

    # Process diagnoses and collect categories from training data
    train_df = process_diagnoses(train_df)
    diag_categories = {}
    for col in ["diag_1_grouped", "diag_2_grouped", "diag_3_grouped"]:
        if col in train_df.columns:
            diag_categories[col] = train_df[col].unique().tolist()
    if test_df is not None:
        test_df = process_diagnoses(test_df, diag_categories)

    # Medication processing
    for df in [train_df, test_df] if test_df is not None else [train_df]:
        for colname in MEDICATION_COLUMNS:
            if colname in df.columns:
                df[colname + 'temp'] = np.where(df[colname].isin(['No', 'Steady']), 0, 1)
        temp_cols = [col + 'temp' for col in MEDICATION_COLUMNS if col + 'temp' in df.columns]
        df['numchange'] = df[temp_cols].sum(axis=1)
        for colname in MEDICATION_COLUMNS:
            if colname in df.columns:
                temp_col = colname + 'temp'
                if temp_col in df.columns:
                    df.drop(columns=temp_col, inplace=True)
                df[colname + '_encoded'] = np.where(df[colname] == 'No', 0, 1)
                df.drop(columns=colname, inplace=True)
        med_cols_encoded = [col + '_encoded' for col in MEDICATION_COLUMNS if col + '_encoded' in df.columns]
        df['num_med'] = df[med_cols_encoded].sum(axis=1)

    # Binary encoding
    for df in [train_df, test_df] if test_df is not None else [train_df]:
        for col, positive_val in [("change", "Ch"), ("diabetesmed", "Yes"), ("gender", "Male"), ("readmitted", "<30")]:
            if col in df.columns:
                if col == "readmitted":
                    df[col] = (df[col] == positive_val).astype(int)
                else:
                    df[f"{col}_encoded"] = (df[col] == positive_val).astype(int)
                    df.drop(columns=[col], inplace=True)
        df['services'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

    # Winsorization bounds computed on training data
    winsor_bounds = {}
    for col, params in [
        ("num_lab_procedures", {"quantiles": [0.25, 0.75]}),
        ("number_outpatient", {"upper": 15}),
        ("number_emergency", {"upper": 12})
    ]:
        if col in train_df.columns:
            if "quantiles" in params:
                q1, q3 = train_df[col].quantile(params["quantiles"])
                iqr = q3 - q1
                winsor_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            else:
                winsor_bounds[col] = (None, params.get("upper"))

    # Apply winsorization
    for col in winsor_bounds:
        train_df = winsorize_column(train_df, col, bounds=winsor_bounds[col])
        if test_df is not None:
            test_df = winsorize_column(test_df, col, bounds=winsor_bounds[col])

    # Categorical encoding with categories from training data
    cat_columns = ['race', 'diag_1_grouped', 'diag_2_grouped', 'diag_3_grouped']
    cat_mappings = {}
    for col in cat_columns:
        if col in train_df.columns:
            cat_mappings[col] = train_df[col].unique().tolist()
            train_df = categorical_encoder(train_df, col)
            if test_df is not None:
                test_df = categorical_encoder(test_df, col, categories=cat_mappings[col])

    logging.getLogger(__name__).info(f"The final training dataframe has {len(train_df.columns)} columns.")
    if test_df is not None:
        logging.getLogger(__name__).info(f"The final test dataframe has {len(test_df.columns)} columns.")
        return train_df.astype('int64'), test_df.astype('int64')
    return train_df.astype('int64'), None