"""
This is a boilerplate pipeline 'drift_simulation'
generated using Kedro 0.19.14
"""
import pandas as pd
import numpy as np

def simulate_drift(test_preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate data drift on a test dataset by modifying specific feature distributions:
    - Flip 5% of 'is_active' from False to True.
    - Add 10 years to 'age'.
    - Decrease 'number_inpatient' by 5 if > 10.
    - Add 3 to 'number_diagnoses'.
    - Randomly add or subtract 2 from 'num_medications' but never go below 0.

    Parameters:
        test_preprocessed_data (pd.DataFrame): Input DataFrame with preprocessed test data.

    Returns:
        pd.DataFrame: New DataFrame with simulated drift applied.
    """
    data = test_preprocessed_data.copy()

    # Flip 5% of False in 'gender_encoded' to True
    num_to_flip = int(len(data) * 0.05)
    false_indices = data[data['gender_encoded'] == 0].sample(n=num_to_flip, random_state=42).index
    data.loc[false_indices, 'gender_encoded'] = 1

    # Add 10 years to 'age'
    data['age'] = data['age'] + 10

    # Decrease 'number_inpatient' by 5 if > 10
    data.loc[data['number_inpatient'] > 10, 'number_inpatient'] -= 5

    # Add 3 to 'number_diagnoses'
    data['number_diagnoses'] = data['number_diagnoses'] + 3

    # Randomly add or subtract 2 to 'num_medications' but keep >= 0
    random_changes = np.random.choice([2, -2], size=len(data), replace=True)
    data['num_medications'] = data['num_medications'] + random_changes
    data['num_medications'] = data['num_medications'].clip(lower=0)

    return data