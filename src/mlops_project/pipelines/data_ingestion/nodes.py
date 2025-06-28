"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.14
"""
import logging
from typing import Any, Dict, Tuple
import hopsworks

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

from mlops_project.constants import(
    MEDICATION_COLUMNS, MEDICATION_VALUE_SET, AGE_VALUE_SET, RACE_VALUE_SET,
    GENDER_VALUE_SET, A1C_RESULT_SET, MAX_GLU_SERUM_SET,
    CHANGE_VALUE_SET, DIABETES_MED_SET, COLS_TO_DROP, READMITTED_SET, MEDICATION_COLUMNS_CORRECTED, 
    CATEGORICAL_FEATURES_DESCRIPTIONS, TARGET_FEATURES_DESCRIPTIONS, NUMERICAL_FEATURES_DESCRIPTIONS, NUMERICAL_COLUMNS
)

logger = logging.getLogger(__name__)


def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    
    Args:
        expectation_suite_name (str): A dictionary with the feature group name and the respective version.
        feature_group (str): Feature group used to construct the expectations.
             
    Returns:
        ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
    """
    
    expectation_suite_diabetes = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )
    

    # numerical features
    if feature_group == 'numerical_features':

        for i in NUMERICAL_COLUMNS:
            expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "int64"},
                )
            )
        # time_in_hospital 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "time_in_hospital",
                        "min_value": 1,
                        "max_value": 14,
                    },
                )
            )
        # admission_type_id 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "admission_type_id",
                        "min_value": 1,
                        "max_value": 8,
                    },
                )
            )
        # discharge_disposition_id 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "discharge_disposition_id",
                        "min_value": 1,
                        "max_value": 32,
                    },
                )
            )
        # admission_source_id 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "admission_source_id",
                        "min_value": 1,
                        "max_value": 25,
                    },
                )
            )
        # number_outpatient 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "number_outpatient",
                        "min_value": 0,
                        "max_value": 100,
                    },
                )
            )
        # number_inpatient 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "number_inpatient",
                        "min_value": 0,
                        "max_value": 100,
                    },
                )
            )
        # num_lab_procedures 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "num_lab_procedures",
                        "min_value": 0
                    },
                )
            )

        # num_procedures 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "num_procedures",
                        "min_value": 0,
                    },
                )
            )

        # num_medications 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "num_medications",
                        "min_value": 0
                    },
                )
            )

        # number_diagnoses 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_min_to_be_between",
                    kwargs={
                        "column": "number_diagnoses",
                        "min_value": 1,
                        "max_value": 20,
                    },
                )
            )
        # encounter_id 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_unique",
                    kwargs={
                        "column": "encounter_id",
                    },
                )
            )

    if feature_group == 'categorical_features':

        for i in MEDICATION_COLUMNS_CORRECTED:
            expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_distinct_values_to_be_in_set",
                    kwargs={"column": i, "value_set": MEDICATION_VALUE_SET},
                )
            )
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "age", "value_set": AGE_VALUE_SET},
            )
        ) 
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "gender", "value_set": GENDER_VALUE_SET},
            )
        ) 
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "race", "value_set": RACE_VALUE_SET},
            )
        ) 
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "a1cresult", "value_set": A1C_RESULT_SET},
            )
        ) 
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "max_glu_serum", "value_set": MAX_GLU_SERUM_SET},
            )
        ) 
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "change", "value_set": CHANGE_VALUE_SET},
            )
        ) 
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "diabetesmed", "value_set": DIABETES_MED_SET},
            )
        ) 
        expectation_suite_diabetes.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={
                        "column": "diag_1"
                    },
                )
            )
    if feature_group == 'target':
        
        expectation_suite_diabetes.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "readmitted", "value_set": READMITTED_SET},
            )
        ) 
     
    return expectation_suite_diabetes

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    
    
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["encounter_id"],
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    # Upload data.
    try:
        # Upload the data to the feature group
        object_feature_group.insert(
            features=data,
            overwrite=False,
                storage="offline",
            write_options={
                "wait_for_job": True,
            },
        )

        # Add feature descriptions
        for feature_name, feature_desc in group_description.items():
            object_feature_group.update_feature_description(feature_name, feature_desc)

        # Configure and update statistics
        object_feature_group.statistics_config = {
            "enabled": True,
            "histograms": True,
            "correlations": True,
        }
        object_feature_group.update_statistics_config()
        object_feature_group.compute_statistics()

        # Return the feature group object
        return object_feature_group

    except Exception as e:
        print(f"Error during feature group creation or data insertion: {e}")
        raise e

import re

def is_valid_feature_name(name):
    # Check if name is valid: starts with letter, contains only lowercase, numbers, underscores
    pattern = r'^[a-z][a-z0-9_]{0,62}$'
    return bool(re.match(pattern, name))




def ingestion(
    df: pd.DataFrame,
    parameters: Dict[str, Any]):

    df = df.drop(columns = COLS_TO_DROP)

    for column in df.columns:
        if not is_valid_feature_name(column):
            print(f"Invalid feature name: {column}. Renaming to {column.replace('-', '_').lower()}")
            df.rename(columns={column: column.replace('-', '_').lower()}, inplace=True)


    df_full= df.drop_duplicates()

    logger.info(f"The dataset contains {len(df_full.columns)} columns.")

    numerical_features = df_full.select_dtypes(exclude=['object','string','category']).columns.tolist()
    categorical_features = df_full.select_dtypes(include=['object','string','category']).columns.tolist()
    categorical_features.remove(parameters["target_column"])


    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations","numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations","categorical_features")
    validation_expectation_suite_target = build_expectation_suite("target_expectations","target")


    for col in df_full.columns:
        if df_full[col].dtype == 'object':
            df_full[col] = df_full[col].astype(str)

    df_full_numeric = df_full[numerical_features]
    df_full_categorical = df_full[["encounter_id"] + categorical_features]
    df_full_target = df_full[["encounter_id"] + [parameters["target_column"]]]

    if parameters["to_feature_store"]:

        object_fs_categorical_features = to_feature_store(
            df_full_categorical,"categorical_features",
            1,"Categorical Features",
            CATEGORICAL_FEATURES_DESCRIPTIONS,
            validation_expectation_suite_categorical,
            credentials["feature_store"]
        )
        object_fs_numerical_features = to_feature_store(
            df_full_numeric,"numerical_features",
            1,"Numerical Features",
            NUMERICAL_FEATURES_DESCRIPTIONS,
            validation_expectation_suite_numerical,
            credentials["feature_store"]
        )
        object_fs_target_features = to_feature_store(
            df_full_target,"target_features",
            1,"Target Features",
            TARGET_FEATURES_DESCRIPTIONS,
            validation_expectation_suite_target,
            credentials["feature_store"]
        )


    return df_full
