"""
This is a boilerplate pipeline 'data_unit_tests'
generated using Kedro 0.19.14
"""

import logging
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import great_expectations as gx

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

logger = logging.getLogger(__name__)

from mlops_project.constants import (
    MEDICATION_COLUMNS, MEDICATION_VALUE_SET, AGE_VALUE_SET, RACE_VALUE_SET,
    GENDER_VALUE_SET, A1C_RESULT_SET, MAX_GLU_SERUM_SET,
    CHANGE_VALUE_SET, DIABETES_MED_SET, READMITTED_SET
)

def get_validation_results(checkpoint_result):
    # validation_result is a dictionary containing one key-value pair
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # Accessing the 'actions_results' from the validation_result_data
    validation_result_ = validation_result_data.get('validation_result', {})

    # Accessing the 'results' from the validation_result_data
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    
    
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        # Process each result dictionary as needed
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')

        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if type(observed_value) is list:
            #sometimes observed_vaue is not iterable
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation

def create_and_add_expectation(
    suite: ExpectationSuite,
    expectation_type: str,
    column: str,
    values: Optional[List[Union[str, int, float]]] = None,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None
) -> ExpectationSuite:
    """
    Create and add an expectation to the given ExpectationSuite.

    Supported expectation types:
        - "expect_column_distinct_values_to_be_in_set"
        - "expect_column_values_to_be_unique"
        - "expect_column_values_to_be_between"
        - "expect_column_values_to_not_be_null"

    Args:
        suite (ExpectationSuite): The ExpectationSuite to which the expectation will be added.
        expectation_type (str): Type of expectation to create.
        column (str): The name of the column to which the expectation applies.
        values (Optional[List[Union[str, int, float]]]): List of allowed values (required for 'expect_column_distinct_values_to_be_in_set').
        min_value (Optional[Union[int, float]]): Minimum allowed value (optional for 'expect_column_values_to_be_between').
        max_value (Optional[Union[int, float]]): Maximum allowed value (optional for 'expect_column_values_to_be_between').

    Returns:
        ExpectationSuite: The updated ExpectationSuite with the new expectation added.

    Raises:
        ValueError: If invalid parameters are provided for a given expectation type.
    """
    if expectation_type == "expect_column_distinct_values_to_be_in_set":
        if not isinstance(values, list):
            raise ValueError("For 'expect_column_distinct_values_to_be_in_set', 'values' must be a list.")
        kwargs = {
            "column": column,
            "value_set": values
        }
    elif expectation_type == "expect_column_values_to_be_unique":
        kwargs = {
            "column": column
        }
    elif expectation_type == "expect_column_values_to_be_between":
        if min_value is None and max_value is None:
            raise ValueError("At least one of 'min_value' or 'max_value' must be provided.")
        kwargs = {
            "column": column,
            "min_value": min_value,
            "max_value": max_value
        }
    elif expectation_type == "expect_column_values_to_not_be_null":
        kwargs = {
            "column": column
        }
    else:
        raise ValueError(f"Unsupported expectation_type: {expectation_type}")

    expectation = ExpectationConfiguration(
        expectation_type=expectation_type,
        kwargs=kwargs
    )

    suite.add_expectation(expectation_configuration=expectation)
    return suite

def test_data(df):
    context = gx.get_context(context_root_dir = "gx")
    datasource_name = "diabetes_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    suite_diabetes = context.add_or_update_expectation_suite(expectation_suite_name="Diabetes")

    for col in MEDICATION_COLUMNS:
        suite_diabetes = create_and_add_expectation(
            suite_diabetes,
            expectation_type="expect_column_distinct_values_to_be_in_set",
            column=col,
            values=MEDICATION_VALUE_SET
        )
    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "age",
        values=AGE_VALUE_SET
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "race",
        values=RACE_VALUE_SET
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "gender",
        values=GENDER_VALUE_SET
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "a1cresult",
        values=A1C_RESULT_SET
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "max_glu_serum",
        values=MAX_GLU_SERUM_SET
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "change",
        values=CHANGE_VALUE_SET
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "diabetesmed",
        values=DIABETES_MED_SET
    )
    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_distinct_values_to_be_in_set", "readmitted",
        values=READMITTED_SET
    )
    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_unique", "encounter_id"
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_not_be_null", "diag_1"
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "time_in_hospital", min_value=1, max_value=14
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "admission_type_id", min_value=1, max_value=8
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "discharge_disposition_id", min_value=1, max_value=32
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "admission_source_id", min_value=1, max_value=25
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "number_outpatient", min_value=0, max_value=100
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "number_inpatient", min_value=0, max_value=100
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "num_lab_procedures", min_value=0
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "num_procedures", min_value=0
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "num_medications", min_value=0
    )

    suite_diabetes = create_and_add_expectation(
        suite_diabetes, "expect_column_values_to_be_between", "number_diagnoses", min_value=1, max_value=20
    )

    context.add_or_update_expectation_suite(expectation_suite=suite_diabetes)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe = df)
        logger.info("The data asset already exists. The required one will be loaded.")
    except:
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe = df)


    checkpoint = gx.checkpoint.SimpleCheckpoint(
    name="checkpoint_marital",
    data_context=context,
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": "Diabetes",
        },
    ],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)

    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")
  

    return df_validation
    