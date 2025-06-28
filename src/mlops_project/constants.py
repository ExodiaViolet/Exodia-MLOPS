MEDICATION_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'tolazamide', 'insulin', 'glyburide_metformin', 'glipizide_metformin'
]

NUMERICAL_COLUMNS = ['encounter_id',
                     'patient_nbr',
                        'admission_type_id',
                        'discharge_disposition_id',
                        'admission_source_id',
                        'time_in_hospital',
                        'num_lab_procedures',
                        'num_procedures',
                        'num_medications',
                        'number_outpatient',
                        'number_emergency',
                        'number_inpatient',
                        'number_diagnoses']

MEDICATION_COLUMNS_CORRECTED = [
    'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'tolazamide', 'insulin', 'glyburide_metformin', 'glipizide_metformin'
]

# Expected values for medication-related columns
MEDICATION_VALUE_SET = ["No", "Steady", "Up", "Down"]

# Expected values for age groups
AGE_VALUE_SET = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
]

# Expected values for race
RACE_VALUE_SET = ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']

# Expected values for gender
GENDER_VALUE_SET = ['Female', 'Male']

# A1C result values
A1C_RESULT_SET = [">7", ">8", "Norm", "none", "?"]

# Max glucose serum values
MAX_GLU_SERUM_SET = [">200", ">300", "Norm", "none", "?"]

# Change and diabetesMed options
CHANGE_VALUE_SET = ["No", "Ch"]
DIABETES_MED_SET = ["Yes", "No"]

READMITTED_SET = ["NO", ">30", "<30"]

COLS_TO_DROP = [
    'acetohexamide', 
    'troglitazone', 
    'examide', 
    'citoglipton', 
    'glimepiride-pioglitazone', 
    'metformin-rosiglitazone', 
    'metformin-pioglitazone', 
    'weight', 
    'payer_code', 
    'medical_specialty'
]

CATEGORICAL_FEATURES_DESCRIPTIONS = {
    "race": "Patient’s race. Validation: String category: 'Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other', or '?'.",
    "gender": "Patient’s gender. Validation: String: 'Male', 'Female', or 'Unknown/Invalid'.",
    "age": "Age range of the patient. Validation: String in format '[x-y)' where x and y are decade boundaries (e.g., '[60-70)').",
    "diag_1": "Primary diagnosis code (ICD-9). Validation: String; either ICD-9 code or numeric.",
    "diag_2": "Secondary diagnosis code (ICD-9). Validation: String; either ICD-9 code or numeric.",
    "diag_3": "Additional diagnosis code (ICD-9). Validation: String; either ICD-9 code or numeric.",
    "max_glu_serum": "Glucose serum test result. Validation: String: 'None', '>200', '>300', 'Norm'.",
    "a1cresult": "A1C test result. Validation: String: 'None', '>7', '>8', 'Norm'.",
    "change": "Indicates if there was a change in diabetes medications. Validation: String: 'Ch' or 'No'.",
    "diabetesmed": "Indicates if any diabetes medication was prescribed. Validation: String: 'Yes' or 'No'.",
    "metformin": "Status of metformin medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "repaglinide": "Status of repaglinide medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "nateglinide": "Status of nateglinide medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "chlorpropamide": "Status of chlorpropamide medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "glimepiride": "Status of glimepiride medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "glipizide": "Status of glipizide medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "glyburide": "Status of glyburide medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "tolbutamide": "Status of tolbutamide medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "pioglitazone": "Status of pioglitazone medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "rosiglitazone": "Status of rosiglitazone medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "acarbose": "Status of acarbose medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "miglitol": "Status of miglitol medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "tolazamide": "Status of tolazamide medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "insulin": "Status of insulin medication. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "glyburide_metformin": "Status of glyburide-metformin combination drug. Validation: String: 'No', 'Steady', 'Up', or 'Down'.",
    "glipizide_metformin": "Status of glipizide-metformin combination drug. Validation: String: 'No', 'Steady', 'Up', or 'Down'."
}
TARGET_FEATURES_DESCRIPTIONS = {
    "readmitted": "Indicates if the patient was readmitted after discharge. Validation: String: '<30' (readmitted within 30 days), '>30', or 'NO'."
}

NUMERICAL_FEATURES_DESCRIPTIONS = {
    "encounter_id": "Unique identifier for the hospital encounter. Validation: Integer, always positive, unique per row.",
    "patient_nbr": "Identifier for the patient (may appear multiple times across encounters). Validation: Integer, always positive, not necessarily unique.",
    "admission_type_id": "Numeric ID indicating the type of hospital admission (e.g., emergency, urgent). Validation: Integer, categorical codes starting from 1.",
    "discharge_disposition_id": "Numeric ID indicating discharge destination (e.g., home, other care facility, death). Validation: Integer, categorical codes starting from 1.",
    "admission_source_id": "Numeric ID representing source of hospital admission (e.g., referral, ER). Validation: Integer, categorical codes starting from 1.",
    "time_in_hospital": "Number of days the patient spent in the hospital during the encounter. Validation: Integer, non-negative (typically in range 1–14).",
    "num_lab_procedures": "Number of lab tests performed during the encounter. Validation: Integer, non-negative.",
    "num_procedures": "Number of procedures (non-lab) performed during the encounter. Validation: Integer, non-negative, typically in range 0–6.",
    "num_medications": "Number of distinct medications administered during the encounter. Validation: Integer, non-negative.",
    "number_outpatient": "Number of outpatient visits in the year before the hospital encounter. Validation: Integer, non-negative.",
    "number_emergency": "Number of emergency visits in the year before the hospital encounter. Validation: Integer, non-negative.",
    "number_inpatient": "Number of inpatient visits in the year before the current hospital encounter. Validation: Integer, non-negative.",
    "number_diagnoses": "Number of diagnoses recorded during the hospital stay. Validation: Integer, non-negative, typically in range 1–16."
}

ADMISSION_TYPE_MAP = {
    '2': '1', 
    '7': '1',
    '6': '5', 
    '8': '5'
}

DISCHARGE_DISPOSITION_MAP = {
    '6': '1',
    '8': '1',
    '9': '1',
    '13': '1',
    '3': '2', 
    '4': '2', 
    '5': '2', 
    '14': '2', 
    '22': '2', 
    '23': '2', 
    '24': '2',
    '12': '10', 
    '15': '10', 
    '16': '10', 
    '17': '10',
    '25': '18', 
    '26': '18'
}

ADMISSION_SOURCE_MAP = {
    '2': '1',
    '3': '1',
    '5': '4',
    '6': '4',
    '10': '4',
    '22': '4',
    '25': '4',
    '15': '9',
    '17': '9',
    '20': '9',
    '21': '9',
    '13': '11',
    '14': '11'
}

A1CRESULT_MAP = {
    '>7': 1, 
    '>8': 1,
    'Norm': 0
}

MAX_GLU_SERUM_MAP = {
    '>200': 1,
    '>300': 1,
    'Norm': 0
}

AGE_MAP = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}