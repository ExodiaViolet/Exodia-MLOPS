MEDICATION_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide',
    'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin'
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