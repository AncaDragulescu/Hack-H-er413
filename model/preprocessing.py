import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# helper function, list difference
# https://stackoverflow.com/questions/6486450/python-compute-list-difference
diff = lambda l1,l2: [x for x in l1 if x not in l2]

# TODO: Dealing with test data that isn't coming from hard drive

# hardcoded input of training data
data = pd.read_csv("C:\\Users\\CodeB\\Documents\\hackher413\\widsdatathon2020\\training_v2.csv")
df = pd.DataFrame(data)

# columns that have text format:
obj_df = df.select_dtypes(include=['object']).copy()
text_cols = list(obj_df.columns.values)

for header in text_cols:
    # make all text values lower case
    # so that "Cardiovascular" and "cardiovascular" will be the same, for example
    df[header] = df[header].str.lower()

    # drop the column if more than 50% of cells is NA
    percent_na = df[header].isna().sum() / len(df[header])
    if percent_na >= 0.5:
        df = df.drop(header, axis=1)
        text_cols = diff(text_cols, [header])
    else:
        # replace text column NA cells with "other"
        df[header] = df[header].fillna(value="other")

# impute 'categorical' NA cells with mode
cat_cols = ["hospital_id", "hospital_death", "elective_surgery", \
"icu_id", "readmission_status", "apache_post_operative", "arf_apache", \
"gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache", \
"intubated_apache", "ventilated_apache", "aids", "cirrhosis", "diabetes_mellitus", \
"hepatic_failure", "immunosuppression", "leukemia", "lymphoma", \
"solid_tumor_with_metastasis"]
for header in cat_cols:
    # drop the column if more than 50% of cells is NA
    percent_na = df[header].isna().sum() / len(df[header])
    if percent_na >= 0.5:
        df = df.drop(header, axis=1)
        cat_cols = diff(cat_cols, [header])
    else:
        # if the percent of NAs is above 10%,
        # add column stating whether imputation happened (1) or not (0)
        if percent_na > 0.1:
            df[header + "_imputed"] = df[header].isna().astype(int)
        # impute categorical NAs w mode
        mode_imputer = SimpleImputer(strategy='most_frequent')
        og = np.array(df[header]).reshape(-1, 1)
        df[header] = mode_imputer.fit_transform(og)


# impute continuous NA cells with mean
all_cols = list(df.columns.values)
cont_cols = diff(all_cols, text_cols)
cont_cols = diff(cont_cols, cat_cols)

for header in cont_cols:
    # drop the column if more than 50% of cells is NA
    percent_na = df[header].isna().sum() / len(df[header])
    if percent_na >= 0.5:
        df = df.drop(header, axis=1)
    else:
        # if the percent of NAs is above 10%,
        # add column stating whether imputation happened (1) or not (0)
        if percent_na > 0.1:
            df[header + "_imputed"] = df[header].isna().astype(int)
        # impute categorical NAs w mo
        mean_imputer = SimpleImputer(strategy='mean')
        og = np.array(df[header]).reshape(-1, 1)
        df[header] = mean_imputer.fit_transform(og)

# text to one hot encoding
# https://pbpython.com/categorical-encoding.html
df = pd.get_dummies(df, columns=text_cols, prefix=text_cols)

# column name cleaning
df.columns = [c.replace(' ', '_').lower() for c in df.columns.values]
df.to_csv("training_v2_cleaned.csv", index=False)
