import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# helper function, list difference
# https://stackoverflow.com/questions/6486450/python-compute-list-difference
diff = lambda l1,l2: [x for x in l1 if x not in l2]

# hardcoded input of training data
data = pd.read_csv("C:\\Users\\CodeB\\Documents\\hackher413\\widsdatathon2020\\training_v2.csv")
df = pd.DataFrame(data)

# keep these columns (based on exploratory visualization)
keep_cols = ["hospital_death", "age", "albumin_apache", "bun_apache", "fio2_apache", "gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache", "heart_rate_apache", "intubated_apache", "map_apache", "paco2_apache", "pao2_apache", "ph_apache", "sodium_apache", "temp_apache", "urineoutput_apache", "ventilated_apache", "wbc_apache", "d1_diasbp_invasive_min", "d1_diasbp_min", "d1_diasbp_noninvasive_min", "d1_heartrate_min", "d1_mbp_invasive_min", "d1_mbp_min", "d1_mbp_noninvasive_min", "d1_resprate_max", "d1_spo2_min", "d1_sysbp_invasive_min", "d1_sysbp_min", "d1_sysbp_noninvasive_min", "d1_temp_max", "h1_diasbp_min", "h1_diasbp_noninvasive_max", "h1_heartrate_min", "h1_mbp_min", "h1_resprate_max", "h1_spo2_min", "h1_sysbp_invasive_min", "h1_sysbp_min", "h1_sysbp_noninvasive_min", "h1_temp_max", "d1_albumin_min", "d1_bun_min", "d1_calcium_max", "d1_creatinine_min", "d1_hco3_max", "d1_inr_min", "d1_lactate_min", "d1_platelets_min", "d1_potassium_max", "d1_sodium_max", "d1_wbc_max", "h1_albumin_min", "h1_bilirubin_min", "h1_bun_min", "h1_calcium_min", "h1_creatinine_min", "h1_hco3_max", "h1_hemaglobin_min", "h1_hematocrit_max", "h1_inr_min", "h1_lactate_min", "h1_platelets_min", "h1_potassium_max", "h1_sodium_min", "h1_wbc_max", "d1_arterial_pco2_min", "d1_arterial_ph_min", "d1_arterial_po2_min", "d1_pao2fio2ratio_min", "h1_arterial_pco2_max", "h1_arterial_ph_max", "h1_arterial_po2_max", "h1_pao2fio2ratio_min", "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob", "aids", "cirrhosis", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma", "solid_tumor_with_metastasis"]
print("KEEP COLS LEN: ", len(keep_cols))
df = df[keep_cols]

### text columns ###
obj_df = df.select_dtypes(include=['object']).copy()
text_cols = list(obj_df.columns.values)

for header in text_cols:
    # make all text values lower case
    # so that "Cardiovascular" and "cardiovascular" will be the same, for example
    df[header] = df[header].str.lower()

    # drop the column if more than 50% of cells is NA
    percent_na = df[header].isna().sum() / len(df[header])
    # convert to categorical
    # https://pbpython.com/categorical-encoding.html
    df[header] = df[header].astype('category')
    df[header] = df[header].cat.codes

### categorical columns ###
cat_cols = ["gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache", \
"intubated_apache", "ventilated_apache", "aids", "cirrhosis", \
"hepatic_failure", "immunosuppression", "leukemia", "lymphoma", \
"solid_tumor_with_metastasis"]
for header in cat_cols:
    # drop the column if more than 50% of cells is NA
    percent_na = df[header].isna().sum() / len(df[header])
    # convert to categorical
    df[header] = df[header].astype('category')
    df[header] = df[header].cat.codes

### continuous columns ###
all_cols = list(df.columns.values)
cont_cols = diff(all_cols, text_cols)
cont_cols = diff(cont_cols, cat_cols)

dict = {}
for header in cont_cols:
    # drop the column if more than 50% of cells is NA
    # drop rows w NAs
    # df = df[df[header].notna()]

    # replace NA w random sample from column
    # https://stackoverflow.com/questions/46384934/pandas-replace-nan-using-random-sampling-of-column-values
    # df = df.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
    # og = df[header].dropna()
    # n = len(df[header])
    # samples = list(og.sample(n, replace=True))
    # mask = list(df[header].isna())
    # # print("mask shape: ", mask.shape)
    # # print("samples shape: ", samples.shape)
    # df[header] = df[header].mask(mask, samples)

    # df[header] = df[header].apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
    # # impute with -1 constant
    impute = SimpleImputer(strategy='constant', fill_value=0)
    arr = np.array(df[header]).reshape(-1, 1)
    df[header] = impute.fit_transform(arr)

print("shape: ", df.shape)

df.to_csv("full_constant_0.csv", index=False)
