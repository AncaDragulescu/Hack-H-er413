import pandas as pd
import json

diff = lambda l1,l2: [x for x in l1 if x not in l2]

cat_cols = ["gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache", \
"intubated_apache", "ventilated_apache", "aids", "cirrhosis", \
"hepatic_failure", "immunosuppression", "leukemia", "lymphoma", \
"solid_tumor_with_metastasis"]

keep_cols = ["age", "albumin_apache", "bun_apache", "fio2_apache", "gcs_eyes_apache", "gcs_motor_apache", "gcs_unable_apache", "gcs_verbal_apache", "heart_rate_apache", "intubated_apache", "map_apache", "paco2_apache", "pao2_apache", "ph_apache", "sodium_apache", "temp_apache", "urineoutput_apache", "ventilated_apache", "wbc_apache", "d1_diasbp_invasive_min", "d1_diasbp_min", "d1_diasbp_noninvasive_min", "d1_heartrate_min", "d1_mbp_invasive_min", "d1_mbp_min", "d1_mbp_noninvasive_min", "d1_resprate_max", "d1_spo2_min", "d1_sysbp_invasive_min", "d1_sysbp_min", "d1_sysbp_noninvasive_min", "d1_temp_max", "h1_diasbp_min", "h1_diasbp_noninvasive_max", "h1_heartrate_min", "h1_mbp_min", "h1_resprate_max", "h1_spo2_min", "h1_sysbp_invasive_min", "h1_sysbp_min", "h1_sysbp_noninvasive_min", "h1_temp_max", "d1_albumin_min", "d1_bun_min", "d1_calcium_max", "d1_creatinine_min", "d1_hco3_max", "d1_inr_min", "d1_lactate_min", "d1_platelets_min", "d1_potassium_max", "d1_sodium_max", "d1_wbc_max", "h1_albumin_min", "h1_bilirubin_min", "h1_bun_min", "h1_calcium_min", "h1_creatinine_min", "h1_hco3_max", "h1_hemaglobin_min", "h1_hematocrit_max", "h1_inr_min", "h1_lactate_min", "h1_platelets_min", "h1_potassium_max", "h1_sodium_min", "h1_wbc_max", "d1_arterial_pco2_min", "d1_arterial_ph_min", "d1_arterial_po2_min", "d1_pao2fio2ratio_min", "h1_arterial_pco2_max", "h1_arterial_ph_max", "h1_arterial_po2_max", "h1_pao2fio2ratio_min", "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob", "aids", "cirrhosis", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma", "solid_tumor_with_metastasis"]

top20_cols = ['age', 'bun_apache', 'gcs_eyes_apache', 'gcs_motor_apache',
       'gcs_verbal_apache', 'heart_rate_apache', 'intubated_apache',
       'ventilated_apache', 'd1_diasbp_min', 'd1_diasbp_noninvasive_min',
       'd1_mbp_min', 'd1_mbp_noninvasive_min', 'd1_resprate_max',
       'd1_spo2_min', 'd1_sysbp_min', 'd1_sysbp_noninvasive_min',
       'h1_resprate_max', 'h1_sysbp_noninvasive_min', 'd1_bun_min',
       'apache_4a_icu_death_prob']

cat_cols = set(top20_cols).intersection(set(cat_cols))

cont_cols = diff(top20_cols, cat_cols)

# means from training data
cont_means = {'age': 62.31287821791894, 'albumin_apache': 2.9039798065704967, 'bun_apache': 25.905594735751748, 'fio2_apache': 0.596969513591312, 'heart_rate_apache': 99.69290067929302, 'map_apache': 88.0207931263834, 'paco2_apache': 42.14930271608169, 'pao2_apache': 131.3086694361759, 'ph_apache': 7.354006583581389, 'sodium_apache': 137.96012124780566, 'temp_apache': 36.41550325471853, 'urineoutput_apache': 1744.7154977483237, 'wbc_apache': 12.125366087686587, 'd1_diasbp_invasive_min': 46.75051519413824, 'd1_diasbp_min': 50.16407706650093, 'd1_diasbp_noninvasive_min': 50.242484707729545, 'd1_heartrate_min': 70.32076150600243, 'd1_mbp_invasive_min': 62.41997317719407, 'd1_mbp_min': 64.86877541897005, 'd1_mbp_noninvasive_min': 64.9436939147122, 'd1_resprate_max': 28.882328568469028, 'd1_spo2_min': 90.45448300677113, 'd1_sysbp_invasive_min': 93.85984538724063, 'd1_sysbp_min': 96.92527776869146, 'd1_sysbp_noninvasive_min': 96.98852616313935, 'd1_temp_max': 37.28425226609097, 'h1_diasbp_min': 62.85040288726789, 'h1_diasbp_noninvasive_max': 75.78147045675095, 'h1_heartrate_min': 83.66661214876844, 'h1_mbp_min': 79.40838267203122, 'h1_resprate_max': 22.629779856726962, 'h1_spo2_min': 95.17916762072988, 'h1_sysbp_invasive_min': 115.02475417879691, 'h1_sysbp_min': 116.32996412722296, 'h1_sysbp_noninvasive_min': 116.54612759368901, 'h1_temp_max': 36.70957612372924, 'd1_albumin_min': 2.9015886515543055, 'd1_bun_min': 23.73862276885502, 'd1_calcium_max': 8.37922431934404, 'd1_creatinine_min': 1.368475944522587, 'd1_hco3_max': 24.36333017129524, 'd1_inr_min': 1.4817790607656494, 'd1_lactate_min': 2.1262346668411243, 'd1_platelets_min': 196.6959471394459, 'd1_potassium_max': 4.251617218932976, 'd1_sodium_max': 139.12729602128377, 'd1_wbc_max': 12.474322724150339, 'h1_albumin_min': 3.026205663319268, 'h1_bilirubin_min': 1.0923814508303076, 'h1_bun_min': 25.72596960081995, 'h1_calcium_min': 8.284019223010915, 'h1_creatinine_min': 1.5349225736809393, 'h1_hco3_max': 22.46218529543249, 'h1_hemaglobin_min': 11.030009922257477, 'h1_hematocrit_max': 33.689262154765416, 'h1_inr_min': 1.4827746775266322, 'h1_lactate_min': 3.0096606206317533, 'h1_platelets_min': 195.86368344727575, 'h1_potassium_max': 4.197429263027052, 'h1_sodium_min': 137.90748748814235, 'h1_wbc_max': 13.453998124584304, 'd1_arterial_pco2_min': 38.46285974725502, 'd1_arterial_ph_min': 7.32482027782321, 'd1_arterial_po2_min': 104.02054539705387, 'd1_pao2fio2ratio_min': 224.03016363919937, 'h1_arterial_pco2_max': 44.73935003761735, 'h1_arterial_ph_max': 7.339200593154733, 'h1_arterial_po2_max': 163.99890724324794, 'h1_pao2fio2ratio_min': 235.95622582820545, 'apache_4a_hospital_death_prob': 0.08667528049458638, 'apache_4a_icu_death_prob': 0.0440574400575709}

def json_to_df(data):
    """
    Convert a JSON to a pandas DataFrame for use with the prediction model.

    Parameters:
     - data: a JSON object

    Returns:
     - a pandas DataFrame with missing values filled in
    """
    # lst = json.loads(data)
    # dict = lst[0]

    # categorical variables get replaced with -1
    for cat in cat_cols:
        if not data[cat]:
            data[cat] = -1

    # continuous variables get replaced with mean
    for cont in cont_cols:
        if not data[cont]:
            data[cont] = cont_means[cont]
    x = pd.DataFrame(data, index=[0])
    return x
