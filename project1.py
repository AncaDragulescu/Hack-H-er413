from flask import Flask, request, render_template, url_for, jsonify, redirect
import json
import app_preprocessing

import pickle

app = Flask(__name__)

with open('input2.json', 'r') as f:
    stats_dict = json.load(f)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', stats_dict=stats_dict)

@app.route("/predict", methods=['GET','POST'])
def pred():
    if request.method == 'POST':
        age = request.form.get("age")
        bun_apache = request.form.get("bun_apache")
        gcs_eyes_apache = request.form.get("gcs_eyes_apache")
        gcs_motor_apache = request.form.get("gcs_motor_apache")
        gcs_verbal_apache = request.form.get("gcs_verbal_apache")
        heart_rate_apache = request.form.get("heart_rate_apache")
        intubated_apache = request.form.get("intubated_apache")
        ventilated_apache = request.form.get("ventilated_apache")
        d1_diasbp_min = request.form.get("d1_diasbp_min")
        d1_diasbp_noninvasive_min = request.form.get("d1_diasbp_noninvasive_min")
        d1_mbp_min = request.form.get("d1_mbp_min")
        d1_mbp_noninvasive_min = request.form.get("d1_mbp_noninvasive_min")
        d1_resprate_max = request.form.get("d1_resprate_max")
        d1_spo2_min = request.form.get("d1_spo2_min")
        d1_sysbp_min = request.form.get("d1_sysbp_min")
        d1_sysbp_noninvasive_min = request.form.get("d1_sysbp_noninvasive_min")
        h1_resprate_max = request.form.get("h1_resprate_max")
        h1_sysbp_noninvasive_min = request.form.get("h1_sysbp_noninvasive_min")
        d1_bun_min = request.form.get("d1_bun_min")
        apache_4a_icu_death_prob = request.form.get("apache_4a_icu_death_prob")

        jsonOutput = {
                "age": age,
                "bun_apache": bun_apache,
                "gcs_eyes_apache": gcs_eyes_apache,
                "gcs_motor_apache": gcs_motor_apache,
                "gcs_verbal_apache": gcs_verbal_apache,
                "heart_rate_apache": heart_rate_apache,
                "intubated_apache": intubated_apache,
                "ventilated_apache": ventilated_apache,
                "d1_diasbp_min": d1_diasbp_min,
                "d1_diasbp_noninvasive_min": d1_diasbp_noninvasive_min,
                "d1_mbp_min": d1_mbp_min,
                "d1_mbp_noninvasive_min": d1_mbp_noninvasive_min,
                "d1_resprate_max": d1_resprate_max,
                "d1_spo2_min": d1_spo2_min,
                "d1_sysbp_min": d1_sysbp_min,
                "d1_sysbp_noninvasive_min": d1_sysbp_noninvasive_min,
                "h1_resprate_max": h1_resprate_max,
                "h1_sysbp_noninvasive_min": h1_sysbp_noninvasive_min,
                "d1_bun_min": d1_bun_min,
                "apache_4a_icu_death_prob": apache_4a_icu_death_prob
            }
        
        df = app_preprocessing.json_to_df(jsonOutput)
        model = pickle.load(open("clf.pkl", "rb"))
        result = model.predict_proba(df)
        val = round(float(list(result)[0][0]),2)
    return render_template('predict.html', output=jsonOutput, prediction=val)

@app.route("/about")
def about():
    return render_template('about.html', title='About')

if __name__ == '__main__':
    app.run(debug=True)