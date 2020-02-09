from flask import Flask, request, render_template, url_for, jsonify, redirect
import json
app = Flask(__name__)

with open('input2.json', 'r') as f:
    stats_dict = json.load(f)
print(stats_dict)

defaultVal = {}
defaultVal["age"]="10"

@app.route("/", methods=['GET','POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = request.form.get('intubated_apache')
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
    return age

@app.route("/about")
def about():
    return render_template('about.html', title='About')

def setDefault(val, str):
    if len(val)==0:
        val = defaultVal[str]

if __name__ == '__main__':
    app.run(debug=True)