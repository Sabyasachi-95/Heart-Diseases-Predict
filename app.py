from flask import Flask,render_template,redirect,url_for,request
import pickle
import numpy as np



import re

app = Flask(__name__)

#heartdiseases model read
filename = open('HeartDiseases/heartdiseasespredictmodel.pkl', 'rb')
clf = pickle.load(filename)
filename.close()

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/hdpredict', methods=['GET','POST'])
def hdpredict():
    if request.method == 'POST':
        na = request.form['na']
        ag = int(request.form['ag'])
        cp = int(request.form['cp'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = float(request.form['exang'])
        sl = float(request.form['sl'])
        
        data = np.array([[ag,cp,chol,fbs,restecg,thalach,exang,sl]])
        my_prediction = clf.predict(data)
        my_prediction_proba = clf.predict_proba(data)[0][1]
        
        return render_template('hdpshow.html',name=na,prediction=my_prediction,proba=my_prediction_proba)
    return render_template('hdp.html')

        
if __name__ == '__main__':
	app.run(debug=True)
