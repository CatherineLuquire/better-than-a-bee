import flask
from flask import render_template
import joblib
import sklearn
import numpy as np
import pandas as pd
import os
import glob
from joblib import load

app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/references')
def references():
    return render_template('references.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/viz1')
def viz1():
    return render_template('viz1.html')

@app.route('/viz2')
def viz2():
    return render_template('viz2.html')

@app.route('/viz3')
def viz3():
    return render_template('viz3.html')

@app.route('/viz4')
def viz4():
    return render_template('viz4.html')

@app.route('/viz5')
def viz5():
    return render_template('viz5.html')

@app.route('/viz6')
def viz6():
    return render_template('viz6.html')

@app.route('/model', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'POST':
        PV_Elastic = joblib.load('models/kaggle_LRE_Elastic_2.sav')
        PV_Lasso = joblib.load('models/kaggle_LRE_Lassso_2.sav')
        PV_LRModel = joblib.load('models/kaggle_LRE_Model_2.sav')
        PV_Ridge = joblib.load('models/kaggle_LRE_Ridge_2.sav')
        HP_Elastic = joblib.load('models/kaggle_LRE_Elastic.sav')
        HP_Lasso = joblib.load('models/kaggle_LRE_Lassso.sav')
        HP_LRModel = joblib.load('models/kaggle_LRE_Linear.sav')
        HP_Ridge = joblib.load('models/kaggle_LRE_Ridge.sav')
        HPX_scaler = joblib.load('models/X_scaler1.sav')
        HPy_scaler = joblib.load('models/y_scaler1.sav')
        PVX_scaler = joblib.load('models/X_scaler2.sav')
        PVy_scaler = joblib.load('models/y_scaler2.sav')

        nCLOTHIANIDIN = flask.request.form['nCLOTHIANIDIN']
        nIMIDACLOPRID = flask.request.form['nIMIDACLOPRID']
        nTHIAMETHOXAM = flask.request.form['nTHIAMETHOXAM']
        nACETAMIPRID = flask.request.form['nACETAMIPRID']
        nTHIACLOPRID = flask.request.form['nTHIACLOPRID']
        nAllNeonic = flask.request.form['nAllNeonic']
        numcol = flask.request.form['numcol']
        totalprod = flask.request.form['totalprod']

        X_PV = pd.DataFrame({'nCLOTHIANIDIN': [nCLOTHIANIDIN], 
                        'nIMIDACLOPRID': [nIMIDACLOPRID],
                        'nTHIAMETHOXAM': [nTHIAMETHOXAM],
                        'nACETAMIPRID': [nACETAMIPRID],
                        'nTHIACLOPRID': [nTHIACLOPRID],
                        'nAllNeonic': [nAllNeonic],
                        'numcol': [numcol],
                        'totalprod': [totalprod]})

        X_HP = pd.DataFrame({'nCLOTHIANIDIN': [nCLOTHIANIDIN], 
                        'nIMIDACLOPRID': [nIMIDACLOPRID],
                        'nTHIAMETHOXAM': [nTHIAMETHOXAM],
                        'nACETAMIPRID': [nACETAMIPRID],
                        'nTHIACLOPRID': [nTHIACLOPRID],
                        'nAllNeonic': [nAllNeonic],
                        'numcol': [numcol]})

        PVX_scaled = PVX_scaler.transform(X_PV)
        HPX_scaled = HPX_scaler.transform(X_HP)

        HPLR_y = int(HPy_scaler.inverse_transform(HP_LRModel.predict(HPX_scaled))[0][0])
        HPE_y = int(HPy_scaler.inverse_transform(HP_Elastic.predict(HPX_scaled))[0])
        HPL_y = int(HPy_scaler.inverse_transform(HP_Lasso.predict(HPX_scaled))[0])
        HPR_y = int(HPy_scaler.inverse_transform(HP_Ridge.predict(HPX_scaled))[0][0])
        PVLR_y = int(PVy_scaler.inverse_transform(PV_LRModel.predict(PVX_scaled))[0][0])
        PVE_y = int(PVy_scaler.inverse_transform(PV_Elastic.predict(PVX_scaled))[0])
        PVL_y = int(PVy_scaler.inverse_transform(PV_Lasso.predict(PVX_scaled))[0])
        PVR_y = int(PVy_scaler.inverse_transform(PV_Ridge.predict(PVX_scaled))[0][0])

        return(flask.render_template('model.html', HPLR_y=HPLR_y, HPE_y=HPE_y, HPL_y=HPL_y, HPR_y=HPR_y, PVLR_y=PVLR_y, PVE_y=PVE_y, PVL_y=PVL_y, PVR_y=PVR_y))
    if flask.request.method == 'GET':
        return(flask.render_template('model.html'))

if __name__ == '__main__':
    app.run()
