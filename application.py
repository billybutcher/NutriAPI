from flask import Flask, request
import pickle
import json
import numpy as np

names = ["1b_Specific Energy","2ai_Caloric","2aii_AMDR Wardlaw","2aiii_AMDR PDRI","2biv1_Protein RENI","2biv2_Micronutrient RENI","2biv3_Vitamin RENI","2biv4_Mineral RENI","3a_IRON","3b_VIT A"]


app = Flask(__name__)
from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

feats = ['CHILD_SEX','IDD_SCORE','AGE','HHID_count','HH_AGE','FOOD_EXPENSE_WEEKLY','NON-FOOD_EXPENSE_WEEKLY','HDD_SCORE','FOOD_INSECURITY','YoungBoys','YoungGirls',
'AverageMonthlyIncome','BEN_4PS','AREA_TYPE','FOODPOOR','JOBSECURITY','FOOD_EXPENSE_WEEKLY_pc','NON-FOOD_EXPENSE_WEEKLY_pc','AverageMonthlyIncome_pc']

kidata = [0,7,5,4,21.5,1457.5,4843.75,8,3,1,0,3,2,0,1,2,364.375,1210.9375,0.75]
kidd = {'CHILD_SEX': 0, 'IDD_SCORE': 7, 'AGE': 5, 'HHID_count': 4, 'HH_AGE': 21.5, 
'FOOD_EXPENSE_WEEKLY': 1457.5, 'NON-FOOD_EXPENSE_WEEKLY': 4843.75, 'HDD_SCORE': 8, 'FOOD_INSECURITY': 3, 
'YoungBoys': 1, 'YoungGirls': 0, 'AverageMonthlyIncome': 3, 'BEN_4PS': 2, 'AREA_TYPE': 0, 'FOODPOOR': 1, 
'JOBSECURITY': 2, 'FOOD_EXPENSE_WEEKLY_pc': 364.375, 'NON-FOOD_EXPENSE_WEEKLY_pc': 1210.9375, 'AverageMonthlyIncome_pc': 0.75}
kiddd = {
    "X":{"CHILD_SEX": 0, "IDD_SCORE": 7, "AGE": 5, "HHID_count": 4, "HH_AGE": 21.5, 
"FOOD_EXPENSE_WEEKLY": 1457.5, "NON-FOOD_EXPENSE_WEEKLY": 4843.75, "HDD_SCORE": 8, "FOOD_INSECURITY": 3, 
"YoungBoys": 1, "YoungGirls": 0, "AverageMonthlyIncome": 3, "BEN_4PS": 2, "AREA_TYPE": 0, "FOODPOOR": 1, 
"JOBSECURITY": 2, "FOOD_EXPENSE_WEEKLY_pc": 364.375, "NON-FOOD_EXPENSE_WEEKLY_pc": 1210.9375, "AverageMonthlyIncome_pc": 0.75}
}


class Kid(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	X = db.Column(db.Text)
	def __repr__(self):
		return self.X

@app.route('/')
def index():
	return 'Welcome to NutriAPI!'

@app.route('/data/<id>')
def get_data(id):
	bakla = Kid.query.get_or_404(id)
	print(bakla.X)
	return {"X":bakla.X}

@app.route('/data', methods = ['POST'])
def send_data():
	boi = Kid(X = request.json['X'])
	db.session.add(boi)
	db.session.commit()
	return {'id':boi.id}

@app.route('/pred/<id>')
def get_pred(id):
	gurl = Kid.query.get_or_404(id)
	x = json.loads(gurl.X)
	x = np.array([x])
	scalerfile = "pickles/scaler.sav"
	scaler = pickle.load(open(scalerfile, 'rb'))
	x = scaler.transform(x)
	preds = {}
	for name in names:
		picklename = "pickles/"+name + ".pickle"
		with open(picklename, 'rb') as file:  
			clf = pickle.load(file)
			preds[name] = clf.predict(x)[0]
	return {"predictions": preds}

