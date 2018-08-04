

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#loading ANN mode

from keras.models import load_model
model = load_model('ANNmodel.h5')
#user input 
age=input("enter age:")
sex=input("enter sex(M/F):")
if sex=='M':
    sex=1
else:
    sex=0
chest_pain=input("enter chest pain:")
blood_pressure=input("enter blood pressure:")
serum_cholestoral=input("enter serum cholestoral:")
fasting_blood_sugar=input("enter fasting blood sugar:")
electrocardiographic=input("enter electrocardiographic:")
max_heart_rate=input("enter max heart rate:")
induced_angina=input("enter induced angina:")
ST_depression=input("enter ST depression:")
slope=input("enter slope:")
vessels=input("enter vessels:")
thal=input("enter thal:")
data=np.array([[age,sex,chest_pain,blood_pressure,serum_cholestoral,fasting_blood_sugar,electrocardiographic,
                max_heart_rate,induced_angina,ST_depression,slope,vessels,thal]])
diagonis=np.around(model.predict(data),1)