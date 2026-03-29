import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import numpy as np

st.set_page_config(page_title="Car Prediction App")

st.title('🚗 Car Price Prediction App')
st.write("This app predicts car price using Machine Learning")
st.write("Enter car details:")
df = pd.read_csv("Cleaned_Dataset.csv")

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['FuelType']=le.fit_transform(df['FuelType'])

x= df[['Age', 'KM', 'FuelType', 'HP', 'MetColor', 'Automatic', 'CC','Doors', 'Weight']]
y=df['Price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=42)
model.fit(x_train,y_train)

age = st.number_input("Age of car", min_value=0)
km = st.number_input("KM driven", min_value=0)
hp = st.number_input("Horsepower (HP)", min_value=0)
metcolor = st.selectbox("Metallic Color", [0, 1])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
automatic = st.selectbox("Automatic", [0, 1])
cc = st.number_input("Engine CC", 0)
doors = st.selectbox("Doors", [2,3,4,5])
weight = st.number_input("Weight", 0)

if fuel == "Petrol":
    fuel_encoded = 2
elif fuel == "Diesel":
    fuel_encoded = 1
else:
    fuel_encoded = 0

if st.button("Predict Price"):
    input_data = np.array([[age, km, hp, metcolor, automatic, cc, doors, weight, fuel_encoded]])
    
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {int(prediction[0])}")