import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import streamlit as st

# Load the passenger data

passengers = pd.read_csv("passengers.csv")

# Update sex column to numerical
passengers.Sex = passengers["Sex"].map({"male": 0, "female": 1})
#print(passengers)

# Fill the nan values in the age column
passengers["Age"].fillna(value = np.mean(passengers.Age), inplace = True)
#print(passengers['Age'].values)
# Create a first class column
passengers["FirstClass"] = passengers.Pclass.apply(lambda x: 1 if x ==1 else 0)

# Create a second class column


passengers["SecondClass"] = passengers.Pclass.apply(lambda x: 1 if x ==2 else 0)
#print(passengers)
# Select the desired features
features = passengers[["Sex","Age","FirstClass","SecondClass"]]
survival = passengers["Survived"]

# Perform train, test, split
train_features, test_features, train_labels, test_labels = train_test_split(features,survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()

train_featues = scaler.fit_transform(train_features)

test_features = scaler.transform(test_features)
# Create and train the model
model = LogisticRegression()
model.fit(train_features,train_labels)

# Score the model on the train data
print(model.score(train_features,train_labels))

# Score the model on the test data
print(model.score(test_features,test_labels))

# Analyze the coefficients
print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))


"""
# Would you survive being on the Titanic?
"""

st.image("https://cyber-breeze.com/wp-content/uploads/2016/07/Titanic-696x456.jpg")

Gender = st.radio("Gender",["Male","Female"])
Gender1 = 0
if Gender == "Male":
    Gender1 == 0
if Gender == "Female":
    Gender1 == 1

Age = st.text_input("Age")


Class = st.radio("Class",["First Class", "Second Class", "Third Class"])
'''For reference, a first class ticket on the Titanic would cost around $5000 today, a second class ticket would cost
around $1000, and a third class ticket would cost around $500'''

first = 0
second = 0

if Class == "First Class":
    first = 1.0
    second = 0.0

if Class == "Second Class":
    second = 1.0
    first = 0.0

enter = st.button("Enter")

if enter == True:
    output = np.array([Gender1,float(Age),first,second])
    output1 = np.array([output])
    scaler.transform(output1)
    '''### Results'''
    if model.predict(output1) == 0:
        st.write("Unfortunately, based on the data you input you probably **would not** survive the Titanic")
    if model.predict(output1) == 1:
        st.write("Congratulations, based on the data you input you probably **would** survive the Titanic")
    st.write("It's important to note that I used logistic regression in my machine learning model, if I had used another classification "
         "model, like K-Means clustering, the output may have been different")


st.write("If you're interested in seeing the data I used click Open table")
Data = st.button("Open table")
Close = st.button("Close table")
if Data == True and Close == False:
    st.table(passengers[["Survived","Pclass","Name","Sex","Age"]])

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
