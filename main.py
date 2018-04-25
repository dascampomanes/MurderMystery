#!/usr/bin/env python


import scipy
import os
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


filename = "database.csv"

columns = [0]
columns.extend(range(3, 17))

dataset = pd.read_csv(filename, usecols=columns)


enc = preprocessing.LabelEncoder()

enc.fit(dataset.Agency_Type)
agency_type = enc.transform(dataset.Agency_Type)

# enc.fit(dataset.City)
# city = enc.transform(dataset.City)
#
# enc.fit(dataset.State)
# state = enc.transform(dataset.State)

enc.fit(dataset.Month)
month = enc.transform(dataset.Month)

enc.fit(dataset.Crime_Type)
crime_type = enc.transform(dataset.Crime_Type)

enc.fit(dataset.Crime_Solved)
crime_solved = enc.transform(dataset.Crime_Solved)

enc.fit(dataset.Victim_Sex)
victim_Sex = enc.transform(dataset.Victim_Sex)

age = []


enc.fit(dataset.Victim_Race)
victim_race = enc.transform(dataset.Victim_Race)

enc.fit(dataset.Perpetrator_Sex)
perpetrator_sex = enc.transform(dataset.Perpetrator_Sex)

enc.fit(dataset.Perpetrator_Race)
perpetrator_race = enc.transform(dataset.Perpetrator_Race)

enc.fit(dataset.Relationship)
relationship = enc.transform(dataset.Relationship)

enc.fit(dataset.Weapon)
weapon = enc.transform(dataset.Weapon)

# enc.fit(dataset.Record_Source)
# record_Source = enc.transform(dataset.Record_Source)

data_matrix = np.c_[agency_type, dataset.Year.values, month, crime_type, crime_solved, victim_Sex, dataset.Victim_Age.values,
               victim_race, perpetrator_sex, dataset.Perpetrator_Age.values, perpetrator_race, relationship, weapon,
               dataset.Additional_Victims_Count.values, dataset.Additional_Perpetrators_Count.values]
data_train, data_test, y_train, y_test = train_test_split(data_matrix, data_matrix[:,3], test_size=0.3)


# testing Naive Bayes
naive_bayes = GaussianNB()

bayes_train_data = np.delete(data_train, 4, 1)
bayes_test_data = np.delete(data_test, 4, 1)

naive_bayes.fit(bayes_train_data, data_train[:, 4])
nb_result = naive_bayes.predict(bayes_test_data)

conf_matrix = confusion_matrix(data_test[:,4], nb_result)

print(conf_matrix)




