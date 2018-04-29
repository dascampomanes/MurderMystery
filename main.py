#!/usr/bin/env python


from tkinter import *

import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import metrics
from scipy import interp

filename = "database.csv"

columns = [0]
columns.extend(range(3, 17))

dataset = pd.read_csv(filename, usecols=columns)

cities_dataset = pd.read_csv(filename, usecols=[1])

enc = preprocessing.LabelEncoder()

enc.fit(dataset.Agency_Type)
agency_type = enc.transform(dataset.Agency_Type)
#['County Police' 'Municipal Police' 'Regional Police' 'Sheriff' 'Special Police' 'State Police' 'Tribal Police']

cities_set = set(cities_dataset.City)
city_enc = preprocessing.LabelEncoder()
city_enc.fit(cities_dataset.City)
cities = city_enc.transform(cities_dataset.City)
#
# enc.fit(dataset.State)
# state = enc.transform(dataset.State)

enc.fit(dataset.Month)
month = enc.transform(dataset.Month)

enc.fit(dataset.Crime_Type)
crime_type = enc.transform(dataset.Crime_Type)

enc.fit(dataset.Crime_Solved)
crime_solved = enc.transform(dataset.Crime_Solved)

victim_sex_enc = preprocessing.LabelEncoder()
victim_sex_enc.fit(dataset.Victim_Sex)
victim_Sex = victim_sex_enc.transform(dataset.Victim_Sex)

age = []

enc.fit(dataset.Victim_Race)
victim_race = enc.transform(dataset.Victim_Race)

enc.fit(dataset.Perpetrator_Sex)
perpetrator_sex = enc.transform(dataset.Perpetrator_Sex)

enc.fit(dataset.Perpetrator_Race)
perpetrator_race = enc.transform(dataset.Perpetrator_Race)

enc.fit(dataset.Relationship)
relationship = enc.transform(dataset.Relationship)

weapon_enc = preprocessing.LabelEncoder()
weapon_enc.fit(dataset.Weapon)
weapon = weapon_enc.transform(dataset.Weapon)
#['Blunt Object' 'Drowning' 'Drugs' 'Explosives' 'Fall' 'Fire' 'Firearm' 'Gun' 'Handgun' 'Knife' 'Poison' 'Rifle' 'Shotgun' 'Strangulation' 'Suffocation' 'Unknown']

data_matrix = np.c_[
    agency_type, dataset.Year.values, month, crime_type, crime_solved, victim_Sex, dataset.Victim_Age.values,
    victim_race, perpetrator_sex, dataset.Perpetrator_Age.values, perpetrator_race, relationship, weapon,
    dataset.Additional_Victims_Count.values, dataset.Additional_Perpetrators_Count.values]

data_train, data_test, y_train, y_test = train_test_split(data_matrix, data_matrix[:, 3], test_size=0.3)

# testing Naive Bayes
naive_bayes = GaussianNB()

bayes_train_data = np.delete(data_train, 4, 1)
bayes_test_data = np.delete(data_test, 4, 1)

naive_bayes.fit(bayes_train_data, data_train[:, 4])
nb_result = naive_bayes.predict(bayes_test_data)

conf_matrix = confusion_matrix(data_test[:, 4], nb_result)
print("partitioning bayes crime solved confusion matrix:")
print(conf_matrix)
print("Accuracy: %0.4f" % (naive_bayes.fit(bayes_train_data, data_train[:, 4]).score(bayes_test_data, data_test[:, 4])))

print("cross validation bayes crime solved confusion matrix:")
#cross validation
scores = []
data_without_target = np.delete(data_matrix, 4, 1)
target = data_matrix[:,4]
k_fold = KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(target):
    naive_bayes.fit(data_without_target[train_indices], target[train_indices])
    result = naive_bayes.predict(data_without_target[test_indices])
    conf_matrix = confusion_matrix(target[test_indices], result)
    print(conf_matrix)
    scores.append(naive_bayes.fit(data_without_target[train_indices], target[train_indices]).score(
        data_without_target[test_indices], target[test_indices]))

print("Accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores), np.std(scores) * 2))

# testing knn
knn = KNeighborsClassifier()

knn_train_data = np.delete(data_train, 8, 1)
knn_test_data = np.delete(data_test, 8, 1)

knn.fit(knn_train_data, data_train[:, 8])
knn_result = knn.predict(knn_test_data)

conf_matrix = confusion_matrix(data_test[:, 8], knn_result)

print("Perpetrator sex confusion matrix:")
print(conf_matrix)

# cross validation knn
scores = []
data_without_target = np.delete(data_matrix, 8, 1)
target = data_matrix[:, 8]
k_fold = KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(target):
    scores.append(
        knn.fit(data_without_target[train_indices], target[train_indices]).score(data_without_target[test_indices],
                                                                                 target[test_indices]))

print("knn Perpetrator sex accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores), np.std(scores) * 2))

for n in range(5, 10, 3):
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = []
    data_without_target = np.delete(data_matrix, 9, 1)
    target = data_matrix[:, 9]
    k_fold = KFold(n_splits=10)
    for train_indices, test_indices in k_fold.split(target):
        scores.append(
            knn.fit(data_without_target[train_indices], target[train_indices]).score(data_without_target[test_indices],
                                                                                     target[test_indices]))

    print("knn, k = " + str(n) + " Perpetrator age accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores), np.std(scores) * 2))

###Decision trees


clf = tree.DecisionTreeClassifier()
scores = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
data_without_target = np.c_[
    agency_type, dataset.Year.values, month, crime_type, victim_Sex, dataset.Victim_Age.values,
    victim_race, weapon,
    dataset.Additional_Victims_Count.values]
target = data_matrix[:,4]
k_fold = KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(target):
	clf.fit(data_without_target[train_indices], target[train_indices])
	result = clf.predict(data_without_target[test_indices])
	conf_matrix = confusion_matrix(target[test_indices], result)
	print(conf_matrix)
	scores.append(clf.fit(data_without_target[train_indices], target[train_indices]).score(data_without_target[test_indices], target[test_indices]))
	probas_ = clf.fit(data_without_target[train_indices], target[train_indices]).predict_proba(data_without_target[test_indices])
    # Compute ROC curve and area the curve
	fpr, tpr, thresholds = metrics.roc_curve(target[test_indices], probas_[:, 1])
	tprs.append(interp(mean_fpr, fpr, tpr))
	tprs[-1][0] = 0.0
	roc_auc = metrics.auc(fpr, tpr)
	aucs.append(roc_auc)
	plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
	i += 1
print("Accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores), np.std(scores)))


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve for Decision tree')
plt.legend(loc="lower right")
plt.show()

clf.fit(data_without_target, target)

names = ["Agency_Type","Year","Month","Crime_Type","Victim_Sex","Victim_Age","Victim_Race","Weapon","Additional_Victims_Count"]
classes = ["crime NOT solved", "crime solved"]
dot_data = tree.export_graphviz(clf, feature_names=names, class_names=classes, out_file=None, max_depth=3, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("MurderMystery")

interactive_data = np.c_[cities, victim_Sex, dataset.Victim_Age.values, weapon]

for n in range(5, 30, 3):
    knn = KNeighborsClassifier(n_neighbors=n)

    scores = []
    data_without_target = np.delete(interactive_data, 3, 1)
    target = interactive_data[:, 3]
    k_fold = KFold(n_splits=5)
    for train_indices, test_indices in k_fold.split(target):
        scores.append(
            knn.fit(data_without_target[train_indices], target[train_indices]).score(data_without_target[test_indices],
                                                                                     target[test_indices]))

    print("knn, k = " + str(n) + " weapon accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores), np.std(scores) * 2))

root = Tk()
root.title("Choose the victim data?")

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
mainframe.pack(pady=100, padx=100)

# Create a Tkinter variable
tkcity = StringVar(root)

cities_set = sorted(cities_set)
# Dictionary with options
tkcity.set('Anchorage')  # set the default option

citypopupMenu = OptionMenu(mainframe, tkcity, *cities_set[:50])
Label(mainframe, text="Choose the city").grid(row=1, column=1)
citypopupMenu.grid(row=2, column=1)

tkgender = StringVar(root)
tkgender.set('Male')  # set the default option

genders = set(dataset.Victim_Sex.values)
genderpopupMenu = OptionMenu(mainframe, tkgender, *genders)
Label(mainframe, text="Choose your gender").grid(row=3, column=1)
genderpopupMenu.grid(row=4, column=1)

tkage = StringVar(root)
tkage.set('30')  # set the default option
ages = range(0, 100)
agepopupMenu = OptionMenu(mainframe, tkage, *ages)
Label(mainframe, text="Choose your age").grid(row=5, column=1)
agepopupMenu.grid(row=6, column=1)


# Ok button trigger
def ok_pressed():
    print(tkcity.get() + tkage.get() + tkgender.get())
    to_predict = [
        [city_enc.transform([tkcity.get()])[0], victim_sex_enc.transform([tkgender.get()])[0], int(tkage.get())]]
    res = knn.predict(to_predict)
    print("The victim will die with a: " + weapon_enc.inverse_transform(res[0]))


okButton = Button(mainframe, text="OK", command=ok_pressed)
okButton.grid(row=7, column=1)

root.mainloop()
