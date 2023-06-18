import pandas as pd # pandas is a dataframe library
import matplotlib.pyplot as plt # matplotlib.pyplot plots data
import numpy as np # numpy provides N-dim object support
# do plotting inline instead of in several windows
df = pd.read_csv("data/pima-data.csv") # load pima-data
df.shape #Dimensions of the data in the file 'col,row'
df.head(5) #Displays first 5 columns
df.tail(5) #Displays last 5 columns
df.isnull().values.any() #Checkking to see if there are any empty spaces in the table
#Function to plot all rows against each other
def plot_corr(df,size=12):
 corr = df.corr() # data frame correlation function
 fig, ax = plt.subplots(figsize = (size,size))
 ax.matshow(corr) # color code the rectangles by correlation value
 plt.xticks(range(len(corr.columns)),corr.columns) # draw x tick marks
 plt.yticks(range(len(corr.columns)),corr.columns) # draw y tick marks
plot_corr(df)
diabetes_map = {True : 1, False : 0} #Storing the column to be changed in a variable
df['diabetes'] = df['diabetes'].map(diabetes_map)
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
total = float(num_true + num_false)
print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true/total)*100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, 
(num_false/total)*100))
from sklearn.model_selection import train_test_split
feature_col_names = 
['num_preg','glucose_conc','diastolic_bp','insulin','bmi','diab_pred','age']
predicted_class_names = ['diabetes']
X = df[feature_col_names].values # predictor feature columns (8 x m ) 
y = df[predicted_class_names].values # predicted class (1 = true, 0 = false) column (1 x m)
split_test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, 
random_state = 42)
print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))
df.head()
print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))
from sklearn.preprocessing import Imputer
# Impute with mean all 0 readings
fill_0 = Imputer(missing_values= 0, strategy = "mean", axis=0)
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)
from sklearn.naive_bayes import GaussianNB
# create Gaussian Bayes model object and train it with the 
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())
# predict values using the training data
nb_predict_train = nb_model.predict(X_train)
# import the performance metrics library
from sklearn import metrics 
# Accuracy
print("Accuracy: {0:.2%}".format(metrics.accuracy_score(y_train,nb_predict_train)))
print()
# predict values using the training data
nb_predict_test = nb_model.predict(X_test)
# import the performance metrics library
from sklearn import metrics 
# Accuracy
print("Accuracy: {0:.2%}".format(metrics.accuracy_score(y_test,nb_predict_test)))
print()
print("Confusion Matrix")
# Note the use of labels for 1 = true and 0 = false to lower right
print("{0}".format(metrics.confusion_matrix(y_test,nb_predict_test, labels=[1,0])))
# Classification report
print("")
print("Classification Report")
print("")
print(metrics.classification_report(y_test,nb_predict_test, labels=[1,0]))
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train.ravel())
rf_predict_train = rf_model.predict(X_train)
# training metrics
print("Accuracy: {0:.2%}".format(metrics.accuracy_score(y_train,rf_predict_train)))
rf_predict_test = rf_model.predict(X_test)
# test metrics
print("Accuracy: {0:.2%}".format(metrics.accuracy_score(y_test,rf_predict_test)))
print("Confusion Matrix")
# Note the use of labels for 1 = true and 0 = false to lower right
print("{0}".format(metrics.confusion_matrix(y_test,rf_predict_test, labels=[1,0])))
# Classification report
print("")
print("Classification Report")
print("")
print(metrics.classification_report(y_test,rf_predict_test, labels=[1,0]))
