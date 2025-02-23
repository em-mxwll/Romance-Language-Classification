## this file implements a language classifier using multinomial naive bayes ## 
    ## a portion of this code is based on the work of Dr. Jonathan Dunn that was shared with LING413 students for week 2's labs. additional sources are listed below

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns

file = "romance_languages_data.csv"

data_df = pd.read_csv(file, header = None) 
data_df = data_df.drop(0) #get rid of the first row because it makes the headers be "0" and "1" instead of "text" and "language"
data_df.columns = ["Text", "Language"]

print(data_df)

#save last 5000 rows for the third step of the paper directions if needed 
data_df_step3 = data_df.tail(5000)
data_df = data_df.iloc[:-5000] #remove the last 5000 rows from the main dataframe 

#get some info about the corpus 
langs = ['fra', 'spa', 'ron', 'ita', 'por']
for lang in langs:
    count = len(data_df[data_df['Language'] == lang])
    print("there are ", count, "rows in ", lang)

#map labels to numbers for the classifier 
    #['fra' = 0, 'ita' = 1, 'por' = 2, 'ron' = 3, 'spa' = 4]
data_df['Language'] = data_df['Language'].map({'fra':0, 'ita':1, 'por':2, 'ron':3, 'spa':4})

print(data_df)

x = data_df['Text'] #features 
y = data_df['Language'] #target 

#90:10 train:test split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

#The train / test arrays should have the same shape
print("Train shape", x_train.shape, y_train.shape) 
print("Test shape", x_test.shape, y_test.shape)

vectorizer = CountVectorizer()
x_train_vectors = vectorizer.fit_transform(x_train)
x_test_vectors = vectorizer.transform(x_test)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('model', MultinomialNB())
])

#trying cross validation because results are suspiciouslly high (like a LOT of 1.00)
cv_scores = cross_val_score(pipeline, x, y, cv = 5, scoring = 'accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation accuracy:", np.mean(cv_scores))


#train the model
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

report = classification_report(y_test, y_pred, digits = 3)

print(report)

#output results as a confusion matrix 
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['fra', 'ita', 'por', 'ron', 'spa'], yticklabels = ['fra', 'ita', 'por', 'ron', 'spa'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naïve Bayes Confusion Matrix')

#save it as a png so i can use it in the paper 
plt.savefig("mnb_confusionmatrix.png", dpi = 300, bbox_inches = 'tight')

plt.show()

## step 3 -- test the better performing model on new data ## 
langs = ['fra', 'spa', 'ron', 'ita', 'por']
for lang in langs:
    count = len(data_df_step3[data_df_step3['Language'] == lang])
    print("there are ", count, "rows in ", lang)

#map labels to numbers for the classifier 
    #['fra' = 0, 'ita' = 1, 'por' = 2, 'ron' = 3, 'spa' = 4]
data_df_step3['Language'] = data_df_step3['Language'].map({'fra':0, 'ita':1, 'por':2, 'ron':3, 'spa':4})

#train the model 

#no need for a train and test split -- this is all test 
x_step3 = data_df_step3['Text']
y_step3 = data_df_step3['Language']

x_step3_vectors = vectorizer.transform(x_step3)

y_pred_step3 = pipeline.predict(x_step3)

report_step3 = classification_report(y_test, y_pred, digits = 3)

print("this is the report for step 3:")
print(report_step3)


#output results as a confusion matrix 
cm_step3 = confusion_matrix(y_step3, y_pred_step3)

plt.figure(figsize=(8,6))
sns.heatmap(cm_step3, annot = True, fmt = 'd', xticklabels = ['fra', 'ita', 'por', 'ron', 'spa'], yticklabels = ['fra', 'ita', 'por', 'ron', 'spa'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naïve Bayes Confusion Matrix for Held Out Data')

#save this new confusion matrix as a png so i can use it in the paper 
plt.savefig("mnb_confusionmatrix_step3.png", dpi = 300, bbox_inches = 'tight')

plt.show()

## sources: ## 
    #https://medium.com/towards-data-science/an-efficient-language-detection-model-using-naive-bayes-85d02b51cfbd 
    #https://www.geeksforgeeks.org/multinomial-naive-bayes/