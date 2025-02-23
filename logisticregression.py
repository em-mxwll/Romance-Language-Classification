## this file builds a logistic regression model to work with romance_languages_data.csv 
    ## a very large portion of this code is based on the work of Dr. Jonathan Dunn that was shared with LING413 students for week 2's labs. 

import pandas as pd
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

file = "romance_languages_data.csv"

data_df = pd.read_csv(file, header = None) 
data_df = data_df.drop(0) #get rid of the first because it makes the headers be "0" and "1" instead of "text" and "language"
data_df.columns = ["Text", "Language"]

print(data_df)

#save last 5000 rows for the third step of the paper directions if needed 
data_df_step3 = data_df.tail(5000)
data_df = data_df.iloc[:-5000] #remove the last 5000 rows from the main dataframe 


#initialize feature extraction 
features = feature_extraction.text.CountVectorizer(input='content', 
                                                encoding='utf-8', 
                                                decode_error='ignore', 
                                                lowercase=True, 
                                                tokenizer = None,
                                                ngram_range=(2, 2), #using bigrams 
                                                analyzer='word', #using words (as opposed to characters)
                                                max_features=10000, #This sets vocab size
                                                )

#Sklearn first fits then transforms
features.fit(data_df.loc[:,"Text"].values)
data_x = features.transform(data_df.loc[:,"Text"].values)
data_y = data_df.loc[:,"Language"] 

#Divide the features into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.10, shuffle=True)

#The train / test arrays should have the same shape
print("Train shape", x_train.shape, y_train.shape) 
print("Test shape", x_test.shape, y_test.shape) 

#STEP 4: Train classifier (logistic regression)
cls = LogisticRegression()
cls.fit(x_train, y_train)
print(cls)

#STEP 5: Use classifier to predict labels on test set
y_pred = cls.predict(x_test)

#Compare predicted and actual labels
report = classification_report(y_test, y_pred, digits = 3)

print(report)

#output results as a confusion matrix 
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['fra', 'ita', 'por', 'ron', 'spa'], yticklabels = ['fra', 'ita', 'por', 'ron', 'spa'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')

#save it as a png so i can use it in the paper 
plt.savefig("lgr_confusionmatrix_words.png", dpi = 300, bbox_inches = 'tight')

plt.show()