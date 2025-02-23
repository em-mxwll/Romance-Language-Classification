## this file takes the original lid201-data.gz corpus from Week 2's lab and shrinks it down to only consider romance languages ##
    ## this choice was made out of 1) personal interest 2) a need to reduce computation and space requirements 
    ## a very large portion of this code is based on the work of Dr. Jonathan Dunn that was shared with LING413 students for week 2's labs. 

import pandas as pd

#STEP 1: Load the corpus
file = "lid201-data.gz" #this file has since been deleted from my local storage (she's too big...)

#load into a dataframe -- using 9000000 rows for now (please pray my computer doesn't explode...)
data_df = pd.read_csv(file, nrows = 9000000, sep = "\t", header = None)

#Add column names and we don't need source information
data_df.columns = ["Text", "Language", "Source"]
data_df.drop("Source", axis = 1, inplace = True)

#We'll just use the ISO3 language codes
data_df.loc[:,"Language"] = [x[:3] for x in data_df.loc[:,"Language"].values]
print(data_df)

#only keep romance languages 
    #french = fra(e) 
    #spanish = spa
    #romanian = ron
    #italian = ita 
    # portuguese = por
romance_languages = ['fra', 'spa', 'ron', 'ita', 'por']
data_df = data_df[data_df['Language'].isin(romance_languages)]

print(data_df)

data_df.to_csv("romance_languages_data.csv", index = False)