# A Supervised Approach to Romance Language Identification
This project makes use of the [OpenLID Projct's language corpus](https://github.com/laurieburchell/open-lid-dataset). 

After processing the data to include only Romance languages (French, Spanish, Romanian, Italian, and Portuguese), it implements two simple classifiers to perform a language identification task. 

**logisticregression.py** implements a word-level bigram-based logistic regression model. **naivebayes.py** implements a word-level unigram-based multinomial naive bayes model. 

Both models output their results as confusion matrices, which are included in this repository. 

## Sources 
Heavily based on lecture and lab content by [Dr. Jonathan Dunn](https://www.jdunn.name/). 
Additional sources include: 
- https://medium.com/towards-data-science/an-efficient-language-detection-model-using-naive-bayes-85d02b51cfbd 
- https://www.geeksforgeeks.org/multinomial-naive-bayes/