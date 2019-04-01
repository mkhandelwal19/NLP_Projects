import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# quoting = 3 will ignore double quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset
#dataset.info()


# ### Cleaning the texts
# clearnign a sample row

import re

dataset['Review'][0]


# remove all characters except a to z or A to Z and add a space
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
review = review.lower()
review = review.split()

import nltk
nltk.download('stopwords')

#Text may contain stop words like ‘the’, ‘is’, ‘are’. 
#Stop words can be filtered from the text to be processed. 
#There is no universal list of stop words in nlp research, 
#however the nltk module contains a list of stop words.

from nltk.corpus import stopwords

#The idea of stemming is a sort of normalizing method. 
from nltk.stem.porter import PorterStemmer

# removes preposition "a" "the" etc
ps = PorterStemmer()

review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

# ### Clean the entire Dataset

# Create a corpus of clean words
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)



# ### Bag of words model

# Sparse matrix - a Matrix with lots of zeros - This is called sparsity
# We try to reduce sparsity using machhine learning model as muchh as possible
# Creating a matrix through the process of tokenization


from sklearn.feature_extraction.text import CountVectorizer
# You can do text cleaning with CountVectorizer 
# max_features = 1500 will filter the non relevant words. 1500 most occuring words will be kept
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
    

# create the dependent variable
y = dataset.iloc[:, 1].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Fitting Naive Bayes to the Training set


#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#classifier = GaussianNB()
classifier = MultinomialNB()
classifier.fit(X_train, y_train)



# Predict the observation for the test set
# y_pred is a vector of prediction 
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]
# y_test consists the real values and y_pred consists of the predicted values


from sklearn.metrics import classification_report, accuracy_score,confusion_matrix


# Create the cofusion matrix
cm = confusion_matrix(y_test, y_pred)

#print accuracy
print("accuracy ", accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))


examples = ['nice service']
example_counts = cv.transform(examples).toarray()
predictions = classifier.predict(example_counts)
predictions
if (predictions[0]==1):
    print("Good Review")
else:
    print("Bad Review")

#new_prediction = (new_prediction > 0.5)
#new_prediction

