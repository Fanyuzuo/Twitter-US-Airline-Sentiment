import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import SGDClassifier
class tweets_ana:
	# from text to feature vectors 
	def count_vec(self, file):
		data = pd.read_csv(file)
		x = data['text']
		x = [word.strip() for word in x]
		y = data['airline_sentiment']
		count_vect = CountVectorizer()
		x_vec = count_vect.fit_transform(x)
		return x_vec, y
	# tfidf - term frequency devide each count by total counts
	# idf - downscale weights for words that occur in many documents
	def tfidf(self, x_vec):
		tfidf_transformer = TfidfTransformer()
		x_tfidf = tfidf_transformer.fit_transform(x_vec)
		return x_tfidf
	# used cleaned tweets and split into train, test, dev sets
	def data_split(self, x_vec, y):
		x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.3, random_state = 50)
		x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state = 50)
		return x_train, x_dev, x_test, y_train, y_dev, y_test
	

if __name__ == '__main__':
	# data = pd.read_csv('clean_tweet.csv')
	# airlines = data['airline'].value_counts()
	# print(airlines)
	x_vec, y = tweets_ana().count_vec('clean_tweet.csv')
	x_tfidf = tweets_ana().tfidf(x_vec)
	x_train, x_dev, x_test, y_train, y_dev, y_test = tweets_ana().data_split(x_tfidf,y)

	clf = MultinomialNB().fit(x_train, y_train)
	predict = clf.predict(x_dev)
	acc = np.mean(predict == y_dev)
	print("naive bayes = ", acc)

	clf = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha=1e-3, max_iter=5, tol=1e-3).fit(x_train, y_train)
	predict = clf.predict(x_dev)
	acc = np.mean(predict == y_dev)
	print('svm = ', acc)


	# print(x_train.shape, x_dev.shape, x_test.shape)
	# print(x_train[1, :])
