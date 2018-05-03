import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import matplotlib.pyplot as plt

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
	# def grid_search
	def grid_search(self, classifier, parameter, x_train, y_train, x_test, y_test):
		clf = GridSearchCV(classifier, parameter)
		clf = clf.fit(x_train, y_train)
		print('best_score=', clf.best_score_)
		predict = clf.predict(x_test)
		print(np.mean(predict == y_test))
		return 
	# def plot_sub_sentiment(self, Airline, file):
	# 	Tweet = pd.read_csv(file)
 #    	df=Tweet[Tweet['airline']==Airline ]
 #    	count=df['airline_sentiment'].value_counts()
 #    	Index = [0,1,2]
 #    	plt.bar(Index,count)
 #    	plt.xticks(Index,['negative','neutral','positive'])
 #    	plt.ylabel('Sentiment Count')
 #    	plt.xlabel('Sentiment')
 #    	plt.title('Sentiment overview of '+Airline)
 #    	plot.show()
 #    	return

if __name__ == '__main__':
	# analysis 
	Tweet = pd.read_csv('clean_tweet.csv')
	sentiment_count=Tweet['airline_sentiment'].value_counts()
	Index = [0,1,2]
	plt.bar(Index,sentiment_count)
	plt.xticks(Index,['negative','neutral','positive'],rotation=0)
	plt.ylabel('sentiment count')
	plt.xlabel('Sentiment')
	plt.title('Overall Sentiment')
	plt.show()
	
	# data = pd.read_csv('clean_tweet.csv')
	# airlines = data['airline'].value_counts()
	# print(airlines)
	x_vec, y = tweets_ana().count_vec('clean_tweet.csv')
	x_tfidf = tweets_ana().tfidf(x_vec)
	x_train, x_dev, x_test, y_train, y_dev, y_test = tweets_ana().data_split(x_tfidf,y)

	# Baseline 
	# Naive Bayes: MultinomiaNB for classification with discrete features
	clf = MultinomialNB().fit(x_train, y_train)
	predict_nb = clf.predict(x_dev)
	acc = np.mean(predict_nb == y_dev)
	print("naive bayes = ", acc)
	# SVM with SGD training
	clf = SGDClassifier(loss = 'hinge', penalty = 'l2', max_iter=5, tol=None).fit(x_train, y_train)
	predict_svm = clf.predict(x_dev)
	acc = np.mean(predict_svm == y_dev)
	print('svm = ', acc)
	# knn
	clf = KNeighborsClassifier(n_neighbors = 5).fit(x_train, y_train)
	predict_knn = clf.predict(x_dev)
	acc = np.mean(predict_knn == y_dev)
	print('knn = ', acc)
	# Logestic Regression
	clf = LogisticRegression(multi_class= 'ovr').fit(x_train, y_train)
	predict_lr = clf.predict(x_dev)
	acc = np.mean(predict_lr == y_dev)
	print('lr = ', acc)
	# Pipeline
	# clf_pipe = Pipeline([('vec', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	# clf_pipe.fit(x_train, y_train)
	# predict = clf_pipe.predict(x_dev)
	# acc = np.mean(predict == y_dev)
	# print('Pipeline', acc)
	print('-----------------Dev set----------------')
	print(metrics.classification_report(y_dev, predict_nb))
	print(metrics.classification_report(y_dev, predict_svm))
	print(metrics.classification_report(y_dev, predict_knn))
	print(metrics.classification_report(y_dev, predict_lr))

	print('-----------------Test set----------------')
	# Gridsearch
	# parameters_knn = {'n_neighbors':[10,15], 'weights':('uniform', 'distance'), 'p':[1,2]}
	
	parameters_knn = {'n_neighbors':[12, 14]}
	knn= KNeighborsClassifier()
	clf = GridSearchCV(knn, parameters_knn)
	clf = clf.fit(x_train, y_train)
	print('best_score=', clf.best_score_)
	predict = clf.predict(x_test)
	print(np.mean(predict == y_test))
	print(metrics.classification_report(y_test,predict, digits=4))

	parameters_lr = {'penalty':('l1', 'l2'), 'C':[1, 10, 100]}
	lr = LogisticRegression(multi_class= 'ovr')
	clf = GridSearchCV(lr, parameters_lr)
	clf = clf.fit(x_train, y_train)
	predict = clf.predict(x_test)
	# print(clf.best_score_, clf.best_estimator_.C, clf.best_estimator_.penalty)
	print(np.mean(predict == y_test))
	print(metrics.classification_report(y_test,predict, digits=4))

	parameters_nb = {'alpha':[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]}
	nb= MultinomialNB()
	clf = GridSearchCV(nb, parameters_nb)
	clf = clf.fit(x_train, y_train)
	print('best_alpha=', clf.best_estimator_.alpha)
	predict = clf.predict(x_test)
	print(np.mean(predict == y_test))
	print(metrics.classification_report(y_test,predict, digits=4))

	clf = SGDClassifier(loss = 'hinge', penalty = 'l2', max_iter=5, tol=None).fit(x_train, y_train)
	predict_svm = clf.predict(x_test)
	acc = np.mean(predict_svm == y_test)
	print('svm = ', acc)
	print(metrics.classification_report(y_test,predict_svm, digits=4))













		









