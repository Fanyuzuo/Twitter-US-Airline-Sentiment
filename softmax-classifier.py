"""
File: classification.py
Language: Python 3
Description: Using training set model is created using SVM, Naive Bayes, 
Neural Networks and the model is tested on testing set and validation set.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CSV_DATA = "./data/cleaned_data"
tweet=pd.read_csv(CSV_DATA)

def fun(Airline):
    data_frame=tweet[tweet['airline']==Airline]#so u created a dataframe with the airline that will be passed on to the function
    count=data_frame['airline_sentiment'].value_counts()#we now count the number of positive and negative tweets by this airline by looking at airline_sentiment column
    val = []
    for i in range(len(count)):
        j = float(count[i]) / 100
        val.append(j)
    calc_softmax = softmax(val)
    
    print (Airline + "\n-----\n")
    print (calc_softmax)
    print ("\n")
    x=[10,20,30]#at x=10 ,20 and 30 we get negative neutral and positive tweets plotted
    plt.bar(x,count)
    plt.xticks(x,['negative','neutral','positive'])
    plt.ylabel('Mood of tweets count')
    plt.xlabel('Sentiment')
    plt.title('Mood count for airline  ' + Airline)
    plt.show()

def analysis_customers():
    Mood_count=tweet['airline_sentiment'].value_counts()#Returns object containing counts of unique values.
    #we now do the visualization 
    x=[10,20,30]#at these x cordinates
    plt.bar(x,Mood_count)#at x=10 we get 
    plt.xticks(x,['negative','neutral','positive'])
    plt.xlabel('Sentiment')
    plt.ylabel('Mood of tweets count')#y-coordinate is labelled this 
    plt.title('Count of Moods')#this will give title to graph 
    plt.show()
    
def classifier():
    print ("=========== Softmax ===========\n")
    analysis_customers()
    fun('US Airways')
    fun('United')   
    fun('American')
    fun('Southwest')
    fun('Delta')
    fun('Virgin America')

def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z))            
            
if __name__ == '__main__':
    classifier()
