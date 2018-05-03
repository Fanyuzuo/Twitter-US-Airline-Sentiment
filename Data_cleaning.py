# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 09:45:32 2018

@author: asra
"""
import nltk
from nltk import word_tokenize
import pandas as pd

 #reading tweet.csv file using pandas 
df= pd.read_csv("Tweets.csv",header=0, usecols=['text', 'airline_sentiment','airline','negativereason'])

#cleaning tweets
df['text']=df['text'].str.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])","")

#transforming postive to 2, netural to 1, negative to 0
sentiment = sorted(df['airline_sentiment'].unique())
sentiment_mapping = dict(zip(sentiment, range(0, len(sentiment) + 1)))
df['airline_sentiment']  = df['airline_sentiment'].map(sentiment_mapping).astype(int)

df.to_csv('clean_tweet.csv',encoding='utf-8')
csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)

#tokenizing text
df['text']=df['text'].apply(nltk.word_tokenize)

# removing stopword 
stopwords = nltk.corpus.stopwords.words('english')
df['text']=df['text'].apply(lambda x: [y for y in x if y not in stopwords])

# Detokenize cleaned text
df['text'] = df['text'].str.join(" ")
