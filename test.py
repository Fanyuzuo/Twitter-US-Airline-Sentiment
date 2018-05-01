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
	x_train, x_dev, x_test, y_train, y_dev, y_test = tweets_ana().data_split(x_vec,y)
	# print(x_train.head(),'\n', y_train.head())
	print(x_train.shape, x_dev.shape, x_test.shape)