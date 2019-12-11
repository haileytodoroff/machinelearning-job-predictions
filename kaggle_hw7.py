# KAGGLE COMPETITION
# AI HW 7
# Hailey Todoroff (ht2450)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics


class Kaggle:
	def __init__(self):
		# mnist train data
		self.mnist_train_data = np.load('mnist-train-images.npy')
		self.mnist_train_labels = np.load('mnist-train-labels.npy')
		# print('mnist train', self.mnist_train_labels[0:5])
		
		# mnist test data
		self.mnist_val_images = np.load('mnist-val-images.npy')
		self.mnist_val_labels = np.load('mnist-val-labels.npy')
		# print('mnist test', self.mnist_val_labels[0:5])
		
		# ai class train data
		self.scan_train_images = np.load('scan-train-images.npy')
		self.scan_train_labels = np.load('scan-train-labels.npy')
		# print('scan train', self.scan_train_labels[0:5])

		# ai class test data
		self.scan_test_data = np.load('scan-test-images.npy')
		# print('scan test', self.scan_test_data[1])

		# # flatten images
		# n_samples = len(self.mnist_train_data)
		# self.data = self.mnist_train_data.reshape((n_samples,-1))

	# def svm(self):
	# 	classifier = svm.SVC(gamma=0.001)

	# 	#split data
	# 	X_train, X_test, y_train, y_test = train_test_split(self.data, self.mnist_train_labels, test_size=0.4, shuffle=False)
	# 	classifier.fit(X_train, y_train)
	# 	print('train score:', classifier.score(X_train, y_train))
	# 	print('test score', classifier.score(X_test, y_test))
	# 	preds = classifier.predict(X_test)		

	# 	print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, preds)))


	# train/test/split, normalize, standardize, feature importance, etc
	def preprocess(self):
		# join data
		data = np.concatenate((self.mnist_train_data, self.scan_train_images))
		data_new = data.reshape((len(data), -1))
		pred_data = self.scan_test_data.reshape((len(self.scan_test_data), -1))
		# print(data_new.shape)
		labels = np.concatenate((self.mnist_train_labels, self.scan_train_labels))
		# print(labels.shape)

		# train test split
		X_train, X_test, y_train, y_test = train_test_split(data_new, labels, test_size=0.4)

		# # train test split on mnist and scan data
		# mX_train, mX_test, my_train, my_test = train_test_split(self.mnist_train_data, self.mnist_train_labels)
		# sX_train, sX_test, sy_train, sy_test = train_test_split(self.scan_train_images, self.scan_train_labels)

		# # config training
		# self.X_train = np.concatenate((mX_train, sX_train), axis=0)
		# self.Y_train = np.concatenate((my_train, sy_train), axis=0)

		# # config testing
		# self.X_test = np.concatenate((mX_test, sX_test), axis=0)
		# self.Y_test = np.concatenate((my_test, sy_test), axis=0)

		# # reshape the train data so that is can be used in sklearn (2d data)
		# size = self.X_train.shape[0]
		# self.resized_X_train = np.reshape(self.X_train, (size, 28*28))

		# # reshape the test data
		# dims = self.X_test.shape
		# self.resized_X_test = np.reshape(self.X_test, (dims[0], dims[1]*dims[2]))

		# # reshape prediction data
		# dims2 = self.scan_test_data.shape
		# self.resized_X_preds = np.reshape(self.scan_test_data, (dims2[0], dims2[1]*dims2[2]))
		return X_train, X_test, y_train, y_test, pred_data


	def randForrest(self, X_train, X_test, y_train, y_test, pred_data):
		rf = RandomForestClassifier(n_estimators=100)
		
		rf.fit(X_train, y_train)
		print('train score', rf.score(X_train, y_train))

		print('test score', rf.score(X_test, y_test))

		preds = rf.predict(pred_data)
		print('predictions', preds[0:5])

		# train_score = rf.score(self.resized_X_train, self.Y_train)
		# test_score = rf.score(self.resized_X_test, self.Y_test)

		# predict
		# preds = rf.predict(self.resized_X_preds)
		
		return preds



if __name__ == '__main__':
	kaggle = Kaggle()

	X_train, X_test, y_train, y_test, pred_data = kaggle.preprocess()

	preds = kaggle.randForrest(X_train, X_test, y_train, y_test, pred_data)

	# train_score, test_score, preds = kaggle.randForrest()
	# print('train score', train_score)
	# print('test score', test_score)

	# print('preds array', preds[0:5])
	predictions = pd.DataFrame(preds)
	# # str_predictions = str(predictions)
	# # for i in str_predictions:
	# # 	i = i[0]
	# # print('new predictions df', str_predictions)
	# # print('preds df', predictions.head())
	# # print('preds df shape', predictions.shape)
	# # print('preds array len', len(preds))

	# # save preds to csv
	# with open('submission.csv', 'wb') as f:
	# 	np.savetxt(f, preds, fmt='%.18e')
	predictions.to_csv('submission.csv', index=True, header=['Category'])


