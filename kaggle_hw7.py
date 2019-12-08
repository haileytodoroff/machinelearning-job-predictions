# KAGGLE COMPETITION
# AI HW 7
# Hailey Todoroff (ht2450)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class Kaggle:
	def __init__(self):
		# mnist train data
		self.mnist_train_data = np.load('mnist-train-images.npy')
		self.mnist_train_labels = np.load('mnist-train-labels.npy')
		
		# mnist test data
		self.mnist_val_images = np.load('mnist-val-images.npy')
		self.mnist_val_labels = np.load('mnist-val-labels.npy')
		
		# ai class train data
		self.scan_train_images = np.load('scan-train-images.npy')
		self.scan_train_labels = np.load('scan-train-labels.npy')

		# ai class test data
		self.scan_test_data = np.load('scan-test-images.npy')


	# train/test/split, normalize, standardize, feature importance, etc
	def preprocess(self):
		# train test split on mnist and scan data
		mX_train, mX_test, my_train, my_test = train_test_split(self.mnist_train_data, self.mnist_train_labels)
		sX_train, sX_test, sy_train, sy_test = train_test_split(self.scan_train_images, self.scan_train_labels)

		print(len(mX_train))
		self.X_train = mX_train.join(sX_train)
		self.Y_train = pd.DataFrame(data=[my_train, sy_train])
		print(self.X_train.head())

		# reshape the train data so that is can be used in sklearn (2d data)
		# self.resized_scan_train_images = np.reshape(self.scan_train_images, (3780, 28*28))

		# # reshape the test data
		# dims = self.mnist_train_data.shape
		# self.resized_mnist_train_data = np.reshape(self.mnist_train_data, (dims[0], dims[1]*dims[2]))

	def randForrest(self):
		rf = RandomForestClassifier()
		rf.fit(self.resized_scan_train_images, self.scan_train_labels)
		rf.fit(self.resized_mnist_train_data, self.mnist_train_labels)

		train_score = rf.score(self.resized_scan_train_images, self.scan_train_labels)
		test_score = rf.score(self.resized_mnist_train_data, self.mnist_train_labels)
		
		return train_score, test_score



if __name__ == '__main__':
	kaggle = Kaggle()
	kaggle.preprocess()
	#print(kaggle.randForrest())



