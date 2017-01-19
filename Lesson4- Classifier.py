from scipy.spatial import distance
from sklearn.metric import accuracy_score
from sklearn import datasets
from sklearn.cross_validation import train_test_split

 
def euc_distance(pt1, pt2):
	return distance.euclidean(pt1,pt2)

class BareClassifier():

	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train
	
	def predict(self, X_test):
		predictions = []
		
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions
		
	def closest(self, row):
		closest_dist = euc_distance(row,self.X_train[0])
		closest_idx = 0
		
		for x in range(1,len(self.X_train)):
			dist = euc_distance(row, self.X_train[x])
			if dist < closest_dist:
				closest_dist = dist
				closest_idx = x
				
		return self.Y_train[closest_idx]
		

iris = datasets.load_iris()

X = iris.data
Y = iris.target

X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.5)

my_classifier = BareClassifier()
my_classifier.fit(X_train,Y_train)
predictions = my_classifier.predict(X_test)

print("Classifier Accuracy:" + accuracy_score(Y_test, first_predictions)
