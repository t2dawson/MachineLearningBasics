from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metric import accuracy_score

iris = datasets.load_iris()

X = iris.data
Y = iris.target

X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.5)

first_classifier = tree.DecisionTreeClassifier()
first_classifier.fit(X_train,Y_train)
first_predictions = first_classifier.predict(X_test)

second_classifier =  KNeighborsClassifier()
second_classifier.fit(X_train, Y_train)
second_predictions = second_classifier.predict(X_test)
 
print("First Classifier Accuracy:" + accuracy_score(Y_test, first_predictions)
print("Second Classifier Accuracy: " + accuracy_score(Y_test, second_predictions)