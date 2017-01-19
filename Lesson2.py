from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

#print out entire training data set
for data in iris.data:
	print(data)

#print out all targets
for target in iris.target:
	print target

idx = [0,50,100]
#remove 1 target and associated data and use the remaining as training data
training_target = np.delete(iris.target,idx)
training_data = np.delete(iris.data, idx, axis = 0)

#use removed idx as training data
test_target = iris.target[idx]
test_data = iris.data[idx]

clf = tree.DecisionTreeClassifier()
clf.fit(training_data,training_target)

#print expected target name
print test_target

#print predicted target name
print(clf.predict(test_data)
