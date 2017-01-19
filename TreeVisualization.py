from sklearn.externals.six import StringIO
import pydot
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data,iris.target)

data_dot = StringIO()
tree.export_graphviz(clf, out_file = data_dot, feature_names = iris.feature_names,
						class_names = iris.target_names, filled = True, rounded= True,
						impurity = False)
						
graph = pydot.graph_from_dot_data(data_dot.getvalue())
graph.write_pdf("iris.pdf")