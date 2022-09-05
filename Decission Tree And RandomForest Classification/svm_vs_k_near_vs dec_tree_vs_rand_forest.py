from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    array(data.data), array(data.target), test_size=0.2)

# accuracy of decison Tree
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
acc_dec_tree = classifier.score(x_test, y_test)

# accuracy of decison Tree
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
acc_ran_for = classifier.score(x_test, y_test)

# accuracy of svm
classifier = SVC(kernel="linear", C=3)
classifier.fit(x_train, y_train)
acc_svm = classifier.score(x_test, y_test)

# accuract of k_nears
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)
acc_k = classifier.score(x_test, y_test)


print("Svm Accuracy : ", acc_svm)
print("K Nearst Accuracy : ", acc_k)
print("Decision Tree Forest", acc_dec_tree)
print("Random Forest Accuracy : ", acc_ran_for)

# the best ones are svm and random forest