from numpy import array
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    array(data.data), array(data.target), test_size=0.2)


# choose linear because is the fatest; c is the soft margin
classifier = SVC(kernel="linear", C=3)

classifier.fit(x_train, y_train)

# accuracy of svm
acc_svm = classifier.score(x_test, y_test)

# accuract of k_nears
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)
acc_k = classifier.score(x_test, y_test)


print("Svm Accuracy : ", acc_svm)
print("K Nearst Accuracy : ", acc_k)
