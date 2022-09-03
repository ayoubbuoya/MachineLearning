from numpy import array
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

'''
    Breast Cancer Dataset : 
        Have You A Benguin Cancer Or Malignant(Bad) Cancer ? 
'''
data = load_breast_cancer()

# print("Features Names: ")
# print(data.feature_names)
# print("Targets Names : ")
# print(data.target_names)

# print("X Data : ")
# print(data.data)
# print("Y Data : ")
# print(data.target)

x_train, x_test, y_train, y_test = train_test_split(
    array(data.data), array(data.target), test_size=0.2)

# we have 2 classes so k = 3 is ok
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(x_train, y_train)

# accuarcy
accuracy = classifier.score(x_test, y_test)
print(accuracy)
