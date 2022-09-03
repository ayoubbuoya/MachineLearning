from numpy import array
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = array([20, 50, 32, 65, 23, 43, 10, 5,
                     22, 35, 29, 5, 56]).reshape(-1, 1)
scores = array([56, 83, 47, 93, 47, 82, 45, 23,
               55, 67, 57, 4, 89]).reshape(-1, 1)


time_train, time_test, score_train, score_test = train_test_split(
    time_studied, scores, test_size=0.2)  # 0.2 means 20% of data for testing


model = LinearRegression()
model.fit(time_train, score_train)

# test model
accuracy = model.score(time_test, score_test)
print(accuracy)

in_data = array([20]).reshape(-1, 1)

if accuracy > 0.5 and accuracy <= 1:
    print(model.predict(in_data))
