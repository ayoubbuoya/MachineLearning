from numpy import array, linspace
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# reshape data because we want them vertically
time_studied = array([20, 50, 32, 65, 23, 43, 10, 5,
                     22, 35, 29, 5, 56]).reshape(-1, 1)
scores = array([56, 83, 47, 93, 47, 82, 45, 78,
               55, 67, 57, 4, 12]).reshape(-1, 1)


model = LinearRegression()
model.fit(time_studied, scores)

data = linspace(0, 70, 100).reshape(-1, 1)
# data = array([80]).reshape(-1, 1)
predicted_score = model.predict(data)
# print(predicted_score)

# show graph without AI
plt.scatter(time_studied, scores)
plt.plot(data, predicted_score, "r")
plt.ylim(0, 100)
plt.show()



def test_model() : 
    pass
