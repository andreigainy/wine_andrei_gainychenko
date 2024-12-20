
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd


wine = datasets.load_wine()


data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['target'] = wine.target

X = data.drop('target', axis=1)
y = data['target']


scaler = StandardScaler()
X = scaler.fit_transform(X)


x_train, y_train = X[:140], y[:140]
x_test, y_test = X[140:], y[140:]

mlp = MLPClassifier(hidden_layer_sizes=(23,),
                    activation='logistic',
                    alpha=1e-4,
                    solver='sgd',
                    tol=1e-4,
                    random_state=0,
                    learning_rate_init=0.1,
                    verbose=True
                    )


mlp.fit(x_train, y_train)


fig, axes = plt.subplots(1, 1, figsize=(8, 6))
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("Number of Iterations")
axes.set_ylabel("Loss")
plt.title("Loss Curve of MLP Classifier")
plt.show()


predictions = mlp.predict(x_test)


accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:}")
