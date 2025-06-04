import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix



dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)


log_reg = LogisticRegression()
log_reg.fit(x_train_scaled, y_train)


y_pred = log_reg.predict(x_test_scaled)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")



def plot_decision_boundary(X, y, model, title, scaler):
    from matplotlib.colors import ListedColormap


    step_size = 0.1

    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=step_size),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=step_size)
    )

    grid = np.array([X1.ravel(), X2.ravel()]).T
    grid_scaled = scaler.transform(grid)
    predictions = model.predict(grid_scaled).reshape(X1.shape)

    plt.contourf(X1, X2, predictions, alpha=0.3, cmap=ListedColormap(('red', 'green')))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(('red', 'green')), edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()



x_train_original = sc.inverse_transform(x_train_scaled)
x_test_original = sc.inverse_transform(x_test_scaled)


plot_decision_boundary(x_train_original, y_train, log_reg, "Logistic Regression (Training Set)", sc)
plot_decision_boundary(x_test_original, y_test, log_reg, "Logistic Regression (Test Set)", sc)
