import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

Classifier = RandomForestClassifier(n_estimators=300,random_state=0)

Classifier.fit(x_train_scaled,y_train)

y_pred = Classifier.predict(x_test_scaled)

print(np.column_stack((y_test,y_pred)))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


def plot_set(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2, Classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i in np.unique(y_set):
        plt.scatter(
            X_set[y_set == i, 0], X_set[y_set == i, 1],
            c=ListedColormap(('red', 'green'))(i), label=i
        )
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

plot_set(x_train_scaled, y_train, 'Random Forest (Training set)')
plot_set(x_test_scaled, y_test, 'Random Forest (Test set)')
