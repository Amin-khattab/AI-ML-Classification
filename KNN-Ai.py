import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

Knn = KNeighborsClassifier(n_neighbors=5,metric="minkowski")
Knn.fit(x_train_scaled,y_train)

y_pred = Knn.predict(x_test_scaled)

print(np.column_stack((y_pred,y_test)))
print("the accuarcy is",accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))


def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = X_set[:, 0], X_set[:, 1]
    x1_min, x1_max = X1.min() - 1, X1.max() + 1
    x2_min, x2_max = X2.min() - 1, X2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    
    Z = Knn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for i, color in zip(np.unique(y_set), ('red', 'green')):
        plt.scatter(X1[y_set == i], X2[y_set == i],
                    c=color, label=f'Class {i}', edgecolor='black', s=50)

    plt.title(title)
    plt.xlabel('Feature 1 (e.g. Age)')
    plt.ylabel('Feature 2 (e.g. Estimated Salary)')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_decision_boundary(x_train_scaled, y_train.values, "KNN (Training set)")


plot_decision_boundary(x_test_scaled, y_test.values, "KNN (Test set)")


