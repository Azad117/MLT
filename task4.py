import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/task4.csv")
X, y = data.drop("label", axis=1), data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

gmm = GaussianMixture(n_components=len(y.unique()), random_state=0)
gmm.fit(X_train)

y_pred = gmm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Predicted:", y_pred.tolist())
print("Accuracy:", acc)


///Input :

feature1,feature2,feature3,label
5.1,3.5,1.4,0
4.9,3.0,1.4,0
6.2,3.4,5.4,1
5.9,3.0,5.1,1
5.5,2.3,4.0,1
4.6,3.6,1.0,0
6.7,3.1,4.7,1
5.0,3.4,1.5,0
6.3,2.7,4.9,1


//Output:

Predicted: [0, 0, 0]
Accuracy: 0.6666666666666666


