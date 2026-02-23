 ##Iris Flower Classification Objective: Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on



from sklearn.datasets import load_iris
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
X= iris.data
y = iris.target

df = pd.DataFrame(X,columns=iris.feature_names)
df['species'] = iris.target_names[y]
df.head()


import seaborn as sns 
import matplotlib.pyplot as plt

sns.pairplot(df,hue='species',kind='kde')
df.hist(figsize=(10,10))
plt.show()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_scaled,y_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)



dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {"Logistic regression": log_reg, "Decision Tree":dt}

for name, model in models.items():
    if name == "Decision Tree":
        y_pred = model.predict(X_test)   # unscaled
    else:
        y_pred = model.predict(X_test_scaled)  # scaled

        print(f"\n{name}")
        print("Accuracy \n",accuracy_score(y_test,y_pred))
        print("Classification report \n", classification_report(y_test,y_pred))
        print("Knn Score \n",knn.score(X_test,y_test))
        print("Confusion matrix \n", confusion_matrix(y_test,y_pred))
        print("models score",model.score(X_test,y_test))






