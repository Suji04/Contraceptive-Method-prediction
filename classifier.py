import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

attributes = ["Wife's age", "Wife's education", "Husband's education", 
              "Number of children", "Wife's religion", "Wife is working",
              "Husband's occupation", "SLI", "Media exposure", "CMC"]

data = pd.read_csv("dataset.csv", names=attributes)


plt.scatter(data["Wife's age"], data["CMC"], alpha = .1) # important
plt.scatter(data["Number of children"], data["CMC"], alpha=.1) # important 
plt.scatter(data["Wife is working"], data["CMC"], alpha=.1)
plt.scatter(data["Husband's education"], data["CMC"], alpha=.1)
plt.scatter(data["Media exposure"], data["CMC"], alpha=.1)
plt.scatter(data["SLI"], data["CMC"], alpha=.1) # important
plt.scatter(data["Wife's education"], data["CMC"], alpha=.1) # important


plt.hist(data["Wife's age"])
plt.hist(data["Wife is working"])
plt.hist(data["SLI"])
plt.hist(data["Husband's education"])
plt.hist(data["Wife's education"])


X = np.float64(data.iloc[:,0:-1])
Y = np.float64(data.iloc[:,9:10])

from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X[:,0:1] = X_sc.fit_transform(X[:,0:1])
X[:,3:4] = X_sc.fit_transform(X[:,3:4])

X2 = np.zeros((1473,4))
X2[:,0:1] = X[:,0:1]
X2[:,1:2] = X[:,3:4]
X2[:,2:3] = X[:,1:2]
X2[:,3:4] = X[:,7:8]


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, train_size=.8, random_state=0)

from sklearn.svm import SVC
model_svm = SVC(kernel = "rbf", gamma = 1.5)
model_svm.fit(X_train, Y_train)


Y_pred = model_svm.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test, Y_pred)
F1 = f1_score(Y_test, Y_pred, average="micro") 

from sklearn.model_selection import cross_val_score
k_fold_acc = cross_val_score(model_svm, X_train, Y_train, cv=10)
K_fold_mean = k_fold_acc.mean()