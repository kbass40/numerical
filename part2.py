from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn

wine = datasets.load_wine()
iris = datasets.load_iris()

def main(dset):
    model = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(dset.data,dset.target, test_size=0.8)
    model.fit(X_train, y_train)

    print "accuracy: " + str(model.score(X_test, y_test))
    y_predicted = model.predict(X_test)

    cm = confusion_matrix(y_test, y_predicted)

    plt.figure(figsize = (15,10.5))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

print "iris",
main(iris)
print "wine",
main(wine)