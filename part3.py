from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import random
  
wine = datasets.load_wine() 
iris = datasets.load_iris()
  

def main(dset):
    X = dset.data 
    y = dset.target 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .8, random_state = random.randint(1,101)) 
    
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
    svm_predictions = svm_model_linear.predict(X_test) 
        
    accuracy = svm_model_linear.score(X_test, y_test)
    print "accuracy: " + str(accuracy)

    cm = confusion_matrix(y_test, svm_predictions)

    plt.figure(figsize = (15,10.5))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

print "wine",
main(wine)
print "iris",
main(iris)