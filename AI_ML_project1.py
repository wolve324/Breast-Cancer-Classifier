
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt  

breast_cancer_data=load_breast_cancer()
print(breast_cancer_data)
x=breast_cancer_data.feature_names
print(x)
training_data,validation_data,training_labels,validation_labels=train_test_split(breast_cancer_data.data,breast_cancer_data.target,test_size=0.2,random_state=100)
print(len(training_data))
print(len(training_labels))
accuracies=[]
for k in range(1,100):
  classifier=KNeighborsClassifier(n_neighbors=k)
  classifier.fit(training_data,training_labels)
  classifier.score(validation_data,validation_labels)
  accuracies.append(classifier.score(validation_data,validation_labels))
k_list=range(1,100)
plt.title("Breast Cancer Classifier Accuracy")
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.plot(k_list,accuracies)
plt.show()



