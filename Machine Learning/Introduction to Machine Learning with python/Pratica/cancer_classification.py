import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
cancer = load_breast_cancer()

# Aqui estamos separando os dados e utilizando o stratify para ter uma distribuição igual em relação as classes
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

training_accuracy = []
test_accuracy = []

#Criando possibilidades com vizinhos de 1 a 10
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    #Salvando a acuracia
    training_accuracy.append(clf.score(X_train, y_train))
    #Salvando a acuracia geral
    test_accuracy.append(clf.score(X_test, y_test))

# Aqui plotamos o gráfico construindo os eixos x e y e nomeando os dados
plt.plot(neighbors_settings, training_accuracy, label='Training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='Test accuracy')
plt.ylabel("Accuracy")
plt.xlabel('N_neighbors')
plt.legend()
plt.show()