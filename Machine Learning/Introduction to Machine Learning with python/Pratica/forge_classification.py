import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Instanciando o algoritmo k-Neighbors
clf = KNeighborsClassifier(n_neighbors=3)
# Passando os dados do conjunto de dados
X,y = mglearn.datasets.make_forge()

# Separando os dados em uma parte de teste e outra de treino
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Ajustando os dados na instancia do algoritmo
clf.fit(X_train, y_train)

# Realizando as predições e medindo a acuracia
print(f'Test set predictions: {clf.predict(X_test)}')
print(f'Test set accuracy: {clf.score(X_test, y_test)}')