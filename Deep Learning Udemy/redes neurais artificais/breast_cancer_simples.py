import pandas as pd
from sklearn.model_selection import train_test_split

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

previsores_treinamento, previsores_test, classe_treinamento, classe_test = train_test_split(previsores, classe, test_size=0.25)
