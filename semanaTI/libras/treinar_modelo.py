import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

try:
    df = pd.read_csv('dados_coletados.csv')
except FileNotFoundError:
    print("Arquivo 'dados_coletados.csv' não encontrado")
    exit()

# B. Separa as Colunas:
# X (Features/Entrada): 63 coordenadas (X, Y, Z dos 21 pontos)
# y (Label/Saída): A coluna 'Sinal' (A, B, C, D)
X = df.drop('Sinal', axis=1) 
y = df['Sinal']

print(f"Total de amostras encontradas: {len(df)}")
print(f"Sinais que a IA irá aprender: {y.unique()}")




# O 'stratify=y' garante que as 4 letras sejam bem distribuídas nas duas partes.
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Amostras para Treino: {len(X_treino)}")
print(f"Amostras para Teste: {len(X_teste)}")


print("\nIniciando treinamento do modelo KNN...")
# O n_neighbors=5 define que a IA verificará os 5 vizinhos mais próximos para classificar um novo gesto.
modelo = KNeighborsClassifier(n_neighbors=3) 
modelo.fit(X_treino, y_treino)



y_predicao = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, y_predicao)

print("\n--- Treinamento Concluído ---")
print(f"Acurácia (Precisão) do Modelo em dados de Teste: {acuracia * 100:.2f}%")
print("\nRelatório Detalhado por Sinal:")
print(classification_report(y_teste, y_predicao))


nome_arquivo_modelo = 'modelo_libras.pkl'
with open(nome_arquivo_modelo, 'wb') as f:
    pickle.dump(modelo, f)

print(f"\nModelo salvo com sucesso como '{nome_arquivo_modelo}'.")
print("Pronto! Agora use 'tradutor_libras.py' para testar na webcam.")