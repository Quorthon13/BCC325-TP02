import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler

# Leitura dos conjuntos de treinamento e teste
data_file = 'sheets/dados_aluguel_todos.csv'

df = pd.read_csv(data_file, sep=';')

# Função para converter o valor formatado em 'Valor do Aluguel' para um float
def format_valor_aluguel(valor_formatado):
    valor_sem_simbolo = valor_formatado.replace('R$', '').replace(".", "").replace(",", ".").strip()

    return float(valor_sem_simbolo)

def format_sim_nao(valor_sim_nao):
    return 1 if valor_sim_nao == 'Sim' else 0

# converte o valor do aluguel em float
df['Valor do Aluguel'] = df['Valor do Aluguel'].apply(format_valor_aluguel)

# converte colunas sim/não em 0 ou 1
sim_nao_columns=['Garagem', 'Mobiliado', 'Área externa', 'Área de serviço']
for column in sim_nao_columns:
    df[column] = df[column].apply(format_sim_nao)

# One-Hot Encoding para variáveis categóricas
nominal_columns = ['Tipo', 'Bairro']
df = pd.get_dummies(df, columns=nominal_columns)

target_column = 'Área externa'
X = df.drop(target_column, axis=1)
y = df[target_column]

def RegressaoLogistica(X, y, tam_teste, draw_graph):
    # Escalonamento dos Dados
    scaler = MinMaxScaler()  # Inicializa o MinMaxScaler
    X_scaled = scaler.fit_transform(X)  # Escalona as características (features)

    # Divide o conjunto de dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=tam_teste)

    # Cria e treina o modelo de Regressão Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Faz as previsões no conjunto de teste
    predicted_value = model.predict(X_test)

    # Calcula a Acurácia e o F1 Score do modelo
    score = accuracy_score(y_test.ravel(), predicted_value) * 100
    measure = f1_score(y_test, predicted_value, average='weighted') * 100
    conf_matrix = confusion_matrix(y_test, predicted_value)
    print('Acurácia:', score)
    print('Medida F1:', measure)
    # print('\nMatriz de Confusão:')
    # print(conf_matrix)
    # Realiza a Validação Cruzada para acurácia média
    scores = cross_val_score(model, X_scaled, y, cv=5)  # cv=5 para validação cruzada com 5 folds
    print('Acurácia média da Validação Cruzada:', scores.mean() * 100)
    print('Desempenho em cada fold:', [score * 100 for score in scores])

    # Realiza a Validação Cruzada e obtém os F1 Scores em cada fold
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1_weighted')
    print('\nF1 Score médio da Validação Cruzada:', scores.mean() * 100)
    print('Desempenho em cada fold:', [score * 100 for score in scores])

    if draw_graph:
        # Cria um DataFrame para comparar os valores reais com os valores previstos
        comparison_df = pd.DataFrame({'Valor Real': y_test, 'Valor Previsto': predicted_value})

        # Mapeando os valores numéricos para "Não" e "Sim"
        comparison_df.sort_values(by='Valor Real', inplace=True)
        comparison_df['Valor Real'] = comparison_df['Valor Real'].map({0: 'Não', 1: 'Sim'})
        comparison_df['Valor Previsto'] = comparison_df['Valor Previsto'].map({0: 'Não', 1: 'Sim'})

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Generate the scatter graph in the first subplot
        ax1.scatter(range(len(comparison_df)), comparison_df['Valor Real'], label='Valor Real', alpha=0.7)
        ax1.scatter(range(len(comparison_df)), comparison_df['Valor Previsto'], label='Valor Previsto', alpha=0.7)
        ax1.set_xlabel('Amostras')
        ax1.set_ylabel(target_column)
        ax1.set_title('Comparação entre Valor Real e Valor Previsto')
        ax1.legend()

        # Generate the confusion matrix as a heatmap in the second subplot
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
        ax2.set_xlabel('Valor Previsto')
        ax2.set_ylabel('Valor Real')
        ax2.set_title('Matriz de Confusão')

        plt.tight_layout()
        plt.show()
    return predicted_value, y_test



print('\n\nRegressão Logística com 10% de proporção de teste')
predicted_value, y_test = RegressaoLogistica(X, y, tam_teste=0.1, draw_graph=True)

print('\n\nRegressão Logística com 20% de proporção de teste')
predicted_value, y_test = RegressaoLogistica(X, y, tam_teste=0.2, draw_graph=True)

print('\n\nRegressão Logística com 30% de proporção de teste')
predicted_value, y_test = RegressaoLogistica(X, y, tam_teste=0.3, draw_graph=True)
