import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

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

print('Número de colunas apõs One-Hot encoding:', df.columns.size)

def ArvoreRegressao(X, y, profundidade, tam_teste, draw_graph):
    #print("Arvore de regressão - profudidade: " + str(profundidade) + ", proporção dos testes: " + str(tam_teste * 100) +
    #      "%, seed:" + str(seed))
    # Divide o conjunto em 2 partes: treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tam_teste)
    # Criando o modelo de árvore de decisão
    model = DecisionTreeClassifier(max_depth=profundidade)
    # Treinando o modelo
    model.fit(X_train, y_train)
    # Fazendo a predição
    predicted_value = model.predict(X_test)


    # A partir disso, calculamos a Acurácia e a F-Score
    score = accuracy_score(y_test.ravel(), predicted_value) * 100
    measure = f1_score(y_test, predicted_value, average='weighted') * 100
    print('Acurácia:', score);
    print('Measure:', measure)

    if draw_graph:
        # Visualizando a árvore de decisão
        plt.figure(figsize=(15, 10))
        plot_tree(model, feature_names=list(X.columns), class_names=['Não', 'Sim'], filled=True, rounded=True)
        plt.show()
        # Criando um DataFrame para comparar os valores reais com os preditos
        comparison_df = pd.DataFrame({'Valor Real': y_test, 'Valor Predito': predicted_value})

        # Ordenando o DataFrame para melhor visualização
        comparison_df.sort_values(by='Valor Real', inplace=True)
        # Mapeando os valores numéricos para "Não" e "Sim"
        comparison_df['Valor Real'] = comparison_df['Valor Real'].map({0: 'Não', 1: 'Sim'})
        comparison_df['Valor Predito'] = comparison_df['Valor Predito'].map({0: 'Não', 1: 'Sim'})
        # Gerando o gráfico de dispersão
        plt.scatter(range(len(comparison_df)), comparison_df['Valor Real'], label='Valor Real', alpha=0.7)
        plt.scatter(range(len(comparison_df)), comparison_df['Valor Predito'], label='Valor Predito', alpha=0.7)

        plt.xlabel('Amostras')
        plt.ylabel(target_column)
        plt.title('O imóvel possui área externa?')
        plt.legend()
        plt.show()

    return predicted_value, y_test


print('\n\nÁrvore de regressão com profundidade 1 e com 30% de proporção de teste' )
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=1, tam_teste=0.3, draw_graph=True)
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=1, tam_teste=0.3, draw_graph=False)
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=1, tam_teste=0.3, draw_graph=False)

print('\n\nÁrvore de regressão com profundidade 2 e com 30% de proporção de teste' )
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=2, tam_teste=0.3, draw_graph=True)
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=2, tam_teste=0.3, draw_graph=False)
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=2, tam_teste=0.3, draw_graph=False)

print('\n\nÁrvore de regressão com profundidade 5 e com 30% de proporção de teste')
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=5, tam_teste=0.3, draw_graph=True)
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=5, tam_teste=0.3, draw_graph=False)
predicted_value, y_test = ArvoreRegressao(X, y, profundidade=5, tam_teste=0.3, draw_graph=False)
