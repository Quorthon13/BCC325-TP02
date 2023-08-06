import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, cross_val_score # Import train_test_split function
from sklearn.metrics import mean_squared_error, r2_score
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

target_column = 'Valor do Aluguel'
X = df.drop(target_column, axis=1)
y = df[target_column]

print('Número de colunas apõs One-Hot encoding:', df.columns.size)

def ArvoreRegressao(X, y, profundidade, tam_teste, draw_graph):
    #print("Arvore de regressão - profudidade: " + str(profundidade) + ", proporção dos testes: " + str(tam_teste * 100) +
    #      "%, seed:" + str(seed))
    # Divide o conjunto em 2 partes: treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tam_teste)
    # Criando o modelo de árvore de decisão
    model = DecisionTreeRegressor(max_depth=profundidade)
    # Treinando o modelo
    model.fit(X_train, y_train)
    # Fazendo a predição
    predicted_value = model.predict(X_test)

    # Calcular o Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predicted_value)
    # Calcular o R-squared (R2)
    r2 = r2_score(y_test, predicted_value)
    print("Mean Squared Error (MSE):", f'{mse:.2f}')
    print("R-squared (R2):", f'{r2:.2f}')

    # Aplicar Validação Cruzada
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_mse_scores = -cv_scores  # O resultado da validação cruzada retorna negativo do MSE, por isso invertemos o sinal

    print("MSE médio da Validação Cruzada:", f'{cv_mse_scores.mean():.2f}')


    if draw_graph:
        # Criando um DataFrame para comparar os valores reais com os preditos
        comparison_df = pd.DataFrame({'Valor Real': y_test, 'Valor Predito': predicted_value})

        # Ordenando o DataFrame para melhor visualização
        comparison_df.sort_values(by='Valor Real', inplace=True)

        # Criando uma figura com dois subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Gerando o gráfico de dispersão no primeiro subplot
        ax1.scatter(range(len(comparison_df)), comparison_df['Valor Real'], label='Valor do Aluguel Real', alpha=0.7)
        ax1.scatter(range(len(comparison_df)), comparison_df['Valor Predito'], label='Valor do Aluguel Predito', alpha=0.7)
        ax1.set_xlabel('Amostras')
        ax1.set_ylabel('Valor')
        ax1.set_title('Comparação entre Valor do Aluguel Real e Valor do Aluguel Predito')
        ax1.legend()

        # Plot da Árvore de Regressão no segundo subplot
        plot_tree(model, feature_names=list(X.columns), filled=True, rounded=True, ax=ax2)
        ax2.set_title('Árvore de Regressão')

        plt.tight_layout()
        plt.show()

    return mse, r2, predicted_value, y_test



print('\n\nÁrvore de regressão com profundidade 1 e com 30% de proporção de teste', )
mse, r2, predicted_value, y_test = ArvoreRegressao(X, y, profundidade=1, tam_teste=0.3, draw_graph=True)

print('\n\nÁrvore de regressão com profundidade 2 e com 30% de proporção de teste', )
mse, r2, predicted_value, y_test = ArvoreRegressao(X, y, profundidade=2, tam_teste=0.3, draw_graph=True)

print('\n\nÁrvore de regressão com profundidade 5 e com 30% de proporção de teste', )
mse, r2, predicted_value, y_test = ArvoreRegressao(X, y, profundidade=5, tam_teste=0.3, draw_graph=True)

print('\n\nÁrvore de regressão com profundidade 10 e com 30% de proporção de teste', )
mse, r2, predicted_value, y_test = ArvoreRegressao(X, y, profundidade=10, tam_teste=0.3, draw_graph=True)

print('\n\nÁrvore de regressão com profundidade 20 e com 30% de proporção de teste', )
mse, r2, predicted_value, y_test = ArvoreRegressao(X, y, profundidade=20, tam_teste=0.3, draw_graph=True)

print('\n\nÁrvore de regressão com profundidade 50 e com 30% de proporção de teste', )
mse, r2, predicted_value, y_test = ArvoreRegressao(X, y, profundidade=50, tam_teste=0.3, draw_graph=True)

print('\n\nÁrvore de regressão com profundidade 100 e com 30% de proporção de teste', )
mse, r2, predicted_value, y_test = ArvoreRegressao(X, y, profundidade=100, tam_teste=0.3, draw_graph=True)