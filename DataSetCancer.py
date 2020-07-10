#!/usr/bin/env python
# coding: utf-8

# ### Fatec Mogi Mirim
# # Projeto de IA - 1 Semestre de 2020
# 
# ## Bibliotecas
# - Pandas
#   - É uma lib Python focada em manipulação de dados
#   - [Site Oficial da Pandas](https://pandas.pydata.org/)
#   - Para instalar a lib utilize o comando abaixo:
#   ```
#   pip install pandas
#   ```
# - NumPy
#   - É uma lib Python de codigo aberto focado em processamento computacional numerico
#   - [Site Oficial da NumPy](https://numpy.org/)
#   - Para instalar a lib utilize o comando abaixo:
#   ```
#   pip install numpy
#   ```
# - Matplotlib
#   - Lib Python focada em criação de gráficos
#   - [Site Oficial da Matplotlib](https://matplotlib.org/)
#   - Para instalar a lib utilize o comando abaixo:
#   ```
#   pip install matplotlib
#   ```
# - Seaborn
#   - Lib Python baseada em Matplotlib focada em visualização de gráficos estatisticos
#   - [Site Oficial da Seaborn](https://matplotlib.org/)
#   - Para instalar a lib utilize o comando abaixo:
#   ```
#   pip install seaborn
#   ```
# - Scikit-learn (Sklearn)
#   - Lib Python que oferece uma gama de ferramentas para fazer analise de dados
#   - [Site Oficial do Scikit-learn](https://scikit-learn.org/stable/)
#   - Para instalar a lib utilize o comando abaixo:
#   ```
#   pip install sklearn
#   ```
#   

# In[1]:


import pandas as pd            
import numpy as np              
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importando os Dados Base

# In[7]:


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer
cancer.keys()
print(cancer['DESCR'])
print(cancer['target_names'])
print(cancer['target'])
print(cancer['feature_names'])
print(cancer['data'])


# ### Organizando a Tabela de forma visivel

# In[8]:


cancer['data'].shape

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()
df_cancer.tail()


# ### Visualizar os dados com a Lib MatPlot

# In[17]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
sns.countplot(df_cancer['target'], label = "Count")


# In[13]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
sns.lmplot('mean area', 'mean smoothness', hue ='target', data = df_cancer, fit_reg=False)


# ### Checando a correlação entre as variaveis

# In[18]:


plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True)


# ### Treinando e avaliando o modelo

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

svc_model = SVC()
svc_model.fit(X_train, y_train)
# Avaliando o modelo
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict))


# ### 1º Melhoramento do Modelo
# 
# Realizando a Normalização de dados através do dimensionamento de recursos (Tipo de normalização baseada em Uni) que classifica os valores dentro do intervalo [0,1].
# Tem por base a seguinte equação:
# ```
# X '= (X - Xmin) / (Xmáx - Xmin)
# ```

# In[26]:


min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_test,y_predict))


# ### 2º Melhoramento do Modelo
# 
# - Parâmetro C: controla o trade-off entre a classificação de pontos de treinamento corretamente e com um limite de decisão suave:
#     - O pequeno C (solto) reduz o custo (penalidade) da classificação incorreta (margem suave)
#     - Grande C (estrito) eleva o custo da classificação incorreta (margem bruta), forçando o modelo para explicar os dados de entrada mais restritos e potencialmente sobrepostos
#      
#      
#      
# - Parâmetro gama: controla até que ponto a influência de um único conjunto de treinamento atinge
#     - Grande gama: alcance próximo (pontos de dados mais próximos têm alto peso)
#     - Gama pequena: longo alcance (mais solução de generalização)

# In[30]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

grid.best_params_
grid.best_estimator_

grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)


# In[31]:


sns.heatmap(cm, annot=True)

print(classification_report(y_test,grid_predictions))


# In[ ]:




