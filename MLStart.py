#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:rgba(0,0,0,.70); text-align:center;font-family:fantasy;text-shadow:2px 1px 2px royalblue;font-size:50px"> INICIANDO EM MACHINE LEARNING </h1><hr>

# <h2 style="color:red;font-family:fantasy;text-shadow:1px 1px 1px black;font-size:30px;text-align:center;">Carrega módulos</h2>

# In[1]:


from sklearn.datasets import load_breast_cancer 
from sklearn.datasets import load_diabetes 
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2 style="color:lightgreen;font-family:fantasy;text-shadow:1px 1px 1px black;text-align:center;font-size:30px"> Carrega classes discretas</h2>

# In[2]:


dataset_cancer = load_breast_cancer()
print(dataset_cancer.feature_names)
print(dataset_cancer.target_names)


# <h2 style="color:lightblue;font-family:fantasy;text-shadow:1px 1px 1px black;text-align:center;font-size:30px">Carrega classes contínuas</h2>

# In[3]:


dataset_diabetes = load_diabetes()
print(dataset_diabetes.feature_names)
print(dataset_diabetes.target)


# <h2 style="color:royalblue;font-family:fantasy;text-shadow:1px 1px 1px black;text-align:center;font-size:30px;">Separa os datasets em treino e teste</h2>

# In[4]:


X_train_can, X_test_can, y_train_can, y_test_can = train_test_split(dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target, random_state=42)

X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target, random_state=42)


# <h2 style="color:gray;font-family:fantasy;text-shadow:1px 1px 1px black;font-size:26px;">Treinamento e avaliação do modelo com SVM</h2>

# In[5]:


training_accuracy = []
test_accuracy = []

kernels = ['linear', 'rbf', 'sigmoid']
for kernel in kernels:
  svm_model = svm.SVC(kernel=kernel)
  
  svm_model.fit(X_train_can, y_train_can)
  training_accuracy.append(svm_model.score(X_train_can, y_train_can))
  test_accuracy.append(svm_model.score(X_test_can, y_test_can))

plt.plot(kernels,training_accuracy, label='Acuracia no conj. treino')
plt.plot(kernels,test_accuracy, label='Acuracia no conj. teste')
plt.ylabel('Accuracy')
plt.xlabel('Kernels')
plt.legend()


# <h2 style="color:gray;font-family:fantasy;text-shadow:1px 1px 1px black;font-size:26px;">Treinamento e avaliação do modelo com árvores de decisão</h2>

# In[6]:


training_accuracy = []
test_accuracy = []

prof_max = range(1,10)

for md in prof_max:
  tree = DecisionTreeClassifier(max_depth=md,random_state=0)
  tree.fit(X_train_can,y_train_can)
  training_accuracy.append(tree.score(X_train_can, y_train_can))
  test_accuracy.append(tree.score(X_test_can, y_test_can))

plt.plot(prof_max,training_accuracy, label='Acuracia no conj. treino')
plt.plot(prof_max,test_accuracy, label='Acuracia no conj. teste')
plt.ylabel('Acuracia')
plt.xlabel('Profundidade Maxima')
plt.legend()


# <h2 style="color:gray;font-family:fantasy;text-shadow:1px 1px 1px black;font-size:26px;">Treinamento e avaliação do modelo com regressão linear</h2>

# In[7]:


training_accuracy = []
test_accuracy = []

for interception in [True, False]:
  regr = LinearRegression(fit_intercept=interception)
  regr.fit(X_train_dia, y_train_dia)
  training_accuracy.append(regr.score(X_train_dia, y_train_dia))
  test_accuracy.append(regr.score(X_test_dia, y_test_dia))

plt.plot(["Interc", "No Interc"],training_accuracy, label='Acuracia no conj. treino')
plt.plot(["Interc", "No Interc"],test_accuracy, label='Acuracia no conj. teste')
plt.ylabel('Acuracia')
plt.xlabel('Fit Intercept')
plt.legend()

