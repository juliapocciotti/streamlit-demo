import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('diabetes.csv')

# titulo
st.title('Prevendo Diabetes')
st.markdown('Modelo feito utilizando [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)')

# cabeçalho
st.subheader('Informações dos dados')

# nome usuário
user_input = st.sidebar.text_input('Digite seu nome')

# escrevendo o nome do usuário
st.write('Paciente: ', user_input)

# Modelo
X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

def get_user_data():
    
    pregnancies = st.sidebar.slider(label = 'Gravidez', 
                                    min_value = 0, 
                                    max_value = 15, 
                                    value = 1) # default value
    
    glicose = st.sidebar.slider(label = 'Glicose', min_value = 0, max_value = 200, value = 110)
    
    blood_pressure = st.sidebar.slider(label = 'Pressão Sanguínea', min_value = 0, max_value = 122, value = 72)
    
    skin_thickness = st.sidebar.slider(label = 'Espessura da pele', min_value = 0, max_value = 99, value = 20)
    
    insulin = st.sidebar.slider(label = 'Insulina', min_value = 0, max_value = 900, value = 30)
   
    BMI = st.sidebar.slider(label = 'Índice de massa corporal', min_value = 0.0,  max_value = 70.0, value = 15.0)
    
    DPF = st.sidebar.slider(label = 'Histórico familiar de diabetes', min_value = 0.0, max_value = 3.0, value = 0.0)
    
    age = st.sidebar.slider(label = 'Idade', min_value = 15, max_value = 100, value = 21)
    
    user_data = {'Gravidez'                       : pregnancies,
                 'Glicose'                        : glicose,
                 'Pressão sanguínea'              : blood_pressure,
                 'Espessura da pele'              : skin_thickness,
                 'Insulina'                       : insulin,
                 'Índice de massa corporal'       : BMI, 
                 'Histórico familiar de diabetes' : DPF,
                 'Idade'                          : age
                }
    
    features = pd.DataFrame(user_data, index=[0])
    
    return features

user_input_variables = get_user_data()
bar = st.bar_chart(user_input_variables.T)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

logisreg = LogisticRegression()
logisreg.fit(X_train, y_train)
y_pred = logisreg.predict(X_test)
acc_logisreg = accuracy_score(y_test, y_pred)

# Métricas e previsão
st.subheader('Acurácia do modelo')

st.write(round(acc_logisreg*100, 2))

user_pred = logisreg.predict(user_input_variables)

st.subheader('Previsão')
st.write(user_pred)

