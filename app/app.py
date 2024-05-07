from io import StringIO

import streamlit as st
import numpy as np
import pandas as pd
import time

#TODO:
# 1) загрузка сигнала
# 2) визуализация и маркировка сигнала по отведениям
# 3) предсказание модели состояния
# 5) предсказание ритма
# 6) фильтрация сигнала
# 7) доп задачи


st.header('🫀ЭКГ-сервис', divider='green')

with st.expander('⬇️Загрузка сигнала'):
    st.markdown('''
        **Заполните предлагаемые поля данных. Далее загрузите сигнал.**
    ''')
    # st.image("https://static.streamlit.io/examples/dice.jpg")
    # st.button('Upload').on_click(show_popup)
    name = st.text_input("Имя пациента:")
    age = st.number_input("Возраст:", min_value=0, max_value=120, step=1)
    gender = option = st.selectbox("Пол:", ("М", "Ж"))
    device = st.text_input("Устройство:")
    uploaded_file = st.file_uploader("Выберите файл:", type=['txt', 'npy'])
    # if uploaded_file is not None:
    #     # To read file as bytes:
    #     bytes_data = uploaded_file.getvalue()
    #     st.write(bytes_data)
    #
    #     # To convert to a string based IO:
    #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #     st.write(stringio)
    #
    #     # To read file as string:
    #     string_data = stringio.read()
    #     st.write(string_data)
    #
    #     # Can be used wherever a "file-like" object is accepted:
    #     dataframe = pd.read_csv(uploaded_file)
    #     st.write(dataframe)

if uploaded_file is not None:
    print(uploaded_file)
    print('FILE')
    print(type(uploaded_file.getvalue()))







