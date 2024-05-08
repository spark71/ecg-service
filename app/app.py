from io import StringIO
import altair as alt
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

with st.expander(':arrow_up:Загрузка сигнала'):
    st.markdown('''
        **Заполните предлагаемые поля данных. Далее загрузите сигнал.**
    ''')
    # st.image("https://static.streamlit.io/examples/dice.jpg")
    # st.button('Upload').on_click(show_popup)
    name = st.text_input("Имя пациента:")
    age = st.number_input("Возраст:", min_value=0, max_value=120, step=1)
    gender = option = st.selectbox("Пол:", ("М", "Ж"))
    height = st.number_input("Рост:", min_value=0, max_value=300, step=1)
    weight = st.number_input("Вес:", min_value=0, max_value=300, step=1)
    date = str(st.date_input("Дата:"))
    device = st.text_input("Устройство:")
    uploaded_file = st.file_uploader("Выберите файл:", type=['txt', 'npy'])
    file_content = None
    if uploaded_file:
        file_content = np.loadtxt(StringIO(uploaded_file.getvalue().decode('utf-8')), dtype=float)
    if (file_content is not None) and (file_content.shape == (1000, 12)):
        st.success('Сигнал успешно загружен.', icon="✅")
        success = True
    else:
        st.error('Файл не загружен. Пожалуйста, проверьте формат данных и потвторите попытку.', icon="🚨")
        success = False

if uploaded_file is not None:
    print(uploaded_file)
    print('FILE')
    print(type(uploaded_file.getvalue()))
    print(type(file_content))
    print(file_content, file_content.shape)

if success:
    with st.expander('📈Графики отведений'):
        st.markdown('Выберите отведения.')

        # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.checkbox('I')
            st.checkbox('II')
            st.checkbox('III')
        with col2:
            st.checkbox('AVR')
            st.checkbox('AVL')
            st.checkbox('AVF')
        with col3:
            st.checkbox('V1')
            st.checkbox('V2')
            st.checkbox('V3')
        with col4:
            st.checkbox('V4')
            st.checkbox('V5')
            st.checkbox('V6')

        st.button("Показать все отведения")
        # Создаем DataFrame с данными для графика
        df = pd.DataFrame({'x': np.arange(len(file_content[:, 0])), 'y': file_content[:, 0]})

        # Создаем интерактивный график сигнала с помощью библиотеки Altair
        # chart = alt.Chart(df).mark_line().encode(
        #     x='x',
        #     y='y',
        #
        # ).properties(
        #     width=600, height=400
        # ).interactive()

        # Создаем интерактивный график сигнала с помощью библиотеки Altair
        chart = alt.Chart(df).mark_line().encode(
            x='x',
            y='y'
        )

        # Наносим вертикальные линии
        vertical_lines = alt.Chart(pd.DataFrame({'x': [10, 40, 70]})).mark_rule(color='red').encode(x='x')

        # Совмещаем график и вертикальные линии
        combined_chart = (chart + vertical_lines).properties(
            width=600,  # задаем ширину графика
            height=300  # задаем высоту графика
        )

        # Отображаем график в Streamlit
        st.write(combined_chart)
        # for lead in range(file_content.shape[0]-1):
        #     # print(lead)
        #     st.line_chart(file_content[:, lead-1], color="#f23c24")

    with st.expander('🧾Диагностическая ифнормация'):
        st.header('Общие сведения о сигнале.')
        st.subheader("Параметры вариабельности сердечного ритма (ВСР)")
        st.subheader("Классификация ЭКГ")

