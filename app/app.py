import base64
from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from ecg.form_schema import DataBytes

api_host = 'http://127.0.0.1:8000/'
#TODO:
# [+] 1) загрузка сигнала
# 2) визуализация и маркировка сигнала по отведениям
# 3) предсказание модели состояния
# 5) предсказание ритма
# 6) фильтрация сигнала
# 7) доп задачи



st.title('🫀ЭКГ-сервис')

with st.expander(':arrow_up:Загрузка сигнала'):
    st.info('Заполните предлагаемые поля данных. Далее загрузите сигнал', icon="ℹ️")
    name = st.text_input("Имя пациента:")
    sr = st.number_input("Частота дискретизации (sample rate):", min_value=0, max_value=10000, step=1)
    age = st.number_input("Возраст:", min_value=0, max_value=120, step=1)
    gender = option = st.selectbox("Пол:", ("М", "Ж"))
    height = st.number_input("Рост:", min_value=0, max_value=300, step=1)
    weight = st.number_input("Вес:", min_value=0, max_value=300, step=1)
    date = str(st.date_input("Дата:"))
    device = st.text_input("Устройство:")
    uploaded_file = st.file_uploader("Выберите файл:", type=['txt', 'npy'])
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1]
    else:
        file_extension = None
    file_content = None

    if file_extension == 'txt':
        file_content = np.loadtxt(StringIO(uploaded_file.getvalue().decode('utf-8')), dtype=float)
        bytes_signal = file_content.tobytes()
        base64_bytes = base64.b64encode(bytes_signal)
        base64_string_ecg_values = base64_bytes.decode("utf-8")

    elif file_extension == 'npy':
        file_content = np.load(uploaded_file)
        print("SHAPE: ", file_content.shape)
        bytes_signal = file_content.tobytes()
        base64_bytes = base64.b64encode(bytes_signal)
        base64_string_ecg_values = base64_bytes.decode("utf-8")

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
    with st.expander('📈Графики'):
        # st.markdown('Выберите отведения.')

        lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        payload = DataBytes(
            sample_rate=sr,
            name=name,
            gender=gender,
            age=age,
            height=height,
            weight=weight,
            device=device,
            ecg_values=base64_string_ecg_values
        )
        data = payload.json().encode('utf-8')

        add_sig_req = requests.post(api_host + 'add_sig_bytes', data=data)
        st.write(add_sig_req.status_code)
        info_res = requests.get(api_host + 'get_signal_info').json()


        with st.container(height=200, border=True):
            st.markdown("〰️ Отведения")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                lead1 = st.checkbox('I')
                lead2 = st.checkbox('II')
                lead3 = st.checkbox('III')
            with col2:
                lead_avr = st.checkbox('AVR')
                lead_avl = st.checkbox('AVL')
                lead_avf = st.checkbox('AVF')
            with col3:
                lead_v1 = st.checkbox('V1')
                lead_v2 = st.checkbox('V2')
                lead_v3 = st.checkbox('V3')
            with col4:
                lead_v4 = st.checkbox('V4')
                lead_v5 = st.checkbox('V5')
                lead_v6 = st.checkbox('V6')

            leads_checkboxes = [lead1, lead2, lead3, lead_avr, lead_avl, lead_avf, lead_v1, lead_v2, lead_v3, lead_v4,
                                lead_v5, lead_v6]

        r_peaks_checkbox = st.checkbox('R-пики')
        r_peaks = info_res['r_peaks']

        def draw_lead(sig_df: pd.DataFrame, lead_name: str) -> st.altair_chart:
            # Создаем интерактивный график сигнала с помощью библиотеки Altair
            chart = alt.Chart(sig_df).mark_line().encode(
                x='time',
                y='mV'
            )
            if r_peaks_checkbox:
                vertical_lines = alt.Chart(pd.DataFrame({'time': r_peaks})).mark_rule(color='red').encode(x='time')

                # Совмещаем график и вертикальные линии
                combined_chart = (chart + vertical_lines).properties(
                    width=670,  # задаем ширину графика
                    height=300,  # задаем высоту графика
                    title = {
                        "text": lead_name,
                        "anchor": "middle",
                        "align": "center",
                    }
                )
            else:
                combined_chart = chart.properties(
                    width=670,
                    height=300,
                    title={
                        "text": lead_name,
                        "anchor": "middle",
                        "align": "center",
                    }
                )
            st.altair_chart(combined_chart.interactive())


        # Отображаем графики
        for i in range(len(leads_checkboxes)):
            # Если проставлен чекбокс отведения
            if leads_checkboxes[i]:
                sig_df = pd.DataFrame({'time': np.arange(len(file_content[:, 0])), 'mV': file_content[:, i]})
                lead_name = lead_names[i]
                draw_lead(sig_df, lead_name)


    with st.expander('🧾Диагностическая ифнормация'):
        st.header('Общие сведения о сигнале', divider="green")
        st.subheader("Параметры вариабельности сердечного ритма (ВСР)")
        st.subheader("Классификация ЭКГ")
        api_host = 'http://127.0.0.1:8000/'
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False

        def click_button():
            st.session_state.clicked = True

        st.button('Запуск', on_click=click_button)
        if st.session_state.clicked:
            st.write('Button clicked!')
            pred_res = requests.get(api_host + 'predict')
            print(pred_res.status_code)
            if pred_res.status_code == 200:
                data = pred_res.json()
                for i in range(len(data['cls_pred'])):
                    st.write(f'{data['cls_pred'][i]} - ' + "{:.2f}%".format(data['cls_probs'][i]*100))

                signal_info_df = pd.DataFrame([list(info_res['time_domain_features'].values())],
                                  columns=list(info_res['time_domain_features'].keys()))
                st.write("Отсчёты R-пиков:", info_res['r_peaks'])
                st.write("Длительности RR-интервалов:", info_res['nn_intervals'])
                st.dataframe(signal_info_df, hide_index=True)
            else:
                st.write("Не удалось классифицировать сигнал")


