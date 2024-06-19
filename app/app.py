import base64
import os
from io import StringIO
from typing import Optional
import altair as alt
import numpy as np
import pandas as pd
import pdfkit
import requests
import streamlit as st
import torch
from dotenv import load_dotenv
from streamlit import session_state as ss
import vl_convert as vlc
from jinja2 import Environment, select_autoescape, FileSystemLoader
from ecg.form_schema import DataBytes
import sys

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")
sys.path.append(ROOT_DIR)
from models.preprocess.lead_filter_methods import gan_preprocess, med_filter, check_baseline

api_host = os.environ.get("API_HOST")


#TODO:
# 5) предсказание ритма

st.set_page_config(
    page_title="ЭКГ",
    page_icon="🧊",
)

st.title('🫀ЭКГ-сервис')

with st.expander(':arrow_up:Загрузка сигнала'):
    st.info('Заполните предлагаемые поля данных. Далее загрузите сигнал', icon="ℹ️")
    name = st.text_input("**Имя пациента:**")
    sr = st.number_input("**Частота дискретизации (sample rate):**", min_value=0, max_value=10000, step=1)
    age = st.number_input("**Возраст:**", min_value=0, max_value=120, step=1)
    gender = option = st.selectbox("**Пол:**", ("М", "Ж"))
    height = st.number_input("**Рост:**", min_value=0, max_value=300, step=1)
    weight = st.number_input("**Вес:**", min_value=0, max_value=300, step=1)
    date = str(st.date_input("**Дата:**"))
    device = st.text_input("**Устройство:**")
    uploaded_file = st.file_uploader("**Выберите файл:**", type=['txt', 'npy'])
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
    # print(uploaded_file)
    print('FILE')
    # print(type(uploaded_file.getvalue()))
    # print(type(file_content))
    # print(file_content, file_content.shape)

if success:
    with st.expander('📈Графики'):
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
        # st.write("Ответ сервиса: ", add_sig_req.status_code)
        info_res = requests.get(api_host + 'get_signal_info').json()
        with st.container(height=200, border=True):
            st.markdown("**〰️ Отведения**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                lead1 = st.checkbox('I', key='cb1')
                lead2 = st.checkbox('II', key='cb2')
                lead3 = st.checkbox('III', key='cb3')
            with col2:
                lead_avr = st.checkbox('AVR', key='cb4')
                lead_avl = st.checkbox('AVL', key='cb5')
                lead_avf = st.checkbox('AVF', key='cb6')
            with col3:
                lead_v1 = st.checkbox('V1', key='cb7')
                lead_v2 = st.checkbox('V2', key='cb8')
                lead_v3 = st.checkbox('V3', key='cb9')
            with col4:
                lead_v4 = st.checkbox('V4', key='cb10')
                lead_v5 = st.checkbox('V5', key='cb11')
                lead_v6 = st.checkbox('V6', key='cb12')


            leads_checkboxes = [lead1, lead2, lead3, lead_avr, lead_avl, lead_avf, lead_v1, lead_v2, lead_v3, lead_v4,
                                lead_v5, lead_v6]

        def change_cb():
            if not all_leads_tumbler:
                ss.cb1 = True
                ss.cb2 = True
                ss.cb3 = True
                ss.cb4 = True
                ss.cb5 = True
                ss.cb6 = True
                ss.cb7 = True
                ss.cb8 = True
                ss.cb9 = True
                ss.cb10 = True
                ss.cb11 = True
                ss.cb12 = True
            else:
                ss.cb1 = False
                ss.cb2 = False
                ss.cb3 = False
                ss.cb4 = False
                ss.cb5 = False
                ss.cb6 = False
                ss.cb7 = False
                ss.cb8 = False
                ss.cb9 = False
                ss.cb10 = False
                ss.cb11 = False
                ss.cb12 = False

        all_leads_tumbler = st.toggle("Все отведения", on_change=change_cb)
        r_peaks_checkbox = st.checkbox('R-пики')

        filter_options = st.multiselect(
            "Выберите фильтрацию",
            ["original", "median", "gan", "cycle_ganx2", "cycle_ganx3", "cycle_ganx4"],
            ["original"]
        )
        # st.write("**Фильтрации:**", filter_options)
        r_peaks = info_res['r_peaks']
        def draw_lead(sig_df: pd.DataFrame, lead_name: str, filter_options: Optional[dict]=None) -> st.altair_chart:
            sig_df['time, sec'] = sig_df.index / 100
            baseline_warning = 'На данном отведении обнаружен дрейф'
            for option in filter_options:
                if 'median' in option:
                    sig_df['mV'] = med_filter(torch.tensor(sig_df['mV'].values))
                if 'gan' in option:
                    sig_df['mV'] = gan_preprocess(torch.tensor(sig_df['mV'].values), inference_count=1)
                if 'ganx' in option:
                    sig_df['mV'] = gan_preprocess(torch.tensor(sig_df['mV'].values), inference_count=int(option[-1]))

            filter_options_title = ' + '.join(filter_options)
            if len(filter_options_title) != 0:
                filter_options_title = ' (' + filter_options_title + ')'
            print(sig_df)
            chart = alt.Chart(sig_df).mark_line().encode(
                x='time, sec:Q',
                y='mV'
            )
            if r_peaks_checkbox:
                vertical_lines = alt.Chart(pd.DataFrame({'R': np.array(r_peaks)/100})).mark_rule(color='red').encode(x='R')
                # Совмещаем график и вертикальные линии
                combined_chart = (chart + vertical_lines).properties(
                    width=670,  # задаем ширину графика
                    height=300,  # задаем высоту графика
                    title = {
                        "text": lead_name + filter_options_title,
                        "anchor": "middle",
                        "align": "center",
                    }
                )
            else:
                combined_chart = chart.properties(
                    width=670,
                    height=300,
                    title={
                        "text": lead_name + filter_options_title,
                        "anchor": "middle",
                        "align": "center",
                    }
                )
            with st.container(border=True):
                if check_baseline(sig_df['mV'].values):
                    st.warning(baseline_warning, icon='🚨')
                st.altair_chart(combined_chart.interactive(), use_container_width=True)
                print('chart saved')
                png_data = vlc.vegalite_to_png(combined_chart.to_json(), scale=2)
                # print(combined_chart.to_json())
                with open(f"static/{lead_name}.png", "wb") as f:
                    f.write(png_data)


        leads_to_report = []
        # Отображаем графики
        for i in range(len(leads_checkboxes)):
            # Если проставлен чекбокс отведения
            if leads_checkboxes[i]:
                sig_df = pd.DataFrame({'time': np.arange(len(file_content[:, 0])), 'mV': file_content[:, i]})
                lead_name = lead_names[i]
                draw_lead(sig_df, lead_name, filter_options)
                leads_to_report.append(lead_name)
        leads_to_report = list(map(lambda x: fr'{os.path.abspath('static')}\{x}.png', leads_to_report))


    # with st.expander('🧾Диагностическая ифнормация'):
    with st.container(border=True):
        st.header('🧾Диагностическая ифнормация', divider="green")
        st.subheader("1. Вариабельность сердечного ритма (ВСР)")

        signal_info_df = pd.DataFrame([list(info_res['time_domain_features'].values())],
                                      columns=list(info_res['time_domain_features'].keys()))
        with st.expander("🔻R-пики"):
            st.write("Отсчёты R-пиков:", info_res['r_peaks'])
            st.write("Длительности RR-интервалов:", info_res['nn_intervals'])
        st.dataframe(signal_info_df, hide_index=True)
        rhythm_statement = ''
        model_rhytm_hrv = st.selectbox(
            "Модель предсказания **ритма** на основе ВСР:",
            ("LGBMClassifier", "LinearSVC"),
            index=None,
            placeholder="Модель",
        )

        if model_rhytm_hrv is not None:
            gender_to_int = (lambda x: 1 if x == 'M' else 0)(gender)
            print("GENDER", gender_to_int)
            pred_rhytm_hrv = requests.get(api_host + f'predict_rhythm_by/{model_rhytm_hrv}/age={age}/gender={gender_to_int}')
            print("Ответ сервиса: ", pred_rhytm_hrv.status_code)
            if pred_rhytm_hrv.status_code == 200:
                data = pred_rhytm_hrv.json()
                # st.write(data)
                if data == "Синусовый ритм":
                    rhythm_statement = f"Результат предсказания: <b>{data}</b>. Отклонений не обнаружено."
                    print(rhythm_statement)
                else:
                    rhythm_statement = f'Результат предсказания: <b>{data}</b>. Является отклонением от медицинской нормы. Требуется осмотр лечащего специалиста'
                st.markdown(f':blue-background[**{data}**]')
            else:
                st.write("Не удалось классифицировать сигнал")
        diagnosis_statement = ''
        model_diagnosis_hrv = st.selectbox(
            "Модель предсказания **диагноза** на основе ВСР:",
            ("LGBMClassifier", "LSTM"),
            index=None,
            placeholder="Модель",
        )




        if model_diagnosis_hrv is not None:
            gender_to_int = (lambda x: 1 if x == 'M' else 0)(gender)
            pred_diagnosis_hrv = requests.get(api_host + f'predict_diagnostic_by/{model_diagnosis_hrv}/age={age}/gender={gender_to_int}')
            print("Ответ сервиса: ", pred_diagnosis_hrv.status_code)
            if pred_diagnosis_hrv.status_code == 200:
                data = pred_diagnosis_hrv.json()
                # st.write(data)
                if data == "Нормальная ЭКГ":
                    diagnosis_statement = f"Результат предсказания: <b>{data}</b>. Отклонений не обнаружено."
                else:
                    diagnosis_statement = f'Результат предсказания: <b>{data}</b>. Является отклонением от медицинской нормы. Требуется осмотр лечащего специалиста'
                st.markdown(f':blue-background[**{data}**]')
            else:
                st.write("Не удалось классифицировать сигнал")



        st.subheader("2. Классификация ЭКГ")
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False

        def click_button():
            st.session_state.clicked = True

        model_option = st.selectbox(
            "Выбор модели:",
            ("resnet1d", "inception1d", "vgg16"),
            index=None,
            placeholder="Модель",
        )
        st.write("В качестве модели классификации выбрана: ", f'`{model_option}`')

        if model_option:
            st.button("🕹️Запуск классификатора", on_click=click_button, disabled=False)
        else:
            st.button("🕹️Запуск классификатора", on_click=click_button, disabled=True)


        clsf_statement = ''
        if st.session_state.clicked:
            pred_res = requests.get(api_host + f'predict_by/{model_option}')
            print("Ответ сервиса: ", pred_res.status_code)
            if pred_res.status_code == 200:
                data = pred_res.json()
                class_description = {
                    "NORM": "Нормальный ЭКС",
                    "STTC": "Изменения в ST-сегменте",
                    "HYP": "Гипертрофия",
                    "CD": "Нарушениe проводимости"
                }
                for i in range(len(data['cls_pred'])):
                    descr = class_description[data['cls_pred'][i]]
                    statement_md = f'**{data['cls_pred'][i]} ({descr})** - ' + "{:.2f}%".format(data['cls_probs'][i]*100)
                    statement_html = f'<b>{data['cls_pred'][i]} ({descr})</b>. Вероятность - ' + "{:.2f}%".format(data['cls_probs'][i]*100) + '<br>'
                    clsf_statement += statement_html
                    st.write(statement_md)
                clsf_statement += "<text style='color:Tomato;'>Рекомендуется обратиться к лечащему специалисту для подтверждения диагноза.</text>"
            else:
                st.write("Не удалось классифицировать сигнал")

            env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape())
            template = env.get_template("report.html")
            html = template.render(
                name=name,
                age=age,
                sample_rate=sr,
                gender=gender,
                date=date,
                height=height,
                weight=weight,
                device=device,
                leads_images=leads_to_report,
                hrv_rhythm_statement=rhythm_statement,
                hrv_diagnosis_statement=diagnosis_statement,
                main_clsf_statement=clsf_statement
            )
            config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
            options = {
                "enable-local-file-access": True,
            }
            pdf = pdfkit.from_string(html, False, configuration=config, options=options)
            st.info("В отчёте будут отображены только выбранные отведения", icon="ℹ️")
            download_pdf_btn = st.download_button(
                "⬇️ Скачать PDF-отчёт",
                data=pdf,
                file_name=f"report_{name.lower()}.pdf",
                mime="application/octet-stream",
            )

