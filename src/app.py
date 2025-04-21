import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch

from output import ml_eng_output, chn_output

def plot_scanpath(scanpath, text, density_pred, cf):
    sac_range = np.append(np.arange(-cf["max_sn_len"] + 3, cf["max_sn_len"] - 1), cf["max_sn_len"] - 1)

    fig = go.Figure()

    # Для всех шагов создаем линию пути
    for timestep in range(len(scanpath) - 1):  # Проходим по всем фиксированным шагам
        cur_loc = scanpath[timestep]
        target_loc = sac_range + cur_loc
        valid_mask = (0 <= target_loc) & (target_loc < len(text))

        # Получаем вероятности для текущего шага
        probs_step = density_pred[timestep][0].detach().squeeze().numpy()
        filtered_probs = probs_step[valid_mask]
        filtered_words = [text[i] for i in target_loc[valid_mask] if i < len(text)]  # Проверка на выход за пределы

        # Добавляем тепловую карту для текущего шага
        fig.add_trace(go.Heatmap(
            z=[filtered_probs],
            x=filtered_words,
            colorscale="YlGnBu",
            showscale=True,
            colorbar=dict(title="Плотность"),
            zmin=0,
            zmax=1
        ))

        # Рисуем стрелки пути взгляда для текущего шага
        y = 1.2  # начальная высота
        if timestep + 1 < len(scanpath):  # Проверка на выход за пределы списка
            dx = scanpath[timestep + 1] - scanpath[timestep]
            if dx <= 0:
                y -= 0.2  # рефиксация
            fig.add_trace(go.Scatter(
                x=[text[scanpath[timestep]], text[scanpath[timestep + 1]]] if scanpath[timestep] < len(text) and scanpath[timestep + 1] < len(text) else [],
                y=[y, y],
                mode="lines+markers+text",
                line=dict(color="blue"),
                marker=dict(size=10),
                text=[str(timestep), str(timestep + 1)],
                textposition="top center",
                showlegend=False
            ))

    # Текущий взгляд (красная стрелка, если нужно)
    if len(scanpath) > 1:
        idx1 = scanpath[-2]
        idx2 = scanpath[-1]
        if idx1 < len(text) and idx2 < len(text):
            fig.add_trace(go.Scatter(
                x=[text[idx1], text[idx2]],
                y=[y, y],
                mode="lines+markers",
                line=dict(color="red", dash="dot"),
                marker=dict(size=12, color='red'),
                showlegend=False
            ))

    fig.update_layout(
        title="Визуализация пути взгляда",
        xaxis_title="Слова",
        yaxis=dict(range=[0, 1.5]),
        height=500
    )

    return fig


# Конфиги
cf_chn = {
    "model_pretrained": "bert-base-chinese",
    "atten_type": 'local-g',
    "max_sn_len": 27,
    "max_sp_len": 40,
    "et_path": "../training_results/BSC/CHN_ET.pth",
    "fn_path": "../training_results/BSC/CHN_FN.pickle"
}

cf_ml = {
    "model_pretrained": "bert-base-multilingual-cased",
    "atten_type": 'local-g',
    "max_sn_len": 27,
    "max_sn_token": 35,
    "max_sp_len": 52,
    "et_path": "../training_results/MECO/ML_ET.pth",
    "fn_path": "../training_results/MECO/ML_FN.pickle"
}

# ——— Интерфейс Streamlit ———
st.set_page_config(layout="wide")
st.title("👁️ Eyettention: Визуализация пути взгляда")

# Язык
lang = st.selectbox("Выберите язык", ["ru", "en", "de", "fr", "zh"])

# Ввод предложения
sentence = st.text_input("Введите предложение")

if sentence.strip():
    with st.spinner("Обрабатываем предложение..."):
        try:
            cf = None
            if lang == "zh":
                scanpath, density_pred, fixated_characters = chn_output(cf_chn, sentence)
                text = fixated_characters
                cf = cf_chn
            else:
                scanpath, density_pred, _ = ml_eng_output(cf_ml, sentence)
                text = sentence.strip().split()
                cf = cf_ml

            # Визуализация полного пути взгляда
            fig = plot_scanpath(scanpath[0], text, density_pred, cf)
            st.plotly_chart(fig, use_container_width=True)

            # Подсветка пути взгляда
            st.markdown("### 🔍 Путь взгляда")
            highlighted_text = ""
            for i, word in enumerate(text):
                if i in scanpath[0]:
                    highlighted_text += f"<span style='background-color: #FFCCCC; padding: 2px; border-radius: 4px;'> <strong>{word}</strong> </span> "
                else:
                    highlighted_text += word + " "
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # Таблица с полным путём взгляда
            st.markdown("### 📊 Таблица фиксаций")
            fixation_data = pd.DataFrame({
                "Шаг": list(range(len(scanpath[0]))),
                "Индекс слова": scanpath[0],
                "Слово": [text[i] if 0 <= i < len(text) else "[UNK]" for i in scanpath[0]]
            })
            st.dataframe(fixation_data)

        except Exception as e:
            st.error(f"Ошибка при обработке: {e}")
