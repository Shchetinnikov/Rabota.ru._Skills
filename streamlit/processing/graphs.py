import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

labels = ['work_skills', 'custom_position', 'city_id', 'schedule',
          'education_name', 'required_experience']
human_labels = ['Навыки', 'Должность', 'Регион', 'График работы',
                'Образовние', 'Опыт', ]


def get_impacts(vacancies, model):
    vacancie = vacancies.loc[0]
    feature_impacts = []

    for label in labels:
        indexes = [index for index, col in enumerate(
            model.feature_names_in_) if label in col]
        feature_coefs = model.coef_[indexes]
        feature_vec = vacancie.values[indexes]
        feature_impacts.append(feature_vec.dot(feature_coefs))

    return np.array(feature_impacts)


def get_percentage_graph(impacts):
    impacts = np.abs(impacts)
    contrib_ratios = np.array(impacts / sum(impacts))
    fig, ax = plt.subplots()
    ax.pie(contrib_ratios,
           labels=human_labels,
           colors=sns.color_palette('pastel')[0:6],
           autopct='%.0f%%'
           )
    return fig


def get_histogram(impacts):
    impacts = np.abs(impacts)
    palette1 = ["blue", "green", "red", "orange", "purple", "brown"]
    df_hist = pd.DataFrame({
        'labels': human_labels,
        'feature_contribution': impacts
        })
    plt.figure(figsize=(8, 6))
    sns.barplot(df_hist,
                x="labels",
                y="feature_contribution",
                fill=True,
                palette=palette1)
    plt.title("График вкладов различных признаков")
    plt.xlabel("Признаки")
    plt.ylabel("Вклад")
    return plt
