import os
import pickle
from pathlib import Path

import fasttext
import numpy as np
import pandas as pd

LOCATION_INDEX = {
    'Москва': 1,
    'Пермь': 10,
    'Краснодар': 16,
    'Оренбургская область': 174,
    'Санкт-Петербург': 2,
    'Курганская область': 268,
    'Екатеринбург': 3,
    'Ижевск': 36,
    'Московская область': 57,
    'Ростов-на-Дону': 6,
}

LOCINDEX_PLACE = {
    1: 0,
    10: 1,
    16: 2,
    174: 3,
    2: 4,
    268: 5,
    3: 6,
    36: 7,
    57: 8,
    6: 9,
}

SCHEDULE_INDEX = {
    'вахта': 0,
    'полный рабочий день': 1,
    'свободный график': 2,
    'сменный график': 3,
    'удаленная работа': 4,
    'частичная занятость': 5,
}

EDUCATION_INDEX = {
    'высшее': 0,
    'высшее (бакалавр)': 1,
    'любое': 2,
    'неполное высшее': 3,
    'среднее': 4,
    'среднее профессиональное': 5,
}

EXP_INDEX = {
    'Более 6 лет': 0,
    'Не указано': 1,
    'Нет опыта': 2,
    'От 1 года до 3 лет': 3,
    'От 3 до 6 лет': 4,
}

ft_path = '/home/ilgiz/dev/rabotaru/data/cc.ru.300.bin'
ft = fasttext.load_model(ft_path)
ft.get_dimension()

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = os.path.join(BASE_DIR, 'models', 'positive_model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

columns_names = model.feature_names_in_


def skills_embedding(skills: list[str]):
    sentence_vectors = np.zeros(300)

    for skill in skills:
        sentence_vector = ft.get_sentence_vector(skill)
        sentence_vectors += sentence_vector

    return sentence_vectors


def preprocess_features(location: str, position: str,
                        education: str, schedule: str,
                        experience: str, skills: list[str]
                        ):
    skill_vec = skills_embedding(skills)
    position_vec = ft.get_word_vector(position)
    location_vec = [0] * len(LOCATION_INDEX)
    location_vec[LOCINDEX_PLACE[LOCATION_INDEX[location]]] = 1
    schedule_vec = [0] * len(SCHEDULE_INDEX)
    schedule_vec[SCHEDULE_INDEX[schedule]] = 1
    education_vec = [0] * len(EDUCATION_INDEX)
    education_vec[EDUCATION_INDEX[education]] = 1
    experience_vec = [0] * len(EXP_INDEX)
    experience_vec[EXP_INDEX[experience]] = 1

    features = np.concatenate((skill_vec, position_vec), axis=0).tolist()
    features += location_vec + schedule_vec + education_vec + experience_vec
    df = pd.DataFrame([features,], columns=columns_names)
    return df


def get_model():
    return model


def get_salary(model, df):
    return model.predict(df)


def print_column_coef():
    coefs = model.coef_
    feature_names = model.feature_names_in_
    for i in range(600, len(feature_names)):
        print(coefs[i], feature_names[i], columns_names[i])
    print(coefs[-1])
    print(len(feature_names))
    print(len(coefs))
