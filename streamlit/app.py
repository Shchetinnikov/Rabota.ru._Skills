import streamlit as st

from processing.feature import (EDUCATION_INDEX, EXP_INDEX, LOCATION_INDEX,
                                SCHEDULE_INDEX, get_model, get_salary,
                                preprocess_features)
from processing.graphs import get_histogram, get_impacts, get_percentage_graph

LOCATIONS = LOCATION_INDEX.keys()
SCHEDULES = SCHEDULE_INDEX.keys()
EDUCATIONS = EDUCATION_INDEX.keys()
EXPERIENCES = EXP_INDEX.keys()


st.title("Cервис предсказания зарплат")

location = st.selectbox('Регион или город федерального значения', LOCATIONS)
position = st.text_input("Профессия")
education = st.radio('Образование', EDUCATIONS)
schedule = st.radio('Выберите график работы:', SCHEDULES)
experience = st.radio('Выберите опыт работы:', EXPERIENCES)


str_skills = st.text_input("Укажите ваши навыки через запятую", value='')
skills = str_skills.split(", ")


st.write(f'Выбраный регион -  :green[**{location}**]')
st.write(f'Ваша профессия - :red[**{position}**]')
st.write(f'Ваше образование - :blue[**{education}**]')
st.write(f'Вы выбрали - :blue[**{schedule}**]')
st.write(f'Ваш опыт - :blue[**{experience}**]')
st.write(f'Выбранные навыки - :blue[**{skills}**]')

button_pressed = st.button("Получить предсказание зарплаты")
model = get_model()

if button_pressed:
    vacancies = preprocess_features(
        location,
        position,
        education,
        schedule,
        experience,
        skills,
        )
    predicted_salaries = get_salary(model, vacancies)

    st.write(f'Предлагаемая зарплата {int(round(predicted_salaries[0], -3))}')

    feature_impacts = get_impacts(vacancies, model)
    percentage_graph = get_percentage_graph(feature_impacts)
    hist = get_histogram(feature_impacts)

    st.pyplot(percentage_graph)
    st.pyplot(hist)
