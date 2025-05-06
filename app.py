import streamlit as st
import pickle
import joblib
import requests


try:
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    music = pickle.load(open('music_list.pkl', 'rb'))
    similarity1 = pickle.load(open('similarity1.pkl', 'rb'))

    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


def recommend_movie(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = [movies.iloc[i[0]].title for i in distances[1:6]]
    return recommended_movie_names

def recommend_music(musics):
    music_index = music[music['title'] == musics].index[0]
    distances = sorted(list(enumerate(similarity1[music_index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = [music.iloc[i[0]].title for i in distances[1:6]]
    return recommended_music_names

st.title("🎬🎧 My Movie & Music Recommender + Sentiment Analyzer")


app_choice = st.radio("Choose an option:", ["🎯 Recommendation", "🧠 Sentiment Analysis"])


if app_choice == "🎯 Recommendation":
    selection_type = st.radio(
        "What do you want to recommend?",
        ["🎥 Film", "🎵 Music"],
        horizontal=True
    )

    if selection_type == "🎥 Film":
        selected_movie = st.selectbox("Select a movie:", movies['title'].values)
        if st.button("Recommend"):
            names = recommend_movie(selected_movie)
            for name in names:
                st.markdown(f"<h4>{name}</h4>", unsafe_allow_html=True)

    elif selection_type == "🎵 Music":
        selected_music = st.selectbox("Select a music title:", music['title'].values)
        if st.button("Recommend"):
            names = recommend_music(selected_music)
            for name in names:
                st.markdown(f"<h4>{name}</h>", unsafe_allow_html=True)


elif app_choice == "🧠 Sentiment Analysis":
    review = st.text_area("Enter your movie review:")
    if st.button("Predict Sentiment"):
        try:
            review_tfidf = vectorizer.transform([review])
            prediction = model.predict(review_tfidf)
            prediction_label = 'Positive' if prediction == 1 else 'Negative'
            if prediction_label == 'Positive':
                st.markdown(
                f"<div style='background-color:#4CAF50; padding:10px; border-radius:8px;'>"
                f"<p style='color:white; font-size:20px;'>The sentiment of the review is: <b>{prediction_label}</b></p>"
                f"</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background-color:#F44336; padding:10px; border-radius:8px;'>"
                    f"<p style='color:white; font-size:20px;'>The sentiment of the review is: <b>{prediction_label}</b></p>"
                    f"</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error making prediction: {e}")
