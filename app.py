import pickle
import streamlit as st
import requests
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
# def fetch_movie_poster(movie_id):
#     url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
#     data = requests.get(url)
#     data = data.json()
#     poster_path = data['poster_path']
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path

def recommend_movie(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        # recommended_movie_posters.append(fetch_movie_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names

def recommend_music(musics):
    music_index = music[music['title'] == musics].index[0]
    distances = sorted(list(enumerate(similarity1[music_index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    for i in distances[1:6]:
        recommended_music_names.append(music.iloc[i[0]].title)
    return recommended_music_names




# try:
#     # Load the saved model and vectorizer
#     # model = joblib.load('sentiment_model.pkl')
#     # vectorizer = joblib.load('tfidf_vectorizer.pkl')
# except Exception as e:
#     st.error(f"Error loading model or vectorizer: {e}")
#     st.stop()


# Load models and data
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
music = pickle.load(open('music_list.pkl', 'rb'))
similarity1 = pickle.load(open('similarity1.pkl', 'rb'))
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def index():
    return render_template('index.html', movies=movies['title'].values, music=music['title'].values)


@app.route('/recommend', methods=['POST'])
def recommend():
    selection_type = request.form.get('selection_type')
    item_name = request.form.get('item_name')

    if selection_type == 'movie':
        recommended_names = recommend_movie(item_name)
    else:
        recommended_names = recommend_music(item_name)

    return {'recommendations': recommended_names}


@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    try:
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)
        label = 'positive' if prediction == 1 else 'negative'
        return {'sentiment': label}
    except Exception as e:
        return {'error': str(e)}
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

    




