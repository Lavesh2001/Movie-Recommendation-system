import streamlit as st
import pickle
import requests

def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=ff4958fccf08898c2e6cc5848aa2ec9a'.format(movie_id))
    data=response.json()
    print(data)
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']


def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    recommended_movies_posters=[]
    for i in movies_list:
        movie_id=movies.iloc[i[0]].movie_id
        # fetch poster of movie from API
        recommended_movies_posters.append(fetch_poster(movie_id))

        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies,recommended_movies_posters

movies=pickle.load(open('movies.pkl','rb'))
movies_list=movies['title'].values

similarity=pickle.load(open('similarity.pkl','rb'))

st.title('Movie Recommender System')

selected_movie_name=st.selectbox("Enter the movie name:",movies_list)

if st.button('Recommend'):
    movies_names,movies_posters=recommend(selected_movie_name)

    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.text(movies_names[0])
        st.image(movies_posters[0])
    with col2:
        st.text(movies_names[1])
        st.image(movies_posters[1])
    with col3:
        st.text(movies_names[2])
        st.image(movies_posters[2])
    with col4:
        st.text(movies_names[3])
        st.image(movies_posters[3])
    with col5:
        st.text(movies_names[4])
        st.image(movies_posters[4])