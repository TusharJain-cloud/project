import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
    return df[df.index ==index]["movie_title"].values[0]

def get_index_from_title(title):
    return df[df.movie_title == title]["index"].values[0]

df = pd.read_csv('main_data.csv')
#print (df.columns)

features = ['director_name','genres','actor_1_name', 'actor_2_name', 'actor_3_name']

cv = CountVectorizer()

count_matrix = cv.fit_transform(df["comb"])

cosine_sim = cosine_similarity(count_matrix)

movie_user_likes = 'Avatar'

movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))



i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[1]))
    i = i + 1
    if i > 10:
        break