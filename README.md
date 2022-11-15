# Movie-Recommendation-systemimport numpy as np
# In[57]:


import numpy as np
import pandas as pd
import ast


# In[58]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[59]:


movies.head(1)


# In[60]:


credits.head(1)
#credits.head(1)['cast'].values


# In[61]:


movies=movies.merge(credits,on='title')


# In[62]:


movies.head(1)


# In[63]:


# genres
# id
# keywords
# title
# overview
# cast
# crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[64]:


movies.head()


# In[65]:


movies.isnull().sum()


# In[66]:


movies.dropna(inplace=True) 


# In[67]:


movies.duplicated().sum()


# In[68]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[69]:


movies['genres']=movies['genres'].apply(convert)


# In[70]:


movies.head()


# In[71]:


movies['keywords']=movies['keywords'].apply(convert)


# In[72]:


movies.head()


# In[73]:


def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i['name'])
        else:
            break
        counter+=1
    return l


# In[74]:


movies['cast']=movies['cast'].apply(convert3)


# In[75]:


movies.head()


# In[76]:


def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


# In[77]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[78]:


movies.head()


# In[79]:


movies['overview'][0]


# In[80]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[81]:


movies.head()


# In[82]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[83]:


movies.head()


# In[84]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[85]:


movies.head()


# In[86]:


new_df=movies[['movie_id','title','tags']]


# In[87]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[88]:


new_df.head()


# In[89]:


new_df['tags'][0]


# In[90]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[91]:


new_df.head()


# In[92]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[93]:


def stem(text):
       y=[]
       for i in text.split():
           y.append(ps.stem(i))
       return " ".join(y)


# In[94]:


new_df['tags']=new_df['tags'].apply(stem)


# In[95]:


new_df['tags'][0]


# In[96]:


new_df['tags'][1]


# In[97]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[98]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[99]:


vectors[0]


# In[100]:


cv.get_feature_names()


# In[101]:


ps.stem('danced')


# In[102]:


from sklearn.metrics.pairwise import cosine_similarity


# In[103]:


similarity=cosine_similarity(vectors)


# In[104]:


similarity[0]


# In[105]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index] 
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
       
    


# In[106]:


recommend('Batman Begins')


# In[107]:


import pickle


# In[108]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[110]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:

