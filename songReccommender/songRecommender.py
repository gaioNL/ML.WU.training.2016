'''
@author: Gaio
@summary: Song reccommender
'''

#Start Graphlab create
import graphlab

#read product review data
song_data = graphlab.SFrame('song_data.gl/')
song_data.head()

#redirect output to notebook
graphlab.canvas.set_target('ipynb')

#histogram of songs played 
len(song_data)
song_data['song'].show()

#find the unique users in the dataset
users = song_data['user_id'].unique()
len(users)

#create a song recommender

#split the data
train_data,test_data = song_data.random_split(.8,seed=0)

#popularity-based reccommender
pop_model = graphlab.popularity_recommender.create(train_data,user_id='user_id',item_id='song')
#recommend for a particular user
pop_model.recommend(users=[users[0]])
pop_model.recommend(users=[users[1]])

#buid recommender with personalization
pers_model = graphlab.item_similarity_recommender.create(train_data,user_id='user_id',item_id='song')
#recommend for a particular user
pers_model.recommend(users=[users[0]])
pers_model.recommend(users=[users[1]])
#get similar items

#compare the models
%matplotlib inline
model_performance = graphlab.recommender.util.compare_models(test_data, [pop_model, pers_model], user_sample=.05)

#assignment

#1
artists=['Kanye West', 'Foo Fighters', 'Taylor Swift','Lady GaGa']
for mrWHo in artists:
    print mrWHo, ': ', len(song_data[song_data['artist']==mrWHo]['user_id'].unique())
    
#2
artists=['Kanye West', 'Foo Fighters', 'Taylor Swift','Lady GaGa']

listenedToWhat = song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')}).sort('total_count',ascending = False)
listenedToWhat[0]#most listened Artist
listenedToWhat[-1]#least popular artist