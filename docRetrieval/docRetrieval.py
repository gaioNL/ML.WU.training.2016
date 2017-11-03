'''
@author: Gaio
@summary: Doc retrieval using tf-idf
'''

#Start Graphlab create
import graphlab

#read product review data
people = graphlab.SFrame('people_wiki.gl/')

#explore data
people.head()

#look for 1 specific person
obama = people[people['name'] == 'Barack Obama']

#compute df for obama
obama['word_count']=graphlab.text_analytics.count_words(obama['text'])
print obama['word_count']

#extract the words count in a new table & sort it
obama_word_count_tb=obama[['word_count']].stack('word_count',new_column_name=['word','count'])
obama_word_count_tb.head()
obama_word_count_tb.sort('count',ascending = False)

#compute tf & idf for the corpus
people['word_count']=graphlab.text_analytics.count_words(people['text'])
tfidf_people=graphlab.text_analytics.tf_idf(people['word_count'])
tfidf_people
people['tfidf']=tfidf_people

#analyze the obama doc
obama = people[people['name'] == 'Barack Obama']
obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending = False)

#calculate distances among people
clinton = people[people['name'] == 'Bill Clinton']
clinton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending = False)

beckham = people[people['name'] == 'David Beckham']
beckham[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending = False)

#check whether obama is closer to Clinton or Beckham
#cosine distance = the lower the distance, the closer the articles are
graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])
graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])

#create the nearest neighbor model for docs retrieval
knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')

#apply the model
#who is the closest to Obama
knn_model.query(obama)

#aahhhhhhhhhhhhhhhhhhhhhhh
arnold = people[people['name'] == 'Arnold Schwarzenegger']
knn_model.query(arnold)

#Assignment
elton = people[people['name'] == 'Elton John']

#compute df for elton
elton['word_count']=graphlab.text_analytics.count_words(elton['text'])
print elton['word_count']

#extract the words count in a new table & sort it
elton_word_count_tb=elton[['word_count']].stack('word_count',new_column_name=['word','count'])
elton_word_count_tb.sort('count',ascending = False).head()

#compute tf & idf for the corpus
people['word_count']=graphlab.text_analytics.count_words(people['text'])
tfidf_people=graphlab.text_analytics.tf_idf(people['word_count'])
tfidf_people
people['tfidf']=tfidf_people

#analyze the elton doc
elton = people[people['name'] == 'Elton John']
elton_tdidf_tb= elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending = False)
elton_tdidf_tb.head()

#elton -victoria
victoria = people[people['name'] == 'Victoria Beckham']
graphlab.distances.cosine(elton['tfidf'][0],victoria['tfidf'][0])#0.9567006376655429
#elton-paul
paul = people[people['name'] == 'Paul McCartney']
graphlab.distances.cosine(elton['tfidf'][0],paul['tfidf'][0])#0.8250310029221779

#create the nearest neighbor model for docs retrieval
knn_model_wordcount = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name')
knn_model_tfidf = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')

#apply the model
#who is the closest to elton

neighbours_list=['Billy Joel','Cliff Richard','Roger Daltrey','George Bush']
elton_wc=knn_model_wordcount.query(elton,k=None)
for mrWHo in neighbours_list:
    print elton_wc[elton_wc['reference_label']==mrWHo]
    
neighbours_list=['Elvis Presley','Tommy Haas','Roger Daltrey','Rod Stewart']
elton_tfidf = knn_model_tfidf.query(elton,k=None)
for mrWHo in neighbours_list:
    print elton_tfidf[elton_tfidf['reference_label']==mrWHo]

neighbours_list=['Stephen Dow Beckham','Louis Molloy','Adrienne Corri','Mary Fitzgerald (artist)']
victoria_wc=knn_model_wordcount.query(victoria,k=None)
for mrWHo in neighbours_list:
    print victoria_wc[victoria_wc['reference_label']==mrWHo]
    
neighbours_list=['Mel B','Caroline Rush','David Beckham','Carrie Reichardt']
victoria_tfidf = knn_model_tfidf.query(victoria,k=None)
for mrWHo in neighbours_list:
    print victoria_tfidf[victoria_tfidf['reference_label']==mrWHo]    


knn_model_tfidf.query(victoria)