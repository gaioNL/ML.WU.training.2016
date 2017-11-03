'''
@author: Gaio
@summary: sentiment analysis with classification models
'''

#Start Graphlab create
import graphlab

#read product review data
products = graphlab.SFrame('amazon_baby.gl/')

#explore data
products.head()

#build the wordcount vector 
products['word_count']=graphlab.text_analytics.count_words(products['review'])
products.head()

#graphlab.canvas.set_target('ipynb')
products['name'].show()

#explore the most reviewed product
popy_reviews=products[products['name']=="Vulli Sophie the Giraffe Teether"]
len(popy_reviews)
popy_reviews['rating'].show(view='Categorical')

#build the sentiment classifier

#define what positive & negative sentiment is
#ignore 3* reviews
products = products[products['rating']!=3]

#positive sentiment is 4* or 5* reviews
products['sentiment'] = products['rating'] >= 4

#train the sentiment classifier

#split the data
train_data, test_data = products.random_split(.8,seed=0)
#create & run the model
sentiment_nodel = graphlab.logistic_classifier.create(train_data,target='sentiment',
                                                      features=['word_count'],
                                                      validation_set=test_data)

#evaluate the model
sentiment_nodel.evaluate(test_data,metric='roc_curve')

sentiment_nodel.show(view='Evaluation')

#analyze sentiment for the popy product
popy_reviews['predicted_sentiment']=sentiment_nodel.predict(popy_reviews,output_type='probability')
popy_reviews.head()

#sort reviews based on predicted sentiment
popy_reviews= popy_reviews.sort('predicted_sentiment',ascending = False)
popy_reviews.head()
popy_reviews[0]['review']#most positive review
popy_reviews[-1]['review']#most negative review

#assignment

#select subset of words to count
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

#v1 basic function
def awesome_count():
    if 'awesome' in products['word_count']:
        return return products['word_count']['awesome']
    else:
        return 0
    
products['awesome']= products['word_count'].apply(awesome_count, skip_undefined=True)

#v2
#lambda x: x['awesome'] if 'awesome' in x else 0L
products['awesome'] = products['word_count'].apply(lambda x: x['awesome'] if 'awesome' in x else 0L)

wordToCount='awesome'
products[wordToCount] = products['word_count'].apply(lambda x: x[wordToCount] if wordToCount in x else 0L)

#v3 loop on words
for wordToCount in selected_words:
    products[wordToCount] = products['word_count'].apply(lambda x: x[wordToCount] if wordToCount in x else 0L)
    
#part 1
for wordToCount in selected_words:
    print wordToCount," : ",products[wordToCount].sum()

train_data,test_data = products.random_split(.8, seed=0)

sentiment_model_sel = graphlab.logistic_classifier.create(train_data,target='sentiment',
                                                      features=selected_words,
                                                      validation_set=test_data)
#part 2
sentiment_model_sel['coefficients'].sort('value',ascending = False)

#part 3
sentiment_nodel.evaluate(test_data,metric='roc_curve')
sentiment_nodel.show(view='Evaluation')
sentiment_model_sel.evaluate(test_data,metric='roc_curve')
sentiment_model_sel.show(view='Evaluation')

#part 4
diaper_champ_reviews = products[products['name']=="Baby Trend Diaper Champ"]
len(diaper_champ_reviews)
diaper_champ_reviews['rating'].show(view='Categorical')

#analyze sentiment for the  product
diaper_champ_reviews['predicted_sentiment']=sentiment_nodel.predict(diaper_champ_reviews,output_type='probability')


#sort reviews based on predicted sentiment
diaper_champ_reviews= diaper_champ_reviews.sort('predicted_sentiment',ascending = False)
diaper_champ_reviews.head()#0.999999937267
diaper_champ_reviews[0]['review']#most positive review
diaper_champ_reviews[-1]['review']#most negative review

sentiment_model_sel.predict(diaper_champ_reviews[0:1], output_type='probability') #[0.796940851290673]