'''
@author: Gaio
@summary: Image Retrieval using Deep Learning
'''

#Start Graphlab create
import graphlab

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

#load images data set
image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')

#create the nearest neighbor model for docs retrieval
knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],label='id')

#use the model to find similar images
#choose a particular image
graphlab.canvas.set_target('ipynb')
cat=image_train[18:19]
#find the nearest neighbours
knn_model.query(cat)

def get_img_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')

cat_neighbours = get_img_from_ids(knn_model.query(cat))
cat_neighbours['image'].show()

car=image_train[8:9]
get_img_from_ids(knn_model.query(car))['image'].show()

show_neighbors = lambda i: get_img_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()

show_neighbors(8)

#assignment

#question 1
train_sketch = image_train['label'].sketch_summary() #1 - bird

#queestion 2 
#create sub-sets
dog_train = image_train[image_train['label']=='dog']
cat_train = image_train[image_train['label']=='cat']
bird_train = image_train[image_train['label']=='bird']
auto_train = image_train[image_train['label']=='automobile']
#create models
dog_model = graphlab.nearest_neighbors.create(dog_train,features=['deep_features'],label='id')
cat_model = graphlab.nearest_neighbors.create(cat_train,features=['deep_features'],label='id')
bird_model = graphlab.nearest_neighbors.create(bird_train,features=['deep_features'],label='id')
auto_model = graphlab.nearest_neighbors.create(auto_train,features=['deep_features'],label='id')

#define function
def get_img_from_data_ids(data_set,query_result):
    return data_set.filter_by(query_result['reference_label'],'id')

#find the nearest cat
test_cat=image_test[0:1]
test_cat['image'].show()

nearest_cat = get_img_from_data_ids(cat_train,cat_model.query(test_cat))
nearest_cat['image'].show()#last cat

#question 3 nearest dog to the 1st cat
nearest_dog_cat = get_img_from_data_ids(image_train,dog_model.query(test_cat))
nearest_dog_cat['image'].show()#4th dog

#questions 4-5
mn_dist_cat = cat_model.query(test_cat)['distance'].mean()#36.15 - 35 to 37
mn_dist_dog = dog_model.query(test_cat)['distance'].mean()#37.77 - 37 to 39

#questions 6 - cat

image_test_dog = image_test[image_test['label']=='dog']
image_test_cat = image_test[image_test['label']=='cat']
image_test_bird = image_test[image_test['label']=='bird']
image_test_auto = image_test[image_test['label']=='automobile']

dog_dog_neighbors = dog_model.query(image_test_dog, k=1)
dog_cat_neighbors = cat_model.query(image_test_dog, k=1)
dog_bird_neighbors = bird_model.query(image_test_dog, k=1)
dog_auto_neighbors = auto_model.query(image_test_dog, k=1)

dog_distances = graphlab.SFrame({'dog-dog': dog_dog_neighbors['distance'],'dog-cat': dog_cat_neighbors['distance'],'dog-bird':dog_bird_neighbors['distance'],'dog-auto':dog_auto_neighbors['distance']})

def is_dog_correct(row):
    if min(row['dog-dog'],row['dog-cat'],row['dog-bird'],row['dog-auto']) == row['dog-dog']:
        return 1
    else:
        return 0

dog_distances.apply(is_dog_correct).sum() #678/100 - 60 to 70
