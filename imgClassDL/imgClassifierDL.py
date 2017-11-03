'''
@author: Gaio
@summary: Image Classifier using Deep Learning
'''

#Start Graphlab create
import graphlab

#load images data set
image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')

#explore the data
graphlab.canvas.set_target('ipynb')
image_train['image'].show()

#train a classifier on the raw image pixels
raw_pixel_model = graphlab.logistic_classifier.create(image_train,target='label',features=['image_array'])
#make a prediction on the 1st 3 imgs
image_test[0:3]['image'].show()
image_test[0:3]['label']
raw_pixel_model.predict(image_test[0:3])#it sucks
#evaluate the model on test data
raw_pixel_model.evaluate(image_test)

#improve model with deep features
dl_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
image_train['deep_features']=dl_model.extract_features(image_train)
dl_features_model=graphlab.logistic_classifier.create(image_train,target='label',features=['deep_features'])
#make a prediction on the 1st 3 imgs
dl_features_model.predict(image_test[0:3])#it rocks
#evaluate the model on test data
dl_features_model.evaluate(image_test)