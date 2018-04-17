Project: 
Nerual Collaborative Filtering based Recommender System.

Test Environment:
Google Cloud

Dependency:
Please make sure you have installed following python package before install
Keras => version 2.1.5
TensorFlow => version 1.7.0
Anaconda2 for Python 2.7
h5py => version 2.7.1

Project Folder:

./*.png => these are image to visualize model structure, we created using keras.

./Data => this folder is to store the user-item ratings produced by our data
pre-processing program. it contains three file: 
	test.rating => leave one out test sets
	test.negative => item with no interaction with users
	train.rating => training sets

./Model_saved => this folder is for store the model we have trained. There is a
default model we have trained to recommend songs, please don't change it.

./output => this folder contains recommendation list of items for a specific
user, for example, "top500_recommend_for52.txt". 
If you want to trained a new model, the hit-rate and ndcg of every epoch are
also be stored in here, for example, "ncf_train_statics_1epoch.csv".

./spotify_data => please put the spotify data in this folder, and make sure the
files like "mpd.slice.000-999.json" are put in ./spotify_data/data/

./visual_model => this folder contains code to visualize using tensorboard. If
you are interesting to visualize the whole model, you can entering the folder
visual_model, and type following commands:
	python visualization.py
	tensorboard --local=./

Project Core Code:
./data_process.py => this python file contain methods that convert
the json file to our actual rating file, then store them into ./Data/

./produce_graph.py => main function to do data process

./neural_recsys.py => this python file contains the core code of building the
whole structure of neural collborative filtering.

./main_recommend.py => this file to use the trained model and recommend 500
trackes for target playlist

Make command:

# this command will produce these rating file I have mentioned, and inside of
# produce_graph.py, you can change the number of playlists you want to process.
# but please make sure you have the enough spotify data for it to process
# 
make data_process 

# this command will train our whole model depends on the rating files 
make train
 
# it will recommend the songs and store them into ./output/
make recommend

# the default of make is to do recommend

Parameters:
All the parameter, like number of layers, learning rate is changable. they are
in the main() function. I didn't use the sys.argv to pass value here, because
there are lots of parameters. If you want to change parameter, please make sure
them have the same dimension.

Contact Info:

If you have any extra questions, please contact juluo@umich.edu. 
