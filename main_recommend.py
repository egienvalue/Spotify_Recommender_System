import numpy as np
import keras
import sys
import scipy.sparse as sp
import math
import json
import os
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.layers import Multiply, Concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from time import time
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.models import load_model
from sklearn.cluster import KMeans

def get_playlists_map(file_prefix, num_play_list):
    if os.path.isfile('play_lists_map.json'):
        play_lists_map = json.load(open('play_lists_map.json', 'r'))
        return play_lists_map
    play_lists_map = {}
    for i in range(num_play_list/1000):
            #mpd.slice.0-999
            filename = file_prefix+"%d-%d" % (i*1000,i*1000+999) + \
                    ".json"
            print "Processing: " + filename
            file_handler = open(filename, "r")
            #print filename 
            data = json.load(file_handler)
            for idx,play_list in enumerate(data["playlists"]):
                #if len(play_list["tracks"]) < 30:
                #    continue
                #print play_list["tracks"][-1]["track_uri"]
                track_uris = []
                for track in play_list["tracks"]:
                    track_uris.append(track["track_uri"])
                play_lists_map[i*1000 + idx] = track_uris
   
    with open('play_lists_map.json', 'w') as f:
        f.write(json.dumps(play_lists_map))
        f.close()

    return play_lists_map

def get_trackuri2num_map(file_prefix, num_play_list):
    if os.path.isfile('track_uri2num_map.json'):
        track_uri2num_map = json.load(open('track_uri2num_map.json', 'r'))
        return track_uri2num_map
    track_uris = []
    track_uri2num_map = {}
    for i in range(num_play_list/1000):
        #mpd.slice.0-999
        filename = file_prefix+"%d-%d" % (i*1000,i*1000+999) + \
                ".json"
        print "Processing: " + filename
        file_handler = open(filename, "r")
        #print filename 
        data = json.load(file_handler)
        for play_list in data["playlists"]:
            #if len(play_list["tracks"]) < 30:
            #    continue
            #print play_list["tracks"][-1]["track_uri"]
            for track in play_list["tracks"]:
                track_uris.append(track["track_uri"])

    track_uris = set(track_uris)
    for idx,track_uri in enumerate(list(track_uris)):
        track_uri2num_map[track_uri] = idx
    with open('track_uri2num_map.json', 'w') as f:
        f.write(json.dumps(track_uri2num_map))
        f.close()
    return track_uri2num_map

def main():

    desired_user_id = int(sys.argv[1])
    num_play_list = 10000
    topK = 500
    model_path = './Model_saved/spotify_NCF_8_[64, 32, 16, 8]_default.h5'
    print('using model : %s' % model_path)
    model = load_model(model_path)
    
    mlp_user_embedding_weights = (next(iter(filter(lambda x: x.name == \
                        'mlp_user_embedding', model.layers))).get_weights())
    
    user_latent_matrix = mlp_user_embedding_weights[0]
    one_user_vector = user_latent_matrix[desired_user_id,:]
    one_user_vector = np.reshape(one_user_vector, (1,32))
    kmeans = KMeans(n_clusters=100, random_state=0, verbose=0).fit(user_latent_matrix)
    desired_user_label = kmeans.predict(one_user_vector)
    user_label = kmeans.labels_
    desired_user_neibors = []
    for user_id,user_label in enumerate(user_label):
        if user_label == desired_user_label:
            desired_user_neibors.append(user_id)


    path = "./spotify_data/data/"
    fileprefix = path + "mpd.slice."
    num_play_list = 10000
    track_set = []
    play_lists_map = get_playlists_map(fileprefix, num_play_list)
    track_uri2num_map = get_trackuri2num_map(fileprefix, num_play_list)
    num2track_uri = {}
    for k,v in track_uri2num_map.items():
        num2track_uri[int(v)] = str(k)
    for user_id in desired_user_neibors:
        track_set += play_lists_map[str(user_id)]
    recommend_track_set = set(track_set)
    
    recommend_track_num = [int(track_uri2num_map[x]) for x in recommend_track_set]
    users = np.full(len(recommend_track_num), desired_user_id, dtype='int32')
    items = np.array(recommend_track_num, dtype='int32')

    results = model.predict([users,items],batch_size=100, verbose=0) 
    results = results.tolist()
    results_t = [(recommend_track_num[idx], x) for idx, x in enumerate(results)]
    results_t = sorted(results_t, key= lambda x : x[1], reverse=True)
    
    topK_rec_track_num = [x[0] for x in results_t[0:topK]]
    topK_rec_track = [num2track_uri[x] for x in topK_rec_track_num]
    rec_output_file = './output/top'+ str(topK)+'_recommend_for_'+str(desired_user_id)+'.txt'
    print('save the results back to: %s' % rec_output_file)
    with open(rec_output_file, 'w') as f:
        for x in topK_rec_track:
            f.write(x + '\n')
        f.close()
    #results = model.predict([users,np.array(items)],
    #            batch_size=100, verbose=0)

if __name__ == "__main__":
    main()
