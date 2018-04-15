import numpy as np
import keras
import sys
import scipy.sparse as sp
import math
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



def build_mlp_model(num_user, num_item, latent_v_dim=8, 
                dense_layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], reg_mf=0
                ):
    # Input layer
    input_user = Input(shape=(1,), dtype='int32', name='user_input')
    input_item = Input(shape=(1,), dtype='int32', name='item_input')
    
    # Embedding layer
    mlp_user_embedding = Embedding(input_dim=num_user, output_dim=dense_layers[0]/2,
                         name='mlp_user_embedding',
                         embeddings_initializer='RandomNormal',
                         embeddings_regularizer=l2(reg_layers[0]), 
                         input_length=1)
    mlp_item_embedding = Embedding(input_dim=num_item, output_dim=dense_layers[0]/2,
                         name='mlp_item_embedding',
                         embeddings_initializer='RandomNormal',
                         embeddings_regularizer=l2(reg_layers[0]), 
                         input_length=1)


    # Multi layer perceptron latent vector
    mlp_user_latent = Flatten()(mlp_user_embedding(input_user))
    mlp_item_latent = Flatten()(mlp_item_embedding(input_item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])
    
    mlp_vector = mlp_cat_latent
    # Build dense layer for model
    for i in range(1,len(dense_layers)):
        layer = Dense(dense_layers[i],
                      activity_regularizer=l2(reg_layers[i]),
                      activation='relu',
                      name='layer%d' % i)
        mlp_vector = layer(mlp_vector)

    result = Dense(1, activation='sigmoid', 
                   kernel_initializer='lecun_uniform',name='result')

    model = Model(input=[input_user,input_item], output=result(mlp_vector))

    return model


def build_gmf_model(num_user, num_item, latent_v_dim=8, 
                dense_layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], reg_mf=0
                ):

    # Input layer
    input_user = Input(shape=(1,), dtype='int32', name='user_input')
    input_item = Input(shape=(1,), dtype='int32', name='item_input')
    
    # Embedding layer
    mf_user_embedding = Embedding(input_dim=num_user, output_dim=latent_v_dim,
                        name='mf_user_embedding',
                        embeddings_initializer='RandomNormal',
                        embeddings_regularizer=l2(reg_mf), input_length=1)
    mf_item_embedding = Embedding(input_dim=num_item, output_dim=latent_v_dim,
                        name='mf_item_embedding',
                        embeddings_initializer='RandomNormal',
                        embeddings_regularizer=l2(reg_mf), input_length=1)

    mf_user_latent = Flatten()(mf_user_embedding(input_user))
    mf_item_latent = Flatten()(mf_item_embedding(input_item))
    mf_merge_latent = Multiply()([mf_user_latent, mf_item_latent])

    result = Dense(1, activation='sigmoid', 
                   kernel_initializer='lecun_uniform',name='result')

    model = Model(input=[input_user,input_item], output=result(mf_merge_latent))

    return model

# Built the Nerual Collaborative Filtering Model:
def build_ncf_model(num_user, num_item, latent_v_dim=8, 
                dense_layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0], reg_mf=0
                ):

    # Input layer
    input_user = Input(shape=(1,), dtype='int32', name='user_input')
    input_item = Input(shape=(1,), dtype='int32', name='item_input')
    
    # Embedding layer
    mf_user_embedding = Embedding(input_dim=num_user, output_dim=latent_v_dim,
                        name='mf_user_embedding',
                        embeddings_initializer='RandomNormal',
                        embeddings_regularizer=l2(reg_mf), input_length=1)
    mf_item_embedding = Embedding(input_dim=num_item, output_dim=latent_v_dim,
                        name='mf_item_embedding',
                        embeddings_initializer='RandomNormal',
                        embeddings_regularizer=l2(reg_mf), input_length=1)
    mlp_user_embedding = Embedding(input_dim=num_user, output_dim=dense_layers[0]/2,
                         name='mlp_user_embedding',
                         embeddings_initializer='RandomNormal',
                         embeddings_regularizer=l2(reg_layers[0]), 
                         input_length=1)
    mlp_item_embedding = Embedding(input_dim=num_item, output_dim=dense_layers[0]/2,
                         name='mlp_item_embedding',
                         embeddings_initializer='RandomNormal',
                         embeddings_regularizer=l2(reg_layers[0]), 
                         input_length=1)

    # Matrix Factorization latent vector
    mf_user_latent = Flatten()(mf_user_embedding(input_user))
    mf_item_latent = Flatten()(mf_item_embedding(input_item))
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    # Multi layer perceptron latent vector
    mlp_user_latent = Flatten()(mlp_user_embedding(input_user))
    mlp_item_latent = Flatten()(mlp_item_embedding(input_item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])
    
    mlp_vector = mlp_cat_latent
    # Build dense layer for model
    for i in range(1,len(dense_layers)):
        layer = Dense(dense_layers[i],
                      activity_regularizer=l2(reg_layers[i]),
                      activation='relu',
                      name='layer%d' % i)
        mlp_vector = layer(mlp_vector)

    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])
    result = Dense(1, activation='sigmoid', 
                   kernel_initializer='lecun_uniform',name='result')

    model = Model(input=[input_user,input_item], output=result(predict_layer))

    return model


def load_pretrain_model():
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('mf_item_embedding').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def load_rating_file_as_matrix(filename):
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()    
    return mat

def load_rating_file_as_list(filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList

def load_data(data_path='./Data', num_neg_sample=4):
    # return user input, iterm input, labels
    test_rating_file = "test.rating"
    train_rating_file = "train.rating"
    test_neg_file = "test.negative" 
    print("Loading Data ...")
    t1 = time()
    train_matrix = load_rating_file_as_matrix(data_path + train_rating_file)
    test_pos_list = load_rating_file_as_list(data_path + test_rating_file)
    test_neg_list = load_negative_file(data_path + test_neg_file)
    num_user, num_item = train_matrix.shape
    print("Complete Load Data: %0.1f s" % (time()-t1))
    print("#user = %d, #item = %d, #train = %d, #test = %d" % (num_user, num_item,
          train_matrix.nnz, len(test_pos_list)))
    return train_matrix, test_pos_list, test_neg_list

def get_train_samples(train_matrix, num_neg_sample):
    user_input, item_input, labels = [], [], []
    num_user, num_item = train_matrix.shape
    for (u, i) in train_matrix.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_neg_sample):
            j = np.random.randint(num_item)
            while train_matrix.has_key((u, j)):
                j = np.random.randint(num_item)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def get_HR(model, test_pos_list, test_neg_list, topN):
    hits = 0
    for idx,leave_one in enumerate(test_pos_list):
        results = []
        num_users = len(test_neg_list[0])+1
        items = list(test_neg_list[idx])
        items.append(leave_one[1])
        users = np.full(num_users, leave_one[0], dtype='int32')
        results = model.predict([users,np.array(items)],
                batch_size=100, verbose=0)
        item_map = {}
        for i in range(len(items)):
            item_map[items[i]] = results[i]
        
        topN_item = [item_tuple[0] for item_tuple in sorted(item_map.items(), key=lambda x:
            x[1], reverse=True)[0:topN]]
        if leave_one[1] in topN_item:
            hits += 1.0

    return hits/len(test_pos_list)

def get_NDCG(model, test_pos_list, test_neg_list, topN):
    ndcgs = []
    for idx,leave_one in enumerate(test_pos_list):
        results = []
        num_users = len(test_neg_list[0])+1
        items = list(test_neg_list[idx])
        items.append(leave_one[1])
        users = np.full(num_users, leave_one[0], dtype='int32')
        results = model.predict([users,np.array(items)],
                batch_size=100, verbose=0)
        item_map = {}
        for i in range(len(items)):
            item_map[items[i]] = results[i]
        
        topN_item = [item_tuple[0] for item_tuple in sorted(item_map.items(), key=lambda x:
            x[1], reverse=True)[0:topN]]
        if leave_one[1] not in topN_item:
            temp = 0
        else:
            for x in range(len(topN_item)):
                if leave_one[1]==topN_item[x]:
                    temp = math.log(2)/math.log(x+2)
        ndcgs.append(temp)

    return np.array(ndcgs).mean()


def main(method):
    
    print 'Choose Model ======> %s' % method
    print 'Start Building and Training'
    model_path = './Model_saved/'
    model_weight_path = './Model_saved/'
    output_path = './output/'
    num_epochs = 30
    batch_size = 256
    latent_v_dim = 8
    dense_layers = [64, 32, 16, 8]
    reg_layers = [0, 0, 0, 0]
    reg_mf = [0]
    num_neg_sample = 4
    learning_rate = 0.001
    learner = 'adam'
    verbose = 1
    
    train_matrix, test_pos_list, test_neg_list = load_data(data_path='./Data/')
    user_input, item_input, labels = get_train_samples(train_matrix,
                                                       num_neg_sample)
    num_user, num_item = train_matrix.shape

    if method == 'NCF':
        model = build_ncf_model(num_user, num_item, latent_v_dim, dense_layers,
            reg_layers, reg_mf)
        #plot_model(model, to_file='ncf_model.png')
    elif method == 'GMF':
        model = build_gmf_model(num_user, num_item, latent_v_dim, dense_layers,
            reg_layers, reg_mf)
        #plot_model(model, to_file='gmf_model.png')
    elif method == 'MLP':
        model = build_mlp_model(num_user, num_item, latent_v_dim, dense_layers,
            reg_layers, reg_mf)
        #plot_model(model, to_file='mlp_model.png')

    
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate),
                loss='binary_crossentropy')
  

    # Train model
    topN = 10
    hit_rate = get_HR(model, test_pos_list, test_neg_list, topN)
    ndcg = get_NDCG(model, test_pos_list, test_neg_list, topN)
    best_hr, best_ndcg = hit_rate, ndcg
    dataset = 'spotify'
    if method == 'NCF':
        model_save_file = model_weight_path + ('%s_NCF_%d_%s_%0.6f.h5' % (dataset, latent_v_dim,
                                                         str(dense_layers),
                                                         best_hr))

    elif method == 'GMF':
        model_save_file = model_weight_path + ('%s_GMF_%d_%s_%0.6f.h5' % (dataset, latent_v_dim,
                                                         str(dense_layers),
                                                         best_hr))
    elif method == 'MLP':
        model_save_file = model_weight_path + ('%s_MLP_%d_%s_%0.6f.h5' % (dataset, latent_v_dim,
                                                         str(dense_layers),
                                                         best_hr))

    csv_hr = []
    csv_ndcg = []
    csv_epoch = []
    csv_loss = []
    csv_time = []
    for epoch in range(num_epochs):
        t1 = time()
        user_input, item_input, labels = get_train_samples(train_matrix,
                                                       num_neg_sample)
        
        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=1,
                         shuffle=True)
        t2 = time()
        hit_rate = get_HR(model, test_pos_list, test_neg_list, topN)
        ndcg = get_NDCG(model, test_pos_list, test_neg_list, topN)
        print("Epoch %d with %.1f s: HR = %0.4f, NDCG = %0.4f, loss = %.4f" % (epoch,
            t2-t1, hit_rate, ndcg, hist.history['loss'][0]))

        if hit_rate > best_hr:
            best_hr, best_ndcg,best_epoch = hit_rate, ndcg, epoch
            if method == 'NCF':
                model_save_file = model_weight_path + ('%s_NCF_%d_%s_%d.h5' % (dataset, latent_v_dim,
                                                         str(dense_layers),
                                                         num_epochs))
            elif method == 'GMF':
                model_save_file = model_weight_path + ('%s_GMF_%d_%s_%d.h5' % (dataset, latent_v_dim,
                                                         str(dense_layers),
                                                         num_epochs))
            elif method == 'MLP':
                model_save_file = model_weight_path + ('%s_MLP_%d_%s_%d.h5' % (dataset, latent_v_dim,
                                                         str(dense_layers),
                                                         num_epochs))
            print 'Save Model back to %s' % model_save_file
            model.save(model_save_file, overwrite=True)


        csv_hr.append('%0.4f' % hit_rate)
        csv_loss.append('%0.4f' % hist.history['loss'][0])
        csv_ndcg.append('%0.4f' % ndcg)
        csv_time.append('%0.4f' % (t2-t1))
        csv_epoch.append(str(epoch))
        
    if method == 'NCF':
        output_file = output_path + 'ncf_train_statics_%depoch.csv' % num_epochs
    elif method == 'GMF':
        output_file = output_path + 'gmf_train_statics_%depoch.csv' % num_epochs
    elif method == 'MLP':
        output_file = output_path + 'mlp_train_statics_%depoch.csv' % num_epochs
    print 'Save training statis back to %s' % output_file
    with open(output_file, 'w') as f:
        ret = 'epoch,' + 'hit_rate,' + 'ndcg,' + 'loss,' + 'time\n' 
        f.write(ret)
        for idx in range(len(csv_hr)):
            ret = csv_epoch[idx] + ',' + csv_hr[idx] + ',' + csv_ndcg[idx] + \
            ',' + csv_loss[idx] + \
            ',' + csv_time[idx] + '\n'
            f.write(ret)
    f.close()

if __name__ == '__main__':
    
    main('NCF')
    main('GMF')
    main('MLP')
