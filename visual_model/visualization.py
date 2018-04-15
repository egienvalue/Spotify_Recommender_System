from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils import plot_model
from keras import backend as K
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd 
import numpy as np
import os
import tensorflow
ROOT_DIR = './'
model_name = './model.json' 


OUTPUT_MODEL_FILE_NAME = os.path.join(ROOT_DIR,'tf.ckpt')

# get the keras model

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./ml-1m_MLP_[64,32,16,8]_1523494203.h5")
#plot_model(model, to_file='./model.png')
# get the tensor name from the embedding layer
print type(iter(model.layers))

#tensor_name = next(iter(filter(lambda x: x.name == 'user_embedding',
#    model.layers))).weights.name
tensor_name = "mlp_embedding"
# the vocabulary
metadata_file_name = os.path.join(ROOT_DIR,tensor_name)

#embedding_df = get_embedding()
embedding_weights = (next(iter(filter(lambda x: x.name == 'user_embedding',
    model.layers))).get_weights())

print embedding_weights[0]
print embedding_weights[0].shape
print type(embedding_weights[0])
#embedding_df.to_csv(metadata_file_name, header=False, columns=[])
df = pd.DataFrame(embedding_weights[0])
print metadata_file_name
df.to_csv(metadata_file_name + ".csv")

saver = tensorflow.train.Saver()
saver.save(K.get_session(), OUTPUT_MODEL_FILE_NAME)

summary_writer = tensorflow.summary.FileWriter(ROOT_DIR)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = tensor_name
embedding.metadata_path = metadata_file_name

projector.visualize_embeddings(summary_writer, config)
