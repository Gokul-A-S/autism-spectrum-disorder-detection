import os.path
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
import numpy as np
import tensorflow as tf

def predict(path):
    print(tf.__version__)
    data=preprocess(path)
    print(os.path.curdir)
    # model = tf.saved_model.load('model')
    # serialize model to JSON


    # later...

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    data=np.array(data)
    print(data.shape)
    print(loaded_model.input_shape)
    # reshaped_data = data.reshape((data.shape[0], 1))
    data = data.reshape(-1, 2000, 1)
    return loaded_model.predict(data)

def preprocess(path):
    res = []
    input_file = np.loadtxt(path)
    indices = np.loadtxt("indices.txt")

    # Initialize ConnectivityMeasure object
    conn_est = ConnectivityMeasure(kind='correlation', discard_diagonal=True)  # Connectivity Estimator

    # List to store connectivity matrices for each subject

    # Compute connectivity matrices for each subject

    # Compute connectivity matrix for the current subject
    conn_matrix = conn_est.fit_transform([input_file])[0]
    data=sym_matrix_to_vec(conn_matrix, discard_diagonal=True)
    # Append the computed connectivity matrix to the list

    # Print the connectivity matrix of the first subject as an example
    print(conn_matrix)
    print(data.shape)
    for i in indices:
        res.append(data[int(i)])
    print(res)
    return res