#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestClassifier
from models import *
from get_feature import *
import pickle as pk

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')


    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    classes = ['Present', 'Unknown', 'Absent']
    num_classes = len(classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features1 = get_features(data_folder, patient_files)
    model = get_toy((100, 313, 1))

    model.compile(optimizer = "adam", 
             loss = "categorical_crossentropy",
             metrics = "accuracy")    
    

    # Train the model.
    if verbose >= 1:
        print('Training model...')
    imputer = SimpleImputer().fit(features1[0]['hw'])
    features1[0]['hw'] = imputer.transform(features1[0]['hw'])
    model.fit([features1[0]['age'],features1[0]['sex'], features1[0]['hw'], features1[0]['preg'], features1[0]['loc'], 
           features1[0]['mel1']], features1[1],
          epochs = 30)    
    
    
    # Define parameters for random forest classifier.
#    n_estimators = 10    # Number of trees in the forest.
#    max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
#    random_state = 123   # Random state; set for reproducibility.

#    imputer = SimpleImputer().fit(features)
#    features = imputer.transform(features)
#    classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # Save the model.
#    save_challenge_model(model_folder, classes, imputer, model)
    save_challenge_model(model_folder, model, classes, m_name = 'toy', mel_shape = (100, 313, 1) )
    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.

#def load_challenge_model(model_folder, verbose):
#    filename = os.path.join(model_folder, 'model.sav')
#    return joblib.load(filename)

def load_challenge_model(model_folder, verbose):
    info_fnm = os.path.join(model_folder, 'desc.pk')
    with open(info_fnm, 'rb') as f:
        info_m = pk.load(f)
#    if info_m['model'] == 'toy' :
#        model = get_toy(info_m['mel_shape'])
#    filename = os.path.join(model_folder, info_m['model'] + '_model.hdf5')
#    model.load_weights(filename)
    return info_m


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.

def run_challenge_model(model, data, recordings, verbose):
    
    if model['model'] == 'toy' :
        model1 = get_toy(model['mel_shape'])
    model1.load_weights(model['model_fnm'])
    
    classes = model['classes']
    # Load features.
    features = get_feature_one(data, verbose = 0)

    features['mel1'] = []
    for i in range(len(recordings)) :
        mel1 = feature_extract_melspec(recordings[i])[0]
        features['mel1'].append(mel1)

    M, N = features['mel1'][0].shape
    for i in range(len(features['mel1'])) :
        features['mel1'][i] = features['mel1'][i].reshape(M,N,1)   
        
    features['mel1'] = np.array(features['mel1'])
#    print(features)
    # Impute missing data.
    res1 = model1.predict([features['age'], features['sex'], features['hw'], features['preg'], features['loc'], features['mel1']])

    # Get classifier probabilities.
    prob1 = res1.mean(axis = 0) ## simple rule for now
    idx = np.argmax(prob1)
    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    labels[idx] = 1

    return classes, labels, prob1


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
#def save_challenge_model(model_folder, classes, imputer, classifier):
#    d = {'classes': classes, 'imputer': imputer, 'classifier': classifier}
#    filename = os.path.join(model_folder, 'model.sav')
#    joblib.dump(d, filename, protocol=0)
    
def save_challenge_model(model_folder, model, classes, m_name, mel_shape = (100, 313, 1)) :
    os.makedirs(model_folder, exist_ok=True)
    info_fnm = os.path.join(model_folder, 'desc.pk')
    filename = os.path.join(model_folder, m_name + '_model.hdf5')
    model.save(filename)
    d = {'model': m_name, 'classes': classes, 'mel_shape': mel_shape, 'model_fnm': filename}    
    with open(info_fnm, 'wb') as f:
        pk.dump(d, f, pk.HIGHEST_PROTOCOL)

        
        
