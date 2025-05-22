import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams

from utilities.model_training_functions import *
from utilities.TFModels import *

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
plt.style.use("ggplot")


if __name__ == "__main__":
    #tf.random.set_seed(42) # Select the random number generator to ensure reproducibility of the results
    #s elect_GPU(0) # Select the GPU to be used
    subject = 1 # Select the subjectt for training
    tested_on = None # Select the subjectt for testing. if none will test on train subject
    subject = f"{subject:02d}"
    ######################### Model I/O #########################
    features = ["MAV", "RMS"] # Select features
    features.extend([f"AR{i+1}" for i in range(6)])
    sensors = [f"sensor {i+1}" for i in range(7)] # select sensors numbers (1~14)
    out_labels = ['Right knee angle']
    input_width = 20
    shift = 1
    label_width = 1
    batch_size = 8
    #################### Models names functions ####################
    # If you create new model, just give it a name and pass the function to the dictionary
    model_name = 'LSTM'
    models_dic = {model_name: create_lstm_model}
    ############ Create pd dataframe to hold the results ############
    r2_results = pd.DataFrame(columns=out_labels)
    rmse_results = pd.DataFrame(columns=out_labels)
    nrmse_results = pd.DataFrame(columns=out_labels)
    
    ################################################
    for subject in range(1, 2):
        subject_test = f"{subject:02d}"
        y_true, y_pred = inference(
            subject=subject_test,
            model_name=model_name,
            models_dic=models_dic,
            input_width=20,
            shift=1,
            label_width=1,
            lr=0.001,
            features=features,
            sensors=sensors,
            out_labels=out_labels,
            batch_size=8
        )
