import json
import re
from typing import *
from warnings import simplefilter

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg

simplefilter(action='ignore', category=FutureWarning)
# Set constant to be used by other modules.
WINDOW_LENGTH = 0.2  # The length of the sliding window in seconds
SLIDING_WINDOW_STRIDE = 0.05  # Sliding window stride in seconds


def get_emg_files(subject: str, trials: List[str]) -> List[List[str]]:
    """This function will return I/O directories stored in two lists.
    """
    # Get I/O data folders
    inputs_path = "E:\Spring25\Capstone\emg1.csv"
    outputs_path = "E:\Spring25\Capstone\\re_ver\emg_post_process_2026"
    # Get inputs directories
    inputs_names = ["E:\Spring25\Capstone\\re_ver\minh_kun_data_only_raw.csv"]
    # Get outputs directories
    output_files = list(
        map(lambda x: f"{outputs_path}\\{x}_final_features.csv", trials))
    return inputs_names, output_files


def load_emg_data(subject: str, emg_file: str) -> pd.DataFrame:

    # Load delsys data
    delsys_data = pd.read_csv(emg_file, header=0)
    # Rename time column.
    # Note that delsys will have atime column for each s
    # ensor input, so set the regex

    # Set time column as the index
    delsys_data.set_index("time", inplace=True)
    # Drop all columns that does not have the string EMG on it (Acc, gyro and X[s]).
    # The time column is set to the index so it will not be removed
    emg = delsys_data
    # Subtract mean only from EMG columns, excluding 'time' and angle columns
    columns_to_skip = ["time", "angle"]
    columns_to_normalize = [col for col in emg.columns if col not in columns_to_skip]

    # Normalize only selected columns
    emg[columns_to_normalize] = emg[columns_to_normalize] - emg[columns_to_normalize].mean()
    # reset time
    return emg


def remove_outlier(data: pd.DataFrame, detect_factor: float = 10, remove_factor: float = 20) -> pd.DataFrame:
    # loop in each column
    for col in data.columns:
        column_data = data.loc[:, col]
        # Find artifact trigger by multiplying the mean of the absolute value of the data by a detecto
        detector = column_data.apply(np.abs).mean()*detect_factor
        # If the absolute value of a data point is above the tigger, divided by remove factor
        # apply method is very slow
        data.loc[:, col] = column_data.apply(
            lambda x: x if np.abs(x) < detector else x/remove_factor)
    print(data.shape)
    return data


def segmant(emg: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return emg.loc[(emg.index >= start) & (emg.index <= end), :]


def emg_filter(window: pd.DataFrame):
    return notch_filter(bandpass_filter(window))


def bandpass_filter(window: Union[pd.DataFrame, np.ndarray], order: int = 4,
                    lowband: int = 20, highband: int = 450) -> np.ndarray:
    fs = 1/0.001  # Hz
    low_pass = lowband/(fs*0.5)
    hig_pass = highband/(fs*0.5)
    b, a = signal.butter(N=order, Wn=[low_pass, hig_pass], btype="bandpass")
    return signal.filtfilt(b, a, window, axis=0)


def notch_filter(window: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    fs = 1/0.001  # sampling frequancy in Hz
    f0 = 50  # Notched frequancy
    Q = 30  # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, window, axis=0)


def get_single_window_features(filtered_window: np.ndarray, features_names: List[str], ar_order: int) -> np.ndarray:

    features_dict: Dict[str, function] = {"RMS": get_RMS,
                                          "MAV": get_MAV,
                                          "WL": wave_length,
                                          "ZC": get_ZC}

    features_function = [features_dict[feature]
                         for feature in features_names if "AR" not in feature]

    features = np.vstack([foo(filtered_window)
                         for foo in features_function]).transpose()

    if ar_order:
        features = np.hstack(
            (features, get_AR_coeffs(filtered_window, ar_order)))
        
    
    return features.flatten()


def get_ZC(window: np.ndarray) -> np.ndarray:
    # returns the indexes of where ZC appear return a tuple with length of 1
    ZC = [len(np.where(np.diff(np.signbit(window[:, col])))[0])
          for col in range(np.shape(window)[-1])]
    return np.array(ZC)


def get_RMS(window: np.ndarray) -> np.ndarray:
    return np.sqrt(sum(n*n for n in window)/len(window))


def get_MAV(window: np.ndarray) -> np.ndarray:
    return sum(abs(n) for n in window)/len(window)


def wave_length(window: np.ndarray) -> np.ndarray:
    return np.sum(abs(np.diff(window, axis=0)), axis=0)


def get_AR_coeffs(window: np.ndarray, num_coeff=6) -> np.ndarray:
    first = True
    for col in range(np.shape(window)[-1]):
        model = AutoReg(window[:, col], lags=num_coeff, old_names=True)
        model_fit = model.fit()
        if first:
            parameters = model_fit.params[1:]
            first = False
        else:
            parameters = np.vstack((parameters, model_fit.params[1:]))
    return parameters


def process_emg(emg, features_names: List[str] = ["RMS", "MAV", "WL", "ZC"],
                ar_order: int = 4, use_DEMG: bool = True) -> pd.DataFrame:
    """
    This function will segmant the data, filter it and apply features extraction methods to return the dataset.
    """
    # Create a copy of the features names. This is necessary if the function will be called multiple times.
    features_copy_list = features_names.copy()
    angle_cols = ["angle"]
    emg_cols = [col for col in emg.columns if col not in angle_cols]
    emg_cols = [col for col in emg_cols if col not in ['time']]
    # Add AR to the features list
    if ar_order > 0:
        features_copy_list.extend([f"AR{i}" for i in range(1, ar_order+1)])
    # Create the df columns from the features_names
    df_col = []
    for emg_num in range(1, len(emg_cols)+1):
        df_col.extend([f"sensor {emg_num} {f}" for f in features_copy_list])
    df_col.extend(angle_cols)
    # initialize the features df
    dataset = pd.DataFrame(columns=df_col)
    print(dataset.shape)
    # Create the first sliding window and mark it's ending position
    start = 0
    end = WINDOW_LENGTH
    time_limit = max(emg.index)
    # Show the user the number of seconds will be converted
    print(f"time_limit: {time_limit}s")
    # start looping using the sliding window
    while end <= time_limit:
        # Segmant the data
        print(emg_cols)
        window= segmant(emg[emg_cols], start, end)
        window_angle = segmant(emg[angle_cols], start, end)
        # Filter segmanted data
        window = emg_filter(window)
        # differentiation the signal if required
        if use_DEMG:
            window = np.diff(window, axis=0)/0.0009
        # Get features
        window = window / np.abs(window).max()
        features = get_single_window_features(
            window, features_copy_list, ar_order)
        # Update data frame
        print(features.shape)
        mean_angles = window_angle.mean().values
        combined_row = np.concatenate([features, mean_angles])
        #print(combined_row.shape)
        dataset.loc[len(dataset)] = combined_row
        # Increment the sliding window by the stride value
        start += SLIDING_WINDOW_STRIDE
        end += SLIDING_WINDOW_STRIDE
    # Create the time column
    dataset['time'] = [np.around(SLIDING_WINDOW_STRIDE*i + WINDOW_LENGTH, 3)
                       for i in range(len(dataset))]
    # Set the time column as the index
    dataset.set_index("time", inplace=True)
    return dataset


def emg_to_features(subject: Union[str, int], trials: List[str],
                    features_names: List[str] = ["RMS", "MAV", "WL", "ZC"],
                    ar_order: int = 4, use_DEMG: bool = True):
    # Get inputs and outputs directories
    try:
        subject = f"{int(subject):02d}"
    except:
        raise 'Subject variable should be a number'
    inputs_names, output_files = get_emg_files(subject, trials)

    for emg_file, output_file, trial in zip(inputs_names, output_files, trials):
        # Load data
        print(f"Loading {trial} data...")
        print(emg_file)
        print(output_file)  
        emg = load_emg_data(subject, emg_file)
        # Remove artifacts
        emg = remove_outlier(emg, detect_factor=10,
                             remove_factor=20)  # Un-efficient method
        # preprocess the data to get features
        dataset = process_emg(emg, features_names, ar_order, use_DEMG)
        # save dataset
        print(dataset.shape)
        dataset.to_csv(output_file)


if __name__ == "__main__":
    ########################### prepare setups ###########################
    trials = ["trial", ]
    # Set the features names
    features_names = ["RMS", "MAV"]
    # set ordewr to zero if you do not want AR feature
    ar_order = 6
    # Use DEMG or sEMG features
    use_DEMG = True
    # If set to false, features will be extracted from the filtered signal (sEMG)

    # Select the subject number. You cn create a 'for' loop to loop through multiple subjects
    # subject = input("Please input subject number: ")
    for subject in [1,]:
        subject = f"{int(subject):02d}"
        print(subject)
        ########################### Convert ###########################
    
        emg_to_features(subject, trials, features_names, ar_order, use_DEMG)

'''
        try:
            # If all subject data files exisit, the dataset will be automatically generated
            from 04_Dataset_gener import *
            get_dataset(subject, trials=trials)
            print("Dataset file been updated successfully.")
        except:
            pass
'''