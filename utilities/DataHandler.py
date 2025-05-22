import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
from typing import *


class DataHandler:
    '''
    A custom class that will make working with data easier.
    '''
    def __init__(self, subject: str, features: List[str], sensors: List[str],
                 out_labels: List[str] = ["Left ankle moment"]):

        # Initiate the subject
        self.subject = subject
        self.features = features
        self.sensors = sensors
        self.out_labels = out_labels

        print(f"Subject: thiss iss handler {self.subject}")
        #print(f"Features: {self.features}")
        #print(f"Sensors: {self.sensors}")
        #print(f"Output labels: {self.out_labels}")
        
        # Load data
        self._data = pd.read_csv(f"E:\\Spring25\\Capstone\\re_ver\\Dataset\\S{subject}_trial_dataset.csv",
                                 index_col="time")
        #print(f"Loaded dataset shape: {self._data.shape}")
        #print(f"Columns in dataset: {self._data.columns.tolist()}")
        

        # Apply low-pass filter to labels
        self._labels_lowpass_filter()

        # Create EMG feature column names
        self.emg_features = []
        for sensor in self.sensors:
            for feature in self.features:
                self.emg_features.append(f'{sensor} {feature}')
        #print(f"EMG features: {self.emg_features}")

        # Separate output labels
        self.angle_cols = list(filter(lambda x: 'angle' in x, self.out_labels))
        self.moment_cols = list(filter(lambda x: 'moment' in x, self.out_labels))
        #print(f"Angle output columns: {self.angle_cols}")
        #print(f"Moment output columns: {self.moment_cols}")

        # Count EMG input features
        self.features_num = len(self.emg_features)
        print(f"Number of EMG features: {self.features_num}")

        # Full list of columns used for modeling (input + output)
        self.model_columns = self.emg_features.copy()
        self.model_columns.extend(self.out_labels)
        print(f"Model columns (X + Y): {self.model_columns}")

        # Identify joint label columns
        self._joints_columns = list(
            filter(lambda x: "sensor" not in x, self._data.columns))
        #print(f"Joint label columns (non-EMG): {self._joints_columns}")

        # Split dataset
        self.infer_set = self._data.iloc[-25:, :].copy()
        self.train_set = self._data.iloc[: int(0.6 * len(self._data)), :].copy()
        self.val_set = self._data.iloc[int(0.6 * len(self._data)): int(0.8 * len(self._data)), :].copy()
        self.test_set = self._data.iloc[int(0.8 * len(self._data)):, :].copy()

        
        print(f"Train set shape: {self.train_set.shape}")
        print(f"Val set shape: {self.val_set.shape}")
        print(f"Test set shape: {self.test_set.shape}")

        self._is_scaler_available = False

        # Scale and filter model columns only
       
        self.infer_set = self._scale(self.infer_set).loc[:, self.model_columns]
        self.full_set = self._scale(self._data).loc[:, self.model_columns]
        self.train_set = self._scale(self.train_set).loc[:, self.model_columns]
        self.val_set = self._scale(self.val_set).loc[:, self.model_columns]
        self.test_set = self._scale(self.test_set).loc[:, self.model_columns]


    @ property
    def subject_weight(self) -> float:
        with open("subject_details.json", "r") as f:
            return json.load(f)[f"S{self.subject}"]["weight"]

    def _scale(self, data):
        '''
        Scale the Dataset
        '''
        # Scale features between 0 and 1
        if not self._is_scaler_available:
            self._features_scaler = MinMaxScaler(feature_range=(0, 1))
            # The scaler will fit only data from the recording periods.
            self._features_scaler.fit(data.loc[:, self.emg_features])
        '''
        
        
        print("Feature Min:", self._features_scaler.data_min_)
        print("Feature Max:", self._features_scaler.data_max_)
        print("Feature Scale:", self._features_scaler.scale_)
        print("Number of EMG features:", len(self.emg_features))
        print("Scaler shape:", self._features_scaler.data_min_.shape)
        '''
        
        data.loc[:, self.emg_features] = self._features_scaler.transform(
                                                data.loc[:, self.emg_features])
        
        
        
        # scale angles
        if not self._is_scaler_available:
            self.angle_scaler = MinMaxScaler(feature_range=(0, 1))
            self.angle_scaler.fit(data.loc[:, self.angle_cols])
            
        '''
           

        print("Scaler fitted to EMG features.")
        print(f"Number of features: {len(self.emg_features)}")
        print(f"Shape of data_min_: {self._features_scaler.data_min_.shape}")
        print(f"Shape of data_max_: {self._features_scaler.data_max_.shape}")
        ''' 
        
        data.loc[:, self.angle_cols] = self.angle_scaler.transform(
                                                    data.loc[:, self.angle_cols])
        
        # Set the scaler value to True to avoid creating new scalers
        self._is_scaler_available = True
        return data
    
    def _labels_lowpass_filter(self, freq=6, fs=20):
        low_pass = freq / (fs / 2)
        b2, a2 = butter(N=6, Wn=low_pass, btype="lowpass")
        # Don't filter the time column
        self._data.iloc[:, -1:] = filtfilt(b2, a2, self._data.iloc[:, -1:], axis=0)
