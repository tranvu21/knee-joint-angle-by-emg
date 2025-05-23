import pandas as pd
import tensorflow as tf
from tensorflow import keras
from utilities.DataHandler import DataHandler
from typing import *
tf.random.set_seed(42)
from pandas.api.types import is_integer_dtype

class WindowGenerator:
    def __init__(self, dataHandler: DataHandler,
                 input_width=10, label_width=1,
                 shift=1, batch_size=128, is_general_model=False):
        '''
        Window object will make it easier to work with the time series dataset, allowing more flexability
        when adjusting the parameters of the input and output sizes and lengths.
        For more details about this part in code go to https://www.tensorflow.org/tutorials/structured_data/time_series
        '''
        self.dataHandler = dataHandler
        self.batch_size = batch_size
        self.out_labels = dataHandler.out_labels
        self.out_nums = len(self.out_labels)
        self._is_general_model = is_general_model
        
        self.features_columns = self.dataHandler.emg_features
        self.features_num = self.dataHandler.features_num
        
        ####### Store the raw data #######
        self.train_set = self.dataHandler.train_set
        self.val_set = self.dataHandler.val_set
        self.test_set = self.dataHandler.test_set
        self.infer_set = self.dataHandler.infer_set
        ####### Window parameters #########
        # Set the the timesteps
        self.input_width = input_width
        self.label_width = label_width
        # Shift parameters shows the difference in timesteps between last input and last output
        self.shift = shift
        # Create one window that hold all inputs and outputs timesteps
        self.total_window_size = input_width + shift
        # features slicer (slice through the time)
        self.input_timestep_slice = slice(0, input_width)
        # Find the output's startig timestep
        self.label_start_timestep = self.total_window_size - self.label_width
        required_rows = input_width + shift
        #print(f"Need at least {required_rows} rows for inference, got {len(self.infer_set)}")
        #print(self.infer_set)
        #print(self.test_set)
        #print("IN THE WINDOW GENERATOR INIT" )
        self.labels_timestep_slice = slice(self.label_start_timestep, None)
        

    def IO_window(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Create tf time-series data set [Batch_size, timestep, features/labels].
        Can handle each side-data separately or provide a full dataset if you want to work with a unidirectional model
        """
        #print(f"[DEBUG] Input data shape: {data.shape}")
        #print(f"[DEBUG] Input data columns: {data.columns}")
        #print(f"[DEBUG] total_window_size: {self.total_window_size}")
        #print(f"[DEBUG] batch_size: {self.batch_size}")
        #print(f"[DEBUG] Input data shape: {data.shape}")
        #print(f"[DEBUG] Input data columns: {data.columns}")
        
        # Inspect the index
        #print(f"[DEBUG] Index type: {type(data.index)}")
        #print(f"[DEBUG] Index dtype: {data.index.dtype}")
        #print(f"[DEBUG] Index values (first 10): {data.index[:10].tolist()}")
        #print(f"[DEBUG] Index is integer: {data.index.is_integer()}")
        #print(f"[DEBUG] Index is monotonic increasing: {data.index.is_monotonic_increasing}")
        #print(f"[DEBUG] Index has duplicates: {data.index.has_duplicates}")
        #print(f"[DEBUG] Index length: {len(data.index)}")
        if data.empty or len(data) < self.total_window_size:
            raise ValueError(f"Input data has {len(data)} rows, but total_window_size is {self.total_window_size}")
        ds = keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1, shuffle=False,
            batch_size=self.batch_size)
        #print(f"[DEBUG] IO_window dataset created, checking size...")
        '''
        
        
        try:
            dataset_size = len(list(ds.as_numpy_iterator()))
            print(f"[DEBUG] IO_window dataset size: {dataset_size}")
        except Exception as e:
            print(f"[ERROR] Failed to iterate dataset: {e}")
            raise
        '''
        return ds
        
        return keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1, shuffle=False,
            batch_size=self.batch_size,)

    def split_window(self, features: tf.data.Dataset) -> List[tf.data.Dataset]:
        '''Take all EMG features and knee angle column. Shape is [Batch_size, timestep, features/labels]'''
        inputs = features[:, self.input_timestep_slice, :-self.out_nums]
        # Predict ankle angle & torque
        labels = features[:, self.labels_timestep_slice, -self.out_nums:]
        #print(f"[DEBUG] inputs shape: {inputs.shape}, labels shape: {labels.shape}")
        #print(f"[DEBUG] input_timestep_slice: {self.input_timestep_slice}, labels_timestep_slice: {self.labels_timestep_slice}")
        #print(f"[DEBUG] out_nums: {self.out_nums}")
        # Slicing doesn't preserve static shape information, so set the shapes manually
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    @classmethod
    def preprocessing(cls, ds: tf.data.Dataset, shuffle=False, 
                      batch_size=None, drop_reminder=False) -> tf.data.Dataset:
        """Will process batched dataset according to the inputs, then returned cached and prefetched the tf.data
        """
        #print(f"[DEBUG] Dataset size before preprocessing: {len(list(ds.as_numpy_iterator()))}")
        # Data are batched, unbatch it
        ds = ds.unbatch()
        # The dataset size is small, so cache in the memory to speed the code runtime
        ds = ds.cache()
        # Shuffle the data
        if shuffle:
            ds = ds.shuffle(buffer_size=16000, reshuffle_each_iteration=True)
        # create batches
        ds = ds.batch(batch_size, drop_remainder=drop_reminder)
        # Prefetch to speed the code runtime
        #print(f"[DEBUG] Dataset size after batching: {len(list(ds.as_numpy_iterator()))}")
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    @ property
    def train_dataset(self) -> tf.data.Dataset:
        """Make the training dataset for the indiviual models from train_01 and train_02.
        """
        train_ds = self.IO_window(self.train_set)
        # Split window data to input and output and store results
        train_ds = train_ds.map(self.split_window)
        if not self._is_general_model:
            # Process the training set
            train_ds = self.preprocessing(train_ds, shuffle=True, 
                                          batch_size=self.batch_size,
                                          drop_reminder=True)
        return train_ds

    @property
    def val_dataset(self) -> tf.data.Dataset:
        """Make the validation dataset for the indiviual models."""
        # Create timeseries dataset
        val_ds = self.IO_window(self.val_set)
        # Split window data to inputs and outputs
        val_ds = val_ds.map(self.split_window)
        if not self._is_general_model:
            # Make the batch size as big as possible to fit the whole dataset in a single batch
            val_ds = self.preprocessing(val_ds, shuffle=False, 
                                        batch_size=160000,
                                        drop_reminder=False)
        return val_ds

    @property
    def evaluation_set(self) -> tf.data.Dataset:
        """Make the evaluation dataset for the indiviual models. In validation set, remove the NaN values but in evaluation set NaN values can stay.
        """
        # Create timeseries dataset
        test_ds = self.IO_window(self.test_set)
        # Split window data to inputs and outputs
        test_ds = test_ds.map(self.split_window)
        # Make the batch size as big as possible to fit the whole dataset in a single batch
        test_ds = self.preprocessing(test_ds, shuffle=False, 
                                     batch_size=160000,
                                     drop_reminder=False)
        return test_ds
    @property
    def inference_set(self) -> tf.data.Dataset:
         # Create timeseries dataset
        print(f"[DEBUG] infer_set shape: {self.infer_set.shape}")
        print(f"[DEBUG] infer_set columns: {self.infer_set.columns}")
        if not is_integer_dtype(self.infer_set.index):
            print("[DEBUG] Resetting infer_set index to RangeIndex")
            infer_set = self.infer_set.reset_index(drop=True)
        else:
            infer_set = self.infer_set
            
        infer_ds = self.IO_window(infer_set)
        # Split window data to inputs and outputs
        infer_ds = infer_ds.map(self.split_window)
        dataset_size = len(list(infer_ds.as_numpy_iterator()))
        print(f"[DEBUG] inference_set size before batching: {dataset_size}")
        # Make the batch size as big as possible to fit the whole dataset in a single batch
        infer_ds = self.preprocessing(infer_ds, shuffle=False, 
                                     batch_size=256,
                                     drop_reminder=False)
        return infer_ds
    
    
    def make_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Make train/val/test datasets for individual models. Test dataset can be used to test GM also.
        """
        return self.train_dataset, self.val_dataset, self.evaluation_set
