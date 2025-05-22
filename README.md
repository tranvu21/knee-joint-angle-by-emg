# Step to run these code of knee-joint-angle-by-emg

## 1. Post process the data 
Use post_process.py to add time, tune the angle
## 2. Extract feature
To creat the feature dataset at /Dataset folder, use 03_EMG_Process.py. Change the path 
## 3. Train model
Train the model by the feature dataset using 05_model.py , result would be recorded at /Result. 
## 4. Infer angle 
User 06_infer.py
