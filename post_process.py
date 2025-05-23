import pandas as pd

# Load CSV
df = pd.read_csv('E:\Spring25\Capstone\emg_log_20250520_001524.csv')  # replace with actual file path
if 'angle' in df.columns:
    min_angle = df['angle'].min()
    df['angle'] = df['angle'] - min_angle
else:
    print("Column 'angle' not found.")
# Step 1: Remove a column
cols_to_drop = ["filtered0", "filtered1", "filtered2", "filtered3", "filtered4", "filtered5", "filtered6", "pulseCount", "rotationCount", "time"]

existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=existing_cols_to_drop, inplace=True)
# Step 2: Transform "Degrees" column
print( df['angle'].min())
print( df['angle'].max())

df['time'] = [round(i * 0.001, 3) for i in range(len(df))]
# Optional: Save the modified DataFrame
df.to_csv("E:\Spring25\Capstone\\re_ver\minh_kun_data_only_raw.csv", index=False)