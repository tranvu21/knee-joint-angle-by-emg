import matplotlib.pyplot as plt
import pandas as pd
from typing import *
import numpy as np

def plot_learning_curve(history, file: str):
    # Do nothing if train stop manually
    if history == None:
        print("No train history was found")
        return
    else:
        # Create the plot
        plt.figure("Learning curve")
        plt.plot(
            history.epoch,
            history.history["loss"],
            history.epoch,
            history.history["val_loss"])
        
        plt.legend(["train loss", "val loss"])
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.draw()
        # Save according on the desired directories
        plt.savefig(file)
        return

def plot_results(y_true, y_pred, out_labels, file: str):
    tick_size = 12
    label_size = 14
    title_size = 20
    plt.figure("Prediction", figsize=(15, 8))
    time = [i / 20 for i in range(len(y_true))]
    true_df = pd.DataFrame(data=y_true, columns=out_labels, dtype=np.float64)
    pred_df = pd.DataFrame(data=y_pred, columns=out_labels, dtype=np.float64)
    
    plot_by_side = ['Right knee angle', 'Right ankle angle', 'Left knee angle','Left ankle angle',
                    'Right knee moment', 'Right ankle moment', 'Left knee moment', 'Left ankle moment']
        
    i = 1
    y_label = 'Angle'
    for col in plot_by_side:
        plt.subplot(2, len(out_labels)//2, i)
        
        plt.plot(time, true_df.loc[:, col], "b-", linewidth=2.5)
        plt.plot(time, pred_df.loc[:, col], "r--", linewidth=1.5,)
        
        if i > 4:
            plt.xlabel("Time [s]", fontsize=label_size)
            
        if i == 1 and y_label == 'Angle':
            plt.ylabel("Angle", fontsize=label_size)
            # plt.legend(["Measurments", "Estimations"], fontsize=label_size)
            
        elif i == 5:
            plt.ylabel("Moment [Nm]", fontsize=label_size)
            
        plt.title(col, fontsize=title_size)
        plt.xlim((0, 15))
        # plt.xticks(ticks=[i//5 for i in time if i%5==0],fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.grid(True)
        
        if i==len(out_labels)//2:
            y_label = 'Moment'
        i += 1
    plt.tight_layout()
    plt.savefig(file)
    plt.draw()
    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_single_output(y_true, y_pred, out_labels, file: str = "prediction_plot.pdf"):
    """
    Plots predicted vs. true values for one or more output labels.
    Saves the plot to the specified file.
    
    Parameters:
        y_true (np.ndarray): Ground truth values [samples, labels]
        y_pred (np.ndarray): Predicted values [samples, labels]
        out_labels (List[str]): List of output label names (columns)
        file (str): File path to save the figure
    """
    time = [i / 20 for i in range(len(y_true))]  # 20 Hz assumed sampling rate
    tick_size = 12
    label_size = 14
    title_size = 18
    
    true_df = pd.DataFrame(data=y_true, columns=out_labels, dtype=float)
    pred_df = pd.DataFrame(data=y_pred, columns=out_labels, dtype=float)

    plt.figure("Prediction", figsize=(10, 4))
    
    for i, col in enumerate(out_labels, start=1):
        plt.subplot(1, len(out_labels), i)
        plt.plot(time, true_df[col], "b-", linewidth=2.0, label="True")
        plt.plot(time, pred_df[col], "r--", linewidth=1.5, label="Predicted")
        plt.xlabel("Time [s]", fontsize=label_size)
        plt.ylabel(col, fontsize=label_size)
        plt.title(col, fontsize=title_size)
        plt.grid(True)
        plt.yticks(fontsize=tick_size)
        plt.xticks(fontsize=tick_size)
        plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(file)
    plt.close()
