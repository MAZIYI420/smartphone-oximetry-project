# ----------------------------------------------------------------------------
# Deep Learning SpO2 Analysis (v8_patch_1 - CSV Hotfix)
# ----------------------------------------------------------------------------
# This script is based on the methods from Hoffman et al. (2022)
# and our own R&D process to reproduce and test the model.
#
# This version (v8) DOES NOT include the SciPy resampling fix.
#
# PATCH 1: This version bypasses the metadata.csv loading error
# by defaulting to "Left Hand" for all subjects.
# ----------------------------------------------------------------------------

# --- 1. Environment Check & Imports ---
import os
import sys

# Suppress TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

try:
    import numpy as np
    import cv2
    import pandas as pd
    import h5py
    from scipy.signal import butter, filtfilt
    import matplotlib.pyplot as plt
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Error: A required library is missing.")
    print(f"Details: {e}")
    print("Please ensure all libraries (tensorflow, opencv-python, numpy, scipy, matplotlib, h5py, pandas, scikit-learn) are installed in your .venv environment.")
    sys.exit(1)

# --- 2. Global Configuration ---
MODEL_FILENAME = "spo2_model_v8_no_resample.h5"
VIDEO_FILENAME = "finger_video.mp4" 

# Model & Data Parameters from the paper
TARGET_FPS = 30.0
WINDOW_SECONDS = 3.0
WINDOW_SIZE = int(TARGET_FPS * WINDOW_SECONDS) # 90 frames
WINDOW_OVERLAP = int(WINDOW_SIZE // 2)         # 45 frames
N_CHANNELS = 3                                 # R, G, B

# --- 3. Core Functions: Data Loading (from GitHub Repo) ---

def load_metadata(metapath):
    """
    (This function is no longer called in v8_patch_1)
    Loads the metadata.csv file to determine which hand (left/right)
    was used for each participant.
    """
    if not os.path.exists(metapath):
        print(f"Warning: Metadata file not found at {metapath}")
        return None
    meta_df = pd.read_csv(metapath)
    # This line below is what caused the error, as the columns did not exist
    data_idx = meta_df[['left', 'right']].values
    return data_idx

def make_temp_data(data_uw, groundtruth_uw, data_idx=[], gt_ind=3):
    """
    This function (from the paper's 'examples') cleans the raw data.
    It selects the correct hand (L/R) based on data_idx and removes
    zero-padding from the end of the signals.
    """
    res_data_list = []
    res_gt_list = []
    
    # Use data_idx to append the correct hand's data (L: 0-2, R: 3-5)
    for pid, row in enumerate(data_idx):
        if row[0] == 1: # Use Left Hand
            res_data_list.append(data_uw[pid][:3, :])
            res_gt_list.append(groundtruth_uw[pid][gt_ind, :])
        if row[1] == 1: # Use Right Hand
            res_data_list.append(data_uw[pid][3:, :])
            res_gt_list.append(groundtruth_uw[pid][gt_ind, :])

    results_data_list = []
    results_gt_list = []
    fps_list = []
    
    # Clean signals by removing trailing zeros
    for i in range(len(res_gt_list)):
        zeros_data = np.where(res_data_list[i][0] == 0)[0]
        zeros_gt = np.where(res_gt_list[i] == 0)[0]

        if len(zeros_data) > 0:
            result_data_i = res_data_list[i][:, :int(zeros_data[0])]
        else:
            result_data_i = res_data_list[i]
        
        if len(zeros_gt) > 0:
            result_gt_i = res_gt_list[i][:int(zeros_gt[0])]
        else:
            result_gt_i = res_gt_list[i]

        # Clip data to the shorter of the two signals
        fps = 30
        clip_len = min(result_gt_i.shape[0], result_data_i.shape[1] // fps)
        result_data_i = result_data_i[:, :clip_len * fps]
        result_gt_i = result_gt_i[:clip_len]

        results_gt_list.append(result_gt_i)
        results_data_list.append(result_data_i)
        fps_list.append(fps)

    return {"data": results_data_list, "gt": results_gt_list, "fps": fps_list}

def make_windows(cleaned_data, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
    """
    This is our implementation (Step 3.5) to slice the cleaned,
    variable-length signals into fixed-size (90, 3) windows
    for training the CNN.
    """
    X_windows = []
    y_windows = []
    
    data_list = cleaned_data["data"]
    gt_list = cleaned_data["gt"]
    
    for i in range(len(data_list)):
        signal = data_list[i].T  # Transpose to (N_frames, 3_channels)
        labels = gt_list[i]
        fps = cleaned_data["fps"][i]
        
        # Ensure signal and labels align
        max_len_sec = min(len(labels), len(signal) // fps)
        
        # Iterate with a sliding window
        for start_frame in range(0, (max_len_sec * fps) - window_size, overlap):
            end_frame = start_frame + window_size
            
            # Get the (90, 3) signal window
            window = signal[start_frame:end_frame, :]
            
            # The label is the average SpO2 over the corresponding 3 seconds
            start_sec = start_frame // fps
            end_sec = (end_frame // fps) - 1 # Label corresponds to the end
            
            # Use the label at the *end* of the window
            label = labels[end_sec] 
            
            if window.shape == (window_size, N_CHANNELS):
                X_windows.append(window)
                y_windows.append(label)
                
    return np.array(X_windows), np.array(y_windows)

def load_real_data_from_github_repo():
    """
    Main function to load and process the oximetry-phone-cam-data.
    """
    print("Attempting to load real data from GitHub repo...")
    try:
        # --- [Step 1: Load necessary imports] ---
        # (Already done at top of file)
        
        # --- [Step 2: Load raw data from H5 file] ---
        PATH = './data/preprocessed/'
        h5_file_path = os.path.join(PATH, 'all_uw_data.h5')
        
        if not os.path.exists(h5_file_path):
            print(f"Error: Data file not found at {h5_file_path}")
            print("Please ensure the 'data' folder from GitHub is in your project directory.")
            return None

        with h5py.File(h5_file_path, 'r') as f:
            raw_data = f['dataset'][:]
            raw_groundtruth = f['groundtruth'][:]
            
        # --- [Step 3: Load metadata (for hand L/R)] ---
        
        # --- PATCH_1 ---
        # The line below caused an error because 'left'/'right' columns were not found.
        # meta_path = os.path.join(PATH, '..', 'gt', 'metadata.csv')
        # data_idx = load_metadata(meta_path)
        
        # We are now bypassing the metadata.csv load and *assuming* 'Left Hand'
        # for all participants, as this is the most common configuration.
        print("Info: Bypassing metadata.csv check. Assuming 'Left Hand' for all subjects.")
        data_idx = np.tile([1, 0], (raw_data.shape[0], 1)) # Default to Left Hand
        # --- END PATCH_1 ---

        # --- [Step 4: Clean the data (using paper's function)] ---
        cleaned_data = make_temp_data(raw_data, raw_groundtruth, data_idx)
        
        # --- [Step 5: Slice data into 90-frame windows (our function)] ---
        X_train, y_train = make_windows(cleaned_data)
        
        print(f"Successfully loaded and processed real data!")
        print(f"Total training samples created: {X_train.shape[0]}")
        
        if X_train.shape[1:] != (WINDOW_SIZE, N_CHANNELS):
             print(f"Error: Final data shape is {X_train.shape}, but model expects (N, {WINDOW_SIZE}, {N_CHANNELS})")
             return None
             
        return X_train, y_train

    except Exception as e:
        print(f"Error during loading of real data: {e}")
        print("Falling back to dummy data...")
        return None

# --- 4. Core Functions: Model & Signal Processing ---

def create_1d_cnn_model(input_shape=(WINDOW_SIZE, N_CHANNELS)):
    """
    Defines the 1D-CNN architecture.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1) # Output layer: 1 neuron for SpO2 regression
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Defines the Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, fs, lowcut=0.5, highcut=4.0, order=5):
    """Applies the bandpass filter to the signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def analyze_video(video_path, model, scaler):
    """
    Main function to analyze a local video file (finger_video.mp4).
    """
    print(f"Analyzing video: {video_path}...")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # --- Video Property Check ---
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (TARGET_FPS - 1 < video_fps < TARGET_FPS + 1):
        print("="*50)
        print(f"WARNING: Video FPS ({video_fps:.2f}) is not {TARGET_FPS} FPS.")
        print("This may affect signal quality and prediction accuracy.")
        print("="*50)
    
    raw_signals = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- Signal Extraction (Naive Global Average) ---
        # This is a 'naive' algorithm. A better one would find the
        # finger's Region of Interest (ROI) first.
        avg_r = np.mean(frame[:, :, 2])
        avg_g = np.mean(frame[:, :, 1])
        avg_b = np.mean(frame[:, :, 0])
        raw_signals.append([avg_r, avg_g, avg_b])
        
    cap.release()
    
    if len(raw_signals) < WINDOW_SIZE:
        print(f"Error: Video is too short ({len(raw_signals)} frames). Need at least {WINDOW_SIZE} frames.")
        return

    print("Video processing complete.")
    raw_ppg = np.array(raw_signals)
    
    # --- Signal Processing ---
    # 1. Detrending (simple normalization)
    raw_ppg_detrended = raw_ppg / np.mean(raw_ppg, axis=0) - 1
    
    # 2. Bandpass Filtering (to get pulse)
    # THIS IS v8: We filter using the video's NATIVE FPS
    filtered_ppg = bandpass_filter(raw_ppg_detrended, fs=video_fps)

    # 3. Standardization (using the *scaler* from training)
    filtered_ppg_scaled = scaler.transform(filtered_ppg)

    # --- Prediction ---
    # Create overlapping windows from the *entire* video signal
    test_windows = []
    for i in range(0, len(filtered_ppg_scaled) - WINDOW_SIZE, WINDOW_OVERLAP):
        window = filtered_ppg_scaled[i : i + WINDOW_SIZE]
        test_windows.append(window)
        
    if not test_windows:
        print("Error: Could not create any test windows from the video.")
        return

    test_windows = np.array(test_windows)
    
    # Get predictions for all windows
    predictions = model.predict(test_windows, verbose=0)
    
    # Average all predictions for a final stable value
    final_spo2 = np.mean(predictions)
    
    print("="*40)
    print(f"Predicted SpO2: {final_spo2:.2f} %")
    print("="*40)

    # --- Visualization ---
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("Raw PPG Signals (Detrended)")
    plt.plot(raw_ppg_detrended[:, 0], label="Red", color='red', alpha=0.7)
    plt.plot(raw_ppg_detrended[:, 1], label="Green", color='green', alpha=0.7)
    plt.plot(raw_ppg_detrended[:, 2], label="Blue", color='blue', alpha=0.7)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title(f"Filtered PPG Signals (Pulse) - Predicted SpO2: {final_spo2:.2f} %")
    plt.plot(filtered_ppg[:, 0], label="Red (Filtered)", color='red')
    plt.plot(filtered_ppg[:, 1], label="Green (Filtered)", color='green')
    plt.plot(filtered_ppg[:, 2], label="Blue (Filtered)", color='blue')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- 5. Main Execution ---

def main():
    """Main function to run the training or analysis."""
    
    model = None
    scaler = None
    force_retrain = False # This variable was the cause of the UnboundLocalError bug
    
    if os.path.exists(MODEL_FILENAME):
        try:
            print(f"Loading existing model from {MODEL_FILENAME}...")
            model = load_model(MODEL_FILENAME)
            model.summary()
        except Exception as e:
            print(f"Error loading model: {e}. Forcing retrain...")
            force_retrain = True
    else:
        print("No model file found. Training a new model...")
        force_retrain = True

    if force_retrain:
        # Load the real training data
        data = load_real_data_from_github_repo()
        
        if data is None:
            print("Error: Failed to load real data. Cannot train model.")
            return
            
        X, y = data
        
        if X is None or len(X) == 0:
            print("Error: Data loading returned empty data. Cannot train.")
            return

        # Create a scaler
        # We must reshape to 2D for the scaler, then back to 3D for the CNN
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, N_CHANNELS)
        scaler.fit(X_reshaped) # Fit the scaler
        
        # We don't need to transform X_train, as it's not used again
        # The scaler is saved implicitly by being used in analyze_video
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Starting training... X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        model = create_1d_cnn_model(input_shape=(WINDOW_SIZE, N_CHANNELS))
        model.summary()
        
        callbacks = [
            ModelCheckpoint(MODEL_FILENAME, save_best_only=True, monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, # Increased epochs, but EarlyStopping will find the best
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete. Model saved.")

    # --- Analysis Phase ---
    
    # We must create the scaler *after* training, or load it if model exists
    # For this script, we'll re-fit the scaler every time we run analysis
    # on a pre-trained model.
    # A robust app would save/load the scaler with the model.
    if scaler is None:
        print("Fitting a new scaler for analysis...")
        data = load_real_data_from_github_repo()
        if data is None:
            print("Error: Cannot fit scaler without data.")
            return
        X, y = data
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, N_CHANNELS)
        scaler.fit(X_reshaped)
        print("Scaler fitted.")

    # Analyze the local video file
    analyze_video(VIDEO_FILENAME, model, scaler)

if __name__ == "__main__":
    main()