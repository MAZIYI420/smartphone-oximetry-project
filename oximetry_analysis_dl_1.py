"""
Deep Learning SpO2 Analysis (v10 - Final Fix)

This script contains a complete deep learning workflow to predict SpO2 from a
finger video. It includes a critical resampling feature to fix inaccuracies
caused by framerate mismatches.

v10 Updates:
- Fixed 'UnboundLocalError: force_retrain' Bug.
- Fixed 'import numpy asnp' typo.

Workflow:
1.  Load real data (load_real_data_from_github_repo)
    - If loading fails, fallback to dummy data (generate_dummy_data)
2.  Train model (create_1d_cnn_model)
    - If model file (spo2_model_v10_final.h5) exists, skip training.
3.  Analyze video (analyze_video)
    - Detect video FPS.
    - If FPS is not 30, resample the signal.
    - Preprocess signal.
    - Load the trained model.
    - Predict SpO2.
    - Display results and plots.
"""

# --- [0. Import Libraries] ---
import os
import time
import h5py
import pandas as pd
import numpy as np # [v10 Fix] Corrected typo
import matplotlib.pyplot as plt

# Attempt to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint
except ImportError:
    print("="*50)
    print("ERROR: TensorFlow library not found.")
    print("Please run in your terminal: pip install tensorflow")
    print("="*50)
    exit()

# Attempt to import CV2 and Scipy
try:
    import cv2
    from scipy.signal import butter, filtfilt, resample
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("="*50)
    print("ERROR: Missing one or more required libraries.")
    print("Please run all of the following commands in your terminal:")
    print("pip install opencv-python")
    print("pip install scipy")
    print("pip install scikit-learn")
    print("="*50)
    exit()

# --- [1. Global Configuration] ---
MODEL_FILENAME = 'spo2_model_v10_final.h5'
VIDEO_FILENAME = 'finger_video.mp4'

# Parameters from the paper
TARGET_FPS = 30.0         # Target FPS (Hz)
DURATION = 3              # 3 seconds
WINDOW_LENGTH = int(TARGET_FPS * DURATION) # 90 frames (WINDOW_LENGTH)

# --- [2. Signal Processing Functions] ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0) # Apply along time axis (axis=0)
    return y

def preprocess_signals_for_prediction(raw_signals, original_fps, target_fps=TARGET_FPS):
    """
    [v9 Update]
    Preprocesses signals for "prediction" (from analyze_video).
    Includes resampling, filtering, and standardization.
    """
    processed_signals = raw_signals.copy()
    num_frames = processed_signals.shape[0]
    
    # 1. [NEW] Resampling
    # If original FPS and target FPS do not match, resample.
    if abs(original_fps - target_fps) > 1.0: # Allow 1 FPS tolerance
        print(f"INFO: Resampling signal from {original_fps:.2f} FPS to {target_fps} FPS...")
        
        # Calculate new number of frames
        target_num_frames = int(num_frames * (target_fps / original_fps))
        
        # Resample R, G, B channels separately
        resampled_r = resample(processed_signals[:, 0], target_num_frames)
        resampled_g = resample(processed_signals[:, 1], target_num_frames)
        resampled_b = resample(processed_signals[:, 2], target_num_frames)
        
        processed_signals = np.stack([resampled_r, resampled_g, resampled_b], axis=1)
        
        # After resampling, we use the new framerate
        fs = target_fps
        print(f"Resampling complete. New signal length: {processed_signals.shape[0]} frames")
    else:
        # FPS is close enough, use original.
        fs = original_fps

    # 2. Filtering
    # (0.5 Hz * 60 = 30 bpm; 4 Hz * 60 = 240 bpm)
    lowcut = 0.5  # (30 bpm)
    highcut = 4.0   # (240 bpm)
    
    filtered_signals = butter_bandpass_filter(processed_signals, lowcut, highcut, fs)

    # 3. Standardization (Z-score)
    # Standard practice for CNN training
    mean = np.mean(filtered_signals, axis=0)
    std = np.std(filtered_signals, axis=0)
    
    # Prevent division by zero
    std[std == 0] = 1e-10
    
    standardized_signals = (filtered_signals - mean) / std
    
    return standardized_signals, filtered_signals # Return both signals for plotting

def preprocess_signals_for_training(X, y):
    """
    Preprocesses signals for "training" (from load_real_data).
    Only applies standardization, as data is already windowed.
    """
    # Assume X shape is (N, 90, 3)
    # StandardScaler expects (n_samples, n_features)
    # First, reshape data
    X_reshaped = X.reshape(-1, X.shape[2]) # (N*90, 3)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # Reshape back to (N, 90, 3)
    X_final = X_scaled.reshape(X.shape)
    
    # We also need to scale the labels (y)
    y_reshaped = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_reshaped)
    
    # Return processed data and the label scaler (for inverse transform)
    return X_final, y_scaled.flatten(), y_scaler


# --- [3. Data Loading Functions] ---

def load_real_data_from_github_repo():
    """
    (v8 - Populated)
    Load, clean, and window data from the GitHub repository.
    """
    print("Attempting to load real data from GitHub repo...")
    try:
        # --- [STEP 1: Necessary Imports] ---
        import h5py
        import os
        import pandas as pd
        import numpy as np # (already imported at top)
        print("All data loading libraries found.")

        # --- [STEP 2: Load Raw H5 Data] ---
        
        # [!] NOTE: Path modified to './' (current directory)
        # Ensure the 'data' folder is in your 'MyOximetryProject' folder.
        H5_PATH = './data/preprocessed/' 
        META_PATH = './data/gt/metadata.csv'

        def load_data_and_groundtruth(h5_path):
            file_path = os.path.join(h5_path, 'all_uw_data.h5')
            if not os.path.exists(file_path):
                print(f"ERROR: Data file not found at {file_path}")
                print("Please ensure you have placed the 'data' folder (from GitHub) in your project folder.")
                return None, None
            
            with h5py.File(file_path, 'r') as f:
                data = f['dataset'][:]
                groundtruth = f['groundtruth'][:]
            print("H5 file loaded successfully.")
            return data, groundtruth
        
        data_uw, groundtruth_uw = load_data_and_groundtruth(H5_PATH)
        if data_uw is None:
            return None # Trigger fallback
            
        # --- [STEP 2.5: Load Metadata (for hand selection)] ---
        def load_metadata(metapath):
            if not os.path.exists(metapath):
                print(f"ERROR: Metadata file not found at {metapath}")
                return None
            meta_df = pd.read_csv(metapath)
            print("Metadata (metadata.csv) loaded successfully.")
            return meta_df
        
        meta_df = load_metadata(META_PATH)
        if meta_df is None:
            return None # Trigger fallback
            
        # Extract hand info from metadata (1=use, 0=do not use)
        data_idx = meta_df[['l_hand_idx', 'r_hand_idx']].values

        # --- [STEP 3: Clean Long Signals (remove 0s)] ---
        # (This is the author's cleaning function)
        
        def make_temp_data(data_uw, groundtruth_uw, data_idx=[], gt_ind = 3):
            res_data_list = []
            res_gt_list = []
            for pid, row in enumerate(data_idx):
                if row[0] == 1: # Use left hand
                    res_data_list.append(data_uw[pid][:3,:]) # Channels 0,1,2
                    res_gt_list.append(groundtruth_uw[pid][gt_ind,:])
                if row[1] == 1: # Use right hand
                    res_data_list.append(data_uw[pid][3:,:]) # Channels 3,4,5
                    res_gt_list.append(groundtruth_uw[pid][gt_ind, :])

            results_data_list = []
            results_gt_list = []
            fps_list = []
            for i in range(len(res_gt_list)):
                # Find zeros (invalid data)
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

                # Clip to match shorter signal
                fps = 30
                clip_len = min(result_gt_i.shape[0], result_data_i.shape[1] // fps)
                result_data_i = result_data_i[:, :clip_len*fps]
                result_gt_i = result_gt_i[:clip_len]

                results_gt_list.append(result_gt_i)
                results_data_list.append(result_data_i)
                fps_list.append(fps)

            print("Long signal cleaning complete.")
            return {"data": results_data_list, "gt": results_gt_list, "fps": fps_list}
            
        # Call cleaning function
        # gt_ind=3 means using SpO2_fast (fast response) as label
        all_seq = make_temp_data(data_uw, groundtruth_uw, gt_ind=3, data_idx=data_idx)

        # --- [STEP 3.5: Windowing (Slicing)] ---
        # (This is our standard overlapping window method for reproduction)
        
        print(f"Starting to slice long signals into {DURATION} sec ({WINDOW_LENGTH} frame) windows...")
        X_list, y_list = [], []
        
        # We use a 50% overlap (hop_size) to augment data (1.5 sec hop)
        hop_size = WINDOW_LENGTH // 2
        
        # Iterate over the N cleaned long signals
        for long_signal_data, long_signal_gt, fps in zip(all_seq["data"], all_seq["gt"], all_seq["fps"]):
            
            # Author's data is (3, L), we must transpose to (L, 3)
            signal = long_signal_data.T 
            
            # Iterate over this long signal
            for i in range(0, signal.shape[0] - WINDOW_LENGTH, hop_size):
                
                # 1. Extract 90-frame signal window
                window_signal = signal[i : i + WINDOW_LENGTH] # (90, 3)
                
                # 2. Find the corresponding label (labels are 1Hz)
                # We take the label corresponding to the midpoint of the window
                gt_index = (i + hop_size) // fps
                if gt_index < len(long_signal_gt):
                    window_label = long_signal_gt[gt_index]
                    
                    # Check if label is valid (not 0)
                    if window_label > 0:
                        X_list.append(window_signal)
                        y_list.append(window_label)

        # Convert to Numpy arrays
        X_train_final = np.array(X_list)
        y_train_final = np.array(y_list)
        
        if X_train_final.shape[0] == 0:
            print("ERROR: No data generated after windowing!")
            return None

        print(f"Windowing complete! Generated {X_train_final.shape[0]} training samples.")
        print(f"Final data shape: X={X_train_final.shape}, y={y_train_final.shape}")
        
        # (IMPORTANT) Return X and y
        return X_train_final, y_train_final
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Will fall back to using dummy data...")
        return None

def generate_dummy_data(num_samples=1000, length=WINDOW_LENGTH, num_channels=3):
    """
    Generates dummy data if loading real data fails.
    """
    print(f"Generating {num_samples} dummy training samples...")
    # Simulate (N, 90, 3) signals
    X = np.random.randn(num_samples, length, num_channels)
    # Simulate (N,) SpO2 labels (between 70 and 100)
    y = np.random.uniform(70, 100, num_samples)
    print("Dummy data generation complete.")
    return X, y

# --- [4. Deep Learning Model] ---

def create_1d_cnn_model(input_shape=(WINDOW_LENGTH, 3)):
    """
    Builds a simple 1D CNN model.
    """
    model = Sequential()
    
    # Conv Layer 1
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    # Conv Layer 2
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Conv Layer 3
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    
    # Fully Connected Layer
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5)) # Dropout layer to prevent overfitting
    
    # Output Layer (Regression task, predicts 1 value)
    model.add(Dense(1, activation='linear')) # Linear activation
    
    # Compile the model
    # We use 'MeanSquaredError' (MSE) as the loss function
    # and 'MeanAbsoluteError' (MAE) as the metric
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
    
    model.summary() # Print model summary
    return model

# --- [5. Video Analysis Function] ---

def analyze_video(video_path, model, y_scaler):
    """
    [v9 Update]
    Loads video, extracts signals, resamples, preprocesses, and predicts with the model.
    """
    print(f"Analyzing video: {video_path}...")
    
    # 1. Check video file
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at '{video_path}'")
        print("Please name your recorded phone video 'finger_video.mp4' and place it in the project folder.")
        return

    # 2. Load video and extract R,G,B signals
    cap = cv2.VideoCapture(video_path)
    
    # [v9 NEW] Get real FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("WARNING: Could not read video FPS. Assuming 30.")
        original_fps = 30.0

    if abs(original_fps - TARGET_FPS) > 1.0:
        print("="*50)
        print(f"WARNING: Video framerate ({original_fps:.2f}) is not {TARGET_FPS} FPS.")
        print("This may affect signal quality. Attempting to resample for fix.")
        print("="*50)

    raw_signals_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # (Key Step) Extract R, G, B mean values
        # [!] NOTE: This is a naive implementation that averages the entire frame.
        # As discussed, this is sensitive to orientation, finger placement, etc.
        # A more robust method would first find the finger ROI.
        
        # OpenCV's order is BGR, we need RGB
        avg_b = np.mean(frame[:, :, 0])
        avg_g = np.mean(frame[:, :, 1])
        avg_r = np.mean(frame[:, :, 2])
        
        raw_signals_list.append([avg_r, avg_g, avg_b])
        frame_count += 1
        
    cap.release()
    
    if frame_count == 0:
        print("ERROR: Video is empty or cannot be read.")
        return

    print("Video processing complete.")
    raw_signals_np = np.array(raw_signals_list)

    # 3. Preprocess signals (including v9 resampling)
    try:
        preprocessed_signals, filtered_signals = preprocess_signals_for_prediction(
            raw_signals_np, 
            original_fps=original_fps,
            target_fps=TARGET_FPS
        )
    except Exception as e:
        print(f"ERROR: Signal preprocessing failed: {e}")
        print("This may be due to a very short video or poor signal quality.")
        return

    # 4. Window the signal into (N, 90, 3) clips
    # We use a 50% overlap (hop_size)
    hop_size = WINDOW_LENGTH // 2
    video_clips = []
    
    for i in range(0, preprocessed_signals.shape[0] - WINDOW_LENGTH, hop_size):
        clip = preprocessed_signals[i : i + WINDOW_LENGTH]
        video_clips.append(clip)
        
    if not video_clips:
        print(f"ERROR: Video is too short to be windowed into {DURATION} sec clips.")
        return
        
    video_clips_np = np.array(video_clips)
    
    # 5. Model Prediction
    # We predict on all clips and take the average
    predicted_scaled_values = model.predict(video_clips_np)
    
    # 6. Invert Scaling
    # We need the 'y_scaler' saved during training to convert the value back to SpO2
    try:
        predicted_spo2_values = y_scaler.inverse_transform(predicted_scaled_values)
    except Exception as e:
        print(f"ERROR: Could not inverse_transform prediction: {e}")
        print("This might mean the y_scaler was not loaded correctly.")
        predicted_spo2_values = predicted_scaled_values # as fallback

    # 7. Average predictions and display result
    final_prediction = np.mean(predicted_spo2_values)
    
    print("\n" + "="*30)
    print(f"  Predicted SpO2: {final_prediction:.2f} %")
    print("="*30 + "\n")

    # 8. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # [v9 Update] Plot X-axis now uses seconds
    time_axis_raw = np.arange(raw_signals_np.shape[0]) / original_fps
    time_axis_filtered = np.arange(filtered_signals.shape[0]) / TARGET_FPS

    # Plot 1: Raw Signals
    ax1.set_title(f"Raw PPG Signals (Original FPS: {original_fps:.2f})")
    ax1.plot(time_axis_raw, raw_signals_np[:, 0], 'r-', alpha=0.7, label="Red")
    ax1.plot(time_axis_raw, raw_signals_np[:, 1], 'g-', alpha=0.7, label="Green")
    ax1.plot(time_axis_raw, raw_signals_np[:, 2], 'b-', alpha=0.7, label="Blue")
    ax1.set_ylabel("Raw Pixel Value")
    ax1.legend()

    # Plot 2: Filtered & Resampled Signals
    ax2.set_title(f"Filtered & Resampled Signals (Target FPS: {TARGET_FPS})")
    ax2.plot(time_axis_filtered, filtered_signals[:, 0], 'r-', label="Red (Filtered)")
    ax2.plot(time_axis_filtered, filtered_signals[:, 1], 'g-', label="Green (Filtered)")
    ax2.plot(time_axis_filtered, filtered_signals[:, 2], 'b-', label="Blue (Filtered)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Signal Amplitude")
    ax2.legend()
    
    plt.tight_layout()
    plt.suptitle(f"Video Analysis: {video_path}\nPredicted SpO2: {final_prediction:.2f} %", y=1.03)
    plt.show()


# --- [6. Main Function] ---

def main():
    """
    Main workflow:
    1. Load data
    2. Train or Load Model
    3. Analyze Video
    """
    
    # 1. Load data
    data_tuple = load_real_data_from_github_repo()
    
    use_dummy_data = False
    if data_tuple is None:
        X_data, y_data = generate_dummy_data()
        use_dummy_data = True
    else:
        X_data, y_data = data_tuple

    # 2. Preprocess (Standardize) Training Data
    # [v8 Update] We now preprocess before training
    print("Standardizing training data...")
    X_train_scaled, y_train_scaled, y_scaler = preprocess_signals_for_training(X_data, y_data)
    print("Training data standardization complete.")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )

    # 3. Train or Load Model
    
    # --- [v10 Fix] ---
    # Initialize force_retrain before checking if model exists
    force_retrain = False
    # --- [End Fix] ---
    
    if os.path.exists(MODEL_FILENAME):
        # Load model
        print(f"Loading trained model from {MODEL_FILENAME}...")
        try:
            model = tf.keras.models.load_model(MODEL_FILENAME)
            print("Model loaded successfully.")
            model.summary()
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            print("Will force model retraining...")
            if os.path.exists(MODEL_FILENAME):
                os.remove(MODEL_FILENAME) # Delete corrupted model file
            force_retrain = True
    else:
        print(f"Model file {MODEL_FILENAME} not found. Starting new training...")
        force_retrain = True

    if force_retrain or not os.path.exists(MODEL_FILENAME):
        if use_dummy_data:
            print("WARNING: Training model on 'dummy data'.")
            print("This model is for testing only. Predictions will be meaningless.")
        else:
            print("Training model on 'real data'...")
            
        model = create_1d_cnn_model(input_shape=(WINDOW_LENGTH, 3))
        
        # Set up a callback to save only the best model
        checkpoint = ModelCheckpoint(
            MODEL_FILENAME, 
            monitor='val_loss', # monitor validation loss
            save_best_only=True, 
            mode='min',
            verbose=1
        )
        
        # Start training
        history = model.fit(
            X_train, y_train,
            epochs=20, # Train for 20 epochs
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint] # Use callbacks
        )
        
        print(f"Training complete. Best model saved to {MODEL_FILENAME}")
        
        # Reload the best model (in case the last epoch wasn't the best)
        model = tf.keras.models.load_model(MODEL_FILENAME)

    # 4. Analyze 'finger_video.mp4'
    # Ensure y_scaler (label scaler) is defined
    if 'y_scaler' not in locals():
        print("ERROR: y_scaler is not defined. Cannot invert prediction scaling.")
        print("This can happen if you are training on dummy data.")
        # Create a dummy scaler so the program can run
        y_scaler = StandardScaler().fit(np.array([[70], [100]]))
        
    analyze_video(VIDEO_FILENAME, model, y_scaler)

if __name__ == "__main__":
    main()
