'''
Hannah Soria - code adapted from Michael Rivera ISS Lab 2

This file uses the previously written code provided to pull out features
to determine what the noise is and records them, saves them to a csv file,
then uses the labeled csv file to train a model utilizing sklearn svc. 
The model is then saved to be used for gesture recognition in 
<gesture_detection_visualization.py>

To train and add entries to the csv, user types in the label to the white
box on the plot, presses shift, records the sound, preses shift again, 
repeats this for desired amount of entires, then presses return to train 
the model. 

--------------------------------------------------------------------------

Records audio input using the microphone. The microphone input is 
processed using and FFT and the results are displayed on a retime spectrogram.

NOTE: Make sure you chose the correct INPUT_DEVICE, the console / terminal 
window will show the inputs you have available to you with indices (e.g., 0, 1, etc.)

You can also use the spectrogram to explore different times of audible sounds. 
For example, notice the different between a nail tap and a knuckle tap on your
laptop, or the table. Compare a clap vs. a whistle.
'''

import queue
import time

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import chirp, get_window

import csv
from matplotlib.widgets import TextBox

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# files
model_file = "gesture_model.pkl"
csv_training = "training.csv"

#  load existing model
if os.path.exists(model_file):
    with open(model_file, "rb") as file:
        model = pickle.load(file)
    print("loaded existing gesture model")
else:
    model = None
    print("no trained gesture model found")

# -----------------------------
# Audio configuration
# -----------------------------
SAMPLE_RATE = 96_000 #44_100 # MBpro should have 96kHz max sample rate
BLOCK_SIZE = 1024 
CHANNELS = 1

# Display settings
# the number of fft bins, so SAMPLE_RATE / 4096 will give the range of frequencies captured in a bin
FFT_SIZE = 4096  
DISPLAY_SECONDS = 10
DISPLAY_MIN_FREQUENCY = 0
DISPLAY_MAX_FREQUENCY = 22050 # at most can be SAMPLE_RATE / 2 (the nyquist frequency)

# CHANGE THESE TO MATCH YOUR SYSTEM
# Set to the below to None to use the system defaults.
# Input is microphone
INPUT_DEVICE = 0 # Set to the index of you laptop mic

# The callback writes microphone data here.
audio_queue = queue.Queue(maxsize=50)

# -----------------------------
# Audio callback
# -----------------------------
def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(status)

    # Put a copy of microphone input into the queue.
    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        # Drop the oldest block if plotting falls behind.
        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Empty:
            pass

# -----------------------------
# Spectrogram state
# -----------------------------
frequency_bins = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)

# The minimum FFT bin we will care about for the Spectrogram display
min_bin = np.searchsorted(
    frequency_bins,
    DISPLAY_MIN_FREQUENCY,
)

# The maximum FFT bin we will care about for the Spectrogram display
max_bin = np.searchsorted(
    frequency_bins,
    DISPLAY_MAX_FREQUENCY,
)

frequency_bins = frequency_bins[min_bin:max_bin]

# Number of FFT columns shown.
column_count = max(
    1,
    int(DISPLAY_SECONDS * SAMPLE_RATE / BLOCK_SIZE),
)

# Will store the data for display in the plot
spectrogram_data = np.full(
    (len(frequency_bins), column_count),
    -100.0,
    dtype=np.float32,
)

# FFT windowing using Hann fxn which smooths the signal
window = get_window("hann", FFT_SIZE)

# Computes an FFT over a block and returns the decibel result for the range
# from min_bin to max_bin (as defined above). 
def compute_fft(block):
    """Return the positive-frequency magnitude spectrum in dB."""
    if len(block) < FFT_SIZE:
        padded = np.zeros(FFT_SIZE, dtype=np.float32)
        padded[-len(block):] = block
        block = padded
    else:
        block = block[-FFT_SIZE:]
    
    # Normalizes the input to remove DC Offset
    block = block - np.mean(block)
    
    # Returns only the positive FFT values
    # the input is the windowing function (hann as specified above) f
    # unction applied to the bloc
    spectrum = np.fft.rfft(block * window)

    # Normalize magnitude and convert to dB.
    magnitude = np.abs(spectrum) / np.sum(window)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
    
    return magnitude_db[min_bin:max_bin]

# PART 2 LAB: MADE BY HANNAH
# this function identifies the sound by capturing the frequencies groups of ranges
def id_sound(previous_column, latest_column, block):
    # how much the frequency changes 
    difference =  latest_column - previous_column

    # finds the number of bins that are changed by a significant amount
    # significant amount determiend to be 20 hear
    changed_bins = np.where(difference > 20)[0]

    total_changed_bins = len(changed_bins)

    # peak frequencies, ignoring below 500 Hz
    peak_mask = (frequency_bins >= 500)
    peak_frequencies = (frequency_bins[peak_mask])
    peak_values = (latest_column[peak_mask])

    if len(peak_values) > 0:
        peak_index = np.argmax(peak_values)
        peak_freq = (peak_frequencies[peak_index])
    else:
        peak_freq = 0

    # convert dB to linear magnitude
    magnitude = 10 ** (latest_column / 20)
    magnitude_sum = np.sum(magnitude) + 1e-10

    # spectral centroid
    spectral_centroid = np.sum(frequency_bins * magnitude) / magnitude_sum

    # spectral spread
    spectral_spread = np.sqrt(np.sum(((frequency_bins - spectral_centroid) ** 2) * magnitude) / magnitude_sum)

    # how strong is the strong freq compared to total spectrum
    peak_strength = np.max(magnitude) / magnitude_sum

    # volume / rms
    rms = np.sqrt(np.mean(block ** 2))
    
    # store the data into a dictionary
    features =  {
        "total_changed_bins": len(changed_bins),
        "peak_freq": float(peak_freq), 
        "rms": float(rms),
        "spectral_centroid": float(spectral_centroid),
        "spectral_spread": float(spectral_spread),
        "peak_strength": float(peak_strength)
    }

    return features

# manually record for training, variables
recording = False
recorded_feats = []
sound_found = []

# aut record once already trained, variables
auto_recording = False
auto_recorded_feats = []
silence_count = 0

# sound detection settings
# recognize the start of a sound based on volume
sound_threshold = 0.03
# period of silence after a sound
silence_frames = 20

recent_feats = []
recent_frame_count = 20

feature_cols = [

    "bins_mean",
    "bins_max",
    "bins_std",

    "peak_freq",
    "peak_count",
    "change_range",

    "rms_mean",
    "rms_max",

    "centroid_mean",
    "centroid_max",

    "spread_mean",
    "spread_max",

    "peak_strength_mean",
    "peak_strength_max",

    "duration",
    "peak_strength_range",
    "peak_strength_std"
]

# MADE BY HANNAH
# function waits for shift to be pressed then 
# it starts a new tracking of a sound until shift is pressed again
def on_key(event):

    global recording
    global model

    # shift is pressed
    if event.key == "shift":

        recording = not recording

        # recording start, new empty dict
        if recording:
            print("recording started")
            recorded_feats.clear()

        # the recording has stopped
        else:
            print("recording stopped")
            if len(recorded_feats) >0:
                
                # Get gesture name from the text box
                gesture = text_box.text.strip()

                # if the box is empty, prompt for a gesture name
                if gesture == "":
                    print("Please enter a gesture name.")
                    return

                # gesture and summary of the recording
                # get the summary from the funcion
                summary = sum_recording(recorded_feats)

                gesture_data = {
                    "gesture": gesture,
                    **summary
                }

                # add the information
                sound_found.append(gesture_data)
                fieldnames = [
                    'gesture',

                    'bins_mean',
                    'bins_max',
                    'bins_std',

                    'peak_freq',
                    'peak_count',
                    'change_range',

                    'rms_mean',
                    'rms_max',

                    'centroid_mean',
                    'centroid_max',

                    'spread_mean',
                    'spread_max',

                    'peak_strength_mean',
                    'peak_strength_max',

                    'duration',
                    'peak_strength_range',
                    'peak_strength_std'
                ]

                #  open the training csv and add the gesture dictionary in
                with open('training.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                    # if the file is empty write the header
                    if f.tell()==0:
                        writer.writeheader()
                    # write the gesture info to the csv
                    writer.writerow(gesture_data)

                # clear dict
                recorded_feats.clear()

                # clear text box
                text_box.set_val("")

    # key for training
    elif event.key == "enter":
        if recording:
            print("stop recording before training")
            return
        
        print("\nfinished collecting training data")

        model = train_model()

# MADE BY HANNAH
def sum_recording(frames):
    # make lists of each of this values for every frame in the recording
    bins = [f["total_changed_bins"] for f in frames]
    peak = [f["peak_freq"] for f in frames]
    rms = [f["rms"] for f in frames]
    centroid = [f["spectral_centroid"] for f in frames]
    spread = [f["spectral_spread"] for f in frames]
    peak_strength = [f["peak_strength"] for f in frames]

    # find significant chagnes in recording
    peaks, properties = find_peaks(
        bins, height=50, distance=5
    )

    # average the features
    features = {
        "bins_mean": np.mean(bins),
        "bins_max": np.max(bins),
        # standard dev to check the flucuation
        "bins_std": np.std(bins),

        "peak_freq": np.median(peak),

        "peak_count": len(peaks),

        "change_range": np.max(bins) - np.min(bins),

        "rms_mean": np.mean(rms),
        "rms_max": np.max(rms),

        "centroid_mean": np.mean(centroid),
        "centroid_max": np.max(centroid),

        "spread_mean": np.mean(spread),
        "spread_max": np.max(spread),

        "peak_strength_mean": np.mean(peak_strength),
        "peak_strength_max": np.max(peak_strength),

        "duration": len(frames) * (BLOCK_SIZE / SAMPLE_RATE),
        "peak_strength_range": np.max(peak_strength) - np.min(peak_strength),
        "peak_strength_std": np.std(peak_strength)
    }

    return features

# MADE BY HANNAH
def recognize_gesture(frames):
    global model

    # there is no model trained
    if model is None:
        return

    # no frames are detected
    if len(frames) == 0:
        return

    summary = sum_recording(frames)

    X_new = pd.DataFrame([[summary[column] for column in feature_cols]], columns = feature_cols)

    print("\n========================================")
    print("LIVE SOUND")
    print("========================================")

    for column in feature_cols:
        print(f"{column:20s}: {summary[column]:.6f}")

    print("----------------------------------------")
    print("NUMBER OF LIVE FRAMES:", len(frames))

    prediction = model.predict(X_new)[0]

    print("----------------------------------------")
    print("PREDICTION:", prediction)
    print("========================================\n")

    sound_text.set_text("Gesture: " + str(prediction))

    return prediction


# MADE BY HANNAH
# function trains the model
def train_model():

    print("loading training data...")

    # check csv exists
    # if not os.path.exists(csv_training):
    #     print("no training.csv file found")
    #     return None

    # load the created csv
    training_data = pd.read_csv("training.csv")

    # verify columns
    required_cols = ["gesture", *feature_cols]

    missing_columns = [column for column in required_cols 
                       if column not in training_data.columns]


    if len(missing_columns) > 0:
        print("\nERROR:")

        print("The training CSV is missing:")

        for column in missing_columns:
            print(" -", column)


        print("\nDelete training.csv and collect "
            "new training data using this version.")

        return None

    # removing any missing vals 
    training_data = training_data.dropna(subset=required_cols)

    # X = numerical features
    # get all of the data excluding the name
    X = training_data[feature_cols]

    # Y = gesture labels
    Y = training_data["gesture"]

    print("number of recordings: ", len(X))
    print("gestures: ", Y.unique())

    # split the data into training and testing data
    # random state randomized what is used for testing and training
    # tratify preserves the split of different gestures
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print("training recordings:", len(X_train))
    print("test recordings:", len(X_test))

    # create the svm
    #  standard scaler standardizes the data around 0 to make it more usable
    model = make_pipeline(
        StandardScaler(),
        SVC(
            class_weight="balanced")
    )

    # train the nvm
    # learn the data
    model.fit(X_train, Y_train)

    # save model
    with open(model_file, "wb") as file:
        pickle.dump(model,file)

    print("gesture model saved")

    print("\nmodel trained")

    # test the model
    predictions = model.predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(Y_test, predictions)

    print("\nresults")

    # zip goes through two lists at the same time and matches their positions
    for actual, predicted in zip(Y_test, predictions):
        print("acutal: ", actual, "   predicted: ", predicted)

    print("\naccuracy: ", accuracy)
    print("\naccuracy:", accuracy)

    print("\nclassification report:")
    print(classification_report(Y_test, predictions))

    print("\nconfusion matrix:")
    print(confusion_matrix(Y_test, predictions))

    return model

# global state for spectrogram
previous_column = None

# EDITED BY HANNAH
# Updates the plot with the latest spectrogram data
# This is called from out animation function down below
def update_plot(_frame):

    global spectrogram_data
    global previous_column
    global auto_recording
    global auto_recorded_feats
    global silence_count

    # Process available microphone blocks
    while True:

        try:
            block = audio_queue.get_nowait()

        except queue.Empty:
            break

        # Convert microphone block to FFT
        latest_column = compute_fft(block)

        # Update spectrogram
        spectrogram_data[:, :-1] = spectrogram_data[:, 1:]
        spectrogram_data[:, -1] = latest_column

        image.set_data(spectrogram_data)

        # Need two FFT columns to compare
        if previous_column is not None:

            # Get information about how much the sound changed
            features = id_sound(previous_column, latest_column, block)

            recent_feats.append(features.copy())

            if len(recent_feats) > recent_frame_count:
                recent_feats.pop(0)
            
            #training
            if recording:
                recorded_feats.append(features.copy())

            # detection
            if not recording and model is not None:

                rms = features["rms"]

                # Start recording a new sound
                if not auto_recording and rms > sound_threshold:

                    auto_recording = True
                    auto_recorded_feats = recent_feats.copy()
                    silence_count = 0

                    print("SOUND DETECTED")

                # recording a sound
                elif auto_recording:
                    auto_recorded_feats.append(features.copy())

                    # check for silence
                    if rms <= sound_threshold:
                        silence_count += 1
                    else:
                        silence_count = 0

                    # sound ended
                    if silence_count >= silence_frames:
                        print("sound ended")

                        recognize_gesture(auto_recorded_feats)

                        auto_recording = False
                        auto_recorded_feats = []
                        silence_count = 0
                        sound_text.set_text("no sound detected")

        # save current fft column
        previous_column = np.copy(latest_column)

    return image, sound_text

# -----------------------------
# Start audio stream and plot
# -----------------------------
print("Available audio devices:")
print(sd.query_devices())
print("\nStarting microphone capture and repeating sweep.")
print("Close the plot window or press Ctrl+C to stop.")

fig, ax = plt.subplots(figsize=(11, 6))
fig.canvas.mpl_connect("key_press_event", on_key)

axbox = plt.axes([0.25, 0.01, 0.5, 0.05])
text_box = TextBox(axbox, "Gesture: ")

image = ax.imshow(
    spectrogram_data,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=[
        -DISPLAY_SECONDS,
        0,
        DISPLAY_MIN_FREQUENCY,
        DISPLAY_MAX_FREQUENCY,
    ],
    cmap="magma",
    vmin=-90,
    vmax=-20,
)

sound_text = ax.text (
    0.05, 0.95, 
    "no sound detected",
    transform=ax.transAxes,
    fontsize=16,
    color="pink",
    verticalalignment="top"
)

ax.set_title("Live Microphone FFT Spectrogram")
ax.set_xlabel("Time relative to now (seconds)")
ax.set_ylabel("Frequency (Hz)")

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Magnitude (dB)")

# Open the audio input and output stream and 
# attach our audio_callback
stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    dtype="float32",
    channels=(CHANNELS, CHANNELS),
    device=(INPUT_DEVICE, None),
    callback=audio_callback,
    latency="low",
)

# Open stream and run animation to update plot
try:
    with stream:
        animation = FuncAnimation(
            fig,
            update_plot,
            interval=50,
            blit=False,
            cache_frame_data=False,
        )

        plt.show()

except KeyboardInterrupt:
    pass

finally:
    plt.close(fig)
    print("Stopped.")