'''
Hannah Soria - code adapted from Michael Rivera ISS Lab 2

This file uses the machine learning model created in 
<gesture_train_model_microphone_spectrogram.py> to actively listen
for a clap, whistle, or hum to then display a corresponding animal
that utilizes this noise as to educate children in a museum for example

This is implemented by constantly waiting for a noise to be detected via 
volume, then capturing it until it quiets, then making a prediction based 
on the model previously created

The interactive element come from making sounds to display the different
screens and learn about the animals
'''

# imports
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.signal import chirp, get_window
from scipy.signal import get_window, find_peaks
import csv
import pandas as pd
import pickle
import os
from p5 import *
import random

# ML gesture model created in previous file
model_file = "gesture_model.pkl"
my_font = None

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

canvas_width = 1100
canvas_height = 700
title_screen = True

# images set to global variable
global seal_img
global dolphin_img
global whale_img

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

# ------------------------------------------------------------------------------
# following code was made or edited by Hannah

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

    # peak frequency
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

# auto record once already trained, variables
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

current_detect = "no sound"
status_message = "listening"

# function creates lists of the features of a recording
def sum_recording(frames):

    # make lists of each of these values for every frame in the recording
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

# Function identifies the sound by collecting new data then using predict
def recognize_gesture(frames):
    global model
    global current_detect
    global status_message

    # there is no model trained
    if model is None:
        return

    # no frames are detected
    if len(frames) == 0:
        return

    # collect a summary of the features
    summary = sum_recording(frames)

    X_new = pd.DataFrame([[summary[column] for column in feature_cols]], columns = feature_cols)

    # prints to help determine the sound, and characteristics of the sound
    # for debugging purposes
    print("\n========================================")
    print("LIVE SOUND")
    print("========================================")

    for column in feature_cols:
        print(f"{column:20s}: {summary[column]:.6f}")

    print("----------------------------------------")
    print("NUMBER OF LIVE FRAMES:", len(frames))

    # predict what the gesture is 
    prediction = model.predict(X_new)[0]

    # convert prediction to a string
    current_detect = str(prediction)

    print("----------------------------------------")
    print("PREDICTION:", prediction)
    print("========================================\n")

    return prediction

# global state for spectrogram
previous_column = None

# EDITED BY HANNAH
# Updates the plot with the latest spectrogram data
# This is called from out animation function down below
def update_plot():

    global previous_column
    global auto_recording
    global auto_recorded_feats
    global silence_count
    global status_message

    # Process available microphone blocks
    while True:

        try:
            block = audio_queue.get_nowait()

        except queue.Empty:
            break

        # Convert microphone block to FFT
        latest_column = compute_fft(block)

        # Need two FFT columns to compare
        if previous_column is not None:

            # Get information about how much the sound changed
            features = id_sound(previous_column, latest_column, block)

            recent_feats.append(features.copy())

            if len(recent_feats) > recent_frame_count:
                recent_feats.pop(0)
        
            # detection
            # a sound is detected by measure that the volume has increased over a threshold
            rms = features["rms"]

            # If the start of a sound is detected
            if not auto_recording and rms > sound_threshold:

                # begin recording
                auto_recording = True
                # grab the recent audio
                auto_recorded_feats = recent_feats.copy()
                # prepare count for cooldown of sound
                silence_count = 0

                print("SOUND DETECTED")

            # recording a sound while it happens
            elif auto_recording:

                # appending the features
                auto_recorded_feats.append(features.copy())

                # check for silence
                # continue on if the threshold remains broken
                if rms <= sound_threshold:
                    silence_count += 1
                else:
                    silence_count = 0

                # sound ended once the count ends
                if silence_count >= silence_frames:
                    print("sound ended")

                    # now determine the gesture
                    recognize_gesture(auto_recorded_feats)

                    # reset the recording variables
                    auto_recording = False
                    auto_recorded_feats = []
                    silence_count = 0

        # save current fft column to be the "new" previous
        previous_column = np.copy(latest_column)

# ------------------------------VISUALIZATION--------------------------------------
# the aim of the visualizationis to mimic an installation in an educational setting for kids
# i.e. a childrens museum or experience, and have them learn what sounds different animals make 
# by making the sounds themselves then learning about it, done with p5 processing

# draws the first screen shown, title card that gives instructions
def draw_title_screen():
    background(6, 29, 84)
    text_font(my_font)
    text_size(50)
    text("Ocean mammals make distinct\n      noises to communicate\n ", 200, 100)
    text_size(30)
    text("try a WHISTLE, CLAP, or HUM\nto explore ocean communication\n ", 300, 300)
    text("Press [space] to begin\n ", 375, 450)
    
# page for the seal sound: CLAP
def seal():
    background(101, 161, 189)
    image(seal_img,300,250)
    text_font(my_font)
    text_size(40)
    text("Male seals CLAP to attract a mate,\nshow physical strength,\nand ward off competitiors!\n ", 100,100)
    text("Try another sound!\n ", 400, 600)

# page for the dolphine sound: WHISTLE
def dolphin():
    background(108, 149, 247)
    # img = load_image("img/dolphin.png")
    image(dolphin_img, 300, 250)
    # text_size(40)
    text_font(my_font)
    text_size(40)
    text("Dolphins WHISTLE to identify themselves,\nstay in contact with their pod, and find each other!\n ", 100, 100)
    text("Try another sound!\n ", 400, 600)

# page for the whale sound: HUM
def whale():
    background(30, 69, 112)
    # img = load_image("img/whale.png")
    image(whale_img, 300, 250)
    # text_size(50)
    text_font(my_font)
    text_size(40)
    text("Whales HUM for all communication\nincluding mating, finding groups over\ndistances, identity, and mapping!", 100, 100)
    text("Try another sound!\n ", 400, 600)

# tracks what the gesture currently detected is and switches the screen accordingly
def draw_detection():
        
    # seal
    if current_detect == "clap":
        seal()

    # dolpin
    elif current_detect == "whistle":
        dolphin()

    # whale
    elif current_detect == "hum":
        whale()

    if current_detect == "no sound":
        display_text = "LISTENING"
    
    else:
        display_text = (str(current_detect).upper())

# on the title screen a spacebar advances to the start of the experience
# this is due to if a sound is detected right away then there is no time to read the instructions
def key_pressed():
    global title_screen
    print("KEY PRESSED:", key)
    if title_screen and key == " ":
        title_screen = False

        # clear any audio data / fresh start
        # also helps with memory, project doesnt fail
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        print("live sound detection started")

# load all of the assests, create canvas, set backgorund
def setup():
    global my_font
    global dolphin_img
    global whale_img
    global seal_img
    size(canvas_width, canvas_height)
    background(15, 20, 30)
    my_font = load_font("txt.ttf")
    seal_img = load_image("img/seal.png")
    dolphin_img = load_image("img/dolphin.png")
    whale_img = load_image("img/whale.png")

# controls which screen is currently displayed
def draw():
    global title_screen
    if title_screen:
        draw_title_screen()
    else:
        update_plot()
        draw_detection()

stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    dtype="float32",
    channels=(CHANNELS,CHANNELS),
    device=(INPUT_DEVICE, None),
    callback=audio_callback,
    latency="low"
)

# run
try:
    with stream:
        run()

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    print("Microphone closed.")