'''
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


# Updates the plot with the latest spectrogram data
# This is called from out animation function down below
def update_plot(_frame):
    global spectrogram_data

    latest_column = None

    # Process all available microphone blocks.
    while True:
        try:
            block = audio_queue.get_nowait()
        except queue.Empty:
            break

        latest_column = compute_fft(block)

    if latest_column is not None:
        spectrogram_data[:, :-1] = spectrogram_data[:, 1:]
        spectrogram_data[:, -1] = latest_column

        image.set_data(spectrogram_data)

    return (image,)


# -----------------------------
# Start audio stream and plot
# -----------------------------
print("Available audio devices:")
print(sd.query_devices())
print("\nStarting microphone capture and repeating sweep.")
print("Close the plot window or press Ctrl+C to stop.")

fig, ax = plt.subplots(figsize=(11, 6))

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
            blit=True,
            cache_frame_data=False,
        )

        plt.show()

except KeyboardInterrupt:
    pass

finally:
    plt.close(fig)
    print("Stopped.")
