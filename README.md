# IIS Lab 2: Acoustic

# Description:
In this lab, the goal is to create a system to detect different audio sounds which are then categorized  accordingly. For this, I utilized Professor Rivera's provided spectrogram code and implemented calculating and recording features, or characteristics of the audio, that are then used to determine what the noise is. The features of the audio, along with a provided label are saved to a csv file, then the labeled csv file is used to train a model utilizing sklearn svc. The model is then saved to be used for gesture recognition in <gesture_detection_visualization.py>

<gesture_detection_visualization> uses the machine learning model created in <gesture_train_model_microphone_spectrogram.py> to actively listen for a clap, whistle, or hum to then display a corresponding animal, along with how they use this sound. A sound is recorded when there is a significant increase in volume until there is prolonged silence. The label is then predicted using the model. The purpose is to educate children on animal communication in a run and interactive way, in a museum for example.

The interactive element comes from making sounds to display the different screens and learn about the animals. I was inspired to make this because I am interesting in creating interactive environments in museum settings. 

AI was used to further understand Professor Rivera's code including specific libraries used that I was unfamiliar with and overall understanding sound complexities and how it is manipulated in his code. I wrote and attempted all of the code I created but sometimes needed to find the correct functions or syntax I was searching for in unfamiliar libraries, such as Sklearn, achieve math to manipulate variables that I did not previously know, or further understand sound features, so I researched and revised with AI assistance where necessary. 

# Run instructions:

### Setup the Virtual Env
1. From the directory of this folder, create a virtual env using: `python3 -m venv env`
2. Activate the virtual environment: 
    - On Mac: <source env/bin/activate> 
    - On Windows: <./env/Scripts/activate.bat>
3. Install the required packages using: `pip install -r requirements.txt`
4. When adding new packages, be sure to update the requirements.txt using: `pip freeze > requirements.txt`

### Running the script 
1. Activate the virtual environment: 
    - On Mac: `source env/bin/activate`. 
    - On Windows: `./env/Scripts/activate.bat`
2. Run the script:
        `python <script_name>.py `

### If running without a previously saved <gesture_model.pkl> and <training.csv>:

1. `python gesture_train_model_microphone_spectrogram.py`

2. To save audio recordings to <training.csv>:
    - Type label into white text box on the plot
        - Press shift
        - Record audio
        - Press shift
        - Repeat until desired training data collected
    - To train the model
        - Press enter

    <training.csv> and <gesture_model.pkl> are both saved now and should be deleted to traina new model with new data

### To run the detection and visualization after training the model in <gesture_train_model_microphone_spectrogram.py>:

1. `python gesture_detection_visualization.py`
    - Press the space bar on the title screen to begin the detection
    - make a whistle, clap, hum noise to use the system

https://github.com/user-attachments/assets/ee6c5a2a-9145-4a31-a2b7-40739242254e
