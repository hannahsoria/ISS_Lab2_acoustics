# IIS Lab 2: Acoustic

# Description:
In this lab, the goal is to create a system to detect different audio sounds that are recognized 
accordingly by the system. For this, I utilized Professor Rivera's provided spectrogram code to 
to pull out features to determine what the noise is and records them, saves them to a csv file,
then uses the labeled csv file to train a model utilizing sklearn svc. The model is then saved to 
be used for gesture recognition in <gesture_detection_visualization.py>

<gesture_detection_visualization> uses the machine learning model created in 
<gesture_train_model_microphone_spectrogram.py> to actively listenfor a clap, whistle, or hum to 
then display a corresponding animal that utilizes this noise as to educate children in a museum for example.

This is implemented by constantly waiting for a noise to be detected via volume, then capturing it 
until it quiets, then making a prediction based on the model previously created.

The interactive element comes from making sounds to display the different screens and learn about the animals. I was inspired to make this because I am interesting in creating interactive environments
in museum settings. 

AI was used to further understand Professor Rivera's code including specific libraries uses I was unfamiliar with, the code structure, and overall understanding sound and how it is manipulated in this project. AI was also used to achieved cleaner code, for example I wrote and attempting all of the code I created but recognized it could have been written better, or needed to find the correct function I was searching for, so I revised with AI. Lastly, I used AI to understand Sklearn implmentation because this was an unfamiliar library for me.

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

