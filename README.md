# Python Setup

## Setup the Virtual Env
1. From the directory of this folder, create a virtual env using: `python3 -m venv env`
2. Activate the virtual environment: 
    - On Mac: `source env/bin/activate`. 
    - On Windows: `./env/Scripts/activate.bat`
3. Install the required packages using: `pip install -r requirements.txt`
4. When adding new packages, be sure to update the requirements.txt using: `pip freeze > requirements.txt`

## Running the script 
1. Activate the virtual environment: 
    - On Mac: `source env/bin/activate`. 
    - On Windows: `./env/Scripts/activate.bat`
2. Run the script:
        `python <script_name>.py `
