# brax_web_app
Website code for the Amazon Mturk study done in IROS 2022 Paper:
"Keeping Humans in the Loop: Teaching via Feedback in Continuous Action Space Environments."
app.py is the main file to be run. 

The weibsite is built with Flask (https://flask.palletsprojects.com/en/2.1.x/), brax (https://github.com/google/brax), and 
PyTorch.  

Note on requirements:
-The requirements.txt file has many of the requirements used but some may be missing or outdated.
-The version of brax used was the version as of December/Janurary 2021 and implimentation
details of the environments are subject to change depending on brax (e.g. input and action space
size)

Note on folders:
To run the app you should have the following local folders:
feedback_data
reward_data
state_data
static
tamer_feedback_data
tamer_reward_data
templates
uid

Note on performance:
The study was conducted on a powerful CPU-based server. The speed of the simulation and site partly depends
on the machine it is being run on. The teaching experience may be affected if the simulation is too slow or
too fast. Please see the time-based parameters throughout app.py to adjsut to suite your setup. Thank you. 