import os

PROJECT_PATH = os.path.dirname(__file__)

# absolute path to mnist.pkl file
DATA_PATH = os.path.join(PROJECT_PATH, 'mnist.pkl')

# absolute path to dir where exeriment data will be saved
SAVE_PATH = os.path.join(PROJECT_PATH, 'result')

try:
    from user_settings import *
except ImportError:
    pass
