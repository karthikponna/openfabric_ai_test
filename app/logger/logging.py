import logging
import os
from datetime import datetime

# Ensure the log directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, '..', 'log')
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a log file name with current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"{timestamp}.log")

# Configure the application-wide logger
logger = logging.getLogger('app_logger')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)