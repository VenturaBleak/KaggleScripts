# Description: This script is used to download data from Kaggle and submit results to Kaggle
import os

from API_object import KaggleAPIConnector


if __name__ == '__main__':
    # You would set these environment variables outside of the script for security reasons
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    kaggle_connector = KaggleAPIConnector('playground-series-s3e24')
    kaggle_connector.download_data()

    print(kaggle_connector.data_dir)

    # Now you can do other tasks like loading the data, and when ready, submit your results
    submission_path = os.path.join(kaggle_connector.data_dir, 'gender_submission.csv')
    kaggle_connector.submit(submission_path, 'Your submission message')