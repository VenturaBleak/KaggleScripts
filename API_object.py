import os
import zipfile
import kaggle

class KaggleAPIConnector:
    def __init__(self, competition_name):
        self.competition_name = competition_name
        self.api = self.authenticate()

        # Directories for downloading and extracting data
        self.data_dir = os.path.join(os.getcwd(), self.competition_name, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def authenticate():
        """Authenticate using the Kaggle API credentials from environment variables."""
        # Assumes that 'KAGGLE_USERNAME' and 'KAGGLE_KEY' are set in the environment
        kaggle.api.authenticate()
        return kaggle.api

    def download_data(self):
        """Download and unzip competition data using the Kaggle API."""
        # only download data if it hasn't been downloaded yet
        if not os.listdir(self.data_dir):
            print(f"Downloading data for {self.competition_name}...")
            self.api.competition_download_files(self.competition_name, path=self.data_dir, force=True, quiet=False)
            self.unzip_data()
            print(f"Data for {self.competition_name} has been downloaded to {self.data_dir}")
        else:
            print(f"Data for {self.competition_name} has already been downloaded to {self.data_dir}")

    def unzip_data(self):
        """Unzip downloaded data."""
        for file in os.listdir(self.data_dir):
            if file.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(self.data_dir, file), 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                os.remove(os.path.join(self.data_dir, file))
        print(f"Data for {self.competition_name} has been downloaded and unzipped to {self.data_dir}")

    def submit(self, file_path, message):
        """Submit the results to the Kaggle competition."""
        print(f"Submitting {file_path} to {self.competition_name}...")
        submission_result = self.api.competition_submit(file_path, message, self.competition_name)
        print(submission_result)

# Usage
if __name__ == '__main__':
    COMPETITION_NAME = 'playground-series-s3e9'

    # You would set these environment variables outside of the script for security reasons
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    kaggle_connector = KaggleAPIConnector(COMPETITION_NAME)
    kaggle_connector.download_data()

    # Now you can do other tasks like loading the data, and when ready, submit your results
    submission_path = os.path.join(kaggle_connector.data_dir, 'submission.csv')
    # kaggle_connector.submit(submission_path, 'Your submission message')