import os
import zipfile
import kaggle

class KaggleAPIConnector:
    def __init__(self, competition_name):
        self.competition_name = competition_name
        self.api = self.authenticate()
        # Directory for downloading and extracting data
        self.data_dir = os.path.join(os.getcwd(), self.competition_name, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def authenticate():
        """Authenticate using the Kaggle API credentials from environment variables."""
        kaggle.api.authenticate()
        return kaggle.api

    def download_data(self):
        """Download and unzip competition data using the Kaggle API."""
        # Only download if the data directory is empty
        if not os.listdir(self.data_dir):
            print(f"Downloading competition data for {self.competition_name}...")
            self.api.competition_download_files(
                self.competition_name,
                path=self.data_dir,
                force=True,
                quiet=False
            )
            self.unzip_data()
            print(f"Competition data for {self.competition_name} has been downloaded to {self.data_dir}")
        else:
            print(f"Competition data for {self.competition_name} has already been downloaded to {self.data_dir}")

    def download_underlying_data(self):
        """
        Download and unzip the underlying dataset.
        Underlying dataset: souradippal/student-bag-price-prediction-dataset
        """
        underlying_file = os.path.join(self.data_dir, "underlying.csv")
        if os.path.exists(underlying_file):
            print(f"Underlying data already exists at {underlying_file}. Skipping download.")
            return

        underlying_dataset = "souradippal/student-bag-price-prediction-dataset"
        print(f"Downloading underlying data from {underlying_dataset}...")
        self.api.dataset_download_files(
            underlying_dataset,
            path=self.data_dir,
            force=True,
            quiet=False
        )
        self.unzip_data()
        self.rename_underlying_file()
        print(f"Underlying data has been downloaded and unzipped to {self.data_dir}")

    def rename_underlying_file(self):
        """Rename the underlying CSV file (that is not 'train.csv' or 'test.csv') to 'underlying.csv'."""
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv') and file not in ['train.csv', 'test.csv', 'underlying.csv']:
                src = os.path.join(self.data_dir, file)
                dst = os.path.join(self.data_dir, 'underlying.csv')
                os.rename(src, dst)
                print(f"Renamed {file} to underlying.csv")
                break

    def unzip_data(self):
        """Unzip any zip files found in the data directory and remove the zip files."""
        for file in os.listdir(self.data_dir):
            if file.endswith('.zip'):
                zip_path = os.path.join(self.data_dir, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                os.remove(zip_path)
        print(f"Data in {self.data_dir} has been unzipped.")

    def submit(self, file_path, message):
        """Submit the results to the Kaggle competition."""
        print(f"Submitting {file_path} to {self.competition_name}...")
        submission_result = self.api.competition_submit(file_path, message, self.competition_name)
        print(submission_result)


# Usage example
if __name__ == '__main__':
    COMPETITION_NAME = 'playground-series-s5e2'
    kaggle_connector = KaggleAPIConnector(COMPETITION_NAME)

    # Download competition data
    kaggle_connector.download_data()

    # Download underlying dataset (will skip if already present)
    kaggle_connector.download_underlying_data()

    # Prompt the user whether to submit the results
    submission_path = os.path.join(kaggle_connector.data_dir, 'submission.csv')
    submit_answer = input("Do you want to submit your results? (y/n): ")
    if submit_answer.strip().lower() == 'y':
        if os.path.exists(submission_path):
            submission_message = input("Enter your submission message: ")
            kaggle_connector.submit(submission_path, submission_message)
        else:
            print("Submission file not found.")
    else:
        print("Submission skipped.")