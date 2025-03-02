import os
import zipfile
import kaggle


class KaggleAPIConnector:
    def __init__(self, competition_name):
        self.competition_name = competition_name
        self.api = self.authenticate()
        self.data_dir = os.path.join(os.getcwd(), self.competition_name, 'data')
        self.submissions_dir = os.path.join(os.getcwd(), self.competition_name, 'submissions')

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.submissions_dir, exist_ok=True)

    @staticmethod
    def authenticate():
        """Authenticate using the Kaggle API credentials from environment variables."""
        kaggle.api.authenticate()
        return kaggle.api

    def download_data(self):
        """Download and unzip competition data using the Kaggle API."""
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

    def unzip_data(self):
        """Unzip any zip files found in the data directory and remove the zip files."""
        for file in os.listdir(self.data_dir):
            if file.endswith('.zip'):
                zip_path = os.path.join(self.data_dir, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                os.remove(zip_path)
        print(f"Data in {self.data_dir} has been unzipped.")

    def get_last_submission_number(self):
        """Find the last available submission number in the submissions directory."""
        existing_files = [f for f in os.listdir(self.submissions_dir) if
                          f.startswith("submission") and f.endswith(".csv")]

        existing_numbers = [
            int(f.replace("submission", "").replace(".csv", ""))
            for f in existing_files if f.replace("submission", "").replace(".csv", "").isdigit()
        ]

        return max(existing_numbers) if existing_numbers else None

    def get_submission_file(self, submission_number):
        """Retrieve the submission file path based on the chosen submission number."""
        file_name = f"submission{submission_number}.csv"
        file_path = os.path.join(self.submissions_dir, file_name)
        return file_path if os.path.exists(file_path) else None

    def submit(self, file_path, message):
        """Submit the selected submission file to the Kaggle competition."""
        print(f"Submitting {file_path} to {self.competition_name}...")
        submission_result = self.api.competition_submit(file_path, message, self.competition_name)
        print(submission_result)


# -----------------------
# User Interaction for Submission
# -----------------------
if __name__ == '__main__':
    COMPETITION_NAME = 'playground-series-s5e3'
    kaggle_connector = KaggleAPIConnector(COMPETITION_NAME)

    # Download competition data
    kaggle_connector.download_data()

    # Find the last available submission
    last_submission_number = kaggle_connector.get_last_submission_number()

    if last_submission_number is None:
        print("No submission files found. Run the model training first.")
    else:
        print(f"The last available submission is: submission{last_submission_number}.csv")

        # Ask user which submission to upload
        while True:
            user_input = input(
                f"Which submission do you want to upload? (1-{last_submission_number}, or 'n' to skip): ").strip()

            if user_input.lower() == 'n':
                print("Submission skipped.")
                break

            if user_input.isdigit():
                submission_number = int(user_input)
                if 1 <= submission_number <= last_submission_number:
                    submission_path = kaggle_connector.get_submission_file(submission_number)
                    if submission_path:
                        submission_message = input("Enter your submission message: ")
                        kaggle_connector.submit(submission_path, submission_message)
                        break
                    else:
                        print(f"Error: submission{submission_number}.csv not found.")
                else:
                    print(f"Invalid input. Please enter a number between 1 and {last_submission_number}.")
            else:
                print("Invalid input. Please enter a valid submission number or 'n' to skip.")