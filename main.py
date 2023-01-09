import os
from src.pipeline import process_scan

def process_all_folders(data_folder: str, out_folder: str):
    subjects = next(os.walk(data_folder))[1]
    for subject in subjects:
        print("Processing data from folder: ", subject, ".", sep='', end=' ')
        in_path = os.path.join(data_folder, subject)
        out_path = os.path.join(out_folder, subject)
        process_scan(in_path, out_path)
        print("Done.")
    print("All subjects processed.")

if __name__ == "__main__":
    data = "data"
    out = "output"
    process_all_folders(data, out)
