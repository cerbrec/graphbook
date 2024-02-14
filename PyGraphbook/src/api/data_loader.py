import os
from src.api import api_util
import logging


ALLOWED_FILE_NAMES = ["weight", "bias"]


# Recursively list all files in directory
def list_files(directory):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    return files


def upload_weights_with_retry(dir_path: str, num_tries=10):
    full_paths = api_util.get_all_full_paths()
    any_left = True

    number_tries_remaining = num_tries

    while any_left:
        number_tries_remaining -= 1
        if number_tries_remaining == 0:
            logging.error("Ran out of tries")
            break

        any_left = False
        for file in list_files(dir_path):
            is_allowed = False
            for allowed_file_type in ALLOWED_FILE_NAMES:
                if allowed_file_type in file:
                    is_allowed = True
                    break

            if not is_allowed:
                continue

            if file.removeprefix(".") in full_paths:
                continue

            # print(f"python3 /Users/drw/cerbrec/graph-compute/scripts/upload_weights.py --weight_file_path {file}")
            os.system(f"python3 /Users/drw/cerbrec/graph-compute/scripts/upload_weights.py --weight_file_path {file}")

            print(file)
            any_left = True

        full_paths = api_util.get_all_full_paths()


if __name__ == "__main__":
    paths = [
        "./public/donut-base-finetuned-docvqa",
        # "./public/flan-t5-base",
        # "./public/flan-t5-large",
        # "./public/tinyllama-1.1B-Chat-v0.2",
        # "./public/phi-2",
        # "./public/Mistral-7B-Instruct-v0.2"
        ]

    print(paths)
    for path in paths:
        upload_weights_with_retry(path)
