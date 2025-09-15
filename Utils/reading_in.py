import os
import pandas as pd
import cv2

def read_in_csv(top_dir, search_term=""):

    if search_term == "":
        # Get list of csv filenames under 'data/segments'
        filepaths = get_filepaths(dir_name=top_dir)
    else:
        filepaths = get_filepaths_with_regex(dir_name=top_dir, search=search_term)

    # Join csv filename and dir path together
    for i in range(len(filepaths)):
        filepaths[i] = os.path.join(top_dir, filepaths[i])

    # Read in all csv files and store in df
    df = pd.concat(map(pd.read_csv, filepaths))

    return df

def read_in_images(top_dir):
    list_of_images = []

    # Get list of filenames in top_dir
    filepaths = get_filepaths(dir_name=top_dir)
    filepaths.sort()

    for i in range(len(filepaths)):
        filepaths[i] = os.path.join(top_dir, filepaths[i])
        # Load image and sort colour channel order
        img = cv2.imread(filepaths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Append image to List
        list_of_images.append(img)

    return list_of_images

def get_filepaths(dir_name):
    fpaths = [f for f in os.listdir(dir_name)]
    return fpaths

def get_filepaths_with_regex(dir_name, search):
    fpaths = [f for f in os.listdir(dir_name) if f.startswith(search)]
    return fpaths

