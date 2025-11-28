import os
import pandas as pd
import numpy as np
import time
import random

def write_split(split,filename):
    with open(filename,"w+") as fp:
        for subject in split:
            for record in subject:
                fp.write(f"{record}")

def split_dataset(root_dir, split_ratio):
    labels_file = os.path.join(root_dir,"labels.csv")
    df = pd.read_csv(labels_file)
    records = df.to_numpy()
    subjects = np.unique(records[:,2])
    records_by_subject = np.array([])
    for subject in subjects:
        subject_records = np.where(subject in records, records)
        records_by_subject = np.append(records_by_subject,subject_records)

    # shuffle the array contents
    np.random.Generator.shuffle(records_by_subject)
    t = f"{time.time()}".replace(".","_")
    filename = os.path.join(root_dir, f"subject_order_{t}.txt")
    write_split(records_by_subject, filename)

    # create the split
    train_count = split_ratio * len(records)
    train_set = []
    remove_idx = []
    enumerate_records_by_subject = enumerate(records_by_subject)
    for idx, subject_record in enumerate_records_by_subject:
        if (len(train_set) >= train_count):
            break

        # if there is little left to fill up the train_set then do not add a large subject
        if (train_count - len(train_set) < len(subject_record)/4):
            continue

        # get the image id and append ".nii" to it
        train_set.append(subject_record[0] + ".nii")
        remove_idx.append(idx)

    # whatever is left-over
    test_set_subjects = np.delete(records_by_subject,remove_idx,axis=0)
    test_set = [record[0] + ".nii" for record in test_set_subjects]

    train_names = os.path.join(root_dir,"train.txt")
    test_names = os.path.join(root_dir,"test.txt")

    with open(train_names,"w+") as fp:
        for image_id in train_set:
            fp.write(f"{image_id}")

    with open(test_names,"w+") as fp:
        for image_id in test_set:
            fp.write(f"{image_id}")

    return train_set,test_set
        
    