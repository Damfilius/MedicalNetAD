import os
import pandas as pd
import numpy as np
import time
import random

def write_split(split,filename):
    with open(filename,"w+") as fp:
        for subject in split:
            fp.write(f"{subject}\n")
            for record in subject:
                fp.write(f"{record}\n")

def split_dataset(root_dir, split_ratio):
    labels_file = os.path.join(root_dir,"labels.csv")
    df = pd.read_csv(labels_file)
    records = df.to_numpy()
    subjects = np.unique(records[:,1])
    records_by_subject = []
    for subject in subjects:
        cond = [subject in record for record in records]
        subject_records = records[cond]
        records_by_subject.append(subject_records)

    # shuffle the array contents
    # records_by_subject = np.array(records_by_subject)
    # np.random.Generator.shuffle(records_by_subject)
    random.shuffle(records_by_subject)

    # dumping the ordering in a file
    t = f"{time.time()}".replace(".","_")
    filename = os.path.join(root_dir, f"subject_order_{t}.txt")
    write_split(records_by_subject, filename)

    # create the split
    train_count = split_ratio * len(records)
    train_set = []
    test_set_records = records_by_subject.copy()
    remove_idx = []
    enumerate_records_by_subject = enumerate(records_by_subject)
    for idx, subject_records in enumerate_records_by_subject:
        if (len(train_set) >= train_count):
            break

        # if there is little left to fill up the train_set then do not add a large subject
        if (train_count - len(train_set) < len(subject_records)/4):
            continue

        # get the image id and append ".nii" to it
        for subject_record in subject_records:
            train_set.append(subject_record[0] + ".nii")
            remove_idx.append(idx)
        
        # whatever is left-over goes to test_set
        test_set_records.remove(subject_records)

    # fetching and storing only the file names
    test_set = []
    for subject_records in test_set_records:
        for record in subject_records:
            test_set.append(record[0] + ".nii")

    # train and test file name files
    train_names = os.path.join(root_dir,"train.txt")
    test_names = os.path.join(root_dir,"test.txt")

    with open(train_names,"w+") as fp:
        for image_id in train_set:
            fp.write(f"{image_id}\n")

    with open(test_names,"w+") as fp:
        for image_id in test_set:
            fp.write(f"{image_id}\n")

    return train_set,test_set
        
    