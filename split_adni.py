import os
import pandas as pd
import numpy as np
import time
import random
import math

def write_split(split,filename):
    with open(filename,"w+") as fp:
        for subject in split:
            fp.write(f"{subject}\n")
            for record in subject:
                fp.write(f"{record}\n")

def get_ratio(records):
    ncn, nmci, nad = 0, 0, 0
    for record in records:
        if record[2] == "CN": ncn += 1
        elif record[2] == "MCI": nmci += 1
        elif record[2] == "AD": nad += 1

    ratio = np.array([ncn, nmci, nad]) / (ncn + nmci + nad)
    return ratio

def is_subject_appropriate(subject_records, full):
    if np.all(full == False):
        return True

    # get the majority class in subject_records
    counts = np.array([0, 0, 0])
    for record in subject_records:
        counts = counts + [record[2] == 'CN', record[2] == 'MCI', record[2] == 'AD']
    max_idx = np.argmax(counts)

    # check if the majority class overlaps with the class that is not yet full
    return not full[max_idx]

def get_split(records_by_subject, size, class_ratio):
    class_sizes = class_ratio * size
    added = np.array([0, 0, 0])
    full = np.array([False, False, False])
    split_set = []

    for idx, subject_records in enumerate(records_by_subject):
        if (len(split_set) >= size):
            break

        if not is_subject_appropriate(subject_records, full):
            continue

        for subject_record in subject_records:
            if subject_record[2] == 'CN' and not full[0]:
                split_set.append(subject_record[0] + ".nii")
                added[0] += 1
            if subject_record[2] == 'MCI' and not full[1]:
                split_set.append(subject_record[0] + ".nii")
                added[1] += 1
            if subject_record[2] == 'AD' and not full[2]:
                split_set.append(subject_record[0] + ".nii")
                added[2] += 1

        full = added > class_sizes
        del records_by_subject[idx]
    
    return split_set, added

def print_info(split, added, name):
    print(f"{name} Split: \nSize: {len(split)} \nCN vs. MCI vs. AD {added} \nRatios: {added / np.sum(added)}\n")

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

    # calculate the ratio of different classes
    class_ratio = get_ratio(records)
    print(f"Ratio CN vs MCI vs AD: {class_ratio}")

    train_size, val_size, test_size = split_ratio * len(records)

    # shuffle the array contents
    # records_by_subject = np.array(records_by_subject)
    # np.random.Generator.shuffle(records_by_subject)
    random.seed(1)
    random.shuffle(records_by_subject)

    # dumping the ordering in a file
    t = f"{time.time()}".replace(".","_")
    filename = os.path.join(root_dir, f"subject_order_{t}.txt")
    write_split(records_by_subject, filename)

    # create the split
    # val_set, val_added = get_split(records_by_subject, val_size, class_ratio)
    val_set, val_added = [], [0,0,0]
    test_set, test_added = get_split(records_by_subject, test_size, class_ratio)
    train_set = []
    train_added = np.array([0, 0, 0])
    for subject_records in records_by_subject:
        for record in subject_records:
            train_set.append(record[0] + ".nii")
            train_added = train_added + [record[2] == 'CN', record[2] == 'MCI', record[2] == 'AD']

    # print information about the splits
    print(f"Total Dataset Size: {len(records)}")
    print_info(train_set, train_added, "Train")
    print_info(val_set, val_added, "Validation")
    print_info(test_set, test_added, "Test")

    # train and test file name files
    train_names = os.path.join(root_dir,"train.txt")
    val_names = os.path.join(root_dir,"val.txt")
    test_names = os.path.join(root_dir,"test.txt")

    with open(train_names,"w+") as fp:
        for image_id in train_set:
            fp.write(f"{image_id}\n")

    with open(val_names,"w+") as fp:
        for image_id in val_set:
            fp.write(f"{image_id}\n")

    with open(test_names,"w+") as fp:
        for image_id in test_set:
            fp.write(f"{image_id}\n")

    return train_set, val_set, test_set

def remove_majority(records_by_subject):
    classes = ['CN','MCI','AD']

    # get statistics
    counts = np.array([0,0,0])
    for subject in records_by_subject:
        for record in subject:
            counts = counts + np.array([record[2] == 'CN', record[2] == 'MCI', record[2] == 'AD'])

    # find the difference between the number of most popular class and the middle popular class
    largest = np.argmax(counts)
    smallest = np.argmin(counts)
    middle = 3 - (largest + smallest)
    diff = counts[largest] - counts[middle]
    print(f"Largest class: {classes[largest]}")
    print(f"Middle class: {classes[middle]}")
    print(f"Smallest class: {classes[smallest]}")
    print(f"Removing {diff} scans")

    # bring down the count of the most popular class
    subject_idx = 0
    while(diff > 0):
        subject = records_by_subject[subject_idx]
        inds = np.arange(len(subject))[::-1]
        for idx in inds:
            record = subject[idx]
            if record[2] != classes[largest]:
                continue

            del subject[idx]
            diff -= 1
            if len(subject) == 0:
                del records_by_subject[subject_idx]
            break
        subject_idx = (subject_idx + 1) % len(records_by_subject)

def split_even(root_dir, split_ratio):
    # read the csv file
    labels_file = os.path.join(root_dir,"labels.csv")
    df = pd.read_csv(labels_file)
    records = df.to_numpy()
    subjects = np.unique(records[:,1])

    # cluster records by subject
    records_by_subject = []
    for subject in subjects:
        subject_records = []
        for record in records:
            if subject in record:
                subject_records.append(record)
        records_by_subject.append(subject_records)

    # balance out the dataset a bit
    remove_majority(records_by_subject)
    
    # print out dataset statistics
    counts = np.array([0,0,0])
    for subject in records_by_subject:
        for record in subject:
            counts = counts + np.array([record[2] == 'CN', record[2] == 'MCI', record[2] == 'AD'])
    total = np.sum(counts)
    ratio = counts / total
    print(f"COUNT CN vs MCI vs AD: {counts}")
    print(f"RATIO CN vs MCI vs AD: {ratio}")

    # shuffle the array contents
    random.seed(1)
    random.shuffle(records_by_subject)

    # get the test and train splits
    train_size, test_size = total * split_ratio 
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}\n")

    test_set, test_added = get_split(records_by_subject, test_size, ratio)
    train_set, train_added = [], np.array([0,0,0])
    for subject in records_by_subject:
        for record in subject:
            train_set.append(f"{record[0]}.nii")
            train_added = train_added + np.array([record[2] == 'CN', record[2] == 'MCI', record[2] == 'AD'])

    # print information about the splits
    print_info(train_set, train_added, "Train")
    print_info(test_set, test_added, "Test")

    # train and test file name files
    train_names = os.path.join(root_dir,"train.txt")
    test_names = os.path.join(root_dir,"test.txt")

    with open(train_names,"w+") as fp:
        for image_id in train_set:
            fp.write(f"{image_id}\n")

    with open(test_names,"w+") as fp:
        for image_id in test_set:
            fp.write(f"{image_id}\n")

    return train_set, test_set
        
    