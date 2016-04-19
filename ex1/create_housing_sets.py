#!/usr/bin/env python3

import random
import os

def create_housing_sets():
    training_filename = 'housing_training.data'
    test_filename = 'housing_test.data'
    if not os.path.isfile(training_filename) or not os.path.isfile(test_filename):
        training_set_size = 400

        lines = list()
        training = list()
        test = list()
        with open('housing.data', 'rb') as f:
            for line in f:
                lines.append(line)
        
        random.shuffle(lines)
        training = lines[0:training_set_size]
        test = lines[training_set_size:]
        del(lines)

        with open(training_filename, 'wb') as f:
            for line in training:
                f.write(line)
        with open(test_filename, 'wb') as f:
            for line in test:
                f.write(line)

if __name__ == '__main__':
    create_housing_sets()
