# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:07:55 2017

@author: zgh
"""

import argparse
import subprocess
import numpy as np
from datetime import datetime
import json
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
print(all_obj_cats)
fin.close()

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'part_belong_to_object.json'), 'r'))
print(all_cats)
