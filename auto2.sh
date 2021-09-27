#!/bin/sh

STUDY_NAME1="ros_test_angle"
STUDY_NAME2="ros_test_pos"

python example.py -sn $STUDY_NAME1 -s mimo -np &&
python example.py -sn $STUDY_NAME2 -s mimo -np
