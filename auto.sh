#!/bin/sh

STUDY_NAME="ros_test"
TARGET_FOLDER="./study/${STUDY_NAME}/history"

python example.py -sn $STUDY_NAME -s mimo -np &&
mv "${TARGET_FOLDER}/error.txt" "${TARGET_FOLDER}/error_0.txt" &&
mv "${TARGET_FOLDER}/h.txt" "${TARGET_FOLDER}/h_0.txt"

for IH in `seq 5 5 100`
do
    python example.py -sn $STUDY_NAME -s mimo -ih $IH -np &&
    mv "${TARGET_FOLDER}/error.txt" "${TARGET_FOLDER}/error_${IH}.txt" &&
    mv "${TARGET_FOLDER}/h.txt" "${TARGET_FOLDER}/h_${IH}.txt"
done