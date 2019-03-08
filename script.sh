#!/usr/bin/env bash

for i in `seq 1 151`;
        do
                python3.5 text_recognition.py -i "sample_images/$i.jpg" -east frozen_east_text_detection.pb
        done