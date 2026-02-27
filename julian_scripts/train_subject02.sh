#!/bin/bash
python main.py -c configs/subject02_julian_stage1.yaml -m train
python main.py -c configs/subject02_julian_stage1.yaml -m render_depth_sequences
python main.py -c configs/subject02_julian_stage2.yaml -m train
