#!/bin/bash
python main.py -c configs/0165_08_stage1.yaml -m train
python main.py -c configs/0165_08_stage1.yaml -m render_depth_sequences
python main.py -c configs/0165_08_stage2.yaml -m train
