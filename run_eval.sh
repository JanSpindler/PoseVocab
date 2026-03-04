#!/bin/bash
PYTORCH_ALLOC_CONF=expandable_segments:True xvfb-run -a python julian_eval.py
