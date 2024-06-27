#!/bin/bash

python -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt
pip install sentencepiece