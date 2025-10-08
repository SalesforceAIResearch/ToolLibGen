#!/usr/bin/env python3
from ast import parse
import json
import sys
import os
import datetime
from collections import defaultdict
import numpy as np
import openai
import time
import argparse
import threading
import inspect
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import read_json, save_json


data = read_json("/Users/murong.yue/Desktop/data/math_103k_processed.json")
new_data = data[:20]
save_json(file_path="/Users/murong.yue/Desktop/LLM4ToolMaking/backup/example_for_aggregation.json", data=new_data)