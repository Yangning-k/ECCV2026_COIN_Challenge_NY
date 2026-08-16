#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(_CODE))
runpy.run_path(str(_CODE / "eval_model.py"), run_name="__main__")
