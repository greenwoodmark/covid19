# -*- coding: utf-8 -*-
"""
testpath.py
"""

from pathlib import Path
print("File      Path:", Path(__file__).absolute())
print("Directory Path:", Path().absolute())