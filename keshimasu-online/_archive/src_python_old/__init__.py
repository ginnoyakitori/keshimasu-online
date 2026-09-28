# src/ai/generator/__init__.py

import sys
import os

# パスを通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import get_char_id, CHAR_LIST, CHAR_TO_ID

# もし他のファイルが CHAR_IDS というリストをインポートして使っている場合の互換性維持
CHAR_IDS = [get_char_id(ch) for ch in CHAR_LIST]