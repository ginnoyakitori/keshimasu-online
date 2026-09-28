# src/utils.py

import json
import os
import numpy as np

# --- 盤面定数（すべてのファイルで共通化） ---
EMPTY = 0
WILD_F = 1
ROWS = 8
COLS = 5
VISIBLE_START_ROW = 3  # 下5段：インデックス3〜7のみ消去可能

# ======================================================
# 1. 漢字ケシマス（国名版）全文字のマスターID定義
# ======================================================
# country_words.json に含まれるすべての文字を一意のIDに変換するリスト
CHAR_LIST = [
    '',   # 0: EMPTY
    'F',  # 1: WILD_F (ワイルドカード)
    # --- 清音 ---
    'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ',
    'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト',
    'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
    'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ',
    'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン', 'ー',
    # --- 濁音 ---
    'ガ', 'ギ', 'グ', 'ゲ', 'ゴ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ',
    'ダ', 'ヂ', 'ヅ', 'デ', 'ド', 'バ', 'ビ', 'ブ', 'ベ', 'ボ',
    'ヴ',
    # --- 半濁音 ---
    'パ', 'ピ', 'プ', 'ペ', 'ポ',
    # --- 小文字（小書きカタカナ） ---
    'ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ッ', 'ャ', 'ュ', 'ョ', 'ヮ'
]

# 高速引換辞書
CHAR_TO_ID = {ch: idx for idx, ch in enumerate(CHAR_LIST)}
ID_TO_CHAR = {idx: ch for idx, ch in enumerate(CHAR_LIST)}

# 万が一リストにない未知の文字が来た場合のフォールバック処理用
def get_char_id(ch):
    """文字を固定マスターID(int8)に変換"""
    if not ch or ch == '.':
        return EMPTY
    return CHAR_TO_ID.get(ch, EMPTY)


# ======================================================
# 2. 国名辞書データのロードとNumba形式変換
# ======================================================
def load_country_words(json_path="country_words.json"):
    """country_words.jsonから国名リストを読み込む"""
    if not os.path.exists(json_path):
        # 存在しない場合のフォールバック（カレントディレクトリ等）
        raise FileNotFoundError(f"辞書ファイルが見つかりません: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_words_matrix(active_words):
    """
    選定された単語リスト（最大20単語）をNumba用の2次元行列(int8)と長さ配列(int32)に変換
    active_words: ['ニホン', 'フランス', ...]
    """
    n_words = len(active_words)
    if n_words == 0:
        return np.zeros((0, 1), dtype=np.int8), np.zeros(0, dtype=np.int32)
        
    max_len = max(len(w) for w in active_words)
    
    words_matrix = np.zeros((n_words, max_len), dtype=np.int8)
    word_lengths = np.zeros(n_words, dtype=np.int32)
    
    for i, word in enumerate(active_words):
        word_lengths[i] = len(word)
        for j, ch in enumerate(word):
            words_matrix[i, j] = get_char_id(ch)
            
    return words_matrix, word_lengths


# ======================================================
# 3. 盤面（2次元リスト ⇔ Numba用numpy配列）の相互変換
# ======================================================
def encode_board(board_strs):
    """
    文字列の2次元リスト(8x5)をNumba高速演算用の np.int8 配列に変換
    ※空白は '' または '.' として扱われます
    """
    arr = np.zeros((ROWS, COLS), dtype=np.int8)
    for r in range(ROWS):
        for c in range(COLS):
            ch = board_strs[r][c] if c < len(board_strs[r]) else ''
            arr[r, c] = get_char_id(ch)
    return arr

def decode_board(board_np):
    """
    Numba用の np.int8 配列(8x5)を人間やクライアント(JS)が読める文字列の2次元リストに逆変換
    """
    board_strs = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            ch = ID_TO_CHAR.get(int(board_np[r, c]), '.')
            row.append(ch)
        board_strs.append(row)
    return board_strs


# ======================================================
# 4. 強化学習(SB3)用：20単語マスク(1 << 20)ビット演算の補助関数
# ======================================================
def get_active_word_mask(used_word_indices):
    """
    すでに使用した単語のローカルインデックス(0〜19)のリストから、
    強化学習ビット演算用の1つの整数マスク値を生成する
    例: [0, 2] -> (1 << 0) | (1 << 2) = 5
    """
    mask = 0
    for idx in used_word_indices:
        if 0 <= idx < 20:
            mask |= (1 << idx)
    return mask

def parse_active_word_mask(mask, n_words=20):
    """
    1つの整数ビットマスクから、どの単語(0〜19)が使用済みかを
    要素数20の 0/1 バイナリNumPy配列(強化学習のObservation用)に分解する
    """
    obs_mask = np.zeros(n_words, dtype=np.int8)
    for i in range(n_words):
        if (mask >> i) & 1:
            obs_mask[i] = 1
    return obs_mask


# ======================================================
# 動作テスト・検証
# ======================================================
if __name__ == "__main__":
    print("--- utils.py 動作確認用テスト ---")
    
    # 1. 盤面エンコード・デコードテスト（濁音・半濁音・小文字入り）
    sample_strs = [
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['フ', 'F', 'シ', 'ル', 'F'],
        ['バ', 'ン', 'グ', 'ラ', 'デ'], # 「バ」「グ」「デ」
        ['シ', 'ュ', 'パ', 'F', 'ジ'], # 「ュ」小文字、「パ」半濁音
        ['ド', 'イ', 'ツ', 'F', 'ー'], # 「ド」
        ['ヴ', 'ェ', 'ネ', 'ツ', 'ィ']  # 「ヴ」「ェ」「ィ」
    ]
    
    encoded = encode_board(sample_strs)
    print("\n[1] 盤面のエンコード結果 (Numbaが読み込む配列):")
    print(encoded)
    
    decoded = decode_board(encoded)
    print("\n[2] デコード（復元）結果:")
    for row in decoded:
        print(" ".join(row))
        
    # 2. 単語マトリックス生成テスト
    sample_words = ['バングラデシュ', 'パナマ', 'ドイツ', 'ブラジル', 'シンガポール']
    matrix, lengths = prepare_words_matrix(sample_words)
    print("\n[3] 単語マトリックス (Numba化):")
    print(matrix)
    print("単語の長さ配列:", lengths)
    
    # 3. マスクのビット演算テスト
    mask_val = get_active_word_mask([0, 2])
    print(f"\n[4] ビットマスク整数値 (0と2フラグON): {mask_val} (バイナリ: {bin(mask_val)})")
    
    obs_array = parse_active_word_mask(mask_val, n_words=5)
    print("強化学習の観測(Observation)用デコード配列:", obs_array)