# src/ai/solver/environment.py

import sys
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# インポートパスを通すための処理（状況に応じて調整してください）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core import apply_move, get_raw_candidates
from utils import ROWS, COLS, EMPTY, parse_active_word_mask


class KeshimasuEnv(gym.Env):
    """
    漢字ケシマス（国名版）の強化学習(Gymnasium)環境。
    
    - 観測空間 (Observation Space):
        - "board": 8x5 の int8 配列（各マスの文字マスターID）
        - "mask": 20要素の int8 配列（単語リスト内のどのローカルインデックスが使用済みか、0=未使用, 1=使用済）
    - 行動空間 (Action Space):
        - Discrete(20): 最大20個のアクティブ単語候補のうち、何番目の候補を処理（消去）するか
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, initial_board, words_matrix, word_lengths):
        """
        Args:
            initial_board (np.ndarray): 8x5 の初期盤面 (int8)
            words_matrix (np.ndarray): 選定された最大20個の単語マトリックス (int8)
            word_lengths (np.ndarray): 各単語の長さ配列 (int32)
        """
        super().__init__()
        
        # パズルの固定データ（エピソード開始時のリセット用として保持）
        self.initial_board = initial_board.copy()
        self.words_matrix = words_matrix
        self.word_lengths = word_lengths
        self.n_words = len(word_lengths)
        
        # 20個の固定アクションスロット（最大20単語制限に基づく）
        assert self.n_words <= 20, "アクティブな単語リストは最大20個である必要があります。"

        # 1. 観測空間の設定 (Multi-Input Policy対応のためのDict空間)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=50, shape=(ROWS, COLS), dtype=np.int8),
            "mask": spaces.Box(low=0, high=1, shape=(20,), dtype=np.int8)
        })

        # 2. 行動空間の設定 (最大20単語からインデックスを選択)
        self.action_space = spaces.Discrete(20)
        
        # 内部状態の初期化
        self.board = None
        self.word_mask_int = 0  # 1つの整数によるビットフラグ管理 (1 << 20)

    def reset(self, seed=None, options=None):
        """環境を初期状態（エピソード開始時）にリセット"""
        super().reset(seed=seed)
        
        # 盤面とビットマスク変数を初期化
        self.board = self.initial_board.copy()
        self.word_mask_int = 0
        
        return self._get_obs(), {}

    def step(self, action):
        """選択された行動（単語の消去スロット）を実行して状態を遷移"""
        # 現盤面から消去可能な生候補を Numba で爆速全列挙
        cands = get_raw_candidates(self.board, self.words_matrix, self.word_lengths)
        
        # 有効なアクションかどうかの検証
        # 1. actionインデックスが、列挙された有効手(cands)の範囲内であるか
        # 2. その手が指す単語IDが、まだ一度も使われていない（ビットフラグが立っていない）か
        is_valid = False
        cand = None
        if action < len(cands):
            cand = cands[action]
            w_idx = cand[0] # 単語のローカルID (0〜19)
            if not (self.word_mask_int & (1 << w_idx)):
                is_valid = True
        
        # 状態遷移および報酬の計算
        if is_valid:
            # Numbaで盤面消去処理＋重力（下に詰める）処理を爆速実行
            self.board = apply_move(self.board, cand)
            
            # 使用済み単語マスクのビットフラグをONに更新
            self.word_mask_int |= (1 << cand[0])
            
            # 報酬設計: 正しく単語を消せたらプラス報酬
            # （消した文字数に応じたボーナスを付与して大きな文字数の消去を促す）
            word_len = cand[6]
            reward = 1.0 + (word_len * 0.1)
            
            # 終了判定（盤面から全ての文字が消えた＝全消しクリア）
            terminated = (np.sum(self.board) == EMPTY)
            truncated = False
        else:
            # 無効な行動を打ってしまった場合（Action Masking使用時は基本的にここには来ない）
            # 即座にペナルティを与えてエピソードを強制終了
            reward = -2.0
            terminated = True
            truncated = False

        # 手詰まり判定の確認（クリアしておらず、次に打てる有効手が1つもない場合）
        if not terminated:
            next_cands = get_raw_candidates(self.board, self.words_matrix, self.word_lengths)
            has_valid_next = False
            for n_cand in next_cands:
                if not (self.word_mask_int & (1 << n_cand[0])):
                    has_valid_next = True
                    break
            
            # クリアしていないのに次に打てる有効手がない＝手詰まり（敗北終了）
            if not has_valid_next:
                terminated = True
                reward -= 1.0  # 手詰まりペナルティ

        return self._get_obs(), reward, terminated, truncated, {}

    def action_masks(self):
        """
        【重要】sb3-contrib の MaskablePPO が参照する有効行動フラグマスク。
        長さ20の boolean 配列を返し、現時点で選択可能なアクションスロットのみを True にする。
        """
        masks = np.zeros(20, dtype=bool)
        cands = get_raw_candidates(self.board, self.words_matrix, self.word_lengths)
        
        # Numbaが列挙した候補のうち、まだ使われていないスロットのアクションのみを True に解放
        for i in range(len(cands)):
            if i >= 20:
                break
            w_idx = cands[i][0]
            if not (self.word_mask_int & (1 << w_idx)):
                masks[i] = True
                
        return masks

    def _get_obs(self):
        """観測空間（AIに渡すステート）のDictを成形"""
        # 1つの整数値として管理しているビットを、utilsを用いて 0/1 のNumPy配列(固定長20)に高速パース
        mask_array = parse_active_word_mask(self.word_mask_int, n_words=20)
        return {
            "board": self.board.copy(),
            "mask": mask_array
        }

    def render(self):
        """コンソール画面に現在の盤面状態を可視化（デバッグ・人間確認用）"""
        from utils import decode_board
        print("\n--- Current Board State ---")
        decoded = decode_board(self.board)
        for row in decoded:
            print(" ".join(row))
        print(f"Active Mask Bits: {bin(self.word_mask_int)}")
        print("---------------------------")


# ======================================================
# 環境の単体動作確認テスト
# ======================================================
if __name__ == "__main__":
    from utils import encode_board, prepare_words_matrix
    
    print("--- environment.py 単体動作検証 ---")
    
    # 1. テスト用の初期配置文字列
    test_strs = [
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.'],
        ['フ', 'F', 'シ', 'ル', 'F'],
        ['ラ', 'ン', 'ス', 'ウ', 'イ'],
        ['イ', 'ン', 'ド', 'F', 'ジ'],
        ['ド', 'イ', 'ツ', 'F', 'ー'],
        ['ロ', 'シ', 'ア', 'イ', 'ロ']
    ]
    # 2. テスト用の20単語制限内のアクティブワード
    test_words = ['フランス', 'インド', 'ドイツ', 'ロシア', 'フィジー', 'カイロ']
    
    # numpy変換
    b_np = encode_board(test_strs)
    w_mat, w_lens = prepare_words_matrix(test_words)
    
    # 環境のインスタンス化
    env = KeshimasuEnv(b_np, w_mat, w_lens)
    obs, info = env.reset()
    
    print("\n初期化完了。初期盤面表示:")
    env.render()
    
    # 有効アクションマスクの取得テスト
    initial_masks = env.action_masks()
    print(f"利用可能なアクションインデックスマスク (MaskablePPO用):\n{initial_masks}")
    
    # 1手モック実行してみる（最初の有効アクションを選択）
    valid_actions = np.where(initial_masks == True)[0]
    if len(valid_actions) > 0:
        chosen_act = valid_actions[0]
        print(f"\nアクションインデックス {chosen_act} を実行します...")
        next_obs, reward, terminated, truncated, _ = env.step(chosen_act)
        print(f"報酬: {reward}, 終了フラグ: {terminated}")
        env.render()