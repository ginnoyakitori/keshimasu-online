# src/ai/solver/train.py

import sys
import os
import random
import numpy as np
import torch

# sb3-contrib から MaskablePPO 関連モジュールをインポート
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

# インポートパスをプロジェクトルートに通す
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from environment import KeshimasuEnv
from utils import load_country_words, prepare_words_matrix, encode_board, decode_board, WILD_F, ROWS, COLS, VISIBLE_START_ROW


def generate_mock_puzzle(all_words, num_words=5):
    """
    【簡易版・逆再生ジェネレータ】
    辞書から単語をランダムに選び、重力の逆プロセスで有効な初期盤面を生成する。
    ※本番環境では、ここで生成した盤面をGA(遺伝的アルゴリズム)で評価・進化させます。
    """
    # 20単語制限に引っかからないよう安全にサンプリング
    chosen_words = random.sample(all_words, min(num_words, 20))
    
    # 完全に同期したutilsの関数でマトリックス化
    words_matrix, word_lengths = prepare_words_matrix(chosen_words)
    
    # 空の盤面から逆算して文字を積み上げる
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    
    for i in range(len(chosen_words)):
        w_len = word_lengths[i]
        word_chars = words_matrix[i, :w_len]
        direction = random.choice([0, 1])  # 0:横, 1:縦
        
        if direction == 0:  # 横方向に配置
            r = random.randint(VISIBLE_START_ROW, ROWS - 1)
            c_start = random.randint(0, COLS - w_len)
            for c_offset in range(w_len):
                c = c_start + c_offset
                # 既存の文字を1つ上に押し上げる（逆重力）
                for row in range(VISIBLE_START_ROW, r):
                    board[row, c] = board[row + 1, c]
                # 空いた隙間に文字を挿入（たまにワイルドカードFを混ぜる）
                board[r, c] = WILD_F if random.random() < 0.1 else word_chars[c_offset]
        else:  # 縦方向に配置
            c = random.randint(0, COLS - 1)
            r_start = random.randint(VISIBLE_START_ROW, ROWS - w_len)
            # その列の既存文字を一気に上に押し上げる
            for row in range(VISIBLE_START_ROW, r_start):
                board[row, c] = board[row + w_len, c]
            # 空いた縦の隙間に単語を流し込む
            for r_offset in range(w_len):
                board[r_start + r_offset, c] = WILD_F if random.random() < 0.1 else word_chars[r_offset]
                
    return board, words_matrix, word_lengths, chosen_words


def main():
    print("=== [1] 国名辞書データの読み込み ===")
    try:
        # プロジェクトルートにあるcountry_words.jsonを読み込み
        json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../country_words.json'))
        all_country_words = load_country_words(json_path)
        print(f"総単語数: {len(all_country_words)} 件をロードしました。")
    except FileNotFoundError as e:
        print(e)
        print("テスト用のフォールバック辞書を使用します。")
        all_country_words = ['フランス', 'ブラジル', 'インド', 'ドイツ', 'ロシア', 'フィジー', 'カイロ', 'イタリア', 'ジャパン']

    print("\n=== [2] 訓練用パズル盤面の生成と環境の構築 ===")
    # 5つの国名を組み込んだ「絶対に解ける」初期パズルを1つ生成
    init_board, w_mat, w_lens, active_words = generate_mock_puzzle(all_country_words, num_words=5)
    
    # Gymnasium環境のインスタンス化
    env = KeshimasuEnv(init_board, w_mat, w_lens)
    
    print("初期訓練用盤面のプレビュー:")
    env.reset()
    env.render()
    print(f"出現する単語リスト (ローカルID順): {active_words}")

    print("\n=== [3] MaskablePPO モデルの初期化 ===")
    # デバイス設定 (CUDAが使えるならGPU、なければCPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # 多入力（Dict型観測空間）に対応した「MultiInputPolicy」を適用
    # Action Masking を有効にするために MaskablePPO を使用
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device=device
    )

    print("\n=== [4] AIのトレーニング開始 ===")
    # 高速なJITコアとAction Maskingにより、少ないステップでも効率よく学習します
    total_timesteps = 10000
    model.learn(total_timesteps=total_timesteps)
    print("トレーニングが正常に完了しました！")

    # 学習済みモデルの保存
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, "keshimasu_ppo_model")
    model.save(model_path)
    print(f"モデルを保存しました: {model_path}.zip")

    print("\n=== [5] 学習済みAIによるテストプレイ（推論） ===")
    # 環境をリセットしてテスト開始
    obs, info = env.reset()
    env.render()
    
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        step_count += 1
        # モデルから有効な行動のマスクを取得
        action_masks = get_action_masks(env)
        
        # マスクを適用して次の行動を予測 (deterministic=True で決定論的な最適手を選択)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # 選択されたアクションに対応する単語名を確認
        # 有効手(cands)のaction番目の要素から単語IDを取得
        from core import get_raw_candidates
        cands = get_raw_candidates(env.board, env.words_matrix, env.word_lengths)
        chosen_word_name = "未知の手/手詰まり"
        if action < len(cands):
            word_idx = cands[action][0]
            if word_idx < len(active_words):
                chosen_word_name = active_words[word_idx]

        print(f"\n🎬 AIの手番 [{step_count}手目]: 行動スロットID {action} -> 「{chosen_word_name}」を選択")
        
        # 環境を進める
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 盤面の描画
        env.render()
        
        done = terminated or truncated
        if terminated:
            # 盤面が全てEMPTY(0)なら全消しクリア成功
            if np.sum(env.board) == 0:
                print(f"🎉 祝！AIがパズルを完璧に全消しクリアしました！ (合計報酬: {total_reward:.2f})")
            else:
                print(f"❌ 手詰まり、または無効な手により途中で終了しました。 (合計報酬: {total_reward:.2f})")


if __name__ == "__main__":
    main()