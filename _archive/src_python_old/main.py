# src/main.py

import os
import sys
import json
import numpy as np

# インポートパスをsrc/直下に通す
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ai/generator')))

from ai.generator.ga_reverse import KeshimasuGAPuzzleGenerator
from utils import encode_board


def main():
    print("======================================================")
    print("  漢字ケシマス（国名版）共進化パズルジェネレータ 起動")
    print("======================================================")

    # --- ハイパーパラメータ設定 ---
    JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../country_words.json'))
    GENERATIONS = 20         # 進化させる世代数（まずは20世代ほどでテスト）
    POPULATION_SIZE = 40     # 1世代あたりのパズル（個体）数
    WORDS_PER_PUZZLE = 6    # 1つのパズルに詰め込む国名数
    MUTATION_RATE = 0.25     # 突然変異確率
    
    if not os.path.exists(JSON_PATH):
        # パスフォールバック（カレントディレクトリ確認用）
        JSON_PATH = "country_words.json"

    print(f"・辞書パス: {JSON_PATH}")
    print(f"・世代数: {GENERATIONS}, 個体数: {POPULATION_SIZE}")
    print(f"・パズル内単語数: {WORDS_PER_PUZZLE}語制限")
    print("------------------------------------------------------")

    # 1. GAジェネレータの初期化
    print(">> 初期ランダム世代（第1世代）のパズルを生成中...")
    ga_engine = KeshimasuGAPuzzleGenerator(
        json_path=JSON_PATH,
        population_size=POPULATION_SIZE,
        num_words_per_puzzle=WORDS_PER_PUZZLE
    )

    # 歴代最高難易度のパズルを記録する変数
    hall_of_fame = {
        "fitness": -1.0,
        "board": None,
        "words": None,
        "generation": 0
    }

    # 2. 共進化メインループ
    for g in range(1, GENERATIONS + 1):
        print(f"\n🌀 [世代 {g} / {GENERATIONS}] パズル評価フェーズ開始...")
        
        # 現世代のパズルをすべてソルバーに解かせ、探索負荷（ノード数）から適応度を算出
        ga_engine.evaluate_population()
        
        # 適応度が高い（難易度が高い）順にソート
        ga_engine.population.sort(key=lambda x: x.fitness, reverse=True)
        
        current_best = ga_engine.population[0]
        current_median = ga_engine.population[POPULATION_SIZE // 2]
        
        print(f"📈 世代 {g} 結果サマリー:")
        print(f"  - 最高難易度スコア（探索ノード数）: {current_best.fitness}")
        print(f"  - 中央値難易度スコア: {current_median.fitness}")
        
        # 歴代最高難易度のパズルが更新されたかチェック（正常に解けたもののみ）
        if current_best.fitness > hall_of_fame["fitness"] and current_best.fitness > 0:
            hall_of_fame["fitness"] = current_best.fitness
            hall_of_fame["board"] = current_best.generated_board_strs
            hall_of_fame["words"] = current_best.active_words
            hall_of_fame["generation"] = g
            print(f"✨ 🏆 歴代最高難易度パズルが更新されました！ (世代 {g})")

        # 最終世代でなければ、選択・交叉・突然変異を行って次世代を生成
        if g < GENERATIONS:
            ga_engine.evolve_one_generation(mutation_rate=MUTATION_RATE)

    print("\n======================================================")
    print("  🎉 共進化ループ完了！ 歴代最高傑作パズルのエクスポート")
    print("======================================================")

    if hall_of_fame["board"] is not None:
        print(f"👑 歴代最難関問題（発見世代: 第{hall_of_fame['generation']}世代）")
        print(f"🔥 ソルバー消費ノード数: {hall_of_fame['fitness']}")
        print(f"📝 採用された国名リスト: {hall_of_fame['words']}")
        print("\n--- 盤面プレビュー ---")
        for row in hall_of_fame["board"]:
            print(" ".join(row))
        print("----------------------")

        # 3. フロントエンド（ブラウザ/Webアプリ）連携用JSONオブジェクトの組み立て
        export_data = {
            "title": f"AI自動生成・国名漢字ケシマス（難易度:{int(hall_of_fame['fitness'])}）",
            "generation": hall_of_fame["generation"],
            "difficulty_score": hall_of_fame["fitness"],
            "words": hall_of_fame["words"],
            "board": hall_of_fame["board"]  # 8行5列の文字列2次元配列
        }

        # 保存先パス（srcと同階層のプロジェクトルート、または適切なWeb公開ディレクトリ）
        export_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../best_puzzle.json'))
        
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n💾 最高傑作パズルをフロントエンド互換JSONとして保存しました！")
        print(f"👉 保存先: {export_path}")
    else:
        print("❌ 警告: 有効な全消しパズルが1つも生成されませんでした。設定（単語数など）を見直してください。")


if __name__ == "__main__":
    main()