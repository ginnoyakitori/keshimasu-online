# 国名ケシマス 2人対戦

## 配置
このフォルダの内容を既存の `keshimasu-ai` ルートへコピーしてください。
生成済み問題は `data/ga_reverse`、国名リストは `data/words/countries.json` を使用します。

## 起動
```bat
npm install
npm start
```
ブラウザで `http://localhost:3000` を開き、片方が部屋を作成し、もう片方がコードで参加します。

## ルール
- 両者に同じ初期盤面を配信
- 各プレイヤーは独立した盤面を操作
- 縦または横の2〜5マスを選択
- Fはワイルドカード
- 同じ国名は1問で1回だけ
- 下5段から開始
- サーバーが正誤、重力、時間、勝敗を確定

## 環境変数
- `PORT`
- `PUZZLE_DIR`
- `COUNTRY_FILE`
- `MATCH_DIR`
