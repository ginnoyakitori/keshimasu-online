# 国名ケシマス 自動マッチング対戦

## 起動
```bat
npm install
npm start
```

1. 2つのブラウザで `http://localhost:3000` を開く
2. 両方で名前を入力し「対戦する」を押す
3. サーバーが待機中の2人を自動マッチング
4. マッチ成立直後に同じ問題を配信
5. 3秒カウントダウン後に同時開始

生成済み問題は `data/ga_reverse`、国名リストは `data/words/countries.json` を使用します。
