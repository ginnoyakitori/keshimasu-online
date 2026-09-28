配置先:
  src/multiplayer/runtime-puzzle-generator.js
  src/multiplayer/match-manager.js

他のファイルは変更不要です。

確認:
  node --check src\multiplayer\runtime-puzzle-generator.js
  node --check src\multiplayer\match-manager.js
  npm start

環境変数（任意）:
  RUNTIME_MIN_F=2
  RUNTIME_MAX_F=8
  RUNTIME_GENERATION_ATTEMPTS=100
