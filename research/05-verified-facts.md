# 05. 設計メモの[未確認]事項 — 確認結果

2026-11-24 に確認した分。出典付き。

## 1. Cloudflare Workers 静的アセットの1ファイル上限

- **確認済み: 25 MiB（Free / Paid 共通）**。設計メモの記憶は正しい。
- 出典: Cloudflare Workers 公式ドキュメント「Limits」— Individual Static Asset file size: 25 MiB（Free / Paid とも）。
  https://developers.cloudflare.com/workers/platform/limits/
- 含意: 重みファイルは1ファイル25MiB未満に収める。超える場合は分割するか R2 等に置く。疎な結合＋半精度で小さく保つ方針は妥当。

## 2. Pathak らの研究コードの公開状況

- **原著者による公式コードリポジトリは、今回の確認では見つからなかった。**
  - Pathak et al. (2018, PRL 120, 024102) の論文自体は APS / arXiv 系で入手可能。
  - 第三者による再実装や講義ノートは存在するが、原著者の公式コードとしての確認は取れていない。
- 含意: 「公開されていた」という記憶は保留。答え合わせの基準は ReservoirPy / ReservoirComputing.jl と、Lorenz・KS の再現結果に置く。

## 3. Zig の WebGPU 対応（最新状況）

- **「ブラウザ向けには弱い」という設計メモの評価は、現時点でも概ね妥当。**
  - Zig の WebGPU は Mach（hexops/mach）の `mach/gpu` が中心だった。Wasm ターゲットではブラウザのネイティブ WebGPU を呼ぶ設計だが、README 上でもブラウザ実装は未完項目に挙がっていた。
  - その後 Mach プロジェクト自体が deprecated（終了）扱いとなり、後継フォーク（例: necromach-gpu-dawn）が Dawn バイナリの維持を引き継いでいる状況。エコシステムは依然としてネイティブ（Dawn）寄り。
  - 出典: https://github.com/mitchellh/mach / https://github.com/Senryoku/necromach-gpu-dawn（2026-11-24 確認）
- 含意: 当初の判断（まず Babylon.js + WebGPU、必要なら Rust/wgpu）で問題なし。

## 4. 第2リザバーによる補正が発散を防げるか

- 文献では未確認（04参照）。**実験で確かめる項目**のまま。
