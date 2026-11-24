# 03. 並列／局所リザバー（時空間カオス・PDE向け）

設計メモの「タイルごとに小さなリザバーを置き、隣同士を結合する」の根拠と実装詳細。

## 起点論文（確立している話）

- Pathak, J., Hunt, B. R., Girvan, M., Lu, Z., & Ott, E. (2018). Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach. *Physical Review Letters*, 120, 024102. https://doi.org/10.1103/PhysRevLett.120.024102
  - 対象: Kuramoto–Sivashinsky (KS) 方程式。空間を分割し、各領域に1つのリザバーを割り当てる並列方式で、任意の広さの時空間カオスを予測できることを示した。
- Pathak, J., et al. (2017). Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data. *Chaos*, 27, 121102.（低次元版の基礎）

## 方式（Pathak ら 2018 の構成）

- 空間領域を重なりのある小区画に分ける。各リザバーは:
  - **入力**: 自分の区画の状態 ＋ 隣接区画の「ハロー（のりしろ）」部分の状態。
  - **出力**: 自分の区画の次ステップの状態だけ。
- 各リザバーは独立に学習できる（リッジ回帰を区画ごとに解く）。全体を1つの巨大リザバーにしないので、次元爆発を避けられる。
- 読み出しには $r$ と $r^2$ の連結特徴を使う（01参照）。入力結合 $W_{in}$ は $[-\sigma, \sigma]$ の一様分布、$A$ は低次数の Erdős–Rényi グラフをスペクトル半径 $\rho$ にスケーリング。
- ハローの幅は、情報が1ステップで伝わる距離（CFL的な伝播速度×dt）をカバーするのが目安。狭すぎると区画境界で破綻する。

## TAYUTAUへの写し方

- 2D流体なら、格子をタイル分割し、各タイルのリザバーが「自タイル＋隣接1〜数セルの帯」を入力にする。これは物理層のステンシル計算と同じ発想で、WGSLのタイル処理とも相性が良い。
- タイル境界の不連続が目立つ場合は、物理層（粗いソルバー）側で平滑化するか、ハイブリッド（04）で吸収する。

## 注意: 部分観測・低次元化データでは大型RCが不安定になりうる（1本の論文の主張だが重要）

- Vlachas, P. R., Pathak, J., Hunt, B. R., Sapsis, T. P., Girvan, M., Ott, E., & Koumoutsakos, P. (2020). Backpropagation Algorithms and Reservoir Computing in Recurrent Neural Networks for the Forecasting of Complex Spatiotemporal Dynamics. *Neural Networks*, 126, 191–217. https://arxiv.org/abs/1910.05266
  - 全状態が観測できる場合は RC が BPTT 系 RNN より高精度・高速。
  - しかし **reduced order（低次元化された）データでは、大型RCは不安定で発散しやすく、BPTT系の方が優れた**、と報告している。
- 含意: 設計メモの「PCA/オートエンコーダで圧縮 → 潜在状態をリザバーで予測」は、まさにこの弱点に触れる可能性がある。対策として:
  - 潜在次元を小さくしすぎない／圧縮誤差を学習データ側にも含める（再構成誤差込みで学習）。
  - NVAR（02）のような決定的で小さなモデルの方が安定する場合がある。両方試して比較する。
  - 物理層とのハイブリッド＋射影（04）で暴走を抑える。

## コードについて

- 原著者による公式コードリポジトリは、2026-11-24 時点の確認では見つからなかった（05参照）。答え合わせは ReservoirPy 等の一般実装＋Lorenz/KS の再現で行う。
