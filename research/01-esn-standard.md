# 01. 標準ESN（Echo State Network）の実装仕様

「確立している話」。ここに書いた範囲は教科書的水準で、AIに論文なしで実装させてよい。

## 出典

- Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks*. GMD Report 148.（ESNの原論文）
- Jaeger, H., & Haas, H. (2004). Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication. *Science*, 304, 78–80.
- Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. *Computer Science Review*, 3(3), 127–149.（総説）
- Lukoševičius, M. (2012). A practical guide to applying echo state networks. In *Neural Networks: Tricks of the Trade* (2nd ed.), LNCS 7700, 659–686.（実装の実務ガイド。まずこれ）
- Lu, Z., Hunt, B. R., & Ott, E. (2018). Attractor reconstruction by machine learning. *Chaos*, 28, 061104.（閉ループ予測の標準形）
- Pathak, J., Lu, Z., Hunt, B. R., Girvan, M., & Ott, E. (2017). Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data. *Chaos*, 27, 121102.

## 更新式（リーク付き）

- 入力 $u(t) \in \mathbb{R}^{D_{in}}$、リザバー状態 $r(t) \in \mathbb{R}^{D_r}$、出力 $y(t) \in \mathbb{R}^{D_{out}}$。
- $$ \tilde{r}(t+1) = \tanh\bigl( A\, r(t) + W_{in}\, u(t+1) \bigr) $$
- $$ r(t+1) = (1-\alpha)\, r(t) + \alpha\, \tilde{r}(t+1) $$
- $$ y(t+1) = W_{out}\, \bar{r}(t+1) $$

ここで $A \in \mathbb{R}^{D_r \times D_r}$ は疎なリカレント結合（固定・学習しない）、
$W_{in} \in \mathbb{R}^{D_r \times D_{in}}$ は入力結合（固定）、$\alpha \in (0,1]$ はリーク率。

- 読み出しの特徴ベクトル $\bar{r}$ は、$r$ そのまま、または $r$ と $r^2$（要素ごと二乗）の連結がよく使われる。Pathak らの一連の研究では二乗項入りが標準。対称性の破れ（入力の符号に対する応答の非対称性）を補うためと説明される。
- $A$ は生成後にスペクトル半径（固有値の絶対値最大）$\rho$ でスケーリングする: $A \leftarrow A \cdot \rho / \rho(A)$。
- バイアス入力（定数1の入力チャンネル）を入れると安定しやすい（Lukoševičius 2012）。

## 学習（読み出しだけをリッジ回帰）

1. 教師信号 $u(t)$ を入力として開ループで回し、ウォッシュアウト期間（初期過渡、目安100〜1000ステップ）の状態を捨てる。
2. 残りの状態 $\bar{r}(t)$ を行列 $R$（$T \times D_{feat}$）、対応する目標 $Y$（$T \times D_{out}$、通常は次ステップの入力 $u(t+1)$）に集める。
3. $$ W_{out} = Y^\top R \, \bigl( R^\top R + \beta I \bigr)^{-1} $$ の形で解く（$\beta$ がリッジ正則化）。数値的には正規方程式を直接解かず、SVDやコレスキー分解を使うのが安全。
4. 予測時は閉ループ: $y(t+1)$ を次の入力 $u(t+1)$ として戻す。

## ハイパーパラメータの目安（出発点。タスク依存）

| 項目 | 目安 | 備考 |
|---|---|---|
| リザバー次元 $D_r$ | 300〜2000（小規模系） | 大きいほど記憶と表現力が上がるが過学習と計算量も増える |
| スペクトル半径 $\rho$ | 0.5〜1.5 | カオス予測では1前後〜やや大きめも使う。ESP（echo state property）の十分条件は $\rho<1$ だが、実務では $\rho \gtrsim 1$ でも動くことが多い |
| 入力スケーリング $\sigma$ | 0.1〜1.0 | $W_{in}$ の一様分布 $[-\sigma, \sigma]$ の幅。非線形の効き具合を決める |
| 疎性（$A$ の結合率） | 1〜10% | 密である必要はない |
| リーク率 $\alpha$ | 0.3〜1.0 | 対象の時間スケールに合わせる。遅い系ほど小さく |
| 正則化 $\beta$ | $10^{-8}$〜$10^{-2}$ | 対数スケールで探索。閉ループ安定性に直結する |
| ウォッシュアウト | 100〜1000ステップ | 状態が入力履歴に同期するまで |

## 落とし穴（実装時にAIへ明示すること）

- **入力の正規化**: 教師データは各次元で平均・分散を揃える（または [-1,1] にスケール）。これを怠ると tanh が飽和して性能が崩れる。
- **閉ループの不安定化**: 開ループの検証誤差が小さくても、閉ループでは誤差がフィードバックされて発散しうる。検証は必ず閉ループの予測可能時間で測る。
- **予測可能時間の測り方**: Lorenz系なら誤差が閾値（例: 標準化誤差 0.4）を超えるまでの時間を、最大リアプノフ指数の逆数（リアプノフ時間）で割って報告する。Lu ら (2018) では数リアプノフ時間〜が目安。
- **訓練時ノイズ**: 教師入力に小さなノイズを加えると閉ループが安定しやすい（04参照）。
- **再現性**: $A$ と $W_{in}$ の乱数シードを固定し、生成方法（分布・疎性・スケーリング手順）をコードに残す。

## 最小検証（論文再現）

1. Lorenz系（$\sigma=10, \rho=28, \beta=8/3$、dt=0.02程度）を教師データにする。
2. $D_r=300$〜500、ウォッシュアウト後に閉ループ予測し、予測可能時間が数リアプノフ時間に届くか確認。
3. 長期では気候（アトラクタの形）が再現されるかも見る（Pathak ら 2017 はリアプノフ指数の再現まで確認している）。

## 参考実装

- ReservoirPy（Python）: https://github.com/reservoirpy/reservoirpy — ESN/NVAR両対応。答え合わせの第一候補。
- ReservoirComputing.jl（Julia）: https://github.com/SciML/ReservoirComputing.jl
