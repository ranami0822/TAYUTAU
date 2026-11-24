# リザバーコンピューティング調査レポート
## Reservoir Computing：基礎理論、物理実装、日本の研究動向、最新動向と課題

作成日：2026年11月24日（調査時点）
言語：日本語
調査方法：学術論文・大学／公的機関のプレスリリース・出版社ページを中心にウェブ検索で確認した。以下、すべての主要な主張に出典（論文タイトル、著者、年、掲載誌またはarXiv番号、URL）を付す。URLは検索で実際に取得したものをそのまま記載する。企業発表については「企業発表であり査読論文ではない」ことを明記する。

---

## Summary（要約）

リザバーコンピューティング（reservoir computing, RC）は、再帰型ニューラルネットワーク（RNN）の訓練困難を回避するために2001–2002年頃に独立に提案された枠組みである。大きな再帰結合系（リザバー）をランダムに生成して固定し、入力で駆動して得られる高次元・非線形な過渡状態から、線形読み出し（readout）だけを線形回帰などで学習する。そのため学習は高速・安定で、閉形式に近い解法が使える反面、性能は固定されたリザバーの力学（メモリと非線形性のトレードオフ、echo state propertyの成立など）に強く依存する。

この「リザバーを訓練しない」という性質が、光、スピントロニクス、メモリスタ、電子回路、力学系・ソフトボディ、培養神経細胞、量子系など、多様な物理系をそのまま計算資源として使う物理リザバーコンピューティング（physical reservoir computing, PRC）への道を開いた。日本は、田中剛平・廣瀬明らの理論・光／複素数値系、中嶋浩平（東京大学）のソフトボディ・量子・生態系リザバー、藤井啓祐との量子リザバー、常木らを含むスピントロニクス連携、角ら（東北大学など）の生体神経リザバー、浅井哲也（北海道大学）とTDKのアナログリザバーチップなど、理論からデバイス・ロボティクスまで層が厚い。

2023–2026年には、(1)新興エレクトロニクスによるPRCの総説（Nature Electronics, 2024）、(2)脳オルガノイドをリザバーとするBrainoware（Nature Electronics, 2023）、(3)培養神経細胞の汎化フィルタ機能（PNAS, 2023、日本のグループ）、(4)クラゲサイボーグ（Nature Communications, 2025、日本）、(5)TDKと北海道大学によるアナログリザバーAIチップ試作とCEATEC 2025出展（2025、企業発表）など、基礎・バイオ・商用化前段の動きが並行している。一方で、スケーラビリティ、逐次処理の非並列性、物理デバイスのばらつき・温度依存性による再現性、タスク依存性、Transformer／LSTMに対する表現力の限界は、批判的文献でも繰り返し指摘されている。RCは「万能な深層学習の代替」ではなく、「時系列・力学系タスクに特化し、学習コストと（物理実装では）推論の省電力性を引き換えに表現力を限定する」枠組みと理解するのが、文献に最も忠実である。

---

## Findings

### 1. 基礎理論：ESNとLSM、読み出し層だけで済む仕組み、echo state property、評価概念

#### 1.1 二つの起源：Echo State NetworkとLiquid State Machine

**Echo State Network（ESN）――Jaeger。**
Herbert Jaegerは技術報告書で、RNNの状態が入力履歴の「エコー（echo）」として理解できる条件を定式化し、出力ユニットへの重みだけを修正する構成的学習法を提案した。検索で確認した報告書本文は「only the weights to output units」を修正すると明記し、原報告をJaeger (2001) *GMD Report 148, German National Research Center for Information Technology* と同定している。
- 出典：Jaeger, H. (2001) “The ‘echo state’ approach to analysing and training recurrent neural networks”, GMD Report 148（2010年訂正版で確認）。 https://www.scribd.com/document/487953134/The-echo-state-approach-to-analysing-and-training-1

**Liquid State Machine（LSM）――Maassら。**
Wolfgang Maass、Thomas Natschläger、Henry Markramは、安定状態への収束を要求せず、高次元力学系の過渡ダイナミクス自体を「universal analog fading memory」として使い、読み出しニューロンが現在状態から現在・過去の入力情報をリアルタイムに抽出できるとした。生物の神経回路（スパイキング・ネットワーク）を強く意識した定式化であり、separation property（異なる入力履歴を異なる状態へ分離する性質）とapproximation property（読み出しが目標関数を近似できる性質）が普遍的計算能力の条件として議論される。
- 出典：Maass, W., Natschläger, T. & Markram, H. (2002) “Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations”, *Neural Computation* 14, 2531–2560. DOI:10.1162/089976602760407955 https://psycnet.apa.org/record/2002-06457-001 ／全文PDF http://www.lsm.tugraz.at/papers/lsm-nc-130.pdf

**統一概念としての「リザバーコンピューティング」。**
Lukoševičius & Jaegerの総説は、ESNとLSMが「RNN（リザバー）をランダムに生成し、読み出しだけを訓練する」という新しいRNN訓練パラダイムを導入し、それが「reservoir computing」として知られるようになり、古典的な全結合訓練RNNを多くのタスクで上回った、と位置づける。
- 出典：Lukoševičius, M. & Jaeger, H. (2009) “Reservoir computing approaches to recurrent neural network training”, *Computer Science Review* 3(3), 127–149. DOI:10.1016/j.cosrev.2009.03.005 https://cenl.ucsd.edu/CompNeuro/Readings/week13/Lukosevicius-Jaeger+Reservoir-computing-recurrent-neural-network+CompSciRev+2019.pdf
- 補足出典：Verstraeten, D., Schrauwen, B., D’Haene, M. & Stroobandt, D. (2007) “An experimental unification of reservoir computing methods”, *Neural Networks* 20, 391–403. ESN、LSM、Backpropagation-Decorrelationを実験的に統一し、「reservoir computing」という総称の定着に寄与した論文として総説類で引用される。 https://www.sciencedirect.com/science/article/pii/S089360800700038X

ESNとLSMの違いは出発点にある。ESNはレート型（sigmoid/tanh）ユニットの離散時間RNNと工学的時系列学習を、LSMはスパイキング神経回路と神経計算の理論を主眼とする。しかし共通構造――(i)固定された高次元非線形力学系、(ii)入力による駆動、(iii)学習可能な読み出し――は同一であり、現在のRC文献は両者を同一パラダイムの二つの起源として扱う。これは上記Lukoševičius & Jaeger (2009)およびTanaka et al. (2019)（後述）のいずれも明記するところである。

#### 1.2 なぜ学習が読み出し層だけで済むのか

仕組みは線形代数に帰着する点が核心である。

1. 入力列 $u(t)$ が固定されたリザバーを駆動し、各時刻に高次元状態 $x(t)$ が得られる。リザバーは入力履歴の非線形変換を「勝手に」行う特徴抽出器として働く。
2. 目標出力 $y(t)$ を $x(t)$ の線形結合で表す問題にすれば、学習は線形回帰（通常はリッジ回帰による正則化付き最小二乗）になる。訓練される重みに循環依存がないため、BPTT（backpropagation through time）のような勾配の消失・爆発、局所解、逐次的な勾配伝播を回避できる。
3. Jaeger & Haasはこの点を明示し、ESNが「only the synaptic connections from the RNN to the output readout neurons are modified by learning」であり、「training an ESN becomes a simple linear regression task」だと述べている。
   - 出典：Jaeger, H. & Haas, H. (2004) “Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication”, *Science* 304(5667), 78–80. DOI:10.1126/science.1091277 https://www.ai.rug.nl/minds/uploads/1912_JaegerHaas04.pdf ／書誌 https://pubmed.ncbi.nlm.nih.gov:443/15064413/
4. 同論文の成果として、Mackey–Glassカオス時系列予測で従来法より精度が2400倍改善（精緻化した変種での値として本文に記載）、無線通信チャネルの等化で信号誤り率が2桁改善したと報告された。ただしこれは特定ベンチマークでの結果であり、汎用的な優位性の証明ではない点に注意が必要である（§6参照）。

Tanakaらの総説も同じ構造を要約する：リザバーは入力を高次元空間へ写像し、読み出しがその高次元状態からパターン解析を行い、「reservoir is fixed and only the readout is trained with a simple method such as linear regression」、ゆえに他のRNNに比べ学習が速く訓練コストが低い。
- 出典：Tanaka, G. et al. (2019) “Recent advances in physical reservoir computing: A review”, *Neural Networks* 115, 100–123. DOI:10.1016/j.neunet.2019.03.005, arXiv:1808.04962 https://arxiv.org/abs/1808.04962?context=cs

#### 1.3 Echo state property（ESP）とfading memory

ESPとは、同じ入力列で駆動すれば、異なる初期状態から出発したリザバー状態軌道が漸近的に一致する性質、すなわち現在の状態が初期条件ではなく入力履歴の関数として一意に定まる性質である。これが成り立たなければ、読み出しは「入力」ではなく「任意の初期状態」を学習してしまう。

- 定義と議論の原典はJaeger (2001)（上記）。
- 実務的な目安としてスペクトル半径（リザバー重み行列の最大固有値絶対値）を1未満にスケーリングすることが広く使われるが、これは必要条件／十分条件との関係が入力に依存するヒューリスティクスであり、厳密な保証ではない。IEEE Technology Navigatorの解説も、十分条件は最大特異値に関する条件であり、スペクトル半径＜1は「standard working heuristic rather than a guarantee」と明記する。
  - 出典（解説）：IEEE Technology Navigator, “Echo state networks”. https://technav.ieee.org/topic/echo-state-networks/
- 再検討論文として、検索結果でも言及される Yildiz, Jaeger & Kiebel (2012) “Re-visiting the echo state property”, *Neural Networks* 35, 1–9 が、ESPの条件を入力依存性まで含めて再定式化した代表作である（書誌はGallicchioらの講義資料経由で確認）。
- LSM側の対応概念はfading memoryとpointwise separationである。Maass et al. (2002)（上記）は、過去入力の情報が徐々に消える（fading）一方で、 saliently異なる入力には敏感である（ただし非カオス的である）ことが必要だと論じる。

主要ハイパーパラメータは、リザバーサイズ、結合の疎性、スペクトル半径、入力スケーリング、leaky-integrator型ではリーク率であり、これらが実効的な時間スケールと安定性（カオスの縁 edge of chaos への近さ）を決める。

#### 1.4 評価概念：メモリ容量、カーネル品質、情報処理容量

**（a）短期／線形メモリ容量（memory capacity, MC）。**
Jaegerが導入した概念で、現在のリザバー状態から $k$ ステップ前の入力 $u(t-k)$ を線形読み出しでどれだけ再構成できるかを全遅延について合計したもの。$N$ ノードのリザバーでは $MC \le N$ であり、線形リザバーで上限に達しうる一方、非線形性を強めると一般に線形メモリは低下する――というメモリと非線形性のトレードオフが理論研究の中心テーマになった。
- 出典：この上限とトレードオフの要約は、Inubushi & Yoshimura系の議論を引く総説的論文 “Reservoir Computing Beyond Memory-Nonlinearity Trade-off”, *Scientific Reports* (2017) の序論に明記される（「Jaeger conducted pioneering studies on short-term linear memory capacity (MC) and showed theoretically that MC ≤ N」）。 https://www.nature.com/articles/s41598-017-10257-6?error=cookies_not_supported&code=0f4426a6-2635-4066-9577-1a0368dc4a3d
- 同義の定義（ランダム入力で $i$ ステップ前の入力を再現する読み出しの相関の総和、上限 $N$）はVerstraeten et al. (2007)経由の解説にも現れる。 https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00068/full

**（b）カーネル品質（kernel quality）と汎化ランク（generalization rank）。**
リザバーを「時間情報を扱えるカーネル」とみなす見方である。異なる入力を異なる状態へ分離する能力（kernel quality rankは高い方がよい）と、似た入力を似た状態に保つ能力（generalization rankは低い方がよい）の差が計算能力（computational ability）の指標として、Legenstein & Maass (2007) 系の研究で提案され、Verstraeten et al. (2007) で実験的に使われた。分離を強めすぎると汎化が失われるため、最適動作点はそのバランスにある。
- 出典：Soriano, M.C. et al. (2015) “Minimal approach to neuro-inspired information processing”, *Frontiers in Computational Neuroscience* 9, 68（kernel quality／generalization rank／memory capacityの定義をまとめて解説）。 https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00068/full
- 原実験出典：Verstraeten et al. (2007)（上記ScienceDirect）。

**（c）情報処理容量（information processing capacity, IPC）。**
Dambreらは、直交多項式基底（Legendre多項式など）で張られる目標関数群について、各関数の再構成容量を定義し、その総和（total capacity）が、fading memory条件の下で線形独立な状態変数の数 $N$ で上界され、それに等しくなりうることを示した。これにより「線形メモリ」「非線形変換」など異なる処理モードへの容量配分と、その普遍的トレードオフを一つの枠組みで測れるようになった。物理リザバーの性能比較で現在も標準的に使われる指標である。
- 出典：Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. (2012) “Information processing capacity of dynamical systems”, *Scientific Reports* 2, 514. DOI:10.1038/srep00514 https://pubmed.ncbi.nlm.nih.gov:443/22816038/

**（d）その他。** 分離性／近似性（LSM）、最大Lyapunov指数や条件付きLyapunov指数による安定性評価（負であれば共通入力同期＝ESPと対応）、NARMAタスク、Mackey–Glass／Lorenz／Kuramoto–Sivashinsky予測、音声数字認識（TI-46等）などのベンチマークが、分野共通の物差しとして使われてきた。Pathakらの大規模時空間カオス予測はその到達点の一つである（§2.6）。

---

### 2. 物理リザバーコンピューティング（PRC）

PRCは、リザバーを数値シミュレーションされたRNNではなく、物理系のダイナミクスそのもので置き換える。訓練しないリザバーであれば、デバイスのばらつきや制御困難性を「欠陥」ではなく計算資源として許容できる、という発想が核にある。Cucchiらのチュートリアルは、RCが「minimal device-to-device variability and control over each unit」といった通常のハードウェアNNの要求を回避できると明言する一方、「well-performing reservoir for a specific task」を用意することは見かけほど容易ではないとも警告する。
- 出典：Cucchi, M., Abreu, S., Ciccone, G., Brunner, D. & Kleemann, H. (2022) “Hands-on reservoir computing: a tutorial for practical implementation”, *Neuromorphic Computing and Engineering* 2(3), 032002. DOI:10.1088/2634-4386/ac7db7 https://research.rug.nl/en/publications/dc6b310f-86fe-4a19-adfa-ee6a915fb26c

**分野の主要レビュー（必読）。**
1. Tanaka, G. et al. (2019) “Recent advances in physical reservoir computing: A review”, *Neural Networks* 115, 100–123, arXiv:1808.04962. リザバーの物理種別――電子、フォトニック、スピントロニック、力学、生物、その他――で分類した標準的レビュー。本レポート§2の分類も主にこれに従う。 https://arxiv.org/html/1808.04962v3
2. Nakajima, K. (2020) “Physical reservoir computing—an introductory perspective”, *Japanese Journal of Applied Physics* 59, 060501. 日本発の入門的展望論文。2023年にJSAP論文賞（Review Paper）を受賞したことが応用物理学会ページで確認できる。 https://www.jsap.or.jp/english/awards/jsap-outstanding-paper-award/recipients45 （書誌は東京大学ページでも確認： https://www.t.u-tokyo.ac.jp/topics/tp2026-09-15-091 ）
3. Liang, X., Tang, J., Zhong, Y., Gao, B., Qian, H. & Wu, H. (2024) “Physical reservoir computing with emerging electronics”, *Nature Electronics* 7, 193–206. 新興エレクトロニクスに焦点を当て、アーキテクチャ、物理ノード、入出力層、ベンチマークと競合性を論じた2024年の主要レビュー。 https://www.mbrl.ae/web/guest/asset-details?documentId=summon____FETCH-LOGICAL-c319t-3ab4b4d683d92889c92903bf2cad69a752532c883b2d43483f3cc27594dcd3ce3
4. Van der Sande, G., Brunner, D. & Soriano, M.C. (2017) “Advances in photonic reservoir computing”, *Nanophotonics* 6, 561–576. フォトニックRC専門レビュー（書誌はIFISCの著者ページで確認）。 https://personal.ifisc.uib-csic.es/miguel/
5. Nakajima, K. & Fischer, I. (eds.) (2021) *Reservoir Computing: Theory, Physical Implementations and Applications*, Springer. 分野の成書（編者としての書誌はte Vrugtの入門章 arXiv:2412.13212 の参考文献で確認）。入門章そのもの：te Vrugt, M. “An introduction to reservoir computing”, in *Artificial Intelligence and Intelligent Matter*, Springer (2025), arXiv:2412.13212 https://arxiv.org/html/2412.13212v1

#### 2.1 光（フォトニクス）

遅延フィードバックを持つ単一非線形ノードを時間多重化して仮想ノード網とする方式が主流である。

- **単一ノード方式の創始**：Appeltant, L. et al. (2011) “Information processing using a single dynamical node as complex system”, *Nature Communications* 2, 468. 単一の非線形ノード（電子回路実装）と遅延フィードバックで大規模リザバーと等価な処理を行い、音声数字認識とNARMA-10で実証した（後者はTanaka et al. 2019の要約による）。 https://personal.ifisc.uib-csic.es/miguel/
- **全光・光電子方式**：Duport, F. et al. (2012) “All-optical reservoir computing”, *Optics Express* 20, 22783–22795；Larger, L. et al. (2012) “Photonic information processing beyond Turing: an optoelectronic implementation of reservoir computing”, *Optics Express* 20, 3241–3250（いずれも書誌はSorianoらの文献リストで確認）。 https://personal.ifisc.uib-csic.es/miguel/
- **高速化**：Brunner, D., Soriano, M.C., Mirasso, C.R. & Fischer, I. (2013) “Parallel photonic information processing at gigabyte per second data rates using transient states”, *Nature Communications* 4, 1364. 半導体レーザーの過渡状態でGbit/s級処理を示した。
- **100万語／秒**：Larger, L. et al. (2017) “High-speed photonic reservoir computing using a time-delay-based architecture: million words per second classification”, *Physical Review X* 7, 011015. 題名通り、時間遅延アーキテクチャで毎秒100万語級の分類をうたう。
  - 以上2件の書誌： https://personal.ifisc.uib-csic.es/miguel/
- **シリコンフォトニクス集積**：Vandoorne, K. et al. (2014) “Experimental demonstration of reservoir computing on a silicon photonics chip”, *Nature Communications* 5（書誌は図書館検索結果で確認。詳細な性能数値は本調査では原論文本文で未確認のため断定しない）。

光方式の利点は帯域・速度と低遅延、課題は入出力変換（光電変換）を含む系全体の消費電力・安定性・集積規模である（§6）。

#### 2.2 スピントロニクス

- **代表作**：Torrejon, J. et al. (2017) “Neuromorphic computing with nanoscale spintronic oscillators”, *Nature* 547(7664), 428–431. DOI:10.1038/nature23011. 単一のスピントルク・ナノオシレータ（磁気トンネル接合）の振幅ダイナミクス（数百nsの緩和による短期記憶と非線形性）を時間多重化してリザバーとし、音声数字認識で「state-of-the-art neural networksと同等の精度」を実験的に示したとNature要旨は述べる。著者には常木澄人（Tsunegi, S.）、薬師寺ら産総研側研究者が含まれ、日仏米連携の成果である。
  - 出典： https://www.nist.gov/publications/neuromorphic-computing-nanoscale-spintronic-oscillators ／ https://pubmed.ncbi.nlm.nih.gov:443/28748930/
- スピントロニクス系は、ナノスケール、低消費電力（オシレータ単体でµW級という議論が原論文にある）、相互結合によるネットワーク化の可能性が魅力とされる一方、ノイズと安定性が当初からの課題だったことが同論文要旨でも明記される。

#### 2.3 メモリスタ／動的メモリスタ

- **代表作**：Du, C., Cai, F., Zidan, M.A., Ma, W., Lee, S.H. & Lu, W.D. (2017) “Reservoir computing using dynamic memristors for temporal information processing”, *Nature Communications* 8, 2204. DOI:10.1038/s41467-017-02337-y. メモリスタ内部のイオン動力学による短期記憶をリザバーとして直接時間領域処理を行い、わずか88個のメモリスタのハードウェアで手書き数字認識と2次非線形動的タスクの解法を実証した。
  - 出典： https://econpapers.repec.org/article/natnatcom/v_3a8_3ay_3a2017_3ai_3a1_3ad_3a10.1038_5fs41467-017-02337-y.htm
- **発展**：Moon, J. et al. (2019) “Temporal data classification and forecasting using a memristor-based reservoir computing system”, *Nature Electronics* 2, 480–487. 時系列分類と予測への拡張（書誌はミシガン大学Luグループの出版リストで確認）。 http://www-personal.umich.edu/~wluee
- **高効率化**：Zhong, Y. et al. (2021) “Dynamic memristor-based reservoir computing for high-efficiency temporal signal processing”, *Nature Communications* 12, 408（書誌は引用文献リスト経由で確認。本調査では本文未読のため、性能の詳細は断定しない）。

#### 2.4 電子回路

- 遅延型単一ノードの最初の実験実装自体が電子（光電子）回路であった（Appeltant et al. 2011、上記）。
- Tanaka et al. (2019) は「Electronic RC」を独立章として、単一ノード遅延系が電子RCとフォトニックRCの双方で広く使われ、その単純さが物理実装上の利点だと整理する。 https://arxiv.org/html/1808.04962v3
- アナログ電子回路による集積化の日本での到達点が、北海道大学・TDKのアナログリザバーAIチップである（§3、§4で詳述）。
- CMOSスパイキング実装（2025）：Duranら “CMOS Implementation of Field Programmable Spiking Neural Network for Hardware Reservoir Computing”, arXiv版2024年11月／*Neuromorphic Computing and Engineering*掲載版。65nm CMOS、100ニューロン、ニューロン当たり21.7 pJ/pulse、約540 NAND2換算面積、FORCE学習とNARMA10・メモリ容量ベンチマークを実チップで検証したと要旨は述べる。数値は著者ら要旨に基づくもので、独立検証ではない。
  - 出典： https://arxiv.org/html/2509.17355v1

#### 2.5 力学系・ソフトロボット（タコ腕など）

- **タコ腕（シミュレーション）**：Nakajima, K., Hauser, H., Kang, R., Guglielmino, E., Caldwell, D.G. & Pfeifer, R. (2013) “A soft body as a reservoir: case studies in a dynamic model of octopus-inspired soft robotic arm”, *Frontiers in Computational Neuroscience* 7, 91. DOI:10.3389/fncom.2013.00091. タコを模したソフトアームの力学モデルをリザバーとし、線形読み出しだけで非線形力学系のエミュレーションと非線形リミットサイクルの埋め込みができることを示した。「身体のダイナミクス自体が計算資源」という形態計算（morphological computation）の代表作。
  - 出典： https://pmc.ncbi.nlm.nih.gov/articles/PMC3705147/
- **タコ腕（実機シリコンアーム）**：Nakajima, K., Hauser, H., Li, T. & Pfeifer, R. (2015) “Information processing via physical soft body”, *Scientific Reports* 5, 10487. DOI:10.1038/srep10487. 実際のシリコン製ソフトアームの身体ダイナミクスで非線形力学系をエミュレートし、身体の非線形性とメモリが計算性能を上げること、適用範囲と限界を標準的機械学習手法との比較で特徴づけた。「soft bodies can also be potentially used as computational resources on top and for free」という結論で知られる。
  - 出典： https://research-information.bris.ac.uk/ws/files/76657528/srep10487.pdf
- 関連：Nakajima, K., Li, T., Hauser, H. & Pfeifer, R. (2014) “Exploiting short-term memory in soft body dynamics as a computational resource”, *Journal of The Royal Society Interface* 11, 20140437（書誌は東京大学ページで確認）。 https://www.t.u-tokyo.ac.jp/topics/tp2026-09-15-091
- テンセグリティ構造による移動（Caluwaerts et al. 2013 “Locomotion without a brain: physical reservoir computing in tensegrity structures”）も、Tanaka／Nakajima系文献で力学RCの例として繰り返し引用される（本調査では原論文本文は未読）。

#### 2.6 ソフトウェアRCの大規模成果（物理ではないが分野を押し上げたもの）

- Pathak, J., Hunt, B., Girvan, M., Lu, Z. & Ott, E. (2018) “Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach”, *Physical Review Letters* 120, 024102. DOI:10.1103/PhysRevLett.120.024102. 並列リザバーを空間領域ごとに割り当てる方式で、Kuramoto–Sivashinsky方程式の時空間カオスを観測データのみからモデルフリーに予測し、任意に大きな系へのスケーラビリティを示した。
  - 出典： https://link.aps.org/accepted/10.1103/PhysRevLett.120.024102

#### 2.7 量子・生物など、その他の物理系

- **量子リザバー（日本発）**：Fujii, K. & Nakajima, K. (2017) “Harnessing disordered-ensemble quantum dynamics for machine learning”, *Physical Review Applied* 8, 024030. DOI:10.1103/PhysRevApplied.8.024030. 乱れた量子多体系のダイナミクスをリザバーとし、指数関数的に大きなHilbert空間を特徴空間として使う量子リザバーコンピューティング（QRC）の創始論文の一つ。5–7量子ビット程度で数百ノード級RNNに匹敵しうるという数値的示唆が紹介資料にある。
  - 出典： https://kaken.nii.ac.jp/en/report/KAKENHI-PROJECT-16KT0019/16KT00192017hokoku/
- **生物（培養神経・オルガノイド）**：§3、§4で詳述（Sumi et al. 2023; Cai et al. 2023 Brainoware）。

| 物理系 | 代表論文 | 何ができたか（論文の主張） |
|---|---|---|
| 光 | Appeltant et al., Nat Commun 2011; Larger et al., PRX 2017 | 単一ノード＋遅延で音声認識・NARMA；時間遅延型で100万語/秒級分類 |
| スピントロニクス | Torrejon et al., Nature 2017 | 単一ナノオシレータで音声数字認識、当時のSOTA級精度 |
| メモリスタ | Du et al., Nat Commun 2017 | 88素子で手書き数字認識と非線形動的タスク |
| 電子回路 | Appeltant et al. 2011；TDK/北大 2025（企業発表） | 遅延型RCの実験原型；アナログ集積チップ試作とリアルタイム学習デモ |
| 力学・ソフトボディ | Nakajima et al., Front 2013; Sci Rep 2015 | タコ腕（模型・実機）で非線形系エミュレーション、リミットサイクル埋め込み |
| 量子 | Fujii & Nakajima, PRApplied 2017 | 少数量子ビットの量子ダイナミクスを時系列学習のリザバーに |
| 生物 | Sumi et al., PNAS 2023; Cai et al., Nat Electron 2023 | 培養神経で音声数字分類と汎化；脳オルガノイドで話者認識・非線形方程式予測 |

---

### 3. 日本の研究動向

#### 3.1 中嶋浩平（東京大学）――ソフトボディから量子・生態系・クラゲまで

中嶋浩平は東京大学大学院情報理工学系研究科知能機械情報学専攻准教授（2022年4月～現職、國吉・中嶋研究室／知能システム情報学研究室）。チューリッヒ大学／ETHでのポスドク時代にEUプロジェクトOCTOPUSでタコ型ソフトロボットの制御を担当し、身体ダイナミクスをリザバーとする着想に至った経緯がインタビューで語られている。
- 出典：東京大学工学部「若手研究者紹介：中嶋浩平准教授」 https://www.t.u-tokyo.ac.jp/topics/tp2026-09-15-091 ／セコム科学技術振興財団インタビュー https://www.secomzaidan.jp/tokutei/advanced-mathematics/interview/nakajima/

代表論文（§2.5、§2.7と重複するものは簡略に再掲する）：
- Nakajima et al. (2013) タコ腕モデル、*Front Comput Neurosci* 7:91。
- Nakajima et al. (2015) シリコンアーム実機、*Sci Rep* 5:10487。
- Fujii & Nakajima (2017) 量子リザバー、*Phys Rev Applied* 8:024030。
- Nakajima, K. (2020) “Physical reservoir computing—an introductory perspective”, *Jpn J Appl Phys* 59:060501（JSAP Review Paper賞、2023）。
- Nakajima, M. et al. (2022) “Physical deep learning with biologically inspired training method: gradient-free approach for physical hardware”, *Nature Communications* 13, 7847. 物理系そのものを深層化する際の、勾配に頼らない学習法（物理深層学習）への展開としてNature Electronicsレビュー（Liang et al. 2024）の参考文献でも引かれる。
  - 書誌出典： https://gw.aguse.jp/?aguse_url=https%3A%2F%2Fwww-nature-com.translate.goog%2Farticles%2Fs41928-024-01133-z%3F_x_tr_sl%3Den%26_x_tr_tl%3Dja%26_x_tr_hl%3Dja%23citeas
- Ushio, M., Watanabe, K., Fukuda, Y., Tokudome, Y. & Nakajima, K. (2023) “Computational capability of ecological dynamics”, *Royal Society Open Science* 10(4), 221614. テトラヒメナを含む生態系ダイナミクス（個体数変動）をリザバーとして使う「生態系AI」の提案。東京大学ページは、温度入力に対する個体数増減をリザバーとし魚種の生息数時系列を予測する構想を紹介する。
  - 出典： https://www.t.u-tokyo.ac.jp/topics/tp2026-09-15-091
- Akashi, N. et al. (2024) “Embedding bifurcations into pneumatic artificial muscle”, *Advanced Science* 11(25), 2304402. 空気圧人工筋肉（ブリヂストンとの共同研究系）に分岐構造を埋め込み、周期運動の生成や刺激応答の切り替えを身体側で担う。
  - 出典： https://www.t.u-tokyo.ac.jp/topics/tp2026-09-15-091
- Owaki, D., Austin, M., Ikeda, S., Okuizumi, K. & Nakajima, K. (2025) クラゲサイボーグ、*Nature Communications* 16, 4642（§4）。
- 理論面では、力学系の情報処理容量、カオス（過渡カオス）の計算能力、LLM内部ダイナミクスと脳活動の比較など、RCの物差しを他分野へ輸出する研究も行っている（Inoue et al. 2022, *Phys Rev Research* 4:013204 “Transient chaos in bidirectional encoder representations from transformers”など、書誌は東京大学ページで確認）。

#### 3.2 田中剛平・廣瀬明・中根了昌ら（東京大学／名古屋工業大学ほか）――理論とハードウェアの体系化

- Tanaka et al. (2019) のPRCレビュー（§2）が国際的な標準参照である。
- 日本語の成書：田中剛平、中根了昌、廣瀬明『リザバーコンピューティング：時系列パターン認識のための高速機械学習の理論とハードウェア』森北出版、2021年3月、207頁、ISBN 978-4-627-85531-1。数理・アルゴリズムとデバイス開発・実装の両面から解説し、Pythonサンプルコードを提供すると国立国会図書館／CiNiiの書誌は述べる。日本語でRCを学ぶ際の第一の体系書である。
  - 出典： https://ndlsearch.ndl.go.jp/books/R100000002-I031334048 ／ https://cir.nii.ac.jp/crid/1970586434872421572
- 田中は現在は名古屋工業大学に所属する研究者情報ページでも同レビューを代表作として掲げる（所属の異動時点はページによる）。 https://researcher.nitech.ac.jp/html/100001183_ronbn_1_en.html
- 廣瀬明（東京大学）の複素数値ニューラルネットワーク／光・電波系の信号処理の流れは、Tanakaレビューの共著構成（IBM東京基礎研究所の中野大樹・山根敏志らとの産学混成チーム）にも表れている。日本では大学（理論）と企業研究所（デバイス）が一つのレビューを共著する形で分野が形成された点が特徴である。

#### 3.3 その他の日本の主要グループ

- **スピントロニクス（産総研ほか）**：Torrejon et al. (2017, *Nature*) の共著に常木澄人、薬師寺啓、福島章雄、久保田均、湯浅新治（産総研）が名を連ねる（§2.2）。日本はスピントロニクス素子側の強みをPRCに接続した。
- **生体神経リザバー（東北大学・公立はこだて未来大学ほか）**：Sumi, T., Yamamoto, H., Katori, Y. et al. (2023) “Biological neurons act as generalization filters in reservoir computing”, *Proceedings of the National Academy of Sciences* 120(25), e2217008120. DOI:10.1073/pnas.2217008120. 培養した生物神経ネットワーク（モジュール構造を埋め込んだ基板上）を光遺伝学とカルシウムイメージングで記録し、線形デコーダで分類できること、数百msの短期記憶を持つこと、音声数字分類に使えること、そして訓練データと別データセットの同カテゴリ分類を可能にする「汎化フィルタ」として働くことを示した。
  - 出典： https://pubmed.ncbi.nlm.nih.gov/37307467/
- **アナログ電子回路（北海道大学・浅井哲也）＋TDK**：§4で詳述。浅井哲也（北海道大学情報科学研究院教授）はTDKのプレスリリースで、今回のチップが「エッジAIの新たな可能性を切り拓く」とコメントしている。
  - 出典： https://www.tdk.com/ja/news_center/press/20251002_01.html
- **量子**：藤井啓祐（大阪大学／理研、当時京都大学ほか）と中嶋のQRC（§2.7）。その後、NMR実装などへの展開がKAKEN報告に記録される。 https://kaken.nii.ac.jp/grant/KAKENHI-PROJECT-15K16076/
- **小脳モデルとしてのRC**：Yamazaki & Tanaka (2007) の小脳RCモデルは、Soriano et al. (2015) でも「RC-like model for the cerebellum」として引用される、日本発の神経科学側の先行研究である。 https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00068/full

---

### 4. 2023年から2026年までの最新動向

注意：2026年の日付を持つ一部のウェブ資料（arXivの将来日付表示など）は検索インデックス上の表記揺れを含むため、本節では査読誌と公的機関発表を優先し、日付が不確かなものはその旨を付す。

#### 4.1 総説・基盤（2023–2024）

- Liang, X. et al. (2024) “Physical reservoir computing with emerging electronics”, *Nature Electronics* 7, 193–206（§2）。PRCが「materialsの動的性質をharvestするneuromorphic computing」であり、高効率計算を狙うと総括する。2024年時点でのベンチマーク比較と競合性評価を含む点で、Tanaka (2019) の後継レビューとして読める。
- Cucchi et al. (2022) のチュートリアル以降、材料科学・化学・電子工学の研究者が自分の材料系でRC実験を組むための実務知（入力符号化、時間多重化、読み出し、容量測定）が標準化されつつあることが、2024年レビューの引用構成からもうかがえる。

#### 4.2 バイオリザバーの跳躍（2023–2025）

- **Brainoware**：Cai, H., Ao, Z., Tian, C. et al. & Guo, F. (2023) “Brain organoid reservoir computing for artificial intelligence”, *Nature Electronics* 6, 1032–1039. DOI:10.1038/s41928-023-01069-w. ヒト脳オルガノイドを高密度多電極アレイ上に載せ、時空間電気刺激で非線形ダイナミクスとfading memoryを得て、音声認識と非線形方程式予測を実証した。二次的な読み解き資料によれば、8話者の日本語母音の話者認識精度は訓練前約51%から2日間の訓練後約78%へ向上したとされる（数値は原論文の報告を要約した資料に基づく）。
  - 出典：原論文 https://www.researchgate.net/publication/376413285_Brain_organoid_reservoir_computing_for_artificial_intelligence ／数値の要約 https://vexpaer.github.io/2026/10/29/readApaper-12/
  - 批判的留保：小規模な概念実証であり、オルガノイド自体が「学習した」と単純化すべきではなく、外部の教師あり読み出しとの複合系である点は読み解き資料も強調する。
- **培養神経の汎化フィルタ**：Sumi et al. (2023, *PNAS*、§3.3)。生体神経リザバーが単なるランダム変換ではなくカテゴリ汎化に寄与するという、設計論に踏み込む結果である。
- **クラゲサイボーグ**：Owaki, D., Austin, M., Ikeda, S., Okuizumi, K. & Nakajima, K. (2025) “Harnessing natural embodied intelligence for spontaneous jellyfish cyborgs”, *Nature Communications* 16, 4642. DOI:10.1038/s41467-025-59889-7. ミズクラゲの筋肉への電気パルスで遊泳を変調し、クラゲの身体を構成要素とする軽量なハイブリッド物理リザバーモデルで全横方向の将来速度を予測できたと、東北大学のプレスリリースは述べる。最も効果的なパルスがクラゲ本来のリズムに合ったものだった点が「自然の身体性を計算に使う」PRCの典型例である。
  - 出典： https://www.tohoku.ac.jp/en/press/exploring_the_seas_with_selfpowered_jellyfish_cyborgs.html

#### 4.3 アルゴリズム側：Next-Generation RCの展開

- 原点：Gauthier, D.J., Bollt, E., Griffith, A. & Barbosa, W.A.S. (2021) “Next generation reservoir computing”, *Nature Communications* 12, 5564. DOI:10.1038/s41467-021-25801-2. RCが非線形ベクトル自己回帰（NVAR）と等価であることを利用し、ランダム行列もリザバーもなしに、遅延観測の多項式特徴だけで同等以上の性能を出し、訓練データと訓練時間をさらに削減した。大学発表では、Lorenz予測で従来RCより33–163倍高速、高精度条件では約100万倍高速、4000ニューロン相当を28ニューロン相当で代替、ウォームアップが数点で済む、といった著者らの主張が紹介されている（数値は発表側の説明であり、タスク依存であることに注意）。
  - 出典： https://www.nature.com/articles/s41467-021-25801-2.pdf ／ https://www.eurekalert.org/news-releases/928952
- 2024–2025年の展開として、カーネル嶺回帰として定式化し無限次元特徴へ拡張する “Infinite-dimensional next-generation reservoir computing” (arXiv:2412.09800, 2024) や、NG-RCが予測だけでなく分類タスクでも従来RCと同等に機能することを初めて示したと主張する “Next-generation reservoir computing validated by classification task” (arXiv:2512.12903, 2025年12月投稿と表示。著者所属としてNICT・浜松ホトニクスが二次資料に現れる) が検索で確認できる。いずれもプレプリント段階として扱うべきである。
  - 出典： https://scirate.com/arxiv/2412.09800 ／ https://scirate.com/arxiv/2512.12903

#### 4.4 商用化・実用化の動き

- **TDK＋北海道大学（2025、日本）**：TDKは2025年10月2日、小脳を模倣したアナログ電子回路によるリザバーAIチップのプロトタイプを北海道大学と共同開発し、CEATEC 2025（2025年10月14–17日）で加速度センサと組み合わせたデモ機を展示すると発表した。デモは、指の動きを加速度センサで測り、出し終わる前にじゃんけんの手を予測して「ユーザーが絶対に勝てないじゃんけん」を実現するもので、個人差をリアルタイム学習する点が売りである。TDK自身が「市場でエッジAI向けリザバーコンピューティングデバイスの製品化が加速することを期待」と述べる段階であり、量産製品の発売発表ではない。
  - 出典（企業発表）： https://www.tdk.com/en/news_center/press/20251002_01.html ／日本語 https://www.tdk.com/ja/news_center/press/20251002_01.html
  - 技術解説（企業記事）：2024年にアナログ回路によるリザバーAIチップ試作に成功したこと、それに先立つ2020年に基本電子部品で大型RCデバイスを開発し「世界最高特性」を確認したというTDK側の説明を含む。いずれもTDKの主張であり、第三者の査読評価ではない。 https://www.tdk.com/ja/featured_stories/entry_083-analog-reservoir-computing-AI-chip.html
- **Quantum Computing Inc.（QCi、米国）**：フォトニックRCプラットフォーム「NeuraWave」をPCIeカード形式で商用注文可能（deployment-ready）とする発表が、業界メディアで報じられている。時系列予測・異常検知・信号処理をエッジ／組み込み向けに狙うという位置づけである。また2025年第2四半期決算資料では、EmuCoreリザバーデバイスを大手自動車メーカーへ販売し、エッジ機械学習のR&Dに使うとされる。
  - 出典（業界メディア／企業決算転載）： https://convergedigest.com/qci-launches-neurawave-photonic-platform/ ／ https://quantumcomputingreport.com/quantum-computing-inc-launches-neurawave-photonic-platform-for-edge-ai-inference/ ／ https://marketwirenews.com/stock/qubt/news/quantum-computing-inc-reports-second-quarter-2025-fi-4877970412762179.html
  - 留保：これらは企業発表に基づく報道であり、独立したベンチマーク論文は本調査では確認できていない。性能・省電力の主張は未検証として扱うこと。
- 全体として、2026年時点で「RC専用チップの量産が主流AI市場を置き換えた」と言える文献的根拠はない。商用化は、センサ近傍の時系列処理というニッチから始まっている、というのが文献と発表の双方に整合する読みである。

---

### 5. エッジAI・低消費電力コンピューティングとの関係

#### 5.1 なぜエッジ向きと言われるのか――主張とその根拠

主張の論理は3段である。(1)学習が線形読み出しのみなので、訓練の計算量・データ量が小さく、現場での再学習（リアルタイム学習）がしやすい。(2)推論では、リザバーのダイナミクスを物理系に任せれば、デジタル積和演算の大部分を「自然現象の伝播」で置き換えられる可能性がある。(3)時系列センサデータはセンサ近傍で処理できれば通信・クラウド往復が不要になる。

この論理自体はTanaka et al. (2019)（学習の速さ＝低訓練コスト、固定リザバーのハードウェア実装適性）およびLiang et al. (2024)（PRC＝材料ダイナミクスによる高効率計算という総括）に明記された、分野の公式見解である。ただし (2) の「系全体で省電力になるか」は、入出力変換・読み出し回路・制御回路を含めた実測でしか決まらず、個別論文の主張をそのまま一般化できない（§6）。

#### 5.2 FPGA実装

- **代表的な最近の実装**： “Streamlined Dataflow FPGA Architectures for Echo State Neural Networks in Time-Series Prediction and Classification Applications”, *ACM Transactions on Reconfigurable Technology and Systems*（ACM DLで要旨を確認）。全層を量子化し計算を完全展開（fully unroll）するデータフロー型アクセラレータで、200リザバーニューロンのESNで遅延9.5 ns、スループット100 Megasamples/秒、電力遅延積72 nWsを達成し、従来FPGA実装および組み込みGPU／CPU比で数桁の改善だと要旨は主張する。数値は著者ら要旨に基づく。
  - 出典： https://dl.acm.org/doi/10.1145/3811539
- FPGAは「固定リザバー＋線形読み出し」という構造と相性がよい（リザバー部の演算パターンは定型的で、学習部は小さい）。一方、完全展開型はリザバー規模を大きくするとリソース消費が急増するトレードオフが要旨でも示唆される（“resource-intensive but excel in latency, throughput, and energy efficiency”）。

#### 5.3 エッジデバイス／チップ実装

- **TDKアナログリザバーチップ（2025）**：センサ（加速度）とチップを組み合わせ、指動作の個人差をその場で学習して予測するデモが、エッジRCのユースケースとして提示された（§4.4）。TDKはRCデバイスを「時間的に変化する情報に特化し、センサデータを直接扱うエッジのような超低電力製品で低消費電力と高速計算を実現」するものと位置づける。これは企業による位置づけであり、消費電力の絶対値（mW等）はプレスリリース本文では本調査で確認できなかった。
  - 出典： https://www.tdk.com/en/news_center/press/20251002_01.html
- **CMOSスパイキングチップ**：Duran et al.（§2.4）。21.7 pJ/pulseというニューロン単体のエネルギー、ADCなし読み出し、SoC統合適性がエッジ向けの根拠として著者らにより提示される。
  - 出典： https://arxiv.org/html/2509.17355v1
- **簡素化RCの探索（2025）**：Simple Cycle Reservoir、簡素化した活性化関数、16ビット低精度演算、遺伝的アルゴリズムによるハイパーパラメータ最適化を組み合わせ、リソース制約下のエッジハードウェア向けに計算・ハードウェアオーバーヘッドを下げつつ競合的精度を保つとする会議論文が2025年に現れている（IEEE MWSCAS 2025）。
  - 出典： https://par.nsf.gov/servlets/purl/10672112

#### 5.4 Transformer／通常のRNNとの比較

| 観点 | RC（ESN等） | LSTM等の全訓練RNN | Transformer |
|---|---|---|---|
| 訓練 | 読み出しの線形回帰のみ。高速・安定・少量データ向き（Jaeger & Haas 2004; Gauthier et al. 2021） | BPTTで全パラメータ訓練。勾配消失・爆発、コスト大 | 全パラメータ訓練。データ・計算とも大規模 |
| 長期依存 | fading memoryが前提のため、原理的に有限の時間窓。マルチタイムスケール化は研究課題 | ゲート機構で長期依存を学習可能 | self-attentionで長距離依存を直接参照可能（ただし計算量は系列長の2次） |
| 表現力 | 固定ランダム特徴に依存。タスクに合えば強いが、合わなければ読み出しでは救えない | タスク適応的に内部表現を学習 | スケールとともに言語等で最高性能 |
| 推論（物理実装時） | 物理ダイナミクスで置換できる可能性。エッジ向き | デジタル演算が中心 | デジタル演算が中心、メモリ・電力大 |
| 得意領域 | 力学系・カオス予測、センサ時系列、音声等のリアルタイム処理 | 汎用系列モデリング | 言語・マルチモーダル等の大規模汎用タスク |

この比較を実証的に裏づける最近の研究として、文字レベル言語モデリングでRCとTransformerを訓練可能パラメータ数を揃えて比較したプレプリントがある。その結論は「transformers excel in prediction quality, whereas reservoir computers remain highly efficient reducing the training and inference speed（要旨の表現）」――すなわち予測品質ではTransformerが優れ、RCは訓練・推論の効率で優れる、というトレードオフである。
- 出典： “Reservoir Computing as a Language Model”, arXiv:2507.15779 https://arxiv.org/html/2507.15779v1
- また、埼玉大学・内田淳史らのグループによる、Transformerのself-attentionを固定リザバーで置き換える研究（*Physical Review Applied*, 2026年8月と二次資料が伝える）でも、n-gram再現では同等でもテスト損失ではTransformerに劣り、エンドツーエンドの速度・エネルギー優位は実証されていない、と二次資料は慎重に報じる。本調査では原論文本文を直接確認できていないため、詳細は未検証として扱う。
  - 出典（二次資料）： https://www.habaw.com/articles/fixed-reservoirs-and-the-limits-of-the-transformer-shaped-hole-in-language-modeling-8c8sdz

結論として、RCの利点は「学習の安さ」と「物理実装時の推論の潜在的安さ」に、限界は「固定特徴ゆえの表現力の天井」と「fading memoryゆえの長期依存の弱さ」に、それぞれ文献的根拠がある。

---

### 6. 未解決の課題：批判的視点

#### 6.1 スケーラビリティと逐次性

- 従来RCは時系列を逐次処理しなければならず並列化できず、高次元リザバーでは密行列のメモリフットプリントが prohibitive になる、という批判が明示的に定式化されている。これを対角線形再帰と複素領域の構造化作用素で解こうとするのがParalESN（Pinna et al., PMLR v306収録論文）であり、序論で「its scalability remains severely constrained by the need to process temporal data sequentially and the prohibitive memory footprint of high-dimensional reservoirs」と述べる。
  - 出典： https://raw.githubusercontent.com/mlresearch/v306/main/assets/pinna26a/pinna26a.pdf
- Pathak et al. (2018) の並列リザバー方式は時空間カオスに限った分割統治であり、汎用系列タスクへの一般解ではない。深層化（DeepESN、Gallicchio & Micheli 2017 “Deep reservoir computing: A critical experimental analysis”, *Neurocomputing* 268, 87–99）も提案されたが、層を重ねるとハイパーパラメータが増え、RC本来の単純さが失われるトレードオフがある（書誌はGallicchioの講義資料で確認）。
  - 出典： https://www.slideshare.net/slideshow/claudio-gallicchio-deep-reservoir-computing-for-structured-data-239336864/239336864

#### 6.2 再現性：物理デバイスのばらつき、温度、ドリフト

- デジタルRCでは訓練済み重みを何百万台にもコピーできるが、物理リザバーではそうはいかない。Ma et al. (2025) はPRCの信頼性問題を正面から論じ、非線形半導体素子は温度によるリーク電流・閾値電圧シフトに弱く、メモリスタは素子間ばらつきが大きく、訓練済み系の複製と大規模展開を妨げると指摘する。さらに「inherently fixed properties of physical systems」のため汎用性が限られ、タスク要求に合わせてダイナミクスを精密に制御することは未解決だと述べる。
  - 出典：Ma et al. (2025) “Versatile and Robust Reservoir Computing with PWM-Driven Heterogeneous R–C Circuits”, *Advanced Science*. https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202416413
- Cucchi et al. (2022) も、良いリザバーを特定タスク向けに用意することは「not as easy as it seems at first sight」とし、実験系ごとの調整負担を課題とする。 https://research.rug.nl/en/publications/dc6b310f-86fe-4a19-adfa-ee6a915fb26c
- 生物リザバーでは個体差・日差がさらに大きく、Brainowareの結果も5個のオルガノイド、3回の独立実験に基づく概念実証であることが読み解き資料で注意されている（§4.2）。

#### 6.3 タスク依存性とハイパーパラメータ

- RCの性能は、スペクトル半径、入力スケーリング、リーク率、リザバーサイズ、ランダムシードに敏感であり、「ランダムでよい」と言いながら実際にはタスクごとの調整が性能を決める。NG-RC（Gauthier et al. 2021）が問題視したのもまさにこの点――「randomly sampled matrices」と「multitude of metaparameters that must be optimized」――であり、NG-RCはそれを除去することでRC自身の弱点を突いた研究だと読める。
  - 出典： https://www.nature.com/articles/s41467-021-25801-2.pdf
- Dambre et al. (2012) の容量理論は裏側から同じことを示す：総容量は状態変数数で上界され、線形メモリを増やせば非線形容量が減るという配分のトレードオフは避けられない。つまり「どんなタスクにも強いリザバー」は原理的に存在せず、タスクの要求（メモリ長×非線形度）に合ったリザバーを選ぶ問題が残る。

#### 6.4 ベンチマークと主張のインフレへの注意

- 分野の初期の成功（Jaeger & Haas 2004の2400倍、Larger 2017の100万語/秒、Gauthier 2021の100万倍高速など）はいずれも特定タスク・特定条件での値であり、論文・発表の文脈を離れて引用されやすい。本レポートでは各数値に出典と条件を付けたが、読者も同様に扱うべきである。
- 物理実装の「省電力」主張の多くは素子単体またはリザバー部の値であり、A/D・D/A変換、読み出し、制御、ホスト計算を含む系全体の実測比較は不足している。これはLiang et al. (2024) がベンチマークと競合性の標準化を課題として扱う理由でもある。
- 理論面では、ESPの必要十分条件が入力依存であること（§1.3）、普遍近似性の結果がfading memoryを持つ関数のクラスに限られることなど、「RCは万能計算機」という言い方には常に前提条件が伴う。

#### 6.5 その他の未解決課題

- **オンライン学習と安定性**：読み出しを逐次更新するFORCE学習等の手法はあるが、物理系のドリフト下での長期安定運用は未確立。
- **大規模言語・マルチモーダルへの拡張**：§5.4の通り、現状ではTransformerに品質で劣るという比較研究が蓄積しつつある。RC単独ではなく、注意機構や状態空間モデルとのハイブリッド（AERC、ResFormer等の試み）が探索されているが、査読済みの決定的成果は本調査では確認できなかった。
- **設計論の欠如**：「どの物理系が、どのタスクに、なぜ良いのか」を予測する設計理論は、容量理論（Dambre）や力学系理論の側から構築途上であり、中嶋らが「Physical Reservoir Computingにおける数理基盤の構築」を掲げるのもこの空白に対応する。 https://www.secomzaidan.jp/tokutei/advanced-mathematics/interview/nakajima/

---

## 最初に読むべき論文5本

1. **Jaeger, H. & Haas, H. (2004) “Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication”, *Science* 304, 78–80.**
   ESNの威力を最もコンパクトに示した3頁の論文。読み出しのみの線形回帰でカオス予測精度が桁違いに改善する様子が、分野の出発点として理解できる。 https://www.ai.rug.nl/minds/uploads/1912_JaegerHaas04.pdf
2. **Maass, W., Natschläger, T. & Markram, H. (2002) “Real-Time Computing Without Stable States”, *Neural Computation* 14, 2531–2560.**
   LSMの原典。過渡ダイナミクス、fading memory、分離性・近似性という、RCの理論的語彙の源泉。ESNと対で読むこと。 http://www.lsm.tugraz.at/papers/lsm-nc-130.pdf
3. **Lukoševičius, M. & Jaeger, H. (2009) “Reservoir computing approaches to recurrent neural network training”, *Computer Science Review* 3, 127–149.**
   ESN／LSMを「RC」として統一し、リザバー生成法と読み出し学習法を体系化した標準総説。実務的な設計指針までカバーする。 https://cenl.ucsd.edu/CompNeuro/Readings/week13/Lukosevicius-Jaeger+Reservoir-computing-recurrent-neural-network+CompSciRev+2019.pdf
4. **Tanaka, G. et al. (2019) “Recent advances in physical reservoir computing: A review”, *Neural Networks* 115, 100–123（arXiv:1808.04962）。**
   物理RCの分類と到達点を一望できる必読レビュー。光・スピン・メモリスタ・力学・生物の各実装へここから辿れる。 https://arxiv.org/abs/1808.04962?context=cs
5. **Dambre, J., Verstraeten, D., Schrauwen, B. & Massar, S. (2012) “Information processing capacity of dynamical systems”, *Scientific Reports* 2, 514.**
   「リザバーの良さ」を測る物差し（容量、メモリと非線形性のトレードオフ）を理論的に与えた論文。物理リザバーを評価する側に回るための基礎になる。 https://pubmed.ncbi.nlm.nih.gov:443/22816038/

次点として：Gauthier et al. (2021) “Next generation reservoir computing”, *Nature Communications* 12, 5564（RCの前提を問い直す批判的発展として）；Nakajima, K. (2020) “Physical reservoir computing—an introductory perspective”, *Jpn J Appl Phys* 59, 060501（日本語圏の研究者によるPRC入門）；Cucchi et al. (2022)（実験を始める人向け）。

---

## Could not verify / Open questions（確認できなかったこと）

- TDKアナログリザバーAIチップの消費電力の絶対値、チップ規模、査読論文としての発表は、本調査ではプレスリリース本文から確認できなかった。「世界最高特性」「世界初」はTDKの主張として扱い、独立検証は未確認である。
- QCi NeuraWave／EmuCoreの性能・省電力・販売実績は企業発表とその転載報道のみで、査読論文や独立ベンチマークは確認できなかった。
- 埼玉大学・内田らによるRC言語モデル（*Physical Review Applied*掲載とされる2026年の研究）は、二次資料経由でのみ確認し、原論文本文は未読である。
- Vandoorne et al. (2014) シリコンフォトニクスRC、Zhong et al. (2021) メモリスタRC、Caluwaerts et al. (2013) テンセグリティRCは書誌・引用経由の確認にとどまり、本文の性能数値は検証していない。
- 2026年日付を帯びた一部のarXiv／ウェブ資料は検索インデックスの表示日付に揺れがあり、本レポートでは2025年12月投稿と表示されたNG-RC分類論文（arXiv:2512.12903）などをプレプリントとして慎重に扱った。
- 本調査はウェブ検索インデックス経由の確認が中心であり、研究室ブラウザでの原論文PDF全文の逐一の精読（verified live）は行っていない。各数値は引用した要旨・発表資料の範囲内でのものである。

---

## Sources（出典一覧）

基礎理論
- Jaeger (2001) GMD Report 148 — https://www.scribd.com/document/487953134/The-echo-state-approach-to-analysing-and-training-1
- Maass, Natschläger & Markram (2002) Neural Computation — https://psycnet.apa.org/record/2002-06457-001 ; http://www.lsm.tugraz.at/papers/lsm-nc-130.pdf
- Jaeger & Haas (2004) Science — https://www.ai.rug.nl/minds/uploads/1912_JaegerHaas04.pdf ; https://pubmed.ncbi.nlm.nih.gov:443/15064413/
- Lukoševičius & Jaeger (2009) Computer Science Review — https://cenl.ucsd.edu/CompNeuro/Readings/week13/Lukosevicius-Jaeger+Reservoir-computing-recurrent-neural-network+CompSciRev+2019.pdf
- Verstraeten et al. (2007) Neural Networks — https://www.sciencedirect.com/science/article/pii/S089360800700038X
- Dambre et al. (2012) Scientific Reports — https://pubmed.ncbi.nlm.nih.gov:443/22816038/
- Soriano et al. (2015) Frontiers in Computational Neuroscience — https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00068/full
- IEEE Technology Navigator, Echo state networks — https://technav.ieee.org/topic/echo-state-networks/
- Reservoir Computing Beyond Memory-Nonlinearity Trade-off (2017) Scientific Reports — https://www.nature.com/articles/s41598-017-10257-6?error=cookies_not_supported&code=0f4426a6-2635-4066-9577-1a0368dc4a3d

物理RC・レビュー
- Tanaka et al. (2019) Neural Networks, arXiv:1808.04962 — https://arxiv.org/abs/1808.04962?context=cs ; https://arxiv.org/html/1808.04962v3
- Cucchi et al. (2022) Neuromorphic Computing and Engineering — https://research.rug.nl/en/publications/dc6b310f-86fe-4a19-adfa-ee6a915fb26c
- Liang et al. (2024) Nature Electronics — https://www.mbrl.ae/web/guest/asset-details?documentId=summon____FETCH-LOGICAL-c319t-3ab4b4d683d92889c92903bf2cad69a752532c883b2d43483f3cc27594dcd3ce3
- Van der Sande et al. (2017) などフォトニクス書誌 — https://personal.ifisc.uib-csic.es/miguel/
- Torrejon et al. (2017) Nature — https://www.nist.gov/publications/neuromorphic-computing-nanoscale-spintronic-oscillators ; https://pubmed.ncbi.nlm.nih.gov:443/28748930/
- Du et al. (2017) Nature Communications — https://econpapers.repec.org/article/natnatcom/v_3a8_3ay_3a2017_3ai_3a1_3ad_3a10.1038_5fs41467-017-02337-y.htm
- Moon et al. (2019) Nature Electronics（書誌） — http://www-personal.umich.edu/~wluee
- Nakajima et al. (2013) Frontiers — https://pmc.ncbi.nlm.nih.gov/articles/PMC3705147/
- Nakajima et al. (2015) Scientific Reports — https://research-information.bris.ac.uk/ws/files/76657528/srep10487.pdf
- Fujii & Nakajima (2017) Physical Review Applied（KAKEN報告） — https://kaken.nii.ac.jp/en/report/KAKENHI-PROJECT-16KT0019/16KT00192017hokoku/
- Pathak et al. (2018) Physical Review Letters — https://link.aps.org/accepted/10.1103/PhysRevLett.120.024102
- te Vrugt, An introduction to reservoir computing, arXiv:2412.13212 — https://arxiv.org/html/2412.13212v1

日本・最新動向・商用化
- 東京大学 中嶋浩平紹介 — https://www.t.u-tokyo.ac.jp/topics/tp2026-09-15-091
- セコム財団 中嶋インタビュー — https://www.secomzaidan.jp/tokutei/advanced-mathematics/interview/nakajima/
- JSAP論文賞（Nakajima 2020） — https://www.jsap.or.jp/english/awards/jsap-outstanding-paper-award/recipients45
- 田中・中根・廣瀬 (2021) 森北出版（NDL） — https://ndlsearch.ndl.go.jp/books/R100000002-I031334048 ; https://cir.nii.ac.jp/crid/1970586434872421572
- Sumi et al. (2023) PNAS — https://pubmed.ncbi.nlm.nih.gov/37307467/
- Cai et al. (2023) Brainoware, Nature Electronics — https://www.researchgate.net/publication/376413285_Brain_organoid_reservoir_computing_for_artificial_intelligence ; 数値要約 https://vexpaer.github.io/2026/10/29/readApaper-12/
- Owaki et al. (2025) Nature Communications（東北大学発表） — https://www.tohoku.ac.jp/en/press/exploring_the_seas_with_selfpowered_jellyfish_cyborgs.html
- Gauthier et al. (2021) Nature Communications — https://www.nature.com/articles/s41467-021-25801-2.pdf ; https://www.eurekalert.org/news-releases/928952
- NG-RC展開（プレプリント） — https://scirate.com/arxiv/2412.09800 ; https://scirate.com/arxiv/2512.12903
- TDKプレスリリース (2025-10-02) — https://www.tdk.com/en/news_center/press/20251002_01.html ; https://www.tdk.com/ja/news_center/press/20251002_01.html ; 解説 https://www.tdk.com/ja/featured_stories/entry_083-analog-reservoir-computing-AI-chip.html
- QCi NeuraWave報道 — https://convergedigest.com/qci-launches-neurawave-photonic-platform/ ; https://quantumcomputingreport.com/quantum-computing-inc-launches-neurawave-photonic-platform-for-edge-ai-inference/

エッジ・比較・課題
- FPGA ESN, ACM TRETS — https://dl.acm.org/doi/10.1145/3811539
- CMOS SNN RC, arXiv:2509.17355 — https://arxiv.org/html/2509.17355v1
- Simplified RC for Edge AI, MWSCAS 2025 — https://par.nsf.gov/servlets/purl/10672112
- Reservoir Computing as a Language Model, arXiv:2507.15779 — https://arxiv.org/html/2507.15779v1
- ParalESN（スケーラビリティ批判） — https://raw.githubusercontent.com/mlresearch/v306/main/assets/pinna26a/pinna26a.pdf
- Ma et al. (2025) Advanced Science（信頼性批判） — https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202416413
- Gallicchio DeepRC講義資料（DeepESN書誌） — https://www.slideshare.net/slideshow/claudio-gallicchio-deep-reservoir-computing-for-structured-data-239336864/239336864
