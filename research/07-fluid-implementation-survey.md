# ブラウザ（WebGPU）で動く流体シミュレータゲームのための技術調査レポート

- 調査日: 2026-11-24
- 前提: 物理層は明示的な数値解法（状態方程式と相変化を担当）、リザバーコンピューティング（ESN等）は粗格子で解像できない部分の補正（クロージャ）と潜在状態の時間発展予測を担当するハイブリッド構成。優先順位は「見た目の自然さ」と「破綻しないこと」であり、工学的精度ではない。
- 表記ルール: 本レポートでは各節で **【確立】**＝複数の独立した論文・実装で繰り返し確認されている話、**【単一論文】**＝1本の論文／1事例だけが根拠の主張、**【空白】**＝今回の調査で事例が見つからなかった領域、を区別する。各主張には出典（タイトル・著者・年・掲載誌／arXiv番号・URL）を付す。URLは調査で実際に取得したものをそのまま記載する。

## Summary（要約）

1. **物理層の土台は今もStable Fluids型の格子法でよい。** 半ラグランジュ移流＋圧力投影（Poisson方程式＋勾配差し引き）＋渦度保存（vorticity confinement）という構成は、1999年以降のリアルタイム流体の事実上の標準であり、WebGLでもWebGPU（WGSL）でも同じパイプラインがそのまま実装されている【確立】。
2. **圧力の扱いは目的で分けるのが正解である。** 見た目優先の格子法では圧力は「非圧縮条件を満たすためのラグランジュ乗数」として投影法で求めるのが標準。一方、粒子法（SPH）で「状態方程式から圧力を直接計算する」場合はTait方程式 `p = B((ρ/ρ0)^γ − 1), γ=7, B=ρ0 c^2/γ` が標準形であり、人工音速 `c ≈ 10 × 最大流速` で密度変動を約1%に抑える運用が定着している【確立】。本プロジェクトのように「状態方程式を物理層が担当する」設計では、この2つを混同しないことが重要である（格子法にTait式をそのまま入れると二重に圧力を数えることになる。§2参照）。
3. **相変化はエンタルピー法が固定格子と最も相性がよい。** 界面を明示追跡せず、エンタルピー `H` を主変数にしてStefan条件を陰に含める定式化は、固定格子・ゲーム向け格子ソルバと構造的に一致する【確立】。ただし氷・水・水蒸気の三相を1つの格子ソルバで「それらしく」扱った公開ゲーム事例は乏しく、商用／同人ゲームの多くは温度閾値による状態遷移ルールで割り切っている【空白に近い】（§3）。
4. **学習補正は「ソルバを残し、解像できない部分だけ学ぶ」が成功パターンである。** Solver-in-the-Loop（Um et al. 2020）とKochkov et al. 2021はいずれも、純粋なブラックボックス予測ではなく、粗いソルバの中の特定の演算（補正項／対流フラックスの補間）を学習に置き換えることで長期安定性と汎化を得ている【確立しつつある】。本プロジェクトの「ESNによるクロージャ」という発想はこの系譜に乗るが、ESNをクロージャとして格子ソルバに組み込んだ直接の成功例はCNN系に比べて少なく、ここが本プロジェクトの新規性であり同時にリスクである（§4）。
5. **潜在空間予測には既知の落とし穴がある。** 全状態が訓練データにある場合はRCがBPTT系RNNより速く正確だが、**縮約（reduced order）データでは大規模RCが不安定化し発散しやすい**ことがVlachas et al. 2020で明示されている【確立に近い重要な警告】。対照的にCAE-ESN（Racca et al. 2023）は潜在次元を物理空間の1%以下まで落として時間正確・統計的に予測できたと報告する【単一論文系統】。両者を併せて読むと、「潜在次元の選び方と安定化機構なしにESNを自律運転させない」が実務原則になる（§5）。
6. **Babylon.jsのWebGPU経路は実用段階にあるが、フォールバック設計が必須である。** `ComputeShader`／`StorageBuffer`はv5.0以降のWebGPU専用機能であり、対応判定は `WebGPUEngine.IsSupportedAsync`、非対応時はWebGLエンジンへ落とすのが公式ドキュメントの示す手順である。2026年11月時点でもWebGPU未対応環境（Firefox on Linux/Android等）は残る【確立】（§6）。
7. **最初にやるべきは学習ではなく「壊れない粗ソルバ＋壊れない補正の差し込み口」の検証である。** 実験順序の提案を§8に置く。

---

## Findings

### 1. リアルタイム流体シミュレーションの基礎 — Stable Fluids以降とWebGPU実装

#### 1.1 標準構成【確立】

- Jos Stam「Stable Fluids」, SIGGRAPH 1999が起点である。1ステップは作用素分割（operator splitting）で、(1)外力付加、(2)半ラグランジュ移流、(3)拡散（粘性）、(4)圧力投影、の順に処理する。半ラグランジュ移流は各格子点から速度場を逆向きに辿り（backtrace: `x_back = x − Δt·u(x)`）、旧場を双線形補間でサンプリングするため、CFL条件に縛られず無条件安定である代わりに、数値拡散で流れが鈍る（blurryになる）ことが知られている。出典: Stam (1999) の解説・再構成 “Stable Fluids”, Algorithmページ, https://algorithm.tamnd.com/stable-fluids/ ; 同趣旨の実装解説 “Solving Navier–Stokes at 60 FPS: Stable Fluids & Helmholtz–Hodge Projection in WebGL”, https://arman1o1.github.io/posts/physics/real-time-navier-stokes-webgl.html
- 圧力投影はHelmholtz–Hodge分解に基づく。移流後の速度 `w` を `w = u + ∇p, ∇·u = 0` と分解すると圧力Poisson方程式 `∇²p = ∇·w` が得られ、これを解いて `u = w − ∇p` と差し引く。ブラウザ実装ではPoisson方程式をJacobi反復（20〜30回程度）で解くのが通例である（例では26回、別例では約20 sweep）。出典: 同上WebGL解説 https://arman1o1.github.io/posts/physics/real-time-navier-stokes-webgl.html
- 渦度保存／渦度閉じ込め（vorticity confinement）はFedkiw, Stam & Jensen (2001)「Visual Simulation of Smoke」がグラフィクス分野に導入した手法として位置づけられ、半ラグランジュ移流の数値拡散で失われる小渦を、渦度（curl）の勾配方向への力で補う。実装上は「curl計算 → 渦度力付加」の追加パスになる。出典: 同上WebGL解説（Fedkiw, Stam & Jensen 2001への言及を含む） https://arman1o1.github.io/posts/physics/real-time-navier-stokes-webgl.html ; WGSLカーネル集 “@vshaders/fluid” のvorticity節 https://www.vshaders.sh/docs/fluid
- 格子配置については、Stam原論文は全成分をセル中心に置くcollocated配置だったが、Fedkiw et al. (2001)以降、速度を面に置くMAC（Marker-and-Cell）格子が煙・流体シミュレーションの標準的な選択肢になっている。出典: “Stable Fluids”, https://algorithm.tamnd.com/stable-fluids/

> 実務メモ（本プロジェクト向け）: 2Dで始めるならcollocated＋Jacobi投影で十分「らしく」なる。MAC化は境界条件と発散の精度が問題になってからでよい。渦度閉じ込め係数は「見た目の生き生きしさ」を決める最重要ノブの1つであり、学習補正を入れる前にこの古典的ノブだけでどこまで出せるかを測るべきである（§8実験1）。

#### 1.2 ブラウザ実装事例【確立】

- **WebGLの代表作**: Pavel Dobryakov（PavelDoGreat）「WebGL-Fluid-Simulation」は、NVIDIA GPU Gemsのソルバ記述に基づくブラウザ実装で、モバイルでも動作し、オープンソースの参照実装として広く使われている。構成は速度・染料（dye）・圧力のdouble framebuffer（ping-pong）、移流・発散・curl・渦度・圧力・勾配差し引きの各フラグメントシェーダパスである。出典: DeepWiki “PavelDoGreat/WebGL-Fluid-Simulation”, https://deepwiki.com/PavelDoGreat/WebGL-Fluid-Simulation およびWebGL実装節 https://deepwiki.com/PavelDoGreat/WebGL-Fluid-Simulation/3.2-webgl-implementation
- **wgpuへの移植例**: 同じDobryakov実装をwgpu（RustのWebGPU実装）へ移植し、ネイティブとブラウザで同一コードを動かす事例がある。数値解法はStamのStable Fluids、フレーム構成はMark Harris「GPU Gems 38」に基づくと明記されている。出典: threers “Fluid”ドキュメント, https://raw.githubusercontent.com/eugenehp/threers/main/docs/fluid.md
- **WebGPU 3Dの事例**: Taos EngineのVolumetric Fluidサンプルは、3D Stable Fluidsソルバ（移流・渦度・外力・発散・圧力・投影の6つのコンピュートカーネル）をWGSLコンピュートシェーダで実装し、格子 `64×96×64`、3Dテクスチャに `TEXTURE_BINDING | STORAGE_BINDING` を付けてping-pong運用し、密度場をレイマーチングで描画する構成を公開している。出典: “Volumetric Fluid — Technical Deep Dive”, https://taosengine.com/book/samples/fluid_test.html
- **WGSLカーネル部品化の事例**: `@vshaders/fluid` はStable Fluidsのカーネル（半ラグランジュ移流、発散・Jacobi圧力緩和・勾配差し引きによる投影、渦度閉じ込め、Gaussian splat）を純粋なWGSLモジュールとして提供し、「流体は単一シェーダではなく小さなパスグラフである」と設計思想を明記している。Jacobi圧力反復は20回がデフォルトの目安とされる。出典: https://www.vshaders.sh/docs/fluid

#### 1.3 WGSLで格子法を書く際の実務上の注意点【確立＋実装知見】

- **ping-pongが必須**: 近傍を読むパスでは、読み取り場と書き込み場を同一にできない。速度・密度は2スロットを用意しホスト側でインデックスを入れ替える（Taos事例、上記）。粒子系のように「自分の要素だけ触る」パスはin-place可だが、近傍参照がある流体パスはdouble-bufferが原則である。出典（一般則）: “WebGPU Compute Use Cases”, https://skillmd.com/skills/impertio-studio/webgpu-impl-compute-usecases
- **ワークグループ同期の限界**: `workgroupBarrier()` はワークグループ内しか同期できない。ワークグループを跨ぐ依存（例: Jacobiの1反復全体）は別パス（別dispatch）に分割するしかない。これが「Jacobi 20〜30パス」という構造の理由である。バリアはワークグループ内で一様（uniform）な制御フロー内でのみ安全で、発散した分岐内での使用は未定義である。出典: “WGSL Compute Shaders”, https://skillmd.com/skills/impertio-studio/webgpu-wgsl-compute-shaders
- **境界チェックは省略不可**: dispatchはワークグループ単位で切り上げられるため、格子サイズがワークグループサイズの倍数でない場合に範囲外アクセスが起きる。`global_invocation_id` の境界チェックを全カーネルに入れる。出典: “Getting Started With WebGPU Compute Shaders”, https://www.newsnix.com/articles/webgpu-compute-shaders-tutorial-getting-started/
- **ストレージ・テクスチャの制約**: Babylon.js経由ではストレージ・テクスチャは「書き込み専用」として扱われ、読む場合は通常テクスチャとして渡し直す（§6）。素のWebGPUでも、同一テクスチャへの同時読み書きは順序が保証されないため入力／出力テクスチャは分ける。出典: Babylon.js公式 “Compute Shaders”（2026-11-24にライブ確認）, https://doc.babylonjs.com/features/featuresDeepDive/materials/shaders/computeShader/
- **精度**: WGSLの基本浮動小数点はf32である。圧力PoissonのJacobi反復はf32で十分回ることが上記各実装で示されているが、長時間積分でのエネルギー drift や、エンタルピーのような「大きな基準値＋小さな差分」を持つ量では桁落ちに注意する。相変化（§3）では温度ではなくエンタルピーの増分側を保持する等の工夫が効く。これは本レポート著者の設計助言であり、特定論文の主張ではない。
- **デバイス上限**: 既定上限として `maxStorageBufferBindingSize = 128 MiB`、`maxBufferSize = 256 MiB`、`maxComputeWorkgroupStorageSize = 16 KiB`、`maxComputeInvocationsPerWorkgroup = 256` 等がある。2D格子（例: 256×256×数場×4byte ≈ 数MB）では全く問題にならないが、3D高解像度＋多数の場では binding size と3Dテクスチャ上限（既定2048）が先に効く。出典: WebGPU上限の整理 “The Limits of WebGPU & WebAssembly in the Browser (November 2026)”, https://ttslab.dev/limits

### 2. 状態方程式と圧縮性 — Tait方程式とゲーム向け圧力の扱い

#### 2.1 Tait方程式の標準形【確立】

弱圧縮性SPH（WCSPH）で使う形は次である。

```
p = B · ((ρ/ρ0)^γ − 1),   γ = 7（水）,   B = ρ0 · c^2 / γ
```

- `ρ0` は静止密度、`c` は（物理的な音速ではなく）人工音速、`B` は剛性パラメータである。出典: “Weakly compressible SPH”, https://algorithm.tamnd.com/weakly-compressible-sph/ ; “SPH: Fluids from Particles”, https://www.mysimulator.uk/articles/sph-fluids/ ; DualSPHysicsの定式化紹介（γ=7、Tait状態方程式で閉じる旨を明記）, https://sfmc22.uca.es/wp-content/uploads/2022/06/ABSTRACT_80.pdf
- 運用則【確立】: 人工音速は最大流速の約10倍（`c ≈ 10·v_max`）に取る。これで相対密度変動 `Δρ/ρ0 ≈ v^2/c^2 ≈ 0.01`、すなわち約1%以内に収まる。水の物理音速（約1480 m/s）をそのまま使うとCFL条件で時間刻みが非現実的に小さくなるため、この「音速を偽る」ことがWCSPHの本質的な割り切りである。出典: 同上2件 https://algorithm.tamnd.com/weakly-compressible-sph/ , https://www.mysimulator.uk/articles/sph-fluids/
- 代償【確立】: 線形系を解かなくてよい代わりに、剛性が高いほど時間刻みを小さくしなければならず、「書き方は単純だが1秒の運動に多数のステップが要る」のがWCSPHの主コストである。出典: https://algorithm.tamnd.com/weakly-compressible-sph/
- 実装バリエーション: Project ChronoのFSI-SPHでは `eos_type` としてTAIT（`p = B[(ρ/ρ0)^γ − 1] + p_base, γ=7`）とISOTHERMAL（線形形 `p = Cs^2 (ρ − ρ0)`）が併記され、自由表面・衝撃・スロッシングにはTAIT、低速の内部流れには線形形で十分とする使い分けが示されている。視覚目的のリアルタイムデモでは線形状態方程式 `p = k(ρ − ρ0)` で足りる場合が多いことも指摘されている。出典: “Chrono::FSI-SPH Parameter Selection Guide”, https://api.chrono.projectchrono.org/manual_fsi_sph_parameter_selection.html ; https://www.mysimulator.uk/articles/sph-fluids/

#### 2.2 ゲーム向け格子法での圧力 — 本プロジェクトへの含意

- 格子法（§1）では圧力は状態方程式から決まる物理量ではなく、非圧縮条件 `∇·u = 0` を課すための乗数場であり、Poisson方程式で毎ステップ求める【確立、§1.1出典】。
- したがって本プロジェクトの設計分担は次のように整理すると破綻しない（本レポートの設計提案）:
  1. **液体内部の圧力**: 投影法が担当（Tait式は使わない）。
  2. **Tait式／線形EOSが効く場所**: (a) 粒子系サブシステムを併用する場合、(b) 気相（水蒸気）や沸騰のような「圧縮性が本質」の現象を格子上で安く表現したい場合に、密度→圧力の局所閉形式として使う、(c) 相変化に伴う体積変化を速度場の発散源（source term）として注入する場合の換算係数として使う。
  3. **やってはいけないこと**: 投影法で得た圧力場にさらにEOS圧力勾配を外力として二重に加えること。やるなら投影の前段（予測速度への寄与）に限定し、投影で必ず引き直す。

### 3. 相変化の数値解法 — エンタルピー法・Stefan問題とゲーム向け簡略化

#### 3.1 Stefan問題とStefan条件【確立】

- Stefan問題は融解・凝固のような移動境界（固液界面）を持つ熱伝導問題の古典的設定で、Josef Stefan（1835–1893）が極地探検の海氷データとの比較を通じて定式化したことに由来する。界面でのエネルギー収支がStefan条件を与える: `ρ_s · L · dX_m/dt = −κ ∂T/∂x|界面`（`L` は融解潜熱、水の氷では `L ≈ 334 kJ/kg` ※出典ページでは単位表記が `334 J/kg` と表示されている箇所があるが、物理的には334 kJ/kg＝334 J/gが正値であり、他出典でも `Lf = 334 J/g` と一致する）。出典: RWTH Aachen講義ノート “Solid-liquid phase change”, https://rwth-mbd.pages.git.nrw/courses/cmm/content/script/chapters/phase_change.html ; 潜熱値の裏付け “Heating Curve Simulator”, https://webtoolyard.com/heating-curve-simulator/
- 水の蒸発潜熱は `Lv = 2,257 J/g` で融解潜熱の約7倍である。ゲームで「沸騰が凍結よりずっと遅い／エネルギーが要る」と感じさせるには、この比を（圧縮してでも）残すのが最も安いリアリティである。出典: https://webtoolyard.com/heating-curve-simulator/

#### 3.2 エンタルピー法の標準的定式化【確立】

- 考え方: エンタルピー `H` を従属変数に取り直すことで、固相・液相・界面に別々の式と界面条件を用意する必要をなくし、**全領域で単一のエネルギー方程式** `ρ ∂H/∂t = ∇·(λ∇T)` を固定格子で解く。Stefan条件は陽に課すのではなく定式化に陰に含まれる。界面位置は解いた後に `H`（または液相率）から事後的に求める。出典: ScienceDirect Topics “Enthalpy Method”概要（Applied Thermal Engineering 2023に基づく）, https://www.sciencedirect.com/topics/engineering/enthalpy-method ; Mochnacki, Lara & Pawlak, “Application of the enthalpy approach in numerical modelling of solidification process”, Scientific Research of the Institute of Mathematics and Computer Science, 2002, 1(1), 163–176, https://amcm.pcz.pl/2002_1/art_22.pdf
- `H` と `T` の関係: 顕熱 `h = ∫ c_p dT` に潜熱寄与 `ΔH = f_l · L`（`f_l` は液相率）を加えたものが全エンタルピーであり、融解温度を挟む区間では `T` が（ほぼ）一定のまま `H` だけが増える——これが加熱曲線のプラトー（温度が止まる区間）の正体であり、ゲームで再現すべき最小の現象である。出典: 同上ScienceDirect概要；Columbia講義資料 “Using enthalpy method to deal with phase change difficulties”, http://www.aml.engineering.columbia.edu/ntm/level3/ch03/html/l3c03s05.html
- 流れと連成させる拡張（enthalpy–porosity法）【確立】: Voller & Prakash (1987)「A fixed grid numerical modelling methodology for convection-diffusion mushy phase-change problems」, International Journal of Heat and Mass Transfer, 30(8), 1709–1719 は、相変化領域（mushy zone）を多孔質媒体とみなし、多孔率（＝液相率）に応じたDarcy型の運動量シンク項で固相域の速度を抑える固定格子法を与えた。ANSYS FluentやOpenFOAMの凝固・融解モデルがこの系譜にある。出典: OpenFOAM v1806リリースノート（Voller & Prakash 1987およびSwaminathan & Voller 1992を典拠として明記）, https://www.openfoam.com/news/main-news/openfoam-v1806/solver-and-physics
- 注意点【単一〜少数論文】: 純物質（融解温度が一点）のエンタルピー法では、界面が階段状に進む、温度履歴に非物理的なプラトーが余分に現れる、といった既知の数値的アーティファクトが報告されており、正則化（融解温度区間を小さく広げる）や局所Newton反復による改良が提案されている。ゲーム用途ではむしろ「融解温度区間を意図的に広げる」ことが、見た目の滑らかさと安定性を両立させる実務的な正則化になる。出典: Luddens, Lothodé & Danaila, “An implicit regularized enthalpy Lattice Boltzmann Method for the Stefan problem”, International Journal of Heat and Mass Transfer誌へ投稿のプレプリント (2025), https://borowiecki.dev/pdf/2505.13097

#### 3.3 グラフィクス／ゲーム向けの割り切り事例

- **「固体＝超高粘性流体」方式**【確立したグラフィクス手法】: Carlson, Mucha, Van Horn & Turk「Melting and Flowing」(SIGGRAPH 2002系統) は、融解・流動・再凝固を、温度等に応じて空間的に粘性を変える非圧縮粘性Navier–Stokes（MAC法の改変）で表現した。固体は別ソルバではなく「粘性が極端に高い流体」として扱うため、蝋・溶岩・氷（水）まで同一ソルバで連続的に遷移できる。本プロジェクトの格子ソルバに相変化を足す際の最も実装量の少ない道は、エンタルピー法で液相率を求め、液相率→粘性／速度抑制（Darcyシンク）に変換するこの方式とのハイブリッドである。出典: Carlson et al., “Melting and Flowing”, https://docslib.org/download/2747190/melting-and-flowing ; Carlson博士論文 “Rigid, Melting, and Flowing Fluid”, Georgia Tech (2004), https://repository.gatech.edu/entities/publication/56f77990-ae6e-4876-aa30-64722b57d1bc/full
- **セルオートマトン方式**【単一系統】: Wei, Li & Kaufman「Melting and Flowing of Viscous Volumes」は、PDEによる厳密な熱・相転移計算を避け、近傍ボクセル間の熱伝導＋3Dセルオートマトンで高粘性固体の融解を表現し、対話速度を優先した。「見た目優先」の割り切りの古典例である。出典: https://cvc.cs.stonybrook.edu/sites/default/files/vislab/2021-05/Melting.pdf
- **粒子法での氷の融解・凍結**【単一系統】: Iwasaki, Uchida, Dobashi & Nishita はSPH等の粒子法に熱伝達・氷⇄水の相転移・融解水の滴流を組み込みCUDAで対話フレームレートを達成したと報告する（ResearchGate経由の要旨で確認）。出典: https://www.researchgate.net/publication/268300180_Computational_Fluid_Dynamics_in_2D_Game_Environments （当該論文の要旨を含むページ）
- **閾値ルール方式（ゲーム／サンドボックス）**【事例・工学モデルではない】: ブラウザ・サンドボックス系では「水は沸点閾値で蒸気に、凝固閾値で氷に、蒸気は冷却で水に戻る」という温度閾値＋局所熱交換ルールで水循環を表現する例があり、湿度・圧力・真の蒸発速度はモデル化しないことが明記されている。Steamで開発中の3Dボクセル・サンドボックス「Base Matter」も「意図的に単純化され、科学・工学解析のためではない」と明言した上で凍結・沸騰・凝縮を実装している。出典: Sandboxels水循環ガイド, https://www.sandboxels.cc/sandboxels-water-cycle/ ; “Base Matter”, Steamストアページ, https://store.steampowered.com/app/5365110/Base_Matter/?snr=1_7_7_230_150_9
- **【空白】**: 氷・水・水蒸気を単一の格子Navier–Stokesソルバ＋エンタルピー法で連成させ、ブラウザ上で対話的に動かした公開事例は、今回の調査では見つからなかった。沸騰（気液相変化＋体積膨張＋気泡）を格子法で扱う研究は工学側には存在するが、ゲーム／WebGPU実装への接続は空白地帯である。本プロジェクトがここを実装すれば、それ自体が新規の実験報告になる。割り切り案（本レポートの提案）: 沸騰は「液相率の逆方向」として扱わず、温度が沸点を超えたセルのエンタルピー超過分を (a) 蒸気染料場への変換、(b) 発散源（膨張）として速度場へ注入、の2つに分ける。潜熱比 `Lv/Lf ≈ 6.8` をそのまま使うと沸騰が遅すぎてゲームにならないため、潜熱は見た目のテンポに合わせて1/5〜1/20に圧縮し、比だけを残すことを推奨する。

### 4. 機械学習によるクロージャ／補正

#### 4.1 Solver-in-the-Loop — Um et al. 2020【中核論文】

- Um, Brand, Fei, Holl & Thuerey, “Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers”, NeurIPS 2020. 粗いPDEソルバの離散化誤差を学習で補正する際、教師データを事前に作って当てはめる方式（NON／PRE）ではなく、**微分可能なソルバを訓練ループに入れ、補正モデル自身が変えた状態分布の上で訓練する**方式（SOL）が決定的に優れることを示した。数百ステップの安定したロールアウトを達成し、非定常後流では平均絶対誤差を73%以上削減した例が報告されている。対象は移流拡散から3D Navier–Stokesまで。出典: NeurIPS 2020論文PDF, https://proceedings.nips.cc/paper_files/paper/2020/file/43e4e6a6f341e00671e123714de019a8-Paper.pdf ; arXiv版 arXiv:2007.00016 のHTML版, https://ar5iv.labs.arxiv.org/abs/2007.00016
- 本プロジェクトへの含意: ESNは微分可能ではない（出力層の線形リッジ回帰のみ学習）ため、BPTTによるSOL訓練はそのまま使えない。代わりに (a) ロールアウトを教師なしで長く回し、自己生成状態を訓練データに混ぜる「DAgger的」データ拡張、(b) 補正をソルバの外ではなく内（例: 渦度力や発散源の項）に置き、物理層が誤差を毎ステップ引き戻す構造にする、の2点でSOLの教訓（分布シフト対策）を移植するのが現実的である（本レポートの設計提案）。

#### 4.2 Kochkov et al. 2021 — 学習でCFDを高速化【中核論文】

- Kochkov, Smith, Alieva, Wang, Brenner & Hoyer, “Machine learning–accelerated computational fluid dynamics”, Proceedings of the National Academy of Sciences, 2021, 118(21), e2101784118. arXiv:2102.01010. 2D乱流について、従来ソルバの「解像度不足で最も壊れる部品」（対流フラックスの補間＝learned interpolation）をCNNに置き換え、端から端まで（end-to-end）で訓練することで、各空間次元で8〜10倍粗い格子でベースラインと同精度、40〜80倍の高速化を達成した。長時間シミュレーションでも安定で、訓練外の強制項・レイノルズ数にも汎化した——ブラックボックス型MLアプローチとの対比が明記されている。出典: PNASオープンアクセス版（PMC）, https://pmc.ncbi.nlm.nih.gov/articles/PMC8166023/ ; arXiv, https://arxiv.org/abs/2102.01010?context=cs
- 構造の要点: 有限体積法＋スタッガード格子＋圧力投影という古典ソルバの骨格は残し、学習は補間係数の生成だけを担う。本プロジェクトの「物理層＋ESN補正」と発想は同じだが、Kochkov系は補正対象が「局所ステンシルの係数」である点が重要である。ESNのような時間方向のモデルで空間ステンシルを補正するのは役割分担として不自然であり、ESNには (a) 時間発展の残差、(b) 潜在状態の予測、を持たせる方が文献の成功パターンに沿う（本レポートの設計提案）。

#### 4.3 乱流サブグリッド（クロージャ）モデルの学習【確立しつつある】

- Beck, Flad & Munz, “Deep neural networks for data-driven LES closure models”, Journal of Computational Physics, 2019（DOI: 10.1016/j.jcp.2019.108910）. DNS（減衰等方性乱流）から「完全なLES閉包項」を厳密に定義して訓練データを作り、局所畳み込みフィルタ型DNNで粗格子量→閉包項の写像を学習した。粗格子原始変数とLES演算子の両方を入力特徴に入れると訓練結果が大きく改善すること、学習した閉包項をそのまま使うと安定性の問題が出るため、データ適応的な渦粘性（eddy viscosity）閉包に翻訳して初めて安定で正確なLESになることが示された。**「a prioriに当てはまってもa posteriori（実際に回すと）安定とは限らない」**という教訓の代表例である。出典: IAEA INIS収録の書誌・要旨, https://inis.iaea.org/records/5xajk-wvh93
- 総説的位置づけ: Duraisamy, “Perspectives on Machine Learning-augmented Reynolds-averaged and Large Eddy Simulation Models of Turbulence”, arXiv:2009.10675 (2020/2021) は、RANS／LESの未閉包項・モデル不一致・サブフィルタスケールを教師あり学習で表すアプローチ群を整理し、汎化可能なML拡張モデルの課題を論じる。出典: https://arxiv.org/pdf/2009.10675
- 微分可能ソルバでLESモデルを学ぶ後続: “Learned turbulence modelling with differentiable fluid solvers: physics-based loss functions and optimisation horizons”, Journal of Fluid Mechanics は、学習モデルをソルバに統合するハイブリッド手法の系譜の中でBeck et al. (2019)等を位置づけ、損失関数の物理的設計と最適化ホライズン（何ステップ先まで見て訓練するか）が性能を左右することを論じる。出典: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/learned-turbulence-modelling-with-differentiable-fluid-solvers-physicsbased-loss-functions-and-optimisation-horizons/28D19239CEDB81A3DA58F32E0E8CB3B2

#### 4.4 リザバーコンピューティングの流体・乱流・ROMへの直接適用（Pathak et al. 2018以外）

- **Physics-constrained／Physics-informed ESN（PI-ESN）**【流体への直接適用】: Doan, Polifke & Magri, “Short- and long-term prediction of a chaotic flow: A physics-constrained reservoir computing approach”, Proceedings of the Royal Society A, 2021, 477, 20210135（プレプリント arXiv:2102.07514／初期版 arXiv:1912.10994 “A physics-aware machine to predict extreme events in turbulence”）。乱流せん断流の低次元モデル（Moehlis–Faisst–Eckhardtの9方程式モデル）に対し、保存則を訓練に組み込んだESNで、極端事象（乱流から準層流への突発遷移）の時刻正確な予測と長期統計の再現を、データのみ／物理のみのどちら単独でもできない形で達成し、ノイズにも頑健だったと報告する。出典: https://arxiv.org/html/2102.07514v2 ; https://arxiv.org/pdf/1912.10994 ; 書誌確認 https://iris.polito.it/retrieve/handle/11583/2995088/155dd8dc-9c55-4f71-82db-491ceb019948/Short-%20and%20long-term%20predictions%20of%20chaotic%20flows%20and%20extreme%20events_%20a%20physics-constrained%20reservoir%20computing%20approach.pdf
- **小データからの極端事象の統計予測**: Racca & Magri 系の研究 “Statistical prediction of extreme events from small datasets” は、極端事象を含まない小さな訓練データでESNを訓練しても、重い裾を持つ統計を外挿的に改善できる場合があると報告する【単一系統】。出典: https://ar5iv.labs.arxiv.org/html/2201.08294
- **部分観測からの予測**: “Model-Free Forecasting of Partially Observable Spatiotemporally Chaotic Systems”, arXiv:2212.00001 は、非線形射影（RBF等）で観測を高次元化してからリザバーに入力することで、方程式を知らず部分観測しかなくても時空間カオス（乱流の特徴を模した系）を予測できるとする【単一論文】。出典: https://arxiv.org/pdf/2212.00001
- **総説での位置づけ**: Pandey, Schumacher & Sreenivasan, “A perspective on machine learning in turbulent flows”, Journal of Turbulence, 2020（DOI: 10.1080/14685248.2020.1757685）は、等方性乱流・Rayleigh–Bénard対流・乱流チャネルの最小流ユニットに対するML適用を論じ、最後のケースでESNによる解析を詳述する。出典: https://1library.net/document/yenj2e1y-a-perspective-on-machine-learning-in-turbulent-flows.html
- **POD＋ESN＋データ同化**: 円柱後流（Re=100、2D非定常層流後流）を対象に、PODで縮約した時間係数をESNで予測し、Ensemble Kalman Filter系のデータ同化でオンライン適応させる “Online model learning with data-assimilated reservoir computers” の系統がある。ノイズ付き観測からでも再構成誤差を下げられるという報告は、ゲームで「プレイヤーの攪拌＝常時入る外乱」がある状況への示唆になる【単一系統・二次資料経由】。出典（レビュー要約）: https://www.themoonlight.io/en/review/online-model-learning-with-data-assimilated-reservoir-computers
- **CAE-ESN**: §5で詳述（Racca, Doan & Magri 2023）。ESNの流体適用として現時点で最も「格子流れ場そのもの」に近い成功例である。
- **【空白に近い点】**: ESNを「格子ソルバ内部のサブグリッド・クロージャ項」として毎ステップ呼び出す形の流体論文は、今回の調査ではCNN系（Beck等）ほど確立した例が見つからなかった。ESNの実績は主に「低次元化された時間発展の予測」側に集中している。したがって本プロジェクトでは、ESNの第一の役割を潜在時間発展に置き、クロージャは「潜在補正→物理場への射影」という間接形から始めるのが文献に忠実である。

### 5. 潜在空間での予測 — 圧縮＋予測と不安定性

#### 5.1 代表的な構成【確立しつつある】

- **Autoencoder＋LSTM**: Eivazi／Hasegawa系を含む系統の代表として、 “Deep Neural Networks for Nonlinear Model Order Reduction of Unsteady Flows”, Physics of Fluids, 2020, 32, 105104（arXiv版 https://arxiv.org/html/2007.00936 ）は、オートエンコーダで非線形次元削減（SVDの代替）を行い、抽出した潜在特徴をLSTMに入れて将来の速度場を予測するAE-LSTMを、POD／DMD系の非侵入型ROMと比較し、決定係数で優位だったと報告する。
- **CNN-AE＋LSTM（乱流チャネル）**: “CNN-AE/LSTM based turbulent flow forecast on low-dimensional latent space” (NeurIPS 2020 ML4PS Workshop) は、CNNオートエンコーダで空間を潜在ベクトルに落とし、LSTMでその時間発展を追うことで高次元乱流チャネル流の統計的一致（時間アンサンブル）を得たと報告する【ワークショップ論文】。出典: https://ml4physicalsciences.github.io/2020/files/NeurIPS_ML4PS_2020_8.pdf
- **CAE-ESN（本プロジェクトに最も近い）**【単一論文だが査読誌】: Racca, Doan & Magri, “Predicting turbulent dynamics with the convolutional autoencoder echo state network”, Journal of Fluid Mechanics, 2023, 975, A2（DOI: 10.1017/jfm.2023.716）。畳み込みオートエンコーダ（CAE）で乱流状態の潜在多様体を求め（空間問題）、ESNで潜在状態の時間発展を予測し（時間問題）、両者を自律力学系CAE-ESNとして結合する。2D Kolmogorov流と3D最小流ユニットで、(i) 物理空間の1%以下の自由度への縮約、(ii) 異なるレイノルズ数での時刻正確・統計的予測、(iii) 元方程式を解く場合の1%以下の計算時間、を報告する。潜在9変数でPOD 100変数より良い再構成が得られたという比較も著者ページで紹介されている。出典: https://resolve.cambridge.org/core/journals/journal-of-fluid-mechanics/article/predicting-turbulent-dynamics-with-the-convolutional-autoencoder-echo-state-network/1E0F75CD94FCB3A1354A09622F8D25CD ; 著者解説 https://www.lucamagri.com/nonlinear-model-reduction.html
- **PCA系**: PCA／PODは線形だが解釈しやすく、潜在係数の時間発展学習の前段として今も使われる。敵対的オートエンコーダ＋敵対的LSTMでCFDのROM予測の発散を抑える試み（Quilodrán-Casas et al., ICLR 2021 SimDL Workshop）は、「潜在分布から外れた予測サンプルを出さない」訓練が長期予測の寿命を延ばすという問題意識を示す【ワークショップ】。出典: https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/2104/2104.06297v2.pdf

#### 5.2 失敗例・不安定性の指摘【重要】

- **Vlachas et al. 2020**: Vlachas, Pathak, Hunt, Sapsis, Girvan, Ott & Koumoutsakos, “Backpropagation algorithms and Reservoir Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal dynamics”, Neural Networks, 2020, 126, 191–217. arXiv:1910.05266。Lorenz-96とKuramoto–Sivashinsky方程式をベンチマークにRCとBPTT（LSTM／GRU等）を比較し、結論は二分される:
  - **全状態データがある場合**: RCが予測性能・長期統計の再現でBPTTを上回り、訓練時間も大幅に短い。
  - **縮約（reduced order）データの場合**: **大規模RCモデルは不安定になりやすく、BPTT系より発散しやすい**。BPTT系RNNの方が縮約系の力学をよく捉える。
  出典: arXiv https://arxiv.org/abs/1910.05266 ; PubMed書誌 https://pubmed.ncbi.nlm.nih.gov/32248008/
- この結果は本プロジェクトへの直接の警告である。「PCA／AEで圧縮した潜在状態をESNで自律予測する」構成は、まさにVlachas et al. が不安定化を観測した条件（reduced order data＋大規模RC）に一致しうる。対策として文献と整合的なのは: (1) リザバーを大きくしすぎない、(2) スペクトル半径・リーク率・リッジ正則化をロールアウト安定性で選ぶ（1ステップ誤差で選ばない）、(3) 予測を自律運転させず、数ステップごとに物理ソルバの状態で再同期（再射影）する、(4) 潜在予測は「物理場の置き換え」ではなく「補正」として使い、振幅に上限（clamp）を付ける、である（(1)(2)はVlachas／Doan系の知見、(3)(4)は本レポートの設計提案）。
- **NVAR／NG-RCという選択肢**: Gauthier, Bollt, Griffith & Barbosa, “Next generation reservoir computing”, Nature Communications, 2021, 12, 5564. arXiv:2106.07688。RCが非線形ベクトル自己回帰（NVAR）と等価であることを示し、ランダム行列なし・メタパラメータ少数・より短い訓練データで同等以上の予測ができると報告する。潜在次元が小さい（数十以下）本プロジェクトの潜在予測では、ESNよりNVARの方が調整が楽で再現性が高い可能性がある有力な対照実験である。出典: https://arxiv.org/abs/2106.07688 ; PMC版 https://pmc.ncbi.nlm.nih.gov/articles/PMC8455577/
- **a priori／a posteriori乖離の一般論**: §4.3のBeck et al. の教訓（オフライン評価が良くてもオンラインで回すと不安定）は潜在予測でも同型に現れる。評価指標は必ず「自由ロールアウトで何ステップ／何秒持つか」と「長期統計（エネルギースペクトル、平均場）が崩れないか」で取るべきである【確立しつつある評価作法】。

### 6. Babylon.js の WebGPU 対応（2026年時点）

以下はBabylon.js公式ドキュメントを2026-11-24に直接確認した内容である（verified live相当: 公式ドキュメントページを取得して確認）。

#### 6.1 ComputeShader / StorageBuffer の現行仕様【確立・公式】

- コンピュートシェーダは **WebGPU専用機能（v5.0以降）で、WebGLは対応しない**。対応可否は `engine.getCaps().supportComputeShaders` で照会する。出典: Babylon.js公式 “Compute Shaders”, https://doc.babylonjs.com/features/featuresDeepDive/materials/shaders/computeShader/
- 生成は `new ComputeShader(name, engine, { computeSource }, { bindingsMapping })`。ブラウザにWGSLリフレクションがないため、`bindingsMapping` で変数名→`{group, binding}` を手で渡す必要がある。実行は `dispatch()` または `dispatchWhenReady()`。シェーダ言語はWGSL必須で、WGSLソースは `ShaderStore.ShadersStoreWGSL` にも格納できる。出典: 同上
- 渡せる型は: 通常テクスチャ（`setTexture`）、ストレージ・テクスチャ（`setStorageTexture`、生成時に `Constants.TEXTURE_CREATIONFLAG_STORAGE`、ヘルパー `RawTexture.CreateRGBAStorageTexture()` / `CreateRStorageTexture()`）、UniformBuffer（`setUniformBuffer`、レイアウトはシェーダ側の並び順と一致させる必要がある）、StorageBuffer（`setStorageBuffer`、任意データの読み書き用）、サンプラー（`setTextureSampler` / `setTexturesSampler`）。出典: 同上
- ストレージ・テクスチャはBabylon.jsの抽象化では書き込み専用であり、書いたものを読みたい場合は通常テクスチャとして渡し直す。Slime simulationの公式サンプルでは、WebGPUに読み書き両用テクスチャがないためTrailMapをストレージバッファで代替し、バッファ⇄テクスチャ変換を自前で挟んだ旨が明記されている——格子場を「テクスチャで描画、バッファで計算」と分ける場合に踏む実務ポイントである。出典: 同上（Slime simulation節）
- `StorageBuffer.read` は既定で `Engine.onEndFrameObservable` 経由（＝レンダーループが回っている前提）で読み戻し、即時読みしたい場合は第4引数 `noDelay = true` を渡す。毎フレームの読み戻しは避ける設計が前提である。出典: 同上（FAQ節）

#### 6.2 CPUに戻さず描画へ渡す方法【確立・公式サンプル】

- 公式のCompute Boidsサンプルが規範パターンである: 粒子位置・速度を保持する2つのping-pongストレージバッファをコンピュートで更新し、そのバッファをそのまま（インスタンシング用）頂点バッファとして描画に使う。条件は、バッファ生成時に **`BUFFER_CREATIONFLAG_VERTEX` フラグを立てる**こと、およびインスタンスを持たないメッシュに `Mesh.forcedInstanceCount` を設定することである。CPUへの読み戻しは一切発生しない。出典: 同上（Compute Boids節）
- 格子流体への応用（本レポートの設計提案）: 速度・温度・液相率・染料をストレージ・テクスチャ／バッファに置き、(a) 描画はフラグメントシェーダから場テクスチャをサンプリングして直接色付けする（読み戻しなし）、(b) 粒子的な見た目（飛沫・蒸気）が欲しい場合のみBoids方式でバッファを頂点として流用する、の二段構えがBabylon.jsの抽象化と最も喧嘩しない。

#### 6.3 WebGPU非対応環境の扱い【確立・公式】

- エンジン生成は非同期である。公式の手順は `await WebGPUEngine.IsSupportedAsync` で判定し、対応すれば `new WebGPUEngine(canvas)` ＋ `await engine.initAsync()`、しなければ通常のWebGL `Engine` を返す、という分岐であり、`EngineFactory.CreateAsync` はWebGPU→WebGL→NullEngineの順に試すヘルパーとして提供される。**シーンはエンジンを後から乗り換えられない**ため、この判定はシーン生成前に終える必要がある。出典: Babylon.js公式 “WebGPU Breaking Changes”, https://doc.babylonjs.com/setup/support/webGPU/webGPUBreakingChanges
- WebGPUではテクスチャ読み取り（readPixels）が非同期になる等、WebGLとの挙動差がある。出典: 同上
- ブラウザ対応状況（2026年11月時点）: WebGPUはChrome/Edge 113+（2023年5月〜、全プラットフォーム）、Safari 26.0（2025年9月、全Appleプラットフォームで既定有効）、Firefox 141（Windows、2025年7月）／145–147（macOS）で既定有効となり、カバー率は約89%とされる一方、Firefox on Linux／Android、Intel Mac版Safari、Windows ARM64版Chrome等に穴が残る。したがって「WebGPU前提・WebGLフォールバックなし」は2026年時点でも一般公開ゲームには早い。出典: “The Limits of WebGPU & WebAssembly in the Browser (November 2026)”, https://ttslab.dev/limits
- フォールバック設計（本レポートの提案）: WebGL2フォールバックではコンピュートが使えないため、Pavel方式のフラグメントシェーダ版ソルバ（§1.2）を別実装として持つか、解像度を落とした同一WGSLロジックのGLSL移植を持つ。Babylon.jsの描画層は共通化できるが、シミュレーション層は二重実装になることを最初から予算に入れるべきである。なおBabylon.js公式のHydraulic erosionサンプルも「WebGL2では計算をCPU側で行う」構成を取っており、フォールバックで計算経路が変わること自体は公式事例の範囲内である。出典: Babylon.js公式 “Compute Shaders”（Hydraulic erosion節）, https://doc.babylonjs.com/features/featuresDeepDive/materials/shaders/computeShader/

### 7. 学習データ生成 — オフライン「本物のソルバー」からの訓練データ実務

論文から直接読み取れる目安と、そこからの設計指針を分けて書く。

#### 7.1 論文に明記された設定【出典付き】

- **Kochkov et al. 2021**: 2D乱流Kolmogorov流、Re=1000のケースで、学習補間モデルを64×64格子で動かして2048×2048のDNSに匹敵する精度を得た例（実効解像度で32倍）等、粗／細の解像度比が具体的に示されている。訓練は粗ソルバと細参照解のペアを端から端までで結ぶ方式であり、「何ステップ先まで通して訓練するか」が設計変数である。出典: https://pmc.ncbi.nlm.nih.gov/articles/PMC8166023/ ; 解説的要約 https://www.alphaxiv.org/abs/2102.01010v1
- **Beck et al. 2019**: 訓練データは減衰等方性乱流のDNSから生成し、「完全なLES閉包項」を離散化演算子込みで厳密に定義して教師とした。データ量そのものより「閉包項の定義の厳密さ」と「入力特徴にLES演算子を含めるか」が結果を分けた。出典: https://inis.iaea.org/records/5xajk-wvh93
- **Um et al. 2020**: 微分可能ソルバで最大128ステップを通した訓練を行い、数百ステップのロールアウトで評価した。すなわち訓練ホライズン（数十〜百ステップ）と評価ホライズン（数百ステップ）を分けて考えるのがこの系統の作法である。出典: https://ar5iv.labs.arxiv.org/abs/2007.00016
- **ESN系のデータ規模感**: Pandey et al. 2020の総説が紹介する乱流チャネル最小ユニットの研究では、100系列×4000時間ステップ＝400,000データ点で3種のネットワークを訓練し、未見系列で評価した。ESN系は「系列の本数×長さ」で考えるのが通例であり、1本の長い系列より多様な初期条件の系列を揃える方が汎化に効く。出典: https://1library.net/document/yenj2e1y-a-perspective-on-machine-learning-in-turbulent-flows.html
- **NVARの省データ性**: Gauthier et al. 2021はNVARがRCよりさらに短い訓練データで済むことを売りにしている。潜在次元が小さい問題では、まずNVARで必要データ量の下限を測るのが効率的である。出典: https://arxiv.org/abs/2106.07688

#### 7.2 実務指針（本レポートの提案。論文の直接の主張ではない）

1. **参照ソルバ**: まずCPU（Python＋NumPy／JAX等）で「遅くてよい高解像度ソルバ」を1つ作る。例: 目標の実時間格子が128×128なら参照は512×512、時間刻みはCFLと粘性・熱拡散の安定条件から決め、実時間側の `Δt` の1/4〜1/8を初期値とする。相変化を含む場合はエンタルピー法の参照実装を同じコードに含め、液相率場もスナップショットに保存する。
2. **保存量**: 速度（u,v）、圧力、温度／エンタルピー、液相率、染料、の全場を、実時間ソルバのステップ間隔に合わせた間引きで保存する。潜在予測用には全スナップショット、クロージャ用には「粗場→細場の残差」が後から計算できる形で保存する。
3. **データ量の初期目安**: ESN／NVARの潜在予測なら、異なる初期条件・外力条件で50〜200系列 × 各2,000〜10,000ステップを最初の目標にする（上記Pandey総説の400k点が1つの基準点）。足りないかは「未見系列での自由ロールアウト持続時間」で判断し、1ステップ誤差では判断しない。
4. **分割**: 訓練／検証の分割は時刻ではなく系列単位で行う（同一系列の前半・後半で分けるとリークして汎化を過大評価する）。
5. **ゲーム固有の分布**: プレイヤー入力（攪拌・加熱・冷却）は訓練分布に最初から含める。SOLの教訓（§4.1）は「モデルが実際に遭遇する状態分布で訓練せよ」であり、ゲームでは外乱込みの分布がそれに当たる。

---

### 8. 最初に試すべき実験順序の提案

前提: 各実験は「次に進む／戻る」の判定基準（ゲート）を満たしてから次へ進む。学習は最後である。

| # | 実験 | 内容 | ゲート（合格条件） |
|---|------|------|-------------------|
| 0 | Babylon WebGPU疎通 | `WebGPUEngine.IsSupportedAsync` 分岐＋`ComputeShader` 1本（場のコピー／定数加算）＋ストレージ・テクスチャ表示。WebGLフォールバック経路も同時に確認 | 対応／非対応の両環境で画面が出る。読み戻しなしで場が表示される |
| 1 | 古典ソルバ単体（2D） | WGSLでStable Fluids一式（移流・Jacobi投影20〜40回・渦度閉じ込め・染料）。解像度128×128と256×256で比較。学習なし | 10分以上連続運転で発散・NaN・エネルギー爆発なし。渦度係数だけで「らしく」見える範囲を記録 |
| 2 | 熱と浮力 | 温度場＋Boussinesq浮力を追加。加熱／冷却ブラシで対流が立つことを確認 | 煙／熱水の上昇が破綻なく持続する |
| 3 | 相変化（融解・凝固のみ） | エンタルピー法で液相率を導入。固相域はDarcyシンク／高粘性で速度抑制（Voller–Prakash／Carlson方式）。沸騰はまだ入れない | 氷が融け、水が凍る過程で温度プラトー（潜熱）が見え、質量・エネルギーが単調に壊れない |
| 4 | 沸騰の割り切り版 | §3.3の提案（エンタルピー超過→蒸気染料＋発散源、潜熱は圧縮） | 沸騰が「暴れるが破綻しない」範囲にパラメータで収まる |
| 5 | 参照データ生成 | CPU高解像度ソルバで§7.2の仕様でデータセットを作成。粗ソルバとの残差を可視化し、「粗格子で何が失われているか」を先に目で見る | 残差が構造（渦・界面近傍）を持ち、ランダムノイズでないことを確認 |
| 6 | 潜在予測の単体評価 | 全場→PCA（線形、基準）および小CAEで潜在化し、ESNとNVARを同じ潜在系列で比較。評価は自由ロールアウト持続時間と長期統計 | Vlachas警告の再現確認: 潜在次元とリザバーサイズを振り、発散する条件を地図化する |
| 7 | 補正としての統合 | 潜在予測／ESN出力を物理場の「置き換え」ではなく補正（渦度力・発散源・潜在射影）として小さなゲインで差し込み、数ステップごとに物理状態で再同期 | 学習なし（実験1）より見た目指標（渦の持続、スペクトル）が改善し、長時間安定性が悪化しない |
| 8 | フォールバック整備と解像度階梯 | WebGL2版（フラグメント方式）の移植、端末性能に応じた解像度／Jacobi回数の自動調整 | 低性能環境でもフレームレートが維持され、見た目の破綻がない |

特に実験6→7の順序が重要である。Vlachas et al. 2020の結果は「潜在＋RCは条件次第で発散する」ことを既に示しており、統合してからデバッグすると原因が物理層か学習層か切り分けられない。単体で発散条件を地図化してから統合すること。

---

## Could not verify / 空白地帯

- **氷・水・水蒸気の三相を単一格子ソルバ＋エンタルピー法で扱うブラウザ／ゲーム公開事例**: 見つからなかった（§3.3）。工学CFD側の沸騰モデルをゲーム解像度へ落とす具体的な割り切りパラメータの公表値も見つからなかった。
- **ESNを格子ソルバ内部のサブグリッド・クロージャとして毎ステップ呼ぶ確立した流体論文**: CNN系に比べ事例が乏しく、今回の調査では確立した例を確認できなかった（§4.4）。本プロジェクトの構成はこの意味で研究的に新規である。
- **Babylon.jsのバージョン固有の最新API変更（2026年の特定マイナーバージョン）**: 公式ドキュメントの現行ページで確認した範囲に限る。特定バージョンのchangelog逐条確認は行っていない。
- **Tait方程式の歴史的出典（Tait 1888／Batchelorの教科書経由の普及、MonaghanのWCSPHへの導入年）**: 標準形と運用則は複数資料で一致したが、原典そのもののページは今回取得していないため、歴史記述は最小限に留めた。
- **Stam (1999) 原論文PDFの直接確認**: 内容は複数の再構成・実装資料で一致することを確認したが、原論文PDF自体は今回開いていない。

## Sources

すべて2026-11-24に調査（index経由。Babylon.js Compute Shadersページのみ取得して直接確認）。

1. Stam (1999) Stable Fluids の再構成・解説 — “Stable Fluids”, https://algorithm.tamnd.com/stable-fluids/
2. WGSL流体カーネル集 — “@vshaders/fluid”, https://www.vshaders.sh/docs/fluid
3. WebGL実装解説 — “Solving Navier–Stokes at 60 FPS: Stable Fluids & Helmholtz–Hodge Projection in WebGL”, https://arman1o1.github.io/posts/physics/real-time-navier-stokes-webgl.html
4. PavelDoGreat/WebGL-Fluid-Simulation 解説 — https://deepwiki.com/PavelDoGreat/WebGL-Fluid-Simulation および https://deepwiki.com/PavelDoGreat/WebGL-Fluid-Simulation/3.2-webgl-implementation
5. wgpu移植 — threers “Fluid”, https://raw.githubusercontent.com/eugenehp/threers/main/docs/fluid.md
6. WebGPU 3D流体 — “Volumetric Fluid — Technical Deep Dive” (Taos Engine), https://taosengine.com/book/samples/fluid_test.html
7. WCSPH／Tait方程式 — “Weakly compressible SPH”, https://algorithm.tamnd.com/weakly-compressible-sph/ ; “SPH: Fluids from Particles”, https://www.mysimulator.uk/articles/sph-fluids/ ; Chrono FSI-SPHガイド, https://api.chrono.projectchrono.org/manual_fsi_sph_parameter_selection.html
8. Stefan問題 — RWTH Aachen “Solid-liquid phase change”, https://rwth-mbd.pages.git.nrw/courses/cmm/content/script/chapters/phase_change.html
9. エンタルピー法 — ScienceDirect Topics “Enthalpy Method”, https://www.sciencedirect.com/topics/engineering/enthalpy-method ; Mochnacki et al. (2002), https://amcm.pcz.pl/2002_1/art_22.pdf ; Luddens et al. (2025), https://borowiecki.dev/pdf/2505.13097
10. Voller & Prakash (1987) の典拠確認 — OpenFOAM v1806, https://www.openfoam.com/news/main-news/openfoam-v1806/solver-and-physics
11. Carlson et al. “Melting and Flowing”, https://docslib.org/download/2747190/melting-and-flowing ; Wei et al. “Melting and Flowing of Viscous Volumes”, https://cvc.cs.stonybrook.edu/sites/default/files/vislab/2021-05/Melting.pdf
12. 潜熱の値 — “Heating Curve Simulator”, https://webtoolyard.com/heating-curve-simulator/
13. ゲームの閾値方式 — Sandboxels水循環, https://www.sandboxels.cc/sandboxels-water-cycle/ ; Base Matter (Steam), https://store.steampowered.com/app/5365110/Base_Matter/?snr=1_7_7_230_150_9
14. Um et al. (2020) Solver-in-the-Loop, NeurIPS 2020, https://proceedings.nips.cc/paper_files/paper/2020/file/43e4e6a6f341e00671e123714de019a8-Paper.pdf ; arXiv:2007.00016, https://ar5iv.labs.arxiv.org/abs/2007.00016
15. Kochkov et al. (2021) PNAS 118(21) e2101784118 / arXiv:2102.01010, https://pmc.ncbi.nlm.nih.gov/articles/PMC8166023/ ; https://arxiv.org/abs/2102.01010?context=cs
16. Beck, Flad & Munz (2019) JCP, https://inis.iaea.org/records/5xajk-wvh93 ; Duraisamy (2020) arXiv:2009.10675, https://arxiv.org/pdf/2009.10675 ; JFM differentiable LES, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/learned-turbulence-modelling-with-differentiable-fluid-solvers-physicsbased-loss-functions-and-optimisation-horizons/28D19239CEDB81A3DA58F32E0E8CB3B2
17. Doan, Polifke & Magri (2021) Proc. R. Soc. A / arXiv:2102.07514, https://arxiv.org/html/2102.07514v2 ; arXiv:1912.10994, https://arxiv.org/pdf/1912.10994
18. Racca, Doan & Magri (2023) CAE-ESN, JFM 975, A2, https://resolve.cambridge.org/core/journals/journal-of-fluid-mechanics/article/predicting-turbulent-dynamics-with-the-convolutional-autoencoder-echo-state-network/1E0F75CD94FCB3A1354A09622F8D25CD
19. Vlachas et al. (2020) Neural Networks 126:191–217 / arXiv:1910.05266, https://arxiv.org/abs/1910.05266 ; https://pubmed.ncbi.nlm.nih.gov/32248008/
20. Gauthier et al. (2021) Nature Communications 12:5564 / arXiv:2106.07688, https://arxiv.org/abs/2106.07688 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8455577/
21. AE-LSTM ROM (2020) Physics of Fluids 32:105104, https://arxiv.org/html/2007.00936 ; CNN-AE/LSTM (NeurIPS 2020 ML4PS), https://ml4physicalsciences.github.io/2020/files/NeurIPS_ML4PS_2020_8.pdf
22. Pandey, Schumacher & Sreenivasan (2020) Journal of Turbulence総説, https://1library.net/document/yenj2e1y-a-perspective-on-machine-learning-in-turbulent-flows.html
23. Babylon.js公式 — “Compute Shaders”, https://doc.babylonjs.com/features/featuresDeepDive/materials/shaders/computeShader/ ; “WebGPU Breaking Changes”, https://doc.babylonjs.com/setup/support/webGPU/webGPUBreakingChanges
24. WebGPU対応状況・上限 (2026-11), https://ttslab.dev/limits
25. WGSL実務 — https://skillmd.com/skills/impertio-studio/webgpu-wgsl-compute-shaders ; https://skillmd.com/skills/impertio-studio/webgpu-impl-compute-usecases ; https://www.newsnix.com/articles/webgpu-compute-shaders-tutorial-getting-started/
