# SSD：整合慣性 × ランダム接続 — 跳躍の最小ハイブリッドモデル

## 0. 目的と要約
- **目的**：整合（決定論・省エネ）と、跳躍（確率論・接続拡張）を単一の時間発展で統合する。
- **要約**：
  - 既存の整合回路（オーム則アナロジー）に、未処理圧リザーバEを導入。
  - Eが閾値を超えると**制約付きランダム接続**でネットワークを再配線（跳躍）。
  - 跳躍後は放熱（E減衰）と慣性調整で再び整合へ。硬直化すれば温度T↑で探索を増やす。

---

## 1. 状態と変数
- 構造ネットワーク：\(G=(\mathcal{S},\mathcal{E}, w)\)（ノード=概念、エッジ重み\(w_{ij}\)）
- 整合慣性：\(\kappa_{ij}(t)\)（経路の通りやすさ／記憶）
- 入力（意味圧）：\(p(t)\)（課題ベクトル or スカラー）
- 未処理圧リザーバ：\(E(t)\)（整合不能の蓄積＝“熱”）
- 温度：\(T(t)\)（探索の強さ）

---

## 2. 連続部：整合ダイナミクス
**通りやすさ**：\(G_{ij}(t)=G_{0,ij}+g\,\kappa_{ij}(t)\)

**整合流**（ネットワーク版オーム則）
\[
 j_{ij}(t)=G_{ij}(t)\,a_{ij}(t),\quad a_{ij}(t)=\langle\phi_i,p(t)\rangle-\langle\phi_j,p(t)\rangle
\]
（単純化：\(a_{ij}(t)=p(t)\)可）

**整合仕事（正味）**
\[
 \dot W_{\text{align}}(t)=\sum_{(i,j)}\big(p(t)\,j_{ij}-\rho\, j_{ij}^2\big)
\]

**慣性更新（学習・忘却）**
\[
 \dot\kappa_{ij}=\eta\,\big[p(t)\,j_{ij}-\rho\,j_{ij}^2\big]-\lambda(\kappa_{ij}-\kappa_{\min})
\]

---

## 3. 未処理圧の蓄積（整合不能の計上）
\[
 \dot E=\alpha\,[\,\|p(t)\|-\|J(t)\|\,]_+ - \beta\,E\quad (\,\|J\|:=\sqrt{\sum j_{ij}^2}\,)
\]
- 整合が追いつかない差分を“熱”として貯蔵。

---

## 4. 跳躍トリガ（確率的発火）
発火強度（ポアソン過程）：
\[
 h(t)=h_0\,\exp\!\Big(\frac{E(t)-\Theta(t)}{\gamma}\Big),\qquad
 \Theta(t)=\Theta_0 + a_1\,\bar{\kappa}(t)-a_2\,F(t)
\]
- \(\bar{\kappa}\)：平均慣性（高いほど跳びにくい）
- \(F\)：疲労などの可逆指標（高いほど跳びやすい）
- 発火確率（ステップ幅\(\Delta t\)）：\(1-\exp(-h\,\Delta t)\)

---

## 5. 跳躍：制約付きランダム接続（構造依存＋ノイズ）
候補集合 \(\mathcal{C}\) から、新規接続先 \(k\) を確率分布で選択：
\[
 \pi(k\mid s,t) \propto \exp\!\Big(\frac{\underbrace{\mathrm{sim}(s,k)}_{\text{構造依存}} + \underbrace{\xi_k}_{\text{ノイズ}}}{T(t)}\Big),\quad \xi_k\sim\mathcal{N}(0,\sigma^2)
\]
- \(T(t)=T_0+c_1E(t)-c_2\,H(\pi_{t^-})\)：**探索温度**（熱↑／ポリシーのエントロピー低↓で温度↑）
- 選ばれたエッジの追加・強化：
\[
 w_{sk}\leftarrow w_{sk}+\Delta w,\qquad \kappa_{sk}\leftarrow \kappa_{sk}+\Delta\kappa^{(+)}
\]
- 過飽和の主経路は微緩和（硬直ほぐし）：
\[
 \kappa_{ij}\leftarrow \kappa_{ij}-\epsilon\cdot \mathbb{I}\{j_{ij}\text{が上位}q\%\}
\]
- 放熱：\(E\leftarrow c_0E\;(0\le c_0<1)\)

---

## 6. 硬直—創発バランサ
- **硬直度**：\(H(\pi)\)（接続ポリシーのエントロピー）。下がりすぎたら \(T\) を底上げ。
- **探索ノイズ**（\(\varepsilon\)-greedy）：
\[
 \varepsilon(t)=\varepsilon_0 + d_1E - d_2\bar{\kappa}
\]
確率 \(\varepsilon\) で完全ランダム接続を混入。

---

## 7. 観測指標（ダッシュボード）
- 整合効率：\(\eta_{\text{align}}=\|J\|/\|p\|\)
- 放熱率：\(-\dot E\)
- 政策エントロピー：\(H(\pi)\)
- 新規接続率：\(|\delta\mathcal{E}|/\Delta t\)
- 創造歩留まり：新接続のうち後続で \(\kappa\) が持続増加する割合

---

## 8. 擬似コード（逐次更新）
```text
for t in 0..T:
  # 整合ステップ
  compute j_ij = (G0_ij + g*kappa_ij) * a_ij(p)
  W_align = sum(p*j_ij - rho*j_ij^2)
  kappa_ij += eta*(p*j_ij - rho*j_ij^2) - lambda*(kappa_ij - kappa_min)
  E += alpha*max(||p||-||J||,0) - beta*E

  # 跳躍判定
  Theta = Theta0 + a1*mean(kappa) - a2*F
  h = h0*exp((E-Theta)/gamma)
  if rand() < 1 - exp(-h*dt):
      # 制約付きランダム接続
      T = T0 + c1*E - c2*entropy(pi_prev)
      sample k ~ pi(k|s,t) ∝ exp((sim(s,k) + noise)/T)
      add edge (s,k): w_sk += Δw ; kappa_sk += Δkappa_plus
      # 硬直ほぐし
      for top-q% edges by j_ij: kappa_ij -= ε
      # 放熱
      E *= c0

  # ε-greedy（完全ランダム接続）
  if rand() < (ε0 + d1*E - d2*mean(kappa)):
      randomly add/rewire a weak edge
```

---

## 9. 実験プロトコル（最小セット）
1. **単一課題**：固定ベクトル \(p\) を入れて、E・T・H(π) の時間発展を観察。
2. **課題系列**：局所相関が低い \(p_t\) を列投入→跳躍頻度と整合効率のトレードオフ計測。
3. **硬直化テスト**：\(\lambda\downarrow\)（忘却遅延）で硬直を誘発→T制御の効果を検証。
4. **創造歩留まり**：新規接続の持続率（\(\kappa\)増加）と分布（近傍vs遠方）を計測。

---

## 10. 応用・解釈
- **閃き**：Eが閾値を超え、遠方ノードと接続→新経路が学習で固定。
- **暴走**：T過大やρ低下でノイズ優位→整合効率低下・熱暴走。
- **冷静性**：β↑, c2↑（エントロピー低下検知で温度制御）で“必要な跳躍のみ”を残す。

---

## 11. 次の拡張
- 多主体の自己励起（Hawkes過程）で社会的カスケード。
- ランドウ自由エネルギーによる相転移表示（\(F_S(\phi; p)\) の極小入れ替え）。
- 物語生成への応用：跳躍＝プロット分岐、整合＝伏線回収の力学。

