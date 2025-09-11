# SSD：整合×跳躍 関数API（v1）

> 目的：整合（決定論パート）と跳躍（確率パート）を**同一の更新器**で扱える汎用API。Nano‑SSD/最小実装に直結する形で定義。

---

## 0. 状態と入力
- **状態** $S=\{G,\kappa,E,F,T,\pi\}$
  - $G=(\mathcal{S},\mathcal{E},w)$：ノード＝戦術/概念、エッジ重み $w$
  - $\kappa_{ij}$：整合慣性（経路の通りやすさ/記憶）
  - $E$：未処理圧（“熱”）／ $F$：疲労など可逆指標／ $T$：探索温度／ $\pi$：接続方策
- **入力** $p$：意味圧（スカラー/ベクトル）
- **刻み** $\Delta t$：離散時間（※連続時間の近似でも可）

---

## 1. 整合（決定論）パートの関数
### 1.1 AlignFlow — 整合流の計算
**型**：`AlignFlow(p, G0, κ; g, ε) -> j`

**式**：\[ j = (G_0 + g\,\kappa)\,p \; + \; \varepsilon \]
- $G_0$：基礎通りやすさ，$g$：慣性利得，$\varepsilon$：微小ノイズ

### 1.2 UpdateKappa — 学習・忘却更新
**型**：`UpdateKappa(κ, p, j; η, ρ, λ, κ_min) -> κ'`

**式**：\[ \dot\kappa_{ij}=\eta\,(p\,j_{ij}-\rho\,j_{ij}^2)-\lambda\,(\kappa_{ij}-\kappa_{\min}),\quad \kappa' = \kappa + \dot\kappa\,\Delta t \]
- $\eta$：学習率，$\rho$：過駆動抑制，$\lambda$：未使用減衰

### 1.3 UpdateHeat — 未処理圧（熱）の更新
**型**：`UpdateHeat(E, p, j; α, β_E) -> E'`

**式**：\[ \dot E = \alpha\,[\,\|p\|-\|J\|\,]_+ - \beta_E\,E,\quad E' = E + \dot E\,\Delta t \]
- $\|J\|=\sqrt{\sum j_{ij}^2}$，$[x]_+=\max(x,0)$

---

## 2. 跳躍（確率）パートの関数
### 2.1 Threshold — 構造依存の閾値
**型**：`Threshold(Θ0, κ̄, F; a1, a2) -> Θ`

**式**：\[ \Theta = \Theta_0 + a_1\,\bar\kappa - a_2\,F \]

### 2.2 JumpRate — 跳躍率（擬ポアソン強度）
**型**：`JumpRate(E, Θ; h0, γ) -> h`

**式**：\[ h = h_0\exp\!\Big(\frac{E-\Theta}{\gamma}\Big) \]

### 2.3 Temperature — 探索温度（硬直検知つき）
**型**：`Temperature(T0, E, H(π); c1, c2) -> T`

**式**：\[ T = T_0 + c_1\,E - c_2\,H(\pi) \]
- $H(\pi)$：方策エントロピー（硬直=低エントロピー）

### 2.4 SampleJump — 跳躍判定
**型**：`SampleJump(h, Δt) -> bool`

**確率**：\[ \mathbb{P}(\text{jump}) = 1 - e^{-h\,\Delta t} \]

---

## 3. 接続再配線（跳躍の実体）
### 3.1 Policy — 制約付きランダム選択
**型**：`Policy(sim(·), T, σ) -> π(k)`

**式**：\[ \pi(k) \propto \exp\!\Big(\frac{\mathrm{sim}(s,k)+\xi_k}{T}\Big),\quad \xi_k\sim\mathcal{N}(0,\sigma^2) \]
- 候補集合はタグ/距離/役割でフィルタした $\mathcal{C}$

### 3.2 Rewire — 新規接続/強化 + 放熱
**型**：`Rewire(G, κ, k; Δw, Δκ, c0) -> (G', κ', E')`

**操作**：$w_{sk}\!\leftarrow w_{sk}+\Delta w,\; \kappa_{sk}\!\leftarrow\kappa_{sk}+\Delta\kappa,\; E\!\leftarrow c_0E$

### 3.3 RelaxTop — 硬直ほぐし
**型**：`RelaxTop(κ, j; q, ε_relax) -> κ'`
- 流量上位 $q\%$ 経路の $\kappa$ を $\varepsilon_{\text{relax}}$ だけ微減

### 3.4 EpsRandom — 低頻度の完全ランダム
**型**：`EpsRandom(ε0, E, κ̄; d1, d2) -> prob`

**式**：$\varepsilon = \varepsilon_0 + d_1E - d_2\bar\kappa$

---

## 4. 行動選択（任意）
**型**：`SelectAction(Q, κ; b) -> a*`

**近似**：$Q(a) \approx U_{\text{immediate}}(a) + b\cdot \kappa_{\text{path}}(a)$

---

## 5. ワンステップ更新オーケストラ
```pseudo
function Step(S, p, dt, params):
  j = AlignFlow(p, G0, κ; g, ε)
  κ = UpdateKappa(κ, p, j; η, ρ, λ, κ_min)
  E = UpdateHeat(E, p, j; α, β_E)

  Θ = Threshold(Θ0, mean(κ), F; a1, a2)
  h = JumpRate(E, Θ; h0, γ)
  T = Temperature(T0, E, entropy(π); c1, c2)

  if SampleJump(h, dt):
      π = Policy(sim(·), T, σ)
      k ~ π
      (G, κ, E) = Rewire(G, κ, k; Δw, Δκ, c0)
      κ = RelaxTop(κ, j; q, ε_relax)
  else if rand() < (ε0 + d1*E - d2*mean(κ)):
      add_weak_random_edge()

  a* = SelectAction(Q, κ; b)  # 任意
  return (S', a*)
```

---

## 6. パラメータ目安（小規模実装）
- 整合：$g\in[0.2,1.5],\; \eta\in[0.1,0.5],\; \lambda\in[0.005,0.05],\; \rho\in[0.1,1.0]$
- 熱：$\alpha\in[0.2,1.0],\; \beta_E\in[0.05,0.3]$
- 閾値：$\Theta_0\in[0.5,2.0],\; a_1\in[0,1.0],\; a_2\in[0,1.0]$
- 跳躍：$h_0\in[0.05,0.5],\; \gamma\in[0.4,1.5]$
- 温度：$T_0\in[0.1,0.7],\; c_1\in[0.1,1.0],\; c_2\in[0.2,1.2],\; \sigma\in[0.05,0.5]$

---

## 7. 「冷静性」ノブ（過跳躍の抑制）
- **放熱強化**：$c_0\downarrow$／**自然減衰強化**：$\beta_E\uparrow$
- **硬直検知強化**：$c_2\uparrow$（低エントロピー時に温度が上がりにくい）
- **閾値底上げ**：$\Theta_0\uparrow,\; a_1\uparrow$（高慣性で跳びにくく）
- **発火曲線緩和**：$\gamma\uparrow$／**ベース強度抑制**：$h_0\downarrow$

---

## 8. 実装ノート
- **離散/連続**：離散ステップでも、擬ポアソン発火（2.4）を使えば連続時間の近似として動作。
- **候補集合**：世界観・役割タグで $\mathcal{C}$ を絞ると暴走を抑えやすい。
- **テレメトリ**：$\|J\|/\|p\|$, $E$, $T$, $H(\pi)$, 跳躍頻度, 新接続の生存率（創造歩留まり）。

---

### 付記：整合—跳躍 統合観
- 整合は省エネの決定論、跳躍は構造再配線の確率論。未処理圧 $E$ が両者の“切替”を担う。
- 「退屈」は低レベル不整合として $E$ に加算し、軽い跳躍で硬直を回避（探索温度 $T$ を自動制御）。

