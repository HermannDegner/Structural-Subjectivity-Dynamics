# SSD：整合×跳躍 関数API（v1）

> **目的**：整合（決定論パート）と跳躍（確率パート）を**同一の更新器**で扱える汎用API。Nano‑SSD/最小実装に直結する形で定義。

---

## 0. 状態と入力の定義

### 0.1. システム状態

$$S = \{G, \kappa, E, F, T, \pi\}$$

**構成要素：**
- $G = (\mathcal{S}, \mathcal{E}, w)$：グラフ構造
  - $\mathcal{S}$：ノード集合（戦術/概念）
  - $\mathcal{E}$：エッジ集合
  - $w_{ij}$：エッジ重み
- $\kappa_{ij}$：整合慣性（経路の通りやすさ/記憶）
- $E$：未処理圧（"熱"）
- $F$：疲労など可逆指標
- $T$：探索温度
- $\pi$：接続方策

### 0.2. 入力・時間パラメータ

- **入力意味圧**：$p \in \mathbb{R}$ または $p \in \mathbb{R}^n$
- **時間刻み**：$\Delta t > 0$（離散時間、連続時間近似でも可）

---

## 1. 整合（決定論）パートの関数群

### 1.1. AlignFlow — 整合流の計算

**関数型：**
```
AlignFlow(p, G₀, κ; g, ε) → j
```

**数学的定義：**
$$j = (G_0 + g \cdot \kappa) \cdot p + \varepsilon$$

**パラメータ：**
- $G_0 \geq 0$：基礎通りやすさ
- $g > 0$：慣性利得係数
- $\varepsilon \sim \mathcal{N}(0, \sigma_{\varepsilon}^2)$：微小ノイズ

**戻り値：**
- $j$：整合流ベクトル/スカラー

### 1.2. UpdateKappa — 学習・忘却更新

**関数型：**
```
UpdateKappa(κ, p, j; η, ρ, λ, κ_min) → κ'
```

**数学的定義：**
$$\frac{d\kappa_{ij}}{dt} = \eta (p \cdot j_{ij} - \rho \cdot j_{ij}^2) - \lambda (\kappa_{ij} - \kappa_{\min})$$

**離散化：**
$$\kappa' = \kappa + \frac{d\kappa}{dt} \cdot \Delta t$$

**パラメータ：**
- $\eta > 0$：学習率
- $\rho \geq 0$：過駆動抑制係数
- $\lambda > 0$：未使用減衰率
- $\kappa_{\min} \geq 0$：慣性下限値

### 1.3. UpdateHeat — 未処理圧（熱）の更新

**関数型：**
```
UpdateHeat(E, p, j; α, β_E) → E'
```

**数学的定義：**
$$\frac{dE}{dt} = \alpha \left[ \|p\| - \|J\| \right]_+ - \beta_E \cdot E$$

**記号説明：**
- $\|J\| = \sqrt{\sum_{ij} j_{ij}^2}$：総整合流のノルム
- $[x]_+ = \max(x, 0)$：正の部分のみ

**離散化：**
$$E' = E + \frac{dE}{dt} \cdot \Delta t$$

**パラメータ：**
- $\alpha > 0$：蓄積係数
- $\beta_E > 0$：自然減衰係数

---

## 2. 跳躍（確率）パートの関数群

### 2.1. Threshold — 構造依存の閾値計算

**関数型：**
```
Threshold(Θ₀, κ̄, F; a₁, a₂) → Θ
```

**数学的定義：**
$$\Theta = \Theta_0 + a_1 \bar{\kappa} - a_2 F$$

**記号説明：**
- $\bar{\kappa} = \frac{1}{|\mathcal{E}|} \sum_{ij} \kappa_{ij}$：平均慣性

**パラメータ：**
- $\Theta_0 > 0$：基本閾値
- $a_1 \geq 0$：慣性による閾値上昇係数
- $a_2 \geq 0$：疲労による閾値低下係数

### 2.2. JumpRate — 跳躍率（擬ポアソン強度）

**関数型：**
```
JumpRate(E, Θ; h₀, γ) → h
```

**数学的定義：**
$$h = h_0 \exp\left(\frac{E - \Theta}{\gamma}\right)$$

**パラメータ：**
- $h_0 > 0$：ベース跳躍強度
- $\gamma > 0$：発火曲線の鋭さ

### 2.3. Temperature — 探索温度（硬直検知付き）

**関数型：**
```
Temperature(T₀, E, H(π); c₁, c₂) → T
```

**数学的定義：**
$$T = T_0 + c_1 E - c_2 H(\pi)$$

**エントロピー：**
$$H(\pi) = -\sum_k \pi_k \log \pi_k$$

**パラメータ：**
- $T_0 > 0$：基本温度
- $c_1 > 0$：熱による温度上昇係数
- $c_2 > 0$：硬直（低エントロピー）による温度上昇係数

### 2.4. SampleJump — 跳躍判定

**関数型：**
```
SampleJump(h, Δt) → bool
```

**確率計算：**
$$P(\text{jump}) = 1 - e^{-h \cdot \Delta t}$$

**実装：**
```python
return random.uniform(0, 1) < (1 - exp(-h * dt))
```

---

## 3. 接続再配線（跳躍の実体）関数群

### 3.1. Policy — 制約付きランダム選択

**関数型：**
```
Policy(sim(·), T, σ; candidates) → π(k)
```

**数学的定義：**
$$\pi(k) \propto \exp\left(\frac{\text{sim}(s,k) + \xi_k}{T}\right)$$

**ノイズ項：**
$$\xi_k \sim \mathcal{N}(0, \sigma^2)$$

**正規化：**
$$\pi(k) = \frac{\exp\left(\frac{\text{sim}(s,k) + \xi_k}{T}\right)}{\sum_{k' \in \mathcal{C}} \exp\left(\frac{\text{sim}(s,k') + \xi_{k'}}{T}\right)}$$

**パラメータ：**
- $\mathcal{C}$：候補集合（タグ/距離/役割でフィルタ済み）
- $\sigma > 0$：ノイズ強度
- $T > 0$：探索温度

### 3.2. Rewire — 新規接続/強化 + 放熱

**関数型：**
```
Rewire(G, κ, E, k; Δw, Δκ, c₀) → (G', κ', E')
```

**更新操作：**
$$w_{sk} \leftarrow w_{sk} + \Delta w$$
$$\kappa_{sk} \leftarrow \kappa_{sk} + \Delta \kappa$$
$$E \leftarrow c_0 \cdot E$$

**パラメータ：**
- $\Delta w > 0$：重み増加量
- $\Delta \kappa > 0$：慣性増加量  
- $0 \leq c_0 < 1$：放熱係数

### 3.3. RelaxTop — 硬直ほぐし

**関数型：**
```
RelaxTop(κ, j; q, ε_relax) → κ'
```

**処理手順：**
1. 流量 $j_{ij}$ の上位 $q\%$ を特定
2. 該当経路の慣性を微減：$\kappa_{ij} \leftarrow \kappa_{ij} - \varepsilon_{\text{relax}}$

**パラメータ：**
- $q \in [0, 100]$：対象パーセンタイル
- $\varepsilon_{\text{relax}} > 0$：緩和量

### 3.4. EpsRandom — 低頻度の完全ランダム

**関数型：**
```
EpsRandom(ε₀, E, κ̄; d₁, d₂) → prob
```

**確率計算：**
$$\varepsilon = \varepsilon_0 + d_1 E - d_2 \bar{\kappa}$$

**パラメータ：**
- $\varepsilon_0 > 0$：ベース探索率
- $d_1 > 0$：熱による探索増加
- $d_2 > 0$：慣性による探索減少

---

## 4. 行動選択（オプショナル）

### 4.1. SelectAction — 慣性考慮行動選択

**関数型：**
```
SelectAction(Q, κ; b) → a*
```

**評価関数近似：**
$$Q(a) \approx U_{\text{immediate}}(a) + b \cdot \kappa_{\text{path}}(a)$$

**パラメータ：**
- $b > 0$：慣性経路の重み係数

---

## 5. ワンステップ更新オーケストラ

### 5.1. メイン更新関数

**関数型：**
```
Step(S, p, dt, params) → (S', a*)
```

### 5.2. 更新アルゴリズム

```python
def Step(S, p, dt, params):
    G, κ, E, F, T, π = S
    
    # === 整合フェーズ（決定論的） ===
    j = AlignFlow(p, params.G0, κ, params.g, params.ε)
    κ = UpdateKappa(κ, p, j, params.η, params.ρ, params.λ, params.κ_min)
    E = UpdateHeat(E, p, j, params.α, params.β_E)
    
    # === 跳躍判定（確率的） ===
    Θ = Threshold(params.Θ0, mean(κ), F, params.a1, params.a2)
    h = JumpRate(E, Θ, params.h0, params.γ)
    T = Temperature(params.T0, E, entropy(π), params.c1, params.c2)
    
    # === 跳躍実行 ===
    if SampleJump(h, dt):
        π = Policy(sim_function, T, params.σ, candidates)
        k = sample(π)
        G, κ, E = Rewire(G, κ, E, k, params.Δw, params.Δκ, params.c0)
        κ = RelaxTop(κ, j, params.q, params.ε_relax)
    
    # === ε-greedy完全ランダム ===
    elif random() < EpsRandom(params.ε0, E, mean(κ), params.d1, params.d2):
        add_weak_random_edge(G)
    
    # === 行動選択（オプション） ===
    a_star = SelectAction(Q_function, κ, params.b)
    
    S_new = (G, κ, E, F, T, π)
    return S_new, a_star
```

---

## 6. パラメータ仕様（小規模実装向け）

### 6.1. パラメータ一覧表

| カテゴリ | 記号 | 既定値 | 範囲 | 物理的意味 |
|----------|------|--------|------|-----------|
| **整合** | $g$ | 0.7 | 0.2–1.5 | 慣性の影響係数 |
| | $\eta$ | 0.3 | 0.1–0.5 | 学習率 |
| | $\lambda$ | 0.02 | 0.005–0.05 | 忘却率 |
| | $\rho$ | 0.1 | 0.0–1.0 | 過駆動抑制 |
| **熱** | $\alpha$ | 0.6 | 0.2–1.0 | 蓄積係数 |
| | $\beta_E$ | 0.15 | 0.05–0.3 | 自然減衰 |
| **閾値** | $\Theta_0$ | 1.0 | 0.5–2.0 | 基本閾値 |
| | $a_1$ | 0.5 | 0.0–1.0 | 慣性補正 |
| | $a_2$ | 0.4 | 0.0–1.0 | 疲労補正 |
| **跳躍** | $h_0$ | 0.2 | 0.05–0.5 | ベース強度 |
| | $\gamma$ | 0.8 | 0.4–1.5 | 発火鋭さ |
| **温度** | $T_0$ | 0.3 | 0.1–0.7 | 基本温度 |
| | $c_1$ | 0.5 | 0.1–1.0 | 熱→温度係数 |
| | $c_2$ | 0.6 | 0.2–1.2 | 硬直検知係数 |
| | $\sigma$ | 0.2 | 0.05–0.5 | ノイズ強度 |

### 6.2. 構造体定義例

```python
@dataclass
class SSDParams:
    # 整合パラメータ
    g: float = 0.7
    eta: float = 0.3
    lambda_: float = 0.02
    rho: float = 0.1
    
    # 熱パラメータ  
    alpha: float = 0.6
    beta_E: float = 0.15
    
    # 閾値パラメータ
    Theta0: float = 1.0
    a1: float = 0.5
    a2: float = 0.4
    
    # 跳躍パラメータ
    h0: float = 0.2
    gamma: float = 0.8
    
    # 温度パラメータ
    T0: float = 0.3
    c1: float = 0.5
    c2: float = 0.6
    sigma: float = 0.2
```

---

## 7. 「冷静性」ノブ（過跳躍の抑制）

### 7.1. 冷静性調整の数学的表現

過度な跳躍を抑制するための調整：

#### 放熱強化
$$c_0 \leftarrow c_0 \cdot (1 - \text{calmness\_factor})$$

#### 自然減衰強化  
$$\beta_E \leftarrow \beta_E \cdot (1 + \text{calmness\_factor})$$

#### 硬直検知強化
$$c_2 \leftarrow c_2 \cdot (1 + \text{rationality\_factor})$$

#### 閾値底上げ
$$\Theta_0 \leftarrow \Theta_0 + \text{stability\_bonus}$$
$$a_1 \leftarrow a_1 \cdot (1 + \text{experience\_weight})$$

#### 発火曲線緩和
$$\gamma \leftarrow \gamma \cdot (1 + \text{smoothness\_factor})$$

### 7.2. 冷静性関数

**関数型：**
```
ApplyCalmness(params, calmness_level) → params'
```

```python
def ApplyCalmness(params, level):
    factor = max(0, min(1, level))
    params.c0 *= (1 - 0.3 * factor)
    params.beta_E *= (1 + 0.5 * factor)  
    params.c2 *= (1 + 0.4 * factor)
    params.Theta0 += 0.2 * factor
    params.gamma *= (1 + 0.3 * factor)
    return params
```

---

## 8. 実装ノート

### 8.1. 離散時間 vs 連続時間

**離散時間実装：**
- フレーム更新: `Step(S, p, 1/60, params)` 
- 擬ポアソン発火で連続時間を近似

**連続時間実装：**
- 微分方程式ソルバー使用
- より正確だが計算コスト高

### 8.2. 候補集合のフィルタリング

$$\mathcal{C} = \{k \in \mathcal{S} : \text{WorldCompatible}(k) \land \text{Distance}(current, k) \leq d_{\max}\}$$

**実装例：**
```python
def get_candidates(G, current, world_tags, max_distance=3):
    candidates = []
    for node in G.nodes():
        if (node.tags & world_tags) and distance(current, node) <= max_distance:
            candidates.append(node)
    return candidates
```

### 8.3. テレメトリ関数

**主要KPI関数：**
```python
def compute_metrics(S, p, j):
    return {
        'alignment_efficiency': norm(j) / norm(p),
        'heat_level': S.E,
        'temperature': S.T,
        'policy_entropy': entropy(S.π),
        'jump_frequency': compute_jump_rate(),
        'creativity_yield': successful_connections / total_connections
    }
```

---

## 9. 統合観：整合—跳躍のハイブリッド力学

### 9.1. 統合の数学的表現

システム全体は以下の状態空間で記述されます：

$$\dot{S} = F_{\text{align}}(S, p) + F_{\text{jump}}(S, p, \xi_t)$$

**成分：**
- $F_{\text{align}}$：決定論的整合ダイナミクス
- $F_{\text{jump}}$：確率的跳躍プロセス  
- $\xi_t$：時間依存ランダムプロセス

### 9.2. エネルギー解釈

**未処理圧 $E$** が整合と跳躍の切り替えを制御：

$\begin{cases}
E < \Theta & \Rightarrow \text{整合モード（決定論的）} \\
E \geq \Theta & \Rightarrow \text{跳躍モード（確率的）}
\end{cases}$

### 9.3. 退屈の力学的解釈

「退屈」は低レベル不整合として $E$ に寄与：

$$E_{\text{boredom}} = \alpha_{\text{bore}} \cdot \max(0, t_{\text{stable}} - t_{\text{threshold}})$$

これにより硬直状態から自動的に脱却する機構が組み込まれます。

---

## 結論

本APIモデルは、構造主観力学の「整合」と「跳躍」を**実装可能な関数群**として定義しました。

### 🎯 達成事項

1. **完全な関数仕様**：すべての処理を型付き関数として定義
2. **ワンステップ統合**：`Step()`による統一的な更新手続き
3. **パラメータ標準化**：実装間での互換性を保証
4. **デバッガビリティ**：各段階でのテレメトリ取得が可能

### 🚀 期待される効果

- **実装の標準化**：異なる環境での一貫した動作
- **デバッグの容易性**：関数単位での動作検証
- **パフォーマンス最適化**：ホットスポットの特定と改善
- **拡張性**：新機能の追加が関数追加で完結

このAPIを基盤として、ゲームAIから教育システムまで、多様な領域でSSDの恩恵を享受できることを期待します。