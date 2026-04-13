# Nano-SSD：能動NPC向けミニマム実装仕様（v1）

> **目的**：SSDの核心（整合／跳躍）を**軽量に蒸留**し、能動NPCがリアルタイムで回せる最小コアを定義する。

---

## 1. コア概念（NPCが保持する最小状態）

### 1.1. 基本変数

| 記号 | 名前 | 型 | 意味 |
|------|------|----|----|
| $p$ | 意味圧 | Vector/Scalar | 外界・相手・目標からの要求強度 |
| $\kappa$ | 整合慣性 | Matrix | 過去の成功経路の通りやすさ |
| $\theta$ | 臨界 | Scalar | 跳躍トリガ閾値：$\|p\| \geq \theta$ で跳躍モードへ |
| $\beta$ | 探索温度 | Scalar | ランダム接続率。熱や硬直度で自動調整 |
| $j$ | 整合流 | Vector | 反応強度 |
| $E$ | 未処理圧 | Scalar | 整合が追いつかない差分の蓄積（"熱"） |
| $F$ | 疲労 | Scalar | 可逆的な処理能力低下 |

### 1.2. ネットワーク構造

$$G = (S, E_{\text{edges}}, w)$$

**推奨規模：**
- **ノード数**：10〜30
- **エッジ数**：30〜100  
- **全処理計算量**：$O(|E_{\text{edges}}|)$

---

## 2. 1ティック更新（擬似コード）

```python
def nano_ssd_update(world, goals, opponent):
    # === 入力処理（観測→意味圧） ===
    p = sense(world, goals, opponent)
    
    # === 整合ステップ（決定論的処理） ===
    j = (G0 + g * kappa) * p + noise(epsilon)
    reward, fatigue_delta = evaluate(j, world)
    
    # 学習・忘却の更新
    kappa += eta * reward - lambda_forget * unused_decay(kappa)
    F = clamp(F + fatigue_delta - rest(world), 0, F_max)
    
    # === 熱の更新（整合不能の計上） ===
    E += alpha * max(abs(p) - abs(j), 0) - beta_E * E
    
    # === 跳躍判定（確率的発火） ===
    Theta = Theta0 + a1 * mean(kappa) - a2 * F
    h = h0 * exp((E - Theta) / gamma)
    
    if random() < (1 - exp(-h * dt)):
        # === 制約付きランダム接続 ===
        C = candidate_nodes(G, context)
        T = T0 + c1 * E - c2 * entropy(policy_prev)
        
        # ソフトマックス選択
        probs = softmax([(sim(current, k) + N(0, sigma**2)) / T 
                        for k in C])
        k = sample(C, probs)
        
        # エッジの追加・強化
        add_or_boost_edge(G, current, k, Delta_w)
        kappa[current, k] += Delta_kappa_plus
        
        # 硬直ほぐし（上位q%の過飽和経路を微減）
        for e in top_q_percent_edges_by_flow(G):
            kappa[e] -= epsilon_relax
            
        # 放熱
        E *= c0
    
    # === 行動選択（軽量） ===
    action = argmax_a(Q(a)) where Q(a) = immediate_utility(a) + b * kappa_path(a)
    execute(action)
```

---

## 3. 簡易方程式と意味

### 3.1. 整合流

$$j = (G_0 + g \cdot \kappa) \cdot p + \varepsilon$$

**意味：** 既存経路での反応。$\kappa \uparrow$ で省エネ化・安定化

**パラメータ：**
- $G_0$：基本通りやすさ
- $g$：慣性の影響係数
- $\varepsilon$：システムノイズ

### 3.2. 熱（未処理圧）の蓄積

$$\frac{dE}{dt} = \alpha \cdot [\|p\| - \|j\|]_+ - \beta_E \cdot E$$

**意味：** 追いつかない分が溜まる。休息や成功で減る

**パラメータ：**
- $\alpha > 0$：蓄積係数
- $\beta_E > 0$：自然減衰係数

### 3.3. 跳躍率（ポアソン過程）

$$h = h_0 \cdot \exp\left(\frac{E - \Theta}{\gamma}\right)$$

**意味：** 閾値超えで非連続接続（確率ジャンプ）

**動的閾値：**
$$\Theta = \Theta_0 + a_1 \cdot \bar{\kappa} - a_2 \cdot F$$

**パラメータ：**
- $h_0$：基本跳躍強度
- $\gamma$：発火曲線の鋭さ
- $a_1 > 0$：慣性が高いほど跳びにくい
- $a_2 > 0$：疲労で跳びやすい

### 3.4. 探索温度（自動制御）

$$T = T_0 + c_1 \cdot E - c_2 \cdot H(\pi)$$

**意味：** 熱・硬直度で探索温度を自動制御

**パラメータ：**
- $T_0$：基本温度
- $c_1$：熱による温度上昇
- $c_2$：硬直（低エントロピー）補正
- $H(\pi) = -\sum_k \pi_k \log \pi_k$：政策エントロピー

---

## 4. 候補選択：制約付きランダム

### 4.1. 候補集合の構築

$$\mathcal{C} = \{k \in S : \text{distance}(current, k) \leq d_{\max} \land \text{compatible}(k, context)\}$$

**制約条件：**
- **距離制約**：グラフ上の距離・タグ一致・役割適合
- **コンテキスト制約**：現在の状況に適合するノード

### 4.2. 確率分布

$$\pi(k) \propto \exp\left(\frac{\text{sim}(current, k) + \xi_k}{T}\right)$$

**成分：**
- $\text{sim}(current, k)$：構造依存の近さ
  - コサイン類似度
  - タグ一致度
  - 地理的近接度
- $\xi_k \sim \mathcal{N}(0, \sigma^2)$：ガウスノイズ

### 4.3. 探索混合（$\varepsilon$-greedy）

$$\varepsilon_{\text{explore}} = \varepsilon_0 + d_1 E - d_2 \bar{\kappa}$$

確率 $\varepsilon_{\text{explore}}$ で完全ランダム接続を混入。

---

## 5. ふるまい接続（BT/GOAP統合）

### 5.1. Behavior Tree統合

```
Selector(
    AlignNode  [when E < Θ],
    LeapNode   [when E ≥ Θ]
)
```

- **AlignNode**：既存戦術の評価・実行・$\kappa$ 更新
- **LeapNode**：1本だけ新戦術/連想を追加 → 直後は低コスト試行
- **$\theta$ ゲート**：閾値による自動切替

### 5.2. GOAP統合

Plannerの評価関数を拡張：

$$\text{Cost}(plan) = \text{BaseCost}(plan) - \alpha_{\kappa} \sum_{edge \in plan} \kappa_{edge} + \alpha_F F$$

**追加要素：**
- $\kappa_{\text{path}}$：高慣性パスを優先
- $F$：疲労ペナルティ
- $E \uparrow$ で探索幅を拡張

### 5.3. Blackboard共有変数

```python
shared_state = {
    'p': current_meaning_pressure,
    'kappa_mean': mean(kappa_matrix),
    'Theta': current_threshold,
    'E': heat_level,
    'F': fatigue_level,
    'T': exploration_temperature,
    'last_jump_timestamp': last_jump_ts
}
```

---

## 6. チューニング・レシピ

### 6.1. 典型的な問題と対策

| **問題症状** | **対策パラメータ** |
|-------------|-------------------|
| **硬直して単調** | $T_0 \uparrow$, $\sigma \uparrow$, $\varepsilon_{\text{relax}} \uparrow$, $c_1 \uparrow$, $\varepsilon_{\text{greedy}} \uparrow$ |
| **暴走して雑** | $T_0 \downarrow$, $\sigma \downarrow$, $\rho \uparrow$ (失敗ペナルティ), $c_2 \uparrow$, $h_0 \downarrow$ |
| **過跳躍** | $\gamma \uparrow$ (滑らか化), $\Theta_0 \uparrow$, $a_1 \uparrow$, クールダウン追加 |
| **考えすぎで動かない** | $g \uparrow$, $G_0 \uparrow$, $\lambda \uparrow$ (忘却促進), $\beta_E \uparrow$ |

### 6.2. デバッグ指標

```python
# リアルタイム監視
debug_metrics = {
    'alignment_efficiency': norm(j) / norm(p),
    'heat_dissipation_rate': -dE_dt,
    'policy_entropy': H(policy),
    'jump_frequency': jumps_per_minute,
    'creativity_yield': successful_new_connections / total_new_connections
}
```

---

## 7. パラメータの目安（ゲーム向け）

### 7.1. 基本パラメータ表

| 記号 | 既定値 | 範囲 | 意味 |
|------|--------|------|------|
| $G_0$ | 0.5 | 0.1–1.0 | 基本通りやすさ |
| $g$ | 0.7 | 0.2–1.5 | $\kappa$ の影響係数 |
| $\eta$ | 0.3 | 0.1–0.5 | 学習率（成功時 $\Delta\kappa$） |
| $\lambda$ | 0.02 | 0.005–0.05 | 忘却（未使用減衰） |
| $\alpha$ | 0.6 | 0.2–1.0 | 熱の蓄積係数 |
| $\beta_E$ | 0.15 | 0.05–0.3 | 熱の自然減衰 |
| $\Theta_0$ | 1.0 | 0.5–2.0 | 基本閾値 |
| $a_1$ | 0.5 | 0–1.0 | $\bar{\kappa}$ の閾値補正 |
| $a_2$ | 0.4 | 0–1.0 | 疲労の閾値補正 |
| $h_0$ | 0.2 | 0.05–0.5 | 跳躍のベース強度 |
| $\gamma$ | 0.8 | 0.4–1.5 | 発火曲線の鋭さ |
| $T_0$ | 0.3 | 0.1–0.7 | 探索温度の下限 |
| $c_1$ | 0.5 | 0.1–1.0 | 熱→温度の利得 |
| $c_2$ | 0.6 | 0.2–1.2 | 硬直補正 |
| $\sigma$ | 0.2 | 0.05–0.5 | ノイズ幅 |

> **注意：** 数値は目安。ゲーム速度やフレームレートに合わせてスケーリングが必要。

---

## 8. 応用例：巡回ガード（5戦術／20エッジ）

### 8.1. ネットワーク構成

**ノード：** $\{\text{巡回}, \text{追跡}, \text{呼集}, \text{威嚇}, \text{待機}\}$

**基本経路：**
$$\text{巡回} \xrightarrow{\kappa=0.8} \text{威嚇} \xrightarrow{\kappa=0.6} \text{追跡}$$

### 8.2. 動作シナリオ

1. **平常時**：$\text{巡回} \rightarrow \text{威嚇} \rightarrow \text{追跡}$ の $\kappa$ が太い
2. **侵入者発生**：$|p|$（侵入者圧）が高まり $E \uparrow$
3. **閾値超え**：跳躍で $\text{呼集}$ への新規接続が出現
4. **強化学習**：成功すると局所的に強化
5. **退屈対応**：長時間イベント無しで $\text{待機} \rightarrow \text{巡回（遠回り)}$ の枝形成

### 8.3. パラメータ調整例

```python
# 警備の基本設定
guard_params = {
    'G0': 0.4,           # やや慎重
    'g': 0.8,            # 経験重視
    'alpha': 0.7,        # 緊張しやすい
    'Theta0': 1.2,       # 跳躍は控えめ
    'T0': 0.25,          # 探索も控えめ
    'c1': 0.6,           # 熱で少し探索増加
    'a1': 0.7            # 経験で更に慎重に
}
```

---

## 9. 計測（テレメトリ）

### 9.1. リアルタイムKPI

$$\text{整合効率} = \frac{\|J\|}{\|p\|}, \quad \text{熱レベル} = E, \quad \text{温度} = T$$

$$\text{エントロピー} = H(\pi), \quad \text{跳躍頻度} = \frac{\text{jumps}}{\Delta t}$$

### 9.2. ダッシュボード表示

```
┌─────────────────── Nano-SSD Monitor ───────────────────┐
│ Alignment Eff: ████████░░ 78%   Heat: ███░░░░░░░ 2.1   │
│ Temperature:   █████░░░░░ 0.45   Entropy: ██████░░░ 0.6│
│ Jump Rate:     ▲ 0.12/sec        Last Jump: 15s ago   │
│ Creativity:    ████████████ 85%  (17/20 connections)  │
└────────────────────────────────────────────────────────┘
```

### 9.3. デバッグ・ログ出力

```python
if debug_mode:
    log(f"t={t:04d} p={p:.2f} j={j:.2f} E={E:.2f} T={T:.2f}")
    log(f"  kappa_mean={mean(kappa):.2f} jump_prob={jump_prob:.3f}")
    if jumped:
        log(f"  JUMP: {source} -> {target} (sim={sim:.2f})")
```

---

## 10. 安全装置（ゲームバランス）

### 10.1. 跳躍制限

```python
# クールダウン制御
min_jump_interval = 3.0  # 最短間隔（秒）
if (current_time - last_jump_time) < min_jump_interval:
    jump_probability *= 0.1  # 大幅減衰
```

### 10.2. 範囲制限

```python
# 世界観適合性チェック
def candidate_nodes(G, context):
    candidates = []
    for node in G.nodes:
        if is_compatible_with_worldview(node, context.world_tags):
            if distance(context.current, node) <= context.max_distance:
                candidates.append(node)
    return candidates
```

### 10.3. 冷静性ノブ

$$\beta_E^{\text{adj}} = \beta_E \cdot (1 + \text{calmness\_factor})$$

$$c_2^{\text{adj}} = c_2 \cdot (1 + \text{rational\_factor})$$

**効果：** 無駄跳躍を抑制し、必要跳躍のみ残す

---

## 11. 実装ヒント（Unity/Unreal）

### 11.1. Unity実装

```csharp
[System.Serializable]
public class NanoSSDParams 
{
    [Range(0.1f, 1.0f)] public float G0 = 0.5f;
    [Range(0.2f, 1.5f)] public float g = 0.7f;
    [Range(0.1f, 0.5f)] public float eta = 0.3f;
    // ... 他のパラメータ
}

public class NanoSSDNPC : MonoBehaviour 
{
    public NanoSSDParams parameters;
    private Matrix kappa;
    private float E, F, T;
    
    void Update() 
    {
        var p = SenseMeaningPressure();
        NanoSSDTick(p);
    }
}
```

**BT統合：**
- Behavior Designer/Apex等のDecoratorに $\theta$ 判定を実装
- Serviceで E, F, T を毎フレーム更新

### 11.2. Unreal実装

```cpp
USTRUCT(BlueprintType)
struct FNanoSSDParams 
{
    GENERATED_BODY()
    
    UPROPERTY(EditAnywhere, meta = (ClampMin = "0.1", ClampMax = "1.0"))
    float G0 = 0.5f;
    
    UPROPERTY(EditAnywhere, meta = (ClampMin = "0.2", ClampMax = "1.5"))  
    float g = 0.7f;
    
    // ... 他のパラメータ
};
```

**BT統合：**
- Blackboardに `E`, `kappa_mean`, `Theta` 等を配置
- BTのServiceで毎フレーム更新
- EQS（Environmental Query System）のスコアに $\kappa$ を加味

---

## 12. 拡張オプション（必要になったら）

### 12.1. 多体相互作用

**Hawkes過程** による集団カスケード：

$$\lambda_i(t) = \mu_i + \sum_{j \neq i} \int_{-\infty}^t \alpha_{ij} e^{-\beta_{ij}(t-s)} dN_j(s)$$

**用途：** 群集AIの連鎖反応、パニックの伝播

### 12.2. 相転移の可視化

**ランドウ自由エネルギー：**

$$F_S(\phi; p) = \frac{1}{2}\alpha(\phi - \phi_0)^2 + \frac{1}{4}\beta(\phi - \phi_0)^4 - p\phi$$

**用途：** NPCの状態遷移（冷静⇔興奮、協力⇔敵対）の可視化

### 12.3. 言語・関係学習

**拡張ノード：** 台詞、態度、関係性ラベルをノードに設定

**応用：** 
- 会話AI：過去の会話の成功パターンから新しい応答を生成
- 社交NPC：他キャラクターとの関係性を学習・更新

---

## まとめ

**Nano-SSD v1** は、構造主観力学の核心を軽量化し、リアルタイムゲーム環境で実行可能なミニマム実装を提供します。

### 🎯 達成した要件

1. **軽量性**：$O(|E_{\text{edges}}|)$ の線形計算量
2. **リアルタイム性**：1フレーム内で完結する更新サイクル  
3. **調整可能性**：豊富なパラメータによる行動特性の制御
4. **拡張性**：BT/GOAPとの自然な統合、追加機能への対応

### 🚀 期待される効果

- **予測可能性と驚き**：整合で安定感、跳躍で意外性
- **個性の表現**：パラメータ調整による多様なNPCキャラクタ
- **動的適応**：環境変化への自律的な学習・調整
- **デバッグ性**：可視化指標による調整の容易さ

このミニマム実装が、より豊かで知的なゲーム体験の基盤となることを期待しています。