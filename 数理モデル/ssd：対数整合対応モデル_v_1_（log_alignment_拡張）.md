# SSD：対数整合対応モデル v1（Log‑Alignment 拡張）

> 目的：既存の「整合×跳躍」数理API／nano_ssdに**対数整合（Weber–Fechner対応）**を明示導入し、非線形入力を線形応答へ安定写像する層を標準化する。

---

## 0. 要旨（TL;DR）
- **Log‑Alignment Layer** を入力処理に追加：\( \hat p = \mathcal{L}_{\log}(p) \)
- 整合流は \( j = (G_0 + g\,\kappa)\,\hat p + \varepsilon \)
- 未処理圧の計上は、スケール整合係数 \(\zeta\) を介して \(\|p\|\) と \(\|j\|\) を比較
- 適応ゲイン \(\alpha_t\) によりダイナミックレンジを自動最適化（Weber–Fechner適応）
- 既存API（AlignFlow / UpdateHeat / Step）に非破壊でオプション追加

---

## 1. Log‑Alignment の定義
### 1.1 変換（符号保持・要素別）
\[\boxed{\quad \hat p_i = \operatorname{sign}(p_i)\,\frac{\log\big(1 + \alpha_t\,|p_i|\big)}{\log b}\quad}\]
- 既定：\(b = e\)（自然対数）、\(\alpha_t > 0\)
- 小信号域：\(\hat p_i \approx \alpha_t\, p_i\)（線形）
- 大信号域：増分感度が \(\propto 1/|p|\) に減衰（飽和防止）

### 1.2 適応ゲイン（環境適応）
\[\alpha_t = \frac{\alpha_0}{\epsilon + \mathrm{EMA}_\tau\big(|p|\big)}\]
- \(\mathrm{EMA}_\tau\) は指数移動平均（時間定数 \(\tau\)）
- 明暗順応・音量順応のアナロジー：入力の平均強度が上がると \(\alpha_t\) が下がり、依然として可変域を確保

> 実装メモ：ベクトル入力は成分別EMAまたはノルムEMAのいずれも可。既定はノルムEMA（低コスト）。

---

## 2. 既存APIへの組み込み
### 2.1 AlignFlow（置換）
**Before**：\( j = (G_0 + g\kappa)\,p + \varepsilon \)

**After**：
\[\boxed{\quad j = (G_0 + g\kappa)\,\mathcal{L}_{\log}(p;\alpha_t,b) + \varepsilon \quad}\]

### 2.2 UpdateHeat（未処理圧E）
スケール不一致を補正するため、\(\zeta\) を導入：
\[\frac{dE}{dt} = \alpha_E\,\Big[\,\|p\| - \zeta\,\|j\|\,\Big]_+ - \beta_E\,E\]
- \(\zeta\) は単位整合係数。既定：\(\zeta = \left\|\,G_0 + g\,\kappa\,\right\|_{\mathrm{op}}\cdot c_b\)
- \(c_b\) は基底変換定数（\(b\neq e\) 時）。実務では**学習またはグリッドサーチ**で良い

> 代替案：log空間の残差 \([\,\|\hat p\| - \|j\|\,]_+\) を用いる方法も提供（切替フラグ）。

### 2.3 Threshold/JumpRate（そのまま）
- 跳躍率：\(h = h_0\exp\big((E-\Theta)/\gamma\big)\) は変更なし
- Log‑Alignment は**整合側の安定化**に効く（Eの成長を抑制）

---

## 3. 仕様（関数API拡張）
```text
AlignFlow(p, G0, κ; g, ε, log_align: Bool=False,
          alpha0=1.0, base=e, ema_tau=50, eps=1e-6,
          adapt_mode={norm|per_channel}) → j, state

UpdateHeat(E, p, j; αE, βE, use_log_residual=False, ζ=None) → E'
```
- `log_align=False` なら従来挙動。
- `ζ=None` の場合は自動推定（起動時キャリブレーション or EMA）。

---

## 4. 数理的性質と保証
1) **単調性**：\(\partial \hat p/ \partial p > 0\)（\(p\neq0\)）で応答は単調
2) **原点近傍線形**：小信号で \(\hat p \approx \alpha_t p\) → 既存学習則と整合
3) **飽和回避**：大信号でも有限増分 → 破綻時のE爆増を緩和
4) **Weber–Fechner整合**：\(\alpha_t\) により**相対変化**の感度が主導

---

## 5. 推奨パラメータ（初期値）
| パラメータ | 既定 | 意味・チューニング |
|---|---:|---|
| `log_align` | True | 本拡張を有効化 |
| `alpha0` | 1.0 | 原点傾き（1推奨）|
| `base` | e | 対数底。dB的解釈なら10 |
| `ema_tau` | 50 | 入力強度の適応時間（tick）|
| `eps` | 1e-6 | 数値安定化 |
| `ζ` | auto | 単位合わせ。自動か学習 |
| `use_log_residual` | False | 需要に応じ切替 |

---

## 6. 参考擬似コード（NumPy想定）
```python
class LogAlign:
    def __init__(self, alpha0=1.0, base=np.e, ema_tau=50, eps=1e-6, adapt_mode='norm'):
        self.alpha0 = alpha0
        self.base = base
        self.ema_tau = ema_tau
        self.eps = eps
        self.adapt_mode = adapt_mode
        self.m = 0.0  # EMA state
        self.log_base = np.log(base)

    def step(self, p):
        # 1) 更新EMA
        x = np.linalg.norm(p) if self.adapt_mode == 'norm' else np.abs(p)
        beta = np.exp(-1.0 / max(1, self.ema_tau))
        self.m = beta * self.m + (1 - beta) * x
        alpha_t = self.alpha0 / (self.eps + (self.m if np.isscalar(self.m) else np.mean(self.m)))
        # 2) 符号保持log変換
        phat = np.sign(p) * np.log1p(alpha_t * np.abs(p)) / self.log_base
        return phat, alpha_t
```

**整合流**
```python
phat, _ = log_layer.step(p) if log_align else (p, None)
j = (G0 + g * kappa) @ phat + epsilon_noise()
```

**未処理圧**
```python
if use_log_residual:
    resid = max(0.0, np.linalg.norm(phat) - np.linalg.norm(j))
else:
    zeta_eff = zeta if zeta is not None else zeta_auto  # 事前推定 or EMA
    resid = max(0.0, np.linalg.norm(p) - zeta_eff * np.linalg.norm(j))
E += alpha_E * resid - beta_E * E
```

---

## 7. 既存モジュールへの統合手順
1) **API拡張**：`AlignFlow` に `log_align` 系引数を追加
2) **State管理**：`LogAlign` のEMA状態を `S` に保持（`S.log_state`）
3) **キャリブレーション**：起動後数十tickは `ζ` をEMAで自動推定
4) **Nano‑SSD**：`sense()` 出力直後に `LogAlign` を適用
5) **ダッシュボード**に `alpha_t` と `residual_mode` を追加

---

## 8. テスト項目（最小）
- **単調性テスト**：ランダムpで \(\|\hat p\|\) が \(\|p\|\) に単調増加
- **原点傾き**：\(\lim_{p\to0} \hat p/p \approx 1\)（`alpha0=1`）
- **飽和抑制**：極大入力でEのピークが従来比で減少
- **適応**：定常入力振幅を上げたとき `alpha_t` が低下
- **後方互換**：`log_align=False` で従来と一致

---

## 9. オプション拡張（将来）
- **局所適応**：チャンネル別EMA（視覚・聴覚の帯域別ダイナミクス）
- **可逆近似**：必要に応じた逆写像 \(\mathcal{L}_{\log}^{-1}\) で物理スケールへ復元
- **スペクトルlog**：STFT後に振幅logを適用（音場モデル向け）

---

## 10. まとめ
- Log‑Alignment を導入することで、**非線形世界を線形に“扱う”**整合プロセスを明示化
- 破綻側（指数跳躍）は従来式を維持 → **安定と変化の双対**をフル表現
- 実装は**後方互換・低侵襲**で、既存コードにオプトイン導入が可能

