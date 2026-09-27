# 07｜把 Q、K、V 算成真正的輸出

上一課只處理一個查詢。現在把 T 個位置排在矩陣裡，同時產生：Q=XWq、K=XWk、V=XWv。單一 head 下，Q、K 的 shape 都是 T×dₖ，因此 QKᵀ 會得到 T×T 的分數表。

第 i 列、第 j 欄代表位置 i 對位置 j 的配對分數。分數可能為負，也不一定加總為 1，所以不能直接當機率。我們逐列套用 softmax：

$$A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV$$

softmax 的定義是 pᵢ=exp(zᵢ)/Σⱼexp(zⱼ)。例如 [0,0] 會變成 [0.5,0.5]。減去同一個常數不改變結果，所以實作先減掉最大值，避免 exp(1000) 溢位。

為什麼除以 √dₖ？在分量近似獨立、尺度穩定的簡化假設下，點積變異會隨維度增加；縮放有助於避免 softmax 過度飽和。這是數值尺度的處理，不是額外的語意規則。

M 是 mask，不可讀取的位置設為負無限大，softmax 後其權重為零；可讀取的位置加零。下一課會解釋哪些位置不能讀。

最後，A 是 T×T，V 是 T×dᵥ，因此 AV 是 T×dᵥ。每個輸出位置都得到一個從可見位置整合而來的向量。

**練習：** 在 `attention` 實驗把所有 scores 加 100，再算 softmax，權重會變嗎？

**答案：** 在浮點誤差範圍內不變；共同因子 exp(100) 在分子分母相消。這也是穩定實作成立的理由。

**閱讀：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，§3.2；原論文是 encoder-decoder 架構，本系列採 decoder-only 教學架構。

---

[課程首頁](../README.md) · [上一課](06-attention-intuition.md) · [下一課](08-causality-heads.md)
