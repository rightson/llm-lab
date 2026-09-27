# 19｜不改 Attention 的答案，能不能少搬一些資料？

朴素 attention 會建立 T×T 的分數與機率矩陣。當 T 很大，這些中間結果在 GPU 高頻寬記憶體 HBM 與晶片內較小的快速儲存之間往返，會消耗大量流量。FlashAttention 的核心想法是分塊計算，減少把巨大中間矩陣寫回 HBM。

先看一列 attention。輸出等於 Σᵢexp(sᵢ)vᵢ / Σᵢexp(sᵢ)。看似必須一次拿到所有分數，其實可以逐塊累積分子與分母。為避免數值溢位，再維護已見最大分數 m。

對新區塊 scores s、values V：

$$m'=\max(m,\max(s)),\quad \alpha=e^{m-m'}$$
$$\ell'=\alpha\ell+\sum_i e^{s_i-m'},\quad u'=\alpha u+\sum_i e^{s_i-m'}v_i$$

處理完所有區塊後輸出 u/ℓ。舊累積值乘 α，是把先前使用的指數基準換成新的最大值。這使我們不必保存整列機率，也能得到相同的數學結果。

執行 `python -m labs.run online`，比較一次算完與每次讀 3 個位置的結果。這是單 query、CPU 的 online-softmax 教學實驗，並不是 FlashAttention GPU kernel，也不能用它宣稱實際加速。

Exact attention 的意思是沒有刻意把 attention 近似成另一個目標；不同浮點運算順序仍可能有微小數值差異。對 dense attention，分塊主要改善資料搬移與中間記憶體，沒有消除所有 token 配對所需的二次運算量，也不等於移除 KV cache。

**練習：** 為何新區塊出現更大的 score 時，不能只更新 m 而不縮放舊的 u、ℓ？

**答案：** 舊值與新值將使用不同的指數基準，直接相加就失去正確比例。

**閱讀：** [FlashAttention](https://arxiv.org/abs/2205.14135)。

---

[課程首頁](../README.md) · [上一課](18-quantization.md) · [下一課](20-pagedattention.md)
