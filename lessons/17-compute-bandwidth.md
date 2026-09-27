# 17｜GPU 很強，為什麼生成還是慢？

計算前，資料必須先送到計算單元。若搬資料比乘加法耗時，增加算力未必加速。理解推論效能，至少需要同時看 FLOPs（浮點運算量）與 bytes（資料搬移量）。

用簡化下界表示一次工作的時間：

$$t\gtrsim\max\left(\frac{\mathrm{FLOPs}}{\text{有效算力}},\frac{\mathrm{bytes}}{\text{有效頻寬}}\right)$$

這不是完整效能模型；實際還有核心啟動、同步與無法完全重疊的工作。Arithmetic intensity 是 FLOPs/byte，表示每搬一份資料可以做多少運算。

單一請求的 decode 每層只加入少量新 token，仍可能需要讀取大部分 dense weights；同一份權重的重用機會少。Prefill 同時處理很多已知 token，矩陣乘法可以讓權重服務更多輸入，通常更容易提高運算利用率。

教學假設：每步需讀取 16GB 權重，有效頻寬 800GB/s，光這部分的搬移下界就是 20ms，對應最多約每秒 50 個單請求步驟。實際還需加上 KV 讀取與其他成本；如果權重流量、快取命中或模型稀疏性不同，就不能套用這個數字。

長 context 時，即使有 KV cache，每個新 query 仍可能讀取很長的 K/V。於是瓶頸可能從權重搬移轉向 KV 流量。較大的 batch 則增加權重重用，但也增加 KV、計算與排程壓力。

**練習：** 算力變兩倍、頻寬不變，在純 memory-bound 假設下能否期待速度兩倍？

**答案：** 不能；先處理搬移量、資料重用或頻寬。若工作原本 compute-bound，結論才可能不同。

**完成標準：** 不把「prefill 必定 compute-bound、decode 必定 memory-bound」當成定律，而能指出 batch、長度與硬體改變後，瓶頸也會改變。

---

[課程首頁](../README.md) · [上一課](16-metrics.md) · [下一課](18-quantization.md)
