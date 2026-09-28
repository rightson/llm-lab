# 24｜入門篇檢核：解釋、驗證，再診斷一個推論系統

這是既有 24 篇入門短講的整合檢核。完整 LLM 理解還需補上模型訓練、泛化、消融與已訓練模型追蹤，詳見 [完整學習規劃](../docs/learning-plan.md)；不能把隨機權重的機制實驗當成語言學習成果。

完成這一課的標準不是記住名詞，而是能對觀察提出可被推翻的假設。先跑不需要外部服務的核心實驗，再把同一套方法搬到自己的推論服務。

## A. 重建計算流程

執行 `python -m labs.run all` 與 `python -m unittest discover -s tests -v`。閱讀 `labs/core.py`，標出 embedding、position、Q/K/V、mask、softmax、residual、FFN 與 LM head。

接著用 `[1,2,3,4,5,6]`，分別做一次完整 forward、先三個後三個、以及先三個後逐個。三種方式的 logits 應在浮點誤差內相同。故意把 cache offset 改成零，觀察哪個不變量會失敗，然後恢復正確實作。

交付一張逐步表：每次輸入幾個 token、每層 K/V shape、最後 logits shape、累積 context 長度。這比只展示生成的句子更能驗證你懂資料如何流動。

## B. 不靠跑分也能驗證的容量算例

假設 8B dense model、BF16 weights、32 層、8 KV heads、head width 128。推導單請求 8192 tokens 需要 1GiB KV；8 個請求需要 8GiB KV。權重約 14.90GiB，兩者合計約 22.90GiB，還沒有 workspace 與其他開銷。

問：24GiB 裝置能否保證支撐這個工作負載？答案是不能。餘量約 1.10GiB，且輸出還會讓 cache 增長。真正的准入上限需根據實作與保留額度決定。

## C. 有推論服務時的進階量測

選一個自己有權使用的模型與端點，記錄確切版本。控制輸出長度與 sampling 設定，逐一改變輸入長度、併發與共享前綴比例。先 warmup，再記錄多次結果；每次只改一個主要因素。

| 觀察 | 待驗證假設 | 區分原因的下一步 |
| --- | --- | --- |
| TTFT 高、ITL 正常 | 排隊或 prefill | 拆 queue time 與 prefill time |
| context 增長後 ITL 惡化 | KV 讀取成本增加 | 固定 batch、改 context，觀察流量 |
| 高併發開始 OOM | KV 或 workspace 不足 | 比對活躍 token 總量與記憶體 |
| 吞吐增加、p95 變差 | batching/排隊取捨 | 固定到達率比較排程策略 |
| 量化後沒有更快 | 核心或其他瓶頸 | 分開量權重流量、核心時間與品質 |

這些是診斷假設，不是僅憑症狀即可確定的原因。若沒有 GPU/端點，完成 A、B 就能交付核心課程成果；C 需標記尚未執行，不能用 CPU 玩具數字替代。

## 自評規準

每項 0–2 分：能完整追蹤一個 token；能說明並驗證 cache 正確性；能估權重/KV容量；能區分 TTFT/ITL/吞吐；能針對症狀設計區分性實驗。8 分以上且 cache 項不為零，代表已建立可用的基礎。

最後用五句話教會另一個人：輸入如何編碼、上下文如何混合、候選如何選出、狀態如何重用、效能如何量測。若仍需要把所有名詞一次列出，回到相應課程重做算例。

---

[課程首頁](../README.md) · [上一課](23-serving-architecture.md)
