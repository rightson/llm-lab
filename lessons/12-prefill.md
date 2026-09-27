# 12｜第一個 token 出現前，模型在忙什麼？

使用者送入的 prompt 可能有幾千個 token。要預測第一個輸出，模型必須先把已知前文轉成可使用的內部表示。這個處理已知輸入的階段稱為 prefill。

Prefill 不是 Transformer 之前的另一個神秘模組。它會執行 embedding、位置處理、所有 Transformer blocks，並留下各層 K/V。最後一個 prompt 位置的 logits 可以用來選出第一個輸出 token。

因為整段 prompt 已知，同一層的許多 token 能用矩陣運算一起處理；因果遮罩仍保證每個位置看不到未來。若系統把長 prompt 切成幾塊處理，則稱 chunked prefill，不必一次把整段送進一個大工作批次。

對 dense attention 的簡化估算：token-wise projections 與 FFN 的成本大致隨 T 增加；全序列 attention 配對則含 T² 項。因此「prompt 變兩倍，時間一定變四倍」並不成立，因為總時間還取決於模型寬度、核心與硬體效率。

| 輸入長度變化 | 理論項的變化 |
| --- | --- |
| T → 2T | token-wise 計算約 2 倍 |
| T → 2T | dense attention 配對約 4 倍 |
| T → 2T | KV 儲存約 2 倍 |

從使用者送出請求到收到第一個 token 的時間叫 TTFT。Prefill 是其中一部分；排隊、tokenization、傳輸與服務端處理也可能影響 TTFT。

**實驗：** `python -m labs.run timing` 比較不同 prompt 長度的 forward 時間。這是 CPU 玩具模型的局部計時，不包含服務端 queue，也不能代表 GPU 生產效能。

**練習：** 長 prompt 的 TTFT 高，能否直接判定 GPU 算 prompt 太慢？

**答案：** 不能。如果縮短排隊時間就改善，原本的問題至少有一部分來自 queue；需要分開量測。

---

[課程首頁](../README.md) · [上一課](11-sampling.md) · [下一課](13-kv-cache.md)
