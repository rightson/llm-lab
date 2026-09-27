# 術語速查

| 術語 | 一句話 | 課程 |
| --- | --- | --- |
| Token | 模型使用的離散文字片段；不一定是一個字 | 03 |
| Vocabulary | 可使用的 token 集合與 ID 對照 | 03 |
| Weight / parameter | 訓練調整、推論時通常固定的數值 | 02 |
| Embedding | 由 ID 查出的學習向量 | 05 |
| Hidden state | token 在某一層的內部向量表示 | 05–10 |
| Q / K / V | 查詢、配對用鍵、被聚合的值 | 06–07 |
| Causal mask | 禁止位置讀取未來 token 的遮罩 | 08 |
| Attention head | 一套獨立的查詢與聚合投影 | 08 |
| FFN | 對每個位置進行非線性加工的網路 | 09 |
| Residual | 把子層更新加回原表示的路徑 | 09 |
| Logit | 正規化前的候選分數 | 10 |
| Softmax | 把分數轉成總和為一的非負分布 | 07、10 |
| Prefill | 處理已知 prompt、建立狀態的階段 | 12 |
| Decode | 沿已生成前綴繼續產生 token 的階段 | 14 |
| KV cache | 每層歷史位置的 K/V 狀態 | 13 |
| TTFT | 請求到第一個輸出 token 的時間 | 16 |
| ITL / TPOT | 相鄰 token 間隔 / 平均每個後續 token 時間 | 16 |
| Throughput | 系統單位時間完成的工作量，需定義單位 | 16 |
| Goodput | 符合指定目標的有效工作量 | 16 |
| HBM | GPU 常用的高頻寬外部記憶體 | 17、19 |
| Quantization | 用較少的離散等級近似數值 | 18 |
| Continuous batching | 在迭代邊界替換完成請求，持續利用批次容量 | 21 |
| Speculative decoding | 先草擬、再由目標模型驗證校正的生成方式 | 22 |
| Prefix caching | 重用相容且相同 token 前綴的計算狀態 | 23 |
| SLO | 對延遲、可用性等服務表現設定的目標 | 16、23 |
