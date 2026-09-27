# 13｜為什麼可以不重算前面的 token？

已生成「重力 是」，現在要繼續。如果每次都把整個前文從頭算一遍，模型會重複產生前面各位置的 K/V。因果模型裡，前面位置的表示不依賴後來新增的 token，因此可以保存並重用。

新 token 仍然需要在每一層計算自己的 Q、K、V。把新 K/V 附加到 cache 後，新的 Q 會查詢整個可見的 K，並混合對應的 V。Cache 省掉舊位置的重算，沒有省掉讀取前文的需求。

為什麼通常存 K/V 而不存 Q？新位置需要的是「自己的 query 對歷史 keys/values」；舊位置的 query 不參與這個新輸出的計算。

| 項目 | 是否隨這次請求改變 |
| --- | --- |
| 模型權重 | 一般推論期間固定 |
| 每層 KV cache | 隨輸入與生成延伸 |
| 新 token 的 hidden state | 每次重新計算 |
| 已輸出的文字 | 供呈現與記錄；不是 KV 的替代品 |

執行 `python -m labs.run cache`。程式會先完整計算六個 tokens，再先做三個 token 的 prefill，後面逐個加入。比較的是**所有位置的 logits**，不只是恰巧選出相同 token。最大差異應接近浮點誤差。

若不相同，先檢查位置 offset、mask、每層 cache 與快取前綴是否一致。只比較 greedy token 可能漏掉錯誤，因為 logits 已偏移但最大值尚未換人。

**練習：** 同樣一句文字，換了模型權重後，能否沿用舊 KV？

**答案：** 不能直接沿用；KV 是特定權重、輸入與位置計算出的狀態。

**閱讀：** [Transformers：How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation)。

---

[課程首頁](../README.md) · [上一課](12-prefill.md) · [下一課](14-decode-stream.md)
