# 21｜如何讓很多請求共享 GPU，又不讓互動卡住？

將多個請求放進同一批次，可以讓一次讀取的權重服務更多 tokens。但生成長度不一：有人一個 token 就結束，有人要幾百個。若整批等最長請求完成才補入新人，部分位置會閒置。

Continuous batching 在迭代邊界移除完成的請求、加入等待中的請求，讓可用位置持續工作。它不是一次把一個請求的未來 token 全部算出來，因果依賴依然存在。

執行 `python -m labs.run scheduling`。四個請求需要 [1,5,1,5] 個 decode 步驟，有兩個 slots，全部在時間零到達：

| 策略 | 完成全部的回合數 | slot 使用率 |
| --- | ---: | ---: |
| Static batching | 10 | 60.0% |
| Continuous batching | 7 | 約 85.7% |

這是每步成本相同的離散模擬。真實 GPU 的 batch 變大後，每步成本不一定不變；prefill、記憶體與到達時間也會影響結果，因此不能把這個比值當作實測加速。

長 prompt 的 prefill 可能佔據執行時間，拖慢既有請求的 decode。Chunked prefill 把大工作切成較小單位，以便交錯執行。切太小會增加管理與啟動開銷，切太大則可能拉高其他串流的 ITL。

排程要明確決定優先目標：短 TTFT、穩定 ITL、公平性、最大吞吐，或某種組合。若到達率長期超過服務率，排隊只會越來越長；必須搭配容量擴充、admission control 或背壓。

**練習：** 為了讓新請求立即開始，永遠優先執行 prefill，會有什麼副作用？

**答案：** 已開始輸出的請求可能長時間收不到後續 token，甚至飢餓。改善 TTFT 可能犧牲 ITL。

**閱讀：** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)，了解 iteration-level scheduling。

---

[課程首頁](../README.md) · [上一課](20-pagedattention.md) · [下一課](22-speculative-decoding.md)
