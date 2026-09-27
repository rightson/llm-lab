# 10｜如何把一個向量變成整個詞彙表的候選分數？

最後一層的 hidden state 仍是一個 d 維向量，並不是文字。生成系統需要比較詞彙表裡每個 token，因此使用 LM head，把 d 維投影到 V 維：

$$z=hW_{\mathrm{out}}$$

Wout 的 shape 是 d×V，z 的每個元素是某個 token 的 logit。Logit 可以是任意實數，不能直接解讀為百分比。套用 softmax 才得到總和為 1 的分布。

假設三個候選分數是 [2,1,0]，softmax 後約為 [0.665,0.245,0.090]。第一個候選最有優勢，但其機率不是 2，也不是 2/3 的簡單比例。

訓練時，每個位置都可能有自己的預測題；生成時，通常只需要這次輸入最後一個位置的 logits，用來選下一個 token。前面位置的 logits 是對前面各自下一步的預測，不能把它們當成整段未來答案。

有些模型讓輸出投影與輸入 embedding 共享權重，稱為 weight tying；本實驗為了清楚區分兩個角色，使用獨立參數。兩者的形狀與用途仍可彼此對應。

從機率最大值選出 token 是 greedy；依分布抽樣則是 sampling。LM head 本身只產生分數，選擇策略在下一課。

**練習：** 詞彙表 50,000 個 token、hidden width 4,096，獨立且沒有 bias 的 LM head 有幾個參數？

**答案：** 4,096×50,000=204,800,000。即使每次只選一個 token，朴素輸出投影仍須為大量候選評分。

**觀察：** `TinyDecoder.forward` 回傳 shape 為「此次新 token 數 × vocab_size」；`logits[-1]` 才是用來產生下一個 token 的那一列。

---

[課程首頁](../README.md) · [上一課](09-transformer-block.md) · [下一課](11-sampling.md)
