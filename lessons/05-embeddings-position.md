# 05｜從 ID 查表，到知道誰在誰前面

Token ID 只是編號，所以模型先用 embedding table 把 ID 換成向量。若詞彙表有 V 個 token、每個向量 d 維，表格 E 的 shape 是 V×d。ID=i 的 token 取出第 i 列 E[i]。

假設教學詞彙表裡「重」是 ID 2，而 E[2]=[0.2,-0.1,0.7]，查表結果就是這個三維向量。這些數字由訓練調整，不是人工寫進去的中文定義。同一個 token 起初查出同一列，後面的 Transformer 才會根據前文更新其表示。

現在考慮「狗追貓」與「貓追狗」。出現的 token 類似，順序卻改變意思。模型因此還需要位置資訊。本實驗採用容易理解的 learned positional embedding：

$$X_t=E[\mathrm{id}_t]+P[t]$$

P[t] 是位置 t 的向量。相同 token 在不同位置，輸入向量就不同。這種方法有一張固定大小的位置表，超過表格範圍不能直接索引。

有些模型使用 RoPE，把位置相關旋轉套用到 Q、K，使注意力分數帶入相對位置關係。它不是「在所有模型的 embedding 後面加一張位置表」。本系列程式刻意採用較簡單的版本；讀懂後才適合換成其他位置機制。

**觀察程式：** 打開 `labs/core.py`，找到 `self.embedding[ids] + self.position[offset:offset+n]`。offset 是這次新 token 的起始位置。若使用 KV cache 卻把 offset 重設成零，同一段文字的完整運算與分段運算就可能不一致。

**練習：** 前面已處理 10 個 token，下一個 token 應取 P[0] 還是 P[10]？

**答案：** 採零起算時取 P[10]。快取省略重算，並沒有把文字的位置重設。

---

[課程首頁](../README.md) · [上一課](04-vectors-matrices.md) · [下一課](06-attention-intuition.md)
