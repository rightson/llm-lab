# 15｜模型放得下，為什麼一聊天還是記憶體不足？

模型權重只是推論記憶體的一部分，還要容納 KV cache、暫存張量、核心 workspace 與服務框架開銷。權重通常由請求共享，KV 則會隨請求數與 context 長度增加。

先估權重：P 個參數，每個 b bytes，大小約 P×b。8B 模型用 BF16，每個參數 2 bytes，權重約 16GB，換成二進位單位約 14.90GiB。1GiB=2³⁰ bytes，不等於十進位 1GB。

對標準、完整注意力、各層配置相同的 KV，未考慮 padding、分頁浪費或壓縮時：

$$M_{KV}=2\times B\times T\times L\times H_{KV}\times d_h\times b$$

2 代表 K 和 V，B 是請求數，T 是每個請求已快取的 token 數，L 是層數，Hkv 是 KV head 數，dh 是每個 head 的寬度，b 是每個元素 bytes。不同請求長度不等時，把 B×T 改成所有快取 token 數的總和。

教學配置：32 層、8 KV heads、head width 128、BF16、8192 tokens：每個請求恰好 1GiB。改成 32 KV heads 則是 4GiB；16 個相同請求的 KV 就分別是 16GiB 或 64GiB。

執行 `python -m labs.run memory` 重算。這些是自訂配置，不代表任一特定商品模型。滑動視窗、混合層、latent cache 等架構需改用自己的儲存公式。

**練習：** 剩餘 12GiB，而每個請求 KV 是 1GiB，能保證同時跑 12 個嗎？

**答案：** 不能。還需預留 workspace、安全餘量與未來生成造成的增長。這只提供理想容量上限。

**完成標準：** 能拆開固定權重與隨工作負載增長的狀態，而不是只用「模型有幾 B」判斷能否運行。

---

[課程首頁](../README.md) · [上一課](14-decode-stream.md) · [下一課](16-metrics.md)
