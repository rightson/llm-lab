# 實驗導覽

從 repo 根目錄執行 `python -m labs.run <名稱>`。所有實驗只需 Python 與 NumPy，不下載模型、不呼叫遠端 API。

| 名稱 | 對應課程 | 應觀察到的結果 | 修改一個變數 |
| --- | --- | --- | --- |
| `tokens` | 03 | 「重力 gravity」為 10 個字元、14 個 UTF-8 bytes；可逆還原 | 換成 emoji 或不同語言 |
| `attention` | 04、06、07 | 權重約 [0.670,0.330]，輸出約 [6.698,6.605] | 把 q 改為 [0,1] |
| `training` | 02 | 交叉熵約從 1.3863 降到 0.4805 | 改語料中 aba/aca 比例 |
| `cache` | 08、13 | full/cached logits 誤差接近 0 | 改 prefill 切分點 |
| `sampling` | 10、11 | T 越高通常越平坦；top-p=.8 保留前兩個 | 改成 top-p=.5 |
| `generation` | 14 | 8 個 token IDs；UTF-8 分段能還原「重力」 | 改 greedy 為抽樣 |
| `memory` | 15 | 32/8/1 個 KV heads 分別 4/1/0.125GiB | context 或 batch 加倍 |
| `timing` | 12、16、17 | 不同長度的 CPU forward 局部計時 | 增加重複次數與模型寬度 |
| `quantization` | 18 | 較少 bits 通常產生較大還原誤差 | 刪掉 outlier 4.0 |
| `online` | 19 | 分塊與 dense attention 數值一致 | block_size 改為 1 或 8 |
| `scheduling` | 21 | static 10 回合，continuous 7 回合 | 改請求長度順序 |
| `speculation` | 22 | 接受率 0.8，校正分布等於 [0.6,0.3,0.1] | 改 draft 分布 |

## 對實驗結果負責

算例、模擬與 benchmark 要分開。`memory` 是理論配置估算；`scheduling` 是等步長模擬；`timing` 才有讀取本機時鐘，但只是 CPU 玩具模型。沒有任何實驗證明某張 GPU、某個 serving framework 或某種量化在你的工作負載會快多少。

`TinyDecoder` 預設使用 NumPy float64，故 cache bytes 與正文 BF16 容量算例不同。它用 concatenate 延伸 cache，會有複製成本；生產引擎通常採不同配置方式。不要用這個實作判斷 PagedAttention 的速度。

## 建議紀錄格式

每次實驗寫四行：我預測什麼；改了什麼；觀察到什麼；哪一個結果會推翻我的解釋。先保留正確版本，再做修改，最後重新執行測試。

核心測試檢查的是資訊因果性、cache 等價性、穩定 softmax、sampling 邊界、記憶體算例、排程 token 守恆、學習損失、分塊 attention 與 speculative 分布校正。通過測試表示這些教學不變量成立，不代表生成能力或生產效能合格。
