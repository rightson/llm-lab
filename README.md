# LLM Lab｜深刻理解 LLM，從直覺走到程式與硬體

**核心目標：能解釋、推導、預測並驗證 LLM 的行為。** 從下一個 token、表示與學習出發，親手訓練小模型，再追到推論、記憶體與自己的 NPU。演算法、程式與 RTL 都服務於同一條理解主線。

適合從基本四則運算開始的讀者；數學與程式先備在需要時補上。繁體中文教材，保留英文技術名稱；[環境入門](docs/getting-started.md) 說明如何執行實驗。

## 先看學習地圖

- **[完整學習規劃](docs/learning-plan.md)**：八階段、40 個學習單元的問題、推導、實驗與理解 gates。
- [進度與交付順序](docs/learning-progress.md)：哪些已有、哪些還缺，以及下一輪先補什麼。
- [教學與實驗標準](docs/lesson-standard.md)：每個機制都要能手算、預測、干預與核對。

| 階段 | 要真正理解的問題 |
| --- | --- |
| A：任務與機率 | 為什麼預測下一個 token 能形成生成模型？ |
| B：表示與運算 | 文字如何變成可學習、可運算的向量？ |
| C：學習與泛化 | 權重怎麼從錯誤中更新，怎麼分辨學會與記住？ |
| D：Attention | 模型如何選擇性使用 context？Q/K/V 如何學來？ |
| E：Transformer | residual、normalization、FFN、多層為何一起使用？ |
| F：生成與能力 | 訓練與生成如何接起來？模型的答案能證明什麼？ |
| G：推論成本 | 為何記憶體、頻寬、精度與排程改變效能？ |
| H：自有 NPU | 如何將同一模型落到數值、資料流與 RTL，並由 Python 驗證？ |

前段先建立可訓練、可評估的小模型；後段用同一份模型 trace 延伸到硬體。每個概念分多次回訪，不要求第一次學 attention 就懂 SRAM/STA。NPU 仍是完整路線的深入終點，NanoNPU 始終只作參考。

## 現有內容與下一個重點

已有 **24 課入門短講、12 個基礎 CPU 實驗、CD01 共同設計實驗與 18 個已通過的 Python 測試**。這些是新學習地圖的基礎，不代表整個深度課程已完成。

目前 Transformer 仍使用隨機權重；真正訓練的是獨立 bigram。接下來依序補：受控資料與 baseline → gradients/backprop → 可訓練 attention/Transformer → 消融與泛化 → 推論品質/成本 → Python/RTL 整合。現有 PE RTL 尚未執行 simulator。

共同設計材料：[主張](docs/codesign-charter.md) · [硬體深入路線](docs/codesign-roadmap.md) · [CD01](codesign/README.md) · [NanoNPU 參考審查](docs/nanonpu-reference-review.md)。完整規劃與尚待實作內容請以進度表為準。

## 現有入門篇能建立什麼基礎

- 追蹤文字 → token IDs → embedding → Transformer → logits → sampling → 文字。
- 手算小型 attention，並驗證完整計算與 KV cache 分段計算的一致性。
- 區分訓練、prefill、decode、streaming，以及模型與服務系統的責任。
- 估算權重/KV 記憶體，辨識 TTFT、ITL、吞吐與排程的取捨。
- 解釋各種加速技術省下什麼、增加什麼成本，以及如何驗證效果。

## 開始實驗

Python 3.10 以上。macOS/Linux 使用：

```bash
git clone https://github.com/rightson/llm-lab.git
cd llm-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m labs.run all
python -m unittest discover -s tests -v
```

Windows 與命令說明見[環境入門](docs/getting-started.md)。安裝 NumPy 後，核心實驗可離線執行。請先讀[實驗導覽](labs/README.md)，知道每個結果代表什麼。

## 既有 24 課索引

以下是已存在的入門短講，含算例與自我檢查。依完整學習地圖選讀並補上新增實驗；先寫出預測，再看結果。原檔名與連結保持可用；第 24 課是入門篇檢核，不是整個深度主線的終點。

| 階段 | 課程 | 實驗 |
| --- | --- | --- |
| 一：文字與學習 | [01 下一個 token](lessons/01-next-token.md) · [02 訓練與推論](lessons/02-training-inference.md) · [03 Tokenizer](lessons/03-tokenization.md) · [04 向量與矩陣](lessons/04-vectors-matrices.md) | `tokens`、`training` |
| 二：上下文如何流動 | [05 Embedding 與位置](lessons/05-embeddings-position.md) · [06 Attention 直覺](lessons/06-attention-intuition.md) · [07 Q/K/V 計算](lessons/07-attention-math.md) · [08 因果遮罩與多頭](lessons/08-causality-heads.md) | `attention` |
| 三：從向量選出答案 | [09 Transformer block](lessons/09-transformer-block.md) · [10 LM head](lessons/10-lm-head.md) · [11 Sampling](lessons/11-sampling.md) · [12 Prefill](lessons/12-prefill.md) | `sampling`、`timing` |
| 四：生成與容量 | [13 KV cache](lessons/13-kv-cache.md) · [14 Decode 與串流](lessons/14-decode-stream.md) · [15 記憶體估算](lessons/15-memory-budget.md) · [16 效能指標](lessons/16-metrics.md) | `cache`、`generation`、`memory` |
| 五：加速的原理 | [17 算力與頻寬](lessons/17-compute-bandwidth.md) · [18 量化](lessons/18-quantization.md) · [19 FlashAttention](lessons/19-flashattention.md) · [20 PagedAttention](lessons/20-pagedattention.md) | `quantization`、`online` |
| 六：推論服務 | [21 Batching 與排程](lessons/21-batching-scheduling.md) · [22 Speculative decoding](lessons/22-speculative-decoding.md) · [23 服務架構](lessons/23-serving-architecture.md) · [24 入門篇檢核](lessons/24-capstone.md) | `scheduling`、`speculation`、整合驗證 |

## 正確理解那張推論流程圖

Prefill 與 decode 是使用同一模型的兩個執行階段；兩者都執行 Transformer。首個輸出通常由 prefill 最後位置的 logits 產生。一般單 token decode 每步選出一個新 token，speculative decoding 則可能在一輪驗證中接受多個。

「第一個 token 慢」只能提示調查 TTFT 的各部分，不能直接確診 prefill；「streaming 慢」也可能受排隊、傳輸或 buffering 影響。Embedding 是學到的表示，並非把文字意義直接翻譯成一串人類已定義的座標。

## 教學範圍與實驗邊界

主線是 decoder-only、dense causal Transformer 的文字推論。`TinyDecoder` 是兩層、單 head、learned position、LayerNorm、ReLU FFN 的隨機權重模型；它示範真實運算關係，但不會生成有意義的自然語言。真正會更新權重的是獨立的 bigram 訓練實驗。

基礎課的多頭/GQA/RoPE、分頁管理與多 GPU 在正文說明；CPU 模擬沒有實作完整生產引擎。Online-softmax 與 speculative mass 實驗只驗證數學，不是 GPU kernel 或實測加速。GPU serving benchmark 放在第 24 課進階作業，需自行提供模型與服務。

基礎篇涵蓋原始圖中的主要概念與必要前置知識；共同設計篇逐步延伸自有 NPU。MoE、混合注意力與多模態不在目前首版範圍；小模型訓練已提前列為理解主線的核心，量化與硬體則要回接它的品質評估。

## 參考與維護

- [原始論文與官方文件](docs/references.md)
- [術語速查](docs/glossary.md)
- [本次驗證紀錄](docs/validation.md)

文中的 toy 數字是教學算例；真正量測需記錄模型、版本、精度與工作負載。課程沒有複製原始社群圖片，其內容以重新推導的文字、表格與示意圖呈現。
