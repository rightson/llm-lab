# LLM Lab｜理解模型、實作演算法、部署 NPU 加速

從 LLM 的詳細設計出發，完整理解訓練與推理如何運作，再親手將演算法寫成程式，最後部署到自行設計的 NPU，驗證正確性與加速效果。

**同一個模型，貫穿三個層面：模型原理 → 演算法程式設計 → NPU 部署與加速。**

## 第一層｜理解 LLM 模型的設計、訓練與推理

先建立能逐步推導運算與解釋設計取捨的直覺。

| 主題 | 要理解到什麼程度 |
| --- | --- |
| 模型設計 | Tokenizer、embedding、位置資訊、Q/K/V、causal attention、multi-head、FFN、residual、normalization、LM head：各自解決什麼問題，如何連接，張量維度如何變化？ |
| 訓練步驟 | 資料切分、序列與 label 對齊、forward、loss、backpropagation、optimizer update、驗證與 checkpoint：一次訓練迭代如何改變參數？ |
| 推理步驟 | Prompt 編碼、prefill、logits、sampling、decode、KV cache、停止條件與 streaming：一個 token 如何產生，哪些狀態被保留？ |
| 能力與限制 | 如何分辨記住資料與泛化？改動 context、mask、模型結構或精度，會如何影響結果？ |

**交付：** 能手算小例子，畫出資料與梯度的流向，說明每一步的必要性，並預測改動後的行為。

## 第二層｜把演算法實作成可訓練、可推理的程式

把第一層的公式、步驟與假設轉成可讀、可測試的 Python 程式。

- 先以 NumPy 實作最小算子與數值 reference，再與框架實作比對。
- 從 baseline、固定視窗模型到小型 Transformer，實作完整訓練迴圈與評估。
- 實作生成與 KV cache，追蹤每層的張量、logits、梯度與狀態。
- 以 gradient check、消融、獨立測試集與錯誤注入，檢驗理解和程式。
- 對同一模型分析 shape、記憶體、運算量與資料搬移，找出值得加速的部分。

**交付：** 一個真正訓練過、能執行推理的小模型，以及可重現的程式、checkpoint、測試與效能基準。

## 第三層｜部署到 NPU，驗證與優化加速

以第二層的模型與工作負載為依據，設計硬體並建立部署路徑。

| 工作 | 具體內容 |
| --- | --- |
| 決定數值與分工 | 量化與混合精度、算子支援範圍、CPU fallback、模型品質預算 |
| 編譯與部署 | 算子 lowering、tiling、layout、記憶體配置、指令或 descriptor、權重載入與 runtime |
| 設計自己的 NPU | PE、運算陣列、SRAM/buffer、DMA、控制與 RTL；依模型需求比較架構 |
| Python/RTL 共同驗證 | Python 驅動模擬 NPU，寫入資料與指令、讀回結果，逐算子與逐層比對 |
| 驗證加速效果 | 同一工作負載比較品質、cycles、資料流量與端到端成本；將傳輸與 fallback 納入 |

**交付：** 同一個小模型在 CPU 與模擬 NPU 共同執行的結果，以及正確性與效能比較。這條路線不需要 tape-out；模擬 cycles、時脈假設與實體量測分別報告。

**llm-lab 是主專案，NanoNPU 僅供參考。** NPU 的架構、ISA 與資料流由我們的 LLM 工作負載推導；參考實作不決定課程路線。

## 三層如何接在一起？以 Attention 為例

| 層面 | 同一個問題逐步深入 |
| --- | --- |
| 理解模型 | 為什麼需要 Q/K/V？mask 如何限制資訊？softmax 後如何混合 V？訓練時參數如何更新？ |
| 實作演算法 | 手算對照 NumPy；接入可訓練模型；實作 cached attention，檢查數值與梯度 |
| NPU 部署加速 | 決定矩陣乘法與 softmax 的分工，安排 tile/KV/SRAM，透過 RTL 比對輸出並分析瓶頸 |

每個主題先理解再實作；有了正確程式與固定工作負載後，再進入硬體加速。全程使用相同的模型與資料，讓每項優化都能追溯到原本的演算法與品質要求。

## 學習地圖與目前進度

[完整規劃](docs/learning-plan.md) 將三個層面展開成 40 個學習單元；[教學標準](docs/lesson-standard.md) 定義手算、實驗與驗收方式；[進度表](docs/learning-progress.md) 區分已完成與待實作內容。

目前已有 24 課入門短講、12 個基礎 CPU 實驗、CD01 整數／陣列模型；前次執行的 18 個 Python 測試通過。現有 Transformer 是隨機權重，訓練實驗為獨立 bigram。完整小模型訓練、NPU 部署與 Python/RTL 整合仍待完成，PE RTL 尚未執行模擬。

下一步優先補齊可訓練小模型與評估，讓第三層取得可信的 reference 與 workload。

## 開始實驗

Python 3.10 以上；安裝 NumPy 後，現有核心實驗可離線執行，不需要 API key 或下載模型。

```bash
git clone https://github.com/rightson/llm-lab.git
cd llm-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m labs.run all
python -m codesign.baseline --size 8 --rows 8
python -m unittest discover -s tests -v
```

Windows 操作與 Python 入門見 [環境說明](docs/getting-started.md)，各實驗的觀察重點見 [實驗導覽](labs/README.md)。

## 既有入門短講

以下是三層課程可重用的基礎材料；更完整的訓練、演算法實作與部署實驗依學習地圖補齊。

| 階段 | 課程 | 實驗 |
| --- | --- | --- |
| 一：文字與學習 | [01 下一個 token](lessons/01-next-token.md) · [02 訓練與推論](lessons/02-training-inference.md) · [03 Tokenizer](lessons/03-tokenization.md) · [04 向量與矩陣](lessons/04-vectors-matrices.md) | `tokens`、`training` |
| 二：上下文如何流動 | [05 Embedding 與位置](lessons/05-embeddings-position.md) · [06 Attention 直覺](lessons/06-attention-intuition.md) · [07 Q/K/V 計算](lessons/07-attention-math.md) · [08 因果遮罩與多頭](lessons/08-causality-heads.md) | `attention` |
| 三：從向量選出答案 | [09 Transformer block](lessons/09-transformer-block.md) · [10 LM head](lessons/10-lm-head.md) · [11 Sampling](lessons/11-sampling.md) · [12 Prefill](lessons/12-prefill.md) | `sampling`、`timing` |
| 四：生成與容量 | [13 KV cache](lessons/13-kv-cache.md) · [14 Decode 與串流](lessons/14-decode-stream.md) · [15 記憶體估算](lessons/15-memory-budget.md) · [16 效能指標](lessons/16-metrics.md) | `cache`、`generation`、`memory` |
| 五：加速的原理 | [17 算力與頻寬](lessons/17-compute-bandwidth.md) · [18 量化](lessons/18-quantization.md) · [19 FlashAttention](lessons/19-flashattention.md) · [20 PagedAttention](lessons/20-pagedattention.md) | `quantization`、`online` |
| 六：推論服務 | [21 Batching 與排程](lessons/21-batching-scheduling.md) · [22 Speculative decoding](lessons/22-speculative-decoding.md) · [23 服務架構](lessons/23-serving-architecture.md) · [24 入門篇檢核](lessons/24-capstone.md) | `scheduling`、`speculation`、整合驗證 |

## 深入資料

- [共同設計主張](docs/codesign-charter.md) · [硬體深入路線](docs/codesign-roadmap.md)
- [CD01：從整數乘加到時脈](codesign/README.md) · [NanoNPU 參考審查](docs/nanonpu-reference-review.md)
- [原始論文與官方文件](docs/references.md) · [術語速查](docs/glossary.md) · [驗證紀錄](docs/validation.md)
