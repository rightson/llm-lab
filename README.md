# LLM Lab｜從零走到演算法與 NPU 共同設計

從「下一個 token 是什麼」開始，逐步算出一個小型 Transformer 的輸出，再理解 KV cache、GPU 瓶頸與多人服務。**24 課基礎正文、12 個 CPU 基礎實驗，加上第一個 NPU 共同設計實驗；18 個 Python 測試**，不需要 API key 或下載模型。

適合能做基本加減乘除、想理解 LLM 如何運作的讀者。數學從向量與矩陣乘法教起；前幾課可先用紙筆完成。程式實驗需要會在終端機執行指令，附有[入門說明](docs/getting-started.md)。正文使用繁體中文，技術名稱保留英文。

## 新主線：LLM 演算法與自己的 NPU 一起設計

**本 repo 是主專案；NanoNPU 僅供參考。** 架構、ISA、資料流與記憶體由我們的 LLM 工作負載推導，並用實驗反覆修正。第一版從 integer GEMM 和 PE 做起，往 compiler、SRAM/DMA、attention、KV、RTL、STA 與實體設計延伸。

- [設計主張與驗收標準](docs/codesign-charter.md)
- [25–72 課延伸路線](docs/codesign-roadmap.md)：48 課規劃；尚未全部完成。
- [CD01：同一個矩陣乘法，從數字到時脈](codesign/README.md)：可立即執行的整數與 mesh 時序模型，含自己的 PE RTL/testbench。
- [NanoNPU 固定版本審查](docs/nanonpu-reference-review.md)：只作比較，沒有將其 ISA/RTL 設為建置依賴。

```bash
python -m codesign.baseline --size 8 --rows 8
python -m codesign.baseline --size 8 --rows 1 --trace
```

共同設計的 Python 模型已驗證；PE RTL/testbench 已撰寫但尚未在 simulator 執行，完整 NPU 與 48 課延伸正文是後續里程碑。建議從第 04 課就交錯做 CD01，不必等 24 課全部讀完才接觸硬體。

## 學完能做什麼

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

## 課程路線

每課是一個完整短講，含推導、算例與自我檢查。建議每次讀 1–2 課，再做對應實驗；先用自己的話解釋觀察，再看答案。全系列可按 6 週、每週 4 課安排，實驗時間依程式熟悉度調整。

| 階段 | 課程 | 實驗 |
| --- | --- | --- |
| 一：文字與學習 | [01 下一個 token](lessons/01-next-token.md) · [02 訓練與推論](lessons/02-training-inference.md) · [03 Tokenizer](lessons/03-tokenization.md) · [04 向量與矩陣](lessons/04-vectors-matrices.md) | `tokens`、`training` |
| 二：上下文如何流動 | [05 Embedding 與位置](lessons/05-embeddings-position.md) · [06 Attention 直覺](lessons/06-attention-intuition.md) · [07 Q/K/V 計算](lessons/07-attention-math.md) · [08 因果遮罩與多頭](lessons/08-causality-heads.md) | `attention` |
| 三：從向量選出答案 | [09 Transformer block](lessons/09-transformer-block.md) · [10 LM head](lessons/10-lm-head.md) · [11 Sampling](lessons/11-sampling.md) · [12 Prefill](lessons/12-prefill.md) | `sampling`、`timing` |
| 四：生成與容量 | [13 KV cache](lessons/13-kv-cache.md) · [14 Decode 與串流](lessons/14-decode-stream.md) · [15 記憶體估算](lessons/15-memory-budget.md) · [16 效能指標](lessons/16-metrics.md) | `cache`、`generation`、`memory` |
| 五：加速的原理 | [17 算力與頻寬](lessons/17-compute-bandwidth.md) · [18 量化](lessons/18-quantization.md) · [19 FlashAttention](lessons/19-flashattention.md) · [20 PagedAttention](lessons/20-pagedattention.md) | `quantization`、`online` |
| 六：推論服務 | [21 Batching 與排程](lessons/21-batching-scheduling.md) · [22 Speculative decoding](lessons/22-speculative-decoding.md) · [23 服務架構](lessons/23-serving-architecture.md) · [24 畢業實驗](lessons/24-capstone.md) | `scheduling`、`speculation`、整合驗證 |

## 正確理解那張推論流程圖

Prefill 與 decode 是使用同一模型的兩個執行階段；兩者都執行 Transformer。首個輸出通常由 prefill 最後位置的 logits 產生。一般單 token decode 每步選出一個新 token，speculative decoding 則可能在一輪驗證中接受多個。

「第一個 token 慢」只能提示調查 TTFT 的各部分，不能直接確診 prefill；「streaming 慢」也可能受排隊、傳輸或 buffering 影響。Embedding 是學到的表示，並非把文字意義直接翻譯成一串人類已定義的座標。

## 教學範圍與實驗邊界

主線是 decoder-only、dense causal Transformer 的文字推論。`TinyDecoder` 是兩層、單 head、learned position、LayerNorm、ReLU FFN 的隨機權重模型；它示範真實運算關係，但不會生成有意義的自然語言。真正會更新權重的是獨立的 bigram 訓練實驗。

基礎課的多頭/GQA/RoPE、分頁管理與多 GPU 在正文說明；CPU 模擬沒有實作完整生產引擎。Online-softmax 與 speculative mass 實驗只驗證數學，不是 GPU kernel 或實測加速。GPU serving benchmark 放在第 24 課進階作業，需自行提供模型與服務。

基礎篇涵蓋原始圖中的主要概念與必要前置知識；共同設計篇逐步延伸自有 NPU。MoE、混合注意力與多模態不在目前首版範圍；小模型訓練與量化品質驗證已列入後續整合里程碑。

## 參考與維護

- [原始論文與官方文件](docs/references.md)
- [術語速查](docs/glossary.md)
- [本次驗證紀錄](docs/validation.md)

文中的 toy 數字是教學算例；真正量測需記錄模型、版本、精度與工作負載。課程沒有複製原始社群圖片，其內容以重新推導的文字、表格與示意圖呈現。
