# 共同設計課程路線：由原理一直追到 RTL 與實體限制

**狀態：01–24 課已完成；下列 25–72 為延伸課程規劃，尚未全部撰寫或實作。** 第一個跨層實驗 CD01 已放在 [codesign](../codesign/README.md)，可先把數學、整數與時序連起來。

原 24 課保留作為 LLM 基礎。後續採交錯學習，不必讀完所有軟體課才碰硬體。例如第 04 課矩陣乘法接 CD01 的小型 PE/mesh，再回到第 06–10 課看同一個 GEMM 出現在 QKV、FFN 與 head 的哪裡。

## 編號與學習順序

| 階段 | 課號與逐課主題 | 階段驗收成果 |
| --- | --- | --- |
| 數字到電路 | 25 二補數與溢位；26 scale、zero point 與 rounding；27 multiplier/adder 位寬；28 FF、組合邏輯與 pipeline；29 PE、reset 與 enable；30 golden model 與 RTL 差分驗證 | 全 INT8 operand 組合、overflow、reset/stall 都有可重現檢查 |
| 矩陣到陣列 | 31 GEMM loop nest；32 weight-stationary；33 output-stationary；34 skew/de-skew 波前；35 padding、K tiling 與 accumulator lifetime；36 array size 與 prefill/decode shape | 至少兩種資料流，在相同 workload 下比較 cycles、有效 MAC 與 bytes |
| 陣列到記憶體 | 37 tensor layout 與 stride；38 SRAM banking/port conflict；39 scratchpad 與 liveness；40 DMA/burst/backpressure；41 double buffering 與相依；42 traffic model 與 roofline | 帶有頻寬/port 限制的排程，能解釋 stall 並保證不覆寫活資料 |
| 程式到硬體 | 43 算子 IR；44 tile lowering；45 instruction/descriptor 規格；46 assembler、地址配置與 relocation；47 hazard、completion 與 fence；48 host runtime 和 fallback | 同一份 matmul 規格自動產生可重現 schedule，bit-accurate model 與 RTL 比對 |
| Transformer 非線性 | 49 reduction/max/sum；50 exp LUT 或近似多項式；51 softmax normalization；52 LayerNorm/RMSNorm/rsqrt；53 RoPE、GELU、SwiGLU；54 fusion 的 liveness 與精度 | 定義每算子誤差範圍、極值測試與回到模型後的品質影響 |
| Attention 與生成 | 55 QKV/GQA layout；56 online-softmax/block attention；57 KV append/read/容量；58 prefill 排程；59 decode GEMV 與權重重用；60 mixed precision/INT4 與品質 | 小型 attention block 功能對齊；prefill 與 decode 分別有成本模型 |
| RTL 到實體 | 61 synthesis 與面積來源；62 SDC、setup/hold 與 pipeline；63 SRAM macro 與 floorplan；64 placement/CTS/routing；65 activity、clock gating 與功耗估計；66 PPA 設計空間搜尋 | 固定 library/corner/constraints，重現至少兩個方案並區分估計與量測 |
| 模型到端到端系統 | 67 小型模型訓練與評估資料；68 PTQ/QAT 與精度分配；69 整圖編譯與 host 分工；70 UART/APB/DMA 等傳輸選擇；71 regression/performance counters；72 端到端生成與架構報告 | 能生成可評估的文本，列出品質、TTFT、TPOT、bytes/token、area/timing 與未完成限制 |

## 建議交錯路線

| 已讀的基礎 | 立即串入的硬體問題 | 可以先做的事 |
| --- | --- | --- |
| 02 訓練、04 矩陣 | 一個乘加如何保存數值？ | 25–30、CD01 PE |
| 06–10 Attention/FFN/head | 同樣 GEMM 如何在陣列流動？ | 31–36、CD01 mesh trace |
| 12–15 Prefill/KV/容量 | 資料放哪裡、每步要讀多少？ | 37–42、55、57–59 |
| 18 量化 | bits 降低以後，誰負責 scale？ | 26–27、47、60、68 |
| 19 Online attention | 運算順序改了，SRAM 和誤差如何變？ | 49–56 |
| 21–23 系統 | CPU/NPU、DMA 與 compiler 如何合作？ | 43–48、69–71 |

這樣每輪都會經歷「公式 → 數值 → 記憶體 → 硬體 → 證據」，而非先背完軟體名詞再背一套 RTL 名詞。

## 分階段里程碑

- M0，已建立：獨立主線、reference audit、INT8 PE 數值模型、mesh 教學時序模型、原創 PE RTL/testbench。Python 已驗證；RTL 執行尚待工具環境。
- M1：完整 GEMM tile engine 與 RTL regression，包括停頓、尾塊和跨 K tile 累加；決定初版資料流。
- M2：受 SRAM/介面約束的 command engine、compiler lowering 與 host runtime。
- M3：Transformer linear ops 可卸載，非線性明確由 host 執行，量測傳輸成本。
- M4：逐步加入 reduction/nonlinear/attention/KV；在端到端品質與成本下決定融合邊界。
- M5：硬體/模型設計空間探索；FPGA 或選定公開製程流程的實作證據。

M1–M5 是後續工作，不因有 roadmap 或參考專案報告就視為完成。不預先承諾所有算子都要硬化；若某項硬化的成本大於收益，保留 host 路徑也是可接受的設計結論。

## 技術縱深的標準

每篇從一個可觀察問題開始，連接到一個正式規格與實驗。必須至少有一個手算例、一張有資訊量的資料流/時序圖或表、一個失敗案例，以及一項可以重跑的檢查。資料流方案要比較至少一個替代方案，不只展示最順利的結果。

例如「Softmax 硬體化」不以 256-entry LUT 收尾；要涵蓋 max reduction、exp 動態範圍、sum 位寬、倒數/除法、跨 tile 累積與 normalization 誤差。LUT 最多替代其中的純量函數，不能單獨完成向量 softmax。

例如「8×8 擴成 16×16」必須同時討論供料帶寬、有效 lane、填充/排空、長線與 timing；PE 數量四倍不代表端到端速度四倍。
