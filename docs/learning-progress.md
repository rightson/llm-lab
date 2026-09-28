# 學習計畫進度與下一輪交付

更新：2026-09-28。學習目標以 [完整規劃](learning-plan.md) 為準。此處刻意區分「已有短講／程式」與「已完成深度學習路線」；章節檔案存在不等於所有理解 gate 已通過。

## 已有資產

| 資產 | 狀態 | 適用與限制 |
| --- | --- | --- |
| lessons/01–24 | 已撰寫入門短講 | 可重用為主線材料；新增的訓練、消融與共同模型 trace 尚未補齊 |
| labs/core.py TinyDecoder | 已有隨機權重 forward/cache | 驗證計算機制；沒有已訓練 Transformer checkpoint |
| labs.run training | 已有可學習 bigram | 不能取代 Transformer backprop/training |
| 12 個基礎 CPU 實驗 | 已執行 | 有 sampling/cache/online-softmax 等機制實驗；不是完整模型能力評測 |
| CD01 integer/mesh | Python 模型已驗證 | 自有教學排程；不代表 NanoNPU 或完整 NPU 的實測時序 |
| codesign/rtl PE/testbench | 已撰寫，尚未編譯/模擬 | 沒有 Python→RTL co-simulation 成功紀錄 |
| 18 個 Python tests | 前次程式驗證全部通過 | 本次僅調整規劃與文件，未宣稱新增執行結果 |
| C01–C40 詳細路線 | 本次完成規劃 | 是有 gate 的學習地圖，不是 40 篇已完成新文章 |
| 48 項硬體深入路線 | 已規劃 | 供主線 H 階段展開；整機/compiler/STA/PPA 尚未完成 |

## 依依賴順序交付，先補模型學習

| 次序 | 下一份交付 | 要解決的缺口 | 完成條件 |
| --- | --- | --- | --- |
| P1 | 受控任務、資料生成與 baseline 套件 | 缺少共用且能驗證 context 需求的資料 | 固定 schema/splits/seeds，train/test 無重複範例洩漏；unigram/bigram 與平衡 chance baseline |
| P2 | Loss/gradient/backprop 教材及小網路 | 「權重如何學來」仍是黑盒 | 有限差分、手算與實作對照；可 overfit tiny batch |
| P3 | 固定視窗 MLP 與可訓練 attention | 缺少機制與任務的直接連結 | 同一資料與評估，比較視窗限制、配對干預與 held-out 表現 |
| P4 | 可訓練小型 Transformer 及 trace | 現有模型只有隨機權重 | 可重現 checkpoint、forward/loss trace、至少一組實質 ablation |
| P5 | 生成、cache、量化回接已訓練模型 | 機制測試尚未連到品質 | 同一 checkpoint 的等價性/誤差/任務指標；另有有授權的自然文字評估 |
| P6 | Tensor workload 與 cost report | 硬體設計缺少模型驅動的固定 workload | QKV/FFN/head shapes、bytes/MAC、prefill/decode 分別列出 |
| P7 | PE→GEMM RTL 與 Python adapter | 尚未把模型執行接到真正 RTL | simulator compile/test、獨立 vectors、stall/reset/tail 測試與輸出比對 |
| P8 | 小型模型共同執行 | 尚無端到端軟硬體驗證鏈 | 明列 offload/fallback、逐層比較、品質與 cycles/traffic 報告 |
| P9 | 更多硬體方案與物理分析 | 需要比較真實設計取捨 | 固定 workload/library/constraints，比較資料流、SRAM、精度；未測項目明列 |

這些是工作順序，不是已建立的排程或已發布內容。P7 的 PE simulator bring-up 可作獨立小任務提早進行，但不能以此取代 P1–P6；完整 NPU 架構選擇要等有模型 workload 證據。

## 三條閱讀路線

| 讀者當下的需求 | 路線 | 達成什麼 |
| --- | --- | --- |
| 先建立完整直覺 | C01–C30，按 A–F gates 前進 | 能解釋模型如何表示、學習、使用 context 與生成 |
| 想理解推論效率 | 上述路線＋C31–C35 | 能連接品質、容量、延遲、頻寬與執行順序 |
| 想自行建構 NPU | 上述路線＋C36–C40，按需展開硬體支線 | Python 模型與自有 RTL 共同驗證，進一步比較實作成本 |

三條路線共用資料、模型與語言，不另開一套與 LLM 無關的硬體主題。完整路線包含全部，但不要求新手第一次學一個機制就掌握全部層級。

## 計畫的範圍管理

新增課題先放入對應單元或附錄，不持續延長必修清單。每完成一個 gate，檢查是否能少用名詞、更精確解釋同一個模型；若不能，優先改善既有教材。以理解與可重現證據決定進度，無須為了湊篇數跳過實驗。
