# 驗證紀錄

驗證日期：2026-09-27。環境：Linux、Python 3.12.14、NumPy 2.3.5。README 所列 Python 3.10+ 是相容性目標；本次沒有逐一測試所有 Python/NumPy 組合或 Windows/macOS。

## 已實際執行

```bash
python3 -m labs.run all
python3 -m unittest discover -s tests -v
```

12 個實驗皆完成，12 個單元測試皆通過。24 課 Markdown 的本機相對連結與 code fence 配對已檢查。

| 檢查 | 本次結果 |
| --- | --- |
| Bigram 學習 | 交叉熵 1.3863 → 0.4805 |
| 完整 forward 與 KV cache | 最大 logits 絕對誤差 8.882×10⁻¹⁶ |
| 因果性 | 新增未來 token 不改變前綴 logits，容許浮點誤差 |
| Cache 分塊 | 測試序列的每個切分点皆與完整 forward 一致 |
| Top-p 門檻 | 保留跨過 threshold 的 token，分布正規化正確 |
| KV 算例 | 8192 tokens、32 層、8 KV heads、128 維、2 bytes → 1GiB |
| 排程模擬 | static 10 回合；continuous 7 回合；token 工作量守恆 |
| Online attention | 與 dense 結果最大差異 5.551×10⁻¹⁷；大 score 案例亦通過 |
| Speculative 校正 | 包含零 support 與 p=q 的案例，均還原目標分布 |

具體浮點差異可因平台而變動，測試以容差比較。CPU timing 數字只用來確認量測流程可執行，不列為 GPU 速度、模型服務 TTFT 或優化技術的效能證據。

## 尚未執行的進階工作

沒有下載/訓練完整 Transformer 語言模型、沒有實際 GPU kernel benchmark、沒有部署推論服務或驗證多卡通訊。第 24 課 C 部分提供的是實測作業，需另外指定自己的模型與端點。

## 2026-09-28：共同設計增量驗證

新增 `tests/test_codesign.py` 的 6 個 Python 測試；加上既有 12 個，共 **18 個測試全部通過**。新增檢查涵蓋 65,536 個 INT8 operand 配對、INT32 wrap、signed requant、算術右移與飽和、enable/reset、不同 shape 的 mesh 結果與獨立 INT64 matmul、有效 MAC 數守恆。

已執行 `python3 -m codesign.baseline --size 8 --rows 8`：得到 512 個有效 MAC、30 個教學模型 cycles、26.67% PE-cycle slot 利用率，整數輸出與 reference 相同。這不含 NanoNPU 整機控制、SRAM/UART 或任何實際頻率，不能當作它的效能數字。

`codesign/rtl/ws_pe.sv` 與 `tb_ws_pe.sv` 為新寫的 reference RTL/testbench。本環境沒有 Icarus Verilog、Verilator 或 Yosys，未執行 RTL compile/simulation/synthesis；不將 Python pass 宣稱為 RTL pass。NanoNPU 只完成 pinned source inspection。
