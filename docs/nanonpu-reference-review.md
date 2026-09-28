# NanoNPU 參考實作審查

日期：2026-09-28。參考 repo：[rightson/nanoNPU](https://github.com/rightson/nanoNPU)，branch `dev`，固定 commit `f876e8e236aed42e3ca4627ca8e43f8d44f1a086`。檔案 blob pins 見 [reference manifest](../codesign/references/nanonpu.json)。

**用途：** 借鑑可讀取的 RTL、數值處理與整合流程，為 llm-lab 的獨立設計提供比較證據。不是把該 repo 的 README 視為我們的規格。這次做的是 source review，沒有跑其整機 simulation 或重新執行後端。

## 實際讀到什麼

| 來源 | Source review 結果 | 對我們的意義 |
| --- | --- | --- |
| `RTL/Systolic Array/PE.sv` | weight-stationary；signed INT8 multiply；32-bit partial sum；activation 向右、psum 向下 | 參考 signedness 與波前；自己建立 bit/cycle contract |
| `RTL/Systolic Array/SA_NxN_top.sv` | top 預設 N_SIZE=8；包含 skew、array、de-skew 與 CU | 可拆解控制與 datapath，但先驗證輸入/輸出的實際 cycle |
| `RTL/npu_top.sv` | SA_SIZE=8；SRAM 註解稱 256×32，但實例為 RAM128x32 | 註解與真正 instance 必須分開；不能直接抄容量表 |
| `Backend/openlane/RTL/npu_project_macro.sv` | SA_SIZE=4、SRAM_ADDR_W=6，傳入 system top | 後端 source set 不是 README 所述 8×8 組態 |
| `Backend/openlane/RTL/npu_top.sv` | 使用 RAM64x32，bias/req 以 valid 串接 | 這個後端預設是 4×4、64×32 data memory，即 256 bytes；不含其餘 buffers |
| `Backend/openlane/RTL/CU.SV` | CONV 等待 fused pipeline；ADD_BIAS、REQ 已標 deprecated 並立即完成 | 發指令順序不能混用不同版本 |
| 開發與後端的 `Req.sv` | 算術都將 b cast 為 signed，64-bit 乘積、算術右移、INT8 飽和；介面不同 | 文首 UINT32/INT65 註解不能取代 actual expression；對負 multiplier 加測試 |
| `Backend/openlane/config.json` | 包含 `.v/.sv/.SV`；CLOCK_PERIOD=50ns、PNR/Signoff SDC 與特定 ECO target | 可學流程，但舊 net 名稱的 ECO 不適用於新 RTL 合成結果 |

後端參數由實際 wrapper 往下傳，不應只看個別模組預設值推定最終 elaborated 配置；此處的結論限於已讀的這組來源與預設設定。

## 應納入課程的實際問題

1. **來源一致性。** 先鎖定一份 file manifest、top、參數、時脈與數值契約，再跑 simulation；不能以兩套 RTL 各自有同名模組為理由隨意混合。
2. **算力單位。** 若每 PE 每 cycle 做一個 MAC，8×8 峰值為 64 MAC/cycle；將一次乘法與一次加法各算一個 op 才是 128 ops/cycle。README 的擴展段落混用這個單位，不能沿用為跑分結論。
3. **停頓契約。** SA_CU 可在 valid_in=0 時停止計數，但 PE/skew 的資料移動需整體檢查。僅看到 counter hold 不能推論所有 pipeline 都正確停住；這是待 simulator 驗證的問題，不在此判定為已證實 bug。
4. **ISA 位元欄位。** README 圖表有欄位重疊；後端 decode 實際為 opcode[31:26]、buf_sel[25:22]、transpose[6]、shift[5:1]、bypass[0]。正式 compiler 要以受測契約為準。我們的初版 ISA 另行設計。
5. **Softmax。** 向量正規化需要跨元素的 reduction；純量 exp LUT 只是其中一部分。不能因有 activation LUT 的建議，就視為具備 LLM attention 支援。
6. **硬體證據層級。** README 同時有 silicon-proven 字樣與預期 2026-11 fabrication 敘述；本次未取得實際矽量測，不將這些用語轉寫成我們已證實的 silicon 結果。

## 可以借鑑與不預先繼承的部分

借鑑 PE 分工、tile buffer、重定量的 bit semantics、分層 testbench、host-to-core 整合及 synthesis/STA 流程。是否採用 weight-stationary、operator fusion、特定陣列大小或介面，則由本專案的 LLM workload 與測量决定。

本次沒有修改 NanoNPU fork，也沒有 vendor 其 RTL。所有後续設計與教材在 llm-lab 主線進行。
