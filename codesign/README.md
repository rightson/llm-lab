# CD01｜同一個矩陣乘法，從數字到時脈

這個 lab 是 llm-lab 自己的 NPU 設計起點。目的是把一個可手算的 GEMM，變成具有明確整數語意、傳輸方向與延遲的設計，再問：這個設計適合什麼 shape？

NanoNPU 只提供參考，無須下載它即可執行本 lab。`baseline.py` 不發送其指令，不是它的 cycle-accurate emulator；`rtl/ws_pe.sv` 是本專案的初始 PE，沒有實作完整 NPU。

## 1. 為什麼乘加還要定義位寬？

Signed INT8 範圍是 −128 到 127。−128×−128=16384，需要超過 8 bits；把結果放回 INT8 會改變答案。我們先用 16-bit 乘積、32-bit partial sum，PE 的輸出加法按二補數 32 bits wrap。

因此 `(2^31−1)+1` 的 register 結果是 `−2^31`。這不是浮點近似，而是明確的整數規格。模型/編譯器可以另外證明工作負載不會溢位，但 testbench 仍要檢查溢位發生時的行為。

輸出重定量另有規格：INT32 accumulator × signed INT32 multiplier，以 INT64 表示，再 arithmetic right shift，最後飽和到 [−128,127]。`−3 >> 1 = −2`，不是向零截斷的 −1；浮點 reference 必須使用相同規則才能 bit-exact。

跨 K tile 的 partial sum 應先累加再重定量。例如兩個 partial sum 都是 1、shift=1：各自右移再加得到 0；先加成 2 再右移得到 1。這就是一個演算法順序會影響硬體正確性的例子。

## 2. 一個 PE 的契約

| 條件 | 時脈邊緣後的狀態 |
| --- | --- |
| reset | weight、activation、psum 全部清零 |
| enable=0 | 所有 register 保持 |
| enable=1 且 load_weight | 更新 weight；保持 activation/psum |
| enable=1 且 compute | 註冊 activation；輸出 incoming_psum + activation×stored_weight |

這裡加上的 enable 是我們自己的介面選擇。未來如果要支援 backpressure，還必須讓 skew registers、控制計數、valid 與 buffer 一起遵循契約；只停一個 PE 不會自動形成正確的整個陣列。

## 3. 2×2 算例：用時間對齊 reduction

令 A=[[1,2],[3,4]]，W=[[5,6],[7,8]]，AW=[[19,22],[43,50]]。PE 的 row 對應 reduction index k，column 對應輸出 n；權重預先留在 PE，activation 向右、partial sum 向下。

對每一個輸入列 m，第 r 個 reduction lane 在 t=m+r 注入 A[m,r]。右移到 column j 還需 j 個 clocks；因此它能在 t=m+r+j 遇到來自上一個 reduction lane 的 partial sum。

| Feed cycle t | row 0 左邊輸入 | row 1 左邊輸入 | bottom outputs 中有效者 |
| --- | ---: | ---: | --- |
| 0 | 1 | 0（bubble） | 無 |
| 1 | 3 | 2 | C[0,0]=19 |
| 2 | 0（bubble） | 4 | C[1,0]=43，C[0,1]=22 |
| 3 | 0（bubble） | 0（bubble） | C[1,1]=50 |

這張表讓你看到「陣列有四個 PE」和「每個 cycle 都完成四個有用 MAC」是不同的事。開始要填充，結束要排空；不同 column 的輸出也不是同一拍到齊。本模型用收集器記錄結果位置，尚未實作 RTL de-skew buffer。

## 4. 執行與修改

從 repo 根目錄執行：

```bash
python -m codesign.baseline --size 8 --rows 8
python -m codesign.baseline --size 8 --rows 1 --trace
python -m codesign.baseline --size 8 --rows 64
python -m unittest discover -s tests -v
```

模型將 S×S 權重載入計為 S cycles，接著 feed/drain 計 M+2S−2 cycles。總計 M+3S−2。程式用逐 cycle 的 register 更新計算輸出，不是只拿矩陣乘法乘上一個估計時間；最後才用 NumPy INT64 matmul 獨立檢查數值。

| S=8，K=N=8 | 有效 MAC | 本模型總 cycles | 有效 MAC / 全部 PE-cycle slots |
| --- | ---: | ---: | ---: |
| M=1 | 64 | 23 | 約 4.35% |
| M=8 | 512 | 30 | 約 26.67% |
| M=64 | 4096 | 86 | 約 74.42% |

這些數字**只屬於本 lab 的 resident-weight 排程**。未含 SRAM loading、bus、指令取出、bias/requant、stall 或實體頻率；不能解讀成 NanoNPU 實測，也不能直接推估完整 LLM tokens/s。M=64 連續串流亦是本模型的能力，不宣稱 NanoNPU 固定 tile CU 支援同一協定。

可重現檢查包括：全部 65,536 個 INT8 乘法組合、wrap/shift/saturation、enable/reset、不同 shape 的波前與 matmul 一致，以及有效 MAC 數守恆。

## 5. 自己的 PE RTL

`rtl/ws_pe.sv` 對應上述 PE 數值與 enable 契約；`rtl/tb_ws_pe.sv` 準備了完整 INT8 乘法配對與 overflow/hold/reset 檢查。有 Icarus Verilog 時可從 repo 根目錄執行：

```bash
mkdir -p /tmp/llm-lab-rtl
iverilog -g2012 -s tb_ws_pe -o /tmp/llm-lab-rtl/pe.vvp codesign/rtl/ws_pe.sv codesign/rtl/tb_ws_pe.sv
vvp /tmp/llm-lab-rtl/pe.vvp
```

本次環境沒有 Icarus/Verilator，**尚未編譯或執行這份 RTL testbench**。Python 測試通過只代表 Python 數值與教學時序檢查成立。後續 M1 的首個 gate 是 RTL compile/simulation，以及與同一份 vectors 的差分比對。

## 6. 從觀察形成下一個設計問題

M 很小時，增加 PE 數量是否划算？要提高 decode 的利用率，可以比較跨請求 batching、不同資料流、GEMV 路徑或較小陣列，而不急著宣布哪個最好。權重載入能不能跨 tile 重用，又會改變 SRAM 壽命與指令排程。

本 lab 的下一步是對同一組 shape 實作 output-stationary 比較，加入跨 K tile accumulator 與 SRAM 流量，再決定我們初版 tile engine 的方案。完整路線見 [共同設計 roadmap](../docs/codesign-roadmap.md)。
