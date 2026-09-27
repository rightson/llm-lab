# 第一次執行 Python 實驗

先確認已安裝 Python 3.10 以上與 Git。在終端機執行 `python3 --version`；Windows 通常使用 `py --version`。若系統找不到指令，需先安裝這些工具。

終端機是輸入指令的地方；資料夾是檔案的位置；`cd` 用來切換資料夾。所有實驗指令都要在含有 README.md、requirements.txt、labs 的 repo 根目錄執行。

## macOS / Linux

依根目錄 README 的命令執行。`.venv` 是這個專案獨立使用的 Python 環境；啟用後，`python` 與 `pip` 使用此環境。每次開新終端機，都要先 `cd llm-lab`，再 `source .venv/bin/activate`。

## Windows PowerShell

```powershell
git clone https://github.com/rightson/llm-lab.git
cd llm-lab
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m labs.run tokens
.venv\Scripts\python.exe -m labs.run all
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

直接呼叫環境裡的 python.exe 即可，不必修改 PowerShell execution policy。

## 三個先看懂的 Python 概念

```python
x = [2, 1]          # list：有順序的資料，索引从 0 開始
print(x[0])         # 顯示 2
for value in x:     # 把每個元素依序交给 value
    print(value)
```

NumPy 的 array 支援矩陣運算。`x @ w` 表示矩陣乘法；`x * w` 通常是逐元素乘法，兩者不同。`shape` 是各維度的大小；`axis=-1` 是最後一個維度。`logits[-1]` 是最後一列，不是負機率。

```python
import numpy as np
x = np.array([[2., 1.]])
w = np.array([[1., 3., 0.], [2., -1., 4.]])
print(x.shape, w.shape)
print(x @ w)        # [[4. 5. 4.]]
```

`python -m labs.run attention` 中的 `-m` 表示把指定模組當作程式執行，`attention` 是實驗名稱。完整清單可執行 `python -m labs.run --help`。

## 常見問題

| 錯誤/現象 | 處理 |
| --- | --- |
| No module named numpy | 用同一個 Python 執行 `-m pip install -r requirements.txt` |
| No module named labs | 切到 repo 根目錄再執行 |
| 第一次沒有網路 | 需先讓環境取得 NumPy；實驗本身不連網 |
| generation 印出數字而非好文章 | 正常：TinyDecoder 沒有訓練語言能力 |
| timing 每次不同 | 正常：CPU 排程、BLAS 執行緒與暖機會影響結果 |

先跑 `tokens` 與 `attention`，確認你能預測一部分輸出，再進行其他實驗。不需要一開始就理解 `core.py` 的全部程式。
