# 原始論文與官方文件

以下是概念來源與進階閱讀，不是逐段翻譯。算例、課程順序與程式是本教材為教學重新設計；文中沒有沿用論文的效能數字作為本專案測量。

| 主題 | 一手資料 | 建議讀法 |
| --- | --- | --- |
| Attention、multi-head、FFN | [Vaswani et al., Attention Is All You Need, 2017](https://arxiv.org/abs/1706.03762) | 完成 07–09 課後讀 §3；區分原始 encoder-decoder 與本課 decoder-only |
| BPE | [Hugging Face LLM Course, BPE](https://huggingface.co/learn/llm-course/en/chapter6/5) | 完成 03 課後追蹤一次 merge；不要把教學切法當特定模型 token IDs |
| RoPE | [Su et al., RoFormer, 2021](https://arxiv.org/abs/2104.09864) | 完成 Q/K 後再看位置如何進入配對 |
| GQA | [Ainslie et al., GQA, 2023](https://arxiv.org/abs/2305.13245) | 用第 15 課公式重算不同 KV heads 的容量 |
| KV cache | [Transformers, How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation) | 對照第 13 課的 per-layer state；文件 main 會持續變動 |
| Quantization | [Lin et al., AWQ, 2023](https://arxiv.org/abs/2306.00978) | 先懂近似誤差，再看 activation-aware 設計 |
| FlashAttention | [Dao et al., FlashAttention, 2022](https://arxiv.org/abs/2205.14135) | 區分數學運算量與 HBM I/O |
| KV 配置 | [Kwon et al., PagedAttention, 2023](https://arxiv.org/abs/2309.06180) | 看 logical/physical blocks 與碎片問題 |
| Iteration-level scheduling | [Yu et al., Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | 比較等待整個 batch 與逐 iteration 排程 |
| Speculative decoding | [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) | 重算接受與拒絕校正，再看多 token 演算法 |
| Prefill/decode 分離 | [Zhong et al., DistServe, 2024](https://arxiv.org/abs/2401.09670) | 看 goodput 目標、干擾與傳輸成本 |

教材建立時已核對上述來源的題名與核心主張。這是一條基礎學習路線，沒有宣稱涵蓋 2026 年所有最新推論方法。
