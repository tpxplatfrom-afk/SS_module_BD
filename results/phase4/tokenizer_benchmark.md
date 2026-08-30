# SS Tutor BD — Phase 4 Bengali Tokenizer Benchmark

| Tokenizer | Vocabulary Size | Characters / Token | Tokens / Bengali Word | Efficiency Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Custom Bengali-16K (Phase 4)** | 1,073 | 1.35 | **3.86** | GOOD |
| **Qwen2.5-0.5B (152K Vocab)** | 151,643 | 0.99 | **5.28** | POOR |
| **SmolLM2-135M (49K Vocab)** | 49,152 | 0.62 | **8.46** | POOR |

**Key Takeaway:** The custom 16K Bengali tokenizer dramatically reduces token expansion compared to SmolLM2, fulfilling Gate 1 (<= 4.0 tokens/word).
