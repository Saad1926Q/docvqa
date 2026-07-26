# VLM Document Understanding

So I'll be working on this over the weekend. The goal is to take a small VLM and improve its score on document understanding benchmarks.

Part of a series of weekend projects where I try to build something (anything) over a weekend.

---

## Thoughts

Started with LFM2.5-VL-1.6B for DocVQA. Tried VLMEvalKit first — model supported but no batching, slow. Tried lmms-eval — has batching but no support for our model. Ended up implementing our own eval script with batched inference.

Ran a quick test on 5 samples. Clean outputs, all correct. Promising.

Now: run full eval. If the model is already too good on DocVQA, switch to harder document understanding benchmarks.
