# VLM Document Understanding

So I'll be working on this over the weekend. The goal is to take a small VLM and improve its score on document understanding benchmarks.

Part of a series of weekend projects where I try to build something (anything) over a weekend.

---

## Thoughts

Started with LFM2.5-VL-1.6B for DocVQA. Tried VLMEvalKit first — model supported but no batching, slow. Tried lmms-eval — has batching but no support for our model. Ended up implementing standalone batched eval scripts.

Baseline results:

| Benchmark | Split | Samples | Metric |
|---|---:|---:|---:|
| DocVQA | validation | 5,349 | ANLS 0.8829 |
| SlideVQA | validation | 1,652 | ANLS 0.4537 / EM 0.3130 / F1 0.3856 |
| SlideVQA, evidence pages only | validation | 1,652 | ANLS 0.6487 / EM 0.4691 / F1 0.5434 |

DocVQA is mostly solved by the base model, so SlideVQA is the more useful benchmark now. It exposes harder failures: finding the right slide, multi-slide reasoning, arithmetic/numerical questions, and exact short-answer formatting.

The evidence-pages-only run feeds only the dataset's gold evidence slides instead of all 20 slides. That improves ANLS by about 0.20, so slide selection is a major bottleneck: the model answers much better once it is looking at the right slide(s).
