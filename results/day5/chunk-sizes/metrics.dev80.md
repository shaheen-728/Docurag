# Development chunk-size comparison

Questions: 80. Vector retrieval. Top k: 5. No reranking.

| Variant | Hit@5 | MRR@5 | Pooled Recall@5 | Precision@5 | Mean retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.000 | 0.981 | 0.998 | 0.410 | 11.93 |
| token256 | 1.000 | 0.972 | 0.998 | 0.408 | 12.24 |
| token512 | 1.000 | 0.959 | 0.998 | 0.375 | 11.41 |
| token1024 | 1.000 | 0.963 | 0.998 | 0.362 | 10.90 |

Scores are macro averages over all 80 development questions.
Labels are assistant-reviewed, not human-verified; partial answer support counts.
Pooled recall uses variant-specific top-5, generation-source, and exact-evidence pools.
It is not exhaustive corpus recall, and pools differ between chunking variants.
Precision counts relevant retrieved chunks; it does not measure answer correctness.
Latency includes database vector retrieval only; it excludes query embedding, HTTP,
generation, and reranking. One fixed-order pass cannot establish a latency winner.
This comparison does not evaluate generated answers or the held-out test set.
No model API calls are made by this reporting script.

Run SHA256: 3ef6abafbf457f2d3cc194691987d03f1068820e859c08b1a9e6ede62916dea5
Review SHA256: 999b9c3e7e8c70c34b28ce710556fbe6c42dff9666b53e4a92923bdcc1749caa
