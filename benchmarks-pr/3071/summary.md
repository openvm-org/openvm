| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3071/fibonacci-bdfbc5b0b45f474f154bc12effa5e6d16674639f.md) | 468 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3071/keccak-bdfbc5b0b45f474f154bc12effa5e6d16674639f.md) | 7,288 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3071/sha2_bench-bdfbc5b0b45f474f154bc12effa5e6d16674639f.md) | 4,753 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3071/regex-bdfbc5b0b45f474f154bc12effa5e6d16674639f.md) | 671 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3071/ecrecover-bdfbc5b0b45f474f154bc12effa5e6d16674639f.md) | 226 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3071/pairing-bdfbc5b0b45f474f154bc12effa5e6d16674639f.md) | 316 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3071/kitchen_sink-bdfbc5b0b45f474f154bc12effa5e6d16674639f.md) | 2,667 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bdfbc5b0b45f474f154bc12effa5e6d16674639f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30156653036)
