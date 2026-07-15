| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-7f9ad2326358ed9f0fd29e29a4b475bacad5b334.md) | 554 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-7f9ad2326358ed9f0fd29e29a4b475bacad5b334.md) | 7,424 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-7f9ad2326358ed9f0fd29e29a4b475bacad5b334.md) | 4,499 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-7f9ad2326358ed9f0fd29e29a4b475bacad5b334.md) | 605 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-7f9ad2326358ed9f0fd29e29a4b475bacad5b334.md) | 223 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-7f9ad2326358ed9f0fd29e29a4b475bacad5b334.md) | 249 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-7f9ad2326358ed9f0fd29e29a4b475bacad5b334.md) | 2,743 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7f9ad2326358ed9f0fd29e29a4b475bacad5b334

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29392308500)
