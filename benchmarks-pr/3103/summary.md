| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-6551ae413e514341bd9047560042395a5602eeaf.md) | 463 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-6551ae413e514341bd9047560042395a5602eeaf.md) | 7,349 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-6551ae413e514341bd9047560042395a5602eeaf.md) | 4,167 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-6551ae413e514341bd9047560042395a5602eeaf.md) | 676 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-6551ae413e514341bd9047560042395a5602eeaf.md) | 201 |  112,210 |  199 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-6551ae413e514341bd9047560042395a5602eeaf.md) | 238 |  592,827 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-6551ae413e514341bd9047560042395a5602eeaf.md) | 2,037 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6551ae413e514341bd9047560042395a5602eeaf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31837126147)
