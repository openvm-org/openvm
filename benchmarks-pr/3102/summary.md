| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-bac02fa2f87e01e6364298d2686e3d886092aac3.md) | 468 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-bac02fa2f87e01e6364298d2686e3d886092aac3.md) | 7,352 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-bac02fa2f87e01e6364298d2686e3d886092aac3.md) | 4,132 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-bac02fa2f87e01e6364298d2686e3d886092aac3.md) | 645 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-bac02fa2f87e01e6364298d2686e3d886092aac3.md) | 221 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-bac02fa2f87e01e6364298d2686e3d886092aac3.md) | 236 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-bac02fa2f87e01e6364298d2686e3d886092aac3.md) | 2,016 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bac02fa2f87e01e6364298d2686e3d886092aac3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31087335444)
