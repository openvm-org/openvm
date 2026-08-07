| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-1aed5eba6657c9d388be3fc0e70236d61d2f5d1e.md) | 479 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-1aed5eba6657c9d388be3fc0e70236d61d2f5d1e.md) | 7,350 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-1aed5eba6657c9d388be3fc0e70236d61d2f5d1e.md) | 4,206 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-1aed5eba6657c9d388be3fc0e70236d61d2f5d1e.md) | 665 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-1aed5eba6657c9d388be3fc0e70236d61d2f5d1e.md) | 223 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-1aed5eba6657c9d388be3fc0e70236d61d2f5d1e.md) | 234 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-1aed5eba6657c9d388be3fc0e70236d61d2f5d1e.md) | 2,045 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1aed5eba6657c9d388be3fc0e70236d61d2f5d1e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31218342449)
