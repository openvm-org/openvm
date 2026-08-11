| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a.md) | 479 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a.md) | 7,500 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a.md) | 4,165 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a.md) | 668 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a.md) | 222 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a.md) | 231 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a.md) | 2,043 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f73fc6c1d323ad68e04d61e53fbd4ad77ef33e8a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31545939402)
