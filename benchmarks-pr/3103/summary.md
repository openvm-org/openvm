| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-b145ccc3ad68785cea453ed4c6f6b99d50523d1f.md) | 470 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-b145ccc3ad68785cea453ed4c6f6b99d50523d1f.md) | 7,317 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-b145ccc3ad68785cea453ed4c6f6b99d50523d1f.md) | 4,155 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-b145ccc3ad68785cea453ed4c6f6b99d50523d1f.md) | 653 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-b145ccc3ad68785cea453ed4c6f6b99d50523d1f.md) | 222 |  112,210 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-b145ccc3ad68785cea453ed4c6f6b99d50523d1f.md) | 232 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-b145ccc3ad68785cea453ed4c6f6b99d50523d1f.md) | 2,020 |  1,979,971 |  524 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b145ccc3ad68785cea453ed4c6f6b99d50523d1f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31222349136)
