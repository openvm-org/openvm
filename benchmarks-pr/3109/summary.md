| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-560c16d38b55ec6257b238e4baacc879e8921818.md) | 483 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-560c16d38b55ec6257b238e4baacc879e8921818.md) | 7,334 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-560c16d38b55ec6257b238e4baacc879e8921818.md) | 4,182 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-560c16d38b55ec6257b238e4baacc879e8921818.md) | 665 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-560c16d38b55ec6257b238e4baacc879e8921818.md) | 222 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-560c16d38b55ec6257b238e4baacc879e8921818.md) | 231 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-560c16d38b55ec6257b238e4baacc879e8921818.md) | 2,033 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/560c16d38b55ec6257b238e4baacc879e8921818

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31545226208)
