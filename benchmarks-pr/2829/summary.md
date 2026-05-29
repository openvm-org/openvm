| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/fibonacci-d045914f1812e57a62fb25209cdc24fb22f377a1.md) | 3,689 |  12,000,265 |  906 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/keccak-d045914f1812e57a62fb25209cdc24fb22f377a1.md) | 17,863 |  18,655,329 |  3,261 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/sha2_bench-d045914f1812e57a62fb25209cdc24fb22f377a1.md) | 10,006 |  14,793,960 |  1,457 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/regex-d045914f1812e57a62fb25209cdc24fb22f377a1.md) | 1,420 |  4,137,067 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/ecrecover-d045914f1812e57a62fb25209cdc24fb22f377a1.md) | 603 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/pairing-d045914f1812e57a62fb25209cdc24fb22f377a1.md) | 885 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/kitchen_sink-d045914f1812e57a62fb25209cdc24fb22f377a1.md) | 3,869 |  2,579,903 |  955 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d045914f1812e57a62fb25209cdc24fb22f377a1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26667347295)
