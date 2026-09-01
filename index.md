| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-fbeabbf544d6120d51002ea5996ddd6b8bc1979e.md) | 1,674 |  12,000,265 |  369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-fbeabbf544d6120d51002ea5996ddd6b8bc1979e.md) | 9,595 |  18,655,329 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-fbeabbf544d6120d51002ea5996ddd6b8bc1979e.md) | 5,256 |  14,793,960 |  590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-fbeabbf544d6120d51002ea5996ddd6b8bc1979e.md) | 694 |  4,137,067 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-fbeabbf544d6120d51002ea5996ddd6b8bc1979e.md) | 440 |  123,583 |  195 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-fbeabbf544d6120d51002ea5996ddd6b8bc1979e.md) | 587 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-fbeabbf544d6120d51002ea5996ddd6b8bc1979e.md) | 2,301 |  2,579,903 |  500 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fbeabbf544d6120d51002ea5996ddd6b8bc1979e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33541035366)
