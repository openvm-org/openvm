| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/fibonacci-55e16837b0165e53bc9345a9cbb5be8522e143fd.md) | 3,971 |  12,000,265 |  1,144 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/keccak-55e16837b0165e53bc9345a9cbb5be8522e143fd.md) | 21,608 |  18,655,329 |  4,577 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/sha2_bench-55e16837b0165e53bc9345a9cbb5be8522e143fd.md) | 9,669 |  14,793,960 |  1,849 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/regex-55e16837b0165e53bc9345a9cbb5be8522e143fd.md) | 1,536 |  4,137,067 |  435 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/ecrecover-55e16837b0165e53bc9345a9cbb5be8522e143fd.md) | 599 |  123,583 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/pairing-55e16837b0165e53bc9345a9cbb5be8522e143fd.md) | 938 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/kitchen_sink-55e16837b0165e53bc9345a9cbb5be8522e143fd.md) | 4,155 |  2,579,903 |  888 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/55e16837b0165e53bc9345a9cbb5be8522e143fd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27365425992)
