| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/fibonacci-3120e056221cba0b34293ae19dcb100453dad569.md) | 3,086 |  12,000,265 |  676 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/keccak-3120e056221cba0b34293ae19dcb100453dad569.md) | 16,298 |  18,655,329 |  3,017 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/sha2_bench-3120e056221cba0b34293ae19dcb100453dad569.md) | 9,097 |  14,793,960 |  1,110 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/regex-3120e056221cba0b34293ae19dcb100453dad569.md) | 1,166 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/ecrecover-3120e056221cba0b34293ae19dcb100453dad569.md) | 605 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/pairing-3120e056221cba0b34293ae19dcb100453dad569.md) | 937 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/kitchen_sink-3120e056221cba0b34293ae19dcb100453dad569.md) | 4,104 |  2,579,903 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3120e056221cba0b34293ae19dcb100453dad569

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27845356748)
