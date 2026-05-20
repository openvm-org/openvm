| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/fibonacci-0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb.md) | 3,765 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/keccak-0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb.md) | 18,609 |  18,655,329 |  3,274 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/sha2_bench-0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb.md) | 10,119 |  14,793,960 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/regex-0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb.md) | 1,403 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/ecrecover-0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb.md) | 600 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/pairing-0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb.md) | 885 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2795/kitchen_sink-0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb.md) | 1,903 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0ec03165baf1d206a5ba5e7e4d2a52799bd0c3cb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26171454863)
