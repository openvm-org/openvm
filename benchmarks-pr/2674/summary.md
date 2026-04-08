| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/fibonacci-f1c4787a5060cdc8ed34a6fda3f4a3851bc728e6.md) | 3,825 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/keccak-f1c4787a5060cdc8ed34a6fda3f4a3851bc728e6.md) | 18,536 |  18,655,329 |  3,332 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/regex-f1c4787a5060cdc8ed34a6fda3f4a3851bc728e6.md) | 1,426 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/ecrecover-f1c4787a5060cdc8ed34a6fda3f4a3851bc728e6.md) | 671 |  123,583 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/pairing-f1c4787a5060cdc8ed34a6fda3f4a3851bc728e6.md) | 905 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/kitchen_sink-f1c4787a5060cdc8ed34a6fda3f4a3851bc728e6.md) | 2,162 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f1c4787a5060cdc8ed34a6fda3f4a3851bc728e6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24158029668)
