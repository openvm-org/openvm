| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-576e26b1adae56953e6b0e51d86c764ecfad60f3.md) | 3,809 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-576e26b1adae56953e6b0e51d86c764ecfad60f3.md) | 18,396 |  18,655,329 |  3,282 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-576e26b1adae56953e6b0e51d86c764ecfad60f3.md) | 1,417 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-576e26b1adae56953e6b0e51d86c764ecfad60f3.md) | 650 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-576e26b1adae56953e6b0e51d86c764ecfad60f3.md) | 900 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-576e26b1adae56953e6b0e51d86c764ecfad60f3.md) | 2,269 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/576e26b1adae56953e6b0e51d86c764ecfad60f3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23806787645)
