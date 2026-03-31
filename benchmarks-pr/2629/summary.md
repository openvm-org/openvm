| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/fibonacci-7984fa182af5ef975bbd228293821af9fe41ca54.md) | 3,783 |  12,000,265 |  926 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/keccak-7984fa182af5ef975bbd228293821af9fe41ca54.md) | 18,548 |  18,655,329 |  3,326 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/regex-7984fa182af5ef975bbd228293821af9fe41ca54.md) | 1,417 |  4,137,067 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/ecrecover-7984fa182af5ef975bbd228293821af9fe41ca54.md) | 648 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/pairing-7984fa182af5ef975bbd228293821af9fe41ca54.md) | 901 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2629/kitchen_sink-7984fa182af5ef975bbd228293821af9fe41ca54.md) | 2,275 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7984fa182af5ef975bbd228293821af9fe41ca54

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23776865772)
