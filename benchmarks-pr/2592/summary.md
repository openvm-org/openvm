| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-12441f001f11586b12844d5ecd5e9a329fc49f5a.md) | 3,842 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-12441f001f11586b12844d5ecd5e9a329fc49f5a.md) | 18,578 |  18,655,329 |  3,296 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-12441f001f11586b12844d5ecd5e9a329fc49f5a.md) | 1,431 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-12441f001f11586b12844d5ecd5e9a329fc49f5a.md) | 641 |  123,583 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-12441f001f11586b12844d5ecd5e9a329fc49f5a.md) | 899 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-12441f001f11586b12844d5ecd5e9a329fc49f5a.md) | 2,292 |  2,579,903 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/12441f001f11586b12844d5ecd5e9a329fc49f5a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23615684238)
