| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-72f35976703272e4d404f7e5ec40d0036b6cc6c1.md) | 3,819 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-72f35976703272e4d404f7e5ec40d0036b6cc6c1.md) | 18,780 |  18,655,329 |  3,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-72f35976703272e4d404f7e5ec40d0036b6cc6c1.md) | 1,446 |  4,137,067 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-72f35976703272e4d404f7e5ec40d0036b6cc6c1.md) | 647 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-72f35976703272e4d404f7e5ec40d0036b6cc6c1.md) | 908 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-72f35976703272e4d404f7e5ec40d0036b6cc6c1.md) | 2,282 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/72f35976703272e4d404f7e5ec40d0036b6cc6c1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23849603287)
