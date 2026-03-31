| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-e86e9654a2adcd1fa1781093eddc419ab4a22c0d.md) | 3,846 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-e86e9654a2adcd1fa1781093eddc419ab4a22c0d.md) | 18,458 |  18,655,329 |  3,306 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-e86e9654a2adcd1fa1781093eddc419ab4a22c0d.md) | 1,418 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-e86e9654a2adcd1fa1781093eddc419ab4a22c0d.md) | 651 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-e86e9654a2adcd1fa1781093eddc419ab4a22c0d.md) | 908 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-e86e9654a2adcd1fa1781093eddc419ab4a22c0d.md) | 2,308 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e86e9654a2adcd1fa1781093eddc419ab4a22c0d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23780803526)
