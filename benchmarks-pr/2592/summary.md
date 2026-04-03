| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-51776902cf5167ec6e6d42ca699ee8bdff99651e.md) | 3,785 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-51776902cf5167ec6e6d42ca699ee8bdff99651e.md) | 18,598 |  18,655,329 |  3,330 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-51776902cf5167ec6e6d42ca699ee8bdff99651e.md) | 1,416 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-51776902cf5167ec6e6d42ca699ee8bdff99651e.md) | 652 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-51776902cf5167ec6e6d42ca699ee8bdff99651e.md) | 912 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-51776902cf5167ec6e6d42ca699ee8bdff99651e.md) | 2,290 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/51776902cf5167ec6e6d42ca699ee8bdff99651e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23957228663)
