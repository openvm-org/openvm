| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-16724dfd3d3c06eb303df78619b14b66af8e265a.md) | 3,789 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-16724dfd3d3c06eb303df78619b14b66af8e265a.md) | 18,365 |  18,655,329 |  3,280 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-16724dfd3d3c06eb303df78619b14b66af8e265a.md) | 1,423 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-16724dfd3d3c06eb303df78619b14b66af8e265a.md) | 658 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-16724dfd3d3c06eb303df78619b14b66af8e265a.md) | 911 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-16724dfd3d3c06eb303df78619b14b66af8e265a.md) | 2,303 |  2,579,903 |  447 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/16724dfd3d3c06eb303df78619b14b66af8e265a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24100869525)
