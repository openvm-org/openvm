| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/fibonacci-68e9f08d2e05f136cd5970cf0c3b96862e27aea7.md) | 3,782 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/keccak-68e9f08d2e05f136cd5970cf0c3b96862e27aea7.md) | 18,525 |  18,655,329 |  3,327 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/regex-68e9f08d2e05f136cd5970cf0c3b96862e27aea7.md) | 1,419 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/ecrecover-68e9f08d2e05f136cd5970cf0c3b96862e27aea7.md) | 644 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/pairing-68e9f08d2e05f136cd5970cf0c3b96862e27aea7.md) | 905 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2690/kitchen_sink-68e9f08d2e05f136cd5970cf0c3b96862e27aea7.md) | 2,200 |  2,579,903 |  444 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/68e9f08d2e05f136cd5970cf0c3b96862e27aea7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24254754195)
