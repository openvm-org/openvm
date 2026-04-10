| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-b2dc1cf0904b8e138a86198120407084de773848.md) | 3,828 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-b2dc1cf0904b8e138a86198120407084de773848.md) | 18,495 |  18,655,329 |  3,312 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-b2dc1cf0904b8e138a86198120407084de773848.md) | 1,439 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-b2dc1cf0904b8e138a86198120407084de773848.md) | 642 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-b2dc1cf0904b8e138a86198120407084de773848.md) | 913 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-b2dc1cf0904b8e138a86198120407084de773848.md) | 2,099 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b2dc1cf0904b8e138a86198120407084de773848

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24261467114)
