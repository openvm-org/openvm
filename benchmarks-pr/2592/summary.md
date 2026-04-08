| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-8080f984ea520f526af6f5a10b7c864bf4bd08b9.md) | 3,789 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-8080f984ea520f526af6f5a10b7c864bf4bd08b9.md) | 18,374 |  18,655,329 |  3,297 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-8080f984ea520f526af6f5a10b7c864bf4bd08b9.md) | 1,428 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-8080f984ea520f526af6f5a10b7c864bf4bd08b9.md) | 646 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-8080f984ea520f526af6f5a10b7c864bf4bd08b9.md) | 902 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-8080f984ea520f526af6f5a10b7c864bf4bd08b9.md) | 2,159 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8080f984ea520f526af6f5a10b7c864bf4bd08b9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24146484344)
