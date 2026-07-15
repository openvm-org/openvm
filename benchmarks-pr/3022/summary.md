| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3022/fibonacci-4b8e3c99bf8d492f2aa7449868cb2d235fda9257.md) | 417 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3022/keccak-4b8e3c99bf8d492f2aa7449868cb2d235fda9257.md) | 8,544 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3022/sha2_bench-4b8e3c99bf8d492f2aa7449868cb2d235fda9257.md) | 3,989 |  11,167,961 |  538 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3022/regex-4b8e3c99bf8d492f2aa7449868cb2d235fda9257.md) | 569 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3022/ecrecover-4b8e3c99bf8d492f2aa7449868cb2d235fda9257.md) | 218 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3022/pairing-4b8e3c99bf8d492f2aa7449868cb2d235fda9257.md) | 267 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3022/kitchen_sink-4b8e3c99bf8d492f2aa7449868cb2d235fda9257.md) | 1,891 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4b8e3c99bf8d492f2aa7449868cb2d235fda9257

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29423426970)
