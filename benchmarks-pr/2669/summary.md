| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/fibonacci-792698873c937200750012b8a65bf6f8ccbb2ca0.md) | 3,807 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/keccak-792698873c937200750012b8a65bf6f8ccbb2ca0.md) | 18,520 |  18,655,329 |  3,317 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/regex-792698873c937200750012b8a65bf6f8ccbb2ca0.md) | 1,418 |  4,137,067 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/ecrecover-792698873c937200750012b8a65bf6f8ccbb2ca0.md) | 645 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/pairing-792698873c937200750012b8a65bf6f8ccbb2ca0.md) | 899 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/kitchen_sink-792698873c937200750012b8a65bf6f8ccbb2ca0.md) | 2,159 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/792698873c937200750012b8a65bf6f8ccbb2ca0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24107590668)
