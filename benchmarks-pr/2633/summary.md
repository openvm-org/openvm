| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2633/fibonacci-be59cd83e60b4825a773e610435fff180fcc510b.md) | 3,820 |  12,000,265 |  940 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2633/keccak-be59cd83e60b4825a773e610435fff180fcc510b.md) | 15,795 |  1,235,218 |  2,221 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2633/regex-be59cd83e60b4825a773e610435fff180fcc510b.md) | 1,407 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2633/ecrecover-be59cd83e60b4825a773e610435fff180fcc510b.md) | 638 |  122,348 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2633/pairing-be59cd83e60b4825a773e610435fff180fcc510b.md) | 923 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2633/kitchen_sink-be59cd83e60b4825a773e610435fff180fcc510b.md) | 2,397 |  154,763 |  422 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/be59cd83e60b4825a773e610435fff180fcc510b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23806708774)
