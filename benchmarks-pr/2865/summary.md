| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2865/fibonacci-f0177b304fd230cb96658dcdb3979198759e331a.md) | 3,978 |  12,000,265 |  1,158 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2865/keccak-f0177b304fd230cb96658dcdb3979198759e331a.md) | 21,608 |  18,655,329 |  4,582 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2865/sha2_bench-f0177b304fd230cb96658dcdb3979198759e331a.md) | 9,621 |  14,793,960 |  1,839 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2865/regex-f0177b304fd230cb96658dcdb3979198759e331a.md) | 1,493 |  4,137,067 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2865/ecrecover-f0177b304fd230cb96658dcdb3979198759e331a.md) | 605 |  123,583 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2865/pairing-f0177b304fd230cb96658dcdb3979198759e331a.md) | 937 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2865/kitchen_sink-f0177b304fd230cb96658dcdb3979198759e331a.md) | 4,119 |  2,579,903 |  884 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f0177b304fd230cb96658dcdb3979198759e331a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27241947652)
