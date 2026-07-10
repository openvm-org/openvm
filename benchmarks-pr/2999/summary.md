| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/fibonacci-0b7e3e30bc80f68c7b3c551f8780aee4ec147a51.md) | 3,029 |  12,000,265 |  679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/keccak-0b7e3e30bc80f68c7b3c551f8780aee4ec147a51.md) | 16,522 |  18,655,329 |  3,052 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/sha2_bench-0b7e3e30bc80f68c7b3c551f8780aee4ec147a51.md) | 9,523 |  14,793,960 |  1,136 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/regex-0b7e3e30bc80f68c7b3c551f8780aee4ec147a51.md) | 1,205 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/ecrecover-0b7e3e30bc80f68c7b3c551f8780aee4ec147a51.md) | 513 |  123,583 |  290 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/pairing-0b7e3e30bc80f68c7b3c551f8780aee4ec147a51.md) | 848 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2999/kitchen_sink-0b7e3e30bc80f68c7b3c551f8780aee4ec147a51.md) | 4,541 |  2,579,903 |  890 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0b7e3e30bc80f68c7b3c551f8780aee4ec147a51

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29069024126)
