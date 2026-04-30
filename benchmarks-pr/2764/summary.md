| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/fibonacci-526b6894cdfc68a4a842406124432d6dfa4854a5.md) | 3,795 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/keccak-526b6894cdfc68a4a842406124432d6dfa4854a5.md) | 18,870 |  18,655,329 |  3,329 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/sha2_bench-526b6894cdfc68a4a842406124432d6dfa4854a5.md) | 9,034 |  14,793,960 |  1,387 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/regex-526b6894cdfc68a4a842406124432d6dfa4854a5.md) | 1,414 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/ecrecover-526b6894cdfc68a4a842406124432d6dfa4854a5.md) | 639 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/pairing-526b6894cdfc68a4a842406124432d6dfa4854a5.md) | 923 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/kitchen_sink-526b6894cdfc68a4a842406124432d6dfa4854a5.md) | 2,050 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/526b6894cdfc68a4a842406124432d6dfa4854a5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25185064470)
