| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/fibonacci-97530bb613b6dfd97fb4d83ee630bf540c29d19f.md) | 2,985 |  12,000,265 |  675 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/keccak-97530bb613b6dfd97fb4d83ee630bf540c29d19f.md) | 16,595 |  18,655,329 |  3,076 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/sha2_bench-97530bb613b6dfd97fb4d83ee630bf540c29d19f.md) | 9,403 |  14,793,960 |  1,123 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/regex-97530bb613b6dfd97fb4d83ee630bf540c29d19f.md) | 1,205 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/ecrecover-97530bb613b6dfd97fb4d83ee630bf540c29d19f.md) | 508 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/pairing-97530bb613b6dfd97fb4d83ee630bf540c29d19f.md) | 852 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/kitchen_sink-97530bb613b6dfd97fb4d83ee630bf540c29d19f.md) | 4,471 |  2,579,903 |  875 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/97530bb613b6dfd97fb4d83ee630bf540c29d19f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29064630109)
