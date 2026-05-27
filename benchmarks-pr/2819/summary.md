| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2819/fibonacci-ae3b7c38ca9f0b4d0184225aee4e0e80727c5557.md) | 3,774 |  12,000,265 |  922 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2819/keccak-ae3b7c38ca9f0b4d0184225aee4e0e80727c5557.md) | 18,231 |  18,655,329 |  3,324 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2819/sha2_bench-ae3b7c38ca9f0b4d0184225aee4e0e80727c5557.md) | 9,896 |  14,793,960 |  1,445 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2819/regex-ae3b7c38ca9f0b4d0184225aee4e0e80727c5557.md) | 1,390 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2819/ecrecover-ae3b7c38ca9f0b4d0184225aee4e0e80727c5557.md) | 601 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2819/pairing-ae3b7c38ca9f0b4d0184225aee4e0e80727c5557.md) | 884 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2819/kitchen_sink-ae3b7c38ca9f0b4d0184225aee4e0e80727c5557.md) | 1,863 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ae3b7c38ca9f0b4d0184225aee4e0e80727c5557

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26516426969)
