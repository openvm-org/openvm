| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2880/fibonacci-48af237c85845868b958662a0ab218d606ff942b.md) | 3,961 |  12,000,265 |  1,141 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2880/keccak-48af237c85845868b958662a0ab218d606ff942b.md) | 22,001 |  18,655,329 |  4,675 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2880/sha2_bench-48af237c85845868b958662a0ab218d606ff942b.md) | 9,626 |  14,793,960 |  1,849 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2880/regex-48af237c85845868b958662a0ab218d606ff942b.md) | 1,500 |  4,137,067 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2880/ecrecover-48af237c85845868b958662a0ab218d606ff942b.md) | 604 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2880/pairing-48af237c85845868b958662a0ab218d606ff942b.md) | 957 |  1,745,757 |  312 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2880/kitchen_sink-48af237c85845868b958662a0ab218d606ff942b.md) | 4,150 |  2,579,903 |  882 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/48af237c85845868b958662a0ab218d606ff942b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27417841007)
