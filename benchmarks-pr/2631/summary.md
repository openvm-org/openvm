| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/fibonacci-d100f0a0d911ff381c339d247862ba35346c6850.md) | 3,836 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/keccak-d100f0a0d911ff381c339d247862ba35346c6850.md) | 18,467 |  18,655,329 |  3,293 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/regex-d100f0a0d911ff381c339d247862ba35346c6850.md) | 1,423 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/ecrecover-d100f0a0d911ff381c339d247862ba35346c6850.md) | 654 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/pairing-d100f0a0d911ff381c339d247862ba35346c6850.md) | 907 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2631/kitchen_sink-d100f0a0d911ff381c339d247862ba35346c6850.md) | 2,266 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d100f0a0d911ff381c339d247862ba35346c6850

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23804433449)
