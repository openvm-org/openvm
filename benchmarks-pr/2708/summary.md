| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/fibonacci-e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6.md) | 3,852 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/keccak-e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6.md) | 18,768 |  18,655,329 |  3,341 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/sha2_bench-e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6.md) | 10,031 |  14,793,960 |  1,421 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/regex-e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6.md) | 1,425 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/ecrecover-e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6.md) | 647 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/pairing-e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6.md) | 911 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2708/kitchen_sink-e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6.md) | 2,154 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e53f3086f8e11fcb4b54aa2ffcb579a33ab899e6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24467715061)
