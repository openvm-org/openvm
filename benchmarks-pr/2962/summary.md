| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2962/fibonacci-88387996b31b9fd04f169f8751e250174dbd405e.md) | 3,150 |  12,000,265 |  698 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2962/keccak-88387996b31b9fd04f169f8751e250174dbd405e.md) | 16,727 |  18,655,329 |  3,106 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2962/sha2_bench-88387996b31b9fd04f169f8751e250174dbd405e.md) | 9,070 |  14,793,960 |  1,113 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2962/regex-88387996b31b9fd04f169f8751e250174dbd405e.md) | 1,160 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2962/ecrecover-88387996b31b9fd04f169f8751e250174dbd405e.md) | 602 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2962/pairing-88387996b31b9fd04f169f8751e250174dbd405e.md) | 954 |  1,745,757 |  310 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2962/kitchen_sink-88387996b31b9fd04f169f8751e250174dbd405e.md) | 4,134 |  2,579,903 |  886 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/88387996b31b9fd04f169f8751e250174dbd405e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28559133212)
