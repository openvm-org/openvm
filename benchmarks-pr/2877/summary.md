| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/fibonacci-7dcf25ffdc724423b32366239a0525f47293a6d5.md) | 3,970 |  12,000,265 |  1,135 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/keccak-7dcf25ffdc724423b32366239a0525f47293a6d5.md) | 21,757 |  18,655,329 |  4,602 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/sha2_bench-7dcf25ffdc724423b32366239a0525f47293a6d5.md) | 9,665 |  14,793,960 |  1,842 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/regex-7dcf25ffdc724423b32366239a0525f47293a6d5.md) | 1,516 |  4,137,067 |  431 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/ecrecover-7dcf25ffdc724423b32366239a0525f47293a6d5.md) | 608 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/pairing-7dcf25ffdc724423b32366239a0525f47293a6d5.md) | 956 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2877/kitchen_sink-7dcf25ffdc724423b32366239a0525f47293a6d5.md) | 4,136 |  2,579,903 |  893 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7dcf25ffdc724423b32366239a0525f47293a6d5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27437609579)
