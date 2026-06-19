| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/fibonacci-247af08fd7a890b656bcdd1c912c9c2ef539f09c.md) | 3,051 |  12,000,265 |  672 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/keccak-247af08fd7a890b656bcdd1c912c9c2ef539f09c.md) | 16,259 |  18,655,329 |  3,033 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/sha2_bench-247af08fd7a890b656bcdd1c912c9c2ef539f09c.md) | 9,203 |  14,793,960 |  1,119 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/regex-247af08fd7a890b656bcdd1c912c9c2ef539f09c.md) | 1,151 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/ecrecover-247af08fd7a890b656bcdd1c912c9c2ef539f09c.md) | 598 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/pairing-247af08fd7a890b656bcdd1c912c9c2ef539f09c.md) | 931 |  1,745,757 |  302 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2910/kitchen_sink-247af08fd7a890b656bcdd1c912c9c2ef539f09c.md) | 4,081 |  2,579,903 |  873 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/247af08fd7a890b656bcdd1c912c9c2ef539f09c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27836329045)
