| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-568d2e63f796c9847423b011eddb28ed77d00043.md) | 3,755 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-568d2e63f796c9847423b011eddb28ed77d00043.md) | 18,383 |  18,655,329 |  3,243 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-568d2e63f796c9847423b011eddb28ed77d00043.md) | 10,192 |  14,793,960 |  1,459 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-568d2e63f796c9847423b011eddb28ed77d00043.md) | 1,394 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-568d2e63f796c9847423b011eddb28ed77d00043.md) | 602 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-568d2e63f796c9847423b011eddb28ed77d00043.md) | 886 |  1,745,757 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-568d2e63f796c9847423b011eddb28ed77d00043.md) | 1,901 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/568d2e63f796c9847423b011eddb28ed77d00043

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26656562828)
