| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-0cc0b0b1c120814d78371083e96a17115c7c1ba9.md) | 3,836 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-0cc0b0b1c120814d78371083e96a17115c7c1ba9.md) | 18,340 |  18,655,329 |  3,277 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/sha2_bench-0cc0b0b1c120814d78371083e96a17115c7c1ba9.md) | 9,857 |  14,793,960 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-0cc0b0b1c120814d78371083e96a17115c7c1ba9.md) | 1,425 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-0cc0b0b1c120814d78371083e96a17115c7c1ba9.md) | 733 |  317,792 |  353 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-0cc0b0b1c120814d78371083e96a17115c7c1ba9.md) | 910 |  1,745,757 |  316 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-0cc0b0b1c120814d78371083e96a17115c7c1ba9.md) | 2,354 |  2,580,026 |  770 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0cc0b0b1c120814d78371083e96a17115c7c1ba9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24265865060)
