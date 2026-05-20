| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 3,752 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 18,723 |  18,655,329 |  3,298 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 10,234 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 1,408 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 608 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 890 |  1,745,757 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 1,898 |  2,579,903 |  412 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 1,783 |  12,000,265 |  414 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 821 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 506 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 639 |  1,745,757 |  133 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-36c26d5ff2809ca7823e2eaabdb0c9fbd915def0.md) | 2,019 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36c26d5ff2809ca7823e2eaabdb0c9fbd915def0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26186822664)
