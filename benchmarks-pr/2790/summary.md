| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 3,771 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 19,134 |  18,655,329 |  3,341 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 10,252 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 1,374 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 612 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 905 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 1,911 |  2,579,903 |  410 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 1,775 |  12,000,265 |  408 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 817 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 511 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 632 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-23c36b36bcc445fe4e16d2f018bde146b1bba70f.md) | 2,031 |  2,579,903 |  399 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/23c36b36bcc445fe4e16d2f018bde146b1bba70f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26649431813)
