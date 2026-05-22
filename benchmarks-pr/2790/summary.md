| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 3,747 |  12,000,265 |  910 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 18,623 |  18,655,329 |  3,291 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 10,249 |  14,793,960 |  1,465 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 1,405 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 602 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 887 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 1,900 |  2,579,903 |  412 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 1,776 |  12,000,265 |  413 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 820 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 509 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 638 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d.md) | 2,041 |  2,579,903 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a0689e34bd62f2a0cbb5406bb89dc6e58c4c963d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26275937063)
