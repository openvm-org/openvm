| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 3,790 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 18,271 |  18,655,329 |  3,229 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 10,376 |  14,793,960 |  1,479 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 1,409 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 607 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 892 |  1,745,757 |  269 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 1,882 |  2,579,903 |  408 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 1,779 |  12,000,265 |  408 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 818 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 507 |  123,583 |  129 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 635 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-0ea17b6b5dde5488869831a6f51a19e1781fc62e.md) | 2,028 |  2,579,903 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0ea17b6b5dde5488869831a6f51a19e1781fc62e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26132185772)
