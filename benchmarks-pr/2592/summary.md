| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 3,785 |  12,000,265 |  925 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 18,734 |  18,655,329 |  3,302 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 10,338 |  14,793,960 |  1,485 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 1,393 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 606 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 898 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 1,896 |  2,579,903 |  413 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 1,772 |  12,000,265 |  407 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 828 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 515 |  123,583 |  132 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 632 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0.md) | 2,033 |  2,579,903 |  397 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5ddd5396a2ac15a84b8300be28f1ba7c7a0bdff0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26195885970)
