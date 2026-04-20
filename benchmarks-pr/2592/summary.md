| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 3,815 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 18,487 |  18,655,329 |  3,302 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 8,906 |  14,793,960 |  1,389 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 1,411 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 639 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 908 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 2,082 |  2,579,903 |  433 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 1,862 |  12,000,265 |  455 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 853 |  4,137,067 |  192 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 551 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 661 |  1,745,757 |  154 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-878ce75c91f912db5f6f6caabd4f6802f8ceb905.md) | 2,204 |  2,579,903 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/878ce75c91f912db5f6f6caabd4f6802f8ceb905

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24685340754)
