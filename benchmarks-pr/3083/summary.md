| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/fibonacci-990c52b6ad14fb12651ada0b20fd997d607c5285.md) | 458 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/keccak-990c52b6ad14fb12651ada0b20fd997d607c5285.md) | 7,235 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/sha2_bench-990c52b6ad14fb12651ada0b20fd997d607c5285.md) | 4,704 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/regex-990c52b6ad14fb12651ada0b20fd997d607c5285.md) | 656 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/ecrecover-990c52b6ad14fb12651ada0b20fd997d607c5285.md) | 231 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/pairing-990c52b6ad14fb12651ada0b20fd997d607c5285.md) | 302 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/kitchen_sink-990c52b6ad14fb12651ada0b20fd997d607c5285.md) | 2,655 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/990c52b6ad14fb12651ada0b20fd997d607c5285

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30484953367)
