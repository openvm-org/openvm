| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 3,855 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 18,732 |  18,655,329 |  3,343 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 8,946 |  14,793,960 |  1,393 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 1,413 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 645 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 904 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 2,081 |  2,579,903 |  432 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 1,872 |  12,000,265 |  453 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 857 |  4,137,067 |  190 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 554 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 662 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-09b49b83538bb2aeab1f2c8559013bd3f203e2bb.md) | 2,210 |  2,579,903 |  423 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/09b49b83538bb2aeab1f2c8559013bd3f203e2bb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24481143098)
