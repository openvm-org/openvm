| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 3,880 |  12,000,265 |  966 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 18,541 |  18,655,329 |  3,305 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 9,094 |  14,793,960 |  1,414 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 1,411 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 646 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 905 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 2,092 |  2,579,903 |  436 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 1,864 |  12,000,265 |  452 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 864 |  4,137,067 |  192 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 554 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 658 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-e4aab2038e0817836f31bc07fa828063ca388ae8.md) | 2,218 |  2,579,903 |  427 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e4aab2038e0817836f31bc07fa828063ca388ae8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24909017352)
