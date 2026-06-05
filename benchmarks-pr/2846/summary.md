| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 3,724 |  12,000,265 | <span style='color: green'>(-3565 [-79.5%])</span> 921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 18,104 |  18,655,329 |  3,295 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 10,010 |  14,793,960 |  1,471 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 1,395 |  4,137,067 | <span style='color: green'>(-11639 [-97.0%])</span> 358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 598 |  123,583 | <span style='color: green'>(-5602 [-95.7%])</span> 254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 877 |  1,745,757 | <span style='color: green'>(-6117 [-95.9%])</span> 263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 3,828 |  2,579,903 |  944 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 1,618 |  12,000,265 |  410 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 667 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 362 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 478 |  1,745,757 |  132 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-593745a627fd762415bc6037da09ad6bb313d4dd.md) | 1,817 |  2,579,903 |  402 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/593745a627fd762415bc6037da09ad6bb313d4dd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27044959598)
