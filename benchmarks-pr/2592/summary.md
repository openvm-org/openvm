| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 3,835 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 18,748 |  18,655,329 |  3,345 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 9,952 |  14,793,960 |  1,414 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 1,419 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 646 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 909 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 2,155 |  2,579,903 |  435 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 1,868 |  12,000,265 |  453 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 867 |  4,137,067 |  193 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 555 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 660 |  1,745,757 |  152 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-dc2c52ce7c237b2446c98b2f624e361f755a9947.md) | 2,285 |  2,579,903 |  426 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc2c52ce7c237b2446c98b2f624e361f755a9947

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24357554185)
