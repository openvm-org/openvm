| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 3,047 |  12,000,265 | <span style='color: green'>(-3815 [-85.0%])</span> 671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 16,390 |  18,655,329 |  3,038 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 9,165 |  14,793,960 |  1,121 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 1,166 |  4,137,067 | <span style='color: green'>(-11648 [-97.1%])</span> 349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 602 |  123,583 | <span style='color: green'>(-5571 [-95.1%])</span> 285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 937 |  1,745,757 | <span style='color: green'>(-6073 [-95.2%])</span> 307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 4,117 |  2,579,903 |  884 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 1,402 |  12,000,265 |  288 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 641 |  4,137,067 |  165 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 392 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 526 |  1,745,757 |  148 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-3d4c9a028e2fbc1165556ad749115a004e872c69.md) | 1,993 |  2,579,903 |  383 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3d4c9a028e2fbc1165556ad749115a004e872c69

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28136446184)
