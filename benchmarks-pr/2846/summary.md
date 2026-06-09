| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-290f44f452020728a7a0788d1b89066a4e848c28.md) | 3,691 |  12,000,265 | <span style='color: green'>(-3571 [-79.6%])</span> 915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-290f44f452020728a7a0788d1b89066a4e848c28.md) | 18,032 |  18,655,329 |  3,278 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-290f44f452020728a7a0788d1b89066a4e848c28.md) | 9,892 |  14,793,960 |  1,445 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-290f44f452020728a7a0788d1b89066a4e848c28.md) | 1,403 |  4,137,067 | <span style='color: green'>(-11638 [-97.0%])</span> 359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-290f44f452020728a7a0788d1b89066a4e848c28.md) | 602 |  123,583 | <span style='color: green'>(-5600 [-95.6%])</span> 256 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-290f44f452020728a7a0788d1b89066a4e848c28.md) | 883 |  1,745,757 | <span style='color: green'>(-6117 [-95.9%])</span> 263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-290f44f452020728a7a0788d1b89066a4e848c28.md) | 3,837 |  2,579,903 |  947 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-290f44f452020728a7a0788d1b89066a4e848c28.md) | 1,619 |  12,000,265 |  410 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-290f44f452020728a7a0788d1b89066a4e848c28.md) | 668 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-290f44f452020728a7a0788d1b89066a4e848c28.md) | 362 |  123,583 |  131 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-290f44f452020728a7a0788d1b89066a4e848c28.md) | 485 |  1,745,757 |  130 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-290f44f452020728a7a0788d1b89066a4e848c28.md) | 1,825 |  2,579,903 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/290f44f452020728a7a0788d1b89066a4e848c28

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27215578428)
