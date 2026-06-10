| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 4,048 |  12,000,265 | <span style='color: green'>(-3318 [-74.0%])</span> 1,168 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 21,795 |  18,655,329 |  4,619 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 9,489 |  14,793,960 |  1,821 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 1,519 |  4,137,067 | <span style='color: green'>(-11564 [-96.4%])</span> 433 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 604 |  123,583 | <span style='color: green'>(-5573 [-95.2%])</span> 283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 939 |  1,745,757 | <span style='color: green'>(-6074 [-95.2%])</span> 306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 4,150 |  2,579,903 |  879 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 1,709 |  12,000,265 |  494 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 720 |  4,137,067 |  196 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 368 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 503 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-70cccbf6f335c093b5f3fd272462b1b8ed5ef291.md) | 2,176 |  2,579,903 |  385 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/70cccbf6f335c093b5f3fd272462b1b8ed5ef291

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27281562352)
