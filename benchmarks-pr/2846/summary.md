| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-36943a836c543c52790152f95a2b7c8fe572056b.md) | 3,031 |  12,000,265 | <span style='color: green'>(-3813 [-85.0%])</span> 673 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-36943a836c543c52790152f95a2b7c8fe572056b.md) | 16,431 |  18,655,329 |  3,033 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-36943a836c543c52790152f95a2b7c8fe572056b.md) | 9,187 |  14,793,960 |  1,122 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-36943a836c543c52790152f95a2b7c8fe572056b.md) | 1,182 |  4,137,067 | <span style='color: green'>(-11642 [-97.0%])</span> 355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-36943a836c543c52790152f95a2b7c8fe572056b.md) | 603 |  123,583 | <span style='color: green'>(-5572 [-95.2%])</span> 284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-36943a836c543c52790152f95a2b7c8fe572056b.md) | 947 |  1,745,757 | <span style='color: green'>(-6068 [-95.1%])</span> 312 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-36943a836c543c52790152f95a2b7c8fe572056b.md) | 4,122 |  2,579,903 |  884 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-36943a836c543c52790152f95a2b7c8fe572056b.md) | 1,403 |  12,000,265 |  290 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-36943a836c543c52790152f95a2b7c8fe572056b.md) | 646 |  4,137,067 |  166 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-36943a836c543c52790152f95a2b7c8fe572056b.md) | 393 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-36943a836c543c52790152f95a2b7c8fe572056b.md) | 533 |  1,745,757 |  149 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-36943a836c543c52790152f95a2b7c8fe572056b.md) | 1,990 |  2,579,903 |  382 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36943a836c543c52790152f95a2b7c8fe572056b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27982585226)
