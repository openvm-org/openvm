| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 3,977 |  12,000,265 | <span style='color: green'>(-3342 [-74.5%])</span> 1,144 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 21,935 |  18,655,329 |  4,645 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 9,806 |  14,793,960 |  1,877 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 1,493 |  4,137,067 | <span style='color: green'>(-11571 [-96.4%])</span> 426 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 604 |  123,583 | <span style='color: green'>(-5574 [-95.2%])</span> 282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 935 |  1,745,757 | <span style='color: green'>(-6072 [-95.2%])</span> 308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 4,111 |  2,579,903 |  877 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 1,708 |  12,000,265 |  492 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 714 |  4,137,067 |  197 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 367 |  123,583 |  143 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 506 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-3f897bcacec733f71d2b9d9ca9df83b245f216b1.md) | 2,170 |  2,579,903 |  382 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3f897bcacec733f71d2b9d9ca9df83b245f216b1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27225067865)
