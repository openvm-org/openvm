| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 3,980 |  12,000,265 | <span style='color: green'>(-3331 [-74.3%])</span> 1,155 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 22,122 |  18,655,329 |  4,685 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 9,579 |  14,793,960 |  1,851 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 1,503 |  4,137,067 | <span style='color: green'>(-11571 [-96.4%])</span> 426 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 603 |  123,583 | <span style='color: green'>(-5580 [-95.3%])</span> 276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 948 |  1,745,757 | <span style='color: green'>(-6074 [-95.2%])</span> 306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 4,147 |  2,579,903 |  887 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 1,709 |  12,000,265 |  494 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 720 |  4,137,067 |  198 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 367 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 503 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-a35d1304f65035c9606d1d2065145a46deb3d914.md) | 2,167 |  2,579,903 |  387 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a35d1304f65035c9606d1d2065145a46deb3d914

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27443271631)
