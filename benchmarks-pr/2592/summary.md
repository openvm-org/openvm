| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 3,857 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 18,653 |  18,655,329 |  3,322 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 9,133 |  14,793,960 |  1,418 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 1,436 |  4,137,067 |  384 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 646 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 912 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 2,087 |  2,579,903 |  435 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 1,863 |  12,000,265 |  449 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 852 |  4,137,067 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 553 |  123,583 |  153 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 655 |  1,745,757 |  154 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-dada909aa0ed06d206be5acaa90233c278a9a95f.md) | 2,207 |  2,579,903 |  426 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dada909aa0ed06d206be5acaa90233c278a9a95f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24738876702)
