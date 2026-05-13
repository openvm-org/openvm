| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-223ff2a648a9cc517817437f5035963a8877df8a.md) | 3,794 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-223ff2a648a9cc517817437f5035963a8877df8a.md) | 18,822 |  18,655,329 |  3,296 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-223ff2a648a9cc517817437f5035963a8877df8a.md) | 9,174 |  14,793,960 |  1,409 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-223ff2a648a9cc517817437f5035963a8877df8a.md) | 1,453 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-223ff2a648a9cc517817437f5035963a8877df8a.md) | 635 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-223ff2a648a9cc517817437f5035963a8877df8a.md) | 909 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-223ff2a648a9cc517817437f5035963a8877df8a.md) | 2,057 |  2,579,903 |  436 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-223ff2a648a9cc517817437f5035963a8877df8a.md) | 1,826 |  12,000,265 |  457 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-223ff2a648a9cc517817437f5035963a8877df8a.md) | 851 |  4,137,067 |  191 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-223ff2a648a9cc517817437f5035963a8877df8a.md) | 543 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-223ff2a648a9cc517817437f5035963a8877df8a.md) | 649 |  1,745,757 |  154 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-223ff2a648a9cc517817437f5035963a8877df8a.md) | 2,154 |  2,579,903 |  423 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/223ff2a648a9cc517817437f5035963a8877df8a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25799488592)
