| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 3,827 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 18,741 |  18,655,329 |  3,346 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 8,901 |  14,793,960 |  1,386 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 1,423 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 647 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 905 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 2,085 |  2,579,903 |  437 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 1,867 |  12,000,265 |  453 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 862 |  4,137,067 |  192 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 555 |  123,583 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 665 |  1,745,757 |  154 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-cfbbed97c95df9d32a51772738f0f218df9def3d.md) | 2,218 |  2,579,903 |  423 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cfbbed97c95df9d32a51772738f0f218df9def3d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24420095913)
