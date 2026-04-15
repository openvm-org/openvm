| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 3,837 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 18,626 |  18,655,329 |  3,315 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 8,911 |  14,793,960 |  1,388 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 1,406 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 643 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 914 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 2,100 |  2,579,903 |  437 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 1,864 |  12,000,265 |  453 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 847 |  4,137,067 |  193 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 554 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 661 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9.md) | 2,219 |  2,579,903 |  423 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/913b4bb84a6b1f4b820c2a9e46a366c5ce4d67f9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24478405511)
