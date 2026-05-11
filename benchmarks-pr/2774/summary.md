| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/fibonacci-a905ccfc1a1388bdce92e3c82a67c74a30efe6af.md) | 3,823 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/keccak-a905ccfc1a1388bdce92e3c82a67c74a30efe6af.md) | 18,433 |  18,655,329 |  3,284 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/sha2_bench-a905ccfc1a1388bdce92e3c82a67c74a30efe6af.md) | 8,952 |  14,793,960 |  1,389 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/regex-a905ccfc1a1388bdce92e3c82a67c74a30efe6af.md) | 1,407 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/ecrecover-a905ccfc1a1388bdce92e3c82a67c74a30efe6af.md) | 633 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/pairing-a905ccfc1a1388bdce92e3c82a67c74a30efe6af.md) | 904 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2774/kitchen_sink-a905ccfc1a1388bdce92e3c82a67c74a30efe6af.md) | 2,095 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a905ccfc1a1388bdce92e3c82a67c74a30efe6af

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25688017829)
