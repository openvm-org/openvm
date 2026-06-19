| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89.md) | 3,093 |  12,000,265 |  681 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89.md) | 16,419 |  18,655,329 |  3,060 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89.md) | 9,205 |  14,793,960 |  1,121 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89.md) | 1,162 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89.md) | 602 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89.md) | 921 |  1,745,757 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89.md) | 4,133 |  2,579,903 |  883 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/279227c4c3d4577701b1b4a2e0bef0bcbf4e0c89

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27848336640)
