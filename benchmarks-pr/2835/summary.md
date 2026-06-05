| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/fibonacci-4f127bf15667a5e9702acea6af57be6968a7ed22.md) | 3,745 |  12,000,265 |  922 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/keccak-4f127bf15667a5e9702acea6af57be6968a7ed22.md) | 18,041 |  18,655,329 |  3,287 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/sha2_bench-4f127bf15667a5e9702acea6af57be6968a7ed22.md) | 10,091 |  14,793,960 |  1,468 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/regex-4f127bf15667a5e9702acea6af57be6968a7ed22.md) | 1,414 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/ecrecover-4f127bf15667a5e9702acea6af57be6968a7ed22.md) | 598 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/pairing-4f127bf15667a5e9702acea6af57be6968a7ed22.md) | 892 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/kitchen_sink-4f127bf15667a5e9702acea6af57be6968a7ed22.md) | 3,903 |  2,579,903 |  969 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4f127bf15667a5e9702acea6af57be6968a7ed22

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27029279659)
