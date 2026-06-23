| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-cf2d645707086ad2908a4de9e164bddae1c003e5.md) | 3,073 |  12,000,265 |  678 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-cf2d645707086ad2908a4de9e164bddae1c003e5.md) | 16,086 |  18,655,329 |  2,983 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-cf2d645707086ad2908a4de9e164bddae1c003e5.md) | 9,084 |  14,793,960 |  1,114 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-cf2d645707086ad2908a4de9e164bddae1c003e5.md) | 1,192 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-cf2d645707086ad2908a4de9e164bddae1c003e5.md) | 602 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-cf2d645707086ad2908a4de9e164bddae1c003e5.md) | 938 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-cf2d645707086ad2908a4de9e164bddae1c003e5.md) | 4,095 |  2,579,903 |  875 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cf2d645707086ad2908a4de9e164bddae1c003e5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28048031224)
