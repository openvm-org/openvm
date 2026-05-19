| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/fibonacci-2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4.md) | 3,767 |  12,000,265 |  911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/keccak-2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4.md) | 18,717 |  18,655,329 |  3,310 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/sha2_bench-2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4.md) | 10,217 |  14,793,960 |  1,468 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/regex-2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4.md) | 1,405 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/ecrecover-2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4.md) | 604 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/pairing-2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4.md) | 890 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2793/kitchen_sink-2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4.md) | 1,904 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ef2c8f150e8c51e46b8078adcdd68c7630d9fb4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26069989299)
