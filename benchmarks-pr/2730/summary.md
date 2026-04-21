| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/fibonacci-1f5db7cd9bee535e39e3dab2eae1f661516bad41.md) | 3,809 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/keccak-1f5db7cd9bee535e39e3dab2eae1f661516bad41.md) | 19,084 |  18,655,329 |  3,391 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/sha2_bench-1f5db7cd9bee535e39e3dab2eae1f661516bad41.md) | 8,965 |  14,793,960 |  1,387 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/regex-1f5db7cd9bee535e39e3dab2eae1f661516bad41.md) | 1,420 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/ecrecover-1f5db7cd9bee535e39e3dab2eae1f661516bad41.md) | 645 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/pairing-1f5db7cd9bee535e39e3dab2eae1f661516bad41.md) | 908 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2730/kitchen_sink-1f5db7cd9bee535e39e3dab2eae1f661516bad41.md) | 2,096 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1f5db7cd9bee535e39e3dab2eae1f661516bad41

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24737908931)
