| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-701d30098007a1884b2247f9e630a5c766ba2aab.md) | 3,863 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-701d30098007a1884b2247f9e630a5c766ba2aab.md) | 18,653 |  18,655,329 |  3,327 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-701d30098007a1884b2247f9e630a5c766ba2aab.md) | 1,437 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-701d30098007a1884b2247f9e630a5c766ba2aab.md) | 656 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-701d30098007a1884b2247f9e630a5c766ba2aab.md) | 920 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-701d30098007a1884b2247f9e630a5c766ba2aab.md) | 2,286 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/701d30098007a1884b2247f9e630a5c766ba2aab

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23960958984)
