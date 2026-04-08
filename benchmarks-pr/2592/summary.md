| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-312fe84a0a8b3973d10c82450a1ce310be21c756.md) | 3,774 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-312fe84a0a8b3973d10c82450a1ce310be21c756.md) | 18,693 |  18,655,329 |  3,356 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-312fe84a0a8b3973d10c82450a1ce310be21c756.md) | 1,428 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-312fe84a0a8b3973d10c82450a1ce310be21c756.md) | 654 |  123,583 |  277 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-312fe84a0a8b3973d10c82450a1ce310be21c756.md) | 910 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-312fe84a0a8b3973d10c82450a1ce310be21c756.md) | 2,171 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/312fe84a0a8b3973d10c82450a1ce310be21c756

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24154998263)
