| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-01db275e2a4a537a28e68bc974afe3db5fcdabbf.md) | 3,842 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-01db275e2a4a537a28e68bc974afe3db5fcdabbf.md) | 18,554 |  18,655,329 |  3,310 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-01db275e2a4a537a28e68bc974afe3db5fcdabbf.md) | 1,429 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-01db275e2a4a537a28e68bc974afe3db5fcdabbf.md) | 648 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-01db275e2a4a537a28e68bc974afe3db5fcdabbf.md) | 901 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-01db275e2a4a537a28e68bc974afe3db5fcdabbf.md) | 2,149 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/01db275e2a4a537a28e68bc974afe3db5fcdabbf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24188246476)
