| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/fibonacci-057bf00629c87712893136bd41b91ab9779c4f3e.md) | 3,869 |  12,000,265 |  964 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/keccak-057bf00629c87712893136bd41b91ab9779c4f3e.md) | 18,697 |  18,655,329 |  3,359 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/regex-057bf00629c87712893136bd41b91ab9779c4f3e.md) | 1,409 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/ecrecover-057bf00629c87712893136bd41b91ab9779c4f3e.md) | 649 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/pairing-057bf00629c87712893136bd41b91ab9779c4f3e.md) | 906 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/kitchen_sink-057bf00629c87712893136bd41b91ab9779c4f3e.md) | 2,145 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/057bf00629c87712893136bd41b91ab9779c4f3e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24203196752)
