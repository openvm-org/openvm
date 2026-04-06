| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-91758cd55f87e12fa6de3e9207f6d0ccd232d9f0.md) | 3,834 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-91758cd55f87e12fa6de3e9207f6d0ccd232d9f0.md) | 18,662 |  18,655,329 |  3,347 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-91758cd55f87e12fa6de3e9207f6d0ccd232d9f0.md) | 1,424 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-91758cd55f87e12fa6de3e9207f6d0ccd232d9f0.md) | 654 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-91758cd55f87e12fa6de3e9207f6d0ccd232d9f0.md) | 913 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-91758cd55f87e12fa6de3e9207f6d0ccd232d9f0.md) | 2,281 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/91758cd55f87e12fa6de3e9207f6d0ccd232d9f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24038630054)
