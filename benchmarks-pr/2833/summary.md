| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-5d2277b0e24eab600bef75692cf25818761b5992.md) | 3,794 |  12,000,265 |  922 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-5d2277b0e24eab600bef75692cf25818761b5992.md) | 18,485 |  18,655,329 |  3,258 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-5d2277b0e24eab600bef75692cf25818761b5992.md) | 10,242 |  14,793,960 |  1,473 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-5d2277b0e24eab600bef75692cf25818761b5992.md) | 1,407 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-5d2277b0e24eab600bef75692cf25818761b5992.md) | 597 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-5d2277b0e24eab600bef75692cf25818761b5992.md) | 893 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-5d2277b0e24eab600bef75692cf25818761b5992.md) | 3,959 |  2,579,903 |  958 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5d2277b0e24eab600bef75692cf25818761b5992

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27167755989)
