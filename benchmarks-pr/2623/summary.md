| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2623/fibonacci-3c892e3e53d2a10a0719aa5c62bd69197e291ae9.md) | 3,766 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2623/keccak-3c892e3e53d2a10a0719aa5c62bd69197e291ae9.md) | 18,465 |  18,655,329 |  3,264 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2623/regex-3c892e3e53d2a10a0719aa5c62bd69197e291ae9.md) | 1,427 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2623/ecrecover-3c892e3e53d2a10a0719aa5c62bd69197e291ae9.md) | 646 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2623/pairing-3c892e3e53d2a10a0719aa5c62bd69197e291ae9.md) | 913 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2623/kitchen_sink-3c892e3e53d2a10a0719aa5c62bd69197e291ae9.md) | 2,277 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3c892e3e53d2a10a0719aa5c62bd69197e291ae9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23567112007)
