| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/fibonacci-fe4d28f1f4474eb83dbde5c95e4a89692b4e745d.md) | 3,820 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/keccak-fe4d28f1f4474eb83dbde5c95e4a89692b4e745d.md) | 18,581 |  18,655,329 |  3,332 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/regex-fe4d28f1f4474eb83dbde5c95e4a89692b4e745d.md) | 1,420 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/ecrecover-fe4d28f1f4474eb83dbde5c95e4a89692b4e745d.md) | 645 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/pairing-fe4d28f1f4474eb83dbde5c95e4a89692b4e745d.md) | 911 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/kitchen_sink-fe4d28f1f4474eb83dbde5c95e4a89692b4e745d.md) | 2,180 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fe4d28f1f4474eb83dbde5c95e4a89692b4e745d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24189238958)
