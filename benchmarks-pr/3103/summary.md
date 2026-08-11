| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-fd841b68d55c627c81b0d0708fc3af4255771eeb.md) | 473 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-fd841b68d55c627c81b0d0708fc3af4255771eeb.md) | 7,320 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-fd841b68d55c627c81b0d0708fc3af4255771eeb.md) | 4,161 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-fd841b68d55c627c81b0d0708fc3af4255771eeb.md) | 654 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-fd841b68d55c627c81b0d0708fc3af4255771eeb.md) | 223 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-fd841b68d55c627c81b0d0708fc3af4255771eeb.md) | 231 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-fd841b68d55c627c81b0d0708fc3af4255771eeb.md) | 2,024 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fd841b68d55c627c81b0d0708fc3af4255771eeb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31502104563)
