| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-1551de2304a91e416d64c389c5b360ff22d04b1a.md) | 466 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-1551de2304a91e416d64c389c5b360ff22d04b1a.md) | 7,319 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-1551de2304a91e416d64c389c5b360ff22d04b1a.md) | 4,764 |  11,167,961 |  535 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-1551de2304a91e416d64c389c5b360ff22d04b1a.md) | 675 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-1551de2304a91e416d64c389c5b360ff22d04b1a.md) | 226 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-1551de2304a91e416d64c389c5b360ff22d04b1a.md) | 325 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-1551de2304a91e416d64c389c5b360ff22d04b1a.md) | 2,672 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1551de2304a91e416d64c389c5b360ff22d04b1a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29878382175)
