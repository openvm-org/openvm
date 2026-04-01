| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-847e4718751286d3ec3049cbcb1af94693d43d78.md) | 3,870 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-847e4718751286d3ec3049cbcb1af94693d43d78.md) | 15,570 |  1,235,218 |  2,179 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-847e4718751286d3ec3049cbcb1af94693d43d78.md) | 1,406 |  4,136,694 |  367 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-847e4718751286d3ec3049cbcb1af94693d43d78.md) | 631 |  122,348 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-847e4718751286d3ec3049cbcb1af94693d43d78.md) | 921 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-847e4718751286d3ec3049cbcb1af94693d43d78.md) | 2,364 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/847e4718751286d3ec3049cbcb1af94693d43d78

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23858945568)
