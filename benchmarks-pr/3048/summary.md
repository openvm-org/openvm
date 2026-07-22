| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/fibonacci-72b7965d3123b824487b99c9bc07793176dd7e13.md) | 471 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/keccak-72b7965d3123b824487b99c9bc07793176dd7e13.md) | 7,365 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/sha2_bench-72b7965d3123b824487b99c9bc07793176dd7e13.md) | 4,688 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/regex-72b7965d3123b824487b99c9bc07793176dd7e13.md) | 668 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/ecrecover-72b7965d3123b824487b99c9bc07793176dd7e13.md) | 226 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/pairing-72b7965d3123b824487b99c9bc07793176dd7e13.md) | 315 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3048/kitchen_sink-72b7965d3123b824487b99c9bc07793176dd7e13.md) | 2,684 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/72b7965d3123b824487b99c9bc07793176dd7e13

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29952904964)
