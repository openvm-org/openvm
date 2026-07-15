| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/fibonacci-1c8bb9ba0e6e6bc330bdf622e065960657bd0273.md) | 408 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/keccak-1c8bb9ba0e6e6bc330bdf622e065960657bd0273.md) | 8,344 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/sha2_bench-1c8bb9ba0e6e6bc330bdf622e065960657bd0273.md) | 3,946 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/regex-1c8bb9ba0e6e6bc330bdf622e065960657bd0273.md) | 569 |  4,090,656 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/ecrecover-1c8bb9ba0e6e6bc330bdf622e065960657bd0273.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/pairing-1c8bb9ba0e6e6bc330bdf622e065960657bd0273.md) | 272 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/kitchen_sink-1c8bb9ba0e6e6bc330bdf622e065960657bd0273.md) | 1,885 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1c8bb9ba0e6e6bc330bdf622e065960657bd0273

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29414417794)
