| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/fibonacci-843d9a056f3bb52250cd6b5941a2c9ce84ef1edb.md) | 481 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/keccak-843d9a056f3bb52250cd6b5941a2c9ce84ef1edb.md) | 7,338 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/sha2_bench-843d9a056f3bb52250cd6b5941a2c9ce84ef1edb.md) | 4,101 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/regex-843d9a056f3bb52250cd6b5941a2c9ce84ef1edb.md) | 661 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/ecrecover-843d9a056f3bb52250cd6b5941a2c9ce84ef1edb.md) | 226 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/pairing-843d9a056f3bb52250cd6b5941a2c9ce84ef1edb.md) | 240 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/kitchen_sink-843d9a056f3bb52250cd6b5941a2c9ce84ef1edb.md) | 2,051 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/843d9a056f3bb52250cd6b5941a2c9ce84ef1edb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30819432739)
