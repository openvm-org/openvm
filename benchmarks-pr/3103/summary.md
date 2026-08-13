| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-d20c0b3823528afb27888a9ca6e90ab2015d0cbd.md) | 464 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-d20c0b3823528afb27888a9ca6e90ab2015d0cbd.md) | 7,523 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-d20c0b3823528afb27888a9ca6e90ab2015d0cbd.md) | 4,209 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-d20c0b3823528afb27888a9ca6e90ab2015d0cbd.md) | 665 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-d20c0b3823528afb27888a9ca6e90ab2015d0cbd.md) | 195 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-d20c0b3823528afb27888a9ca6e90ab2015d0cbd.md) | 233 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-d20c0b3823528afb27888a9ca6e90ab2015d0cbd.md) | 2,030 |  1,979,971 |  528 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d20c0b3823528afb27888a9ca6e90ab2015d0cbd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31731766915)
