| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3018/fibonacci-d79159dec1a34799750402f35070d011d5f956df.md) | 461 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3018/keccak-d79159dec1a34799750402f35070d011d5f956df.md) | 8,720 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3018/sha2_bench-d79159dec1a34799750402f35070d011d5f956df.md) | 4,075 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3018/regex-d79159dec1a34799750402f35070d011d5f956df.md) | 570 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3018/ecrecover-d79159dec1a34799750402f35070d011d5f956df.md) | 220 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3018/pairing-d79159dec1a34799750402f35070d011d5f956df.md) | 290 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3018/kitchen_sink-d79159dec1a34799750402f35070d011d5f956df.md) | 1,962 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d79159dec1a34799750402f35070d011d5f956df

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29362432575)
