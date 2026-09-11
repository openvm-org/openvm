| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/fibonacci-ef17ba605a9c877d559fad7880cd45dbf1d31f21.md) | 479 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/keccak-ef17ba605a9c877d559fad7880cd45dbf1d31f21.md) | 7,645 |  14,365,133 |  1,626 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/sha2_bench-ef17ba605a9c877d559fad7880cd45dbf1d31f21.md) | 4,328 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/regex-ef17ba605a9c877d559fad7880cd45dbf1d31f21.md) | 745 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/ecrecover-ef17ba605a9c877d559fad7880cd45dbf1d31f21.md) | 207 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/pairing-ef17ba605a9c877d559fad7880cd45dbf1d31f21.md) | 251 |  592,827 |  172 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/kitchen_sink-ef17ba605a9c877d559fad7880cd45dbf1d31f21.md) | 2,279 |  1,979,971 |  479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ef17ba605a9c877d559fad7880cd45dbf1d31f21

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34580367422)
