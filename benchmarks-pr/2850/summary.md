| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-34325145d0d54120ef22aa1663e3c60ceef44cbb.md) | 5,262 |  4,000,051 |  440 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-34325145d0d54120ef22aa1663e3c60ceef44cbb.md) | 18,670 |  14,365,133 |  2,380 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-34325145d0d54120ef22aa1663e3c60ceef44cbb.md) | 12,677 |  11,167,961 |  1,425 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-34325145d0d54120ef22aa1663e3c60ceef44cbb.md) | 3,618 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-34325145d0d54120ef22aa1663e3c60ceef44cbb.md) | 1,965 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-34325145d0d54120ef22aa1663e3c60ceef44cbb.md) | 2,090 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-34325145d0d54120ef22aa1663e3c60ceef44cbb.md) | 6,005 |  1,979,971 |  943 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/34325145d0d54120ef22aa1663e3c60ceef44cbb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27301833555)
