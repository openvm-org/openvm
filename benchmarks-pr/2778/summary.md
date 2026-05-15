| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-1fab9d05f2db7333111943f62e6779febee801dd.md) | 1,428 |  4,000,051 |  440 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-1fab9d05f2db7333111943f62e6779febee801dd.md) | 13,335 |  14,365,133 |  2,197 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-1fab9d05f2db7333111943f62e6779febee801dd.md) | 9,092 |  11,167,961 |  1,431 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-1fab9d05f2db7333111943f62e6779febee801dd.md) | 1,351 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-1fab9d05f2db7333111943f62e6779febee801dd.md) | 473 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-1fab9d05f2db7333111943f62e6779febee801dd.md) | 594 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-1fab9d05f2db7333111943f62e6779febee801dd.md) | 1,795 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1fab9d05f2db7333111943f62e6779febee801dd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25925819441)
