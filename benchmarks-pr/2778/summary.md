| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-84251e31c4bcd95e28ee1c90350a76fe045b2436.md) | 1,406 |  4,000,051 |  430 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-84251e31c4bcd95e28ee1c90350a76fe045b2436.md) | 13,348 |  14,365,133 |  2,201 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-84251e31c4bcd95e28ee1c90350a76fe045b2436.md) | 9,090 |  11,167,961 |  1,428 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-84251e31c4bcd95e28ee1c90350a76fe045b2436.md) | 1,367 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-84251e31c4bcd95e28ee1c90350a76fe045b2436.md) | 469 |  112,210 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-84251e31c4bcd95e28ee1c90350a76fe045b2436.md) | 589 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-84251e31c4bcd95e28ee1c90350a76fe045b2436.md) | 2,198 |  1,979,971 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/84251e31c4bcd95e28ee1c90350a76fe045b2436

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25970880948)
