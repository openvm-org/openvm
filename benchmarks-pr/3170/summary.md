| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/fibonacci-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 483 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/keccak-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 7,707 |  14,365,133 |  1,622 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/sha2_bench-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 4,306 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/regex-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 745 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/ecrecover-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 207 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/pairing-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 258 |  592,827 |  175 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/kitchen_sink-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 2,262 |  1,979,971 |  478 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/fibonacci_e2e-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 758 |  4,000,053 |  225 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/regex_e2e-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 1,051 |  4,090,658 |  206 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/ecrecover_e2e-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 514 |  112,212 |  178 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/pairing_e2e-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 555 |  592,829 |  164 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3170/kitchen_sink_e2e-d02834f0e57869e0adaf77c3eca223e97176f930.md) | 2,493 |  1,979,973 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d02834f0e57869e0adaf77c3eca223e97176f930

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38981196511)
