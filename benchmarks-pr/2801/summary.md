| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/fibonacci-2c8452e3e29f24f4907170b2b3e4d4a3872644c9.md) | 1,554 |  4,000,051 |  433 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/keccak-2c8452e3e29f24f4907170b2b3e4d4a3872644c9.md) | 13,897 |  14,365,133 |  2,368 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/sha2_bench-2c8452e3e29f24f4907170b2b3e4d4a3872644c9.md) | 9,120 |  11,167,961 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/regex-2c8452e3e29f24f4907170b2b3e4d4a3872644c9.md) | 1,571 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/ecrecover-2c8452e3e29f24f4907170b2b3e4d4a3872644c9.md) | 485 |  112,210 |  259 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/pairing-2c8452e3e29f24f4907170b2b3e4d4a3872644c9.md) | 605 |  592,827 |  258 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/kitchen_sink-2c8452e3e29f24f4907170b2b3e4d4a3872644c9.md) | 1,999 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c8452e3e29f24f4907170b2b3e4d4a3872644c9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26884938833)
