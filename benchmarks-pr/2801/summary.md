| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/fibonacci-d7316468a192f0f04a55f6942665a1326f73ddc3.md) | 1,561 |  4,000,051 |  439 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/keccak-d7316468a192f0f04a55f6942665a1326f73ddc3.md) | 13,963 |  14,365,133 |  2,379 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/sha2_bench-d7316468a192f0f04a55f6942665a1326f73ddc3.md) | 9,120 |  11,167,961 |  1,405 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/regex-d7316468a192f0f04a55f6942665a1326f73ddc3.md) | 1,579 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/ecrecover-d7316468a192f0f04a55f6942665a1326f73ddc3.md) | 482 |  112,210 |  263 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/pairing-d7316468a192f0f04a55f6942665a1326f73ddc3.md) | 608 |  592,827 |  256 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/kitchen_sink-d7316468a192f0f04a55f6942665a1326f73ddc3.md) | 2,017 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d7316468a192f0f04a55f6942665a1326f73ddc3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26896340198)
