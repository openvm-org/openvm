| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-7e18de6c999f94107e5b3776f218fb7ee055ca3f.md) | 464 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-7e18de6c999f94107e5b3776f218fb7ee055ca3f.md) | 8,835 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-7e18de6c999f94107e5b3776f218fb7ee055ca3f.md) | 3,933 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-7e18de6c999f94107e5b3776f218fb7ee055ca3f.md) | 502 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-7e18de6c999f94107e5b3776f218fb7ee055ca3f.md) | 216 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-7e18de6c999f94107e5b3776f218fb7ee055ca3f.md) | 271 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-7e18de6c999f94107e5b3776f218fb7ee055ca3f.md) | 1,927 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7e18de6c999f94107e5b3776f218fb7ee055ca3f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29371557969)
