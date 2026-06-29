| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-90998169819ca531b6812fc94ba46a1404fee36a.md) | 1,035 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-90998169819ca531b6812fc94ba46a1404fee36a.md) | 15,531 |  14,365,133 |  2,975 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-90998169819ca531b6812fc94ba46a1404fee36a.md) | 8,266 |  11,167,961 |  1,016 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-90998169819ca531b6812fc94ba46a1404fee36a.md) | 1,172 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-90998169819ca531b6812fc94ba46a1404fee36a.md) | 437 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-90998169819ca531b6812fc94ba46a1404fee36a.md) | 600 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-90998169819ca531b6812fc94ba46a1404fee36a.md) | 3,873 |  1,979,971 |  855 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/90998169819ca531b6812fc94ba46a1404fee36a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28398799030)
