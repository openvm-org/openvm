| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-69a5774cf7ff2f86df4b3c50c87ccf11310d521a.md) | 1,666 |  4,000,051 |  530 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-69a5774cf7ff2f86df4b3c50c87ccf11310d521a.md) | 16,306 |  14,365,133 |  3,029 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-69a5774cf7ff2f86df4b3c50c87ccf11310d521a.md) | 10,431 |  11,167,961 |  1,941 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-69a5774cf7ff2f86df4b3c50c87ccf11310d521a.md) | 1,543 |  4,090,656 |  432 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-69a5774cf7ff2f86df4b3c50c87ccf11310d521a.md) | 482 |  112,210 |  312 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-69a5774cf7ff2f86df4b3c50c87ccf11310d521a.md) | 616 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-69a5774cf7ff2f86df4b3c50c87ccf11310d521a.md) | 4,010 |  1,979,971 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/69a5774cf7ff2f86df4b3c50c87ccf11310d521a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27429250975)
