| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-c41a88af1cede70508422b267fafa24996835766.md) | 412 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-c41a88af1cede70508422b267fafa24996835766.md) | 8,543 |  14,365,133 |  1,510 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-c41a88af1cede70508422b267fafa24996835766.md) | 4,201 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-c41a88af1cede70508422b267fafa24996835766.md) | 576 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-c41a88af1cede70508422b267fafa24996835766.md) | 218 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-c41a88af1cede70508422b267fafa24996835766.md) | 293 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-c41a88af1cede70508422b267fafa24996835766.md) | 1,922 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c41a88af1cede70508422b267fafa24996835766

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29762596804)
