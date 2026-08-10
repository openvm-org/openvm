| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-a8a7d3324e058ea25575d316dd00599d3c88b4a8.md) | 463 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-a8a7d3324e058ea25575d316dd00599d3c88b4a8.md) | 7,375 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-a8a7d3324e058ea25575d316dd00599d3c88b4a8.md) | 4,168 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-a8a7d3324e058ea25575d316dd00599d3c88b4a8.md) | 653 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-a8a7d3324e058ea25575d316dd00599d3c88b4a8.md) | 223 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-a8a7d3324e058ea25575d316dd00599d3c88b4a8.md) | 233 |  592,827 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-a8a7d3324e058ea25575d316dd00599d3c88b4a8.md) | 2,015 |  1,979,971 |  525 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a8a7d3324e058ea25575d316dd00599d3c88b4a8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31438058382)
