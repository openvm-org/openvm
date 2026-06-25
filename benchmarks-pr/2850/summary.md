| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-72ab75b770fd85360d3a0f82c274dbd88ccc1ee0.md) | 3,308 |  4,000,051 |  395 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-72ab75b770fd85360d3a0f82c274dbd88ccc1ee0.md) | 20,761 |  14,365,133 |  3,089 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-72ab75b770fd85360d3a0f82c274dbd88ccc1ee0.md) | 10,345 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-72ab75b770fd85360d3a0f82c274dbd88ccc1ee0.md) | 2,623 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-72ab75b770fd85360d3a0f82c274dbd88ccc1ee0.md) | 1,959 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-72ab75b770fd85360d3a0f82c274dbd88ccc1ee0.md) | 2,142 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-72ab75b770fd85360d3a0f82c274dbd88ccc1ee0.md) | 5,596 |  1,979,971 |  865 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/72ab75b770fd85360d3a0f82c274dbd88ccc1ee0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28177255703)
