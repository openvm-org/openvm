| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2881/fibonacci-36054f43cde087bddf5f720102b9130be924236c.md) | 1,566 |  4,000,051 | <span style='color: green'>(-4050 [-90.3%])</span> 436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2881/keccak-36054f43cde087bddf5f720102b9130be924236c.md) | 14,026 |  14,365,133 |  2,407 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2881/sha2_bench-36054f43cde087bddf5f720102b9130be924236c.md) | 8,848 |  11,167,961 |  1,407 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2881/regex-36054f43cde087bddf5f720102b9130be924236c.md) | 1,475 |  4,090,656 | <span style='color: green'>(-11639 [-97.0%])</span> 358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2881/ecrecover-36054f43cde087bddf5f720102b9130be924236c.md) | 472 |  112,210 | <span style='color: green'>(-5589 [-95.4%])</span> 267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2881/pairing-36054f43cde087bddf5f720102b9130be924236c.md) | 593 |  592,827 | <span style='color: green'>(-6127 [-96.0%])</span> 253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2881/kitchen_sink-36054f43cde087bddf5f720102b9130be924236c.md) | 3,748 |  1,979,971 |  929 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36054f43cde087bddf5f720102b9130be924236c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27375695519)
