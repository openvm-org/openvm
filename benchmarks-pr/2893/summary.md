| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2893/fibonacci-9c20ab525136038b408af60d87709bbffbbb1872.md) | 1,659 |  4,000,051 |  529 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2893/keccak-9c20ab525136038b408af60d87709bbffbbb1872.md) | 15,974 |  14,365,133 |  3,008 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2893/sha2_bench-9c20ab525136038b408af60d87709bbffbbb1872.md) | 10,343 |  11,167,961 |  1,920 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2893/regex-9c20ab525136038b408af60d87709bbffbbb1872.md) | 1,526 |  4,090,656 |  426 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2893/ecrecover-9c20ab525136038b408af60d87709bbffbbb1872.md) | 481 |  112,210 |  301 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2893/pairing-9c20ab525136038b408af60d87709bbffbbb1872.md) | 623 |  592,827 |  291 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2893/kitchen_sink-9c20ab525136038b408af60d87709bbffbbb1872.md) | 3,949 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c20ab525136038b408af60d87709bbffbbb1872

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27579645010)
