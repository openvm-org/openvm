| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/fibonacci-b1d1e976122dacf6af95aa1f2c0fb470fb99589d.md) | 471 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/keccak-b1d1e976122dacf6af95aa1f2c0fb470fb99589d.md) | 7,275 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/sha2_bench-b1d1e976122dacf6af95aa1f2c0fb470fb99589d.md) | 4,776 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/regex-b1d1e976122dacf6af95aa1f2c0fb470fb99589d.md) | 669 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/ecrecover-b1d1e976122dacf6af95aa1f2c0fb470fb99589d.md) | 229 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/pairing-b1d1e976122dacf6af95aa1f2c0fb470fb99589d.md) | 309 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/kitchen_sink-b1d1e976122dacf6af95aa1f2c0fb470fb99589d.md) | 2,654 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b1d1e976122dacf6af95aa1f2c0fb470fb99589d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30306335525)
