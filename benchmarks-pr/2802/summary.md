| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-4f3eaf870da012bcefb0be884967ee28ae365fdb.md) | 1,567 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-4f3eaf870da012bcefb0be884967ee28ae365fdb.md) | 14,022 |  14,365,133 |  2,357 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-4f3eaf870da012bcefb0be884967ee28ae365fdb.md) | 9,213 |  11,167,961 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-4f3eaf870da012bcefb0be884967ee28ae365fdb.md) | 1,616 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-4f3eaf870da012bcefb0be884967ee28ae365fdb.md) | 490 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-4f3eaf870da012bcefb0be884967ee28ae365fdb.md) | 612 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-4f3eaf870da012bcefb0be884967ee28ae365fdb.md) | 2,176 |  1,979,971 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4f3eaf870da012bcefb0be884967ee28ae365fdb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26875466538)
