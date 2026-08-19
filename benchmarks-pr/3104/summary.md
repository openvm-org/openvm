| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-f1776e0582e4c6c57bfa6b20824d9c5ff71b6653.md) | 445 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-f1776e0582e4c6c57bfa6b20824d9c5ff71b6653.md) | 7,242 |  14,365,133 |  1,619 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-f1776e0582e4c6c57bfa6b20824d9c5ff71b6653.md) | 4,146 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-f1776e0582e4c6c57bfa6b20824d9c5ff71b6653.md) | 700 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-f1776e0582e4c6c57bfa6b20824d9c5ff71b6653.md) | 210 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-f1776e0582e4c6c57bfa6b20824d9c5ff71b6653.md) | 241 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-f1776e0582e4c6c57bfa6b20824d9c5ff71b6653.md) | 2,193 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f1776e0582e4c6c57bfa6b20824d9c5ff71b6653

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32314280940)
