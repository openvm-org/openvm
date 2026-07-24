| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90.md) | 470 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90.md) | 7,321 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90.md) | 4,772 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90.md) | 665 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90.md) | 231 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90.md) | 312 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90.md) | 2,671 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6e271dabcbbbc45eed2eb3fc8746f5d6cb728c90

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30088328152)
