| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-4bca1c4d7077460540e1c0b3df833a7aaba86d11.md) | 1,909 |  4,000,051 |  543 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-4bca1c4d7077460540e1c0b3df833a7aaba86d11.md) | 13,678 |  14,365,133 |  2,252 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-4bca1c4d7077460540e1c0b3df833a7aaba86d11.md) | 9,764 |  11,167,961 |  1,459 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-4bca1c4d7077460540e1c0b3df833a7aaba86d11.md) | 1,597 |  4,090,656 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-4bca1c4d7077460540e1c0b3df833a7aaba86d11.md) | 645 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-4bca1c4d7077460540e1c0b3df833a7aaba86d11.md) | 752 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-4bca1c4d7077460540e1c0b3df833a7aaba86d11.md) | 2,043 |  1,979,971 |  428 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4bca1c4d7077460540e1c0b3df833a7aaba86d11

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25751597993)
