| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-7f15e22bbcf442661096b974513481d56bca3ad7.md) | 410 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-7f15e22bbcf442661096b974513481d56bca3ad7.md) | 8,584 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-7f15e22bbcf442661096b974513481d56bca3ad7.md) | 4,201 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-7f15e22bbcf442661096b974513481d56bca3ad7.md) | 580 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-7f15e22bbcf442661096b974513481d56bca3ad7.md) | 219 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-7f15e22bbcf442661096b974513481d56bca3ad7.md) | 284 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-7f15e22bbcf442661096b974513481d56bca3ad7.md) | 1,920 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7f15e22bbcf442661096b974513481d56bca3ad7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29656107251)
