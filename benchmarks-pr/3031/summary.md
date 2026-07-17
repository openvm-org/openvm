| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/fibonacci-b16dc9ff7a65b61b3950caafb28379bde01d923a.md) | 412 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/keccak-b16dc9ff7a65b61b3950caafb28379bde01d923a.md) | 8,551 |  14,365,133 |  1,512 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/sha2_bench-b16dc9ff7a65b61b3950caafb28379bde01d923a.md) | 4,180 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/regex-b16dc9ff7a65b61b3950caafb28379bde01d923a.md) | 573 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/ecrecover-b16dc9ff7a65b61b3950caafb28379bde01d923a.md) | 216 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/pairing-b16dc9ff7a65b61b3950caafb28379bde01d923a.md) | 280 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/kitchen_sink-b16dc9ff7a65b61b3950caafb28379bde01d923a.md) | 1,919 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b16dc9ff7a65b61b3950caafb28379bde01d923a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29573489640)
