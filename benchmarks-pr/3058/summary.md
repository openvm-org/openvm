| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-4a79440cbb524d3092f5596a504c95ad27a30d17.md) | 464 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-4a79440cbb524d3092f5596a504c95ad27a30d17.md) | 7,255 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-4a79440cbb524d3092f5596a504c95ad27a30d17.md) | 4,723 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-4a79440cbb524d3092f5596a504c95ad27a30d17.md) | 674 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-4a79440cbb524d3092f5596a504c95ad27a30d17.md) | 226 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-4a79440cbb524d3092f5596a504c95ad27a30d17.md) | 313 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-4a79440cbb524d3092f5596a504c95ad27a30d17.md) | 2,673 |  1,979,971 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4a79440cbb524d3092f5596a504c95ad27a30d17

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30284743816)
