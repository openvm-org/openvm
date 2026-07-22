| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-e34ae0af04bb0b1a7f9d1a797d0a23c480efd821.md) | 470 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-e34ae0af04bb0b1a7f9d1a797d0a23c480efd821.md) | 7,249 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-e34ae0af04bb0b1a7f9d1a797d0a23c480efd821.md) | 4,634 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-e34ae0af04bb0b1a7f9d1a797d0a23c480efd821.md) | 674 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-e34ae0af04bb0b1a7f9d1a797d0a23c480efd821.md) | 225 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-e34ae0af04bb0b1a7f9d1a797d0a23c480efd821.md) | 320 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-e34ae0af04bb0b1a7f9d1a797d0a23c480efd821.md) | 2,668 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e34ae0af04bb0b1a7f9d1a797d0a23c480efd821

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29905328100)
