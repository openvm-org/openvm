| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/fibonacci-60aa2e4baa1cef173956fcc804e52e04ff33e853.md) | 407 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/keccak-60aa2e4baa1cef173956fcc804e52e04ff33e853.md) | 8,336 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/sha2_bench-60aa2e4baa1cef173956fcc804e52e04ff33e853.md) | 3,983 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/regex-60aa2e4baa1cef173956fcc804e52e04ff33e853.md) | 570 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/ecrecover-60aa2e4baa1cef173956fcc804e52e04ff33e853.md) | 217 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/pairing-60aa2e4baa1cef173956fcc804e52e04ff33e853.md) | 262 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/kitchen_sink-60aa2e4baa1cef173956fcc804e52e04ff33e853.md) | 1,881 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/60aa2e4baa1cef173956fcc804e52e04ff33e853

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29429713239)
