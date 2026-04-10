| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-d9e3f4b0404466663a03263d884dac2ea11e5cde.md) | 3,822 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-d9e3f4b0404466663a03263d884dac2ea11e5cde.md) | 18,950 |  18,655,329 |  3,373 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/sha2_bench-d9e3f4b0404466663a03263d884dac2ea11e5cde.md) | 8,926 |  14,793,960 |  1,389 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-d9e3f4b0404466663a03263d884dac2ea11e5cde.md) | 1,426 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-d9e3f4b0404466663a03263d884dac2ea11e5cde.md) | 637 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-d9e3f4b0404466663a03263d884dac2ea11e5cde.md) | 901 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-d9e3f4b0404466663a03263d884dac2ea11e5cde.md) | 2,078 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d9e3f4b0404466663a03263d884dac2ea11e5cde

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24267346098)
