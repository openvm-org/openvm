| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 3,911 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 18,854 |  18,655,329 |  3,325 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 10,097 |  14,793,960 |  1,458 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 1,378 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 597 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 886 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 1,889 |  2,579,903 |  408 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 1,781 |  12,000,265 |  409 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 818 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 513 |  123,583 |  129 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 635 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0.md) | 2,033 |  2,579,903 |  401 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/163d10cdf5aa0cde19bb3ab0b8bad4a8854512d0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26305310141)
