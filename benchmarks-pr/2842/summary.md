| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/fibonacci-06061ee0ca0137f2af58721ac9af9378d56e1882.md) | 3,786 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/keccak-06061ee0ca0137f2af58721ac9af9378d56e1882.md) | 18,598 |  18,655,329 |  3,268 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/sha2_bench-06061ee0ca0137f2af58721ac9af9378d56e1882.md) | 10,108 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/regex-06061ee0ca0137f2af58721ac9af9378d56e1882.md) | 1,403 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/ecrecover-06061ee0ca0137f2af58721ac9af9378d56e1882.md) | 597 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/pairing-06061ee0ca0137f2af58721ac9af9378d56e1882.md) | 893 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2842/kitchen_sink-06061ee0ca0137f2af58721ac9af9378d56e1882.md) | 1,905 |  2,579,903 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/06061ee0ca0137f2af58721ac9af9378d56e1882

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26972902189)
