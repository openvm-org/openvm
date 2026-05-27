| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/fibonacci-5ee9ad38b87eb2be33172537c8dd3a1993f54303.md) | 3,792 |  12,000,265 |  924 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/keccak-5ee9ad38b87eb2be33172537c8dd3a1993f54303.md) | 18,593 |  18,655,329 |  3,281 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/sha2_bench-5ee9ad38b87eb2be33172537c8dd3a1993f54303.md) | 10,221 |  14,793,960 |  1,469 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/regex-5ee9ad38b87eb2be33172537c8dd3a1993f54303.md) | 1,395 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/ecrecover-5ee9ad38b87eb2be33172537c8dd3a1993f54303.md) | 602 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/pairing-5ee9ad38b87eb2be33172537c8dd3a1993f54303.md) | 883 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2823/kitchen_sink-5ee9ad38b87eb2be33172537c8dd3a1993f54303.md) | 1,901 |  2,579,903 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5ee9ad38b87eb2be33172537c8dd3a1993f54303

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26543051434)
