| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/fibonacci-aad70783ed3f1be2bc7115293b69c45c3228eae9.md) | 3,759 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/keccak-aad70783ed3f1be2bc7115293b69c45c3228eae9.md) | 18,040 |  18,655,329 |  3,277 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/sha2_bench-aad70783ed3f1be2bc7115293b69c45c3228eae9.md) | 9,829 |  14,793,960 |  1,438 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/regex-aad70783ed3f1be2bc7115293b69c45c3228eae9.md) | 1,392 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/ecrecover-aad70783ed3f1be2bc7115293b69c45c3228eae9.md) | 603 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/pairing-aad70783ed3f1be2bc7115293b69c45c3228eae9.md) | 885 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/kitchen_sink-aad70783ed3f1be2bc7115293b69c45c3228eae9.md) | 3,832 |  2,579,903 |  945 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aad70783ed3f1be2bc7115293b69c45c3228eae9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27069277443)
