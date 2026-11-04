| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 1,662 |  12,000,265 |  366 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 9,667 |  18,655,329 |  1,552 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 5,326 |  14,793,960 |  588 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 691 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 424 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 593 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-05f3b73e7294fbb6d38155abafc98f1fe2bcfb73.md) | 2,312 |  2,579,903 |  497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/05f3b73e7294fbb6d38155abafc98f1fe2bcfb73

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41874373258)
