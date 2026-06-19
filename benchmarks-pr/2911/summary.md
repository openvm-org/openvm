| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2911/fibonacci-8289aee81b966042c09aadc3480517253949bdf5.md) | 3,066 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2911/keccak-8289aee81b966042c09aadc3480517253949bdf5.md) | 16,359 |  18,655,329 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2911/sha2_bench-8289aee81b966042c09aadc3480517253949bdf5.md) | 9,235 |  14,793,960 |  1,130 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2911/regex-8289aee81b966042c09aadc3480517253949bdf5.md) | 1,167 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2911/ecrecover-8289aee81b966042c09aadc3480517253949bdf5.md) | 605 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2911/pairing-8289aee81b966042c09aadc3480517253949bdf5.md) | 937 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2911/kitchen_sink-8289aee81b966042c09aadc3480517253949bdf5.md) | 4,067 |  2,579,903 |  870 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8289aee81b966042c09aadc3480517253949bdf5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27838617542)
