| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-02c0e74201bdd7501cdecb73659b12d148ce5240.md) | 3,842 |  12,000,265 |  957 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-02c0e74201bdd7501cdecb73659b12d148ce5240.md) | 18,734 |  18,655,329 |  3,339 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-02c0e74201bdd7501cdecb73659b12d148ce5240.md) | 1,403 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-02c0e74201bdd7501cdecb73659b12d148ce5240.md) | 645 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-02c0e74201bdd7501cdecb73659b12d148ce5240.md) | 909 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-02c0e74201bdd7501cdecb73659b12d148ce5240.md) | 2,147 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/02c0e74201bdd7501cdecb73659b12d148ce5240

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24186030312)
