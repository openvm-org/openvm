| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/fibonacci-9bddcdb4080547b052dcaef57efeb56baa57753d.md) | 3,759 |  12,000,265 |  920 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/keccak-9bddcdb4080547b052dcaef57efeb56baa57753d.md) | 18,476 |  18,655,329 |  3,255 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/sha2_bench-9bddcdb4080547b052dcaef57efeb56baa57753d.md) | 10,273 |  14,793,960 |  1,471 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/regex-9bddcdb4080547b052dcaef57efeb56baa57753d.md) | 1,405 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/ecrecover-9bddcdb4080547b052dcaef57efeb56baa57753d.md) | 600 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/pairing-9bddcdb4080547b052dcaef57efeb56baa57753d.md) | 887 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/kitchen_sink-9bddcdb4080547b052dcaef57efeb56baa57753d.md) | 1,899 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9bddcdb4080547b052dcaef57efeb56baa57753d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26279457779)
