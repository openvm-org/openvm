| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/fibonacci-4b9978b350a9035da48c8a1b17173facec17a770.md) | 3,917 |  12,000,265 |  967 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/keccak-4b9978b350a9035da48c8a1b17173facec17a770.md) | 18,363 |  18,655,329 |  3,286 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/regex-4b9978b350a9035da48c8a1b17173facec17a770.md) | 1,425 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/ecrecover-4b9978b350a9035da48c8a1b17173facec17a770.md) | 650 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/pairing-4b9978b350a9035da48c8a1b17173facec17a770.md) | 904 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/kitchen_sink-4b9978b350a9035da48c8a1b17173facec17a770.md) | 2,314 |  2,579,903 |  447 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4b9978b350a9035da48c8a1b17173facec17a770

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24095847002)
