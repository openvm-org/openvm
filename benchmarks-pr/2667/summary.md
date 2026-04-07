| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/fibonacci-82cf15880b3509448ff1fc8a0e1abd900af11cd9.md) | 3,852 |  12,000,265 |  964 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/keccak-82cf15880b3509448ff1fc8a0e1abd900af11cd9.md) | 18,635 |  18,655,329 |  3,362 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/regex-82cf15880b3509448ff1fc8a0e1abd900af11cd9.md) | 1,426 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/ecrecover-82cf15880b3509448ff1fc8a0e1abd900af11cd9.md) | 648 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/pairing-82cf15880b3509448ff1fc8a0e1abd900af11cd9.md) | 906 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2667/kitchen_sink-82cf15880b3509448ff1fc8a0e1abd900af11cd9.md) | 2,274 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/82cf15880b3509448ff1fc8a0e1abd900af11cd9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24087708365)
