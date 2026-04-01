| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/fibonacci-5646b115a82fa6d70cb4eb117a17fa288962f23b.md) | 3,873 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/keccak-5646b115a82fa6d70cb4eb117a17fa288962f23b.md) | 15,620 |  1,235,218 |  2,184 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/regex-5646b115a82fa6d70cb4eb117a17fa288962f23b.md) | 1,418 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/ecrecover-5646b115a82fa6d70cb4eb117a17fa288962f23b.md) | 637 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/pairing-5646b115a82fa6d70cb4eb117a17fa288962f23b.md) | 913 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2650/kitchen_sink-5646b115a82fa6d70cb4eb117a17fa288962f23b.md) | 2,374 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5646b115a82fa6d70cb4eb117a17fa288962f23b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23871527282)
