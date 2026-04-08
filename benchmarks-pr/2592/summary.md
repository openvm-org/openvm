| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-dd42a1999ecfc3dc780fe72d381e941179ecc751.md) | 3,827 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-dd42a1999ecfc3dc780fe72d381e941179ecc751.md) | 18,532 |  18,655,329 |  3,323 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-dd42a1999ecfc3dc780fe72d381e941179ecc751.md) | 1,423 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-dd42a1999ecfc3dc780fe72d381e941179ecc751.md) | 642 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-dd42a1999ecfc3dc780fe72d381e941179ecc751.md) | 905 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-dd42a1999ecfc3dc780fe72d381e941179ecc751.md) | 2,158 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dd42a1999ecfc3dc780fe72d381e941179ecc751

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24153761182)
