| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/fibonacci-5b578258738a355fae7e51bae01be4bdc4f34016.md) | 3,821 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/keccak-5b578258738a355fae7e51bae01be4bdc4f34016.md) | 15,847 |  1,235,218 |  2,224 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/regex-5b578258738a355fae7e51bae01be4bdc4f34016.md) | 1,435 |  4,136,694 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/ecrecover-5b578258738a355fae7e51bae01be4bdc4f34016.md) | 637 |  122,348 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/pairing-5b578258738a355fae7e51bae01be4bdc4f34016.md) | 922 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2648/kitchen_sink-5b578258738a355fae7e51bae01be4bdc4f34016.md) | 2,398 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5b578258738a355fae7e51bae01be4bdc4f34016

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23866098160)
