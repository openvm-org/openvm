| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/fibonacci-93fbe75466327604b8797ac9b6bff325b604724c.md) | 3,787 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/keccak-93fbe75466327604b8797ac9b6bff325b604724c.md) | 15,664 |  1,235,218 |  2,195 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/regex-93fbe75466327604b8797ac9b6bff325b604724c.md) | 1,422 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/ecrecover-93fbe75466327604b8797ac9b6bff325b604724c.md) | 635 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/pairing-93fbe75466327604b8797ac9b6bff325b604724c.md) | 946 |  1,745,757 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/kitchen_sink-93fbe75466327604b8797ac9b6bff325b604724c.md) | 2,371 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/93fbe75466327604b8797ac9b6bff325b604724c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23621545263)
