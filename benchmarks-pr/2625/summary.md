| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-c4c5172c5124fb56c292a1a3acf4eefaba978648.md) | 3,849 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-c4c5172c5124fb56c292a1a3acf4eefaba978648.md) | 15,821 |  1,235,218 |  2,193 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-c4c5172c5124fb56c292a1a3acf4eefaba978648.md) | 1,421 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-c4c5172c5124fb56c292a1a3acf4eefaba978648.md) | 634 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-c4c5172c5124fb56c292a1a3acf4eefaba978648.md) | 930 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-c4c5172c5124fb56c292a1a3acf4eefaba978648.md) | 2,382 |  154,763 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c4c5172c5124fb56c292a1a3acf4eefaba978648

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23820065624)
