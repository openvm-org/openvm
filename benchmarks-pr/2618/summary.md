| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/fibonacci-12cffe87fc45fbf2909942b42ecc8a82e92bb62d.md) | 4,150 |  12,000,265 |  1,352 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/keccak-12cffe87fc45fbf2909942b42ecc8a82e92bb62d.md) | 19,188 |  1,235,218 |  3,378 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/regex-12cffe87fc45fbf2909942b42ecc8a82e92bb62d.md) | 1,594 |  4,136,694 |  519 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/ecrecover-12cffe87fc45fbf2909942b42ecc8a82e92bb62d.md) | 650 |  122,348 |  334 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/pairing-12cffe87fc45fbf2909942b42ecc8a82e92bb62d.md) | 1,063 |  1,745,757 |  345 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/kitchen_sink-12cffe87fc45fbf2909942b42ecc8a82e92bb62d.md) | 3,303 |  154,763 |  723 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/12cffe87fc45fbf2909942b42ecc8a82e92bb62d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23560191496)
