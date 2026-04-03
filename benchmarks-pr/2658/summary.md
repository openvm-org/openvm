| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/fibonacci-a718dea835acdd25e70cb75860ed9d824cd162f0.md) | 3,823 |  12,000,265 |  940 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/keccak-a718dea835acdd25e70cb75860ed9d824cd162f0.md) | 15,607 |  1,235,218 |  2,181 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/regex-a718dea835acdd25e70cb75860ed9d824cd162f0.md) | 1,425 |  4,136,694 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/ecrecover-a718dea835acdd25e70cb75860ed9d824cd162f0.md) | 646 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/pairing-a718dea835acdd25e70cb75860ed9d824cd162f0.md) | 921 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2658/kitchen_sink-a718dea835acdd25e70cb75860ed9d824cd162f0.md) | 2,368 |  154,763 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a718dea835acdd25e70cb75860ed9d824cd162f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23958751404)
