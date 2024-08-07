# Benchmark for Predicate

| Total Cells | Total Prove (ms) | Main Trace Gen (ms) | Perm Trace Gen (ms) |
|-------------|------------------|---------------------|---------------------|
| 19_726_336  | 710.00           | 212.00              | 8.15                |

| Calc Quotient Values (ms) | Rest of Prove (ms) |
|---------------------------|---------------------|
| 29.20                     | 460.65              |

### AIR metrics

| Name                   | Rows   | Cells      | Prep Cols | Main Cols | Perm Cols |
|------------------------|--------|------------|-----------|-----------|-----------|
| PageIndexScanInputAir  | 32_768 | 10_354_688 | 0         | [97, 147] | [72]      |
| PageIndexScanOutputAir | 32_768 | 8_716_288  | 0         | [97, 97]  | [72]      |
| RangeCheckerGateAir    | 65_536 | 655_360    | 0         | [2]       | [8]       |

Commit: [f0b6a873a1874531137376ce34f679f9e89a10c8](https://github.com/axiom-crypto/afs-prototype/commit/f0b6a873a1874531137376ce34f679f9e89a10c8)
AWS Instance Type: [r7g.8xlarge](https://instances.vantage.sh/aws/ec2/r7g.8xlarge)
[Benchmark Workflow](https://github.com/axiom-crypto/afs-prototype/actions/runs/10292200820)
