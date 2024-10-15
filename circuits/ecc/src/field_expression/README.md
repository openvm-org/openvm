### Example usage

Ec Double chip:

```     
let x1 = ExprBuilder::new_input(builder.clone());
let y1 = ExprBuilder::new_input(builder.clone());
let nom = (x1.clone() * x1.clone()).scalar_mul(3);
let denom = y1.scalar_mul(2);
let lambda = nom / denom;
let mut x3 = lambda.clone() * lambda.clone() - x1.clone() - x1.clone();
x3.save();
let mut y3 = lambda * (x1 - x3) - y1;
y3.save();
```

### TODO
- [ ] auto save on add/sub/mul. Need to track the max_overflow_limbs.
- [ ] select op: `select(flag, a, b) -> a if flag else b`.