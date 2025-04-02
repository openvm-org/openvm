#[derive(Clone, Copy, Debug)]
pub enum Term {
    Cell(usize),
    Constant(isize),
}

#[derive(Clone, Debug)]
pub struct AirExpression {
    pub sum: Vec<Vec<Term>>,
}

impl AirExpression {
    pub fn plus(&self, other: &Self) -> Self {
        let mut result = self.sum.clone();
        result.extend(other.sum.clone());
        Self { sum: result }
    }

    pub fn minus(&self, other: &Self) -> Self {
        let mut result = self.sum.clone();
        result.extend(other.sum.iter().map(|product| {
            let mut result = product.clone();
            result.push(Term::Constant(-1));
            result
        }));
        Self { sum: result }
    }

    pub fn times(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        for left in self.sum.iter() {
            for right in other.sum.iter() {
                let mut product = left.clone();
                product.extend(right.clone());
                result.push(product);
            }
        }
        Self { sum: result }
    }

    pub fn negate(&self) -> Self {
        Self::zero().minus(self)
    }

    pub fn constant(value: isize) -> Self {
        Self {
            sum: vec![vec![Term::Constant(value)]],
        }
    }

    pub fn zero() -> Self {
        Self { sum: vec![] }
    }

    pub fn one() -> Self {
        Self { sum: vec![vec![]] }
    }

    pub fn single_cell(cell: usize) -> Self {
        Self {
            sum: vec![vec![Term::Cell(cell)]],
        }
    }
}

pub struct Constraint {
    pub left: AirExpression,
    pub right: AirExpression,
}

#[derive(Clone, Copy)]
pub enum Bus {
    Function,
    Reference,
    Array,
}

#[derive(Clone, Copy)]
pub enum Direction {
    Send,
    Receive,
}

pub struct Interaction {
    pub bus: Bus,
    pub direction: Direction,
    pub multiplicity: AirExpression,
    pub fields: Vec<AirExpression>,
}
