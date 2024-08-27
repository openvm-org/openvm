
pub struct VmChipIoCols<T> {
    pub opcode: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub start_timestamp: T,
}

pub struct ModularMultiplicationVmCols<T> {
    pub io: VmChipIoCols<T>,
    pub sub: ModularMultiplicationBigIntCols<T>,
    pub enabled: T,
    pub is_div: T,
}

impl<T: Clone> VmChipIoCols<T> {
    fn from_slice(slice: &[T]) -> Self {
        Self {
            opcode: slice[0].clone(),
            a: slice[1].clone(),
            b: slice[2].clone(),
            c: slice[3].clone(),
            d: slice[4].clone(),
            e: slice[5].clone(),
            start_timestamp: slice[6].clone(),
        }
    }

    fn flatten(&self) -> Vec<T> {
        vec![
            self.opcode.clone(),
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.d.clone(),
            self.e.clone(),
            self.start_timestamp.clone(),
        ]
    }

    fn get_width() -> usize {
        7
    }
}

/*impl <T: Clone> ModularMultiplicationVmCols<T> {
    fn from_slice(slice: &[T], air: &ModularMultiplicationVmAir) {
        let mut start = 0;
        let mut end = 0;

        end += VmChipIoCols::<T>::get_width();
        let io = VmChipIoCols::from_slice(&slice[start..end]);
        start = end;

        end += ModularMultiplicationBigIntCols::<T>::get_width(&air.air);
        let sub = ModularMultiplicationBigIntCols::from_slice(&slice[start..end], &air.air);
        start = end;

        let enabled = slice[start].clone();
        //let is_div =
    }
}*/
