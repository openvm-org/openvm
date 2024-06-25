use super::CpuOptions;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuIoCols<T> {
    pub clock_cycle: T,
    pub pc: T,

    pub opcode: T,
    pub op_a: T,
    pub op_b: T,
    pub op_c: T,
    pub d: T,
    pub e: T,
}

impl<T: Clone> CpuIoCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            clock_cycle: slc[0].clone(),
            pc: slc[1].clone(),
            opcode: slc[2].clone(),
            op_a: slc[3].clone(),
            op_b: slc[4].clone(),
            op_c: slc[5].clone(),
            d: slc[6].clone(),
            e: slc[7].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.clock_cycle.clone(),
            self.pc.clone(),
            self.opcode.clone(),
            self.op_a.clone(),
            self.op_b.clone(),
            self.op_c.clone(),
            self.d.clone(),
            self.e.clone(),
        ]
    }

    pub fn get_width() -> usize {
        8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAccessCols<T> {
    pub enabled: T,

    pub address_space: T,
    pub is_immediate: T,
    pub is_zero_aux: T,

    pub address: T,

    pub data: T,
}

impl<T: Clone> MemoryAccessCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            enabled: slc[0].clone(),
            address_space: slc[1].clone(),
            is_immediate: slc[2].clone(),
            is_zero_aux: slc[3].clone(),
            address: slc[4].clone(),
            data: slc[5].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.enabled.clone(),
            self.address_space.clone(),
            self.is_immediate.clone(),
            self.is_zero_aux.clone(),
            self.address.clone(),
            self.data.clone(),
        ]
    }

    pub fn get_width() -> usize {
        6
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuAuxCols<T> {
    pub operation_flags: Vec<T>,
    pub read1: MemoryAccessCols<T>,
    pub read2: MemoryAccessCols<T>,
    pub write: MemoryAccessCols<T>,
    pub beq_check: T,
    pub is_equal_aux: T,
}

impl<T: Clone> CpuAuxCols<T> {
    pub fn from_slice(slc: &[T], options: CpuOptions) -> Self {
        let mut start = 0;
        let mut end = options.num_operations();
        let operation_flags = slc[start..end].to_vec();

        start = end;
        end += MemoryAccessCols::<T>::get_width();
        let read1 = MemoryAccessCols::<T>::from_slice(&slc[start..end]);

        start = end;
        end += MemoryAccessCols::<T>::get_width();
        let read2 = MemoryAccessCols::<T>::from_slice(&slc[start..end]);

        start = end;
        end += MemoryAccessCols::<T>::get_width();
        let write = MemoryAccessCols::<T>::from_slice(&slc[start..end]);

        let beq_check = slc[end].clone();
        let is_equal_aux = slc[end + 1].clone();

        Self {
            operation_flags,
            read1,
            read2,
            write,
            beq_check,
            is_equal_aux,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.operation_flags.clone();
        flattened.extend(self.read1.flatten());
        flattened.extend(self.read2.flatten());
        flattened.extend(self.write.flatten());
        flattened.push(self.beq_check.clone());
        flattened.push(self.is_equal_aux.clone());
        flattened
    }

    pub fn get_width(options: CpuOptions) -> usize {
        options.num_operations() + (3 * MemoryAccessCols::<T>::get_width()) + 2
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuCols<T> {
    pub io: CpuIoCols<T>,
    pub aux: CpuAuxCols<T>,
}

impl<T: Clone> CpuCols<T> {
    pub fn from_slice(slc: &[T], options: CpuOptions) -> Self {
        let io = CpuIoCols::<T>::from_slice(&slc[..CpuIoCols::<T>::get_width()]);
        let aux = CpuAuxCols::<T>::from_slice(&slc[CpuIoCols::<T>::get_width()..], options);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(options: CpuOptions) -> usize {
        CpuIoCols::<T>::get_width() + CpuAuxCols::<T>::get_width(options)
    }
}
