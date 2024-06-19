use super::CPUOptions;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CPUIOCols<T> {
    pub clock_cycle: T,
    pub pc: T,

    pub opcode: T,
    pub op_a: T,
    pub op_b: T,
    pub op_c: T,
    pub as_b: T,
    pub as_c: T,
}

impl<T: Clone> CPUIOCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            clock_cycle: slc[0].clone(),
            pc: slc[1].clone(),
            opcode: slc[2].clone(),
            op_a: slc[3].clone(),
            op_b: slc[4].clone(),
            op_c: slc[5].clone(),
            as_b: slc[6].clone(),
            as_c: slc[7].clone(),
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
            self.as_b.clone(),
            self.as_c.clone(),
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

    pub value: T,
}

impl<T: Clone> MemoryAccessCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            enabled: slc[0].clone(),
            address_space: slc[1].clone(),
            is_immediate: slc[2].clone(),
            is_zero_aux: slc[3].clone(),
            address: slc[4].clone(),
            value: slc[5].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.enabled.clone(),
            self.address_space.clone(),
            self.is_immediate.clone(),
            self.is_zero_aux.clone(),
            self.address.clone(),
            self.value.clone(),
        ]
    }

    pub fn get_width() -> usize {
        6
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CPUAuxCols<T> {
    pub operation_flags: Vec<T>,
    pub read1: MemoryAccessCols<T>,
    pub read2: MemoryAccessCols<T>,
    pub write: MemoryAccessCols<T>,
    pub beq_check: T,
    pub is_equal_aux: T,
}

impl<T: Clone> CPUAuxCols<T> {
    pub fn from_slice(slc: &[T], options: CPUOptions) -> Self {
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

    pub fn get_width(options: CPUOptions) -> usize {
        options.num_operations() + (3 * MemoryAccessCols::<T>::get_width()) + 2
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CPUCols<T> {
    pub io: CPUIOCols<T>,
    pub aux: CPUAuxCols<T>,
}

impl<T: Clone> CPUCols<T> {
    pub fn from_slice(slc: &[T], options: CPUOptions) -> Self {
        let io = CPUIOCols::<T>::from_slice(&slc[..CPUIOCols::<T>::get_width()]);
        let aux = CPUAuxCols::<T>::from_slice(&slc[CPUIOCols::<T>::get_width()..], options);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(options: CPUOptions) -> usize {
        CPUIOCols::<T>::get_width() + CPUAuxCols::<T>::get_width(options)
    }
}
