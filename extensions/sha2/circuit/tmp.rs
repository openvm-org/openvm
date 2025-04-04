#[derive(Clone)]
pub struct ShaVmRoundColsRef<'a, T> {
    pub control: ShaVmControlColsRef<'a, T>,
    pub inner: ShaRoundColsRef<'a, T>,
    pub read_aux: &'a MemoryReadAuxCols<T>,
}
impl<'a, T> ShaVmRoundColsRef<'a, T> {
    pub fn from<C: ShaChipConfig>(slice: &'a [T]) -> Self {
        let control_length = <ShaVmControlColsRef<'a, T>>::width::<C>();
        let (control_slice, slice) = slice.split_at(control_length);
        let control_slice = <ShaVmControlColsRef<'a, T>>::from::<C>(control_slice);
        let inner_length = <ShaRoundColsRef<'a, T>>::width::<C>();
        let (inner_slice, slice) = slice.split_at(inner_length);
        let inner_slice = <ShaRoundColsRef<'a, T>>::from::<C>(inner_slice);
        let read_aux_length = <MemoryReadAuxCols<T>>::width();
        let (read_aux_slice, slice) = slice.split_at(read_aux_length);
        Self {
            control: control_slice,
            inner: inner_slice,
            read_aux: {
                use core::borrow::Borrow;
                read_aux_slice.borrow()
            },
        }
    }
    pub const fn width<C: ShaChipConfig>() -> usize {
        0 + <ShaVmControlColsRef<'a, T>>::width::<C>()
            + <ShaRoundColsRef<'a, T>>::width::<C>()
            + <MemoryReadAuxCols<T>>::width()
    }
}
impl<'b, T> ShaVmRoundColsRef<'b, T> {
    pub fn from_mut<'a, C: ShaChipConfig>(other: &'b ShaVmRoundColsRefMut<'a, T>) -> Self {
        Self {
            control: <ShaVmControlColsRef<'b, T>>::from_mut::<C>(&other.control),
            inner: <ShaRoundColsRef<'b, T>>::from_mut::<C>(&other.inner),
            read_aux: other.read_aux,
        }
    }
}
pub struct ShaVmRoundColsRefMut<'a, T> {
    pub control: ShaVmControlColsRefMut<'a, T>,
    pub inner: ShaRoundColsRefMut<'a, T>,
    pub read_aux: &'a mut MemoryReadAuxCols<T>,
}
impl<'a, T> ShaVmRoundColsRefMut<'a, T> {
    pub fn from<C: ShaChipConfig>(slice: &'a mut [T]) -> Self {
        let control_length = <ShaVmControlColsRefMut<'a, T>>::width::<C>();
        let (control_slice, slice) = slice.split_at_mut(control_length);
        let control_slice = <ShaVmControlColsRefMut<'a, T>>::from::<C>(control_slice);
        let inner_length = <ShaRoundColsRefMut<'a, T>>::width::<C>();
        let (inner_slice, slice) = slice.split_at_mut(inner_length);
        let inner_slice = <ShaRoundColsRefMut<'a, T>>::from::<C>(inner_slice);
        let read_aux_length = <MemoryReadAuxCols<T>>::width();
        let (read_aux_slice, slice) = slice.split_at_mut(read_aux_length);
        Self {
            control: control_slice,
            inner: inner_slice,
            read_aux: {
                use core::borrow::BorrowMut;
                read_aux_slice.borrow_mut()
            },
        }
    }
    pub const fn width<C: ShaChipConfig>() -> usize {
        0 + <ShaVmControlColsRefMut<'a, T>>::width::<C>()
            + <ShaRoundColsRefMut<'a, T>>::width::<C>()
            + <MemoryReadAuxCols<T>>::width()
    }
}
#[derive(Clone)]
pub struct ShaVmDigestColsRef<'a, T> {
    pub control: ShaVmControlColsRef<'a, T>,
    pub inner: ShaDigestColsRef<'a, T>,
    pub from_state: &'a ExecutionState<T>,
    pub rd_ptr: &'a T,
    pub rs1_ptr: &'a T,
    pub rs2_ptr: &'a T,
    pub dst_ptr: ndarray::ArrayView1<'a, T>,
    pub src_ptr: ndarray::ArrayView1<'a, T>,
    pub len_data: ndarray::ArrayView1<'a, T>,
    pub register_reads_aux: ndarray::ArrayView1<'a, MemoryReadAuxCols<T>>,
    pub writes_aux_base: &'a MemoryBaseAuxCols<T>,
    pub writes_aux_prev_data: ndarray::ArrayView1<'a, T>,
}
impl<'a, T> ShaVmDigestColsRef<'a, T> {
    pub fn from<C: ShaChipConfig>(slice: &'a [T]) -> Self {
        let control_length = <ShaVmControlColsRef<'a, T>>::width::<C>();
        let (control_slice, slice) = slice.split_at(control_length);
        let control_slice = <ShaVmControlColsRef<'a, T>>::from::<C>(control_slice);
        let inner_length = <ShaDigestColsRef<'a, T>>::width::<C>();
        let (inner_slice, slice) = slice.split_at(inner_length);
        let inner_slice = <ShaDigestColsRef<'a, T>>::from::<C>(inner_slice);
        let from_state_length = <ExecutionState<T>>::width();
        let (from_state_slice, slice) = slice.split_at(from_state_length);
        let rd_ptr_length = 1;
        let (rd_ptr_slice, slice) = slice.split_at(rd_ptr_length);
        let rs1_ptr_length = 1;
        let (rs1_ptr_slice, slice) = slice.split_at(rs1_ptr_length);
        let rs2_ptr_length = 1;
        let (rs2_ptr_slice, slice) = slice.split_at(rs2_ptr_length);
        let (dst_ptr_slice, slice) = slice.split_at(1 * RV32_REGISTER_NUM_LIMBS);
        let dst_ptr_slice =
            ndarray::ArrayView1::from_shape((RV32_REGISTER_NUM_LIMBS), dst_ptr_slice).unwrap();
        let (src_ptr_slice, slice) = slice.split_at(1 * RV32_REGISTER_NUM_LIMBS);
        let src_ptr_slice =
            ndarray::ArrayView1::from_shape((RV32_REGISTER_NUM_LIMBS), src_ptr_slice).unwrap();
        let (len_data_slice, slice) = slice.split_at(1 * RV32_REGISTER_NUM_LIMBS);
        let len_data_slice =
            ndarray::ArrayView1::from_shape((RV32_REGISTER_NUM_LIMBS), len_data_slice).unwrap();
        let (register_reads_aux_slice, slice) =
            slice.split_at(<MemoryReadAuxCols<T>>::width() * SHA_REGISTER_READS);
        let register_reads_aux_slice: &[MemoryReadAuxCols<T>] =
            unsafe { &*(slice as *const [T] as *const [MemoryReadAuxCols<T>]) };
        let register_reads_aux_slice =
            ndarray::ArrayView1::from_shape((SHA_REGISTER_READS), register_reads_aux_slice)
                .unwrap();
        let writes_aux_base_length = <MemoryBaseAuxCols<T>>::width();
        let (writes_aux_base_slice, slice) = slice.split_at(writes_aux_base_length);
        let (writes_aux_prev_data_slice, slice) = slice.split_at(1 * C::WRITE_SIZE);
        let writes_aux_prev_data_slice =
            ndarray::ArrayView1::from_shape((C::WRITE_SIZE), writes_aux_prev_data_slice).unwrap();
        Self {
            control: control_slice,
            inner: inner_slice,
            from_state: {
                use core::borrow::Borrow;
                from_state_slice.borrow()
            },
            rd_ptr: &rd_ptr_slice[0],
            rs1_ptr: &rs1_ptr_slice[0],
            rs2_ptr: &rs2_ptr_slice[0],
            dst_ptr: dst_ptr_slice,
            src_ptr: src_ptr_slice,
            len_data: len_data_slice,
            register_reads_aux: register_reads_aux_slice,
            writes_aux_base: {
                use core::borrow::Borrow;
                writes_aux_base_slice.borrow()
            },
            writes_aux_prev_data: writes_aux_prev_data_slice,
        }
    }
    pub const fn width<C: ShaChipConfig>() -> usize {
        0 + <ShaVmControlColsRef<'a, T>>::width::<C>()
            + <ShaDigestColsRef<'a, T>>::width::<C>()
            + <ExecutionState<T>>::width()
            + 1
            + 1
            + 1
            + 1 * RV32_REGISTER_NUM_LIMBS
            + 1 * RV32_REGISTER_NUM_LIMBS
            + 1 * RV32_REGISTER_NUM_LIMBS
            + <MemoryReadAuxCols<T>>::width() * SHA_REGISTER_READS
            + <MemoryBaseAuxCols<T>>::width()
            + 1 * C::WRITE_SIZE
    }
}
impl<'b, T> ShaVmDigestColsRef<'b, T> {
    pub fn from_mut<'a, C: ShaChipConfig>(other: &'b ShaVmDigestColsRefMut<'a, T>) -> Self {
        Self {
            control: <ShaVmControlColsRef<'b, T>>::from_mut::<C>(&other.control),
            inner: <ShaDigestColsRef<'b, T>>::from_mut::<C>(&other.inner),
            from_state: other.from_state,
            rd_ptr: &other.rd_ptr,
            rs1_ptr: &other.rs1_ptr,
            rs2_ptr: &other.rs2_ptr,
            dst_ptr: other.dst_ptr.view(),
            src_ptr: other.src_ptr.view(),
            len_data: other.len_data.view(),
            register_reads_aux: other.register_reads_aux.view(),
            writes_aux_base: other.writes_aux_base,
            writes_aux_prev_data: other.writes_aux_prev_data.view(),
        }
    }
}
pub struct ShaVmDigestColsRefMut<'a, T> {
    pub control: ShaVmControlColsRefMut<'a, T>,
    pub inner: ShaDigestColsRefMut<'a, T>,
    pub from_state: &'a mut ExecutionState<T>,
    pub rd_ptr: &'a mut T,
    pub rs1_ptr: &'a mut T,
    pub rs2_ptr: &'a mut T,
    pub dst_ptr: ndarray::ArrayViewMut1<'a, T>,
    pub src_ptr: ndarray::ArrayViewMut1<'a, T>,
    pub len_data: ndarray::ArrayViewMut1<'a, T>,
    pub register_reads_aux: ndarray::ArrayViewMut1<'a, MemoryReadAuxCols<T>>,
    pub writes_aux_base: &'a mut MemoryBaseAuxCols<T>,
    pub writes_aux_prev_data: ndarray::ArrayViewMut1<'a, T>,
}
impl<'a, T> ShaVmDigestColsRefMut<'a, T> {
    pub fn from<C: ShaChipConfig>(slice: &'a mut [T]) -> Self {
        let control_length = <ShaVmControlColsRefMut<'a, T>>::width::<C>();
        let (control_slice, slice) = slice.split_at_mut(control_length);
        let control_slice = <ShaVmControlColsRefMut<'a, T>>::from::<C>(control_slice);
        let inner_length = <ShaDigestColsRefMut<'a, T>>::width::<C>();
        let (inner_slice, slice) = slice.split_at_mut(inner_length);
        let inner_slice = <ShaDigestColsRefMut<'a, T>>::from::<C>(inner_slice);
        let from_state_length = <ExecutionState<T>>::width();
        let (from_state_slice, slice) = slice.split_at_mut(from_state_length);
        let rd_ptr_length = 1;
        let (rd_ptr_slice, slice) = slice.split_at_mut(rd_ptr_length);
        let rs1_ptr_length = 1;
        let (rs1_ptr_slice, slice) = slice.split_at_mut(rs1_ptr_length);
        let rs2_ptr_length = 1;
        let (rs2_ptr_slice, slice) = slice.split_at_mut(rs2_ptr_length);
        let (dst_ptr_slice, slice) = slice.split_at_mut(1 * RV32_REGISTER_NUM_LIMBS);
        let dst_ptr_slice =
            ndarray::ArrayViewMut1::from_shape((RV32_REGISTER_NUM_LIMBS), dst_ptr_slice).unwrap();
        let (src_ptr_slice, slice) = slice.split_at_mut(1 * RV32_REGISTER_NUM_LIMBS);
        let src_ptr_slice =
            ndarray::ArrayViewMut1::from_shape((RV32_REGISTER_NUM_LIMBS), src_ptr_slice).unwrap();
        let (len_data_slice, slice) = slice.split_at_mut(1 * RV32_REGISTER_NUM_LIMBS);
        let len_data_slice =
            ndarray::ArrayViewMut1::from_shape((RV32_REGISTER_NUM_LIMBS), len_data_slice).unwrap();
        let (register_reads_aux_slice, slice) =
            slice.split_at_mut(<MemoryReadAuxCols<T>>::width() * SHA_REGISTER_READS);
        let register_reads_aux_slice: &mut [MemoryReadAuxCols<T>] =
            unsafe { &mut *(slice as *mut [T] as *mut [MemoryReadAuxCols<T>]) };
        let register_reads_aux_slice =
            ndarray::ArrayViewMut1::from_shape((SHA_REGISTER_READS), register_reads_aux_slice)
                .unwrap();
        let writes_aux_base_length = <MemoryBaseAuxCols<T>>::width();
        let (writes_aux_base_slice, slice) = slice.split_at_mut(writes_aux_base_length);
        let (writes_aux_prev_data_slice, slice) = slice.split_at_mut(1 * C::WRITE_SIZE);
        let writes_aux_prev_data_slice =
            ndarray::ArrayViewMut1::from_shape((C::WRITE_SIZE), writes_aux_prev_data_slice)
                .unwrap();
        Self {
            control: control_slice,
            inner: inner_slice,
            from_state: {
                use core::borrow::BorrowMut;
                from_state_slice.borrow_mut()
            },
            rd_ptr: &mut rd_ptr_slice[0],
            rs1_ptr: &mut rs1_ptr_slice[0],
            rs2_ptr: &mut rs2_ptr_slice[0],
            dst_ptr: dst_ptr_slice,
            src_ptr: src_ptr_slice,
            len_data: len_data_slice,
            register_reads_aux: register_reads_aux_slice,
            writes_aux_base: {
                use core::borrow::BorrowMut;
                writes_aux_base_slice.borrow_mut()
            },
            writes_aux_prev_data: writes_aux_prev_data_slice,
        }
    }
    pub const fn width<C: ShaChipConfig>() -> usize {
        0 + <ShaVmControlColsRefMut<'a, T>>::width::<C>()
            + <ShaDigestColsRefMut<'a, T>>::width::<C>()
            + <ExecutionState<T>>::width()
            + 1
            + 1
            + 1
            + 1 * RV32_REGISTER_NUM_LIMBS
            + 1 * RV32_REGISTER_NUM_LIMBS
            + 1 * RV32_REGISTER_NUM_LIMBS
            + <MemoryReadAuxCols<T>>::width() * SHA_REGISTER_READS
            + <MemoryBaseAuxCols<T>>::width()
            + 1 * C::WRITE_SIZE
    }
}
impl<
        T,
        const WORD_BITS: usize,
        const WORD_U8S: usize,
        const WORD_U16S: usize,
        const HASH_WORDS: usize,
        const ROUNDS_PER_ROW: usize,
        const ROUNDS_PER_ROW_MINUS_ONE: usize,
        const ROW_VAR_CNT: usize,
        const WRITE_SIZE: usize,
    >
    core::borrow::Borrow<
        ShaVmDigestCols<
            T,
            WORD_BITS,
            WORD_U8S,
            WORD_U16S,
            HASH_WORDS,
            ROUNDS_PER_ROW,
            ROUNDS_PER_ROW_MINUS_ONE,
            ROW_VAR_CNT,
            WRITE_SIZE,
        >,
    > for [T]
{
    fn borrow(
        &self,
    ) -> &ShaVmDigestCols<
        T,
        WORD_BITS,
        WORD_U8S,
        WORD_U16S,
        HASH_WORDS,
        ROUNDS_PER_ROW,
        ROUNDS_PER_ROW_MINUS_ONE,
        ROW_VAR_CNT,
        WRITE_SIZE,
    > {
        if true {
            match (
                &self.len(),
                &ShaVmDigestCols::<
                    T,
                    WORD_BITS,
                    WORD_U8S,
                    WORD_U16S,
                    HASH_WORDS,
                    ROUNDS_PER_ROW,
                    ROUNDS_PER_ROW_MINUS_ONE,
                    ROW_VAR_CNT,
                    WRITE_SIZE,
                >::width(),
            ) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        }
        let (prefix, shorts, _suffix) = unsafe {
            self.align_to::<ShaVmDigestCols<
                T,
                WORD_BITS,
                WORD_U8S,
                WORD_U16S,
                HASH_WORDS,
                ROUNDS_PER_ROW,
                ROUNDS_PER_ROW_MINUS_ONE,
                ROW_VAR_CNT,
                WRITE_SIZE,
            >>()
        };
        if true {
            if !prefix.is_empty() {
                {
                    ::core::panicking::panic_fmt(format_args!("Alignment should match"));
                }
            }
        }
        if true {
            match (&shorts.len(), &1) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        }
        &shorts[0]
    }
}
impl<
        T,
        const WORD_BITS: usize,
        const WORD_U8S: usize,
        const WORD_U16S: usize,
        const HASH_WORDS: usize,
        const ROUNDS_PER_ROW: usize,
        const ROUNDS_PER_ROW_MINUS_ONE: usize,
        const ROW_VAR_CNT: usize,
        const WRITE_SIZE: usize,
    >
    core::borrow::BorrowMut<
        ShaVmDigestCols<
            T,
            WORD_BITS,
            WORD_U8S,
            WORD_U16S,
            HASH_WORDS,
            ROUNDS_PER_ROW,
            ROUNDS_PER_ROW_MINUS_ONE,
            ROW_VAR_CNT,
            WRITE_SIZE,
        >,
    > for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut ShaVmDigestCols<
        T,
        WORD_BITS,
        WORD_U8S,
        WORD_U16S,
        HASH_WORDS,
        ROUNDS_PER_ROW,
        ROUNDS_PER_ROW_MINUS_ONE,
        ROW_VAR_CNT,
        WRITE_SIZE,
    > {
        if true {
            match (
                &self.len(),
                &ShaVmDigestCols::<
                    T,
                    WORD_BITS,
                    WORD_U8S,
                    WORD_U16S,
                    HASH_WORDS,
                    ROUNDS_PER_ROW,
                    ROUNDS_PER_ROW_MINUS_ONE,
                    ROW_VAR_CNT,
                    WRITE_SIZE,
                >::width(),
            ) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        }
        let (prefix, shorts, _suffix) = unsafe {
            self.align_to_mut::<ShaVmDigestCols<
                T,
                WORD_BITS,
                WORD_U8S,
                WORD_U16S,
                HASH_WORDS,
                ROUNDS_PER_ROW,
                ROUNDS_PER_ROW_MINUS_ONE,
                ROW_VAR_CNT,
                WRITE_SIZE,
            >>()
        };
        if true {
            if !prefix.is_empty() {
                {
                    ::core::panicking::panic_fmt(format_args!("Alignment should match"));
                }
            }
        }
        if true {
            match (&shorts.len(), &1) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        }
        &mut shorts[0]
    }
}
impl<
        T,
        const WORD_BITS: usize,
        const WORD_U8S: usize,
        const WORD_U16S: usize,
        const HASH_WORDS: usize,
        const ROUNDS_PER_ROW: usize,
        const ROUNDS_PER_ROW_MINUS_ONE: usize,
        const ROW_VAR_CNT: usize,
        const WRITE_SIZE: usize,
    >
    ShaVmDigestCols<
        T,
        WORD_BITS,
        WORD_U8S,
        WORD_U16S,
        HASH_WORDS,
        ROUNDS_PER_ROW,
        ROUNDS_PER_ROW_MINUS_ONE,
        ROW_VAR_CNT,
        WRITE_SIZE,
    >
{
    pub const fn width() -> usize {
        std::mem::size_of::<
            ShaVmDigestCols<
                u8,
                WORD_BITS,
                WORD_U8S,
                WORD_U16S,
                HASH_WORDS,
                ROUNDS_PER_ROW,
                ROUNDS_PER_ROW_MINUS_ONE,
                ROW_VAR_CNT,
                WRITE_SIZE,
            >,
        >()
    }
}
#[derive(Clone)]
pub struct ShaVmControlColsRef<'a, T> {
    pub len: &'a T,
    pub cur_timestamp: &'a T,
    pub read_ptr: &'a T,
    pub pad_flags: ndarray::ArrayView1<'a, T>,
    pub padding_occurred: &'a T,
}
impl<'a, T> ShaVmControlColsRef<'a, T> {
    pub fn from<C: ShaChipConfig>(slice: &'a [T]) -> Self {
        let len_length = 1;
        let (len_slice, slice) = slice.split_at(len_length);
        let cur_timestamp_length = 1;
        let (cur_timestamp_slice, slice) = slice.split_at(cur_timestamp_length);
        let read_ptr_length = 1;
        let (read_ptr_slice, slice) = slice.split_at(read_ptr_length);
        let (pad_flags_slice, slice) = slice.split_at(1 * 6);
        let pad_flags_slice = ndarray::ArrayView1::from_shape((6), pad_flags_slice).unwrap();
        let padding_occurred_length = 1;
        let (padding_occurred_slice, slice) = slice.split_at(padding_occurred_length);
        Self {
            len: &len_slice[0],
            cur_timestamp: &cur_timestamp_slice[0],
            read_ptr: &read_ptr_slice[0],
            pad_flags: pad_flags_slice,
            padding_occurred: &padding_occurred_slice[0],
        }
    }
    pub const fn width<C: ShaChipConfig>() -> usize {
        0 + 1 + 1 + 1 + 1 * 6 + 1
    }
}
impl<'b, T> ShaVmControlColsRef<'b, T> {
    pub fn from_mut<'a, C: ShaChipConfig>(other: &'b ShaVmControlColsRefMut<'a, T>) -> Self {
        Self {
            len: &other.len,
            cur_timestamp: &other.cur_timestamp,
            read_ptr: &other.read_ptr,
            pad_flags: other.pad_flags.view(),
            padding_occurred: &other.padding_occurred,
        }
    }
}
pub struct ShaVmControlColsRefMut<'a, T> {
    pub len: &'a mut T,
    pub cur_timestamp: &'a mut T,
    pub read_ptr: &'a mut T,
    pub pad_flags: ndarray::ArrayViewMut1<'a, T>,
    pub padding_occurred: &'a mut T,
}
impl<'a, T> ShaVmControlColsRefMut<'a, T> {
    pub fn from<C: ShaChipConfig>(slice: &'a mut [T]) -> Self {
        let len_length = 1;
        let (len_slice, slice) = slice.split_at_mut(len_length);
        let cur_timestamp_length = 1;
        let (cur_timestamp_slice, slice) = slice.split_at_mut(cur_timestamp_length);
        let read_ptr_length = 1;
        let (read_ptr_slice, slice) = slice.split_at_mut(read_ptr_length);
        let (pad_flags_slice, slice) = slice.split_at_mut(1 * 6);
        let pad_flags_slice = ndarray::ArrayViewMut1::from_shape((6), pad_flags_slice).unwrap();
        let padding_occurred_length = 1;
        let (padding_occurred_slice, slice) = slice.split_at_mut(padding_occurred_length);
        Self {
            len: &mut len_slice[0],
            cur_timestamp: &mut cur_timestamp_slice[0],
            read_ptr: &mut read_ptr_slice[0],
            pad_flags: pad_flags_slice,
            padding_occurred: &mut padding_occurred_slice[0],
        }
    }
    pub const fn width<C: ShaChipConfig>() -> usize {
        0 + 1 + 1 + 1 + 1 * 6 + 1
    }
}
