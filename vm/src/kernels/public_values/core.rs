use std::sync::Mutex;

use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::{
    instruction::Instruction, PublishOpcode, PublishOpcode::PUBLISH, UsizeOpcode,
};
use p3_air::{AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, MinimalInstruction,
        Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
    },
    kernels::public_values::columns::PublicValuesCoreColsView,
};
pub(crate) type AdapterInterface<F> = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 0, 1, 1>;
pub(crate) type AdapterInterfaceReads<F> = <AdapterInterface<F> as VmAdapterInterface<F>>::Reads;

#[derive(Copy, Clone, Debug)]
pub struct PublicValuesCoreAir {
    /// Number of custom public values to publish.
    pub num_custom_pvs: usize,
    offset: usize,
}

impl PublicValuesCoreAir {
    pub fn new(num_custom_pvs: usize, offset: usize) -> Self {
        Self {
            num_custom_pvs,
            offset,
        }
    }
}

impl<F: Field> BaseAir<F> for PublicValuesCoreAir {
    fn width(&self) -> usize {
        3 + self.num_custom_pvs
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for PublicValuesCoreAir {
    fn num_public_values(&self) -> usize {
        self.num_custom_pvs
    }
}

impl<AB: InteractionBuilder + AirBuilderWithPublicValues> VmCoreAir<AB, AdapterInterface<AB::Expr>>
    for PublicValuesCoreAir
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, AdapterInterface<AB::Expr>> {
        let cols = PublicValuesCoreColsView::<_, &AB::Var>::borrow(local_core);
        debug_assert_eq!(cols.width(), BaseAir::<AB::F>::width(self));
        let is_valid = *cols.is_valid;
        let value = *cols.value;
        let index = *cols.index;

        let mut sum_flags = AB::Expr::ZERO;
        let mut match_public_value_index = AB::Expr::ZERO;
        let mut match_public_value = AB::Expr::ZERO;
        for (i, &&flag) in cols.custom_pv_flags.iter().enumerate() {
            builder.assert_bool(flag.into());
            sum_flags += flag.into();
            match_public_value_index += flag.into() * AB::F::from_canonical_usize(i);
            match_public_value += flag.into() * builder.public_values()[i].into();
        }

        let mut when_publish = builder.when(is_valid);
        when_publish.assert_one(sum_flags);
        when_publish.assert_eq(index, match_public_value_index);
        when_publish.assert_eq(value, match_public_value);

        AdapterAirContext {
            to_pc: None,
            reads: [[value.into()], [index.into()]],
            writes: [],
            instruction: MinimalInstruction {
                is_valid: is_valid.into(),
                opcode: AB::Expr::from_canonical_usize(PUBLISH.as_usize() + self.offset),
            },
        }
    }
}

#[derive(Debug)]
pub struct PublicValuesRecord<F> {
    value: F,
    index: F,
}

/// ATTENTION: If a specific public value is not provided, a default 0 will be used when generating
/// the proof but in the perspective of constraints, it could be any value.
pub struct PublicValuesCoreChip<F> {
    air: PublicValuesCoreAir,
    // Mutex is to make the struct Sync. But it actually won't be accessed by multiple threads.
    custom_pvs: Mutex<Vec<Option<F>>>,
}

impl<F: PrimeField32> PublicValuesCoreChip<F> {
    pub fn new(num_custom_pvs: usize, offset: usize) -> Self {
        Self {
            air: PublicValuesCoreAir {
                num_custom_pvs,
                offset,
            },
            custom_pvs: Mutex::new(vec![None; num_custom_pvs]),
        }
    }
    pub fn get_custom_public_values(&self) -> Vec<Option<F>> {
        self.custom_pvs.lock().unwrap().clone()
    }
}

impl<F: PrimeField32> VmCoreChip<F, AdapterInterface<F>> for PublicValuesCoreChip<F> {
    type Record = PublicValuesRecord<F>;
    type Air = PublicValuesCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        reads: AdapterInterfaceReads<F>,
    ) -> Result<(AdapterRuntimeContext<F, AdapterInterface<F>>, Self::Record)> {
        let [[value], [index]] = reads;
        {
            let idx: usize = index.as_canonical_u32() as usize;
            let mut custom_pvs = self.custom_pvs.lock().unwrap();

            if custom_pvs[idx].is_none() {
                custom_pvs[idx] = Some(value);
            } else {
                // Not a hard constraint violation when publishing the same value twice but the
                // program should avoid that.
                panic!("Custom public value {} already set", idx);
            }
        }
        let output = AdapterRuntimeContext {
            to_pc: None,
            writes: [],
        };
        let record = Self::Record { value, index };
        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", PublishOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let mut cols = PublicValuesCoreColsView::<_, &mut F>::borrow_mut(row_slice);
        debug_assert_eq!(cols.width(), BaseAir::<F>::width(&self.air));
        *cols.is_valid = F::ONE;
        *cols.value = record.value;
        *cols.index = record.index;
        let idx: usize = record.index.as_canonical_u32() as usize;
        // Assumption: row_slice is initialized with 0s.
        *cols.custom_pv_flags[idx] = F::ONE;
    }

    fn generate_public_values(&self) -> Vec<F> {
        self.get_custom_public_values()
            .into_iter()
            .map(|x| x.unwrap_or(F::ZERO))
            .collect()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
