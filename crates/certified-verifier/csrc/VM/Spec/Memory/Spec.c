// Lean compiler output
// Module: VM.Spec.Memory.Spec
// Imports: public import Init public meta import Init public import Fundamentals.Spec.Poseidon2 public import VM.Spec.Machine.BabyBear public import VM.Spec.Machine.Memory public import VM.Spec.Memory.Events public import VM.Spec.Memory.Record public import VM.Spec.Airs.System.MemoryMerkleAir.View public import VM.Spec.Airs.System.PersistentBoundaryAir.View
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lp_mathlib_ZMod_val(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lp_mathlib_ZMod_decidableEq(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_decidableEq___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Array_instDecidableEqImpl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_instField___redArg(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_compressWithCapacity(lean_object*, lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_commRing(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_addressHeight;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_overallHeight;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_compressDigest(lean_object*, lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_leafDigest(lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafDigest(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafDigest___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_merkleNode(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_merkleNode___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryImageMerkleRoot(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memorySubtreeDigest(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memorySubtreeDigest___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_nodeLabel(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_nodeLabel___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryLeafValues(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryLeafValues___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default___closed__0;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode;
static const lean_closure_object lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ZMod_decidableEq___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2013265921) << 1) | 1))} };
static const lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq___closed__0_value;
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_parentNodeOfExpansion(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_parentNodeOfExpansion___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_leftChildNodeOfExpansion(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_rightChildNodeOfExpansion(lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord(lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord___closed__0;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryNodeOfEvent(lean_object*);
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_addressHeight(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(26u);
return v___x_1_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_overallHeight(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(29u);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_compressDigest(lean_object* v_left_3_, lean_object* v_right_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_fst_6_; 
v___x_5_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_compressWithCapacity(v_left_3_, v_right_4_);
v_fst_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_fst_6_);
lean_dec_ref(v___x_5_);
return v_fst_6_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_unsigned_to_nat(2013265921u);
v___x_8_ = lp_mathlib_ZMod_instField___redArg(v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0);
v___x_10_ = lp_mathlib_Field_toSemifield___redArg(v___x_9_);
return v___x_10_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest(void){
_start:
{
lean_object* v___x_11_; lean_object* v_toCommSemiring_12_; lean_object* v___x_13_; lean_object* v_toZero_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_11_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_12_ = lean_ctor_get(v___x_11_, 0);
lean_inc_ref(v_toCommSemiring_12_);
v___x_13_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_12_);
v_toZero_14_ = lean_ctor_get(v___x_13_, 1);
lean_inc_n(v_toZero_14_, 8);
lean_dec_ref(v___x_13_);
v___x_15_ = lean_unsigned_to_nat(8u);
v___x_16_ = lean_mk_empty_array_with_capacity(v___x_15_);
v___x_17_ = lean_array_push(v___x_16_, v_toZero_14_);
v___x_18_ = lean_array_push(v___x_17_, v_toZero_14_);
v___x_19_ = lean_array_push(v___x_18_, v_toZero_14_);
v___x_20_ = lean_array_push(v___x_19_, v_toZero_14_);
v___x_21_ = lean_array_push(v___x_20_, v_toZero_14_);
v___x_22_ = lean_array_push(v___x_21_, v_toZero_14_);
v___x_23_ = lean_array_push(v___x_22_, v_toZero_14_);
v___x_24_ = lean_array_push(v___x_23_, v_toZero_14_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_leafDigest(lean_object* v_values_25_){
_start:
{
lean_object* v___x_26_; lean_object* v_toCommSemiring_27_; lean_object* v___x_28_; lean_object* v_toZero_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_26_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_27_ = lean_ctor_get(v___x_26_, 0);
lean_inc_ref(v_toCommSemiring_27_);
v___x_28_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_27_);
v_toZero_29_ = lean_ctor_get(v___x_28_, 1);
lean_inc_n(v_toZero_29_, 8);
lean_dec_ref(v___x_28_);
v___x_30_ = lean_unsigned_to_nat(8u);
v___x_31_ = lean_mk_empty_array_with_capacity(v___x_30_);
v___x_32_ = lean_array_push(v___x_31_, v_toZero_29_);
v___x_33_ = lean_array_push(v___x_32_, v_toZero_29_);
v___x_34_ = lean_array_push(v___x_33_, v_toZero_29_);
v___x_35_ = lean_array_push(v___x_34_, v_toZero_29_);
v___x_36_ = lean_array_push(v___x_35_, v_toZero_29_);
v___x_37_ = lean_array_push(v___x_36_, v_toZero_29_);
v___x_38_ = lean_array_push(v___x_37_, v_toZero_29_);
v___x_39_ = lean_array_push(v___x_38_, v_toZero_29_);
v___x_40_ = lp_openvm_x2dfv_VM_Spec_Memory_compressDigest(v_values_25_, v___x_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_41_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__0);
v___x_42_ = lp_mathlib_Field_toDivisionRing___redArg(v___x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues(lean_object* v_memory_43_, lean_object* v_addrSpace_44_, lean_object* v_leafLabel_45_){
_start:
{
lean_object* v___x_46_; lean_object* v_toRing_47_; lean_object* v___x_48_; lean_object* v_toAddMonoidWithOne_49_; lean_object* v_toNatCast_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_46_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0);
v_toRing_47_ = lean_ctor_get(v___x_46_, 0);
lean_inc_ref(v_toRing_47_);
v___x_48_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_47_);
v_toAddMonoidWithOne_49_ = lean_ctor_get(v___x_48_, 1);
lean_inc_ref(v_toAddMonoidWithOne_49_);
lean_dec_ref(v___x_48_);
v_toNatCast_50_ = lean_ctor_get(v_toAddMonoidWithOne_49_, 0);
lean_inc_n(v_toNatCast_50_, 9);
lean_dec_ref(v_toAddMonoidWithOne_49_);
v___x_51_ = lean_unsigned_to_nat(8u);
v___x_52_ = lean_unsigned_to_nat(1u);
v___x_53_ = lean_nat_add(v_addrSpace_44_, v___x_52_);
v___x_54_ = lean_apply_1(v_toNatCast_50_, v___x_53_);
v___x_55_ = lean_nat_mul(v_leafLabel_45_, v___x_51_);
lean_inc(v___x_55_);
v___x_56_ = lean_apply_1(v_toNatCast_50_, v___x_55_);
lean_inc_ref_n(v_memory_43_, 7);
lean_inc_n(v___x_54_, 7);
v___x_57_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_56_);
v___x_58_ = lean_nat_add(v___x_55_, v___x_52_);
v___x_59_ = lean_apply_1(v_toNatCast_50_, v___x_58_);
v___x_60_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_59_);
v___x_61_ = lean_unsigned_to_nat(2u);
v___x_62_ = lean_nat_add(v___x_55_, v___x_61_);
v___x_63_ = lean_apply_1(v_toNatCast_50_, v___x_62_);
v___x_64_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_63_);
v___x_65_ = lean_unsigned_to_nat(3u);
v___x_66_ = lean_nat_add(v___x_55_, v___x_65_);
v___x_67_ = lean_apply_1(v_toNatCast_50_, v___x_66_);
v___x_68_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_67_);
v___x_69_ = lean_unsigned_to_nat(4u);
v___x_70_ = lean_nat_add(v___x_55_, v___x_69_);
v___x_71_ = lean_apply_1(v_toNatCast_50_, v___x_70_);
v___x_72_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_71_);
v___x_73_ = lean_unsigned_to_nat(5u);
v___x_74_ = lean_nat_add(v___x_55_, v___x_73_);
v___x_75_ = lean_apply_1(v_toNatCast_50_, v___x_74_);
v___x_76_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_75_);
v___x_77_ = lean_unsigned_to_nat(6u);
v___x_78_ = lean_nat_add(v___x_55_, v___x_77_);
v___x_79_ = lean_apply_1(v_toNatCast_50_, v___x_78_);
v___x_80_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_79_);
v___x_81_ = lean_unsigned_to_nat(7u);
v___x_82_ = lean_nat_add(v___x_55_, v___x_81_);
lean_dec(v___x_55_);
v___x_83_ = lean_apply_1(v_toNatCast_50_, v___x_82_);
v___x_84_ = lean_apply_2(v_memory_43_, v___x_54_, v___x_83_);
v___x_85_ = lean_mk_empty_array_with_capacity(v___x_51_);
v___x_86_ = lean_array_push(v___x_85_, v___x_57_);
v___x_87_ = lean_array_push(v___x_86_, v___x_60_);
v___x_88_ = lean_array_push(v___x_87_, v___x_64_);
v___x_89_ = lean_array_push(v___x_88_, v___x_68_);
v___x_90_ = lean_array_push(v___x_89_, v___x_72_);
v___x_91_ = lean_array_push(v___x_90_, v___x_76_);
v___x_92_ = lean_array_push(v___x_91_, v___x_80_);
v___x_93_ = lean_array_push(v___x_92_, v___x_84_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___boxed(lean_object* v_memory_94_, lean_object* v_addrSpace_95_, lean_object* v_leafLabel_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues(v_memory_94_, v_addrSpace_95_, v_leafLabel_96_);
lean_dec(v_leafLabel_96_);
lean_dec(v_addrSpace_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafDigest(lean_object* v_memory_98_, lean_object* v_label_99_){
_start:
{
lean_object* v_leavesPerAddressSpace_100_; lean_object* v___x_101_; lean_object* v_addrSpace_102_; lean_object* v_leafLabel_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v_leavesPerAddressSpace_100_ = lean_unsigned_to_nat(67108864u);
v___x_101_ = lean_unsigned_to_nat(26u);
v_addrSpace_102_ = lean_nat_shiftr(v_label_99_, v___x_101_);
v_leafLabel_103_ = lean_nat_mod(v_label_99_, v_leavesPerAddressSpace_100_);
v___x_104_ = lean_unsigned_to_nat(8u);
v___x_105_ = lean_unsigned_to_nat(1u);
v___x_106_ = lean_nat_add(v_addrSpace_102_, v___x_105_);
lean_dec(v_addrSpace_102_);
v___x_107_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_106_);
v___x_108_ = lean_nat_mul(v_leafLabel_103_, v___x_104_);
lean_dec(v_leafLabel_103_);
lean_inc(v___x_108_);
v___x_109_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_108_);
lean_inc_ref_n(v_memory_98_, 7);
lean_inc_n(v___x_107_, 7);
v___x_110_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_109_);
v___x_111_ = lean_nat_add(v___x_108_, v___x_105_);
v___x_112_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_111_);
v___x_113_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_112_);
v___x_114_ = lean_unsigned_to_nat(2u);
v___x_115_ = lean_nat_add(v___x_108_, v___x_114_);
v___x_116_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_115_);
v___x_117_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_116_);
v___x_118_ = lean_unsigned_to_nat(3u);
v___x_119_ = lean_nat_add(v___x_108_, v___x_118_);
v___x_120_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_119_);
v___x_121_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_120_);
v___x_122_ = lean_unsigned_to_nat(4u);
v___x_123_ = lean_nat_add(v___x_108_, v___x_122_);
v___x_124_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_123_);
v___x_125_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_124_);
v___x_126_ = lean_unsigned_to_nat(5u);
v___x_127_ = lean_nat_add(v___x_108_, v___x_126_);
v___x_128_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_127_);
v___x_129_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_128_);
v___x_130_ = lean_unsigned_to_nat(6u);
v___x_131_ = lean_nat_add(v___x_108_, v___x_130_);
v___x_132_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_131_);
v___x_133_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_132_);
v___x_134_ = lean_unsigned_to_nat(7u);
v___x_135_ = lean_nat_add(v___x_108_, v___x_134_);
lean_dec(v___x_108_);
v___x_136_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_135_);
v___x_137_ = lean_apply_2(v_memory_98_, v___x_107_, v___x_136_);
v___x_138_ = lean_mk_empty_array_with_capacity(v___x_104_);
v___x_139_ = lean_array_push(v___x_138_, v___x_110_);
v___x_140_ = lean_array_push(v___x_139_, v___x_113_);
v___x_141_ = lean_array_push(v___x_140_, v___x_117_);
v___x_142_ = lean_array_push(v___x_141_, v___x_121_);
v___x_143_ = lean_array_push(v___x_142_, v___x_125_);
v___x_144_ = lean_array_push(v___x_143_, v___x_129_);
v___x_145_ = lean_array_push(v___x_144_, v___x_133_);
v___x_146_ = lean_array_push(v___x_145_, v___x_137_);
v___x_147_ = lp_openvm_x2dfv_VM_Spec_Memory_leafDigest(v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafDigest___boxed(lean_object* v_memory_148_, lean_object* v_label_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafDigest(v_memory_148_, v_label_149_);
lean_dec(v_label_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_merkleNode(lean_object* v_x_151_, lean_object* v_x_152_, lean_object* v_x_153_){
_start:
{
lean_object* v_zero_154_; uint8_t v_isZero_155_; 
v_zero_154_ = lean_unsigned_to_nat(0u);
v_isZero_155_ = lean_nat_dec_eq(v_x_151_, v_zero_154_);
if (v_isZero_155_ == 1)
{
lean_object* v___x_156_; 
v___x_156_ = lean_apply_1(v_x_152_, v_x_153_);
return v___x_156_;
}
else
{
lean_object* v_one_157_; lean_object* v_n_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v_one_157_ = lean_unsigned_to_nat(1u);
v_n_158_ = lean_nat_sub(v_x_151_, v_one_157_);
v___x_159_ = lean_unsigned_to_nat(2u);
v___x_160_ = lean_nat_mul(v___x_159_, v_x_153_);
lean_dec(v_x_153_);
lean_inc(v___x_160_);
lean_inc_ref(v_x_152_);
v___x_161_ = lp_openvm_x2dfv_VM_Spec_Memory_merkleNode(v_n_158_, v_x_152_, v___x_160_);
v___x_162_ = lean_nat_add(v___x_160_, v_one_157_);
lean_dec(v___x_160_);
v___x_163_ = lp_openvm_x2dfv_VM_Spec_Memory_merkleNode(v_n_158_, v_x_152_, v___x_162_);
lean_dec(v_n_158_);
v___x_164_ = lp_openvm_x2dfv_VM_Spec_Memory_compressDigest(v___x_161_, v___x_163_);
return v___x_164_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_merkleNode___boxed(lean_object* v_x_165_, lean_object* v_x_166_, lean_object* v_x_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_openvm_x2dfv_VM_Spec_Memory_merkleNode(v_x_165_, v_x_166_, v_x_167_);
lean_dec(v_x_165_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memoryImageMerkleRoot(lean_object* v_memory_169_){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_170_ = lean_unsigned_to_nat(29u);
v___x_171_ = lean_alloc_closure((void*)(lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafDigest___boxed), 2, 1);
lean_closure_set(v___x_171_, 0, v_memory_169_);
v___x_172_ = lean_unsigned_to_nat(0u);
v___x_173_ = lp_openvm_x2dfv_VM_Spec_Memory_merkleNode(v___x_170_, v___x_171_, v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memorySubtreeDigest(lean_object* v_memory_174_, lean_object* v_height_175_, lean_object* v_label_176_){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = lean_alloc_closure((void*)(lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafDigest___boxed), 2, 1);
lean_closure_set(v___x_177_, 0, v_memory_174_);
v___x_178_ = lp_openvm_x2dfv_VM_Spec_Memory_merkleNode(v_height_175_, v___x_177_, v_label_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_memorySubtreeDigest___boxed(lean_object* v_memory_179_, lean_object* v_height_180_, lean_object* v_label_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_openvm_x2dfv_VM_Spec_Memory_memorySubtreeDigest(v_memory_179_, v_height_180_, v_label_181_);
lean_dec(v_height_180_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_nodeLabel(lean_object* v_height_183_, lean_object* v_asLabel_184_, lean_object* v_addressLabel_185_){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_186_ = lean_unsigned_to_nat(2013265921u);
v___x_187_ = lp_mathlib_ZMod_val(v___x_186_, v_asLabel_184_);
v___x_188_ = lean_unsigned_to_nat(2u);
v___x_189_ = lean_unsigned_to_nat(26u);
v___x_190_ = lean_nat_sub(v___x_189_, v_height_183_);
v___x_191_ = lean_nat_pow(v___x_188_, v___x_190_);
lean_dec(v___x_190_);
v___x_192_ = lean_nat_mul(v___x_187_, v___x_191_);
lean_dec(v___x_191_);
lean_dec(v___x_187_);
v___x_193_ = lp_mathlib_ZMod_val(v___x_186_, v_addressLabel_185_);
v___x_194_ = lean_nat_add(v___x_192_, v___x_193_);
lean_dec(v___x_193_);
lean_dec(v___x_192_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_nodeLabel___boxed(lean_object* v_height_195_, lean_object* v_asLabel_196_, lean_object* v_addressLabel_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_openvm_x2dfv_VM_Spec_Memory_nodeLabel(v_height_195_, v_asLabel_196_, v_addressLabel_197_);
lean_dec(v_addressLabel_197_);
lean_dec(v_asLabel_196_);
lean_dec(v_height_195_);
return v_res_198_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0(void){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = lean_unsigned_to_nat(2u);
v___x_200_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel(lean_object* v_event_201_){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v_toRing_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v_toAddMonoidWithOne_207_; lean_object* v_toSub_208_; lean_object* v_height_209_; lean_object* v_height__section_210_; lean_object* v_parent__as__label_211_; lean_object* v_parent__address__label_212_; lean_object* v_toOne_213_; lean_object* v___x_214_; lean_object* v_toCommSemiring_215_; lean_object* v___x_216_; lean_object* v_toMul_217_; lean_object* v_toAdd_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_202_ = lean_unsigned_to_nat(2013265921u);
v___x_203_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0);
v_toRing_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc_ref(v_toRing_204_);
v___x_205_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_204_);
v___x_206_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_205_);
v_toAddMonoidWithOne_207_ = lean_ctor_get(v___x_205_, 1);
lean_inc_ref(v_toAddMonoidWithOne_207_);
lean_dec_ref(v___x_205_);
v_toSub_208_ = lean_ctor_get(v___x_206_, 2);
lean_inc_n(v_toSub_208_, 2);
lean_dec_ref(v___x_206_);
v_height_209_ = lean_ctor_get(v_event_201_, 1);
lean_inc(v_height_209_);
v_height__section_210_ = lean_ctor_get(v_event_201_, 2);
lean_inc_n(v_height__section_210_, 2);
v_parent__as__label_211_ = lean_ctor_get(v_event_201_, 3);
lean_inc(v_parent__as__label_211_);
v_parent__address__label_212_ = lean_ctor_get(v_event_201_, 4);
lean_inc(v_parent__address__label_212_);
lean_dec_ref(v_event_201_);
v_toOne_213_ = lean_ctor_get(v_toAddMonoidWithOne_207_, 2);
lean_inc_n(v_toOne_213_, 2);
lean_dec_ref(v_toAddMonoidWithOne_207_);
v___x_214_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_215_ = lean_ctor_get(v___x_214_, 0);
lean_inc_ref(v_toCommSemiring_215_);
v___x_216_ = lp_mathlib_instDistribOfSemiring___redArg(v_toCommSemiring_215_);
v_toMul_217_ = lean_ctor_get(v___x_216_, 0);
lean_inc_n(v_toMul_217_, 2);
v_toAdd_218_ = lean_ctor_get(v___x_216_, 1);
lean_inc(v_toAdd_218_);
lean_dec_ref(v___x_216_);
v___x_219_ = lean_apply_2(v_toSub_208_, v_height_209_, v_toOne_213_);
v___x_220_ = lp_mathlib_ZMod_val(v___x_202_, v___x_219_);
lean_dec(v___x_219_);
v___x_221_ = lean_apply_2(v_toAdd_218_, v_toOne_213_, v_height__section_210_);
v___x_222_ = lean_apply_2(v_toMul_217_, v_parent__as__label_211_, v___x_221_);
v___x_223_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0);
v___x_224_ = lean_apply_2(v_toSub_208_, v___x_223_, v_height__section_210_);
v___x_225_ = lean_apply_2(v_toMul_217_, v_parent__address__label_212_, v___x_224_);
v___x_226_ = lp_openvm_x2dfv_VM_Spec_Memory_nodeLabel(v___x_220_, v___x_222_, v___x_225_);
lean_dec(v___x_225_);
lean_dec(v___x_222_);
lean_dec(v___x_220_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(lean_object* v_record_227_, lean_object* v_idx_228_){
_start:
{
lean_object* v_data_229_; lean_object* v___x_230_; 
v_data_229_ = lean_ctor_get(v_record_227_, 2);
v___x_230_ = l_List_get_x3fInternal___redArg(v_data_229_, v_idx_228_);
if (lean_obj_tag(v___x_230_) == 0)
{
lean_object* v___x_231_; lean_object* v_toCommSemiring_232_; lean_object* v___x_233_; lean_object* v_toZero_234_; 
v___x_231_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_232_ = lean_ctor_get(v___x_231_, 0);
lean_inc_ref(v_toCommSemiring_232_);
v___x_233_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_232_);
v_toZero_234_ = lean_ctor_get(v___x_233_, 1);
lean_inc(v_toZero_234_);
lean_dec_ref(v___x_233_);
return v_toZero_234_;
}
else
{
lean_object* v_val_235_; 
v_val_235_ = lean_ctor_get(v___x_230_, 0);
lean_inc(v_val_235_);
lean_dec_ref_known(v___x_230_, 1);
return v_val_235_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord___boxed(lean_object* v_record_236_, lean_object* v_idx_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_record_236_, v_idx_237_);
lean_dec_ref(v_record_236_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryLeafValues(lean_object* v_lo_239_, lean_object* v_hi_240_){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_241_ = lean_unsigned_to_nat(8u);
v___x_242_ = lean_unsigned_to_nat(0u);
v___x_243_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_lo_239_, v___x_242_);
v___x_244_ = lean_unsigned_to_nat(1u);
v___x_245_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_lo_239_, v___x_244_);
v___x_246_ = lean_unsigned_to_nat(2u);
v___x_247_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_lo_239_, v___x_246_);
v___x_248_ = lean_unsigned_to_nat(3u);
v___x_249_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_lo_239_, v___x_248_);
v___x_250_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_hi_240_, v___x_242_);
v___x_251_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_hi_240_, v___x_244_);
v___x_252_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_hi_240_, v___x_246_);
v___x_253_ = lp_openvm_x2dfv_VM_Spec_Memory_recordDataWord(v_hi_240_, v___x_248_);
v___x_254_ = lean_mk_empty_array_with_capacity(v___x_241_);
v___x_255_ = lean_array_push(v___x_254_, v___x_243_);
v___x_256_ = lean_array_push(v___x_255_, v___x_245_);
v___x_257_ = lean_array_push(v___x_256_, v___x_247_);
v___x_258_ = lean_array_push(v___x_257_, v___x_249_);
v___x_259_ = lean_array_push(v___x_258_, v___x_250_);
v___x_260_ = lean_array_push(v___x_259_, v___x_251_);
v___x_261_ = lean_array_push(v___x_260_, v___x_252_);
v___x_262_ = lean_array_push(v___x_261_, v___x_253_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryLeafValues___boxed(lean_object* v_lo_263_, lean_object* v_hi_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_openvm_x2dfv_VM_Spec_Memory_boundaryLeafValues(v_lo_263_, v_hi_264_);
lean_dec_ref(v_hi_264_);
lean_dec_ref(v_lo_263_);
return v_res_265_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default___closed__0(void){
_start:
{
lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_266_ = lean_unsigned_to_nat(2013265921u);
v___x_267_ = lp_mathlib_ZMod_commRing(v___x_266_);
return v___x_267_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default(void){
_start:
{
lean_object* v___x_268_; lean_object* v_toSemiring_269_; lean_object* v___x_270_; lean_object* v_toZero_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_268_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default___closed__0);
v_toSemiring_269_ = lean_ctor_get(v___x_268_, 0);
lean_inc_ref(v_toSemiring_269_);
v___x_270_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toSemiring_269_);
v_toZero_271_ = lean_ctor_get(v___x_270_, 1);
lean_inc_n(v_toZero_271_, 5);
lean_dec_ref(v___x_270_);
v___x_272_ = lean_unsigned_to_nat(8u);
v___x_273_ = lean_mk_array(v___x_272_, v_toZero_271_);
v___x_274_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_274_, 0, v_toZero_271_);
lean_ctor_set(v___x_274_, 1, v_toZero_271_);
lean_ctor_set(v___x_274_, 2, v_toZero_271_);
lean_ctor_set(v___x_274_, 3, v_toZero_271_);
lean_ctor_set(v___x_274_, 4, v___x_273_);
return v___x_274_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode(void){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default;
return v___x_275_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq(lean_object* v_x_278_, lean_object* v_x_279_){
_start:
{
lean_object* v_direction_280_; lean_object* v_height_281_; lean_object* v_asLabel_282_; lean_object* v_addressLabel_283_; lean_object* v_hash_284_; lean_object* v_direction_285_; lean_object* v_height_286_; lean_object* v_asLabel_287_; lean_object* v_addressLabel_288_; lean_object* v_hash_289_; lean_object* v___x_290_; uint8_t v___x_291_; 
v_direction_280_ = lean_ctor_get(v_x_278_, 0);
v_height_281_ = lean_ctor_get(v_x_278_, 1);
v_asLabel_282_ = lean_ctor_get(v_x_278_, 2);
v_addressLabel_283_ = lean_ctor_get(v_x_278_, 3);
v_hash_284_ = lean_ctor_get(v_x_278_, 4);
v_direction_285_ = lean_ctor_get(v_x_279_, 0);
v_height_286_ = lean_ctor_get(v_x_279_, 1);
v_asLabel_287_ = lean_ctor_get(v_x_279_, 2);
v_addressLabel_288_ = lean_ctor_get(v_x_279_, 3);
v_hash_289_ = lean_ctor_get(v_x_279_, 4);
v___x_290_ = lean_unsigned_to_nat(2013265921u);
v___x_291_ = lp_mathlib_ZMod_decidableEq(v___x_290_, v_direction_280_, v_direction_285_);
if (v___x_291_ == 0)
{
return v___x_291_;
}
else
{
uint8_t v___x_292_; 
v___x_292_ = lp_mathlib_ZMod_decidableEq(v___x_290_, v_height_281_, v_height_286_);
if (v___x_292_ == 0)
{
return v___x_292_;
}
else
{
uint8_t v___x_293_; 
v___x_293_ = lp_mathlib_ZMod_decidableEq(v___x_290_, v_asLabel_282_, v_asLabel_287_);
if (v___x_293_ == 0)
{
return v___x_293_;
}
else
{
uint8_t v___x_294_; 
v___x_294_ = lp_mathlib_ZMod_decidableEq(v___x_290_, v_addressLabel_283_, v_addressLabel_288_);
if (v___x_294_ == 0)
{
return v___x_294_;
}
else
{
lean_object* v___x_295_; uint8_t v___x_296_; 
v___x_295_ = ((lean_object*)(lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq___closed__0));
v___x_296_ = l_Array_instDecidableEqImpl___redArg(v___x_295_, v_hash_284_, v_hash_289_);
return v___x_296_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq___boxed(lean_object* v_x_297_, lean_object* v_x_298_){
_start:
{
uint8_t v_res_299_; lean_object* v_r_300_; 
v_res_299_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq(v_x_297_, v_x_298_);
lean_dec_ref(v_x_298_);
lean_dec_ref(v_x_297_);
v_r_300_ = lean_box(v_res_299_);
return v_r_300_;
}
}
LEAN_EXPORT uint8_t lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode(lean_object* v_x_301_, lean_object* v_x_302_){
_start:
{
uint8_t v___x_303_; 
v___x_303_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode_decEq(v_x_301_, v_x_302_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode___boxed(lean_object* v_x_304_, lean_object* v_x_305_){
_start:
{
uint8_t v_res_306_; lean_object* v_r_307_; 
v_res_306_ = lp_openvm_x2dfv_VM_Spec_Memory_instDecidableEqMerkleBusNode(v_x_304_, v_x_305_);
lean_dec_ref(v_x_305_);
lean_dec_ref(v_x_304_);
v_r_307_ = lean_box(v_res_306_);
return v_r_307_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_parentNodeOfExpansion(lean_object* v_event_308_){
_start:
{
lean_object* v_direction_309_; lean_object* v_height_310_; lean_object* v_parent__as__label_311_; lean_object* v_parent__address__label_312_; lean_object* v_parent__hash_313_; lean_object* v___x_314_; 
v_direction_309_ = lean_ctor_get(v_event_308_, 0);
v_height_310_ = lean_ctor_get(v_event_308_, 1);
v_parent__as__label_311_ = lean_ctor_get(v_event_308_, 3);
v_parent__address__label_312_ = lean_ctor_get(v_event_308_, 4);
v_parent__hash_313_ = lean_ctor_get(v_event_308_, 5);
lean_inc_ref(v_parent__hash_313_);
lean_inc(v_parent__address__label_312_);
lean_inc(v_parent__as__label_311_);
lean_inc(v_height_310_);
lean_inc(v_direction_309_);
v___x_314_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_314_, 0, v_direction_309_);
lean_ctor_set(v___x_314_, 1, v_height_310_);
lean_ctor_set(v___x_314_, 2, v_parent__as__label_311_);
lean_ctor_set(v___x_314_, 3, v_parent__address__label_312_);
lean_ctor_set(v___x_314_, 4, v_parent__hash_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_parentNodeOfExpansion___boxed(lean_object* v_event_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_openvm_x2dfv_VM_Spec_Memory_parentNodeOfExpansion(v_event_315_);
lean_dec_ref(v_event_315_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_leftChildNodeOfExpansion(lean_object* v_event_317_){
_start:
{
lean_object* v___x_318_; lean_object* v_toCommSemiring_319_; lean_object* v___x_320_; lean_object* v_toMul_321_; lean_object* v_toAdd_322_; lean_object* v_direction_323_; lean_object* v_height_324_; lean_object* v_height__section_325_; lean_object* v_parent__as__label_326_; lean_object* v_parent__address__label_327_; lean_object* v_left__child__hash_328_; lean_object* v_left__direction__different_329_; lean_object* v___x_330_; lean_object* v_toRing_331_; lean_object* v___x_332_; lean_object* v_toAddMonoidWithOne_333_; lean_object* v___x_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_351_; 
v___x_318_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc_ref(v_toCommSemiring_319_);
v___x_320_ = lp_mathlib_instDistribOfSemiring___redArg(v_toCommSemiring_319_);
v_toMul_321_ = lean_ctor_get(v___x_320_, 0);
lean_inc(v_toMul_321_);
v_toAdd_322_ = lean_ctor_get(v___x_320_, 1);
lean_inc(v_toAdd_322_);
lean_dec_ref(v___x_320_);
v_direction_323_ = lean_ctor_get(v_event_317_, 0);
lean_inc(v_direction_323_);
v_height_324_ = lean_ctor_get(v_event_317_, 1);
lean_inc(v_height_324_);
v_height__section_325_ = lean_ctor_get(v_event_317_, 2);
lean_inc(v_height__section_325_);
v_parent__as__label_326_ = lean_ctor_get(v_event_317_, 3);
lean_inc(v_parent__as__label_326_);
v_parent__address__label_327_ = lean_ctor_get(v_event_317_, 4);
lean_inc(v_parent__address__label_327_);
v_left__child__hash_328_ = lean_ctor_get(v_event_317_, 6);
lean_inc_ref(v_left__child__hash_328_);
v_left__direction__different_329_ = lean_ctor_get(v_event_317_, 8);
lean_inc(v_left__direction__different_329_);
lean_dec_ref(v_event_317_);
v___x_330_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0);
v_toRing_331_ = lean_ctor_get(v___x_330_, 0);
lean_inc_ref(v_toRing_331_);
v___x_332_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_331_);
v_toAddMonoidWithOne_333_ = lean_ctor_get(v___x_332_, 1);
lean_inc_ref(v_toAddMonoidWithOne_333_);
v___x_334_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_332_);
v_isSharedCheck_351_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_351_ == 0)
{
lean_object* v_unused_352_; lean_object* v_unused_353_; lean_object* v_unused_354_; lean_object* v_unused_355_; lean_object* v_unused_356_; 
v_unused_352_ = lean_ctor_get(v___x_332_, 4);
lean_dec(v_unused_352_);
v_unused_353_ = lean_ctor_get(v___x_332_, 3);
lean_dec(v_unused_353_);
v_unused_354_ = lean_ctor_get(v___x_332_, 2);
lean_dec(v_unused_354_);
v_unused_355_ = lean_ctor_get(v___x_332_, 1);
lean_dec(v_unused_355_);
v_unused_356_ = lean_ctor_get(v___x_332_, 0);
lean_dec(v_unused_356_);
v___x_336_ = v___x_332_;
v_isShared_337_ = v_isSharedCheck_351_;
goto v_resetjp_335_;
}
else
{
lean_dec(v___x_332_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_351_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v_toSub_338_; lean_object* v_toOne_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_349_; 
v_toSub_338_ = lean_ctor_get(v___x_334_, 2);
lean_inc_n(v_toSub_338_, 2);
lean_dec_ref(v___x_334_);
v_toOne_339_ = lean_ctor_get(v_toAddMonoidWithOne_333_, 2);
lean_inc_n(v_toOne_339_, 2);
lean_dec_ref(v_toAddMonoidWithOne_333_);
v___x_340_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0);
lean_inc_n(v_toMul_321_, 2);
v___x_341_ = lean_apply_2(v_toMul_321_, v_left__direction__different_329_, v___x_340_);
lean_inc(v_toAdd_322_);
v___x_342_ = lean_apply_2(v_toAdd_322_, v_direction_323_, v___x_341_);
v___x_343_ = lean_apply_2(v_toSub_338_, v_height_324_, v_toOne_339_);
lean_inc(v_height__section_325_);
v___x_344_ = lean_apply_2(v_toAdd_322_, v_toOne_339_, v_height__section_325_);
v___x_345_ = lean_apply_2(v_toMul_321_, v_parent__as__label_326_, v___x_344_);
v___x_346_ = lean_apply_2(v_toSub_338_, v___x_340_, v_height__section_325_);
v___x_347_ = lean_apply_2(v_toMul_321_, v_parent__address__label_327_, v___x_346_);
if (v_isShared_337_ == 0)
{
lean_ctor_set(v___x_336_, 4, v_left__child__hash_328_);
lean_ctor_set(v___x_336_, 3, v___x_347_);
lean_ctor_set(v___x_336_, 2, v___x_345_);
lean_ctor_set(v___x_336_, 1, v___x_343_);
lean_ctor_set(v___x_336_, 0, v___x_342_);
v___x_349_ = v___x_336_;
goto v_reusejp_348_;
}
else
{
lean_object* v_reuseFailAlloc_350_; 
v_reuseFailAlloc_350_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_350_, 0, v___x_342_);
lean_ctor_set(v_reuseFailAlloc_350_, 1, v___x_343_);
lean_ctor_set(v_reuseFailAlloc_350_, 2, v___x_345_);
lean_ctor_set(v_reuseFailAlloc_350_, 3, v___x_347_);
lean_ctor_set(v_reuseFailAlloc_350_, 4, v_left__child__hash_328_);
v___x_349_ = v_reuseFailAlloc_350_;
goto v_reusejp_348_;
}
v_reusejp_348_:
{
return v___x_349_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_rightChildNodeOfExpansion(lean_object* v_event_357_){
_start:
{
lean_object* v___x_358_; lean_object* v_toCommSemiring_359_; lean_object* v___x_360_; lean_object* v_toMul_361_; lean_object* v_toAdd_362_; lean_object* v_direction_363_; lean_object* v_height_364_; lean_object* v_height__section_365_; lean_object* v_parent__as__label_366_; lean_object* v_parent__address__label_367_; lean_object* v_right__child__hash_368_; lean_object* v_right__direction__different_369_; lean_object* v___x_370_; lean_object* v_toRing_371_; lean_object* v___x_372_; lean_object* v_toAddMonoidWithOne_373_; lean_object* v___x_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_394_; 
v___x_358_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_359_ = lean_ctor_get(v___x_358_, 0);
lean_inc_ref(v_toCommSemiring_359_);
v___x_360_ = lp_mathlib_instDistribOfSemiring___redArg(v_toCommSemiring_359_);
v_toMul_361_ = lean_ctor_get(v___x_360_, 0);
lean_inc(v_toMul_361_);
v_toAdd_362_ = lean_ctor_get(v___x_360_, 1);
lean_inc(v_toAdd_362_);
lean_dec_ref(v___x_360_);
v_direction_363_ = lean_ctor_get(v_event_357_, 0);
lean_inc(v_direction_363_);
v_height_364_ = lean_ctor_get(v_event_357_, 1);
lean_inc(v_height_364_);
v_height__section_365_ = lean_ctor_get(v_event_357_, 2);
lean_inc(v_height__section_365_);
v_parent__as__label_366_ = lean_ctor_get(v_event_357_, 3);
lean_inc(v_parent__as__label_366_);
v_parent__address__label_367_ = lean_ctor_get(v_event_357_, 4);
lean_inc(v_parent__address__label_367_);
v_right__child__hash_368_ = lean_ctor_get(v_event_357_, 7);
lean_inc_ref(v_right__child__hash_368_);
v_right__direction__different_369_ = lean_ctor_get(v_event_357_, 9);
lean_inc(v_right__direction__different_369_);
lean_dec_ref(v_event_357_);
v___x_370_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0);
v_toRing_371_ = lean_ctor_get(v___x_370_, 0);
lean_inc_ref(v_toRing_371_);
v___x_372_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_371_);
v_toAddMonoidWithOne_373_ = lean_ctor_get(v___x_372_, 1);
lean_inc_ref(v_toAddMonoidWithOne_373_);
v___x_374_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_372_);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_394_ == 0)
{
lean_object* v_unused_395_; lean_object* v_unused_396_; lean_object* v_unused_397_; lean_object* v_unused_398_; lean_object* v_unused_399_; 
v_unused_395_ = lean_ctor_get(v___x_372_, 4);
lean_dec(v_unused_395_);
v_unused_396_ = lean_ctor_get(v___x_372_, 3);
lean_dec(v_unused_396_);
v_unused_397_ = lean_ctor_get(v___x_372_, 2);
lean_dec(v_unused_397_);
v_unused_398_ = lean_ctor_get(v___x_372_, 1);
lean_dec(v_unused_398_);
v_unused_399_ = lean_ctor_get(v___x_372_, 0);
lean_dec(v_unused_399_);
v___x_376_ = v___x_372_;
v_isShared_377_ = v_isSharedCheck_394_;
goto v_resetjp_375_;
}
else
{
lean_dec(v___x_372_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_394_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v_toSub_378_; lean_object* v_toOne_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_392_; 
v_toSub_378_ = lean_ctor_get(v___x_374_, 2);
lean_inc_n(v_toSub_378_, 3);
lean_dec_ref(v___x_374_);
v_toOne_379_ = lean_ctor_get(v_toAddMonoidWithOne_373_, 2);
lean_inc_n(v_toOne_379_, 3);
lean_dec_ref(v_toAddMonoidWithOne_373_);
v___x_380_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_leftExpansionNodeLabel___closed__0);
lean_inc_n(v_toMul_361_, 2);
v___x_381_ = lean_apply_2(v_toMul_361_, v_right__direction__different_369_, v___x_380_);
lean_inc_n(v_toAdd_362_, 3);
v___x_382_ = lean_apply_2(v_toAdd_362_, v_direction_363_, v___x_381_);
v___x_383_ = lean_apply_2(v_toSub_378_, v_height_364_, v_toOne_379_);
lean_inc_n(v_height__section_365_, 3);
v___x_384_ = lean_apply_2(v_toAdd_362_, v_toOne_379_, v_height__section_365_);
v___x_385_ = lean_apply_2(v_toMul_361_, v_parent__as__label_366_, v___x_384_);
v___x_386_ = lean_apply_2(v_toAdd_362_, v___x_385_, v_height__section_365_);
v___x_387_ = lean_apply_2(v_toSub_378_, v___x_380_, v_height__section_365_);
v___x_388_ = lean_apply_2(v_toMul_361_, v_parent__address__label_367_, v___x_387_);
v___x_389_ = lean_apply_2(v_toSub_378_, v_toOne_379_, v_height__section_365_);
v___x_390_ = lean_apply_2(v_toAdd_362_, v___x_388_, v___x_389_);
if (v_isShared_377_ == 0)
{
lean_ctor_set(v___x_376_, 4, v_right__child__hash_368_);
lean_ctor_set(v___x_376_, 3, v___x_390_);
lean_ctor_set(v___x_376_, 2, v___x_386_);
lean_ctor_set(v___x_376_, 1, v___x_383_);
lean_ctor_set(v___x_376_, 0, v___x_382_);
v___x_392_ = v___x_376_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_382_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v___x_383_);
lean_ctor_set(v_reuseFailAlloc_393_, 2, v___x_386_);
lean_ctor_set(v_reuseFailAlloc_393_, 3, v___x_390_);
lean_ctor_set(v_reuseFailAlloc_393_, 4, v_right__child__hash_368_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0(void){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; 
v___x_400_ = lean_unsigned_to_nat(8u);
v___x_401_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord(lean_object* v_event_402_){
_start:
{
lean_object* v_address__space_403_; lean_object* v_leaf__label_404_; lean_object* v_values_405_; lean_object* v_timestamps_406_; lean_object* v___x_407_; lean_object* v_toCommSemiring_408_; lean_object* v___x_409_; lean_object* v_toMul_410_; lean_object* v___x_412_; uint8_t v_isShared_413_; uint8_t v_isSharedCheck_433_; 
v_address__space_403_ = lean_ctor_get(v_event_402_, 1);
lean_inc(v_address__space_403_);
v_leaf__label_404_ = lean_ctor_get(v_event_402_, 2);
lean_inc(v_leaf__label_404_);
v_values_405_ = lean_ctor_get(v_event_402_, 3);
lean_inc_ref(v_values_405_);
v_timestamps_406_ = lean_ctor_get(v_event_402_, 5);
lean_inc_ref(v_timestamps_406_);
lean_dec_ref(v_event_402_);
v___x_407_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_408_ = lean_ctor_get(v___x_407_, 0);
lean_inc_ref(v_toCommSemiring_408_);
v___x_409_ = lp_mathlib_instDistribOfSemiring___redArg(v_toCommSemiring_408_);
v_toMul_410_ = lean_ctor_get(v___x_409_, 0);
v_isSharedCheck_433_ = !lean_is_exclusive(v___x_409_);
if (v_isSharedCheck_433_ == 0)
{
lean_object* v_unused_434_; 
v_unused_434_ = lean_ctor_get(v___x_409_, 1);
lean_dec(v_unused_434_);
v___x_412_ = v___x_409_;
v_isShared_413_ = v_isSharedCheck_433_;
goto v_resetjp_411_;
}
else
{
lean_inc(v_toMul_410_);
lean_dec(v___x_409_);
v___x_412_ = lean_box(0);
v_isShared_413_ = v_isSharedCheck_433_;
goto v_resetjp_411_;
}
v_resetjp_411_:
{
lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_426_; 
v___x_414_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0);
v___x_415_ = lean_apply_2(v_toMul_410_, v_leaf__label_404_, v___x_414_);
v___x_416_ = lean_unsigned_to_nat(0u);
v___x_417_ = lean_array_fget(v_values_405_, v___x_416_);
v___x_418_ = lean_unsigned_to_nat(1u);
v___x_419_ = lean_array_fget(v_values_405_, v___x_418_);
v___x_420_ = lean_unsigned_to_nat(2u);
v___x_421_ = lean_array_fget(v_values_405_, v___x_420_);
v___x_422_ = lean_unsigned_to_nat(3u);
v___x_423_ = lean_array_fget(v_values_405_, v___x_422_);
lean_dec_ref(v_values_405_);
v___x_424_ = lean_box(0);
if (v_isShared_413_ == 0)
{
lean_ctor_set_tag(v___x_412_, 1);
lean_ctor_set(v___x_412_, 1, v___x_424_);
lean_ctor_set(v___x_412_, 0, v___x_423_);
v___x_426_ = v___x_412_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_432_; 
v_reuseFailAlloc_432_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_432_, 0, v___x_423_);
lean_ctor_set(v_reuseFailAlloc_432_, 1, v___x_424_);
v___x_426_ = v_reuseFailAlloc_432_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_427_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_427_, 0, v___x_421_);
lean_ctor_set(v___x_427_, 1, v___x_426_);
v___x_428_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_428_, 0, v___x_419_);
lean_ctor_set(v___x_428_, 1, v___x_427_);
v___x_429_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_417_);
lean_ctor_set(v___x_429_, 1, v___x_428_);
v___x_430_ = lean_array_fget(v_timestamps_406_, v___x_416_);
lean_dec_ref(v_timestamps_406_);
v___x_431_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_431_, 0, v_address__space_403_);
lean_ctor_set(v___x_431_, 1, v___x_415_);
lean_ctor_set(v___x_431_, 2, v___x_429_);
lean_ctor_set(v___x_431_, 3, v___x_430_);
return v___x_431_;
}
}
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord___closed__0(void){
_start:
{
lean_object* v___x_435_; lean_object* v___x_436_; 
v___x_435_ = lean_unsigned_to_nat(4u);
v___x_436_ = lp_swirl_x2dfv_Nat_cast___at___00Fundamentals_BabyBear_fbbFieldOps_spec__0(v___x_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord(lean_object* v_event_437_){
_start:
{
lean_object* v_address__space_438_; lean_object* v_leaf__label_439_; lean_object* v_values_440_; lean_object* v_timestamps_441_; lean_object* v___x_442_; lean_object* v_toCommSemiring_443_; lean_object* v___x_444_; lean_object* v_toMul_445_; lean_object* v_toAdd_446_; lean_object* v___x_448_; uint8_t v_isShared_449_; uint8_t v_isSharedCheck_472_; 
v_address__space_438_ = lean_ctor_get(v_event_437_, 1);
lean_inc(v_address__space_438_);
v_leaf__label_439_ = lean_ctor_get(v_event_437_, 2);
lean_inc(v_leaf__label_439_);
v_values_440_ = lean_ctor_get(v_event_437_, 3);
lean_inc_ref(v_values_440_);
v_timestamps_441_ = lean_ctor_get(v_event_437_, 5);
lean_inc_ref(v_timestamps_441_);
lean_dec_ref(v_event_437_);
v___x_442_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_443_ = lean_ctor_get(v___x_442_, 0);
lean_inc_ref(v_toCommSemiring_443_);
v___x_444_ = lp_mathlib_instDistribOfSemiring___redArg(v_toCommSemiring_443_);
v_toMul_445_ = lean_ctor_get(v___x_444_, 0);
v_toAdd_446_ = lean_ctor_get(v___x_444_, 1);
v_isSharedCheck_472_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_472_ == 0)
{
v___x_448_ = v___x_444_;
v_isShared_449_ = v_isSharedCheck_472_;
goto v_resetjp_447_;
}
else
{
lean_inc(v_toAdd_446_);
lean_inc(v_toMul_445_);
lean_dec(v___x_444_);
v___x_448_ = lean_box(0);
v_isShared_449_ = v_isSharedCheck_472_;
goto v_resetjp_447_;
}
v_resetjp_447_:
{
lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_464_; 
v___x_450_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventLoRecord___closed__0);
v___x_451_ = lean_apply_2(v_toMul_445_, v_leaf__label_439_, v___x_450_);
v___x_452_ = lean_unsigned_to_nat(4u);
v___x_453_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_boundaryEventHiRecord___closed__0);
v___x_454_ = lean_apply_2(v_toAdd_446_, v___x_451_, v___x_453_);
v___x_455_ = lean_array_fget(v_values_440_, v___x_452_);
v___x_456_ = lean_unsigned_to_nat(5u);
v___x_457_ = lean_array_fget(v_values_440_, v___x_456_);
v___x_458_ = lean_unsigned_to_nat(6u);
v___x_459_ = lean_array_fget(v_values_440_, v___x_458_);
v___x_460_ = lean_unsigned_to_nat(7u);
v___x_461_ = lean_array_fget(v_values_440_, v___x_460_);
lean_dec_ref(v_values_440_);
v___x_462_ = lean_box(0);
if (v_isShared_449_ == 0)
{
lean_ctor_set_tag(v___x_448_, 1);
lean_ctor_set(v___x_448_, 1, v___x_462_);
lean_ctor_set(v___x_448_, 0, v___x_461_);
v___x_464_ = v___x_448_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v___x_461_);
lean_ctor_set(v_reuseFailAlloc_471_, 1, v___x_462_);
v___x_464_ = v_reuseFailAlloc_471_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; 
v___x_465_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_465_, 0, v___x_459_);
lean_ctor_set(v___x_465_, 1, v___x_464_);
v___x_466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_466_, 0, v___x_457_);
lean_ctor_set(v___x_466_, 1, v___x_465_);
v___x_467_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_467_, 0, v___x_455_);
lean_ctor_set(v___x_467_, 1, v___x_466_);
v___x_468_ = lean_unsigned_to_nat(1u);
v___x_469_ = lean_array_fget(v_timestamps_441_, v___x_468_);
lean_dec_ref(v_timestamps_441_);
v___x_470_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_470_, 0, v_address__space_438_);
lean_ctor_set(v___x_470_, 1, v___x_454_);
lean_ctor_set(v___x_470_, 2, v___x_467_);
lean_ctor_set(v___x_470_, 3, v___x_469_);
return v___x_470_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Memory_boundaryNodeOfEvent(lean_object* v_event_473_){
_start:
{
lean_object* v_direction_474_; lean_object* v_address__space_475_; lean_object* v_leaf__label_476_; lean_object* v_hash_477_; lean_object* v___x_478_; lean_object* v_toCommSemiring_479_; lean_object* v___x_480_; lean_object* v_toZero_481_; lean_object* v___x_482_; lean_object* v_toRing_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v_toAddMonoidWithOne_486_; lean_object* v___x_488_; uint8_t v_isShared_489_; uint8_t v_isSharedCheck_496_; 
v_direction_474_ = lean_ctor_get(v_event_473_, 0);
lean_inc(v_direction_474_);
v_address__space_475_ = lean_ctor_get(v_event_473_, 1);
lean_inc(v_address__space_475_);
v_leaf__label_476_ = lean_ctor_get(v_event_473_, 2);
lean_inc(v_leaf__label_476_);
v_hash_477_ = lean_ctor_get(v_event_473_, 4);
lean_inc_ref(v_hash_477_);
lean_dec_ref(v_event_473_);
v___x_478_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1, &lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest___closed__1);
v_toCommSemiring_479_ = lean_ctor_get(v___x_478_, 0);
lean_inc_ref(v_toCommSemiring_479_);
v___x_480_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_479_);
v_toZero_481_ = lean_ctor_get(v___x_480_, 1);
lean_inc(v_toZero_481_);
lean_dec_ref(v___x_480_);
v___x_482_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0, &lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Memory_memoryLeafValues___closed__0);
v_toRing_483_ = lean_ctor_get(v___x_482_, 0);
lean_inc_ref(v_toRing_483_);
v___x_484_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_483_);
v___x_485_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_484_);
v_toAddMonoidWithOne_486_ = lean_ctor_get(v___x_484_, 1);
v_isSharedCheck_496_ = !lean_is_exclusive(v___x_484_);
if (v_isSharedCheck_496_ == 0)
{
lean_object* v_unused_497_; lean_object* v_unused_498_; lean_object* v_unused_499_; lean_object* v_unused_500_; 
v_unused_497_ = lean_ctor_get(v___x_484_, 4);
lean_dec(v_unused_497_);
v_unused_498_ = lean_ctor_get(v___x_484_, 3);
lean_dec(v_unused_498_);
v_unused_499_ = lean_ctor_get(v___x_484_, 2);
lean_dec(v_unused_499_);
v_unused_500_ = lean_ctor_get(v___x_484_, 0);
lean_dec(v_unused_500_);
v___x_488_ = v___x_484_;
v_isShared_489_ = v_isSharedCheck_496_;
goto v_resetjp_487_;
}
else
{
lean_inc(v_toAddMonoidWithOne_486_);
lean_dec(v___x_484_);
v___x_488_ = lean_box(0);
v_isShared_489_ = v_isSharedCheck_496_;
goto v_resetjp_487_;
}
v_resetjp_487_:
{
lean_object* v_toSub_490_; lean_object* v_toOne_491_; lean_object* v___x_492_; lean_object* v___x_494_; 
v_toSub_490_ = lean_ctor_get(v___x_485_, 2);
lean_inc(v_toSub_490_);
lean_dec_ref(v___x_485_);
v_toOne_491_ = lean_ctor_get(v_toAddMonoidWithOne_486_, 2);
lean_inc(v_toOne_491_);
lean_dec_ref(v_toAddMonoidWithOne_486_);
v___x_492_ = lean_apply_2(v_toSub_490_, v_address__space_475_, v_toOne_491_);
if (v_isShared_489_ == 0)
{
lean_ctor_set(v___x_488_, 4, v_hash_477_);
lean_ctor_set(v___x_488_, 3, v_leaf__label_476_);
lean_ctor_set(v___x_488_, 2, v___x_492_);
lean_ctor_set(v___x_488_, 1, v_toZero_481_);
lean_ctor_set(v___x_488_, 0, v_direction_474_);
v___x_494_ = v___x_488_;
goto v_reusejp_493_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v_direction_474_);
lean_ctor_set(v_reuseFailAlloc_495_, 1, v_toZero_481_);
lean_ctor_set(v_reuseFailAlloc_495_, 2, v___x_492_);
lean_ctor_set(v_reuseFailAlloc_495_, 3, v_leaf__label_476_);
lean_ctor_set(v_reuseFailAlloc_495_, 4, v_hash_477_);
v___x_494_ = v_reuseFailAlloc_495_;
goto v_reusejp_493_;
}
v_reusejp_493_:
{
return v___x_494_;
}
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_BabyBear(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Memory(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Memory_Events(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Memory_Record(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Airs_System_PersistentBoundaryAir_View(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Memory_Spec(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Machine_BabyBear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Machine_Memory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Memory_Events(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Memory_Record(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Airs_System_PersistentBoundaryAir_View(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_VM_Spec_Memory_addressHeight = _init_lp_openvm_x2dfv_VM_Spec_Memory_addressHeight();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_addressHeight);
lp_openvm_x2dfv_VM_Spec_Memory_overallHeight = _init_lp_openvm_x2dfv_VM_Spec_Memory_overallHeight();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_overallHeight);
lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest = _init_lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_zeroDigest);
lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default = _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode_default);
lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode = _init_lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode();
lean_mark_persistent(lp_openvm_x2dfv_VM_Spec_Memory_instInhabitedMerkleBusNode);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
