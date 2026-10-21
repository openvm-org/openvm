// Lean compiler output
// Module: VM.Spec.Airs.System.MemoryMerkleAir.View
// Imports: public import Init public meta import Init public import Fundamentals.Spec.BabyBear.Field public import VM.Spec.Airs.System.MemoryMerkleAir.Extraction.Schema public import VM.Spec.Memory.Events
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
lean_object* lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(lean_object*);
lean_object* lp_mathlib_ZMod_instField___redArg(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_ZMod_decidableEq(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__0;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_finalRoot(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_finalRoot___boxed(lean_object*);
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__1;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__2;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__3;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__4;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__5;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__6;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__7;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__8;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__9;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__10;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__11;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__12;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__13;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__14;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__15;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__16;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__17;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__18;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__19;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__20;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__21;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__22;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__23;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__24;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__25;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__26;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__27;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__28;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__29;
static lean_once_cell_t lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__30;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_filterMapTR_go___at___00VM_Spec_Airs_System_MemoryMerkleAir_View_events_spec__0(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_events___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_events___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_events___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_events(lean_object*);
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(2013265921u);
v___x_2_ = lp_mathlib_ZMod_instField___redArg(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__0, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__0);
v___x_4_ = lp_mathlib_Field_toSemifield___redArg(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot(lean_object* v_publicValues_5_){
_start:
{
lean_object* v___x_6_; lean_object* v_toCommSemiring_7_; lean_object* v___x_8_; lean_object* v_toZero_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_6_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1);
v_toCommSemiring_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_toCommSemiring_7_);
v___x_8_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_7_);
v_toZero_9_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_toZero_9_);
lean_dec_ref(v___x_8_);
v___x_10_ = lean_unsigned_to_nat(8u);
v___x_11_ = lean_unsigned_to_nat(0u);
v___x_12_ = l_List_getD___redArg(v_publicValues_5_, v___x_11_, v_toZero_9_);
v___x_13_ = lean_unsigned_to_nat(1u);
v___x_14_ = l_List_getD___redArg(v_publicValues_5_, v___x_13_, v_toZero_9_);
v___x_15_ = lean_unsigned_to_nat(2u);
v___x_16_ = l_List_getD___redArg(v_publicValues_5_, v___x_15_, v_toZero_9_);
v___x_17_ = lean_unsigned_to_nat(3u);
v___x_18_ = l_List_getD___redArg(v_publicValues_5_, v___x_17_, v_toZero_9_);
v___x_19_ = lean_unsigned_to_nat(4u);
v___x_20_ = l_List_getD___redArg(v_publicValues_5_, v___x_19_, v_toZero_9_);
v___x_21_ = lean_unsigned_to_nat(5u);
v___x_22_ = l_List_getD___redArg(v_publicValues_5_, v___x_21_, v_toZero_9_);
v___x_23_ = lean_unsigned_to_nat(6u);
v___x_24_ = l_List_getD___redArg(v_publicValues_5_, v___x_23_, v_toZero_9_);
v___x_25_ = lean_unsigned_to_nat(7u);
v___x_26_ = l_List_getD___redArg(v_publicValues_5_, v___x_25_, v_toZero_9_);
lean_dec(v_toZero_9_);
v___x_27_ = lean_mk_empty_array_with_capacity(v___x_10_);
v___x_28_ = lean_array_push(v___x_27_, v___x_12_);
v___x_29_ = lean_array_push(v___x_28_, v___x_14_);
v___x_30_ = lean_array_push(v___x_29_, v___x_16_);
v___x_31_ = lean_array_push(v___x_30_, v___x_18_);
v___x_32_ = lean_array_push(v___x_31_, v___x_20_);
v___x_33_ = lean_array_push(v___x_32_, v___x_22_);
v___x_34_ = lean_array_push(v___x_33_, v___x_24_);
v___x_35_ = lean_array_push(v___x_34_, v___x_26_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___boxed(lean_object* v_publicValues_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot(v_publicValues_36_);
lean_dec(v_publicValues_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_finalRoot(lean_object* v_publicValues_38_){
_start:
{
lean_object* v___x_39_; lean_object* v_toCommSemiring_40_; lean_object* v___x_41_; lean_object* v_toZero_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_39_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1);
v_toCommSemiring_40_ = lean_ctor_get(v___x_39_, 0);
lean_inc_ref(v_toCommSemiring_40_);
v___x_41_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_40_);
v_toZero_42_ = lean_ctor_get(v___x_41_, 1);
lean_inc(v_toZero_42_);
lean_dec_ref(v___x_41_);
v___x_43_ = lean_unsigned_to_nat(8u);
v___x_44_ = l_List_getD___redArg(v_publicValues_38_, v___x_43_, v_toZero_42_);
v___x_45_ = lean_unsigned_to_nat(9u);
v___x_46_ = l_List_getD___redArg(v_publicValues_38_, v___x_45_, v_toZero_42_);
v___x_47_ = lean_unsigned_to_nat(10u);
v___x_48_ = l_List_getD___redArg(v_publicValues_38_, v___x_47_, v_toZero_42_);
v___x_49_ = lean_unsigned_to_nat(11u);
v___x_50_ = l_List_getD___redArg(v_publicValues_38_, v___x_49_, v_toZero_42_);
v___x_51_ = lean_unsigned_to_nat(12u);
v___x_52_ = l_List_getD___redArg(v_publicValues_38_, v___x_51_, v_toZero_42_);
v___x_53_ = lean_unsigned_to_nat(13u);
v___x_54_ = l_List_getD___redArg(v_publicValues_38_, v___x_53_, v_toZero_42_);
v___x_55_ = lean_unsigned_to_nat(14u);
v___x_56_ = l_List_getD___redArg(v_publicValues_38_, v___x_55_, v_toZero_42_);
v___x_57_ = lean_unsigned_to_nat(15u);
v___x_58_ = l_List_getD___redArg(v_publicValues_38_, v___x_57_, v_toZero_42_);
lean_dec(v_toZero_42_);
v___x_59_ = lean_mk_empty_array_with_capacity(v___x_43_);
v___x_60_ = lean_array_push(v___x_59_, v___x_44_);
v___x_61_ = lean_array_push(v___x_60_, v___x_46_);
v___x_62_ = lean_array_push(v___x_61_, v___x_48_);
v___x_63_ = lean_array_push(v___x_62_, v___x_50_);
v___x_64_ = lean_array_push(v___x_63_, v___x_52_);
v___x_65_ = lean_array_push(v___x_64_, v___x_54_);
v___x_66_ = lean_array_push(v___x_65_, v___x_56_);
v___x_67_ = lean_array_push(v___x_66_, v___x_58_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_finalRoot___boxed(lean_object* v_publicValues_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_finalRoot(v_publicValues_68_);
lean_dec(v_publicValues_68_);
return v_res_69_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = lean_unsigned_to_nat(0u);
v___x_71_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_70_);
return v___x_71_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__1(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_72_ = lean_unsigned_to_nat(2u);
v___x_73_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_72_);
return v___x_73_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__2(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_74_ = lean_unsigned_to_nat(1u);
v___x_75_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_74_);
return v___x_75_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__3(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_76_ = lean_unsigned_to_nat(5u);
v___x_77_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_76_);
return v___x_77_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__4(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_78_ = lean_unsigned_to_nat(6u);
v___x_79_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_78_);
return v___x_79_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__5(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = lean_unsigned_to_nat(7u);
v___x_81_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_80_);
return v___x_81_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__6(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_82_ = lean_unsigned_to_nat(8u);
v___x_83_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_82_);
return v___x_83_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__7(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_84_ = lean_unsigned_to_nat(9u);
v___x_85_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_84_);
return v___x_85_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__8(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = lean_unsigned_to_nat(10u);
v___x_87_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_86_);
return v___x_87_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__9(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = lean_unsigned_to_nat(11u);
v___x_89_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_88_);
return v___x_89_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__10(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_90_ = lean_unsigned_to_nat(12u);
v___x_91_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_90_);
return v___x_91_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__11(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_92_ = lean_unsigned_to_nat(13u);
v___x_93_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_92_);
return v___x_93_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__12(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lean_unsigned_to_nat(14u);
v___x_95_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_94_);
return v___x_95_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__13(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_96_ = lean_unsigned_to_nat(15u);
v___x_97_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_96_);
return v___x_97_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__14(void){
_start:
{
lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_98_ = lean_unsigned_to_nat(16u);
v___x_99_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_98_);
return v___x_99_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__15(void){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = lean_unsigned_to_nat(17u);
v___x_101_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_100_);
return v___x_101_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__16(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_102_ = lean_unsigned_to_nat(18u);
v___x_103_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_102_);
return v___x_103_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__17(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = lean_unsigned_to_nat(19u);
v___x_105_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_104_);
return v___x_105_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__18(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = lean_unsigned_to_nat(20u);
v___x_107_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_106_);
return v___x_107_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__19(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = lean_unsigned_to_nat(21u);
v___x_109_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_108_);
return v___x_109_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__20(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_110_ = lean_unsigned_to_nat(22u);
v___x_111_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_110_);
return v___x_111_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__21(void){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_112_ = lean_unsigned_to_nat(23u);
v___x_113_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_112_);
return v___x_113_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__22(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_114_ = lean_unsigned_to_nat(24u);
v___x_115_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_114_);
return v___x_115_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__23(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = lean_unsigned_to_nat(25u);
v___x_117_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_116_);
return v___x_117_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__24(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = lean_unsigned_to_nat(26u);
v___x_119_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_118_);
return v___x_119_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__25(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = lean_unsigned_to_nat(27u);
v___x_121_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_120_);
return v___x_121_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__26(void){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_122_ = lean_unsigned_to_nat(28u);
v___x_123_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_122_);
return v___x_123_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__27(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = lean_unsigned_to_nat(29u);
v___x_125_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_124_);
return v___x_125_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__28(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = lean_unsigned_to_nat(30u);
v___x_127_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_126_);
return v___x_127_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__29(void){
_start:
{
lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_128_ = lean_unsigned_to_nat(31u);
v___x_129_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_128_);
return v___x_129_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__30(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_130_ = lean_unsigned_to_nat(32u);
v___x_131_ = lp_swirl_x2dfv_Fundamentals_Air_ColumnRef_commonMain___redArg(v___x_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow(lean_object* v_trace_132_, lean_object* v_row_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_134_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0);
lean_inc_n(v_row_133_, 30);
lean_inc_ref_n(v_trace_132_, 30);
v___x_135_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_134_, v_row_133_);
v___x_136_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__1, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__1);
v___x_137_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_136_, v_row_133_);
v___x_138_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__2, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__2_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__2);
v___x_139_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_138_, v_row_133_);
v___x_140_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__3, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__3_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__3);
v___x_141_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_140_, v_row_133_);
v___x_142_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__4, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__4_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__4);
v___x_143_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_142_, v_row_133_);
v___x_144_ = lean_unsigned_to_nat(8u);
v___x_145_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__5, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__5_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__5);
v___x_146_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_145_, v_row_133_);
v___x_147_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__6, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__6_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__6);
v___x_148_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_147_, v_row_133_);
v___x_149_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__7, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__7_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__7);
v___x_150_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_149_, v_row_133_);
v___x_151_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__8, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__8_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__8);
v___x_152_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_151_, v_row_133_);
v___x_153_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__9, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__9_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__9);
v___x_154_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_153_, v_row_133_);
v___x_155_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__10, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__10_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__10);
v___x_156_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_155_, v_row_133_);
v___x_157_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__11, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__11_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__11);
v___x_158_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_157_, v_row_133_);
v___x_159_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__12, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__12_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__12);
v___x_160_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_159_, v_row_133_);
v___x_161_ = lean_mk_empty_array_with_capacity(v___x_144_);
lean_inc_ref_n(v___x_161_, 2);
v___x_162_ = lean_array_push(v___x_161_, v___x_146_);
v___x_163_ = lean_array_push(v___x_162_, v___x_148_);
v___x_164_ = lean_array_push(v___x_163_, v___x_150_);
v___x_165_ = lean_array_push(v___x_164_, v___x_152_);
v___x_166_ = lean_array_push(v___x_165_, v___x_154_);
v___x_167_ = lean_array_push(v___x_166_, v___x_156_);
v___x_168_ = lean_array_push(v___x_167_, v___x_158_);
v___x_169_ = lean_array_push(v___x_168_, v___x_160_);
v___x_170_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__13, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__13_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__13);
v___x_171_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_170_, v_row_133_);
v___x_172_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__14, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__14_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__14);
v___x_173_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_172_, v_row_133_);
v___x_174_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__15, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__15_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__15);
v___x_175_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_174_, v_row_133_);
v___x_176_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__16, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__16_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__16);
v___x_177_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_176_, v_row_133_);
v___x_178_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__17, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__17_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__17);
v___x_179_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_178_, v_row_133_);
v___x_180_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__18, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__18_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__18);
v___x_181_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_180_, v_row_133_);
v___x_182_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__19, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__19_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__19);
v___x_183_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_182_, v_row_133_);
v___x_184_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__20, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__20_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__20);
v___x_185_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_184_, v_row_133_);
v___x_186_ = lean_array_push(v___x_161_, v___x_171_);
v___x_187_ = lean_array_push(v___x_186_, v___x_173_);
v___x_188_ = lean_array_push(v___x_187_, v___x_175_);
v___x_189_ = lean_array_push(v___x_188_, v___x_177_);
v___x_190_ = lean_array_push(v___x_189_, v___x_179_);
v___x_191_ = lean_array_push(v___x_190_, v___x_181_);
v___x_192_ = lean_array_push(v___x_191_, v___x_183_);
v___x_193_ = lean_array_push(v___x_192_, v___x_185_);
v___x_194_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__21, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__21_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__21);
v___x_195_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_194_, v_row_133_);
v___x_196_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__22, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__22_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__22);
v___x_197_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_196_, v_row_133_);
v___x_198_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__23, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__23_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__23);
v___x_199_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_198_, v_row_133_);
v___x_200_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__24, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__24_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__24);
v___x_201_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_200_, v_row_133_);
v___x_202_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__25, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__25_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__25);
v___x_203_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_202_, v_row_133_);
v___x_204_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__26, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__26_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__26);
v___x_205_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_204_, v_row_133_);
v___x_206_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__27, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__27_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__27);
v___x_207_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_206_, v_row_133_);
v___x_208_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__28, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__28_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__28);
v___x_209_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_208_, v_row_133_);
v___x_210_ = lean_array_push(v___x_161_, v___x_195_);
v___x_211_ = lean_array_push(v___x_210_, v___x_197_);
v___x_212_ = lean_array_push(v___x_211_, v___x_199_);
v___x_213_ = lean_array_push(v___x_212_, v___x_201_);
v___x_214_ = lean_array_push(v___x_213_, v___x_203_);
v___x_215_ = lean_array_push(v___x_214_, v___x_205_);
v___x_216_ = lean_array_push(v___x_215_, v___x_207_);
v___x_217_ = lean_array_push(v___x_216_, v___x_209_);
v___x_218_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__29, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__29_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__29);
v___x_219_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_218_, v_row_133_);
v___x_220_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__30, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__30_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__30);
v___x_221_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_132_, v___x_220_, v_row_133_);
v___x_222_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_222_, 0, v___x_135_);
lean_ctor_set(v___x_222_, 1, v___x_137_);
lean_ctor_set(v___x_222_, 2, v___x_139_);
lean_ctor_set(v___x_222_, 3, v___x_141_);
lean_ctor_set(v___x_222_, 4, v___x_143_);
lean_ctor_set(v___x_222_, 5, v___x_169_);
lean_ctor_set(v___x_222_, 6, v___x_193_);
lean_ctor_set(v___x_222_, 7, v___x_217_);
lean_ctor_set(v___x_222_, 8, v___x_219_);
lean_ctor_set(v___x_222_, 9, v___x_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_List_filterMapTR_go___at___00VM_Spec_Airs_System_MemoryMerkleAir_View_events_spec__0(lean_object* v_trace_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
if (lean_obj_tag(v_a_224_) == 0)
{
lean_object* v___x_226_; 
lean_dec_ref(v_trace_223_);
v___x_226_ = lean_array_to_list(v_a_225_);
return v___x_226_;
}
else
{
lean_object* v_head_227_; lean_object* v_tail_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v_toCommSemiring_231_; lean_object* v___x_232_; lean_object* v_toZero_233_; lean_object* v___x_234_; lean_object* v___x_235_; uint8_t v___x_236_; 
v_head_227_ = lean_ctor_get(v_a_224_, 0);
lean_inc_n(v_head_227_, 2);
v_tail_228_ = lean_ctor_get(v_a_224_, 1);
lean_inc(v_tail_228_);
lean_dec_ref_known(v_a_224_, 2);
v___x_229_ = lean_unsigned_to_nat(2013265921u);
v___x_230_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_initialRoot___closed__1);
v_toCommSemiring_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc_ref(v_toCommSemiring_231_);
v___x_232_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_toCommSemiring_231_);
v_toZero_233_ = lean_ctor_get(v___x_232_, 1);
lean_inc(v_toZero_233_);
lean_dec_ref(v___x_232_);
v___x_234_ = lean_obj_once(&lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0, &lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0_once, _init_lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow___closed__0);
lean_inc_ref(v_trace_223_);
v___x_235_ = lp_swirl_x2dfv_Fundamentals_Air_Trace_col___redArg(v_trace_223_, v___x_234_, v_head_227_);
v___x_236_ = lp_mathlib_ZMod_decidableEq(v___x_229_, v___x_235_, v_toZero_233_);
lean_dec(v_toZero_233_);
lean_dec(v___x_235_);
if (v___x_236_ == 0)
{
lean_object* v___x_237_; lean_object* v___x_238_; 
lean_inc_ref(v_trace_223_);
v___x_237_ = lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_eventAtRow(v_trace_223_, v_head_227_);
v___x_238_ = lean_array_push(v_a_225_, v___x_237_);
v_a_224_ = v_tail_228_;
v_a_225_ = v___x_238_;
goto _start;
}
else
{
lean_dec(v_head_227_);
v_a_224_ = v_tail_228_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_events(lean_object* v_trace_243_){
_start:
{
lean_object* v_height_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v_height_244_ = lean_ctor_get(v_trace_243_, 0);
lean_inc(v_height_244_);
v___x_245_ = l_List_finRange(v_height_244_);
v___x_246_ = ((lean_object*)(lp_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View_events___closed__0));
v___x_247_ = lp_openvm_x2dfv_List_filterMapTR_go___at___00VM_Spec_Airs_System_MemoryMerkleAir_View_events_spec__0(v_trace_243_, v___x_245_, v___x_246_);
return v___x_247_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear_Field(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_Extraction_Schema(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Memory_Events(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_View(uint8_t builtin) {
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
res = initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Airs_System_MemoryMerkleAir_Extraction_Schema(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Memory_Events(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
