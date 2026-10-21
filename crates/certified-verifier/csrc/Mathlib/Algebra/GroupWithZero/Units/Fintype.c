// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Units.Fintype
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Int.Units public import Mathlib.Data.Fintype.Prod public import Mathlib.Data.Fintype.Sum public import Mathlib.SetTheory.Cardinal.Finite public import Mathlib.Algebra.GroupWithZero.Units.Equiv
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
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* l_Int_instDecidableEq___boxed(lean_object*, lean_object*);
uint8_t lp_mathlib_Units_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_ndinsert___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_unitsEquivProdSubtype(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_UnitsInt_fintype___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UnitsInt_fintype___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_UnitsInt_fintype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_UnitsInt_fintype___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_UnitsInt_fintype___closed__0 = (const lean_object*)&lp_mathlib_UnitsInt_fintype___closed__0_value;
static lean_once_cell_t lp_mathlib_UnitsInt_fintype___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_UnitsInt_fintype___closed__1;
static lean_once_cell_t lp_mathlib_UnitsInt_fintype___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_UnitsInt_fintype___closed__2;
static lean_once_cell_t lp_mathlib_UnitsInt_fintype___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_UnitsInt_fintype___closed__3;
static lean_once_cell_t lp_mathlib_UnitsInt_fintype___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_UnitsInt_fintype___closed__4;
static lean_once_cell_t lp_mathlib_UnitsInt_fintype___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_UnitsInt_fintype___closed__5;
static lean_once_cell_t lp_mathlib_UnitsInt_fintype___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_UnitsInt_fintype___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_UnitsInt_fintype;
LEAN_EXPORT uint8_t lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_UnitsInt_fintype___lam__0(lean_object* v_a_1_, lean_object* v_b_2_){
_start:
{
lean_object* v___x_3_; uint8_t v___x_4_; 
v___x_3_ = lean_alloc_closure((void*)(l_Int_instDecidableEq___boxed), 2, 0);
v___x_4_ = lp_mathlib_Units_instDecidableEq___redArg(v___x_3_, v_a_1_, v_b_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UnitsInt_fintype___lam__0___boxed(lean_object* v_a_5_, lean_object* v_b_6_){
_start:
{
uint8_t v_res_7_; lean_object* v_r_8_; 
v_res_7_ = lp_mathlib_UnitsInt_fintype___lam__0(v_a_5_, v_b_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__1(void){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lean_unsigned_to_nat(1u);
v___x_11_ = lean_nat_to_int(v___x_10_);
return v___x_11_;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__2(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_obj_once(&lp_mathlib_UnitsInt_fintype___closed__1, &lp_mathlib_UnitsInt_fintype___closed__1_once, _init_lp_mathlib_UnitsInt_fintype___closed__1);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
lean_ctor_set(v___x_13_, 1, v___x_12_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__3(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_obj_once(&lp_mathlib_UnitsInt_fintype___closed__1, &lp_mathlib_UnitsInt_fintype___closed__1_once, _init_lp_mathlib_UnitsInt_fintype___closed__1);
v___x_15_ = lean_int_neg(v___x_14_);
return v___x_15_;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__4(void){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_16_ = lean_obj_once(&lp_mathlib_UnitsInt_fintype___closed__3, &lp_mathlib_UnitsInt_fintype___closed__3_once, _init_lp_mathlib_UnitsInt_fintype___closed__3);
v___x_17_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
lean_ctor_set(v___x_17_, 1, v___x_16_);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__5(void){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_18_ = lean_box(0);
v___x_19_ = lean_obj_once(&lp_mathlib_UnitsInt_fintype___closed__4, &lp_mathlib_UnitsInt_fintype___closed__4_once, _init_lp_mathlib_UnitsInt_fintype___closed__4);
v___x_20_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___x_18_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__6(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___f_23_; lean_object* v___x_24_; 
v___x_21_ = lean_obj_once(&lp_mathlib_UnitsInt_fintype___closed__5, &lp_mathlib_UnitsInt_fintype___closed__5_once, _init_lp_mathlib_UnitsInt_fintype___closed__5);
v___x_22_ = lean_obj_once(&lp_mathlib_UnitsInt_fintype___closed__2, &lp_mathlib_UnitsInt_fintype___closed__2_once, _init_lp_mathlib_UnitsInt_fintype___closed__2);
v___f_23_ = ((lean_object*)(lp_mathlib_UnitsInt_fintype___closed__0));
v___x_24_ = lp_mathlib_Multiset_ndinsert___redArg(v___f_23_, v___x_22_, v___x_21_);
return v___x_24_;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype(void){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_obj_once(&lp_mathlib_UnitsInt_fintype___closed__6, &lp_mathlib_UnitsInt_fintype___closed__6_once, _init_lp_mathlib_UnitsInt_fintype___closed__6);
return v___x_25_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0(lean_object* v_toMul_26_, lean_object* v_inst_27_, lean_object* v_toOne_28_, lean_object* v_a_29_){
_start:
{
lean_object* v_fst_30_; lean_object* v_snd_31_; lean_object* v___x_32_; lean_object* v___x_33_; uint8_t v___x_34_; 
v_fst_30_ = lean_ctor_get(v_a_29_, 0);
lean_inc_n(v_fst_30_, 2);
v_snd_31_ = lean_ctor_get(v_a_29_, 1);
lean_inc_n(v_snd_31_, 2);
lean_dec_ref(v_a_29_);
lean_inc(v_toMul_26_);
v___x_32_ = lean_apply_2(v_toMul_26_, v_fst_30_, v_snd_31_);
lean_inc_ref(v_inst_27_);
lean_inc(v_toOne_28_);
v___x_33_ = lean_apply_2(v_inst_27_, v___x_32_, v_toOne_28_);
v___x_34_ = lean_unbox(v___x_33_);
if (v___x_34_ == 0)
{
uint8_t v___x_35_; 
lean_dec(v_snd_31_);
lean_dec(v_fst_30_);
lean_dec(v_toOne_28_);
lean_dec_ref(v_inst_27_);
lean_dec(v_toMul_26_);
v___x_35_ = lean_unbox(v___x_33_);
return v___x_35_;
}
else
{
lean_object* v___x_36_; lean_object* v___x_37_; uint8_t v___x_38_; 
v___x_36_ = lean_apply_2(v_toMul_26_, v_snd_31_, v_fst_30_);
v___x_37_ = lean_apply_2(v_inst_27_, v___x_36_, v_toOne_28_);
v___x_38_ = lean_unbox(v___x_37_);
return v___x_38_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0___boxed(lean_object* v_toMul_39_, lean_object* v_inst_40_, lean_object* v_toOne_41_, lean_object* v_a_42_){
_start:
{
uint8_t v_res_43_; lean_object* v_r_44_; 
v_res_43_ = lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0(v_toMul_39_, v_inst_40_, v_toOne_41_, v_a_42_);
v_r_44_ = lean_box(v_res_43_);
return v_r_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v_toOne_50_; lean_object* v_toMul_51_; lean_object* v___f_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_48_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_45_);
v___x_49_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_48_);
v_toOne_50_ = lean_ctor_get(v___x_49_, 0);
lean_inc(v_toOne_50_);
v_toMul_51_ = lean_ctor_get(v___x_49_, 1);
lean_inc(v_toMul_51_);
lean_dec_ref(v___x_49_);
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_52_, 0, v_toMul_51_);
lean_closure_set(v___f_52_, 1, v_inst_47_);
lean_closure_set(v___f_52_, 2, v_toOne_50_);
lean_inc(v_inst_46_);
v___x_53_ = lp_mathlib_Multiset_product___redArg(v_inst_46_, v_inst_46_);
v___x_54_ = lp_mathlib_Subtype_fintype___redArg(v___f_52_, v___x_53_);
v___x_55_ = lp_mathlib_unitsEquivProdSubtype(lean_box(0), v_inst_45_);
v___x_56_ = lp_mathlib_Equiv_symm___redArg(v___x_55_);
v___x_57_ = lp_mathlib_Fintype_ofEquiv___redArg(v___x_54_, v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___boxed(lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(v_inst_58_, v_inst_59_, v_inst_60_);
lean_dec_ref(v_inst_58_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(v_inst_63_, v_inst_64_, v_inst_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___boxed(lean_object* v_00_u03b1_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_instFintypeUnitsOfDecidableEq(v_00_u03b1_67_, v_inst_68_, v_inst_69_, v_inst_70_);
lean_dec_ref(v_inst_68_);
return v_res_71_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_UnitsInt_fintype = _init_lp_mathlib_UnitsInt_fintype();
lean_mark_persistent(lp_mathlib_UnitsInt_fintype);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Fintype(builtin);
}
#ifdef __cplusplus
}
#endif
