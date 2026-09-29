// Lean compiler output
// Module: Mathlib.Data.Nat.Factors
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Ring.List public import Mathlib.Data.Nat.GCD.Basic public import Mathlib.Data.Nat.Prime.Basic public import Mathlib.Data.List.Prime public import Mathlib.Data.List.Sort public import Mathlib.Data.List.Perm.Subperm
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_minFac(lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactorsList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactorsList___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactorsList(lean_object* v_x_1_){
_start:
{
lean_object* v_zero_2_; uint8_t v_isZero_3_; 
v_zero_2_ = lean_unsigned_to_nat(0u);
v_isZero_3_ = lean_nat_dec_eq(v_x_1_, v_zero_2_);
if (v_isZero_3_ == 1)
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
else
{
lean_object* v_one_5_; lean_object* v_n_6_; uint8_t v_isZero_7_; 
v_one_5_ = lean_unsigned_to_nat(1u);
v_n_6_ = lean_nat_sub(v_x_1_, v_one_5_);
v_isZero_7_ = lean_nat_dec_eq(v_n_6_, v_zero_2_);
if (v_isZero_7_ == 1)
{
lean_object* v___x_8_; 
lean_dec(v_n_6_);
v___x_8_ = lean_box(0);
return v___x_8_;
}
else
{
lean_object* v_n_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v_m_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v_n_9_ = lean_nat_sub(v_n_6_, v_one_5_);
lean_dec(v_n_6_);
v___x_10_ = lean_unsigned_to_nat(2u);
v___x_11_ = lean_nat_add(v_n_9_, v___x_10_);
lean_dec(v_n_9_);
v_m_12_ = lp_mathlib_Nat_minFac(v___x_11_);
v___x_13_ = lean_nat_div(v___x_11_, v_m_12_);
lean_dec(v___x_11_);
v___x_14_ = lp_mathlib_Nat_primeFactorsList(v___x_13_);
lean_dec(v___x_13_);
v___x_15_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_15_, 0, v_m_12_);
lean_ctor_set(v___x_15_, 1, v___x_14_);
return v___x_15_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactorsList___boxed(lean_object* v_x_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Nat_primeFactorsList(v_x_16_);
lean_dec(v_x_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter___redArg(lean_object* v_x_18_, lean_object* v_h__1_19_, lean_object* v_h__2_20_, lean_object* v_h__3_21_){
_start:
{
lean_object* v_zero_22_; uint8_t v_isZero_23_; 
v_zero_22_ = lean_unsigned_to_nat(0u);
v_isZero_23_ = lean_nat_dec_eq(v_x_18_, v_zero_22_);
if (v_isZero_23_ == 1)
{
lean_object* v___x_24_; lean_object* v___x_25_; 
lean_dec(v_h__3_21_);
lean_dec(v_h__2_20_);
v___x_24_ = lean_box(0);
v___x_25_ = lean_apply_1(v_h__1_19_, v___x_24_);
return v___x_25_;
}
else
{
lean_object* v_one_26_; lean_object* v_n_27_; uint8_t v_isZero_28_; 
lean_dec(v_h__1_19_);
v_one_26_ = lean_unsigned_to_nat(1u);
v_n_27_ = lean_nat_sub(v_x_18_, v_one_26_);
v_isZero_28_ = lean_nat_dec_eq(v_n_27_, v_zero_22_);
if (v_isZero_28_ == 1)
{
lean_object* v___x_29_; lean_object* v___x_30_; 
lean_dec(v_n_27_);
lean_dec(v_h__3_21_);
v___x_29_ = lean_box(0);
v___x_30_ = lean_apply_1(v_h__2_20_, v___x_29_);
return v___x_30_;
}
else
{
lean_object* v_n_31_; lean_object* v___x_32_; 
lean_dec(v_h__2_20_);
v_n_31_ = lean_nat_sub(v_n_27_, v_one_26_);
lean_dec(v_n_27_);
v___x_32_ = lean_apply_1(v_h__3_21_, v_n_31_);
return v___x_32_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter___redArg___boxed(lean_object* v_x_33_, lean_object* v_h__1_34_, lean_object* v_h__2_35_, lean_object* v_h__3_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter___redArg(v_x_33_, v_h__1_34_, v_h__2_35_, v_h__3_36_);
lean_dec(v_x_33_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter(lean_object* v_motive_38_, lean_object* v_x_39_, lean_object* v_h__1_40_, lean_object* v_h__2_41_, lean_object* v_h__3_42_){
_start:
{
lean_object* v_zero_43_; uint8_t v_isZero_44_; 
v_zero_43_ = lean_unsigned_to_nat(0u);
v_isZero_44_ = lean_nat_dec_eq(v_x_39_, v_zero_43_);
if (v_isZero_44_ == 1)
{
lean_object* v___x_45_; lean_object* v___x_46_; 
lean_dec(v_h__3_42_);
lean_dec(v_h__2_41_);
v___x_45_ = lean_box(0);
v___x_46_ = lean_apply_1(v_h__1_40_, v___x_45_);
return v___x_46_;
}
else
{
lean_object* v_one_47_; lean_object* v_n_48_; uint8_t v_isZero_49_; 
lean_dec(v_h__1_40_);
v_one_47_ = lean_unsigned_to_nat(1u);
v_n_48_ = lean_nat_sub(v_x_39_, v_one_47_);
v_isZero_49_ = lean_nat_dec_eq(v_n_48_, v_zero_43_);
if (v_isZero_49_ == 1)
{
lean_object* v___x_50_; lean_object* v___x_51_; 
lean_dec(v_n_48_);
lean_dec(v_h__3_42_);
v___x_50_ = lean_box(0);
v___x_51_ = lean_apply_1(v_h__2_41_, v___x_50_);
return v___x_51_;
}
else
{
lean_object* v_n_52_; lean_object* v___x_53_; 
lean_dec(v_h__2_41_);
v_n_52_ = lean_nat_sub(v_n_48_, v_one_47_);
lean_dec(v_n_48_);
v___x_53_ = lean_apply_1(v_h__3_42_, v_n_52_);
return v___x_53_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter___boxed(lean_object* v_motive_54_, lean_object* v_x_55_, lean_object* v_h__1_56_, lean_object* v_h__2_57_, lean_object* v_h__3_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib___private_Mathlib_Data_Nat_Factors_0__Nat_primeFactorsList_match__1_splitter(v_motive_54_, v_x_55_, v_h__1_56_, v_h__2_57_, v_h__3_58_);
lean_dec(v_x_55_);
return v_res_59_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_List(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_GCD_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Prime(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Sort(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Subperm(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factors(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_GCD_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Prime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Subperm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Factors(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_List(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_GCD_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Prime(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Sort(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Subperm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Factors(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_GCD_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Prime_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Prime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Subperm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Factors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Factors(builtin);
}
#ifdef __cplusplus
}
#endif
