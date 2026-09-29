// Lean compiler output
// Module: Mathlib.Data.Nat.Find
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Basic public import Mathlib.Tactic.Push public import Batteries.Tactic.Init
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
lean_object* l_WellFounded_fixC___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_findX___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_findX(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_find___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_find(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_findGreatest___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_findGreatest(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_findX___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_m_2_, lean_object* v_IH_3_, lean_object* v_al_4_){
_start:
{
lean_object* v___x_5_; uint8_t v___x_6_; 
lean_inc(v_m_2_);
v___x_5_ = lean_apply_1(v_inst_1_, v_m_2_);
v___x_6_ = lean_unbox(v___x_5_);
if (v___x_6_ == 0)
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_7_ = lean_unsigned_to_nat(1u);
v___x_8_ = lean_nat_add(v_m_2_, v___x_7_);
lean_dec(v_m_2_);
v___x_9_ = lean_apply_3(v_IH_3_, v___x_8_, lean_box(0), lean_box(0));
return v___x_9_;
}
else
{
lean_dec_ref(v_IH_3_);
return v_m_2_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_findX___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; lean_object* v___x_17__overap_13_; lean_object* v___x_14_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Nat_findX___redArg___lam__0), 4, 1);
lean_closure_set(v___f_11_, 0, v_inst_10_);
v___x_12_ = lean_unsigned_to_nat(0u);
v___x_17__overap_13_ = l_WellFounded_fixC___redArg(v___f_11_, v___x_12_);
v___x_14_ = lean_apply_1(v___x_17__overap_13_, lean_box(0));
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_findX(lean_object* v_p_15_, lean_object* v_inst_16_, lean_object* v_H_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Nat_findX___redArg(v_inst_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_find___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_Nat_findX___redArg(v_inst_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_find(lean_object* v_p_21_, lean_object* v_inst_22_, lean_object* v_H_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Nat_findX___redArg(v_inst_22_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_findGreatest___redArg(lean_object* v_inst_25_, lean_object* v_x_26_){
_start:
{
lean_object* v_zero_27_; uint8_t v_isZero_28_; 
v_zero_27_ = lean_unsigned_to_nat(0u);
v_isZero_28_ = lean_nat_dec_eq(v_x_26_, v_zero_27_);
if (v_isZero_28_ == 1)
{
lean_dec(v_x_26_);
lean_dec_ref(v_inst_25_);
return v_zero_27_;
}
else
{
lean_object* v_one_29_; lean_object* v_n_30_; lean_object* v___x_31_; lean_object* v___x_32_; uint8_t v___x_33_; 
v_one_29_ = lean_unsigned_to_nat(1u);
v_n_30_ = lean_nat_sub(v_x_26_, v_one_29_);
lean_dec(v_x_26_);
v___x_31_ = lean_nat_add(v_n_30_, v_one_29_);
lean_inc_ref(v_inst_25_);
lean_inc(v___x_31_);
v___x_32_ = lean_apply_1(v_inst_25_, v___x_31_);
v___x_33_ = lean_unbox(v___x_32_);
if (v___x_33_ == 0)
{
lean_dec(v___x_31_);
v_x_26_ = v_n_30_;
goto _start;
}
else
{
lean_dec(v_n_30_);
lean_dec_ref(v_inst_25_);
return v___x_31_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_findGreatest(lean_object* v_P_35_, lean_object* v_inst_36_, lean_object* v_x_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Nat_findGreatest___redArg(v_inst_36_, v_x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter___redArg(lean_object* v_x_39_, lean_object* v_h__1_40_, lean_object* v_h__2_41_){
_start:
{
lean_object* v_zero_42_; uint8_t v_isZero_43_; 
v_zero_42_ = lean_unsigned_to_nat(0u);
v_isZero_43_ = lean_nat_dec_eq(v_x_39_, v_zero_42_);
if (v_isZero_43_ == 1)
{
lean_object* v___x_44_; lean_object* v___x_45_; 
lean_dec(v_h__2_41_);
v___x_44_ = lean_box(0);
v___x_45_ = lean_apply_1(v_h__1_40_, v___x_44_);
return v___x_45_;
}
else
{
lean_object* v_one_46_; lean_object* v_n_47_; lean_object* v___x_48_; 
lean_dec(v_h__1_40_);
v_one_46_ = lean_unsigned_to_nat(1u);
v_n_47_ = lean_nat_sub(v_x_39_, v_one_46_);
v___x_48_ = lean_apply_1(v_h__2_41_, v_n_47_);
return v___x_48_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter___redArg___boxed(lean_object* v_x_49_, lean_object* v_h__1_50_, lean_object* v_h__2_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter___redArg(v_x_49_, v_h__1_50_, v_h__2_51_);
lean_dec(v_x_49_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter(lean_object* v_motive_53_, lean_object* v_x_54_, lean_object* v_h__1_55_, lean_object* v_h__2_56_){
_start:
{
lean_object* v_zero_57_; uint8_t v_isZero_58_; 
v_zero_57_ = lean_unsigned_to_nat(0u);
v_isZero_58_ = lean_nat_dec_eq(v_x_54_, v_zero_57_);
if (v_isZero_58_ == 1)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
lean_dec(v_h__2_56_);
v___x_59_ = lean_box(0);
v___x_60_ = lean_apply_1(v_h__1_55_, v___x_59_);
return v___x_60_;
}
else
{
lean_object* v_one_61_; lean_object* v_n_62_; lean_object* v___x_63_; 
lean_dec(v_h__1_55_);
v_one_61_ = lean_unsigned_to_nat(1u);
v_n_62_ = lean_nat_sub(v_x_54_, v_one_61_);
v___x_63_ = lean_apply_1(v_h__2_56_, v_n_62_);
return v___x_63_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter___boxed(lean_object* v_motive_64_, lean_object* v_x_65_, lean_object* v_h__1_66_, lean_object* v_h__2_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib___private_Mathlib_Data_Nat_Find_0__Nat_findGreatest_match__1_splitter(v_motive_64_, v_x_65_, v_h__1_66_, v_h__2_67_);
lean_dec(v_x_65_);
return v_res_68_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
}
#ifdef __cplusplus
}
#endif
