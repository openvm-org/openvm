// Lean compiler output
// Module: Mathlib.Algebra.Group.Fin.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.NeZero public import Mathlib.Data.Nat.Cast.Defs public import Mathlib.Data.Fin.Rev
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
lean_object* l_Fin_add___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* l_Fin_neg___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Fin_sub___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommSemigroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instInvolutiveNeg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddLeftCancelSemigroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddRightCancelSemigroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommSemigroup(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_2_, 0, v_n_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid___redArg(lean_object* v_n_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
lean_inc(v_n_3_);
v___x_4_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_4_, 0, v_n_3_);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_nat_mod(v___x_5_, v_n_3_);
lean_dec(v_n_3_);
lean_inc_ref(v___x_4_);
lean_inc(v___x_6_);
v___x_7_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_7_, 0, lean_box(0));
lean_closure_set(v___x_7_, 1, v___x_6_);
lean_closure_set(v___x_7_, 2, v___x_4_);
v___x_8_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_8_, 0, v___x_6_);
lean_ctor_set(v___x_8_, 1, v___x_4_);
lean_ctor_set(v___x_8_, 2, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommMonoid(lean_object* v_n_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_Fin_addCommMonoid___redArg(v_n_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0(lean_object* v_n_12_, lean_object* v_i_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_nat_mod(v_i_13_, v_n_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0___boxed(lean_object* v_n_15_, lean_object* v_i_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0(v_n_15_, v_i_16_);
lean_dec(v_i_16_);
lean_dec(v_n_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne___redArg(lean_object* v_n_18_){
_start:
{
lean_object* v___f_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
lean_inc_n(v_n_18_, 2);
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instAddMonoidWithOne___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_19_, 0, v_n_18_);
v___x_20_ = lp_mathlib_Fin_addCommMonoid___redArg(v_n_18_);
v___x_21_ = lean_unsigned_to_nat(1u);
v___x_22_ = lean_nat_mod(v___x_21_, v_n_18_);
lean_dec(v_n_18_);
v___x_23_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_23_, 0, v___f_19_);
lean_ctor_set(v___x_23_, 1, v___x_20_);
lean_ctor_set(v___x_23_, 2, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddMonoidWithOne(lean_object* v_n_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Fin_instAddMonoidWithOne___redArg(v_n_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup___redArg(lean_object* v_n_27_){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v_toZero_31_; lean_object* v___f_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
lean_inc_n(v_n_27_, 3);
v___x_28_ = lp_mathlib_Fin_addCommMonoid___redArg(v_n_27_);
v___x_29_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_28_);
v___x_30_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_29_);
v_toZero_31_ = lean_ctor_get(v___x_30_, 0);
lean_inc_n(v_toZero_31_, 2);
lean_dec_ref(v___x_30_);
v___f_32_ = lean_alloc_closure((void*)(l_Fin_neg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_32_, 0, v_n_27_);
v___x_33_ = lean_alloc_closure((void*)(l_Fin_sub___boxed), 3, 1);
lean_closure_set(v___x_33_, 0, v_n_27_);
v___x_34_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_34_, 0, v_n_27_);
lean_inc_ref(v___x_34_);
v___x_35_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_35_, 0, lean_box(0));
lean_closure_set(v___x_35_, 1, v_toZero_31_);
lean_closure_set(v___x_35_, 2, v___x_34_);
lean_inc_ref(v___f_32_);
v___x_36_ = lean_alloc_closure((void*)(lp_mathlib_zsmulRec___boxed), 7, 5);
lean_closure_set(v___x_36_, 0, lean_box(0));
lean_closure_set(v___x_36_, 1, v_toZero_31_);
lean_closure_set(v___x_36_, 2, v___x_34_);
lean_closure_set(v___x_36_, 3, v___f_32_);
lean_closure_set(v___x_36_, 4, v___x_35_);
v___x_37_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_37_, 0, v___x_28_);
lean_ctor_set(v___x_37_, 1, v___f_32_);
lean_ctor_set(v___x_37_, 2, v___x_33_);
lean_ctor_set(v___x_37_, 3, v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addCommGroup(lean_object* v_n_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Fin_addCommGroup___redArg(v_n_38_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instInvolutiveNeg(lean_object* v_n_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(l_Fin_neg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_42_, 0, v_n_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddLeftCancelSemigroup(lean_object* v_n_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_44_, 0, v_n_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instAddRightCancelSemigroup(lean_object* v_n_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_alloc_closure((void*)(l_Fin_add___boxed), 3, 1);
lean_closure_set(v___x_46_, 0, v_n_45_);
return v___x_46_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Rev(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Rev(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Rev(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Rev(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Fin_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
