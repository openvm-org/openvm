// Lean compiler output
// Module: Mathlib.Data.Nat.Cast.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Divisibility.Hom public import Mathlib.Algebra.Group.Even public import Mathlib.Algebra.Group.Nat.Hom public import Mathlib.Algebra.Ring.Hom.Defs public import Mathlib.Algebra.Ring.Nat
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
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_castAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_castAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_castRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_castRingHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_uniqueRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_uniqueRingHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instNatCast___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instNatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instNatCast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_castAddMonoidHom___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toNatCast_2_; lean_object* v___x_3_; 
v_toNatCast_2_ = lean_ctor_get(v_inst_1_, 0);
lean_inc(v_toNatCast_2_);
lean_dec_ref(v_inst_1_);
v___x_3_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_3_, 0, lean_box(0));
lean_closure_set(v___x_3_, 1, v_toNatCast_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_castAddMonoidHom(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_Nat_castAddMonoidHom___redArg(v_inst_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_castRingHom___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; lean_object* v_toNatCast_9_; lean_object* v___x_10_; 
v___x_8_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_7_);
v_toNatCast_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_toNatCast_9_);
lean_dec_ref(v___x_8_);
v___x_10_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_10_, 0, lean_box(0));
lean_closure_set(v___x_10_, 1, v_toNatCast_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_castRingHom(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Nat_castRingHom___redArg(v_inst_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_uniqueRingHom___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Nat_castRingHom___redArg(v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_uniqueRingHom(lean_object* v_R_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Nat_castRingHom___redArg(v_inst_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instNatCast___redArg___lam__0(lean_object* v_inst_19_, lean_object* v_n_20_, lean_object* v_x_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_apply_2(v_inst_19_, v_x_21_, v_n_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instNatCast___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instNatCast___redArg___lam__0), 3, 1);
lean_closure_set(v___f_24_, 0, v_inst_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instNatCast(lean_object* v_00_u03b1_25_, lean_object* v_00_u03c0_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___f_28_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instNatCast___redArg___lam__0), 3, 1);
lean_closure_set(v___f_28_, 0, v_inst_27_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat___redArg___lam__0(lean_object* v_inst_29_, lean_object* v_x_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lean_apply_1(v_inst_29_, v_x_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOfNat___redArg___lam__0), 2, 1);
lean_closure_set(v___f_33_, 0, v_inst_32_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat(lean_object* v_00_u03b1_34_, lean_object* v_00_u03c0_35_, lean_object* v_n_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOfNat___redArg___lam__0), 2, 1);
lean_closure_set(v___f_38_, 0, v_inst_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instOfNat___boxed(lean_object* v_00_u03b1_39_, lean_object* v_00_u03c0_40_, lean_object* v_n_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Pi_instOfNat(v_00_u03b1_39_, v_00_u03c0_40_, v_n_41_, v_inst_42_);
lean_dec(v_n_41_);
return v_res_43_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Divisibility_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Even(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Divisibility_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Divisibility_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Even(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Divisibility_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Cast_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
