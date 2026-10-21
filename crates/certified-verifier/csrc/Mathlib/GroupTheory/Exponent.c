// Lean compiler output
// Module: Mathlib.GroupTheory.Exponent
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GCDMonoid.Finset public import Mathlib.Algebra.GCDMonoid.Nat public import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset public import Mathlib.Data.Nat.Factorization.LCM public import Mathlib.GroupTheory.OrderOfElement public import Mathlib.Tactic.Peel
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
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc_ref(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_commMonoidOfExponentTwo___redArg(v_inst_2_);
lean_dec_ref(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo(lean_object* v_G_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_hG_7_){
_start:
{
lean_inc_ref(v_inst_5_);
return v_inst_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commMonoidOfExponentTwo___boxed(lean_object* v_G_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_hG_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_commMonoidOfExponentTwo(v_G_8_, v_inst_9_, v_inst_10_, v_hG_11_);
lean_dec_ref(v_inst_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo___redArg(lean_object* v_inst_13_){
_start:
{
lean_inc_ref(v_inst_13_);
return v_inst_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo___redArg___boxed(lean_object* v_inst_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_addCommMonoidOfExponentTwo___redArg(v_inst_14_);
lean_dec_ref(v_inst_14_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo(lean_object* v_G_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_hG_19_){
_start:
{
lean_inc_ref(v_inst_17_);
return v_inst_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommMonoidOfExponentTwo___boxed(lean_object* v_G_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_hG_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_addCommMonoidOfExponentTwo(v_G_20_, v_inst_21_, v_inst_22_, v_hG_23_);
lean_dec_ref(v_inst_21_);
return v_res_24_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_LCM(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Peel(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Exponent(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_LCM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Peel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Exponent(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_LCM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Peel(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Exponent(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_GroupWithZero_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorization_LCM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_OrderOfElement(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Peel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Exponent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Exponent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Exponent(builtin);
}
#ifdef __cplusplus
}
#endif
