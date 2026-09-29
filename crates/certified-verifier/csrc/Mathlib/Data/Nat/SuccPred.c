// Lean compiler output
// Module: Mathlib.Data.Nat.SuccPred
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Algebra.Ring.Nat public import Mathlib.Algebra.Order.Monoid.Unbundled.WithTop public import Mathlib.Algebra.Order.Sub.Unbundled.Basic public import Mathlib.Algebra.Order.SuccPred public import Mathlib.Data.Fin.Basic public import Mathlib.Order.Nat public import Mathlib.Order.SuccPred.Archimedean public import Mathlib.Order.SuccPred.WithBot
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
lean_object* l_Nat_pred___boxed(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Nat_instSuccOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_instSuccOrder___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instSuccOrder___closed__0 = (const lean_object*)&lp_mathlib_Nat_instSuccOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instSuccOrder = (const lean_object*)&lp_mathlib_Nat_instSuccOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instSuccAddOrder = (const lean_object*)&lp_mathlib_Nat_instSuccOrder___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_instPredOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_pred___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instPredOrder___closed__0 = (const lean_object*)&lp_mathlib_Nat_instPredOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instPredOrder = (const lean_object*)&lp_mathlib_Nat_instPredOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instPredSubOrder = (const lean_object*)&lp_mathlib_Nat_instPredOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_unsigned_to_nat(1u);
v___x_3_ = lean_nat_add(v_n_1_, v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instSuccOrder___lam__0___boxed(lean_object* v_n_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_Nat_instSuccOrder___lam__0(v_n_4_);
lean_dec(v_n_4_);
return v_res_5_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Unbundled_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_SuccPred(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Unbundled_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_WithBot(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Unbundled_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Unbundled_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
}
#ifdef __cplusplus
}
#endif
