// Lean compiler output
// Module: Mathlib.Data.Finset.NatAntidiagonal
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Antidiag.Prod public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Data.Multiset.NatAntidiagonal
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
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_List_Nat_antidiagonal___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_Nat_antidiagonal___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0 = (const lean_object*)&lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Finset_Nat_instHasAntidiagonal = (const lean_object*)&lp_mathlib_Finset_Nat_instHasAntidiagonal___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_Nat_antidiagonalEquivFin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___closed__0 = (const lean_object*)&lp_mathlib_Finset_Nat_antidiagonalEquivFin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0(lean_object* v_x_3_){
_start:
{
lean_object* v_fst_4_; 
v_fst_4_ = lean_ctor_get(v_x_3_, 0);
lean_inc(v_fst_4_);
return v_fst_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__0(v_x_5_);
lean_dec_ref(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1(lean_object* v_n_7_, lean_object* v_x_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_nat_sub(v_n_7_, v_x_8_);
v___x_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_10_, 0, v_x_8_);
lean_ctor_set(v___x_10_, 1, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1___boxed(lean_object* v_n_11_, lean_object* v_x_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1(v_n_11_, v_x_12_);
lean_dec(v_n_11_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Nat_antidiagonalEquivFin(lean_object* v_n_15_){
_start:
{
lean_object* v___f_16_; lean_object* v___f_17_; lean_object* v___x_18_; 
v___f_16_ = ((lean_object*)(lp_mathlib_Finset_Nat_antidiagonalEquivFin___closed__0));
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_Finset_Nat_antidiagonalEquivFin___lam__1___boxed), 2, 1);
lean_closure_set(v___f_17_, 0, v_n_15_);
v___x_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_18_, 0, v___f_16_);
lean_ctor_set(v___x_18_, 1, v___f_17_);
return v___x_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_NatAntidiagonal(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_NatAntidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_NatAntidiagonal(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Antidiag_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_NatAntidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_NatAntidiagonal(builtin);
}
#ifdef __cplusplus
}
#endif
