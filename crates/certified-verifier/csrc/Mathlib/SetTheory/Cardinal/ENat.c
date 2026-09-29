// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.ENat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Hom.Ring public import Mathlib.Data.ENat.SuccOrder public import Mathlib.SetTheory.Cardinal.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Cardinal_instCoeENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_ofENat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_instCoeENat___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_instCoeENat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_instCoeENat = (const lean_object*)&lp_mathlib_Cardinal_instCoeENat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_ofENatHom = (const lean_object*)&lp_mathlib_Cardinal_instCoeENat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0(lean_object* v_a_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0___boxed(lean_object* v_a_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_Nat_cast___at___00Cardinal_ofENat_spec__0(v_a_3_);
lean_dec(v_a_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat(lean_object* v_x_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_ofENat___boxed(lean_object* v_x_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Cardinal_ofENat(v_x_7_);
lean_dec(v_x_7_);
return v_res_8_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_SuccOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_SuccOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Hom_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_SuccOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Cardinal_ENat(builtin);
}
#ifdef __cplusplus
}
#endif
