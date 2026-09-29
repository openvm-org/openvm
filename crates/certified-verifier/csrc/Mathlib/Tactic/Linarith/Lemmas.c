// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Lemmas
// Imports: public import Init public meta import Init public meta import Batteries.Tactic.Lint.Basic public meta import Mathlib.Data.Ineq public import Mathlib.Data.Ineq public import Mathlib.Data.Nat.Cast.Order.Ring public meta import Mathlib.Tactic.ToAdditive
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Linarith"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mul_eq"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__3_value),LEAN_SCALAR_PTR_LITERAL(110, 226, 37, 17, 1, 97, 213, 110)}};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "mul_nonpos"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__5_value),LEAN_SCALAR_PTR_LITERAL(98, 6, 17, 252, 182, 17, 126, 17)}};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "mul_neg"};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__7_value),LEAN_SCALAR_PTR_LITERAL(72, 102, 201, 109, 157, 25, 172, 68)}};
static const lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName(uint8_t v_x_22_){
_start:
{
switch(v_x_22_)
{
case 0:
{
lean_object* v___x_23_; 
v___x_23_ = ((lean_object*)(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__4));
return v___x_23_;
}
case 1:
{
lean_object* v___x_24_; 
v___x_24_ = ((lean_object*)(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__6));
return v___x_24_;
}
default: 
{
lean_object* v___x_25_; 
v___x_25_ = ((lean_object*)(lp_mathlib_Mathlib_Ineq_toConstMulName___closed__8));
return v___x_25_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Ineq_toConstMulName___boxed(lean_object* v_x_26_){
_start:
{
uint8_t v_x_97__boxed_27_; lean_object* v_res_28_; 
v_x_97__boxed_27_ = lean_unbox(v_x_26_);
v_res_28_ = lp_mathlib_Mathlib_Ineq_toConstMulName(v_x_97__boxed_27_);
return v_res_28_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Ineq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
