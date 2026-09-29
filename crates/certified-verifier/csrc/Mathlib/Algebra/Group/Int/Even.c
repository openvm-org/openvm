// Lean compiler output
// Module: Mathlib.Algebra.Group.Int.Even
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Int.Defs public import Mathlib.Algebra.Group.Nat.Even public import Mathlib.Data.Int.Sqrt public import Mathlib.Tactic.Attr.Core
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
lean_object* lp_mathlib_Int_sqrt(lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
lean_object* lean_int_emod(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Int_instDecidablePredEven___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_instDecidablePredEven___closed__0;
static lean_once_cell_t lp_mathlib_Int_instDecidablePredEven___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_instDecidablePredEven___closed__1;
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredEven(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredEven___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredIsSquare(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredIsSquare___boxed(lean_object*);
static lean_object* _init_lp_mathlib_Int_instDecidablePredEven___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(2u);
v___x_2_ = lean_nat_to_int(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_Int_instDecidablePredEven___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_unsigned_to_nat(0u);
v___x_4_ = lean_nat_to_int(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredEven(lean_object* v_x_5_){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; uint8_t v___x_9_; 
v___x_6_ = lean_obj_once(&lp_mathlib_Int_instDecidablePredEven___closed__0, &lp_mathlib_Int_instDecidablePredEven___closed__0_once, _init_lp_mathlib_Int_instDecidablePredEven___closed__0);
v___x_7_ = lean_int_emod(v_x_5_, v___x_6_);
v___x_8_ = lean_obj_once(&lp_mathlib_Int_instDecidablePredEven___closed__1, &lp_mathlib_Int_instDecidablePredEven___closed__1_once, _init_lp_mathlib_Int_instDecidablePredEven___closed__1);
v___x_9_ = lean_int_dec_eq(v___x_7_, v___x_8_);
lean_dec(v___x_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredEven___boxed(lean_object* v_x_10_){
_start:
{
uint8_t v_res_11_; lean_object* v_r_12_; 
v_res_11_ = lp_mathlib_Int_instDecidablePredEven(v_x_10_);
lean_dec(v_x_10_);
v_r_12_ = lean_box(v_res_11_);
return v_r_12_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredIsSquare(lean_object* v_m_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; uint8_t v___x_16_; 
v___x_14_ = lp_mathlib_Int_sqrt(v_m_13_);
v___x_15_ = lean_int_mul(v___x_14_, v___x_14_);
lean_dec(v___x_14_);
v___x_16_ = lean_int_dec_eq(v___x_15_, v_m_13_);
lean_dec(v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredIsSquare___boxed(lean_object* v_m_17_){
_start:
{
uint8_t v_res_18_; lean_object* v_r_19_; 
v_res_18_ = lp_mathlib_Int_instDecidablePredIsSquare(v_m_17_);
lean_dec(v_m_17_);
v_r_19_ = lean_box(v_res_18_);
return v_r_19_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Even(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Sqrt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Even(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Int_Even(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Even(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Sqrt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Even(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Int_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Int_Even(builtin);
}
#ifdef __cplusplus
}
#endif
