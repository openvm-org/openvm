// Lean compiler output
// Module: Mathlib.Algebra.Group.TypeTags.Finite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.TypeTags.Basic public import Mathlib.Basic.Finite.Defs public import Mathlib.Data.Fintype.Card
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
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
static lean_once_cell_t lp_mathlib_Additive_fintype___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Additive_fintype___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Additive_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Additive_fintype(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Multiplicative_fintype___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Multiplicative_fintype___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_fintype(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Additive_fintype___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_fintype___redArg(lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_obj_once(&lp_mathlib_Additive_fintype___redArg___closed__0, &lp_mathlib_Additive_fintype___redArg___closed__0_once, _init_lp_mathlib_Additive_fintype___redArg___closed__0);
v___x_4_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_2_, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Additive_fintype(lean_object* v_00_u03b1_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Additive_fintype___redArg(v_inst_6_);
return v___x_7_;
}
}
static lean_object* _init_lp_mathlib_Multiplicative_fintype___redArg___closed__0(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_fintype___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lean_obj_once(&lp_mathlib_Multiplicative_fintype___redArg___closed__0, &lp_mathlib_Multiplicative_fintype___redArg___closed__0_once, _init_lp_mathlib_Multiplicative_fintype___redArg___closed__0);
v___x_11_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_9_, v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiplicative_fintype(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_Multiplicative_fintype___redArg(v_inst_13_);
return v___x_14_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Finite(builtin);
}
#ifdef __cplusplus
}
#endif
