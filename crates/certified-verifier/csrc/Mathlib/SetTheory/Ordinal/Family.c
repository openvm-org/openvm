// Lean compiler output
// Module: Mathlib.SetTheory.Ordinal.Family
// Imports: public import Init public meta import Init public import Mathlib.SetTheory.Ordinal.Arithmetic
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
lean_object* lp_mathlib_Ordinal_typein(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Ordinal_typein(lean_box(0), lean_box(0), lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(lean_object* v_f_2_, lean_object* v_i_3_){
_start:
{
lean_object* v___x_4_; lean_object* v_toRelEmbedding_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_4_ = lean_obj_once(&lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0, &lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0_once, _init_lp_mathlib_Ordinal_familyOfBFamily_x27___redArg___closed__0);
v_toRelEmbedding_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc(v_toRelEmbedding_5_);
v___x_6_ = lean_apply_1(v_toRelEmbedding_5_, v_i_3_);
v___x_7_ = lean_apply_2(v_f_2_, v___x_6_, lean_box(0));
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27(lean_object* v_00_u03b1_8_, lean_object* v_00_u03b9_9_, lean_object* v_r_10_, lean_object* v_inst_11_, lean_object* v_o_12_, lean_object* v_ho_13_, lean_object* v_f_14_, lean_object* v_i_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(v_f_14_, v_i_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily_x27___boxed(lean_object* v_00_u03b1_17_, lean_object* v_00_u03b9_18_, lean_object* v_r_19_, lean_object* v_inst_20_, lean_object* v_o_21_, lean_object* v_ho_22_, lean_object* v_f_23_, lean_object* v_i_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Ordinal_familyOfBFamily_x27(v_00_u03b1_17_, v_00_u03b9_18_, v_r_19_, v_inst_20_, v_o_21_, v_ho_22_, v_f_23_, v_i_24_);
lean_dec(v_o_21_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___redArg(lean_object* v_f_26_, lean_object* v_a_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(v_f_26_, v_a_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily(lean_object* v_00_u03b1_29_, lean_object* v_o_30_, lean_object* v_f_31_, lean_object* v_a_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Ordinal_familyOfBFamily_x27___redArg(v_f_31_, v_a_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_familyOfBFamily___boxed(lean_object* v_00_u03b1_34_, lean_object* v_o_35_, lean_object* v_f_36_, lean_object* v_a_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Ordinal_familyOfBFamily(v_00_u03b1_34_, v_o_35_, v_f_36_, v_a_37_);
lean_dec(v_o_35_);
return v_res_38_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Arithmetic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Family(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Arithmetic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Ordinal_Family(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Arithmetic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Family(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Ordinal_Arithmetic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Family(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Ordinal_Family(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Ordinal_Family(builtin);
}
#ifdef __cplusplus
}
#endif
