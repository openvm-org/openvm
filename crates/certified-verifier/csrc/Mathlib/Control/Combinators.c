// Lean compiler output
// Module: Mathlib.Control.Combinators
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_id___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_joinM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_joinM___redArg___closed__0 = (const lean_object*)&lp_mathlib_joinM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_joinM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_joinM(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_condM___redArg___lam__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_condM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_condM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_condM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_joinM___redArg(lean_object* v_inst_2_, lean_object* v_a_3_){
_start:
{
lean_object* v_toBind_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v_toBind_4_ = lean_ctor_get(v_inst_2_, 1);
lean_inc(v_toBind_4_);
lean_dec_ref(v_inst_2_);
v___x_5_ = ((lean_object*)(lp_mathlib_joinM___redArg___closed__0));
v___x_6_ = lean_apply_4(v_toBind_4_, lean_box(0), lean_box(0), v_a_3_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_joinM(lean_object* v_m_7_, lean_object* v_inst_8_, lean_object* v_00_u03b1_9_, lean_object* v_a_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_joinM___redArg(v_inst_8_, v_a_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_condM___redArg___lam__0(lean_object* v_fm_12_, lean_object* v_tm_13_, uint8_t v_b_14_){
_start:
{
if (v_b_14_ == 0)
{
lean_inc(v_fm_12_);
return v_fm_12_;
}
else
{
lean_inc(v_tm_13_);
return v_tm_13_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_condM___redArg___lam__0___boxed(lean_object* v_fm_15_, lean_object* v_tm_16_, lean_object* v_b_17_){
_start:
{
uint8_t v_b_boxed_18_; lean_object* v_res_19_; 
v_b_boxed_18_ = lean_unbox(v_b_17_);
v_res_19_ = lp_mathlib_condM___redArg___lam__0(v_fm_15_, v_tm_16_, v_b_boxed_18_);
lean_dec(v_tm_16_);
lean_dec(v_fm_15_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_condM___redArg(lean_object* v_inst_20_, lean_object* v_mbool_21_, lean_object* v_tm_22_, lean_object* v_fm_23_){
_start:
{
lean_object* v_toBind_24_; lean_object* v___f_25_; lean_object* v___x_26_; 
v_toBind_24_ = lean_ctor_get(v_inst_20_, 1);
lean_inc(v_toBind_24_);
lean_dec_ref(v_inst_20_);
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_condM___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_25_, 0, v_fm_23_);
lean_closure_set(v___f_25_, 1, v_tm_22_);
v___x_26_ = lean_apply_4(v_toBind_24_, lean_box(0), lean_box(0), v_mbool_21_, v___f_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_condM(lean_object* v_m_27_, lean_object* v_inst_28_, lean_object* v_00_u03b1_29_, lean_object* v_mbool_30_, lean_object* v_tm_31_, lean_object* v_fm_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_condM___redArg(v_inst_28_, v_mbool_30_, v_tm_31_, v_fm_32_);
return v___x_33_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Control_Combinators(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Control_Combinators(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Combinators(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Combinators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Control_Combinators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Control_Combinators(builtin);
}
#ifdef __cplusplus
}
#endif
