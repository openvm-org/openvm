// Lean compiler output
// Module: Mathlib.Control.Monad.Basic
// Imports: public import Init public meta import Init public import Mathlib.Logic.Equiv.Defs public import Batteries.Lean.Except
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
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_StateT_eval___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_StateT_eval___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_StateT_eval___redArg___closed__0 = (const lean_object*)&lp_mathlib_StateT_eval___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval___redArg___lam__0(lean_object* v_self_1_){
_start:
{
lean_object* v_fst_2_; 
v_fst_2_ = lean_ctor_get(v_self_1_, 0);
lean_inc(v_fst_2_);
return v_fst_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval___redArg___lam__0___boxed(lean_object* v_self_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_StateT_eval___redArg___lam__0(v_self_3_);
lean_dec_ref(v_self_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval___redArg(lean_object* v_inst_6_, lean_object* v_cmd_7_, lean_object* v_s_8_){
_start:
{
lean_object* v_map_9_; lean_object* v___f_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v_map_9_ = lean_ctor_get(v_inst_6_, 0);
lean_inc(v_map_9_);
lean_dec_ref(v_inst_6_);
v___f_10_ = ((lean_object*)(lp_mathlib_StateT_eval___redArg___closed__0));
v___x_11_ = lean_apply_1(v_cmd_7_, v_s_8_);
v___x_12_ = lean_apply_4(v_map_9_, lean_box(0), lean_box(0), v___f_10_, v___x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StateT_eval(lean_object* v_00_u03b1_13_, lean_object* v_00_u03c3_14_, lean_object* v_m_15_, lean_object* v_inst_16_, lean_object* v_cmd_17_, lean_object* v_s_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_StateT_eval___redArg(v_inst_16_, v_cmd_17_, v_s_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv___redArg(lean_object* v_F_20_){
_start:
{
lean_inc_ref(v_F_20_);
return v_F_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv___redArg___boxed(lean_object* v_F_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_StateT_equiv___redArg(v_F_21_);
lean_dec_ref(v_F_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv(lean_object* v_00_u03c3_u2081_23_, lean_object* v_00_u03b1_u2081_24_, lean_object* v_00_u03c3_u2082_25_, lean_object* v_00_u03b1_u2082_26_, lean_object* v_m_u2081_27_, lean_object* v_m_u2082_28_, lean_object* v_F_29_){
_start:
{
lean_inc_ref(v_F_29_);
return v_F_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StateT_equiv___boxed(lean_object* v_00_u03c3_u2081_30_, lean_object* v_00_u03b1_u2081_31_, lean_object* v_00_u03c3_u2082_32_, lean_object* v_00_u03b1_u2082_33_, lean_object* v_m_u2081_34_, lean_object* v_m_u2082_35_, lean_object* v_F_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_StateT_equiv(v_00_u03c3_u2081_30_, v_00_u03b1_u2081_31_, v_00_u03c3_u2082_32_, v_00_u03b1_u2082_33_, v_m_u2081_34_, v_m_u2082_35_, v_F_36_);
lean_dec_ref(v_F_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv___redArg(lean_object* v_F_38_){
_start:
{
lean_inc_ref(v_F_38_);
return v_F_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv___redArg___boxed(lean_object* v_F_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_ReaderT_equiv___redArg(v_F_39_);
lean_dec_ref(v_F_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv(lean_object* v_00_u03c1_u2081_41_, lean_object* v_00_u03b1_u2081_42_, lean_object* v_00_u03c1_u2082_43_, lean_object* v_00_u03b1_u2082_44_, lean_object* v_m_u2081_45_, lean_object* v_m_u2082_46_, lean_object* v_F_47_){
_start:
{
lean_inc_ref(v_F_47_);
return v_F_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ReaderT_equiv___boxed(lean_object* v_00_u03c1_u2081_48_, lean_object* v_00_u03b1_u2081_49_, lean_object* v_00_u03c1_u2082_50_, lean_object* v_00_u03b1_u2082_51_, lean_object* v_m_u2081_52_, lean_object* v_m_u2082_53_, lean_object* v_F_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_ReaderT_equiv(v_00_u03c1_u2081_48_, v_00_u03b1_u2081_49_, v_00_u03c1_u2082_50_, v_00_u03b1_u2082_51_, v_m_u2081_52_, v_m_u2082_53_, v_F_54_);
lean_dec_ref(v_F_54_);
return v_res_55_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Except(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Control_Monad_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Except(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Control_Monad_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Except(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Monad_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Except(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Monad_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Control_Monad_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Control_Monad_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
