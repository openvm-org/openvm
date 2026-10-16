// Lean compiler output
// Module: Batteries.Lean.Syntax
// Imports: public import Init public meta import Init public import Lean.Syntax
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
lean_object* l_Lean_Syntax_replaceM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_batteries_Lean_TSyntax_replaceM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_TSyntax_replaceM___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_TSyntax_replaceM___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_TSyntax_replaceM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___redArg___lam__0(lean_object* v_raw_1_){
_start:
{
lean_inc(v_raw_1_);
return v_raw_1_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___redArg___lam__0___boxed(lean_object* v_raw_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_batteries_Lean_TSyntax_replaceM___redArg___lam__0(v_raw_2_);
lean_dec(v_raw_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___redArg(lean_object* v_inst_5_, lean_object* v_f_6_, lean_object* v_stx_7_){
_start:
{
lean_object* v_toApplicative_8_; lean_object* v_toFunctor_9_; lean_object* v_map_10_; lean_object* v___f_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v_toApplicative_8_ = lean_ctor_get(v_inst_5_, 0);
v_toFunctor_9_ = lean_ctor_get(v_toApplicative_8_, 0);
v_map_10_ = lean_ctor_get(v_toFunctor_9_, 0);
lean_inc(v_map_10_);
v___f_11_ = ((lean_object*)(lp_batteries_Lean_TSyntax_replaceM___redArg___closed__0));
v___x_12_ = l_Lean_Syntax_replaceM___redArg(v_inst_5_, v_f_6_, v_stx_7_);
v___x_13_ = lean_apply_4(v_map_10_, lean_box(0), lean_box(0), v___f_11_, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM(lean_object* v_M_14_, lean_object* v_k_15_, lean_object* v_inst_16_, lean_object* v_f_17_, lean_object* v_stx_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_batteries_Lean_TSyntax_replaceM___redArg(v_inst_16_, v_f_17_, v_stx_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TSyntax_replaceM___boxed(lean_object* v_M_20_, lean_object* v_k_21_, lean_object* v_inst_22_, lean_object* v_f_23_, lean_object* v_stx_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_batteries_Lean_TSyntax_replaceM(v_M_20_, v_k_21_, v_inst_22_, v_f_23_, v_stx_24_);
lean_dec(v_k_21_);
return v_res_25_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Syntax(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Syntax(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Syntax(uint8_t builtin) {
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
lean_object* initialize_Lean_Syntax(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Syntax(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Syntax(builtin);
}
#ifdef __cplusplus
}
#endif
