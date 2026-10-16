// Lean compiler output
// Module: Aesop.Script.Tactic
// Imports: public import Init public meta import Init public import Lean.Meta.Basic
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
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Tactic_instToMessageData___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_Tactic_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Script_Tactic_instToMessageData___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_Tactic_instToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_Tactic_instToMessageData___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Script_Tactic_instToMessageData = (const lean_object*)&lp_aesop_Aesop_Script_Tactic_instToMessageData___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Tactic_unstructured(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Tactic_structured(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Tactic_instToMessageData___lam__0(lean_object* v_t_1_){
_start:
{
lean_object* v_uTactic_2_; lean_object* v___x_3_; 
v_uTactic_2_ = lean_ctor_get(v_t_1_, 0);
lean_inc(v_uTactic_2_);
lean_dec_ref(v_t_1_);
v___x_3_ = l_Lean_MessageData_ofSyntax(v_uTactic_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Tactic_unstructured(lean_object* v_uTactic_6_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_box(0);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v_uTactic_6_);
lean_ctor_set(v___x_8_, 1, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Tactic_structured(lean_object* v_uTactic_9_, lean_object* v_sTactic_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_11_, 0, v_sTactic_10_);
v___x_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_12_, 0, v_uTactic_9_);
lean_ctor_set(v___x_12_, 1, v___x_11_);
return v___x_12_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_Tactic(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_Tactic(builtin);
}
#ifdef __cplusplus
}
#endif
