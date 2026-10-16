// Lean compiler output
// Module: Aesop.BuiltinRules
// Imports: public import Init public meta import Init public import Aesop.BuiltinRules.Assumption public import Aesop.BuiltinRules.ApplyHyps public import Aesop.BuiltinRules.DestructProducts public import Aesop.BuiltinRules.Ext public import Aesop.BuiltinRules.Intros public import Aesop.BuiltinRules.Rfl public import Aesop.BuiltinRules.Split public import Aesop.BuiltinRules.Subst
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
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Assumption(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_ApplyHyps(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_DestructProducts(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Ext(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Intros(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Rfl(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Split(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Subst(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_BuiltinRules(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Assumption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_ApplyHyps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_DestructProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Intros(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Split(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Subst(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_BuiltinRules(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_BuiltinRules_Assumption(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules_ApplyHyps(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules_DestructProducts(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules_Ext(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules_Intros(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules_Rfl(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules_Split(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules_Subst(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BuiltinRules(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_Assumption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_ApplyHyps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_DestructProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_Intros(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_Split(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules_Subst(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_BuiltinRules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_BuiltinRules(builtin);
}
#ifdef __cplusplus
}
#endif
