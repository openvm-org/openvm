// Lean compiler output
// Module: Swirl.Protocol.Noninteractive.Ops
// Imports: public import Init public import Swirl.Protocol.Noninteractive.Core public import Fundamentals.FieldOps
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Swirl_Protocol_Noninteractive_Core(uint8_t builtin);
lean_object* initialize_Fundamentals_FieldOps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Swirl_Protocol_Noninteractive_Ops(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Swirl_Protocol_Noninteractive_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Fundamentals_FieldOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
