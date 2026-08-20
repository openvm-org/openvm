// Lean compiler output
// Module: Swirl.Protocol.Noninteractive.Core
// Imports: public import Init public import Swirl.Protocol.Noninteractive.Config public import Swirl.Protocol.Noninteractive.Runtime.Core
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
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit___redArg(lean_object*);
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observe___redArg(lean_object*);
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observe___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observe(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_Swirl_Protocol_Noninteractive_TranscriptEvent_observeCommit(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Swirl_Protocol_Noninteractive_Config(uint8_t builtin);
lean_object* initialize_Swirl_Protocol_Noninteractive_Runtime_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Swirl_Protocol_Noninteractive_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Swirl_Protocol_Noninteractive_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Swirl_Protocol_Noninteractive_Runtime_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
