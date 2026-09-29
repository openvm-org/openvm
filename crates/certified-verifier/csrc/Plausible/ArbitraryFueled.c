// Lean compiler output
// Module: Plausible.ArbitraryFueled
// Imports: public import Init public meta import Init public import Plausible.Arbitrary
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
lean_object* lp_plausible_Plausible_Gen_sized___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_sized___boxed), 4, 2);
lean_closure_set(v___x_2_, 0, lean_box(0));
lean_closure_set(v___x_2_, 1, v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instArbitraryOfArbitraryFueled(lean_object* v_00_u03b1_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_sized___boxed), 4, 2);
lean_closure_set(v___x_5_, 0, lean_box(0));
lean_closure_set(v___x_5_, 1, v_inst_4_);
return v___x_5_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible_Arbitrary(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_ArbitraryFueled(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Arbitrary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_ArbitraryFueled(uint8_t builtin) {
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
lean_object* initialize_plausible_Plausible_Arbitrary(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_ArbitraryFueled(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Arbitrary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_ArbitraryFueled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_ArbitraryFueled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_ArbitraryFueled(builtin);
}
#ifdef __cplusplus
}
#endif
