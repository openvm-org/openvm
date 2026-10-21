// Lean compiler output
// Module: Mathlib.Data.Set.BooleanAlgebra
// Imports: public import Init public meta import Init public import Mathlib.Order.CompleteBooleanAlgebra
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
lean_object* lp_mathlib_Set_instBooleanAlgebra(lean_object*);
static lean_once_cell_t lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0;
static const lean_ctor_object lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1 = (const lean_object*)&lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instOrderTop(lean_object*);
static lean_object* _init_lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Set_instBooleanAlgebra(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra(lean_object* v_00_u03b1_3_){
_start:
{
lean_object* v___x_4_; lean_object* v_toDistribLattice_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_4_ = lean_obj_once(&lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0, &lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0_once, _init_lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0);
v_toDistribLattice_5_ = lean_ctor_get(v___x_4_, 0);
v___x_6_ = ((lean_object*)(lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1));
lean_inc_ref(v_toDistribLattice_5_);
v___x_7_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_7_, 0, v_toDistribLattice_5_);
lean_ctor_set(v___x_7_, 1, lean_box(0));
lean_ctor_set(v___x_7_, 2, lean_box(0));
lean_ctor_set(v___x_7_, 3, v___x_6_);
v___x_8_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, lean_box(0));
lean_ctor_set(v___x_8_, 2, lean_box(0));
lean_ctor_set(v___x_8_, 3, lean_box(0));
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instOrderTop(lean_object* v_00_u03b1_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_box(0);
return v___x_10_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(builtin);
}
#ifdef __cplusplus
}
#endif
