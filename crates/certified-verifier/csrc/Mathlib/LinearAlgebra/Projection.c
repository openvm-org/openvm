// Lean compiler output
// Module: Mathlib.LinearAlgebra.Projection
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.Quotient.Basic public import Mathlib.LinearAlgebra.Prod public import Mathlib.Algebra.Module.Submodule.Invariant public import Mathlib.LinearAlgebra.GeneralLinearGroup.Basic public import Mathlib.Algebra.Ring.Idempotent
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
lean_object* lp_mathlib_LinearMap_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SMulMemClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0 = (const lean_object*)&lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Submodule_isIdempotentElemEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___closed__1 = (const lean_object*)&lp_mathlib_Submodule_isIdempotentElemEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Submodule_isIdempotentElemEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0_value),((lean_object*)&lp_mathlib_Submodule_isIdempotentElemEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___closed__2 = (const lean_object*)&lp_mathlib_Submodule_isIdempotentElemEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0(lean_object* v_f_2_, lean_object* v___y_3_){
_start:
{
lean_object* v___f_4_; lean_object* v___x_5_; 
v___f_4_ = ((lean_object*)(lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0));
v___x_5_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v_f_2_, v___f_4_, v___y_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv(lean_object* v_R_11_, lean_object* v_inst_12_, lean_object* v_E_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_p_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = ((lean_object*)(lp_mathlib_Submodule_isIdempotentElemEquiv___closed__2));
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___boxed(lean_object* v_R_18_, lean_object* v_inst_19_, lean_object* v_E_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_p_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Submodule_isIdempotentElemEquiv(v_R_18_, v_inst_19_, v_E_20_, v_inst_21_, v_inst_22_, v_p_23_);
lean_dec(v_inst_22_);
lean_dec_ref(v_inst_21_);
lean_dec_ref(v_inst_19_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___redArg(lean_object* v_f_25_){
_start:
{
lean_object* v___f_26_; 
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_26_, 0, v_f_25_);
return v___f_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict(lean_object* v_S_27_, lean_object* v_inst_28_, lean_object* v_M_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_m_32_, lean_object* v_f_33_, lean_object* v_h_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_35_, 0, v_f_33_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___boxed(lean_object* v_S_36_, lean_object* v_inst_37_, lean_object* v_M_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_m_41_, lean_object* v_f_42_, lean_object* v_h_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_LinearMap_IsProj_codRestrict(v_S_36_, v_inst_37_, v_M_38_, v_inst_39_, v_inst_40_, v_m_41_, v_f_42_, v_h_43_);
lean_dec(v_inst_40_);
lean_dec_ref(v_inst_39_);
lean_dec_ref(v_inst_37_);
return v_res_44_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Idempotent(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Projection(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Projection(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Idempotent(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Projection(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Projection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Projection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Projection(builtin);
}
#ifdef __cplusplus
}
#endif
