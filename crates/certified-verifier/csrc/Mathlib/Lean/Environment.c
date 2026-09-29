// Lean compiler output
// Module: Mathlib.Lean.Environment
// Imports: public import Init public meta import Init public import Lean.Environment import Mathlib.Tactic.Linter.Header
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
lean_object* l_Lean_Environment_findAsync_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_AsyncConstantInfo_toConstantVal(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValWithKind_x3f(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValWithKind_x3f___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValOfKind_x3f(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValOfKind_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findTheoremConstVal_x3f(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValWithKind_x3f(lean_object* v_env_1_, lean_object* v_decl_2_, uint8_t v_skipRealize_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = l_Lean_Environment_findAsync_x3f(v_env_1_, v_decl_2_, v_skipRealize_3_);
if (lean_obj_tag(v___x_4_) == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
else
{
lean_object* v_val_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_17_; 
v_val_6_ = lean_ctor_get(v___x_4_, 0);
v_isSharedCheck_17_ = !lean_is_exclusive(v___x_4_);
if (v_isSharedCheck_17_ == 0)
{
v___x_8_ = v___x_4_;
v_isShared_9_ = v_isSharedCheck_17_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_val_6_);
lean_dec(v___x_4_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_17_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
uint8_t v_kind_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_15_; 
v_kind_10_ = lean_ctor_get_uint8(v_val_6_, sizeof(void*)*3);
v___x_11_ = l_Lean_AsyncConstantInfo_toConstantVal(v_val_6_);
v___x_12_ = lean_box(v_kind_10_);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v___x_11_);
lean_ctor_set(v___x_13_, 1, v___x_12_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 0, v___x_13_);
v___x_15_ = v___x_8_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v___x_13_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValWithKind_x3f___boxed(lean_object* v_env_18_, lean_object* v_decl_19_, lean_object* v_skipRealize_20_){
_start:
{
uint8_t v_skipRealize_boxed_21_; lean_object* v_res_22_; 
v_skipRealize_boxed_21_ = lean_unbox(v_skipRealize_20_);
v_res_22_ = lp_mathlib_Lean_Environment_findConstValWithKind_x3f(v_env_18_, v_decl_19_, v_skipRealize_boxed_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValOfKind_x3f(lean_object* v_env_23_, lean_object* v_p_24_, lean_object* v_decl_25_, uint8_t v_skipRealize_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = l_Lean_Environment_findAsync_x3f(v_env_23_, v_decl_25_, v_skipRealize_26_);
if (lean_obj_tag(v___x_27_) == 0)
{
lean_object* v___x_28_; 
lean_dec_ref(v_p_24_);
v___x_28_ = lean_box(0);
return v___x_28_;
}
else
{
lean_object* v_val_29_; lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_42_; 
v_val_29_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_42_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_42_ == 0)
{
v___x_31_ = v___x_27_;
v_isShared_32_ = v_isSharedCheck_42_;
goto v_resetjp_30_;
}
else
{
lean_inc(v_val_29_);
lean_dec(v___x_27_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_42_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
uint8_t v_kind_33_; lean_object* v___x_34_; lean_object* v___x_35_; uint8_t v___x_36_; 
v_kind_33_ = lean_ctor_get_uint8(v_val_29_, sizeof(void*)*3);
v___x_34_ = lean_box(v_kind_33_);
v___x_35_ = lean_apply_1(v_p_24_, v___x_34_);
v___x_36_ = lean_unbox(v___x_35_);
if (v___x_36_ == 0)
{
lean_object* v___x_37_; 
lean_del_object(v___x_31_);
lean_dec(v_val_29_);
v___x_37_ = lean_box(0);
return v___x_37_;
}
else
{
lean_object* v___x_38_; lean_object* v___x_40_; 
v___x_38_ = l_Lean_AsyncConstantInfo_toConstantVal(v_val_29_);
if (v_isShared_32_ == 0)
{
lean_ctor_set(v___x_31_, 0, v___x_38_);
v___x_40_ = v___x_31_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v___x_38_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findConstValOfKind_x3f___boxed(lean_object* v_env_43_, lean_object* v_p_44_, lean_object* v_decl_45_, lean_object* v_skipRealize_46_){
_start:
{
uint8_t v_skipRealize_boxed_47_; lean_object* v_res_48_; 
v_skipRealize_boxed_47_ = lean_unbox(v_skipRealize_46_);
v_res_48_ = lp_mathlib_Lean_Environment_findConstValOfKind_x3f(v_env_43_, v_p_44_, v_decl_45_, v_skipRealize_boxed_47_);
return v_res_48_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___lam__0(uint8_t v_x_49_){
_start:
{
if (v_x_49_ == 1)
{
uint8_t v___x_50_; 
v___x_50_ = 1;
return v___x_50_;
}
else
{
uint8_t v___x_51_; 
v___x_51_ = 0;
return v___x_51_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___lam__0___boxed(lean_object* v_x_52_){
_start:
{
uint8_t v_x_26__boxed_53_; uint8_t v_res_54_; lean_object* v_r_55_; 
v_x_26__boxed_53_ = lean_unbox(v_x_52_);
v_res_54_ = lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___lam__0(v_x_26__boxed_53_);
v_r_55_ = lean_box(v_res_54_);
return v_r_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findTheoremConstVal_x3f(lean_object* v_env_57_, lean_object* v_decl_58_, uint8_t v_skipRealize_59_){
_start:
{
lean_object* v___f_60_; lean_object* v___x_61_; 
v___f_60_ = ((lean_object*)(lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___closed__0));
v___x_61_ = lp_mathlib_Lean_Environment_findConstValOfKind_x3f(v_env_57_, v___f_60_, v_decl_58_, v_skipRealize_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Environment_findTheoremConstVal_x3f___boxed(lean_object* v_env_62_, lean_object* v_decl_63_, lean_object* v_skipRealize_64_){
_start:
{
uint8_t v_skipRealize_boxed_65_; lean_object* v_res_66_; 
v_skipRealize_boxed_65_ = lean_unbox(v_skipRealize_64_);
v_res_66_ = lp_mathlib_Lean_Environment_findTheoremConstVal_x3f(v_env_62_, v_decl_63_, v_skipRealize_boxed_65_);
return v_res_66_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Environment(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Environment(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Environment(uint8_t builtin) {
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
lean_object* initialize_Lean_Environment(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Environment(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Environment(builtin);
}
#ifdef __cplusplus
}
#endif
