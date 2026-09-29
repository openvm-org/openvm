// Lean compiler output
// Module: Mathlib.Algebra.Group.Units.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.Opposite public import Mathlib.Algebra.Group.Units.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Units_opEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Units_opEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Units_opEquiv___closed__0 = (const lean_object*)&lp_mathlib_Units_opEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Units_opEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Units_opEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Units_opEquiv___closed__1 = (const lean_object*)&lp_mathlib_Units_opEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Units_opEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Units_opEquiv___closed__0_value),((lean_object*)&lp_mathlib_Units_opEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Units_opEquiv___closed__2 = (const lean_object*)&lp_mathlib_Units_opEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_AddUnits_opEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddUnits_opEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddUnits_opEquiv___closed__0 = (const lean_object*)&lp_mathlib_AddUnits_opEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_AddUnits_opEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddUnits_opEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddUnits_opEquiv___closed__1 = (const lean_object*)&lp_mathlib_AddUnits_opEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_AddUnits_opEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddUnits_opEquiv___closed__0_value),((lean_object*)&lp_mathlib_AddUnits_opEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_AddUnits_opEquiv___closed__2 = (const lean_object*)&lp_mathlib_AddUnits_opEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv___lam__0(lean_object* v_u_1_){
_start:
{
lean_object* v_val_2_; lean_object* v_inv_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_10_; 
v_val_2_ = lean_ctor_get(v_u_1_, 0);
v_inv_3_ = lean_ctor_get(v_u_1_, 1);
v_isSharedCheck_10_ = !lean_is_exclusive(v_u_1_);
if (v_isSharedCheck_10_ == 0)
{
v___x_5_ = v_u_1_;
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_inv_3_);
lean_inc(v_val_2_);
lean_dec(v_u_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_8_; 
if (v_isShared_6_ == 0)
{
v___x_8_ = v___x_5_;
goto v_reusejp_7_;
}
else
{
lean_object* v_reuseFailAlloc_9_; 
v_reuseFailAlloc_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_9_, 0, v_val_2_);
lean_ctor_set(v_reuseFailAlloc_9_, 1, v_inv_3_);
v___x_8_ = v_reuseFailAlloc_9_;
goto v_reusejp_7_;
}
v_reusejp_7_:
{
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv___lam__1(lean_object* v_X_11_){
_start:
{
lean_object* v_val_12_; lean_object* v_inv_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_20_; 
v_val_12_ = lean_ctor_get(v_X_11_, 0);
v_inv_13_ = lean_ctor_get(v_X_11_, 1);
v_isSharedCheck_20_ = !lean_is_exclusive(v_X_11_);
if (v_isSharedCheck_20_ == 0)
{
v___x_15_ = v_X_11_;
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_inv_13_);
lean_inc(v_val_12_);
lean_dec(v_X_11_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_18_; 
if (v_isShared_16_ == 0)
{
v___x_18_ = v___x_15_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_19_; 
v_reuseFailAlloc_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_19_, 0, v_val_12_);
lean_ctor_set(v_reuseFailAlloc_19_, 1, v_inv_13_);
v___x_18_ = v_reuseFailAlloc_19_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
return v___x_18_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv(lean_object* v_M_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = ((lean_object*)(lp_mathlib_Units_opEquiv___closed__2));
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_opEquiv___boxed(lean_object* v_M_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Units_opEquiv(v_M_29_, v_inst_30_);
lean_dec_ref(v_inst_30_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv___lam__0(lean_object* v_u_32_){
_start:
{
lean_object* v_val_33_; lean_object* v_neg_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_41_; 
v_val_33_ = lean_ctor_get(v_u_32_, 0);
v_neg_34_ = lean_ctor_get(v_u_32_, 1);
v_isSharedCheck_41_ = !lean_is_exclusive(v_u_32_);
if (v_isSharedCheck_41_ == 0)
{
v___x_36_ = v_u_32_;
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_neg_34_);
lean_inc(v_val_33_);
lean_dec(v_u_32_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_39_; 
if (v_isShared_37_ == 0)
{
v___x_39_ = v___x_36_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v_val_33_);
lean_ctor_set(v_reuseFailAlloc_40_, 1, v_neg_34_);
v___x_39_ = v_reuseFailAlloc_40_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
return v___x_39_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv___lam__1(lean_object* v_X_42_){
_start:
{
lean_object* v_val_43_; lean_object* v_neg_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_51_; 
v_val_43_ = lean_ctor_get(v_X_42_, 0);
v_neg_44_ = lean_ctor_get(v_X_42_, 1);
v_isSharedCheck_51_ = !lean_is_exclusive(v_X_42_);
if (v_isSharedCheck_51_ == 0)
{
v___x_46_ = v_X_42_;
v_isShared_47_ = v_isSharedCheck_51_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_neg_44_);
lean_inc(v_val_43_);
lean_dec(v_X_42_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_51_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___x_49_; 
if (v_isShared_47_ == 0)
{
v___x_49_ = v___x_46_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v_val_43_);
lean_ctor_set(v_reuseFailAlloc_50_, 1, v_neg_44_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv(lean_object* v_M_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = ((lean_object*)(lp_mathlib_AddUnits_opEquiv___closed__2));
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_opEquiv___boxed(lean_object* v_M_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_AddUnits_opEquiv(v_M_60_, v_inst_61_);
lean_dec_ref(v_inst_61_);
return v_res_62_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
