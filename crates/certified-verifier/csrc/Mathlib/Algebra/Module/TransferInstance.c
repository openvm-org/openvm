// Lean compiler output
// Module: Mathlib.Algebra.Module.TransferInstance
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Action.TransferInstance public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Algebra.Module.Torsion.Free public import Mathlib.Algebra.NoZeroSMulDivisors.Defs
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
lean_object* lp_mathlib_Equiv_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_module___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_linearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_linearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_linearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_module___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_module___redArg(lean_object* v_inst_1_, lean_object* v_e_2_){
_start:
{
lean_object* v___f_3_; 
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_3_, 0, v_e_2_);
lean_closure_set(v___f_3_, 1, v_inst_1_);
return v___f_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_module(lean_object* v_R_4_, lean_object* v_00_u03b1_5_, lean_object* v_00_u03b2_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_e_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_12_, 0, v_e_11_);
lean_closure_set(v___f_12_, 1, v_inst_10_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_module___boxed(lean_object* v_R_13_, lean_object* v_00_u03b1_14_, lean_object* v_00_u03b2_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_e_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_AddEquiv_module(v_R_13_, v_00_u03b1_14_, v_00_u03b2_15_, v_inst_16_, v_inst_17_, v_inst_18_, v_inst_19_, v_e_20_);
lean_dec_ref(v_inst_18_);
lean_dec_ref(v_inst_17_);
lean_dec_ref(v_inst_16_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_linearEquiv___redArg(lean_object* v_e_22_){
_start:
{
lean_object* v_toFun_23_; lean_object* v_invFun_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_31_; 
v_toFun_23_ = lean_ctor_get(v_e_22_, 0);
v_invFun_24_ = lean_ctor_get(v_e_22_, 1);
v_isSharedCheck_31_ = !lean_is_exclusive(v_e_22_);
if (v_isSharedCheck_31_ == 0)
{
v___x_26_ = v_e_22_;
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_invFun_24_);
lean_inc(v_toFun_23_);
lean_dec(v_e_22_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___x_29_; 
if (v_isShared_27_ == 0)
{
v___x_29_ = v___x_26_;
goto v_reusejp_28_;
}
else
{
lean_object* v_reuseFailAlloc_30_; 
v_reuseFailAlloc_30_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_30_, 0, v_toFun_23_);
lean_ctor_set(v_reuseFailAlloc_30_, 1, v_invFun_24_);
v___x_29_ = v_reuseFailAlloc_30_;
goto v_reusejp_28_;
}
v_reusejp_28_:
{
return v___x_29_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_linearEquiv(lean_object* v_R_32_, lean_object* v_00_u03b1_33_, lean_object* v_00_u03b2_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_e_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_AddEquiv_linearEquiv___redArg(v_e_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_linearEquiv___boxed(lean_object* v_R_41_, lean_object* v_00_u03b1_42_, lean_object* v_00_u03b2_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_e_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_AddEquiv_linearEquiv(v_R_41_, v_00_u03b1_42_, v_00_u03b2_43_, v_inst_44_, v_inst_45_, v_inst_46_, v_inst_47_, v_e_48_);
lean_dec(v_inst_47_);
lean_dec_ref(v_inst_46_);
lean_dec_ref(v_inst_45_);
lean_dec_ref(v_inst_44_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_module___redArg(lean_object* v_inst_50_, lean_object* v_e_51_){
_start:
{
lean_object* v___f_52_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_52_, 0, v_e_51_);
lean_closure_set(v___f_52_, 1, v_inst_50_);
return v___f_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_module(lean_object* v_R_53_, lean_object* v_00_u03b1_54_, lean_object* v_00_u03b2_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_e_60_){
_start:
{
lean_object* v___f_61_; 
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_smul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_61_, 0, v_e_60_);
lean_closure_set(v___f_61_, 1, v_inst_59_);
return v___f_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_module___boxed(lean_object* v_R_62_, lean_object* v_00_u03b1_63_, lean_object* v_00_u03b2_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_e_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_Equiv_module(v_R_62_, v_00_u03b1_63_, v_00_u03b2_64_, v_inst_65_, v_inst_66_, v_inst_67_, v_inst_68_, v_e_69_);
lean_dec_ref(v_inst_67_);
lean_dec_ref(v_inst_66_);
lean_dec_ref(v_inst_65_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearEquiv___redArg(lean_object* v_e_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_AddEquiv_linearEquiv___redArg(v_e_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearEquiv(lean_object* v_R_73_, lean_object* v_00_u03b1_74_, lean_object* v_00_u03b2_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_e_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_AddEquiv_linearEquiv___redArg(v_e_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_linearEquiv___boxed(lean_object* v_R_82_, lean_object* v_00_u03b1_83_, lean_object* v_00_u03b2_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_e_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_Equiv_linearEquiv(v_R_82_, v_00_u03b1_83_, v_00_u03b2_84_, v_inst_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_e_89_);
lean_dec(v_inst_88_);
lean_dec_ref(v_inst_87_);
lean_dec_ref(v_inst_86_);
lean_dec_ref(v_inst_85_);
return v_res_90_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_NoZeroSMulDivisors_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_NoZeroSMulDivisors_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_TransferInstance(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_NoZeroSMulDivisors_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_TransferInstance(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Torsion_Free(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NoZeroSMulDivisors_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_TransferInstance(builtin);
}
#ifdef __cplusplus
}
#endif
