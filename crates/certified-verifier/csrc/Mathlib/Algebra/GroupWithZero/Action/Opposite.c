// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Faithful public import Mathlib.Algebra.Group.Action.Opposite public import Mathlib.Algebra.GroupWithZero.Action.Defs public import Mathlib.Algebra.GroupWithZero.NeZero
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
lean_object* lp_mathlib_MulOpposite_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulZeroClass___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___f_2_; 
v___f_2_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2_, 0, v_inst_1_);
return v___f_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulZeroClass(lean_object* v_M_3_, lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulZeroClass___boxed(lean_object* v_M_8_, lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_MulOpposite_instSMulZeroClass(v_M_8_, v_00_u03b1_9_, v_inst_10_, v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulWithZero___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_14_, 0, v_inst_13_);
return v___f_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulWithZero(lean_object* v_M_15_, lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSMulWithZero___boxed(lean_object* v_M_21_, lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_MulOpposite_instSMulWithZero(v_M_21_, v_00_u03b1_22_, v_inst_23_, v_inst_24_, v_inst_25_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_23_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulActionWithZero___redArg(lean_object* v_inst_27_){
_start:
{
lean_object* v___f_28_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_28_, 0, v_inst_27_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulActionWithZero(lean_object* v_M_29_, lean_object* v_00_u03b1_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___f_34_; 
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_34_, 0, v_inst_33_);
return v___f_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulActionWithZero___boxed(lean_object* v_M_35_, lean_object* v_00_u03b1_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_MulOpposite_instMulActionWithZero(v_M_35_, v_00_u03b1_36_, v_inst_37_, v_inst_38_, v_inst_39_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistribMulAction___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistribMulAction(lean_object* v_M_43_, lean_object* v_00_u03b1_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___f_48_; 
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_48_, 0, v_inst_47_);
return v___f_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDistribMulAction___boxed(lean_object* v_M_49_, lean_object* v_00_u03b1_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_MulOpposite_instDistribMulAction(v_M_49_, v_00_u03b1_50_, v_inst_51_, v_inst_52_, v_inst_53_);
lean_dec_ref(v_inst_52_);
lean_dec_ref(v_inst_51_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulDistribMulAction___redArg(lean_object* v_inst_55_){
_start:
{
lean_object* v___f_56_; 
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_56_, 0, v_inst_55_);
return v___f_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulDistribMulAction(lean_object* v_M_57_, lean_object* v_00_u03b1_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___f_62_; 
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_62_, 0, v_inst_61_);
return v___f_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulDistribMulAction___boxed(lean_object* v_M_63_, lean_object* v_00_u03b1_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_MulOpposite_instMulDistribMulAction(v_M_63_, v_00_u03b1_64_, v_inst_65_, v_inst_66_, v_inst_67_);
lean_dec_ref(v_inst_66_);
lean_dec_ref(v_inst_65_);
return v_res_68_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
