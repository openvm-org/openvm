// Lean compiler output
// Module: Mathlib.Algebra.Ring.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Commute.Defs public import Mathlib.Algebra.Group.Hom.Instances public import Mathlib.Algebra.Group.SelfInv public import Mathlib.Algebra.GroupWithZero.NeZero public import Mathlib.Algebra.Opposites public import Mathlib.Algebra.Ring.Defs public import Mathlib.Tactic.TFAE
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instNeg___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_flip___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulLeft(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulRight(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instHasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instHasDistribNeg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instHasDistribNeg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulLeft___redArg___lam__0(lean_object* v_toMul_1_, lean_object* v_r_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toMul_1_, v_r_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulLeft___redArg(lean_object* v_inst_5_, lean_object* v_r_6_){
_start:
{
lean_object* v_toMul_7_; lean_object* v___f_8_; 
v_toMul_7_ = lean_ctor_get(v_inst_5_, 0);
lean_inc(v_toMul_7_);
lean_dec_ref(v_inst_5_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_AddHom_mulLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_8_, 0, v_toMul_7_);
lean_closure_set(v___f_8_, 1, v_r_6_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulLeft(lean_object* v_R_9_, lean_object* v_inst_10_, lean_object* v_r_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_AddHom_mulLeft___redArg(v_inst_10_, v_r_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulRight___redArg___lam__0(lean_object* v_toMul_13_, lean_object* v_r_14_, lean_object* v_a_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_apply_2(v_toMul_13_, v_a_15_, v_r_14_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulRight___redArg(lean_object* v_inst_17_, lean_object* v_r_18_){
_start:
{
lean_object* v_toMul_19_; lean_object* v___f_20_; 
v_toMul_19_ = lean_ctor_get(v_inst_17_, 0);
lean_inc(v_toMul_19_);
lean_dec_ref(v_inst_17_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_AddHom_mulRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_20_, 0, v_toMul_19_);
lean_closure_set(v___f_20_, 1, v_r_18_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulRight(lean_object* v_R_21_, lean_object* v_inst_22_, lean_object* v_r_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_AddHom_mulRight___redArg(v_inst_22_, v_r_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft___redArg(lean_object* v_inst_25_, lean_object* v_r_26_){
_start:
{
lean_object* v___x_27_; lean_object* v_toMul_28_; lean_object* v___f_29_; 
v___x_27_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_25_);
v_toMul_28_ = lean_ctor_get(v___x_27_, 0);
lean_inc(v_toMul_28_);
lean_dec_ref(v___x_27_);
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_AddHom_mulLeft___redArg___lam__0), 3, 2);
lean_closure_set(v___f_29_, 0, v_toMul_28_);
lean_closure_set(v___f_29_, 1, v_r_26_);
return v___f_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft(lean_object* v_R_30_, lean_object* v_inst_31_, lean_object* v_r_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_AddMonoidHom_mulLeft___redArg(v_inst_31_, v_r_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight___redArg(lean_object* v_inst_34_, lean_object* v_r_35_){
_start:
{
lean_object* v___x_36_; lean_object* v_toMul_37_; lean_object* v___f_38_; 
v___x_36_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_34_);
v_toMul_37_ = lean_ctor_get(v___x_36_, 0);
lean_inc(v_toMul_37_);
lean_dec_ref(v___x_36_);
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_AddHom_mulRight___redArg___lam__0), 3, 2);
lean_closure_set(v___f_38_, 0, v_toMul_37_);
lean_closure_set(v___f_38_, 1, v_r_35_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight(lean_object* v_R_39_, lean_object* v_inst_40_, lean_object* v_r_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_AddMonoidHom_mulRight___redArg(v_inst_40_, v_r_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mul___redArg(lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft), 3, 2);
lean_closure_set(v___x_44_, 0, lean_box(0));
lean_closure_set(v___x_44_, 1, v_inst_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mul(lean_object* v_R_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft), 3, 2);
lean_closure_set(v___x_47_, 0, lean_box(0));
lean_closure_set(v___x_47_, 1, v_inst_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulLeft___redArg(lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft), 3, 2);
lean_closure_set(v___x_49_, 0, lean_box(0));
lean_closure_set(v___x_49_, 1, v_inst_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulLeft(lean_object* v_R_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft), 3, 2);
lean_closure_set(v___x_52_, 0, lean_box(0));
lean_closure_set(v___x_52_, 1, v_inst_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulRight___redArg(lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; lean_object* v___f_55_; 
v___x_54_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft), 3, 2);
lean_closure_set(v___x_54_, 0, lean_box(0));
lean_closure_set(v___x_54_, 1, v_inst_53_);
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_55_, 0, v___x_54_);
return v___f_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_mulRight(lean_object* v_R_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_AddMonoid_End_mulRight___redArg(v_inst_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instHasDistribNeg___redArg(lean_object* v_inst_59_){
_start:
{
lean_object* v___f_60_; 
v___f_60_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_60_, 0, v_inst_59_);
return v___f_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instHasDistribNeg(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___f_64_; 
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_64_, 0, v_inst_63_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instHasDistribNeg___boxed(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_MulOpposite_instHasDistribNeg(v_00_u03b1_65_, v_inst_66_, v_inst_67_);
lean_dec(v_inst_66_);
return v_res_68_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TFAE(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TFAE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TFAE(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TFAE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
