// Lean compiler output
// Module: Mathlib.Algebra.Ring.Associator
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Basic public import Mathlib.Algebra.Ring.Opposite public import Mathlib.Tactic.Abel
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
lean_object* lp_mathlib_AddMonoidHom_mulLeft(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_mulLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_compr_u2082___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associator___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft_u2083___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft_u2083___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft_u2083(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight_u2083___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight_u2083___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight_u2083(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_associator___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_associator___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_associator(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associator___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_y_3_, lean_object* v_z_4_){
_start:
{
lean_object* v_toAddCommGroup_5_; lean_object* v_toSub_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v_toMul_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v_toAddCommGroup_5_ = lean_ctor_get(v_inst_1_, 0);
v_toSub_6_ = lean_ctor_get(v_toAddCommGroup_5_, 2);
lean_inc(v_toSub_6_);
v___x_7_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_1_);
v___x_8_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v___x_7_);
v_toMul_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc_n(v_toMul_9_, 4);
lean_dec_ref(v___x_8_);
lean_inc(v_y_3_);
lean_inc(v_x_2_);
v___x_10_ = lean_apply_2(v_toMul_9_, v_x_2_, v_y_3_);
lean_inc(v_z_4_);
v___x_11_ = lean_apply_2(v_toMul_9_, v___x_10_, v_z_4_);
v___x_12_ = lean_apply_2(v_toMul_9_, v_y_3_, v_z_4_);
v___x_13_ = lean_apply_2(v_toMul_9_, v_x_2_, v___x_12_);
v___x_14_ = lean_apply_2(v_toSub_6_, v___x_11_, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associator(lean_object* v_R_15_, lean_object* v_inst_16_, lean_object* v_x_17_, lean_object* v_y_18_, lean_object* v_z_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_associator___redArg(v_inst_16_, v_x_17_, v_y_18_, v_z_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft_u2083___redArg___lam__0(lean_object* v_inst_21_, lean_object* v_x_22_, lean_object* v___y_23_, lean_object* v___y_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_40__overap_27_; lean_object* v___x_28_; 
lean_inc_ref(v_inst_21_);
v___x_25_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft), 3, 2);
lean_closure_set(v___x_25_, 0, lean_box(0));
lean_closure_set(v___x_25_, 1, v_inst_21_);
v___x_26_ = lp_mathlib_AddMonoidHom_mulLeft___redArg(v_inst_21_, v_x_22_);
v___x_40__overap_27_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_26_, v___x_25_, v___y_23_);
v___x_28_ = lean_apply_1(v___x_40__overap_27_, v___y_24_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft_u2083___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; 
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft_u2083___redArg___lam__0), 4, 1);
lean_closure_set(v___f_30_, 0, v_inst_29_);
return v___f_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulLeft_u2083(lean_object* v_R_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft_u2083___redArg___lam__0), 4, 1);
lean_closure_set(v___f_33_, 0, v_inst_32_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight_u2083___redArg___lam__0(lean_object* v_inst_34_, lean_object* v_x_35_, lean_object* v___y_36_, lean_object* v___y_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_34__overap_40_; lean_object* v___x_41_; 
lean_inc_ref(v_inst_34_);
v___x_38_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulLeft), 3, 2);
lean_closure_set(v___x_38_, 0, lean_box(0));
lean_closure_set(v___x_38_, 1, v_inst_34_);
v___x_39_ = lp_mathlib_AddMonoidHom_mulLeft___redArg(v_inst_34_, v_x_35_);
v___x_34__overap_40_ = lp_mathlib_AddMonoidHom_compr_u2082___redArg(v___x_38_, v___x_39_);
v___x_41_ = lean_apply_2(v___x_34__overap_40_, v___y_36_, v___y_37_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight_u2083___redArg(lean_object* v_inst_42_){
_start:
{
lean_object* v___f_43_; 
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulRight_u2083___redArg___lam__0), 4, 1);
lean_closure_set(v___f_43_, 0, v_inst_42_);
return v___f_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulRight_u2083(lean_object* v_R_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___f_46_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mulRight_u2083___redArg___lam__0), 4, 1);
lean_closure_set(v___f_46_, 0, v_inst_45_);
return v___f_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_associator___redArg___lam__0(lean_object* v_toAddCommGroup_47_, lean_object* v___x_48_, lean_object* v_x_49_, lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
lean_object* v_toSub_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v_toSub_52_ = lean_ctor_get(v_toAddCommGroup_47_, 2);
lean_inc(v_toSub_52_);
lean_dec_ref(v_toAddCommGroup_47_);
lean_inc(v___y_51_);
lean_inc(v___y_50_);
lean_inc(v_x_49_);
lean_inc_ref(v___x_48_);
v___x_53_ = lp_mathlib_AddMonoidHom_mulLeft_u2083___redArg___lam__0(v___x_48_, v_x_49_, v___y_50_, v___y_51_);
v___x_54_ = lp_mathlib_AddMonoidHom_mulRight_u2083___redArg___lam__0(v___x_48_, v_x_49_, v___y_50_, v___y_51_);
v___x_55_ = lean_apply_2(v_toSub_52_, v___x_53_, v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_associator___redArg(lean_object* v_inst_56_){
_start:
{
lean_object* v_toAddCommGroup_57_; lean_object* v___x_58_; lean_object* v___f_59_; 
v_toAddCommGroup_57_ = lean_ctor_get(v_inst_56_, 0);
lean_inc_ref(v_toAddCommGroup_57_);
v___x_58_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_56_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_associator___redArg___lam__0), 5, 2);
lean_closure_set(v___f_59_, 0, v_toAddCommGroup_57_);
lean_closure_set(v___f_59_, 1, v___x_58_);
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_associator(lean_object* v_R_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_AddMonoidHom_associator___redArg(v_inst_61_);
return v___x_62_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Associator(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Associator(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Associator(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Associator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Associator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Associator(builtin);
}
#ifdef __cplusplus
}
#endif
