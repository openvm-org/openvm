// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.AlgebraMap
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Pi public import Mathlib.Algebra.Algebra.Prod public import Mathlib.Algebra.Algebra.Subalgebra.Lattice public import Mathlib.Algebra.Algebra.Tower public import Mathlib.Algebra.MonoidAlgebra.Basic public import Mathlib.Algebra.Polynomial.Eval.Algebra public import Mathlib.Algebra.Polynomial.Eval.Degree public import Mathlib.Algebra.Polynomial.Monomial
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
lean_object* lp_mathlib_Polynomial_eval_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Polynomial_toFinsuppIso(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_aevalTower___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_aevalTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_aevalTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lp_mathlib_Polynomial_toFinsuppIso(lean_box(0), v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg___redArg___boxed(lean_object* v_inst_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_Polynomial_toFinsuppIsoAlg___redArg(v_inst_3_);
lean_dec_ref(v_inst_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg(lean_object* v_R_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Polynomial_toFinsuppIso(lean_box(0), v_inst_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_toFinsuppIsoAlg___boxed(lean_object* v_R_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Polynomial_toFinsuppIsoAlg(v_R_8_, v_inst_9_);
lean_dec_ref(v_inst_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom___redArg___lam__0(lean_object* v_f_11_, lean_object* v___y_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_apply_1(v_f_11_, v___y_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom___redArg(lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_, lean_object* v_b_17_){
_start:
{
lean_object* v___f_18_; lean_object* v___x_19_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_eval_u2082AlgHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_18_, 0, v_f_16_);
v___x_19_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_eval_u2082___boxed), 7, 6);
lean_closure_set(v___x_19_, 0, lean_box(0));
lean_closure_set(v___x_19_, 1, lean_box(0));
lean_closure_set(v___x_19_, 2, v_inst_14_);
lean_closure_set(v___x_19_, 3, v_inst_15_);
lean_closure_set(v___x_19_, 4, v___f_18_);
lean_closure_set(v___x_19_, 5, v_b_17_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom(lean_object* v_R_20_, lean_object* v_A_21_, lean_object* v_B_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_f_28_, lean_object* v_b_29_, lean_object* v_hf_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Polynomial_eval_u2082AlgHom___redArg(v_inst_24_, v_inst_25_, v_f_28_, v_b_29_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_eval_u2082AlgHom___boxed(lean_object* v_R_32_, lean_object* v_A_33_, lean_object* v_B_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_f_40_, lean_object* v_b_41_, lean_object* v_hf_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Polynomial_eval_u2082AlgHom(v_R_32_, v_A_33_, v_B_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_inst_39_, v_f_40_, v_b_41_, v_hf_42_);
lean_dec_ref(v_inst_39_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_35_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_aevalTower___redArg(lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_f_46_, lean_object* v_x_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Polynomial_eval_u2082AlgHom___redArg(v_inst_44_, v_inst_45_, v_f_46_, v_x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_aevalTower(lean_object* v_R_49_, lean_object* v_S_50_, lean_object* v_A_x27_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_f_57_, lean_object* v_x_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Polynomial_eval_u2082AlgHom___redArg(v_inst_52_, v_inst_53_, v_f_57_, v_x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_aevalTower___boxed(lean_object* v_R_60_, lean_object* v_S_61_, lean_object* v_A_x27_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_f_68_, lean_object* v_x_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_Polynomial_aevalTower(v_R_60_, v_S_61_, v_A_x27_62_, v_inst_63_, v_inst_64_, v_inst_65_, v_inst_66_, v_inst_67_, v_f_68_, v_x_69_);
lean_dec_ref(v_inst_67_);
lean_dec_ref(v_inst_66_);
lean_dec_ref(v_inst_65_);
return v_res_70_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Algebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Degree(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Monomial(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Monomial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Algebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Degree(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Monomial(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_Degree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Monomial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(builtin);
}
#ifdef __cplusplus
}
#endif
