// Lean compiler output
// Module: Mathlib.LinearAlgebra.LinearIndependent.Lemmas
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.Tuple.Reflection public import Mathlib.LinearAlgebra.Dual.Defs public import Mathlib.LinearAlgebra.Finsupp.SumProd public import Mathlib.LinearAlgebra.LinearIndependent.Basic public import Mathlib.LinearAlgebra.Pi public import Mathlib.Logic.Equiv.Fin.Rotate public import Mathlib.Tactic.FinCases public import Mathlib.Tactic.Module public import Mathlib.Tactic.ModuleNF public import Mathlib.Tactic.Abel public import Mathlib.Tactic.NormNum.Ineq import Mathlib.Algebra.Module.Torsion.Field
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
lean_object* l_Fin_cases___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_tail___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_equiv__linearIndependent___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_equiv__linearIndependent___redArg___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_equiv__linearIndependent___redArg___closed__0 = (const lean_object*)&lp_mathlib_equiv__linearIndependent___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg___lam__0(lean_object* v_n_1_, lean_object* v_s_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
lean_inc(v_s_2_);
lean_inc(v_n_1_);
v___x_3_ = lean_alloc_closure((void*)(lp_mathlib_Fin_tail___boxed), 4, 3);
lean_closure_set(v___x_3_, 0, v_n_1_);
lean_closure_set(v___x_3_, 1, lean_box(0));
lean_closure_set(v___x_3_, 2, v_s_2_);
v___x_4_ = lean_unsigned_to_nat(1u);
v___x_5_ = lean_nat_add(v_n_1_, v___x_4_);
lean_dec(v_n_1_);
v___x_6_ = lean_unsigned_to_nat(0u);
v___x_7_ = lean_nat_mod(v___x_6_, v___x_5_);
lean_dec(v___x_5_);
v___x_8_ = lean_apply_1(v_s_2_, v___x_7_);
v___x_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_9_, 0, v___x_3_);
lean_ctor_set(v___x_9_, 1, v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg___lam__1(lean_object* v_s_10_, lean_object* v___y_11_){
_start:
{
lean_object* v_fst_12_; lean_object* v_snd_13_; lean_object* v___x_14_; 
v_fst_12_ = lean_ctor_get(v_s_10_, 0);
lean_inc(v_fst_12_);
v_snd_13_ = lean_ctor_get(v_s_10_, 1);
lean_inc(v_snd_13_);
lean_dec_ref(v_s_10_);
v___x_14_ = l_Fin_cases___redArg(v_snd_13_, v_fst_12_, v___y_11_);
lean_dec(v_snd_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg___lam__1___boxed(lean_object* v_s_15_, lean_object* v___y_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_equiv__linearIndependent___redArg___lam__1(v_s_15_, v___y_16_);
lean_dec(v___y_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___redArg(lean_object* v_n_19_){
_start:
{
lean_object* v___f_20_; lean_object* v___f_21_; lean_object* v___x_22_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_equiv__linearIndependent___redArg___lam__0), 2, 1);
lean_closure_set(v___f_20_, 0, v_n_19_);
v___f_21_ = ((lean_object*)(lp_mathlib_equiv__linearIndependent___redArg___closed__0));
v___x_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_22_, 0, v___f_20_);
lean_ctor_set(v___x_22_, 1, v___f_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent(lean_object* v_K_23_, lean_object* v_V_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_n_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_equiv__linearIndependent___redArg(v_n_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_equiv__linearIndependent___boxed(lean_object* v_K_30_, lean_object* v_V_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_n_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_equiv__linearIndependent(v_K_30_, v_V_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_n_35_);
lean_dec(v_inst_34_);
lean_dec_ref(v_inst_33_);
lean_dec_ref(v_inst_32_);
return v_res_36_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ModuleNF(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ModuleNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ModuleNF(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Tuple_Reflection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Dual_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Fin_Rotate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ModuleNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
