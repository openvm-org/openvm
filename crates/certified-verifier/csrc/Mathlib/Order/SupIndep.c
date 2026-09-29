// Lean compiler output
// Module: Mathlib.Order.SupIndep
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Lattice.Union public import Mathlib.Data.Finset.Lattice.Prod public import Mathlib.Data.Finset.Sigma public import Mathlib.Data.Fintype.Basic public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Order.CompleteLatticeIntervals public import Mathlib.Order.ModularLattice public import Mathlib.Tactic.FinCases
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
lean_object* lp_mathlib_Multiset_erase___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_sup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___lam__0(lean_object* v_f_1_, lean_object* v_inst_2_, lean_object* v_s_3_, lean_object* v_toSemilatticeSup_4_, lean_object* v_inst_5_, lean_object* v_inf_6_, lean_object* v_inst_7_, lean_object* v_a_8_, lean_object* v_h_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; uint8_t v___x_15_; 
lean_inc(v_f_1_);
lean_inc(v_a_8_);
v___x_10_ = lean_apply_1(v_f_1_, v_a_8_);
v___x_11_ = lp_mathlib_Multiset_erase___redArg(v_inst_2_, v_s_3_, v_a_8_);
lean_inc(v_inst_5_);
v___x_12_ = lp_mathlib_Finset_sup___redArg(v_toSemilatticeSup_4_, v_inst_5_, v___x_11_, v_f_1_);
v___x_13_ = lean_apply_2(v_inf_6_, v___x_10_, v___x_12_);
v___x_14_ = lean_apply_2(v_inst_7_, v___x_13_, v_inst_5_);
v___x_15_ = lean_unbox(v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___lam__0___boxed(lean_object* v_f_16_, lean_object* v_inst_17_, lean_object* v_s_18_, lean_object* v_toSemilatticeSup_19_, lean_object* v_inst_20_, lean_object* v_inf_21_, lean_object* v_inst_22_, lean_object* v_a_23_, lean_object* v_h_24_){
_start:
{
uint8_t v_res_25_; lean_object* v_r_26_; 
v_res_25_ = lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___lam__0(v_f_16_, v_inst_17_, v_s_18_, v_toSemilatticeSup_19_, v_inst_20_, v_inf_21_, v_inst_22_, v_a_23_, v_h_24_);
v_r_26_ = lean_box(v_res_25_);
return v_r_26_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg(lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_s_29_, lean_object* v_f_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_toSemilatticeSup_33_; lean_object* v_inf_34_; lean_object* v___f_35_; uint8_t v___x_36_; 
v_toSemilatticeSup_33_ = lean_ctor_get(v_inst_27_, 0);
lean_inc_ref(v_toSemilatticeSup_33_);
v_inf_34_ = lean_ctor_get(v_inst_27_, 1);
lean_inc(v_inf_34_);
lean_dec_ref(v_inst_27_);
lean_inc(v_s_29_);
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___lam__0___boxed), 9, 7);
lean_closure_set(v___f_35_, 0, v_f_30_);
lean_closure_set(v___f_35_, 1, v_inst_31_);
lean_closure_set(v___f_35_, 2, v_s_29_);
lean_closure_set(v___f_35_, 3, v_toSemilatticeSup_33_);
lean_closure_set(v___f_35_, 4, v_inst_28_);
lean_closure_set(v___f_35_, 5, v_inf_34_);
lean_closure_set(v___f_35_, 6, v_inst_32_);
v___x_36_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_s_29_, v___f_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg___boxed(lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_s_39_, lean_object* v_f_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
uint8_t v_res_43_; lean_object* v_r_44_; 
v_res_43_ = lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg(v_inst_37_, v_inst_38_, v_s_39_, v_f_40_, v_inst_41_, v_inst_42_);
v_r_44_ = lean_box(v_res_43_);
return v_r_44_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq(lean_object* v_00_u03b1_45_, lean_object* v_00_u03b9_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_s_49_, lean_object* v_f_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___redArg(v_inst_47_, v_inst_48_, v_s_49_, v_f_50_, v_inst_51_, v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq___boxed(lean_object* v_00_u03b1_54_, lean_object* v_00_u03b9_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_s_58_, lean_object* v_f_59_, lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
uint8_t v_res_62_; lean_object* v_r_63_; 
v_res_62_ = lp_mathlib_Finset_instDecidableSupIndepOfDecidableEq(v_00_u03b1_54_, v_00_u03b9_55_, v_inst_56_, v_inst_57_, v_s_58_, v_f_59_, v_inst_60_, v_inst_61_);
v_r_63_ = lean_box(v_res_62_);
return v_r_63_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ModularLattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SupIndep(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ModularLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SupIndep(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ModularLattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SupIndep(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ModularLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SupIndep(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SupIndep(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SupIndep(builtin);
}
#ifdef __cplusplus
}
#endif
