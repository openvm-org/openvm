// Lean compiler output
// Module: Mathlib.Data.Fin.Tuple.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.Rev public import Mathlib.Data.Nat.Find public import Mathlib.Order.Fin.Basic public import Batteries.Data.Fin.Lemmas import Mathlib.Data.Set.Insert
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_succAbove___redArg(lean_object*, lean_object*);
lean_object* l_Fin_cases___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Fin_addCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Fin_consEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_consEquiv___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_consEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fin_consEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consCases___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consCases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_append___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_append___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_append(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_append___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_init___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_init(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_init___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocCases___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocCases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_find___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_find(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_piFinTwoEquiv___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_piFinTwoEquiv___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_piFinTwoEquiv___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_piFinTwoEquiv___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_piFinTwoEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_piFinTwoEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_piFinTwoEquiv___closed__0 = (const lean_object*)&lp_mathlib_piFinTwoEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_piFinTwoEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_piFinTwoEquiv___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_piFinTwoEquiv___closed__1 = (const lean_object*)&lp_mathlib_piFinTwoEquiv___closed__1_value;
static const lean_closure_object lp_mathlib_piFinTwoEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_piFinTwoEquiv___lam__3___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_piFinTwoEquiv___closed__1_value)} };
static const lean_object* lp_mathlib_piFinTwoEquiv___closed__2 = (const lean_object*)&lp_mathlib_piFinTwoEquiv___closed__2_value;
static const lean_ctor_object lp_mathlib_piFinTwoEquiv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_piFinTwoEquiv___closed__0_value),((lean_object*)&lp_mathlib_piFinTwoEquiv___closed__2_value)}};
static const lean_object* lp_mathlib_piFinTwoEquiv___closed__3 = (const lean_object*)&lp_mathlib_piFinTwoEquiv___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail___redArg(lean_object* v_q_1_, lean_object* v_i_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = l_Fin_succ___redArg(v_i_2_);
v___x_4_ = lean_apply_1(v_q_1_, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail___redArg___boxed(lean_object* v_q_5_, lean_object* v_i_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Fin_tail___redArg(v_q_5_, v_i_6_);
lean_dec(v_i_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail(lean_object* v_n_8_, lean_object* v_00_u03b1_9_, lean_object* v_q_10_, lean_object* v_i_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Fin_tail___redArg(v_q_10_, v_i_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_tail___boxed(lean_object* v_n_13_, lean_object* v_00_u03b1_14_, lean_object* v_q_15_, lean_object* v_i_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Fin_tail(v_n_13_, v_00_u03b1_14_, v_q_15_, v_i_16_);
lean_dec(v_i_16_);
lean_dec(v_n_13_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons___redArg(lean_object* v_x_18_, lean_object* v_p_19_, lean_object* v_j_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = l_Fin_cases___redArg(v_x_18_, v_p_19_, v_j_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons___redArg___boxed(lean_object* v_x_22_, lean_object* v_p_23_, lean_object* v_j_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Fin_cons___redArg(v_x_22_, v_p_23_, v_j_24_);
lean_dec(v_j_24_);
lean_dec(v_x_22_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons(lean_object* v_n_26_, lean_object* v_00_u03b1_27_, lean_object* v_x_28_, lean_object* v_p_29_, lean_object* v_j_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = l_Fin_cases___redArg(v_x_28_, v_p_29_, v_j_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_cons___boxed(lean_object* v_n_32_, lean_object* v_00_u03b1_33_, lean_object* v_x_34_, lean_object* v_p_35_, lean_object* v_j_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_Fin_cons(v_n_32_, v_00_u03b1_33_, v_x_34_, v_p_35_, v_j_36_);
lean_dec(v_j_36_);
lean_dec(v_x_34_);
lean_dec(v_n_32_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg___lam__0(lean_object* v_f_38_, lean_object* v___y_39_){
_start:
{
lean_object* v_fst_40_; lean_object* v_snd_41_; lean_object* v___x_42_; 
v_fst_40_ = lean_ctor_get(v_f_38_, 0);
lean_inc(v_fst_40_);
v_snd_41_ = lean_ctor_get(v_f_38_, 1);
lean_inc(v_snd_41_);
lean_dec_ref(v_f_38_);
v___x_42_ = l_Fin_cases___redArg(v_fst_40_, v_snd_41_, v___y_39_);
lean_dec(v_fst_40_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg___lam__0___boxed(lean_object* v_f_43_, lean_object* v___y_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_Fin_consEquiv___redArg___lam__0(v_f_43_, v___y_44_);
lean_dec(v___y_44_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg___lam__1(lean_object* v_n_46_, lean_object* v_f_47_){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_48_ = lean_unsigned_to_nat(1u);
v___x_49_ = lean_nat_add(v_n_46_, v___x_48_);
v___x_50_ = lean_unsigned_to_nat(0u);
v___x_51_ = lean_nat_mod(v___x_50_, v___x_49_);
lean_dec(v___x_49_);
lean_inc(v_f_47_);
v___x_52_ = lean_apply_1(v_f_47_, v___x_51_);
v___x_53_ = lean_alloc_closure((void*)(lp_mathlib_Fin_tail___boxed), 4, 3);
lean_closure_set(v___x_53_, 0, v_n_46_);
lean_closure_set(v___x_53_, 1, lean_box(0));
lean_closure_set(v___x_53_, 2, v_f_47_);
v___x_54_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_52_);
lean_ctor_set(v___x_54_, 1, v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv___redArg(lean_object* v_n_56_){
_start:
{
lean_object* v___f_57_; lean_object* v___f_58_; lean_object* v___x_59_; 
v___f_57_ = ((lean_object*)(lp_mathlib_Fin_consEquiv___redArg___closed__0));
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_Fin_consEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_58_, 0, v_n_56_);
v___x_59_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_59_, 0, v___f_57_);
lean_ctor_set(v___x_59_, 1, v___f_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consEquiv(lean_object* v_n_60_, lean_object* v_00_u03b1_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_Fin_consEquiv___redArg(v_n_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consCases___redArg(lean_object* v_n_63_, lean_object* v_cons_64_, lean_object* v_x_65_){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_66_ = lean_unsigned_to_nat(1u);
v___x_67_ = lean_nat_add(v_n_63_, v___x_66_);
v___x_68_ = lean_unsigned_to_nat(0u);
v___x_69_ = lean_nat_mod(v___x_68_, v___x_67_);
lean_dec(v___x_67_);
lean_inc(v_x_65_);
v___x_70_ = lean_apply_1(v_x_65_, v___x_69_);
v___x_71_ = lean_alloc_closure((void*)(lp_mathlib_Fin_tail___boxed), 4, 3);
lean_closure_set(v___x_71_, 0, v_n_63_);
lean_closure_set(v___x_71_, 1, lean_box(0));
lean_closure_set(v___x_71_, 2, v_x_65_);
v___x_72_ = lean_apply_2(v_cons_64_, v___x_70_, v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consCases(lean_object* v_n_73_, lean_object* v_00_u03b1_74_, lean_object* v_motive_75_, lean_object* v_cons_76_, lean_object* v_x_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Fin_consCases___redArg(v_n_73_, v_cons_76_, v_x_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___redArg(lean_object* v_elim0_79_, lean_object* v_cons_80_, lean_object* v_x_81_, lean_object* v_x_82_){
_start:
{
lean_object* v_zero_83_; uint8_t v_isZero_84_; 
v_zero_83_ = lean_unsigned_to_nat(0u);
v_isZero_84_ = lean_nat_dec_eq(v_x_81_, v_zero_83_);
if (v_isZero_84_ == 1)
{
lean_dec(v_x_82_);
lean_dec(v_cons_80_);
return v_elim0_79_;
}
else
{
lean_object* v_one_85_; lean_object* v_n_86_; lean_object* v___f_87_; lean_object* v___x_88_; 
v_one_85_ = lean_unsigned_to_nat(1u);
v_n_86_ = lean_nat_sub(v_x_81_, v_one_85_);
lean_inc(v_n_86_);
v___f_87_ = lean_alloc_closure((void*)(lp_mathlib_Fin_consInduction___redArg___lam__0), 5, 3);
lean_closure_set(v___f_87_, 0, v_elim0_79_);
lean_closure_set(v___f_87_, 1, v_cons_80_);
lean_closure_set(v___f_87_, 2, v_n_86_);
v___x_88_ = lp_mathlib_Fin_consCases___redArg(v_n_86_, v___f_87_, v_x_82_);
return v___x_88_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___redArg___lam__0(lean_object* v_elim0_89_, lean_object* v_cons_90_, lean_object* v_n_91_, lean_object* v_x_92_, lean_object* v_x_93_){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
lean_inc(v_x_93_);
lean_inc(v_cons_90_);
v___x_94_ = lp_mathlib_Fin_consInduction___redArg(v_elim0_89_, v_cons_90_, v_n_91_, v_x_93_);
v___x_95_ = lean_apply_4(v_cons_90_, v_n_91_, v_x_92_, v_x_93_, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___redArg___boxed(lean_object* v_elim0_96_, lean_object* v_cons_97_, lean_object* v_x_98_, lean_object* v_x_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Fin_consInduction___redArg(v_elim0_96_, v_cons_97_, v_x_98_, v_x_99_);
lean_dec(v_x_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction(lean_object* v_00_u03b1_101_, lean_object* v_motive_102_, lean_object* v_elim0_103_, lean_object* v_cons_104_, lean_object* v_x_105_, lean_object* v_x_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_Fin_consInduction___redArg(v_elim0_103_, v_cons_104_, v_x_105_, v_x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_consInduction___boxed(lean_object* v_00_u03b1_108_, lean_object* v_motive_109_, lean_object* v_elim0_110_, lean_object* v_cons_111_, lean_object* v_x_112_, lean_object* v_x_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_Fin_consInduction(v_00_u03b1_108_, v_motive_109_, v_elim0_110_, v_cons_111_, v_x_112_, v_x_113_);
lean_dec(v_x_112_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_append___redArg(lean_object* v_m_115_, lean_object* v_a_116_, lean_object* v_b_117_, lean_object* v_i_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = l_Fin_addCases___redArg(v_m_115_, v_a_116_, v_b_117_, v_i_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_append___redArg___boxed(lean_object* v_m_120_, lean_object* v_a_121_, lean_object* v_b_122_, lean_object* v_i_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_Fin_append___redArg(v_m_120_, v_a_121_, v_b_122_, v_i_123_);
lean_dec(v_m_120_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_append(lean_object* v_m_125_, lean_object* v_n_126_, lean_object* v_00_u03b1_127_, lean_object* v_a_128_, lean_object* v_b_129_, lean_object* v_i_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = l_Fin_addCases___redArg(v_m_125_, v_a_128_, v_b_129_, v_i_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_append___boxed(lean_object* v_m_132_, lean_object* v_n_133_, lean_object* v_00_u03b1_134_, lean_object* v_a_135_, lean_object* v_b_136_, lean_object* v_i_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Fin_append(v_m_132_, v_n_133_, v_00_u03b1_134_, v_a_135_, v_b_136_, v_i_137_);
lean_dec(v_n_133_);
lean_dec(v_m_132_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat___redArg(lean_object* v_n_139_, lean_object* v_a_140_, lean_object* v_x_141_){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_142_ = lean_nat_mod(v_x_141_, v_n_139_);
v___x_143_ = lean_apply_1(v_a_140_, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat___redArg___boxed(lean_object* v_n_144_, lean_object* v_a_145_, lean_object* v_x_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_Fin_repeat___redArg(v_n_144_, v_a_145_, v_x_146_);
lean_dec(v_x_146_);
lean_dec(v_n_144_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat(lean_object* v_n_148_, lean_object* v_00_u03b1_149_, lean_object* v_m_150_, lean_object* v_a_151_, lean_object* v_x_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_Fin_repeat___redArg(v_n_148_, v_a_151_, v_x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_repeat___boxed(lean_object* v_n_154_, lean_object* v_00_u03b1_155_, lean_object* v_m_156_, lean_object* v_a_157_, lean_object* v_x_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Fin_repeat(v_n_154_, v_00_u03b1_155_, v_m_156_, v_a_157_, v_x_158_);
lean_dec(v_x_158_);
lean_dec(v_m_156_);
lean_dec(v_n_154_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_init___redArg(lean_object* v_q_160_, lean_object* v_i_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_apply_1(v_q_160_, v_i_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_init(lean_object* v_n_163_, lean_object* v_00_u03b1_164_, lean_object* v_q_165_, lean_object* v_i_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_apply_1(v_q_165_, v_i_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_init___boxed(lean_object* v_n_168_, lean_object* v_00_u03b1_169_, lean_object* v_q_170_, lean_object* v_i_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Fin_init(v_n_168_, v_00_u03b1_169_, v_q_170_, v_i_171_);
lean_dec(v_n_168_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc___redArg(lean_object* v_n_173_, lean_object* v_p_174_, lean_object* v_x_175_, lean_object* v_i_176_){
_start:
{
uint8_t v___x_177_; 
v___x_177_ = lean_nat_dec_lt(v_i_176_, v_n_173_);
if (v___x_177_ == 0)
{
lean_dec(v_i_176_);
lean_dec(v_p_174_);
lean_inc(v_x_175_);
return v_x_175_;
}
else
{
lean_object* v___x_178_; 
v___x_178_ = lean_apply_1(v_p_174_, v_i_176_);
return v___x_178_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc___redArg___boxed(lean_object* v_n_179_, lean_object* v_p_180_, lean_object* v_x_181_, lean_object* v_i_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_Fin_snoc___redArg(v_n_179_, v_p_180_, v_x_181_, v_i_182_);
lean_dec(v_x_181_);
lean_dec(v_n_179_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc(lean_object* v_n_184_, lean_object* v_00_u03b1_185_, lean_object* v_p_186_, lean_object* v_x_187_, lean_object* v_i_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_Fin_snoc___redArg(v_n_184_, v_p_186_, v_x_187_, v_i_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snoc___boxed(lean_object* v_n_190_, lean_object* v_00_u03b1_191_, lean_object* v_p_192_, lean_object* v_x_193_, lean_object* v_i_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_Fin_snoc(v_n_190_, v_00_u03b1_191_, v_p_192_, v_x_193_, v_i_194_);
lean_dec(v_x_193_);
lean_dec(v_n_190_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg___lam__0(lean_object* v_n_196_, lean_object* v_f_197_, lean_object* v_x_198_){
_start:
{
lean_object* v_fst_199_; lean_object* v_snd_200_; lean_object* v___x_201_; 
v_fst_199_ = lean_ctor_get(v_f_197_, 0);
lean_inc(v_fst_199_);
v_snd_200_ = lean_ctor_get(v_f_197_, 1);
lean_inc(v_snd_200_);
lean_dec_ref(v_f_197_);
v___x_201_ = lp_mathlib_Fin_snoc___redArg(v_n_196_, v_snd_200_, v_fst_199_, v_x_198_);
lean_dec(v_fst_199_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg___lam__0___boxed(lean_object* v_n_202_, lean_object* v_f_203_, lean_object* v_x_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_Fin_snocEquiv___redArg___lam__0(v_n_202_, v_f_203_, v_x_204_);
lean_dec(v_n_202_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg___lam__1(lean_object* v_n_206_, lean_object* v_f_207_){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; 
lean_inc(v_f_207_);
lean_inc(v_n_206_);
v___x_208_ = lean_apply_1(v_f_207_, v_n_206_);
v___x_209_ = lean_alloc_closure((void*)(lp_mathlib_Fin_init___boxed), 4, 3);
lean_closure_set(v___x_209_, 0, v_n_206_);
lean_closure_set(v___x_209_, 1, lean_box(0));
lean_closure_set(v___x_209_, 2, v_f_207_);
v___x_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_210_, 0, v___x_208_);
lean_ctor_set(v___x_210_, 1, v___x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv___redArg(lean_object* v_n_211_){
_start:
{
lean_object* v___f_212_; lean_object* v___f_213_; lean_object* v___x_214_; 
lean_inc(v_n_211_);
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_Fin_snocEquiv___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_212_, 0, v_n_211_);
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_Fin_snocEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_213_, 0, v_n_211_);
v___x_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_214_, 0, v___f_212_);
lean_ctor_set(v___x_214_, 1, v___f_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocEquiv(lean_object* v_n_215_, lean_object* v_00_u03b1_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lp_mathlib_Fin_snocEquiv___redArg(v_n_215_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocCases___redArg(lean_object* v_n_218_, lean_object* v_snoc_219_, lean_object* v_x_220_){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
lean_inc(v_x_220_);
lean_inc(v_n_218_);
v___x_221_ = lean_alloc_closure((void*)(lp_mathlib_Fin_init___boxed), 4, 3);
lean_closure_set(v___x_221_, 0, v_n_218_);
lean_closure_set(v___x_221_, 1, lean_box(0));
lean_closure_set(v___x_221_, 2, v_x_220_);
v___x_222_ = lean_apply_1(v_x_220_, v_n_218_);
v___x_223_ = lean_apply_2(v_snoc_219_, v___x_221_, v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocCases(lean_object* v_n_224_, lean_object* v_00_u03b1_225_, lean_object* v_motive_226_, lean_object* v_snoc_227_, lean_object* v_x_228_){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
lean_inc(v_x_228_);
lean_inc(v_n_224_);
v___x_229_ = lean_alloc_closure((void*)(lp_mathlib_Fin_init___boxed), 4, 3);
lean_closure_set(v___x_229_, 0, v_n_224_);
lean_closure_set(v___x_229_, 1, lean_box(0));
lean_closure_set(v___x_229_, 2, v_x_228_);
v___x_230_ = lean_apply_1(v_x_228_, v_n_224_);
v___x_231_ = lean_apply_2(v_snoc_227_, v___x_229_, v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction___redArg(lean_object* v_elim0_232_, lean_object* v_snoc_233_, lean_object* v_x_234_, lean_object* v_x_235_){
_start:
{
lean_object* v_zero_236_; uint8_t v_isZero_237_; 
v_zero_236_ = lean_unsigned_to_nat(0u);
v_isZero_237_ = lean_nat_dec_eq(v_x_234_, v_zero_236_);
if (v_isZero_237_ == 1)
{
lean_dec(v_x_235_);
lean_dec(v_snoc_233_);
lean_inc(v_elim0_232_);
return v_elim0_232_;
}
else
{
lean_object* v_one_238_; lean_object* v_n_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; 
v_one_238_ = lean_unsigned_to_nat(1u);
v_n_239_ = lean_nat_sub(v_x_234_, v_one_238_);
lean_inc(v_x_235_);
lean_inc_n(v_n_239_, 2);
v___x_240_ = lean_alloc_closure((void*)(lp_mathlib_Fin_init___boxed), 4, 3);
lean_closure_set(v___x_240_, 0, v_n_239_);
lean_closure_set(v___x_240_, 1, lean_box(0));
lean_closure_set(v___x_240_, 2, v_x_235_);
v___x_241_ = lean_apply_1(v_x_235_, v_n_239_);
lean_inc_ref(v___x_240_);
lean_inc(v_snoc_233_);
v___x_242_ = lp_mathlib_Fin_snocInduction___redArg(v_elim0_232_, v_snoc_233_, v_n_239_, v___x_240_);
v___x_243_ = lean_apply_4(v_snoc_233_, v_n_239_, v___x_240_, v___x_241_, v___x_242_);
return v___x_243_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction___redArg___boxed(lean_object* v_elim0_244_, lean_object* v_snoc_245_, lean_object* v_x_246_, lean_object* v_x_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_Fin_snocInduction___redArg(v_elim0_244_, v_snoc_245_, v_x_246_, v_x_247_);
lean_dec(v_x_246_);
lean_dec(v_elim0_244_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction(lean_object* v_00_u03b1_249_, lean_object* v_motive_250_, lean_object* v_elim0_251_, lean_object* v_snoc_252_, lean_object* v_x_253_, lean_object* v_x_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_Fin_snocInduction___redArg(v_elim0_251_, v_snoc_252_, v_x_253_, v_x_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_snocInduction___boxed(lean_object* v_00_u03b1_256_, lean_object* v_motive_257_, lean_object* v_elim0_258_, lean_object* v_snoc_259_, lean_object* v_x_260_, lean_object* v_x_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_Fin_snocInduction(v_00_u03b1_256_, v_motive_257_, v_elim0_258_, v_snoc_259_, v_x_260_, v_x_261_);
lean_dec(v_x_260_);
lean_dec(v_elim0_258_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases___redArg(lean_object* v_i_263_, lean_object* v_x_264_, lean_object* v_p_265_, lean_object* v_j_266_){
_start:
{
uint8_t v___x_267_; 
v___x_267_ = lean_nat_dec_eq(v_j_266_, v_i_263_);
if (v___x_267_ == 0)
{
uint8_t v___x_268_; 
v___x_268_ = lean_nat_dec_lt(v_j_266_, v_i_263_);
if (v___x_268_ == 0)
{
lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_269_ = lean_unsigned_to_nat(1u);
v___x_270_ = lean_nat_sub(v_j_266_, v___x_269_);
lean_dec(v_j_266_);
v___x_271_ = lean_apply_1(v_p_265_, v___x_270_);
return v___x_271_;
}
else
{
lean_object* v___x_272_; 
v___x_272_ = lean_apply_1(v_p_265_, v_j_266_);
return v___x_272_;
}
}
else
{
lean_dec(v_j_266_);
lean_dec(v_p_265_);
lean_inc(v_x_264_);
return v_x_264_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases___redArg___boxed(lean_object* v_i_273_, lean_object* v_x_274_, lean_object* v_p_275_, lean_object* v_j_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_Fin_succAboveCases___redArg(v_i_273_, v_x_274_, v_p_275_, v_j_276_);
lean_dec(v_x_274_);
lean_dec(v_i_273_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases(lean_object* v_n_278_, lean_object* v_00_u03b1_279_, lean_object* v_i_280_, lean_object* v_x_281_, lean_object* v_p_282_, lean_object* v_j_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_Fin_succAboveCases___redArg(v_i_280_, v_x_281_, v_p_282_, v_j_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveCases___boxed(lean_object* v_n_285_, lean_object* v_00_u03b1_286_, lean_object* v_i_287_, lean_object* v_x_288_, lean_object* v_p_289_, lean_object* v_j_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Fin_succAboveCases(v_n_285_, v_00_u03b1_286_, v_i_287_, v_x_288_, v_p_289_, v_j_290_);
lean_dec(v_x_288_);
lean_dec(v_i_287_);
lean_dec(v_n_285_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth___redArg(lean_object* v_p_292_, lean_object* v_f_293_, lean_object* v_i_294_){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_295_ = lp_mathlib_Fin_succAbove___redArg(v_p_292_, v_i_294_);
v___x_296_ = lean_apply_1(v_f_293_, v___x_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth___redArg___boxed(lean_object* v_p_297_, lean_object* v_f_298_, lean_object* v_i_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_Fin_removeNth___redArg(v_p_297_, v_f_298_, v_i_299_);
lean_dec(v_i_299_);
lean_dec(v_p_297_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth(lean_object* v_n_301_, lean_object* v_00_u03b1_302_, lean_object* v_p_303_, lean_object* v_f_304_, lean_object* v_i_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_Fin_removeNth___redArg(v_p_303_, v_f_304_, v_i_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_removeNth___boxed(lean_object* v_n_307_, lean_object* v_00_u03b1_308_, lean_object* v_p_309_, lean_object* v_f_310_, lean_object* v_i_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_Fin_removeNth(v_n_307_, v_00_u03b1_308_, v_p_309_, v_f_310_, v_i_311_);
lean_dec(v_i_311_);
lean_dec(v_p_309_);
lean_dec(v_n_307_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth___redArg(lean_object* v_i_313_, lean_object* v_x_314_, lean_object* v_p_315_, lean_object* v_j_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_Fin_succAboveCases___redArg(v_i_313_, v_x_314_, v_p_315_, v_j_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth___redArg___boxed(lean_object* v_i_318_, lean_object* v_x_319_, lean_object* v_p_320_, lean_object* v_j_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_Fin_insertNth___redArg(v_i_318_, v_x_319_, v_p_320_, v_j_321_);
lean_dec(v_x_319_);
lean_dec(v_i_318_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth(lean_object* v_n_323_, lean_object* v_00_u03b1_324_, lean_object* v_i_325_, lean_object* v_x_326_, lean_object* v_p_327_, lean_object* v_j_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lp_mathlib_Fin_succAboveCases___redArg(v_i_325_, v_x_326_, v_p_327_, v_j_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNth___boxed(lean_object* v_n_330_, lean_object* v_00_u03b1_331_, lean_object* v_i_332_, lean_object* v_x_333_, lean_object* v_p_334_, lean_object* v_j_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Fin_insertNth(v_n_330_, v_00_u03b1_331_, v_i_332_, v_x_333_, v_p_334_, v_j_335_);
lean_dec(v_x_333_);
lean_dec(v_i_332_);
lean_dec(v_n_330_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg___lam__0(lean_object* v_p_337_, lean_object* v_f_338_, lean_object* v___y_339_){
_start:
{
lean_object* v_fst_340_; lean_object* v_snd_341_; lean_object* v___x_342_; 
v_fst_340_ = lean_ctor_get(v_f_338_, 0);
lean_inc(v_fst_340_);
v_snd_341_ = lean_ctor_get(v_f_338_, 1);
lean_inc(v_snd_341_);
lean_dec_ref(v_f_338_);
v___x_342_ = lp_mathlib_Fin_succAboveCases___redArg(v_p_337_, v_fst_340_, v_snd_341_, v___y_339_);
lean_dec(v_fst_340_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg___lam__0___boxed(lean_object* v_p_343_, lean_object* v_f_344_, lean_object* v___y_345_){
_start:
{
lean_object* v_res_346_; 
v_res_346_ = lp_mathlib_Fin_insertNthEquiv___redArg___lam__0(v_p_343_, v_f_344_, v___y_345_);
lean_dec(v_p_343_);
return v_res_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg___lam__1(lean_object* v_p_347_, lean_object* v_n_348_, lean_object* v_f_349_){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
lean_inc(v_f_349_);
lean_inc(v_p_347_);
v___x_350_ = lean_apply_1(v_f_349_, v_p_347_);
v___x_351_ = lean_alloc_closure((void*)(lp_mathlib_Fin_removeNth___boxed), 5, 4);
lean_closure_set(v___x_351_, 0, v_n_348_);
lean_closure_set(v___x_351_, 1, lean_box(0));
lean_closure_set(v___x_351_, 2, v_p_347_);
lean_closure_set(v___x_351_, 3, v_f_349_);
v___x_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_350_);
lean_ctor_set(v___x_352_, 1, v___x_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv___redArg(lean_object* v_n_353_, lean_object* v_p_354_){
_start:
{
lean_object* v___f_355_; lean_object* v___f_356_; lean_object* v___x_357_; 
lean_inc(v_p_354_);
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_Fin_insertNthEquiv___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_355_, 0, v_p_354_);
v___f_356_ = lean_alloc_closure((void*)(lp_mathlib_Fin_insertNthEquiv___redArg___lam__1), 3, 2);
lean_closure_set(v___f_356_, 0, v_p_354_);
lean_closure_set(v___f_356_, 1, v_n_353_);
v___x_357_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_357_, 0, v___f_355_);
lean_ctor_set(v___x_357_, 1, v___f_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_insertNthEquiv(lean_object* v_n_358_, lean_object* v_00_u03b1_359_, lean_object* v_p_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_mathlib_Fin_insertNthEquiv___redArg(v_n_358_, v_p_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(lean_object* v_n_362_, lean_object* v_inst_363_, lean_object* v_m_364_){
_start:
{
lean_object* v_zero_365_; uint8_t v_isZero_366_; lean_object* v_one_367_; lean_object* v_n_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; uint8_t v___x_372_; 
v_zero_365_ = lean_unsigned_to_nat(0u);
v_isZero_366_ = lean_nat_dec_eq(v_m_364_, v_zero_365_);
v_one_367_ = lean_unsigned_to_nat(1u);
v_n_368_ = lean_nat_sub(v_m_364_, v_one_367_);
lean_dec(v_m_364_);
v___x_369_ = lean_nat_add(v_n_368_, v_one_367_);
v___x_370_ = lean_nat_sub(v_n_362_, v___x_369_);
lean_dec(v___x_369_);
lean_inc_ref(v_inst_363_);
lean_inc(v___x_370_);
v___x_371_ = lean_apply_1(v_inst_363_, v___x_370_);
v___x_372_ = lean_unbox(v___x_371_);
if (v___x_372_ == 0)
{
lean_dec(v___x_370_);
v_m_364_ = v_n_368_;
goto _start;
}
else
{
lean_dec(v_n_368_);
lean_dec_ref(v_inst_363_);
return v___x_370_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg___boxed(lean_object* v_n_374_, lean_object* v_inst_375_, lean_object* v_m_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(v_n_374_, v_inst_375_, v_m_376_);
lean_dec(v_n_374_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go(lean_object* v_n_378_, lean_object* v_p_379_, lean_object* v_inst_380_, lean_object* v_h_381_, lean_object* v_m_382_, lean_object* v_hj_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(v_n_378_, v_inst_380_, v_m_382_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___boxed(lean_object* v_n_385_, lean_object* v_p_386_, lean_object* v_inst_387_, lean_object* v_h_388_, lean_object* v_m_389_, lean_object* v_hj_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go(v_n_385_, v_p_386_, v_inst_387_, v_h_388_, v_m_389_, v_hj_390_);
lean_dec(v_n_385_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX___redArg(lean_object* v_n_392_, lean_object* v_inst_393_){
_start:
{
lean_object* v___x_394_; 
lean_inc(v_n_392_);
v___x_394_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(v_n_392_, v_inst_393_, v_n_392_);
lean_dec(v_n_392_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX(lean_object* v_n_395_, lean_object* v_p_396_, lean_object* v_inst_397_, lean_object* v_h_398_){
_start:
{
lean_object* v___x_399_; 
lean_inc(v_n_395_);
v___x_399_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(v_n_395_, v_inst_397_, v_n_395_);
lean_dec(v_n_395_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_find___redArg(lean_object* v_n_400_, lean_object* v_inst_401_){
_start:
{
lean_object* v___x_402_; 
lean_inc(v_n_400_);
v___x_402_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(v_n_400_, v_inst_401_, v_n_400_);
lean_dec(v_n_400_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_find(lean_object* v_n_403_, lean_object* v_p_404_, lean_object* v_inst_405_, lean_object* v_h_406_){
_start:
{
lean_object* v___x_407_; 
lean_inc(v_n_403_);
v___x_407_ = lp_mathlib___private_Mathlib_Data_Fin_Tuple_Basic_0__Fin_findX_go___redArg(v_n_403_, v_inst_405_, v_n_403_);
lean_dec(v_n_403_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth___redArg(lean_object* v_j_408_, lean_object* v_op_409_, lean_object* v_g_410_, lean_object* v_k_411_){
_start:
{
uint8_t v___x_412_; 
v___x_412_ = lean_nat_dec_lt(v_k_411_, v_j_408_);
if (v___x_412_ == 0)
{
uint8_t v___x_413_; 
v___x_413_ = lean_nat_dec_eq(v_k_411_, v_j_408_);
if (v___x_413_ == 0)
{
lean_object* v___x_414_; lean_object* v___x_415_; 
lean_dec(v_op_409_);
v___x_414_ = l_Fin_succ___redArg(v_k_411_);
lean_dec(v_k_411_);
v___x_415_ = lean_apply_1(v_g_410_, v___x_414_);
return v___x_415_;
}
else
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
lean_inc(v_g_410_);
lean_inc(v_k_411_);
v___x_416_ = lean_apply_1(v_g_410_, v_k_411_);
v___x_417_ = l_Fin_succ___redArg(v_k_411_);
lean_dec(v_k_411_);
v___x_418_ = lean_apply_1(v_g_410_, v___x_417_);
v___x_419_ = lean_apply_2(v_op_409_, v___x_416_, v___x_418_);
return v___x_419_;
}
}
else
{
lean_object* v___x_420_; 
lean_dec(v_op_409_);
v___x_420_ = lean_apply_1(v_g_410_, v_k_411_);
return v___x_420_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth___redArg___boxed(lean_object* v_j_421_, lean_object* v_op_422_, lean_object* v_g_423_, lean_object* v_k_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_Fin_contractNth___redArg(v_j_421_, v_op_422_, v_g_423_, v_k_424_);
lean_dec(v_j_421_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth(lean_object* v_n_426_, lean_object* v_00_u03b1_427_, lean_object* v_j_428_, lean_object* v_op_429_, lean_object* v_g_430_, lean_object* v_k_431_){
_start:
{
lean_object* v___x_432_; 
v___x_432_ = lp_mathlib_Fin_contractNth___redArg(v_j_428_, v_op_429_, v_g_430_, v_k_431_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_contractNth___boxed(lean_object* v_n_433_, lean_object* v_00_u03b1_434_, lean_object* v_j_435_, lean_object* v_op_436_, lean_object* v_g_437_, lean_object* v_k_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_Fin_contractNth(v_n_433_, v_00_u03b1_434_, v_j_435_, v_op_436_, v_g_437_, v_k_438_);
lean_dec(v_j_435_);
lean_dec(v_n_433_);
return v_res_439_;
}
}
static lean_object* _init_lp_mathlib_piFinTwoEquiv___lam__0___closed__0(void){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_440_ = lean_unsigned_to_nat(2u);
v___x_441_ = lean_unsigned_to_nat(0u);
v___x_442_ = lean_nat_mod(v___x_441_, v___x_440_);
return v___x_442_;
}
}
static lean_object* _init_lp_mathlib_piFinTwoEquiv___lam__0___closed__1(void){
_start:
{
lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_443_ = lean_unsigned_to_nat(2u);
v___x_444_ = lean_unsigned_to_nat(1u);
v___x_445_ = lean_nat_mod(v___x_444_, v___x_443_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__0(lean_object* v_f_446_){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_447_ = lean_obj_once(&lp_mathlib_piFinTwoEquiv___lam__0___closed__0, &lp_mathlib_piFinTwoEquiv___lam__0___closed__0_once, _init_lp_mathlib_piFinTwoEquiv___lam__0___closed__0);
lean_inc(v_f_446_);
v___x_448_ = lean_apply_1(v_f_446_, v___x_447_);
v___x_449_ = lean_obj_once(&lp_mathlib_piFinTwoEquiv___lam__0___closed__1, &lp_mathlib_piFinTwoEquiv___lam__0___closed__1_once, _init_lp_mathlib_piFinTwoEquiv___lam__0___closed__1);
v___x_450_ = lean_apply_1(v_f_446_, v___x_449_);
v___x_451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_451_, 0, v___x_448_);
lean_ctor_set(v___x_451_, 1, v___x_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__1(lean_object* v___y_452_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__1___boxed(lean_object* v___y_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_mathlib_piFinTwoEquiv___lam__1(v___y_453_);
lean_dec(v___y_453_);
return v_res_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__2(lean_object* v_snd_455_, lean_object* v___f_456_, lean_object* v___y_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = l_Fin_cases___redArg(v_snd_455_, v___f_456_, v___y_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__2___boxed(lean_object* v_snd_459_, lean_object* v___f_460_, lean_object* v___y_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_piFinTwoEquiv___lam__2(v_snd_459_, v___f_460_, v___y_461_);
lean_dec(v___y_461_);
lean_dec(v_snd_459_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__3(lean_object* v___f_463_, lean_object* v_p_464_, lean_object* v___y_465_){
_start:
{
lean_object* v_fst_466_; lean_object* v_snd_467_; lean_object* v___f_468_; lean_object* v___x_469_; 
v_fst_466_ = lean_ctor_get(v_p_464_, 0);
lean_inc(v_fst_466_);
v_snd_467_ = lean_ctor_get(v_p_464_, 1);
lean_inc(v_snd_467_);
lean_dec_ref(v_p_464_);
v___f_468_ = lean_alloc_closure((void*)(lp_mathlib_piFinTwoEquiv___lam__2___boxed), 3, 2);
lean_closure_set(v___f_468_, 0, v_snd_467_);
lean_closure_set(v___f_468_, 1, v___f_463_);
v___x_469_ = l_Fin_cases___redArg(v_fst_466_, v___f_468_, v___y_465_);
lean_dec(v_fst_466_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv___lam__3___boxed(lean_object* v___f_470_, lean_object* v_p_471_, lean_object* v___y_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_piFinTwoEquiv___lam__3(v___f_470_, v_p_471_, v___y_472_);
lean_dec(v___y_472_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_piFinTwoEquiv(lean_object* v_00_u03b1_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = ((lean_object*)(lp_mathlib_piFinTwoEquiv___closed__3));
return v___x_482_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Rev(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Rev(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_Rev(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Find(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Rev(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fin_Tuple_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
