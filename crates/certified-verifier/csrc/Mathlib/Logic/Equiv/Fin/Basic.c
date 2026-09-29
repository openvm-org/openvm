// Lean compiler output
// Module: Mathlib.Logic.Equiv.Fin.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Fin.VecNotation public import Mathlib.Logic.Embedding.Set public import Mathlib.Logic.Equiv.Option public import Mathlib.Data.Int.Init public import Batteries.Data.Fin.Lemmas
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
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Option_casesOn_x27___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_piFinTwoEquiv(lean_object*);
lean_object* l_Fin_addCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Fin_natAdd___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumComm(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodCongrRight___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Fin_cases___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_int_ediv(lean_object*, lean_object*);
lean_object* lp_mathlib_Int_natMod(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_succAbove___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_succAboveCases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_embeddingCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_optionEmbeddingEquiv(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_prodEquivPiFinTwo___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_prodEquivPiFinTwo___closed__0;
static lean_once_cell_t lp_mathlib_prodEquivPiFinTwo___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_prodEquivPiFinTwo___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_prodEquivPiFinTwo(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_finTwoArrowEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finTwoArrowEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finTwoArrowEquiv___closed__0 = (const lean_object*)&lp_mathlib_finTwoArrowEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_finTwoArrowEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finTwoArrowEquiv___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_finTwoArrowEquiv___closed__0_value)} };
static const lean_object* lp_mathlib_finTwoArrowEquiv___closed__1 = (const lean_object*)&lp_mathlib_finTwoArrowEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_finSuccEquiv_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finSuccEquiv_x27___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finSuccEquiv_x27___closed__0 = (const lean_object*)&lp_mathlib_finSuccEquiv_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__6(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccAboveEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquivLast(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingFinSucc___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingFinSucc(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__3___boxed(lean_object*);
static const lean_closure_object lp_mathlib_finSumFinEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finSumFinEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finSumFinEquiv___closed__0 = (const lean_object*)&lp_mathlib_finSumFinEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_finSumFinEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finSumFinEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finSumFinEquiv___closed__1 = (const lean_object*)&lp_mathlib_finSumFinEquiv___closed__1_value;
static const lean_closure_object lp_mathlib_finSumFinEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finSumFinEquiv___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finSumFinEquiv___closed__2 = (const lean_object*)&lp_mathlib_finSumFinEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_finSumNatEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_finSumNatEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_finSumNatEquiv___closed__0 = (const lean_object*)&lp_mathlib_finSumNatEquiv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv(lean_object*);
static lean_once_cell_t lp_mathlib_finAddFlip___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_finAddFlip___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_finAddFlip(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_castLEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_castLEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_castLEquiv___closed__0 = (const lean_object*)&lp_mathlib_Fin_castLEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Fin_castLEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Fin_castLEquiv___closed__0_value),((lean_object*)&lp_mathlib_Fin_castLEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Fin_castLEquiv___closed__1 = (const lean_object*)&lp_mathlib_Fin_castLEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Fin_succFunEquiv___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_succFunEquiv___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_succFunEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_succFunEquiv___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_succFunEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fin_succFunEquiv___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Fin_succFunEquiv___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_succFunEquiv___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_prodEquivPiFinTwo___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_piFinTwoEquiv(lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_prodEquivPiFinTwo___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_obj_once(&lp_mathlib_prodEquivPiFinTwo___closed__0, &lp_mathlib_prodEquivPiFinTwo___closed__0_once, _init_lp_mathlib_prodEquivPiFinTwo___closed__0);
v___x_3_ = lp_mathlib_Equiv_symm___redArg(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_prodEquivPiFinTwo(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_obj_once(&lp_mathlib_prodEquivPiFinTwo___closed__1, &lp_mathlib_prodEquivPiFinTwo___closed__1_once, _init_lp_mathlib_prodEquivPiFinTwo___closed__1);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__0(lean_object* v___y_7_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__0___boxed(lean_object* v___y_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_finTwoArrowEquiv___lam__0(v___y_8_);
lean_dec(v___y_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__1(lean_object* v_snd_10_, lean_object* v___f_11_, lean_object* v___y_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = l_Fin_cases___redArg(v_snd_10_, v___f_11_, v___y_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__1___boxed(lean_object* v_snd_14_, lean_object* v___f_15_, lean_object* v___y_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_finTwoArrowEquiv___lam__1(v_snd_14_, v___f_15_, v___y_16_);
lean_dec(v___y_16_);
lean_dec(v_snd_14_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__2(lean_object* v___f_18_, lean_object* v_x_19_, lean_object* v___y_20_){
_start:
{
lean_object* v_fst_21_; lean_object* v_snd_22_; lean_object* v___f_23_; lean_object* v___x_24_; 
v_fst_21_ = lean_ctor_get(v_x_19_, 0);
lean_inc(v_fst_21_);
v_snd_22_ = lean_ctor_get(v_x_19_, 1);
lean_inc(v_snd_22_);
lean_dec_ref(v_x_19_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_finTwoArrowEquiv___lam__1___boxed), 3, 2);
lean_closure_set(v___f_23_, 0, v_snd_22_);
lean_closure_set(v___f_23_, 1, v___f_18_);
v___x_24_ = l_Fin_cases___redArg(v_fst_21_, v___f_23_, v___y_20_);
lean_dec(v_fst_21_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv___lam__2___boxed(lean_object* v___f_25_, lean_object* v_x_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_finTwoArrowEquiv___lam__2(v___f_25_, v_x_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finTwoArrowEquiv(lean_object* v_00_u03b1_32_){
_start:
{
lean_object* v___x_33_; lean_object* v_toFun_34_; lean_object* v___f_35_; lean_object* v___x_36_; 
v___x_33_ = lean_obj_once(&lp_mathlib_prodEquivPiFinTwo___closed__0, &lp_mathlib_prodEquivPiFinTwo___closed__0_once, _init_lp_mathlib_prodEquivPiFinTwo___closed__0);
v_toFun_34_ = lean_ctor_get(v___x_33_, 0);
v___f_35_ = ((lean_object*)(lp_mathlib_finTwoArrowEquiv___closed__1));
lean_inc(v_toFun_34_);
v___x_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_36_, 0, v_toFun_34_);
lean_ctor_set(v___x_36_, 1, v___f_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__0(lean_object* v_n_37_, lean_object* v_i_38_, lean_object* v_x_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
lean_inc(v_i_38_);
v___x_40_ = lean_alloc_closure((void*)(lp_mathlib_Fin_succAbove___boxed), 3, 2);
lean_closure_set(v___x_40_, 0, v_n_37_);
lean_closure_set(v___x_40_, 1, v_i_38_);
v___x_41_ = lp_mathlib_Option_casesOn_x27___redArg(v_x_39_, v_i_38_, v___x_40_);
lean_dec(v_i_38_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__1(lean_object* v_val_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_43_, 0, v_val_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__2(lean_object* v_i_44_, lean_object* v___x_45_, lean_object* v___f_46_, lean_object* v___y_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Fin_succAboveCases___redArg(v_i_44_, v___x_45_, v___f_46_, v___y_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27___lam__2___boxed(lean_object* v_i_49_, lean_object* v___x_50_, lean_object* v___f_51_, lean_object* v___y_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_finSuccEquiv_x27___lam__2(v_i_49_, v___x_50_, v___f_51_, v___y_52_);
lean_dec(v___x_50_);
lean_dec(v_i_49_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv_x27(lean_object* v_n_55_, lean_object* v_i_56_){
_start:
{
lean_object* v___f_57_; lean_object* v___f_58_; lean_object* v___x_59_; lean_object* v___f_60_; lean_object* v___x_61_; 
lean_inc(v_i_56_);
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_finSuccEquiv_x27___lam__0), 3, 2);
lean_closure_set(v___f_57_, 0, v_n_55_);
lean_closure_set(v___f_57_, 1, v_i_56_);
v___f_58_ = ((lean_object*)(lp_mathlib_finSuccEquiv_x27___closed__0));
v___x_59_ = lean_box(0);
v___f_60_ = lean_alloc_closure((void*)(lp_mathlib_finSuccEquiv_x27___lam__2___boxed), 4, 3);
lean_closure_set(v___f_60_, 0, v_i_56_);
lean_closure_set(v___f_60_, 1, v___x_59_);
lean_closure_set(v___f_60_, 2, v___f_58_);
v___x_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_61_, 0, v___f_60_);
lean_ctor_set(v___x_61_, 1, v___f_57_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquiv(lean_object* v_n_62_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_63_ = lean_unsigned_to_nat(1u);
v___x_64_ = lean_nat_add(v_n_62_, v___x_63_);
v___x_65_ = lean_unsigned_to_nat(0u);
v___x_66_ = lean_nat_mod(v___x_65_, v___x_64_);
lean_dec(v___x_64_);
v___x_67_ = lp_mathlib_finSuccEquiv_x27(v_n_62_, v___x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__0(lean_object* v_e_68_, lean_object* v_a_69_){
_start:
{
lean_object* v_toFun_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v_toFun_70_ = lean_ctor_get(v_e_68_, 0);
lean_inc(v_toFun_70_);
lean_dec_ref(v_e_68_);
v___x_71_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_71_, 0, v_a_69_);
v___x_72_ = lean_apply_1(v_toFun_70_, v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__1(lean_object* v_e_73_, lean_object* v_b_74_){
_start:
{
lean_object* v___x_75_; lean_object* v_toFun_76_; lean_object* v___x_77_; lean_object* v_val_78_; 
v___x_75_ = lp_mathlib_Equiv_symm___redArg(v_e_73_);
v_toFun_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_toFun_76_);
lean_dec_ref(v___x_75_);
v___x_77_ = lean_apply_1(v_toFun_76_, v_b_74_);
v_val_78_ = lean_ctor_get(v___x_77_, 0);
lean_inc(v_val_78_);
lean_dec(v___x_77_);
return v_val_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__2(lean_object* v_e_79_){
_start:
{
lean_object* v___f_80_; lean_object* v___f_81_; lean_object* v___x_82_; 
lean_inc_ref(v_e_79_);
v___f_80_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_80_, 0, v_e_79_);
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__1), 2, 1);
lean_closure_set(v___f_81_, 0, v_e_79_);
v___x_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_82_, 0, v___f_80_);
lean_ctor_set(v___x_82_, 1, v___f_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__3(lean_object* v_e_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_toFun_85_; lean_object* v___x_86_; 
v_toFun_85_ = lean_ctor_get(v_e_83_, 0);
lean_inc(v_toFun_85_);
lean_dec_ref(v_e_83_);
v___x_86_ = lean_apply_1(v_toFun_85_, v___y_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__4(lean_object* v_x_87_, lean_object* v___f_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_mathlib_Option_casesOn_x27___redArg(v_a_89_, v_x_87_, v___f_88_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__4___boxed(lean_object* v_x_91_, lean_object* v___f_92_, lean_object* v_a_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__4(v_x_91_, v___f_92_, v_a_93_);
lean_dec(v_x_91_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__5(lean_object* v_x_95_, lean_object* v_e_96_, lean_object* v_b_97_){
_start:
{
uint8_t v___x_98_; 
v___x_98_ = lean_nat_dec_eq(v_b_97_, v_x_95_);
if (v___x_98_ == 0)
{
lean_object* v___x_99_; lean_object* v_toFun_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_99_ = lp_mathlib_Equiv_symm___redArg(v_e_96_);
v_toFun_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_toFun_100_);
lean_dec_ref(v___x_99_);
v___x_101_ = lean_apply_1(v_toFun_100_, v_b_97_);
v___x_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
return v___x_102_;
}
else
{
lean_object* v___x_103_; 
lean_dec(v_b_97_);
lean_dec_ref(v_e_96_);
v___x_103_ = lean_box(0);
return v___x_103_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__5___boxed(lean_object* v_x_104_, lean_object* v_e_105_, lean_object* v_b_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__5(v_x_104_, v_e_105_, v_b_106_);
lean_dec(v_x_104_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__6(lean_object* v_x_108_, lean_object* v_e_109_){
_start:
{
lean_object* v___f_110_; lean_object* v___f_111_; lean_object* v___f_112_; lean_object* v___x_113_; 
lean_inc_ref(v_e_109_);
v___f_110_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__3), 2, 1);
lean_closure_set(v___f_110_, 0, v_e_109_);
lean_inc(v_x_108_);
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__4___boxed), 3, 2);
lean_closure_set(v___f_111_, 0, v_x_108_);
lean_closure_set(v___f_111_, 1, v___f_110_);
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__5___boxed), 3, 2);
lean_closure_set(v___f_112_, 0, v_x_108_);
lean_closure_set(v___f_112_, 1, v_e_109_);
v___x_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_113_, 0, v___f_111_);
lean_ctor_set(v___x_113_, 1, v___f_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg(lean_object* v_x_115_){
_start:
{
lean_object* v___f_116_; lean_object* v___f_117_; lean_object* v___x_118_; 
v___f_116_ = ((lean_object*)(lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___closed__0));
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg___lam__6), 2, 1);
lean_closure_set(v___f_117_, 0, v_x_115_);
v___x_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_118_, 0, v___f_116_);
lean_ctor_set(v___x_118_, 1, v___f_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0(lean_object* v___x_119_, lean_object* v_00_u03b1_120_, lean_object* v_x_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg(v_x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___boxed(lean_object* v___x_123_, lean_object* v_00_u03b1_124_, lean_object* v_x_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0(v___x_123_, v_00_u03b1_124_, v_x_125_);
lean_dec(v___x_123_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccAboveEquiv(lean_object* v_n_127_, lean_object* v_p_128_){
_start:
{
lean_object* v___x_129_; lean_object* v_toFun_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
lean_inc(v_p_128_);
v___x_129_ = lp_mathlib_Equiv_optionSubtype___at___00finSuccAboveEquiv_spec__0___redArg(v_p_128_);
v_toFun_130_ = lean_ctor_get(v___x_129_, 0);
lean_inc(v_toFun_130_);
lean_dec_ref(v___x_129_);
v___x_131_ = lp_mathlib_finSuccEquiv_x27(v_n_127_, v_p_128_);
v___x_132_ = lp_mathlib_Equiv_symm___redArg(v___x_131_);
v___x_133_ = lean_apply_1(v_toFun_130_, v___x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSuccEquivLast(lean_object* v_n_134_){
_start:
{
lean_object* v___x_135_; 
lean_inc(v_n_134_);
v___x_135_ = lp_mathlib_finSuccEquiv_x27(v_n_134_, v_n_134_);
return v___x_135_;
}
}
static lean_object* _init_lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__0(void){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__1(void){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib_Function_Embedding_optionEmbeddingEquiv(lean_box(0), lean_box(0));
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingFinSucc___redArg(lean_object* v_n_138_){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_139_ = lp_mathlib_finSuccEquiv(v_n_138_);
v___x_140_ = lean_obj_once(&lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__0, &lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__0_once, _init_lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__0);
v___x_141_ = lp_mathlib_Equiv_embeddingCongr___redArg(v___x_139_, v___x_140_);
v___x_142_ = lean_obj_once(&lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__1, &lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__1_once, _init_lp_mathlib_Equiv_embeddingFinSucc___redArg___closed__1);
v___x_143_ = lp_mathlib_Equiv_trans___redArg(v___x_141_, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingFinSucc(lean_object* v_n_144_, lean_object* v_00_u03b9_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_mathlib_Equiv_embeddingFinSucc___redArg(v_n_144_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__0(lean_object* v_val_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_148_, 0, v_val_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__1(lean_object* v_val_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_150_, 0, v_val_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__2(lean_object* v_m_151_, lean_object* v___f_152_, lean_object* v___f_153_, lean_object* v_i_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = l_Fin_addCases___redArg(v_m_151_, v___f_152_, v___f_153_, v_i_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__2___boxed(lean_object* v_m_156_, lean_object* v___f_157_, lean_object* v___f_158_, lean_object* v_i_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_finSumFinEquiv___lam__2(v_m_156_, v___f_157_, v___f_158_, v_i_159_);
lean_dec(v_m_156_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__3(lean_object* v___y_161_){
_start:
{
lean_inc(v___y_161_);
return v___y_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv___lam__3___boxed(lean_object* v___y_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_finSumFinEquiv___lam__3(v___y_162_);
lean_dec(v___y_162_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumFinEquiv(lean_object* v_m_167_, lean_object* v_n_168_){
_start:
{
lean_object* v___f_169_; lean_object* v___f_170_; lean_object* v___f_171_; lean_object* v___f_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___f_169_ = ((lean_object*)(lp_mathlib_finSumFinEquiv___closed__0));
v___f_170_ = ((lean_object*)(lp_mathlib_finSumFinEquiv___closed__1));
lean_inc(v_m_167_);
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_finSumFinEquiv___lam__2___boxed), 4, 3);
lean_closure_set(v___f_171_, 0, v_m_167_);
lean_closure_set(v___f_171_, 1, v___f_169_);
lean_closure_set(v___f_171_, 2, v___f_170_);
v___f_172_ = ((lean_object*)(lp_mathlib_finSumFinEquiv___closed__2));
v___x_173_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_173_, 0, v_n_168_);
lean_closure_set(v___x_173_, 1, v_m_167_);
v___x_174_ = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(v___x_174_, 0, lean_box(0));
lean_closure_set(v___x_174_, 1, lean_box(0));
lean_closure_set(v___x_174_, 2, lean_box(0));
lean_closure_set(v___x_174_, 3, v___f_172_);
lean_closure_set(v___x_174_, 4, v___x_173_);
v___x_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
lean_ctor_set(v___x_175_, 1, v___f_171_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__0(lean_object* v_self_176_){
_start:
{
lean_inc(v_self_176_);
return v_self_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__0___boxed(lean_object* v_self_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_finSumNatEquiv___lam__0(v_self_177_);
lean_dec(v_self_177_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__1(lean_object* v_n_179_, lean_object* v_x_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lean_nat_add(v_n_179_, v_x_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__1___boxed(lean_object* v_n_182_, lean_object* v_x_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_finSumNatEquiv___lam__1(v_n_182_, v_x_183_);
lean_dec(v_x_183_);
lean_dec(v_n_182_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__2(lean_object* v_n_185_, lean_object* v_i_186_){
_start:
{
uint8_t v___x_187_; 
v___x_187_ = lean_nat_dec_lt(v_i_186_, v_n_185_);
if (v___x_187_ == 0)
{
lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_188_ = lean_nat_sub(v_i_186_, v_n_185_);
lean_dec(v_i_186_);
v___x_189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
return v___x_189_;
}
else
{
lean_object* v___x_190_; 
v___x_190_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_190_, 0, v_i_186_);
return v___x_190_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv___lam__2___boxed(lean_object* v_n_191_, lean_object* v_i_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_finSumNatEquiv___lam__2(v_n_191_, v_i_192_);
lean_dec(v_n_191_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finSumNatEquiv(lean_object* v_n_195_){
_start:
{
lean_object* v___f_196_; lean_object* v___f_197_; lean_object* v___f_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___f_196_ = ((lean_object*)(lp_mathlib_finSumNatEquiv___closed__0));
lean_inc(v_n_195_);
v___f_197_ = lean_alloc_closure((void*)(lp_mathlib_finSumNatEquiv___lam__1___boxed), 2, 1);
lean_closure_set(v___f_197_, 0, v_n_195_);
v___f_198_ = lean_alloc_closure((void*)(lp_mathlib_finSumNatEquiv___lam__2___boxed), 2, 1);
lean_closure_set(v___f_198_, 0, v_n_195_);
v___x_199_ = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(v___x_199_, 0, lean_box(0));
lean_closure_set(v___x_199_, 1, lean_box(0));
lean_closure_set(v___x_199_, 2, lean_box(0));
lean_closure_set(v___x_199_, 3, v___f_196_);
lean_closure_set(v___x_199_, 4, v___f_197_);
v___x_200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
lean_ctor_set(v___x_200_, 1, v___f_198_);
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib_finAddFlip___closed__0(void){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_Equiv_sumComm(lean_box(0), lean_box(0));
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finAddFlip(lean_object* v_m_202_, lean_object* v_n_203_){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
lean_inc(v_n_203_);
lean_inc(v_m_202_);
v___x_204_ = lp_mathlib_finSumFinEquiv(v_m_202_, v_n_203_);
v___x_205_ = lp_mathlib_Equiv_symm___redArg(v___x_204_);
v___x_206_ = lean_obj_once(&lp_mathlib_finAddFlip___closed__0, &lp_mathlib_finAddFlip___closed__0_once, _init_lp_mathlib_finAddFlip___closed__0);
v___x_207_ = lp_mathlib_Equiv_trans___redArg(v___x_205_, v___x_206_);
v___x_208_ = lp_mathlib_finSumFinEquiv(v_n_203_, v_m_202_);
v___x_209_ = lp_mathlib_Equiv_trans___redArg(v___x_207_, v___x_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__0(lean_object* v_n_210_, lean_object* v_x_211_){
_start:
{
lean_object* v_fst_212_; lean_object* v_snd_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v_fst_212_ = lean_ctor_get(v_x_211_, 0);
v_snd_213_ = lean_ctor_get(v_x_211_, 1);
v___x_214_ = lean_nat_mul(v_n_210_, v_fst_212_);
v___x_215_ = lean_nat_add(v_snd_213_, v___x_214_);
lean_dec(v___x_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__0___boxed(lean_object* v_n_216_, lean_object* v_x_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_finProdFinEquiv___redArg___lam__0(v_n_216_, v_x_217_);
lean_dec_ref(v_x_217_);
lean_dec(v_n_216_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__1(lean_object* v_n_219_, lean_object* v_x_220_){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_221_ = lean_nat_div(v_x_220_, v_n_219_);
v___x_222_ = lean_nat_mod(v_x_220_, v_n_219_);
v___x_223_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_223_, 0, v___x_221_);
lean_ctor_set(v___x_223_, 1, v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg___lam__1___boxed(lean_object* v_n_224_, lean_object* v_x_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib_finProdFinEquiv___redArg___lam__1(v_n_224_, v_x_225_);
lean_dec(v_x_225_);
lean_dec(v_n_224_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___redArg(lean_object* v_n_227_){
_start:
{
lean_object* v___f_228_; lean_object* v___f_229_; lean_object* v___x_230_; 
lean_inc(v_n_227_);
v___f_228_ = lean_alloc_closure((void*)(lp_mathlib_finProdFinEquiv___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_228_, 0, v_n_227_);
v___f_229_ = lean_alloc_closure((void*)(lp_mathlib_finProdFinEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_229_, 0, v_n_227_);
v___x_230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_230_, 0, v___f_228_);
lean_ctor_set(v___x_230_, 1, v___f_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv(lean_object* v_m_231_, lean_object* v_n_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_finProdFinEquiv___redArg(v_n_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finProdFinEquiv___boxed(lean_object* v_m_234_, lean_object* v_n_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_finProdFinEquiv(v_m_234_, v_n_235_);
lean_dec(v_m_234_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__0(lean_object* v_n_237_, lean_object* v_a_238_){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_239_ = lean_nat_div(v_a_238_, v_n_237_);
v___x_240_ = lean_nat_mod(v_a_238_, v_n_237_);
v___x_241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_239_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__0___boxed(lean_object* v_n_242_, lean_object* v_a_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_Nat_divModEquiv___redArg___lam__0(v_n_242_, v_a_243_);
lean_dec(v_a_243_);
lean_dec(v_n_242_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__1(lean_object* v_n_245_, lean_object* v_p_246_){
_start:
{
lean_object* v_fst_247_; lean_object* v_snd_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v_fst_247_ = lean_ctor_get(v_p_246_, 0);
v_snd_248_ = lean_ctor_get(v_p_246_, 1);
v___x_249_ = lean_nat_mul(v_fst_247_, v_n_245_);
v___x_250_ = lean_nat_add(v___x_249_, v_snd_248_);
lean_dec(v___x_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg___lam__1___boxed(lean_object* v_n_251_, lean_object* v_p_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib_Nat_divModEquiv___redArg___lam__1(v_n_251_, v_p_252_);
lean_dec_ref(v_p_252_);
lean_dec(v_n_251_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv___redArg(lean_object* v_n_254_){
_start:
{
lean_object* v___f_255_; lean_object* v___f_256_; lean_object* v___x_257_; 
lean_inc(v_n_254_);
v___f_255_ = lean_alloc_closure((void*)(lp_mathlib_Nat_divModEquiv___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_255_, 0, v_n_254_);
v___f_256_ = lean_alloc_closure((void*)(lp_mathlib_Nat_divModEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_256_, 0, v_n_254_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v___f_255_);
lean_ctor_set(v___x_257_, 1, v___f_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_divModEquiv(lean_object* v_n_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lp_mathlib_Nat_divModEquiv___redArg(v_n_258_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg___lam__0(lean_object* v_n_261_, lean_object* v_p_262_){
_start:
{
lean_object* v_fst_263_; lean_object* v_snd_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; 
v_fst_263_ = lean_ctor_get(v_p_262_, 0);
lean_inc(v_fst_263_);
v_snd_264_ = lean_ctor_get(v_p_262_, 1);
lean_inc(v_snd_264_);
lean_dec_ref(v_p_262_);
v___x_265_ = lean_nat_to_int(v_n_261_);
v___x_266_ = lean_int_mul(v_fst_263_, v___x_265_);
lean_dec(v___x_265_);
lean_dec(v_fst_263_);
v___x_267_ = lean_nat_to_int(v_snd_264_);
v___x_268_ = lean_int_add(v___x_266_, v___x_267_);
lean_dec(v___x_267_);
lean_dec(v___x_266_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg___lam__1(lean_object* v_n_269_, lean_object* v_a_270_){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
lean_inc(v_n_269_);
v___x_271_ = lean_nat_to_int(v_n_269_);
v___x_272_ = lean_int_ediv(v_a_270_, v___x_271_);
v___x_273_ = lp_mathlib_Int_natMod(v_a_270_, v___x_271_);
lean_dec(v___x_271_);
v___x_274_ = lean_nat_mod(v___x_273_, v_n_269_);
lean_dec(v_n_269_);
lean_dec(v___x_273_);
v___x_275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_272_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg___lam__1___boxed(lean_object* v_n_276_, lean_object* v_a_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_Int_divModEquiv___redArg___lam__1(v_n_276_, v_a_277_);
lean_dec(v_a_277_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv___redArg(lean_object* v_n_279_){
_start:
{
lean_object* v___f_280_; lean_object* v___f_281_; lean_object* v___x_282_; 
lean_inc(v_n_279_);
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_Int_divModEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_280_, 0, v_n_279_);
v___f_281_ = lean_alloc_closure((void*)(lp_mathlib_Int_divModEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_281_, 0, v_n_279_);
v___x_282_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_282_, 0, v___f_281_);
lean_ctor_set(v___x_282_, 1, v___f_280_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_divModEquiv(lean_object* v_n_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_mathlib_Int_divModEquiv___redArg(v_n_283_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv___lam__0(lean_object* v_i_286_){
_start:
{
lean_inc(v_i_286_);
return v_i_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv___lam__0___boxed(lean_object* v_i_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_Fin_castLEquiv___lam__0(v_i_287_);
lean_dec(v_i_287_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv(lean_object* v_n_292_, lean_object* v_m_293_, lean_object* v_h_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = ((lean_object*)(lp_mathlib_Fin_castLEquiv___closed__1));
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEquiv___boxed(lean_object* v_n_296_, lean_object* v_m_297_, lean_object* v_h_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Fin_castLEquiv(v_n_296_, v_m_297_, v_h_298_);
lean_dec(v_m_297_);
lean_dec(v_n_296_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__0(lean_object* v_m_300_, lean_object* v_fg_301_, lean_object* v___y_302_){
_start:
{
lean_object* v_fst_303_; lean_object* v_snd_304_; lean_object* v___x_305_; 
v_fst_303_ = lean_ctor_get(v_fg_301_, 0);
lean_inc(v_fst_303_);
v_snd_304_ = lean_ctor_get(v_fg_301_, 1);
lean_inc(v_snd_304_);
lean_dec_ref(v_fg_301_);
v___x_305_ = l_Fin_addCases___redArg(v_m_300_, v_fst_303_, v_snd_304_, v___y_302_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__0___boxed(lean_object* v_m_306_, lean_object* v_fg_307_, lean_object* v___y_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_Fin_appendEquiv___redArg___lam__0(v_m_306_, v_fg_307_, v___y_308_);
lean_dec(v_m_306_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__1(lean_object* v_f_310_, lean_object* v_i_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lean_apply_1(v_f_310_, v_i_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__2(lean_object* v_m_313_, lean_object* v_f_314_, lean_object* v_i_315_){
_start:
{
lean_object* v___x_316_; lean_object* v___x_317_; 
v___x_316_ = lean_nat_add(v_m_313_, v_i_315_);
v___x_317_ = lean_apply_1(v_f_314_, v___x_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__2___boxed(lean_object* v_m_318_, lean_object* v_f_319_, lean_object* v_i_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_Fin_appendEquiv___redArg___lam__2(v_m_318_, v_f_319_, v_i_320_);
lean_dec(v_i_320_);
lean_dec(v_m_318_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg___lam__3(lean_object* v_m_322_, lean_object* v_f_323_){
_start:
{
lean_object* v___f_324_; lean_object* v___f_325_; lean_object* v___x_326_; 
lean_inc(v_f_323_);
v___f_324_ = lean_alloc_closure((void*)(lp_mathlib_Fin_appendEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_324_, 0, v_f_323_);
v___f_325_ = lean_alloc_closure((void*)(lp_mathlib_Fin_appendEquiv___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_325_, 0, v_m_322_);
lean_closure_set(v___f_325_, 1, v_f_323_);
v___x_326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_326_, 0, v___f_324_);
lean_ctor_set(v___x_326_, 1, v___f_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___redArg(lean_object* v_m_327_){
_start:
{
lean_object* v___f_328_; lean_object* v___f_329_; lean_object* v___x_330_; 
lean_inc(v_m_327_);
v___f_328_ = lean_alloc_closure((void*)(lp_mathlib_Fin_appendEquiv___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_328_, 0, v_m_327_);
v___f_329_ = lean_alloc_closure((void*)(lp_mathlib_Fin_appendEquiv___redArg___lam__3), 2, 1);
lean_closure_set(v___f_329_, 0, v_m_327_);
v___x_330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_330_, 0, v___f_328_);
lean_ctor_set(v___x_330_, 1, v___f_329_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv(lean_object* v_00_u03b1_331_, lean_object* v_m_332_, lean_object* v_n_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lp_mathlib_Fin_appendEquiv___redArg(v_m_332_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_appendEquiv___boxed(lean_object* v_00_u03b1_335_, lean_object* v_m_336_, lean_object* v_n_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib_Fin_appendEquiv(v_00_u03b1_335_, v_m_336_, v_n_337_);
lean_dec(v_n_337_);
return v_res_338_;
}
}
static lean_object* _init_lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0___closed__0(void){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; 
v___x_339_ = lean_unsigned_to_nat(1u);
v___x_340_ = lean_unsigned_to_nat(0u);
v___x_341_ = lean_nat_mod(v___x_340_, v___x_339_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0(lean_object* v_f_342_){
_start:
{
lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_343_ = lean_obj_once(&lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0___closed__0, &lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0___closed__0_once, _init_lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__0___closed__0);
v___x_344_ = lean_apply_1(v_f_342_, v___x_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__1(lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_inc(v___y_345_);
return v___y_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__1___boxed(lean_object* v___y_347_, lean_object* v___y_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___lam__1(v___y_347_, v___y_348_);
lean_dec(v___y_348_);
lean_dec(v___y_347_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0(lean_object* v_00_u03b2_355_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = ((lean_object*)(lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0___closed__2));
return v___x_356_;
}
}
static lean_object* _init_lp_mathlib_Fin_succFunEquiv___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_mathlib_Equiv_piUnique___at___00Fin_succFunEquiv_spec__0(lean_box(0));
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv___redArg___lam__0(lean_object* v_x_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lean_obj_once(&lp_mathlib_Fin_succFunEquiv___redArg___lam__0___closed__0, &lp_mathlib_Fin_succFunEquiv___redArg___lam__0___closed__0_once, _init_lp_mathlib_Fin_succFunEquiv___redArg___lam__0___closed__0);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv___redArg___lam__0___boxed(lean_object* v_x_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_Fin_succFunEquiv___redArg___lam__0(v_x_360_);
lean_dec(v_x_360_);
return v_res_361_;
}
}
static lean_object* _init_lp_mathlib_Fin_succFunEquiv___redArg___closed__1(void){
_start:
{
lean_object* v___f_363_; lean_object* v___x_364_; 
v___f_363_ = ((lean_object*)(lp_mathlib_Fin_succFunEquiv___redArg___closed__0));
v___x_364_ = lp_mathlib_Equiv_prodCongrRight___redArg(v___f_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv___redArg(lean_object* v_n_365_){
_start:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_366_ = lp_mathlib_Fin_appendEquiv___redArg(v_n_365_);
v___x_367_ = lp_mathlib_Equiv_symm___redArg(v___x_366_);
v___x_368_ = lean_obj_once(&lp_mathlib_Fin_succFunEquiv___redArg___closed__1, &lp_mathlib_Fin_succFunEquiv___redArg___closed__1_once, _init_lp_mathlib_Fin_succFunEquiv___redArg___closed__1);
v___x_369_ = lp_mathlib_Equiv_trans___redArg(v___x_367_, v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succFunEquiv(lean_object* v_00_u03b1_370_, lean_object* v_n_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_Fin_succFunEquiv___redArg(v_n_371_);
return v___x_372_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Option(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Fin_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Fin_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Fin_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
