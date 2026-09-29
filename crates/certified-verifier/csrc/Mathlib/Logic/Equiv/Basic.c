// Lean compiler output
// Module: Mathlib.Logic.Equiv.Basic
// Imports: public import Init public meta import Init public import Mathlib.Logic.Equiv.Option public import Mathlib.Logic.Equiv.Sum public import Mathlib.Logic.Function.Conjugate public import Mathlib.Tactic.Lift public import Mathlib.Data.Int.Notation
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
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_Sigma_uncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_curry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaAssoc(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_optionIsSomeEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrLeft_x27___redArg(lean_object*);
lean_object* l_Int_negSucc___boxed(lean_object*);
lean_object* l_Int_ofNat___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_sigmaEquivProd(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumCompl___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_uniqueSigma___redArg(lean_object*);
lean_object* lp_mathlib_Pi_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_unique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piOptionEquivProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piOptionEquivProd___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piOptionEquivProd___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piOptionEquivProd___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_piOptionEquivProd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piOptionEquivProd___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piOptionEquivProd___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piOptionEquivProd___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_piOptionEquivProd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piOptionEquivProd___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piOptionEquivProd___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_piOptionEquivProd___closed__2 = (const lean_object*)&lp_mathlib_Equiv_piOptionEquivProd___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeCongr___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypePreimage___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypePreimage___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypePreimage___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypePreimage___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piComm___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piComm___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piComm___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piComm___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_piComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piComm___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piComm___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_piComm___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piComm___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piComm(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piCurry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sigma_curry, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_piCurry___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piCurry___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_piCurry___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sigma_uncurry, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_piCurry___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piCurry___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_piCurry___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piCurry___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piCurry___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_piCurry___closed__2 = (const lean_object*)&lp_mathlib_Equiv_piCurry___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCurry(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofFiberEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofFiberEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__3___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaNatSucc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaNatSucc___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaNatSucc___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaNatSucc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaNatSucc___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaNatSucc___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaNatSucc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaNatSucc___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaNatSucc___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaNatSucc___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaNatSucc___lam__3___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaNatSucc___closed__3 = (const lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__3_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaNatSucc___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sigma_map, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__2_value),((lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__3_value)} };
static const lean_object* lp_mathlib_Equiv_sigmaNatSucc___closed__4 = (const lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__4_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaNatSucc___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__1_value),((lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__4_value)} };
static const lean_object* lp_mathlib_Equiv_sigmaNatSucc___closed__5 = (const lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__5_value;
static const lean_ctor_object lp_mathlib_Equiv_sigmaNatSucc___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__5_value)}};
static const lean_object* lp_mathlib_Equiv_sigmaNatSucc___closed__6 = (const lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc(lean_object*);
static const lean_ctor_object lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_natEquivNatSumPUnit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___closed__0 = (const lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_natEquivNatSumPUnit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_natEquivNatSumPUnit___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___closed__1 = (const lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_natEquivNatSumPUnit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_sigmaNatSucc___closed__2_value),((lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__1_value)} };
static const lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___closed__2 = (const lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__2_value;
static const lean_ctor_object lp_mathlib_Equiv_natEquivNatSumPUnit___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__0_value),((lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__2_value)}};
static const lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___closed__3 = (const lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit = (const lean_object*)&lp_mathlib_Equiv_natEquivNatSumPUnit___closed__3_value;
static lean_once_cell_t lp_mathlib_Equiv_natSumPUnitEquivNat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_natSumPUnitEquivNat___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natSumPUnitEquivNat;
static lean_once_cell_t lp_mathlib_Equiv_intEquivNatSumNat___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_intEquivNatSumNat___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_intEquivNatSumNat___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_intEquivNatSumNat___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_intEquivNatSumNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_intEquivNatSumNat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_intEquivNatSumNat___closed__0 = (const lean_object*)&lp_mathlib_Equiv_intEquivNatSumNat___closed__0_value;
static lean_once_cell_t lp_mathlib_Equiv_intEquivNatSumNat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_intEquivNatSumNat___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_intEquivNatSumNat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_intEquivNatSumNat___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_intEquivNatSumNat;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_subtypeEquivRight___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeEquivRight___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_subtypeEquivRight___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeEquivRight___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeUnivEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeUnivEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeUnivEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeUnivEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeUnivEquiv___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeUnivEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeUnivEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeUnivEquiv___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeUnivEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeUnivEquiv___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeUnivEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeUnivEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquiv___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeSigmaEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeSigmaEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeSigmaEquiv___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeSigmaEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquiv___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeSigmaEquiv___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquiv(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__3;
static lean_once_cell_t lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaOptionEquivOfSome(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piEquivSubtypeSigma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piEquivSubtypeSigma___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piEquivSubtypeSigma___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_piEquivSubtypeSigma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piEquivSubtypeSigma___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piEquivSubtypeSigma___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_piEquivSubtypeSigma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piEquivSubtypeSigma___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piEquivSubtypeSigma___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma___closed__2 = (const lean_object*)&lp_mathlib_Equiv_piEquivSubtypeSigma___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePiEquivPi___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePiEquivPi___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypePiEquivPi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypePiEquivPi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypePiEquivPi___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypePiEquivPi___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_subtypePiEquivPi___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypePiEquivPi___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypePiEquivPi___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypePiEquivPi___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypePiEquivPi___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypePiEquivPi___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypePiEquivPi___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_subtypePiEquivPi___closed__2 = (const lean_object*)&lp_mathlib_Equiv_subtypePiEquivPi___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePiEquivPi(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaAssocProd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaAssocProd___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_sigmaAssocProd___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaAssocProd___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_sigmaAssocProd___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaAssocProd___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_sigmaAssocProd___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaAssocProd___closed__3;
static lean_once_cell_t lp_mathlib_Equiv_sigmaAssocProd___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaAssocProd___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssocProd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaSubtype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaSubtype___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaSubtype___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtypeEq___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtypeEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_subtypeEquivCodomain___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivCodomain___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_subtypeEquivCodomain___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_subtypeEquivCodomain___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivCodomain___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivCodomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomain___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swapCore___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swapCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setValue___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setValue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Involutive_toPerm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Involutive_toPerm(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSigmaFiber___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSigmaFiber(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piCongrFiberwise___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSet___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piCongrSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piCongrSet___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piCongrSet___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piCongrSet___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_piCongrSet___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_piCongrSet___closed__0_value),((lean_object*)&lp_mathlib_Equiv_piCongrSet___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_piCongrSet___closed__1 = (const lean_object*)&lp_mathlib_Equiv_piCongrSet___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equivOfSubsingletonOfSubsingleton___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_equivOfSubsingletonOfSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_uniqueUniqueEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_uniqueUniqueEquiv___elam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_uniqueUniqueEquiv___closed__0 = (const lean_object*)&lp_mathlib_uniqueUniqueEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_uniqueUniqueEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_uniqueUniqueEquiv___elam__1___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_uniqueUniqueEquiv___closed__1 = (const lean_object*)&lp_mathlib_uniqueUniqueEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_uniqueUniqueEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_uniqueUniqueEquiv___closed__0_value),((lean_object*)&lp_mathlib_uniqueUniqueEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_uniqueUniqueEquiv___closed__2 = (const lean_object*)&lp_mathlib_uniqueUniqueEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique___elam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique___elam__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd___lam__0(lean_object* v_f_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3_, 0, v_a_2_);
v___x_4_ = lean_apply_1(v_f_1_, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd___lam__1(lean_object* v_f_5_){
_start:
{
lean_object* v___f_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
lean_inc(v_f_5_);
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piOptionEquivProd___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_f_5_);
v___x_7_ = lean_box(0);
v___x_8_ = lean_apply_1(v_f_5_, v___x_7_);
v___x_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v___f_6_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd___lam__2(lean_object* v_x_10_, lean_object* v_a_11_){
_start:
{
if (lean_obj_tag(v_a_11_) == 0)
{
lean_object* v_fst_12_; 
v_fst_12_ = lean_ctor_get(v_x_10_, 0);
lean_inc(v_fst_12_);
lean_dec_ref(v_x_10_);
return v_fst_12_;
}
else
{
lean_object* v_a_13_; lean_object* v_snd_14_; lean_object* v___x_15_; 
v_a_13_ = lean_ctor_get(v_a_11_, 0);
lean_inc(v_a_13_);
lean_dec_ref_known(v_a_11_, 1);
v_snd_14_ = lean_ctor_get(v_x_10_, 1);
lean_inc(v_snd_14_);
lean_dec_ref(v_x_10_);
v___x_15_ = lean_apply_1(v_snd_14_, v_a_13_);
return v___x_15_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piOptionEquivProd(lean_object* v_00_u03b1_21_, lean_object* v_00_u03b2_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = ((lean_object*)(lp_mathlib_Equiv_piOptionEquivProd___closed__2));
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeCongr___redArg(lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_e_26_, lean_object* v_f_27_){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_28_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_24_);
v___x_29_ = lp_mathlib_Equiv_symm___redArg(v___x_28_);
v___x_30_ = lp_mathlib_Equiv_sumCongr___redArg(v_e_26_, v_f_27_);
v___x_31_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_25_);
v___x_32_ = lp_mathlib_Equiv_trans___redArg(v___x_30_, v___x_31_);
v___x_33_ = lp_mathlib_Equiv_trans___redArg(v___x_29_, v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeCongr(lean_object* v_00_u03b1_34_, lean_object* v_p_35_, lean_object* v_q_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_e_39_, lean_object* v_f_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Equiv_subtypeCongr___redArg(v_inst_37_, v_inst_38_, v_e_39_, v_f_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongr___redArg(lean_object* v_inst_42_, lean_object* v_ep_43_, lean_object* v_en_44_){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v_toFun_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_45_ = lp_mathlib_Equiv_sumCompl___redArg(v_inst_42_);
lean_inc_ref(v___x_45_);
v___x_46_ = lp_mathlib_Equiv_equivCongr___redArg(v___x_45_, v___x_45_);
v_toFun_47_ = lean_ctor_get(v___x_46_, 0);
lean_inc(v_toFun_47_);
lean_dec_ref(v___x_46_);
v___x_48_ = lp_mathlib_Equiv_sumCongr___redArg(v_ep_43_, v_en_44_);
v___x_49_ = lean_apply_1(v_toFun_47_, v___x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_subtypeCongr(lean_object* v_00_u03b5_50_, lean_object* v_p_51_, lean_object* v_inst_52_, lean_object* v_ep_53_, lean_object* v_en_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Equiv_Perm_subtypeCongr___redArg(v_inst_52_, v_ep_53_, v_en_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage___redArg___lam__0(lean_object* v_x_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_apply_1(v_x_56_, v_a_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage___redArg___lam__1(lean_object* v_inst_59_, lean_object* v_x_u2080_60_, lean_object* v_x_61_, lean_object* v___y_62_){
_start:
{
lean_object* v___x_63_; uint8_t v___x_64_; 
lean_inc(v___y_62_);
v___x_63_ = lean_apply_1(v_inst_59_, v___y_62_);
v___x_64_ = lean_unbox(v___x_63_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; 
lean_dec(v_x_u2080_60_);
v___x_65_ = lean_apply_1(v_x_61_, v___y_62_);
return v___x_65_;
}
else
{
lean_object* v___x_66_; 
lean_dec(v_x_61_);
v___x_66_ = lean_apply_1(v_x_u2080_60_, v___y_62_);
return v___x_66_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage___redArg(lean_object* v_inst_68_, lean_object* v_x_u2080_69_){
_start:
{
lean_object* v___f_70_; lean_object* v___f_71_; lean_object* v___x_72_; 
v___f_70_ = ((lean_object*)(lp_mathlib_Equiv_subtypePreimage___redArg___closed__0));
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subtypePreimage___redArg___lam__1), 4, 2);
lean_closure_set(v___f_71_, 0, v_inst_68_);
lean_closure_set(v___f_71_, 1, v_x_u2080_69_);
v___x_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_72_, 0, v___f_70_);
lean_ctor_set(v___x_72_, 1, v___f_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePreimage(lean_object* v_00_u03b1_73_, lean_object* v_00_u03b2_74_, lean_object* v_p_75_, lean_object* v_inst_76_, lean_object* v_x_u2080_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Equiv_subtypePreimage___redArg(v_inst_76_, v_x_u2080_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight___redArg___lam__0(lean_object* v_F_79_, lean_object* v_a_80_, lean_object* v___y_81_){
_start:
{
lean_object* v___x_82_; lean_object* v_toFun_83_; lean_object* v___x_84_; 
v___x_82_ = lean_apply_1(v_F_79_, v_a_80_);
v_toFun_83_ = lean_ctor_get(v___x_82_, 0);
lean_inc(v_toFun_83_);
lean_dec_ref(v___x_82_);
v___x_84_ = lean_apply_1(v_toFun_83_, v___y_81_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight___redArg___lam__1(lean_object* v_F_85_, lean_object* v_a_86_, lean_object* v___y_87_){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v_toFun_90_; lean_object* v___x_91_; 
v___x_88_ = lean_apply_1(v_F_85_, v_a_86_);
v___x_89_ = lp_mathlib_Equiv_symm___redArg(v___x_88_);
v_toFun_90_ = lean_ctor_get(v___x_89_, 0);
lean_inc(v_toFun_90_);
lean_dec_ref(v___x_89_);
v___x_91_ = lean_apply_1(v_toFun_90_, v___y_87_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight___redArg(lean_object* v_F_92_){
_start:
{
lean_object* v___f_93_; lean_object* v___f_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
lean_inc_ref(v_F_92_);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_93_, 0, v_F_92_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piCongrRight___redArg___lam__1), 3, 1);
lean_closure_set(v___f_94_, 0, v_F_92_);
v___x_95_ = lean_alloc_closure((void*)(lp_mathlib_Pi_map), 6, 4);
lean_closure_set(v___x_95_, 0, lean_box(0));
lean_closure_set(v___x_95_, 1, lean_box(0));
lean_closure_set(v___x_95_, 2, lean_box(0));
lean_closure_set(v___x_95_, 3, v___f_93_);
v___x_96_ = lean_alloc_closure((void*)(lp_mathlib_Pi_map), 6, 4);
lean_closure_set(v___x_96_, 0, lean_box(0));
lean_closure_set(v___x_96_, 1, lean_box(0));
lean_closure_set(v___x_96_, 2, lean_box(0));
lean_closure_set(v___x_96_, 3, v___f_94_);
v___x_97_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_95_);
lean_ctor_set(v___x_97_, 1, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrRight(lean_object* v_00_u03b1_98_, lean_object* v_00_u03b2_u2081_99_, lean_object* v_00_u03b2_u2082_100_, lean_object* v_F_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Equiv_piCongrRight___redArg(v_F_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piComm___lam__0(lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lean_apply_2(v___y_103_, v___y_105_, v___y_104_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piComm(lean_object* v_00_u03b1_110_, lean_object* v_00_u03b2_111_, lean_object* v_00_u03c6_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = ((lean_object*)(lp_mathlib_Equiv_piComm___closed__1));
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCurry(lean_object* v_00_u03b1_119_, lean_object* v_00_u03b2_120_, lean_object* v_00_u03b3_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = ((lean_object*)(lp_mathlib_Equiv_piCurry___closed__2));
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofFiberEquiv___redArg(lean_object* v_f_123_, lean_object* v_g_124_, lean_object* v_e_125_){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_126_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_f_123_);
v___x_127_ = lp_mathlib_Equiv_symm___redArg(v___x_126_);
v___x_128_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v_e_125_);
v___x_129_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_g_124_);
v___x_130_ = lp_mathlib_Equiv_trans___redArg(v___x_128_, v___x_129_);
v___x_131_ = lp_mathlib_Equiv_trans___redArg(v___x_127_, v___x_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofFiberEquiv(lean_object* v_00_u03b1_132_, lean_object* v_00_u03b2_133_, lean_object* v_00_u03b3_134_, lean_object* v_f_135_, lean_object* v_g_136_, lean_object* v_e_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_Equiv_ofFiberEquiv___redArg(v_f_135_, v_g_136_, v_e_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__0(lean_object* v_x_139_){
_start:
{
lean_object* v_n_140_; lean_object* v_a_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_154_; 
v_n_140_ = lean_ctor_get(v_x_139_, 0);
v_a_141_ = lean_ctor_get(v_x_139_, 1);
v_isSharedCheck_154_ = !lean_is_exclusive(v_x_139_);
if (v_isSharedCheck_154_ == 0)
{
v___x_143_ = v_x_139_;
v_isShared_144_ = v_isSharedCheck_154_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_a_141_);
lean_inc(v_n_140_);
lean_dec(v_x_139_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_154_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v_zero_145_; uint8_t v_isZero_146_; 
v_zero_145_ = lean_unsigned_to_nat(0u);
v_isZero_146_ = lean_nat_dec_eq(v_n_140_, v_zero_145_);
if (v_isZero_146_ == 1)
{
lean_object* v___x_147_; 
lean_del_object(v___x_143_);
lean_dec(v_n_140_);
v___x_147_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_147_, 0, v_a_141_);
return v___x_147_;
}
else
{
lean_object* v_one_148_; lean_object* v_n_149_; lean_object* v___x_151_; 
v_one_148_ = lean_unsigned_to_nat(1u);
v_n_149_ = lean_nat_sub(v_n_140_, v_one_148_);
lean_dec(v_n_140_);
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 0, v_n_149_);
v___x_151_ = v___x_143_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_n_149_);
lean_ctor_set(v_reuseFailAlloc_153_, 1, v_a_141_);
v___x_151_ = v_reuseFailAlloc_153_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
lean_object* v___x_152_; 
v___x_152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
return v___x_152_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__1(lean_object* v_snd_155_){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = lean_unsigned_to_nat(0u);
v___x_157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v_snd_155_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__2(lean_object* v_n_158_){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_159_ = lean_unsigned_to_nat(1u);
v___x_160_ = lean_nat_add(v_n_158_, v___x_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__2___boxed(lean_object* v_n_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_Equiv_sigmaNatSucc___lam__2(v_n_161_);
lean_dec(v_n_161_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__3(lean_object* v_x_163_, lean_object* v___y_164_){
_start:
{
lean_inc(v___y_164_);
return v___y_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc___lam__3___boxed(lean_object* v_x_165_, lean_object* v___y_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_Equiv_sigmaNatSucc___lam__3(v_x_165_, v___y_166_);
lean_dec(v___y_166_);
lean_dec(v_x_165_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaNatSucc(lean_object* v_f_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = ((lean_object*)(lp_mathlib_Equiv_sigmaNatSucc___closed__6));
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0(lean_object* v_n_185_){
_start:
{
lean_object* v_zero_186_; uint8_t v_isZero_187_; 
v_zero_186_ = lean_unsigned_to_nat(0u);
v_isZero_187_ = lean_nat_dec_eq(v_n_185_, v_zero_186_);
if (v_isZero_187_ == 1)
{
lean_object* v___x_188_; 
v___x_188_ = ((lean_object*)(lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0___closed__0));
return v___x_188_;
}
else
{
lean_object* v_one_189_; lean_object* v_val_190_; lean_object* v___x_191_; 
v_one_189_ = lean_unsigned_to_nat(1u);
v_val_190_ = lean_nat_sub(v_n_185_, v_one_189_);
v___x_191_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_191_, 0, v_val_190_);
return v___x_191_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0___boxed(lean_object* v_n_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_Equiv_natEquivNatSumPUnit___lam__0(v_n_192_);
lean_dec(v_n_192_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_natEquivNatSumPUnit___lam__2(lean_object* v_x_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_unsigned_to_nat(0u);
return v___x_195_;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumPUnitEquivNat___closed__0(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = ((lean_object*)(lp_mathlib_Equiv_natEquivNatSumPUnit));
v___x_206_ = lp_mathlib_Equiv_symm___redArg(v___x_205_);
return v___x_206_;
}
}
static lean_object* _init_lp_mathlib_Equiv_natSumPUnitEquivNat(void){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lean_obj_once(&lp_mathlib_Equiv_natSumPUnitEquivNat___closed__0, &lp_mathlib_Equiv_natSumPUnitEquivNat___closed__0_once, _init_lp_mathlib_Equiv_natSumPUnitEquivNat___closed__0);
return v___x_207_;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNatSumNat___lam__0___closed__0(void){
_start:
{
lean_object* v_natZero_208_; lean_object* v_intZero_209_; 
v_natZero_208_ = lean_unsigned_to_nat(0u);
v_intZero_209_ = lean_nat_to_int(v_natZero_208_);
return v_intZero_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_intEquivNatSumNat___lam__0(lean_object* v_z_210_){
_start:
{
lean_object* v_intZero_211_; uint8_t v_isNeg_212_; 
v_intZero_211_ = lean_obj_once(&lp_mathlib_Equiv_intEquivNatSumNat___lam__0___closed__0, &lp_mathlib_Equiv_intEquivNatSumNat___lam__0___closed__0_once, _init_lp_mathlib_Equiv_intEquivNatSumNat___lam__0___closed__0);
v_isNeg_212_ = lean_int_dec_lt(v_z_210_, v_intZero_211_);
if (v_isNeg_212_ == 0)
{
lean_object* v_val_213_; lean_object* v___x_214_; 
v_val_213_ = lean_nat_abs(v_z_210_);
v___x_214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_214_, 0, v_val_213_);
return v___x_214_;
}
else
{
lean_object* v_abs_215_; lean_object* v_one_216_; lean_object* v_val_217_; lean_object* v___x_218_; 
v_abs_215_ = lean_nat_abs(v_z_210_);
v_one_216_ = lean_unsigned_to_nat(1u);
v_val_217_ = lean_nat_sub(v_abs_215_, v_one_216_);
lean_dec(v_abs_215_);
v___x_218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_218_, 0, v_val_217_);
return v___x_218_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_intEquivNatSumNat___lam__0___boxed(lean_object* v_z_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_Equiv_intEquivNatSumNat___lam__0(v_z_219_);
lean_dec(v_z_219_);
return v_res_220_;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNatSumNat___closed__1(void){
_start:
{
lean_object* v___f_222_; lean_object* v___f_223_; lean_object* v___x_224_; 
v___f_222_ = lean_alloc_closure((void*)(l_Int_negSucc___boxed), 1, 0);
v___f_223_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
v___x_224_ = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(v___x_224_, 0, lean_box(0));
lean_closure_set(v___x_224_, 1, lean_box(0));
lean_closure_set(v___x_224_, 2, lean_box(0));
lean_closure_set(v___x_224_, 3, v___f_223_);
lean_closure_set(v___x_224_, 4, v___f_222_);
return v___x_224_;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNatSumNat___closed__2(void){
_start:
{
lean_object* v___x_225_; lean_object* v___f_226_; lean_object* v___x_227_; 
v___x_225_ = lean_obj_once(&lp_mathlib_Equiv_intEquivNatSumNat___closed__1, &lp_mathlib_Equiv_intEquivNatSumNat___closed__1_once, _init_lp_mathlib_Equiv_intEquivNatSumNat___closed__1);
v___f_226_ = ((lean_object*)(lp_mathlib_Equiv_intEquivNatSumNat___closed__0));
v___x_227_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_227_, 0, v___f_226_);
lean_ctor_set(v___x_227_, 1, v___x_225_);
return v___x_227_;
}
}
static lean_object* _init_lp_mathlib_Equiv_intEquivNatSumNat(void){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lean_obj_once(&lp_mathlib_Equiv_intEquivNatSumNat___closed__2, &lp_mathlib_Equiv_intEquivNatSumNat___closed__2_once, _init_lp_mathlib_Equiv_intEquivNatSumNat___closed__2);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__0___redArg(lean_object* v_e_229_, lean_object* v_h_230_){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v_toFun_233_; lean_object* v___x_234_; 
v___x_231_ = lp_mathlib_Equiv_symm___redArg(v_e_229_);
v___x_232_ = lp_mathlib_Equiv_symm___redArg(v___x_231_);
v_toFun_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_toFun_233_);
lean_dec_ref(v___x_232_);
v___x_234_ = lean_apply_1(v_toFun_233_, v_h_230_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__0(lean_object* v_00_u03b1_235_, lean_object* v_00_u03b2_236_, lean_object* v_e_237_, lean_object* v_h_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_Equiv_uniqueCongr___elam__0___redArg(v_e_237_, v_h_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__1___redArg(lean_object* v_e_240_, lean_object* v_h_241_){
_start:
{
lean_object* v___x_242_; lean_object* v_toFun_243_; lean_object* v___x_244_; 
v___x_242_ = lp_mathlib_Equiv_symm___redArg(v_e_240_);
v_toFun_243_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_toFun_243_);
lean_dec_ref(v___x_242_);
v___x_244_ = lean_apply_1(v_toFun_243_, v_h_241_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___elam__1(lean_object* v_00_u03b2_245_, lean_object* v_00_u03b1_246_, lean_object* v_e_247_, lean_object* v_h_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_Equiv_uniqueCongr___elam__1___redArg(v_e_247_, v_h_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr___redArg(lean_object* v_e_250_){
_start:
{
lean_object* v___f_251_; lean_object* v___f_252_; lean_object* v___x_253_; 
lean_inc_ref(v_e_250_);
v___f_251_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_uniqueCongr___elam__0), 4, 3);
lean_closure_set(v___f_251_, 0, lean_box(0));
lean_closure_set(v___f_251_, 1, lean_box(0));
lean_closure_set(v___f_251_, 2, v_e_250_);
v___f_252_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_uniqueCongr___elam__1), 4, 3);
lean_closure_set(v___f_252_, 0, lean_box(0));
lean_closure_set(v___f_252_, 1, lean_box(0));
lean_closure_set(v___f_252_, 2, v_e_250_);
v___x_253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_253_, 0, v___f_251_);
lean_ctor_set(v___x_253_, 1, v___f_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueCongr(lean_object* v_00_u03b1_254_, lean_object* v_00_u03b2_255_, lean_object* v_e_256_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lp_mathlib_Equiv_uniqueCongr___redArg(v_e_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg___lam__0(lean_object* v_e_258_, lean_object* v_a_259_){
_start:
{
lean_object* v_toFun_260_; lean_object* v___x_261_; 
v_toFun_260_ = lean_ctor_get(v_e_258_, 0);
lean_inc(v_toFun_260_);
lean_dec_ref(v_e_258_);
v___x_261_ = lean_apply_1(v_toFun_260_, v_a_259_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg___lam__1(lean_object* v_e_262_, lean_object* v_b_263_){
_start:
{
lean_object* v___x_264_; lean_object* v_toFun_265_; lean_object* v___x_266_; 
v___x_264_ = lp_mathlib_Equiv_symm___redArg(v_e_262_);
v_toFun_265_ = lean_ctor_get(v___x_264_, 0);
lean_inc(v_toFun_265_);
lean_dec_ref(v___x_264_);
v___x_266_ = lean_apply_1(v_toFun_265_, v_b_263_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg(lean_object* v_e_267_){
_start:
{
lean_object* v___f_268_; lean_object* v___f_269_; lean_object* v___x_270_; 
lean_inc_ref(v_e_267_);
v___f_268_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subtypeEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_268_, 0, v_e_267_);
v___f_269_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subtypeEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_269_, 0, v_e_267_);
v___x_270_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_270_, 0, v___f_268_);
lean_ctor_set(v___x_270_, 1, v___f_269_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquiv(lean_object* v_00_u03b1_271_, lean_object* v_00_u03b2_272_, lean_object* v_p_273_, lean_object* v_q_274_, lean_object* v_e_275_, lean_object* v_h_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v_e_275_);
return v___x_277_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeEquivRight___closed__0(void){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_278_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeEquivRight___closed__1(void){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_279_ = lean_obj_once(&lp_mathlib_Equiv_subtypeEquivRight___closed__0, &lp_mathlib_Equiv_subtypeEquivRight___closed__0_once, _init_lp_mathlib_Equiv_subtypeEquivRight___closed__0);
v___x_280_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v___x_279_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivRight(lean_object* v_00_u03b1_281_, lean_object* v_p_282_, lean_object* v_q_283_, lean_object* v_e_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lean_obj_once(&lp_mathlib_Equiv_subtypeEquivRight___closed__1, &lp_mathlib_Equiv_subtypeEquivRight___closed__1_once, _init_lp_mathlib_Equiv_subtypeEquivRight___closed__1);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype___redArg(lean_object* v_e_286_){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v_e_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype(lean_object* v_00_u03b1_288_, lean_object* v_00_u03b2_289_, lean_object* v_p_290_, lean_object* v_e_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v_e_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype_x27___redArg(lean_object* v_e_293_){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_294_ = lp_mathlib_Equiv_symm___redArg(v_e_293_);
v___x_295_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v___x_294_);
v___x_296_ = lp_mathlib_Equiv_symm___redArg(v___x_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivOfSubtype_x27(lean_object* v_00_u03b1_297_, lean_object* v_00_u03b2_298_, lean_object* v_p_299_, lean_object* v_e_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_mathlib_Equiv_subtypeEquivOfSubtype_x27___redArg(v_e_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object* v_00_u03b1_302_, lean_object* v_p_303_, lean_object* v_q_304_, lean_object* v_h_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lean_obj_once(&lp_mathlib_Equiv_subtypeEquivRight___closed__1, &lp_mathlib_Equiv_subtypeEquivRight___closed__1_once, _init_lp_mathlib_Equiv_subtypeEquivRight___closed__1);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___lam__0(lean_object* v_a_307_){
_start:
{
lean_inc(v_a_307_);
return v_a_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___lam__0___boxed(lean_object* v_a_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___lam__0(v_a_308_);
lean_dec(v_a_308_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists(lean_object* v_00_u03b1_313_, lean_object* v_p_314_, lean_object* v_q_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = ((lean_object*)(lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__1));
return v___x_316_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__0(void){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists(lean_box(0), lean_box(0), lean_box(0));
return v___x_317_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1(void){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Equiv_subtypeEquivRight(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_318_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2(void){
_start:
{
lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_319_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1);
v___x_320_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__0, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__0_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__0);
v___x_321_ = lp_mathlib_Equiv_trans___redArg(v___x_320_, v___x_319_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter(lean_object* v_00_u03b1_322_, lean_object* v_p_323_, lean_object* v_q_324_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2);
return v___x_325_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__0(void){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter(lean_box(0), lean_box(0), lean_box(0));
return v___x_326_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__1(void){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_327_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1);
v___x_328_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__0, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__0_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__0);
v___x_329_ = lp_mathlib_Equiv_trans___redArg(v___x_328_, v___x_327_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSubtypeEquivSubtype(lean_object* v_00_u03b1_330_, lean_object* v_p_331_, lean_object* v_q_332_, lean_object* v_h_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__1, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__1_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtype___closed__1);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeUnivEquiv___lam__0(lean_object* v_x_335_){
_start:
{
lean_inc(v_x_335_);
return v_x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeUnivEquiv___lam__0___boxed(lean_object* v_x_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Equiv_subtypeUnivEquiv___lam__0(v_x_336_);
lean_dec(v_x_336_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeUnivEquiv(lean_object* v_00_u03b1_341_, lean_object* v_p_342_, lean_object* v_h_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = ((lean_object*)(lp_mathlib_Equiv_subtypeUnivEquiv___closed__1));
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquiv___lam__0(lean_object* v_x_345_){
_start:
{
lean_object* v_fst_346_; lean_object* v_snd_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_354_; 
v_fst_346_ = lean_ctor_get(v_x_345_, 0);
v_snd_347_ = lean_ctor_get(v_x_345_, 1);
v_isSharedCheck_354_ = !lean_is_exclusive(v_x_345_);
if (v_isSharedCheck_354_ == 0)
{
v___x_349_ = v_x_345_;
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_snd_347_);
lean_inc(v_fst_346_);
lean_dec(v_x_345_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_352_; 
if (v_isShared_350_ == 0)
{
v___x_352_ = v___x_349_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v_fst_346_);
lean_ctor_set(v_reuseFailAlloc_353_, 1, v_snd_347_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquiv(lean_object* v_00_u03b1_358_, lean_object* v_p_359_, lean_object* v_q_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = ((lean_object*)(lp_mathlib_Equiv_subtypeSigmaEquiv___closed__1));
return v___x_361_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__0(void){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = lp_mathlib_Equiv_subtypeSigmaEquiv(lean_box(0), lean_box(0), lean_box(0));
return v___x_362_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__1(void){
_start:
{
lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_363_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__0, &lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__0_once, _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__0);
v___x_364_ = lp_mathlib_Equiv_symm___redArg(v___x_363_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__2(void){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lp_mathlib_Equiv_subtypeUnivEquiv(lean_box(0), lean_box(0), lean_box(0));
return v___x_365_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__3(void){
_start:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_366_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__2, &lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__2_once, _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__2);
v___x_367_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__1, &lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__1_once, _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__1);
v___x_368_ = lp_mathlib_Equiv_trans___redArg(v___x_367_, v___x_366_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset(lean_object* v_00_u03b1_369_, lean_object* v_p_370_, lean_object* v_q_371_, lean_object* v_h_372_){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__3, &lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__3_once, _init_lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset___closed__3);
return v___x_373_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg___closed__0(void){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg(lean_object* v_f_375_){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_376_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg___closed__0, &lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg___closed__0_once, _init_lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg___closed__0);
v___x_377_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_f_375_);
v___x_378_ = lp_mathlib_Equiv_trans___redArg(v___x_376_, v___x_377_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquiv(lean_object* v_00_u03b1_379_, lean_object* v_00_u03b2_380_, lean_object* v_f_381_, lean_object* v_p_382_, lean_object* v_h_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib_Equiv_sigmaSubtypeFiberEquiv___redArg(v_f_381_);
return v___x_384_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_385_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__2);
v___x_386_ = lp_mathlib_Equiv_symm___redArg(v___x_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0(lean_object* v_y_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___closed__0, &lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___closed__0_once, _init_lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___closed__0);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0___boxed(lean_object* v_y_389_){
_start:
{
lean_object* v_res_390_; 
v_res_390_ = lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__0(v_y_389_);
lean_dec(v_y_389_);
return v_res_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__1(lean_object* v_f_391_, lean_object* v_x_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lean_apply_1(v_f_391_, v_x_392_);
return v___x_393_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__1(void){
_start:
{
lean_object* v___f_395_; lean_object* v___x_396_; 
v___f_395_ = ((lean_object*)(lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__0));
v___x_396_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v___f_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg(lean_object* v_f_397_){
_start:
{
lean_object* v___f_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; 
v___f_398_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_398_, 0, v_f_397_);
v___x_399_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__1, &lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__1_once, _init_lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg___closed__1);
v___x_400_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v___f_398_);
v___x_401_ = lp_mathlib_Equiv_trans___redArg(v___x_399_, v___x_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype(lean_object* v_00_u03b1_402_, lean_object* v_00_u03b2_403_, lean_object* v_f_404_, lean_object* v_p_405_, lean_object* v_q_406_, lean_object* v_h_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lp_mathlib_Equiv_sigmaSubtypeFiberEquivSubtype___redArg(v_f_404_);
return v___x_408_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__0(void){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lp_mathlib_Equiv_sigmaSubtypeEquivOfSubset(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_409_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__1(void){
_start:
{
lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_410_ = lean_obj_once(&lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__0, &lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__0_once, _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__0);
v___x_411_ = lp_mathlib_Equiv_symm___redArg(v___x_410_);
return v___x_411_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__2(void){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_Equiv_optionIsSomeEquiv(lean_box(0));
return v___x_412_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__3(void){
_start:
{
lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_413_ = lean_obj_once(&lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__2, &lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__2_once, _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__2);
v___x_414_ = lp_mathlib_Equiv_sigmaCongrLeft_x27___redArg(v___x_413_);
return v___x_414_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__4(void){
_start:
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v___x_415_ = lean_obj_once(&lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__3, &lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__3_once, _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__3);
v___x_416_ = lean_obj_once(&lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__1, &lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__1_once, _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__1);
v___x_417_ = lp_mathlib_Equiv_trans___redArg(v___x_416_, v___x_415_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaOptionEquivOfSome(lean_object* v_00_u03b1_418_, lean_object* v_p_419_, lean_object* v_h_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lean_obj_once(&lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__4, &lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__4_once, _init_lp_mathlib_Equiv_sigmaOptionEquivOfSome___closed__4);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma___lam__0(lean_object* v_f_422_, lean_object* v___y_423_){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; 
lean_inc(v___y_423_);
v___x_424_ = lean_apply_1(v_f_422_, v___y_423_);
v___x_425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_425_, 0, v___y_423_);
lean_ctor_set(v___x_425_, 1, v___x_424_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma___lam__1(lean_object* v_f_426_, lean_object* v_i_427_){
_start:
{
lean_object* v___x_428_; lean_object* v_snd_429_; 
v___x_428_ = lean_apply_1(v_f_426_, v_i_427_);
v_snd_429_ = lean_ctor_get(v___x_428_, 1);
lean_inc(v_snd_429_);
lean_dec_ref(v___x_428_);
return v_snd_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivSubtypeSigma(lean_object* v_00_u03b9_435_, lean_object* v_00_u03c0_436_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = ((lean_object*)(lp_mathlib_Equiv_piEquivSubtypeSigma___closed__2));
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePiEquivPi___lam__0(lean_object* v_f_438_, lean_object* v_a_439_){
_start:
{
lean_object* v___x_440_; 
v___x_440_ = lean_apply_1(v_f_438_, v_a_439_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePiEquivPi___lam__1(lean_object* v_f_441_, lean_object* v___y_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lean_apply_1(v_f_441_, v___y_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypePiEquivPi(lean_object* v_00_u03b1_449_, lean_object* v_00_u03b2_450_, lean_object* v_p_451_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = ((lean_object*)(lp_mathlib_Equiv_subtypePiEquivPi___closed__2));
return v___x_452_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaAssocProd___closed__0(void){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = lp_mathlib_Equiv_sigmaEquivProd(lean_box(0), lean_box(0));
return v___x_453_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaAssocProd___closed__1(void){
_start:
{
lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_454_ = lean_obj_once(&lp_mathlib_Equiv_sigmaAssocProd___closed__0, &lp_mathlib_Equiv_sigmaAssocProd___closed__0_once, _init_lp_mathlib_Equiv_sigmaAssocProd___closed__0);
v___x_455_ = lp_mathlib_Equiv_symm___redArg(v___x_454_);
return v___x_455_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaAssocProd___closed__2(void){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_456_ = lean_obj_once(&lp_mathlib_Equiv_sigmaAssocProd___closed__1, &lp_mathlib_Equiv_sigmaAssocProd___closed__1_once, _init_lp_mathlib_Equiv_sigmaAssocProd___closed__1);
v___x_457_ = lp_mathlib_Equiv_sigmaCongrLeft_x27___redArg(v___x_456_);
return v___x_457_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaAssocProd___closed__3(void){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lp_mathlib_Equiv_sigmaAssoc(lean_box(0), lean_box(0), lean_box(0));
return v___x_458_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaAssocProd___closed__4(void){
_start:
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
v___x_459_ = lean_obj_once(&lp_mathlib_Equiv_sigmaAssocProd___closed__3, &lp_mathlib_Equiv_sigmaAssocProd___closed__3_once, _init_lp_mathlib_Equiv_sigmaAssocProd___closed__3);
v___x_460_ = lean_obj_once(&lp_mathlib_Equiv_sigmaAssocProd___closed__2, &lp_mathlib_Equiv_sigmaAssocProd___closed__2_once, _init_lp_mathlib_Equiv_sigmaAssocProd___closed__2);
v___x_461_ = lp_mathlib_Equiv_trans___redArg(v___x_460_, v___x_459_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaAssocProd(lean_object* v_00_u03b1_462_, lean_object* v_00_u03b2_463_, lean_object* v_00_u03b3_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lean_obj_once(&lp_mathlib_Equiv_sigmaAssocProd___closed__4, &lp_mathlib_Equiv_sigmaAssocProd___closed__4_once, _init_lp_mathlib_Equiv_sigmaAssocProd___closed__4);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg___lam__0(lean_object* v_x_466_){
_start:
{
lean_object* v_snd_467_; 
v_snd_467_ = lean_ctor_get(v_x_466_, 1);
lean_inc(v_snd_467_);
return v_snd_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg___lam__0___boxed(lean_object* v_x_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_mathlib_Equiv_sigmaSubtype___redArg___lam__0(v_x_468_);
lean_dec_ref(v_x_468_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg___lam__1(lean_object* v_a_470_, lean_object* v_b_471_){
_start:
{
lean_object* v___x_472_; 
v___x_472_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_472_, 0, v_a_470_);
lean_ctor_set(v___x_472_, 1, v_b_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype___redArg(lean_object* v_a_474_){
_start:
{
lean_object* v___f_475_; lean_object* v___f_476_; lean_object* v___x_477_; 
v___f_475_ = ((lean_object*)(lp_mathlib_Equiv_sigmaSubtype___redArg___closed__0));
v___f_476_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaSubtype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_476_, 0, v_a_474_);
v___x_477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_477_, 0, v___f_475_);
lean_ctor_set(v___x_477_, 1, v___f_476_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSubtype(lean_object* v_00_u03b1_478_, lean_object* v_00_u03b2_479_, lean_object* v_a_480_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lp_mathlib_Equiv_sigmaSubtype___redArg(v_a_480_);
return v___x_481_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__0(void){
_start:
{
lean_object* v___x_482_; lean_object* v___x_483_; 
v___x_482_ = lean_obj_once(&lp_mathlib_Equiv_sigmaAssocProd___closed__3, &lp_mathlib_Equiv_sigmaAssocProd___closed__3_once, _init_lp_mathlib_Equiv_sigmaAssocProd___closed__3);
v___x_483_ = lp_mathlib_Equiv_symm___redArg(v___x_482_);
return v___x_483_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__1(void){
_start:
{
lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_484_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__0, &lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__0_once, _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__0);
v___x_485_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v___x_484_);
return v___x_485_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__2(void){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Equiv_subtypeSigmaEquiv(lean_box(0), lean_box(0), lean_box(0));
return v___x_486_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3(void){
_start:
{
lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; 
v___x_487_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__2, &lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__2_once, _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__2);
v___x_488_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__1, &lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__1_once, _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__1);
v___x_489_ = lp_mathlib_Equiv_trans___redArg(v___x_488_, v___x_487_);
return v___x_489_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4(void){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_mathlib_Equiv_cast(lean_box(0), lean_box(0), lean_box(0));
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___redArg(lean_object* v_uniq_491_){
_start:
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_492_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3, &lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3_once, _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3);
v___x_493_ = lp_mathlib_Equiv_uniqueSigma___redArg(v_uniq_491_);
v___x_494_ = lp_mathlib_Equiv_trans___redArg(v___x_492_, v___x_493_);
v___x_495_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4, &lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4_once, _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4);
v___x_496_ = lp_mathlib_Equiv_trans___redArg(v___x_494_, v___x_495_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype(lean_object* v_00_u03b1_497_, lean_object* v_00_u03b2_498_, lean_object* v_00_u03b3_499_, lean_object* v_p_500_, lean_object* v_uniq_501_, lean_object* v_a_502_, lean_object* v_b_503_, lean_object* v_h_504_){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = lp_mathlib_Equiv_sigmaSigmaSubtype___redArg(v_uniq_501_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___boxed(lean_object* v_00_u03b1_506_, lean_object* v_00_u03b2_507_, lean_object* v_00_u03b3_508_, lean_object* v_p_509_, lean_object* v_uniq_510_, lean_object* v_a_511_, lean_object* v_b_512_, lean_object* v_h_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_Equiv_sigmaSigmaSubtype(v_00_u03b1_506_, v_00_u03b2_507_, v_00_u03b3_508_, v_p_509_, v_uniq_510_, v_a_511_, v_b_512_, v_h_513_);
lean_dec(v_b_512_);
lean_dec(v_a_511_);
return v_res_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__1(lean_object* v___x_515_, lean_object* v_b_516_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_517_, 0, v___x_515_);
lean_ctor_set(v___x_517_, 1, v_b_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__0(lean_object* v_p_518_){
_start:
{
lean_object* v_snd_519_; 
v_snd_519_ = lean_ctor_get(v_p_518_, 1);
lean_inc(v_snd_519_);
return v_snd_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_p_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__0(v_p_520_);
lean_dec_ref(v_p_520_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg(lean_object* v___x_523_){
_start:
{
lean_object* v___f_524_; lean_object* v___f_525_; lean_object* v___x_526_; 
v___f_524_ = ((lean_object*)(lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___closed__0));
v___f_525_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg___lam__1), 2, 1);
lean_closure_set(v___f_525_, 0, v___x_523_);
v___x_526_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_526_, 0, v___f_524_);
lean_ctor_set(v___x_526_, 1, v___f_525_);
return v___x_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0___redArg(lean_object* v___x_527_){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_528_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3, &lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3_once, _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__3);
v___x_529_ = lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg(v___x_527_);
v___x_530_ = lp_mathlib_Equiv_trans___redArg(v___x_528_, v___x_529_);
v___x_531_ = lean_obj_once(&lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4, &lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4_once, _init_lp_mathlib_Equiv_sigmaSigmaSubtype___redArg___closed__4);
v___x_532_ = lp_mathlib_Equiv_trans___redArg(v___x_530_, v___x_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtypeEq___redArg(lean_object* v_a_533_, lean_object* v_b_534_){
_start:
{
lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_535_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_535_, 0, v_a_533_);
lean_ctor_set(v___x_535_, 1, v_b_534_);
v___x_536_ = lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0___redArg(v___x_535_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtypeEq(lean_object* v_00_u03b1_537_, lean_object* v_00_u03b2_538_, lean_object* v_00_u03b3_539_, lean_object* v_a_540_, lean_object* v_b_541_){
_start:
{
lean_object* v___x_542_; 
v___x_542_ = lp_mathlib_Equiv_sigmaSigmaSubtypeEq___redArg(v_a_540_, v_b_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0(lean_object* v_00_u03b1_543_, lean_object* v_00_u03b2_544_, lean_object* v___x_545_, lean_object* v_00_u03b2_546_){
_start:
{
lean_object* v___x_547_; 
v___x_547_ = lp_mathlib_Equiv_uniqueSigma___at___00Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0_spec__0___redArg(v___x_545_);
return v___x_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0(lean_object* v_00_u03b1_548_, lean_object* v_00_u03b2_549_, lean_object* v___x_550_, lean_object* v_00_u03b3_551_, lean_object* v_p_552_, lean_object* v_a_553_, lean_object* v_b_554_, lean_object* v_h_555_){
_start:
{
lean_object* v___x_556_; 
v___x_556_ = lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0___redArg(v___x_550_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0___boxed(lean_object* v_00_u03b1_557_, lean_object* v_00_u03b2_558_, lean_object* v___x_559_, lean_object* v_00_u03b3_560_, lean_object* v_p_561_, lean_object* v_a_562_, lean_object* v_b_563_, lean_object* v_h_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_mathlib_Equiv_sigmaSigmaSubtype___at___00Equiv_sigmaSigmaSubtypeEq_spec__0(v_00_u03b1_557_, v_00_u03b2_558_, v___x_559_, v_00_u03b3_560_, v_p_561_, v_a_562_, v_b_563_, v_h_564_);
lean_dec(v_b_563_);
lean_dec(v_a_562_);
return v_res_565_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_subtypeEquivCodomain___redArg___lam__0(lean_object* v_inst_566_, lean_object* v_x_567_, lean_object* v_a_568_){
_start:
{
lean_object* v___x_569_; uint8_t v___x_570_; 
v___x_569_ = lean_apply_2(v_inst_566_, v_a_568_, v_x_567_);
v___x_570_ = lean_unbox(v___x_569_);
if (v___x_570_ == 0)
{
uint8_t v___x_571_; 
v___x_571_ = 1;
return v___x_571_;
}
else
{
uint8_t v___x_572_; 
v___x_572_ = 0;
return v___x_572_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivCodomain___redArg___lam__0___boxed(lean_object* v_inst_573_, lean_object* v_x_574_, lean_object* v_a_575_){
_start:
{
uint8_t v_res_576_; lean_object* v_r_577_; 
v_res_576_ = lp_mathlib_Equiv_subtypeEquivCodomain___redArg___lam__0(v_inst_573_, v_x_574_, v_a_575_);
v_r_577_ = lean_box(v_res_576_);
return v_r_577_;
}
}
static lean_object* _init_lp_mathlib_Equiv_subtypeEquivCodomain___redArg___closed__0(void){
_start:
{
lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_578_ = lean_obj_once(&lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1, &lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1_once, _init_lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeInter___closed__1);
v___x_579_ = lp_mathlib_Equiv_symm___redArg(v___x_578_);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivCodomain___redArg(lean_object* v_inst_580_, lean_object* v_x_581_, lean_object* v_f_582_){
_start:
{
lean_object* v___x_583_; lean_object* v_toFun_584_; lean_object* v___f_585_; lean_object* v___x_586_; lean_object* v_this_587_; lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_583_ = lean_obj_once(&lp_mathlib_Equiv_subtypeEquivCodomain___redArg___closed__0, &lp_mathlib_Equiv_subtypeEquivCodomain___redArg___closed__0_once, _init_lp_mathlib_Equiv_subtypeEquivCodomain___redArg___closed__0);
v_toFun_584_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_x_581_);
v___f_585_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_subtypeEquivCodomain___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_585_, 0, v_inst_580_);
lean_closure_set(v___f_585_, 1, v_x_581_);
v___x_586_ = lp_mathlib_Equiv_subtypePreimage___redArg(v___f_585_, v_f_582_);
lean_inc(v_toFun_584_);
v_this_587_ = lean_apply_1(v_toFun_584_, v_x_581_);
v___x_588_ = lp_mathlib_Equiv_piUnique___redArg(v_this_587_);
v___x_589_ = lp_mathlib_Equiv_trans___redArg(v___x_586_, v___x_588_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeEquivCodomain(lean_object* v_X_590_, lean_object* v_Y_591_, lean_object* v_inst_592_, lean_object* v_x_593_, lean_object* v_f_594_){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lp_mathlib_Equiv_subtypeEquivCodomain___redArg(v_inst_592_, v_x_593_, v_f_594_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomain___redArg(lean_object* v_e_596_, lean_object* v_inst_597_, lean_object* v_f_598_){
_start:
{
lean_object* v___x_599_; lean_object* v_toFun_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
lean_inc_ref(v_f_598_);
v___x_599_ = lp_mathlib_Equiv_equivCongr___redArg(v_f_598_, v_f_598_);
v_toFun_600_ = lean_ctor_get(v___x_599_, 0);
lean_inc(v_toFun_600_);
lean_dec_ref(v___x_599_);
v___x_601_ = lean_apply_1(v_toFun_600_, v_e_596_);
v___x_602_ = lean_obj_once(&lp_mathlib_Equiv_subtypeEquivRight___closed__0, &lp_mathlib_Equiv_subtypeEquivRight___closed__0_once, _init_lp_mathlib_Equiv_subtypeEquivRight___closed__0);
v___x_603_ = lp_mathlib_Equiv_Perm_subtypeCongr___redArg(v_inst_597_, v___x_601_, v___x_602_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_extendDomain(lean_object* v_00_u03b1_x27_604_, lean_object* v_00_u03b2_x27_605_, lean_object* v_e_606_, lean_object* v_p_607_, lean_object* v_inst_608_, lean_object* v_f_609_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lp_mathlib_Equiv_Perm_extendDomain___redArg(v_e_606_, v_inst_608_, v_f_609_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype(lean_object* v_00_u03b1_611_, lean_object* v_p_u2081_612_, lean_object* v_s_u2081_613_, lean_object* v_s_u2082_614_, lean_object* v_p_u2082_615_, lean_object* v_hp_u2082_616_, lean_object* v_h_617_){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = ((lean_object*)(lp_mathlib_Equiv_subtypeSubtypeEquivSubtypeExists___closed__1));
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swapCore___redArg(lean_object* v_inst_619_, lean_object* v_a_620_, lean_object* v_b_621_, lean_object* v_r_622_){
_start:
{
lean_object* v___x_623_; uint8_t v___x_624_; 
lean_inc_ref(v_inst_619_);
lean_inc(v_a_620_);
lean_inc(v_r_622_);
v___x_623_ = lean_apply_2(v_inst_619_, v_r_622_, v_a_620_);
v___x_624_ = lean_unbox(v___x_623_);
if (v___x_624_ == 0)
{
lean_object* v___x_625_; uint8_t v___x_626_; 
lean_inc(v_r_622_);
v___x_625_ = lean_apply_2(v_inst_619_, v_r_622_, v_b_621_);
v___x_626_ = lean_unbox(v___x_625_);
if (v___x_626_ == 0)
{
lean_dec(v_a_620_);
return v_r_622_;
}
else
{
lean_dec(v_r_622_);
return v_a_620_;
}
}
else
{
lean_dec(v_r_622_);
lean_dec(v_a_620_);
lean_dec_ref(v_inst_619_);
return v_b_621_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swapCore(lean_object* v_00_u03b1_627_, lean_object* v_inst_628_, lean_object* v_a_629_, lean_object* v_b_630_, lean_object* v_r_631_){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lp_mathlib_Equiv_swapCore___redArg(v_inst_628_, v_a_629_, v_b_630_, v_r_631_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swap___redArg(lean_object* v_inst_633_, lean_object* v_a_634_, lean_object* v_b_635_){
_start:
{
lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_636_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_swapCore), 5, 4);
lean_closure_set(v___x_636_, 0, lean_box(0));
lean_closure_set(v___x_636_, 1, v_inst_633_);
lean_closure_set(v___x_636_, 2, v_a_634_);
lean_closure_set(v___x_636_, 3, v_b_635_);
lean_inc_ref(v___x_636_);
v___x_637_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_637_, 0, v___x_636_);
lean_ctor_set(v___x_637_, 1, v___x_636_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_swap(lean_object* v_00_u03b1_638_, lean_object* v_inst_639_, lean_object* v_a_640_, lean_object* v_b_641_){
_start:
{
lean_object* v___x_642_; 
v___x_642_ = lp_mathlib_Equiv_swap___redArg(v_inst_639_, v_a_640_, v_b_641_);
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setValue___redArg(lean_object* v_inst_643_, lean_object* v_f_644_, lean_object* v_a_645_, lean_object* v_b_646_){
_start:
{
lean_object* v___x_647_; lean_object* v_toFun_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
lean_inc_ref(v_f_644_);
v___x_647_ = lp_mathlib_Equiv_symm___redArg(v_f_644_);
v_toFun_648_ = lean_ctor_get(v___x_647_, 0);
lean_inc(v_toFun_648_);
lean_dec_ref(v___x_647_);
v___x_649_ = lean_apply_1(v_toFun_648_, v_b_646_);
v___x_650_ = lp_mathlib_Equiv_swap___redArg(v_inst_643_, v_a_645_, v___x_649_);
v___x_651_ = lp_mathlib_Equiv_trans___redArg(v___x_650_, v_f_644_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_setValue(lean_object* v_00_u03b1_652_, lean_object* v_00_u03b2_653_, lean_object* v_inst_654_, lean_object* v_f_655_, lean_object* v_a_656_, lean_object* v_b_657_){
_start:
{
lean_object* v___x_658_; 
v___x_658_ = lp_mathlib_Equiv_setValue___redArg(v_inst_654_, v_f_655_, v_a_656_, v_b_657_);
return v___x_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Involutive_toPerm___redArg(lean_object* v_f_659_){
_start:
{
lean_object* v___x_660_; 
lean_inc(v_f_659_);
v___x_660_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_660_, 0, v_f_659_);
lean_ctor_set(v___x_660_, 1, v_f_659_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Involutive_toPerm(lean_object* v_00_u03b1_661_, lean_object* v_f_662_, lean_object* v_h_663_){
_start:
{
lean_object* v___x_664_; 
lean_inc(v_f_662_);
v___x_664_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_664_, 0, v_f_662_);
lean_ctor_set(v___x_664_, 1, v_f_662_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg___lam__0(lean_object* v_e_665_, lean_object* v_f_666_, lean_object* v_x_667_){
_start:
{
lean_object* v___x_668_; lean_object* v_toFun_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_668_ = lp_mathlib_Equiv_symm___redArg(v_e_665_);
v_toFun_669_ = lean_ctor_get(v___x_668_, 0);
lean_inc(v_toFun_669_);
lean_dec_ref(v___x_668_);
v___x_670_ = lean_apply_1(v_toFun_669_, v_x_667_);
v___x_671_ = lean_apply_1(v_f_666_, v___x_670_);
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg___lam__1(lean_object* v_e_672_, lean_object* v_f_673_, lean_object* v_x_674_){
_start:
{
lean_object* v_toFun_675_; lean_object* v___x_676_; lean_object* v___x_677_; 
v_toFun_675_ = lean_ctor_get(v_e_672_, 0);
lean_inc(v_toFun_675_);
lean_dec_ref(v_e_672_);
v___x_676_ = lean_apply_1(v_toFun_675_, v_x_674_);
v___x_677_ = lean_apply_1(v_f_673_, v___x_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27___redArg(lean_object* v_e_678_){
_start:
{
lean_object* v___f_679_; lean_object* v___f_680_; lean_object* v___x_681_; 
lean_inc_ref(v_e_678_);
v___f_679_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piCongrLeft_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_679_, 0, v_e_678_);
v___f_680_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piCongrLeft_x27___redArg___lam__1), 3, 1);
lean_closure_set(v___f_680_, 0, v_e_678_);
v___x_681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_681_, 0, v___f_679_);
lean_ctor_set(v___x_681_, 1, v___f_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft_x27(lean_object* v_00_u03b1_682_, lean_object* v_00_u03b2_683_, lean_object* v_P_684_, lean_object* v_e_685_){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_mathlib_Equiv_piCongrLeft_x27___redArg(v_e_685_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft___redArg(lean_object* v_e_687_){
_start:
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v___x_688_ = lp_mathlib_Equiv_symm___redArg(v_e_687_);
v___x_689_ = lp_mathlib_Equiv_piCongrLeft_x27___redArg(v___x_688_);
v___x_690_ = lp_mathlib_Equiv_symm___redArg(v___x_689_);
return v___x_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrLeft(lean_object* v_00_u03b1_691_, lean_object* v_00_u03b2_692_, lean_object* v_P_693_, lean_object* v_e_694_){
_start:
{
lean_object* v___x_695_; 
v___x_695_ = lp_mathlib_Equiv_piCongrLeft___redArg(v_e_694_);
return v___x_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr___redArg(lean_object* v_h_u2081_696_, lean_object* v_h_u2082_697_){
_start:
{
lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; 
v___x_698_ = lp_mathlib_Equiv_piCongrRight___redArg(v_h_u2082_697_);
v___x_699_ = lp_mathlib_Equiv_piCongrLeft___redArg(v_h_u2081_696_);
v___x_700_ = lp_mathlib_Equiv_trans___redArg(v___x_698_, v___x_699_);
return v___x_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr(lean_object* v_00_u03b1_701_, lean_object* v_00_u03b2_702_, lean_object* v_W_703_, lean_object* v_Z_704_, lean_object* v_h_u2081_705_, lean_object* v_h_u2082_706_){
_start:
{
lean_object* v___x_707_; 
v___x_707_ = lp_mathlib_Equiv_piCongr___redArg(v_h_u2081_705_, v_h_u2082_706_);
return v___x_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr_x27___redArg___lam__0(lean_object* v_h_u2082_708_, lean_object* v_b_709_){
_start:
{
lean_object* v___x_710_; lean_object* v___x_711_; 
v___x_710_ = lean_apply_1(v_h_u2082_708_, v_b_709_);
v___x_711_ = lp_mathlib_Equiv_symm___redArg(v___x_710_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr_x27___redArg(lean_object* v_h_u2081_712_, lean_object* v_h_u2082_713_){
_start:
{
lean_object* v___f_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; 
v___f_714_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piCongr_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_714_, 0, v_h_u2082_713_);
v___x_715_ = lp_mathlib_Equiv_symm___redArg(v_h_u2081_712_);
v___x_716_ = lp_mathlib_Equiv_piCongr___redArg(v___x_715_, v___f_714_);
v___x_717_ = lp_mathlib_Equiv_symm___redArg(v___x_716_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongr_x27(lean_object* v_00_u03b1_718_, lean_object* v_00_u03b2_719_, lean_object* v_W_720_, lean_object* v_Z_721_, lean_object* v_h_u2081_722_, lean_object* v_h_u2082_723_){
_start:
{
lean_object* v___x_724_; 
v___x_724_ = lp_mathlib_Equiv_piCongr_x27___redArg(v_h_u2081_722_, v_h_u2082_723_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSigmaFiber___redArg(lean_object* v_f_725_, lean_object* v_e_726_){
_start:
{
lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; 
v___x_727_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v_f_725_);
v___x_728_ = lp_mathlib_Equiv_piCongrLeft___redArg(v___x_727_);
v___x_729_ = lp_mathlib_Equiv_piCongrRight___redArg(v_e_726_);
v___x_730_ = lp_mathlib_Equiv_trans___redArg(v___x_728_, v___x_729_);
return v___x_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSigmaFiber(lean_object* v_00_u03b1_731_, lean_object* v_00_u03b2_732_, lean_object* v_f_733_, lean_object* v_00_u03b3_u2081_734_, lean_object* v_00_u03b3_u2082_735_, lean_object* v_e_736_){
_start:
{
lean_object* v___x_737_; 
v___x_737_ = lp_mathlib_Equiv_piCongrSigmaFiber___redArg(v_f_733_, v_e_736_);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg___lam__0(lean_object* v_x_738_){
_start:
{
lean_object* v___x_739_; 
v___x_739_ = lean_obj_once(&lp_mathlib_Equiv_subtypeEquivRight___closed__0, &lp_mathlib_Equiv_subtypeEquivRight___closed__0_once, _init_lp_mathlib_Equiv_subtypeEquivRight___closed__0);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg___lam__0___boxed(lean_object* v_x_740_){
_start:
{
lean_object* v_res_741_; 
v_res_741_ = lp_mathlib_Equiv_piCongrFiberwise___redArg___lam__0(v_x_740_);
lean_dec(v_x_740_);
return v_res_741_;
}
}
static lean_object* _init_lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__1(void){
_start:
{
lean_object* v___x_743_; 
v___x_743_ = lp_mathlib_Equiv_piCurry(lean_box(0), lean_box(0), lean_box(0));
return v___x_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise___redArg(lean_object* v_f_744_, lean_object* v_e_745_){
_start:
{
lean_object* v___f_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; 
v___f_746_ = ((lean_object*)(lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__0));
v___x_747_ = lp_mathlib_Equiv_piCongrSigmaFiber___redArg(v_f_744_, v___f_746_);
v___x_748_ = lp_mathlib_Equiv_symm___redArg(v___x_747_);
v___x_749_ = lean_obj_once(&lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__1, &lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__1_once, _init_lp_mathlib_Equiv_piCongrFiberwise___redArg___closed__1);
v___x_750_ = lp_mathlib_Equiv_trans___redArg(v___x_748_, v___x_749_);
v___x_751_ = lp_mathlib_Equiv_piCongrRight___redArg(v_e_745_);
v___x_752_ = lp_mathlib_Equiv_trans___redArg(v___x_750_, v___x_751_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrFiberwise(lean_object* v_00_u03b1_753_, lean_object* v_00_u03b2_754_, lean_object* v_00_u03b3_u2081_755_, lean_object* v_00_u03b3_u2082_756_, lean_object* v_f_757_, lean_object* v_e_758_){
_start:
{
lean_object* v___x_759_; 
v___x_759_ = lp_mathlib_Equiv_piCongrFiberwise___redArg(v_f_757_, v_e_758_);
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSet___lam__0(lean_object* v_f_760_, lean_object* v_i_761_){
_start:
{
lean_object* v___x_762_; 
v___x_762_ = lean_apply_1(v_f_760_, v_i_761_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piCongrSet(lean_object* v_00_u03b1_766_, lean_object* v_W_767_, lean_object* v_s_768_, lean_object* v_t_769_, lean_object* v_h_770_){
_start:
{
lean_object* v___x_771_; 
v___x_771_ = ((lean_object*)(lp_mathlib_Equiv_piCongrSet___closed__1));
return v___x_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_equivOfSubsingletonOfSubsingleton___redArg(lean_object* v_f_772_, lean_object* v_g_773_){
_start:
{
lean_object* v___x_774_; 
v___x_774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_774_, 0, v_f_772_);
lean_ctor_set(v___x_774_, 1, v_g_773_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_equivOfSubsingletonOfSubsingleton(lean_object* v_00_u03b1_775_, lean_object* v_00_u03b2_776_, lean_object* v_inst_777_, lean_object* v_inst_778_, lean_object* v_f_779_, lean_object* v_g_780_){
_start:
{
lean_object* v___x_781_; 
v___x_781_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_781_, 0, v_f_779_);
lean_ctor_set(v___x_781_, 1, v_g_780_);
return v___x_781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0___redArg(lean_object* v_h_782_){
_start:
{
lean_inc(v_h_782_);
return v_h_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0___redArg___boxed(lean_object* v_h_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_mathlib_uniqueUniqueEquiv___elam__0___redArg(v_h_783_);
lean_dec(v_h_783_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0(lean_object* v_00_u03b1_785_, lean_object* v_h_786_){
_start:
{
lean_inc(v_h_786_);
return v_h_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__0___boxed(lean_object* v_00_u03b1_787_, lean_object* v_h_788_){
_start:
{
lean_object* v_res_789_; 
v_res_789_ = lp_mathlib_uniqueUniqueEquiv___elam__0(v_00_u03b1_787_, v_h_788_);
lean_dec(v_h_788_);
return v_res_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1___redArg(lean_object* v_h_790_){
_start:
{
lean_inc(v_h_790_);
return v_h_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1___redArg___boxed(lean_object* v_h_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_mathlib_uniqueUniqueEquiv___elam__1___redArg(v_h_791_);
lean_dec(v_h_791_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1(lean_object* v_00_u03b1_793_, lean_object* v_h_794_){
_start:
{
lean_inc(v_h_794_);
return v_h_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv___elam__1___boxed(lean_object* v_00_u03b1_795_, lean_object* v_h_796_){
_start:
{
lean_object* v_res_797_; 
v_res_797_ = lp_mathlib_uniqueUniqueEquiv___elam__1(v_00_u03b1_795_, v_h_796_);
lean_dec(v_h_796_);
return v_res_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueUniqueEquiv(lean_object* v_00_u03b1_803_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = ((lean_object*)(lp_mathlib_uniqueUniqueEquiv___closed__2));
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__1(lean_object* v_x_805_, lean_object* v_x_806_){
_start:
{
lean_inc(v_x_805_);
return v_x_805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__1___boxed(lean_object* v_x_807_, lean_object* v_x_808_){
_start:
{
lean_object* v_res_809_; 
v_res_809_ = lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__1(v_x_807_, v_x_808_);
lean_dec(v_x_808_);
lean_dec(v_x_807_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__0(lean_object* v_inst_810_, lean_object* v_x_811_){
_start:
{
lean_inc(v_inst_810_);
return v_inst_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__0___boxed(lean_object* v_inst_812_, lean_object* v_x_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__0(v_inst_812_, v_x_813_);
lean_dec(v_x_813_);
lean_dec(v_inst_812_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg(lean_object* v_x_815_, lean_object* v_inst_816_){
_start:
{
lean_object* v___f_817_; lean_object* v___f_818_; lean_object* v___x_819_; 
v___f_817_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_817_, 0, v_inst_816_);
v___f_818_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_818_, 0, v_x_815_);
v___x_819_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_819_, 0, v___f_817_);
lean_ctor_set(v___x_819_, 1, v___f_818_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique___elam__0(lean_object* v_00_u03b1_820_, lean_object* v_00_u03b2_821_, lean_object* v_inst_822_, lean_object* v_x_823_){
_start:
{
lean_object* v___x_824_; 
v___x_824_ = lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg(v_x_823_, v_inst_822_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique___redArg(lean_object* v_inst_825_){
_start:
{
lean_object* v___f_826_; lean_object* v___x_827_; lean_object* v___x_828_; 
lean_inc(v_inst_825_);
v___f_826_ = lean_alloc_closure((void*)(lp_mathlib_uniqueEquivEquivUnique___elam__0), 4, 3);
lean_closure_set(v___f_826_, 0, lean_box(0));
lean_closure_set(v___f_826_, 1, lean_box(0));
lean_closure_set(v___f_826_, 2, v_inst_825_);
v___x_827_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_unique), 4, 3);
lean_closure_set(v___x_827_, 0, lean_box(0));
lean_closure_set(v___x_827_, 1, lean_box(0));
lean_closure_set(v___x_827_, 2, v_inst_825_);
v___x_828_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_828_, 0, v___f_826_);
lean_ctor_set(v___x_828_, 1, v___x_827_);
return v___x_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique(lean_object* v_00_u03b1_829_, lean_object* v_00_u03b2_830_, lean_object* v_inst_831_){
_start:
{
lean_object* v___x_832_; 
v___x_832_ = lp_mathlib_uniqueEquivEquivUnique___redArg(v_inst_831_);
return v___x_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0(lean_object* v_00_u03b1_833_, lean_object* v_00_u03b2_834_, lean_object* v_x_835_, lean_object* v_inst_836_){
_start:
{
lean_object* v___x_837_; 
v___x_837_ = lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg(v_x_835_, v_inst_836_);
return v___x_837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueEquivEquivUnique___elam__0___redArg(lean_object* v_inst_838_, lean_object* v_x_839_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = lp_mathlib_Equiv_ofUnique___at___00uniqueEquivEquivUnique___elam__0_spec__0___redArg(v_x_839_, v_inst_838_);
return v___x_840_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Option(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Sum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Conjugate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Conjugate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_natSumPUnitEquivNat = _init_lp_mathlib_Equiv_natSumPUnitEquivNat();
lean_mark_persistent(lp_mathlib_Equiv_natSumPUnitEquivNat);
lp_mathlib_Equiv_intEquivNatSumNat = _init_lp_mathlib_Equiv_intEquivNatSumNat();
lean_mark_persistent(lp_mathlib_Equiv_intEquivNatSumNat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Conjugate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Conjugate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
