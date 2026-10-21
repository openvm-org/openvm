// Lean compiler output
// Module: Mathlib.Logic.Equiv.Prod
// Imports: public import Init public meta import Init public import Mathlib.Logic.Equiv.Defs public import Mathlib.Tactic.Contrapose public import Mathlib.Util.CompileInductive
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
lean_object* lp_mathlib_Equiv_equivOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Prod_swap(lean_object*, lean_object*, lean_object*);
lean_object* l_Prod_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivPUnit___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_elim___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_plift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProd___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_pprodEquivProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_pprodEquivProd___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_pprodEquivProd___closed__0 = (const lean_object*)&lp_mathlib_Equiv_pprodEquivProd___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_pprodEquivProd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_pprodEquivProd___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_pprodEquivProd___closed__1 = (const lean_object*)&lp_mathlib_Equiv_pprodEquivProd___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_pprodEquivProd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_pprodEquivProd___closed__0_value),((lean_object*)&lp_mathlib_Equiv_pprodEquivProd___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_pprodEquivProd___closed__2 = (const lean_object*)&lp_mathlib_Equiv_pprodEquivProd___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_pprodProd___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_pprodProd___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodProd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPProd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_pprodEquivProdPLift___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_pprodEquivProdPLift___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_pprodEquivProdPLift___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_pprodEquivProdPLift___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_pprodEquivProdPLift___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_pprodEquivProdPLift___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProdPLift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_prodComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Prod_swap, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_prodComm___closed__0 = (const lean_object*)&lp_mathlib_Equiv_prodComm___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_prodComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_prodComm___closed__0_value),((lean_object*)&lp_mathlib_Equiv_prodComm___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_prodComm___closed__1 = (const lean_object*)&lp_mathlib_Equiv_prodComm___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodAssoc___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodAssoc___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_prodAssoc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_prodAssoc___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_prodAssoc___closed__0 = (const lean_object*)&lp_mathlib_Equiv_prodAssoc___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_prodAssoc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_prodAssoc___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_prodAssoc___closed__1 = (const lean_object*)&lp_mathlib_Equiv_prodAssoc___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_prodAssoc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_prodAssoc___closed__0_value),((lean_object*)&lp_mathlib_Equiv_prodAssoc___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_prodAssoc___closed__2 = (const lean_object*)&lp_mathlib_Equiv_prodAssoc___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodAssoc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodProdProdComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodProdProdComm___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_prodProdProdComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_prodProdProdComm___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_prodProdProdComm___closed__0 = (const lean_object*)&lp_mathlib_Equiv_prodProdProdComm___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_prodProdProdComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_prodProdProdComm___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_prodProdProdComm___closed__1 = (const lean_object*)&lp_mathlib_Equiv_prodProdProdComm___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_prodProdProdComm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_prodProdProdComm___closed__0_value),((lean_object*)&lp_mathlib_Equiv_prodProdProdComm___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_prodProdProdComm___closed__2 = (const lean_object*)&lp_mathlib_Equiv_prodProdProdComm___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodProdProdComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_curry___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_curry___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_curry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_curry___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_curry___closed__0 = (const lean_object*)&lp_mathlib_Equiv_curry___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_curry___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_curry___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_curry___closed__1 = (const lean_object*)&lp_mathlib_Equiv_curry___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_curry___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_curry___closed__0_value),((lean_object*)&lp_mathlib_Equiv_curry___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_curry___closed__2 = (const lean_object*)&lp_mathlib_Equiv_curry___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_curry(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_prodPUnit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_prodPUnit___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_prodPUnit___closed__0 = (const lean_object*)&lp_mathlib_Equiv_prodPUnit___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_prodPUnit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_prodPUnit___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_prodPUnit___closed__1 = (const lean_object*)&lp_mathlib_Equiv_prodPUnit___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_prodPUnit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_prodPUnit___closed__0_value),((lean_object*)&lp_mathlib_Equiv_prodPUnit___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_prodPUnit___closed__2 = (const lean_object*)&lp_mathlib_Equiv_prodPUnit___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_punitProd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_punitProd___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_punitProd___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_punitProd___closed__1;
static lean_once_cell_t lp_mathlib_Equiv_punitProd___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_punitProd___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_punitProd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaPUnit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaPUnit___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaPUnit___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaPUnit___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaPUnit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaPUnit___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaPUnit___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sigmaPUnit___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sigmaPUnit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sigmaPUnit___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaPUnit___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sigmaPUnit___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sigmaPUnit___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_prodUnique___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodUnique___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodUnique(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_uniqueProd___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_uniqueProd___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueProd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaUnique___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_sigmaUnique___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_sigmaUnique___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaUnique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_uniqueSigma___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_uniqueSigma___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_uniqueSigma___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_uniqueSigma___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_prodEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmpty(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_emptyProd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_emptyProd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptyProd(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_prodPEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodPEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPEmpty(lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_pemptyProd___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_pemptyProd___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pemptyProd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__3(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__0 = (const lean_object*)&lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__3, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__1 = (const lean_object*)&lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__0_value),((lean_object*)&lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__2 = (const lean_object*)&lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__3(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumPiEquivProdPi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumPiEquivProdPi___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumPiEquivProdPi___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumPiEquivProdPi___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumPiEquivProdPi___lam__3, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumPiEquivProdPi___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sumPiEquivProdPi___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumPiEquivProdPi___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumPiEquivProdPi___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumPiEquivProdPi___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_prodPiEquivSumPi___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodPiEquivSumPi___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_prodPiEquivSumPi___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_prodPiEquivSumPi___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPiEquivSumPi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__3(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__3, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__4___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sumProdDistrib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumProdDistrib___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumProdDistrib___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sumProdDistrib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumProdDistrib___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumProdDistrib___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_sumProdDistrib___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumProdDistrib___lam__3, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumProdDistrib___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_sumProdDistrib___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumProdDistrib___lam__4___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sumProdDistrib___closed__3 = (const lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__3_value;
static const lean_closure_object lp_mathlib_Equiv_sumProdDistrib___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sumProdDistrib___lam__5, .m_arity = 4, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__1_value),((lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__3_value),((lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_sumProdDistrib___closed__4 = (const lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__4_value;
static const lean_ctor_object lp_mathlib_Equiv_sumProdDistrib___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__4_value)}};
static const lean_object* lp_mathlib_Equiv_sumProdDistrib___closed__5 = (const lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaProdDistrib___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaProdDistrib___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sigmaProdDistrib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaProdDistrib___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaProdDistrib___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sigmaProdDistrib___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_sigmaProdDistrib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sigmaProdDistrib___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sigmaProdDistrib___closed__1 = (const lean_object*)&lp_mathlib_Equiv_sigmaProdDistrib___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_sigmaProdDistrib___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_sigmaProdDistrib___closed__0_value),((lean_object*)&lp_mathlib_Equiv_sigmaProdDistrib___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_sigmaProdDistrib___closed__2 = (const lean_object*)&lp_mathlib_Equiv_sigmaProdDistrib___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaProdDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_boolProdEquivSum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolProdEquivSum___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolProdEquivSum___closed__0 = (const lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_boolProdEquivSum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolProdEquivSum___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolProdEquivSum___closed__1 = (const lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_boolProdEquivSum___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolProdEquivSum___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolProdEquivSum___closed__2 = (const lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__2_value;
static const lean_closure_object lp_mathlib_Equiv_boolProdEquivSum___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Sum_elim, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__1_value),((lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__2_value)} };
static const lean_object* lp_mathlib_Equiv_boolProdEquivSum___closed__3 = (const lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__3_value;
static const lean_ctor_object lp_mathlib_Equiv_boolProdEquivSum___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__0_value),((lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__3_value)}};
static const lean_object* lp_mathlib_Equiv_boolProdEquivSum___closed__4 = (const lean_object*)&lp_mathlib_Equiv_boolProdEquivSum___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd___lam__1(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_boolArrowEquivProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolArrowEquivProd___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolArrowEquivProd___closed__0 = (const lean_object*)&lp_mathlib_Equiv_boolArrowEquivProd___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_boolArrowEquivProd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_boolArrowEquivProd___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_boolArrowEquivProd___closed__1 = (const lean_object*)&lp_mathlib_Equiv_boolArrowEquivProd___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_boolArrowEquivProd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_boolArrowEquivProd___closed__0_value),((lean_object*)&lp_mathlib_Equiv_boolArrowEquivProd___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_boolArrowEquivProd___closed__2 = (const lean_object*)&lp_mathlib_Equiv_boolArrowEquivProd___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquivSigma___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeSigmaEquivSigma___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquivSigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivProd___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeProdEquivProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeProdEquivProd___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeProdEquivProd___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeProdEquivProd___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeProdEquivProd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeProdEquivProd___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeProdEquivProd___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeProdEquivProd___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeProdEquivProd___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodSubtypeFstEquivSubtypeProd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__2 = (const lean_object*)&lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funSplitAt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funSplitAt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_subsingletonProdSelfEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_subsingletonProdSelfEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_subsingletonProdSelfEquiv___closed__0 = (const lean_object*)&lp_mathlib_subsingletonProdSelfEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_subsingletonProdSelfEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_subsingletonProdSelfEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_subsingletonProdSelfEquiv___closed__1 = (const lean_object*)&lp_mathlib_subsingletonProdSelfEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_subsingletonProdSelfEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_subsingletonProdSelfEquiv___closed__0_value),((lean_object*)&lp_mathlib_subsingletonProdSelfEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_subsingletonProdSelfEquiv___closed__2 = (const lean_object*)&lp_mathlib_subsingletonProdSelfEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_optionProdEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_optionProdEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_optionProdEquiv___closed__0 = (const lean_object*)&lp_mathlib_optionProdEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_optionProdEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_optionProdEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_optionProdEquiv___closed__1 = (const lean_object*)&lp_mathlib_optionProdEquiv___closed__1_value;
static const lean_closure_object lp_mathlib_optionProdEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_optionProdEquiv___lam__3, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_optionProdEquiv___closed__1_value),((lean_object*)&lp_mathlib_Equiv_sumProdDistrib___closed__3_value)} };
static const lean_object* lp_mathlib_optionProdEquiv___closed__2 = (const lean_object*)&lp_mathlib_optionProdEquiv___closed__2_value;
static const lean_ctor_object lp_mathlib_optionProdEquiv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_optionProdEquiv___closed__0_value),((lean_object*)&lp_mathlib_optionProdEquiv___closed__2_value)}};
static const lean_object* lp_mathlib_optionProdEquiv___closed__3 = (const lean_object*)&lp_mathlib_optionProdEquiv___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProd___lam__0(lean_object* v_x_1_){
_start:
{
lean_object* v_fst_2_; lean_object* v_snd_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_10_; 
v_fst_2_ = lean_ctor_get(v_x_1_, 0);
v_snd_3_ = lean_ctor_get(v_x_1_, 1);
v_isSharedCheck_10_ = !lean_is_exclusive(v_x_1_);
if (v_isSharedCheck_10_ == 0)
{
v___x_5_ = v_x_1_;
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_snd_3_);
lean_inc(v_fst_2_);
lean_dec(v_x_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_10_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_8_; 
if (v_isShared_6_ == 0)
{
v___x_8_ = v___x_5_;
goto v_reusejp_7_;
}
else
{
lean_object* v_reuseFailAlloc_9_; 
v_reuseFailAlloc_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_9_, 0, v_fst_2_);
lean_ctor_set(v_reuseFailAlloc_9_, 1, v_snd_3_);
v___x_8_ = v_reuseFailAlloc_9_;
goto v_reusejp_7_;
}
v_reusejp_7_:
{
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProd___lam__1(lean_object* v_x_11_){
_start:
{
lean_object* v_fst_12_; lean_object* v_snd_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_20_; 
v_fst_12_ = lean_ctor_get(v_x_11_, 0);
v_snd_13_ = lean_ctor_get(v_x_11_, 1);
v_isSharedCheck_20_ = !lean_is_exclusive(v_x_11_);
if (v_isSharedCheck_20_ == 0)
{
v___x_15_ = v_x_11_;
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_snd_13_);
lean_inc(v_fst_12_);
lean_dec(v_x_11_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_20_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_18_; 
if (v_isShared_16_ == 0)
{
v___x_18_ = v___x_15_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_19_; 
v_reuseFailAlloc_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_19_, 0, v_fst_12_);
lean_ctor_set(v_reuseFailAlloc_19_, 1, v_snd_13_);
v___x_18_ = v_reuseFailAlloc_19_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
return v___x_18_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProd(lean_object* v_00_u03b1_26_, lean_object* v_00_u03b2_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = ((lean_object*)(lp_mathlib_Equiv_pprodEquivProd___closed__2));
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr___redArg___lam__0(lean_object* v_e_u2081_29_, lean_object* v_e_u2082_30_, lean_object* v_x_31_){
_start:
{
lean_object* v_fst_32_; lean_object* v_snd_33_; lean_object* v___x_35_; uint8_t v_isShared_36_; uint8_t v_isSharedCheck_44_; 
v_fst_32_ = lean_ctor_get(v_x_31_, 0);
v_snd_33_ = lean_ctor_get(v_x_31_, 1);
v_isSharedCheck_44_ = !lean_is_exclusive(v_x_31_);
if (v_isSharedCheck_44_ == 0)
{
v___x_35_ = v_x_31_;
v_isShared_36_ = v_isSharedCheck_44_;
goto v_resetjp_34_;
}
else
{
lean_inc(v_snd_33_);
lean_inc(v_fst_32_);
lean_dec(v_x_31_);
v___x_35_ = lean_box(0);
v_isShared_36_ = v_isSharedCheck_44_;
goto v_resetjp_34_;
}
v_resetjp_34_:
{
lean_object* v_toFun_37_; lean_object* v_toFun_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_42_; 
v_toFun_37_ = lean_ctor_get(v_e_u2081_29_, 0);
lean_inc(v_toFun_37_);
lean_dec_ref(v_e_u2081_29_);
v_toFun_38_ = lean_ctor_get(v_e_u2082_30_, 0);
lean_inc(v_toFun_38_);
lean_dec_ref(v_e_u2082_30_);
v___x_39_ = lean_apply_1(v_toFun_37_, v_fst_32_);
v___x_40_ = lean_apply_1(v_toFun_38_, v_snd_33_);
if (v_isShared_36_ == 0)
{
lean_ctor_set(v___x_35_, 1, v___x_40_);
lean_ctor_set(v___x_35_, 0, v___x_39_);
v___x_42_ = v___x_35_;
goto v_reusejp_41_;
}
else
{
lean_object* v_reuseFailAlloc_43_; 
v_reuseFailAlloc_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_43_, 0, v___x_39_);
lean_ctor_set(v_reuseFailAlloc_43_, 1, v___x_40_);
v___x_42_ = v_reuseFailAlloc_43_;
goto v_reusejp_41_;
}
v_reusejp_41_:
{
return v___x_42_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr___redArg___lam__1(lean_object* v_e_u2081_45_, lean_object* v_e_u2082_46_, lean_object* v_x_47_){
_start:
{
lean_object* v_fst_48_; lean_object* v_snd_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_62_; 
v_fst_48_ = lean_ctor_get(v_x_47_, 0);
v_snd_49_ = lean_ctor_get(v_x_47_, 1);
v_isSharedCheck_62_ = !lean_is_exclusive(v_x_47_);
if (v_isSharedCheck_62_ == 0)
{
v___x_51_ = v_x_47_;
v_isShared_52_ = v_isSharedCheck_62_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_snd_49_);
lean_inc(v_fst_48_);
lean_dec(v_x_47_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_62_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_53_; lean_object* v_toFun_54_; lean_object* v___x_55_; lean_object* v_toFun_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_60_; 
v___x_53_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_45_);
v_toFun_54_ = lean_ctor_get(v___x_53_, 0);
lean_inc(v_toFun_54_);
lean_dec_ref(v___x_53_);
v___x_55_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_46_);
v_toFun_56_ = lean_ctor_get(v___x_55_, 0);
lean_inc(v_toFun_56_);
lean_dec_ref(v___x_55_);
v___x_57_ = lean_apply_1(v_toFun_54_, v_fst_48_);
v___x_58_ = lean_apply_1(v_toFun_56_, v_snd_49_);
if (v_isShared_52_ == 0)
{
lean_ctor_set(v___x_51_, 1, v___x_58_);
lean_ctor_set(v___x_51_, 0, v___x_57_);
v___x_60_ = v___x_51_;
goto v_reusejp_59_;
}
else
{
lean_object* v_reuseFailAlloc_61_; 
v_reuseFailAlloc_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_61_, 0, v___x_57_);
lean_ctor_set(v_reuseFailAlloc_61_, 1, v___x_58_);
v___x_60_ = v_reuseFailAlloc_61_;
goto v_reusejp_59_;
}
v_reusejp_59_:
{
return v___x_60_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr___redArg(lean_object* v_e_u2081_63_, lean_object* v_e_u2082_64_){
_start:
{
lean_object* v___f_65_; lean_object* v___f_66_; lean_object* v___x_67_; 
lean_inc_ref(v_e_u2082_64_);
lean_inc_ref(v_e_u2081_63_);
v___f_65_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_pprodCongr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_65_, 0, v_e_u2081_63_);
lean_closure_set(v___f_65_, 1, v_e_u2082_64_);
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_pprodCongr___redArg___lam__1), 3, 2);
lean_closure_set(v___f_66_, 0, v_e_u2081_63_);
lean_closure_set(v___f_66_, 1, v_e_u2082_64_);
v___x_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_67_, 0, v___f_65_);
lean_ctor_set(v___x_67_, 1, v___f_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodCongr(lean_object* v_00_u03b1_68_, lean_object* v_00_u03b2_69_, lean_object* v_00_u03b3_70_, lean_object* v_00_u03b4_71_, lean_object* v_e_u2081_72_, lean_object* v_e_u2082_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Equiv_pprodCongr___redArg(v_e_u2081_72_, v_e_u2082_73_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_Equiv_pprodProd___redArg___closed__0(void){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Equiv_pprodEquivProd(lean_box(0), lean_box(0));
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodProd___redArg(lean_object* v_ea_76_, lean_object* v_eb_77_){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_78_ = lp_mathlib_Equiv_pprodCongr___redArg(v_ea_76_, v_eb_77_);
v___x_79_ = lean_obj_once(&lp_mathlib_Equiv_pprodProd___redArg___closed__0, &lp_mathlib_Equiv_pprodProd___redArg___closed__0_once, _init_lp_mathlib_Equiv_pprodProd___redArg___closed__0);
v___x_80_ = lp_mathlib_Equiv_trans___redArg(v___x_78_, v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodProd(lean_object* v_00_u03b1_u2081_81_, lean_object* v_00_u03b2_u2081_82_, lean_object* v_00_u03b1_u2082_83_, lean_object* v_00_u03b2_u2082_84_, lean_object* v_ea_85_, lean_object* v_eb_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_Equiv_pprodProd___redArg(v_ea_85_, v_eb_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPProd___redArg(lean_object* v_ea_88_, lean_object* v_eb_89_){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_90_ = lp_mathlib_Equiv_symm___redArg(v_ea_88_);
v___x_91_ = lp_mathlib_Equiv_symm___redArg(v_eb_89_);
v___x_92_ = lp_mathlib_Equiv_pprodProd___redArg(v___x_90_, v___x_91_);
v___x_93_ = lp_mathlib_Equiv_symm___redArg(v___x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPProd(lean_object* v_00_u03b1_u2082_94_, lean_object* v_00_u03b2_u2082_95_, lean_object* v_00_u03b1_u2081_96_, lean_object* v_00_u03b2_u2081_97_, lean_object* v_ea_98_, lean_object* v_eb_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_Equiv_prodPProd___redArg(v_ea_98_, v_eb_99_);
return v___x_100_;
}
}
static lean_object* _init_lp_mathlib_Equiv_pprodEquivProdPLift___closed__0(void){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Equiv_plift(lean_box(0));
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_Equiv_pprodEquivProdPLift___closed__1(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_102_ = lean_obj_once(&lp_mathlib_Equiv_pprodEquivProdPLift___closed__0, &lp_mathlib_Equiv_pprodEquivProdPLift___closed__0_once, _init_lp_mathlib_Equiv_pprodEquivProdPLift___closed__0);
v___x_103_ = lp_mathlib_Equiv_symm___redArg(v___x_102_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Equiv_pprodEquivProdPLift___closed__2(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = lean_obj_once(&lp_mathlib_Equiv_pprodEquivProdPLift___closed__1, &lp_mathlib_Equiv_pprodEquivProdPLift___closed__1_once, _init_lp_mathlib_Equiv_pprodEquivProdPLift___closed__1);
v___x_105_ = lp_mathlib_Equiv_pprodProd___redArg(v___x_104_, v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pprodEquivProdPLift(lean_object* v_00_u03b1_106_, lean_object* v_00_u03b2_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lean_obj_once(&lp_mathlib_Equiv_pprodEquivProdPLift___closed__2, &lp_mathlib_Equiv_pprodEquivProdPLift___closed__2_once, _init_lp_mathlib_Equiv_pprodEquivProdPLift___closed__2);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg___lam__0(lean_object* v_e_u2081_109_, lean_object* v___y_110_){
_start:
{
lean_object* v_toFun_111_; lean_object* v___x_112_; 
v_toFun_111_ = lean_ctor_get(v_e_u2081_109_, 0);
lean_inc(v_toFun_111_);
lean_dec_ref(v_e_u2081_109_);
v___x_112_ = lean_apply_1(v_toFun_111_, v___y_110_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg___lam__1(lean_object* v_e_u2082_113_, lean_object* v___y_114_){
_start:
{
lean_object* v_toFun_115_; lean_object* v___x_116_; 
v_toFun_115_ = lean_ctor_get(v_e_u2082_113_, 0);
lean_inc(v_toFun_115_);
lean_dec_ref(v_e_u2082_113_);
v___x_116_ = lean_apply_1(v_toFun_115_, v___y_114_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg___lam__2(lean_object* v___x_117_, lean_object* v___y_118_){
_start:
{
lean_object* v_toFun_119_; lean_object* v___x_120_; 
v_toFun_119_ = lean_ctor_get(v___x_117_, 0);
lean_inc(v_toFun_119_);
lean_dec_ref(v___x_117_);
v___x_120_ = lean_apply_1(v_toFun_119_, v___y_118_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object* v_e_u2081_121_, lean_object* v_e_u2082_122_){
_start:
{
lean_object* v___f_123_; lean_object* v___f_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___f_127_; lean_object* v___x_128_; lean_object* v___f_129_; lean_object* v___x_130_; lean_object* v___x_131_; 
lean_inc_ref(v_e_u2081_121_);
v___f_123_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_123_, 0, v_e_u2081_121_);
lean_inc_ref(v_e_u2082_122_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_124_, 0, v_e_u2082_122_);
v___x_125_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_125_, 0, lean_box(0));
lean_closure_set(v___x_125_, 1, lean_box(0));
lean_closure_set(v___x_125_, 2, lean_box(0));
lean_closure_set(v___x_125_, 3, lean_box(0));
lean_closure_set(v___x_125_, 4, v___f_123_);
lean_closure_set(v___x_125_, 5, v___f_124_);
v___x_126_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_121_);
v___f_127_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_127_, 0, v___x_126_);
v___x_128_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_122_);
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongr___redArg___lam__2), 2, 1);
lean_closure_set(v___f_129_, 0, v___x_128_);
v___x_130_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_130_, 0, lean_box(0));
lean_closure_set(v___x_130_, 1, lean_box(0));
lean_closure_set(v___x_130_, 2, lean_box(0));
lean_closure_set(v___x_130_, 3, lean_box(0));
lean_closure_set(v___x_130_, 4, v___f_127_);
lean_closure_set(v___x_130_, 5, v___f_129_);
v___x_131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_125_);
lean_ctor_set(v___x_131_, 1, v___x_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongr(lean_object* v_00_u03b1_u2081_132_, lean_object* v_00_u03b1_u2082_133_, lean_object* v_00_u03b2_u2081_134_, lean_object* v_00_u03b2_u2082_135_, lean_object* v_e_u2081_136_, lean_object* v_e_u2082_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_Equiv_prodCongr___redArg(v_e_u2081_136_, v_e_u2082_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodComm(lean_object* v_00_u03b1_142_, lean_object* v_00_u03b2_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = ((lean_object*)(lp_mathlib_Equiv_prodComm___closed__1));
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodAssoc___lam__0(lean_object* v_p_145_){
_start:
{
lean_object* v_fst_146_; lean_object* v_snd_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_163_; 
v_fst_146_ = lean_ctor_get(v_p_145_, 0);
v_snd_147_ = lean_ctor_get(v_p_145_, 1);
v_isSharedCheck_163_ = !lean_is_exclusive(v_p_145_);
if (v_isSharedCheck_163_ == 0)
{
v___x_149_ = v_p_145_;
v_isShared_150_ = v_isSharedCheck_163_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_snd_147_);
lean_inc(v_fst_146_);
lean_dec(v_p_145_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_163_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v_fst_151_; lean_object* v_snd_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_162_; 
v_fst_151_ = lean_ctor_get(v_fst_146_, 0);
v_snd_152_ = lean_ctor_get(v_fst_146_, 1);
v_isSharedCheck_162_ = !lean_is_exclusive(v_fst_146_);
if (v_isSharedCheck_162_ == 0)
{
v___x_154_ = v_fst_146_;
v_isShared_155_ = v_isSharedCheck_162_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_snd_152_);
lean_inc(v_fst_151_);
lean_dec(v_fst_146_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_162_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 1, v_snd_147_);
lean_ctor_set(v___x_154_, 0, v_snd_152_);
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_snd_152_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v_snd_147_);
v___x_157_ = v_reuseFailAlloc_161_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
lean_object* v___x_159_; 
if (v_isShared_150_ == 0)
{
lean_ctor_set(v___x_149_, 1, v___x_157_);
lean_ctor_set(v___x_149_, 0, v_fst_151_);
v___x_159_ = v___x_149_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v_fst_151_);
lean_ctor_set(v_reuseFailAlloc_160_, 1, v___x_157_);
v___x_159_ = v_reuseFailAlloc_160_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
return v___x_159_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodAssoc___lam__1(lean_object* v_p_164_){
_start:
{
lean_object* v_snd_165_; lean_object* v_fst_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_182_; 
v_snd_165_ = lean_ctor_get(v_p_164_, 1);
v_fst_166_ = lean_ctor_get(v_p_164_, 0);
v_isSharedCheck_182_ = !lean_is_exclusive(v_p_164_);
if (v_isSharedCheck_182_ == 0)
{
v___x_168_ = v_p_164_;
v_isShared_169_ = v_isSharedCheck_182_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_snd_165_);
lean_inc(v_fst_166_);
lean_dec(v_p_164_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_182_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v_fst_170_; lean_object* v_snd_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_181_; 
v_fst_170_ = lean_ctor_get(v_snd_165_, 0);
v_snd_171_ = lean_ctor_get(v_snd_165_, 1);
v_isSharedCheck_181_ = !lean_is_exclusive(v_snd_165_);
if (v_isSharedCheck_181_ == 0)
{
v___x_173_ = v_snd_165_;
v_isShared_174_ = v_isSharedCheck_181_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_snd_171_);
lean_inc(v_fst_170_);
lean_dec(v_snd_165_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_181_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_176_; 
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 1, v_fst_170_);
lean_ctor_set(v___x_173_, 0, v_fst_166_);
v___x_176_ = v___x_173_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v_fst_166_);
lean_ctor_set(v_reuseFailAlloc_180_, 1, v_fst_170_);
v___x_176_ = v_reuseFailAlloc_180_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
lean_object* v___x_178_; 
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 1, v_snd_171_);
lean_ctor_set(v___x_168_, 0, v___x_176_);
v___x_178_ = v___x_168_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v___x_176_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v_snd_171_);
v___x_178_ = v_reuseFailAlloc_179_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
return v___x_178_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodAssoc(lean_object* v_00_u03b1_188_, lean_object* v_00_u03b2_189_, lean_object* v_00_u03b3_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = ((lean_object*)(lp_mathlib_Equiv_prodAssoc___closed__2));
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodProdProdComm___lam__0(lean_object* v_abcd_192_){
_start:
{
lean_object* v_fst_193_; lean_object* v_snd_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_219_; 
v_fst_193_ = lean_ctor_get(v_abcd_192_, 0);
v_snd_194_ = lean_ctor_get(v_abcd_192_, 1);
v_isSharedCheck_219_ = !lean_is_exclusive(v_abcd_192_);
if (v_isSharedCheck_219_ == 0)
{
v___x_196_ = v_abcd_192_;
v_isShared_197_ = v_isSharedCheck_219_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_snd_194_);
lean_inc(v_fst_193_);
lean_dec(v_abcd_192_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_219_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v_fst_198_; lean_object* v_snd_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_218_; 
v_fst_198_ = lean_ctor_get(v_fst_193_, 0);
v_snd_199_ = lean_ctor_get(v_fst_193_, 1);
v_isSharedCheck_218_ = !lean_is_exclusive(v_fst_193_);
if (v_isSharedCheck_218_ == 0)
{
v___x_201_ = v_fst_193_;
v_isShared_202_ = v_isSharedCheck_218_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_snd_199_);
lean_inc(v_fst_198_);
lean_dec(v_fst_193_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_218_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v_fst_203_; lean_object* v_snd_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_217_; 
v_fst_203_ = lean_ctor_get(v_snd_194_, 0);
v_snd_204_ = lean_ctor_get(v_snd_194_, 1);
v_isSharedCheck_217_ = !lean_is_exclusive(v_snd_194_);
if (v_isSharedCheck_217_ == 0)
{
v___x_206_ = v_snd_194_;
v_isShared_207_ = v_isSharedCheck_217_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_snd_204_);
lean_inc(v_fst_203_);
lean_dec(v_snd_194_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_217_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_209_; 
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 1, v_fst_203_);
lean_ctor_set(v___x_206_, 0, v_fst_198_);
v___x_209_ = v___x_206_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v_fst_198_);
lean_ctor_set(v_reuseFailAlloc_216_, 1, v_fst_203_);
v___x_209_ = v_reuseFailAlloc_216_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
lean_object* v___x_211_; 
if (v_isShared_202_ == 0)
{
lean_ctor_set(v___x_201_, 1, v_snd_204_);
lean_ctor_set(v___x_201_, 0, v_snd_199_);
v___x_211_ = v___x_201_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_snd_199_);
lean_ctor_set(v_reuseFailAlloc_215_, 1, v_snd_204_);
v___x_211_ = v_reuseFailAlloc_215_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_213_; 
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 1, v___x_211_);
lean_ctor_set(v___x_196_, 0, v___x_209_);
v___x_213_ = v___x_196_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_214_; 
v_reuseFailAlloc_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_214_, 0, v___x_209_);
lean_ctor_set(v_reuseFailAlloc_214_, 1, v___x_211_);
v___x_213_ = v_reuseFailAlloc_214_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
return v___x_213_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodProdProdComm___lam__1(lean_object* v_acbd_220_){
_start:
{
lean_object* v_fst_221_; lean_object* v_snd_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_247_; 
v_fst_221_ = lean_ctor_get(v_acbd_220_, 0);
v_snd_222_ = lean_ctor_get(v_acbd_220_, 1);
v_isSharedCheck_247_ = !lean_is_exclusive(v_acbd_220_);
if (v_isSharedCheck_247_ == 0)
{
v___x_224_ = v_acbd_220_;
v_isShared_225_ = v_isSharedCheck_247_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_snd_222_);
lean_inc(v_fst_221_);
lean_dec(v_acbd_220_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_247_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v_fst_226_; lean_object* v_snd_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_246_; 
v_fst_226_ = lean_ctor_get(v_fst_221_, 0);
v_snd_227_ = lean_ctor_get(v_fst_221_, 1);
v_isSharedCheck_246_ = !lean_is_exclusive(v_fst_221_);
if (v_isSharedCheck_246_ == 0)
{
v___x_229_ = v_fst_221_;
v_isShared_230_ = v_isSharedCheck_246_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_snd_227_);
lean_inc(v_fst_226_);
lean_dec(v_fst_221_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_246_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v_fst_231_; lean_object* v_snd_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_245_; 
v_fst_231_ = lean_ctor_get(v_snd_222_, 0);
v_snd_232_ = lean_ctor_get(v_snd_222_, 1);
v_isSharedCheck_245_ = !lean_is_exclusive(v_snd_222_);
if (v_isSharedCheck_245_ == 0)
{
v___x_234_ = v_snd_222_;
v_isShared_235_ = v_isSharedCheck_245_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_snd_232_);
lean_inc(v_fst_231_);
lean_dec(v_snd_222_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_245_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_237_; 
if (v_isShared_235_ == 0)
{
lean_ctor_set(v___x_234_, 1, v_fst_231_);
lean_ctor_set(v___x_234_, 0, v_fst_226_);
v___x_237_ = v___x_234_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v_fst_226_);
lean_ctor_set(v_reuseFailAlloc_244_, 1, v_fst_231_);
v___x_237_ = v_reuseFailAlloc_244_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
lean_object* v___x_239_; 
if (v_isShared_230_ == 0)
{
lean_ctor_set(v___x_229_, 1, v_snd_232_);
lean_ctor_set(v___x_229_, 0, v_snd_227_);
v___x_239_ = v___x_229_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_snd_227_);
lean_ctor_set(v_reuseFailAlloc_243_, 1, v_snd_232_);
v___x_239_ = v_reuseFailAlloc_243_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
lean_object* v___x_241_; 
if (v_isShared_225_ == 0)
{
lean_ctor_set(v___x_224_, 1, v___x_239_);
lean_ctor_set(v___x_224_, 0, v___x_237_);
v___x_241_ = v___x_224_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_237_);
lean_ctor_set(v_reuseFailAlloc_242_, 1, v___x_239_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodProdProdComm(lean_object* v_00_u03b1_253_, lean_object* v_00_u03b2_254_, lean_object* v_00_u03b3_255_, lean_object* v_00_u03b4_256_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = ((lean_object*)(lp_mathlib_Equiv_prodProdProdComm___closed__2));
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_curry___lam__0(lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_261_, 0, v___y_259_);
lean_ctor_set(v___x_261_, 1, v___y_260_);
v___x_262_ = lean_apply_1(v___y_258_, v___x_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_curry___lam__1(lean_object* v___y_263_, lean_object* v___y_264_){
_start:
{
lean_object* v_fst_265_; lean_object* v_snd_266_; lean_object* v___x_267_; 
v_fst_265_ = lean_ctor_get(v___y_264_, 0);
lean_inc(v_fst_265_);
v_snd_266_ = lean_ctor_get(v___y_264_, 1);
lean_inc(v_snd_266_);
lean_dec_ref(v___y_264_);
v___x_267_ = lean_apply_2(v___y_263_, v_fst_265_, v_snd_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_curry(lean_object* v_00_u03b1_273_, lean_object* v_00_u03b2_274_, lean_object* v_00_u03b3_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = ((lean_object*)(lp_mathlib_Equiv_curry___closed__2));
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit___lam__0(lean_object* v_p_277_){
_start:
{
lean_object* v_fst_278_; 
v_fst_278_ = lean_ctor_get(v_p_277_, 0);
lean_inc(v_fst_278_);
return v_fst_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit___lam__0___boxed(lean_object* v_p_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_Equiv_prodPUnit___lam__0(v_p_279_);
lean_dec_ref(v_p_279_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit___lam__1(lean_object* v_a_281_){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_282_ = lean_box(0);
v___x_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_283_, 0, v_a_281_);
lean_ctor_set(v___x_283_, 1, v___x_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPUnit(lean_object* v_00_u03b1_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = ((lean_object*)(lp_mathlib_Equiv_prodPUnit___closed__2));
return v___x_290_;
}
}
static lean_object* _init_lp_mathlib_Equiv_punitProd___closed__0(void){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return v___x_291_;
}
}
static lean_object* _init_lp_mathlib_Equiv_punitProd___closed__1(void){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Equiv_prodPUnit(lean_box(0));
return v___x_292_;
}
}
static lean_object* _init_lp_mathlib_Equiv_punitProd___closed__2(void){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_293_ = lean_obj_once(&lp_mathlib_Equiv_punitProd___closed__1, &lp_mathlib_Equiv_punitProd___closed__1_once, _init_lp_mathlib_Equiv_punitProd___closed__1);
v___x_294_ = lean_obj_once(&lp_mathlib_Equiv_punitProd___closed__0, &lp_mathlib_Equiv_punitProd___closed__0_once, _init_lp_mathlib_Equiv_punitProd___closed__0);
v___x_295_ = lp_mathlib_Equiv_trans___redArg(v___x_294_, v___x_293_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_punitProd(lean_object* v_00_u03b1_296_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lean_obj_once(&lp_mathlib_Equiv_punitProd___closed__2, &lp_mathlib_Equiv_punitProd___closed__2_once, _init_lp_mathlib_Equiv_punitProd___closed__2);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit___lam__0(lean_object* v_p_298_){
_start:
{
lean_object* v_fst_299_; 
v_fst_299_ = lean_ctor_get(v_p_298_, 0);
lean_inc(v_fst_299_);
return v_fst_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit___lam__0___boxed(lean_object* v_p_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_Equiv_sigmaPUnit___lam__0(v_p_300_);
lean_dec_ref(v_p_300_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit___lam__1(lean_object* v_a_302_){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_303_ = lean_box(0);
v___x_304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_304_, 0, v_a_302_);
lean_ctor_set(v___x_304_, 1, v___x_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaPUnit(lean_object* v_00_u03b1_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = ((lean_object*)(lp_mathlib_Equiv_sigmaPUnit___closed__2));
return v___x_311_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodUnique___redArg___closed__0(void){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodUnique___redArg(lean_object* v_inst_313_){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_314_ = lean_obj_once(&lp_mathlib_Equiv_prodUnique___redArg___closed__0, &lp_mathlib_Equiv_prodUnique___redArg___closed__0_once, _init_lp_mathlib_Equiv_prodUnique___redArg___closed__0);
v___x_315_ = lp_mathlib_Equiv_equivPUnit___redArg(v_inst_313_);
v___x_316_ = lp_mathlib_Equiv_prodCongr___redArg(v___x_314_, v___x_315_);
v___x_317_ = lean_obj_once(&lp_mathlib_Equiv_punitProd___closed__1, &lp_mathlib_Equiv_punitProd___closed__1_once, _init_lp_mathlib_Equiv_punitProd___closed__1);
v___x_318_ = lp_mathlib_Equiv_trans___redArg(v___x_316_, v___x_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodUnique(lean_object* v_00_u03b1_319_, lean_object* v_00_u03b2_320_, lean_object* v_inst_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_Equiv_prodUnique___redArg(v_inst_321_);
return v___x_322_;
}
}
static lean_object* _init_lp_mathlib_Equiv_uniqueProd___redArg___closed__0(void){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_Equiv_punitProd(lean_box(0));
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueProd___redArg(lean_object* v_inst_324_){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_325_ = lp_mathlib_Equiv_equivPUnit___redArg(v_inst_324_);
v___x_326_ = lean_obj_once(&lp_mathlib_Equiv_prodUnique___redArg___closed__0, &lp_mathlib_Equiv_prodUnique___redArg___closed__0_once, _init_lp_mathlib_Equiv_prodUnique___redArg___closed__0);
v___x_327_ = lp_mathlib_Equiv_prodCongr___redArg(v___x_325_, v___x_326_);
v___x_328_ = lean_obj_once(&lp_mathlib_Equiv_uniqueProd___redArg___closed__0, &lp_mathlib_Equiv_uniqueProd___redArg___closed__0_once, _init_lp_mathlib_Equiv_uniqueProd___redArg___closed__0);
v___x_329_ = lp_mathlib_Equiv_trans___redArg(v___x_327_, v___x_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueProd(lean_object* v_00_u03b1_330_, lean_object* v_00_u03b2_331_, lean_object* v_inst_332_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lp_mathlib_Equiv_uniqueProd___redArg(v_inst_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaUnique___redArg___lam__0(lean_object* v_inst_334_, lean_object* v_a_335_){
_start:
{
lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_336_ = lean_apply_1(v_inst_334_, v_a_335_);
v___x_337_ = lp_mathlib_Equiv_equivPUnit___redArg(v___x_336_);
return v___x_337_;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaUnique___redArg___closed__0(void){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lp_mathlib_Equiv_sigmaPUnit(lean_box(0));
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaUnique___redArg(lean_object* v_inst_339_){
_start:
{
lean_object* v___f_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; 
v___f_340_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaUnique___redArg___lam__0), 2, 1);
lean_closure_set(v___f_340_, 0, v_inst_339_);
v___x_341_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v___f_340_);
v___x_342_ = lean_obj_once(&lp_mathlib_Equiv_sigmaUnique___redArg___closed__0, &lp_mathlib_Equiv_sigmaUnique___redArg___closed__0_once, _init_lp_mathlib_Equiv_sigmaUnique___redArg___closed__0);
v___x_343_ = lp_mathlib_Equiv_trans___redArg(v___x_341_, v___x_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaUnique(lean_object* v_00_u03b1_344_, lean_object* v_00_u03b2_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_Equiv_sigmaUnique___redArg(v_inst_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg___lam__0(lean_object* v_p_348_){
_start:
{
lean_object* v_snd_349_; 
v_snd_349_ = lean_ctor_get(v_p_348_, 1);
lean_inc(v_snd_349_);
return v_snd_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg___lam__0___boxed(lean_object* v_p_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_Equiv_uniqueSigma___redArg___lam__0(v_p_350_);
lean_dec_ref(v_p_350_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg___lam__1(lean_object* v_inst_352_, lean_object* v_b_353_){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_354_, 0, v_inst_352_);
lean_ctor_set(v___x_354_, 1, v_b_353_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma___redArg(lean_object* v_inst_356_){
_start:
{
lean_object* v___f_357_; lean_object* v___f_358_; lean_object* v___x_359_; 
v___f_357_ = ((lean_object*)(lp_mathlib_Equiv_uniqueSigma___redArg___closed__0));
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_uniqueSigma___redArg___lam__1), 2, 1);
lean_closure_set(v___f_358_, 0, v_inst_356_);
v___x_359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_359_, 0, v___f_357_);
lean_ctor_set(v___x_359_, 1, v___f_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueSigma(lean_object* v_00_u03b1_360_, lean_object* v_00_u03b2_361_, lean_object* v_inst_362_){
_start:
{
lean_object* v___x_363_; 
v___x_363_ = lp_mathlib_Equiv_uniqueSigma___redArg(v_inst_362_);
return v___x_363_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodEmpty___closed__0(void){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmpty(lean_object* v_00_u03b1_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lean_obj_once(&lp_mathlib_Equiv_prodEmpty___closed__0, &lp_mathlib_Equiv_prodEmpty___closed__0_once, _init_lp_mathlib_Equiv_prodEmpty___closed__0);
return v___x_366_;
}
}
static lean_object* _init_lp_mathlib_Equiv_emptyProd___closed__0(void){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptyProd(lean_object* v_00_u03b1_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lean_obj_once(&lp_mathlib_Equiv_emptyProd___closed__0, &lp_mathlib_Equiv_emptyProd___closed__0_once, _init_lp_mathlib_Equiv_emptyProd___closed__0);
return v___x_369_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodPEmpty___closed__0(void){
_start:
{
lean_object* v___x_370_; 
v___x_370_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPEmpty(lean_object* v_00_u03b1_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lean_obj_once(&lp_mathlib_Equiv_prodPEmpty___closed__0, &lp_mathlib_Equiv_prodPEmpty___closed__0_once, _init_lp_mathlib_Equiv_prodPEmpty___closed__0);
return v___x_372_;
}
}
static lean_object* _init_lp_mathlib_Equiv_pemptyProd___closed__0(void){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_pemptyProd(lean_object* v_00_u03b1_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lean_obj_once(&lp_mathlib_Equiv_pemptyProd___closed__0, &lp_mathlib_Equiv_pemptyProd___closed__0_once, _init_lp_mathlib_Equiv_pemptyProd___closed__0);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft___redArg___lam__0(lean_object* v_e_376_, lean_object* v_ab_377_){
_start:
{
lean_object* v_fst_378_; lean_object* v_snd_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_389_; 
v_fst_378_ = lean_ctor_get(v_ab_377_, 0);
v_snd_379_ = lean_ctor_get(v_ab_377_, 1);
v_isSharedCheck_389_ = !lean_is_exclusive(v_ab_377_);
if (v_isSharedCheck_389_ == 0)
{
v___x_381_ = v_ab_377_;
v_isShared_382_ = v_isSharedCheck_389_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_snd_379_);
lean_inc(v_fst_378_);
lean_dec(v_ab_377_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_389_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_383_; lean_object* v_toFun_384_; lean_object* v___x_385_; lean_object* v___x_387_; 
lean_inc(v_snd_379_);
v___x_383_ = lean_apply_1(v_e_376_, v_snd_379_);
v_toFun_384_ = lean_ctor_get(v___x_383_, 0);
lean_inc(v_toFun_384_);
lean_dec_ref(v___x_383_);
v___x_385_ = lean_apply_1(v_toFun_384_, v_fst_378_);
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 0, v___x_385_);
v___x_387_ = v___x_381_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v___x_385_);
lean_ctor_set(v_reuseFailAlloc_388_, 1, v_snd_379_);
v___x_387_ = v_reuseFailAlloc_388_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
return v___x_387_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft___redArg___lam__1(lean_object* v_e_390_, lean_object* v_ab_391_){
_start:
{
lean_object* v_fst_392_; lean_object* v_snd_393_; lean_object* v___x_395_; uint8_t v_isShared_396_; uint8_t v_isSharedCheck_404_; 
v_fst_392_ = lean_ctor_get(v_ab_391_, 0);
v_snd_393_ = lean_ctor_get(v_ab_391_, 1);
v_isSharedCheck_404_ = !lean_is_exclusive(v_ab_391_);
if (v_isSharedCheck_404_ == 0)
{
v___x_395_ = v_ab_391_;
v_isShared_396_ = v_isSharedCheck_404_;
goto v_resetjp_394_;
}
else
{
lean_inc(v_snd_393_);
lean_inc(v_fst_392_);
lean_dec(v_ab_391_);
v___x_395_ = lean_box(0);
v_isShared_396_ = v_isSharedCheck_404_;
goto v_resetjp_394_;
}
v_resetjp_394_:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v_toFun_399_; lean_object* v___x_400_; lean_object* v___x_402_; 
lean_inc(v_snd_393_);
v___x_397_ = lean_apply_1(v_e_390_, v_snd_393_);
v___x_398_ = lp_mathlib_Equiv_symm___redArg(v___x_397_);
v_toFun_399_ = lean_ctor_get(v___x_398_, 0);
lean_inc(v_toFun_399_);
lean_dec_ref(v___x_398_);
v___x_400_ = lean_apply_1(v_toFun_399_, v_fst_392_);
if (v_isShared_396_ == 0)
{
lean_ctor_set(v___x_395_, 0, v___x_400_);
v___x_402_ = v___x_395_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v___x_400_);
lean_ctor_set(v_reuseFailAlloc_403_, 1, v_snd_393_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft___redArg(lean_object* v_e_405_){
_start:
{
lean_object* v___f_406_; lean_object* v___f_407_; lean_object* v___x_408_; 
lean_inc_ref(v_e_405_);
v___f_406_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongrLeft___redArg___lam__0), 2, 1);
lean_closure_set(v___f_406_, 0, v_e_405_);
v___f_407_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongrLeft___redArg___lam__1), 2, 1);
lean_closure_set(v___f_407_, 0, v_e_405_);
v___x_408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_408_, 0, v___f_406_);
lean_ctor_set(v___x_408_, 1, v___f_407_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrLeft(lean_object* v_00_u03b1_u2081_409_, lean_object* v_00_u03b2_u2081_410_, lean_object* v_00_u03b2_u2082_411_, lean_object* v_e_412_){
_start:
{
lean_object* v___x_413_; 
v___x_413_ = lp_mathlib_Equiv_prodCongrLeft___redArg(v_e_412_);
return v___x_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight___redArg___lam__0(lean_object* v_e_414_, lean_object* v_ab_415_){
_start:
{
lean_object* v_fst_416_; lean_object* v_snd_417_; lean_object* v___x_419_; uint8_t v_isShared_420_; uint8_t v_isSharedCheck_427_; 
v_fst_416_ = lean_ctor_get(v_ab_415_, 0);
v_snd_417_ = lean_ctor_get(v_ab_415_, 1);
v_isSharedCheck_427_ = !lean_is_exclusive(v_ab_415_);
if (v_isSharedCheck_427_ == 0)
{
v___x_419_ = v_ab_415_;
v_isShared_420_ = v_isSharedCheck_427_;
goto v_resetjp_418_;
}
else
{
lean_inc(v_snd_417_);
lean_inc(v_fst_416_);
lean_dec(v_ab_415_);
v___x_419_ = lean_box(0);
v_isShared_420_ = v_isSharedCheck_427_;
goto v_resetjp_418_;
}
v_resetjp_418_:
{
lean_object* v___x_421_; lean_object* v_toFun_422_; lean_object* v___x_423_; lean_object* v___x_425_; 
lean_inc(v_fst_416_);
v___x_421_ = lean_apply_1(v_e_414_, v_fst_416_);
v_toFun_422_ = lean_ctor_get(v___x_421_, 0);
lean_inc(v_toFun_422_);
lean_dec_ref(v___x_421_);
v___x_423_ = lean_apply_1(v_toFun_422_, v_snd_417_);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 1, v___x_423_);
v___x_425_ = v___x_419_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v_fst_416_);
lean_ctor_set(v_reuseFailAlloc_426_, 1, v___x_423_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight___redArg___lam__1(lean_object* v_e_428_, lean_object* v_ab_429_){
_start:
{
lean_object* v_fst_430_; lean_object* v_snd_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_442_; 
v_fst_430_ = lean_ctor_get(v_ab_429_, 0);
v_snd_431_ = lean_ctor_get(v_ab_429_, 1);
v_isSharedCheck_442_ = !lean_is_exclusive(v_ab_429_);
if (v_isSharedCheck_442_ == 0)
{
v___x_433_ = v_ab_429_;
v_isShared_434_ = v_isSharedCheck_442_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_snd_431_);
lean_inc(v_fst_430_);
lean_dec(v_ab_429_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_442_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v_toFun_437_; lean_object* v___x_438_; lean_object* v___x_440_; 
lean_inc(v_fst_430_);
v___x_435_ = lean_apply_1(v_e_428_, v_fst_430_);
v___x_436_ = lp_mathlib_Equiv_symm___redArg(v___x_435_);
v_toFun_437_ = lean_ctor_get(v___x_436_, 0);
lean_inc(v_toFun_437_);
lean_dec_ref(v___x_436_);
v___x_438_ = lean_apply_1(v_toFun_437_, v_snd_431_);
if (v_isShared_434_ == 0)
{
lean_ctor_set(v___x_433_, 1, v___x_438_);
v___x_440_ = v___x_433_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v_fst_430_);
lean_ctor_set(v_reuseFailAlloc_441_, 1, v___x_438_);
v___x_440_ = v_reuseFailAlloc_441_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
return v___x_440_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight___redArg(lean_object* v_e_443_){
_start:
{
lean_object* v___f_444_; lean_object* v___f_445_; lean_object* v___x_446_; 
lean_inc_ref(v_e_443_);
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongrRight___redArg___lam__0), 2, 1);
lean_closure_set(v___f_444_, 0, v_e_443_);
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodCongrRight___redArg___lam__1), 2, 1);
lean_closure_set(v___f_445_, 0, v_e_443_);
v___x_446_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_446_, 0, v___f_444_);
lean_ctor_set(v___x_446_, 1, v___f_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodCongrRight(lean_object* v_00_u03b1_u2081_447_, lean_object* v_00_u03b2_u2081_448_, lean_object* v_00_u03b2_u2082_449_, lean_object* v_e_450_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lp_mathlib_Equiv_prodCongrRight___redArg(v_e_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear___redArg___lam__0(lean_object* v_e_u2081_452_, lean_object* v_e_u2082_453_, lean_object* v_x_454_){
_start:
{
lean_object* v_fst_455_; lean_object* v_snd_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_468_; 
v_fst_455_ = lean_ctor_get(v_x_454_, 0);
v_snd_456_ = lean_ctor_get(v_x_454_, 1);
v_isSharedCheck_468_ = !lean_is_exclusive(v_x_454_);
if (v_isSharedCheck_468_ == 0)
{
v___x_458_ = v_x_454_;
v_isShared_459_ = v_isSharedCheck_468_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_snd_456_);
lean_inc(v_fst_455_);
lean_dec(v_x_454_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_468_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v_toFun_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v_toFun_463_; lean_object* v___x_464_; lean_object* v___x_466_; 
v_toFun_460_ = lean_ctor_get(v_e_u2081_452_, 0);
lean_inc(v_toFun_460_);
lean_dec_ref(v_e_u2081_452_);
lean_inc(v_fst_455_);
v___x_461_ = lean_apply_1(v_toFun_460_, v_fst_455_);
v___x_462_ = lean_apply_1(v_e_u2082_453_, v_fst_455_);
v_toFun_463_ = lean_ctor_get(v___x_462_, 0);
lean_inc(v_toFun_463_);
lean_dec_ref(v___x_462_);
v___x_464_ = lean_apply_1(v_toFun_463_, v_snd_456_);
if (v_isShared_459_ == 0)
{
lean_ctor_set(v___x_458_, 1, v___x_464_);
lean_ctor_set(v___x_458_, 0, v___x_461_);
v___x_466_ = v___x_458_;
goto v_reusejp_465_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v___x_461_);
lean_ctor_set(v_reuseFailAlloc_467_, 1, v___x_464_);
v___x_466_ = v_reuseFailAlloc_467_;
goto v_reusejp_465_;
}
v_reusejp_465_:
{
return v___x_466_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear___redArg___lam__1(lean_object* v_e_u2081_469_, lean_object* v_e_u2082_470_, lean_object* v_y_471_){
_start:
{
lean_object* v_fst_472_; lean_object* v_snd_473_; lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_487_; 
v_fst_472_ = lean_ctor_get(v_y_471_, 0);
v_snd_473_ = lean_ctor_get(v_y_471_, 1);
v_isSharedCheck_487_ = !lean_is_exclusive(v_y_471_);
if (v_isSharedCheck_487_ == 0)
{
v___x_475_ = v_y_471_;
v_isShared_476_ = v_isSharedCheck_487_;
goto v_resetjp_474_;
}
else
{
lean_inc(v_snd_473_);
lean_inc(v_fst_472_);
lean_dec(v_y_471_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_487_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___x_477_; lean_object* v_toFun_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v_toFun_482_; lean_object* v___x_483_; lean_object* v___x_485_; 
v___x_477_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_469_);
v_toFun_478_ = lean_ctor_get(v___x_477_, 0);
lean_inc(v_toFun_478_);
lean_dec_ref(v___x_477_);
v___x_479_ = lean_apply_1(v_toFun_478_, v_fst_472_);
lean_inc(v___x_479_);
v___x_480_ = lean_apply_1(v_e_u2082_470_, v___x_479_);
v___x_481_ = lp_mathlib_Equiv_symm___redArg(v___x_480_);
v_toFun_482_ = lean_ctor_get(v___x_481_, 0);
lean_inc(v_toFun_482_);
lean_dec_ref(v___x_481_);
v___x_483_ = lean_apply_1(v_toFun_482_, v_snd_473_);
if (v_isShared_476_ == 0)
{
lean_ctor_set(v___x_475_, 1, v___x_483_);
lean_ctor_set(v___x_475_, 0, v___x_479_);
v___x_485_ = v___x_475_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v___x_479_);
lean_ctor_set(v_reuseFailAlloc_486_, 1, v___x_483_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear___redArg(lean_object* v_e_u2081_488_, lean_object* v_e_u2082_489_){
_start:
{
lean_object* v___f_490_; lean_object* v___f_491_; lean_object* v___x_492_; 
lean_inc_ref(v_e_u2082_489_);
lean_inc_ref(v_e_u2081_488_);
v___f_490_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodShear___redArg___lam__0), 3, 2);
lean_closure_set(v___f_490_, 0, v_e_u2081_488_);
lean_closure_set(v___f_490_, 1, v_e_u2082_489_);
v___f_491_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodShear___redArg___lam__1), 3, 2);
lean_closure_set(v___f_491_, 0, v_e_u2081_488_);
lean_closure_set(v___f_491_, 1, v_e_u2082_489_);
v___x_492_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_492_, 0, v___f_490_);
lean_ctor_set(v___x_492_, 1, v___f_491_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodShear(lean_object* v_00_u03b1_u2081_493_, lean_object* v_00_u03b1_u2082_494_, lean_object* v_00_u03b2_u2081_495_, lean_object* v_00_u03b2_u2082_496_, lean_object* v_e_u2081_497_, lean_object* v_e_u2082_498_){
_start:
{
lean_object* v___x_499_; 
v___x_499_ = lp_mathlib_Equiv_prodShear___redArg(v_e_u2081_497_, v_e_u2082_498_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight___redArg___lam__0(lean_object* v_inst_500_, lean_object* v_a_501_, lean_object* v_e_502_, lean_object* v_ab_503_){
_start:
{
lean_object* v_fst_504_; lean_object* v_snd_505_; lean_object* v___x_506_; uint8_t v___x_507_; 
v_fst_504_ = lean_ctor_get(v_ab_503_, 0);
v_snd_505_ = lean_ctor_get(v_ab_503_, 1);
lean_inc(v_a_501_);
lean_inc(v_fst_504_);
v___x_506_ = lean_apply_2(v_inst_500_, v_fst_504_, v_a_501_);
v___x_507_ = lean_unbox(v___x_506_);
if (v___x_507_ == 0)
{
lean_dec_ref(v_e_502_);
lean_dec(v_a_501_);
return v_ab_503_;
}
else
{
lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_516_; 
lean_inc(v_snd_505_);
v_isSharedCheck_516_ = !lean_is_exclusive(v_ab_503_);
if (v_isSharedCheck_516_ == 0)
{
lean_object* v_unused_517_; lean_object* v_unused_518_; 
v_unused_517_ = lean_ctor_get(v_ab_503_, 1);
lean_dec(v_unused_517_);
v_unused_518_ = lean_ctor_get(v_ab_503_, 0);
lean_dec(v_unused_518_);
v___x_509_ = v_ab_503_;
v_isShared_510_ = v_isSharedCheck_516_;
goto v_resetjp_508_;
}
else
{
lean_dec(v_ab_503_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_516_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v_toFun_511_; lean_object* v___x_512_; lean_object* v___x_514_; 
v_toFun_511_ = lean_ctor_get(v_e_502_, 0);
lean_inc(v_toFun_511_);
lean_dec_ref(v_e_502_);
v___x_512_ = lean_apply_1(v_toFun_511_, v_snd_505_);
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 1, v___x_512_);
lean_ctor_set(v___x_509_, 0, v_a_501_);
v___x_514_ = v___x_509_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v_a_501_);
lean_ctor_set(v_reuseFailAlloc_515_, 1, v___x_512_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight___redArg___lam__1(lean_object* v_inst_519_, lean_object* v_a_520_, lean_object* v_e_521_, lean_object* v_ab_522_){
_start:
{
lean_object* v_fst_523_; lean_object* v_snd_524_; lean_object* v___x_525_; uint8_t v___x_526_; 
v_fst_523_ = lean_ctor_get(v_ab_522_, 0);
v_snd_524_ = lean_ctor_get(v_ab_522_, 1);
lean_inc(v_a_520_);
lean_inc(v_fst_523_);
v___x_525_ = lean_apply_2(v_inst_519_, v_fst_523_, v_a_520_);
v___x_526_ = lean_unbox(v___x_525_);
if (v___x_526_ == 0)
{
lean_dec_ref(v_e_521_);
lean_dec(v_a_520_);
return v_ab_522_;
}
else
{
lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_536_; 
lean_inc(v_snd_524_);
v_isSharedCheck_536_ = !lean_is_exclusive(v_ab_522_);
if (v_isSharedCheck_536_ == 0)
{
lean_object* v_unused_537_; lean_object* v_unused_538_; 
v_unused_537_ = lean_ctor_get(v_ab_522_, 1);
lean_dec(v_unused_537_);
v_unused_538_ = lean_ctor_get(v_ab_522_, 0);
lean_dec(v_unused_538_);
v___x_528_ = v_ab_522_;
v_isShared_529_ = v_isSharedCheck_536_;
goto v_resetjp_527_;
}
else
{
lean_dec(v_ab_522_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_536_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v___x_530_; lean_object* v_toFun_531_; lean_object* v___x_532_; lean_object* v___x_534_; 
v___x_530_ = lp_mathlib_Equiv_symm___redArg(v_e_521_);
v_toFun_531_ = lean_ctor_get(v___x_530_, 0);
lean_inc(v_toFun_531_);
lean_dec_ref(v___x_530_);
v___x_532_ = lean_apply_1(v_toFun_531_, v_snd_524_);
if (v_isShared_529_ == 0)
{
lean_ctor_set(v___x_528_, 1, v___x_532_);
lean_ctor_set(v___x_528_, 0, v_a_520_);
v___x_534_ = v___x_528_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_520_);
lean_ctor_set(v_reuseFailAlloc_535_, 1, v___x_532_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight___redArg(lean_object* v_inst_539_, lean_object* v_a_540_, lean_object* v_e_541_){
_start:
{
lean_object* v___f_542_; lean_object* v___f_543_; lean_object* v___x_544_; 
lean_inc_ref(v_e_541_);
lean_inc(v_a_540_);
lean_inc_ref(v_inst_539_);
v___f_542_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_prodExtendRight___redArg___lam__0), 4, 3);
lean_closure_set(v___f_542_, 0, v_inst_539_);
lean_closure_set(v___f_542_, 1, v_a_540_);
lean_closure_set(v___f_542_, 2, v_e_541_);
v___f_543_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_prodExtendRight___redArg___lam__1), 4, 3);
lean_closure_set(v___f_543_, 0, v_inst_539_);
lean_closure_set(v___f_543_, 1, v_a_540_);
lean_closure_set(v___f_543_, 2, v_e_541_);
v___x_544_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_544_, 0, v___f_542_);
lean_ctor_set(v___x_544_, 1, v___f_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_prodExtendRight(lean_object* v_00_u03b1_u2081_545_, lean_object* v_00_u03b2_u2081_546_, lean_object* v_inst_547_, lean_object* v_a_548_, lean_object* v_e_549_){
_start:
{
lean_object* v___x_550_; 
v___x_550_ = lp_mathlib_Equiv_Perm_prodExtendRight___redArg(v_inst_547_, v_a_548_, v_e_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__0(lean_object* v_f_551_, lean_object* v_c_552_){
_start:
{
lean_object* v___x_553_; lean_object* v_fst_554_; 
v___x_553_ = lean_apply_1(v_f_551_, v_c_552_);
v_fst_554_ = lean_ctor_get(v___x_553_, 0);
lean_inc(v_fst_554_);
lean_dec_ref(v___x_553_);
return v_fst_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__1(lean_object* v_f_555_, lean_object* v_c_556_){
_start:
{
lean_object* v___x_557_; lean_object* v_snd_558_; 
v___x_557_ = lean_apply_1(v_f_555_, v_c_556_);
v_snd_558_ = lean_ctor_get(v___x_557_, 1);
lean_inc(v_snd_558_);
lean_dec_ref(v___x_557_);
return v_snd_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__2(lean_object* v_f_559_){
_start:
{
lean_object* v___f_560_; lean_object* v___f_561_; lean_object* v___x_562_; 
lean_inc_ref(v_f_559_);
v___f_560_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__0), 2, 1);
lean_closure_set(v___f_560_, 0, v_f_559_);
v___f_561_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__1), 2, 1);
lean_closure_set(v___f_561_, 0, v_f_559_);
v___x_562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_562_, 0, v___f_560_);
lean_ctor_set(v___x_562_, 1, v___f_561_);
return v___x_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow___lam__3(lean_object* v_p_563_, lean_object* v_c_564_){
_start:
{
lean_object* v_fst_565_; lean_object* v_snd_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_575_; 
v_fst_565_ = lean_ctor_get(v_p_563_, 0);
v_snd_566_ = lean_ctor_get(v_p_563_, 1);
v_isSharedCheck_575_ = !lean_is_exclusive(v_p_563_);
if (v_isSharedCheck_575_ == 0)
{
v___x_568_ = v_p_563_;
v_isShared_569_ = v_isSharedCheck_575_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_snd_566_);
lean_inc(v_fst_565_);
lean_dec(v_p_563_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_575_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_573_; 
lean_inc(v_c_564_);
v___x_570_ = lean_apply_1(v_fst_565_, v_c_564_);
v___x_571_ = lean_apply_1(v_snd_566_, v_c_564_);
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 1, v___x_571_);
lean_ctor_set(v___x_568_, 0, v___x_570_);
v___x_573_ = v___x_568_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v___x_570_);
lean_ctor_set(v_reuseFailAlloc_574_, 1, v___x_571_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
return v___x_573_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_arrowProdEquivProdArrow(lean_object* v_00_u03b1_581_, lean_object* v_00_u03b2_582_, lean_object* v_00_u03b3_583_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = ((lean_object*)(lp_mathlib_Equiv_arrowProdEquivProdArrow___closed__2));
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__0(lean_object* v_f_585_, lean_object* v_i_586_){
_start:
{
lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_587_, 0, v_i_586_);
v___x_588_ = lean_apply_1(v_f_585_, v___x_587_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__1(lean_object* v_f_589_, lean_object* v_i_x27_590_){
_start:
{
lean_object* v___x_591_; lean_object* v___x_592_; 
v___x_591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_591_, 0, v_i_x27_590_);
v___x_592_ = lean_apply_1(v_f_589_, v___x_591_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__2(lean_object* v_f_593_){
_start:
{
lean_object* v___f_594_; lean_object* v___f_595_; lean_object* v___x_596_; 
lean_inc(v_f_593_);
v___f_594_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumPiEquivProdPi___lam__0), 2, 1);
lean_closure_set(v___f_594_, 0, v_f_593_);
v___f_595_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumPiEquivProdPi___lam__1), 2, 1);
lean_closure_set(v___f_595_, 0, v_f_593_);
v___x_596_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_596_, 0, v___f_594_);
lean_ctor_set(v___x_596_, 1, v___f_595_);
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi___lam__3(lean_object* v_g_597_, lean_object* v_t_598_){
_start:
{
lean_object* v_fst_599_; lean_object* v_snd_600_; lean_object* v___x_601_; 
v_fst_599_ = lean_ctor_get(v_g_597_, 0);
lean_inc(v_fst_599_);
v_snd_600_ = lean_ctor_get(v_g_597_, 1);
lean_inc(v_snd_600_);
lean_dec_ref(v_g_597_);
v___x_601_ = lp_mathlib_Sum_rec___redArg_00___x40_Mathlib_Util_CompileInductive_1115872101____hygCtx___hyg_3_(v_fst_599_, v_snd_600_, v_t_598_);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPiEquivProdPi(lean_object* v_00_u03b9_607_, lean_object* v_00_u03b9_x27_608_, lean_object* v_00_u03c0_609_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = ((lean_object*)(lp_mathlib_Equiv_sumPiEquivProdPi___closed__2));
return v___x_610_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodPiEquivSumPi___closed__0(void){
_start:
{
lean_object* v___x_611_; 
v___x_611_ = lp_mathlib_Equiv_sumPiEquivProdPi(lean_box(0), lean_box(0), lean_box(0));
return v___x_611_;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodPiEquivSumPi___closed__1(void){
_start:
{
lean_object* v___x_612_; lean_object* v___x_613_; 
v___x_612_ = lean_obj_once(&lp_mathlib_Equiv_prodPiEquivSumPi___closed__0, &lp_mathlib_Equiv_prodPiEquivSumPi___closed__0_once, _init_lp_mathlib_Equiv_prodPiEquivSumPi___closed__0);
v___x_613_ = lp_mathlib_Equiv_symm___redArg(v___x_612_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodPiEquivSumPi(lean_object* v_00_u03b9_614_, lean_object* v_00_u03b9_x27_615_, lean_object* v_00_u03c0_616_, lean_object* v_00_u03c0_x27_617_){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = lean_obj_once(&lp_mathlib_Equiv_prodPiEquivSumPi___closed__1, &lp_mathlib_Equiv_prodPiEquivSumPi___closed__1_once, _init_lp_mathlib_Equiv_prodPiEquivSumPi___closed__1);
return v___x_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__0(lean_object* v_f_619_, lean_object* v___y_620_){
_start:
{
lean_object* v___x_621_; lean_object* v___x_622_; 
v___x_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_621_, 0, v___y_620_);
v___x_622_ = lean_apply_1(v_f_619_, v___x_621_);
return v___x_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__1(lean_object* v_f_623_, lean_object* v___y_624_){
_start:
{
lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_625_, 0, v___y_624_);
v___x_626_ = lean_apply_1(v_f_623_, v___x_625_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__2(lean_object* v_f_627_){
_start:
{
lean_object* v___f_628_; lean_object* v___f_629_; lean_object* v___x_630_; 
lean_inc(v_f_627_);
v___f_628_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__0), 2, 1);
lean_closure_set(v___f_628_, 0, v_f_627_);
v___f_629_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__1), 2, 1);
lean_closure_set(v___f_629_, 0, v_f_627_);
v___x_630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_630_, 0, v___f_629_);
lean_ctor_set(v___x_630_, 1, v___f_628_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow___lam__3(lean_object* v_p_631_, lean_object* v___y_632_){
_start:
{
lean_object* v_fst_633_; lean_object* v_snd_634_; lean_object* v___x_635_; 
v_fst_633_ = lean_ctor_get(v_p_631_, 0);
lean_inc(v_fst_633_);
v_snd_634_ = lean_ctor_get(v_p_631_, 1);
lean_inc(v_snd_634_);
lean_dec_ref(v_p_631_);
v___x_635_ = l_Sum_elim___redArg(v_fst_633_, v_snd_634_, v___y_632_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumArrowEquivProdArrow(lean_object* v_00_u03b1_641_, lean_object* v_00_u03b2_642_, lean_object* v_00_u03b3_643_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = ((lean_object*)(lp_mathlib_Equiv_sumArrowEquivProdArrow___closed__2));
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__0(lean_object* v_snd_645_, lean_object* v_x_646_){
_start:
{
lean_object* v___x_647_; 
v___x_647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_647_, 0, v_x_646_);
lean_ctor_set(v___x_647_, 1, v_snd_645_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__2(lean_object* v_p_648_){
_start:
{
lean_object* v_fst_649_; lean_object* v_snd_650_; lean_object* v___f_651_; lean_object* v___x_652_; 
v_fst_649_ = lean_ctor_get(v_p_648_, 0);
lean_inc(v_fst_649_);
v_snd_650_ = lean_ctor_get(v_p_648_, 1);
lean_inc(v_snd_650_);
lean_dec_ref(v_p_648_);
v___f_651_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumProdDistrib___lam__0), 2, 1);
lean_closure_set(v___f_651_, 0, v_snd_650_);
lean_inc_ref(v___f_651_);
v___x_652_ = l_Sum_map___redArg(v___f_651_, v___f_651_, v_fst_649_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__1(lean_object* v_val_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_654_, 0, v_val_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__3(lean_object* v_val_655_){
_start:
{
lean_object* v___x_656_; 
v___x_656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_656_, 0, v_val_655_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__4(lean_object* v___y_657_){
_start:
{
lean_inc(v___y_657_);
return v___y_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__4___boxed(lean_object* v___y_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_mathlib_Equiv_sumProdDistrib___lam__4(v___y_658_);
lean_dec(v___y_658_);
return v_res_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib___lam__5(lean_object* v___f_660_, lean_object* v___f_661_, lean_object* v___f_662_, lean_object* v_s_663_){
_start:
{
lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; 
lean_inc(v___f_661_);
v___x_664_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_664_, 0, lean_box(0));
lean_closure_set(v___x_664_, 1, lean_box(0));
lean_closure_set(v___x_664_, 2, lean_box(0));
lean_closure_set(v___x_664_, 3, lean_box(0));
lean_closure_set(v___x_664_, 4, v___f_660_);
lean_closure_set(v___x_664_, 5, v___f_661_);
v___x_665_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_665_, 0, lean_box(0));
lean_closure_set(v___x_665_, 1, lean_box(0));
lean_closure_set(v___x_665_, 2, lean_box(0));
lean_closure_set(v___x_665_, 3, lean_box(0));
lean_closure_set(v___x_665_, 4, v___f_662_);
lean_closure_set(v___x_665_, 5, v___f_661_);
v___x_666_ = l_Sum_elim___redArg(v___x_664_, v___x_665_, v_s_663_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumProdDistrib(lean_object* v_00_u03b1_678_, lean_object* v_00_u03b2_679_, lean_object* v_00_u03b3_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = ((lean_object*)(lp_mathlib_Equiv_sumProdDistrib___closed__5));
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaProdDistrib___lam__0(lean_object* v_p_682_){
_start:
{
lean_object* v_fst_683_; lean_object* v_snd_684_; lean_object* v___x_686_; uint8_t v_isShared_687_; uint8_t v_isSharedCheck_700_; 
v_fst_683_ = lean_ctor_get(v_p_682_, 0);
v_snd_684_ = lean_ctor_get(v_p_682_, 1);
v_isSharedCheck_700_ = !lean_is_exclusive(v_p_682_);
if (v_isSharedCheck_700_ == 0)
{
v___x_686_ = v_p_682_;
v_isShared_687_ = v_isSharedCheck_700_;
goto v_resetjp_685_;
}
else
{
lean_inc(v_snd_684_);
lean_inc(v_fst_683_);
lean_dec(v_p_682_);
v___x_686_ = lean_box(0);
v_isShared_687_ = v_isSharedCheck_700_;
goto v_resetjp_685_;
}
v_resetjp_685_:
{
lean_object* v_fst_688_; lean_object* v_snd_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_699_; 
v_fst_688_ = lean_ctor_get(v_fst_683_, 0);
v_snd_689_ = lean_ctor_get(v_fst_683_, 1);
v_isSharedCheck_699_ = !lean_is_exclusive(v_fst_683_);
if (v_isSharedCheck_699_ == 0)
{
v___x_691_ = v_fst_683_;
v_isShared_692_ = v_isSharedCheck_699_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_snd_689_);
lean_inc(v_fst_688_);
lean_dec(v_fst_683_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_699_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v___x_694_; 
if (v_isShared_687_ == 0)
{
lean_ctor_set(v___x_686_, 0, v_snd_689_);
v___x_694_ = v___x_686_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_698_; 
v_reuseFailAlloc_698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_698_, 0, v_snd_689_);
lean_ctor_set(v_reuseFailAlloc_698_, 1, v_snd_684_);
v___x_694_ = v_reuseFailAlloc_698_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
lean_object* v___x_696_; 
if (v_isShared_692_ == 0)
{
lean_ctor_set(v___x_691_, 1, v___x_694_);
v___x_696_ = v___x_691_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v_fst_688_);
lean_ctor_set(v_reuseFailAlloc_697_, 1, v___x_694_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaProdDistrib___lam__1(lean_object* v_p_701_){
_start:
{
lean_object* v_snd_702_; lean_object* v_fst_703_; lean_object* v___x_705_; uint8_t v_isShared_706_; uint8_t v_isSharedCheck_719_; 
v_snd_702_ = lean_ctor_get(v_p_701_, 1);
v_fst_703_ = lean_ctor_get(v_p_701_, 0);
v_isSharedCheck_719_ = !lean_is_exclusive(v_p_701_);
if (v_isSharedCheck_719_ == 0)
{
v___x_705_ = v_p_701_;
v_isShared_706_ = v_isSharedCheck_719_;
goto v_resetjp_704_;
}
else
{
lean_inc(v_snd_702_);
lean_inc(v_fst_703_);
lean_dec(v_p_701_);
v___x_705_ = lean_box(0);
v_isShared_706_ = v_isSharedCheck_719_;
goto v_resetjp_704_;
}
v_resetjp_704_:
{
lean_object* v_fst_707_; lean_object* v_snd_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_718_; 
v_fst_707_ = lean_ctor_get(v_snd_702_, 0);
v_snd_708_ = lean_ctor_get(v_snd_702_, 1);
v_isSharedCheck_718_ = !lean_is_exclusive(v_snd_702_);
if (v_isSharedCheck_718_ == 0)
{
v___x_710_ = v_snd_702_;
v_isShared_711_ = v_isSharedCheck_718_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_snd_708_);
lean_inc(v_fst_707_);
lean_dec(v_snd_702_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_718_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_713_; 
if (v_isShared_706_ == 0)
{
lean_ctor_set(v___x_705_, 1, v_fst_707_);
v___x_713_ = v___x_705_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_fst_703_);
lean_ctor_set(v_reuseFailAlloc_717_, 1, v_fst_707_);
v___x_713_ = v_reuseFailAlloc_717_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
lean_object* v___x_715_; 
if (v_isShared_711_ == 0)
{
lean_ctor_set(v___x_710_, 0, v___x_713_);
v___x_715_ = v___x_710_;
goto v_reusejp_714_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v___x_713_);
lean_ctor_set(v_reuseFailAlloc_716_, 1, v_snd_708_);
v___x_715_ = v_reuseFailAlloc_716_;
goto v_reusejp_714_;
}
v_reusejp_714_:
{
return v___x_715_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaProdDistrib(lean_object* v_00_u03b9_725_, lean_object* v_00_u03b1_726_, lean_object* v_00_u03b2_727_){
_start:
{
lean_object* v___x_728_; 
v___x_728_ = ((lean_object*)(lp_mathlib_Equiv_sigmaProdDistrib___closed__2));
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__0(lean_object* v_p_729_){
_start:
{
lean_object* v_fst_730_; uint8_t v___x_731_; 
v_fst_730_ = lean_ctor_get(v_p_729_, 0);
v___x_731_ = lean_unbox(v_fst_730_);
if (v___x_731_ == 0)
{
lean_object* v_snd_732_; lean_object* v___x_733_; 
v_snd_732_ = lean_ctor_get(v_p_729_, 1);
lean_inc(v_snd_732_);
v___x_733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_733_, 0, v_snd_732_);
return v___x_733_;
}
else
{
lean_object* v_snd_734_; lean_object* v___x_735_; 
v_snd_734_ = lean_ctor_get(v_p_729_, 1);
lean_inc(v_snd_734_);
v___x_735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_735_, 0, v_snd_734_);
return v___x_735_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__0___boxed(lean_object* v_p_736_){
_start:
{
lean_object* v_res_737_; 
v_res_737_ = lp_mathlib_Equiv_boolProdEquivSum___lam__0(v_p_736_);
lean_dec_ref(v_p_736_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__1(lean_object* v_snd_738_){
_start:
{
uint8_t v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; 
v___x_739_ = 0;
v___x_740_ = lean_box(v___x_739_);
v___x_741_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_741_, 0, v___x_740_);
lean_ctor_set(v___x_741_, 1, v_snd_738_);
return v___x_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum___lam__2(lean_object* v_snd_742_){
_start:
{
uint8_t v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; 
v___x_743_ = 1;
v___x_744_ = lean_box(v___x_743_);
v___x_745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
lean_ctor_set(v___x_745_, 1, v_snd_742_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolProdEquivSum(lean_object* v_00_u03b1_755_){
_start:
{
lean_object* v___x_756_; 
v___x_756_ = ((lean_object*)(lp_mathlib_Equiv_boolProdEquivSum___closed__4));
return v___x_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd___lam__0(lean_object* v_f_757_){
_start:
{
uint8_t v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; uint8_t v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_758_ = 0;
v___x_759_ = lean_box(v___x_758_);
lean_inc(v_f_757_);
v___x_760_ = lean_apply_1(v_f_757_, v___x_759_);
v___x_761_ = 1;
v___x_762_ = lean_box(v___x_761_);
v___x_763_ = lean_apply_1(v_f_757_, v___x_762_);
v___x_764_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_764_, 0, v___x_760_);
lean_ctor_set(v___x_764_, 1, v___x_763_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd___lam__1(lean_object* v_p_765_, uint8_t v_b_766_){
_start:
{
if (v_b_766_ == 0)
{
lean_object* v_fst_767_; 
v_fst_767_ = lean_ctor_get(v_p_765_, 0);
lean_inc(v_fst_767_);
return v_fst_767_;
}
else
{
lean_object* v_snd_768_; 
v_snd_768_ = lean_ctor_get(v_p_765_, 1);
lean_inc(v_snd_768_);
return v_snd_768_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd___lam__1___boxed(lean_object* v_p_769_, lean_object* v_b_770_){
_start:
{
uint8_t v_b_boxed_771_; lean_object* v_res_772_; 
v_b_boxed_771_ = lean_unbox(v_b_770_);
v_res_772_ = lp_mathlib_Equiv_boolArrowEquivProd___lam__1(v_p_769_, v_b_boxed_771_);
lean_dec_ref(v_p_769_);
return v_res_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolArrowEquivProd(lean_object* v_00_u03b1_778_){
_start:
{
lean_object* v___x_779_; 
v___x_779_ = ((lean_object*)(lp_mathlib_Equiv_boolArrowEquivProd___closed__2));
return v___x_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquivSigma___lam__0(lean_object* v_x_780_){
_start:
{
lean_object* v_fst_781_; lean_object* v_snd_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_789_; 
v_fst_781_ = lean_ctor_get(v_x_780_, 0);
v_snd_782_ = lean_ctor_get(v_x_780_, 1);
v_isSharedCheck_789_ = !lean_is_exclusive(v_x_780_);
if (v_isSharedCheck_789_ == 0)
{
v___x_784_ = v_x_780_;
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_snd_782_);
lean_inc(v_fst_781_);
lean_dec(v_x_780_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_789_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_787_; 
if (v_isShared_785_ == 0)
{
v___x_787_ = v___x_784_;
goto v_reusejp_786_;
}
else
{
lean_object* v_reuseFailAlloc_788_; 
v_reuseFailAlloc_788_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_788_, 0, v_fst_781_);
lean_ctor_set(v_reuseFailAlloc_788_, 1, v_snd_782_);
v___x_787_ = v_reuseFailAlloc_788_;
goto v_reusejp_786_;
}
v_reusejp_786_:
{
return v___x_787_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSigmaEquivSigma(lean_object* v_00_u03b1_793_, lean_object* v_00_u03b2_794_, lean_object* v_p_795_, lean_object* v_q_796_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = ((lean_object*)(lp_mathlib_Equiv_subtypeSigmaEquivSigma___closed__1));
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivProd___lam__0(lean_object* v_x_798_){
_start:
{
lean_object* v_fst_799_; lean_object* v_snd_800_; lean_object* v___x_802_; uint8_t v_isShared_803_; uint8_t v_isSharedCheck_807_; 
v_fst_799_ = lean_ctor_get(v_x_798_, 0);
v_snd_800_ = lean_ctor_get(v_x_798_, 1);
v_isSharedCheck_807_ = !lean_is_exclusive(v_x_798_);
if (v_isSharedCheck_807_ == 0)
{
v___x_802_ = v_x_798_;
v_isShared_803_ = v_isSharedCheck_807_;
goto v_resetjp_801_;
}
else
{
lean_inc(v_snd_800_);
lean_inc(v_fst_799_);
lean_dec(v_x_798_);
v___x_802_ = lean_box(0);
v_isShared_803_ = v_isSharedCheck_807_;
goto v_resetjp_801_;
}
v_resetjp_801_:
{
lean_object* v___x_805_; 
if (v_isShared_803_ == 0)
{
v___x_805_ = v___x_802_;
goto v_reusejp_804_;
}
else
{
lean_object* v_reuseFailAlloc_806_; 
v_reuseFailAlloc_806_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_806_, 0, v_fst_799_);
lean_ctor_set(v_reuseFailAlloc_806_, 1, v_snd_800_);
v___x_805_ = v_reuseFailAlloc_806_;
goto v_reusejp_804_;
}
v_reusejp_804_:
{
return v___x_805_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object* v_00_u03b1_811_, lean_object* v_00_u03b2_812_, lean_object* v_p_813_, lean_object* v_q_814_){
_start:
{
lean_object* v___x_815_; 
v___x_815_ = ((lean_object*)(lp_mathlib_Equiv_subtypeProdEquivProd___closed__1));
return v___x_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodSubtypeFstEquivSubtypeProd(lean_object* v_00_u03b1_816_, lean_object* v_00_u03b2_817_, lean_object* v_p_818_){
_start:
{
lean_object* v___x_819_; 
v___x_819_ = ((lean_object*)(lp_mathlib_Equiv_subtypeProdEquivProd___closed__1));
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___lam__0(lean_object* v_x_820_){
_start:
{
lean_object* v_fst_821_; lean_object* v_snd_822_; lean_object* v___x_824_; uint8_t v_isShared_825_; uint8_t v_isSharedCheck_829_; 
v_fst_821_ = lean_ctor_get(v_x_820_, 0);
v_snd_822_ = lean_ctor_get(v_x_820_, 1);
v_isSharedCheck_829_ = !lean_is_exclusive(v_x_820_);
if (v_isSharedCheck_829_ == 0)
{
v___x_824_ = v_x_820_;
v_isShared_825_ = v_isSharedCheck_829_;
goto v_resetjp_823_;
}
else
{
lean_inc(v_snd_822_);
lean_inc(v_fst_821_);
lean_dec(v_x_820_);
v___x_824_ = lean_box(0);
v_isShared_825_ = v_isSharedCheck_829_;
goto v_resetjp_823_;
}
v_resetjp_823_:
{
lean_object* v___x_827_; 
if (v_isShared_825_ == 0)
{
v___x_827_ = v___x_824_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_828_; 
v_reuseFailAlloc_828_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_828_, 0, v_fst_821_);
lean_ctor_set(v_reuseFailAlloc_828_, 1, v_snd_822_);
v___x_827_ = v_reuseFailAlloc_828_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
return v___x_827_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___lam__1(lean_object* v_x_830_){
_start:
{
lean_object* v_fst_831_; lean_object* v_snd_832_; lean_object* v___x_834_; uint8_t v_isShared_835_; uint8_t v_isSharedCheck_839_; 
v_fst_831_ = lean_ctor_get(v_x_830_, 0);
v_snd_832_ = lean_ctor_get(v_x_830_, 1);
v_isSharedCheck_839_ = !lean_is_exclusive(v_x_830_);
if (v_isSharedCheck_839_ == 0)
{
v___x_834_ = v_x_830_;
v_isShared_835_ = v_isSharedCheck_839_;
goto v_resetjp_833_;
}
else
{
lean_inc(v_snd_832_);
lean_inc(v_fst_831_);
lean_dec(v_x_830_);
v___x_834_ = lean_box(0);
v_isShared_835_ = v_isSharedCheck_839_;
goto v_resetjp_833_;
}
v_resetjp_833_:
{
lean_object* v___x_837_; 
if (v_isShared_835_ == 0)
{
v___x_837_ = v___x_834_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_838_; 
v_reuseFailAlloc_838_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_838_, 0, v_fst_831_);
lean_ctor_set(v_reuseFailAlloc_838_, 1, v_snd_832_);
v___x_837_ = v_reuseFailAlloc_838_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
return v___x_837_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype(lean_object* v_00_u03b1_845_, lean_object* v_00_u03b2_846_, lean_object* v_p_847_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = ((lean_object*)(lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype___closed__2));
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__0(lean_object* v_f_849_, lean_object* v_x_850_){
_start:
{
lean_object* v___x_851_; 
v___x_851_ = lean_apply_1(v_f_849_, v_x_850_);
return v___x_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__2(lean_object* v_f_852_){
_start:
{
lean_object* v___f_853_; lean_object* v___x_854_; 
v___f_853_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_853_, 0, v_f_852_);
lean_inc_ref(v___f_853_);
v___x_854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_854_, 0, v___f_853_);
lean_ctor_set(v___x_854_, 1, v___f_853_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__1(lean_object* v_inst_855_, lean_object* v_f_856_, lean_object* v_x_857_){
_start:
{
lean_object* v___x_858_; uint8_t v___x_859_; 
lean_inc(v_x_857_);
v___x_858_ = lean_apply_1(v_inst_855_, v_x_857_);
v___x_859_ = lean_unbox(v___x_858_);
if (v___x_859_ == 0)
{
lean_object* v_snd_860_; lean_object* v___x_861_; 
v_snd_860_ = lean_ctor_get(v_f_856_, 1);
lean_inc(v_snd_860_);
lean_dec_ref(v_f_856_);
v___x_861_ = lean_apply_1(v_snd_860_, v_x_857_);
return v___x_861_;
}
else
{
lean_object* v_fst_862_; lean_object* v___x_863_; 
v_fst_862_ = lean_ctor_get(v_f_856_, 0);
lean_inc(v_fst_862_);
lean_dec_ref(v_f_856_);
v___x_863_ = lean_apply_1(v_fst_862_, v_x_857_);
return v___x_863_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg(lean_object* v_inst_865_){
_start:
{
lean_object* v___f_866_; lean_object* v___f_867_; lean_object* v___x_868_; 
v___f_866_ = ((lean_object*)(lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___closed__0));
v___f_867_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg___lam__1), 3, 1);
lean_closure_set(v___f_867_, 0, v_inst_865_);
v___x_868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_868_, 0, v___f_866_);
lean_ctor_set(v___x_868_, 1, v___f_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piEquivPiSubtypeProd(lean_object* v_00_u03b1_869_, lean_object* v_p_870_, lean_object* v_00_u03b2_871_, lean_object* v_inst_872_){
_start:
{
lean_object* v___x_873_; 
v___x_873_ = lp_mathlib_Equiv_piEquivPiSubtypeProd___redArg(v_inst_872_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg___lam__0(lean_object* v_f_874_, lean_object* v_j_875_){
_start:
{
lean_object* v___x_876_; 
v___x_876_ = lean_apply_1(v_f_874_, v_j_875_);
return v___x_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg___lam__1(lean_object* v_i_877_, lean_object* v_f_878_){
_start:
{
lean_object* v___f_879_; lean_object* v___x_880_; lean_object* v___x_881_; 
lean_inc(v_f_878_);
v___f_879_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piSplitAt___redArg___lam__0), 2, 1);
lean_closure_set(v___f_879_, 0, v_f_878_);
v___x_880_ = lean_apply_1(v_f_878_, v_i_877_);
v___x_881_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_881_, 0, v___x_880_);
lean_ctor_set(v___x_881_, 1, v___f_879_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg___lam__2(lean_object* v_inst_882_, lean_object* v_i_883_, lean_object* v_f_884_, lean_object* v_j_885_){
_start:
{
lean_object* v___x_886_; uint8_t v___x_887_; 
lean_inc(v_j_885_);
v___x_886_ = lean_apply_2(v_inst_882_, v_j_885_, v_i_883_);
v___x_887_ = lean_unbox(v___x_886_);
if (v___x_887_ == 0)
{
lean_object* v_snd_888_; lean_object* v___x_889_; 
v_snd_888_ = lean_ctor_get(v_f_884_, 1);
lean_inc(v_snd_888_);
lean_dec_ref(v_f_884_);
v___x_889_ = lean_apply_1(v_snd_888_, v_j_885_);
return v___x_889_;
}
else
{
lean_object* v_fst_890_; 
lean_dec(v_j_885_);
v_fst_890_ = lean_ctor_get(v_f_884_, 0);
lean_inc(v_fst_890_);
lean_dec_ref(v_f_884_);
return v_fst_890_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt___redArg(lean_object* v_inst_891_, lean_object* v_i_892_){
_start:
{
lean_object* v___f_893_; lean_object* v___f_894_; lean_object* v___x_895_; 
lean_inc(v_i_892_);
v___f_893_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piSplitAt___redArg___lam__1), 2, 1);
lean_closure_set(v___f_893_, 0, v_i_892_);
v___f_894_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_piSplitAt___redArg___lam__2), 4, 2);
lean_closure_set(v___f_894_, 0, v_inst_891_);
lean_closure_set(v___f_894_, 1, v_i_892_);
v___x_895_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_895_, 0, v___f_893_);
lean_ctor_set(v___x_895_, 1, v___f_894_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piSplitAt(lean_object* v_00_u03b1_896_, lean_object* v_inst_897_, lean_object* v_i_898_, lean_object* v_00_u03b2_899_){
_start:
{
lean_object* v___x_900_; 
v___x_900_ = lp_mathlib_Equiv_piSplitAt___redArg(v_inst_897_, v_i_898_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funSplitAt___redArg(lean_object* v_inst_901_, lean_object* v_i_902_){
_start:
{
lean_object* v___x_903_; 
v___x_903_ = lp_mathlib_Equiv_piSplitAt___redArg(v_inst_901_, v_i_902_);
return v___x_903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_funSplitAt(lean_object* v_00_u03b1_904_, lean_object* v_inst_905_, lean_object* v_i_906_, lean_object* v_00_u03b2_907_){
_start:
{
lean_object* v___x_908_; 
v___x_908_ = lp_mathlib_Equiv_piSplitAt___redArg(v_inst_905_, v_i_906_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv___lam__0(lean_object* v_p_909_){
_start:
{
lean_object* v_fst_910_; 
v_fst_910_ = lean_ctor_get(v_p_909_, 0);
lean_inc(v_fst_910_);
return v_fst_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv___lam__0___boxed(lean_object* v_p_911_){
_start:
{
lean_object* v_res_912_; 
v_res_912_ = lp_mathlib_subsingletonProdSelfEquiv___lam__0(v_p_911_);
lean_dec_ref(v_p_911_);
return v_res_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv___lam__1(lean_object* v_a_913_){
_start:
{
lean_object* v___x_914_; 
lean_inc(v_a_913_);
v___x_914_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_914_, 0, v_a_913_);
lean_ctor_set(v___x_914_, 1, v_a_913_);
return v___x_914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subsingletonProdSelfEquiv(lean_object* v_00_u03b1_920_, lean_object* v_inst_921_){
_start:
{
lean_object* v___x_922_; 
v___x_922_ = ((lean_object*)(lp_mathlib_subsingletonProdSelfEquiv___closed__2));
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv___lam__0(lean_object* v_x_923_){
_start:
{
lean_object* v_fst_924_; 
v_fst_924_ = lean_ctor_get(v_x_923_, 0);
lean_inc(v_fst_924_);
if (lean_obj_tag(v_fst_924_) == 0)
{
lean_object* v_snd_925_; lean_object* v___x_926_; 
v_snd_925_ = lean_ctor_get(v_x_923_, 1);
lean_inc(v_snd_925_);
lean_dec_ref(v_x_923_);
v___x_926_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_926_, 0, v_snd_925_);
return v___x_926_;
}
else
{
lean_object* v_snd_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_942_; 
v_snd_927_ = lean_ctor_get(v_x_923_, 1);
v_isSharedCheck_942_ = !lean_is_exclusive(v_x_923_);
if (v_isSharedCheck_942_ == 0)
{
lean_object* v_unused_943_; 
v_unused_943_ = lean_ctor_get(v_x_923_, 0);
lean_dec(v_unused_943_);
v___x_929_ = v_x_923_;
v_isShared_930_ = v_isSharedCheck_942_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_snd_927_);
lean_dec(v_x_923_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_942_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v_a_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_941_; 
v_a_931_ = lean_ctor_get(v_fst_924_, 0);
v_isSharedCheck_941_ = !lean_is_exclusive(v_fst_924_);
if (v_isSharedCheck_941_ == 0)
{
v___x_933_ = v_fst_924_;
v_isShared_934_ = v_isSharedCheck_941_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_a_931_);
lean_dec(v_fst_924_);
v___x_933_ = lean_box(0);
v_isShared_934_ = v_isSharedCheck_941_;
goto v_resetjp_932_;
}
v_resetjp_932_:
{
lean_object* v___x_936_; 
if (v_isShared_930_ == 0)
{
lean_ctor_set(v___x_929_, 0, v_a_931_);
v___x_936_ = v___x_929_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v_a_931_);
lean_ctor_set(v_reuseFailAlloc_940_, 1, v_snd_927_);
v___x_936_ = v_reuseFailAlloc_940_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
lean_object* v___x_938_; 
if (v_isShared_934_ == 0)
{
lean_ctor_set(v___x_933_, 0, v___x_936_);
v___x_938_ = v___x_933_;
goto v_reusejp_937_;
}
else
{
lean_object* v_reuseFailAlloc_939_; 
v_reuseFailAlloc_939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_939_, 0, v___x_936_);
v___x_938_ = v_reuseFailAlloc_939_;
goto v_reusejp_937_;
}
v_reusejp_937_:
{
return v___x_938_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv___lam__1(lean_object* v_val_944_){
_start:
{
lean_object* v___x_945_; 
v___x_945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_945_, 0, v_val_944_);
return v___x_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv___lam__3(lean_object* v___f_946_, lean_object* v___f_947_, lean_object* v_x_948_){
_start:
{
if (lean_obj_tag(v_x_948_) == 0)
{
lean_object* v_snd_949_; lean_object* v___x_950_; lean_object* v___x_951_; 
lean_dec(v___f_947_);
lean_dec_ref(v___f_946_);
v_snd_949_ = lean_ctor_get(v_x_948_, 0);
lean_inc(v_snd_949_);
lean_dec_ref_known(v_x_948_, 1);
v___x_950_ = lean_box(0);
v___x_951_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_951_, 0, v___x_950_);
lean_ctor_set(v___x_951_, 1, v_snd_949_);
return v___x_951_;
}
else
{
lean_object* v_a_952_; lean_object* v___x_953_; 
v_a_952_ = lean_ctor_get(v_x_948_, 0);
lean_inc(v_a_952_);
lean_dec_ref_known(v_x_948_, 1);
v___x_953_ = l_Prod_map___redArg(v___f_946_, v___f_947_, v_a_952_);
return v___x_953_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_optionProdEquiv(lean_object* v_00_u03b1_962_, lean_object* v_00_u03b2_963_){
_start:
{
lean_object* v___x_964_; 
v___x_964_ = ((lean_object*)(lp_mathlib_optionProdEquiv___closed__3));
return v___x_964_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
