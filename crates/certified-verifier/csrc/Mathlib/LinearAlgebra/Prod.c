// Lean compiler output
// Module: Mathlib.LinearAlgebra.Prod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Prod public import Mathlib.Algebra.Group.Graph public import Mathlib.LinearAlgebra.Span.Basic
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
lean_object* lp_mathlib_Function_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
lean_object* l_Prod_swap(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_uniqueProd___redArg(lean_object*);
lean_object* lp_mathlib_AddEquiv_toLinearEquiv___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_toAddEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_prodAssoc(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_fst___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_snd___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_prodEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_prodEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_prodEquiv___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_prodEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_LinearMap_prodEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_prodEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_prodEquiv___closed__1 = (const lean_object*)&lp_mathlib_LinearMap_prodEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_LinearMap_prodEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_LinearMap_prodEquiv___closed__0_value),((lean_object*)&lp_mathlib_LinearMap_prodEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_LinearMap_prodEquiv___closed__2 = (const lean_object*)&lp_mathlib_LinearMap_prodEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_inl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_inl___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_inl___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapLinear___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_prodMapLinear___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_prodMapLinear___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_prodMapLinear___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_prodMapLinear___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapLinear___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_fstEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_fstEquiv___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_fstEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_fstEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_sndEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_sndEquiv___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_sndEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_sndEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_prodComm___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_prodComm___closed__0;
static const lean_closure_object lp_mathlib_LinearEquiv_prodComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Prod_swap, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_LinearEquiv_prodComm___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv_prodComm___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_LinearEquiv_prodAssoc___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearEquiv_prodAssoc___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodAssoc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_LinearEquiv_prodProdProdComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_prodProdProdComm___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___closed__0 = (const lean_object*)&lp_mathlib_LinearEquiv_prodProdProdComm___closed__0_value;
static const lean_closure_object lp_mathlib_LinearEquiv_prodProdProdComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearEquiv_prodProdProdComm___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___closed__1 = (const lean_object*)&lp_mathlib_LinearEquiv_prodProdProdComm___closed__1_value;
static const lean_ctor_object lp_mathlib_LinearEquiv_prodProdProdComm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_LinearEquiv_prodProdProdComm___closed__0_value),((lean_object*)&lp_mathlib_LinearEquiv_prodProdProdComm___closed__1_value)}};
static const lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___closed__2 = (const lean_object*)&lp_mathlib_LinearEquiv_prodProdProdComm___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_graph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_graph___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst___lam__0(lean_object* v_self_1_){
_start:
{
lean_object* v_fst_2_; 
v_fst_2_ = lean_ctor_get(v_self_1_, 0);
lean_inc(v_fst_2_);
return v_fst_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst___lam__0___boxed(lean_object* v_self_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_LinearMap_fst___lam__0(v_self_3_);
lean_dec_ref(v_self_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst(lean_object* v_R_6_, lean_object* v_M_7_, lean_object* v_M_u2082_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = ((lean_object*)(lp_mathlib_LinearMap_fst___closed__0));
return v___f_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_fst___boxed(lean_object* v_R_15_, lean_object* v_M_16_, lean_object* v_M_u2082_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_LinearMap_fst(v_R_15_, v_M_16_, v_M_u2082_17_, v_inst_18_, v_inst_19_, v_inst_20_, v_inst_21_, v_inst_22_);
lean_dec(v_inst_22_);
lean_dec(v_inst_21_);
lean_dec_ref(v_inst_20_);
lean_dec_ref(v_inst_19_);
lean_dec_ref(v_inst_18_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd___lam__0(lean_object* v_self_24_){
_start:
{
lean_object* v_snd_25_; 
v_snd_25_ = lean_ctor_get(v_self_24_, 1);
lean_inc(v_snd_25_);
return v_snd_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd___lam__0___boxed(lean_object* v_self_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_LinearMap_snd___lam__0(v_self_26_);
lean_dec_ref(v_self_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd(lean_object* v_R_29_, lean_object* v_M_30_, lean_object* v_M_u2082_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___f_37_; 
v___f_37_ = ((lean_object*)(lp_mathlib_LinearMap_snd___closed__0));
return v___f_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_snd___boxed(lean_object* v_R_38_, lean_object* v_M_39_, lean_object* v_M_u2082_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_LinearMap_snd(v_R_38_, v_M_39_, v_M_u2082_40_, v_inst_41_, v_inst_42_, v_inst_43_, v_inst_44_, v_inst_45_);
lean_dec(v_inst_45_);
lean_dec(v_inst_44_);
lean_dec_ref(v_inst_43_);
lean_dec_ref(v_inst_42_);
lean_dec_ref(v_inst_41_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___redArg___lam__0(lean_object* v_f_47_, lean_object* v___y_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_apply_1(v_f_47_, v___y_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___redArg___lam__1(lean_object* v_g_50_, lean_object* v___y_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_apply_1(v_g_50_, v___y_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___redArg(lean_object* v_f_53_, lean_object* v_g_54_){
_start:
{
lean_object* v___f_55_; lean_object* v___f_56_; lean_object* v___x_57_; 
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_prod___redArg___lam__0), 2, 1);
lean_closure_set(v___f_55_, 0, v_f_53_);
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_prod___redArg___lam__1), 2, 1);
lean_closure_set(v___f_56_, 0, v_g_54_);
v___x_57_ = lean_alloc_closure((void*)(lp_mathlib_Function_prod), 6, 5);
lean_closure_set(v___x_57_, 0, lean_box(0));
lean_closure_set(v___x_57_, 1, lean_box(0));
lean_closure_set(v___x_57_, 2, lean_box(0));
lean_closure_set(v___x_57_, 3, v___f_55_);
lean_closure_set(v___x_57_, 4, v___f_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod(lean_object* v_R_58_, lean_object* v_M_59_, lean_object* v_M_u2082_60_, lean_object* v_M_u2083_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_f_69_, lean_object* v_g_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_LinearMap_prod___redArg(v_f_69_, v_g_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prod___boxed(lean_object* v_R_72_, lean_object* v_M_73_, lean_object* v_M_u2082_74_, lean_object* v_M_u2083_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_f_83_, lean_object* v_g_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_LinearMap_prod(v_R_72_, v_M_73_, v_M_u2082_74_, v_M_u2083_75_, v_inst_76_, v_inst_77_, v_inst_78_, v_inst_79_, v_inst_80_, v_inst_81_, v_inst_82_, v_f_83_, v_g_84_);
lean_dec(v_inst_82_);
lean_dec(v_inst_81_);
lean_dec(v_inst_80_);
lean_dec_ref(v_inst_79_);
lean_dec_ref(v_inst_78_);
lean_dec_ref(v_inst_77_);
lean_dec_ref(v_inst_76_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv___lam__0(lean_object* v_f_86_, lean_object* v___y_87_){
_start:
{
lean_object* v_fst_88_; lean_object* v_snd_89_; lean_object* v___x_72__overap_90_; lean_object* v___x_91_; 
v_fst_88_ = lean_ctor_get(v_f_86_, 0);
lean_inc(v_fst_88_);
v_snd_89_ = lean_ctor_get(v_f_86_, 1);
lean_inc(v_snd_89_);
lean_dec_ref(v_f_86_);
v___x_72__overap_90_ = lp_mathlib_LinearMap_prod___redArg(v_fst_88_, v_snd_89_);
v___x_91_ = lean_apply_1(v___x_72__overap_90_, v___y_87_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv___lam__1(lean_object* v_f_92_){
_start:
{
lean_object* v___f_93_; lean_object* v___f_94_; lean_object* v___f_95_; lean_object* v___f_96_; lean_object* v___x_97_; 
v___f_93_ = ((lean_object*)(lp_mathlib_LinearMap_fst___closed__0));
lean_inc_ref(v_f_92_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_94_, 0, v_f_92_);
lean_closure_set(v___f_94_, 1, v___f_93_);
v___f_95_ = ((lean_object*)(lp_mathlib_LinearMap_snd___closed__0));
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_96_, 0, v_f_92_);
lean_closure_set(v___f_96_, 1, v___f_95_);
v___x_97_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_97_, 0, v___f_94_);
lean_ctor_set(v___x_97_, 1, v___f_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv(lean_object* v_R_103_, lean_object* v_M_104_, lean_object* v_M_u2082_105_, lean_object* v_M_u2083_106_, lean_object* v_S_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = ((lean_object*)(lp_mathlib_LinearMap_prodEquiv___closed__2));
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodEquiv___boxed(lean_object** _args){
lean_object* v_R_121_ = _args[0];
lean_object* v_M_122_ = _args[1];
lean_object* v_M_u2082_123_ = _args[2];
lean_object* v_M_u2083_124_ = _args[3];
lean_object* v_S_125_ = _args[4];
lean_object* v_inst_126_ = _args[5];
lean_object* v_inst_127_ = _args[6];
lean_object* v_inst_128_ = _args[7];
lean_object* v_inst_129_ = _args[8];
lean_object* v_inst_130_ = _args[9];
lean_object* v_inst_131_ = _args[10];
lean_object* v_inst_132_ = _args[11];
lean_object* v_inst_133_ = _args[12];
lean_object* v_inst_134_ = _args[13];
lean_object* v_inst_135_ = _args[14];
lean_object* v_inst_136_ = _args[15];
lean_object* v_inst_137_ = _args[16];
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_LinearMap_prodEquiv(v_R_121_, v_M_122_, v_M_u2082_123_, v_M_u2083_124_, v_S_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_inst_129_, v_inst_130_, v_inst_131_, v_inst_132_, v_inst_133_, v_inst_134_, v_inst_135_, v_inst_136_, v_inst_137_);
lean_dec(v_inst_135_);
lean_dec(v_inst_134_);
lean_dec(v_inst_133_);
lean_dec(v_inst_132_);
lean_dec(v_inst_131_);
lean_dec_ref(v_inst_130_);
lean_dec_ref(v_inst_129_);
lean_dec_ref(v_inst_128_);
lean_dec_ref(v_inst_127_);
lean_dec_ref(v_inst_126_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg___lam__0(lean_object* v_toZero_139_, lean_object* v_x_140_){
_start:
{
lean_inc(v_toZero_139_);
return v_toZero_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg___lam__0___boxed(lean_object* v_toZero_141_, lean_object* v_x_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_LinearMap_inl___redArg___lam__0(v_toZero_141_, v_x_142_);
lean_dec(v_x_142_);
lean_dec(v_toZero_141_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg(lean_object* v_inst_145_){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v_toZero_148_; lean_object* v___f_149_; lean_object* v___f_150_; lean_object* v___x_151_; 
v___x_146_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_145_);
v___x_147_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_146_);
v_toZero_148_ = lean_ctor_get(v___x_147_, 0);
lean_inc(v_toZero_148_);
lean_dec_ref(v___x_147_);
v___f_149_ = ((lean_object*)(lp_mathlib_LinearMap_inl___redArg___closed__0));
v___f_150_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_inl___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_150_, 0, v_toZero_148_);
v___x_151_ = lp_mathlib_LinearMap_prod___redArg(v___f_149_, v___f_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___redArg___boxed(lean_object* v_inst_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_LinearMap_inl___redArg(v_inst_152_);
lean_dec_ref(v_inst_152_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl(lean_object* v_R_154_, lean_object* v_M_155_, lean_object* v_M_u2082_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_LinearMap_inl___redArg(v_inst_159_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inl___boxed(lean_object* v_R_163_, lean_object* v_M_164_, lean_object* v_M_u2082_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_LinearMap_inl(v_R_163_, v_M_164_, v_M_u2082_165_, v_inst_166_, v_inst_167_, v_inst_168_, v_inst_169_, v_inst_170_);
lean_dec(v_inst_170_);
lean_dec(v_inst_169_);
lean_dec_ref(v_inst_168_);
lean_dec_ref(v_inst_167_);
lean_dec_ref(v_inst_166_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr___redArg(lean_object* v_inst_172_){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v_toZero_175_; lean_object* v___f_176_; lean_object* v___f_177_; lean_object* v___x_178_; 
v___x_173_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_172_);
v___x_174_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_173_);
v_toZero_175_ = lean_ctor_get(v___x_174_, 0);
lean_inc(v_toZero_175_);
lean_dec_ref(v___x_174_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_inl___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_176_, 0, v_toZero_175_);
v___f_177_ = ((lean_object*)(lp_mathlib_LinearMap_inl___redArg___closed__0));
v___x_178_ = lp_mathlib_LinearMap_prod___redArg(v___f_176_, v___f_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr___redArg___boxed(lean_object* v_inst_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_LinearMap_inr___redArg(v_inst_179_);
lean_dec_ref(v_inst_179_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr(lean_object* v_R_181_, lean_object* v_M_182_, lean_object* v_M_u2082_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_LinearMap_inr___redArg(v_inst_185_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_inr___boxed(lean_object* v_R_190_, lean_object* v_M_191_, lean_object* v_M_u2082_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_LinearMap_inr(v_R_190_, v_M_191_, v_M_u2082_192_, v_inst_193_, v_inst_194_, v_inst_195_, v_inst_196_, v_inst_197_);
lean_dec(v_inst_197_);
lean_dec(v_inst_196_);
lean_dec_ref(v_inst_195_);
lean_dec_ref(v_inst_194_);
lean_dec_ref(v_inst_193_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod___redArg___lam__0(lean_object* v___f_199_, lean_object* v_f_200_, lean_object* v___f_201_, lean_object* v_g_202_, lean_object* v_toAdd_203_, lean_object* v___y_204_){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
lean_inc_ref(v___y_204_);
v___x_205_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___f_199_, v_f_200_, v___y_204_);
v___x_206_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___f_201_, v_g_202_, v___y_204_);
v___x_207_ = lean_apply_2(v_toAdd_203_, v___x_205_, v___x_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod___redArg(lean_object* v_inst_208_, lean_object* v_f_209_, lean_object* v_g_210_){
_start:
{
lean_object* v_toAdd_211_; lean_object* v___f_212_; lean_object* v___f_213_; lean_object* v___f_214_; 
v_toAdd_211_ = lean_ctor_get(v_inst_208_, 1);
lean_inc(v_toAdd_211_);
lean_dec_ref(v_inst_208_);
v___f_212_ = ((lean_object*)(lp_mathlib_LinearMap_fst___closed__0));
v___f_213_ = ((lean_object*)(lp_mathlib_LinearMap_snd___closed__0));
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_coprod___redArg___lam__0), 6, 5);
lean_closure_set(v___f_214_, 0, v___f_212_);
lean_closure_set(v___f_214_, 1, v_f_209_);
lean_closure_set(v___f_214_, 2, v___f_213_);
lean_closure_set(v___f_214_, 3, v_g_210_);
lean_closure_set(v___f_214_, 4, v_toAdd_211_);
return v___f_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod(lean_object* v_R_215_, lean_object* v_M_216_, lean_object* v_M_u2082_217_, lean_object* v_M_u2083_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_f_226_, lean_object* v_g_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_mathlib_LinearMap_coprod___redArg(v_inst_222_, v_f_226_, v_g_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprod___boxed(lean_object* v_R_229_, lean_object* v_M_230_, lean_object* v_M_u2082_231_, lean_object* v_M_u2083_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_f_240_, lean_object* v_g_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_LinearMap_coprod(v_R_229_, v_M_230_, v_M_u2082_231_, v_M_u2083_232_, v_inst_233_, v_inst_234_, v_inst_235_, v_inst_236_, v_inst_237_, v_inst_238_, v_inst_239_, v_f_240_, v_g_241_);
lean_dec(v_inst_239_);
lean_dec(v_inst_238_);
lean_dec(v_inst_237_);
lean_dec_ref(v_inst_235_);
lean_dec_ref(v_inst_234_);
lean_dec_ref(v_inst_233_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg___lam__0(lean_object* v_inst_243_, lean_object* v_f_244_, lean_object* v___y_245_){
_start:
{
lean_object* v_fst_246_; lean_object* v_snd_247_; lean_object* v___x_78__overap_248_; lean_object* v___x_249_; 
v_fst_246_ = lean_ctor_get(v_f_244_, 0);
lean_inc(v_fst_246_);
v_snd_247_ = lean_ctor_get(v_f_244_, 1);
lean_inc(v_snd_247_);
lean_dec_ref(v_f_244_);
v___x_78__overap_248_ = lp_mathlib_LinearMap_coprod___redArg(v_inst_243_, v_fst_246_, v_snd_247_);
v___x_249_ = lean_apply_1(v___x_78__overap_248_, v___y_245_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg___lam__1(lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_f_252_){
_start:
{
lean_object* v___x_253_; lean_object* v___f_254_; lean_object* v___x_255_; lean_object* v___f_256_; lean_object* v___x_257_; 
v___x_253_ = lp_mathlib_LinearMap_inl___redArg(v_inst_250_);
lean_inc(v_f_252_);
v___f_254_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_254_, 0, v___x_253_);
lean_closure_set(v___f_254_, 1, v_f_252_);
v___x_255_ = lp_mathlib_LinearMap_inr___redArg(v_inst_251_);
v___f_256_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_256_, 0, v___x_255_);
lean_closure_set(v___f_256_, 1, v_f_252_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v___f_254_);
lean_ctor_set(v___x_257_, 1, v___f_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg___lam__1___boxed(lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_f_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_LinearMap_coprodEquiv___redArg___lam__1(v_inst_258_, v_inst_259_, v_f_260_);
lean_dec_ref(v_inst_259_);
lean_dec_ref(v_inst_258_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___redArg(lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___f_265_; lean_object* v___f_266_; lean_object* v___x_267_; 
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_coprodEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_265_, 0, v_inst_264_);
v___f_266_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_coprodEquiv___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_266_, 0, v_inst_263_);
lean_closure_set(v___f_266_, 1, v_inst_262_);
v___x_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_267_, 0, v___f_265_);
lean_ctor_set(v___x_267_, 1, v___f_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv(lean_object* v_R_268_, lean_object* v_M_269_, lean_object* v_M_u2082_270_, lean_object* v_M_u2083_271_, lean_object* v_S_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_LinearMap_coprodEquiv___redArg(v_inst_275_, v_inst_276_, v_inst_277_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_coprodEquiv___boxed(lean_object* v_R_284_, lean_object* v_M_285_, lean_object* v_M_u2082_286_, lean_object* v_M_u2083_287_, lean_object* v_S_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_LinearMap_coprodEquiv(v_R_284_, v_M_285_, v_M_u2082_286_, v_M_u2083_287_, v_S_288_, v_inst_289_, v_inst_290_, v_inst_291_, v_inst_292_, v_inst_293_, v_inst_294_, v_inst_295_, v_inst_296_, v_inst_297_, v_inst_298_);
lean_dec(v_inst_297_);
lean_dec(v_inst_296_);
lean_dec(v_inst_295_);
lean_dec(v_inst_294_);
lean_dec_ref(v_inst_290_);
lean_dec_ref(v_inst_289_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMap___redArg(lean_object* v_f_300_, lean_object* v_g_301_){
_start:
{
lean_object* v___f_302_; lean_object* v___f_303_; lean_object* v___f_304_; lean_object* v___f_305_; lean_object* v___x_306_; 
v___f_302_ = ((lean_object*)(lp_mathlib_LinearMap_fst___closed__0));
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_303_, 0, v___f_302_);
lean_closure_set(v___f_303_, 1, v_f_300_);
v___f_304_ = ((lean_object*)(lp_mathlib_LinearMap_snd___closed__0));
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_305_, 0, v___f_304_);
lean_closure_set(v___f_305_, 1, v_g_301_);
v___x_306_ = lp_mathlib_LinearMap_prod___redArg(v___f_303_, v___f_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMap(lean_object* v_R_307_, lean_object* v_M_308_, lean_object* v_M_u2082_309_, lean_object* v_M_u2083_310_, lean_object* v_M_u2084_311_, lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_f_321_, lean_object* v_g_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_LinearMap_prodMap___redArg(v_f_321_, v_g_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMap___boxed(lean_object* v_R_324_, lean_object* v_M_325_, lean_object* v_M_u2082_326_, lean_object* v_M_u2083_327_, lean_object* v_M_u2084_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_inst_337_, lean_object* v_f_338_, lean_object* v_g_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_LinearMap_prodMap(v_R_324_, v_M_325_, v_M_u2082_326_, v_M_u2083_327_, v_M_u2084_328_, v_inst_329_, v_inst_330_, v_inst_331_, v_inst_332_, v_inst_333_, v_inst_334_, v_inst_335_, v_inst_336_, v_inst_337_, v_f_338_, v_g_339_);
lean_dec(v_inst_337_);
lean_dec(v_inst_336_);
lean_dec(v_inst_335_);
lean_dec(v_inst_334_);
lean_dec_ref(v_inst_333_);
lean_dec_ref(v_inst_332_);
lean_dec_ref(v_inst_331_);
lean_dec_ref(v_inst_330_);
lean_dec_ref(v_inst_329_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapLinear___lam__0(lean_object* v_f_341_, lean_object* v___y_342_){
_start:
{
lean_object* v_fst_343_; lean_object* v_snd_344_; lean_object* v___x_63__overap_345_; lean_object* v___x_346_; 
v_fst_343_ = lean_ctor_get(v_f_341_, 0);
lean_inc(v_fst_343_);
v_snd_344_ = lean_ctor_get(v_f_341_, 1);
lean_inc(v_snd_344_);
lean_dec_ref(v_f_341_);
v___x_63__overap_345_ = lp_mathlib_LinearMap_prodMap___redArg(v_fst_343_, v_snd_344_);
v___x_346_ = lean_apply_1(v___x_63__overap_345_, v___y_342_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapLinear(lean_object* v_R_348_, lean_object* v_M_349_, lean_object* v_M_u2082_350_, lean_object* v_M_u2083_351_, lean_object* v_M_u2084_352_, lean_object* v_S_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_){
_start:
{
lean_object* v___f_368_; 
v___f_368_ = ((lean_object*)(lp_mathlib_LinearMap_prodMapLinear___closed__0));
return v___f_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapLinear___boxed(lean_object** _args){
lean_object* v_R_369_ = _args[0];
lean_object* v_M_370_ = _args[1];
lean_object* v_M_u2082_371_ = _args[2];
lean_object* v_M_u2083_372_ = _args[3];
lean_object* v_M_u2084_373_ = _args[4];
lean_object* v_S_374_ = _args[5];
lean_object* v_inst_375_ = _args[6];
lean_object* v_inst_376_ = _args[7];
lean_object* v_inst_377_ = _args[8];
lean_object* v_inst_378_ = _args[9];
lean_object* v_inst_379_ = _args[10];
lean_object* v_inst_380_ = _args[11];
lean_object* v_inst_381_ = _args[12];
lean_object* v_inst_382_ = _args[13];
lean_object* v_inst_383_ = _args[14];
lean_object* v_inst_384_ = _args[15];
lean_object* v_inst_385_ = _args[16];
lean_object* v_inst_386_ = _args[17];
lean_object* v_inst_387_ = _args[18];
lean_object* v_inst_388_ = _args[19];
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_LinearMap_prodMapLinear(v_R_369_, v_M_370_, v_M_u2082_371_, v_M_u2083_372_, v_M_u2084_373_, v_S_374_, v_inst_375_, v_inst_376_, v_inst_377_, v_inst_378_, v_inst_379_, v_inst_380_, v_inst_381_, v_inst_382_, v_inst_383_, v_inst_384_, v_inst_385_, v_inst_386_, v_inst_387_, v_inst_388_);
lean_dec(v_inst_386_);
lean_dec(v_inst_385_);
lean_dec(v_inst_384_);
lean_dec(v_inst_383_);
lean_dec(v_inst_382_);
lean_dec(v_inst_381_);
lean_dec_ref(v_inst_380_);
lean_dec_ref(v_inst_379_);
lean_dec_ref(v_inst_378_);
lean_dec_ref(v_inst_377_);
lean_dec_ref(v_inst_376_);
lean_dec_ref(v_inst_375_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapRingHom(lean_object* v_R_390_, lean_object* v_M_391_, lean_object* v_M_u2082_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_inst_397_){
_start:
{
lean_object* v___f_398_; 
v___f_398_ = ((lean_object*)(lp_mathlib_LinearMap_prodMapLinear___closed__0));
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapRingHom___boxed(lean_object* v_R_399_, lean_object* v_M_400_, lean_object* v_M_u2082_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_LinearMap_prodMapRingHom(v_R_399_, v_M_400_, v_M_u2082_401_, v_inst_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_inst_406_);
lean_dec(v_inst_406_);
lean_dec(v_inst_405_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_inst_402_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapAlgHom(lean_object* v_R_408_, lean_object* v_M_409_, lean_object* v_M_u2082_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_inst_415_){
_start:
{
lean_object* v___f_416_; 
v___f_416_ = ((lean_object*)(lp_mathlib_LinearMap_prodMapLinear___closed__0));
return v___f_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_prodMapAlgHom___boxed(lean_object* v_R_417_, lean_object* v_M_418_, lean_object* v_M_u2082_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_LinearMap_prodMapAlgHom(v_R_417_, v_M_418_, v_M_u2082_419_, v_inst_420_, v_inst_421_, v_inst_422_, v_inst_423_, v_inst_424_);
lean_dec(v_inst_424_);
lean_dec(v_inst_423_);
lean_dec_ref(v_inst_422_);
lean_dec_ref(v_inst_421_);
lean_dec_ref(v_inst_420_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fst(lean_object* v_R_426_, lean_object* v_M_427_, lean_object* v_M_u2082_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lean_box(0);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fst___boxed(lean_object* v_R_435_, lean_object* v_M_436_, lean_object* v_M_u2082_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_){
_start:
{
lean_object* v_res_443_; 
v_res_443_ = lp_mathlib_Submodule_fst(v_R_435_, v_M_436_, v_M_u2082_437_, v_inst_438_, v_inst_439_, v_inst_440_, v_inst_441_, v_inst_442_);
lean_dec(v_inst_442_);
lean_dec(v_inst_441_);
lean_dec_ref(v_inst_440_);
lean_dec_ref(v_inst_439_);
lean_dec_ref(v_inst_438_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___lam__0(lean_object* v_x_444_){
_start:
{
lean_object* v_fst_445_; 
v_fst_445_ = lean_ctor_get(v_x_444_, 0);
lean_inc(v_fst_445_);
return v_fst_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___lam__0___boxed(lean_object* v_x_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_Submodule_fstEquiv___redArg___lam__0(v_x_446_);
lean_dec_ref(v_x_446_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___lam__1(lean_object* v_toZero_448_, lean_object* v_m_449_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_450_, 0, v_m_449_);
lean_ctor_set(v___x_450_, 1, v_toZero_448_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg(lean_object* v_inst_452_){
_start:
{
lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v_toZero_455_; lean_object* v___x_457_; uint8_t v_isShared_458_; uint8_t v_isSharedCheck_464_; 
v___x_453_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_452_);
v___x_454_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_453_);
v_toZero_455_ = lean_ctor_get(v___x_454_, 0);
v_isSharedCheck_464_ = !lean_is_exclusive(v___x_454_);
if (v_isSharedCheck_464_ == 0)
{
lean_object* v_unused_465_; 
v_unused_465_ = lean_ctor_get(v___x_454_, 1);
lean_dec(v_unused_465_);
v___x_457_ = v___x_454_;
v_isShared_458_ = v_isSharedCheck_464_;
goto v_resetjp_456_;
}
else
{
lean_inc(v_toZero_455_);
lean_dec(v___x_454_);
v___x_457_ = lean_box(0);
v_isShared_458_ = v_isSharedCheck_464_;
goto v_resetjp_456_;
}
v_resetjp_456_:
{
lean_object* v___f_459_; lean_object* v___f_460_; lean_object* v___x_462_; 
v___f_459_ = ((lean_object*)(lp_mathlib_Submodule_fstEquiv___redArg___closed__0));
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_fstEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_460_, 0, v_toZero_455_);
if (v_isShared_458_ == 0)
{
lean_ctor_set(v___x_457_, 1, v___f_460_);
lean_ctor_set(v___x_457_, 0, v___f_459_);
v___x_462_ = v___x_457_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_463_; 
v_reuseFailAlloc_463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_463_, 0, v___f_459_);
lean_ctor_set(v_reuseFailAlloc_463_, 1, v___f_460_);
v___x_462_ = v_reuseFailAlloc_463_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
return v___x_462_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___redArg___boxed(lean_object* v_inst_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_Submodule_fstEquiv___redArg(v_inst_466_);
lean_dec_ref(v_inst_466_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv(lean_object* v_R_468_, lean_object* v_M_469_, lean_object* v_M_u2082_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_Submodule_fstEquiv___redArg(v_inst_473_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_fstEquiv___boxed(lean_object* v_R_477_, lean_object* v_M_478_, lean_object* v_M_u2082_479_, lean_object* v_inst_480_, lean_object* v_inst_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_mathlib_Submodule_fstEquiv(v_R_477_, v_M_478_, v_M_u2082_479_, v_inst_480_, v_inst_481_, v_inst_482_, v_inst_483_, v_inst_484_);
lean_dec(v_inst_484_);
lean_dec(v_inst_483_);
lean_dec_ref(v_inst_482_);
lean_dec_ref(v_inst_481_);
lean_dec_ref(v_inst_480_);
return v_res_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_snd(lean_object* v_R_486_, lean_object* v_M_487_, lean_object* v_M_u2082_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lean_box(0);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_snd___boxed(lean_object* v_R_495_, lean_object* v_M_496_, lean_object* v_M_u2082_497_, lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_inst_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib_Submodule_snd(v_R_495_, v_M_496_, v_M_u2082_497_, v_inst_498_, v_inst_499_, v_inst_500_, v_inst_501_, v_inst_502_);
lean_dec(v_inst_502_);
lean_dec(v_inst_501_);
lean_dec_ref(v_inst_500_);
lean_dec_ref(v_inst_499_);
lean_dec_ref(v_inst_498_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___lam__0(lean_object* v_x_504_){
_start:
{
lean_object* v_snd_505_; 
v_snd_505_ = lean_ctor_get(v_x_504_, 1);
lean_inc(v_snd_505_);
return v_snd_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___lam__0___boxed(lean_object* v_x_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_Submodule_sndEquiv___redArg___lam__0(v_x_506_);
lean_dec_ref(v_x_506_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___lam__1(lean_object* v_toZero_508_, lean_object* v_n_509_){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_510_, 0, v_toZero_508_);
lean_ctor_set(v___x_510_, 1, v_n_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg(lean_object* v_inst_512_){
_start:
{
lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v_toZero_515_; lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_524_; 
v___x_513_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_512_);
v___x_514_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_513_);
v_toZero_515_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_524_ == 0)
{
lean_object* v_unused_525_; 
v_unused_525_ = lean_ctor_get(v___x_514_, 1);
lean_dec(v_unused_525_);
v___x_517_ = v___x_514_;
v_isShared_518_ = v_isSharedCheck_524_;
goto v_resetjp_516_;
}
else
{
lean_inc(v_toZero_515_);
lean_dec(v___x_514_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_524_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v___f_519_; lean_object* v___f_520_; lean_object* v___x_522_; 
v___f_519_ = ((lean_object*)(lp_mathlib_Submodule_sndEquiv___redArg___closed__0));
v___f_520_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_sndEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_520_, 0, v_toZero_515_);
if (v_isShared_518_ == 0)
{
lean_ctor_set(v___x_517_, 1, v___f_520_);
lean_ctor_set(v___x_517_, 0, v___f_519_);
v___x_522_ = v___x_517_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v___f_519_);
lean_ctor_set(v_reuseFailAlloc_523_, 1, v___f_520_);
v___x_522_ = v_reuseFailAlloc_523_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
return v___x_522_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___redArg___boxed(lean_object* v_inst_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_Submodule_sndEquiv___redArg(v_inst_526_);
lean_dec_ref(v_inst_526_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv(lean_object* v_R_528_, lean_object* v_M_529_, lean_object* v_M_u2082_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_inst_535_){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = lp_mathlib_Submodule_sndEquiv___redArg(v_inst_532_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_sndEquiv___boxed(lean_object* v_R_537_, lean_object* v_M_538_, lean_object* v_M_u2082_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_inst_544_){
_start:
{
lean_object* v_res_545_; 
v_res_545_ = lp_mathlib_Submodule_sndEquiv(v_R_537_, v_M_538_, v_M_u2082_539_, v_inst_540_, v_inst_541_, v_inst_542_, v_inst_543_, v_inst_544_);
lean_dec(v_inst_544_);
lean_dec(v_inst_543_);
lean_dec_ref(v_inst_542_);
lean_dec_ref(v_inst_541_);
lean_dec_ref(v_inst_540_);
return v_res_545_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_prodComm___closed__0(void){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodComm(lean_object* v_R_548_, lean_object* v_M_549_, lean_object* v_N_550_, lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_inst_555_){
_start:
{
lean_object* v___x_556_; lean_object* v_invFun_557_; lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_556_ = lean_obj_once(&lp_mathlib_LinearEquiv_prodComm___closed__0, &lp_mathlib_LinearEquiv_prodComm___closed__0_once, _init_lp_mathlib_LinearEquiv_prodComm___closed__0);
v_invFun_557_ = lean_ctor_get(v___x_556_, 1);
v___x_558_ = ((lean_object*)(lp_mathlib_LinearEquiv_prodComm___closed__1));
lean_inc(v_invFun_557_);
v___x_559_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_559_, 0, v___x_558_);
lean_ctor_set(v___x_559_, 1, v_invFun_557_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodComm___boxed(lean_object* v_R_560_, lean_object* v_M_561_, lean_object* v_N_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_inst_566_, lean_object* v_inst_567_){
_start:
{
lean_object* v_res_568_; 
v_res_568_ = lp_mathlib_LinearEquiv_prodComm(v_R_560_, v_M_561_, v_N_562_, v_inst_563_, v_inst_564_, v_inst_565_, v_inst_566_, v_inst_567_);
lean_dec(v_inst_567_);
lean_dec(v_inst_566_);
lean_dec_ref(v_inst_565_);
lean_dec_ref(v_inst_564_);
lean_dec_ref(v_inst_563_);
return v_res_568_;
}
}
static lean_object* _init_lp_mathlib_LinearEquiv_prodAssoc___closed__0(void){
_start:
{
lean_object* v___x_569_; 
v___x_569_ = lp_mathlib_Equiv_prodAssoc(lean_box(0), lean_box(0), lean_box(0));
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodAssoc(lean_object* v_R_570_, lean_object* v_M_u2081_571_, lean_object* v_M_u2082_572_, lean_object* v_M_u2083_573_, lean_object* v_inst_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_inst_580_){
_start:
{
lean_object* v___x_581_; lean_object* v_toFun_582_; lean_object* v_invFun_583_; lean_object* v___x_584_; 
v___x_581_ = lean_obj_once(&lp_mathlib_LinearEquiv_prodAssoc___closed__0, &lp_mathlib_LinearEquiv_prodAssoc___closed__0_once, _init_lp_mathlib_LinearEquiv_prodAssoc___closed__0);
v_toFun_582_ = lean_ctor_get(v___x_581_, 0);
v_invFun_583_ = lean_ctor_get(v___x_581_, 1);
lean_inc(v_invFun_583_);
lean_inc(v_toFun_582_);
v___x_584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_584_, 0, v_toFun_582_);
lean_ctor_set(v___x_584_, 1, v_invFun_583_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodAssoc___boxed(lean_object* v_R_585_, lean_object* v_M_u2081_586_, lean_object* v_M_u2082_587_, lean_object* v_M_u2083_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_inst_595_){
_start:
{
lean_object* v_res_596_; 
v_res_596_ = lp_mathlib_LinearEquiv_prodAssoc(v_R_585_, v_M_u2081_586_, v_M_u2082_587_, v_M_u2083_588_, v_inst_589_, v_inst_590_, v_inst_591_, v_inst_592_, v_inst_593_, v_inst_594_, v_inst_595_);
lean_dec(v_inst_595_);
lean_dec(v_inst_594_);
lean_dec(v_inst_593_);
lean_dec_ref(v_inst_592_);
lean_dec_ref(v_inst_591_);
lean_dec_ref(v_inst_590_);
lean_dec_ref(v_inst_589_);
return v_res_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg___lam__0(lean_object* v_toNeg_597_, lean_object* v_x_598_){
_start:
{
lean_object* v_fst_599_; lean_object* v_snd_600_; lean_object* v___x_602_; uint8_t v_isShared_603_; uint8_t v_isSharedCheck_608_; 
v_fst_599_ = lean_ctor_get(v_x_598_, 0);
v_snd_600_ = lean_ctor_get(v_x_598_, 1);
v_isSharedCheck_608_ = !lean_is_exclusive(v_x_598_);
if (v_isSharedCheck_608_ == 0)
{
v___x_602_ = v_x_598_;
v_isShared_603_ = v_isSharedCheck_608_;
goto v_resetjp_601_;
}
else
{
lean_inc(v_snd_600_);
lean_inc(v_fst_599_);
lean_dec(v_x_598_);
v___x_602_ = lean_box(0);
v_isShared_603_ = v_isSharedCheck_608_;
goto v_resetjp_601_;
}
v_resetjp_601_:
{
lean_object* v___x_604_; lean_object* v___x_606_; 
v___x_604_ = lean_apply_1(v_toNeg_597_, v_snd_600_);
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 1, v_fst_599_);
lean_ctor_set(v___x_602_, 0, v___x_604_);
v___x_606_ = v___x_602_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_604_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v_fst_599_);
v___x_606_ = v_reuseFailAlloc_607_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
return v___x_606_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg___lam__1(lean_object* v_toNeg_609_, lean_object* v_x_610_){
_start:
{
lean_object* v_fst_611_; lean_object* v_snd_612_; lean_object* v___x_614_; uint8_t v_isShared_615_; uint8_t v_isSharedCheck_620_; 
v_fst_611_ = lean_ctor_get(v_x_610_, 0);
v_snd_612_ = lean_ctor_get(v_x_610_, 1);
v_isSharedCheck_620_ = !lean_is_exclusive(v_x_610_);
if (v_isSharedCheck_620_ == 0)
{
v___x_614_ = v_x_610_;
v_isShared_615_ = v_isSharedCheck_620_;
goto v_resetjp_613_;
}
else
{
lean_inc(v_snd_612_);
lean_inc(v_fst_611_);
lean_dec(v_x_610_);
v___x_614_ = lean_box(0);
v_isShared_615_ = v_isSharedCheck_620_;
goto v_resetjp_613_;
}
v_resetjp_613_:
{
lean_object* v___x_616_; lean_object* v___x_618_; 
v___x_616_ = lean_apply_1(v_toNeg_609_, v_fst_611_);
if (v_isShared_615_ == 0)
{
lean_ctor_set(v___x_614_, 1, v___x_616_);
lean_ctor_set(v___x_614_, 0, v_snd_612_);
v___x_618_ = v___x_614_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_snd_612_);
lean_ctor_set(v_reuseFailAlloc_619_, 1, v___x_616_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg(lean_object* v_inst_621_){
_start:
{
lean_object* v___x_622_; lean_object* v_toNeg_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_632_; 
v___x_622_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_621_);
v_toNeg_623_ = lean_ctor_get(v___x_622_, 1);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_632_ == 0)
{
lean_object* v_unused_633_; 
v_unused_633_ = lean_ctor_get(v___x_622_, 0);
lean_dec(v_unused_633_);
v___x_625_ = v___x_622_;
v_isShared_626_ = v_isSharedCheck_632_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_toNeg_623_);
lean_dec(v___x_622_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_632_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___f_627_; lean_object* v___f_628_; lean_object* v___x_630_; 
lean_inc(v_toNeg_623_);
v___f_627_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_skewSwap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_627_, 0, v_toNeg_623_);
v___f_628_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_skewSwap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_628_, 0, v_toNeg_623_);
if (v_isShared_626_ == 0)
{
lean_ctor_set(v___x_625_, 1, v___f_628_);
lean_ctor_set(v___x_625_, 0, v___f_627_);
v___x_630_ = v___x_625_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v___f_627_);
lean_ctor_set(v_reuseFailAlloc_631_, 1, v___f_628_);
v___x_630_ = v_reuseFailAlloc_631_;
goto v_reusejp_629_;
}
v_reusejp_629_:
{
return v___x_630_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___redArg___boxed(lean_object* v_inst_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_mathlib_LinearEquiv_skewSwap___redArg(v_inst_634_);
lean_dec_ref(v_inst_634_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap(lean_object* v_R_636_, lean_object* v_M_637_, lean_object* v_N_638_, lean_object* v_inst_639_, lean_object* v_inst_640_, lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_inst_643_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lp_mathlib_LinearEquiv_skewSwap___redArg(v_inst_641_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewSwap___boxed(lean_object* v_R_645_, lean_object* v_M_646_, lean_object* v_N_647_, lean_object* v_inst_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_mathlib_LinearEquiv_skewSwap(v_R_645_, v_M_646_, v_N_647_, v_inst_648_, v_inst_649_, v_inst_650_, v_inst_651_, v_inst_652_);
lean_dec(v_inst_652_);
lean_dec(v_inst_651_);
lean_dec_ref(v_inst_650_);
lean_dec_ref(v_inst_649_);
lean_dec_ref(v_inst_648_);
return v_res_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___lam__0(lean_object* v_mnmn_654_){
_start:
{
lean_object* v_fst_655_; lean_object* v_snd_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_681_; 
v_fst_655_ = lean_ctor_get(v_mnmn_654_, 0);
v_snd_656_ = lean_ctor_get(v_mnmn_654_, 1);
v_isSharedCheck_681_ = !lean_is_exclusive(v_mnmn_654_);
if (v_isSharedCheck_681_ == 0)
{
v___x_658_ = v_mnmn_654_;
v_isShared_659_ = v_isSharedCheck_681_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_snd_656_);
lean_inc(v_fst_655_);
lean_dec(v_mnmn_654_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_681_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v_fst_660_; lean_object* v_snd_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_680_; 
v_fst_660_ = lean_ctor_get(v_fst_655_, 0);
v_snd_661_ = lean_ctor_get(v_fst_655_, 1);
v_isSharedCheck_680_ = !lean_is_exclusive(v_fst_655_);
if (v_isSharedCheck_680_ == 0)
{
v___x_663_ = v_fst_655_;
v_isShared_664_ = v_isSharedCheck_680_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_snd_661_);
lean_inc(v_fst_660_);
lean_dec(v_fst_655_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_680_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v_fst_665_; lean_object* v_snd_666_; lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_679_; 
v_fst_665_ = lean_ctor_get(v_snd_656_, 0);
v_snd_666_ = lean_ctor_get(v_snd_656_, 1);
v_isSharedCheck_679_ = !lean_is_exclusive(v_snd_656_);
if (v_isSharedCheck_679_ == 0)
{
v___x_668_ = v_snd_656_;
v_isShared_669_ = v_isSharedCheck_679_;
goto v_resetjp_667_;
}
else
{
lean_inc(v_snd_666_);
lean_inc(v_fst_665_);
lean_dec(v_snd_656_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_679_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v___x_671_; 
if (v_isShared_669_ == 0)
{
lean_ctor_set(v___x_668_, 1, v_fst_665_);
lean_ctor_set(v___x_668_, 0, v_fst_660_);
v___x_671_ = v___x_668_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v_fst_660_);
lean_ctor_set(v_reuseFailAlloc_678_, 1, v_fst_665_);
v___x_671_ = v_reuseFailAlloc_678_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
lean_object* v___x_673_; 
if (v_isShared_664_ == 0)
{
lean_ctor_set(v___x_663_, 1, v_snd_666_);
lean_ctor_set(v___x_663_, 0, v_snd_661_);
v___x_673_ = v___x_663_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_snd_661_);
lean_ctor_set(v_reuseFailAlloc_677_, 1, v_snd_666_);
v___x_673_ = v_reuseFailAlloc_677_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
lean_object* v___x_675_; 
if (v_isShared_659_ == 0)
{
lean_ctor_set(v___x_658_, 1, v___x_673_);
lean_ctor_set(v___x_658_, 0, v___x_671_);
v___x_675_ = v___x_658_;
goto v_reusejp_674_;
}
else
{
lean_object* v_reuseFailAlloc_676_; 
v_reuseFailAlloc_676_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_676_, 0, v___x_671_);
lean_ctor_set(v_reuseFailAlloc_676_, 1, v___x_673_);
v___x_675_ = v_reuseFailAlloc_676_;
goto v_reusejp_674_;
}
v_reusejp_674_:
{
return v___x_675_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___lam__1(lean_object* v_mmnn_682_){
_start:
{
lean_object* v_fst_683_; lean_object* v_snd_684_; lean_object* v___x_686_; uint8_t v_isShared_687_; uint8_t v_isSharedCheck_709_; 
v_fst_683_ = lean_ctor_get(v_mmnn_682_, 0);
v_snd_684_ = lean_ctor_get(v_mmnn_682_, 1);
v_isSharedCheck_709_ = !lean_is_exclusive(v_mmnn_682_);
if (v_isSharedCheck_709_ == 0)
{
v___x_686_ = v_mmnn_682_;
v_isShared_687_ = v_isSharedCheck_709_;
goto v_resetjp_685_;
}
else
{
lean_inc(v_snd_684_);
lean_inc(v_fst_683_);
lean_dec(v_mmnn_682_);
v___x_686_ = lean_box(0);
v_isShared_687_ = v_isSharedCheck_709_;
goto v_resetjp_685_;
}
v_resetjp_685_:
{
lean_object* v_fst_688_; lean_object* v_snd_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_708_; 
v_fst_688_ = lean_ctor_get(v_fst_683_, 0);
v_snd_689_ = lean_ctor_get(v_fst_683_, 1);
v_isSharedCheck_708_ = !lean_is_exclusive(v_fst_683_);
if (v_isSharedCheck_708_ == 0)
{
v___x_691_ = v_fst_683_;
v_isShared_692_ = v_isSharedCheck_708_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_snd_689_);
lean_inc(v_fst_688_);
lean_dec(v_fst_683_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_708_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v_fst_693_; lean_object* v_snd_694_; lean_object* v___x_696_; uint8_t v_isShared_697_; uint8_t v_isSharedCheck_707_; 
v_fst_693_ = lean_ctor_get(v_snd_684_, 0);
v_snd_694_ = lean_ctor_get(v_snd_684_, 1);
v_isSharedCheck_707_ = !lean_is_exclusive(v_snd_684_);
if (v_isSharedCheck_707_ == 0)
{
v___x_696_ = v_snd_684_;
v_isShared_697_ = v_isSharedCheck_707_;
goto v_resetjp_695_;
}
else
{
lean_inc(v_snd_694_);
lean_inc(v_fst_693_);
lean_dec(v_snd_684_);
v___x_696_ = lean_box(0);
v_isShared_697_ = v_isSharedCheck_707_;
goto v_resetjp_695_;
}
v_resetjp_695_:
{
lean_object* v___x_699_; 
if (v_isShared_697_ == 0)
{
lean_ctor_set(v___x_696_, 1, v_fst_693_);
lean_ctor_set(v___x_696_, 0, v_fst_688_);
v___x_699_ = v___x_696_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_706_; 
v_reuseFailAlloc_706_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_706_, 0, v_fst_688_);
lean_ctor_set(v_reuseFailAlloc_706_, 1, v_fst_693_);
v___x_699_ = v_reuseFailAlloc_706_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
lean_object* v___x_701_; 
if (v_isShared_692_ == 0)
{
lean_ctor_set(v___x_691_, 1, v_snd_694_);
lean_ctor_set(v___x_691_, 0, v_snd_689_);
v___x_701_ = v___x_691_;
goto v_reusejp_700_;
}
else
{
lean_object* v_reuseFailAlloc_705_; 
v_reuseFailAlloc_705_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_705_, 0, v_snd_689_);
lean_ctor_set(v_reuseFailAlloc_705_, 1, v_snd_694_);
v___x_701_ = v_reuseFailAlloc_705_;
goto v_reusejp_700_;
}
v_reusejp_700_:
{
lean_object* v___x_703_; 
if (v_isShared_687_ == 0)
{
lean_ctor_set(v___x_686_, 1, v___x_701_);
lean_ctor_set(v___x_686_, 0, v___x_699_);
v___x_703_ = v___x_686_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v___x_699_);
lean_ctor_set(v_reuseFailAlloc_704_, 1, v___x_701_);
v___x_703_ = v_reuseFailAlloc_704_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
return v___x_703_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm(lean_object* v_R_715_, lean_object* v_M_716_, lean_object* v_M_u2082_717_, lean_object* v_M_u2083_718_, lean_object* v_M_u2084_719_, lean_object* v_inst_720_, lean_object* v_inst_721_, lean_object* v_inst_722_, lean_object* v_inst_723_, lean_object* v_inst_724_, lean_object* v_inst_725_, lean_object* v_inst_726_, lean_object* v_inst_727_, lean_object* v_inst_728_){
_start:
{
lean_object* v___x_729_; 
v___x_729_ = ((lean_object*)(lp_mathlib_LinearEquiv_prodProdProdComm___closed__2));
return v___x_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodProdProdComm___boxed(lean_object* v_R_730_, lean_object* v_M_731_, lean_object* v_M_u2082_732_, lean_object* v_M_u2083_733_, lean_object* v_M_u2084_734_, lean_object* v_inst_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_inst_739_, lean_object* v_inst_740_, lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v_inst_743_){
_start:
{
lean_object* v_res_744_; 
v_res_744_ = lp_mathlib_LinearEquiv_prodProdProdComm(v_R_730_, v_M_731_, v_M_u2082_732_, v_M_u2083_733_, v_M_u2084_734_, v_inst_735_, v_inst_736_, v_inst_737_, v_inst_738_, v_inst_739_, v_inst_740_, v_inst_741_, v_inst_742_, v_inst_743_);
lean_dec(v_inst_743_);
lean_dec(v_inst_742_);
lean_dec(v_inst_741_);
lean_dec(v_inst_740_);
lean_dec_ref(v_inst_739_);
lean_dec_ref(v_inst_738_);
lean_dec_ref(v_inst_737_);
lean_dec_ref(v_inst_736_);
lean_dec_ref(v_inst_735_);
return v_res_744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodCongr___redArg(lean_object* v_e_u2081_745_, lean_object* v_e_u2082_746_){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v_toFun_750_; lean_object* v_invFun_751_; lean_object* v___x_753_; uint8_t v_isShared_754_; uint8_t v_isSharedCheck_758_; 
v___x_747_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_e_u2081_745_);
v___x_748_ = lp_mathlib_LinearEquiv_toAddEquiv___redArg(v_e_u2082_746_);
v___x_749_ = lp_mathlib_Equiv_prodCongr___redArg(v___x_747_, v___x_748_);
v_toFun_750_ = lean_ctor_get(v___x_749_, 0);
v_invFun_751_ = lean_ctor_get(v___x_749_, 1);
v_isSharedCheck_758_ = !lean_is_exclusive(v___x_749_);
if (v_isSharedCheck_758_ == 0)
{
v___x_753_ = v___x_749_;
v_isShared_754_ = v_isSharedCheck_758_;
goto v_resetjp_752_;
}
else
{
lean_inc(v_invFun_751_);
lean_inc(v_toFun_750_);
lean_dec(v___x_749_);
v___x_753_ = lean_box(0);
v_isShared_754_ = v_isSharedCheck_758_;
goto v_resetjp_752_;
}
v_resetjp_752_:
{
lean_object* v___x_756_; 
if (v_isShared_754_ == 0)
{
v___x_756_ = v___x_753_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v_toFun_750_);
lean_ctor_set(v_reuseFailAlloc_757_, 1, v_invFun_751_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodCongr(lean_object* v_R_759_, lean_object* v_M_760_, lean_object* v_M_u2082_761_, lean_object* v_M_u2083_762_, lean_object* v_M_u2084_763_, lean_object* v_inst_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_inst_767_, lean_object* v_inst_768_, lean_object* v_module__M_769_, lean_object* v_module__M_u2082_770_, lean_object* v_module__M_u2083_771_, lean_object* v_module__M_u2084_772_, lean_object* v_e_u2081_773_, lean_object* v_e_u2082_774_){
_start:
{
lean_object* v___x_775_; 
v___x_775_ = lp_mathlib_LinearEquiv_prodCongr___redArg(v_e_u2081_773_, v_e_u2082_774_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodCongr___boxed(lean_object* v_R_776_, lean_object* v_M_777_, lean_object* v_M_u2082_778_, lean_object* v_M_u2083_779_, lean_object* v_M_u2084_780_, lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_inst_784_, lean_object* v_inst_785_, lean_object* v_module__M_786_, lean_object* v_module__M_u2082_787_, lean_object* v_module__M_u2083_788_, lean_object* v_module__M_u2084_789_, lean_object* v_e_u2081_790_, lean_object* v_e_u2082_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_mathlib_LinearEquiv_prodCongr(v_R_776_, v_M_777_, v_M_u2082_778_, v_M_u2083_779_, v_M_u2084_780_, v_inst_781_, v_inst_782_, v_inst_783_, v_inst_784_, v_inst_785_, v_module__M_786_, v_module__M_u2082_787_, v_module__M_u2083_788_, v_module__M_u2084_789_, v_e_u2081_790_, v_e_u2082_791_);
lean_dec(v_module__M_u2084_789_);
lean_dec(v_module__M_u2083_788_);
lean_dec(v_module__M_u2082_787_);
lean_dec(v_module__M_786_);
lean_dec_ref(v_inst_785_);
lean_dec_ref(v_inst_784_);
lean_dec_ref(v_inst_783_);
lean_dec_ref(v_inst_782_);
lean_dec_ref(v_inst_781_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___redArg___lam__0(lean_object* v_e_u2081_793_, lean_object* v_e_u2082_794_, lean_object* v_f_795_, lean_object* v_toSub_796_, lean_object* v_p_797_){
_start:
{
lean_object* v_fst_798_; lean_object* v_snd_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_814_; 
v_fst_798_ = lean_ctor_get(v_p_797_, 0);
v_snd_799_ = lean_ctor_get(v_p_797_, 1);
v_isSharedCheck_814_ = !lean_is_exclusive(v_p_797_);
if (v_isSharedCheck_814_ == 0)
{
v___x_801_ = v_p_797_;
v_isShared_802_ = v_isSharedCheck_814_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_snd_799_);
lean_inc(v_fst_798_);
lean_dec(v_p_797_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_814_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v___x_803_; lean_object* v_toLinearMap_804_; lean_object* v___x_805_; lean_object* v_toLinearMap_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_812_; 
v___x_803_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_u2081_793_);
v_toLinearMap_804_ = lean_ctor_get(v___x_803_, 0);
lean_inc(v_toLinearMap_804_);
lean_dec_ref(v___x_803_);
v___x_805_ = lp_mathlib_LinearEquiv_symm___redArg(v_e_u2082_794_);
v_toLinearMap_806_ = lean_ctor_get(v___x_805_, 0);
lean_inc(v_toLinearMap_806_);
lean_dec_ref(v___x_805_);
v___x_807_ = lean_apply_1(v_toLinearMap_804_, v_fst_798_);
lean_inc(v___x_807_);
v___x_808_ = lean_apply_1(v_f_795_, v___x_807_);
v___x_809_ = lean_apply_2(v_toSub_796_, v_snd_799_, v___x_808_);
v___x_810_ = lean_apply_1(v_toLinearMap_806_, v___x_809_);
if (v_isShared_802_ == 0)
{
lean_ctor_set(v___x_801_, 1, v___x_810_);
lean_ctor_set(v___x_801_, 0, v___x_807_);
v___x_812_ = v___x_801_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v___x_807_);
lean_ctor_set(v_reuseFailAlloc_813_, 1, v___x_810_);
v___x_812_ = v_reuseFailAlloc_813_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
return v___x_812_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___redArg___lam__1(lean_object* v___f_815_, lean_object* v_toLinearMap_816_, lean_object* v___f_817_, lean_object* v_f_818_, lean_object* v_toAdd_819_, lean_object* v___y_820_){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; 
lean_inc_ref(v___y_820_);
v___x_821_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___f_815_, v_toLinearMap_816_, v___y_820_);
v___x_822_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___f_817_, v_f_818_, v___y_820_);
v___x_823_ = lean_apply_2(v_toAdd_819_, v___x_821_, v___x_822_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___redArg(lean_object* v_inst_824_, lean_object* v_e_u2081_825_, lean_object* v_e_u2082_826_, lean_object* v_f_827_){
_start:
{
lean_object* v_toAddMonoid_828_; lean_object* v_toSub_829_; lean_object* v_toLinearMap_830_; lean_object* v_toLinearMap_831_; lean_object* v_toAdd_832_; lean_object* v___f_833_; lean_object* v___f_834_; lean_object* v___f_835_; lean_object* v___f_836_; lean_object* v___f_837_; lean_object* v___x_838_; lean_object* v___x_839_; 
v_toAddMonoid_828_ = lean_ctor_get(v_inst_824_, 0);
lean_inc_ref(v_toAddMonoid_828_);
v_toSub_829_ = lean_ctor_get(v_inst_824_, 2);
lean_inc(v_toSub_829_);
lean_dec_ref(v_inst_824_);
v_toLinearMap_830_ = lean_ctor_get(v_e_u2081_825_, 0);
lean_inc(v_toLinearMap_830_);
v_toLinearMap_831_ = lean_ctor_get(v_e_u2082_826_, 0);
lean_inc(v_toLinearMap_831_);
v_toAdd_832_ = lean_ctor_get(v_toAddMonoid_828_, 1);
lean_inc(v_toAdd_832_);
lean_dec_ref(v_toAddMonoid_828_);
lean_inc(v_f_827_);
v___f_833_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_skewProd___redArg___lam__0), 5, 4);
lean_closure_set(v___f_833_, 0, v_e_u2081_825_);
lean_closure_set(v___f_833_, 1, v_e_u2082_826_);
lean_closure_set(v___f_833_, 2, v_f_827_);
lean_closure_set(v___f_833_, 3, v_toSub_829_);
v___f_834_ = ((lean_object*)(lp_mathlib_LinearMap_fst___closed__0));
v___f_835_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_835_, 0, v___f_834_);
lean_closure_set(v___f_835_, 1, v_toLinearMap_830_);
v___f_836_ = ((lean_object*)(lp_mathlib_LinearMap_snd___closed__0));
v___f_837_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_skewProd___redArg___lam__1), 6, 5);
lean_closure_set(v___f_837_, 0, v___f_836_);
lean_closure_set(v___f_837_, 1, v_toLinearMap_831_);
lean_closure_set(v___f_837_, 2, v___f_834_);
lean_closure_set(v___f_837_, 3, v_f_827_);
lean_closure_set(v___f_837_, 4, v_toAdd_832_);
v___x_838_ = lp_mathlib_LinearMap_prod___redArg(v___f_835_, v___f_837_);
v___x_839_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_839_, 0, v___x_838_);
lean_ctor_set(v___x_839_, 1, v___f_833_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd(lean_object* v_R_840_, lean_object* v_M_841_, lean_object* v_M_u2082_842_, lean_object* v_M_u2083_843_, lean_object* v_M_u2084_844_, lean_object* v_inst_845_, lean_object* v_inst_846_, lean_object* v_inst_847_, lean_object* v_inst_848_, lean_object* v_inst_849_, lean_object* v_module__M_850_, lean_object* v_module__M_u2082_851_, lean_object* v_module__M_u2083_852_, lean_object* v_module__M_u2084_853_, lean_object* v_e_u2081_854_, lean_object* v_e_u2082_855_, lean_object* v_f_856_){
_start:
{
lean_object* v___x_857_; 
v___x_857_ = lp_mathlib_LinearEquiv_skewProd___redArg(v_inst_849_, v_e_u2081_854_, v_e_u2082_855_, v_f_856_);
return v___x_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_skewProd___boxed(lean_object** _args){
lean_object* v_R_858_ = _args[0];
lean_object* v_M_859_ = _args[1];
lean_object* v_M_u2082_860_ = _args[2];
lean_object* v_M_u2083_861_ = _args[3];
lean_object* v_M_u2084_862_ = _args[4];
lean_object* v_inst_863_ = _args[5];
lean_object* v_inst_864_ = _args[6];
lean_object* v_inst_865_ = _args[7];
lean_object* v_inst_866_ = _args[8];
lean_object* v_inst_867_ = _args[9];
lean_object* v_module__M_868_ = _args[10];
lean_object* v_module__M_u2082_869_ = _args[11];
lean_object* v_module__M_u2083_870_ = _args[12];
lean_object* v_module__M_u2084_871_ = _args[13];
lean_object* v_e_u2081_872_ = _args[14];
lean_object* v_e_u2082_873_ = _args[15];
lean_object* v_f_874_ = _args[16];
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_LinearEquiv_skewProd(v_R_858_, v_M_859_, v_M_u2082_860_, v_M_u2083_861_, v_M_u2084_862_, v_inst_863_, v_inst_864_, v_inst_865_, v_inst_866_, v_inst_867_, v_module__M_868_, v_module__M_u2082_869_, v_module__M_u2083_870_, v_module__M_u2084_871_, v_e_u2081_872_, v_e_u2082_873_, v_f_874_);
lean_dec(v_module__M_u2084_871_);
lean_dec(v_module__M_u2083_870_);
lean_dec(v_module__M_u2082_869_);
lean_dec(v_module__M_868_);
lean_dec_ref(v_inst_866_);
lean_dec_ref(v_inst_865_);
lean_dec_ref(v_inst_864_);
lean_dec_ref(v_inst_863_);
return v_res_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueProd___redArg(lean_object* v_inst_876_){
_start:
{
lean_object* v___x_877_; lean_object* v___x_878_; 
v___x_877_ = lp_mathlib_Equiv_uniqueProd___redArg(v_inst_876_);
v___x_878_ = lp_mathlib_AddEquiv_toLinearEquiv___redArg(v___x_877_);
return v___x_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueProd(lean_object* v_R_879_, lean_object* v_M_880_, lean_object* v_M_u2082_881_, lean_object* v_inst_882_, lean_object* v_inst_883_, lean_object* v_inst_884_, lean_object* v_inst_885_, lean_object* v_inst_886_, lean_object* v_inst_887_){
_start:
{
lean_object* v___x_888_; 
v___x_888_ = lp_mathlib_LinearEquiv_uniqueProd___redArg(v_inst_887_);
return v___x_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_uniqueProd___boxed(lean_object* v_R_889_, lean_object* v_M_890_, lean_object* v_M_u2082_891_, lean_object* v_inst_892_, lean_object* v_inst_893_, lean_object* v_inst_894_, lean_object* v_inst_895_, lean_object* v_inst_896_, lean_object* v_inst_897_){
_start:
{
lean_object* v_res_898_; 
v_res_898_ = lp_mathlib_LinearEquiv_uniqueProd(v_R_889_, v_M_890_, v_M_u2082_891_, v_inst_892_, v_inst_893_, v_inst_894_, v_inst_895_, v_inst_896_, v_inst_897_);
lean_dec(v_inst_896_);
lean_dec(v_inst_895_);
lean_dec_ref(v_inst_894_);
lean_dec_ref(v_inst_893_);
lean_dec_ref(v_inst_892_);
return v_res_898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodUnique___redArg(lean_object* v_inst_899_){
_start:
{
lean_object* v___x_900_; lean_object* v___x_901_; 
v___x_900_ = lp_mathlib_Equiv_prodUnique___redArg(v_inst_899_);
v___x_901_ = lp_mathlib_AddEquiv_toLinearEquiv___redArg(v___x_900_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodUnique(lean_object* v_R_902_, lean_object* v_M_903_, lean_object* v_M_u2082_904_, lean_object* v_inst_905_, lean_object* v_inst_906_, lean_object* v_inst_907_, lean_object* v_inst_908_, lean_object* v_inst_909_, lean_object* v_inst_910_){
_start:
{
lean_object* v___x_911_; 
v___x_911_ = lp_mathlib_LinearEquiv_prodUnique___redArg(v_inst_910_);
return v___x_911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_prodUnique___boxed(lean_object* v_R_912_, lean_object* v_M_913_, lean_object* v_M_u2082_914_, lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_inst_918_, lean_object* v_inst_919_, lean_object* v_inst_920_){
_start:
{
lean_object* v_res_921_; 
v_res_921_ = lp_mathlib_LinearEquiv_prodUnique(v_R_912_, v_M_913_, v_M_u2082_914_, v_inst_915_, v_inst_916_, v_inst_917_, v_inst_918_, v_inst_919_, v_inst_920_);
lean_dec(v_inst_919_);
lean_dec(v_inst_918_);
lean_dec_ref(v_inst_917_);
lean_dec_ref(v_inst_916_);
lean_dec_ref(v_inst_915_);
return v_res_921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_graph(lean_object* v_R_922_, lean_object* v_M_923_, lean_object* v_M_u2082_924_, lean_object* v_inst_925_, lean_object* v_inst_926_, lean_object* v_inst_927_, lean_object* v_inst_928_, lean_object* v_inst_929_, lean_object* v_f_930_){
_start:
{
lean_object* v___x_931_; 
v___x_931_ = lean_box(0);
return v___x_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_graph___boxed(lean_object* v_R_932_, lean_object* v_M_933_, lean_object* v_M_u2082_934_, lean_object* v_inst_935_, lean_object* v_inst_936_, lean_object* v_inst_937_, lean_object* v_inst_938_, lean_object* v_inst_939_, lean_object* v_f_940_){
_start:
{
lean_object* v_res_941_; 
v_res_941_ = lp_mathlib_LinearMap_graph(v_R_932_, v_M_933_, v_M_u2082_934_, v_inst_935_, v_inst_936_, v_inst_937_, v_inst_938_, v_inst_939_, v_f_940_);
lean_dec(v_f_940_);
lean_dec(v_inst_939_);
lean_dec(v_inst_938_);
lean_dec_ref(v_inst_937_);
lean_dec_ref(v_inst_936_);
lean_dec_ref(v_inst_935_);
return v_res_941_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Graph(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Graph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Graph(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Graph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
