// Lean compiler output
// Module: Mathlib.Data.Matrix.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Opposite public import Mathlib.Algebra.Algebra.Pi public import Mathlib.Algebra.BigOperators.RingEquiv public import Mathlib.Basic.Finite.Prod public import Mathlib.Data.Matrix.Mul public import Mathlib.GroupTheory.DedekindFinite public import Mathlib.LinearAlgebra.Pi
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
uint8_t lp_mathlib_Fintype_decidablePiFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_transpose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_op___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_unop___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_piFinset___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_diagonal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Matrix_diag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_ofLinearEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_ofLinearEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Matrix_diagAddMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Matrix_diag, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Matrix_diagAddMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_Matrix_diagAddMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Matrix_scalar___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_const___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Matrix_scalar___redArg___closed__0 = (const lean_object*)&lp_mathlib_Matrix_scalar___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalar___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAlgebra___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalarAlgHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalarAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalarAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrixLinear___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrixLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrixLinear___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mapMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mapMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingEquiv_mopMatrix___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_unop___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RingEquiv_mopMatrix___lam__0___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_mopMatrix___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingEquiv_mopMatrix___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_op___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RingEquiv_mopMatrix___lam__1___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_mopMatrix___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingEquiv_mopMatrix___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_mopMatrix___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_mopMatrix___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_mopMatrix___closed__0_value;
static const lean_closure_object lp_mathlib_RingEquiv_mopMatrix___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingEquiv_mopMatrix___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingEquiv_mopMatrix___closed__1 = (const lean_object*)&lp_mathlib_RingEquiv_mopMatrix___closed__1_value;
static const lean_ctor_object lp_mathlib_RingEquiv_mopMatrix___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RingEquiv_mopMatrix___closed__0_value),((lean_object*)&lp_mathlib_RingEquiv_mopMatrix___closed__1_value)}};
static const lean_object* lp_mathlib_RingEquiv_mopMatrix___closed__2 = (const lean_object*)&lp_mathlib_RingEquiv_mopMatrix___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapMatrix___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_matrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_matrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_matrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_matrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_matrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_matrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_matrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subring_matrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_matrix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_matrix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_piEquiv___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_piEquiv___lam__3___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Matrix_piEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Matrix_piEquiv___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Matrix_piEquiv___closed__0 = (const lean_object*)&lp_mathlib_Matrix_piEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Matrix_piEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Matrix_piEquiv___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Matrix_piEquiv___closed__1 = (const lean_object*)&lp_mathlib_Matrix_piEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Matrix_piEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_piEquiv___closed__0_value),((lean_object*)&lp_mathlib_Matrix_piEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Matrix_piEquiv___closed__2 = (const lean_object*)&lp_mathlib_Matrix_piEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_piAddEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_piAddEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Matrix_transposeAddEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Matrix_transpose, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Matrix_transposeAddEquiv___closed__0 = (const lean_object*)&lp_mathlib_Matrix_transposeAddEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Matrix_transposeAddEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Matrix_transposeAddEquiv___closed__0_value),((lean_object*)&lp_mathlib_Matrix_transposeAddEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Matrix_transposeAddEquiv___closed__1 = (const lean_object*)&lp_mathlib_Matrix_transposeAddEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Matrix_transposeRingEquiv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Matrix_transposeRingEquiv___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; uint8_t v___x_6_; 
v___x_5_ = lean_apply_2(v_inst_1_, v___y_3_, v___y_4_);
v___x_6_ = lean_unbox(v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___redArg___lam__0___boxed(lean_object* v_inst_7_, lean_object* v_a_8_, lean_object* v___y_9_, lean_object* v___y_10_){
_start:
{
uint8_t v_res_11_; lean_object* v_r_12_; 
v_res_11_ = lp_mathlib_Matrix_decidableEq___redArg___lam__0(v_inst_7_, v_a_8_, v___y_9_, v___y_10_);
lean_dec(v_a_8_);
v_r_12_ = lean_box(v_res_11_);
return v_r_12_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq___redArg___lam__1(lean_object* v___f_13_, lean_object* v_inst_14_, lean_object* v_a_15_, lean_object* v_a_16_, lean_object* v_b_17_){
_start:
{
uint8_t v___x_18_; 
v___x_18_ = lp_mathlib_Fintype_decidablePiFintype___redArg(v___f_13_, v_inst_14_, v_a_16_, v_b_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___redArg___lam__1___boxed(lean_object* v___f_19_, lean_object* v_inst_20_, lean_object* v_a_21_, lean_object* v_a_22_, lean_object* v_b_23_){
_start:
{
uint8_t v_res_24_; lean_object* v_r_25_; 
v_res_24_ = lp_mathlib_Matrix_decidableEq___redArg___lam__1(v___f_19_, v_inst_20_, v_a_21_, v_a_22_, v_b_23_);
lean_dec(v_a_21_);
v_r_25_ = lean_box(v_res_24_);
return v_r_25_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq___redArg(lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_a_29_, lean_object* v_b_30_){
_start:
{
lean_object* v___f_31_; lean_object* v___f_32_; uint8_t v___x_33_; 
v___f_31_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_decidableEq___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_31_, 0, v_inst_26_);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_decidableEq___redArg___lam__1___boxed), 5, 2);
lean_closure_set(v___f_32_, 0, v___f_31_);
lean_closure_set(v___f_32_, 1, v_inst_28_);
v___x_33_ = lp_mathlib_Fintype_decidablePiFintype___redArg(v___f_32_, v_inst_27_, v_a_29_, v_b_30_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___redArg___boxed(lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_a_37_, lean_object* v_b_38_){
_start:
{
uint8_t v_res_39_; lean_object* v_r_40_; 
v_res_39_ = lp_mathlib_Matrix_decidableEq___redArg(v_inst_34_, v_inst_35_, v_inst_36_, v_a_37_, v_b_38_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Matrix_decidableEq(lean_object* v_m_41_, lean_object* v_n_42_, lean_object* v_00_u03b1_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_a_47_, lean_object* v_b_48_){
_start:
{
uint8_t v___x_49_; 
v___x_49_ = lp_mathlib_Matrix_decidableEq___redArg(v_inst_44_, v_inst_45_, v_inst_46_, v_a_47_, v_b_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_decidableEq___boxed(lean_object* v_m_50_, lean_object* v_n_51_, lean_object* v_00_u03b1_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_a_56_, lean_object* v_b_57_){
_start:
{
uint8_t v_res_58_; lean_object* v_r_59_; 
v_res_58_ = lp_mathlib_Matrix_decidableEq(v_m_50_, v_n_51_, v_00_u03b1_52_, v_inst_53_, v_inst_54_, v_inst_55_, v_a_56_, v_b_57_);
v_r_59_ = lean_box(v_res_58_);
return v_r_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0(lean_object* v_inst_60_, lean_object* v_x_61_){
_start:
{
lean_inc(v_inst_60_);
return v_inst_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0___boxed(lean_object* v_inst_62_, lean_object* v_x_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0(v_inst_62_, v_x_63_);
lean_dec(v_x_63_);
lean_dec(v_inst_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1(lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v___f_67_, lean_object* v_x_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Fintype_piFinset___redArg(v_inst_65_, v_inst_66_, v___f_67_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1___boxed(lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v___f_72_, lean_object* v_x_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1(v_inst_70_, v_inst_71_, v___f_72_, v_x_73_);
lean_dec(v_x_73_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg(lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v___f_80_; lean_object* v___f_81_; lean_object* v___x_82_; 
v___f_80_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_80_, 0, v_inst_79_);
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_81_, 0, v_inst_78_);
lean_closure_set(v___f_81_, 1, v_inst_77_);
lean_closure_set(v___f_81_, 2, v___f_80_);
v___x_82_ = lp_mathlib_Fintype_piFinset___redArg(v_inst_76_, v_inst_75_, v___f_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1(lean_object* v_n_83_, lean_object* v_m_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_00_u03b1_89_, lean_object* v_inst_90_){
_start:
{
lean_object* v___f_91_; lean_object* v___f_92_; lean_object* v___x_93_; 
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_91_, 0, v_inst_90_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_92_, 0, v_inst_88_);
lean_closure_set(v___f_92_, 1, v_inst_87_);
lean_closure_set(v___f_92_, 2, v___f_91_);
v___x_93_ = lp_mathlib_Fintype_piFinset___redArg(v_inst_86_, v_inst_85_, v___f_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq___redArg(lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v___f_99_; lean_object* v___f_100_; lean_object* v___x_101_; 
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_99_, 0, v_inst_98_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_instFintypeOfDecidableEq___aux__1___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_100_, 0, v_inst_97_);
lean_closure_set(v___f_100_, 1, v_inst_96_);
lean_closure_set(v___f_100_, 2, v___f_99_);
v___x_101_ = lp_mathlib_Fintype_piFinset___redArg(v_inst_95_, v_inst_94_, v___f_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instFintypeOfDecidableEq(lean_object* v_n_102_, lean_object* v_m_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_00_u03b1_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_Matrix_instFintypeOfDecidableEq___redArg(v_inst_104_, v_inst_105_, v_inst_106_, v_inst_107_, v_inst_109_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_Matrix_ofLinearEquiv___closed__0(void){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofLinearEquiv(lean_object* v_m_112_, lean_object* v_n_113_, lean_object* v_R_114_, lean_object* v_00_u03b1_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_){
_start:
{
lean_object* v___x_119_; lean_object* v_toFun_120_; lean_object* v_invFun_121_; lean_object* v___x_122_; 
v___x_119_ = lean_obj_once(&lp_mathlib_Matrix_ofLinearEquiv___closed__0, &lp_mathlib_Matrix_ofLinearEquiv___closed__0_once, _init_lp_mathlib_Matrix_ofLinearEquiv___closed__0);
v_toFun_120_ = lean_ctor_get(v___x_119_, 0);
v_invFun_121_ = lean_ctor_get(v___x_119_, 1);
lean_inc(v_invFun_121_);
lean_inc(v_toFun_120_);
v___x_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_122_, 0, v_toFun_120_);
lean_ctor_set(v___x_122_, 1, v_invFun_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_ofLinearEquiv___boxed(lean_object* v_m_123_, lean_object* v_n_124_, lean_object* v_R_125_, lean_object* v_00_u03b1_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_Matrix_ofLinearEquiv(v_m_123_, v_n_124_, v_R_125_, v_00_u03b1_126_, v_inst_127_, v_inst_128_, v_inst_129_);
lean_dec(v_inst_129_);
lean_dec_ref(v_inst_128_);
lean_dec_ref(v_inst_127_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAddMonoidHom___redArg(lean_object* v_inst_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___x_133_; lean_object* v_toZero_134_; lean_object* v___x_135_; 
v___x_133_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_132_);
v_toZero_134_ = lean_ctor_get(v___x_133_, 0);
lean_inc(v_toZero_134_);
lean_dec_ref(v___x_133_);
v___x_135_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal), 7, 4);
lean_closure_set(v___x_135_, 0, lean_box(0));
lean_closure_set(v___x_135_, 1, lean_box(0));
lean_closure_set(v___x_135_, 2, v_inst_131_);
lean_closure_set(v___x_135_, 3, v_toZero_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAddMonoidHom(lean_object* v_n_136_, lean_object* v_00_u03b1_137_, lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_mathlib_Matrix_diagonalAddMonoidHom___redArg(v_inst_138_, v_inst_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap___redArg(lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_143_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_142_);
v___x_144_ = lp_mathlib_Matrix_diagonalAddMonoidHom___redArg(v_inst_141_, v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap___redArg___boxed(lean_object* v_inst_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_Matrix_diagonalLinearMap___redArg(v_inst_145_, v_inst_146_);
lean_dec_ref(v_inst_146_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap(lean_object* v_n_148_, lean_object* v_R_149_, lean_object* v_00_u03b1_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Matrix_diagonalLinearMap___redArg(v_inst_151_, v_inst_153_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalLinearMap___boxed(lean_object* v_n_156_, lean_object* v_R_157_, lean_object* v_00_u03b1_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_Matrix_diagonalLinearMap(v_n_156_, v_R_157_, v_00_u03b1_158_, v_inst_159_, v_inst_160_, v_inst_161_, v_inst_162_);
lean_dec(v_inst_162_);
lean_dec_ref(v_inst_161_);
lean_dec_ref(v_inst_160_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagAddMonoidHom(lean_object* v_n_165_, lean_object* v_00_u03b1_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = ((lean_object*)(lp_mathlib_Matrix_diagAddMonoidHom___closed__0));
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagAddMonoidHom___boxed(lean_object* v_n_169_, lean_object* v_00_u03b1_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Matrix_diagAddMonoidHom(v_n_169_, v_00_u03b1_170_, v_inst_171_);
lean_dec_ref(v_inst_171_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagLinearMap(lean_object* v_n_173_, lean_object* v_R_174_, lean_object* v_00_u03b1_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = ((lean_object*)(lp_mathlib_Matrix_diagAddMonoidHom___closed__0));
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagLinearMap___boxed(lean_object* v_n_180_, lean_object* v_R_181_, lean_object* v_00_u03b1_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Matrix_diagLinearMap(v_n_180_, v_R_181_, v_00_u03b1_182_, v_inst_183_, v_inst_184_, v_inst_185_);
lean_dec(v_inst_185_);
lean_dec_ref(v_inst_184_);
lean_dec_ref(v_inst_183_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalRingHom___redArg(lean_object* v_inst_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_189_; lean_object* v___x_190_; lean_object* v_toZero_191_; lean_object* v___x_192_; 
v_toNonUnitalNonAssocSemiring_189_ = lean_ctor_get(v_inst_187_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_189_);
lean_dec_ref(v_inst_187_);
v___x_190_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_toNonUnitalNonAssocSemiring_189_);
v_toZero_191_ = lean_ctor_get(v___x_190_, 1);
lean_inc(v_toZero_191_);
lean_dec_ref(v___x_190_);
v___x_192_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal), 7, 4);
lean_closure_set(v___x_192_, 0, lean_box(0));
lean_closure_set(v___x_192_, 1, lean_box(0));
lean_closure_set(v___x_192_, 2, v_inst_188_);
lean_closure_set(v___x_192_, 3, v_toZero_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalRingHom(lean_object* v_n_193_, lean_object* v_00_u03b1_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_Matrix_diagonalRingHom___redArg(v_inst_195_, v_inst_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalRingHom___boxed(lean_object* v_n_199_, lean_object* v_00_u03b1_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Matrix_diagonalRingHom(v_n_199_, v_00_u03b1_200_, v_inst_201_, v_inst_202_, v_inst_203_);
lean_dec(v_inst_202_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalar___redArg(lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___f_211_; 
v___x_208_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_206_);
v___x_209_ = lp_mathlib_Matrix_diagonalRingHom___redArg(v___x_208_, v_inst_207_);
v___x_210_ = ((lean_object*)(lp_mathlib_Matrix_scalar___redArg___closed__0));
v___f_211_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_211_, 0, v___x_210_);
lean_closure_set(v___f_211_, 1, v___x_209_);
return v___f_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalar(lean_object* v_00_u03b1_212_, lean_object* v_inst_213_, lean_object* v_n_214_, lean_object* v_inst_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lp_mathlib_Matrix_scalar___redArg(v_inst_213_, v_inst_215_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalar___boxed(lean_object* v_00_u03b1_218_, lean_object* v_inst_219_, lean_object* v_n_220_, lean_object* v_inst_221_, lean_object* v_inst_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Matrix_scalar(v_00_u03b1_218_, v_inst_219_, v_n_220_, v_inst_221_, v_inst_222_);
lean_dec(v_inst_222_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAlgebra___redArg(lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v_toSMul_227_; lean_object* v_algebraMap_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_238_; 
v_toSMul_227_ = lean_ctor_get(v_inst_226_, 0);
v_algebraMap_228_ = lean_ctor_get(v_inst_226_, 1);
v_isSharedCheck_238_ = !lean_is_exclusive(v_inst_226_);
if (v_isSharedCheck_238_ == 0)
{
v___x_230_ = v_inst_226_;
v_isShared_231_ = v_isSharedCheck_238_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_algebraMap_228_);
lean_inc(v_toSMul_227_);
lean_dec(v_inst_226_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_238_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___f_232_; lean_object* v___x_233_; lean_object* v___f_234_; lean_object* v___x_236_; 
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_smul___redArg___lam__0), 5, 1);
lean_closure_set(v___f_232_, 0, v_toSMul_227_);
v___x_233_ = lp_mathlib_Matrix_scalar___redArg(v_inst_225_, v_inst_224_);
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_234_, 0, v_algebraMap_228_);
lean_closure_set(v___f_234_, 1, v___x_233_);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 1, v___f_234_);
lean_ctor_set(v___x_230_, 0, v___f_232_);
v___x_236_ = v___x_230_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v___f_232_);
lean_ctor_set(v_reuseFailAlloc_237_, 1, v___f_234_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAlgebra(lean_object* v_n_239_, lean_object* v_R_240_, lean_object* v_00_u03b1_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_Matrix_instAlgebra___redArg(v_inst_243_, v_inst_245_, v_inst_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_instAlgebra___boxed(lean_object* v_n_248_, lean_object* v_R_249_, lean_object* v_00_u03b1_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_Matrix_instAlgebra(v_n_248_, v_R_249_, v_00_u03b1_250_, v_inst_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_inst_255_);
lean_dec_ref(v_inst_253_);
lean_dec(v_inst_251_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAlgHom___redArg(lean_object* v_inst_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v___x_259_; lean_object* v_toZero_260_; lean_object* v___x_261_; 
v___x_259_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_258_);
v_toZero_260_ = lean_ctor_get(v___x_259_, 1);
lean_inc(v_toZero_260_);
lean_dec_ref(v___x_259_);
v___x_261_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_diagonal), 7, 4);
lean_closure_set(v___x_261_, 0, lean_box(0));
lean_closure_set(v___x_261_, 1, lean_box(0));
lean_closure_set(v___x_261_, 2, v_inst_257_);
lean_closure_set(v___x_261_, 3, v_toZero_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAlgHom(lean_object* v_n_262_, lean_object* v_R_263_, lean_object* v_00_u03b1_264_, lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lp_mathlib_Matrix_diagonalAlgHom___redArg(v_inst_266_, v_inst_268_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_diagonalAlgHom___boxed(lean_object* v_n_271_, lean_object* v_R_272_, lean_object* v_00_u03b1_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_Matrix_diagonalAlgHom(v_n_271_, v_R_272_, v_00_u03b1_273_, v_inst_274_, v_inst_275_, v_inst_276_, v_inst_277_, v_inst_278_);
lean_dec_ref(v_inst_278_);
lean_dec_ref(v_inst_276_);
lean_dec(v_inst_274_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalarAlgHom___redArg(lean_object* v_inst_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_mathlib_Matrix_scalar___redArg(v_inst_281_, v_inst_280_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalarAlgHom(lean_object* v_n_283_, lean_object* v_R_284_, lean_object* v_00_u03b1_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_Matrix_scalar___redArg(v_inst_289_, v_inst_287_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_scalarAlgHom___boxed(lean_object* v_n_292_, lean_object* v_R_293_, lean_object* v_00_u03b1_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_Matrix_scalarAlgHom(v_n_292_, v_R_293_, v_00_u03b1_294_, v_inst_295_, v_inst_296_, v_inst_297_, v_inst_298_, v_inst_299_);
lean_dec_ref(v_inst_299_);
lean_dec_ref(v_inst_297_);
lean_dec(v_inst_295_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom___redArg___lam__0(lean_object* v_i_301_, lean_object* v_j_302_, lean_object* v_M_303_){
_start:
{
lean_object* v___x_304_; 
v___x_304_ = lean_apply_2(v_M_303_, v_i_301_, v_j_302_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom___redArg(lean_object* v_i_305_, lean_object* v_j_306_){
_start:
{
lean_object* v___f_307_; 
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_entryAddHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_307_, 0, v_i_305_);
lean_closure_set(v___f_307_, 1, v_j_306_);
return v___f_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom(lean_object* v_m_308_, lean_object* v_n_309_, lean_object* v_00_u03b1_310_, lean_object* v_inst_311_, lean_object* v_i_312_, lean_object* v_j_313_){
_start:
{
lean_object* v___f_314_; 
v___f_314_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_entryAddHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_314_, 0, v_i_312_);
lean_closure_set(v___f_314_, 1, v_j_313_);
return v___f_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddHom___boxed(lean_object* v_m_315_, lean_object* v_n_316_, lean_object* v_00_u03b1_317_, lean_object* v_inst_318_, lean_object* v_i_319_, lean_object* v_j_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_Matrix_entryAddHom(v_m_315_, v_n_316_, v_00_u03b1_317_, v_inst_318_, v_i_319_, v_j_320_);
lean_dec(v_inst_318_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddMonoidHom___redArg(lean_object* v_i_322_, lean_object* v_j_323_){
_start:
{
lean_object* v___f_324_; 
v___f_324_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_entryAddHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_324_, 0, v_i_322_);
lean_closure_set(v___f_324_, 1, v_j_323_);
return v___f_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddMonoidHom(lean_object* v_m_325_, lean_object* v_n_326_, lean_object* v_00_u03b1_327_, lean_object* v_inst_328_, lean_object* v_i_329_, lean_object* v_j_330_){
_start:
{
lean_object* v___f_331_; 
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_entryAddHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_331_, 0, v_i_329_);
lean_closure_set(v___f_331_, 1, v_j_330_);
return v___f_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryAddMonoidHom___boxed(lean_object* v_m_332_, lean_object* v_n_333_, lean_object* v_00_u03b1_334_, lean_object* v_inst_335_, lean_object* v_i_336_, lean_object* v_j_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib_Matrix_entryAddMonoidHom(v_m_332_, v_n_333_, v_00_u03b1_334_, v_inst_335_, v_i_336_, v_j_337_);
lean_dec_ref(v_inst_335_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryLinearMap___redArg(lean_object* v_i_339_, lean_object* v_j_340_){
_start:
{
lean_object* v___f_341_; 
v___f_341_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_entryAddHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_341_, 0, v_i_339_);
lean_closure_set(v___f_341_, 1, v_j_340_);
return v___f_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryLinearMap(lean_object* v_m_342_, lean_object* v_n_343_, lean_object* v_R_344_, lean_object* v_00_u03b1_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_i_349_, lean_object* v_j_350_){
_start:
{
lean_object* v___f_351_; 
v___f_351_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_entryAddHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_351_, 0, v_i_349_);
lean_closure_set(v___f_351_, 1, v_j_350_);
return v___f_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_entryLinearMap___boxed(lean_object* v_m_352_, lean_object* v_n_353_, lean_object* v_R_354_, lean_object* v_00_u03b1_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_inst_358_, lean_object* v_i_359_, lean_object* v_j_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_Matrix_entryLinearMap(v_m_352_, v_n_353_, v_R_354_, v_00_u03b1_355_, v_inst_356_, v_inst_357_, v_inst_358_, v_i_359_, v_j_360_);
lean_dec(v_inst_358_);
lean_dec_ref(v_inst_357_);
lean_dec_ref(v_inst_356_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__0(lean_object* v_f_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_toFun_364_; lean_object* v___x_365_; 
v_toFun_364_ = lean_ctor_get(v_f_362_, 0);
lean_inc(v_toFun_364_);
lean_dec_ref(v_f_362_);
v___x_365_ = lean_apply_1(v_toFun_364_, v___y_363_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__1(lean_object* v___f_366_, lean_object* v_M_367_, lean_object* v___y_368_, lean_object* v___y_369_){
_start:
{
lean_object* v___x_370_; 
v___x_370_ = lp_mathlib_Matrix_map___redArg(v_M_367_, v___f_366_, v___y_368_, v___y_369_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__2(lean_object* v___x_371_, lean_object* v___y_372_){
_start:
{
lean_object* v_toFun_373_; lean_object* v___x_374_; 
v_toFun_373_ = lean_ctor_get(v___x_371_, 0);
lean_inc(v_toFun_373_);
lean_dec_ref(v___x_371_);
v___x_374_ = lean_apply_1(v_toFun_373_, v___y_372_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg___lam__3(lean_object* v_f_375_, lean_object* v_M_376_, lean_object* v___y_377_, lean_object* v___y_378_){
_start:
{
lean_object* v___x_379_; lean_object* v___f_380_; lean_object* v___x_381_; 
v___x_379_ = lp_mathlib_Equiv_symm___redArg(v_f_375_);
v___f_380_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__2), 2, 1);
lean_closure_set(v___f_380_, 0, v___x_379_);
v___x_381_ = lp_mathlib_Matrix_map___redArg(v_M_376_, v___f_380_, v___y_377_, v___y_378_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix___redArg(lean_object* v_f_382_){
_start:
{
lean_object* v___f_383_; lean_object* v___f_384_; lean_object* v___f_385_; lean_object* v___x_386_; 
lean_inc_ref(v_f_382_);
v___f_383_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_383_, 0, v_f_382_);
v___f_384_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_384_, 0, v___f_383_);
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__3), 4, 1);
lean_closure_set(v___f_385_, 0, v_f_382_);
v___x_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_386_, 0, v___f_384_);
lean_ctor_set(v___x_386_, 1, v___f_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_mapMatrix(lean_object* v_m_387_, lean_object* v_n_388_, lean_object* v_00_u03b1_389_, lean_object* v_00_u03b2_390_, lean_object* v_f_391_){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lp_mathlib_Equiv_mapMatrix___redArg(v_f_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix___redArg___lam__0(lean_object* v_f_393_, lean_object* v___y_394_){
_start:
{
lean_object* v___x_395_; 
v___x_395_ = lean_apply_1(v_f_393_, v___y_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix___redArg(lean_object* v_f_396_){
_start:
{
lean_object* v___f_397_; lean_object* v___f_398_; 
v___f_397_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_397_, 0, v_f_396_);
v___f_398_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_398_, 0, v___f_397_);
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix(lean_object* v_m_399_, lean_object* v_n_400_, lean_object* v_00_u03b1_401_, lean_object* v_00_u03b2_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_f_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_AddMonoidHom_mapMatrix___redArg(v_f_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mapMatrix___boxed(lean_object* v_m_407_, lean_object* v_n_408_, lean_object* v_00_u03b1_409_, lean_object* v_00_u03b2_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_f_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_AddMonoidHom_mapMatrix(v_m_407_, v_n_408_, v_00_u03b1_409_, v_00_u03b2_410_, v_inst_411_, v_inst_412_, v_f_413_);
lean_dec_ref(v_inst_412_);
lean_dec_ref(v_inst_411_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapMatrix___redArg(lean_object* v_f_415_){
_start:
{
lean_object* v___f_416_; lean_object* v___f_417_; lean_object* v___f_418_; lean_object* v___x_419_; 
lean_inc_ref(v_f_415_);
v___f_416_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_416_, 0, v_f_415_);
v___f_417_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_417_, 0, v___f_416_);
v___f_418_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__3), 4, 1);
lean_closure_set(v___f_418_, 0, v_f_415_);
v___x_419_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_419_, 0, v___f_417_);
lean_ctor_set(v___x_419_, 1, v___f_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapMatrix(lean_object* v_m_420_, lean_object* v_n_421_, lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_f_426_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lp_mathlib_AddEquiv_mapMatrix___redArg(v_f_426_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapMatrix___boxed(lean_object* v_m_428_, lean_object* v_n_429_, lean_object* v_00_u03b1_430_, lean_object* v_00_u03b2_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_f_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_AddEquiv_mapMatrix(v_m_428_, v_n_429_, v_00_u03b1_430_, v_00_u03b2_431_, v_inst_432_, v_inst_433_, v_f_434_);
lean_dec(v_inst_433_);
lean_dec(v_inst_432_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrix___redArg(lean_object* v_f_436_){
_start:
{
lean_object* v___f_437_; lean_object* v___f_438_; 
v___f_437_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_437_, 0, v_f_436_);
v___f_438_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_438_, 0, v___f_437_);
return v___f_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrix(lean_object* v_m_439_, lean_object* v_n_440_, lean_object* v_R_441_, lean_object* v_S_442_, lean_object* v_00_u03b1_443_, lean_object* v_00_u03b2_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_00_u03c3_u1d63_u209b_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_f_452_){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = lp_mathlib_LinearMap_mapMatrix___redArg(v_f_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrix___boxed(lean_object* v_m_454_, lean_object* v_n_455_, lean_object* v_R_456_, lean_object* v_S_457_, lean_object* v_00_u03b1_458_, lean_object* v_00_u03b2_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_00_u03c3_u1d63_u209b_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_f_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_mathlib_LinearMap_mapMatrix(v_m_454_, v_n_455_, v_R_456_, v_S_457_, v_00_u03b1_458_, v_00_u03b2_459_, v_inst_460_, v_inst_461_, v_00_u03c3_u1d63_u209b_462_, v_inst_463_, v_inst_464_, v_inst_465_, v_inst_466_, v_f_467_);
lean_dec(v_inst_466_);
lean_dec(v_inst_465_);
lean_dec_ref(v_inst_464_);
lean_dec_ref(v_inst_463_);
lean_dec(v_00_u03c3_u1d63_u209b_462_);
lean_dec_ref(v_inst_461_);
lean_dec_ref(v_inst_460_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrixLinear___redArg(lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_00_u03c3_u1d63_u209b_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_mapMatrix___boxed), 14, 13);
lean_closure_set(v___x_476_, 0, lean_box(0));
lean_closure_set(v___x_476_, 1, lean_box(0));
lean_closure_set(v___x_476_, 2, lean_box(0));
lean_closure_set(v___x_476_, 3, lean_box(0));
lean_closure_set(v___x_476_, 4, lean_box(0));
lean_closure_set(v___x_476_, 5, lean_box(0));
lean_closure_set(v___x_476_, 6, v_inst_469_);
lean_closure_set(v___x_476_, 7, v_inst_470_);
lean_closure_set(v___x_476_, 8, v_00_u03c3_u1d63_u209b_471_);
lean_closure_set(v___x_476_, 9, v_inst_472_);
lean_closure_set(v___x_476_, 10, v_inst_473_);
lean_closure_set(v___x_476_, 11, v_inst_474_);
lean_closure_set(v___x_476_, 12, v_inst_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrixLinear(lean_object* v_m_477_, lean_object* v_n_478_, lean_object* v_R_479_, lean_object* v_S_480_, lean_object* v_A_481_, lean_object* v_00_u03b1_482_, lean_object* v_00_u03b2_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_00_u03c3_u1d63_u209b_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_){
_start:
{
lean_object* v___x_494_; 
v___x_494_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_mapMatrix___boxed), 14, 13);
lean_closure_set(v___x_494_, 0, lean_box(0));
lean_closure_set(v___x_494_, 1, lean_box(0));
lean_closure_set(v___x_494_, 2, lean_box(0));
lean_closure_set(v___x_494_, 3, lean_box(0));
lean_closure_set(v___x_494_, 4, lean_box(0));
lean_closure_set(v___x_494_, 5, lean_box(0));
lean_closure_set(v___x_494_, 6, v_inst_484_);
lean_closure_set(v___x_494_, 7, v_inst_485_);
lean_closure_set(v___x_494_, 8, v_00_u03c3_u1d63_u209b_486_);
lean_closure_set(v___x_494_, 9, v_inst_487_);
lean_closure_set(v___x_494_, 10, v_inst_488_);
lean_closure_set(v___x_494_, 11, v_inst_489_);
lean_closure_set(v___x_494_, 12, v_inst_490_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_mapMatrixLinear___boxed(lean_object** _args){
lean_object* v_m_495_ = _args[0];
lean_object* v_n_496_ = _args[1];
lean_object* v_R_497_ = _args[2];
lean_object* v_S_498_ = _args[3];
lean_object* v_A_499_ = _args[4];
lean_object* v_00_u03b1_500_ = _args[5];
lean_object* v_00_u03b2_501_ = _args[6];
lean_object* v_inst_502_ = _args[7];
lean_object* v_inst_503_ = _args[8];
lean_object* v_00_u03c3_u1d63_u209b_504_ = _args[9];
lean_object* v_inst_505_ = _args[10];
lean_object* v_inst_506_ = _args[11];
lean_object* v_inst_507_ = _args[12];
lean_object* v_inst_508_ = _args[13];
lean_object* v_inst_509_ = _args[14];
lean_object* v_inst_510_ = _args[15];
lean_object* v_inst_511_ = _args[16];
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib_LinearMap_mapMatrixLinear(v_m_495_, v_n_496_, v_R_497_, v_S_498_, v_A_499_, v_00_u03b1_500_, v_00_u03b2_501_, v_inst_502_, v_inst_503_, v_00_u03c3_u1d63_u209b_504_, v_inst_505_, v_inst_506_, v_inst_507_, v_inst_508_, v_inst_509_, v_inst_510_, v_inst_511_);
lean_dec(v_inst_510_);
lean_dec_ref(v_inst_509_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__0(lean_object* v_f_513_, lean_object* v___y_514_){
_start:
{
lean_object* v_toLinearMap_515_; lean_object* v___x_516_; 
v_toLinearMap_515_ = lean_ctor_get(v_f_513_, 0);
lean_inc(v_toLinearMap_515_);
lean_dec_ref(v_f_513_);
v___x_516_ = lean_apply_1(v_toLinearMap_515_, v___y_514_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__2(lean_object* v___x_517_, lean_object* v___y_518_){
_start:
{
lean_object* v_toLinearMap_519_; lean_object* v___x_520_; 
v_toLinearMap_519_ = lean_ctor_get(v___x_517_, 0);
lean_inc(v_toLinearMap_519_);
lean_dec_ref(v___x_517_);
v___x_520_ = lean_apply_1(v_toLinearMap_519_, v___y_518_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__1(lean_object* v_f_521_, lean_object* v_M_522_, lean_object* v___y_523_, lean_object* v___y_524_){
_start:
{
lean_object* v___x_525_; lean_object* v___f_526_; lean_object* v___x_527_; 
v___x_525_ = lp_mathlib_LinearEquiv_symm___redArg(v_f_521_);
v___f_526_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__2), 2, 1);
lean_closure_set(v___f_526_, 0, v___x_525_);
v___x_527_ = lp_mathlib_Matrix_map___redArg(v_M_522_, v___f_526_, v___y_523_, v___y_524_);
return v___x_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___redArg(lean_object* v_f_528_){
_start:
{
lean_object* v___f_529_; lean_object* v___f_530_; lean_object* v___f_531_; lean_object* v___x_532_; 
lean_inc_ref(v_f_528_);
v___f_529_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_529_, 0, v_f_528_);
v___f_530_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_530_, 0, v___f_529_);
v___f_531_ = lean_alloc_closure((void*)(lp_mathlib_LinearEquiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_531_, 0, v_f_528_);
v___x_532_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_532_, 0, v___f_530_);
lean_ctor_set(v___x_532_, 1, v___f_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix(lean_object* v_m_533_, lean_object* v_n_534_, lean_object* v_R_535_, lean_object* v_S_536_, lean_object* v_00_u03b1_537_, lean_object* v_00_u03b2_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_00_u03c3_u1d63_u209b_545_, lean_object* v_00_u03c3_u209b_u1d63_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_f_549_){
_start:
{
lean_object* v___x_550_; 
v___x_550_ = lp_mathlib_LinearEquiv_mapMatrix___redArg(v_f_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearEquiv_mapMatrix___boxed(lean_object** _args){
lean_object* v_m_551_ = _args[0];
lean_object* v_n_552_ = _args[1];
lean_object* v_R_553_ = _args[2];
lean_object* v_S_554_ = _args[3];
lean_object* v_00_u03b1_555_ = _args[4];
lean_object* v_00_u03b2_556_ = _args[5];
lean_object* v_inst_557_ = _args[6];
lean_object* v_inst_558_ = _args[7];
lean_object* v_inst_559_ = _args[8];
lean_object* v_inst_560_ = _args[9];
lean_object* v_inst_561_ = _args[10];
lean_object* v_inst_562_ = _args[11];
lean_object* v_00_u03c3_u1d63_u209b_563_ = _args[12];
lean_object* v_00_u03c3_u209b_u1d63_564_ = _args[13];
lean_object* v_inst_565_ = _args[14];
lean_object* v_inst_566_ = _args[15];
lean_object* v_f_567_ = _args[16];
_start:
{
lean_object* v_res_568_; 
v_res_568_ = lp_mathlib_LinearEquiv_mapMatrix(v_m_551_, v_n_552_, v_R_553_, v_S_554_, v_00_u03b1_555_, v_00_u03b2_556_, v_inst_557_, v_inst_558_, v_inst_559_, v_inst_560_, v_inst_561_, v_inst_562_, v_00_u03c3_u1d63_u209b_563_, v_00_u03c3_u209b_u1d63_564_, v_inst_565_, v_inst_566_, v_f_567_);
lean_dec(v_00_u03c3_u209b_u1d63_564_);
lean_dec(v_00_u03c3_u1d63_u209b_563_);
lean_dec(v_inst_562_);
lean_dec(v_inst_561_);
lean_dec_ref(v_inst_560_);
lean_dec_ref(v_inst_559_);
lean_dec_ref(v_inst_558_);
lean_dec_ref(v_inst_557_);
return v_res_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mapMatrix___redArg(lean_object* v_f_569_){
_start:
{
lean_object* v___f_570_; lean_object* v___f_571_; 
v___f_570_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_570_, 0, v_f_569_);
v___f_571_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_571_, 0, v___f_570_);
return v___f_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mapMatrix(lean_object* v_m_572_, lean_object* v_00_u03b1_573_, lean_object* v_00_u03b2_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_f_579_){
_start:
{
lean_object* v___x_580_; 
v___x_580_ = lp_mathlib_RingHom_mapMatrix___redArg(v_f_579_);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_mapMatrix___boxed(lean_object* v_m_581_, lean_object* v_00_u03b1_582_, lean_object* v_00_u03b2_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_inst_587_, lean_object* v_f_588_){
_start:
{
lean_object* v_res_589_; 
v_res_589_ = lp_mathlib_RingHom_mapMatrix(v_m_581_, v_00_u03b1_582_, v_00_u03b2_583_, v_inst_584_, v_inst_585_, v_inst_586_, v_inst_587_, v_f_588_);
lean_dec_ref(v_inst_587_);
lean_dec_ref(v_inst_586_);
lean_dec_ref(v_inst_585_);
lean_dec(v_inst_584_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mapMatrix___redArg(lean_object* v_f_590_){
_start:
{
lean_object* v___f_591_; lean_object* v___f_592_; lean_object* v___f_593_; lean_object* v___x_594_; 
lean_inc_ref(v_f_590_);
v___f_591_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_591_, 0, v_f_590_);
v___f_592_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_592_, 0, v___f_591_);
v___f_593_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__3), 4, 1);
lean_closure_set(v___f_593_, 0, v_f_590_);
v___x_594_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_594_, 0, v___f_592_);
lean_ctor_set(v___x_594_, 1, v___f_593_);
return v___x_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mapMatrix(lean_object* v_m_595_, lean_object* v_00_u03b1_596_, lean_object* v_00_u03b2_597_, lean_object* v_inst_598_, lean_object* v_inst_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_f_602_){
_start:
{
lean_object* v___x_603_; 
v___x_603_ = lp_mathlib_RingEquiv_mapMatrix___redArg(v_f_602_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mapMatrix___boxed(lean_object* v_m_604_, lean_object* v_00_u03b1_605_, lean_object* v_00_u03b2_606_, lean_object* v_inst_607_, lean_object* v_inst_608_, lean_object* v_inst_609_, lean_object* v_inst_610_, lean_object* v_f_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib_RingEquiv_mapMatrix(v_m_604_, v_00_u03b1_605_, v_00_u03b2_606_, v_inst_607_, v_inst_608_, v_inst_609_, v_inst_610_, v_f_611_);
lean_dec_ref(v_inst_610_);
lean_dec_ref(v_inst_609_);
lean_dec_ref(v_inst_608_);
lean_dec(v_inst_607_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix___lam__0(lean_object* v_M_614_, lean_object* v___y_615_, lean_object* v___y_616_){
_start:
{
lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; 
v___x_617_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_transpose), 6, 4);
lean_closure_set(v___x_617_, 0, lean_box(0));
lean_closure_set(v___x_617_, 1, lean_box(0));
lean_closure_set(v___x_617_, 2, lean_box(0));
lean_closure_set(v___x_617_, 3, v_M_614_);
v___x_618_ = ((lean_object*)(lp_mathlib_RingEquiv_mopMatrix___lam__0___closed__0));
v___x_619_ = lp_mathlib_Matrix_map___redArg(v___x_617_, v___x_618_, v___y_615_, v___y_616_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix___lam__1(lean_object* v_M_621_, lean_object* v___y_622_, lean_object* v___y_623_){
_start:
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_624_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_transpose), 6, 4);
lean_closure_set(v___x_624_, 0, lean_box(0));
lean_closure_set(v___x_624_, 1, lean_box(0));
lean_closure_set(v___x_624_, 2, lean_box(0));
lean_closure_set(v___x_624_, 3, v_M_621_);
v___x_625_ = ((lean_object*)(lp_mathlib_RingEquiv_mopMatrix___lam__1___closed__0));
v___x_626_ = lp_mathlib_Matrix_map___redArg(v___x_624_, v___x_625_, v___y_622_, v___y_623_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix(lean_object* v_m_632_, lean_object* v_inst_633_, lean_object* v_00_u03b1_634_, lean_object* v_inst_635_, lean_object* v_inst_636_){
_start:
{
lean_object* v___x_637_; 
v___x_637_ = ((lean_object*)(lp_mathlib_RingEquiv_mopMatrix___closed__2));
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_mopMatrix___boxed(lean_object* v_m_638_, lean_object* v_inst_639_, lean_object* v_00_u03b1_640_, lean_object* v_inst_641_, lean_object* v_inst_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_mathlib_RingEquiv_mopMatrix(v_m_638_, v_inst_639_, v_00_u03b1_640_, v_inst_641_, v_inst_642_);
lean_dec_ref(v_inst_642_);
lean_dec(v_inst_641_);
lean_dec(v_inst_639_);
return v_res_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapMatrix___redArg(lean_object* v_f_644_){
_start:
{
lean_object* v___f_645_; lean_object* v___f_646_; 
v___f_645_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_645_, 0, v_f_644_);
v___f_646_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_646_, 0, v___f_645_);
return v___f_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapMatrix(lean_object* v_m_647_, lean_object* v_R_648_, lean_object* v_00_u03b1_649_, lean_object* v_00_u03b2_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_inst_657_, lean_object* v_f_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_mathlib_AlgHom_mapMatrix___redArg(v_f_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_mapMatrix___boxed(lean_object* v_m_660_, lean_object* v_R_661_, lean_object* v_00_u03b1_662_, lean_object* v_00_u03b2_663_, lean_object* v_inst_664_, lean_object* v_inst_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_inst_668_, lean_object* v_inst_669_, lean_object* v_inst_670_, lean_object* v_f_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_AlgHom_mapMatrix(v_m_660_, v_R_661_, v_00_u03b1_662_, v_00_u03b2_663_, v_inst_664_, v_inst_665_, v_inst_666_, v_inst_667_, v_inst_668_, v_inst_669_, v_inst_670_, v_f_671_);
lean_dec_ref(v_inst_670_);
lean_dec_ref(v_inst_669_);
lean_dec_ref(v_inst_668_);
lean_dec_ref(v_inst_667_);
lean_dec_ref(v_inst_666_);
lean_dec_ref(v_inst_665_);
lean_dec(v_inst_664_);
return v_res_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapMatrix___redArg(lean_object* v_f_673_){
_start:
{
lean_object* v___f_674_; lean_object* v___f_675_; lean_object* v___f_676_; lean_object* v___x_677_; 
lean_inc_ref(v_f_673_);
v___f_674_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__0), 2, 1);
lean_closure_set(v___f_674_, 0, v_f_673_);
v___f_675_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__1), 4, 1);
lean_closure_set(v___f_675_, 0, v___f_674_);
v___f_676_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_mapMatrix___redArg___lam__3), 4, 1);
lean_closure_set(v___f_676_, 0, v_f_673_);
v___x_677_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_677_, 0, v___f_675_);
lean_ctor_set(v___x_677_, 1, v___f_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapMatrix(lean_object* v_m_678_, lean_object* v_R_679_, lean_object* v_00_u03b1_680_, lean_object* v_00_u03b2_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_f_689_){
_start:
{
lean_object* v___x_690_; 
v___x_690_ = lp_mathlib_AlgEquiv_mapMatrix___redArg(v_f_689_);
return v___x_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mapMatrix___boxed(lean_object* v_m_691_, lean_object* v_R_692_, lean_object* v_00_u03b1_693_, lean_object* v_00_u03b2_694_, lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_inst_697_, lean_object* v_inst_698_, lean_object* v_inst_699_, lean_object* v_inst_700_, lean_object* v_inst_701_, lean_object* v_f_702_){
_start:
{
lean_object* v_res_703_; 
v_res_703_ = lp_mathlib_AlgEquiv_mapMatrix(v_m_691_, v_R_692_, v_00_u03b1_693_, v_00_u03b2_694_, v_inst_695_, v_inst_696_, v_inst_697_, v_inst_698_, v_inst_699_, v_inst_700_, v_inst_701_, v_f_702_);
lean_dec_ref(v_inst_701_);
lean_dec_ref(v_inst_700_);
lean_dec_ref(v_inst_699_);
lean_dec_ref(v_inst_698_);
lean_dec_ref(v_inst_697_);
lean_dec_ref(v_inst_696_);
lean_dec(v_inst_695_);
return v_res_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix___redArg(lean_object* v_inst_704_, lean_object* v_inst_705_){
_start:
{
lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v_toMul_708_; lean_object* v_toAddCommMonoid_709_; lean_object* v___x_710_; 
v___x_706_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_705_);
lean_inc_ref(v___x_706_);
v___x_707_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v___x_706_);
v_toMul_708_ = lean_ctor_get(v___x_707_, 0);
lean_inc(v_toMul_708_);
lean_dec_ref(v___x_707_);
v_toAddCommMonoid_709_ = lean_ctor_get(v___x_706_, 0);
lean_inc_ref(v_toAddCommMonoid_709_);
lean_dec_ref(v___x_706_);
v___x_710_ = lp_mathlib_RingEquiv_mopMatrix(lean_box(0), v_inst_704_, lean_box(0), v_toMul_708_, v_toAddCommMonoid_709_);
lean_dec_ref(v_toAddCommMonoid_709_);
lean_dec(v_toMul_708_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix___redArg___boxed(lean_object* v_inst_711_, lean_object* v_inst_712_){
_start:
{
lean_object* v_res_713_; 
v_res_713_ = lp_mathlib_AlgEquiv_mopMatrix___redArg(v_inst_711_, v_inst_712_);
lean_dec_ref(v_inst_712_);
lean_dec(v_inst_711_);
return v_res_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix(lean_object* v_m_714_, lean_object* v_R_715_, lean_object* v_00_u03b1_716_, lean_object* v_inst_717_, lean_object* v_inst_718_, lean_object* v_inst_719_, lean_object* v_inst_720_, lean_object* v_inst_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lp_mathlib_AlgEquiv_mopMatrix___redArg(v_inst_717_, v_inst_720_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_mopMatrix___boxed(lean_object* v_m_723_, lean_object* v_R_724_, lean_object* v_00_u03b1_725_, lean_object* v_inst_726_, lean_object* v_inst_727_, lean_object* v_inst_728_, lean_object* v_inst_729_, lean_object* v_inst_730_){
_start:
{
lean_object* v_res_731_; 
v_res_731_ = lp_mathlib_AlgEquiv_mopMatrix(v_m_723_, v_R_724_, v_00_u03b1_725_, v_inst_726_, v_inst_727_, v_inst_728_, v_inst_729_, v_inst_730_);
lean_dec_ref(v_inst_730_);
lean_dec_ref(v_inst_729_);
lean_dec_ref(v_inst_728_);
lean_dec_ref(v_inst_727_);
lean_dec(v_inst_726_);
return v_res_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_matrix(lean_object* v_m_732_, lean_object* v_n_733_, lean_object* v_A_734_, lean_object* v_inst_735_, lean_object* v_S_736_){
_start:
{
lean_object* v___x_737_; 
v___x_737_ = lean_box(0);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_matrix___boxed(lean_object* v_m_738_, lean_object* v_n_739_, lean_object* v_A_740_, lean_object* v_inst_741_, lean_object* v_S_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_mathlib_AddSubmonoid_matrix(v_m_738_, v_n_739_, v_A_740_, v_inst_741_, v_S_742_);
lean_dec_ref(v_inst_741_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_matrix(lean_object* v_m_744_, lean_object* v_n_745_, lean_object* v_A_746_, lean_object* v_inst_747_, lean_object* v_S_748_){
_start:
{
lean_object* v___x_749_; 
v___x_749_ = lean_box(0);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_matrix___boxed(lean_object* v_m_750_, lean_object* v_n_751_, lean_object* v_A_752_, lean_object* v_inst_753_, lean_object* v_S_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_AddSubgroup_matrix(v_m_750_, v_n_751_, v_A_752_, v_inst_753_, v_S_754_);
lean_dec_ref(v_inst_753_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_matrix(lean_object* v_n_756_, lean_object* v_R_757_, lean_object* v_inst_758_, lean_object* v_inst_759_, lean_object* v_inst_760_, lean_object* v_S_761_){
_start:
{
lean_object* v___x_762_; 
v___x_762_ = lean_box(0);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_matrix___boxed(lean_object* v_n_763_, lean_object* v_R_764_, lean_object* v_inst_765_, lean_object* v_inst_766_, lean_object* v_inst_767_, lean_object* v_S_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_mathlib_Subsemiring_matrix(v_n_763_, v_R_764_, v_inst_765_, v_inst_766_, v_inst_767_, v_S_768_);
lean_dec_ref(v_inst_767_);
lean_dec(v_inst_766_);
lean_dec_ref(v_inst_765_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_matrix(lean_object* v_n_770_, lean_object* v_R_771_, lean_object* v_inst_772_, lean_object* v_inst_773_, lean_object* v_inst_774_, lean_object* v_S_775_){
_start:
{
lean_object* v___x_776_; 
v___x_776_ = lean_box(0);
return v___x_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subring_matrix___boxed(lean_object* v_n_777_, lean_object* v_R_778_, lean_object* v_inst_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_S_782_){
_start:
{
lean_object* v_res_783_; 
v_res_783_ = lp_mathlib_Subring_matrix(v_n_777_, v_R_778_, v_inst_779_, v_inst_780_, v_inst_781_, v_S_782_);
lean_dec_ref(v_inst_781_);
lean_dec(v_inst_780_);
lean_dec_ref(v_inst_779_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_matrix(lean_object* v_m_784_, lean_object* v_n_785_, lean_object* v_R_786_, lean_object* v_M_787_, lean_object* v_inst_788_, lean_object* v_inst_789_, lean_object* v_inst_790_, lean_object* v_S_791_){
_start:
{
lean_object* v___x_792_; 
v___x_792_ = lean_box(0);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_matrix___boxed(lean_object* v_m_793_, lean_object* v_n_794_, lean_object* v_R_795_, lean_object* v_M_796_, lean_object* v_inst_797_, lean_object* v_inst_798_, lean_object* v_inst_799_, lean_object* v_S_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_mathlib_Submodule_matrix(v_m_793_, v_n_794_, v_R_795_, v_M_796_, v_inst_797_, v_inst_798_, v_inst_799_, v_S_800_);
lean_dec(v_inst_799_);
lean_dec_ref(v_inst_798_);
lean_dec_ref(v_inst_797_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__0(lean_object* v_i_802_, lean_object* v_x_803_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lean_apply_1(v_x_803_, v_i_802_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__1(lean_object* v_f_805_, lean_object* v_i_806_, lean_object* v___y_807_, lean_object* v___y_808_){
_start:
{
lean_object* v___f_809_; lean_object* v___x_810_; 
v___f_809_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_piEquiv___lam__0), 2, 1);
lean_closure_set(v___f_809_, 0, v_i_806_);
v___x_810_ = lp_mathlib_Matrix_map___redArg(v_f_805_, v___f_809_, v___y_807_, v___y_808_);
return v___x_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__2(lean_object* v_f_811_, lean_object* v_j_812_, lean_object* v_k_813_, lean_object* v_i_814_){
_start:
{
lean_object* v___x_815_; 
v___x_815_ = lean_apply_3(v_f_811_, v_i_814_, v_j_812_, v_k_813_);
return v___x_815_;
}
}
static lean_object* _init_lp_mathlib_Matrix_piEquiv___lam__3___closed__0(void){
_start:
{
lean_object* v___x_816_; 
v___x_816_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv___lam__3(lean_object* v_f_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_){
_start:
{
lean_object* v___x_821_; lean_object* v_toFun_822_; lean_object* v___f_823_; lean_object* v___x_824_; 
v___x_821_ = lean_obj_once(&lp_mathlib_Matrix_piEquiv___lam__3___closed__0, &lp_mathlib_Matrix_piEquiv___lam__3___closed__0_once, _init_lp_mathlib_Matrix_piEquiv___lam__3___closed__0);
v_toFun_822_ = lean_ctor_get(v___x_821_, 0);
v___f_823_ = lean_alloc_closure((void*)(lp_mathlib_Matrix_piEquiv___lam__2), 4, 1);
lean_closure_set(v___f_823_, 0, v_f_817_);
lean_inc(v_toFun_822_);
v___x_824_ = lean_apply_4(v_toFun_822_, v___f_823_, v___y_818_, v___y_819_, v___y_820_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piEquiv(lean_object* v_m_830_, lean_object* v_n_831_, lean_object* v_00_u03b9_832_, lean_object* v_00_u03b2_833_){
_start:
{
lean_object* v___x_834_; 
v___x_834_ = ((lean_object*)(lp_mathlib_Matrix_piEquiv___closed__2));
return v___x_834_;
}
}
static lean_object* _init_lp_mathlib_Matrix_piAddEquiv___closed__0(void){
_start:
{
lean_object* v___x_835_; 
v___x_835_ = lp_mathlib_Matrix_piEquiv(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAddEquiv(lean_object* v_m_836_, lean_object* v_n_837_, lean_object* v_00_u03b9_838_, lean_object* v_00_u03b2_839_, lean_object* v_inst_840_){
_start:
{
lean_object* v___x_841_; 
v___x_841_ = lean_obj_once(&lp_mathlib_Matrix_piAddEquiv___closed__0, &lp_mathlib_Matrix_piAddEquiv___closed__0_once, _init_lp_mathlib_Matrix_piAddEquiv___closed__0);
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAddEquiv___boxed(lean_object* v_m_842_, lean_object* v_n_843_, lean_object* v_00_u03b9_844_, lean_object* v_00_u03b2_845_, lean_object* v_inst_846_){
_start:
{
lean_object* v_res_847_; 
v_res_847_ = lp_mathlib_Matrix_piAddEquiv(v_m_842_, v_n_843_, v_00_u03b9_844_, v_00_u03b2_845_, v_inst_846_);
lean_dec(v_inst_846_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piLinearEquiv(lean_object* v_m_848_, lean_object* v_n_849_, lean_object* v_00_u03b9_850_, lean_object* v_00_u03b2_851_, lean_object* v_R_852_, lean_object* v_inst_853_, lean_object* v_inst_854_, lean_object* v_inst_855_){
_start:
{
lean_object* v___x_856_; lean_object* v_toFun_857_; lean_object* v_invFun_858_; lean_object* v___x_859_; 
v___x_856_ = lean_obj_once(&lp_mathlib_Matrix_piAddEquiv___closed__0, &lp_mathlib_Matrix_piAddEquiv___closed__0_once, _init_lp_mathlib_Matrix_piAddEquiv___closed__0);
v_toFun_857_ = lean_ctor_get(v___x_856_, 0);
v_invFun_858_ = lean_ctor_get(v___x_856_, 1);
lean_inc(v_invFun_858_);
lean_inc(v_toFun_857_);
v___x_859_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_859_, 0, v_toFun_857_);
lean_ctor_set(v___x_859_, 1, v_invFun_858_);
return v___x_859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piLinearEquiv___boxed(lean_object* v_m_860_, lean_object* v_n_861_, lean_object* v_00_u03b9_862_, lean_object* v_00_u03b2_863_, lean_object* v_R_864_, lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_inst_867_){
_start:
{
lean_object* v_res_868_; 
v_res_868_ = lp_mathlib_Matrix_piLinearEquiv(v_m_860_, v_n_861_, v_00_u03b9_862_, v_00_u03b2_863_, v_R_864_, v_inst_865_, v_inst_866_, v_inst_867_);
lean_dec(v_inst_867_);
lean_dec_ref(v_inst_866_);
lean_dec_ref(v_inst_865_);
return v_res_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piRingEquiv(lean_object* v_n_869_, lean_object* v_00_u03b9_870_, lean_object* v_00_u03b2_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_){
_start:
{
lean_object* v___x_875_; 
v___x_875_ = lean_obj_once(&lp_mathlib_Matrix_piAddEquiv___closed__0, &lp_mathlib_Matrix_piAddEquiv___closed__0_once, _init_lp_mathlib_Matrix_piAddEquiv___closed__0);
return v___x_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piRingEquiv___boxed(lean_object* v_n_876_, lean_object* v_00_u03b9_877_, lean_object* v_00_u03b2_878_, lean_object* v_inst_879_, lean_object* v_inst_880_, lean_object* v_inst_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_mathlib_Matrix_piRingEquiv(v_n_876_, v_00_u03b9_877_, v_00_u03b2_878_, v_inst_879_, v_inst_880_, v_inst_881_);
lean_dec(v_inst_881_);
lean_dec(v_inst_880_);
lean_dec_ref(v_inst_879_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAlgEquiv(lean_object* v_n_883_, lean_object* v_00_u03b9_884_, lean_object* v_00_u03b2_885_, lean_object* v_R_886_, lean_object* v_inst_887_, lean_object* v_inst_888_, lean_object* v_inst_889_, lean_object* v_inst_890_, lean_object* v_inst_891_){
_start:
{
lean_object* v___x_892_; 
v___x_892_ = lean_obj_once(&lp_mathlib_Matrix_piAddEquiv___closed__0, &lp_mathlib_Matrix_piAddEquiv___closed__0_once, _init_lp_mathlib_Matrix_piAddEquiv___closed__0);
return v___x_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_piAlgEquiv___boxed(lean_object* v_n_893_, lean_object* v_00_u03b9_894_, lean_object* v_00_u03b2_895_, lean_object* v_R_896_, lean_object* v_inst_897_, lean_object* v_inst_898_, lean_object* v_inst_899_, lean_object* v_inst_900_, lean_object* v_inst_901_){
_start:
{
lean_object* v_res_902_; 
v_res_902_ = lp_mathlib_Matrix_piAlgEquiv(v_n_893_, v_00_u03b9_894_, v_00_u03b2_895_, v_R_896_, v_inst_897_, v_inst_898_, v_inst_899_, v_inst_900_, v_inst_901_);
lean_dec_ref(v_inst_901_);
lean_dec(v_inst_900_);
lean_dec_ref(v_inst_899_);
lean_dec_ref(v_inst_898_);
lean_dec_ref(v_inst_897_);
return v_res_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAddEquiv(lean_object* v_m_906_, lean_object* v_n_907_, lean_object* v_00_u03b1_908_, lean_object* v_inst_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = ((lean_object*)(lp_mathlib_Matrix_transposeAddEquiv___closed__1));
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAddEquiv___boxed(lean_object* v_m_911_, lean_object* v_n_912_, lean_object* v_00_u03b1_913_, lean_object* v_inst_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_Matrix_transposeAddEquiv(v_m_911_, v_n_912_, v_00_u03b1_913_, v_inst_914_);
lean_dec(v_inst_914_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv___redArg(lean_object* v_inst_916_){
_start:
{
lean_object* v_toAdd_917_; lean_object* v___x_918_; lean_object* v_toFun_919_; lean_object* v_invFun_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_927_; 
v_toAdd_917_ = lean_ctor_get(v_inst_916_, 1);
v___x_918_ = lp_mathlib_Matrix_transposeAddEquiv(lean_box(0), lean_box(0), lean_box(0), v_toAdd_917_);
v_toFun_919_ = lean_ctor_get(v___x_918_, 0);
v_invFun_920_ = lean_ctor_get(v___x_918_, 1);
v_isSharedCheck_927_ = !lean_is_exclusive(v___x_918_);
if (v_isSharedCheck_927_ == 0)
{
v___x_922_ = v___x_918_;
v_isShared_923_ = v_isSharedCheck_927_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_invFun_920_);
lean_inc(v_toFun_919_);
lean_dec(v___x_918_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_927_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_925_; 
if (v_isShared_923_ == 0)
{
v___x_925_ = v___x_922_;
goto v_reusejp_924_;
}
else
{
lean_object* v_reuseFailAlloc_926_; 
v_reuseFailAlloc_926_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_926_, 0, v_toFun_919_);
lean_ctor_set(v_reuseFailAlloc_926_, 1, v_invFun_920_);
v___x_925_ = v_reuseFailAlloc_926_;
goto v_reusejp_924_;
}
v_reusejp_924_:
{
return v___x_925_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv___redArg___boxed(lean_object* v_inst_928_){
_start:
{
lean_object* v_res_929_; 
v_res_929_ = lp_mathlib_Matrix_transposeLinearEquiv___redArg(v_inst_928_);
lean_dec_ref(v_inst_928_);
return v_res_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv(lean_object* v_m_930_, lean_object* v_n_931_, lean_object* v_R_932_, lean_object* v_00_u03b1_933_, lean_object* v_inst_934_, lean_object* v_inst_935_, lean_object* v_inst_936_){
_start:
{
lean_object* v___x_937_; 
v___x_937_ = lp_mathlib_Matrix_transposeLinearEquiv___redArg(v_inst_935_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeLinearEquiv___boxed(lean_object* v_m_938_, lean_object* v_n_939_, lean_object* v_R_940_, lean_object* v_00_u03b1_941_, lean_object* v_inst_942_, lean_object* v_inst_943_, lean_object* v_inst_944_){
_start:
{
lean_object* v_res_945_; 
v_res_945_ = lp_mathlib_Matrix_transposeLinearEquiv(v_m_938_, v_n_939_, v_R_940_, v_00_u03b1_941_, v_inst_942_, v_inst_943_, v_inst_944_);
lean_dec(v_inst_944_);
lean_dec_ref(v_inst_943_);
lean_dec_ref(v_inst_942_);
return v_res_945_;
}
}
static lean_object* _init_lp_mathlib_Matrix_transposeRingEquiv___redArg___closed__0(void){
_start:
{
lean_object* v___x_946_; 
v___x_946_ = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return v___x_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv___redArg(lean_object* v_inst_947_){
_start:
{
lean_object* v_toAdd_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; 
v_toAdd_948_ = lean_ctor_get(v_inst_947_, 1);
v___x_949_ = lp_mathlib_Matrix_transposeAddEquiv(lean_box(0), lean_box(0), lean_box(0), v_toAdd_948_);
v___x_950_ = lean_obj_once(&lp_mathlib_Matrix_transposeRingEquiv___redArg___closed__0, &lp_mathlib_Matrix_transposeRingEquiv___redArg___closed__0_once, _init_lp_mathlib_Matrix_transposeRingEquiv___redArg___closed__0);
v___x_951_ = lp_mathlib_Equiv_trans___redArg(v___x_949_, v___x_950_);
return v___x_951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv___redArg___boxed(lean_object* v_inst_952_){
_start:
{
lean_object* v_res_953_; 
v_res_953_ = lp_mathlib_Matrix_transposeRingEquiv___redArg(v_inst_952_);
lean_dec_ref(v_inst_952_);
return v_res_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv(lean_object* v_m_954_, lean_object* v_00_u03b1_955_, lean_object* v_inst_956_, lean_object* v_inst_957_, lean_object* v_inst_958_){
_start:
{
lean_object* v___x_959_; 
v___x_959_ = lp_mathlib_Matrix_transposeRingEquiv___redArg(v_inst_956_);
return v___x_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeRingEquiv___boxed(lean_object* v_m_960_, lean_object* v_00_u03b1_961_, lean_object* v_inst_962_, lean_object* v_inst_963_, lean_object* v_inst_964_){
_start:
{
lean_object* v_res_965_; 
v_res_965_ = lp_mathlib_Matrix_transposeRingEquiv(v_m_960_, v_00_u03b1_961_, v_inst_962_, v_inst_963_, v_inst_964_);
lean_dec(v_inst_964_);
lean_dec(v_inst_963_);
lean_dec_ref(v_inst_962_);
return v_res_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv___redArg(lean_object* v_inst_966_){
_start:
{
lean_object* v_toAddCommMonoid_967_; lean_object* v___x_968_; 
v_toAddCommMonoid_967_ = lean_ctor_get(v_inst_966_, 0);
v___x_968_ = lp_mathlib_Matrix_transposeRingEquiv___redArg(v_toAddCommMonoid_967_);
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv___redArg___boxed(lean_object* v_inst_969_){
_start:
{
lean_object* v_res_970_; 
v_res_970_ = lp_mathlib_Matrix_transposeAlgEquiv___redArg(v_inst_969_);
lean_dec_ref(v_inst_969_);
return v_res_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv(lean_object* v_m_971_, lean_object* v_R_972_, lean_object* v_00_u03b1_973_, lean_object* v_inst_974_, lean_object* v_inst_975_, lean_object* v_inst_976_, lean_object* v_inst_977_, lean_object* v_inst_978_){
_start:
{
lean_object* v___x_979_; 
v___x_979_ = lp_mathlib_Matrix_transposeAlgEquiv___redArg(v_inst_975_);
return v___x_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Matrix_transposeAlgEquiv___boxed(lean_object* v_m_980_, lean_object* v_R_981_, lean_object* v_00_u03b1_982_, lean_object* v_inst_983_, lean_object* v_inst_984_, lean_object* v_inst_985_, lean_object* v_inst_986_, lean_object* v_inst_987_){
_start:
{
lean_object* v_res_988_; 
v_res_988_ = lp_mathlib_Matrix_transposeAlgEquiv(v_m_980_, v_R_981_, v_00_u03b1_982_, v_inst_983_, v_inst_984_, v_inst_985_, v_inst_986_, v_inst_987_);
lean_dec_ref(v_inst_987_);
lean_dec_ref(v_inst_986_);
lean_dec(v_inst_985_);
lean_dec_ref(v_inst_984_);
lean_dec_ref(v_inst_983_);
return v_res_988_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_RingEquiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Matrix_Mul(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_DedekindFinite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Matrix_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_RingEquiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Matrix_Mul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_DedekindFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Matrix_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_RingEquiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Matrix_Mul(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_DedekindFinite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Matrix_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_RingEquiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Matrix_Mul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_DedekindFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Matrix_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Matrix_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Matrix_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
