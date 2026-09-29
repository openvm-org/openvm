// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Oracle.SimplexAlgorithm.Gauss
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.Linarith.Oracle.SimplexAlgorithm.Datatypes public import Mathlib.Tactic.Linarith.Oracle.SimplexAlgorithm.Datatypes
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
lean_object* l_Rat_neg(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Array_findIdx_x3f_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_panic___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
extern lean_object* l_instInhabitedRat;
lean_object* l_outOfBounds___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_While_0__repeatM_erased___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_filterMapTR_go___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Linarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SimplexAlgorithm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Gauss"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "getTableauImp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(90, 19, 252, 206, 85, 151, 142, 50)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(78, 172, 83, 255, 173, 212, 79, 145)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value_aux_4),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(3, 130, 163, 123, 246, 233, 96, 154)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__3___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__2___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_unsigned_to_nat(0u);
v___x_2_ = l_Rat_instNatCast___lam__0(v___x_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0(lean_object* v___x_3_, lean_object* v___x_4_, lean_object* v___x_5_, lean_object* v_n_6_, lean_object* v_col_7_, lean_object* v_m_8_, lean_object* v_inst_9_, lean_object* v_a_10_, lean_object* v_x_11_, lean_object* v___y_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_){
_start:
{
lean_object* v___y_18_; uint8_t v___x_32_; 
v___x_32_ = lean_nat_dec_lt(v_a_10_, v_n_6_);
if (v___x_32_ == 0)
{
lean_dec_ref(v_inst_9_);
lean_dec(v_m_8_);
lean_dec(v_col_7_);
lean_dec(v_n_6_);
goto v___jp_30_;
}
else
{
uint8_t v___x_33_; 
v___x_33_ = lean_nat_dec_lt(v_col_7_, v_m_8_);
if (v___x_33_ == 0)
{
lean_dec_ref(v_inst_9_);
lean_dec(v_m_8_);
lean_dec(v_col_7_);
lean_dec(v_n_6_);
goto v___jp_30_;
}
else
{
lean_object* v_getElem_34_; lean_object* v___x_35_; 
v_getElem_34_ = lean_ctor_get(v_inst_9_, 0);
lean_inc_ref(v_getElem_34_);
lean_dec_ref(v_inst_9_);
lean_inc(v_a_10_);
lean_inc(v___y_13_);
v___x_35_ = lean_apply_5(v_getElem_34_, v_n_6_, v_m_8_, v___y_13_, v_a_10_, v_col_7_);
v___y_18_ = v___x_35_;
goto v___jp_17_;
}
}
v___jp_17_:
{
lean_object* v___x_19_; uint8_t v___x_20_; 
v___x_19_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___closed__0);
v___x_20_ = l_instDecidableEqRat_decEq(v___y_18_, v___x_19_);
lean_dec_ref(v___y_18_);
if (v___x_20_ == 0)
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
lean_dec_ref(v___x_4_);
v___x_21_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_21_, 0, v_a_10_);
v___x_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
v___x_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_23_, 0, v___x_22_);
lean_ctor_set(v___x_23_, 1, v___x_3_);
v___x_24_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_24_, 0, v___x_23_);
v___x_25_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_25_, 0, v___x_24_);
lean_ctor_set(v___x_25_, 1, v___y_13_);
v___x_26_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_26_, 0, v___x_25_);
return v___x_26_;
}
else
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
lean_dec(v_a_10_);
v___x_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_27_, 0, v___x_4_);
v___x_28_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_28_, 0, v___x_27_);
lean_ctor_set(v___x_28_, 1, v___y_13_);
v___x_29_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_29_, 0, v___x_28_);
return v___x_29_;
}
}
v___jp_30_:
{
lean_object* v___x_31_; 
v___x_31_ = l_outOfBounds___redArg(v___x_5_);
v___y_18_ = v___x_31_;
goto v___jp_17_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___boxed(lean_object* v___x_36_, lean_object* v___x_37_, lean_object* v___x_38_, lean_object* v_n_39_, lean_object* v_col_40_, lean_object* v_m_41_, lean_object* v_inst_42_, lean_object* v_a_43_, lean_object* v_x_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0(v___x_36_, v___x_37_, v___x_38_, v_n_39_, v_col_40_, v_m_41_, v_inst_42_, v_a_43_, v_x_44_, v___y_45_, v___y_46_, v___y_47_, v___y_48_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
lean_dec_ref(v___y_45_);
lean_dec_ref(v___x_38_);
return v_res_50_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__0(void){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = l_instMonadEIO(lean_box(0));
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_52_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__0);
v___x_53_ = l_StateRefT_x27_instMonad___redArg(v___x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg(lean_object* v_n_59_, lean_object* v_m_60_, lean_object* v_inst_61_, lean_object* v_rowStart_62_, lean_object* v_col_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_){
_start:
{
lean_object* v___x_68_; lean_object* v_toApplicative_69_; lean_object* v_toFunctor_70_; lean_object* v_toSeq_71_; lean_object* v_toSeqLeft_72_; lean_object* v_toSeqRight_73_; lean_object* v___f_74_; lean_object* v___f_75_; lean_object* v___f_76_; lean_object* v___f_77_; lean_object* v___x_78_; lean_object* v___f_79_; lean_object* v___f_80_; lean_object* v___f_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___f_85_; lean_object* v___f_86_; lean_object* v___f_87_; lean_object* v___f_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___f_100_; lean_object* v___x_1516__overap_101_; lean_object* v___x_102_; 
v___x_68_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1);
v_toApplicative_69_ = lean_ctor_get(v___x_68_, 0);
v_toFunctor_70_ = lean_ctor_get(v_toApplicative_69_, 0);
v_toSeq_71_ = lean_ctor_get(v_toApplicative_69_, 2);
v_toSeqLeft_72_ = lean_ctor_get(v_toApplicative_69_, 3);
v_toSeqRight_73_ = lean_ctor_get(v_toApplicative_69_, 4);
v___f_74_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__2));
v___f_75_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_70_, 2);
v___f_76_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_76_, 0, v_toFunctor_70_);
v___f_77_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_77_, 0, v_toFunctor_70_);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___f_76_);
lean_ctor_set(v___x_78_, 1, v___f_77_);
lean_inc(v_toSeqRight_73_);
v___f_79_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_79_, 0, v_toSeqRight_73_);
lean_inc(v_toSeqLeft_72_);
v___f_80_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_80_, 0, v_toSeqLeft_72_);
lean_inc(v_toSeq_71_);
v___f_81_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_81_, 0, v_toSeq_71_);
v___x_82_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_82_, 0, v___x_78_);
lean_ctor_set(v___x_82_, 1, v___f_74_);
lean_ctor_set(v___x_82_, 2, v___f_81_);
lean_ctor_set(v___x_82_, 3, v___f_80_);
lean_ctor_set(v___x_82_, 4, v___f_79_);
v___x_83_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v___f_75_);
v___x_84_ = l_instInhabitedRat;
lean_inc_ref_n(v___x_83_, 6);
v___f_85_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_85_, 0, v___x_83_);
v___f_86_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_86_, 0, v___x_83_);
v___f_87_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_87_, 0, v___x_83_);
v___f_88_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_88_, 0, v___x_83_);
v___x_89_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_89_, 0, lean_box(0));
lean_closure_set(v___x_89_, 1, lean_box(0));
lean_closure_set(v___x_89_, 2, v___x_83_);
v___x_90_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v___f_85_);
v___x_91_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_91_, 0, lean_box(0));
lean_closure_set(v___x_91_, 1, lean_box(0));
lean_closure_set(v___x_91_, 2, v___x_83_);
v___x_92_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_92_, 0, v___x_90_);
lean_ctor_set(v___x_92_, 1, v___x_91_);
lean_ctor_set(v___x_92_, 2, v___f_86_);
lean_ctor_set(v___x_92_, 3, v___f_87_);
lean_ctor_set(v___x_92_, 4, v___f_88_);
v___x_93_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_93_, 0, lean_box(0));
lean_closure_set(v___x_93_, 1, lean_box(0));
lean_closure_set(v___x_93_, 2, v___x_83_);
v___x_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_92_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = lean_unsigned_to_nat(1u);
lean_inc(v_n_59_);
lean_inc(v_rowStart_62_);
v___x_96_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_96_, 0, v_rowStart_62_);
lean_ctor_set(v___x_96_, 1, v_n_59_);
lean_ctor_set(v___x_96_, 2, v___x_95_);
v___x_97_ = lean_box(0);
v___x_98_ = lean_box(0);
v___x_99_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__4));
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___lam__0___boxed), 14, 7);
lean_closure_set(v___f_100_, 0, v___x_98_);
lean_closure_set(v___f_100_, 1, v___x_99_);
lean_closure_set(v___f_100_, 2, v___x_84_);
lean_closure_set(v___f_100_, 3, v_n_59_);
lean_closure_set(v___f_100_, 4, v_col_63_);
lean_closure_set(v___f_100_, 5, v_m_60_);
lean_closure_set(v___f_100_, 6, v_inst_61_);
v___x_1516__overap_101_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_94_, v___x_96_, v___f_100_, v___x_99_, v_rowStart_62_, lean_box(0), lean_box(0));
lean_inc(v_a_66_);
lean_inc_ref(v_a_65_);
v___x_102_ = lean_apply_4(v___x_1516__overap_101_, v_a_64_, v_a_65_, v_a_66_, lean_box(0));
if (lean_obj_tag(v___x_102_) == 0)
{
lean_object* v_a_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_129_; 
v_a_103_ = lean_ctor_get(v___x_102_, 0);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_102_);
if (v_isSharedCheck_129_ == 0)
{
v___x_105_ = v___x_102_;
v_isShared_106_ = v_isSharedCheck_129_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_a_103_);
lean_dec(v___x_102_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_129_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v_fst_107_; lean_object* v_fst_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_127_; 
v_fst_107_ = lean_ctor_get(v_a_103_, 0);
lean_inc(v_fst_107_);
v_fst_108_ = lean_ctor_get(v_fst_107_, 0);
v_isSharedCheck_127_ = !lean_is_exclusive(v_fst_107_);
if (v_isSharedCheck_127_ == 0)
{
lean_object* v_unused_128_; 
v_unused_128_ = lean_ctor_get(v_fst_107_, 1);
lean_dec(v_unused_128_);
v___x_110_ = v_fst_107_;
v_isShared_111_ = v_isSharedCheck_127_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_fst_108_);
lean_dec(v_fst_107_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_127_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
if (lean_obj_tag(v_fst_108_) == 0)
{
lean_object* v_snd_112_; lean_object* v___x_114_; 
v_snd_112_ = lean_ctor_get(v_a_103_, 1);
lean_inc(v_snd_112_);
lean_dec(v_a_103_);
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 1, v_snd_112_);
lean_ctor_set(v___x_110_, 0, v___x_97_);
v___x_114_ = v___x_110_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v___x_97_);
lean_ctor_set(v_reuseFailAlloc_118_, 1, v_snd_112_);
v___x_114_ = v_reuseFailAlloc_118_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
lean_object* v___x_116_; 
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 0, v___x_114_);
v___x_116_ = v___x_105_;
goto v_reusejp_115_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___x_114_);
v___x_116_ = v_reuseFailAlloc_117_;
goto v_reusejp_115_;
}
v_reusejp_115_:
{
return v___x_116_;
}
}
}
else
{
lean_object* v_snd_119_; lean_object* v_val_120_; lean_object* v___x_122_; 
v_snd_119_ = lean_ctor_get(v_a_103_, 1);
lean_inc(v_snd_119_);
lean_dec(v_a_103_);
v_val_120_ = lean_ctor_get(v_fst_108_, 0);
lean_inc(v_val_120_);
lean_dec_ref_known(v_fst_108_, 1);
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 1, v_snd_119_);
lean_ctor_set(v___x_110_, 0, v_val_120_);
v___x_122_ = v___x_110_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v_val_120_);
lean_ctor_set(v_reuseFailAlloc_126_, 1, v_snd_119_);
v___x_122_ = v_reuseFailAlloc_126_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
lean_object* v___x_124_; 
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 0, v___x_122_);
v___x_124_ = v___x_105_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v___x_122_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
return v___x_124_;
}
}
}
}
}
}
else
{
lean_object* v_a_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_137_; 
v_a_130_ = lean_ctor_get(v___x_102_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_102_);
if (v_isSharedCheck_137_ == 0)
{
v___x_132_ = v___x_102_;
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
else
{
lean_inc(v_a_130_);
lean_dec(v___x_102_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_135_; 
if (v_isShared_133_ == 0)
{
v___x_135_ = v___x_132_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_a_130_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___boxed(lean_object* v_n_138_, lean_object* v_m_139_, lean_object* v_inst_140_, lean_object* v_rowStart_141_, lean_object* v_col_142_, lean_object* v_a_143_, lean_object* v_a_144_, lean_object* v_a_145_, lean_object* v_a_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg(v_n_138_, v_m_139_, v_inst_140_, v_rowStart_141_, v_col_142_, v_a_143_, v_a_144_, v_a_145_);
lean_dec(v_a_145_);
lean_dec_ref(v_a_144_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow(lean_object* v_n_148_, lean_object* v_m_149_, lean_object* v_matType_150_, lean_object* v_inst_151_, lean_object* v_rowStart_152_, lean_object* v_col_153_, lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg(v_n_148_, v_m_149_, v_inst_151_, v_rowStart_152_, v_col_153_, v_a_154_, v_a_155_, v_a_156_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___boxed(lean_object* v_n_159_, lean_object* v_m_160_, lean_object* v_matType_161_, lean_object* v_inst_162_, lean_object* v_rowStart_163_, lean_object* v_col_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow(v_n_159_, v_m_160_, v_matType_161_, v_inst_162_, v_rowStart_163_, v_col_164_, v_a_165_, v_a_166_, v_a_167_);
lean_dec(v_a_167_);
lean_dec_ref(v_a_166_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__0(lean_object* v_fst_170_, lean_object* v_inst_171_, lean_object* v_n_172_, lean_object* v_m_173_, lean_object* v___x_174_, lean_object* v___x_175_, lean_object* v_snd_176_, lean_object* v_a_177_, lean_object* v_x_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_){
_start:
{
lean_object* v___y_185_; uint8_t v___x_191_; lean_object* v___y_193_; 
v___x_191_ = lean_nat_dec_eq(v_a_177_, v_fst_170_);
if (v___x_191_ == 0)
{
lean_object* v___x_199_; uint8_t v___x_202_; 
v___x_199_ = l_instInhabitedRat;
v___x_202_ = lean_nat_dec_lt(v_a_177_, v_n_172_);
if (v___x_202_ == 0)
{
lean_dec(v_snd_176_);
goto v___jp_200_;
}
else
{
uint8_t v___x_203_; 
v___x_203_ = lean_nat_dec_lt(v_snd_176_, v_m_173_);
if (v___x_203_ == 0)
{
lean_dec(v_snd_176_);
goto v___jp_200_;
}
else
{
lean_object* v_getElem_204_; lean_object* v___x_205_; 
v_getElem_204_ = lean_ctor_get(v_inst_171_, 0);
lean_inc_ref(v_getElem_204_);
lean_inc(v_a_177_);
lean_inc(v___y_180_);
lean_inc(v_m_173_);
lean_inc(v_n_172_);
v___x_205_ = lean_apply_5(v_getElem_204_, v_n_172_, v_m_173_, v___y_180_, v_a_177_, v_snd_176_);
v___y_193_ = v___x_205_;
goto v___jp_192_;
}
}
v___jp_200_:
{
lean_object* v___x_201_; 
v___x_201_ = l_outOfBounds___redArg(v___x_199_);
v___y_193_ = v___x_201_;
goto v___jp_192_;
}
}
else
{
lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
lean_dec(v_a_177_);
lean_dec(v_snd_176_);
lean_dec(v___x_175_);
lean_dec(v_m_173_);
lean_dec(v_n_172_);
lean_dec_ref(v_inst_171_);
lean_dec(v_fst_170_);
v___x_206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_206_, 0, v___x_174_);
v___x_207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
lean_ctor_set(v___x_207_, 1, v___y_180_);
v___x_208_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
return v___x_208_;
}
v___jp_184_:
{
lean_object* v_subtractRow_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v_subtractRow_186_ = lean_ctor_get(v_inst_171_, 5);
lean_inc(v_subtractRow_186_);
lean_dec_ref(v_inst_171_);
v___x_187_ = lean_apply_6(v_subtractRow_186_, v_n_172_, v_m_173_, v___y_180_, v_fst_170_, v_a_177_, v___y_185_);
v___x_188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_188_, 0, v___x_174_);
v___x_189_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
lean_ctor_set(v___x_189_, 1, v___x_187_);
v___x_190_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_190_, 0, v___x_189_);
return v___x_190_;
}
v___jp_192_:
{
lean_object* v___x_194_; uint8_t v___x_195_; 
v___x_194_ = l_Rat_instNatCast___lam__0(v___x_175_);
v___x_195_ = l_instDecidableEqRat_decEq(v___y_193_, v___x_194_);
lean_dec_ref(v___x_194_);
if (v___x_195_ == 0)
{
v___y_185_ = v___y_193_;
goto v___jp_184_;
}
else
{
if (v___x_191_ == 0)
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
lean_dec_ref(v___y_193_);
lean_dec(v_a_177_);
lean_dec(v_m_173_);
lean_dec(v_n_172_);
lean_dec_ref(v_inst_171_);
lean_dec(v_fst_170_);
v___x_196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_196_, 0, v___x_174_);
v___x_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v___y_180_);
v___x_198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_198_, 0, v___x_197_);
return v___x_198_;
}
else
{
v___y_185_ = v___y_193_;
goto v___jp_184_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__0___boxed(lean_object* v_fst_209_, lean_object* v_inst_210_, lean_object* v_n_211_, lean_object* v_m_212_, lean_object* v___x_213_, lean_object* v___x_214_, lean_object* v_snd_215_, lean_object* v_a_216_, lean_object* v_x_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__0(v_fst_209_, v_inst_210_, v_n_211_, v_m_212_, v___x_213_, v___x_214_, v_snd_215_, v_a_216_, v_x_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1(lean_object* v___x_237_, lean_object* v_n_238_, lean_object* v_inst_239_, lean_object* v_m_240_, lean_object* v___x_241_, lean_object* v_b_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v_a_248_; lean_object* v_snd_249_; lean_object* v_snd_253_; lean_object* v_snd_254_; lean_object* v_fst_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_373_; 
v_snd_253_ = lean_ctor_get(v_b_242_, 1);
lean_inc(v_snd_253_);
v_snd_254_ = lean_ctor_get(v_snd_253_, 1);
lean_inc(v_snd_254_);
v_fst_255_ = lean_ctor_get(v_b_242_, 0);
v_isSharedCheck_373_ = !lean_is_exclusive(v_b_242_);
if (v_isSharedCheck_373_ == 0)
{
lean_object* v_unused_374_; 
v_unused_374_ = lean_ctor_get(v_b_242_, 1);
lean_dec(v_unused_374_);
v___x_257_ = v_b_242_;
v_isShared_258_ = v_isSharedCheck_373_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_fst_255_);
lean_dec(v_b_242_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_373_;
goto v_resetjp_256_;
}
v___jp_247_:
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_250_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_250_, 0, v_a_248_);
v___x_251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v_snd_249_);
v___x_252_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
return v___x_252_;
}
v_resetjp_256_:
{
lean_object* v_fst_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_371_; 
v_fst_259_ = lean_ctor_get(v_snd_253_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v_snd_253_);
if (v_isSharedCheck_371_ == 0)
{
lean_object* v_unused_372_; 
v_unused_372_ = lean_ctor_get(v_snd_253_, 1);
lean_dec(v_unused_372_);
v___x_261_ = v_snd_253_;
v_isShared_262_ = v_isSharedCheck_371_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_fst_259_);
lean_dec(v_snd_253_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_371_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v_fst_263_; lean_object* v_snd_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_370_; 
v_fst_263_ = lean_ctor_get(v_snd_254_, 0);
v_snd_264_ = lean_ctor_get(v_snd_254_, 1);
v_isSharedCheck_370_ = !lean_is_exclusive(v_snd_254_);
if (v_isSharedCheck_370_ == 0)
{
v___x_266_ = v_snd_254_;
v_isShared_267_ = v_isSharedCheck_370_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_snd_264_);
lean_inc(v_fst_263_);
lean_dec(v_snd_254_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_370_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___y_269_; lean_object* v___y_304_; lean_object* v___y_305_; lean_object* v___y_306_; uint8_t v___y_310_; uint8_t v___x_368_; 
v___x_368_ = lean_nat_dec_lt(v_fst_263_, v_n_238_);
if (v___x_368_ == 0)
{
v___y_310_ = v___x_368_;
goto v___jp_309_;
}
else
{
uint8_t v___x_369_; 
v___x_369_ = lean_nat_dec_lt(v_snd_264_, v_m_240_);
v___y_310_ = v___x_369_;
goto v___jp_309_;
}
v___jp_268_:
{
lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___f_273_; lean_object* v___x_8850__overap_274_; lean_object* v___x_275_; 
v___x_270_ = lean_unsigned_to_nat(1u);
lean_inc(v_n_238_);
lean_inc_n(v___x_237_, 2);
v___x_271_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_271_, 0, v___x_237_);
lean_ctor_set(v___x_271_, 1, v_n_238_);
lean_ctor_set(v___x_271_, 2, v___x_270_);
v___x_272_ = lean_box(0);
lean_inc(v_snd_264_);
lean_inc(v_fst_263_);
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__0___boxed), 14, 7);
lean_closure_set(v___f_273_, 0, v_fst_263_);
lean_closure_set(v___f_273_, 1, v_inst_239_);
lean_closure_set(v___f_273_, 2, v_n_238_);
lean_closure_set(v___f_273_, 3, v_m_240_);
lean_closure_set(v___f_273_, 4, v___x_272_);
lean_closure_set(v___f_273_, 5, v___x_237_);
lean_closure_set(v___f_273_, 6, v_snd_264_);
v___x_8850__overap_274_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_241_, v___x_271_, v___f_273_, v___x_272_, v___x_237_, lean_box(0), lean_box(0));
lean_inc(v___y_245_);
lean_inc_ref(v___y_244_);
v___x_275_ = lean_apply_4(v___x_8850__overap_274_, v___y_269_, v___y_244_, v___y_245_, lean_box(0));
if (lean_obj_tag(v___x_275_) == 0)
{
lean_object* v_a_276_; lean_object* v_snd_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_293_; 
v_a_276_ = lean_ctor_get(v___x_275_, 0);
lean_inc(v_a_276_);
lean_dec_ref_known(v___x_275_, 1);
v_snd_277_ = lean_ctor_get(v_a_276_, 1);
v_isSharedCheck_293_ = !lean_is_exclusive(v_a_276_);
if (v_isSharedCheck_293_ == 0)
{
lean_object* v_unused_294_; 
v_unused_294_ = lean_ctor_get(v_a_276_, 0);
lean_dec(v_unused_294_);
v___x_279_ = v_a_276_;
v_isShared_280_ = v_isSharedCheck_293_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_snd_277_);
lean_dec(v_a_276_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_293_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_285_; 
lean_inc(v_snd_264_);
v___x_281_ = lean_array_push(v_fst_259_, v_snd_264_);
v___x_282_ = lean_nat_add(v_fst_263_, v___x_270_);
lean_dec(v_fst_263_);
v___x_283_ = lean_nat_add(v_snd_264_, v___x_270_);
lean_dec(v_snd_264_);
if (v_isShared_280_ == 0)
{
lean_ctor_set(v___x_279_, 1, v___x_283_);
lean_ctor_set(v___x_279_, 0, v___x_282_);
v___x_285_ = v___x_279_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v___x_282_);
lean_ctor_set(v_reuseFailAlloc_292_, 1, v___x_283_);
v___x_285_ = v_reuseFailAlloc_292_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
lean_object* v___x_287_; 
if (v_isShared_267_ == 0)
{
lean_ctor_set(v___x_266_, 1, v___x_285_);
lean_ctor_set(v___x_266_, 0, v___x_281_);
v___x_287_ = v___x_266_;
goto v_reusejp_286_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v___x_281_);
lean_ctor_set(v_reuseFailAlloc_291_, 1, v___x_285_);
v___x_287_ = v_reuseFailAlloc_291_;
goto v_reusejp_286_;
}
v_reusejp_286_:
{
lean_object* v___x_289_; 
if (v_isShared_262_ == 0)
{
lean_ctor_set(v___x_261_, 1, v___x_287_);
lean_ctor_set(v___x_261_, 0, v_fst_255_);
v___x_289_ = v___x_261_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_fst_255_);
lean_ctor_set(v_reuseFailAlloc_290_, 1, v___x_287_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
v_a_248_ = v___x_289_;
v_snd_249_ = v_snd_277_;
goto v___jp_247_;
}
}
}
}
}
else
{
lean_object* v_a_295_; lean_object* v___x_297_; uint8_t v_isShared_298_; uint8_t v_isSharedCheck_302_; 
lean_del_object(v___x_266_);
lean_dec(v_snd_264_);
lean_dec(v_fst_263_);
lean_del_object(v___x_261_);
lean_dec(v_fst_259_);
lean_dec(v_fst_255_);
v_a_295_ = lean_ctor_get(v___x_275_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_275_);
if (v_isSharedCheck_302_ == 0)
{
v___x_297_ = v___x_275_;
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
else
{
lean_inc(v_a_295_);
lean_dec(v___x_275_);
v___x_297_ = lean_box(0);
v_isShared_298_ = v_isSharedCheck_302_;
goto v_resetjp_296_;
}
v_resetjp_296_:
{
lean_object* v___x_300_; 
if (v_isShared_298_ == 0)
{
v___x_300_ = v___x_297_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v_a_295_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
v___jp_303_:
{
lean_object* v___x_307_; lean_object* v___x_308_; 
v___x_307_ = l_outOfBounds___redArg(v___y_305_);
lean_inc(v_fst_263_);
lean_inc(v_m_240_);
lean_inc(v_n_238_);
v___x_308_ = lean_apply_5(v___y_304_, v_n_238_, v_m_240_, v___y_306_, v_fst_263_, v___x_307_);
v___y_269_ = v___x_308_;
goto v___jp_268_;
}
v___jp_309_:
{
if (v___y_310_ == 0)
{
lean_object* v___x_312_; 
lean_del_object(v___x_266_);
lean_del_object(v___x_261_);
lean_dec_ref(v___x_241_);
lean_dec(v_m_240_);
lean_dec_ref(v_inst_239_);
lean_dec(v_n_238_);
lean_dec(v___x_237_);
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 1, v_snd_264_);
lean_ctor_set(v___x_257_, 0, v_fst_263_);
v___x_312_ = v___x_257_;
goto v_reusejp_311_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_318_, 0, v_fst_263_);
lean_ctor_set(v_reuseFailAlloc_318_, 1, v_snd_264_);
v___x_312_ = v_reuseFailAlloc_318_;
goto v_reusejp_311_;
}
v_reusejp_311_:
{
lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; 
v___x_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_313_, 0, v_fst_259_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
v___x_314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_314_, 0, v_fst_255_);
lean_ctor_set(v___x_314_, 1, v___x_313_);
v___x_315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_315_, 0, v___x_314_);
v___x_316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v___y_243_);
v___x_317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_317_, 0, v___x_316_);
return v___x_317_;
}
}
else
{
lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_319_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___closed__6));
v___x_320_ = l_Lean_Name_toString(v___x_319_, v___y_310_);
v___x_321_ = l_Lean_Core_checkSystem(v___x_320_, v___y_244_, v___y_245_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_object* v___x_322_; 
lean_dec_ref_known(v___x_321_, 1);
lean_inc(v_snd_264_);
lean_inc(v_fst_263_);
lean_inc_ref(v_inst_239_);
lean_inc(v_m_240_);
lean_inc(v_n_238_);
v___x_322_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg(v_n_238_, v_m_240_, v_inst_239_, v_fst_263_, v_snd_264_, v___y_243_, v___y_244_, v___y_245_);
if (lean_obj_tag(v___x_322_) == 0)
{
lean_object* v_a_323_; lean_object* v_fst_324_; 
v_a_323_ = lean_ctor_get(v___x_322_, 0);
lean_inc(v_a_323_);
lean_dec_ref_known(v___x_322_, 1);
v_fst_324_ = lean_ctor_get(v_a_323_, 0);
if (lean_obj_tag(v_fst_324_) == 0)
{
lean_object* v_snd_325_; lean_object* v___x_327_; uint8_t v_isShared_328_; uint8_t v_isSharedCheck_339_; 
lean_del_object(v___x_266_);
lean_del_object(v___x_261_);
lean_dec_ref(v___x_241_);
lean_dec(v_m_240_);
lean_dec_ref(v_inst_239_);
lean_dec(v_n_238_);
lean_dec(v___x_237_);
v_snd_325_ = lean_ctor_get(v_a_323_, 1);
v_isSharedCheck_339_ = !lean_is_exclusive(v_a_323_);
if (v_isSharedCheck_339_ == 0)
{
lean_object* v_unused_340_; 
v_unused_340_ = lean_ctor_get(v_a_323_, 0);
lean_dec(v_unused_340_);
v___x_327_ = v_a_323_;
v_isShared_328_ = v_isSharedCheck_339_;
goto v_resetjp_326_;
}
else
{
lean_inc(v_snd_325_);
lean_dec(v_a_323_);
v___x_327_ = lean_box(0);
v_isShared_328_ = v_isSharedCheck_339_;
goto v_resetjp_326_;
}
v_resetjp_326_:
{
lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_333_; 
lean_inc(v_snd_264_);
v___x_329_ = lean_array_push(v_fst_255_, v_snd_264_);
v___x_330_ = lean_unsigned_to_nat(1u);
v___x_331_ = lean_nat_add(v_snd_264_, v___x_330_);
lean_dec(v_snd_264_);
if (v_isShared_328_ == 0)
{
lean_ctor_set(v___x_327_, 1, v___x_331_);
lean_ctor_set(v___x_327_, 0, v_fst_263_);
v___x_333_ = v___x_327_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v_fst_263_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v___x_331_);
v___x_333_ = v_reuseFailAlloc_338_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
lean_object* v___x_335_; 
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 1, v___x_333_);
lean_ctor_set(v___x_257_, 0, v_fst_259_);
v___x_335_ = v___x_257_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v_fst_259_);
lean_ctor_set(v_reuseFailAlloc_337_, 1, v___x_333_);
v___x_335_ = v_reuseFailAlloc_337_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_336_; 
v___x_336_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_329_);
lean_ctor_set(v___x_336_, 1, v___x_335_);
v_a_248_ = v___x_336_;
v_snd_249_ = v_snd_325_;
goto v___jp_247_;
}
}
}
}
else
{
lean_object* v_snd_341_; lean_object* v_val_342_; lean_object* v_getElem_343_; lean_object* v_swapRows_344_; lean_object* v_divideRow_345_; lean_object* v___x_346_; lean_object* v___x_347_; uint8_t v___x_348_; 
lean_inc_ref(v_fst_324_);
lean_del_object(v___x_257_);
v_snd_341_ = lean_ctor_get(v_a_323_, 1);
lean_inc(v_snd_341_);
lean_dec(v_a_323_);
v_val_342_ = lean_ctor_get(v_fst_324_, 0);
lean_inc(v_val_342_);
lean_dec_ref_known(v_fst_324_, 1);
v_getElem_343_ = lean_ctor_get(v_inst_239_, 0);
v_swapRows_344_ = lean_ctor_get(v_inst_239_, 4);
v_divideRow_345_ = lean_ctor_get(v_inst_239_, 6);
lean_inc(v_swapRows_344_);
lean_inc(v_fst_263_);
lean_inc(v_m_240_);
lean_inc(v_n_238_);
v___x_346_ = lean_apply_5(v_swapRows_344_, v_n_238_, v_m_240_, v_snd_341_, v_fst_263_, v_val_342_);
v___x_347_ = l_instInhabitedRat;
v___x_348_ = lean_nat_dec_lt(v_fst_263_, v_n_238_);
if (v___x_348_ == 0)
{
lean_inc(v_divideRow_345_);
v___y_304_ = v_divideRow_345_;
v___y_305_ = v___x_347_;
v___y_306_ = v___x_346_;
goto v___jp_303_;
}
else
{
uint8_t v___x_349_; 
v___x_349_ = lean_nat_dec_lt(v_snd_264_, v_m_240_);
if (v___x_349_ == 0)
{
lean_inc(v_divideRow_345_);
v___y_304_ = v_divideRow_345_;
v___y_305_ = v___x_347_;
v___y_306_ = v___x_346_;
goto v___jp_303_;
}
else
{
lean_object* v___x_350_; lean_object* v___x_351_; 
lean_inc_ref(v_getElem_343_);
lean_inc(v_snd_264_);
lean_inc_n(v_fst_263_, 2);
lean_inc(v___x_346_);
lean_inc_n(v_m_240_, 2);
lean_inc_n(v_n_238_, 2);
v___x_350_ = lean_apply_5(v_getElem_343_, v_n_238_, v_m_240_, v___x_346_, v_fst_263_, v_snd_264_);
lean_inc(v_divideRow_345_);
v___x_351_ = lean_apply_5(v_divideRow_345_, v_n_238_, v_m_240_, v___x_346_, v_fst_263_, v___x_350_);
v___y_269_ = v___x_351_;
goto v___jp_268_;
}
}
}
}
else
{
lean_object* v_a_352_; lean_object* v___x_354_; uint8_t v_isShared_355_; uint8_t v_isSharedCheck_359_; 
lean_del_object(v___x_266_);
lean_dec(v_snd_264_);
lean_dec(v_fst_263_);
lean_del_object(v___x_261_);
lean_dec(v_fst_259_);
lean_del_object(v___x_257_);
lean_dec(v_fst_255_);
lean_dec_ref(v___x_241_);
lean_dec(v_m_240_);
lean_dec_ref(v_inst_239_);
lean_dec(v_n_238_);
lean_dec(v___x_237_);
v_a_352_ = lean_ctor_get(v___x_322_, 0);
v_isSharedCheck_359_ = !lean_is_exclusive(v___x_322_);
if (v_isSharedCheck_359_ == 0)
{
v___x_354_ = v___x_322_;
v_isShared_355_ = v_isSharedCheck_359_;
goto v_resetjp_353_;
}
else
{
lean_inc(v_a_352_);
lean_dec(v___x_322_);
v___x_354_ = lean_box(0);
v_isShared_355_ = v_isSharedCheck_359_;
goto v_resetjp_353_;
}
v_resetjp_353_:
{
lean_object* v___x_357_; 
if (v_isShared_355_ == 0)
{
v___x_357_ = v___x_354_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v_a_352_);
v___x_357_ = v_reuseFailAlloc_358_;
goto v_reusejp_356_;
}
v_reusejp_356_:
{
return v___x_357_;
}
}
}
}
else
{
lean_object* v_a_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_367_; 
lean_del_object(v___x_266_);
lean_dec(v_snd_264_);
lean_dec(v_fst_263_);
lean_del_object(v___x_261_);
lean_dec(v_fst_259_);
lean_del_object(v___x_257_);
lean_dec(v_fst_255_);
lean_dec(v___y_243_);
lean_dec_ref(v___x_241_);
lean_dec(v_m_240_);
lean_dec_ref(v_inst_239_);
lean_dec(v_n_238_);
lean_dec(v___x_237_);
v_a_360_ = lean_ctor_get(v___x_321_, 0);
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_321_);
if (v_isSharedCheck_367_ == 0)
{
v___x_362_ = v___x_321_;
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_a_360_);
lean_dec(v___x_321_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_365_; 
if (v_isShared_363_ == 0)
{
v___x_365_ = v___x_362_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_a_360_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
return v___x_365_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___boxed(lean_object* v___x_375_, lean_object* v_n_376_, lean_object* v_inst_377_, lean_object* v_m_378_, lean_object* v___x_379_, lean_object* v_b_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1(v___x_375_, v_n_376_, v_inst_377_, v_m_378_, v___x_379_, v_b_380_, v___y_381_, v___y_382_, v___y_383_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
return v_res_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__2(lean_object* v_a_386_, lean_object* v_x_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_393_ = lean_array_push(v___y_388_, v_a_386_);
v___x_394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
v___x_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_394_);
lean_ctor_set(v___x_395_, 1, v___y_389_);
v___x_396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_396_, 0, v___x_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__2___boxed(lean_object* v_a_397_, lean_object* v_x_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__2(v_a_397_, v_x_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
return v_res_404_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__3(lean_object* v_fst_405_, lean_object* v_x_406_){
_start:
{
uint8_t v___x_407_; 
v___x_407_ = lean_nat_dec_eq(v_x_406_, v_fst_405_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__3___boxed(lean_object* v_fst_408_, lean_object* v_x_409_){
_start:
{
uint8_t v_res_410_; lean_object* v_r_411_; 
v_res_410_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__3(v_fst_408_, v_x_409_);
lean_dec(v_x_409_);
lean_dec(v_fst_408_);
v_r_411_ = lean_box(v_res_410_);
return v_r_411_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__3(void){
_start:
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_415_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__2));
v___x_416_ = lean_unsigned_to_nat(14u);
v___x_417_ = lean_unsigned_to_nat(22u);
v___x_418_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__1));
v___x_419_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__0));
v___x_420_ = l_mkPanicMessageWithDecl(v___x_419_, v___x_418_, v___x_417_, v___x_416_, v___x_415_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4(lean_object* v___x_421_, lean_object* v_fst_422_, lean_object* v_fst_423_, lean_object* v___x_424_, lean_object* v_x_425_){
_start:
{
lean_object* v_snd_426_; lean_object* v_fst_427_; lean_object* v___x_429_; uint8_t v_isShared_430_; uint8_t v_isSharedCheck_455_; 
v_snd_426_ = lean_ctor_get(v_x_425_, 1);
v_fst_427_ = lean_ctor_get(v_x_425_, 0);
v_isSharedCheck_455_ = !lean_is_exclusive(v_x_425_);
if (v_isSharedCheck_455_ == 0)
{
v___x_429_ = v_x_425_;
v_isShared_430_ = v_isSharedCheck_455_;
goto v_resetjp_428_;
}
else
{
lean_inc(v_snd_426_);
lean_inc(v_fst_427_);
lean_dec(v_x_425_);
v___x_429_ = lean_box(0);
v_isShared_430_ = v_isSharedCheck_455_;
goto v_resetjp_428_;
}
v_resetjp_428_:
{
lean_object* v_fst_431_; lean_object* v_snd_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_454_; 
v_fst_431_ = lean_ctor_get(v_snd_426_, 0);
v_snd_432_ = lean_ctor_get(v_snd_426_, 1);
v_isSharedCheck_454_ = !lean_is_exclusive(v_snd_426_);
if (v_isSharedCheck_454_ == 0)
{
v___x_434_ = v_snd_426_;
v_isShared_435_ = v_isSharedCheck_454_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_snd_432_);
lean_inc(v_fst_431_);
lean_dec(v_snd_426_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_454_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___y_437_; lean_object* v___x_446_; uint8_t v___x_447_; 
v___x_446_ = lean_array_get_borrowed(v___x_421_, v_fst_422_, v_fst_427_);
v___x_447_ = lean_nat_dec_eq(v_fst_431_, v___x_446_);
if (v___x_447_ == 0)
{
lean_object* v___f_448_; lean_object* v___x_449_; 
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_448_, 0, v_fst_431_);
v___x_449_ = l_Array_findIdx_x3f_loop___redArg(v___f_448_, v_fst_423_, v___x_424_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_450_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__3, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___closed__3);
v___x_451_ = l_panic___redArg(v___x_421_, v___x_450_);
v___y_437_ = v___x_451_;
goto v___jp_436_;
}
else
{
lean_object* v_val_452_; 
v_val_452_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_val_452_);
lean_dec_ref_known(v___x_449_, 1);
v___y_437_ = v_val_452_;
goto v___jp_436_;
}
}
else
{
lean_object* v___x_453_; 
lean_del_object(v___x_434_);
lean_dec(v_snd_432_);
lean_dec(v_fst_431_);
lean_del_object(v___x_429_);
lean_dec(v_fst_427_);
lean_dec(v___x_424_);
v___x_453_ = lean_box(0);
return v___x_453_;
}
v___jp_436_:
{
lean_object* v___x_438_; lean_object* v___x_440_; 
v___x_438_ = l_Rat_neg(v_snd_432_);
if (v_isShared_435_ == 0)
{
lean_ctor_set(v___x_434_, 1, v___x_438_);
lean_ctor_set(v___x_434_, 0, v___y_437_);
v___x_440_ = v___x_434_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v___y_437_);
lean_ctor_set(v_reuseFailAlloc_445_, 1, v___x_438_);
v___x_440_ = v_reuseFailAlloc_445_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
lean_object* v___x_442_; 
if (v_isShared_430_ == 0)
{
lean_ctor_set(v___x_429_, 1, v___x_440_);
v___x_442_ = v___x_429_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_444_; 
v_reuseFailAlloc_444_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_444_, 0, v_fst_427_);
lean_ctor_set(v_reuseFailAlloc_444_, 1, v___x_440_);
v___x_442_ = v_reuseFailAlloc_444_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
lean_object* v___x_443_; 
v___x_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
return v___x_443_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___boxed(lean_object* v___x_456_, lean_object* v_fst_457_, lean_object* v_fst_458_, lean_object* v___x_459_, lean_object* v_x_460_){
_start:
{
lean_object* v_res_461_; 
v_res_461_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4(v___x_456_, v_fst_457_, v_fst_458_, v___x_459_, v_x_460_);
lean_dec_ref(v_fst_458_);
lean_dec(v_fst_457_);
lean_dec(v___x_456_);
return v_res_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg(lean_object* v_n_473_, lean_object* v_m_474_, lean_object* v_inst_475_, lean_object* v_a_476_, lean_object* v_a_477_, lean_object* v_a_478_){
_start:
{
lean_object* v___x_480_; lean_object* v_toApplicative_481_; lean_object* v_toFunctor_482_; lean_object* v_toSeq_483_; lean_object* v_toSeqLeft_484_; lean_object* v_toSeqRight_485_; lean_object* v___f_486_; lean_object* v___f_487_; lean_object* v___f_488_; lean_object* v___f_489_; lean_object* v___x_490_; lean_object* v___f_491_; lean_object* v___f_492_; lean_object* v___f_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v_free_497_; lean_object* v___f_498_; lean_object* v___f_499_; lean_object* v___f_500_; lean_object* v___f_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___f_508_; lean_object* v___x_509_; lean_object* v___x_8234__overap_510_; lean_object* v___x_511_; 
v___x_480_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__1);
v_toApplicative_481_ = lean_ctor_get(v___x_480_, 0);
v_toFunctor_482_ = lean_ctor_get(v_toApplicative_481_, 0);
v_toSeq_483_ = lean_ctor_get(v_toApplicative_481_, 2);
v_toSeqLeft_484_ = lean_ctor_get(v_toApplicative_481_, 3);
v_toSeqRight_485_ = lean_ctor_get(v_toApplicative_481_, 4);
v___f_486_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__2));
v___f_487_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_findNonzeroRow___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_482_, 2);
v___f_488_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_488_, 0, v_toFunctor_482_);
v___f_489_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_489_, 0, v_toFunctor_482_);
v___x_490_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_490_, 0, v___f_488_);
lean_ctor_set(v___x_490_, 1, v___f_489_);
lean_inc(v_toSeqRight_485_);
v___f_491_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_491_, 0, v_toSeqRight_485_);
lean_inc(v_toSeqLeft_484_);
v___f_492_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_492_, 0, v_toSeqLeft_484_);
lean_inc(v_toSeq_483_);
v___f_493_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_493_, 0, v_toSeq_483_);
v___x_494_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_494_, 0, v___x_490_);
lean_ctor_set(v___x_494_, 1, v___f_486_);
lean_ctor_set(v___x_494_, 2, v___f_493_);
lean_ctor_set(v___x_494_, 3, v___f_492_);
lean_ctor_set(v___x_494_, 4, v___f_491_);
v___x_495_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_495_, 0, v___x_494_);
lean_ctor_set(v___x_495_, 1, v___f_487_);
v___x_496_ = lean_unsigned_to_nat(0u);
v_free_497_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__0));
lean_inc_ref_n(v___x_495_, 6);
v___f_498_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_498_, 0, v___x_495_);
v___f_499_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_499_, 0, v___x_495_);
v___f_500_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_500_, 0, v___x_495_);
v___f_501_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_501_, 0, v___x_495_);
v___x_502_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_502_, 0, lean_box(0));
lean_closure_set(v___x_502_, 1, lean_box(0));
lean_closure_set(v___x_502_, 2, v___x_495_);
v___x_503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_502_);
lean_ctor_set(v___x_503_, 1, v___f_498_);
v___x_504_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_504_, 0, lean_box(0));
lean_closure_set(v___x_504_, 1, lean_box(0));
lean_closure_set(v___x_504_, 2, v___x_495_);
v___x_505_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_505_, 0, v___x_503_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
lean_ctor_set(v___x_505_, 2, v___f_499_);
lean_ctor_set(v___x_505_, 3, v___f_500_);
lean_ctor_set(v___x_505_, 4, v___f_501_);
v___x_506_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_506_, 0, lean_box(0));
lean_closure_set(v___x_506_, 1, lean_box(0));
lean_closure_set(v___x_506_, 2, v___x_495_);
v___x_507_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_507_, 0, v___x_505_);
lean_ctor_set(v___x_507_, 1, v___x_506_);
lean_inc_ref_n(v___x_507_, 2);
lean_inc(v_m_474_);
lean_inc_ref(v_inst_475_);
lean_inc(v_n_473_);
v___f_508_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__1___boxed), 10, 5);
lean_closure_set(v___f_508_, 0, v___x_496_);
lean_closure_set(v___f_508_, 1, v_n_473_);
lean_closure_set(v___f_508_, 2, v_inst_475_);
lean_closure_set(v___f_508_, 3, v_m_474_);
lean_closure_set(v___f_508_, 4, v___x_507_);
v___x_509_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__3));
v___x_8234__overap_510_ = l___private_Init_While_0__repeatM_erased___redArg(v___x_507_, v___f_508_, v___x_509_);
lean_inc(v_a_478_);
lean_inc_ref(v_a_477_);
v___x_511_ = lean_apply_4(v___x_8234__overap_510_, v_a_476_, v_a_477_, v_a_478_, lean_box(0));
if (lean_obj_tag(v___x_511_) == 0)
{
lean_object* v_a_512_; lean_object* v_fst_513_; lean_object* v_snd_514_; lean_object* v_snd_515_; lean_object* v_snd_516_; lean_object* v_fst_517_; lean_object* v_fst_518_; lean_object* v_snd_519_; lean_object* v___f_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_8257__overap_523_; lean_object* v___x_524_; 
v_a_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc(v_a_512_);
lean_dec_ref_known(v___x_511_, 1);
v_fst_513_ = lean_ctor_get(v_a_512_, 0);
lean_inc(v_fst_513_);
v_snd_514_ = lean_ctor_get(v_fst_513_, 1);
lean_inc(v_snd_514_);
v_snd_515_ = lean_ctor_get(v_snd_514_, 1);
lean_inc(v_snd_515_);
v_snd_516_ = lean_ctor_get(v_a_512_, 1);
lean_inc(v_snd_516_);
lean_dec(v_a_512_);
v_fst_517_ = lean_ctor_get(v_fst_513_, 0);
lean_inc(v_fst_517_);
lean_dec(v_fst_513_);
v_fst_518_ = lean_ctor_get(v_snd_514_, 0);
lean_inc(v_fst_518_);
lean_dec(v_snd_514_);
v_snd_519_ = lean_ctor_get(v_snd_515_, 1);
lean_inc_n(v_snd_519_, 2);
lean_dec(v_snd_515_);
v___f_520_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___closed__4));
v___x_521_ = lean_unsigned_to_nat(1u);
lean_inc(v_m_474_);
v___x_522_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_522_, 0, v_snd_519_);
lean_ctor_set(v___x_522_, 1, v_m_474_);
lean_ctor_set(v___x_522_, 2, v___x_521_);
v___x_8257__overap_523_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_507_, v___x_522_, v___f_520_, v_fst_517_, v_snd_519_, lean_box(0), lean_box(0));
lean_inc(v_a_478_);
lean_inc_ref(v_a_477_);
v___x_524_ = lean_apply_4(v___x_8257__overap_523_, v_snd_516_, v_a_477_, v_a_478_, lean_box(0));
if (lean_obj_tag(v___x_524_) == 0)
{
lean_object* v_a_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_550_; 
v_a_525_ = lean_ctor_get(v___x_524_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_524_);
if (v_isSharedCheck_550_ == 0)
{
v___x_527_ = v___x_524_;
v_isShared_528_ = v_isSharedCheck_550_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_a_525_);
lean_dec(v___x_524_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_550_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v_fst_529_; lean_object* v_snd_530_; lean_object* v___x_532_; uint8_t v_isShared_533_; uint8_t v_isSharedCheck_549_; 
v_fst_529_ = lean_ctor_get(v_a_525_, 0);
v_snd_530_ = lean_ctor_get(v_a_525_, 1);
v_isSharedCheck_549_ = !lean_is_exclusive(v_a_525_);
if (v_isSharedCheck_549_ == 0)
{
v___x_532_ = v_a_525_;
v_isShared_533_ = v_isSharedCheck_549_;
goto v_resetjp_531_;
}
else
{
lean_inc(v_snd_530_);
lean_inc(v_fst_529_);
lean_dec(v_a_525_);
v___x_532_ = lean_box(0);
v_isShared_533_ = v_isSharedCheck_549_;
goto v_resetjp_531_;
}
v_resetjp_531_:
{
lean_object* v_getValues_534_; lean_object* v_ofValues_535_; lean_object* v___f_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_544_; 
v_getValues_534_ = lean_ctor_get(v_inst_475_, 2);
lean_inc_ref(v_getValues_534_);
v_ofValues_535_ = lean_ctor_get(v_inst_475_, 3);
lean_inc(v_ofValues_535_);
lean_dec_ref(v_inst_475_);
lean_inc(v_fst_529_);
lean_inc(v_fst_518_);
v___f_536_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___lam__4___boxed), 5, 4);
lean_closure_set(v___f_536_, 0, v___x_496_);
lean_closure_set(v___f_536_, 1, v_fst_518_);
lean_closure_set(v___f_536_, 2, v_fst_529_);
lean_closure_set(v___f_536_, 3, v___x_496_);
lean_inc(v_snd_530_);
v___x_537_ = lean_apply_3(v_getValues_534_, v_n_473_, v_m_474_, v_snd_530_);
v___x_538_ = l_List_filterMapTR_go___redArg(v___f_536_, v___x_537_, v_free_497_);
v___x_539_ = lean_array_get_size(v_fst_518_);
v___x_540_ = lean_array_get_size(v_fst_529_);
v___x_541_ = lean_apply_3(v_ofValues_535_, v___x_539_, v___x_540_, v___x_538_);
v___x_542_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_542_, 0, v_fst_518_);
lean_ctor_set(v___x_542_, 1, v_fst_529_);
lean_ctor_set(v___x_542_, 2, v___x_541_);
if (v_isShared_533_ == 0)
{
lean_ctor_set(v___x_532_, 0, v___x_542_);
v___x_544_ = v___x_532_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_548_; 
v_reuseFailAlloc_548_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_548_, 0, v___x_542_);
lean_ctor_set(v_reuseFailAlloc_548_, 1, v_snd_530_);
v___x_544_ = v_reuseFailAlloc_548_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
lean_object* v___x_546_; 
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 0, v___x_544_);
v___x_546_ = v___x_527_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v___x_544_);
v___x_546_ = v_reuseFailAlloc_547_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
return v___x_546_;
}
}
}
}
}
else
{
lean_object* v_a_551_; lean_object* v___x_553_; uint8_t v_isShared_554_; uint8_t v_isSharedCheck_558_; 
lean_dec(v_fst_518_);
lean_dec_ref(v_inst_475_);
lean_dec(v_m_474_);
lean_dec(v_n_473_);
v_a_551_ = lean_ctor_get(v___x_524_, 0);
v_isSharedCheck_558_ = !lean_is_exclusive(v___x_524_);
if (v_isSharedCheck_558_ == 0)
{
v___x_553_ = v___x_524_;
v_isShared_554_ = v_isSharedCheck_558_;
goto v_resetjp_552_;
}
else
{
lean_inc(v_a_551_);
lean_dec(v___x_524_);
v___x_553_ = lean_box(0);
v_isShared_554_ = v_isSharedCheck_558_;
goto v_resetjp_552_;
}
v_resetjp_552_:
{
lean_object* v___x_556_; 
if (v_isShared_554_ == 0)
{
v___x_556_ = v___x_553_;
goto v_reusejp_555_;
}
else
{
lean_object* v_reuseFailAlloc_557_; 
v_reuseFailAlloc_557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_557_, 0, v_a_551_);
v___x_556_ = v_reuseFailAlloc_557_;
goto v_reusejp_555_;
}
v_reusejp_555_:
{
return v___x_556_;
}
}
}
}
else
{
lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_566_; 
lean_dec_ref_known(v___x_507_, 2);
lean_dec_ref(v_inst_475_);
lean_dec(v_m_474_);
lean_dec(v_n_473_);
v_a_559_ = lean_ctor_get(v___x_511_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_511_);
if (v_isSharedCheck_566_ == 0)
{
v___x_561_ = v___x_511_;
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_511_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_564_; 
if (v_isShared_562_ == 0)
{
v___x_564_ = v___x_561_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v_a_559_);
v___x_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
return v___x_564_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg___boxed(lean_object* v_n_567_, lean_object* v_m_568_, lean_object* v_inst_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg(v_n_567_, v_m_568_, v_inst_569_, v_a_570_, v_a_571_, v_a_572_);
lean_dec(v_a_572_);
lean_dec_ref(v_a_571_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp(lean_object* v_n_575_, lean_object* v_m_576_, lean_object* v_matType_577_, lean_object* v_inst_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_){
_start:
{
lean_object* v___x_583_; 
v___x_583_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg(v_n_575_, v_m_576_, v_inst_578_, v_a_579_, v_a_580_, v_a_581_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___boxed(lean_object* v_n_584_, lean_object* v_m_585_, lean_object* v_matType_586_, lean_object* v_inst_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp(v_n_584_, v_m_585_, v_matType_586_, v_inst_587_, v_a_588_, v_a_589_, v_a_590_);
lean_dec(v_a_590_);
lean_dec_ref(v_a_589_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg(lean_object* v_n_593_, lean_object* v_m_594_, lean_object* v_inst_595_, lean_object* v_A_596_, lean_object* v_a_597_, lean_object* v_a_598_){
_start:
{
lean_object* v___x_600_; 
v___x_600_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableauImp___redArg(v_n_593_, v_m_594_, v_inst_595_, v_A_596_, v_a_597_, v_a_598_);
if (lean_obj_tag(v___x_600_) == 0)
{
lean_object* v_a_601_; lean_object* v___x_603_; uint8_t v_isShared_604_; uint8_t v_isSharedCheck_609_; 
v_a_601_ = lean_ctor_get(v___x_600_, 0);
v_isSharedCheck_609_ = !lean_is_exclusive(v___x_600_);
if (v_isSharedCheck_609_ == 0)
{
v___x_603_ = v___x_600_;
v_isShared_604_ = v_isSharedCheck_609_;
goto v_resetjp_602_;
}
else
{
lean_inc(v_a_601_);
lean_dec(v___x_600_);
v___x_603_ = lean_box(0);
v_isShared_604_ = v_isSharedCheck_609_;
goto v_resetjp_602_;
}
v_resetjp_602_:
{
lean_object* v_fst_605_; lean_object* v___x_607_; 
v_fst_605_ = lean_ctor_get(v_a_601_, 0);
lean_inc(v_fst_605_);
lean_dec(v_a_601_);
if (v_isShared_604_ == 0)
{
lean_ctor_set(v___x_603_, 0, v_fst_605_);
v___x_607_ = v___x_603_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v_fst_605_);
v___x_607_ = v_reuseFailAlloc_608_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
return v___x_607_;
}
}
}
else
{
lean_object* v_a_610_; lean_object* v___x_612_; uint8_t v_isShared_613_; uint8_t v_isSharedCheck_617_; 
v_a_610_ = lean_ctor_get(v___x_600_, 0);
v_isSharedCheck_617_ = !lean_is_exclusive(v___x_600_);
if (v_isSharedCheck_617_ == 0)
{
v___x_612_ = v___x_600_;
v_isShared_613_ = v_isSharedCheck_617_;
goto v_resetjp_611_;
}
else
{
lean_inc(v_a_610_);
lean_dec(v___x_600_);
v___x_612_ = lean_box(0);
v_isShared_613_ = v_isSharedCheck_617_;
goto v_resetjp_611_;
}
v_resetjp_611_:
{
lean_object* v___x_615_; 
if (v_isShared_613_ == 0)
{
v___x_615_ = v___x_612_;
goto v_reusejp_614_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v_a_610_);
v___x_615_ = v_reuseFailAlloc_616_;
goto v_reusejp_614_;
}
v_reusejp_614_:
{
return v___x_615_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg___boxed(lean_object* v_n_618_, lean_object* v_m_619_, lean_object* v_inst_620_, lean_object* v_A_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_){
_start:
{
lean_object* v_res_625_; 
v_res_625_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg(v_n_618_, v_m_619_, v_inst_620_, v_A_621_, v_a_622_, v_a_623_);
lean_dec(v_a_623_);
lean_dec_ref(v_a_622_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau(lean_object* v_n_626_, lean_object* v_m_627_, lean_object* v_matType_628_, lean_object* v_inst_629_, lean_object* v_A_630_, lean_object* v_a_631_, lean_object* v_a_632_){
_start:
{
lean_object* v___x_634_; 
v___x_634_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg(v_n_626_, v_m_627_, v_inst_629_, v_A_630_, v_a_631_, v_a_632_);
return v___x_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___boxed(lean_object* v_n_635_, lean_object* v_m_636_, lean_object* v_matType_637_, lean_object* v_inst_638_, lean_object* v_A_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau(v_n_635_, v_m_636_, v_matType_637_, v_inst_638_, v_A_639_, v_a_640_, v_a_641_);
lean_dec(v_a_641_);
lean_dec_ref(v_a_640_);
return v_res_643_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(builtin);
}
#ifdef __cplusplus
}
#endif
