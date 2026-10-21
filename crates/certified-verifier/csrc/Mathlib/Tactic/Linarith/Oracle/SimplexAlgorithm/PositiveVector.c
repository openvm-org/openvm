// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Oracle.SimplexAlgorithm.PositiveVector
// Imports: public import Init public meta import Init public meta import Lean.Meta.Basic public import Mathlib.Tactic.Linarith.Oracle.SimplexAlgorithm.Gauss public import Mathlib.Tactic.Linarith.Oracle.SimplexAlgorithm.SimplexAlgorithm
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
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
lean_object* l_Rat_neg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* l_outOfBounds___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_instInhabitedRat;
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__7;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__2___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__13;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__14_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Simplex Algorithm failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__0(lean_object* v___x_1_, lean_object* v___x_2_, lean_object* v_idx_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_4_ = lean_unsigned_to_nat(2u);
v___x_5_ = lean_nat_add(v_idx_3_, v___x_4_);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_1_);
v___x_7_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_7_, 0, v___x_2_);
lean_ctor_set(v___x_7_, 1, v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__0___boxed(lean_object* v___x_8_, lean_object* v___x_9_, lean_object* v_idx_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__0(v___x_8_, v___x_9_, v_idx_10_);
lean_dec(v_idx_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__1(lean_object* v___x_12_, lean_object* v___x_13_, lean_object* v___x_14_, lean_object* v_i_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_16_ = lean_nat_add(v_i_15_, v___x_12_);
v___x_17_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
lean_ctor_set(v___x_17_, 1, v___x_13_);
v___x_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_18_, 0, v___x_14_);
lean_ctor_set(v___x_18_, 1, v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__1___boxed(lean_object* v___x_19_, lean_object* v___x_20_, lean_object* v___x_21_, lean_object* v_i_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__1(v___x_19_, v___x_20_, v___x_21_, v_i_22_);
lean_dec(v_i_22_);
lean_dec(v___x_19_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__2(lean_object* v___x_24_, lean_object* v_x_25_){
_start:
{
lean_object* v_snd_26_; lean_object* v_fst_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_45_; 
v_snd_26_ = lean_ctor_get(v_x_25_, 1);
v_fst_27_ = lean_ctor_get(v_x_25_, 0);
v_isSharedCheck_45_ = !lean_is_exclusive(v_x_25_);
if (v_isSharedCheck_45_ == 0)
{
v___x_29_ = v_x_25_;
v_isShared_30_ = v_isSharedCheck_45_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_snd_26_);
lean_inc(v_fst_27_);
lean_dec(v_x_25_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_45_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v_fst_31_; lean_object* v_snd_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_44_; 
v_fst_31_ = lean_ctor_get(v_snd_26_, 0);
v_snd_32_ = lean_ctor_get(v_snd_26_, 1);
v_isSharedCheck_44_ = !lean_is_exclusive(v_snd_26_);
if (v_isSharedCheck_44_ == 0)
{
v___x_34_ = v_snd_26_;
v_isShared_35_ = v_isSharedCheck_44_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_snd_32_);
lean_inc(v_fst_31_);
lean_dec(v_snd_26_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_44_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_39_; 
v___x_36_ = lean_nat_add(v_fst_27_, v___x_24_);
lean_dec(v_fst_27_);
v___x_37_ = lean_nat_add(v_fst_31_, v___x_24_);
lean_dec(v_fst_31_);
if (v_isShared_35_ == 0)
{
lean_ctor_set(v___x_34_, 0, v___x_37_);
v___x_39_ = v___x_34_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_43_; 
v_reuseFailAlloc_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_43_, 0, v___x_37_);
lean_ctor_set(v_reuseFailAlloc_43_, 1, v_snd_32_);
v___x_39_ = v_reuseFailAlloc_43_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
lean_object* v___x_41_; 
if (v_isShared_30_ == 0)
{
lean_ctor_set(v___x_29_, 1, v___x_39_);
lean_ctor_set(v___x_29_, 0, v___x_36_);
v___x_41_ = v___x_29_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_36_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v___x_39_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__2___boxed(lean_object* v___x_46_, lean_object* v_x_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__2(v___x_46_, v_x_47_);
lean_dec(v___x_46_);
return v_res_48_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = lean_unsigned_to_nat(1u);
v___x_50_ = l_Rat_instNatCast___lam__0(v___x_49_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__1(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___f_53_; 
v___x_51_ = lean_unsigned_to_nat(0u);
v___x_52_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0);
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_53_, 0, v___x_52_);
lean_closure_set(v___f_53_, 1, v___x_51_);
return v___f_53_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0);
v___x_55_ = l_Rat_neg(v___x_54_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__3(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_56_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2);
v___x_57_ = lean_unsigned_to_nat(0u);
v___x_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v___x_56_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__4(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_59_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0);
v___x_60_ = lean_unsigned_to_nat(1u);
v___x_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v___x_59_);
return v___x_61_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__5(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_62_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__4, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__4);
v___x_63_ = lean_unsigned_to_nat(1u);
v___x_64_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_64_, 0, v___x_63_);
lean_ctor_set(v___x_64_, 1, v___x_62_);
return v___x_64_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__6(void){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_65_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__3);
v___x_66_ = lean_unsigned_to_nat(0u);
v___x_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v___x_65_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__7(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___f_71_; 
v___x_68_ = lean_unsigned_to_nat(1u);
v___x_69_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__0);
v___x_70_ = lean_unsigned_to_nat(2u);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_71_, 0, v___x_70_);
lean_closure_set(v___f_71_, 1, v___x_69_);
lean_closure_set(v___f_71_, 2, v___x_68_);
return v___f_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg(lean_object* v_inst_74_, lean_object* v_n_75_, lean_object* v_m_76_, lean_object* v_A_77_, lean_object* v_strictIndexes_78_){
_start:
{
lean_object* v___x_79_; lean_object* v___f_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v_getValues_85_; lean_object* v_ofValues_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v_objectiveRow_93_; lean_object* v___f_94_; lean_object* v___f_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v_constraintRow_99_; lean_object* v___x_100_; lean_object* v_valuesA_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_79_ = lean_unsigned_to_nat(1u);
v___f_80_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__1);
v___x_81_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__2);
v___x_82_ = lean_box(0);
v___x_83_ = l_List_mapTR_loop___redArg(v___f_80_, v_strictIndexes_78_, v___x_82_);
v___x_84_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__5, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__5);
v_getValues_85_ = lean_ctor_get(v_inst_74_, 2);
lean_inc_ref(v_getValues_85_);
v_ofValues_86_ = lean_ctor_get(v_inst_74_, 3);
lean_inc(v_ofValues_86_);
lean_dec_ref(v_inst_74_);
v___x_87_ = lean_unsigned_to_nat(2u);
v___x_88_ = lean_nat_add(v_m_76_, v___x_87_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set(v___x_89_, 1, v___x_81_);
v___x_90_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_79_);
lean_ctor_set(v___x_90_, 1, v___x_89_);
v___x_91_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_82_);
v___x_92_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__6);
v_objectiveRow_93_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_objectiveRow_93_, 0, v___x_92_);
lean_ctor_set(v_objectiveRow_93_, 1, v___x_83_);
v___f_94_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__7, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__7);
v___f_95_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg___closed__8));
v___x_96_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_84_);
lean_ctor_set(v___x_96_, 1, v___x_91_);
lean_inc_n(v_m_76_, 2);
v___x_97_ = l_List_range(v_m_76_);
v___x_98_ = l_List_mapTR_loop___redArg(v___f_94_, v___x_97_, v___x_82_);
v_constraintRow_99_ = l_List_appendTR___redArg(v___x_96_, v___x_98_);
lean_inc(v_n_75_);
v___x_100_ = lean_apply_3(v_getValues_85_, v_n_75_, v_m_76_, v_A_77_);
v_valuesA_101_ = l_List_mapTR_loop___redArg(v___f_95_, v___x_100_, v___x_82_);
v___x_102_ = lean_nat_add(v_n_75_, v___x_87_);
lean_dec(v_n_75_);
v___x_103_ = lean_unsigned_to_nat(3u);
v___x_104_ = lean_nat_add(v_m_76_, v___x_103_);
lean_dec(v_m_76_);
v___x_105_ = l_List_appendTR___redArg(v_objectiveRow_93_, v_constraintRow_99_);
v___x_106_ = l_List_appendTR___redArg(v___x_105_, v_valuesA_101_);
v___x_107_ = lean_apply_3(v_ofValues_86_, v___x_102_, v___x_104_, v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP(lean_object* v_matType_108_, lean_object* v_inst_109_, lean_object* v_n_110_, lean_object* v_m_111_, lean_object* v_A_112_, lean_object* v_strictIndexes_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg(v_inst_109_, v_n_110_, v_m_111_, v_A_112_, v_strictIndexes_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___lam__0(lean_object* v_basic_115_, lean_object* v___x_116_, lean_object* v___x_117_, lean_object* v___x_118_, lean_object* v___x_119_, lean_object* v_inst_120_, lean_object* v_mat_121_, lean_object* v_i_122_, lean_object* v_h_123_, lean_object* v_____s_124_){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___y_129_; uint8_t v___x_134_; 
v___x_125_ = lean_array_fget_borrowed(v_basic_115_, v_i_122_);
v___x_126_ = lean_unsigned_to_nat(2u);
v___x_127_ = lean_nat_sub(v___x_125_, v___x_126_);
v___x_134_ = lean_nat_dec_lt(v_i_122_, v___x_117_);
if (v___x_134_ == 0)
{
lean_dec(v_i_122_);
lean_dec(v_mat_121_);
lean_dec_ref(v_inst_120_);
lean_dec(v___x_118_);
lean_dec(v___x_117_);
goto v___jp_132_;
}
else
{
lean_object* v___x_135_; uint8_t v___x_136_; 
v___x_135_ = lean_nat_sub(v___x_118_, v___x_119_);
v___x_136_ = lean_nat_dec_lt(v___x_135_, v___x_118_);
if (v___x_136_ == 0)
{
lean_dec(v___x_135_);
lean_dec(v_i_122_);
lean_dec(v_mat_121_);
lean_dec_ref(v_inst_120_);
lean_dec(v___x_118_);
lean_dec(v___x_117_);
goto v___jp_132_;
}
else
{
lean_object* v_getElem_137_; lean_object* v___x_138_; 
v_getElem_137_ = lean_ctor_get(v_inst_120_, 0);
lean_inc_ref(v_getElem_137_);
lean_dec_ref(v_inst_120_);
v___x_138_ = lean_apply_5(v_getElem_137_, v___x_117_, v___x_118_, v_mat_121_, v_i_122_, v___x_135_);
v___y_129_ = v___x_138_;
goto v___jp_128_;
}
}
v___jp_128_:
{
lean_object* v_ans_130_; lean_object* v___x_131_; 
v_ans_130_ = lean_array_set(v_____s_124_, v___x_127_, v___y_129_);
lean_dec(v___x_127_);
v___x_131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_131_, 0, v_ans_130_);
return v___x_131_;
}
v___jp_132_:
{
lean_object* v___x_133_; 
v___x_133_ = l_outOfBounds___redArg(v___x_116_);
v___y_129_ = v___x_133_;
goto v___jp_128_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___lam__0___boxed(lean_object* v_basic_139_, lean_object* v___x_140_, lean_object* v___x_141_, lean_object* v___x_142_, lean_object* v___x_143_, lean_object* v_inst_144_, lean_object* v_mat_145_, lean_object* v_i_146_, lean_object* v_h_147_, lean_object* v_____s_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___lam__0(v_basic_139_, v___x_140_, v___x_141_, v___x_142_, v___x_143_, v_inst_144_, v_mat_145_, v_i_146_, v_h_147_, v_____s_148_);
lean_dec(v___x_143_);
lean_dec_ref(v___x_140_);
lean_dec_ref(v_basic_139_);
return v_res_149_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__10(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_169_ = lean_unsigned_to_nat(0u);
v___x_170_ = l_Rat_instNatCast___lam__0(v___x_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg(lean_object* v_inst_171_, lean_object* v_tableau_172_){
_start:
{
lean_object* v___x_173_; lean_object* v_basic_174_; lean_object* v_free_175_; lean_object* v_mat_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_194_; 
v___x_173_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__9));
v_basic_174_ = lean_ctor_get(v_tableau_172_, 0);
v_free_175_ = lean_ctor_get(v_tableau_172_, 1);
v_mat_176_ = lean_ctor_get(v_tableau_172_, 2);
v_isSharedCheck_194_ = !lean_is_exclusive(v_tableau_172_);
if (v_isSharedCheck_194_ == 0)
{
v___x_178_ = v_tableau_172_;
v_isShared_179_ = v_isSharedCheck_194_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_mat_176_);
lean_inc(v_free_175_);
lean_inc(v_basic_174_);
lean_dec(v_tableau_172_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_194_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v_ans_187_; lean_object* v___x_188_; lean_object* v___f_189_; lean_object* v___x_191_; 
v___x_180_ = l_instInhabitedRat;
v___x_181_ = lean_array_get_size(v_basic_174_);
v___x_182_ = lean_array_get_size(v_free_175_);
lean_dec_ref(v_free_175_);
v___x_183_ = lean_nat_add(v___x_181_, v___x_182_);
v___x_184_ = lean_unsigned_to_nat(3u);
v___x_185_ = lean_nat_sub(v___x_183_, v___x_184_);
lean_dec(v___x_183_);
v___x_186_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__10, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___closed__10);
v_ans_187_ = lean_mk_array(v___x_185_, v___x_186_);
v___x_188_ = lean_unsigned_to_nat(1u);
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg___lam__0___boxed), 10, 7);
lean_closure_set(v___f_189_, 0, v_basic_174_);
lean_closure_set(v___f_189_, 1, v___x_180_);
lean_closure_set(v___f_189_, 2, v___x_181_);
lean_closure_set(v___f_189_, 3, v___x_182_);
lean_closure_set(v___f_189_, 4, v___x_188_);
lean_closure_set(v___f_189_, 5, v_inst_171_);
lean_closure_set(v___f_189_, 6, v_mat_176_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 2, v___x_188_);
lean_ctor_set(v___x_178_, 1, v___x_181_);
lean_ctor_set(v___x_178_, 0, v___x_188_);
v___x_191_ = v___x_178_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v___x_188_);
lean_ctor_set(v_reuseFailAlloc_193_, 1, v___x_181_);
lean_ctor_set(v_reuseFailAlloc_193_, 2, v___x_188_);
v___x_191_ = v_reuseFailAlloc_193_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
lean_object* v___x_192_; 
v___x_192_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_173_, v___x_191_, v___f_189_, v_ans_187_, v___x_188_, lean_box(0), lean_box(0));
return v___x_192_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution(lean_object* v_matType_195_, lean_object* v_inst_196_, lean_object* v_tableau_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg(v_inst_196_, v_tableau_197_);
return v___x_198_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__0(void){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = l_instMonadEIO(lean_box(0));
return v___x_199_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__1(void){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_200_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__0);
v___x_201_ = l_StateRefT_x27_instMonad___redArg(v___x_200_);
return v___x_201_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__8(void){
_start:
{
lean_object* v___x_208_; lean_object* v___f_209_; 
v___x_208_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_209_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_209_, 0, v___x_208_);
return v___f_209_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__9(void){
_start:
{
lean_object* v___x_210_; lean_object* v___f_211_; 
v___x_210_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_211_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_211_, 0, v___x_210_);
return v___f_211_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10(void){
_start:
{
lean_object* v___f_212_; lean_object* v___f_213_; lean_object* v___x_214_; 
v___f_212_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__9, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__9);
v___f_213_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__8, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__8);
v___x_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_214_, 0, v___f_213_);
lean_ctor_set(v___x_214_, 1, v___f_212_);
return v___x_214_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__11(void){
_start:
{
lean_object* v___x_215_; lean_object* v___f_216_; 
v___x_215_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10);
v___f_216_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_216_, 0, v___x_215_);
return v___f_216_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__12(void){
_start:
{
lean_object* v___x_217_; lean_object* v___f_218_; 
v___x_217_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__10);
v___f_218_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_218_, 0, v___x_217_);
return v___f_218_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__13(void){
_start:
{
lean_object* v___f_219_; lean_object* v___f_220_; lean_object* v___x_221_; 
v___f_219_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__12, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__12);
v___f_220_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__11, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__11);
v___x_221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_221_, 0, v___f_220_);
lean_ctor_set(v___x_221_, 1, v___f_219_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__16(void){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_224_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_225_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__7));
v___x_226_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__15));
v___x_227_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_226_, v___x_225_, v___x_224_);
return v___x_227_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__17(void){
_start:
{
lean_object* v___x_228_; lean_object* v___f_229_; lean_object* v___f_230_; lean_object* v___x_231_; 
v___x_228_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__16, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__16);
v___f_229_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__6));
v___f_230_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__14));
v___x_231_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_230_, v___f_229_, v___x_228_);
return v___x_231_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__19(void){
_start:
{
lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_233_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__18));
v___x_234_ = l_Lean_stringToMessageData(v___x_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg(lean_object* v_n_235_, lean_object* v_m_236_, lean_object* v_inst_237_, lean_object* v_A_238_, lean_object* v_strictIndexes_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v___x_245_; lean_object* v_toApplicative_246_; lean_object* v_toFunctor_247_; lean_object* v_toSeq_248_; lean_object* v_toSeqLeft_249_; lean_object* v_toSeqRight_250_; lean_object* v___f_251_; lean_object* v___f_252_; lean_object* v___f_253_; lean_object* v___f_254_; lean_object* v___x_255_; lean_object* v___f_256_; lean_object* v___f_257_; lean_object* v___f_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v_toApplicative_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_333_; 
v___x_245_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__1);
v_toApplicative_246_ = lean_ctor_get(v___x_245_, 0);
v_toFunctor_247_ = lean_ctor_get(v_toApplicative_246_, 0);
v_toSeq_248_ = lean_ctor_get(v_toApplicative_246_, 2);
v_toSeqLeft_249_ = lean_ctor_get(v_toApplicative_246_, 3);
v_toSeqRight_250_ = lean_ctor_get(v_toApplicative_246_, 4);
v___f_251_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__2));
v___f_252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_247_, 2);
v___f_253_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_253_, 0, v_toFunctor_247_);
v___f_254_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_254_, 0, v_toFunctor_247_);
v___x_255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_255_, 0, v___f_253_);
lean_ctor_set(v___x_255_, 1, v___f_254_);
lean_inc(v_toSeqRight_250_);
v___f_256_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_256_, 0, v_toSeqRight_250_);
lean_inc(v_toSeqLeft_249_);
v___f_257_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_257_, 0, v_toSeqLeft_249_);
lean_inc(v_toSeq_248_);
v___f_258_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_258_, 0, v_toSeq_248_);
v___x_259_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_259_, 0, v___x_255_);
lean_ctor_set(v___x_259_, 1, v___f_251_);
lean_ctor_set(v___x_259_, 2, v___f_258_);
lean_ctor_set(v___x_259_, 3, v___f_257_);
lean_ctor_set(v___x_259_, 4, v___f_256_);
v___x_260_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_259_);
lean_ctor_set(v___x_260_, 1, v___f_252_);
v___x_261_ = l_StateRefT_x27_instMonad___redArg(v___x_260_);
v_toApplicative_262_ = lean_ctor_get(v___x_261_, 0);
v_isSharedCheck_333_ = !lean_is_exclusive(v___x_261_);
if (v_isSharedCheck_333_ == 0)
{
lean_object* v_unused_334_; 
v_unused_334_ = lean_ctor_get(v___x_261_, 1);
lean_dec(v_unused_334_);
v___x_264_ = v___x_261_;
v_isShared_265_ = v_isSharedCheck_333_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_toApplicative_262_);
lean_dec(v___x_261_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_333_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v_toFunctor_266_; lean_object* v_toSeq_267_; lean_object* v_toSeqLeft_268_; lean_object* v_toSeqRight_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_331_; 
v_toFunctor_266_ = lean_ctor_get(v_toApplicative_262_, 0);
v_toSeq_267_ = lean_ctor_get(v_toApplicative_262_, 2);
v_toSeqLeft_268_ = lean_ctor_get(v_toApplicative_262_, 3);
v_toSeqRight_269_ = lean_ctor_get(v_toApplicative_262_, 4);
v_isSharedCheck_331_ = !lean_is_exclusive(v_toApplicative_262_);
if (v_isSharedCheck_331_ == 0)
{
lean_object* v_unused_332_; 
v_unused_332_ = lean_ctor_get(v_toApplicative_262_, 1);
lean_dec(v_unused_332_);
v___x_271_ = v_toApplicative_262_;
v_isShared_272_ = v_isSharedCheck_331_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_toSeqRight_269_);
lean_inc(v_toSeqLeft_268_);
lean_inc(v_toSeq_267_);
lean_inc(v_toFunctor_266_);
lean_dec(v_toApplicative_262_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_331_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
lean_object* v___f_273_; lean_object* v___f_274_; lean_object* v___f_275_; lean_object* v___f_276_; lean_object* v___x_277_; lean_object* v___f_278_; lean_object* v___f_279_; lean_object* v___f_280_; lean_object* v___x_282_; 
v___f_273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__4));
v___f_274_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__5));
lean_inc_ref(v_toFunctor_266_);
v___f_275_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_275_, 0, v_toFunctor_266_);
v___f_276_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_276_, 0, v_toFunctor_266_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v___f_275_);
lean_ctor_set(v___x_277_, 1, v___f_276_);
v___f_278_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_278_, 0, v_toSeqRight_269_);
v___f_279_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_279_, 0, v_toSeqLeft_268_);
v___f_280_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_280_, 0, v_toSeq_267_);
if (v_isShared_272_ == 0)
{
lean_ctor_set(v___x_271_, 4, v___f_278_);
lean_ctor_set(v___x_271_, 3, v___f_279_);
lean_ctor_set(v___x_271_, 2, v___f_280_);
lean_ctor_set(v___x_271_, 1, v___f_273_);
lean_ctor_set(v___x_271_, 0, v___x_277_);
v___x_282_ = v___x_271_;
goto v_reusejp_281_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v___x_277_);
lean_ctor_set(v_reuseFailAlloc_330_, 1, v___f_273_);
lean_ctor_set(v_reuseFailAlloc_330_, 2, v___f_280_);
lean_ctor_set(v_reuseFailAlloc_330_, 3, v___f_279_);
lean_ctor_set(v_reuseFailAlloc_330_, 4, v___f_278_);
v___x_282_ = v_reuseFailAlloc_330_;
goto v_reusejp_281_;
}
v_reusejp_281_:
{
lean_object* v___x_284_; 
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 1, v___f_274_);
lean_ctor_set(v___x_264_, 0, v___x_282_);
v___x_284_ = v___x_264_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v___x_282_);
lean_ctor_set(v_reuseFailAlloc_329_, 1, v___f_274_);
v___x_284_ = v_reuseFailAlloc_329_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
lean_object* v_B_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; 
lean_inc(v_m_236_);
lean_inc(v_n_235_);
lean_inc_ref_n(v_inst_237_, 2);
v_B_285_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_stateLP___redArg(v_inst_237_, v_n_235_, v_m_236_, v_A_238_, v_strictIndexes_239_);
v___x_286_ = lean_unsigned_to_nat(2u);
v___x_287_ = lean_nat_add(v_n_235_, v___x_286_);
lean_dec(v_n_235_);
v___x_288_ = lean_unsigned_to_nat(3u);
v___x_289_ = lean_nat_add(v_m_236_, v___x_288_);
lean_dec(v_m_236_);
v___x_290_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_Gauss_getTableau___redArg(v___x_287_, v___x_289_, v_inst_237_, v_B_285_, v_a_242_, v_a_243_);
if (lean_obj_tag(v___x_290_) == 0)
{
lean_object* v_a_291_; lean_object* v___x_292_; 
v_a_291_ = lean_ctor_get(v___x_290_, 0);
lean_inc(v_a_291_);
lean_dec_ref_known(v___x_290_, 1);
lean_inc_ref(v_inst_237_);
v___x_292_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg(v_inst_237_, v_a_291_, v_a_242_, v_a_243_);
if (lean_obj_tag(v___x_292_) == 0)
{
lean_object* v_a_293_; lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_312_; 
v_a_293_ = lean_ctor_get(v___x_292_, 0);
v_isSharedCheck_312_ = !lean_is_exclusive(v___x_292_);
if (v_isSharedCheck_312_ == 0)
{
v___x_295_ = v___x_292_;
v_isShared_296_ = v_isSharedCheck_312_;
goto v_resetjp_294_;
}
else
{
lean_inc(v_a_293_);
lean_dec(v___x_292_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_312_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
lean_object* v_fst_297_; 
v_fst_297_ = lean_ctor_get(v_a_293_, 0);
if (lean_obj_tag(v_fst_297_) == 0)
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v_toMonadRef_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_608__overap_305_; lean_object* v___x_306_; 
lean_del_object(v___x_295_);
lean_dec(v_a_293_);
lean_dec_ref(v_inst_237_);
v___x_298_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__13, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__13);
v___x_299_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__17, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__17);
v_toMonadRef_300_ = lean_ctor_get(v___x_299_, 0);
v___x_301_ = l_Lean_Meta_instAddMessageContextMetaM;
lean_inc_ref(v___x_284_);
v___x_302_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_301_, v___x_284_);
lean_inc_ref(v_toMonadRef_300_);
v___x_303_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_303_, 0, v___x_298_);
lean_ctor_set(v___x_303_, 1, v_toMonadRef_300_);
lean_ctor_set(v___x_303_, 2, v___x_302_);
v___x_304_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__19, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___closed__19);
v___x_608__overap_305_ = l_Lean_throwError___redArg(v___x_284_, v___x_303_, v___x_304_);
lean_inc(v_a_243_);
lean_inc_ref(v_a_242_);
lean_inc(v_a_241_);
lean_inc_ref(v_a_240_);
v___x_306_ = lean_apply_5(v___x_608__overap_305_, v_a_240_, v_a_241_, v_a_242_, v_a_243_, lean_box(0));
return v___x_306_;
}
else
{
lean_object* v_snd_307_; lean_object* v___x_308_; lean_object* v___x_310_; 
lean_dec_ref(v___x_284_);
v_snd_307_ = lean_ctor_get(v_a_293_, 1);
lean_inc(v_snd_307_);
lean_dec(v_a_293_);
v___x_308_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_extractSolution___redArg(v_inst_237_, v_snd_307_);
if (v_isShared_296_ == 0)
{
lean_ctor_set(v___x_295_, 0, v___x_308_);
v___x_310_ = v___x_295_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v___x_308_);
v___x_310_ = v_reuseFailAlloc_311_;
goto v_reusejp_309_;
}
v_reusejp_309_:
{
return v___x_310_;
}
}
}
}
else
{
lean_object* v_a_313_; lean_object* v___x_315_; uint8_t v_isShared_316_; uint8_t v_isSharedCheck_320_; 
lean_dec_ref(v___x_284_);
lean_dec_ref(v_inst_237_);
v_a_313_ = lean_ctor_get(v___x_292_, 0);
v_isSharedCheck_320_ = !lean_is_exclusive(v___x_292_);
if (v_isSharedCheck_320_ == 0)
{
v___x_315_ = v___x_292_;
v_isShared_316_ = v_isSharedCheck_320_;
goto v_resetjp_314_;
}
else
{
lean_inc(v_a_313_);
lean_dec(v___x_292_);
v___x_315_ = lean_box(0);
v_isShared_316_ = v_isSharedCheck_320_;
goto v_resetjp_314_;
}
v_resetjp_314_:
{
lean_object* v___x_318_; 
if (v_isShared_316_ == 0)
{
v___x_318_ = v___x_315_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v_a_313_);
v___x_318_ = v_reuseFailAlloc_319_;
goto v_reusejp_317_;
}
v_reusejp_317_:
{
return v___x_318_;
}
}
}
}
else
{
lean_object* v_a_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_328_; 
lean_dec_ref(v___x_284_);
lean_dec_ref(v_inst_237_);
v_a_321_ = lean_ctor_get(v___x_290_, 0);
v_isSharedCheck_328_ = !lean_is_exclusive(v___x_290_);
if (v_isSharedCheck_328_ == 0)
{
v___x_323_ = v___x_290_;
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_a_321_);
lean_dec(v___x_290_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v___x_326_; 
if (v_isShared_324_ == 0)
{
v___x_326_ = v___x_323_;
goto v_reusejp_325_;
}
else
{
lean_object* v_reuseFailAlloc_327_; 
v_reuseFailAlloc_327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_327_, 0, v_a_321_);
v___x_326_ = v_reuseFailAlloc_327_;
goto v_reusejp_325_;
}
v_reusejp_325_:
{
return v___x_326_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg___boxed(lean_object* v_n_335_, lean_object* v_m_336_, lean_object* v_inst_337_, lean_object* v_A_338_, lean_object* v_strictIndexes_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg(v_n_335_, v_m_336_, v_inst_337_, v_A_338_, v_strictIndexes_339_, v_a_340_, v_a_341_, v_a_342_, v_a_343_);
lean_dec(v_a_343_);
lean_dec_ref(v_a_342_);
lean_dec(v_a_341_);
lean_dec_ref(v_a_340_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector(lean_object* v_n_346_, lean_object* v_m_347_, lean_object* v_matType_348_, lean_object* v_inst_349_, lean_object* v_A_350_, lean_object* v_strictIndexes_351_, lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___redArg(v_n_346_, v_m_347_, v_inst_349_, v_A_350_, v_strictIndexes_351_, v_a_352_, v_a_353_, v_a_354_, v_a_355_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector___boxed(lean_object* v_n_358_, lean_object* v_m_359_, lean_object* v_matType_360_, lean_object* v_inst_361_, lean_object* v_A_362_, lean_object* v_strictIndexes_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_, lean_object* v_a_367_, lean_object* v_a_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_findPositiveVector(v_n_358_, v_m_359_, v_matType_360_, v_inst_361_, v_A_362_, v_strictIndexes_363_, v_a_364_, v_a_365_, v_a_366_, v_a_367_);
lean_dec(v_a_367_);
lean_dec_ref(v_a_366_);
lean_dec(v_a_365_);
lean_dec_ref(v_a_364_);
return v_res_369_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_PositiveVector(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_PositiveVector(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_PositiveVector(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Gauss(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_PositiveVector(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_PositiveVector(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_PositiveVector(builtin);
}
#ifdef __cplusplus
}
#endif
