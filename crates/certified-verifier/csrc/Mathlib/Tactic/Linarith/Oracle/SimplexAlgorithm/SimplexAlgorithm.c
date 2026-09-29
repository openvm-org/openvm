// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Oracle.SimplexAlgorithm.SimplexAlgorithm
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
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Rat_div(lean_object*, lean_object*);
uint8_t l_Rat_blt(lean_object*, lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_outOfBounds___redArg(lean_object*);
lean_object* l_Rat_neg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t l_Rat_instDecidableLe(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
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
lean_object* l_ExceptT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ExceptT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ExceptT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ExceptT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ExceptT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ExceptT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ExceptT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_instInhabitedRat;
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
lean_object* l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_panic___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Nat_Control_0__Nat_allM_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_While_0__repeatM_erased___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__11;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Linarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SimplexAlgorithm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "runSimplexAlgorithm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 101, 49, 147, 63, 179, 122, 68)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(90, 19, 252, 206, 85, 151, 142, 50)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(113, 205, 151, 167, 55, 201, 183, 180)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__0(lean_object* v___x_1_, lean_object* v___x_2_, lean_object* v_xs_3_, lean_object* v_i_4_){
_start:
{
lean_object* v_fst_5_; lean_object* v_snd_6_; uint8_t v___x_7_; 
v_fst_5_ = lean_ctor_get(v_i_4_, 0);
v_snd_6_ = lean_ctor_get(v_i_4_, 1);
v___x_7_ = lean_nat_dec_lt(v_fst_5_, v___x_1_);
if (v___x_7_ == 0)
{
return v___x_7_;
}
else
{
uint8_t v___x_8_; 
v___x_8_ = lean_nat_dec_lt(v_snd_6_, v___x_2_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__0___boxed(lean_object* v___x_9_, lean_object* v___x_10_, lean_object* v_xs_11_, lean_object* v_i_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__0(v___x_9_, v___x_10_, v_xs_11_, v_i_12_);
lean_dec_ref(v_i_12_);
lean_dec(v_xs_11_);
lean_dec(v___x_10_);
lean_dec(v___x_9_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__1(lean_object* v_exitIdx_15_, lean_object* v_setElem_16_, lean_object* v___x_17_, lean_object* v___x_18_, lean_object* v_enterIdx_19_, lean_object* v_subtractRow_20_, lean_object* v___y_21_, lean_object* v___x_22_, lean_object* v___f_23_, lean_object* v___x_24_, lean_object* v_getElem_25_, lean_object* v_a_26_, lean_object* v_x_27_, lean_object* v___y_28_){
_start:
{
lean_object* v___y_30_; lean_object* v_mat_31_; lean_object* v___y_35_; uint8_t v___x_37_; lean_object* v___y_39_; 
v___x_37_ = lean_nat_dec_eq(v_a_26_, v_exitIdx_15_);
if (v___x_37_ == 0)
{
lean_object* v___x_43_; lean_object* v___x_44_; uint8_t v___x_45_; 
lean_inc(v_enterIdx_19_);
lean_inc(v_a_26_);
v___x_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_43_, 0, v_a_26_);
lean_ctor_set(v___x_43_, 1, v_enterIdx_19_);
lean_inc(v___y_28_);
v___x_44_ = lean_apply_2(v___f_23_, v___y_28_, v___x_43_);
v___x_45_ = lean_unbox(v___x_44_);
if (v___x_45_ == 0)
{
lean_object* v___x_46_; 
lean_dec_ref(v_getElem_25_);
v___x_46_ = l_outOfBounds___redArg(v___x_24_);
v___y_39_ = v___x_46_;
goto v___jp_38_;
}
else
{
lean_object* v___x_47_; 
lean_inc(v_enterIdx_19_);
lean_inc(v_a_26_);
lean_inc(v___y_28_);
lean_inc(v___x_18_);
lean_inc(v___x_17_);
v___x_47_ = lean_apply_5(v_getElem_25_, v___x_17_, v___x_18_, v___y_28_, v_a_26_, v_enterIdx_19_);
v___y_39_ = v___x_47_;
goto v___jp_38_;
}
}
else
{
lean_object* v___x_48_; 
lean_dec(v_a_26_);
lean_dec_ref(v_getElem_25_);
lean_dec_ref(v___f_23_);
lean_dec(v___x_22_);
lean_dec_ref(v___y_21_);
lean_dec(v_subtractRow_20_);
lean_dec(v_enterIdx_19_);
lean_dec(v___x_18_);
lean_dec(v___x_17_);
lean_dec(v_setElem_16_);
lean_dec(v_exitIdx_15_);
v___x_48_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_48_, 0, v___y_28_);
return v___x_48_;
}
v___jp_29_:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_apply_6(v_setElem_16_, v___x_17_, v___x_18_, v_mat_31_, v_a_26_, v_enterIdx_19_, v___y_30_);
v___x_33_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
return v___x_33_;
}
v___jp_34_:
{
lean_object* v___x_36_; 
lean_inc_ref(v___y_35_);
lean_inc(v_a_26_);
lean_inc(v___x_18_);
lean_inc(v___x_17_);
v___x_36_ = lean_apply_6(v_subtractRow_20_, v___x_17_, v___x_18_, v___y_28_, v_exitIdx_15_, v_a_26_, v___y_35_);
v___y_30_ = v___y_35_;
v_mat_31_ = v___x_36_;
goto v___jp_29_;
}
v___jp_38_:
{
lean_object* v___x_40_; lean_object* v___x_41_; uint8_t v___x_42_; 
v___x_40_ = l_Rat_div(v___y_39_, v___y_21_);
lean_dec_ref(v___y_39_);
v___x_41_ = l_Rat_instNatCast___lam__0(v___x_22_);
v___x_42_ = l_instDecidableEqRat_decEq(v___x_40_, v___x_41_);
lean_dec_ref(v___x_41_);
if (v___x_42_ == 0)
{
v___y_35_ = v___x_40_;
goto v___jp_34_;
}
else
{
if (v___x_37_ == 0)
{
lean_dec(v_subtractRow_20_);
lean_dec(v_exitIdx_15_);
v___y_30_ = v___x_40_;
v_mat_31_ = v___y_28_;
goto v___jp_29_;
}
else
{
v___y_35_ = v___x_40_;
goto v___jp_34_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__1___boxed(lean_object* v_exitIdx_49_, lean_object* v_setElem_50_, lean_object* v___x_51_, lean_object* v___x_52_, lean_object* v_enterIdx_53_, lean_object* v_subtractRow_54_, lean_object* v___y_55_, lean_object* v___x_56_, lean_object* v___f_57_, lean_object* v___x_58_, lean_object* v_getElem_59_, lean_object* v_a_60_, lean_object* v_x_61_, lean_object* v___y_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__1(v_exitIdx_49_, v_setElem_50_, v___x_51_, v___x_52_, v_enterIdx_53_, v_subtractRow_54_, v___y_55_, v___x_56_, v___f_57_, v___x_58_, v_getElem_59_, v_a_60_, v_x_61_, v___y_62_);
lean_dec_ref(v___x_58_);
return v_res_63_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__10(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = lean_unsigned_to_nat(1u);
v___x_84_ = l_Rat_instNatCast___lam__0(v___x_83_);
return v___x_84_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__11(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__10, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__10);
v___x_86_ = l_Rat_neg(v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg(lean_object* v_inst_89_, lean_object* v_exitIdx_90_, lean_object* v_enterIdx_91_, lean_object* v_a_92_){
_start:
{
lean_object* v___x_94_; lean_object* v_basic_95_; lean_object* v_free_96_; lean_object* v_mat_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_135_; 
v___x_94_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__9));
v_basic_95_ = lean_ctor_get(v_a_92_, 0);
v_free_96_ = lean_ctor_get(v_a_92_, 1);
v_mat_97_ = lean_ctor_get(v_a_92_, 2);
v_isSharedCheck_135_ = !lean_is_exclusive(v_a_92_);
if (v_isSharedCheck_135_ == 0)
{
v___x_99_ = v_a_92_;
v_isShared_100_ = v_isSharedCheck_135_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_mat_97_);
lean_inc(v_free_96_);
lean_inc(v_basic_95_);
lean_dec(v_a_92_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_135_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___f_105_; lean_object* v___y_107_; lean_object* v___x_130_; uint8_t v___x_131_; 
v___x_101_ = l_instInhabitedRat;
v___x_102_ = lean_unsigned_to_nat(0u);
v___x_103_ = lean_array_get_size(v_basic_95_);
v___x_104_ = lean_array_get_size(v_free_96_);
v___f_105_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_105_, 0, v___x_103_);
lean_closure_set(v___f_105_, 1, v___x_104_);
lean_inc(v_enterIdx_91_);
lean_inc(v_exitIdx_90_);
v___x_130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_130_, 0, v_exitIdx_90_);
lean_ctor_set(v___x_130_, 1, v_enterIdx_91_);
v___x_131_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__0(v___x_103_, v___x_104_, v_mat_97_, v___x_130_);
lean_dec_ref_known(v___x_130_, 2);
if (v___x_131_ == 0)
{
lean_object* v___x_132_; 
v___x_132_ = l_outOfBounds___redArg(v___x_101_);
v___y_107_ = v___x_132_;
goto v___jp_106_;
}
else
{
lean_object* v_getElem_133_; lean_object* v___x_134_; 
v_getElem_133_ = lean_ctor_get(v_inst_89_, 0);
lean_inc_ref(v_getElem_133_);
lean_inc(v_enterIdx_91_);
lean_inc(v_exitIdx_90_);
lean_inc(v_mat_97_);
v___x_134_ = lean_apply_5(v_getElem_133_, v___x_103_, v___x_104_, v_mat_97_, v_exitIdx_90_, v_enterIdx_91_);
v___y_107_ = v___x_134_;
goto v___jp_106_;
}
v___jp_106_:
{
lean_object* v_getElem_108_; lean_object* v_setElem_109_; lean_object* v_subtractRow_110_; lean_object* v_divideRow_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___f_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_125_; 
v_getElem_108_ = lean_ctor_get(v_inst_89_, 0);
lean_inc_ref(v_getElem_108_);
v_setElem_109_ = lean_ctor_get(v_inst_89_, 1);
lean_inc_n(v_setElem_109_, 2);
v_subtractRow_110_ = lean_ctor_get(v_inst_89_, 5);
lean_inc(v_subtractRow_110_);
v_divideRow_111_ = lean_ctor_get(v_inst_89_, 6);
lean_inc(v_divideRow_111_);
lean_dec_ref(v_inst_89_);
v___x_112_ = lean_unsigned_to_nat(1u);
v___x_113_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_113_, 0, v___x_102_);
lean_ctor_set(v___x_113_, 1, v___x_103_);
lean_ctor_set(v___x_113_, 2, v___x_112_);
lean_inc_ref(v___y_107_);
lean_inc_n(v_enterIdx_91_, 2);
lean_inc_n(v_exitIdx_90_, 3);
v___f_114_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___lam__1___boxed), 14, 11);
lean_closure_set(v___f_114_, 0, v_exitIdx_90_);
lean_closure_set(v___f_114_, 1, v_setElem_109_);
lean_closure_set(v___f_114_, 2, v___x_103_);
lean_closure_set(v___f_114_, 3, v___x_104_);
lean_closure_set(v___f_114_, 4, v_enterIdx_91_);
lean_closure_set(v___f_114_, 5, v_subtractRow_110_);
lean_closure_set(v___f_114_, 6, v___y_107_);
lean_closure_set(v___f_114_, 7, v___x_102_);
lean_closure_set(v___f_114_, 8, v___f_105_);
lean_closure_set(v___f_114_, 9, v___x_101_);
lean_closure_set(v___f_114_, 10, v_getElem_108_);
v___x_115_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_94_, v___x_113_, v___f_114_, v_mat_97_, v___x_102_, lean_box(0), lean_box(0));
v___x_116_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__11, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__11);
v___x_117_ = lean_apply_6(v_setElem_109_, v___x_103_, v___x_104_, v___x_115_, v_exitIdx_90_, v_enterIdx_91_, v___x_116_);
v___x_118_ = l_Rat_neg(v___y_107_);
v___x_119_ = lean_apply_5(v_divideRow_111_, v___x_103_, v___x_104_, v___x_117_, v_exitIdx_90_, v___x_118_);
v___x_120_ = lean_array_get_borrowed(v___x_102_, v_free_96_, v_enterIdx_91_);
lean_inc(v___x_120_);
lean_inc_ref(v_basic_95_);
v___x_121_ = lean_array_set(v_basic_95_, v_exitIdx_90_, v___x_120_);
v___x_122_ = lean_array_get(v___x_102_, v_basic_95_, v_exitIdx_90_);
lean_dec(v_exitIdx_90_);
lean_dec_ref(v_basic_95_);
v___x_123_ = lean_array_set(v_free_96_, v_enterIdx_91_, v___x_122_);
lean_dec(v_enterIdx_91_);
if (v_isShared_100_ == 0)
{
lean_ctor_set(v___x_99_, 2, v___x_119_);
lean_ctor_set(v___x_99_, 1, v___x_123_);
lean_ctor_set(v___x_99_, 0, v___x_121_);
v___x_125_ = v___x_99_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v___x_121_);
lean_ctor_set(v_reuseFailAlloc_129_, 1, v___x_123_);
lean_ctor_set(v_reuseFailAlloc_129_, 2, v___x_119_);
v___x_125_ = v_reuseFailAlloc_129_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_126_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__12));
v___x_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v___x_125_);
v___x_128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
return v___x_128_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___boxed(lean_object* v_inst_136_, lean_object* v_exitIdx_137_, lean_object* v_enterIdx_138_, lean_object* v_a_139_, lean_object* v_a_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg(v_inst_136_, v_exitIdx_137_, v_enterIdx_138_, v_a_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation(lean_object* v_matType_142_, lean_object* v_inst_143_, lean_object* v_exitIdx_144_, lean_object* v_enterIdx_145_, lean_object* v_a_146_, lean_object* v_a_147_, lean_object* v_a_148_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg(v_inst_143_, v_exitIdx_144_, v_enterIdx_145_, v_a_146_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___boxed(lean_object* v_matType_151_, lean_object* v_inst_152_, lean_object* v_exitIdx_153_, lean_object* v_enterIdx_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation(v_matType_151_, v_inst_152_, v_exitIdx_153_, v_enterIdx_154_, v_a_155_, v_a_156_, v_a_157_);
lean_dec(v_a_157_);
lean_dec_ref(v_a_156_);
return v_res_159_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_160_ = lean_unsigned_to_nat(0u);
v___x_161_ = l_Rat_instNatCast___lam__0(v___x_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0(lean_object* v___x_162_, lean_object* v_lastIdx_163_, lean_object* v_inst_164_, lean_object* v_i_165_, lean_object* v_x_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
lean_object* v_basic_171_; lean_object* v_free_172_; lean_object* v_mat_173_; lean_object* v___x_174_; lean_object* v___y_176_; lean_object* v___x_184_; uint8_t v___x_185_; 
v_basic_171_ = lean_ctor_get(v___y_167_, 0);
v_free_172_ = lean_ctor_get(v___y_167_, 1);
v_mat_173_ = lean_ctor_get(v___y_167_, 2);
v___x_174_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0);
v___x_184_ = lean_array_get_size(v_basic_171_);
v___x_185_ = lean_nat_dec_lt(v_i_165_, v___x_184_);
if (v___x_185_ == 0)
{
lean_dec(v_i_165_);
lean_dec_ref(v_inst_164_);
lean_dec(v_lastIdx_163_);
goto v___jp_182_;
}
else
{
lean_object* v___x_186_; uint8_t v___x_187_; 
v___x_186_ = lean_array_get_size(v_free_172_);
v___x_187_ = lean_nat_dec_lt(v_lastIdx_163_, v___x_186_);
if (v___x_187_ == 0)
{
lean_dec(v_i_165_);
lean_dec_ref(v_inst_164_);
lean_dec(v_lastIdx_163_);
goto v___jp_182_;
}
else
{
lean_object* v_getElem_188_; lean_object* v___x_189_; 
v_getElem_188_ = lean_ctor_get(v_inst_164_, 0);
lean_inc_ref(v_getElem_188_);
lean_dec_ref(v_inst_164_);
lean_inc(v_mat_173_);
v___x_189_ = lean_apply_5(v_getElem_188_, v___x_184_, v___x_186_, v_mat_173_, v_i_165_, v_lastIdx_163_);
v___y_176_ = v___x_189_;
goto v___jp_175_;
}
}
v___jp_175_:
{
uint8_t v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_177_ = l_Rat_instDecidableLe(v___x_174_, v___y_176_);
v___x_178_ = lean_box(v___x_177_);
v___x_179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
v___x_180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
lean_ctor_set(v___x_180_, 1, v___y_167_);
v___x_181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
return v___x_181_;
}
v___jp_182_:
{
lean_object* v___x_183_; 
v___x_183_ = l_outOfBounds___redArg(v___x_162_);
v___y_176_ = v___x_183_;
goto v___jp_175_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___boxed(lean_object* v___x_190_, lean_object* v_lastIdx_191_, lean_object* v_inst_192_, lean_object* v_i_193_, lean_object* v_x_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0(v___x_190_, v_lastIdx_191_, v_inst_192_, v_i_193_, v_x_194_, v___y_195_, v___y_196_, v___y_197_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
lean_dec_ref(v___x_190_);
return v_res_199_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__0(void){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = l_instMonadEIO(lean_box(0));
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_201_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__0);
v___x_202_ = l_StateRefT_x27_instMonad___redArg(v___x_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg(lean_object* v_inst_205_, lean_object* v_a_206_, lean_object* v_a_207_, lean_object* v_a_208_){
_start:
{
lean_object* v___x_210_; lean_object* v_toApplicative_211_; lean_object* v_toFunctor_212_; lean_object* v_toSeq_213_; lean_object* v_toSeqLeft_214_; lean_object* v_toSeqRight_215_; lean_object* v___f_216_; lean_object* v___f_217_; lean_object* v___f_218_; lean_object* v___f_219_; lean_object* v___x_220_; lean_object* v___f_221_; lean_object* v___f_222_; lean_object* v___f_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___f_226_; lean_object* v___f_227_; lean_object* v___f_228_; lean_object* v___f_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___f_236_; lean_object* v___f_237_; lean_object* v___f_238_; lean_object* v___f_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v_basic_246_; lean_object* v_free_247_; lean_object* v_mat_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v_lastIdx_252_; lean_object* v___f_253_; lean_object* v___x_254_; lean_object* v___x_3863__overap_255_; lean_object* v___x_256_; 
v___x_210_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1);
v_toApplicative_211_ = lean_ctor_get(v___x_210_, 0);
v_toFunctor_212_ = lean_ctor_get(v_toApplicative_211_, 0);
v_toSeq_213_ = lean_ctor_get(v_toApplicative_211_, 2);
v_toSeqLeft_214_ = lean_ctor_get(v_toApplicative_211_, 3);
v_toSeqRight_215_ = lean_ctor_get(v_toApplicative_211_, 4);
v___f_216_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__2));
v___f_217_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_212_, 2);
v___f_218_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_218_, 0, v_toFunctor_212_);
v___f_219_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_219_, 0, v_toFunctor_212_);
v___x_220_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_220_, 0, v___f_218_);
lean_ctor_set(v___x_220_, 1, v___f_219_);
lean_inc(v_toSeqRight_215_);
v___f_221_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_221_, 0, v_toSeqRight_215_);
lean_inc(v_toSeqLeft_214_);
v___f_222_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_222_, 0, v_toSeqLeft_214_);
lean_inc(v_toSeq_213_);
v___f_223_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_223_, 0, v_toSeq_213_);
v___x_224_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_224_, 0, v___x_220_);
lean_ctor_set(v___x_224_, 1, v___f_216_);
lean_ctor_set(v___x_224_, 2, v___f_223_);
lean_ctor_set(v___x_224_, 3, v___f_222_);
lean_ctor_set(v___x_224_, 4, v___f_221_);
v___x_225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v___f_217_);
lean_inc_ref_n(v___x_225_, 6);
v___f_226_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_226_, 0, v___x_225_);
v___f_227_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_227_, 0, v___x_225_);
v___f_228_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_228_, 0, v___x_225_);
v___f_229_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_229_, 0, v___x_225_);
v___x_230_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_230_, 0, lean_box(0));
lean_closure_set(v___x_230_, 1, lean_box(0));
lean_closure_set(v___x_230_, 2, v___x_225_);
v___x_231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
lean_ctor_set(v___x_231_, 1, v___f_226_);
v___x_232_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_232_, 0, lean_box(0));
lean_closure_set(v___x_232_, 1, lean_box(0));
lean_closure_set(v___x_232_, 2, v___x_225_);
v___x_233_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_233_, 0, v___x_231_);
lean_ctor_set(v___x_233_, 1, v___x_232_);
lean_ctor_set(v___x_233_, 2, v___f_227_);
lean_ctor_set(v___x_233_, 3, v___f_228_);
lean_ctor_set(v___x_233_, 4, v___f_229_);
v___x_234_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_234_, 0, lean_box(0));
lean_closure_set(v___x_234_, 1, lean_box(0));
lean_closure_set(v___x_234_, 2, v___x_225_);
v___x_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_233_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
lean_inc_ref_n(v___x_235_, 6);
v___f_236_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_236_, 0, v___x_235_);
v___f_237_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__4), 5, 1);
lean_closure_set(v___f_237_, 0, v___x_235_);
v___f_238_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__7), 5, 1);
lean_closure_set(v___f_238_, 0, v___x_235_);
v___f_239_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__9), 5, 1);
lean_closure_set(v___f_239_, 0, v___x_235_);
v___x_240_ = lean_alloc_closure((void*)(l_ExceptT_map), 7, 3);
lean_closure_set(v___x_240_, 0, lean_box(0));
lean_closure_set(v___x_240_, 1, lean_box(0));
lean_closure_set(v___x_240_, 2, v___x_235_);
v___x_241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v___f_236_);
v___x_242_ = lean_alloc_closure((void*)(l_ExceptT_pure), 5, 3);
lean_closure_set(v___x_242_, 0, lean_box(0));
lean_closure_set(v___x_242_, 1, lean_box(0));
lean_closure_set(v___x_242_, 2, v___x_235_);
v___x_243_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_243_, 0, v___x_241_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
lean_ctor_set(v___x_243_, 2, v___f_237_);
lean_ctor_set(v___x_243_, 3, v___f_238_);
lean_ctor_set(v___x_243_, 4, v___f_239_);
v___x_244_ = lean_alloc_closure((void*)(l_ExceptT_bind), 7, 3);
lean_closure_set(v___x_244_, 0, lean_box(0));
lean_closure_set(v___x_244_, 1, lean_box(0));
lean_closure_set(v___x_244_, 2, v___x_235_);
v___x_245_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_243_);
lean_ctor_set(v___x_245_, 1, v___x_244_);
v_basic_246_ = lean_ctor_get(v_a_206_, 0);
v_free_247_ = lean_ctor_get(v_a_206_, 1);
v_mat_248_ = lean_ctor_get(v_a_206_, 2);
lean_inc(v_mat_248_);
v___x_249_ = l_instInhabitedRat;
v___x_250_ = lean_array_get_size(v_free_247_);
v___x_251_ = lean_unsigned_to_nat(1u);
v_lastIdx_252_ = lean_nat_sub(v___x_250_, v___x_251_);
lean_inc_ref(v_inst_205_);
lean_inc(v_lastIdx_252_);
v___f_253_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_253_, 0, v___x_249_);
lean_closure_set(v___f_253_, 1, v_lastIdx_252_);
lean_closure_set(v___f_253_, 2, v_inst_205_);
v___x_254_ = lean_array_get_size(v_basic_246_);
v___x_3863__overap_255_ = l___private_Init_Data_Nat_Control_0__Nat_allM_loop(lean_box(0), v___x_245_, v___x_254_, v___f_253_, v___x_254_, lean_box(0));
lean_inc(v_a_208_);
lean_inc_ref(v_a_207_);
v___x_256_ = lean_apply_4(v___x_3863__overap_255_, v_a_206_, v_a_207_, v_a_208_, lean_box(0));
if (lean_obj_tag(v___x_256_) == 0)
{
lean_object* v_a_257_; lean_object* v_fst_258_; lean_object* v_snd_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_284_; 
v_a_257_ = lean_ctor_get(v___x_256_, 0);
lean_inc(v_a_257_);
v_fst_258_ = lean_ctor_get(v_a_257_, 0);
v_snd_259_ = lean_ctor_get(v_a_257_, 1);
v_isSharedCheck_284_ = !lean_is_exclusive(v_a_257_);
if (v_isSharedCheck_284_ == 0)
{
v___x_261_ = v_a_257_;
v_isShared_262_ = v_isSharedCheck_284_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_snd_259_);
lean_inc(v_fst_258_);
lean_dec(v_a_257_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_284_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
uint8_t v___y_264_; 
if (lean_obj_tag(v_fst_258_) == 0)
{
lean_dec_ref_known(v_fst_258_, 1);
lean_del_object(v___x_261_);
lean_dec(v_snd_259_);
lean_dec(v_lastIdx_252_);
lean_dec(v_mat_248_);
lean_dec_ref(v_inst_205_);
return v___x_256_;
}
else
{
lean_object* v_a_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___y_275_; uint8_t v___x_280_; 
lean_dec_ref_known(v___x_256_, 1);
v_a_271_ = lean_ctor_get(v_fst_258_, 0);
lean_inc(v_a_271_);
lean_dec_ref_known(v_fst_258_, 1);
v___x_272_ = lean_unsigned_to_nat(0u);
v___x_273_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0);
v___x_280_ = lean_nat_dec_lt(v___x_272_, v___x_254_);
if (v___x_280_ == 0)
{
lean_dec(v_lastIdx_252_);
lean_dec(v_mat_248_);
lean_dec_ref(v_inst_205_);
goto v___jp_278_;
}
else
{
uint8_t v___x_281_; 
v___x_281_ = lean_nat_dec_lt(v_lastIdx_252_, v___x_250_);
if (v___x_281_ == 0)
{
lean_dec(v_lastIdx_252_);
lean_dec(v_mat_248_);
lean_dec_ref(v_inst_205_);
goto v___jp_278_;
}
else
{
lean_object* v_getElem_282_; lean_object* v___x_283_; 
v_getElem_282_ = lean_ctor_get(v_inst_205_, 0);
lean_inc_ref(v_getElem_282_);
lean_dec_ref(v_inst_205_);
v___x_283_ = lean_apply_5(v_getElem_282_, v___x_254_, v___x_250_, v_mat_248_, v___x_272_, v_lastIdx_252_);
v___y_275_ = v___x_283_;
goto v___jp_274_;
}
}
v___jp_274_:
{
uint8_t v___x_276_; 
v___x_276_ = l_Rat_blt(v___x_273_, v___y_275_);
if (v___x_276_ == 0)
{
lean_dec(v_a_271_);
v___y_264_ = v___x_276_;
goto v___jp_263_;
}
else
{
uint8_t v___x_277_; 
v___x_277_ = lean_unbox(v_a_271_);
lean_dec(v_a_271_);
v___y_264_ = v___x_277_;
goto v___jp_263_;
}
}
v___jp_278_:
{
lean_object* v___x_279_; 
v___x_279_ = l_outOfBounds___redArg(v___x_249_);
v___y_275_ = v___x_279_;
goto v___jp_274_;
}
}
v___jp_263_:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_268_; 
v___x_265_ = lean_box(v___y_264_);
v___x_266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_266_, 0, v___x_265_);
if (v_isShared_262_ == 0)
{
lean_ctor_set(v___x_261_, 0, v___x_266_);
v___x_268_ = v___x_261_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v___x_266_);
lean_ctor_set(v_reuseFailAlloc_270_, 1, v_snd_259_);
v___x_268_ = v_reuseFailAlloc_270_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
lean_object* v___x_269_; 
v___x_269_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_269_, 0, v___x_268_);
return v___x_269_;
}
}
}
}
else
{
lean_dec(v_lastIdx_252_);
lean_dec(v_mat_248_);
lean_dec_ref(v_inst_205_);
return v___x_256_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___boxed(lean_object* v_inst_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg(v_inst_285_, v_a_286_, v_a_287_, v_a_288_);
lean_dec(v_a_288_);
lean_dec_ref(v_a_287_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess(lean_object* v_matType_291_, lean_object* v_inst_292_, lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg(v_inst_292_, v_a_293_, v_a_294_, v_a_295_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___boxed(lean_object* v_matType_298_, lean_object* v_inst_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess(v_matType_298_, v_inst_299_, v_a_300_, v_a_301_, v_a_302_);
lean_dec(v_a_302_);
lean_dec_ref(v_a_301_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___lam__0(lean_object* v___x_305_, lean_object* v_minIdx_306_, lean_object* v___x_307_, lean_object* v_inst_308_, lean_object* v_a_309_, lean_object* v_x_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v_fst_316_; lean_object* v_snd_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_357_; 
v_fst_316_ = lean_ctor_get(v___y_311_, 0);
v_snd_317_ = lean_ctor_get(v___y_311_, 1);
v_isSharedCheck_357_ = !lean_is_exclusive(v___y_311_);
if (v_isSharedCheck_357_ == 0)
{
v___x_319_ = v___y_311_;
v_isShared_320_ = v_isSharedCheck_357_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_snd_317_);
lean_inc(v_fst_316_);
lean_dec(v___y_311_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_357_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v_basic_321_; lean_object* v_free_322_; lean_object* v_mat_323_; uint8_t v___y_325_; lean_object* v___x_345_; lean_object* v___y_347_; lean_object* v___x_351_; uint8_t v___x_352_; 
v_basic_321_ = lean_ctor_get(v___y_312_, 0);
v_free_322_ = lean_ctor_get(v___y_312_, 1);
v_mat_323_ = lean_ctor_get(v___y_312_, 2);
lean_inc(v_minIdx_306_);
v___x_345_ = l_Rat_instNatCast___lam__0(v_minIdx_306_);
v___x_351_ = lean_array_get_size(v_basic_321_);
v___x_352_ = lean_nat_dec_lt(v_minIdx_306_, v___x_351_);
if (v___x_352_ == 0)
{
lean_dec_ref(v_inst_308_);
lean_dec(v_minIdx_306_);
goto v___jp_349_;
}
else
{
lean_object* v___x_353_; uint8_t v___x_354_; 
v___x_353_ = lean_array_get_size(v_free_322_);
v___x_354_ = lean_nat_dec_lt(v_a_309_, v___x_353_);
if (v___x_354_ == 0)
{
lean_dec_ref(v_inst_308_);
lean_dec(v_minIdx_306_);
goto v___jp_349_;
}
else
{
lean_object* v_getElem_355_; lean_object* v___x_356_; 
v_getElem_355_ = lean_ctor_get(v_inst_308_, 0);
lean_inc_ref(v_getElem_355_);
lean_dec_ref(v_inst_308_);
lean_inc(v_a_309_);
lean_inc(v_mat_323_);
v___x_356_ = lean_apply_5(v_getElem_355_, v___x_351_, v___x_353_, v_mat_323_, v_minIdx_306_, v_a_309_);
v___y_347_ = v___x_356_;
goto v___jp_346_;
}
}
v___jp_324_:
{
if (v___y_325_ == 0)
{
lean_object* v___x_327_; 
lean_dec(v_a_309_);
if (v_isShared_320_ == 0)
{
v___x_327_ = v___x_319_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v_fst_316_);
lean_ctor_set(v_reuseFailAlloc_332_, 1, v_snd_317_);
v___x_327_ = v_reuseFailAlloc_332_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_328_, 0, v___x_327_);
v___x_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
v___x_330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
lean_ctor_set(v___x_330_, 1, v___y_312_);
v___x_331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
return v___x_331_;
}
}
else
{
lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_336_; 
lean_dec(v_snd_317_);
lean_dec(v_fst_316_);
lean_inc(v_a_309_);
v___x_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_333_, 0, v_a_309_);
v___x_334_ = lean_array_get_borrowed(v___x_305_, v_free_322_, v_a_309_);
lean_dec(v_a_309_);
lean_inc(v___x_334_);
if (v_isShared_320_ == 0)
{
lean_ctor_set(v___x_319_, 1, v___x_334_);
lean_ctor_set(v___x_319_, 0, v___x_333_);
v___x_336_ = v___x_319_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v___x_333_);
lean_ctor_set(v_reuseFailAlloc_341_, 1, v___x_334_);
v___x_336_ = v_reuseFailAlloc_341_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_337_, 0, v___x_336_);
v___x_338_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
v___x_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_338_);
lean_ctor_set(v___x_339_, 1, v___y_312_);
v___x_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
return v___x_340_;
}
}
}
v___jp_342_:
{
lean_object* v___x_343_; uint8_t v___x_344_; 
v___x_343_ = lean_array_get_borrowed(v___x_305_, v_free_322_, v_a_309_);
v___x_344_ = lean_nat_dec_lt(v___x_343_, v_snd_317_);
v___y_325_ = v___x_344_;
goto v___jp_324_;
}
v___jp_346_:
{
uint8_t v___x_348_; 
v___x_348_ = l_Rat_blt(v___x_345_, v___y_347_);
if (v___x_348_ == 0)
{
v___y_325_ = v___x_348_;
goto v___jp_324_;
}
else
{
if (lean_obj_tag(v_fst_316_) == 0)
{
if (v___x_348_ == 0)
{
goto v___jp_342_;
}
else
{
v___y_325_ = v___x_348_;
goto v___jp_324_;
}
}
else
{
goto v___jp_342_;
}
}
}
v___jp_349_:
{
lean_object* v___x_350_; 
v___x_350_ = l_outOfBounds___redArg(v___x_307_);
v___y_347_ = v___x_350_;
goto v___jp_346_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___lam__0___boxed(lean_object* v___x_358_, lean_object* v_minIdx_359_, lean_object* v___x_360_, lean_object* v_inst_361_, lean_object* v_a_362_, lean_object* v_x_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___lam__0(v___x_358_, v_minIdx_359_, v___x_360_, v_inst_361_, v_a_362_, v_x_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_);
lean_dec(v___y_367_);
lean_dec_ref(v___y_366_);
lean_dec_ref(v___x_360_);
lean_dec(v___x_358_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg(lean_object* v_inst_375_, lean_object* v_a_376_, lean_object* v_a_377_, lean_object* v_a_378_){
_start:
{
lean_object* v___x_380_; lean_object* v_toApplicative_381_; lean_object* v_toFunctor_382_; lean_object* v_toSeq_383_; lean_object* v_toSeqLeft_384_; lean_object* v_toSeqRight_385_; lean_object* v___f_386_; lean_object* v___f_387_; lean_object* v___f_388_; lean_object* v___f_389_; lean_object* v___x_390_; lean_object* v___f_391_; lean_object* v___f_392_; lean_object* v___f_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___f_396_; lean_object* v___f_397_; lean_object* v___f_398_; lean_object* v___f_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___f_406_; lean_object* v___f_407_; lean_object* v___f_408_; lean_object* v___f_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v_free_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___f_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_4109__overap_425_; lean_object* v___x_426_; 
v___x_380_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1);
v_toApplicative_381_ = lean_ctor_get(v___x_380_, 0);
v_toFunctor_382_ = lean_ctor_get(v_toApplicative_381_, 0);
v_toSeq_383_ = lean_ctor_get(v_toApplicative_381_, 2);
v_toSeqLeft_384_ = lean_ctor_get(v_toApplicative_381_, 3);
v_toSeqRight_385_ = lean_ctor_get(v_toApplicative_381_, 4);
v___f_386_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__2));
v___f_387_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_382_, 2);
v___f_388_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_388_, 0, v_toFunctor_382_);
v___f_389_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_389_, 0, v_toFunctor_382_);
v___x_390_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_390_, 0, v___f_388_);
lean_ctor_set(v___x_390_, 1, v___f_389_);
lean_inc(v_toSeqRight_385_);
v___f_391_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_391_, 0, v_toSeqRight_385_);
lean_inc(v_toSeqLeft_384_);
v___f_392_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_392_, 0, v_toSeqLeft_384_);
lean_inc(v_toSeq_383_);
v___f_393_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_393_, 0, v_toSeq_383_);
v___x_394_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_394_, 0, v___x_390_);
lean_ctor_set(v___x_394_, 1, v___f_386_);
lean_ctor_set(v___x_394_, 2, v___f_393_);
lean_ctor_set(v___x_394_, 3, v___f_392_);
lean_ctor_set(v___x_394_, 4, v___f_391_);
v___x_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_394_);
lean_ctor_set(v___x_395_, 1, v___f_387_);
lean_inc_ref_n(v___x_395_, 6);
v___f_396_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_396_, 0, v___x_395_);
v___f_397_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_397_, 0, v___x_395_);
v___f_398_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_398_, 0, v___x_395_);
v___f_399_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_399_, 0, v___x_395_);
v___x_400_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_400_, 0, lean_box(0));
lean_closure_set(v___x_400_, 1, lean_box(0));
lean_closure_set(v___x_400_, 2, v___x_395_);
v___x_401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_401_, 0, v___x_400_);
lean_ctor_set(v___x_401_, 1, v___f_396_);
v___x_402_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_402_, 0, lean_box(0));
lean_closure_set(v___x_402_, 1, lean_box(0));
lean_closure_set(v___x_402_, 2, v___x_395_);
v___x_403_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_403_, 0, v___x_401_);
lean_ctor_set(v___x_403_, 1, v___x_402_);
lean_ctor_set(v___x_403_, 2, v___f_397_);
lean_ctor_set(v___x_403_, 3, v___f_398_);
lean_ctor_set(v___x_403_, 4, v___f_399_);
v___x_404_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_404_, 0, lean_box(0));
lean_closure_set(v___x_404_, 1, lean_box(0));
lean_closure_set(v___x_404_, 2, v___x_395_);
v___x_405_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_403_);
lean_ctor_set(v___x_405_, 1, v___x_404_);
lean_inc_ref_n(v___x_405_, 6);
v___f_406_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_406_, 0, v___x_405_);
v___f_407_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__4), 5, 1);
lean_closure_set(v___f_407_, 0, v___x_405_);
v___f_408_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__7), 5, 1);
lean_closure_set(v___f_408_, 0, v___x_405_);
v___f_409_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__9), 5, 1);
lean_closure_set(v___f_409_, 0, v___x_405_);
v___x_410_ = lean_alloc_closure((void*)(l_ExceptT_map), 7, 3);
lean_closure_set(v___x_410_, 0, lean_box(0));
lean_closure_set(v___x_410_, 1, lean_box(0));
lean_closure_set(v___x_410_, 2, v___x_405_);
v___x_411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_410_);
lean_ctor_set(v___x_411_, 1, v___f_406_);
v___x_412_ = lean_alloc_closure((void*)(l_ExceptT_pure), 5, 3);
lean_closure_set(v___x_412_, 0, lean_box(0));
lean_closure_set(v___x_412_, 1, lean_box(0));
lean_closure_set(v___x_412_, 2, v___x_405_);
v___x_413_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_413_, 0, v___x_411_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
lean_ctor_set(v___x_413_, 2, v___f_407_);
lean_ctor_set(v___x_413_, 3, v___f_408_);
lean_ctor_set(v___x_413_, 4, v___f_409_);
v___x_414_ = lean_alloc_closure((void*)(l_ExceptT_bind), 7, 3);
lean_closure_set(v___x_414_, 0, lean_box(0));
lean_closure_set(v___x_414_, 1, lean_box(0));
lean_closure_set(v___x_414_, 2, v___x_405_);
v___x_415_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_413_);
lean_ctor_set(v___x_415_, 1, v___x_414_);
v_free_416_ = lean_ctor_get(v_a_376_, 1);
v___x_417_ = lean_unsigned_to_nat(0u);
v___x_418_ = l_instInhabitedRat;
v___f_419_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___lam__0___boxed), 11, 4);
lean_closure_set(v___f_419_, 0, v___x_417_);
lean_closure_set(v___f_419_, 1, v___x_417_);
lean_closure_set(v___f_419_, 2, v___x_418_);
lean_closure_set(v___f_419_, 3, v_inst_375_);
v___x_420_ = lean_array_get_size(v_free_416_);
v___x_421_ = lean_unsigned_to_nat(1u);
v___x_422_ = lean_nat_sub(v___x_420_, v___x_421_);
v___x_423_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_423_, 0, v___x_417_);
lean_ctor_set(v___x_423_, 1, v___x_422_);
lean_ctor_set(v___x_423_, 2, v___x_421_);
v___x_424_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__0));
v___x_4109__overap_425_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_415_, v___x_423_, v___f_419_, v___x_424_, v___x_417_, lean_box(0), lean_box(0));
lean_inc(v_a_378_);
lean_inc_ref(v_a_377_);
v___x_426_ = lean_apply_4(v___x_4109__overap_425_, v_a_376_, v_a_377_, v_a_378_, lean_box(0));
if (lean_obj_tag(v___x_426_) == 0)
{
lean_object* v_a_427_; lean_object* v___x_429_; uint8_t v_isShared_430_; uint8_t v_isSharedCheck_482_; 
v_a_427_ = lean_ctor_get(v___x_426_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_426_);
if (v_isSharedCheck_482_ == 0)
{
v___x_429_ = v___x_426_;
v_isShared_430_ = v_isSharedCheck_482_;
goto v_resetjp_428_;
}
else
{
lean_inc(v_a_427_);
lean_dec(v___x_426_);
v___x_429_ = lean_box(0);
v_isShared_430_ = v_isSharedCheck_482_;
goto v_resetjp_428_;
}
v_resetjp_428_:
{
lean_object* v_fst_431_; 
v_fst_431_ = lean_ctor_get(v_a_427_, 0);
lean_inc(v_fst_431_);
if (lean_obj_tag(v_fst_431_) == 0)
{
lean_object* v_snd_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_450_; 
v_snd_432_ = lean_ctor_get(v_a_427_, 1);
v_isSharedCheck_450_ = !lean_is_exclusive(v_a_427_);
if (v_isSharedCheck_450_ == 0)
{
lean_object* v_unused_451_; 
v_unused_451_ = lean_ctor_get(v_a_427_, 0);
lean_dec(v_unused_451_);
v___x_434_ = v_a_427_;
v_isShared_435_ = v_isSharedCheck_450_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_snd_432_);
lean_dec(v_a_427_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_450_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_449_; 
v_a_436_ = lean_ctor_get(v_fst_431_, 0);
v_isSharedCheck_449_ = !lean_is_exclusive(v_fst_431_);
if (v_isSharedCheck_449_ == 0)
{
v___x_438_ = v_fst_431_;
v_isShared_439_ = v_isSharedCheck_449_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v_fst_431_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_449_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_441_; 
if (v_isShared_439_ == 0)
{
v___x_441_ = v___x_438_;
goto v_reusejp_440_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_a_436_);
v___x_441_ = v_reuseFailAlloc_448_;
goto v_reusejp_440_;
}
v_reusejp_440_:
{
lean_object* v___x_443_; 
if (v_isShared_435_ == 0)
{
lean_ctor_set(v___x_434_, 0, v___x_441_);
v___x_443_ = v___x_434_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_447_; 
v_reuseFailAlloc_447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_447_, 0, v___x_441_);
lean_ctor_set(v_reuseFailAlloc_447_, 1, v_snd_432_);
v___x_443_ = v_reuseFailAlloc_447_;
goto v_reusejp_442_;
}
v_reusejp_442_:
{
lean_object* v___x_445_; 
if (v_isShared_430_ == 0)
{
lean_ctor_set(v___x_429_, 0, v___x_443_);
v___x_445_ = v___x_429_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v___x_443_);
v___x_445_ = v_reuseFailAlloc_446_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
return v___x_445_;
}
}
}
}
}
}
else
{
lean_object* v_a_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_481_; 
v_a_452_ = lean_ctor_get(v_fst_431_, 0);
v_isSharedCheck_481_ = !lean_is_exclusive(v_fst_431_);
if (v_isSharedCheck_481_ == 0)
{
v___x_454_ = v_fst_431_;
v_isShared_455_ = v_isSharedCheck_481_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_a_452_);
lean_dec(v_fst_431_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_481_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v_fst_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_479_; 
v_fst_456_ = lean_ctor_get(v_a_452_, 0);
v_isSharedCheck_479_ = !lean_is_exclusive(v_a_452_);
if (v_isSharedCheck_479_ == 0)
{
lean_object* v_unused_480_; 
v_unused_480_ = lean_ctor_get(v_a_452_, 1);
lean_dec(v_unused_480_);
v___x_458_ = v_a_452_;
v_isShared_459_ = v_isSharedCheck_479_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_fst_456_);
lean_dec(v_a_452_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_479_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
if (lean_obj_tag(v_fst_456_) == 0)
{
lean_object* v_snd_460_; lean_object* v___x_461_; lean_object* v___x_463_; 
lean_del_object(v___x_454_);
v_snd_460_ = lean_ctor_get(v_a_427_, 1);
lean_inc(v_snd_460_);
lean_dec(v_a_427_);
v___x_461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___closed__1));
if (v_isShared_459_ == 0)
{
lean_ctor_set(v___x_458_, 1, v_snd_460_);
lean_ctor_set(v___x_458_, 0, v___x_461_);
v___x_463_ = v___x_458_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v___x_461_);
lean_ctor_set(v_reuseFailAlloc_467_, 1, v_snd_460_);
v___x_463_ = v_reuseFailAlloc_467_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
lean_object* v___x_465_; 
if (v_isShared_430_ == 0)
{
lean_ctor_set(v___x_429_, 0, v___x_463_);
v___x_465_ = v___x_429_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_466_; 
v_reuseFailAlloc_466_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_466_, 0, v___x_463_);
v___x_465_ = v_reuseFailAlloc_466_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
return v___x_465_;
}
}
}
else
{
lean_object* v_snd_468_; lean_object* v_val_469_; lean_object* v___x_471_; 
v_snd_468_ = lean_ctor_get(v_a_427_, 1);
lean_inc(v_snd_468_);
lean_dec(v_a_427_);
v_val_469_ = lean_ctor_get(v_fst_456_, 0);
lean_inc(v_val_469_);
lean_dec_ref_known(v_fst_456_, 1);
if (v_isShared_455_ == 0)
{
lean_ctor_set(v___x_454_, 0, v_val_469_);
v___x_471_ = v___x_454_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_478_; 
v_reuseFailAlloc_478_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_478_, 0, v_val_469_);
v___x_471_ = v_reuseFailAlloc_478_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
lean_object* v___x_473_; 
if (v_isShared_459_ == 0)
{
lean_ctor_set(v___x_458_, 1, v_snd_468_);
lean_ctor_set(v___x_458_, 0, v___x_471_);
v___x_473_ = v___x_458_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v___x_471_);
lean_ctor_set(v_reuseFailAlloc_477_, 1, v_snd_468_);
v___x_473_ = v_reuseFailAlloc_477_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
lean_object* v___x_475_; 
if (v_isShared_430_ == 0)
{
lean_ctor_set(v___x_429_, 0, v___x_473_);
v___x_475_ = v___x_429_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_476_; 
v_reuseFailAlloc_476_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_476_, 0, v___x_473_);
v___x_475_ = v_reuseFailAlloc_476_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
return v___x_475_;
}
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_490_; 
v_a_483_ = lean_ctor_get(v___x_426_, 0);
v_isSharedCheck_490_ = !lean_is_exclusive(v___x_426_);
if (v_isSharedCheck_490_ == 0)
{
v___x_485_ = v___x_426_;
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_a_483_);
lean_dec(v___x_426_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___x_488_; 
if (v_isShared_486_ == 0)
{
v___x_488_ = v___x_485_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v_a_483_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg___boxed(lean_object* v_inst_491_, lean_object* v_a_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg(v_inst_491_, v_a_492_, v_a_493_, v_a_494_);
lean_dec(v_a_494_);
lean_dec_ref(v_a_493_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar(lean_object* v_matType_497_, lean_object* v_inst_498_, lean_object* v_a_499_, lean_object* v_a_500_, lean_object* v_a_501_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg(v_inst_498_, v_a_499_, v_a_500_, v_a_501_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___boxed(lean_object* v_matType_504_, lean_object* v_inst_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_, lean_object* v_a_509_){
_start:
{
lean_object* v_res_510_; 
v_res_510_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar(v_matType_504_, v_inst_505_, v_a_506_, v_a_507_, v_a_508_);
lean_dec(v_a_508_);
lean_dec_ref(v_a_507_);
return v_res_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___lam__0(lean_object* v___x_511_, lean_object* v___x_512_, lean_object* v_enterIdx_513_, lean_object* v_inst_514_, lean_object* v_minCoef_515_, lean_object* v___x_516_, lean_object* v_a_517_, lean_object* v_x_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_){
_start:
{
lean_object* v_snd_524_; lean_object* v_fst_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_607_; 
v_snd_524_ = lean_ctor_get(v___y_519_, 1);
v_fst_525_ = lean_ctor_get(v___y_519_, 0);
v_isSharedCheck_607_ = !lean_is_exclusive(v___y_519_);
if (v_isSharedCheck_607_ == 0)
{
v___x_527_ = v___y_519_;
v_isShared_528_ = v_isSharedCheck_607_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_snd_524_);
lean_inc(v_fst_525_);
lean_dec(v___y_519_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_607_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v_fst_529_; lean_object* v_snd_530_; lean_object* v___x_532_; uint8_t v_isShared_533_; uint8_t v_isSharedCheck_606_; 
v_fst_529_ = lean_ctor_get(v_snd_524_, 0);
v_snd_530_ = lean_ctor_get(v_snd_524_, 1);
v_isSharedCheck_606_ = !lean_is_exclusive(v_snd_524_);
if (v_isSharedCheck_606_ == 0)
{
v___x_532_ = v_snd_524_;
v_isShared_533_ = v_isSharedCheck_606_;
goto v_resetjp_531_;
}
else
{
lean_inc(v_snd_530_);
lean_inc(v_fst_529_);
lean_dec(v_snd_524_);
v___x_532_ = lean_box(0);
v_isShared_533_ = v_isSharedCheck_606_;
goto v_resetjp_531_;
}
v_resetjp_531_:
{
lean_object* v_basic_534_; lean_object* v_free_535_; lean_object* v_mat_536_; lean_object* v___y_538_; lean_object* v___y_552_; uint8_t v___y_553_; lean_object* v___y_561_; lean_object* v___y_562_; lean_object* v___y_569_; lean_object* v___y_572_; lean_object* v___y_583_; lean_object* v___x_600_; uint8_t v___x_601_; 
v_basic_534_ = lean_ctor_get(v___y_520_, 0);
v_free_535_ = lean_ctor_get(v___y_520_, 1);
v_mat_536_ = lean_ctor_get(v___y_520_, 2);
v___x_600_ = lean_array_get_size(v_basic_534_);
v___x_601_ = lean_nat_dec_lt(v_a_517_, v___x_600_);
if (v___x_601_ == 0)
{
goto v___jp_598_;
}
else
{
lean_object* v___x_602_; uint8_t v___x_603_; 
v___x_602_ = lean_array_get_size(v_free_535_);
v___x_603_ = lean_nat_dec_lt(v_enterIdx_513_, v___x_602_);
if (v___x_603_ == 0)
{
goto v___jp_598_;
}
else
{
lean_object* v_getElem_604_; lean_object* v___x_605_; 
v_getElem_604_ = lean_ctor_get(v_inst_514_, 0);
lean_inc_ref(v_getElem_604_);
lean_inc(v_enterIdx_513_);
lean_inc(v_a_517_);
lean_inc(v_mat_536_);
v___x_605_ = lean_apply_5(v_getElem_604_, v___x_600_, v___x_602_, v_mat_536_, v_a_517_, v_enterIdx_513_);
v___y_583_ = v___x_605_;
goto v___jp_582_;
}
}
v___jp_537_:
{
lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_542_; 
lean_inc(v_a_517_);
v___x_539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_539_, 0, v_a_517_);
v___x_540_ = lean_array_get_borrowed(v___x_511_, v_basic_534_, v_a_517_);
lean_dec(v_a_517_);
lean_inc(v___x_540_);
if (v_isShared_533_ == 0)
{
lean_ctor_set(v___x_532_, 1, v___x_540_);
lean_ctor_set(v___x_532_, 0, v___y_538_);
v___x_542_ = v___x_532_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v___y_538_);
lean_ctor_set(v_reuseFailAlloc_550_, 1, v___x_540_);
v___x_542_ = v_reuseFailAlloc_550_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
lean_object* v___x_544_; 
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 1, v___x_542_);
lean_ctor_set(v___x_527_, 0, v___x_539_);
v___x_544_ = v___x_527_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v___x_539_);
lean_ctor_set(v_reuseFailAlloc_549_, 1, v___x_542_);
v___x_544_ = v_reuseFailAlloc_549_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_545_, 0, v___x_544_);
v___x_546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_546_, 0, v___x_545_);
v___x_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_547_, 0, v___x_546_);
lean_ctor_set(v___x_547_, 1, v___y_520_);
v___x_548_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_548_, 0, v___x_547_);
return v___x_548_;
}
}
}
v___jp_551_:
{
if (v___y_553_ == 0)
{
lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; 
lean_dec_ref(v___y_552_);
lean_del_object(v___x_532_);
lean_del_object(v___x_527_);
lean_dec(v_a_517_);
v___x_554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_554_, 0, v_fst_529_);
lean_ctor_set(v___x_554_, 1, v_snd_530_);
v___x_555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_555_, 0, v_fst_525_);
lean_ctor_set(v___x_555_, 1, v___x_554_);
v___x_556_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_556_, 0, v___x_555_);
v___x_557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_557_, 0, v___x_556_);
v___x_558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_557_);
lean_ctor_set(v___x_558_, 1, v___y_520_);
v___x_559_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_559_, 0, v___x_558_);
return v___x_559_;
}
else
{
lean_dec(v_snd_530_);
lean_dec(v_fst_529_);
lean_dec(v_fst_525_);
v___y_538_ = v___y_552_;
goto v___jp_537_;
}
}
v___jp_560_:
{
lean_object* v___x_563_; 
v___x_563_ = l_Rat_div(v___y_561_, v___y_562_);
lean_dec_ref(v___y_561_);
if (lean_obj_tag(v_fst_525_) == 0)
{
lean_dec(v_snd_530_);
lean_dec(v_fst_529_);
v___y_538_ = v___x_563_;
goto v___jp_537_;
}
else
{
uint8_t v___x_564_; 
lean_inc(v_fst_529_);
lean_inc_ref(v___x_563_);
v___x_564_ = l_Rat_blt(v___x_563_, v_fst_529_);
if (v___x_564_ == 0)
{
uint8_t v___x_565_; 
v___x_565_ = l_instDecidableEqRat_decEq(v___x_563_, v_fst_529_);
if (v___x_565_ == 0)
{
v___y_552_ = v___x_563_;
v___y_553_ = v___x_565_;
goto v___jp_551_;
}
else
{
lean_object* v___x_566_; uint8_t v___x_567_; 
v___x_566_ = lean_array_get_borrowed(v___x_511_, v_basic_534_, v_a_517_);
v___x_567_ = lean_nat_dec_lt(v___x_566_, v_snd_530_);
v___y_552_ = v___x_563_;
v___y_553_ = v___x_567_;
goto v___jp_551_;
}
}
else
{
lean_dec_ref_known(v_fst_525_, 1);
lean_dec(v_snd_530_);
lean_dec(v_fst_529_);
v___y_538_ = v___x_563_;
goto v___jp_537_;
}
}
}
v___jp_568_:
{
lean_object* v___x_570_; 
v___x_570_ = l_outOfBounds___redArg(v___x_512_);
v___y_561_ = v___y_569_;
v___y_562_ = v___x_570_;
goto v___jp_560_;
}
v___jp_571_:
{
lean_object* v___x_573_; lean_object* v___x_574_; uint8_t v___x_575_; 
v___x_573_ = l_Rat_neg(v___y_572_);
v___x_574_ = lean_array_get_size(v_basic_534_);
v___x_575_ = lean_nat_dec_lt(v_a_517_, v___x_574_);
if (v___x_575_ == 0)
{
lean_dec_ref(v_inst_514_);
lean_dec(v_enterIdx_513_);
v___y_569_ = v___x_573_;
goto v___jp_568_;
}
else
{
lean_object* v___x_576_; uint8_t v___x_577_; 
v___x_576_ = lean_array_get_size(v_free_535_);
v___x_577_ = lean_nat_dec_lt(v_enterIdx_513_, v___x_576_);
if (v___x_577_ == 0)
{
lean_dec_ref(v_inst_514_);
lean_dec(v_enterIdx_513_);
v___y_569_ = v___x_573_;
goto v___jp_568_;
}
else
{
lean_object* v_getElem_578_; lean_object* v___x_579_; 
v_getElem_578_ = lean_ctor_get(v_inst_514_, 0);
lean_inc_ref(v_getElem_578_);
lean_dec_ref(v_inst_514_);
lean_inc(v_a_517_);
lean_inc(v_mat_536_);
v___x_579_ = lean_apply_5(v_getElem_578_, v___x_574_, v___x_576_, v_mat_536_, v_a_517_, v_enterIdx_513_);
v___y_561_ = v___x_573_;
v___y_562_ = v___x_579_;
goto v___jp_560_;
}
}
}
v___jp_580_:
{
lean_object* v___x_581_; 
v___x_581_ = l_outOfBounds___redArg(v___x_512_);
v___y_572_ = v___x_581_;
goto v___jp_571_;
}
v___jp_582_:
{
uint8_t v___x_584_; 
v___x_584_ = l_Rat_instDecidableLe(v_minCoef_515_, v___y_583_);
if (v___x_584_ == 0)
{
lean_object* v___x_585_; uint8_t v___x_586_; 
v___x_585_ = lean_array_get_size(v_basic_534_);
v___x_586_ = lean_nat_dec_lt(v_a_517_, v___x_585_);
if (v___x_586_ == 0)
{
goto v___jp_580_;
}
else
{
lean_object* v___x_587_; lean_object* v___x_588_; uint8_t v___x_589_; 
v___x_587_ = lean_array_get_size(v_free_535_);
v___x_588_ = lean_nat_sub(v___x_587_, v___x_516_);
v___x_589_ = lean_nat_dec_lt(v___x_588_, v___x_587_);
if (v___x_589_ == 0)
{
lean_dec(v___x_588_);
goto v___jp_580_;
}
else
{
lean_object* v_getElem_590_; lean_object* v___x_591_; 
v_getElem_590_ = lean_ctor_get(v_inst_514_, 0);
lean_inc_ref(v_getElem_590_);
lean_inc(v_a_517_);
lean_inc(v_mat_536_);
v___x_591_ = lean_apply_5(v_getElem_590_, v___x_585_, v___x_587_, v_mat_536_, v_a_517_, v___x_588_);
v___y_572_ = v___x_591_;
goto v___jp_571_;
}
}
}
else
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; 
lean_del_object(v___x_532_);
lean_del_object(v___x_527_);
lean_dec(v_a_517_);
lean_dec_ref(v_inst_514_);
lean_dec(v_enterIdx_513_);
v___x_592_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_592_, 0, v_fst_529_);
lean_ctor_set(v___x_592_, 1, v_snd_530_);
v___x_593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_593_, 0, v_fst_525_);
lean_ctor_set(v___x_593_, 1, v___x_592_);
v___x_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_594_, 0, v___x_593_);
v___x_595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_595_, 0, v___x_594_);
v___x_596_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_595_);
lean_ctor_set(v___x_596_, 1, v___y_520_);
v___x_597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_597_, 0, v___x_596_);
return v___x_597_;
}
}
v___jp_598_:
{
lean_object* v___x_599_; 
v___x_599_ = l_outOfBounds___redArg(v___x_512_);
v___y_583_ = v___x_599_;
goto v___jp_582_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___lam__0___boxed(lean_object* v___x_608_, lean_object* v___x_609_, lean_object* v_enterIdx_610_, lean_object* v_inst_611_, lean_object* v_minCoef_612_, lean_object* v___x_613_, lean_object* v_a_614_, lean_object* v_x_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___lam__0(v___x_608_, v___x_609_, v_enterIdx_610_, v_inst_611_, v_minCoef_612_, v___x_613_, v_a_614_, v_x_615_, v___y_616_, v___y_617_, v___y_618_, v___y_619_);
lean_dec(v___y_619_);
lean_dec_ref(v___y_618_);
lean_dec(v___x_613_);
lean_dec_ref(v___x_609_);
lean_dec(v___x_608_);
return v_res_621_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__0(void){
_start:
{
lean_object* v___x_622_; lean_object* v_minCoef_623_; lean_object* v___x_624_; 
v___x_622_ = lean_unsigned_to_nat(0u);
v_minCoef_623_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0);
v___x_624_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_624_, 0, v_minCoef_623_);
lean_ctor_set(v___x_624_, 1, v___x_622_);
return v___x_624_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__1(void){
_start:
{
lean_object* v___x_625_; lean_object* v_exitIdxOpt_626_; lean_object* v___x_627_; 
v___x_625_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__0);
v_exitIdxOpt_626_ = lean_box(0);
v___x_627_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_627_, 0, v_exitIdxOpt_626_);
lean_ctor_set(v___x_627_, 1, v___x_625_);
return v___x_627_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__5(void){
_start:
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; 
v___x_631_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__4));
v___x_632_ = lean_unsigned_to_nat(14u);
v___x_633_ = lean_unsigned_to_nat(22u);
v___x_634_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__3));
v___x_635_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__2));
v___x_636_ = l_mkPanicMessageWithDecl(v___x_635_, v___x_634_, v___x_633_, v___x_632_, v___x_631_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg(lean_object* v_inst_637_, lean_object* v_enterIdx_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_){
_start:
{
lean_object* v___x_643_; lean_object* v_toApplicative_644_; lean_object* v_toFunctor_645_; lean_object* v_toSeq_646_; lean_object* v_toSeqLeft_647_; lean_object* v_toSeqRight_648_; lean_object* v___f_649_; lean_object* v___f_650_; lean_object* v___f_651_; lean_object* v___f_652_; lean_object* v___x_653_; lean_object* v___f_654_; lean_object* v___f_655_; lean_object* v___f_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___f_659_; lean_object* v___f_660_; lean_object* v___f_661_; lean_object* v___f_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___f_669_; lean_object* v___f_670_; lean_object* v___f_671_; lean_object* v___f_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v_basic_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v_minCoef_682_; lean_object* v___x_683_; lean_object* v___f_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_6573__overap_688_; lean_object* v___x_689_; 
v___x_643_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1);
v_toApplicative_644_ = lean_ctor_get(v___x_643_, 0);
v_toFunctor_645_ = lean_ctor_get(v_toApplicative_644_, 0);
v_toSeq_646_ = lean_ctor_get(v_toApplicative_644_, 2);
v_toSeqLeft_647_ = lean_ctor_get(v_toApplicative_644_, 3);
v_toSeqRight_648_ = lean_ctor_get(v_toApplicative_644_, 4);
v___f_649_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__2));
v___f_650_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_645_, 2);
v___f_651_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_651_, 0, v_toFunctor_645_);
v___f_652_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_652_, 0, v_toFunctor_645_);
v___x_653_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_653_, 0, v___f_651_);
lean_ctor_set(v___x_653_, 1, v___f_652_);
lean_inc(v_toSeqRight_648_);
v___f_654_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_654_, 0, v_toSeqRight_648_);
lean_inc(v_toSeqLeft_647_);
v___f_655_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_655_, 0, v_toSeqLeft_647_);
lean_inc(v_toSeq_646_);
v___f_656_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_656_, 0, v_toSeq_646_);
v___x_657_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_657_, 0, v___x_653_);
lean_ctor_set(v___x_657_, 1, v___f_649_);
lean_ctor_set(v___x_657_, 2, v___f_656_);
lean_ctor_set(v___x_657_, 3, v___f_655_);
lean_ctor_set(v___x_657_, 4, v___f_654_);
v___x_658_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_658_, 0, v___x_657_);
lean_ctor_set(v___x_658_, 1, v___f_650_);
lean_inc_ref_n(v___x_658_, 6);
v___f_659_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_659_, 0, v___x_658_);
v___f_660_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_660_, 0, v___x_658_);
v___f_661_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_661_, 0, v___x_658_);
v___f_662_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_662_, 0, v___x_658_);
v___x_663_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_663_, 0, lean_box(0));
lean_closure_set(v___x_663_, 1, lean_box(0));
lean_closure_set(v___x_663_, 2, v___x_658_);
v___x_664_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_664_, 0, v___x_663_);
lean_ctor_set(v___x_664_, 1, v___f_659_);
v___x_665_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_665_, 0, lean_box(0));
lean_closure_set(v___x_665_, 1, lean_box(0));
lean_closure_set(v___x_665_, 2, v___x_658_);
v___x_666_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_666_, 0, v___x_664_);
lean_ctor_set(v___x_666_, 1, v___x_665_);
lean_ctor_set(v___x_666_, 2, v___f_660_);
lean_ctor_set(v___x_666_, 3, v___f_661_);
lean_ctor_set(v___x_666_, 4, v___f_662_);
v___x_667_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_667_, 0, lean_box(0));
lean_closure_set(v___x_667_, 1, lean_box(0));
lean_closure_set(v___x_667_, 2, v___x_658_);
v___x_668_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_668_, 0, v___x_666_);
lean_ctor_set(v___x_668_, 1, v___x_667_);
lean_inc_ref_n(v___x_668_, 6);
v___f_669_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_669_, 0, v___x_668_);
v___f_670_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__4), 5, 1);
lean_closure_set(v___f_670_, 0, v___x_668_);
v___f_671_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__7), 5, 1);
lean_closure_set(v___f_671_, 0, v___x_668_);
v___f_672_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__9), 5, 1);
lean_closure_set(v___f_672_, 0, v___x_668_);
v___x_673_ = lean_alloc_closure((void*)(l_ExceptT_map), 7, 3);
lean_closure_set(v___x_673_, 0, lean_box(0));
lean_closure_set(v___x_673_, 1, lean_box(0));
lean_closure_set(v___x_673_, 2, v___x_668_);
v___x_674_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_674_, 0, v___x_673_);
lean_ctor_set(v___x_674_, 1, v___f_669_);
v___x_675_ = lean_alloc_closure((void*)(l_ExceptT_pure), 5, 3);
lean_closure_set(v___x_675_, 0, lean_box(0));
lean_closure_set(v___x_675_, 1, lean_box(0));
lean_closure_set(v___x_675_, 2, v___x_668_);
v___x_676_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_676_, 0, v___x_674_);
lean_ctor_set(v___x_676_, 1, v___x_675_);
lean_ctor_set(v___x_676_, 2, v___f_670_);
lean_ctor_set(v___x_676_, 3, v___f_671_);
lean_ctor_set(v___x_676_, 4, v___f_672_);
v___x_677_ = lean_alloc_closure((void*)(l_ExceptT_bind), 7, 3);
lean_closure_set(v___x_677_, 0, lean_box(0));
lean_closure_set(v___x_677_, 1, lean_box(0));
lean_closure_set(v___x_677_, 2, v___x_668_);
v___x_678_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_678_, 0, v___x_676_);
lean_ctor_set(v___x_678_, 1, v___x_677_);
v_basic_679_ = lean_ctor_get(v_a_639_, 0);
v___x_680_ = l_instInhabitedRat;
v___x_681_ = lean_unsigned_to_nat(0u);
v_minCoef_682_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___lam__0___closed__0);
v___x_683_ = lean_unsigned_to_nat(1u);
v___f_684_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___lam__0___boxed), 13, 6);
lean_closure_set(v___f_684_, 0, v___x_681_);
lean_closure_set(v___f_684_, 1, v___x_680_);
lean_closure_set(v___f_684_, 2, v_enterIdx_638_);
lean_closure_set(v___f_684_, 3, v_inst_637_);
lean_closure_set(v___f_684_, 4, v_minCoef_682_);
lean_closure_set(v___f_684_, 5, v___x_683_);
v___x_685_ = lean_array_get_size(v_basic_679_);
v___x_686_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_686_, 0, v___x_683_);
lean_ctor_set(v___x_686_, 1, v___x_685_);
lean_ctor_set(v___x_686_, 2, v___x_683_);
v___x_687_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__1);
v___x_6573__overap_688_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_678_, v___x_686_, v___f_684_, v___x_687_, v___x_683_, lean_box(0), lean_box(0));
lean_inc(v_a_641_);
lean_inc_ref(v_a_640_);
v___x_689_ = lean_apply_4(v___x_6573__overap_688_, v_a_639_, v_a_640_, v_a_641_, lean_box(0));
if (lean_obj_tag(v___x_689_) == 0)
{
lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_724_; 
v_a_690_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_724_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_724_ == 0)
{
v___x_692_ = v___x_689_;
v_isShared_693_ = v_isSharedCheck_724_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_689_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_724_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v_fst_694_; lean_object* v_snd_695_; lean_object* v___x_697_; uint8_t v_isShared_698_; uint8_t v_isSharedCheck_723_; 
v_fst_694_ = lean_ctor_get(v_a_690_, 0);
v_snd_695_ = lean_ctor_get(v_a_690_, 1);
v_isSharedCheck_723_ = !lean_is_exclusive(v_a_690_);
if (v_isSharedCheck_723_ == 0)
{
v___x_697_ = v_a_690_;
v_isShared_698_ = v_isSharedCheck_723_;
goto v_resetjp_696_;
}
else
{
lean_inc(v_snd_695_);
lean_inc(v_fst_694_);
lean_dec(v_a_690_);
v___x_697_ = lean_box(0);
v_isShared_698_ = v_isSharedCheck_723_;
goto v_resetjp_696_;
}
v_resetjp_696_:
{
lean_object* v___y_700_; 
if (lean_obj_tag(v_fst_694_) == 0)
{
lean_object* v_a_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_717_; 
lean_del_object(v___x_697_);
lean_del_object(v___x_692_);
v_a_708_ = lean_ctor_get(v_fst_694_, 0);
v_isSharedCheck_717_ = !lean_is_exclusive(v_fst_694_);
if (v_isSharedCheck_717_ == 0)
{
v___x_710_ = v_fst_694_;
v_isShared_711_ = v_isSharedCheck_717_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_a_708_);
lean_dec(v_fst_694_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_717_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_713_; 
if (v_isShared_711_ == 0)
{
v___x_713_ = v___x_710_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v_a_708_);
v___x_713_ = v_reuseFailAlloc_716_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
lean_object* v___x_714_; lean_object* v___x_715_; 
v___x_714_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_714_, 0, v___x_713_);
lean_ctor_set(v___x_714_, 1, v_snd_695_);
v___x_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_715_, 0, v___x_714_);
return v___x_715_;
}
}
}
else
{
lean_object* v_a_718_; lean_object* v_fst_719_; 
v_a_718_ = lean_ctor_get(v_fst_694_, 0);
lean_inc(v_a_718_);
lean_dec_ref_known(v_fst_694_, 1);
v_fst_719_ = lean_ctor_get(v_a_718_, 0);
lean_inc(v_fst_719_);
lean_dec(v_a_718_);
if (lean_obj_tag(v_fst_719_) == 0)
{
lean_object* v___x_720_; lean_object* v___x_721_; 
v___x_720_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__5, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___closed__5);
v___x_721_ = l_panic___redArg(v___x_681_, v___x_720_);
v___y_700_ = v___x_721_;
goto v___jp_699_;
}
else
{
lean_object* v_val_722_; 
v_val_722_ = lean_ctor_get(v_fst_719_, 0);
lean_inc(v_val_722_);
lean_dec_ref_known(v_fst_719_, 1);
v___y_700_ = v_val_722_;
goto v___jp_699_;
}
}
v___jp_699_:
{
lean_object* v___x_701_; lean_object* v___x_703_; 
v___x_701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_701_, 0, v___y_700_);
if (v_isShared_698_ == 0)
{
lean_ctor_set(v___x_697_, 0, v___x_701_);
v___x_703_ = v___x_697_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_707_; 
v_reuseFailAlloc_707_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_707_, 0, v___x_701_);
lean_ctor_set(v_reuseFailAlloc_707_, 1, v_snd_695_);
v___x_703_ = v_reuseFailAlloc_707_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
lean_object* v___x_705_; 
if (v_isShared_693_ == 0)
{
lean_ctor_set(v___x_692_, 0, v___x_703_);
v___x_705_ = v___x_692_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_706_; 
v_reuseFailAlloc_706_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_706_, 0, v___x_703_);
v___x_705_ = v_reuseFailAlloc_706_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
return v___x_705_;
}
}
}
}
}
}
else
{
lean_object* v_a_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_732_; 
v_a_725_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_732_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_732_ == 0)
{
v___x_727_ = v___x_689_;
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_a_725_);
lean_dec(v___x_689_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v___x_730_; 
if (v_isShared_728_ == 0)
{
v___x_730_ = v___x_727_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v_a_725_);
v___x_730_ = v_reuseFailAlloc_731_;
goto v_reusejp_729_;
}
v_reusejp_729_:
{
return v___x_730_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg___boxed(lean_object* v_inst_733_, lean_object* v_enterIdx_734_, lean_object* v_a_735_, lean_object* v_a_736_, lean_object* v_a_737_, lean_object* v_a_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg(v_inst_733_, v_enterIdx_734_, v_a_735_, v_a_736_, v_a_737_);
lean_dec(v_a_737_);
lean_dec_ref(v_a_736_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar(lean_object* v_matType_740_, lean_object* v_inst_741_, lean_object* v_enterIdx_742_, lean_object* v_a_743_, lean_object* v_a_744_, lean_object* v_a_745_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg(v_inst_741_, v_enterIdx_742_, v_a_743_, v_a_744_, v_a_745_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___boxed(lean_object* v_matType_748_, lean_object* v_inst_749_, lean_object* v_enterIdx_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar(v_matType_748_, v_inst_749_, v_enterIdx_750_, v_a_751_, v_a_752_, v_a_753_);
lean_dec(v_a_753_);
lean_dec_ref(v_a_752_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___redArg(lean_object* v_inst_756_, lean_object* v_a_757_, lean_object* v_a_758_, lean_object* v_a_759_){
_start:
{
lean_object* v___x_761_; 
lean_inc_ref(v_inst_756_);
v___x_761_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseEnteringVar___redArg(v_inst_756_, v_a_757_, v_a_758_, v_a_759_);
if (lean_obj_tag(v___x_761_) == 0)
{
lean_object* v_a_762_; lean_object* v___x_764_; uint8_t v_isShared_765_; uint8_t v_isSharedCheck_852_; 
v_a_762_ = lean_ctor_get(v___x_761_, 0);
v_isSharedCheck_852_ = !lean_is_exclusive(v___x_761_);
if (v_isSharedCheck_852_ == 0)
{
v___x_764_ = v___x_761_;
v_isShared_765_ = v_isSharedCheck_852_;
goto v_resetjp_763_;
}
else
{
lean_inc(v_a_762_);
lean_dec(v___x_761_);
v___x_764_ = lean_box(0);
v_isShared_765_ = v_isSharedCheck_852_;
goto v_resetjp_763_;
}
v_resetjp_763_:
{
lean_object* v_fst_766_; 
v_fst_766_ = lean_ctor_get(v_a_762_, 0);
lean_inc(v_fst_766_);
if (lean_obj_tag(v_fst_766_) == 0)
{
lean_object* v_snd_767_; lean_object* v___x_769_; uint8_t v_isShared_770_; uint8_t v_isSharedCheck_785_; 
lean_dec_ref(v_inst_756_);
v_snd_767_ = lean_ctor_get(v_a_762_, 1);
v_isSharedCheck_785_ = !lean_is_exclusive(v_a_762_);
if (v_isSharedCheck_785_ == 0)
{
lean_object* v_unused_786_; 
v_unused_786_ = lean_ctor_get(v_a_762_, 0);
lean_dec(v_unused_786_);
v___x_769_ = v_a_762_;
v_isShared_770_ = v_isSharedCheck_785_;
goto v_resetjp_768_;
}
else
{
lean_inc(v_snd_767_);
lean_dec(v_a_762_);
v___x_769_ = lean_box(0);
v_isShared_770_ = v_isSharedCheck_785_;
goto v_resetjp_768_;
}
v_resetjp_768_:
{
lean_object* v_a_771_; lean_object* v___x_773_; uint8_t v_isShared_774_; uint8_t v_isSharedCheck_784_; 
v_a_771_ = lean_ctor_get(v_fst_766_, 0);
v_isSharedCheck_784_ = !lean_is_exclusive(v_fst_766_);
if (v_isSharedCheck_784_ == 0)
{
v___x_773_ = v_fst_766_;
v_isShared_774_ = v_isSharedCheck_784_;
goto v_resetjp_772_;
}
else
{
lean_inc(v_a_771_);
lean_dec(v_fst_766_);
v___x_773_ = lean_box(0);
v_isShared_774_ = v_isSharedCheck_784_;
goto v_resetjp_772_;
}
v_resetjp_772_:
{
lean_object* v___x_776_; 
if (v_isShared_774_ == 0)
{
v___x_776_ = v___x_773_;
goto v_reusejp_775_;
}
else
{
lean_object* v_reuseFailAlloc_783_; 
v_reuseFailAlloc_783_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_783_, 0, v_a_771_);
v___x_776_ = v_reuseFailAlloc_783_;
goto v_reusejp_775_;
}
v_reusejp_775_:
{
lean_object* v___x_778_; 
if (v_isShared_770_ == 0)
{
lean_ctor_set(v___x_769_, 0, v___x_776_);
v___x_778_ = v___x_769_;
goto v_reusejp_777_;
}
else
{
lean_object* v_reuseFailAlloc_782_; 
v_reuseFailAlloc_782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_782_, 0, v___x_776_);
lean_ctor_set(v_reuseFailAlloc_782_, 1, v_snd_767_);
v___x_778_ = v_reuseFailAlloc_782_;
goto v_reusejp_777_;
}
v_reusejp_777_:
{
lean_object* v___x_780_; 
if (v_isShared_765_ == 0)
{
lean_ctor_set(v___x_764_, 0, v___x_778_);
v___x_780_ = v___x_764_;
goto v_reusejp_779_;
}
else
{
lean_object* v_reuseFailAlloc_781_; 
v_reuseFailAlloc_781_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_781_, 0, v___x_778_);
v___x_780_ = v_reuseFailAlloc_781_;
goto v_reusejp_779_;
}
v_reusejp_779_:
{
return v___x_780_;
}
}
}
}
}
}
else
{
lean_object* v_snd_787_; lean_object* v___x_789_; uint8_t v_isShared_790_; uint8_t v_isSharedCheck_850_; 
lean_del_object(v___x_764_);
v_snd_787_ = lean_ctor_get(v_a_762_, 1);
v_isSharedCheck_850_ = !lean_is_exclusive(v_a_762_);
if (v_isSharedCheck_850_ == 0)
{
lean_object* v_unused_851_; 
v_unused_851_ = lean_ctor_get(v_a_762_, 0);
lean_dec(v_unused_851_);
v___x_789_ = v_a_762_;
v_isShared_790_ = v_isSharedCheck_850_;
goto v_resetjp_788_;
}
else
{
lean_inc(v_snd_787_);
lean_dec(v_a_762_);
v___x_789_ = lean_box(0);
v_isShared_790_ = v_isSharedCheck_850_;
goto v_resetjp_788_;
}
v_resetjp_788_:
{
lean_object* v_a_791_; lean_object* v___x_792_; 
v_a_791_ = lean_ctor_get(v_fst_766_, 0);
lean_inc_n(v_a_791_, 2);
lean_dec_ref_known(v_fst_766_, 1);
v___x_792_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_chooseExitingVar___redArg(v_inst_756_, v_a_791_, v_snd_787_, v_a_758_, v_a_759_);
if (lean_obj_tag(v___x_792_) == 0)
{
lean_object* v_a_793_; lean_object* v___x_795_; uint8_t v_isShared_796_; uint8_t v_isSharedCheck_841_; 
v_a_793_ = lean_ctor_get(v___x_792_, 0);
v_isSharedCheck_841_ = !lean_is_exclusive(v___x_792_);
if (v_isSharedCheck_841_ == 0)
{
v___x_795_ = v___x_792_;
v_isShared_796_ = v_isSharedCheck_841_;
goto v_resetjp_794_;
}
else
{
lean_inc(v_a_793_);
lean_dec(v___x_792_);
v___x_795_ = lean_box(0);
v_isShared_796_ = v_isSharedCheck_841_;
goto v_resetjp_794_;
}
v_resetjp_794_:
{
lean_object* v_fst_797_; 
v_fst_797_ = lean_ctor_get(v_a_793_, 0);
lean_inc(v_fst_797_);
if (lean_obj_tag(v_fst_797_) == 0)
{
lean_object* v_snd_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_816_; 
lean_dec(v_a_791_);
lean_del_object(v___x_789_);
v_snd_798_ = lean_ctor_get(v_a_793_, 1);
v_isSharedCheck_816_ = !lean_is_exclusive(v_a_793_);
if (v_isSharedCheck_816_ == 0)
{
lean_object* v_unused_817_; 
v_unused_817_ = lean_ctor_get(v_a_793_, 0);
lean_dec(v_unused_817_);
v___x_800_ = v_a_793_;
v_isShared_801_ = v_isSharedCheck_816_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_snd_798_);
lean_dec(v_a_793_);
v___x_800_ = lean_box(0);
v_isShared_801_ = v_isSharedCheck_816_;
goto v_resetjp_799_;
}
v_resetjp_799_:
{
lean_object* v_a_802_; lean_object* v___x_804_; uint8_t v_isShared_805_; uint8_t v_isSharedCheck_815_; 
v_a_802_ = lean_ctor_get(v_fst_797_, 0);
v_isSharedCheck_815_ = !lean_is_exclusive(v_fst_797_);
if (v_isSharedCheck_815_ == 0)
{
v___x_804_ = v_fst_797_;
v_isShared_805_ = v_isSharedCheck_815_;
goto v_resetjp_803_;
}
else
{
lean_inc(v_a_802_);
lean_dec(v_fst_797_);
v___x_804_ = lean_box(0);
v_isShared_805_ = v_isSharedCheck_815_;
goto v_resetjp_803_;
}
v_resetjp_803_:
{
lean_object* v___x_807_; 
if (v_isShared_805_ == 0)
{
v___x_807_ = v___x_804_;
goto v_reusejp_806_;
}
else
{
lean_object* v_reuseFailAlloc_814_; 
v_reuseFailAlloc_814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_814_, 0, v_a_802_);
v___x_807_ = v_reuseFailAlloc_814_;
goto v_reusejp_806_;
}
v_reusejp_806_:
{
lean_object* v___x_809_; 
if (v_isShared_801_ == 0)
{
lean_ctor_set(v___x_800_, 0, v___x_807_);
v___x_809_ = v___x_800_;
goto v_reusejp_808_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v___x_807_);
lean_ctor_set(v_reuseFailAlloc_813_, 1, v_snd_798_);
v___x_809_ = v_reuseFailAlloc_813_;
goto v_reusejp_808_;
}
v_reusejp_808_:
{
lean_object* v___x_811_; 
if (v_isShared_796_ == 0)
{
lean_ctor_set(v___x_795_, 0, v___x_809_);
v___x_811_ = v___x_795_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v___x_809_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
return v___x_811_;
}
}
}
}
}
}
else
{
lean_object* v_snd_818_; lean_object* v___x_820_; uint8_t v_isShared_821_; uint8_t v_isSharedCheck_839_; 
v_snd_818_ = lean_ctor_get(v_a_793_, 1);
v_isSharedCheck_839_ = !lean_is_exclusive(v_a_793_);
if (v_isSharedCheck_839_ == 0)
{
lean_object* v_unused_840_; 
v_unused_840_ = lean_ctor_get(v_a_793_, 0);
lean_dec(v_unused_840_);
v___x_820_ = v_a_793_;
v_isShared_821_ = v_isSharedCheck_839_;
goto v_resetjp_819_;
}
else
{
lean_inc(v_snd_818_);
lean_dec(v_a_793_);
v___x_820_ = lean_box(0);
v_isShared_821_ = v_isSharedCheck_839_;
goto v_resetjp_819_;
}
v_resetjp_819_:
{
lean_object* v_a_822_; lean_object* v___x_824_; uint8_t v_isShared_825_; uint8_t v_isSharedCheck_838_; 
v_a_822_ = lean_ctor_get(v_fst_797_, 0);
v_isSharedCheck_838_ = !lean_is_exclusive(v_fst_797_);
if (v_isSharedCheck_838_ == 0)
{
v___x_824_ = v_fst_797_;
v_isShared_825_ = v_isSharedCheck_838_;
goto v_resetjp_823_;
}
else
{
lean_inc(v_a_822_);
lean_dec(v_fst_797_);
v___x_824_ = lean_box(0);
v_isShared_825_ = v_isSharedCheck_838_;
goto v_resetjp_823_;
}
v_resetjp_823_:
{
lean_object* v___x_827_; 
if (v_isShared_821_ == 0)
{
lean_ctor_set(v___x_820_, 1, v_a_791_);
lean_ctor_set(v___x_820_, 0, v_a_822_);
v___x_827_ = v___x_820_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_837_; 
v_reuseFailAlloc_837_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_837_, 0, v_a_822_);
lean_ctor_set(v_reuseFailAlloc_837_, 1, v_a_791_);
v___x_827_ = v_reuseFailAlloc_837_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
lean_object* v___x_829_; 
if (v_isShared_825_ == 0)
{
lean_ctor_set(v___x_824_, 0, v___x_827_);
v___x_829_ = v___x_824_;
goto v_reusejp_828_;
}
else
{
lean_object* v_reuseFailAlloc_836_; 
v_reuseFailAlloc_836_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_836_, 0, v___x_827_);
v___x_829_ = v_reuseFailAlloc_836_;
goto v_reusejp_828_;
}
v_reusejp_828_:
{
lean_object* v___x_831_; 
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v_snd_818_);
lean_ctor_set(v___x_789_, 0, v___x_829_);
v___x_831_ = v___x_789_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_835_; 
v_reuseFailAlloc_835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_835_, 0, v___x_829_);
lean_ctor_set(v_reuseFailAlloc_835_, 1, v_snd_818_);
v___x_831_ = v_reuseFailAlloc_835_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
lean_object* v___x_833_; 
if (v_isShared_796_ == 0)
{
lean_ctor_set(v___x_795_, 0, v___x_831_);
v___x_833_ = v___x_795_;
goto v_reusejp_832_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_831_);
v___x_833_ = v_reuseFailAlloc_834_;
goto v_reusejp_832_;
}
v_reusejp_832_:
{
return v___x_833_;
}
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_842_; lean_object* v___x_844_; uint8_t v_isShared_845_; uint8_t v_isSharedCheck_849_; 
lean_dec(v_a_791_);
lean_del_object(v___x_789_);
v_a_842_ = lean_ctor_get(v___x_792_, 0);
v_isSharedCheck_849_ = !lean_is_exclusive(v___x_792_);
if (v_isSharedCheck_849_ == 0)
{
v___x_844_ = v___x_792_;
v_isShared_845_ = v_isSharedCheck_849_;
goto v_resetjp_843_;
}
else
{
lean_inc(v_a_842_);
lean_dec(v___x_792_);
v___x_844_ = lean_box(0);
v_isShared_845_ = v_isSharedCheck_849_;
goto v_resetjp_843_;
}
v_resetjp_843_:
{
lean_object* v___x_847_; 
if (v_isShared_845_ == 0)
{
v___x_847_ = v___x_844_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_848_; 
v_reuseFailAlloc_848_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_848_, 0, v_a_842_);
v___x_847_ = v_reuseFailAlloc_848_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
return v___x_847_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_853_; lean_object* v___x_855_; uint8_t v_isShared_856_; uint8_t v_isSharedCheck_860_; 
lean_dec_ref(v_inst_756_);
v_a_853_ = lean_ctor_get(v___x_761_, 0);
v_isSharedCheck_860_ = !lean_is_exclusive(v___x_761_);
if (v_isSharedCheck_860_ == 0)
{
v___x_855_ = v___x_761_;
v_isShared_856_ = v_isSharedCheck_860_;
goto v_resetjp_854_;
}
else
{
lean_inc(v_a_853_);
lean_dec(v___x_761_);
v___x_855_ = lean_box(0);
v_isShared_856_ = v_isSharedCheck_860_;
goto v_resetjp_854_;
}
v_resetjp_854_:
{
lean_object* v___x_858_; 
if (v_isShared_856_ == 0)
{
v___x_858_ = v___x_855_;
goto v_reusejp_857_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v_a_853_);
v___x_858_ = v_reuseFailAlloc_859_;
goto v_reusejp_857_;
}
v_reusejp_857_:
{
return v___x_858_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___redArg___boxed(lean_object* v_inst_861_, lean_object* v_a_862_, lean_object* v_a_863_, lean_object* v_a_864_, lean_object* v_a_865_){
_start:
{
lean_object* v_res_866_; 
v_res_866_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___redArg(v_inst_861_, v_a_862_, v_a_863_, v_a_864_);
lean_dec(v_a_864_);
lean_dec_ref(v_a_863_);
return v_res_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots(lean_object* v_matType_867_, lean_object* v_inst_868_, lean_object* v_a_869_, lean_object* v_a_870_, lean_object* v_a_871_){
_start:
{
lean_object* v___x_873_; 
v___x_873_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___redArg(v_inst_868_, v_a_869_, v_a_870_, v_a_871_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___boxed(lean_object* v_matType_874_, lean_object* v_inst_875_, lean_object* v_a_876_, lean_object* v_a_877_, lean_object* v_a_878_, lean_object* v_a_879_){
_start:
{
lean_object* v_res_880_; 
v_res_880_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots(v_matType_874_, v_inst_875_, v_a_876_, v_a_877_, v_a_878_);
lean_dec(v_a_878_);
lean_dec_ref(v_a_877_);
return v_res_880_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__6(void){
_start:
{
uint8_t v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_892_ = 1;
v___x_893_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__5));
v___x_894_ = l_Lean_Name_toString(v___x_893_, v___x_892_);
return v___x_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0(lean_object* v_inst_895_, lean_object* v___x_896_, lean_object* v_b_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_){
_start:
{
lean_object* v_a_903_; lean_object* v_snd_904_; lean_object* v___x_908_; 
lean_inc_ref(v_inst_895_);
v___x_908_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg(v_inst_895_, v___y_898_, v___y_899_, v___y_900_);
if (lean_obj_tag(v___x_908_) == 0)
{
lean_object* v_a_909_; lean_object* v___x_911_; uint8_t v_isShared_912_; uint8_t v_isSharedCheck_999_; 
v_a_909_ = lean_ctor_get(v___x_908_, 0);
v_isSharedCheck_999_ = !lean_is_exclusive(v___x_908_);
if (v_isSharedCheck_999_ == 0)
{
v___x_911_ = v___x_908_;
v_isShared_912_ = v_isSharedCheck_999_;
goto v_resetjp_910_;
}
else
{
lean_inc(v_a_909_);
lean_dec(v___x_908_);
v___x_911_ = lean_box(0);
v_isShared_912_ = v_isSharedCheck_999_;
goto v_resetjp_910_;
}
v_resetjp_910_:
{
lean_object* v_fst_913_; 
v_fst_913_ = lean_ctor_get(v_a_909_, 0);
lean_inc(v_fst_913_);
if (lean_obj_tag(v_fst_913_) == 0)
{
lean_object* v_snd_914_; lean_object* v_a_915_; 
lean_del_object(v___x_911_);
lean_dec_ref(v_inst_895_);
v_snd_914_ = lean_ctor_get(v_a_909_, 1);
lean_inc(v_snd_914_);
lean_dec(v_a_909_);
v_a_915_ = lean_ctor_get(v_fst_913_, 0);
lean_inc(v_a_915_);
lean_dec_ref_known(v_fst_913_, 1);
v_a_903_ = v_a_915_;
v_snd_904_ = v_snd_914_;
goto v___jp_902_;
}
else
{
lean_object* v_a_916_; lean_object* v___x_918_; uint8_t v_isShared_919_; uint8_t v_isSharedCheck_998_; 
v_a_916_ = lean_ctor_get(v_fst_913_, 0);
v_isSharedCheck_998_ = !lean_is_exclusive(v_fst_913_);
if (v_isSharedCheck_998_ == 0)
{
v___x_918_ = v_fst_913_;
v_isShared_919_ = v_isSharedCheck_998_;
goto v_resetjp_917_;
}
else
{
lean_inc(v_a_916_);
lean_dec(v_fst_913_);
v___x_918_ = lean_box(0);
v_isShared_919_ = v_isSharedCheck_998_;
goto v_resetjp_917_;
}
v_resetjp_917_:
{
uint8_t v___x_920_; 
v___x_920_ = lean_unbox(v_a_916_);
lean_dec(v_a_916_);
if (v___x_920_ == 0)
{
lean_object* v_snd_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
lean_del_object(v___x_918_);
lean_del_object(v___x_911_);
v_snd_921_ = lean_ctor_get(v_a_909_, 1);
lean_inc(v_snd_921_);
lean_dec(v_a_909_);
v___x_922_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__6, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___closed__6);
v___x_923_ = l_Lean_Core_checkSystem(v___x_922_, v___y_899_, v___y_900_);
if (lean_obj_tag(v___x_923_) == 0)
{
lean_object* v___x_924_; 
lean_dec_ref_known(v___x_923_, 1);
lean_inc_ref(v_inst_895_);
v___x_924_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_choosePivots___redArg(v_inst_895_, v_snd_921_, v___y_899_, v___y_900_);
if (lean_obj_tag(v___x_924_) == 0)
{
lean_object* v_a_925_; lean_object* v_fst_926_; 
v_a_925_ = lean_ctor_get(v___x_924_, 0);
lean_inc(v_a_925_);
lean_dec_ref_known(v___x_924_, 1);
v_fst_926_ = lean_ctor_get(v_a_925_, 0);
lean_inc(v_fst_926_);
if (lean_obj_tag(v_fst_926_) == 0)
{
lean_object* v_snd_927_; lean_object* v_a_928_; 
lean_dec_ref(v_inst_895_);
v_snd_927_ = lean_ctor_get(v_a_925_, 1);
lean_inc(v_snd_927_);
lean_dec(v_a_925_);
v_a_928_ = lean_ctor_get(v_fst_926_, 0);
lean_inc(v_a_928_);
lean_dec_ref_known(v_fst_926_, 1);
v_a_903_ = v_a_928_;
v_snd_904_ = v_snd_927_;
goto v___jp_902_;
}
else
{
lean_object* v_a_929_; lean_object* v___x_931_; uint8_t v_isShared_932_; uint8_t v_isSharedCheck_965_; 
v_a_929_ = lean_ctor_get(v_fst_926_, 0);
v_isSharedCheck_965_ = !lean_is_exclusive(v_fst_926_);
if (v_isSharedCheck_965_ == 0)
{
v___x_931_ = v_fst_926_;
v_isShared_932_ = v_isSharedCheck_965_;
goto v_resetjp_930_;
}
else
{
lean_inc(v_a_929_);
lean_dec(v_fst_926_);
v___x_931_ = lean_box(0);
v_isShared_932_ = v_isSharedCheck_965_;
goto v_resetjp_930_;
}
v_resetjp_930_:
{
lean_object* v_snd_933_; lean_object* v_fst_934_; lean_object* v_snd_935_; lean_object* v___x_936_; lean_object* v_a_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_964_; 
v_snd_933_ = lean_ctor_get(v_a_925_, 1);
lean_inc(v_snd_933_);
lean_dec(v_a_925_);
v_fst_934_ = lean_ctor_get(v_a_929_, 0);
lean_inc(v_fst_934_);
v_snd_935_ = lean_ctor_get(v_a_929_, 1);
lean_inc(v_snd_935_);
lean_dec(v_a_929_);
v___x_936_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg(v_inst_895_, v_fst_934_, v_snd_935_, v_snd_933_);
v_a_937_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_964_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_964_ == 0)
{
v___x_939_ = v___x_936_;
v_isShared_940_ = v_isSharedCheck_964_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_a_937_);
lean_dec(v___x_936_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_964_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
lean_object* v_fst_941_; lean_object* v_snd_942_; lean_object* v___x_944_; uint8_t v_isShared_945_; uint8_t v_isSharedCheck_963_; 
v_fst_941_ = lean_ctor_get(v_a_937_, 0);
v_snd_942_ = lean_ctor_get(v_a_937_, 1);
v_isSharedCheck_963_ = !lean_is_exclusive(v_a_937_);
if (v_isSharedCheck_963_ == 0)
{
v___x_944_ = v_a_937_;
v_isShared_945_ = v_isSharedCheck_963_;
goto v_resetjp_943_;
}
else
{
lean_inc(v_snd_942_);
lean_inc(v_fst_941_);
lean_dec(v_a_937_);
v___x_944_ = lean_box(0);
v_isShared_945_ = v_isSharedCheck_963_;
goto v_resetjp_943_;
}
v_resetjp_943_:
{
lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_961_; 
v_isSharedCheck_961_ = !lean_is_exclusive(v_fst_941_);
if (v_isSharedCheck_961_ == 0)
{
lean_object* v_unused_962_; 
v_unused_962_ = lean_ctor_get(v_fst_941_, 0);
lean_dec(v_unused_962_);
v___x_947_ = v_fst_941_;
v_isShared_948_ = v_isSharedCheck_961_;
goto v_resetjp_946_;
}
else
{
lean_dec(v_fst_941_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_961_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
lean_object* v___x_950_; 
if (v_isShared_932_ == 0)
{
lean_ctor_set_tag(v___x_931_, 0);
lean_ctor_set(v___x_931_, 0, v___x_896_);
v___x_950_ = v___x_931_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v___x_896_);
v___x_950_ = v_reuseFailAlloc_960_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
lean_object* v___x_952_; 
if (v_isShared_948_ == 0)
{
lean_ctor_set(v___x_947_, 0, v___x_950_);
v___x_952_ = v___x_947_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_959_; 
v_reuseFailAlloc_959_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_959_, 0, v___x_950_);
v___x_952_ = v_reuseFailAlloc_959_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
lean_object* v___x_954_; 
if (v_isShared_945_ == 0)
{
lean_ctor_set(v___x_944_, 0, v___x_952_);
v___x_954_ = v___x_944_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_958_; 
v_reuseFailAlloc_958_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_958_, 0, v___x_952_);
lean_ctor_set(v_reuseFailAlloc_958_, 1, v_snd_942_);
v___x_954_ = v_reuseFailAlloc_958_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
lean_object* v___x_956_; 
if (v_isShared_940_ == 0)
{
lean_ctor_set(v___x_939_, 0, v___x_954_);
v___x_956_ = v___x_939_;
goto v_reusejp_955_;
}
else
{
lean_object* v_reuseFailAlloc_957_; 
v_reuseFailAlloc_957_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_957_, 0, v___x_954_);
v___x_956_ = v_reuseFailAlloc_957_;
goto v_reusejp_955_;
}
v_reusejp_955_:
{
return v___x_956_;
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
else
{
lean_object* v_a_966_; lean_object* v___x_968_; uint8_t v_isShared_969_; uint8_t v_isSharedCheck_973_; 
lean_dec_ref(v_inst_895_);
v_a_966_ = lean_ctor_get(v___x_924_, 0);
v_isSharedCheck_973_ = !lean_is_exclusive(v___x_924_);
if (v_isSharedCheck_973_ == 0)
{
v___x_968_ = v___x_924_;
v_isShared_969_ = v_isSharedCheck_973_;
goto v_resetjp_967_;
}
else
{
lean_inc(v_a_966_);
lean_dec(v___x_924_);
v___x_968_ = lean_box(0);
v_isShared_969_ = v_isSharedCheck_973_;
goto v_resetjp_967_;
}
v_resetjp_967_:
{
lean_object* v___x_971_; 
if (v_isShared_969_ == 0)
{
v___x_971_ = v___x_968_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_972_; 
v_reuseFailAlloc_972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_972_, 0, v_a_966_);
v___x_971_ = v_reuseFailAlloc_972_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
return v___x_971_;
}
}
}
}
else
{
lean_object* v_a_974_; lean_object* v___x_976_; uint8_t v_isShared_977_; uint8_t v_isSharedCheck_981_; 
lean_dec(v_snd_921_);
lean_dec_ref(v_inst_895_);
v_a_974_ = lean_ctor_get(v___x_923_, 0);
v_isSharedCheck_981_ = !lean_is_exclusive(v___x_923_);
if (v_isSharedCheck_981_ == 0)
{
v___x_976_ = v___x_923_;
v_isShared_977_ = v_isSharedCheck_981_;
goto v_resetjp_975_;
}
else
{
lean_inc(v_a_974_);
lean_dec(v___x_923_);
v___x_976_ = lean_box(0);
v_isShared_977_ = v_isSharedCheck_981_;
goto v_resetjp_975_;
}
v_resetjp_975_:
{
lean_object* v___x_979_; 
if (v_isShared_977_ == 0)
{
v___x_979_ = v___x_976_;
goto v_reusejp_978_;
}
else
{
lean_object* v_reuseFailAlloc_980_; 
v_reuseFailAlloc_980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_980_, 0, v_a_974_);
v___x_979_ = v_reuseFailAlloc_980_;
goto v_reusejp_978_;
}
v_reusejp_978_:
{
return v___x_979_;
}
}
}
}
else
{
lean_object* v_snd_982_; lean_object* v___x_984_; uint8_t v_isShared_985_; uint8_t v_isSharedCheck_996_; 
lean_dec_ref(v_inst_895_);
v_snd_982_ = lean_ctor_get(v_a_909_, 1);
v_isSharedCheck_996_ = !lean_is_exclusive(v_a_909_);
if (v_isSharedCheck_996_ == 0)
{
lean_object* v_unused_997_; 
v_unused_997_ = lean_ctor_get(v_a_909_, 0);
lean_dec(v_unused_997_);
v___x_984_ = v_a_909_;
v_isShared_985_ = v_isSharedCheck_996_;
goto v_resetjp_983_;
}
else
{
lean_inc(v_snd_982_);
lean_dec(v_a_909_);
v___x_984_ = lean_box(0);
v_isShared_985_ = v_isSharedCheck_996_;
goto v_resetjp_983_;
}
v_resetjp_983_:
{
lean_object* v___x_986_; lean_object* v___x_988_; 
v___x_986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_986_, 0, v___x_896_);
if (v_isShared_919_ == 0)
{
lean_ctor_set(v___x_918_, 0, v___x_986_);
v___x_988_ = v___x_918_;
goto v_reusejp_987_;
}
else
{
lean_object* v_reuseFailAlloc_995_; 
v_reuseFailAlloc_995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_995_, 0, v___x_986_);
v___x_988_ = v_reuseFailAlloc_995_;
goto v_reusejp_987_;
}
v_reusejp_987_:
{
lean_object* v___x_990_; 
if (v_isShared_985_ == 0)
{
lean_ctor_set(v___x_984_, 0, v___x_988_);
v___x_990_ = v___x_984_;
goto v_reusejp_989_;
}
else
{
lean_object* v_reuseFailAlloc_994_; 
v_reuseFailAlloc_994_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_994_, 0, v___x_988_);
lean_ctor_set(v_reuseFailAlloc_994_, 1, v_snd_982_);
v___x_990_ = v_reuseFailAlloc_994_;
goto v_reusejp_989_;
}
v_reusejp_989_:
{
lean_object* v___x_992_; 
if (v_isShared_912_ == 0)
{
lean_ctor_set(v___x_911_, 0, v___x_990_);
v___x_992_ = v___x_911_;
goto v_reusejp_991_;
}
else
{
lean_object* v_reuseFailAlloc_993_; 
v_reuseFailAlloc_993_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_993_, 0, v___x_990_);
v___x_992_ = v_reuseFailAlloc_993_;
goto v_reusejp_991_;
}
v_reusejp_991_:
{
return v___x_992_;
}
}
}
}
}
}
}
}
}
else
{
lean_object* v_a_1000_; lean_object* v___x_1002_; uint8_t v_isShared_1003_; uint8_t v_isSharedCheck_1007_; 
lean_dec_ref(v_inst_895_);
v_a_1000_ = lean_ctor_get(v___x_908_, 0);
v_isSharedCheck_1007_ = !lean_is_exclusive(v___x_908_);
if (v_isSharedCheck_1007_ == 0)
{
v___x_1002_ = v___x_908_;
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
else
{
lean_inc(v_a_1000_);
lean_dec(v___x_908_);
v___x_1002_ = lean_box(0);
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
v_resetjp_1001_:
{
lean_object* v___x_1005_; 
if (v_isShared_1003_ == 0)
{
v___x_1005_ = v___x_1002_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v_a_1000_);
v___x_1005_ = v_reuseFailAlloc_1006_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
return v___x_1005_;
}
}
}
v___jp_902_:
{
lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; 
v___x_905_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_905_, 0, v_a_903_);
v___x_906_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_906_, 0, v___x_905_);
lean_ctor_set(v___x_906_, 1, v_snd_904_);
v___x_907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_907_, 0, v___x_906_);
return v___x_907_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___boxed(lean_object* v_inst_1008_, lean_object* v___x_1009_, lean_object* v_b_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_){
_start:
{
lean_object* v_res_1015_; 
v_res_1015_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0(v_inst_1008_, v___x_1009_, v_b_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
return v_res_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg(lean_object* v_inst_1016_, lean_object* v_a_1017_, lean_object* v_a_1018_, lean_object* v_a_1019_){
_start:
{
lean_object* v___x_1021_; lean_object* v_toApplicative_1022_; lean_object* v_toFunctor_1023_; lean_object* v_toSeq_1024_; lean_object* v_toSeqLeft_1025_; lean_object* v_toSeqRight_1026_; lean_object* v___f_1027_; lean_object* v___f_1028_; lean_object* v___f_1029_; lean_object* v___f_1030_; lean_object* v___x_1031_; lean_object* v___f_1032_; lean_object* v___f_1033_; lean_object* v___f_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___f_1037_; lean_object* v___f_1038_; lean_object* v___f_1039_; lean_object* v___f_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___f_1047_; lean_object* v___f_1048_; lean_object* v___f_1049_; lean_object* v___f_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___f_1058_; lean_object* v___x_5399__overap_1059_; lean_object* v___x_1060_; 
v___x_1021_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__1);
v_toApplicative_1022_ = lean_ctor_get(v___x_1021_, 0);
v_toFunctor_1023_ = lean_ctor_get(v_toApplicative_1022_, 0);
v_toSeq_1024_ = lean_ctor_get(v_toApplicative_1022_, 2);
v_toSeqLeft_1025_ = lean_ctor_get(v_toApplicative_1022_, 3);
v_toSeqRight_1026_ = lean_ctor_get(v_toApplicative_1022_, 4);
v___f_1027_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__2));
v___f_1028_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_checkSuccess___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_1023_, 2);
v___f_1029_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1029_, 0, v_toFunctor_1023_);
v___f_1030_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1030_, 0, v_toFunctor_1023_);
v___x_1031_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1031_, 0, v___f_1029_);
lean_ctor_set(v___x_1031_, 1, v___f_1030_);
lean_inc(v_toSeqRight_1026_);
v___f_1032_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1032_, 0, v_toSeqRight_1026_);
lean_inc(v_toSeqLeft_1025_);
v___f_1033_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1033_, 0, v_toSeqLeft_1025_);
lean_inc(v_toSeq_1024_);
v___f_1034_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1034_, 0, v_toSeq_1024_);
v___x_1035_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1035_, 0, v___x_1031_);
lean_ctor_set(v___x_1035_, 1, v___f_1027_);
lean_ctor_set(v___x_1035_, 2, v___f_1034_);
lean_ctor_set(v___x_1035_, 3, v___f_1033_);
lean_ctor_set(v___x_1035_, 4, v___f_1032_);
v___x_1036_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1036_, 0, v___x_1035_);
lean_ctor_set(v___x_1036_, 1, v___f_1028_);
lean_inc_ref_n(v___x_1036_, 6);
v___f_1037_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1037_, 0, v___x_1036_);
v___f_1038_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1038_, 0, v___x_1036_);
v___f_1039_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_1039_, 0, v___x_1036_);
v___f_1040_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_1040_, 0, v___x_1036_);
v___x_1041_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_1041_, 0, lean_box(0));
lean_closure_set(v___x_1041_, 1, lean_box(0));
lean_closure_set(v___x_1041_, 2, v___x_1036_);
v___x_1042_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1042_, 0, v___x_1041_);
lean_ctor_set(v___x_1042_, 1, v___f_1037_);
v___x_1043_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_1043_, 0, lean_box(0));
lean_closure_set(v___x_1043_, 1, lean_box(0));
lean_closure_set(v___x_1043_, 2, v___x_1036_);
v___x_1044_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1044_, 0, v___x_1042_);
lean_ctor_set(v___x_1044_, 1, v___x_1043_);
lean_ctor_set(v___x_1044_, 2, v___f_1038_);
lean_ctor_set(v___x_1044_, 3, v___f_1039_);
lean_ctor_set(v___x_1044_, 4, v___f_1040_);
v___x_1045_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_1045_, 0, lean_box(0));
lean_closure_set(v___x_1045_, 1, lean_box(0));
lean_closure_set(v___x_1045_, 2, v___x_1036_);
v___x_1046_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1044_);
lean_ctor_set(v___x_1046_, 1, v___x_1045_);
lean_inc_ref_n(v___x_1046_, 6);
v___f_1047_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_1047_, 0, v___x_1046_);
v___f_1048_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__4), 5, 1);
lean_closure_set(v___f_1048_, 0, v___x_1046_);
v___f_1049_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__7), 5, 1);
lean_closure_set(v___f_1049_, 0, v___x_1046_);
v___f_1050_ = lean_alloc_closure((void*)(l_ExceptT_instMonad___redArg___lam__9), 5, 1);
lean_closure_set(v___f_1050_, 0, v___x_1046_);
v___x_1051_ = lean_alloc_closure((void*)(l_ExceptT_map), 7, 3);
lean_closure_set(v___x_1051_, 0, lean_box(0));
lean_closure_set(v___x_1051_, 1, lean_box(0));
lean_closure_set(v___x_1051_, 2, v___x_1046_);
v___x_1052_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1052_, 0, v___x_1051_);
lean_ctor_set(v___x_1052_, 1, v___f_1047_);
v___x_1053_ = lean_alloc_closure((void*)(l_ExceptT_pure), 5, 3);
lean_closure_set(v___x_1053_, 0, lean_box(0));
lean_closure_set(v___x_1053_, 1, lean_box(0));
lean_closure_set(v___x_1053_, 2, v___x_1046_);
v___x_1054_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1054_, 0, v___x_1052_);
lean_ctor_set(v___x_1054_, 1, v___x_1053_);
lean_ctor_set(v___x_1054_, 2, v___f_1048_);
lean_ctor_set(v___x_1054_, 3, v___f_1049_);
lean_ctor_set(v___x_1054_, 4, v___f_1050_);
v___x_1055_ = lean_alloc_closure((void*)(l_ExceptT_bind), 7, 3);
lean_closure_set(v___x_1055_, 0, lean_box(0));
lean_closure_set(v___x_1055_, 1, lean_box(0));
lean_closure_set(v___x_1055_, 2, v___x_1046_);
v___x_1056_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1054_);
lean_ctor_set(v___x_1056_, 1, v___x_1055_);
v___x_1057_ = lean_box(0);
v___f_1058_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1058_, 0, v_inst_1016_);
lean_closure_set(v___f_1058_, 1, v___x_1057_);
v___x_5399__overap_1059_ = l___private_Init_While_0__repeatM_erased___redArg(v___x_1056_, v___f_1058_, v___x_1057_);
lean_inc(v_a_1019_);
lean_inc_ref(v_a_1018_);
v___x_1060_ = lean_apply_4(v___x_5399__overap_1059_, v_a_1017_, v_a_1018_, v_a_1019_, lean_box(0));
if (lean_obj_tag(v___x_1060_) == 0)
{
lean_object* v_a_1061_; lean_object* v_fst_1062_; 
v_a_1061_ = lean_ctor_get(v___x_1060_, 0);
lean_inc(v_a_1061_);
v_fst_1062_ = lean_ctor_get(v_a_1061_, 0);
if (lean_obj_tag(v_fst_1062_) == 0)
{
lean_dec(v_a_1061_);
return v___x_1060_;
}
else
{
lean_object* v___x_1064_; uint8_t v_isShared_1065_; uint8_t v_isSharedCheck_1079_; 
v_isSharedCheck_1079_ = !lean_is_exclusive(v___x_1060_);
if (v_isSharedCheck_1079_ == 0)
{
lean_object* v_unused_1080_; 
v_unused_1080_ = lean_ctor_get(v___x_1060_, 0);
lean_dec(v_unused_1080_);
v___x_1064_ = v___x_1060_;
v_isShared_1065_ = v_isSharedCheck_1079_;
goto v_resetjp_1063_;
}
else
{
lean_dec(v___x_1060_);
v___x_1064_ = lean_box(0);
v_isShared_1065_ = v_isSharedCheck_1079_;
goto v_resetjp_1063_;
}
v_resetjp_1063_:
{
lean_object* v_snd_1066_; lean_object* v___x_1068_; uint8_t v_isShared_1069_; uint8_t v_isSharedCheck_1077_; 
v_snd_1066_ = lean_ctor_get(v_a_1061_, 1);
v_isSharedCheck_1077_ = !lean_is_exclusive(v_a_1061_);
if (v_isSharedCheck_1077_ == 0)
{
lean_object* v_unused_1078_; 
v_unused_1078_ = lean_ctor_get(v_a_1061_, 0);
lean_dec(v_unused_1078_);
v___x_1068_ = v_a_1061_;
v_isShared_1069_ = v_isSharedCheck_1077_;
goto v_resetjp_1067_;
}
else
{
lean_inc(v_snd_1066_);
lean_dec(v_a_1061_);
v___x_1068_ = lean_box(0);
v_isShared_1069_ = v_isSharedCheck_1077_;
goto v_resetjp_1067_;
}
v_resetjp_1067_:
{
lean_object* v___x_1070_; lean_object* v___x_1072_; 
v___x_1070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_doPivotOperation___redArg___closed__12));
if (v_isShared_1069_ == 0)
{
lean_ctor_set(v___x_1068_, 0, v___x_1070_);
v___x_1072_ = v___x_1068_;
goto v_reusejp_1071_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v___x_1070_);
lean_ctor_set(v_reuseFailAlloc_1076_, 1, v_snd_1066_);
v___x_1072_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1071_;
}
v_reusejp_1071_:
{
lean_object* v___x_1074_; 
if (v_isShared_1065_ == 0)
{
lean_ctor_set(v___x_1064_, 0, v___x_1072_);
v___x_1074_ = v___x_1064_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v___x_1072_);
v___x_1074_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
return v___x_1074_;
}
}
}
}
}
}
else
{
return v___x_1060_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg___boxed(lean_object* v_inst_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_, lean_object* v_a_1084_, lean_object* v_a_1085_){
_start:
{
lean_object* v_res_1086_; 
v_res_1086_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg(v_inst_1081_, v_a_1082_, v_a_1083_, v_a_1084_);
lean_dec(v_a_1084_);
lean_dec_ref(v_a_1083_);
return v_res_1086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm(lean_object* v_matType_1087_, lean_object* v_inst_1088_, lean_object* v_a_1089_, lean_object* v_a_1090_, lean_object* v_a_1091_){
_start:
{
lean_object* v___x_1093_; 
v___x_1093_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___redArg(v_inst_1088_, v_a_1089_, v_a_1090_, v_a_1091_);
return v___x_1093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm___boxed(lean_object* v_matType_1094_, lean_object* v_inst_1095_, lean_object* v_a_1096_, lean_object* v_a_1097_, lean_object* v_a_1098_, lean_object* v_a_1099_){
_start:
{
lean_object* v_res_1100_; 
v_res_1100_ = lp_mathlib_Mathlib_Tactic_Linarith_SimplexAlgorithm_runSimplexAlgorithm(v_matType_1094_, v_inst_1095_, v_a_1096_, v_a_1097_, v_a_1098_);
lean_dec(v_a_1098_);
lean_dec_ref(v_a_1097_);
return v_res_1100_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_Datatypes(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Oracle_SimplexAlgorithm_SimplexAlgorithm(builtin);
}
#ifdef __cplusplus
}
#endif
