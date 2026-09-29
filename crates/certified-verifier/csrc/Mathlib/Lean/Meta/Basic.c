// Lean compiler output
// Module: Mathlib.Lean.Meta.Basic
// Imports: public import Init public meta import Init public import Mathlib.Init public import Lean.Meta.AppBuilder public import Lean.Meta.Coe
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_getLast_x21___redArg(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_coerceSimple_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LOption_toOption___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withNewMCtxDepth___redArg(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_mkMVar(lean_object*);
lean_object* l_Lean_Meta_trySynthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Expr_abstractM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isForall(lean_object*);
lean_object* l_Lean_Meta_coerceToFunction_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isSort(lean_object*);
lean_object* l_Lean_Meta_coerceToSort_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Failed to find "};
static const lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__2;
static const lean_string_object lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = " as the type of a parameter of "};
static const lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__4;
static const lean_string_object lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Failed"};
static const lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__1;
static const lean_string_object lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Failed: "};
static const lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__3;
static const lean_string_object lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = " is not the type of a function."};
static const lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_pureIsDefEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_pureIsDefEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___closed__1;
static const lean_closure_object lp_mathlib_Lean_Meta_pureIsDefEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_pureIsDefEq___closed__2_value;
static const lean_closure_object lp_mathlib_Lean_Meta_pureIsDefEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_pureIsDefEq___closed__3_value;
static const lean_closure_object lp_mathlib_Lean_Meta_pureIsDefEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_pureIsDefEq___closed__4_value;
static const lean_closure_object lp_mathlib_Lean_Meta_pureIsDefEq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_pureIsDefEq___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_pureIsDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_mkRel___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_Meta_mkRel___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRel___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_mkRel___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_mkRel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_Meta_mkRel___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRel___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_mkRel___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Lean_Meta_mkRel___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRel___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_mkRel___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_mkRel___closed__2_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_Lean_Meta_mkRel___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRel___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRel___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRel___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "failed to assign synthesized type class instance "};
static const lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\nto"};
static const lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "inst"};
static const lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(170, 188, 240, 205, 110, 63, 170, 91)}};
static const lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_ensureHasType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Expected"};
static const lean_object* lp_mathlib_Lean_Meta_ensureHasType___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_ensureHasType___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ensureHasType___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ensureHasType___closed__1;
static const lean_string_object lp_mathlib_Lean_Meta_ensureHasType___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "\nto have type"};
static const lean_object* lp_mathlib_Lean_Meta_ensureHasType___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_ensureHasType___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ensureHasType___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ensureHasType___closed__3;
static const lean_string_object lp_mathlib_Lean_Meta_ensureHasType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "\n or to be coercible to it"};
static const lean_object* lp_mathlib_Lean_Meta_ensureHasType___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_ensureHasType___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ensureHasType___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ensureHasType___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_ensureIsFunction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "\nof type"};
static const lean_object* lp_mathlib_Lean_Meta_ensureIsFunction___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_ensureIsFunction___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ensureIsFunction___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ensureIsFunction___closed__1;
static const lean_string_object lp_mathlib_Lean_Meta_ensureIsFunction___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "\nto be a function, or to be coercible to a function"};
static const lean_object* lp_mathlib_Lean_Meta_ensureIsFunction___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_ensureIsFunction___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ensureIsFunction___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ensureIsFunction___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsFunction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsFunction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_ensureIsSort___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "\nto be a Sort, or to be coercible to a Sort"};
static const lean_object* lp_mathlib_Lean_Meta_ensureIsSort___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_ensureIsSort___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ensureIsSort___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ensureIsSort___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsSort(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsSort___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg___lam__0(lean_object* v_a_1_, lean_object* v_mctx_2_, lean_object* v_a_x3f_3_){
_start:
{
lean_object* v___x_5_; lean_object* v_cache_6_; lean_object* v_zetaDeltaFVarIds_7_; lean_object* v_postponed_8_; lean_object* v_diag_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_19_; 
v___x_5_ = lean_st_ref_take(v_a_1_);
v_cache_6_ = lean_ctor_get(v___x_5_, 1);
v_zetaDeltaFVarIds_7_ = lean_ctor_get(v___x_5_, 2);
v_postponed_8_ = lean_ctor_get(v___x_5_, 3);
v_diag_9_ = lean_ctor_get(v___x_5_, 4);
v_isSharedCheck_19_ = !lean_is_exclusive(v___x_5_);
if (v_isSharedCheck_19_ == 0)
{
lean_object* v_unused_20_; 
v_unused_20_ = lean_ctor_get(v___x_5_, 0);
lean_dec(v_unused_20_);
v___x_11_ = v___x_5_;
v_isShared_12_ = v_isSharedCheck_19_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_diag_9_);
lean_inc(v_postponed_8_);
lean_inc(v_zetaDeltaFVarIds_7_);
lean_inc(v_cache_6_);
lean_dec(v___x_5_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_19_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___x_14_; 
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 0, v_mctx_2_);
v___x_14_ = v___x_11_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v_mctx_2_);
lean_ctor_set(v_reuseFailAlloc_18_, 1, v_cache_6_);
lean_ctor_set(v_reuseFailAlloc_18_, 2, v_zetaDeltaFVarIds_7_);
lean_ctor_set(v_reuseFailAlloc_18_, 3, v_postponed_8_);
lean_ctor_set(v_reuseFailAlloc_18_, 4, v_diag_9_);
v___x_14_ = v_reuseFailAlloc_18_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_15_ = lean_st_ref_set(v_a_1_, v___x_14_);
v___x_16_ = lean_box(0);
v___x_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
return v___x_17_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg___lam__0___boxed(lean_object* v_a_21_, lean_object* v_mctx_22_, lean_object* v_a_x3f_23_, lean_object* v___y_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Lean_Meta_preservingMCtx___redArg___lam__0(v_a_21_, v_mctx_22_, v_a_x3f_23_);
lean_dec(v_a_x3f_23_);
lean_dec(v_a_21_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg(lean_object* v_x_26_, lean_object* v_a_27_, lean_object* v_a_28_, lean_object* v_a_29_, lean_object* v_a_30_){
_start:
{
lean_object* v___x_32_; lean_object* v_mctx_33_; lean_object* v_r_34_; 
v___x_32_ = lean_st_ref_get(v_a_28_);
v_mctx_33_ = lean_ctor_get(v___x_32_, 0);
lean_inc_ref(v_mctx_33_);
lean_dec(v___x_32_);
lean_inc(v_a_30_);
lean_inc_ref(v_a_29_);
lean_inc(v_a_28_);
lean_inc_ref(v_a_27_);
v_r_34_ = lean_apply_5(v_x_26_, v_a_27_, v_a_28_, v_a_29_, v_a_30_, lean_box(0));
if (lean_obj_tag(v_r_34_) == 0)
{
lean_object* v_a_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_51_; 
v_a_35_ = lean_ctor_get(v_r_34_, 0);
v_isSharedCheck_51_ = !lean_is_exclusive(v_r_34_);
if (v_isSharedCheck_51_ == 0)
{
v___x_37_ = v_r_34_;
v_isShared_38_ = v_isSharedCheck_51_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_a_35_);
lean_dec(v_r_34_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_51_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_40_; 
lean_inc(v_a_35_);
if (v_isShared_38_ == 0)
{
lean_ctor_set_tag(v___x_37_, 1);
v___x_40_ = v___x_37_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v_a_35_);
v___x_40_ = v_reuseFailAlloc_50_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
lean_object* v___x_41_; lean_object* v___x_43_; uint8_t v_isShared_44_; uint8_t v_isSharedCheck_48_; 
v___x_41_ = lp_mathlib_Lean_Meta_preservingMCtx___redArg___lam__0(v_a_28_, v_mctx_33_, v___x_40_);
lean_dec_ref(v___x_40_);
v_isSharedCheck_48_ = !lean_is_exclusive(v___x_41_);
if (v_isSharedCheck_48_ == 0)
{
lean_object* v_unused_49_; 
v_unused_49_ = lean_ctor_get(v___x_41_, 0);
lean_dec(v_unused_49_);
v___x_43_ = v___x_41_;
v_isShared_44_ = v_isSharedCheck_48_;
goto v_resetjp_42_;
}
else
{
lean_dec(v___x_41_);
v___x_43_ = lean_box(0);
v_isShared_44_ = v_isSharedCheck_48_;
goto v_resetjp_42_;
}
v_resetjp_42_:
{
lean_object* v___x_46_; 
if (v_isShared_44_ == 0)
{
lean_ctor_set(v___x_43_, 0, v_a_35_);
v___x_46_ = v___x_43_;
goto v_reusejp_45_;
}
else
{
lean_object* v_reuseFailAlloc_47_; 
v_reuseFailAlloc_47_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_47_, 0, v_a_35_);
v___x_46_ = v_reuseFailAlloc_47_;
goto v_reusejp_45_;
}
v_reusejp_45_:
{
return v___x_46_;
}
}
}
}
}
else
{
lean_object* v_a_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_56_; uint8_t v_isShared_57_; uint8_t v_isSharedCheck_61_; 
v_a_52_ = lean_ctor_get(v_r_34_, 0);
lean_inc(v_a_52_);
lean_dec_ref_known(v_r_34_, 1);
v___x_53_ = lean_box(0);
v___x_54_ = lp_mathlib_Lean_Meta_preservingMCtx___redArg___lam__0(v_a_28_, v_mctx_33_, v___x_53_);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_54_);
if (v_isSharedCheck_61_ == 0)
{
lean_object* v_unused_62_; 
v_unused_62_ = lean_ctor_get(v___x_54_, 0);
lean_dec(v_unused_62_);
v___x_56_ = v___x_54_;
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
else
{
lean_dec(v___x_54_);
v___x_56_ = lean_box(0);
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
v_resetjp_55_:
{
lean_object* v___x_59_; 
if (v_isShared_57_ == 0)
{
lean_ctor_set_tag(v___x_56_, 1);
lean_ctor_set(v___x_56_, 0, v_a_52_);
v___x_59_ = v___x_56_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v_a_52_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___redArg___boxed(lean_object* v_x_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Lean_Meta_preservingMCtx___redArg(v_x_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
lean_dec(v_a_67_);
lean_dec_ref(v_a_66_);
lean_dec(v_a_65_);
lean_dec_ref(v_a_64_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx(lean_object* v_00_u03b1_70_, lean_object* v_x_71_, lean_object* v_a_72_, lean_object* v_a_73_, lean_object* v_a_74_, lean_object* v_a_75_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_Lean_Meta_preservingMCtx___redArg(v_x_71_, v_a_72_, v_a_73_, v_a_74_, v_a_75_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_preservingMCtx___boxed(lean_object* v_00_u03b1_78_, lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_Lean_Meta_preservingMCtx(v_00_u03b1_78_, v_x_79_, v_a_80_, v_a_81_, v_a_82_, v_a_83_);
lean_dec(v_a_83_);
lean_dec_ref(v_a_82_);
lean_dec(v_a_81_);
lean_dec_ref(v_a_80_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0_spec__0(lean_object* v_msgData_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_){
_start:
{
lean_object* v___x_92_; lean_object* v_env_93_; lean_object* v___x_94_; lean_object* v_mctx_95_; lean_object* v_lctx_96_; lean_object* v_options_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_92_ = lean_st_ref_get(v___y_90_);
v_env_93_ = lean_ctor_get(v___x_92_, 0);
lean_inc_ref(v_env_93_);
lean_dec(v___x_92_);
v___x_94_ = lean_st_ref_get(v___y_88_);
v_mctx_95_ = lean_ctor_get(v___x_94_, 0);
lean_inc_ref(v_mctx_95_);
lean_dec(v___x_94_);
v_lctx_96_ = lean_ctor_get(v___y_87_, 2);
v_options_97_ = lean_ctor_get(v___y_89_, 2);
lean_inc_ref(v_options_97_);
lean_inc_ref(v_lctx_96_);
v___x_98_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_98_, 0, v_env_93_);
lean_ctor_set(v___x_98_, 1, v_mctx_95_);
lean_ctor_set(v___x_98_, 2, v_lctx_96_);
lean_ctor_set(v___x_98_, 3, v_options_97_);
v___x_99_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_msgData_86_);
v___x_100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0_spec__0___boxed(lean_object* v_msgData_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0_spec__0(v_msgData_101_, v___y_102_, v___y_103_, v___y_104_, v___y_105_);
lean_dec(v___y_105_);
lean_dec_ref(v___y_104_);
lean_dec(v___y_103_);
lean_dec_ref(v___y_102_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(lean_object* v_msg_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
lean_object* v_ref_114_; lean_object* v___x_115_; lean_object* v_a_116_; lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_124_; 
v_ref_114_ = lean_ctor_get(v___y_111_, 5);
v___x_115_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0_spec__0(v_msg_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
v_a_116_ = lean_ctor_get(v___x_115_, 0);
v_isSharedCheck_124_ = !lean_is_exclusive(v___x_115_);
if (v_isSharedCheck_124_ == 0)
{
v___x_118_ = v___x_115_;
v_isShared_119_ = v_isSharedCheck_124_;
goto v_resetjp_117_;
}
else
{
lean_inc(v_a_116_);
lean_dec(v___x_115_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_124_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
lean_object* v___x_120_; lean_object* v___x_122_; 
lean_inc(v_ref_114_);
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v_ref_114_);
lean_ctor_set(v___x_120_, 1, v_a_116_);
if (v_isShared_119_ == 0)
{
lean_ctor_set_tag(v___x_118_, 1);
lean_ctor_set(v___x_118_, 0, v___x_120_);
v___x_122_ = v___x_118_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v___x_120_);
v___x_122_ = v_reuseFailAlloc_123_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
return v___x_122_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg___boxed(lean_object* v_msg_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v_msg_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___lam__0(lean_object* v_fst_132_, lean_object* v_fst_133_, lean_object* v_fst_134_, lean_object* v_fst_135_, lean_object* v_snd_136_, lean_object* v_____r_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_143_ = l_Array_append___redArg(v_fst_132_, v_fst_133_);
v___x_144_ = l_Array_append___redArg(v_fst_134_, v_fst_135_);
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_144_);
lean_ctor_set(v___x_145_, 1, v_snd_136_);
v___x_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_143_);
lean_ctor_set(v___x_146_, 1, v___x_145_);
v___x_147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
v___x_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___lam__0___boxed(lean_object* v_fst_149_, lean_object* v_fst_150_, lean_object* v_fst_151_, lean_object* v_fst_152_, lean_object* v_snd_153_, lean_object* v_____r_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___lam__0(v_fst_149_, v_fst_150_, v_fst_151_, v_fst_152_, v_snd_153_, v_____r_154_, v___y_155_, v___y_156_, v___y_157_, v___y_158_);
lean_dec(v___y_158_);
lean_dec_ref(v___y_157_);
lean_dec(v___y_156_);
lean_dec_ref(v___y_155_);
lean_dec_ref(v_fst_152_);
lean_dec_ref(v_fst_150_);
return v_res_160_;
}
}
static lean_object* _init_lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__2(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = ((lean_object*)(lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__1));
v___x_165_ = l_Lean_stringToMessageData(v___x_164_);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__4(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_167_ = ((lean_object*)(lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__3));
v___x_168_ = l_Lean_stringToMessageData(v___x_167_);
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__6(void){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__5));
v___x_171_ = l_Lean_stringToMessageData(v___x_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg(lean_object* v_t_172_, uint8_t v_kind_173_, lean_object* v_e_174_, lean_object* v_a_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_){
_start:
{
lean_object* v___y_182_; lean_object* v_fst_202_; lean_object* v_snd_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_316_; 
v_fst_202_ = lean_ctor_get(v_a_175_, 0);
v_snd_203_ = lean_ctor_get(v_a_175_, 1);
v_isSharedCheck_316_ = !lean_is_exclusive(v_a_175_);
if (v_isSharedCheck_316_ == 0)
{
v___x_205_ = v_a_175_;
v_isShared_206_ = v_isSharedCheck_316_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_snd_203_);
lean_inc(v_fst_202_);
lean_dec(v_a_175_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_316_;
goto v_resetjp_204_;
}
v___jp_181_:
{
if (lean_obj_tag(v___y_182_) == 0)
{
lean_object* v_a_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_193_; 
v_a_183_ = lean_ctor_get(v___y_182_, 0);
v_isSharedCheck_193_ = !lean_is_exclusive(v___y_182_);
if (v_isSharedCheck_193_ == 0)
{
v___x_185_ = v___y_182_;
v_isShared_186_ = v_isSharedCheck_193_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_a_183_);
lean_dec(v___y_182_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_193_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
if (lean_obj_tag(v_a_183_) == 0)
{
lean_object* v_a_187_; lean_object* v___x_189_; 
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_a_187_ = lean_ctor_get(v_a_183_, 0);
lean_inc(v_a_187_);
lean_dec_ref_known(v_a_183_, 1);
if (v_isShared_186_ == 0)
{
lean_ctor_set(v___x_185_, 0, v_a_187_);
v___x_189_ = v___x_185_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_190_; 
v_reuseFailAlloc_190_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_190_, 0, v_a_187_);
v___x_189_ = v_reuseFailAlloc_190_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
return v___x_189_;
}
}
else
{
lean_object* v_a_191_; 
lean_del_object(v___x_185_);
v_a_191_ = lean_ctor_get(v_a_183_, 0);
lean_inc(v_a_191_);
lean_dec_ref_known(v_a_183_, 1);
v_a_175_ = v_a_191_;
goto _start;
}
}
}
else
{
lean_object* v_a_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_201_; 
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_a_194_ = lean_ctor_get(v___y_182_, 0);
v_isSharedCheck_201_ = !lean_is_exclusive(v___y_182_);
if (v_isSharedCheck_201_ == 0)
{
v___x_196_ = v___y_182_;
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_a_194_);
lean_dec(v___y_182_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_199_; 
if (v_isShared_197_ == 0)
{
v___x_199_ = v___x_196_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v_a_194_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
return v___x_199_;
}
}
}
}
v_resetjp_204_:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_207_ = l_Lean_instInhabitedExpr;
lean_inc(v_fst_202_);
v___x_208_ = lean_array_to_list(v_fst_202_);
v___x_209_ = l_List_getLast_x21___redArg(v___x_207_, v___x_208_);
lean_dec(v___x_208_);
lean_inc(v___y_179_);
lean_inc_ref(v___y_178_);
lean_inc(v___y_177_);
lean_inc_ref(v___y_176_);
v___x_210_ = lean_infer_type(v___x_209_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_object* v_a_211_; lean_object* v___x_212_; 
v_a_211_ = lean_ctor_get(v___x_210_, 0);
lean_inc(v_a_211_);
lean_dec_ref_known(v___x_210_, 1);
lean_inc_ref(v_t_172_);
v___x_212_ = l_Lean_Meta_isExprDefEq(v_a_211_, v_t_172_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
if (lean_obj_tag(v___x_212_) == 0)
{
lean_object* v_a_213_; lean_object* v___x_215_; uint8_t v_isShared_216_; uint8_t v_isSharedCheck_299_; 
v_a_213_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_299_ == 0)
{
v___x_215_ = v___x_212_;
v_isShared_216_ = v_isSharedCheck_299_;
goto v_resetjp_214_;
}
else
{
lean_inc(v_a_213_);
lean_dec(v___x_212_);
v___x_215_ = lean_box(0);
v_isShared_216_ = v_isSharedCheck_299_;
goto v_resetjp_214_;
}
v_resetjp_214_:
{
uint8_t v___x_217_; 
v___x_217_ = lean_unbox(v_a_213_);
lean_dec(v_a_213_);
if (v___x_217_ == 0)
{
lean_object* v_fst_218_; lean_object* v_snd_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
lean_del_object(v___x_215_);
lean_del_object(v___x_205_);
v_fst_218_ = lean_ctor_get(v_snd_203_, 0);
lean_inc(v_fst_218_);
v_snd_219_ = lean_ctor_get(v_snd_203_, 1);
lean_inc(v_snd_219_);
lean_dec(v_snd_203_);
v___x_220_ = lean_unsigned_to_nat(1u);
v___x_221_ = ((lean_object*)(lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__0));
v___x_222_ = l_Lean_Meta_forallMetaTelescopeReducing(v_snd_219_, v___x_221_, v_kind_173_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
if (lean_obj_tag(v___x_222_) == 0)
{
lean_object* v_a_223_; lean_object* v_snd_224_; lean_object* v_fst_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_283_; 
v_a_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_a_223_);
lean_dec_ref_known(v___x_222_, 1);
v_snd_224_ = lean_ctor_get(v_a_223_, 1);
v_fst_225_ = lean_ctor_get(v_a_223_, 0);
v_isSharedCheck_283_ = !lean_is_exclusive(v_a_223_);
if (v_isSharedCheck_283_ == 0)
{
v___x_227_ = v_a_223_;
v_isShared_228_ = v_isSharedCheck_283_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_snd_224_);
lean_inc(v_fst_225_);
lean_dec(v_a_223_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_283_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v_fst_229_; lean_object* v_snd_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_282_; 
v_fst_229_ = lean_ctor_get(v_snd_224_, 0);
v_snd_230_ = lean_ctor_get(v_snd_224_, 1);
v_isSharedCheck_282_ = !lean_is_exclusive(v_snd_224_);
if (v_isSharedCheck_282_ == 0)
{
v___x_232_ = v_snd_224_;
v_isShared_233_ = v_isSharedCheck_282_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_snd_230_);
lean_inc(v_fst_229_);
lean_dec(v_snd_224_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_282_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___x_234_; uint8_t v___x_235_; 
v___x_234_ = lean_array_get_size(v_fst_225_);
v___x_235_ = lean_nat_dec_eq(v___x_234_, v___x_220_);
if (v___x_235_ == 0)
{
lean_object* v___x_236_; 
lean_inc_ref(v_t_172_);
v___x_236_ = l_Lean_Meta_ppExpr(v_t_172_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
if (lean_obj_tag(v___x_236_) == 0)
{
lean_object* v_a_237_; lean_object* v___x_238_; 
v_a_237_ = lean_ctor_get(v___x_236_, 0);
lean_inc(v_a_237_);
lean_dec_ref_known(v___x_236_, 1);
lean_inc_ref(v_e_174_);
v___x_238_ = l_Lean_Meta_ppExpr(v_e_174_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
if (lean_obj_tag(v___x_238_) == 0)
{
lean_object* v_a_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_243_; 
v_a_239_ = lean_ctor_get(v___x_238_, 0);
lean_inc(v_a_239_);
lean_dec_ref_known(v___x_238_, 1);
v___x_240_ = lean_obj_once(&lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__2, &lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__2_once, _init_lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__2);
v___x_241_ = l_Lean_MessageData_ofFormat(v_a_237_);
if (v_isShared_233_ == 0)
{
lean_ctor_set_tag(v___x_232_, 7);
lean_ctor_set(v___x_232_, 1, v___x_241_);
lean_ctor_set(v___x_232_, 0, v___x_240_);
v___x_243_ = v___x_232_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v___x_240_);
lean_ctor_set(v_reuseFailAlloc_263_, 1, v___x_241_);
v___x_243_ = v_reuseFailAlloc_263_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
lean_object* v___x_244_; lean_object* v___x_246_; 
v___x_244_ = lean_obj_once(&lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__4, &lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__4_once, _init_lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__4);
if (v_isShared_228_ == 0)
{
lean_ctor_set_tag(v___x_227_, 7);
lean_ctor_set(v___x_227_, 1, v___x_244_);
lean_ctor_set(v___x_227_, 0, v___x_243_);
v___x_246_ = v___x_227_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_262_, 1, v___x_244_);
v___x_246_ = v_reuseFailAlloc_262_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_247_ = l_Lean_MessageData_ofFormat(v_a_239_);
v___x_248_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_246_);
lean_ctor_set(v___x_248_, 1, v___x_247_);
v___x_249_ = lean_obj_once(&lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__6, &lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__6_once, _init_lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__6);
v___x_250_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_248_);
lean_ctor_set(v___x_250_, 1, v___x_249_);
v___x_251_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v___x_250_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
if (lean_obj_tag(v___x_251_) == 0)
{
lean_object* v_a_252_; lean_object* v___x_253_; 
v_a_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc(v_a_252_);
lean_dec_ref_known(v___x_251_, 1);
v___x_253_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___lam__0(v_fst_202_, v_fst_225_, v_fst_218_, v_fst_229_, v_snd_230_, v_a_252_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
lean_dec(v_fst_229_);
lean_dec(v_fst_225_);
v___y_182_ = v___x_253_;
goto v___jp_181_;
}
else
{
lean_object* v_a_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_261_; 
lean_dec(v_snd_230_);
lean_dec(v_fst_229_);
lean_dec(v_fst_225_);
lean_dec(v_fst_218_);
lean_dec(v_fst_202_);
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_a_254_ = lean_ctor_get(v___x_251_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v___x_251_);
if (v_isSharedCheck_261_ == 0)
{
v___x_256_ = v___x_251_;
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_a_254_);
lean_dec(v___x_251_);
v___x_256_ = lean_box(0);
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
v_resetjp_255_:
{
lean_object* v___x_259_; 
if (v_isShared_257_ == 0)
{
v___x_259_ = v___x_256_;
goto v_reusejp_258_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v_a_254_);
v___x_259_ = v_reuseFailAlloc_260_;
goto v_reusejp_258_;
}
v_reusejp_258_:
{
return v___x_259_;
}
}
}
}
}
}
else
{
lean_object* v_a_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_271_; 
lean_dec(v_a_237_);
lean_del_object(v___x_232_);
lean_dec(v_snd_230_);
lean_dec(v_fst_229_);
lean_del_object(v___x_227_);
lean_dec(v_fst_225_);
lean_dec(v_fst_218_);
lean_dec(v_fst_202_);
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_a_264_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_271_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_271_ == 0)
{
v___x_266_ = v___x_238_;
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_a_264_);
lean_dec(v___x_238_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___x_269_; 
if (v_isShared_267_ == 0)
{
v___x_269_ = v___x_266_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v_a_264_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
return v___x_269_;
}
}
}
}
else
{
lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_279_; 
lean_del_object(v___x_232_);
lean_dec(v_snd_230_);
lean_dec(v_fst_229_);
lean_del_object(v___x_227_);
lean_dec(v_fst_225_);
lean_dec(v_fst_218_);
lean_dec(v_fst_202_);
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_a_272_ = lean_ctor_get(v___x_236_, 0);
v_isSharedCheck_279_ = !lean_is_exclusive(v___x_236_);
if (v_isSharedCheck_279_ == 0)
{
v___x_274_ = v___x_236_;
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_dec(v___x_236_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_277_; 
if (v_isShared_275_ == 0)
{
v___x_277_ = v___x_274_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v_a_272_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
else
{
lean_object* v___x_280_; lean_object* v___x_281_; 
lean_del_object(v___x_232_);
lean_del_object(v___x_227_);
v___x_280_ = lean_box(0);
v___x_281_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___lam__0(v_fst_202_, v_fst_225_, v_fst_218_, v_fst_229_, v_snd_230_, v___x_280_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
lean_dec(v_fst_229_);
lean_dec(v_fst_225_);
v___y_182_ = v___x_281_;
goto v___jp_181_;
}
}
}
}
else
{
lean_dec(v_fst_218_);
lean_dec(v_fst_202_);
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
return v___x_222_;
}
}
else
{
lean_object* v_fst_284_; lean_object* v_snd_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_298_; 
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_fst_284_ = lean_ctor_get(v_snd_203_, 0);
v_snd_285_ = lean_ctor_get(v_snd_203_, 1);
v_isSharedCheck_298_ = !lean_is_exclusive(v_snd_203_);
if (v_isSharedCheck_298_ == 0)
{
v___x_287_ = v_snd_203_;
v_isShared_288_ = v_isSharedCheck_298_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_snd_285_);
lean_inc(v_fst_284_);
lean_dec(v_snd_203_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_298_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_290_; 
if (v_isShared_288_ == 0)
{
v___x_290_ = v___x_287_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v_fst_284_);
lean_ctor_set(v_reuseFailAlloc_297_, 1, v_snd_285_);
v___x_290_ = v_reuseFailAlloc_297_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
lean_object* v___x_292_; 
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 1, v___x_290_);
v___x_292_ = v___x_205_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v_fst_202_);
lean_ctor_set(v_reuseFailAlloc_296_, 1, v___x_290_);
v___x_292_ = v_reuseFailAlloc_296_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
lean_object* v___x_294_; 
if (v_isShared_216_ == 0)
{
lean_ctor_set(v___x_215_, 0, v___x_292_);
v___x_294_ = v___x_215_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_295_; 
v_reuseFailAlloc_295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_295_, 0, v___x_292_);
v___x_294_ = v_reuseFailAlloc_295_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
return v___x_294_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_307_; 
lean_del_object(v___x_205_);
lean_dec(v_snd_203_);
lean_dec(v_fst_202_);
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_a_300_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_307_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_307_ == 0)
{
v___x_302_ = v___x_212_;
v_isShared_303_ = v_isSharedCheck_307_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_a_300_);
lean_dec(v___x_212_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_307_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v___x_305_; 
if (v_isShared_303_ == 0)
{
v___x_305_ = v___x_302_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v_a_300_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
}
}
else
{
lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_315_; 
lean_del_object(v___x_205_);
lean_dec(v_snd_203_);
lean_dec(v_fst_202_);
lean_dec_ref(v_e_174_);
lean_dec_ref(v_t_172_);
v_a_308_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_315_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_315_ == 0)
{
v___x_310_ = v___x_210_;
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_210_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_313_; 
if (v_isShared_311_ == 0)
{
v___x_313_ = v___x_310_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_314_; 
v_reuseFailAlloc_314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_314_, 0, v_a_308_);
v___x_313_ = v_reuseFailAlloc_314_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
return v___x_313_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___boxed(lean_object* v_t_317_, lean_object* v_kind_318_, lean_object* v_e_319_, lean_object* v_a_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
uint8_t v_kind_boxed_326_; lean_object* v_res_327_; 
v_kind_boxed_326_ = lean_unbox(v_kind_318_);
v_res_327_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg(v_t_317_, v_kind_boxed_326_, v_e_319_, v_a_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
return v_res_327_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__1(void){
_start:
{
lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_329_ = ((lean_object*)(lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__0));
v___x_330_ = l_Lean_stringToMessageData(v___x_329_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__3(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = ((lean_object*)(lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__2));
v___x_333_ = l_Lean_stringToMessageData(v___x_332_);
return v___x_333_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__5(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__4));
v___x_336_ = l_Lean_stringToMessageData(v___x_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq(lean_object* v_e_337_, lean_object* v_t_338_, uint8_t v_kind_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_345_ = lean_unsigned_to_nat(1u);
v___x_346_ = ((lean_object*)(lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg___closed__0));
lean_inc_ref(v_e_337_);
v___x_347_ = l_Lean_Meta_forallMetaTelescopeReducing(v_e_337_, v___x_346_, v_kind_339_, v_a_340_, v_a_341_, v_a_342_, v_a_343_);
if (lean_obj_tag(v___x_347_) == 0)
{
lean_object* v_a_348_; lean_object* v___y_350_; lean_object* v___y_351_; lean_object* v___y_352_; lean_object* v___y_353_; lean_object* v_fst_381_; lean_object* v___x_382_; uint8_t v___x_383_; 
v_a_348_ = lean_ctor_get(v___x_347_, 0);
lean_inc(v_a_348_);
lean_dec_ref_known(v___x_347_, 1);
v_fst_381_ = lean_ctor_get(v_a_348_, 0);
v___x_382_ = lean_array_get_size(v_fst_381_);
v___x_383_ = lean_nat_dec_eq(v___x_382_, v___x_345_);
if (v___x_383_ == 0)
{
lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_425_; 
lean_dec_ref(v_t_338_);
v_isSharedCheck_425_ = !lean_is_exclusive(v_a_348_);
if (v_isSharedCheck_425_ == 0)
{
lean_object* v_unused_426_; lean_object* v_unused_427_; 
v_unused_426_ = lean_ctor_get(v_a_348_, 1);
lean_dec(v_unused_426_);
v_unused_427_ = lean_ctor_get(v_a_348_, 0);
lean_dec(v_unused_427_);
v___x_385_ = v_a_348_;
v_isShared_386_ = v_isSharedCheck_425_;
goto v_resetjp_384_;
}
else
{
lean_dec(v_a_348_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_425_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_387_; uint8_t v___x_388_; 
v___x_387_ = lean_unsigned_to_nat(0u);
v___x_388_ = lean_nat_dec_eq(v___x_382_, v___x_387_);
if (v___x_388_ == 0)
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v_a_391_; lean_object* v___x_393_; uint8_t v_isShared_394_; uint8_t v_isSharedCheck_398_; 
lean_del_object(v___x_385_);
lean_dec_ref(v_e_337_);
v___x_389_ = lean_obj_once(&lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__1, &lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__1_once, _init_lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__1);
v___x_390_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v___x_389_, v_a_340_, v_a_341_, v_a_342_, v_a_343_);
v_a_391_ = lean_ctor_get(v___x_390_, 0);
v_isSharedCheck_398_ = !lean_is_exclusive(v___x_390_);
if (v_isSharedCheck_398_ == 0)
{
v___x_393_ = v___x_390_;
v_isShared_394_ = v_isSharedCheck_398_;
goto v_resetjp_392_;
}
else
{
lean_inc(v_a_391_);
lean_dec(v___x_390_);
v___x_393_ = lean_box(0);
v_isShared_394_ = v_isSharedCheck_398_;
goto v_resetjp_392_;
}
v_resetjp_392_:
{
lean_object* v___x_396_; 
if (v_isShared_394_ == 0)
{
v___x_396_ = v___x_393_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v_a_391_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
else
{
lean_object* v___x_399_; 
v___x_399_ = l_Lean_Meta_ppExpr(v_e_337_, v_a_340_, v_a_341_, v_a_342_, v_a_343_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_404_; 
v_a_400_ = lean_ctor_get(v___x_399_, 0);
lean_inc(v_a_400_);
lean_dec_ref_known(v___x_399_, 1);
v___x_401_ = lean_obj_once(&lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__3, &lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__3_once, _init_lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__3);
v___x_402_ = l_Lean_MessageData_ofFormat(v_a_400_);
if (v_isShared_386_ == 0)
{
lean_ctor_set_tag(v___x_385_, 7);
lean_ctor_set(v___x_385_, 1, v___x_402_);
lean_ctor_set(v___x_385_, 0, v___x_401_);
v___x_404_ = v___x_385_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v___x_401_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v___x_402_);
v___x_404_ = v_reuseFailAlloc_416_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v_a_408_; lean_object* v___x_410_; uint8_t v_isShared_411_; uint8_t v_isSharedCheck_415_; 
v___x_405_ = lean_obj_once(&lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__5, &lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__5_once, _init_lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___closed__5);
v___x_406_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_404_);
lean_ctor_set(v___x_406_, 1, v___x_405_);
v___x_407_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v___x_406_, v_a_340_, v_a_341_, v_a_342_, v_a_343_);
v_a_408_ = lean_ctor_get(v___x_407_, 0);
v_isSharedCheck_415_ = !lean_is_exclusive(v___x_407_);
if (v_isSharedCheck_415_ == 0)
{
v___x_410_ = v___x_407_;
v_isShared_411_ = v_isSharedCheck_415_;
goto v_resetjp_409_;
}
else
{
lean_inc(v_a_408_);
lean_dec(v___x_407_);
v___x_410_ = lean_box(0);
v_isShared_411_ = v_isSharedCheck_415_;
goto v_resetjp_409_;
}
v_resetjp_409_:
{
lean_object* v___x_413_; 
if (v_isShared_411_ == 0)
{
v___x_413_ = v___x_410_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_a_408_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
return v___x_413_;
}
}
}
}
else
{
lean_object* v_a_417_; lean_object* v___x_419_; uint8_t v_isShared_420_; uint8_t v_isSharedCheck_424_; 
lean_del_object(v___x_385_);
v_a_417_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_424_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_424_ == 0)
{
v___x_419_ = v___x_399_;
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
else
{
lean_inc(v_a_417_);
lean_dec(v___x_399_);
v___x_419_ = lean_box(0);
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
v_resetjp_418_:
{
lean_object* v___x_422_; 
if (v_isShared_420_ == 0)
{
v___x_422_ = v___x_419_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_a_417_);
v___x_422_ = v_reuseFailAlloc_423_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
return v___x_422_;
}
}
}
}
}
}
else
{
v___y_350_ = v_a_340_;
v___y_351_ = v_a_341_;
v___y_352_ = v_a_342_;
v___y_353_ = v_a_343_;
goto v___jp_349_;
}
v___jp_349_:
{
lean_object* v___x_354_; 
v___x_354_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg(v_t_338_, v_kind_339_, v_e_337_, v_a_348_, v___y_350_, v___y_351_, v___y_352_, v___y_353_);
if (lean_obj_tag(v___x_354_) == 0)
{
lean_object* v_a_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_380_; 
v_a_355_ = lean_ctor_get(v___x_354_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v___x_354_);
if (v_isSharedCheck_380_ == 0)
{
v___x_357_ = v___x_354_;
v_isShared_358_ = v_isSharedCheck_380_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_a_355_);
lean_dec(v___x_354_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_380_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v_snd_359_; lean_object* v_fst_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_379_; 
v_snd_359_ = lean_ctor_get(v_a_355_, 1);
v_fst_360_ = lean_ctor_get(v_a_355_, 0);
v_isSharedCheck_379_ = !lean_is_exclusive(v_a_355_);
if (v_isSharedCheck_379_ == 0)
{
v___x_362_ = v_a_355_;
v_isShared_363_ = v_isSharedCheck_379_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_snd_359_);
lean_inc(v_fst_360_);
lean_dec(v_a_355_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_379_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v_fst_364_; lean_object* v_snd_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_378_; 
v_fst_364_ = lean_ctor_get(v_snd_359_, 0);
v_snd_365_ = lean_ctor_get(v_snd_359_, 1);
v_isSharedCheck_378_ = !lean_is_exclusive(v_snd_359_);
if (v_isSharedCheck_378_ == 0)
{
v___x_367_ = v_snd_359_;
v_isShared_368_ = v_isSharedCheck_378_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_snd_365_);
lean_inc(v_fst_364_);
lean_dec(v_snd_359_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_378_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
lean_object* v___x_370_; 
if (v_isShared_368_ == 0)
{
v___x_370_ = v___x_367_;
goto v_reusejp_369_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_fst_364_);
lean_ctor_set(v_reuseFailAlloc_377_, 1, v_snd_365_);
v___x_370_ = v_reuseFailAlloc_377_;
goto v_reusejp_369_;
}
v_reusejp_369_:
{
lean_object* v___x_372_; 
if (v_isShared_363_ == 0)
{
lean_ctor_set(v___x_362_, 1, v___x_370_);
v___x_372_ = v___x_362_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v_fst_360_);
lean_ctor_set(v_reuseFailAlloc_376_, 1, v___x_370_);
v___x_372_ = v_reuseFailAlloc_376_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
lean_object* v___x_374_; 
if (v_isShared_358_ == 0)
{
lean_ctor_set(v___x_357_, 0, v___x_372_);
v___x_374_ = v___x_357_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v___x_372_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
}
}
}
}
}
else
{
return v___x_354_;
}
}
}
else
{
lean_dec_ref(v_t_338_);
lean_dec_ref(v_e_337_);
return v___x_347_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq___boxed(lean_object* v_e_428_, lean_object* v_t_429_, lean_object* v_kind_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_){
_start:
{
uint8_t v_kind_boxed_436_; lean_object* v_res_437_; 
v_kind_boxed_436_ = lean_unbox(v_kind_430_);
v_res_437_ = lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq(v_e_428_, v_t_429_, v_kind_boxed_436_, v_a_431_, v_a_432_, v_a_433_, v_a_434_);
lean_dec(v_a_434_);
lean_dec_ref(v_a_433_);
lean_dec(v_a_432_);
lean_dec_ref(v_a_431_);
return v_res_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0(lean_object* v_00_u03b1_438_, lean_object* v_msg_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
lean_object* v___x_445_; 
v___x_445_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v_msg_439_, v___y_440_, v___y_441_, v___y_442_, v___y_443_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___boxed(lean_object* v_00_u03b1_446_, lean_object* v_msg_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0(v_00_u03b1_446_, v_msg_447_, v___y_448_, v___y_449_, v___y_450_, v___y_451_);
lean_dec(v___y_451_);
lean_dec_ref(v___y_450_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1(lean_object* v_t_454_, uint8_t v_kind_455_, lean_object* v_e_456_, lean_object* v_inst_457_, lean_object* v_a_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___redArg(v_t_454_, v_kind_455_, v_e_456_, v_a_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1___boxed(lean_object* v_t_465_, lean_object* v_kind_466_, lean_object* v_e_467_, lean_object* v_inst_468_, lean_object* v_a_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
uint8_t v_kind_boxed_475_; lean_object* v_res_476_; 
v_kind_boxed_475_ = lean_unbox(v_kind_466_);
v_res_476_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__1(v_t_465_, v_kind_boxed_475_, v_e_467_, v_inst_468_, v_a_469_, v___y_470_, v___y_471_, v___y_472_, v___y_473_);
lean_dec(v___y_473_);
lean_dec_ref(v___y_472_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
return v_res_476_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_pureIsDefEq___closed__0(void){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = l_instMonadEIO(lean_box(0));
return v___x_477_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_pureIsDefEq___closed__1(void){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; 
v___x_478_ = lean_obj_once(&lp_mathlib_Lean_Meta_pureIsDefEq___closed__0, &lp_mathlib_Lean_Meta_pureIsDefEq___closed__0_once, _init_lp_mathlib_Lean_Meta_pureIsDefEq___closed__0);
v___x_479_ = l_StateRefT_x27_instMonad___redArg(v___x_478_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_pureIsDefEq(lean_object* v_e_u2081_484_, lean_object* v_e_u2082_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_){
_start:
{
lean_object* v___x_491_; lean_object* v_toApplicative_492_; lean_object* v_toFunctor_493_; lean_object* v_toSeq_494_; lean_object* v_toSeqLeft_495_; lean_object* v_toSeqRight_496_; lean_object* v___f_497_; lean_object* v___f_498_; lean_object* v___f_499_; lean_object* v___f_500_; lean_object* v___x_501_; lean_object* v___f_502_; lean_object* v___f_503_; lean_object* v___f_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v_toApplicative_510_; lean_object* v_toFunctor_511_; lean_object* v_toSeq_512_; lean_object* v_toSeqLeft_513_; lean_object* v_toSeqRight_514_; lean_object* v___f_515_; lean_object* v___f_516_; lean_object* v___x_517_; lean_object* v___f_518_; lean_object* v___f_519_; lean_object* v___f_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v_toApplicative_524_; lean_object* v___x_526_; uint8_t v_isShared_527_; uint8_t v_isSharedCheck_555_; 
v___x_491_ = lean_obj_once(&lp_mathlib_Lean_Meta_pureIsDefEq___closed__1, &lp_mathlib_Lean_Meta_pureIsDefEq___closed__1_once, _init_lp_mathlib_Lean_Meta_pureIsDefEq___closed__1);
v_toApplicative_492_ = lean_ctor_get(v___x_491_, 0);
v_toFunctor_493_ = lean_ctor_get(v_toApplicative_492_, 0);
v_toSeq_494_ = lean_ctor_get(v_toApplicative_492_, 2);
v_toSeqLeft_495_ = lean_ctor_get(v_toApplicative_492_, 3);
v_toSeqRight_496_ = lean_ctor_get(v_toApplicative_492_, 4);
v___f_497_ = ((lean_object*)(lp_mathlib_Lean_Meta_pureIsDefEq___closed__2));
v___f_498_ = ((lean_object*)(lp_mathlib_Lean_Meta_pureIsDefEq___closed__3));
lean_inc_ref_n(v_toFunctor_493_, 2);
v___f_499_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_499_, 0, v_toFunctor_493_);
v___f_500_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_500_, 0, v_toFunctor_493_);
v___x_501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_501_, 0, v___f_499_);
lean_ctor_set(v___x_501_, 1, v___f_500_);
lean_inc(v_toSeqRight_496_);
v___f_502_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_502_, 0, v_toSeqRight_496_);
lean_inc(v_toSeqLeft_495_);
v___f_503_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_503_, 0, v_toSeqLeft_495_);
lean_inc(v_toSeq_494_);
v___f_504_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_504_, 0, v_toSeq_494_);
v___x_505_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_505_, 0, v___x_501_);
lean_ctor_set(v___x_505_, 1, v___f_497_);
lean_ctor_set(v___x_505_, 2, v___f_504_);
lean_ctor_set(v___x_505_, 3, v___f_503_);
lean_ctor_set(v___x_505_, 4, v___f_502_);
v___x_506_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_506_, 0, v___x_505_);
lean_ctor_set(v___x_506_, 1, v___f_498_);
v___x_507_ = l_StateRefT_x27_instMonad___redArg(v___x_506_);
v___x_508_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_508_, 0, lean_box(0));
lean_closure_set(v___x_508_, 1, lean_box(0));
lean_closure_set(v___x_508_, 2, v___x_507_);
v___x_509_ = l_instMonadControlTOfPure___redArg(v___x_508_);
v_toApplicative_510_ = lean_ctor_get(v___x_491_, 0);
v_toFunctor_511_ = lean_ctor_get(v_toApplicative_510_, 0);
v_toSeq_512_ = lean_ctor_get(v_toApplicative_510_, 2);
v_toSeqLeft_513_ = lean_ctor_get(v_toApplicative_510_, 3);
v_toSeqRight_514_ = lean_ctor_get(v_toApplicative_510_, 4);
lean_inc_ref_n(v_toFunctor_511_, 2);
v___f_515_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_515_, 0, v_toFunctor_511_);
v___f_516_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_516_, 0, v_toFunctor_511_);
v___x_517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_517_, 0, v___f_515_);
lean_ctor_set(v___x_517_, 1, v___f_516_);
lean_inc(v_toSeqRight_514_);
v___f_518_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_518_, 0, v_toSeqRight_514_);
lean_inc(v_toSeqLeft_513_);
v___f_519_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_519_, 0, v_toSeqLeft_513_);
lean_inc(v_toSeq_512_);
v___f_520_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_520_, 0, v_toSeq_512_);
v___x_521_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_521_, 0, v___x_517_);
lean_ctor_set(v___x_521_, 1, v___f_497_);
lean_ctor_set(v___x_521_, 2, v___f_520_);
lean_ctor_set(v___x_521_, 3, v___f_519_);
lean_ctor_set(v___x_521_, 4, v___f_518_);
v___x_522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_522_, 0, v___x_521_);
lean_ctor_set(v___x_522_, 1, v___f_498_);
v___x_523_ = l_StateRefT_x27_instMonad___redArg(v___x_522_);
v_toApplicative_524_ = lean_ctor_get(v___x_523_, 0);
v_isSharedCheck_555_ = !lean_is_exclusive(v___x_523_);
if (v_isSharedCheck_555_ == 0)
{
lean_object* v_unused_556_; 
v_unused_556_ = lean_ctor_get(v___x_523_, 1);
lean_dec(v_unused_556_);
v___x_526_ = v___x_523_;
v_isShared_527_ = v_isSharedCheck_555_;
goto v_resetjp_525_;
}
else
{
lean_inc(v_toApplicative_524_);
lean_dec(v___x_523_);
v___x_526_ = lean_box(0);
v_isShared_527_ = v_isSharedCheck_555_;
goto v_resetjp_525_;
}
v_resetjp_525_:
{
lean_object* v_toFunctor_528_; lean_object* v_toSeq_529_; lean_object* v_toSeqLeft_530_; lean_object* v_toSeqRight_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_553_; 
v_toFunctor_528_ = lean_ctor_get(v_toApplicative_524_, 0);
v_toSeq_529_ = lean_ctor_get(v_toApplicative_524_, 2);
v_toSeqLeft_530_ = lean_ctor_get(v_toApplicative_524_, 3);
v_toSeqRight_531_ = lean_ctor_get(v_toApplicative_524_, 4);
v_isSharedCheck_553_ = !lean_is_exclusive(v_toApplicative_524_);
if (v_isSharedCheck_553_ == 0)
{
lean_object* v_unused_554_; 
v_unused_554_ = lean_ctor_get(v_toApplicative_524_, 1);
lean_dec(v_unused_554_);
v___x_533_ = v_toApplicative_524_;
v_isShared_534_ = v_isSharedCheck_553_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_toSeqRight_531_);
lean_inc(v_toSeqLeft_530_);
lean_inc(v_toSeq_529_);
lean_inc(v_toFunctor_528_);
lean_dec(v_toApplicative_524_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_553_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v___f_535_; lean_object* v___f_536_; lean_object* v___f_537_; lean_object* v___f_538_; lean_object* v___x_539_; lean_object* v___f_540_; lean_object* v___f_541_; lean_object* v___f_542_; lean_object* v___x_544_; 
v___f_535_ = ((lean_object*)(lp_mathlib_Lean_Meta_pureIsDefEq___closed__4));
v___f_536_ = ((lean_object*)(lp_mathlib_Lean_Meta_pureIsDefEq___closed__5));
lean_inc_ref(v_toFunctor_528_);
v___f_537_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_537_, 0, v_toFunctor_528_);
v___f_538_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_538_, 0, v_toFunctor_528_);
v___x_539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_539_, 0, v___f_537_);
lean_ctor_set(v___x_539_, 1, v___f_538_);
v___f_540_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_540_, 0, v_toSeqRight_531_);
v___f_541_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_541_, 0, v_toSeqLeft_530_);
v___f_542_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_542_, 0, v_toSeq_529_);
if (v_isShared_534_ == 0)
{
lean_ctor_set(v___x_533_, 4, v___f_540_);
lean_ctor_set(v___x_533_, 3, v___f_541_);
lean_ctor_set(v___x_533_, 2, v___f_542_);
lean_ctor_set(v___x_533_, 1, v___f_535_);
lean_ctor_set(v___x_533_, 0, v___x_539_);
v___x_544_ = v___x_533_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v___x_539_);
lean_ctor_set(v_reuseFailAlloc_552_, 1, v___f_535_);
lean_ctor_set(v_reuseFailAlloc_552_, 2, v___f_542_);
lean_ctor_set(v_reuseFailAlloc_552_, 3, v___f_541_);
lean_ctor_set(v_reuseFailAlloc_552_, 4, v___f_540_);
v___x_544_ = v_reuseFailAlloc_552_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
lean_object* v___x_546_; 
if (v_isShared_527_ == 0)
{
lean_ctor_set(v___x_526_, 1, v___f_536_);
lean_ctor_set(v___x_526_, 0, v___x_544_);
v___x_546_ = v___x_526_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v___x_544_);
lean_ctor_set(v_reuseFailAlloc_551_, 1, v___f_536_);
v___x_546_ = v_reuseFailAlloc_551_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
lean_object* v___x_547_; uint8_t v___x_548_; lean_object* v___x_10__overap_549_; lean_object* v___x_550_; 
v___x_547_ = lean_alloc_closure((void*)(l_Lean_Meta_isDefEq___boxed), 7, 2);
lean_closure_set(v___x_547_, 0, v_e_u2081_484_);
lean_closure_set(v___x_547_, 1, v_e_u2082_485_);
v___x_548_ = 0;
v___x_10__overap_549_ = l_Lean_Meta_withNewMCtxDepth___redArg(v___x_509_, v___x_546_, v___x_547_, v___x_548_);
lean_inc(v_a_489_);
lean_inc_ref(v_a_488_);
lean_inc(v_a_487_);
lean_inc_ref(v_a_486_);
v___x_550_ = lean_apply_5(v___x_10__overap_549_, v_a_486_, v_a_487_, v_a_488_, v_a_489_, lean_box(0));
return v___x_550_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_pureIsDefEq___boxed(lean_object* v_e_u2081_557_, lean_object* v_e_u2082_558_, lean_object* v_a_559_, lean_object* v_a_560_, lean_object* v_a_561_, lean_object* v_a_562_, lean_object* v_a_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_Lean_Meta_pureIsDefEq(v_e_u2081_557_, v_e_u2082_558_, v_a_559_, v_a_560_, v_a_561_, v_a_562_);
lean_dec(v_a_562_);
lean_dec_ref(v_a_561_);
lean_dec(v_a_560_);
lean_dec_ref(v_a_559_);
return v_res_564_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRel___closed__4(void){
_start:
{
lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_571_ = lean_box(0);
v___x_572_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRel___closed__3));
v___x_573_ = l_Lean_Expr_const___override(v___x_572_, v___x_571_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRel(lean_object* v_n_574_, lean_object* v_lhs_575_, lean_object* v_rhs_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_){
_start:
{
lean_object* v___x_582_; uint8_t v___x_583_; 
v___x_582_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRel___closed__1));
v___x_583_ = lean_name_eq(v_n_574_, v___x_582_);
if (v___x_583_ == 0)
{
lean_object* v___x_584_; uint8_t v___x_585_; 
v___x_584_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRel___closed__3));
v___x_585_ = lean_name_eq(v_n_574_, v___x_584_);
if (v___x_585_ == 0)
{
lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; 
v___x_586_ = lean_unsigned_to_nat(2u);
v___x_587_ = lean_mk_empty_array_with_capacity(v___x_586_);
v___x_588_ = lean_array_push(v___x_587_, v_lhs_575_);
v___x_589_ = lean_array_push(v___x_588_, v_rhs_576_);
v___x_590_ = l_Lean_Meta_mkAppM(v_n_574_, v___x_589_, v_a_577_, v_a_578_, v_a_579_, v_a_580_);
return v___x_590_;
}
else
{
lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
lean_dec(v_n_574_);
v___x_591_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRel___closed__4, &lp_mathlib_Lean_Meta_mkRel___closed__4_once, _init_lp_mathlib_Lean_Meta_mkRel___closed__4);
v___x_592_ = l_Lean_mkAppB(v___x_591_, v_lhs_575_, v_rhs_576_);
v___x_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
return v___x_593_;
}
}
else
{
lean_object* v___x_594_; 
lean_dec(v_n_574_);
v___x_594_ = l_Lean_Meta_mkEq(v_lhs_575_, v_rhs_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_);
return v___x_594_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRel___boxed(lean_object* v_n_595_, lean_object* v_lhs_596_, lean_object* v_rhs_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_mathlib_Lean_Meta_mkRel(v_n_595_, v_lhs_596_, v_rhs_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_);
lean_dec(v_a_601_);
lean_dec_ref(v_a_600_);
lean_dec(v_a_599_);
lean_dec_ref(v_a_598_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___lam__0(lean_object* v_k_604_, lean_object* v_b_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_){
_start:
{
lean_object* v___x_611_; 
lean_inc(v___y_609_);
lean_inc_ref(v___y_608_);
lean_inc(v___y_607_);
lean_inc_ref(v___y_606_);
v___x_611_ = lean_apply_6(v_k_604_, v_b_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_, lean_box(0));
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___lam__0___boxed(lean_object* v_k_612_, lean_object* v_b_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___lam__0(v_k_612_, v_b_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_);
lean_dec(v___y_617_);
lean_dec_ref(v___y_616_);
lean_dec(v___y_615_);
lean_dec_ref(v___y_614_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg(lean_object* v_name_620_, lean_object* v_type_621_, lean_object* v_val_622_, lean_object* v_k_623_, uint8_t v_nondep_624_, uint8_t v_kind_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_){
_start:
{
lean_object* v___f_631_; lean_object* v___x_632_; 
v___f_631_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_631_, 0, v_k_623_);
v___x_632_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_620_, v_type_621_, v_val_622_, v___f_631_, v_nondep_624_, v_kind_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_);
if (lean_obj_tag(v___x_632_) == 0)
{
lean_object* v_a_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_640_; 
v_a_633_ = lean_ctor_get(v___x_632_, 0);
v_isSharedCheck_640_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_640_ == 0)
{
v___x_635_ = v___x_632_;
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_a_633_);
lean_dec(v___x_632_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_640_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_638_; 
if (v_isShared_636_ == 0)
{
v___x_638_ = v___x_635_;
goto v_reusejp_637_;
}
else
{
lean_object* v_reuseFailAlloc_639_; 
v_reuseFailAlloc_639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_639_, 0, v_a_633_);
v___x_638_ = v_reuseFailAlloc_639_;
goto v_reusejp_637_;
}
v_reusejp_637_:
{
return v___x_638_;
}
}
}
else
{
lean_object* v_a_641_; lean_object* v___x_643_; uint8_t v_isShared_644_; uint8_t v_isSharedCheck_648_; 
v_a_641_ = lean_ctor_get(v___x_632_, 0);
v_isSharedCheck_648_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_648_ == 0)
{
v___x_643_ = v___x_632_;
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
else
{
lean_inc(v_a_641_);
lean_dec(v___x_632_);
v___x_643_ = lean_box(0);
v_isShared_644_ = v_isSharedCheck_648_;
goto v_resetjp_642_;
}
v_resetjp_642_:
{
lean_object* v___x_646_; 
if (v_isShared_644_ == 0)
{
v___x_646_ = v___x_643_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_647_; 
v_reuseFailAlloc_647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_647_, 0, v_a_641_);
v___x_646_ = v_reuseFailAlloc_647_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
return v___x_646_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg___boxed(lean_object* v_name_649_, lean_object* v_type_650_, lean_object* v_val_651_, lean_object* v_k_652_, lean_object* v_nondep_653_, lean_object* v_kind_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_){
_start:
{
uint8_t v_nondep_boxed_660_; uint8_t v_kind_boxed_661_; lean_object* v_res_662_; 
v_nondep_boxed_660_ = lean_unbox(v_nondep_653_);
v_kind_boxed_661_ = lean_unbox(v_kind_654_);
v_res_662_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg(v_name_649_, v_type_650_, v_val_651_, v_k_652_, v_nondep_boxed_660_, v_kind_boxed_661_, v___y_655_, v___y_656_, v___y_657_, v___y_658_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec(v___y_656_);
lean_dec_ref(v___y_655_);
return v_res_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0(lean_object* v_00_u03b1_663_, lean_object* v_name_664_, lean_object* v_type_665_, lean_object* v_val_666_, lean_object* v_k_667_, uint8_t v_nondep_668_, uint8_t v_kind_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_){
_start:
{
lean_object* v___x_675_; 
v___x_675_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg(v_name_664_, v_type_665_, v_val_666_, v_k_667_, v_nondep_668_, v_kind_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_);
return v___x_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___boxed(lean_object* v_00_u03b1_676_, lean_object* v_name_677_, lean_object* v_type_678_, lean_object* v_val_679_, lean_object* v_k_680_, lean_object* v_nondep_681_, lean_object* v_kind_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_){
_start:
{
uint8_t v_nondep_boxed_688_; uint8_t v_kind_boxed_689_; lean_object* v_res_690_; 
v_nondep_boxed_688_ = lean_unbox(v_nondep_681_);
v_kind_boxed_689_ = lean_unbox(v_kind_682_);
v_res_690_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0(v_00_u03b1_676_, v_name_677_, v_type_678_, v_val_679_, v_k_680_, v_nondep_boxed_688_, v_kind_boxed_689_, v___y_683_, v___y_684_, v___y_685_, v___y_686_);
lean_dec(v___y_686_);
lean_dec_ref(v___y_685_);
lean_dec(v___y_684_);
lean_dec_ref(v___y_683_);
return v_res_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___lam__0(lean_object* v_k_691_, lean_object* v_instE_692_, lean_object* v_inst_x27_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_){
_start:
{
lean_object* v___x_699_; 
lean_inc(v___y_697_);
lean_inc_ref(v___y_696_);
lean_inc(v___y_695_);
lean_inc_ref(v___y_694_);
v___x_699_ = lean_apply_5(v_k_691_, v___y_694_, v___y_695_, v___y_696_, v___y_697_, lean_box(0));
if (lean_obj_tag(v___x_699_) == 0)
{
lean_object* v_a_700_; lean_object* v_fst_701_; lean_object* v_snd_702_; lean_object* v___x_704_; uint8_t v_isShared_705_; uint8_t v_isSharedCheck_730_; 
v_a_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_a_700_);
lean_dec_ref_known(v___x_699_, 1);
v_fst_701_ = lean_ctor_get(v_a_700_, 0);
v_snd_702_ = lean_ctor_get(v_a_700_, 1);
v_isSharedCheck_730_ = !lean_is_exclusive(v_a_700_);
if (v_isSharedCheck_730_ == 0)
{
v___x_704_ = v_a_700_;
v_isShared_705_ = v_isSharedCheck_730_;
goto v_resetjp_703_;
}
else
{
lean_inc(v_snd_702_);
lean_inc(v_fst_701_);
lean_dec(v_a_700_);
v___x_704_ = lean_box(0);
v_isShared_705_ = v_isSharedCheck_730_;
goto v_resetjp_703_;
}
v_resetjp_703_:
{
lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; 
v___x_706_ = lean_unsigned_to_nat(1u);
v___x_707_ = lean_mk_empty_array_with_capacity(v___x_706_);
v___x_708_ = lean_array_push(v___x_707_, v_inst_x27_693_);
v___x_709_ = l_Lean_Expr_abstractM(v_fst_701_, v___x_708_, v___y_694_, v___y_695_, v___y_696_, v___y_697_);
lean_dec_ref(v___x_708_);
if (lean_obj_tag(v___x_709_) == 0)
{
lean_object* v_a_710_; lean_object* v___x_712_; uint8_t v_isShared_713_; uint8_t v_isSharedCheck_721_; 
v_a_710_ = lean_ctor_get(v___x_709_, 0);
v_isSharedCheck_721_ = !lean_is_exclusive(v___x_709_);
if (v_isSharedCheck_721_ == 0)
{
v___x_712_ = v___x_709_;
v_isShared_713_ = v_isSharedCheck_721_;
goto v_resetjp_711_;
}
else
{
lean_inc(v_a_710_);
lean_dec(v___x_709_);
v___x_712_ = lean_box(0);
v_isShared_713_ = v_isSharedCheck_721_;
goto v_resetjp_711_;
}
v_resetjp_711_:
{
lean_object* v___x_714_; lean_object* v___x_716_; 
v___x_714_ = lean_expr_instantiate1(v_a_710_, v_instE_692_);
lean_dec(v_a_710_);
if (v_isShared_705_ == 0)
{
lean_ctor_set(v___x_704_, 0, v___x_714_);
v___x_716_ = v___x_704_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_720_; 
v_reuseFailAlloc_720_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_720_, 0, v___x_714_);
lean_ctor_set(v_reuseFailAlloc_720_, 1, v_snd_702_);
v___x_716_ = v_reuseFailAlloc_720_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
lean_object* v___x_718_; 
if (v_isShared_713_ == 0)
{
lean_ctor_set(v___x_712_, 0, v___x_716_);
v___x_718_ = v___x_712_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_719_; 
v_reuseFailAlloc_719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_719_, 0, v___x_716_);
v___x_718_ = v_reuseFailAlloc_719_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
return v___x_718_;
}
}
}
}
else
{
lean_object* v_a_722_; lean_object* v___x_724_; uint8_t v_isShared_725_; uint8_t v_isSharedCheck_729_; 
lean_del_object(v___x_704_);
lean_dec(v_snd_702_);
v_a_722_ = lean_ctor_get(v___x_709_, 0);
v_isSharedCheck_729_ = !lean_is_exclusive(v___x_709_);
if (v_isSharedCheck_729_ == 0)
{
v___x_724_ = v___x_709_;
v_isShared_725_ = v_isSharedCheck_729_;
goto v_resetjp_723_;
}
else
{
lean_inc(v_a_722_);
lean_dec(v___x_709_);
v___x_724_ = lean_box(0);
v_isShared_725_ = v_isSharedCheck_729_;
goto v_resetjp_723_;
}
v_resetjp_723_:
{
lean_object* v___x_727_; 
if (v_isShared_725_ == 0)
{
v___x_727_ = v___x_724_;
goto v_reusejp_726_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v_a_722_);
v___x_727_ = v_reuseFailAlloc_728_;
goto v_reusejp_726_;
}
v_reusejp_726_:
{
return v___x_727_;
}
}
}
}
}
else
{
lean_dec_ref(v_inst_x27_693_);
return v___x_699_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___lam__0___boxed(lean_object* v_k_731_, lean_object* v_instE_732_, lean_object* v_inst_x27_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___lam__0(v_k_731_, v_instE_732_, v_inst_x27_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
lean_dec(v___y_735_);
lean_dec_ref(v___y_734_);
lean_dec_ref(v_instE_732_);
return v_res_739_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__1(void){
_start:
{
lean_object* v___x_741_; lean_object* v___x_742_; 
v___x_741_ = ((lean_object*)(lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__0));
v___x_742_ = l_Lean_stringToMessageData(v___x_741_);
return v___x_742_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__3(void){
_start:
{
lean_object* v___x_744_; lean_object* v___x_745_; 
v___x_744_ = ((lean_object*)(lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__2));
v___x_745_ = l_Lean_stringToMessageData(v___x_744_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg(lean_object* v_inst_749_, lean_object* v_k_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_){
_start:
{
lean_object* v_instE_756_; lean_object* v___x_757_; 
v_instE_756_ = l_Lean_mkMVar(v_inst_749_);
lean_inc(v_a_754_);
lean_inc_ref(v_a_753_);
lean_inc(v_a_752_);
lean_inc_ref(v_a_751_);
lean_inc_ref(v_instE_756_);
v___x_757_ = lean_infer_type(v_instE_756_, v_a_751_, v_a_752_, v_a_753_, v_a_754_);
if (lean_obj_tag(v___x_757_) == 0)
{
lean_object* v_a_758_; lean_object* v___x_759_; lean_object* v___x_760_; 
v_a_758_ = lean_ctor_get(v___x_757_, 0);
lean_inc(v_a_758_);
lean_dec_ref_known(v___x_757_, 1);
v___x_759_ = lean_box(0);
v___x_760_ = l_Lean_Meta_trySynthInstance(v_a_758_, v___x_759_, v_a_751_, v_a_752_, v_a_753_, v_a_754_);
if (lean_obj_tag(v___x_760_) == 0)
{
lean_object* v_a_761_; 
v_a_761_ = lean_ctor_get(v___x_760_, 0);
lean_inc(v_a_761_);
lean_dec_ref_known(v___x_760_, 1);
if (lean_obj_tag(v_a_761_) == 1)
{
lean_object* v_a_762_; lean_object* v___x_763_; 
v_a_762_ = lean_ctor_get(v_a_761_, 0);
lean_inc_n(v_a_762_, 2);
lean_dec_ref_known(v_a_761_, 1);
lean_inc_ref(v_instE_756_);
v___x_763_ = l_Lean_Meta_isExprDefEq(v_instE_756_, v_a_762_, v_a_751_, v_a_752_, v_a_753_, v_a_754_);
if (lean_obj_tag(v___x_763_) == 0)
{
lean_object* v_a_764_; uint8_t v___x_765_; 
v_a_764_ = lean_ctor_get(v___x_763_, 0);
lean_inc(v_a_764_);
lean_dec_ref_known(v___x_763_, 1);
v___x_765_ = lean_unbox(v_a_764_);
lean_dec(v_a_764_);
if (v___x_765_ == 0)
{
lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v_a_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_781_; 
lean_dec_ref(v_k_750_);
v___x_766_ = lean_obj_once(&lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__1, &lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__1);
v___x_767_ = l_Lean_indentExpr(v_a_762_);
v___x_768_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_768_, 0, v___x_766_);
lean_ctor_set(v___x_768_, 1, v___x_767_);
v___x_769_ = lean_obj_once(&lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__3, &lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__3);
v___x_770_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_770_, 0, v___x_768_);
lean_ctor_set(v___x_770_, 1, v___x_769_);
v___x_771_ = l_Lean_indentExpr(v_instE_756_);
v___x_772_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_772_, 0, v___x_770_);
lean_ctor_set(v___x_772_, 1, v___x_771_);
v___x_773_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v___x_772_, v_a_751_, v_a_752_, v_a_753_, v_a_754_);
v_a_774_ = lean_ctor_get(v___x_773_, 0);
v_isSharedCheck_781_ = !lean_is_exclusive(v___x_773_);
if (v_isSharedCheck_781_ == 0)
{
v___x_776_ = v___x_773_;
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_a_774_);
lean_dec(v___x_773_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_781_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___x_779_; 
if (v_isShared_777_ == 0)
{
v___x_779_ = v___x_776_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_780_, 0, v_a_774_);
v___x_779_ = v_reuseFailAlloc_780_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
return v___x_779_;
}
}
}
else
{
lean_object* v___x_782_; 
lean_dec(v_a_762_);
lean_dec_ref(v_instE_756_);
lean_inc(v_a_754_);
lean_inc_ref(v_a_753_);
lean_inc(v_a_752_);
lean_inc_ref(v_a_751_);
v___x_782_ = lean_apply_5(v_k_750_, v_a_751_, v_a_752_, v_a_753_, v_a_754_, lean_box(0));
return v___x_782_;
}
}
else
{
lean_object* v_a_783_; lean_object* v___x_785_; uint8_t v_isShared_786_; uint8_t v_isSharedCheck_790_; 
lean_dec(v_a_762_);
lean_dec_ref(v_instE_756_);
lean_dec_ref(v_k_750_);
v_a_783_ = lean_ctor_get(v___x_763_, 0);
v_isSharedCheck_790_ = !lean_is_exclusive(v___x_763_);
if (v_isSharedCheck_790_ == 0)
{
v___x_785_ = v___x_763_;
v_isShared_786_ = v_isSharedCheck_790_;
goto v_resetjp_784_;
}
else
{
lean_inc(v_a_783_);
lean_dec(v___x_763_);
v___x_785_ = lean_box(0);
v_isShared_786_ = v_isSharedCheck_790_;
goto v_resetjp_784_;
}
v_resetjp_784_:
{
lean_object* v___x_788_; 
if (v_isShared_786_ == 0)
{
v___x_788_ = v___x_785_;
goto v_reusejp_787_;
}
else
{
lean_object* v_reuseFailAlloc_789_; 
v_reuseFailAlloc_789_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_789_, 0, v_a_783_);
v___x_788_ = v_reuseFailAlloc_789_;
goto v_reusejp_787_;
}
v_reusejp_787_:
{
return v___x_788_;
}
}
}
}
else
{
lean_object* v___x_791_; 
lean_dec(v_a_761_);
lean_inc(v_a_754_);
lean_inc_ref(v_a_753_);
lean_inc(v_a_752_);
lean_inc_ref(v_a_751_);
lean_inc_ref(v_instE_756_);
v___x_791_ = lean_infer_type(v_instE_756_, v_a_751_, v_a_752_, v_a_753_, v_a_754_);
if (lean_obj_tag(v___x_791_) == 0)
{
lean_object* v_a_792_; lean_object* v___f_793_; lean_object* v___x_794_; uint8_t v___x_795_; uint8_t v___x_796_; lean_object* v___x_797_; 
v_a_792_ = lean_ctor_get(v___x_791_, 0);
lean_inc(v_a_792_);
lean_dec_ref_known(v___x_791_, 1);
lean_inc_ref(v_instE_756_);
v___f_793_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_793_, 0, v_k_750_);
lean_closure_set(v___f_793_, 1, v_instE_756_);
v___x_794_ = ((lean_object*)(lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___closed__5));
v___x_795_ = 0;
v___x_796_ = 0;
v___x_797_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_withEnsuringLocalInstance_spec__0___redArg(v___x_794_, v_a_792_, v_instE_756_, v___f_793_, v___x_795_, v___x_796_, v_a_751_, v_a_752_, v_a_753_, v_a_754_);
return v___x_797_;
}
else
{
lean_object* v_a_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_805_; 
lean_dec_ref(v_instE_756_);
lean_dec_ref(v_k_750_);
v_a_798_ = lean_ctor_get(v___x_791_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_805_ == 0)
{
v___x_800_ = v___x_791_;
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_a_798_);
lean_dec(v___x_791_);
v___x_800_ = lean_box(0);
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
v_resetjp_799_:
{
lean_object* v___x_803_; 
if (v_isShared_801_ == 0)
{
v___x_803_ = v___x_800_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v_a_798_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
}
}
else
{
lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_813_; 
lean_dec_ref(v_instE_756_);
lean_dec_ref(v_k_750_);
v_a_806_ = lean_ctor_get(v___x_760_, 0);
v_isSharedCheck_813_ = !lean_is_exclusive(v___x_760_);
if (v_isSharedCheck_813_ == 0)
{
v___x_808_ = v___x_760_;
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_760_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v___x_811_; 
if (v_isShared_809_ == 0)
{
v___x_811_ = v___x_808_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v_a_806_);
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
else
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_821_; 
lean_dec_ref(v_instE_756_);
lean_dec_ref(v_k_750_);
v_a_814_ = lean_ctor_get(v___x_757_, 0);
v_isSharedCheck_821_ = !lean_is_exclusive(v___x_757_);
if (v_isSharedCheck_821_ == 0)
{
v___x_816_ = v___x_757_;
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_757_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_821_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v___x_819_; 
if (v_isShared_817_ == 0)
{
v___x_819_ = v___x_816_;
goto v_reusejp_818_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v_a_814_);
v___x_819_ = v_reuseFailAlloc_820_;
goto v_reusejp_818_;
}
v_reusejp_818_:
{
return v___x_819_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg___boxed(lean_object* v_inst_822_, lean_object* v_k_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_){
_start:
{
lean_object* v_res_829_; 
v_res_829_ = lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg(v_inst_822_, v_k_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_);
lean_dec(v_a_827_);
lean_dec_ref(v_a_826_);
lean_dec(v_a_825_);
lean_dec_ref(v_a_824_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance(lean_object* v_00_u03b1_830_, lean_object* v_inst_831_, lean_object* v_k_832_, lean_object* v_a_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_){
_start:
{
lean_object* v___x_838_; 
v___x_838_ = lp_mathlib_Lean_Meta_withEnsuringLocalInstance___redArg(v_inst_831_, v_k_832_, v_a_833_, v_a_834_, v_a_835_, v_a_836_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withEnsuringLocalInstance___boxed(lean_object* v_00_u03b1_839_, lean_object* v_inst_840_, lean_object* v_k_841_, lean_object* v_a_842_, lean_object* v_a_843_, lean_object* v_a_844_, lean_object* v_a_845_, lean_object* v_a_846_){
_start:
{
lean_object* v_res_847_; 
v_res_847_ = lp_mathlib_Lean_Meta_withEnsuringLocalInstance(v_00_u03b1_839_, v_inst_840_, v_k_841_, v_a_842_, v_a_843_, v_a_844_, v_a_845_);
lean_dec(v_a_845_);
lean_dec_ref(v_a_844_);
lean_dec(v_a_843_);
lean_dec_ref(v_a_842_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___redArg(lean_object* v_k_848_, uint8_t v_allowLevelAssignments_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v___x_855_; 
v___x_855_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_849_, v_k_848_, v___y_850_, v___y_851_, v___y_852_, v___y_853_);
if (lean_obj_tag(v___x_855_) == 0)
{
lean_object* v_a_856_; lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_863_; 
v_a_856_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_863_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_863_ == 0)
{
v___x_858_ = v___x_855_;
v_isShared_859_ = v_isSharedCheck_863_;
goto v_resetjp_857_;
}
else
{
lean_inc(v_a_856_);
lean_dec(v___x_855_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_863_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
lean_object* v___x_861_; 
if (v_isShared_859_ == 0)
{
v___x_861_ = v___x_858_;
goto v_reusejp_860_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v_a_856_);
v___x_861_ = v_reuseFailAlloc_862_;
goto v_reusejp_860_;
}
v_reusejp_860_:
{
return v___x_861_;
}
}
}
else
{
lean_object* v_a_864_; lean_object* v___x_866_; uint8_t v_isShared_867_; uint8_t v_isSharedCheck_871_; 
v_a_864_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_871_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_871_ == 0)
{
v___x_866_ = v___x_855_;
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
else
{
lean_inc(v_a_864_);
lean_dec(v___x_855_);
v___x_866_ = lean_box(0);
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
v_resetjp_865_:
{
lean_object* v___x_869_; 
if (v_isShared_867_ == 0)
{
v___x_869_ = v___x_866_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v_a_864_);
v___x_869_ = v_reuseFailAlloc_870_;
goto v_reusejp_868_;
}
v_reusejp_868_:
{
return v___x_869_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___redArg___boxed(lean_object* v_k_872_, lean_object* v_allowLevelAssignments_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_879_; lean_object* v_res_880_; 
v_allowLevelAssignments_boxed_879_ = lean_unbox(v_allowLevelAssignments_873_);
v_res_880_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___redArg(v_k_872_, v_allowLevelAssignments_boxed_879_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v_res_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0(lean_object* v_00_u03b1_881_, lean_object* v_k_882_, uint8_t v_allowLevelAssignments_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_){
_start:
{
lean_object* v___x_889_; 
v___x_889_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___redArg(v_k_882_, v_allowLevelAssignments_883_, v___y_884_, v___y_885_, v___y_886_, v___y_887_);
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___boxed(lean_object* v_00_u03b1_890_, lean_object* v_k_891_, lean_object* v_allowLevelAssignments_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_898_; lean_object* v_res_899_; 
v_allowLevelAssignments_boxed_898_ = lean_unbox(v_allowLevelAssignments_892_);
v_res_899_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0(v_00_u03b1_890_, v_k_891_, v_allowLevelAssignments_boxed_898_, v___y_893_, v___y_894_, v___y_895_, v___y_896_);
lean_dec(v___y_896_);
lean_dec_ref(v___y_895_);
lean_dec(v___y_894_);
lean_dec_ref(v___y_893_);
return v_res_899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType___lam__0(lean_object* v_a_900_, lean_object* v_expectedType_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_){
_start:
{
lean_object* v___x_907_; 
v___x_907_ = l_Lean_Meta_isExprDefEq(v_a_900_, v_expectedType_901_, v___y_902_, v___y_903_, v___y_904_, v___y_905_);
return v___x_907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType___lam__0___boxed(lean_object* v_a_908_, lean_object* v_expectedType_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_Lean_Meta_ensureHasType___lam__0(v_a_908_, v_expectedType_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
return v_res_915_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ensureHasType___closed__1(void){
_start:
{
lean_object* v___x_917_; lean_object* v___x_918_; 
v___x_917_ = ((lean_object*)(lp_mathlib_Lean_Meta_ensureHasType___closed__0));
v___x_918_ = l_Lean_stringToMessageData(v___x_917_);
return v___x_918_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ensureHasType___closed__3(void){
_start:
{
lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_920_ = ((lean_object*)(lp_mathlib_Lean_Meta_ensureHasType___closed__2));
v___x_921_ = l_Lean_stringToMessageData(v___x_920_);
return v___x_921_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ensureHasType___closed__5(void){
_start:
{
lean_object* v___x_923_; lean_object* v___x_924_; 
v___x_923_ = ((lean_object*)(lp_mathlib_Lean_Meta_ensureHasType___closed__4));
v___x_924_ = l_Lean_stringToMessageData(v___x_923_);
return v___x_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType(lean_object* v_e_925_, lean_object* v_expectedType_926_, lean_object* v_a_927_, lean_object* v_a_928_, lean_object* v_a_929_, lean_object* v_a_930_){
_start:
{
lean_object* v___x_932_; 
lean_inc(v_a_930_);
lean_inc_ref(v_a_929_);
lean_inc(v_a_928_);
lean_inc_ref(v_a_927_);
lean_inc_ref(v_e_925_);
v___x_932_ = lean_infer_type(v_e_925_, v_a_927_, v_a_928_, v_a_929_, v_a_930_);
if (lean_obj_tag(v___x_932_) == 0)
{
lean_object* v_a_933_; lean_object* v___f_934_; uint8_t v___x_935_; lean_object* v___x_936_; 
v_a_933_ = lean_ctor_get(v___x_932_, 0);
lean_inc_n(v_a_933_, 2);
lean_dec_ref_known(v___x_932_, 1);
lean_inc_ref(v_expectedType_926_);
v___f_934_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_ensureHasType___lam__0___boxed), 7, 2);
lean_closure_set(v___f_934_, 0, v_a_933_);
lean_closure_set(v___f_934_, 1, v_expectedType_926_);
v___x_935_ = 0;
v___x_936_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_ensureHasType_spec__0___redArg(v___f_934_, v___x_935_, v_a_927_, v_a_928_, v_a_929_, v_a_930_);
if (lean_obj_tag(v___x_936_) == 0)
{
lean_object* v_a_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_976_; 
v_a_937_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_976_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_976_ == 0)
{
v___x_939_ = v___x_936_;
v_isShared_940_ = v_isSharedCheck_976_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_a_937_);
lean_dec(v___x_936_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_976_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
uint8_t v___x_941_; 
v___x_941_ = lean_unbox(v_a_937_);
lean_dec(v_a_937_);
if (v___x_941_ == 0)
{
lean_object* v___x_942_; 
lean_del_object(v___x_939_);
lean_inc_ref(v_e_925_);
v___x_942_ = l_Lean_Meta_coerceSimple_x3f(v_e_925_, v_expectedType_926_, v_a_927_, v_a_928_, v_a_929_, v_a_930_);
if (lean_obj_tag(v___x_942_) == 0)
{
lean_object* v_a_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_964_; 
v_a_943_ = lean_ctor_get(v___x_942_, 0);
v_isSharedCheck_964_ = !lean_is_exclusive(v___x_942_);
if (v_isSharedCheck_964_ == 0)
{
v___x_945_ = v___x_942_;
v_isShared_946_ = v_isSharedCheck_964_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_a_943_);
lean_dec(v___x_942_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_964_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
lean_object* v___x_947_; 
v___x_947_ = l_Lean_LOption_toOption___redArg(v_a_943_);
if (lean_obj_tag(v___x_947_) == 0)
{
lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; 
lean_del_object(v___x_945_);
v___x_948_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureHasType___closed__1, &lp_mathlib_Lean_Meta_ensureHasType___closed__1_once, _init_lp_mathlib_Lean_Meta_ensureHasType___closed__1);
v___x_949_ = l_Lean_MessageData_ofExpr(v_e_925_);
v___x_950_ = l_Lean_indentD(v___x_949_);
v___x_951_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_951_, 0, v___x_948_);
lean_ctor_set(v___x_951_, 1, v___x_950_);
v___x_952_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureHasType___closed__3, &lp_mathlib_Lean_Meta_ensureHasType___closed__3_once, _init_lp_mathlib_Lean_Meta_ensureHasType___closed__3);
v___x_953_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_953_, 0, v___x_951_);
lean_ctor_set(v___x_953_, 1, v___x_952_);
v___x_954_ = l_Lean_MessageData_ofExpr(v_a_933_);
v___x_955_ = l_Lean_indentD(v___x_954_);
v___x_956_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_956_, 0, v___x_953_);
lean_ctor_set(v___x_956_, 1, v___x_955_);
v___x_957_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureHasType___closed__5, &lp_mathlib_Lean_Meta_ensureHasType___closed__5_once, _init_lp_mathlib_Lean_Meta_ensureHasType___closed__5);
v___x_958_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_958_, 0, v___x_956_);
lean_ctor_set(v___x_958_, 1, v___x_957_);
v___x_959_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v___x_958_, v_a_927_, v_a_928_, v_a_929_, v_a_930_);
return v___x_959_;
}
else
{
lean_object* v_val_960_; lean_object* v___x_962_; 
lean_dec(v_a_933_);
lean_dec_ref(v_e_925_);
v_val_960_ = lean_ctor_get(v___x_947_, 0);
lean_inc(v_val_960_);
lean_dec_ref_known(v___x_947_, 1);
if (v_isShared_946_ == 0)
{
lean_ctor_set(v___x_945_, 0, v_val_960_);
v___x_962_ = v___x_945_;
goto v_reusejp_961_;
}
else
{
lean_object* v_reuseFailAlloc_963_; 
v_reuseFailAlloc_963_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_963_, 0, v_val_960_);
v___x_962_ = v_reuseFailAlloc_963_;
goto v_reusejp_961_;
}
v_reusejp_961_:
{
return v___x_962_;
}
}
}
}
else
{
lean_object* v_a_965_; lean_object* v___x_967_; uint8_t v_isShared_968_; uint8_t v_isSharedCheck_972_; 
lean_dec(v_a_933_);
lean_dec_ref(v_e_925_);
v_a_965_ = lean_ctor_get(v___x_942_, 0);
v_isSharedCheck_972_ = !lean_is_exclusive(v___x_942_);
if (v_isSharedCheck_972_ == 0)
{
v___x_967_ = v___x_942_;
v_isShared_968_ = v_isSharedCheck_972_;
goto v_resetjp_966_;
}
else
{
lean_inc(v_a_965_);
lean_dec(v___x_942_);
v___x_967_ = lean_box(0);
v_isShared_968_ = v_isSharedCheck_972_;
goto v_resetjp_966_;
}
v_resetjp_966_:
{
lean_object* v___x_970_; 
if (v_isShared_968_ == 0)
{
v___x_970_ = v___x_967_;
goto v_reusejp_969_;
}
else
{
lean_object* v_reuseFailAlloc_971_; 
v_reuseFailAlloc_971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_971_, 0, v_a_965_);
v___x_970_ = v_reuseFailAlloc_971_;
goto v_reusejp_969_;
}
v_reusejp_969_:
{
return v___x_970_;
}
}
}
}
else
{
lean_object* v___x_974_; 
lean_dec(v_a_933_);
lean_dec_ref(v_expectedType_926_);
if (v_isShared_940_ == 0)
{
lean_ctor_set(v___x_939_, 0, v_e_925_);
v___x_974_ = v___x_939_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v_e_925_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
else
{
lean_object* v_a_977_; lean_object* v___x_979_; uint8_t v_isShared_980_; uint8_t v_isSharedCheck_984_; 
lean_dec(v_a_933_);
lean_dec_ref(v_expectedType_926_);
lean_dec_ref(v_e_925_);
v_a_977_ = lean_ctor_get(v___x_936_, 0);
v_isSharedCheck_984_ = !lean_is_exclusive(v___x_936_);
if (v_isSharedCheck_984_ == 0)
{
v___x_979_ = v___x_936_;
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
else
{
lean_inc(v_a_977_);
lean_dec(v___x_936_);
v___x_979_ = lean_box(0);
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
v_resetjp_978_:
{
lean_object* v___x_982_; 
if (v_isShared_980_ == 0)
{
v___x_982_ = v___x_979_;
goto v_reusejp_981_;
}
else
{
lean_object* v_reuseFailAlloc_983_; 
v_reuseFailAlloc_983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_983_, 0, v_a_977_);
v___x_982_ = v_reuseFailAlloc_983_;
goto v_reusejp_981_;
}
v_reusejp_981_:
{
return v___x_982_;
}
}
}
}
else
{
lean_dec_ref(v_expectedType_926_);
lean_dec_ref(v_e_925_);
return v___x_932_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureHasType___boxed(lean_object* v_e_985_, lean_object* v_expectedType_986_, lean_object* v_a_987_, lean_object* v_a_988_, lean_object* v_a_989_, lean_object* v_a_990_, lean_object* v_a_991_){
_start:
{
lean_object* v_res_992_; 
v_res_992_ = lp_mathlib_Lean_Meta_ensureHasType(v_e_985_, v_expectedType_986_, v_a_987_, v_a_988_, v_a_989_, v_a_990_);
lean_dec(v_a_990_);
lean_dec_ref(v_a_989_);
lean_dec(v_a_988_);
lean_dec_ref(v_a_987_);
return v_res_992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg(lean_object* v_e_993_, lean_object* v___y_994_){
_start:
{
uint8_t v___x_996_; 
v___x_996_ = l_Lean_Expr_hasMVar(v_e_993_);
if (v___x_996_ == 0)
{
lean_object* v___x_997_; 
v___x_997_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_997_, 0, v_e_993_);
return v___x_997_;
}
else
{
lean_object* v___x_998_; lean_object* v_mctx_999_; lean_object* v___x_1000_; lean_object* v_fst_1001_; lean_object* v_snd_1002_; lean_object* v___x_1003_; lean_object* v_cache_1004_; lean_object* v_zetaDeltaFVarIds_1005_; lean_object* v_postponed_1006_; lean_object* v_diag_1007_; lean_object* v___x_1009_; uint8_t v_isShared_1010_; uint8_t v_isSharedCheck_1016_; 
v___x_998_ = lean_st_ref_get(v___y_994_);
v_mctx_999_ = lean_ctor_get(v___x_998_, 0);
lean_inc_ref(v_mctx_999_);
lean_dec(v___x_998_);
v___x_1000_ = l_Lean_instantiateMVarsCore(v_mctx_999_, v_e_993_);
v_fst_1001_ = lean_ctor_get(v___x_1000_, 0);
lean_inc(v_fst_1001_);
v_snd_1002_ = lean_ctor_get(v___x_1000_, 1);
lean_inc(v_snd_1002_);
lean_dec_ref(v___x_1000_);
v___x_1003_ = lean_st_ref_take(v___y_994_);
v_cache_1004_ = lean_ctor_get(v___x_1003_, 1);
v_zetaDeltaFVarIds_1005_ = lean_ctor_get(v___x_1003_, 2);
v_postponed_1006_ = lean_ctor_get(v___x_1003_, 3);
v_diag_1007_ = lean_ctor_get(v___x_1003_, 4);
v_isSharedCheck_1016_ = !lean_is_exclusive(v___x_1003_);
if (v_isSharedCheck_1016_ == 0)
{
lean_object* v_unused_1017_; 
v_unused_1017_ = lean_ctor_get(v___x_1003_, 0);
lean_dec(v_unused_1017_);
v___x_1009_ = v___x_1003_;
v_isShared_1010_ = v_isSharedCheck_1016_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_diag_1007_);
lean_inc(v_postponed_1006_);
lean_inc(v_zetaDeltaFVarIds_1005_);
lean_inc(v_cache_1004_);
lean_dec(v___x_1003_);
v___x_1009_ = lean_box(0);
v_isShared_1010_ = v_isSharedCheck_1016_;
goto v_resetjp_1008_;
}
v_resetjp_1008_:
{
lean_object* v___x_1012_; 
if (v_isShared_1010_ == 0)
{
lean_ctor_set(v___x_1009_, 0, v_snd_1002_);
v___x_1012_ = v___x_1009_;
goto v_reusejp_1011_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1015_, 0, v_snd_1002_);
lean_ctor_set(v_reuseFailAlloc_1015_, 1, v_cache_1004_);
lean_ctor_set(v_reuseFailAlloc_1015_, 2, v_zetaDeltaFVarIds_1005_);
lean_ctor_set(v_reuseFailAlloc_1015_, 3, v_postponed_1006_);
lean_ctor_set(v_reuseFailAlloc_1015_, 4, v_diag_1007_);
v___x_1012_ = v_reuseFailAlloc_1015_;
goto v_reusejp_1011_;
}
v_reusejp_1011_:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1013_ = lean_st_ref_set(v___y_994_, v___x_1012_);
v___x_1014_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1014_, 0, v_fst_1001_);
return v___x_1014_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg___boxed(lean_object* v_e_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_){
_start:
{
lean_object* v_res_1021_; 
v_res_1021_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg(v_e_1018_, v___y_1019_);
lean_dec(v___y_1019_);
return v_res_1021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0(lean_object* v_e_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_){
_start:
{
lean_object* v___x_1028_; 
v___x_1028_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg(v_e_1022_, v___y_1024_);
return v___x_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___boxed(lean_object* v_e_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_){
_start:
{
lean_object* v_res_1035_; 
v_res_1035_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0(v_e_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_);
lean_dec(v___y_1033_);
lean_dec_ref(v___y_1032_);
lean_dec(v___y_1031_);
lean_dec_ref(v___y_1030_);
return v_res_1035_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ensureIsFunction___closed__1(void){
_start:
{
lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1037_ = ((lean_object*)(lp_mathlib_Lean_Meta_ensureIsFunction___closed__0));
v___x_1038_ = l_Lean_stringToMessageData(v___x_1037_);
return v___x_1038_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ensureIsFunction___closed__3(void){
_start:
{
lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1040_ = ((lean_object*)(lp_mathlib_Lean_Meta_ensureIsFunction___closed__2));
v___x_1041_ = l_Lean_stringToMessageData(v___x_1040_);
return v___x_1041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsFunction(lean_object* v_e_1042_, lean_object* v_a_1043_, lean_object* v_a_1044_, lean_object* v_a_1045_, lean_object* v_a_1046_){
_start:
{
lean_object* v___x_1048_; 
lean_inc(v_a_1046_);
lean_inc_ref(v_a_1045_);
lean_inc(v_a_1044_);
lean_inc_ref(v_a_1043_);
lean_inc_ref(v_e_1042_);
v___x_1048_ = lean_infer_type(v_e_1042_, v_a_1043_, v_a_1044_, v_a_1045_, v_a_1046_);
if (lean_obj_tag(v___x_1048_) == 0)
{
lean_object* v_a_1049_; lean_object* v___x_1050_; lean_object* v_a_1051_; lean_object* v___x_1052_; 
v_a_1049_ = lean_ctor_get(v___x_1048_, 0);
lean_inc(v_a_1049_);
lean_dec_ref_known(v___x_1048_, 1);
v___x_1050_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg(v_a_1049_, v_a_1044_);
v_a_1051_ = lean_ctor_get(v___x_1050_, 0);
lean_inc(v_a_1051_);
lean_dec_ref(v___x_1050_);
lean_inc(v_a_1046_);
lean_inc_ref(v_a_1045_);
lean_inc(v_a_1044_);
lean_inc_ref(v_a_1043_);
v___x_1052_ = lean_whnf(v_a_1051_, v_a_1043_, v_a_1044_, v_a_1045_, v_a_1046_);
if (lean_obj_tag(v___x_1052_) == 0)
{
lean_object* v_a_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1091_; 
v_a_1053_ = lean_ctor_get(v___x_1052_, 0);
v_isSharedCheck_1091_ = !lean_is_exclusive(v___x_1052_);
if (v_isSharedCheck_1091_ == 0)
{
v___x_1055_ = v___x_1052_;
v_isShared_1056_ = v_isSharedCheck_1091_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_a_1053_);
lean_dec(v___x_1052_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1091_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
uint8_t v___x_1057_; 
v___x_1057_ = l_Lean_Expr_isForall(v_a_1053_);
if (v___x_1057_ == 0)
{
lean_object* v___x_1058_; 
lean_del_object(v___x_1055_);
lean_inc_ref(v_e_1042_);
v___x_1058_ = l_Lean_Meta_coerceToFunction_x3f(v_e_1042_, v_a_1043_, v_a_1044_, v_a_1045_, v_a_1046_);
if (lean_obj_tag(v___x_1058_) == 0)
{
lean_object* v_a_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1079_; 
v_a_1059_ = lean_ctor_get(v___x_1058_, 0);
v_isSharedCheck_1079_ = !lean_is_exclusive(v___x_1058_);
if (v_isSharedCheck_1079_ == 0)
{
v___x_1061_ = v___x_1058_;
v_isShared_1062_ = v_isSharedCheck_1079_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_a_1059_);
lean_dec(v___x_1058_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1079_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
if (lean_obj_tag(v_a_1059_) == 0)
{
lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; 
lean_del_object(v___x_1061_);
v___x_1063_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureHasType___closed__1, &lp_mathlib_Lean_Meta_ensureHasType___closed__1_once, _init_lp_mathlib_Lean_Meta_ensureHasType___closed__1);
v___x_1064_ = l_Lean_MessageData_ofExpr(v_e_1042_);
v___x_1065_ = l_Lean_indentD(v___x_1064_);
v___x_1066_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1066_, 0, v___x_1063_);
lean_ctor_set(v___x_1066_, 1, v___x_1065_);
v___x_1067_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureIsFunction___closed__1, &lp_mathlib_Lean_Meta_ensureIsFunction___closed__1_once, _init_lp_mathlib_Lean_Meta_ensureIsFunction___closed__1);
v___x_1068_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1068_, 0, v___x_1066_);
lean_ctor_set(v___x_1068_, 1, v___x_1067_);
v___x_1069_ = l_Lean_MessageData_ofExpr(v_a_1053_);
v___x_1070_ = l_Lean_indentD(v___x_1069_);
v___x_1071_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1071_, 0, v___x_1068_);
lean_ctor_set(v___x_1071_, 1, v___x_1070_);
v___x_1072_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureIsFunction___closed__3, &lp_mathlib_Lean_Meta_ensureIsFunction___closed__3_once, _init_lp_mathlib_Lean_Meta_ensureIsFunction___closed__3);
v___x_1073_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1073_, 0, v___x_1071_);
lean_ctor_set(v___x_1073_, 1, v___x_1072_);
v___x_1074_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v___x_1073_, v_a_1043_, v_a_1044_, v_a_1045_, v_a_1046_);
return v___x_1074_;
}
else
{
lean_object* v_val_1075_; lean_object* v___x_1077_; 
lean_dec(v_a_1053_);
lean_dec_ref(v_e_1042_);
v_val_1075_ = lean_ctor_get(v_a_1059_, 0);
lean_inc(v_val_1075_);
lean_dec_ref_known(v_a_1059_, 1);
if (v_isShared_1062_ == 0)
{
lean_ctor_set(v___x_1061_, 0, v_val_1075_);
v___x_1077_ = v___x_1061_;
goto v_reusejp_1076_;
}
else
{
lean_object* v_reuseFailAlloc_1078_; 
v_reuseFailAlloc_1078_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1078_, 0, v_val_1075_);
v___x_1077_ = v_reuseFailAlloc_1078_;
goto v_reusejp_1076_;
}
v_reusejp_1076_:
{
return v___x_1077_;
}
}
}
}
else
{
lean_object* v_a_1080_; lean_object* v___x_1082_; uint8_t v_isShared_1083_; uint8_t v_isSharedCheck_1087_; 
lean_dec(v_a_1053_);
lean_dec_ref(v_e_1042_);
v_a_1080_ = lean_ctor_get(v___x_1058_, 0);
v_isSharedCheck_1087_ = !lean_is_exclusive(v___x_1058_);
if (v_isSharedCheck_1087_ == 0)
{
v___x_1082_ = v___x_1058_;
v_isShared_1083_ = v_isSharedCheck_1087_;
goto v_resetjp_1081_;
}
else
{
lean_inc(v_a_1080_);
lean_dec(v___x_1058_);
v___x_1082_ = lean_box(0);
v_isShared_1083_ = v_isSharedCheck_1087_;
goto v_resetjp_1081_;
}
v_resetjp_1081_:
{
lean_object* v___x_1085_; 
if (v_isShared_1083_ == 0)
{
v___x_1085_ = v___x_1082_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1086_; 
v_reuseFailAlloc_1086_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1086_, 0, v_a_1080_);
v___x_1085_ = v_reuseFailAlloc_1086_;
goto v_reusejp_1084_;
}
v_reusejp_1084_:
{
return v___x_1085_;
}
}
}
}
else
{
lean_object* v___x_1089_; 
lean_dec(v_a_1053_);
if (v_isShared_1056_ == 0)
{
lean_ctor_set(v___x_1055_, 0, v_e_1042_);
v___x_1089_ = v___x_1055_;
goto v_reusejp_1088_;
}
else
{
lean_object* v_reuseFailAlloc_1090_; 
v_reuseFailAlloc_1090_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1090_, 0, v_e_1042_);
v___x_1089_ = v_reuseFailAlloc_1090_;
goto v_reusejp_1088_;
}
v_reusejp_1088_:
{
return v___x_1089_;
}
}
}
}
else
{
lean_dec_ref(v_e_1042_);
return v___x_1052_;
}
}
else
{
lean_dec_ref(v_e_1042_);
return v___x_1048_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsFunction___boxed(lean_object* v_e_1092_, lean_object* v_a_1093_, lean_object* v_a_1094_, lean_object* v_a_1095_, lean_object* v_a_1096_, lean_object* v_a_1097_){
_start:
{
lean_object* v_res_1098_; 
v_res_1098_ = lp_mathlib_Lean_Meta_ensureIsFunction(v_e_1092_, v_a_1093_, v_a_1094_, v_a_1095_, v_a_1096_);
lean_dec(v_a_1096_);
lean_dec_ref(v_a_1095_);
lean_dec(v_a_1094_);
lean_dec_ref(v_a_1093_);
return v_res_1098_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ensureIsSort___closed__1(void){
_start:
{
lean_object* v___x_1100_; lean_object* v___x_1101_; 
v___x_1100_ = ((lean_object*)(lp_mathlib_Lean_Meta_ensureIsSort___closed__0));
v___x_1101_ = l_Lean_stringToMessageData(v___x_1100_);
return v___x_1101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsSort(lean_object* v_e_1102_, lean_object* v_a_1103_, lean_object* v_a_1104_, lean_object* v_a_1105_, lean_object* v_a_1106_){
_start:
{
lean_object* v___x_1108_; 
lean_inc(v_a_1106_);
lean_inc_ref(v_a_1105_);
lean_inc(v_a_1104_);
lean_inc_ref(v_a_1103_);
lean_inc_ref(v_e_1102_);
v___x_1108_ = lean_infer_type(v_e_1102_, v_a_1103_, v_a_1104_, v_a_1105_, v_a_1106_);
if (lean_obj_tag(v___x_1108_) == 0)
{
lean_object* v_a_1109_; lean_object* v___x_1110_; lean_object* v_a_1111_; lean_object* v___x_1112_; 
v_a_1109_ = lean_ctor_get(v___x_1108_, 0);
lean_inc(v_a_1109_);
lean_dec_ref_known(v___x_1108_, 1);
v___x_1110_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_ensureIsFunction_spec__0___redArg(v_a_1109_, v_a_1104_);
v_a_1111_ = lean_ctor_get(v___x_1110_, 0);
lean_inc(v_a_1111_);
lean_dec_ref(v___x_1110_);
lean_inc(v_a_1106_);
lean_inc_ref(v_a_1105_);
lean_inc(v_a_1104_);
lean_inc_ref(v_a_1103_);
v___x_1112_ = lean_whnf(v_a_1111_, v_a_1103_, v_a_1104_, v_a_1105_, v_a_1106_);
if (lean_obj_tag(v___x_1112_) == 0)
{
lean_object* v_a_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1151_; 
v_a_1113_ = lean_ctor_get(v___x_1112_, 0);
v_isSharedCheck_1151_ = !lean_is_exclusive(v___x_1112_);
if (v_isSharedCheck_1151_ == 0)
{
v___x_1115_ = v___x_1112_;
v_isShared_1116_ = v_isSharedCheck_1151_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_a_1113_);
lean_dec(v___x_1112_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1151_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
uint8_t v___x_1117_; 
v___x_1117_ = l_Lean_Expr_isSort(v_a_1113_);
if (v___x_1117_ == 0)
{
lean_object* v___x_1118_; 
lean_del_object(v___x_1115_);
lean_inc_ref(v_e_1102_);
v___x_1118_ = l_Lean_Meta_coerceToSort_x3f(v_e_1102_, v_a_1103_, v_a_1104_, v_a_1105_, v_a_1106_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_a_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1139_; 
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1139_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1139_ == 0)
{
v___x_1121_ = v___x_1118_;
v_isShared_1122_ = v_isSharedCheck_1139_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_a_1119_);
lean_dec(v___x_1118_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1139_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
if (lean_obj_tag(v_a_1119_) == 0)
{
lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; 
lean_del_object(v___x_1121_);
v___x_1123_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureHasType___closed__1, &lp_mathlib_Lean_Meta_ensureHasType___closed__1_once, _init_lp_mathlib_Lean_Meta_ensureHasType___closed__1);
v___x_1124_ = l_Lean_MessageData_ofExpr(v_e_1102_);
v___x_1125_ = l_Lean_indentD(v___x_1124_);
v___x_1126_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1126_, 0, v___x_1123_);
lean_ctor_set(v___x_1126_, 1, v___x_1125_);
v___x_1127_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureIsFunction___closed__1, &lp_mathlib_Lean_Meta_ensureIsFunction___closed__1_once, _init_lp_mathlib_Lean_Meta_ensureIsFunction___closed__1);
v___x_1128_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1128_, 0, v___x_1126_);
lean_ctor_set(v___x_1128_, 1, v___x_1127_);
v___x_1129_ = l_Lean_MessageData_ofExpr(v_a_1113_);
v___x_1130_ = l_Lean_indentD(v___x_1129_);
v___x_1131_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1131_, 0, v___x_1128_);
lean_ctor_set(v___x_1131_, 1, v___x_1130_);
v___x_1132_ = lean_obj_once(&lp_mathlib_Lean_Meta_ensureIsSort___closed__1, &lp_mathlib_Lean_Meta_ensureIsSort___closed__1_once, _init_lp_mathlib_Lean_Meta_ensureIsSort___closed__1);
v___x_1133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1131_);
lean_ctor_set(v___x_1133_, 1, v___x_1132_);
v___x_1134_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_forallMetaTelescopeReducingUntilDefEq_spec__0___redArg(v___x_1133_, v_a_1103_, v_a_1104_, v_a_1105_, v_a_1106_);
return v___x_1134_;
}
else
{
lean_object* v_val_1135_; lean_object* v___x_1137_; 
lean_dec(v_a_1113_);
lean_dec_ref(v_e_1102_);
v_val_1135_ = lean_ctor_get(v_a_1119_, 0);
lean_inc(v_val_1135_);
lean_dec_ref_known(v_a_1119_, 1);
if (v_isShared_1122_ == 0)
{
lean_ctor_set(v___x_1121_, 0, v_val_1135_);
v___x_1137_ = v___x_1121_;
goto v_reusejp_1136_;
}
else
{
lean_object* v_reuseFailAlloc_1138_; 
v_reuseFailAlloc_1138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1138_, 0, v_val_1135_);
v___x_1137_ = v_reuseFailAlloc_1138_;
goto v_reusejp_1136_;
}
v_reusejp_1136_:
{
return v___x_1137_;
}
}
}
}
else
{
lean_object* v_a_1140_; lean_object* v___x_1142_; uint8_t v_isShared_1143_; uint8_t v_isSharedCheck_1147_; 
lean_dec(v_a_1113_);
lean_dec_ref(v_e_1102_);
v_a_1140_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1147_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1142_ = v___x_1118_;
v_isShared_1143_ = v_isSharedCheck_1147_;
goto v_resetjp_1141_;
}
else
{
lean_inc(v_a_1140_);
lean_dec(v___x_1118_);
v___x_1142_ = lean_box(0);
v_isShared_1143_ = v_isSharedCheck_1147_;
goto v_resetjp_1141_;
}
v_resetjp_1141_:
{
lean_object* v___x_1145_; 
if (v_isShared_1143_ == 0)
{
v___x_1145_ = v___x_1142_;
goto v_reusejp_1144_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v_a_1140_);
v___x_1145_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1144_;
}
v_reusejp_1144_:
{
return v___x_1145_;
}
}
}
}
else
{
lean_object* v___x_1149_; 
lean_dec(v_a_1113_);
if (v_isShared_1116_ == 0)
{
lean_ctor_set(v___x_1115_, 0, v_e_1102_);
v___x_1149_ = v___x_1115_;
goto v_reusejp_1148_;
}
else
{
lean_object* v_reuseFailAlloc_1150_; 
v_reuseFailAlloc_1150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1150_, 0, v_e_1102_);
v___x_1149_ = v_reuseFailAlloc_1150_;
goto v_reusejp_1148_;
}
v_reusejp_1148_:
{
return v___x_1149_;
}
}
}
}
else
{
lean_dec_ref(v_e_1102_);
return v___x_1112_;
}
}
else
{
lean_dec_ref(v_e_1102_);
return v___x_1108_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ensureIsSort___boxed(lean_object* v_e_1152_, lean_object* v_a_1153_, lean_object* v_a_1154_, lean_object* v_a_1155_, lean_object* v_a_1156_, lean_object* v_a_1157_){
_start:
{
lean_object* v_res_1158_; 
v_res_1158_ = lp_mathlib_Lean_Meta_ensureIsSort(v_e_1152_, v_a_1153_, v_a_1154_, v_a_1155_, v_a_1156_);
lean_dec(v_a_1156_);
lean_dec_ref(v_a_1155_);
lean_dec(v_a_1154_);
lean_dec_ref(v_a_1153_);
return v_res_1158_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_AppBuilder(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Coe(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_AppBuilder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_AppBuilder(uint8_t builtin);
lean_object* initialize_Lean_Meta_Coe(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_AppBuilder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
