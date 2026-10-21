// Lean compiler output
// Module: Mathlib.Tactic.Coe
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Meta_coerce_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
lean_object* l_Lean_Elab_Term_tryPostpone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Meta_coerceToSort_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_ensureHasType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_etaExpanded_x3f(lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* l_Lean_Elab_Term_withExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_coerceToFunction_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__5___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = ") must have a non-dependent function type, not"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__5;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = ") must have a function type, not"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "CoeImpl"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term(↑)"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__3_value),LEAN_SCALAR_PTR_LITERAL(10, 185, 64, 11, 127, 227, 35, 222)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value_aux_3),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__4_value),LEAN_SCALAR_PTR_LITERAL(208, 132, 153, 133, 247, 190, 3, 1)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__6_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↑"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__9_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__11_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__11_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__13_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__15_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "cannot coerce"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "\nto type"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term(⇑)"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__3_value),LEAN_SCALAR_PTR_LITERAL(10, 185, 64, 11, 127, 227, 35, 222)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(226, 149, 188, 57, 230, 78, 27, 15)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⇑"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__3_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__4_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__13_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__5_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "cannot coerce to function"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term(↥)"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__3_value),LEAN_SCALAR_PTR_LITERAL(10, 185, 64, 11, 127, 227, 35, 222)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 110, 131, 9, 47, 82, 186, 47)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↥"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__3_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__7_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__4_value),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__13_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__5_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "cannot coerce to sort"};
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Lean_Expr_hasMVar(v_e_1_);
if (v___x_4_ == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_e_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; lean_object* v_mctx_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; lean_object* v_cache_12_; lean_object* v_zetaDeltaFVarIds_13_; lean_object* v_postponed_14_; lean_object* v_diag_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_6_ = lean_st_ref_get(v___y_2_);
v_mctx_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_mctx_7_);
lean_dec(v___x_6_);
v___x_8_ = l_Lean_instantiateMVarsCore(v_mctx_7_, v_e_1_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v___x_8_);
v___x_11_ = lean_st_ref_take(v___y_2_);
v_cache_12_ = lean_ctor_get(v___x_11_, 1);
v_zetaDeltaFVarIds_13_ = lean_ctor_get(v___x_11_, 2);
v_postponed_14_ = lean_ctor_get(v___x_11_, 3);
v_diag_15_ = lean_ctor_get(v___x_11_, 4);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_11_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_diag_15_);
lean_inc(v_postponed_14_);
lean_inc(v_zetaDeltaFVarIds_13_);
lean_inc(v_cache_12_);
lean_dec(v___x_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v_snd_10_);
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_snd_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_cache_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_zetaDeltaFVarIds_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_postponed_14_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_diag_15_);
v___x_20_ = v_reuseFailAlloc_23_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_st_ref_set(v___y_2_, v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_fst_9_);
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___redArg(v_e_30_, v___y_34_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___boxed(lean_object* v_e_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0(v_e_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___lam__0(lean_object* v_mkCoe_48_, lean_object* v_body_49_, lean_object* v_x_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___x_58_; 
lean_inc(v___y_56_);
lean_inc_ref(v___y_55_);
lean_inc(v___y_54_);
lean_inc_ref(v___y_53_);
lean_inc(v___y_52_);
lean_inc_ref(v___y_51_);
lean_inc_ref(v_x_50_);
v___x_58_ = lean_apply_9(v_mkCoe_48_, v_body_49_, v_x_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_, lean_box(0));
if (lean_obj_tag(v___x_58_) == 0)
{
lean_object* v_a_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; uint8_t v___x_63_; uint8_t v___x_64_; uint8_t v___x_65_; lean_object* v___x_66_; 
v_a_59_ = lean_ctor_get(v___x_58_, 0);
lean_inc(v_a_59_);
lean_dec_ref_known(v___x_58_, 1);
v___x_60_ = lean_unsigned_to_nat(1u);
v___x_61_ = lean_mk_empty_array_with_capacity(v___x_60_);
v___x_62_ = lean_array_push(v___x_61_, v_x_50_);
v___x_63_ = 0;
v___x_64_ = 1;
v___x_65_ = 1;
v___x_66_ = l_Lean_Meta_mkLambdaFVars(v___x_62_, v_a_59_, v___x_63_, v___x_64_, v___x_63_, v___x_64_, v___x_65_, v___y_53_, v___y_54_, v___y_55_, v___y_56_);
lean_dec_ref(v___x_62_);
return v___x_66_;
}
else
{
lean_dec_ref(v_x_50_);
return v___x_58_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___lam__0___boxed(lean_object* v_mkCoe_67_, lean_object* v_body_68_, lean_object* v_x_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___lam__0(v_mkCoe_67_, v_body_68_, v_x_69_, v___y_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___lam__0(lean_object* v_k_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v_b_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_){
_start:
{
lean_object* v___x_87_; 
lean_inc(v___y_85_);
lean_inc_ref(v___y_84_);
lean_inc(v___y_83_);
lean_inc_ref(v___y_82_);
lean_inc(v___y_80_);
lean_inc_ref(v___y_79_);
v___x_87_ = lean_apply_8(v_k_78_, v_b_81_, v___y_79_, v___y_80_, v___y_82_, v___y_83_, v___y_84_, v___y_85_, lean_box(0));
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___lam__0___boxed(lean_object* v_k_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v_b_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___lam__0(v_k_88_, v___y_89_, v___y_90_, v_b_91_, v___y_92_, v___y_93_, v___y_94_, v___y_95_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
lean_dec(v___y_93_);
lean_dec_ref(v___y_92_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg(lean_object* v_name_98_, uint8_t v_bi_99_, lean_object* v_type_100_, lean_object* v_k_101_, uint8_t v_kind_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_){
_start:
{
lean_object* v___f_110_; lean_object* v___x_111_; 
lean_inc(v___y_104_);
lean_inc_ref(v___y_103_);
v___f_110_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_110_, 0, v_k_101_);
lean_closure_set(v___f_110_, 1, v___y_103_);
lean_closure_set(v___f_110_, 2, v___y_104_);
v___x_111_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_98_, v_bi_99_, v_type_100_, v___f_110_, v_kind_102_, v___y_105_, v___y_106_, v___y_107_, v___y_108_);
if (lean_obj_tag(v___x_111_) == 0)
{
return v___x_111_;
}
else
{
lean_object* v_a_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_119_; 
v_a_112_ = lean_ctor_get(v___x_111_, 0);
v_isSharedCheck_119_ = !lean_is_exclusive(v___x_111_);
if (v_isSharedCheck_119_ == 0)
{
v___x_114_ = v___x_111_;
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_a_112_);
lean_dec(v___x_111_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_117_; 
if (v_isShared_115_ == 0)
{
v___x_117_ = v___x_114_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v_a_112_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg___boxed(lean_object* v_name_120_, lean_object* v_bi_121_, lean_object* v_type_122_, lean_object* v_k_123_, lean_object* v_kind_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
uint8_t v_bi_boxed_132_; uint8_t v_kind_boxed_133_; lean_object* v_res_134_; 
v_bi_boxed_132_ = lean_unbox(v_bi_121_);
v_kind_boxed_133_ = lean_unbox(v_kind_124_);
v_res_134_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg(v_name_120_, v_bi_boxed_132_, v_type_122_, v_k_123_, v_kind_boxed_133_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
lean_dec(v___y_130_);
lean_dec_ref(v___y_129_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___redArg(lean_object* v_name_135_, lean_object* v_type_136_, lean_object* v_k_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_){
_start:
{
uint8_t v___x_145_; uint8_t v___x_146_; lean_object* v___x_147_; 
v___x_145_ = 0;
v___x_146_ = 0;
v___x_147_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg(v_name_135_, v___x_145_, v_type_136_, v_k_137_, v___x_146_, v___y_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_, v___y_143_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___redArg___boxed(lean_object* v_name_148_, lean_object* v_type_149_, lean_object* v_k_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___redArg(v_name_148_, v_type_149_, v_k_150_, v___y_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_);
lean_dec(v___y_156_);
lean_dec_ref(v___y_155_);
lean_dec(v___y_154_);
lean_dec_ref(v___y_153_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
return v_res_158_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__5(lean_object* v_opts_159_, lean_object* v_opt_160_){
_start:
{
lean_object* v_name_161_; lean_object* v_defValue_162_; lean_object* v_map_163_; lean_object* v___x_164_; 
v_name_161_ = lean_ctor_get(v_opt_160_, 0);
v_defValue_162_ = lean_ctor_get(v_opt_160_, 1);
v_map_163_ = lean_ctor_get(v_opts_159_, 0);
v___x_164_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_163_, v_name_161_);
if (lean_obj_tag(v___x_164_) == 0)
{
uint8_t v___x_165_; 
v___x_165_ = lean_unbox(v_defValue_162_);
return v___x_165_;
}
else
{
lean_object* v_val_166_; 
v_val_166_ = lean_ctor_get(v___x_164_, 0);
lean_inc(v_val_166_);
lean_dec_ref_known(v___x_164_, 1);
if (lean_obj_tag(v_val_166_) == 1)
{
uint8_t v_v_167_; 
v_v_167_ = lean_ctor_get_uint8(v_val_166_, 0);
lean_dec_ref_known(v_val_166_, 0);
return v_v_167_;
}
else
{
uint8_t v___x_168_; 
lean_dec(v_val_166_);
v___x_168_ = lean_unbox(v_defValue_162_);
return v___x_168_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__5___boxed(lean_object* v_opts_169_, lean_object* v_opt_170_){
_start:
{
uint8_t v_res_171_; lean_object* v_r_172_; 
v_res_171_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__5(v_opts_169_, v_opt_170_);
lean_dec_ref(v_opt_170_);
lean_dec_ref(v_opts_169_);
v_r_172_ = lean_box(v_res_171_);
return v_r_172_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0(void){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_173_ = lean_box(1);
v___x_174_ = l_Lean_MessageData_ofFormat(v___x_173_);
return v___x_174_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__3(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_178_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__2));
v___x_179_ = l_Lean_MessageData_ofFormat(v___x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6(lean_object* v_x_180_, lean_object* v_x_181_){
_start:
{
if (lean_obj_tag(v_x_181_) == 0)
{
return v_x_180_;
}
else
{
lean_object* v_head_182_; lean_object* v_tail_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_205_; 
v_head_182_ = lean_ctor_get(v_x_181_, 0);
v_tail_183_ = lean_ctor_get(v_x_181_, 1);
v_isSharedCheck_205_ = !lean_is_exclusive(v_x_181_);
if (v_isSharedCheck_205_ == 0)
{
v___x_185_ = v_x_181_;
v_isShared_186_ = v_isSharedCheck_205_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_tail_183_);
lean_inc(v_head_182_);
lean_dec(v_x_181_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_205_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v_before_187_; lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_203_; 
v_before_187_ = lean_ctor_get(v_head_182_, 0);
v_isSharedCheck_203_ = !lean_is_exclusive(v_head_182_);
if (v_isSharedCheck_203_ == 0)
{
lean_object* v_unused_204_; 
v_unused_204_ = lean_ctor_get(v_head_182_, 1);
lean_dec(v_unused_204_);
v___x_189_ = v_head_182_;
v_isShared_190_ = v_isSharedCheck_203_;
goto v_resetjp_188_;
}
else
{
lean_inc(v_before_187_);
lean_dec(v_head_182_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_203_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___x_191_; lean_object* v___x_193_; 
v___x_191_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0);
if (v_isShared_190_ == 0)
{
lean_ctor_set_tag(v___x_189_, 7);
lean_ctor_set(v___x_189_, 1, v___x_191_);
lean_ctor_set(v___x_189_, 0, v_x_180_);
v___x_193_ = v___x_189_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v_x_180_);
lean_ctor_set(v_reuseFailAlloc_202_, 1, v___x_191_);
v___x_193_ = v_reuseFailAlloc_202_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
lean_object* v___x_194_; lean_object* v___x_196_; 
v___x_194_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__3);
if (v_isShared_186_ == 0)
{
lean_ctor_set_tag(v___x_185_, 7);
lean_ctor_set(v___x_185_, 1, v___x_194_);
lean_ctor_set(v___x_185_, 0, v___x_193_);
v___x_196_ = v___x_185_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v___x_193_);
lean_ctor_set(v_reuseFailAlloc_201_, 1, v___x_194_);
v___x_196_ = v_reuseFailAlloc_201_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_197_ = l_Lean_MessageData_ofSyntax(v_before_187_);
v___x_198_ = l_Lean_indentD(v___x_197_);
v___x_199_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_196_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
v_x_180_ = v___x_199_;
v_x_181_ = v_tail_183_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__1));
v___x_210_ = l_Lean_MessageData_ofFormat(v___x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg(lean_object* v_msgData_211_, lean_object* v_macroStack_212_, lean_object* v___y_213_){
_start:
{
lean_object* v_options_215_; lean_object* v___x_216_; uint8_t v___x_217_; 
v_options_215_ = lean_ctor_get(v___y_213_, 2);
v___x_216_ = l_Lean_Elab_pp_macroStack;
v___x_217_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__5(v_options_215_, v___x_216_);
if (v___x_217_ == 0)
{
lean_object* v___x_218_; 
lean_dec(v_macroStack_212_);
v___x_218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_218_, 0, v_msgData_211_);
return v___x_218_;
}
else
{
if (lean_obj_tag(v_macroStack_212_) == 0)
{
lean_object* v___x_219_; 
v___x_219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_219_, 0, v_msgData_211_);
return v___x_219_;
}
else
{
lean_object* v_head_220_; lean_object* v_after_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_236_; 
v_head_220_ = lean_ctor_get(v_macroStack_212_, 0);
lean_inc(v_head_220_);
v_after_221_ = lean_ctor_get(v_head_220_, 1);
v_isSharedCheck_236_ = !lean_is_exclusive(v_head_220_);
if (v_isSharedCheck_236_ == 0)
{
lean_object* v_unused_237_; 
v_unused_237_ = lean_ctor_get(v_head_220_, 0);
lean_dec(v_unused_237_);
v___x_223_ = v_head_220_;
v_isShared_224_ = v_isSharedCheck_236_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_after_221_);
lean_dec(v_head_220_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_236_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_227_; 
v___x_225_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6___closed__0);
if (v_isShared_224_ == 0)
{
lean_ctor_set_tag(v___x_223_, 7);
lean_ctor_set(v___x_223_, 1, v___x_225_);
lean_ctor_set(v___x_223_, 0, v_msgData_211_);
v___x_227_ = v___x_223_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v_msgData_211_);
lean_ctor_set(v_reuseFailAlloc_235_, 1, v___x_225_);
v___x_227_ = v_reuseFailAlloc_235_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v_msgData_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_228_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___closed__2);
v___x_229_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_227_);
lean_ctor_set(v___x_229_, 1, v___x_228_);
v___x_230_ = l_Lean_MessageData_ofSyntax(v_after_221_);
v___x_231_ = l_Lean_indentD(v___x_230_);
v_msgData_232_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_232_, 0, v___x_229_);
lean_ctor_set(v_msgData_232_, 1, v___x_231_);
v___x_233_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4_spec__6(v_msgData_232_, v_macroStack_212_);
v___x_234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
return v___x_234_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg___boxed(lean_object* v_msgData_238_, lean_object* v_macroStack_239_, lean_object* v___y_240_, lean_object* v___y_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg(v_msgData_238_, v_macroStack_239_, v___y_240_);
lean_dec_ref(v___y_240_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__3(lean_object* v_msgData_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_){
_start:
{
lean_object* v___x_249_; lean_object* v_env_250_; lean_object* v___x_251_; lean_object* v_mctx_252_; lean_object* v_lctx_253_; lean_object* v_options_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_249_ = lean_st_ref_get(v___y_247_);
v_env_250_ = lean_ctor_get(v___x_249_, 0);
lean_inc_ref(v_env_250_);
lean_dec(v___x_249_);
v___x_251_ = lean_st_ref_get(v___y_245_);
v_mctx_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc_ref(v_mctx_252_);
lean_dec(v___x_251_);
v_lctx_253_ = lean_ctor_get(v___y_244_, 2);
v_options_254_ = lean_ctor_get(v___y_246_, 2);
lean_inc_ref(v_options_254_);
lean_inc_ref(v_lctx_253_);
v___x_255_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_255_, 0, v_env_250_);
lean_ctor_set(v___x_255_, 1, v_mctx_252_);
lean_ctor_set(v___x_255_, 2, v_lctx_253_);
lean_ctor_set(v___x_255_, 3, v_options_254_);
v___x_256_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
lean_ctor_set(v___x_256_, 1, v_msgData_243_);
v___x_257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__3___boxed(lean_object* v_msgData_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__3(v_msgData_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(lean_object* v_msg_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v_ref_273_; lean_object* v___x_274_; lean_object* v_a_275_; lean_object* v_macroStack_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v_a_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_287_; 
v_ref_273_ = lean_ctor_get(v___y_270_, 5);
v___x_274_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__3(v_msg_265_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
v_a_275_ = lean_ctor_get(v___x_274_, 0);
lean_inc(v_a_275_);
lean_dec_ref(v___x_274_);
v_macroStack_276_ = lean_ctor_get(v___y_266_, 1);
v___x_277_ = l_Lean_Elab_getBetterRef(v_ref_273_, v_macroStack_276_);
lean_inc(v_macroStack_276_);
v___x_278_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg(v_a_275_, v_macroStack_276_, v___y_270_);
v_a_279_ = lean_ctor_get(v___x_278_, 0);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_287_ == 0)
{
v___x_281_ = v___x_278_;
v_isShared_282_ = v_isSharedCheck_287_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_a_279_);
lean_dec(v___x_278_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_287_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_283_; lean_object* v___x_285_; 
v___x_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_277_);
lean_ctor_set(v___x_283_, 1, v_a_279_);
if (v_isShared_282_ == 0)
{
lean_ctor_set_tag(v___x_281_, 1);
lean_ctor_set(v___x_281_, 0, v___x_283_);
v___x_285_ = v___x_281_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v___x_283_);
v___x_285_ = v_reuseFailAlloc_286_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
return v___x_285_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg___boxed(lean_object* v_msg_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(v_msg_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_);
lean_dec(v___y_294_);
lean_dec_ref(v___y_293_);
lean_dec(v___y_292_);
lean_dec_ref(v___y_291_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
return v_res_296_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3(void){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_301_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__2));
v___x_302_ = l_Lean_stringToMessageData(v___x_301_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__5(void){
_start:
{
lean_object* v___x_304_; lean_object* v___x_305_; 
v___x_304_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__4));
v___x_305_ = l_Lean_stringToMessageData(v___x_304_);
return v___x_305_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__7(void){
_start:
{
lean_object* v___x_307_; lean_object* v___x_308_; 
v___x_307_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__6));
v___x_308_ = l_Lean_stringToMessageData(v___x_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe(lean_object* v_sym_309_, lean_object* v_expectedType_310_, lean_object* v_mkCoe_311_, lean_object* v_a_312_, lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_, lean_object* v_a_317_){
_start:
{
lean_object* v___x_319_; lean_object* v_a_320_; 
v___x_319_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__0___redArg(v_expectedType_310_, v_a_315_);
v_a_320_ = lean_ctor_get(v___x_319_, 0);
lean_inc(v_a_320_);
lean_dec_ref(v___x_319_);
if (lean_obj_tag(v_a_320_) == 7)
{
lean_object* v_binderType_321_; lean_object* v_body_322_; lean_object* v___f_323_; lean_object* v___y_325_; lean_object* v___y_326_; lean_object* v___y_327_; lean_object* v___y_328_; lean_object* v___y_329_; lean_object* v___y_330_; lean_object* v___y_345_; lean_object* v___y_346_; lean_object* v___y_347_; lean_object* v___y_348_; lean_object* v___y_349_; lean_object* v___y_350_; uint8_t v___x_361_; 
v_binderType_321_ = lean_ctor_get(v_a_320_, 1);
v_body_322_ = lean_ctor_get(v_a_320_, 2);
lean_inc_ref(v_body_322_);
v___f_323_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___lam__0___boxed), 10, 2);
lean_closure_set(v___f_323_, 0, v_mkCoe_311_);
lean_closure_set(v___f_323_, 1, v_body_322_);
v___x_361_ = l_Lean_Expr_hasLooseBVars(v_body_322_);
if (v___x_361_ == 0)
{
lean_inc_ref(v_binderType_321_);
lean_dec_ref_known(v_a_320_, 3);
lean_dec_ref(v_sym_309_);
v___y_345_ = v_a_312_;
v___y_346_ = v_a_313_;
v___y_347_ = v_a_314_;
v___y_348_ = v_a_315_;
v___y_349_ = v_a_316_;
v___y_350_ = v_a_317_;
goto v___jp_344_;
}
else
{
lean_object* v___x_362_; 
lean_dec_ref(v___f_323_);
v___x_362_ = l_Lean_Elab_Term_tryPostpone(v_a_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_);
if (lean_obj_tag(v___x_362_) == 0)
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v_a_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_378_; 
lean_dec_ref_known(v___x_362_, 1);
v___x_363_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3, &lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3);
v___x_364_ = l_Lean_stringToMessageData(v_sym_309_);
v___x_365_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_363_);
lean_ctor_set(v___x_365_, 1, v___x_364_);
v___x_366_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__5, &lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__5_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__5);
v___x_367_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_367_, 0, v___x_365_);
lean_ctor_set(v___x_367_, 1, v___x_366_);
v___x_368_ = l_Lean_indentExpr(v_a_320_);
v___x_369_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_367_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
v___x_370_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(v___x_369_, v_a_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_);
v_a_371_ = lean_ctor_get(v___x_370_, 0);
v_isSharedCheck_378_ = !lean_is_exclusive(v___x_370_);
if (v_isSharedCheck_378_ == 0)
{
v___x_373_ = v___x_370_;
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_a_371_);
lean_dec(v___x_370_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_376_; 
if (v_isShared_374_ == 0)
{
v___x_376_ = v___x_373_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_a_371_);
v___x_376_ = v_reuseFailAlloc_377_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
return v___x_376_;
}
}
}
else
{
lean_object* v_a_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_386_; 
lean_dec_ref_known(v_a_320_, 3);
lean_dec_ref(v_sym_309_);
v_a_379_ = lean_ctor_get(v___x_362_, 0);
v_isSharedCheck_386_ = !lean_is_exclusive(v___x_362_);
if (v_isSharedCheck_386_ == 0)
{
v___x_381_ = v___x_362_;
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_a_379_);
lean_dec(v___x_362_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_384_; 
if (v_isShared_382_ == 0)
{
v___x_384_ = v___x_381_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v_a_379_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
}
}
v___jp_324_:
{
lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_331_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__1));
v___x_332_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___redArg(v___x_331_, v_binderType_321_, v___f_323_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_, v___y_330_);
if (lean_obj_tag(v___x_332_) == 0)
{
lean_object* v_a_333_; lean_object* v___x_334_; 
v_a_333_ = lean_ctor_get(v___x_332_, 0);
lean_inc(v_a_333_);
v___x_334_ = l_Lean_Expr_etaExpanded_x3f(v_a_333_);
if (lean_obj_tag(v___x_334_) == 0)
{
return v___x_332_;
}
else
{
lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_342_; 
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_342_ == 0)
{
lean_object* v_unused_343_; 
v_unused_343_ = lean_ctor_get(v___x_332_, 0);
lean_dec(v_unused_343_);
v___x_336_ = v___x_332_;
v_isShared_337_ = v_isSharedCheck_342_;
goto v_resetjp_335_;
}
else
{
lean_dec(v___x_332_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_342_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v_val_338_; lean_object* v___x_340_; 
v_val_338_ = lean_ctor_get(v___x_334_, 0);
lean_inc(v_val_338_);
lean_dec_ref_known(v___x_334_, 1);
if (v_isShared_337_ == 0)
{
lean_ctor_set(v___x_336_, 0, v_val_338_);
v___x_340_ = v___x_336_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_val_338_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
return v___x_340_;
}
}
}
}
else
{
return v___x_332_;
}
}
v___jp_344_:
{
uint8_t v___x_351_; 
v___x_351_ = l_Lean_Expr_hasExprMVar(v_binderType_321_);
if (v___x_351_ == 0)
{
v___y_325_ = v___y_345_;
v___y_326_ = v___y_346_;
v___y_327_ = v___y_347_;
v___y_328_ = v___y_348_;
v___y_329_ = v___y_349_;
v___y_330_ = v___y_350_;
goto v___jp_324_;
}
else
{
lean_object* v___x_352_; 
v___x_352_ = l_Lean_Elab_Term_tryPostpone(v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_, v___y_350_);
if (lean_obj_tag(v___x_352_) == 0)
{
lean_dec_ref_known(v___x_352_, 1);
v___y_325_ = v___y_345_;
v___y_326_ = v___y_346_;
v___y_327_ = v___y_347_;
v___y_328_ = v___y_348_;
v___y_329_ = v___y_349_;
v___y_330_ = v___y_350_;
goto v___jp_324_;
}
else
{
lean_object* v_a_353_; lean_object* v___x_355_; uint8_t v_isShared_356_; uint8_t v_isSharedCheck_360_; 
lean_dec_ref(v___f_323_);
lean_dec_ref(v_binderType_321_);
v_a_353_ = lean_ctor_get(v___x_352_, 0);
v_isSharedCheck_360_ = !lean_is_exclusive(v___x_352_);
if (v_isSharedCheck_360_ == 0)
{
v___x_355_ = v___x_352_;
v_isShared_356_ = v_isSharedCheck_360_;
goto v_resetjp_354_;
}
else
{
lean_inc(v_a_353_);
lean_dec(v___x_352_);
v___x_355_ = lean_box(0);
v_isShared_356_ = v_isSharedCheck_360_;
goto v_resetjp_354_;
}
v_resetjp_354_:
{
lean_object* v___x_358_; 
if (v_isShared_356_ == 0)
{
v___x_358_ = v___x_355_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v_a_353_);
v___x_358_ = v_reuseFailAlloc_359_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
return v___x_358_;
}
}
}
}
}
}
else
{
lean_object* v___x_387_; 
lean_dec_ref(v_mkCoe_311_);
v___x_387_ = l_Lean_Elab_Term_tryPostpone(v_a_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; 
lean_dec_ref_known(v___x_387_, 1);
v___x_388_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3, &lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__3);
v___x_389_ = l_Lean_stringToMessageData(v_sym_309_);
v___x_390_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_388_);
lean_ctor_set(v___x_390_, 1, v___x_389_);
v___x_391_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__7, &lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__7_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___closed__7);
v___x_392_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_390_);
lean_ctor_set(v___x_392_, 1, v___x_391_);
v___x_393_ = l_Lean_indentExpr(v_a_320_);
v___x_394_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_392_);
lean_ctor_set(v___x_394_, 1, v___x_393_);
v___x_395_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(v___x_394_, v_a_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_);
return v___x_395_;
}
else
{
lean_object* v_a_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_403_; 
lean_dec(v_a_320_);
lean_dec_ref(v_sym_309_);
v_a_396_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_403_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_403_ == 0)
{
v___x_398_ = v___x_387_;
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_a_396_);
lean_dec(v___x_387_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
lean_object* v___x_401_; 
if (v_isShared_399_ == 0)
{
v___x_401_ = v___x_398_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v_a_396_);
v___x_401_ = v_reuseFailAlloc_402_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
return v___x_401_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe___boxed(lean_object* v_sym_404_, lean_object* v_expectedType_405_, lean_object* v_mkCoe_406_, lean_object* v_a_407_, lean_object* v_a_408_, lean_object* v_a_409_, lean_object* v_a_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe(v_sym_404_, v_expectedType_405_, v_mkCoe_406_, v_a_407_, v_a_408_, v_a_409_, v_a_410_, v_a_411_, v_a_412_);
lean_dec(v_a_412_);
lean_dec_ref(v_a_411_);
lean_dec(v_a_410_);
lean_dec_ref(v_a_409_);
lean_dec(v_a_408_);
lean_dec_ref(v_a_407_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1(lean_object* v_00_u03b1_415_, lean_object* v_name_416_, uint8_t v_bi_417_, lean_object* v_type_418_, lean_object* v_k_419_, uint8_t v_kind_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___redArg(v_name_416_, v_bi_417_, v_type_418_, v_k_419_, v_kind_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1___boxed(lean_object* v_00_u03b1_429_, lean_object* v_name_430_, lean_object* v_bi_431_, lean_object* v_type_432_, lean_object* v_k_433_, lean_object* v_kind_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
uint8_t v_bi_boxed_442_; uint8_t v_kind_boxed_443_; lean_object* v_res_444_; 
v_bi_boxed_442_ = lean_unbox(v_bi_431_);
v_kind_boxed_443_ = lean_unbox(v_kind_434_);
v_res_444_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1_spec__1(v_00_u03b1_429_, v_name_430_, v_bi_boxed_442_, v_type_432_, v_k_433_, v_kind_boxed_443_, v___y_435_, v___y_436_, v___y_437_, v___y_438_, v___y_439_, v___y_440_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
lean_dec(v___y_436_);
lean_dec_ref(v___y_435_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1(lean_object* v_00_u03b1_445_, lean_object* v_name_446_, lean_object* v_type_447_, lean_object* v_k_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___redArg(v_name_446_, v_type_447_, v_k_448_, v___y_449_, v___y_450_, v___y_451_, v___y_452_, v___y_453_, v___y_454_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1___boxed(lean_object* v_00_u03b1_457_, lean_object* v_name_458_, lean_object* v_type_459_, lean_object* v_k_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__1(v_00_u03b1_457_, v_name_458_, v_type_459_, v_k_460_, v___y_461_, v___y_462_, v___y_463_, v___y_464_, v___y_465_, v___y_466_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
lean_dec(v___y_464_);
lean_dec_ref(v___y_463_);
lean_dec(v___y_462_);
lean_dec_ref(v___y_461_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2(lean_object* v_00_u03b1_469_, lean_object* v_msg_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(v_msg_470_, v___y_471_, v___y_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___boxed(lean_object* v_00_u03b1_479_, lean_object* v_msg_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_){
_start:
{
lean_object* v_res_488_; 
v_res_488_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2(v_00_u03b1_479_, v_msg_480_, v___y_481_, v___y_482_, v___y_483_, v___y_484_, v___y_485_, v___y_486_);
lean_dec(v___y_486_);
lean_dec_ref(v___y_485_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec(v___y_482_);
lean_dec_ref(v___y_481_);
return v_res_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4(lean_object* v_msgData_489_, lean_object* v_macroStack_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___redArg(v_msgData_489_, v_macroStack_490_, v___y_495_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4___boxed(lean_object* v_msgData_499_, lean_object* v_macroStack_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2_spec__4(v_msgData_499_, v_macroStack_500_, v___y_501_, v___y_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
lean_dec(v___y_506_);
lean_dec_ref(v___y_505_);
lean_dec(v___y_504_);
lean_dec_ref(v___y_503_);
lean_dec(v___y_502_);
lean_dec_ref(v___y_501_);
return v_res_508_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; 
v___x_544_ = lean_box(0);
v___x_545_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_546_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_546_, 0, v___x_545_);
lean_ctor_set(v___x_546_, 1, v___x_544_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg(){
_start:
{
lean_object* v___x_548_; lean_object* v___x_549_; 
v___x_548_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___closed__0);
v___x_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_549_, 0, v___x_548_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg___boxed(lean_object* v___y_550_){
_start:
{
lean_object* v_res_551_; 
v_res_551_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg();
return v_res_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0(lean_object* v_00_u03b1_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg();
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___boxed(lean_object* v_00_u03b1_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_){
_start:
{
lean_object* v_res_569_; 
v_res_569_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0(v_00_u03b1_561_, v___y_562_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_);
lean_dec(v___y_567_);
lean_dec_ref(v___y_566_);
lean_dec(v___y_565_);
lean_dec_ref(v___y_564_);
lean_dec(v___y_563_);
lean_dec_ref(v___y_562_);
return v_res_569_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_571_; lean_object* v___x_572_; 
v___x_571_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__0));
v___x_572_ = l_Lean_stringToMessageData(v___x_571_);
return v___x_572_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__3(void){
_start:
{
lean_object* v___x_574_; lean_object* v___x_575_; 
v___x_574_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__2));
v___x_575_ = l_Lean_stringToMessageData(v___x_574_);
return v___x_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0(lean_object* v_b_576_, lean_object* v_x_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_){
_start:
{
uint8_t v___x_612_; 
v___x_612_ = l_Lean_Expr_hasExprMVar(v_b_576_);
if (v___x_612_ == 0)
{
goto v___jp_585_;
}
else
{
lean_object* v___x_613_; 
v___x_613_ = l_Lean_Elab_Term_tryPostpone(v___y_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
if (lean_obj_tag(v___x_613_) == 0)
{
lean_dec_ref_known(v___x_613_, 1);
goto v___jp_585_;
}
else
{
lean_object* v_a_614_; lean_object* v___x_616_; uint8_t v_isShared_617_; uint8_t v_isSharedCheck_621_; 
lean_dec_ref(v_x_577_);
lean_dec_ref(v_b_576_);
v_a_614_ = lean_ctor_get(v___x_613_, 0);
v_isSharedCheck_621_ = !lean_is_exclusive(v___x_613_);
if (v_isSharedCheck_621_ == 0)
{
v___x_616_ = v___x_613_;
v_isShared_617_ = v_isSharedCheck_621_;
goto v_resetjp_615_;
}
else
{
lean_inc(v_a_614_);
lean_dec(v___x_613_);
v___x_616_ = lean_box(0);
v_isShared_617_ = v_isSharedCheck_621_;
goto v_resetjp_615_;
}
v_resetjp_615_:
{
lean_object* v___x_619_; 
if (v_isShared_617_ == 0)
{
v___x_619_ = v___x_616_;
goto v_reusejp_618_;
}
else
{
lean_object* v_reuseFailAlloc_620_; 
v_reuseFailAlloc_620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_620_, 0, v_a_614_);
v___x_619_ = v_reuseFailAlloc_620_;
goto v_reusejp_618_;
}
v_reusejp_618_:
{
return v___x_619_;
}
}
}
}
v___jp_585_:
{
lean_object* v___x_586_; 
lean_inc_ref(v_b_576_);
lean_inc_ref(v_x_577_);
v___x_586_ = l_Lean_Meta_coerce_x3f(v_x_577_, v_b_576_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
if (lean_obj_tag(v___x_586_) == 0)
{
lean_object* v_a_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_603_; 
v_a_587_ = lean_ctor_get(v___x_586_, 0);
v_isSharedCheck_603_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_603_ == 0)
{
v___x_589_ = v___x_586_;
v_isShared_590_ = v_isSharedCheck_603_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_a_587_);
lean_dec(v___x_586_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_603_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
if (lean_obj_tag(v_a_587_) == 1)
{
lean_object* v_a_591_; lean_object* v___x_593_; 
lean_dec_ref(v_x_577_);
lean_dec_ref(v_b_576_);
v_a_591_ = lean_ctor_get(v_a_587_, 0);
lean_inc(v_a_591_);
lean_dec_ref_known(v_a_587_, 1);
if (v_isShared_590_ == 0)
{
lean_ctor_set(v___x_589_, 0, v_a_591_);
v___x_593_ = v___x_589_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v_a_591_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
else
{
lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
lean_del_object(v___x_589_);
lean_dec(v_a_587_);
v___x_595_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__1, &lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__1_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__1);
v___x_596_ = l_Lean_indentExpr(v_x_577_);
v___x_597_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_597_, 0, v___x_595_);
lean_ctor_set(v___x_597_, 1, v___x_596_);
v___x_598_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__3, &lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__3_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___closed__3);
v___x_599_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_599_, 0, v___x_597_);
lean_ctor_set(v___x_599_, 1, v___x_598_);
v___x_600_ = l_Lean_indentExpr(v_b_576_);
v___x_601_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_601_, 0, v___x_599_);
lean_ctor_set(v___x_601_, 1, v___x_600_);
v___x_602_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(v___x_601_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
return v___x_602_;
}
}
}
else
{
lean_object* v_a_604_; lean_object* v___x_606_; uint8_t v_isShared_607_; uint8_t v_isSharedCheck_611_; 
lean_dec_ref(v_x_577_);
lean_dec_ref(v_b_576_);
v_a_604_ = lean_ctor_get(v___x_586_, 0);
v_isSharedCheck_611_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_611_ == 0)
{
v___x_606_ = v___x_586_;
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
else
{
lean_inc(v_a_604_);
lean_dec(v___x_586_);
v___x_606_ = lean_box(0);
v_isShared_607_ = v_isSharedCheck_611_;
goto v_resetjp_605_;
}
v_resetjp_605_:
{
lean_object* v___x_609_; 
if (v_isShared_607_ == 0)
{
v___x_609_ = v___x_606_;
goto v_reusejp_608_;
}
else
{
lean_object* v_reuseFailAlloc_610_; 
v_reuseFailAlloc_610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_610_, 0, v_a_604_);
v___x_609_ = v_reuseFailAlloc_610_;
goto v_reusejp_608_;
}
v_reusejp_608_:
{
return v___x_609_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0___boxed(lean_object* v_b_622_, lean_object* v_x_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_){
_start:
{
lean_object* v_res_631_; 
v_res_631_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__0(v_b_622_, v_x_623_, v___y_624_, v___y_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_);
lean_dec(v___y_629_);
lean_dec_ref(v___y_628_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
lean_dec(v___y_625_);
lean_dec_ref(v___y_624_);
return v_res_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__1(lean_object* v_stx_632_, lean_object* v___f_633_, lean_object* v_expectedType_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_){
_start:
{
lean_object* v___x_642_; uint8_t v___x_643_; 
v___x_642_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__5));
v___x_643_ = l_Lean_Syntax_isOfKind(v_stx_632_, v___x_642_);
if (v___x_643_ == 0)
{
lean_object* v___x_644_; 
lean_dec_ref(v_expectedType_634_);
lean_dec_ref(v___f_633_);
v___x_644_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg();
return v___x_644_;
}
else
{
lean_object* v___x_645_; lean_object* v___x_646_; 
v___x_645_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u2191_x29___closed__9));
v___x_646_ = lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe(v___x_645_, v_expectedType_634_, v___f_633_, v___y_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_, v___y_640_);
return v___x_646_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__1___boxed(lean_object* v_stx_647_, lean_object* v___f_648_, lean_object* v_expectedType_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
lean_object* v_res_657_; 
v_res_657_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__1(v_stx_647_, v___f_648_, v_expectedType_649_, v___y_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
lean_dec(v___y_653_);
lean_dec_ref(v___y_652_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1(lean_object* v_stx_659_, lean_object* v_expectedType_x3f_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_){
_start:
{
lean_object* v___f_668_; lean_object* v___f_669_; lean_object* v___x_670_; 
v___f_668_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___closed__0));
v___f_669_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___lam__1___boxed), 10, 2);
lean_closure_set(v___f_669_, 0, v_stx_659_);
lean_closure_set(v___f_669_, 1, v___f_668_);
v___x_670_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_660_, v___f_669_, v_a_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_);
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1___boxed(lean_object* v_stx_671_, lean_object* v_expectedType_x3f_672_, lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_, lean_object* v_a_676_, lean_object* v_a_677_, lean_object* v_a_678_, lean_object* v_a_679_){
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1(v_stx_671_, v_expectedType_x3f_672_, v_a_673_, v_a_674_, v_a_675_, v_a_676_, v_a_677_, v_a_678_);
lean_dec(v_a_678_);
lean_dec_ref(v_a_677_);
lean_dec(v_a_676_);
lean_dec_ref(v_a_675_);
lean_dec(v_a_674_);
lean_dec_ref(v_a_673_);
return v_res_680_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_705_; lean_object* v___x_706_; 
v___x_705_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__0));
v___x_706_ = l_Lean_stringToMessageData(v___x_705_);
return v___x_706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0(lean_object* v_b_707_, lean_object* v_x_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_){
_start:
{
lean_object* v___x_716_; 
lean_inc_ref(v_x_708_);
v___x_716_ = l_Lean_Meta_coerceToFunction_x3f(v_x_708_, v___y_711_, v___y_712_, v___y_713_, v___y_714_);
if (lean_obj_tag(v___x_716_) == 0)
{
lean_object* v_a_717_; 
v_a_717_ = lean_ctor_get(v___x_716_, 0);
lean_inc(v_a_717_);
lean_dec_ref_known(v___x_716_, 1);
if (lean_obj_tag(v_a_717_) == 1)
{
lean_object* v_val_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_727_; 
lean_dec_ref(v_x_708_);
v_val_718_ = lean_ctor_get(v_a_717_, 0);
v_isSharedCheck_727_ = !lean_is_exclusive(v_a_717_);
if (v_isSharedCheck_727_ == 0)
{
v___x_720_ = v_a_717_;
v_isShared_721_ = v_isSharedCheck_727_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_val_718_);
lean_dec(v_a_717_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_727_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_723_; 
if (v_isShared_721_ == 0)
{
lean_ctor_set(v___x_720_, 0, v_b_707_);
v___x_723_ = v___x_720_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_726_; 
v_reuseFailAlloc_726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_726_, 0, v_b_707_);
v___x_723_ = v_reuseFailAlloc_726_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_724_ = lean_box(0);
v___x_725_ = l_Lean_Elab_Term_ensureHasType(v___x_723_, v_val_718_, v___x_724_, v___x_724_, v___y_709_, v___y_710_, v___y_711_, v___y_712_, v___y_713_, v___y_714_);
return v___x_725_;
}
}
}
else
{
lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; 
lean_dec(v_a_717_);
lean_dec_ref(v_b_707_);
v___x_728_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__1, &lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__1_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___closed__1);
v___x_729_ = l_Lean_indentExpr(v_x_708_);
v___x_730_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_728_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(v___x_730_, v___y_709_, v___y_710_, v___y_711_, v___y_712_, v___y_713_, v___y_714_);
return v___x_731_;
}
}
else
{
lean_object* v_a_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_739_; 
lean_dec_ref(v_x_708_);
lean_dec_ref(v_b_707_);
v_a_732_ = lean_ctor_get(v___x_716_, 0);
v_isSharedCheck_739_ = !lean_is_exclusive(v___x_716_);
if (v_isSharedCheck_739_ == 0)
{
v___x_734_ = v___x_716_;
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_a_732_);
lean_dec(v___x_716_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_737_; 
if (v_isShared_735_ == 0)
{
v___x_737_ = v___x_734_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_738_; 
v_reuseFailAlloc_738_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_738_, 0, v_a_732_);
v___x_737_ = v_reuseFailAlloc_738_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
return v___x_737_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0___boxed(lean_object* v_b_740_, lean_object* v_x_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_){
_start:
{
lean_object* v_res_749_; 
v_res_749_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__0(v_b_740_, v_x_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_, v___y_746_, v___y_747_);
lean_dec(v___y_747_);
lean_dec_ref(v___y_746_);
lean_dec(v___y_745_);
lean_dec_ref(v___y_744_);
lean_dec(v___y_743_);
lean_dec_ref(v___y_742_);
return v_res_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__1(lean_object* v_stx_750_, lean_object* v___f_751_, lean_object* v_expectedType_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_){
_start:
{
lean_object* v___x_760_; uint8_t v___x_761_; 
v___x_760_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__1));
v___x_761_ = l_Lean_Syntax_isOfKind(v_stx_750_, v___x_760_);
if (v___x_761_ == 0)
{
lean_object* v___x_762_; 
lean_dec_ref(v_expectedType_752_);
lean_dec_ref(v___f_751_);
v___x_762_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg();
return v___x_762_;
}
else
{
lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_763_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21d1_x29___closed__2));
v___x_764_ = lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe(v___x_763_, v_expectedType_752_, v___f_751_, v___y_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_, v___y_758_);
return v___x_764_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__1___boxed(lean_object* v_stx_765_, lean_object* v___f_766_, lean_object* v_expectedType_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__1(v_stx_765_, v___f_766_, v_expectedType_767_, v___y_768_, v___y_769_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
lean_dec(v___y_769_);
lean_dec_ref(v___y_768_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1(lean_object* v_stx_777_, lean_object* v_expectedType_x3f_778_, lean_object* v_a_779_, lean_object* v_a_780_, lean_object* v_a_781_, lean_object* v_a_782_, lean_object* v_a_783_, lean_object* v_a_784_){
_start:
{
lean_object* v___f_786_; lean_object* v___f_787_; lean_object* v___x_788_; 
v___f_786_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___closed__0));
v___f_787_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___lam__1___boxed), 10, 2);
lean_closure_set(v___f_787_, 0, v_stx_777_);
lean_closure_set(v___f_787_, 1, v___f_786_);
v___x_788_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_778_, v___f_787_, v_a_779_, v_a_780_, v_a_781_, v_a_782_, v_a_783_, v_a_784_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1___boxed(lean_object* v_stx_789_, lean_object* v_expectedType_x3f_790_, lean_object* v_a_791_, lean_object* v_a_792_, lean_object* v_a_793_, lean_object* v_a_794_, lean_object* v_a_795_, lean_object* v_a_796_, lean_object* v_a_797_){
_start:
{
lean_object* v_res_798_; 
v_res_798_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21d1_x29__1(v_stx_789_, v_expectedType_x3f_790_, v_a_791_, v_a_792_, v_a_793_, v_a_794_, v_a_795_, v_a_796_);
lean_dec(v_a_796_);
lean_dec_ref(v_a_795_);
lean_dec(v_a_794_);
lean_dec_ref(v_a_793_);
lean_dec(v_a_792_);
lean_dec_ref(v_a_791_);
return v_res_798_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_823_; lean_object* v___x_824_; 
v___x_823_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__0));
v___x_824_ = l_Lean_stringToMessageData(v___x_823_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0(lean_object* v_b_825_, lean_object* v_x_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_){
_start:
{
lean_object* v___x_834_; 
lean_inc_ref(v_x_826_);
v___x_834_ = l_Lean_Meta_coerceToSort_x3f(v_x_826_, v___y_829_, v___y_830_, v___y_831_, v___y_832_);
if (lean_obj_tag(v___x_834_) == 0)
{
lean_object* v_a_835_; 
v_a_835_ = lean_ctor_get(v___x_834_, 0);
lean_inc(v_a_835_);
lean_dec_ref_known(v___x_834_, 1);
if (lean_obj_tag(v_a_835_) == 1)
{
lean_object* v_val_836_; lean_object* v___x_838_; uint8_t v_isShared_839_; uint8_t v_isSharedCheck_845_; 
lean_dec_ref(v_x_826_);
v_val_836_ = lean_ctor_get(v_a_835_, 0);
v_isSharedCheck_845_ = !lean_is_exclusive(v_a_835_);
if (v_isSharedCheck_845_ == 0)
{
v___x_838_ = v_a_835_;
v_isShared_839_ = v_isSharedCheck_845_;
goto v_resetjp_837_;
}
else
{
lean_inc(v_val_836_);
lean_dec(v_a_835_);
v___x_838_ = lean_box(0);
v_isShared_839_ = v_isSharedCheck_845_;
goto v_resetjp_837_;
}
v_resetjp_837_:
{
lean_object* v___x_841_; 
if (v_isShared_839_ == 0)
{
lean_ctor_set(v___x_838_, 0, v_b_825_);
v___x_841_ = v___x_838_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v_b_825_);
v___x_841_ = v_reuseFailAlloc_844_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
lean_object* v___x_842_; lean_object* v___x_843_; 
v___x_842_ = lean_box(0);
v___x_843_ = l_Lean_Elab_Term_ensureHasType(v___x_841_, v_val_836_, v___x_842_, v___x_842_, v___y_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_);
return v___x_843_;
}
}
}
else
{
lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; 
lean_dec(v_a_835_);
lean_dec_ref(v_b_825_);
v___x_846_ = lean_obj_once(&lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__1, &lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__1_once, _init_lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___closed__1);
v___x_847_ = l_Lean_indentExpr(v_x_826_);
v___x_848_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_848_, 0, v___x_846_);
lean_ctor_set(v___x_848_, 1, v___x_847_);
v___x_849_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe_spec__2___redArg(v___x_848_, v___y_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_);
return v___x_849_;
}
}
else
{
lean_object* v_a_850_; lean_object* v___x_852_; uint8_t v_isShared_853_; uint8_t v_isSharedCheck_857_; 
lean_dec_ref(v_x_826_);
lean_dec_ref(v_b_825_);
v_a_850_ = lean_ctor_get(v___x_834_, 0);
v_isSharedCheck_857_ = !lean_is_exclusive(v___x_834_);
if (v_isSharedCheck_857_ == 0)
{
v___x_852_ = v___x_834_;
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
else
{
lean_inc(v_a_850_);
lean_dec(v___x_834_);
v___x_852_ = lean_box(0);
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
v_resetjp_851_:
{
lean_object* v___x_855_; 
if (v_isShared_853_ == 0)
{
v___x_855_ = v___x_852_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v_a_850_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
return v___x_855_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0___boxed(lean_object* v_b_858_, lean_object* v_x_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_){
_start:
{
lean_object* v_res_867_; 
v_res_867_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__0(v_b_858_, v_x_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_, v___y_865_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
lean_dec(v___y_863_);
lean_dec_ref(v___y_862_);
lean_dec(v___y_861_);
lean_dec_ref(v___y_860_);
return v_res_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__1(lean_object* v_stx_868_, lean_object* v___f_869_, lean_object* v_expectedType_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_){
_start:
{
lean_object* v___x_878_; uint8_t v___x_879_; 
v___x_878_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__1));
v___x_879_ = l_Lean_Syntax_isOfKind(v_stx_868_, v___x_878_);
if (v___x_879_ == 0)
{
lean_object* v___x_880_; 
lean_dec_ref(v_expectedType_870_);
lean_dec_ref(v___f_869_);
v___x_880_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u2191_x29__1_spec__0___redArg();
return v___x_880_;
}
else
{
lean_object* v___x_881_; lean_object* v___x_882_; 
v___x_881_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl_term_x28_u21a5_x29___closed__2));
v___x_882_ = lp_mathlib_Lean_Elab_Term_CoeImpl_elabPartiallyAppliedCoe(v___x_881_, v_expectedType_870_, v___f_869_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
return v___x_882_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__1___boxed(lean_object* v_stx_883_, lean_object* v___f_884_, lean_object* v_expectedType_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_){
_start:
{
lean_object* v_res_893_; 
v_res_893_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__1(v_stx_883_, v___f_884_, v_expectedType_885_, v___y_886_, v___y_887_, v___y_888_, v___y_889_, v___y_890_, v___y_891_);
lean_dec(v___y_891_);
lean_dec_ref(v___y_890_);
lean_dec(v___y_889_);
lean_dec_ref(v___y_888_);
lean_dec(v___y_887_);
lean_dec_ref(v___y_886_);
return v_res_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1(lean_object* v_stx_895_, lean_object* v_expectedType_x3f_896_, lean_object* v_a_897_, lean_object* v_a_898_, lean_object* v_a_899_, lean_object* v_a_900_, lean_object* v_a_901_, lean_object* v_a_902_){
_start:
{
lean_object* v___f_904_; lean_object* v___f_905_; lean_object* v___x_906_; 
v___f_904_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___closed__0));
v___f_905_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___lam__1___boxed), 10, 2);
lean_closure_set(v___f_905_, 0, v_stx_895_);
lean_closure_set(v___f_905_, 1, v___f_904_);
v___x_906_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_896_, v___f_905_, v_a_897_, v_a_898_, v_a_899_, v_a_900_, v_a_901_, v_a_902_);
return v___x_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1___boxed(lean_object* v_stx_907_, lean_object* v_expectedType_x3f_908_, lean_object* v_a_909_, lean_object* v_a_910_, lean_object* v_a_911_, lean_object* v_a_912_, lean_object* v_a_913_, lean_object* v_a_914_, lean_object* v_a_915_){
_start:
{
lean_object* v_res_916_; 
v_res_916_ = lp_mathlib_Lean_Elab_Term_CoeImpl___aux__Mathlib__Tactic__Coe______elabRules__Lean__Elab__Term__CoeImpl__term_x28_u21a5_x29__1(v_stx_907_, v_expectedType_x3f_908_, v_a_909_, v_a_910_, v_a_911_, v_a_912_, v_a_913_, v_a_914_);
lean_dec(v_a_914_);
lean_dec_ref(v_a_913_);
lean_dec(v_a_912_);
lean_dec_ref(v_a_911_);
lean_dec(v_a_910_);
lean_dec_ref(v_a_909_);
return v_res_916_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Coe(builtin);
}
#ifdef __cplusplus
}
#endif
