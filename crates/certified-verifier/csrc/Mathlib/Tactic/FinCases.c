// Lean compiler output
// Module: Mathlib.Tactic.FinCases
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.Core public meta import Mathlib.Lean.Expr.Basic public import Mathlib.Data.Finset.Attr public import Mathlib.Data.Fintype.Defs public meta import Mathlib.Tactic.ToDual
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_MVarId_cases(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_setUserName___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_Tactic_getFVarId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_MVarId_assert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_allGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_focus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 104, .m_capacity = 104, .m_length = 94, .m_data = "Hypothesis must be of type `x ∈ (A : List α)`, `x ∈ (A : Finset α)`, or `x ∈ (A : Multiset α)`"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Membership"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mem"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Multiset"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "unexpected number of cases"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_unfoldCases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_unfoldCases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___at___00Lean_Elab_Tactic_finCasesAt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___at___00Lean_Elab_Tactic_finCasesAt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Fintype"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 129, 114, 60, 203, 137, 135, 143)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "elems"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 129, 114, 60, 203, 137, 135, 143)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 101, 225, 201, 119, 19, 141, 18)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 217, 109, 94, 255, 55, 82, 109)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(224, 90, 126, 237, 128, 148, 153, 69)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "complete"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 129, 114, 60, 203, 137, 135, 143)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(225, 96, 51, 148, 8, 144, 12, 108)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "this"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(38, 116, 214, 236, 212, 160, 188, 150)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "finCases"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__3_value),LEAN_SCALAR_PTR_LITERAL(81, 122, 97, 52, 141, 152, 225, 169)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "fin_cases "};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__9_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__11_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "token"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__12_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__11_value),LEAN_SCALAR_PTR_LITERAL(46, 123, 149, 63, 0, 221, 179, 78)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__11_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__11_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__13_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__15_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__16_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__17 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__18_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__19_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__20 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__20_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__20_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__21 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__21_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__18_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__19_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__21_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__22 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__22_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__10_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__15_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__22_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__23 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__23_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__23_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__24 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__24_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__25 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__25_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__25_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__26 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__26_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__27 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__27_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__27_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__28 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__28_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__28_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__22_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__29 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__29_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__26_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__29_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__30 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__30_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__24_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__30_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__31 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__31_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_finCases___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__31_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases___closed__32 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__32_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Tactic_finCases = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_finCases___closed__32_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___lam__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__3(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___boxed__const__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__0));
v___x_3_ = l_Lean_stringToMessageData(v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___redArg(lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_e_11_){
_start:
{
lean_object* v_toApplicative_15_; lean_object* v_toPure_16_; lean_object* v_00_u03b1_18_; lean_object* v___x_24_; lean_object* v_fst_25_; 
v_toApplicative_15_ = lean_ctor_get(v_inst_9_, 0);
v_toPure_16_ = lean_ctor_get(v_toApplicative_15_, 1);
v___x_24_ = l_Lean_Expr_getAppFnArgs(v_e_11_);
v_fst_25_ = lean_ctor_get(v___x_24_, 0);
lean_inc(v_fst_25_);
if (lean_obj_tag(v_fst_25_) == 1)
{
lean_object* v_pre_26_; 
v_pre_26_ = lean_ctor_get(v_fst_25_, 0);
lean_inc(v_pre_26_);
if (lean_obj_tag(v_pre_26_) == 1)
{
lean_object* v_pre_27_; 
v_pre_27_ = lean_ctor_get(v_pre_26_, 0);
if (lean_obj_tag(v_pre_27_) == 0)
{
lean_object* v_snd_28_; lean_object* v_str_29_; lean_object* v_str_30_; lean_object* v___x_31_; uint8_t v___x_32_; 
v_snd_28_ = lean_ctor_get(v___x_24_, 1);
lean_inc(v_snd_28_);
lean_dec_ref(v___x_24_);
v_str_29_ = lean_ctor_get(v_fst_25_, 1);
lean_inc_ref(v_str_29_);
lean_dec_ref_known(v_fst_25_, 2);
v_str_30_ = lean_ctor_get(v_pre_26_, 1);
lean_inc_ref(v_str_30_);
lean_dec_ref_known(v_pre_26_, 2);
v___x_31_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__2));
v___x_32_ = lean_string_dec_eq(v_str_30_, v___x_31_);
lean_dec_ref(v_str_30_);
if (v___x_32_ == 0)
{
lean_inc(v_toPure_16_);
lean_dec_ref(v_str_29_);
lean_dec(v_snd_28_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
goto v___jp_21_;
}
else
{
lean_object* v___x_33_; uint8_t v___x_34_; 
v___x_33_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__3));
v___x_34_ = lean_string_dec_eq(v_str_29_, v___x_33_);
lean_dec_ref(v_str_29_);
if (v___x_34_ == 0)
{
lean_inc(v_toPure_16_);
lean_dec(v_snd_28_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
goto v___jp_21_;
}
else
{
lean_object* v___x_35_; lean_object* v___x_36_; uint8_t v___x_37_; 
v___x_35_ = lean_array_get_size(v_snd_28_);
v___x_36_ = lean_unsigned_to_nat(5u);
v___x_37_ = lean_nat_dec_eq(v___x_35_, v___x_36_);
if (v___x_37_ == 0)
{
lean_inc(v_toPure_16_);
lean_dec(v_snd_28_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
goto v___jp_21_;
}
else
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v_fst_41_; 
v___x_38_ = lean_unsigned_to_nat(1u);
v___x_39_ = lean_array_fget(v_snd_28_, v___x_38_);
lean_dec(v_snd_28_);
v___x_40_ = l_Lean_Expr_getAppFnArgs(v___x_39_);
v_fst_41_ = lean_ctor_get(v___x_40_, 0);
lean_inc(v_fst_41_);
if (lean_obj_tag(v_fst_41_) == 1)
{
lean_object* v_pre_42_; 
v_pre_42_ = lean_ctor_get(v_fst_41_, 0);
if (lean_obj_tag(v_pre_42_) == 0)
{
lean_object* v_snd_43_; lean_object* v_str_44_; lean_object* v___x_45_; uint8_t v___x_46_; 
v_snd_43_ = lean_ctor_get(v___x_40_, 1);
lean_inc(v_snd_43_);
lean_dec_ref(v___x_40_);
v_str_44_ = lean_ctor_get(v_fst_41_, 1);
lean_inc_ref(v_str_44_);
lean_dec_ref_known(v_fst_41_, 2);
v___x_45_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__4));
v___x_46_ = lean_string_dec_eq(v_str_44_, v___x_45_);
if (v___x_46_ == 0)
{
lean_object* v___x_47_; uint8_t v___x_48_; 
v___x_47_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__5));
v___x_48_ = lean_string_dec_eq(v_str_44_, v___x_47_);
if (v___x_48_ == 0)
{
lean_object* v___x_49_; uint8_t v___x_50_; 
v___x_49_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__6));
v___x_50_ = lean_string_dec_eq(v_str_44_, v___x_49_);
lean_dec_ref(v_str_44_);
if (v___x_50_ == 0)
{
lean_dec(v_snd_43_);
goto v___jp_12_;
}
else
{
lean_object* v___x_51_; uint8_t v___x_52_; 
v___x_51_ = lean_array_get_size(v_snd_43_);
v___x_52_ = lean_nat_dec_eq(v___x_51_, v___x_38_);
if (v___x_52_ == 0)
{
lean_dec(v_snd_43_);
goto v___jp_12_;
}
else
{
lean_object* v___x_53_; lean_object* v___x_54_; 
lean_inc(v_toPure_16_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
v___x_53_ = lean_unsigned_to_nat(0u);
v___x_54_ = lean_array_fget(v_snd_43_, v___x_53_);
lean_dec(v_snd_43_);
v_00_u03b1_18_ = v___x_54_;
goto v___jp_17_;
}
}
}
else
{
lean_object* v___x_55_; uint8_t v___x_56_; 
lean_dec_ref(v_str_44_);
v___x_55_ = lean_array_get_size(v_snd_43_);
v___x_56_ = lean_nat_dec_eq(v___x_55_, v___x_38_);
if (v___x_56_ == 0)
{
lean_dec(v_snd_43_);
goto v___jp_12_;
}
else
{
lean_object* v___x_57_; lean_object* v___x_58_; 
lean_inc(v_toPure_16_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
v___x_57_ = lean_unsigned_to_nat(0u);
v___x_58_ = lean_array_fget(v_snd_43_, v___x_57_);
lean_dec(v_snd_43_);
v_00_u03b1_18_ = v___x_58_;
goto v___jp_17_;
}
}
}
else
{
lean_object* v___x_59_; uint8_t v___x_60_; 
lean_dec_ref(v_str_44_);
v___x_59_ = lean_array_get_size(v_snd_43_);
v___x_60_ = lean_nat_dec_eq(v___x_59_, v___x_38_);
if (v___x_60_ == 0)
{
lean_dec(v_snd_43_);
goto v___jp_12_;
}
else
{
lean_object* v___x_61_; lean_object* v___x_62_; 
lean_inc(v_toPure_16_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
v___x_61_ = lean_unsigned_to_nat(0u);
v___x_62_ = lean_array_fget(v_snd_43_, v___x_61_);
lean_dec(v_snd_43_);
v_00_u03b1_18_ = v___x_62_;
goto v___jp_17_;
}
}
}
else
{
lean_dec_ref_known(v_fst_41_, 2);
lean_dec_ref(v___x_40_);
goto v___jp_12_;
}
}
else
{
lean_dec(v_fst_41_);
lean_dec_ref(v___x_40_);
goto v___jp_12_;
}
}
}
}
}
else
{
lean_inc(v_toPure_16_);
lean_dec_ref_known(v_pre_26_, 2);
lean_dec_ref_known(v_fst_25_, 2);
lean_dec_ref(v___x_24_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
goto v___jp_21_;
}
}
else
{
lean_inc(v_toPure_16_);
lean_dec(v_pre_26_);
lean_dec_ref_known(v_fst_25_, 2);
lean_dec_ref(v___x_24_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
goto v___jp_21_;
}
}
else
{
lean_inc(v_toPure_16_);
lean_dec(v_fst_25_);
lean_dec_ref(v___x_24_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
goto v___jp_21_;
}
v___jp_12_:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1, &lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1);
v___x_14_ = l_Lean_throwError___redArg(v_inst_9_, v_inst_10_, v___x_13_);
return v___x_14_;
}
v___jp_17_:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_19_, 0, v_00_u03b1_18_);
v___x_20_ = lean_apply_2(v_toPure_16_, lean_box(0), v___x_19_);
return v___x_20_;
}
v___jp_21_:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = lean_box(0);
v___x_23_ = lean_apply_2(v_toPure_16_, lean_box(0), v___x_22_);
return v___x_23_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType(lean_object* v_m_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_e_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_Lean_Elab_Tactic_getMemType___redArg(v_inst_64_, v_inst_65_, v_e_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0_spec__0(lean_object* v_msgData_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
lean_object* v___x_74_; lean_object* v_env_75_; lean_object* v___x_76_; lean_object* v_mctx_77_; lean_object* v_lctx_78_; lean_object* v_options_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_74_ = lean_st_ref_get(v___y_72_);
v_env_75_ = lean_ctor_get(v___x_74_, 0);
lean_inc_ref(v_env_75_);
lean_dec(v___x_74_);
v___x_76_ = lean_st_ref_get(v___y_70_);
v_mctx_77_ = lean_ctor_get(v___x_76_, 0);
lean_inc_ref(v_mctx_77_);
lean_dec(v___x_76_);
v_lctx_78_ = lean_ctor_get(v___y_69_, 2);
v_options_79_ = lean_ctor_get(v___y_71_, 2);
lean_inc_ref(v_options_79_);
lean_inc_ref(v_lctx_78_);
v___x_80_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_80_, 0, v_env_75_);
lean_ctor_set(v___x_80_, 1, v_mctx_77_);
lean_ctor_set(v___x_80_, 2, v_lctx_78_);
lean_ctor_set(v___x_80_, 3, v_options_79_);
v___x_81_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v_msgData_68_);
v___x_82_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0_spec__0___boxed(lean_object* v_msgData_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0_spec__0(v_msgData_83_, v___y_84_, v___y_85_, v___y_86_, v___y_87_);
lean_dec(v___y_87_);
lean_dec_ref(v___y_86_);
lean_dec(v___y_85_);
lean_dec_ref(v___y_84_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg(lean_object* v_msg_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_){
_start:
{
lean_object* v_ref_96_; lean_object* v___x_97_; lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_106_; 
v_ref_96_ = lean_ctor_get(v___y_93_, 5);
v___x_97_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0_spec__0(v_msg_90_, v___y_91_, v___y_92_, v___y_93_, v___y_94_);
v_a_98_ = lean_ctor_get(v___x_97_, 0);
v_isSharedCheck_106_ = !lean_is_exclusive(v___x_97_);
if (v_isSharedCheck_106_ == 0)
{
v___x_100_ = v___x_97_;
v_isShared_101_ = v_isSharedCheck_106_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_dec(v___x_97_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_106_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_102_; lean_object* v___x_104_; 
lean_inc(v_ref_96_);
v___x_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_102_, 0, v_ref_96_);
lean_ctor_set(v___x_102_, 1, v_a_98_);
if (v_isShared_101_ == 0)
{
lean_ctor_set_tag(v___x_100_, 1);
lean_ctor_set(v___x_100_, 0, v___x_102_);
v___x_104_ = v___x_100_;
goto v_reusejp_103_;
}
else
{
lean_object* v_reuseFailAlloc_105_; 
v_reuseFailAlloc_105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_105_, 0, v___x_102_);
v___x_104_ = v_reuseFailAlloc_105_;
goto v_reusejp_103_;
}
v_reusejp_103_:
{
return v___x_104_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg___boxed(lean_object* v_msg_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg(v_msg_107_, v___y_108_, v___y_109_, v___y_110_, v___y_111_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
return v_res_113_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__2(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__1));
v___x_118_ = l_Lean_stringToMessageData(v___x_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_unfoldCases(lean_object* v_g_119_, lean_object* v_h_120_, lean_object* v_userNamePre_121_, lean_object* v_counter_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v___y_129_; uint8_t v___y_130_; lean_object* v_a_135_; lean_object* v___x_138_; lean_object* v___x_139_; uint8_t v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_138_ = lean_unsigned_to_nat(0u);
v___x_139_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__0));
v___x_140_ = 0;
v___x_141_ = lean_box(0);
v___x_142_ = l_Lean_MVarId_cases(v_g_119_, v_h_120_, v___x_139_, v___x_140_, v___x_141_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
if (lean_obj_tag(v___x_142_) == 0)
{
lean_object* v_a_143_; lean_object* v___x_144_; lean_object* v___x_145_; uint8_t v___x_146_; 
v_a_143_ = lean_ctor_get(v___x_142_, 0);
lean_inc(v_a_143_);
lean_dec_ref_known(v___x_142_, 1);
v___x_144_ = lean_array_get_size(v_a_143_);
v___x_145_ = lean_unsigned_to_nat(2u);
v___x_146_ = lean_nat_dec_eq(v___x_144_, v___x_145_);
if (v___x_146_ == 0)
{
lean_object* v___x_147_; lean_object* v___x_148_; 
lean_dec(v_a_143_);
lean_dec(v_counter_122_);
lean_dec(v_userNamePre_121_);
v___x_147_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__2, &lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__2_once, _init_lp_mathlib_Lean_Elab_Tactic_unfoldCases___closed__2);
v___x_148_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg(v___x_147_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
if (lean_obj_tag(v___x_148_) == 0)
{
return v___x_148_;
}
else
{
lean_object* v_a_149_; 
v_a_149_ = lean_ctor_get(v___x_148_, 0);
lean_inc(v_a_149_);
lean_dec_ref_known(v___x_148_, 1);
v_a_135_ = v_a_149_;
goto v___jp_134_;
}
}
else
{
lean_object* v___x_150_; lean_object* v_toInductionSubgoal_151_; lean_object* v_mvarId_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_150_ = lean_array_fget_borrowed(v_a_143_, v___x_138_);
v_toInductionSubgoal_151_ = lean_ctor_get(v___x_150_, 0);
v_mvarId_152_ = lean_ctor_get(v_toInductionSubgoal_151_, 0);
lean_inc_n(v_mvarId_152_, 2);
lean_inc(v_counter_122_);
v___x_153_ = l_Nat_reprFast(v_counter_122_);
lean_inc(v_userNamePre_121_);
v___x_154_ = l_Lean_Name_str___override(v_userNamePre_121_, v___x_153_);
v___x_155_ = l_Lean_MVarId_setUserName___redArg(v_mvarId_152_, v___x_154_, v_a_124_);
if (lean_obj_tag(v___x_155_) == 0)
{
lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v_toInductionSubgoal_158_; lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_181_; 
lean_dec_ref_known(v___x_155_, 1);
v___x_156_ = lean_unsigned_to_nat(1u);
v___x_157_ = lean_array_fget(v_a_143_, v___x_156_);
lean_dec(v_a_143_);
v_toInductionSubgoal_158_ = lean_ctor_get(v___x_157_, 0);
v_isSharedCheck_181_ = !lean_is_exclusive(v___x_157_);
if (v_isSharedCheck_181_ == 0)
{
lean_object* v_unused_182_; 
v_unused_182_ = lean_ctor_get(v___x_157_, 1);
lean_dec(v_unused_182_);
v___x_160_ = v___x_157_;
v_isShared_161_ = v_isSharedCheck_181_;
goto v_resetjp_159_;
}
else
{
lean_inc(v_toInductionSubgoal_158_);
lean_dec(v___x_157_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_181_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
lean_object* v_mvarId_162_; lean_object* v_fields_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v_mvarId_162_ = lean_ctor_get(v_toInductionSubgoal_158_, 0);
lean_inc(v_mvarId_162_);
v_fields_163_ = lean_ctor_get(v_toInductionSubgoal_158_, 1);
lean_inc_ref(v_fields_163_);
lean_dec_ref(v_toInductionSubgoal_158_);
v___x_164_ = l_Lean_instInhabitedExpr;
v___x_165_ = lean_array_get(v___x_164_, v_fields_163_, v___x_145_);
lean_dec_ref(v_fields_163_);
v___x_166_ = l_Lean_Expr_fvarId_x21(v___x_165_);
lean_dec(v___x_165_);
v___x_167_ = lean_nat_add(v_counter_122_, v___x_156_);
lean_dec(v_counter_122_);
v___x_168_ = lp_mathlib_Lean_Elab_Tactic_unfoldCases(v_mvarId_162_, v___x_166_, v_userNamePre_121_, v___x_167_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
if (lean_obj_tag(v___x_168_) == 0)
{
lean_object* v_a_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_179_; 
v_a_169_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_179_ == 0)
{
v___x_171_ = v___x_168_;
v_isShared_172_ = v_isSharedCheck_179_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_a_169_);
lean_dec(v___x_168_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_179_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_174_; 
if (v_isShared_161_ == 0)
{
lean_ctor_set_tag(v___x_160_, 1);
lean_ctor_set(v___x_160_, 1, v_a_169_);
lean_ctor_set(v___x_160_, 0, v_mvarId_152_);
v___x_174_ = v___x_160_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v_mvarId_152_);
lean_ctor_set(v_reuseFailAlloc_178_, 1, v_a_169_);
v___x_174_ = v_reuseFailAlloc_178_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
lean_object* v___x_176_; 
if (v_isShared_172_ == 0)
{
lean_ctor_set(v___x_171_, 0, v___x_174_);
v___x_176_ = v___x_171_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v___x_174_);
v___x_176_ = v_reuseFailAlloc_177_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
return v___x_176_;
}
}
}
}
else
{
lean_object* v_a_180_; 
lean_del_object(v___x_160_);
lean_dec(v_mvarId_152_);
v_a_180_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_a_180_);
lean_dec_ref_known(v___x_168_, 1);
v_a_135_ = v_a_180_;
goto v___jp_134_;
}
}
}
else
{
lean_object* v_a_183_; 
lean_dec(v_mvarId_152_);
lean_dec(v_a_143_);
lean_dec(v_counter_122_);
lean_dec(v_userNamePre_121_);
v_a_183_ = lean_ctor_get(v___x_155_, 0);
lean_inc(v_a_183_);
lean_dec_ref_known(v___x_155_, 1);
v_a_135_ = v_a_183_;
goto v___jp_134_;
}
}
}
else
{
lean_object* v_a_184_; lean_object* v___x_186_; uint8_t v_isShared_187_; uint8_t v_isSharedCheck_191_; 
lean_dec(v_counter_122_);
lean_dec(v_userNamePre_121_);
v_a_184_ = lean_ctor_get(v___x_142_, 0);
v_isSharedCheck_191_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_191_ == 0)
{
v___x_186_ = v___x_142_;
v_isShared_187_ = v_isSharedCheck_191_;
goto v_resetjp_185_;
}
else
{
lean_inc(v_a_184_);
lean_dec(v___x_142_);
v___x_186_ = lean_box(0);
v_isShared_187_ = v_isSharedCheck_191_;
goto v_resetjp_185_;
}
v_resetjp_185_:
{
lean_object* v___x_189_; 
if (v_isShared_187_ == 0)
{
v___x_189_ = v___x_186_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_190_; 
v_reuseFailAlloc_190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_190_, 0, v_a_184_);
v___x_189_ = v_reuseFailAlloc_190_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
return v___x_189_;
}
}
}
v___jp_128_:
{
if (v___y_130_ == 0)
{
lean_object* v___x_131_; lean_object* v___x_132_; 
lean_dec_ref(v___y_129_);
v___x_131_ = lean_box(0);
v___x_132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
return v___x_132_;
}
else
{
lean_object* v___x_133_; 
v___x_133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_133_, 0, v___y_129_);
return v___x_133_;
}
}
v___jp_134_:
{
uint8_t v___x_136_; 
v___x_136_ = l_Lean_Exception_isInterrupt(v_a_135_);
if (v___x_136_ == 0)
{
uint8_t v___x_137_; 
lean_inc_ref(v_a_135_);
v___x_137_ = l_Lean_Exception_isRuntime(v_a_135_);
v___y_129_ = v_a_135_;
v___y_130_ = v___x_137_;
goto v___jp_128_;
}
else
{
v___y_129_ = v_a_135_;
v___y_130_ = v___x_136_;
goto v___jp_128_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_unfoldCases___boxed(lean_object* v_g_192_, lean_object* v_h_193_, lean_object* v_userNamePre_194_, lean_object* v_counter_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_, lean_object* v_a_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Lean_Elab_Tactic_unfoldCases(v_g_192_, v_h_193_, v_userNamePre_194_, v_counter_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_);
lean_dec(v_a_199_);
lean_dec_ref(v_a_198_);
lean_dec(v_a_197_);
lean_dec_ref(v_a_196_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0(lean_object* v_00_u03b1_202_, lean_object* v_msg_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg(v_msg_203_, v___y_204_, v___y_205_, v___y_206_, v___y_207_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___boxed(lean_object* v_00_u03b1_210_, lean_object* v_msg_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0(v_00_u03b1_210_, v_msg_211_, v___y_212_, v___y_213_, v___y_214_, v___y_215_);
lean_dec(v___y_215_);
lean_dec_ref(v___y_214_);
lean_dec(v___y_213_);
lean_dec_ref(v___y_212_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___redArg(lean_object* v_e_218_, lean_object* v___y_219_){
_start:
{
uint8_t v___x_221_; 
v___x_221_ = l_Lean_Expr_hasMVar(v_e_218_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; 
v___x_222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_222_, 0, v_e_218_);
return v___x_222_;
}
else
{
lean_object* v___x_223_; lean_object* v_mctx_224_; lean_object* v___x_225_; lean_object* v_fst_226_; lean_object* v_snd_227_; lean_object* v___x_228_; lean_object* v_cache_229_; lean_object* v_zetaDeltaFVarIds_230_; lean_object* v_postponed_231_; lean_object* v_diag_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_241_; 
v___x_223_ = lean_st_ref_get(v___y_219_);
v_mctx_224_ = lean_ctor_get(v___x_223_, 0);
lean_inc_ref(v_mctx_224_);
lean_dec(v___x_223_);
v___x_225_ = l_Lean_instantiateMVarsCore(v_mctx_224_, v_e_218_);
v_fst_226_ = lean_ctor_get(v___x_225_, 0);
lean_inc(v_fst_226_);
v_snd_227_ = lean_ctor_get(v___x_225_, 1);
lean_inc(v_snd_227_);
lean_dec_ref(v___x_225_);
v___x_228_ = lean_st_ref_take(v___y_219_);
v_cache_229_ = lean_ctor_get(v___x_228_, 1);
v_zetaDeltaFVarIds_230_ = lean_ctor_get(v___x_228_, 2);
v_postponed_231_ = lean_ctor_get(v___x_228_, 3);
v_diag_232_ = lean_ctor_get(v___x_228_, 4);
v_isSharedCheck_241_ = !lean_is_exclusive(v___x_228_);
if (v_isSharedCheck_241_ == 0)
{
lean_object* v_unused_242_; 
v_unused_242_ = lean_ctor_get(v___x_228_, 0);
lean_dec(v_unused_242_);
v___x_234_ = v___x_228_;
v_isShared_235_ = v_isSharedCheck_241_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_diag_232_);
lean_inc(v_postponed_231_);
lean_inc(v_zetaDeltaFVarIds_230_);
lean_inc(v_cache_229_);
lean_dec(v___x_228_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_241_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_237_; 
if (v_isShared_235_ == 0)
{
lean_ctor_set(v___x_234_, 0, v_snd_227_);
v___x_237_ = v___x_234_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_240_; 
v_reuseFailAlloc_240_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_240_, 0, v_snd_227_);
lean_ctor_set(v_reuseFailAlloc_240_, 1, v_cache_229_);
lean_ctor_set(v_reuseFailAlloc_240_, 2, v_zetaDeltaFVarIds_230_);
lean_ctor_set(v_reuseFailAlloc_240_, 3, v_postponed_231_);
lean_ctor_set(v_reuseFailAlloc_240_, 4, v_diag_232_);
v___x_237_ = v_reuseFailAlloc_240_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lean_st_ref_set(v___y_219_, v___x_237_);
v___x_239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_239_, 0, v_fst_226_);
return v___x_239_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___redArg___boxed(lean_object* v_e_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___redArg(v_e_243_, v___y_244_);
lean_dec(v___y_244_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1(lean_object* v_e_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___redArg(v_e_247_, v___y_249_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___boxed(lean_object* v_e_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1(v_e_254_, v___y_255_, v___y_256_, v___y_257_, v___y_258_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
lean_dec(v___y_256_);
lean_dec_ref(v___y_255_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___redArg(lean_object* v_mvarId_261_, lean_object* v_x_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_261_, v_x_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_);
if (lean_obj_tag(v___x_268_) == 0)
{
lean_object* v_a_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_276_; 
v_a_269_ = lean_ctor_get(v___x_268_, 0);
v_isSharedCheck_276_ = !lean_is_exclusive(v___x_268_);
if (v_isSharedCheck_276_ == 0)
{
v___x_271_ = v___x_268_;
v_isShared_272_ = v_isSharedCheck_276_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_a_269_);
lean_dec(v___x_268_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_276_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
lean_object* v___x_274_; 
if (v_isShared_272_ == 0)
{
v___x_274_ = v___x_271_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v_a_269_);
v___x_274_ = v_reuseFailAlloc_275_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
return v___x_274_;
}
}
}
else
{
lean_object* v_a_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_284_; 
v_a_277_ = lean_ctor_get(v___x_268_, 0);
v_isSharedCheck_284_ = !lean_is_exclusive(v___x_268_);
if (v_isSharedCheck_284_ == 0)
{
v___x_279_ = v___x_268_;
v_isShared_280_ = v_isSharedCheck_284_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_a_277_);
lean_dec(v___x_268_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_284_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
lean_object* v___x_282_; 
if (v_isShared_280_ == 0)
{
v___x_282_ = v___x_279_;
goto v_reusejp_281_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v_a_277_);
v___x_282_ = v_reuseFailAlloc_283_;
goto v_reusejp_281_;
}
v_reusejp_281_:
{
return v___x_282_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___redArg___boxed(lean_object* v_mvarId_285_, lean_object* v_x_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___redArg(v_mvarId_285_, v_x_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
return v_res_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2(lean_object* v_00_u03b1_293_, lean_object* v_mvarId_294_, lean_object* v_x_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___redArg(v_mvarId_294_, v_x_295_, v___y_296_, v___y_297_, v___y_298_, v___y_299_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___boxed(lean_object* v_00_u03b1_302_, lean_object* v_mvarId_303_, lean_object* v_x_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2(v_00_u03b1_302_, v_mvarId_303_, v_x_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
lean_dec(v___y_306_);
lean_dec_ref(v___y_305_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___at___00Lean_Elab_Tactic_finCasesAt_spec__0(lean_object* v_e_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_){
_start:
{
lean_object* v_00_u03b1_324_; lean_object* v___x_327_; lean_object* v_fst_328_; 
v___x_327_ = l_Lean_Expr_getAppFnArgs(v_e_311_);
v_fst_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_fst_328_);
if (lean_obj_tag(v_fst_328_) == 1)
{
lean_object* v_pre_329_; 
v_pre_329_ = lean_ctor_get(v_fst_328_, 0);
lean_inc(v_pre_329_);
if (lean_obj_tag(v_pre_329_) == 1)
{
lean_object* v_pre_330_; 
v_pre_330_ = lean_ctor_get(v_pre_329_, 0);
if (lean_obj_tag(v_pre_330_) == 0)
{
lean_object* v_snd_331_; lean_object* v_str_332_; lean_object* v_str_333_; lean_object* v___x_334_; uint8_t v___x_335_; 
v_snd_331_ = lean_ctor_get(v___x_327_, 1);
lean_inc(v_snd_331_);
lean_dec_ref(v___x_327_);
v_str_332_ = lean_ctor_get(v_fst_328_, 1);
lean_inc_ref(v_str_332_);
lean_dec_ref_known(v_fst_328_, 2);
v_str_333_ = lean_ctor_get(v_pre_329_, 1);
lean_inc_ref(v_str_333_);
lean_dec_ref_known(v_pre_329_, 2);
v___x_334_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__2));
v___x_335_ = lean_string_dec_eq(v_str_333_, v___x_334_);
lean_dec_ref(v_str_333_);
if (v___x_335_ == 0)
{
lean_dec_ref(v_str_332_);
lean_dec(v_snd_331_);
goto v___jp_317_;
}
else
{
lean_object* v___x_336_; uint8_t v___x_337_; 
v___x_336_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__3));
v___x_337_ = lean_string_dec_eq(v_str_332_, v___x_336_);
lean_dec_ref(v_str_332_);
if (v___x_337_ == 0)
{
lean_dec(v_snd_331_);
goto v___jp_317_;
}
else
{
lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_338_ = lean_array_get_size(v_snd_331_);
v___x_339_ = lean_unsigned_to_nat(5u);
v___x_340_ = lean_nat_dec_eq(v___x_338_, v___x_339_);
if (v___x_340_ == 0)
{
lean_dec(v_snd_331_);
goto v___jp_317_;
}
else
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v_fst_344_; 
v___x_341_ = lean_unsigned_to_nat(1u);
v___x_342_ = lean_array_fget(v_snd_331_, v___x_341_);
lean_dec(v_snd_331_);
v___x_343_ = l_Lean_Expr_getAppFnArgs(v___x_342_);
v_fst_344_ = lean_ctor_get(v___x_343_, 0);
lean_inc(v_fst_344_);
if (lean_obj_tag(v_fst_344_) == 1)
{
lean_object* v_pre_345_; 
v_pre_345_ = lean_ctor_get(v_fst_344_, 0);
if (lean_obj_tag(v_pre_345_) == 0)
{
lean_object* v_snd_346_; lean_object* v_str_347_; lean_object* v___x_348_; uint8_t v___x_349_; 
v_snd_346_ = lean_ctor_get(v___x_343_, 1);
lean_inc(v_snd_346_);
lean_dec_ref(v___x_343_);
v_str_347_ = lean_ctor_get(v_fst_344_, 1);
lean_inc_ref(v_str_347_);
lean_dec_ref_known(v_fst_344_, 2);
v___x_348_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__4));
v___x_349_ = lean_string_dec_eq(v_str_347_, v___x_348_);
if (v___x_349_ == 0)
{
lean_object* v___x_350_; uint8_t v___x_351_; 
v___x_350_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__5));
v___x_351_ = lean_string_dec_eq(v_str_347_, v___x_350_);
if (v___x_351_ == 0)
{
lean_object* v___x_352_; uint8_t v___x_353_; 
v___x_352_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__6));
v___x_353_ = lean_string_dec_eq(v_str_347_, v___x_352_);
lean_dec_ref(v_str_347_);
if (v___x_353_ == 0)
{
lean_dec(v_snd_346_);
goto v___jp_320_;
}
else
{
lean_object* v___x_354_; uint8_t v___x_355_; 
v___x_354_ = lean_array_get_size(v_snd_346_);
v___x_355_ = lean_nat_dec_eq(v___x_354_, v___x_341_);
if (v___x_355_ == 0)
{
lean_dec(v_snd_346_);
goto v___jp_320_;
}
else
{
lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_356_ = lean_unsigned_to_nat(0u);
v___x_357_ = lean_array_fget(v_snd_346_, v___x_356_);
lean_dec(v_snd_346_);
v_00_u03b1_324_ = v___x_357_;
goto v___jp_323_;
}
}
}
else
{
lean_object* v___x_358_; uint8_t v___x_359_; 
lean_dec_ref(v_str_347_);
v___x_358_ = lean_array_get_size(v_snd_346_);
v___x_359_ = lean_nat_dec_eq(v___x_358_, v___x_341_);
if (v___x_359_ == 0)
{
lean_dec(v_snd_346_);
goto v___jp_320_;
}
else
{
lean_object* v___x_360_; lean_object* v___x_361_; 
v___x_360_ = lean_unsigned_to_nat(0u);
v___x_361_ = lean_array_fget(v_snd_346_, v___x_360_);
lean_dec(v_snd_346_);
v_00_u03b1_324_ = v___x_361_;
goto v___jp_323_;
}
}
}
else
{
lean_object* v___x_362_; uint8_t v___x_363_; 
lean_dec_ref(v_str_347_);
v___x_362_ = lean_array_get_size(v_snd_346_);
v___x_363_ = lean_nat_dec_eq(v___x_362_, v___x_341_);
if (v___x_363_ == 0)
{
lean_dec(v_snd_346_);
goto v___jp_320_;
}
else
{
lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_364_ = lean_unsigned_to_nat(0u);
v___x_365_ = lean_array_fget(v_snd_346_, v___x_364_);
lean_dec(v_snd_346_);
v_00_u03b1_324_ = v___x_365_;
goto v___jp_323_;
}
}
}
else
{
lean_dec_ref_known(v_fst_344_, 2);
lean_dec_ref(v___x_343_);
goto v___jp_320_;
}
}
else
{
lean_dec(v_fst_344_);
lean_dec_ref(v___x_343_);
goto v___jp_320_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_329_, 2);
lean_dec_ref_known(v_fst_328_, 2);
lean_dec_ref(v___x_327_);
goto v___jp_317_;
}
}
else
{
lean_dec(v_pre_329_);
lean_dec_ref_known(v_fst_328_, 2);
lean_dec_ref(v___x_327_);
goto v___jp_317_;
}
}
else
{
lean_dec(v_fst_328_);
lean_dec_ref(v___x_327_);
goto v___jp_317_;
}
v___jp_317_:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = lean_box(0);
v___x_319_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_319_, 0, v___x_318_);
return v___x_319_;
}
v___jp_320_:
{
lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_321_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1, &lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_Tactic_getMemType___redArg___closed__1);
v___x_322_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_unfoldCases_spec__0___redArg(v___x_321_, v___y_312_, v___y_313_, v___y_314_, v___y_315_);
return v___x_322_;
}
v___jp_323_:
{
lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_325_, 0, v_00_u03b1_324_);
v___x_326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_326_, 0, v___x_325_);
return v___x_326_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMemType___at___00Lean_Elab_Tactic_finCasesAt_spec__0___boxed(lean_object* v_e_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_Lean_Elab_Tactic_getMemType___at___00Lean_Elab_Tactic_finCasesAt_spec__0(v_e_366_, v___y_367_, v___y_368_, v___y_369_, v___y_370_);
lean_dec(v___y_370_);
lean_dec_ref(v___y_369_);
lean_dec(v___y_368_);
lean_dec_ref(v___y_367_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0(lean_object* v_hyp_390_, lean_object* v_g_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_){
_start:
{
lean_object* v___y_398_; lean_object* v___x_528_; 
lean_inc(v_hyp_390_);
v___x_528_ = l_Lean_FVarId_getType___redArg(v_hyp_390_, v___y_392_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_528_) == 0)
{
lean_object* v_a_529_; lean_object* v___x_530_; 
v_a_529_ = lean_ctor_get(v___x_528_, 0);
lean_inc(v_a_529_);
lean_dec_ref_known(v___x_528_, 1);
v___x_530_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_finCasesAt_spec__1___redArg(v_a_529_, v___y_393_);
v___y_398_ = v___x_530_;
goto v___jp_397_;
}
else
{
v___y_398_ = v___x_528_;
goto v___jp_397_;
}
v___jp_397_:
{
if (lean_obj_tag(v___y_398_) == 0)
{
lean_object* v_a_399_; lean_object* v___x_400_; 
v_a_399_ = lean_ctor_get(v___y_398_, 0);
lean_inc_n(v_a_399_, 2);
lean_dec_ref_known(v___y_398_, 1);
v___x_400_ = lp_mathlib_Lean_Elab_Tactic_getMemType___at___00Lean_Elab_Tactic_finCasesAt_spec__0(v_a_399_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_400_) == 0)
{
lean_object* v_a_401_; 
v_a_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_a_401_);
lean_dec_ref_known(v___x_400_, 1);
if (lean_obj_tag(v_a_401_) == 0)
{
lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_402_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__1));
v___x_403_ = lean_unsigned_to_nat(1u);
v___x_404_ = lean_mk_empty_array_with_capacity(v___x_403_);
lean_inc(v_a_399_);
v___x_405_ = lean_array_push(v___x_404_, v_a_399_);
v___x_406_ = l_Lean_Meta_mkAppM(v___x_402_, v___x_405_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_406_) == 0)
{
lean_object* v_a_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
v_a_407_ = lean_ctor_get(v___x_406_, 0);
lean_inc(v_a_407_);
lean_dec_ref_known(v___x_406_, 1);
v___x_408_ = lean_box(0);
v___x_409_ = l_Lean_Meta_synthInstance(v_a_407_, v___x_408_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_409_) == 0)
{
lean_object* v_a_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v_a_410_ = lean_ctor_get(v___x_409_, 0);
lean_inc(v_a_410_);
lean_dec_ref_known(v___x_409_, 1);
v___x_411_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__3));
v___x_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_412_, 0, v_a_399_);
v___x_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_413_, 0, v_a_410_);
v___x_414_ = lean_unsigned_to_nat(2u);
v___x_415_ = lean_mk_empty_array_with_capacity(v___x_414_);
lean_inc_ref(v___x_412_);
lean_inc_ref(v___x_415_);
v___x_416_ = lean_array_push(v___x_415_, v___x_412_);
lean_inc_ref(v___x_413_);
v___x_417_ = lean_array_push(v___x_416_, v___x_413_);
v___x_418_ = l_Lean_Meta_mkAppOptM(v___x_411_, v___x_417_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_object* v_a_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v_a_419_ = lean_ctor_get(v___x_418_, 0);
lean_inc(v_a_419_);
lean_dec_ref_known(v___x_418_, 1);
v___x_420_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__4));
v___x_421_ = l_Lean_Expr_fvar___override(v_hyp_390_);
v___x_422_ = lean_array_push(v___x_415_, v_a_419_);
lean_inc_ref(v___x_421_);
v___x_423_ = lean_array_push(v___x_422_, v___x_421_);
v___x_424_ = l_Lean_Meta_mkAppM(v___x_420_, v___x_423_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_424_) == 0)
{
lean_object* v_a_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v_a_425_ = lean_ctor_get(v___x_424_, 0);
lean_inc(v_a_425_);
lean_dec_ref_known(v___x_424_, 1);
v___x_426_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__6));
v___x_427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_427_, 0, v___x_421_);
v___x_428_ = lean_unsigned_to_nat(3u);
v___x_429_ = lean_mk_empty_array_with_capacity(v___x_428_);
v___x_430_ = lean_array_push(v___x_429_, v___x_412_);
v___x_431_ = lean_array_push(v___x_430_, v___x_413_);
v___x_432_ = lean_array_push(v___x_431_, v___x_427_);
v___x_433_ = l_Lean_Meta_mkAppOptM(v___x_426_, v___x_432_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_433_) == 0)
{
lean_object* v_a_434_; lean_object* v___x_435_; lean_object* v___x_436_; 
v_a_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_434_);
lean_dec_ref_known(v___x_433_, 1);
v___x_435_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___closed__8));
v___x_436_ = l_Lean_MVarId_assert(v_g_391_, v___x_435_, v_a_425_, v_a_434_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_436_) == 0)
{
lean_object* v_a_437_; uint8_t v___x_438_; lean_object* v___x_439_; 
v_a_437_ = lean_ctor_get(v___x_436_, 0);
lean_inc(v_a_437_);
lean_dec_ref_known(v___x_436_, 1);
v___x_438_ = 1;
v___x_439_ = l_Lean_Meta_intro1Core(v_a_437_, v___x_438_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_439_) == 0)
{
lean_object* v_a_440_; lean_object* v_fst_441_; lean_object* v_snd_442_; lean_object* v___x_443_; 
v_a_440_ = lean_ctor_get(v___x_439_, 0);
lean_inc(v_a_440_);
lean_dec_ref_known(v___x_439_, 1);
v_fst_441_ = lean_ctor_get(v_a_440_, 0);
lean_inc(v_fst_441_);
v_snd_442_ = lean_ctor_get(v_a_440_, 1);
lean_inc(v_snd_442_);
lean_dec(v_a_440_);
v___x_443_ = lp_mathlib_Lean_Elab_Tactic_finCasesAt(v_snd_442_, v_fst_441_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
return v___x_443_;
}
else
{
lean_object* v_a_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_451_; 
v_a_444_ = lean_ctor_get(v___x_439_, 0);
v_isSharedCheck_451_ = !lean_is_exclusive(v___x_439_);
if (v_isSharedCheck_451_ == 0)
{
v___x_446_ = v___x_439_;
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_a_444_);
lean_dec(v___x_439_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_449_; 
if (v_isShared_447_ == 0)
{
v___x_449_ = v___x_446_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v_a_444_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
}
else
{
lean_object* v_a_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_459_; 
v_a_452_ = lean_ctor_get(v___x_436_, 0);
v_isSharedCheck_459_ = !lean_is_exclusive(v___x_436_);
if (v_isSharedCheck_459_ == 0)
{
v___x_454_ = v___x_436_;
v_isShared_455_ = v_isSharedCheck_459_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_a_452_);
lean_dec(v___x_436_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_459_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v___x_457_; 
if (v_isShared_455_ == 0)
{
v___x_457_ = v___x_454_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v_a_452_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
else
{
lean_object* v_a_460_; lean_object* v___x_462_; uint8_t v_isShared_463_; uint8_t v_isSharedCheck_467_; 
lean_dec(v_a_425_);
lean_dec(v_g_391_);
v_a_460_ = lean_ctor_get(v___x_433_, 0);
v_isSharedCheck_467_ = !lean_is_exclusive(v___x_433_);
if (v_isSharedCheck_467_ == 0)
{
v___x_462_ = v___x_433_;
v_isShared_463_ = v_isSharedCheck_467_;
goto v_resetjp_461_;
}
else
{
lean_inc(v_a_460_);
lean_dec(v___x_433_);
v___x_462_ = lean_box(0);
v_isShared_463_ = v_isSharedCheck_467_;
goto v_resetjp_461_;
}
v_resetjp_461_:
{
lean_object* v___x_465_; 
if (v_isShared_463_ == 0)
{
v___x_465_ = v___x_462_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_466_; 
v_reuseFailAlloc_466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_466_, 0, v_a_460_);
v___x_465_ = v_reuseFailAlloc_466_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
return v___x_465_;
}
}
}
}
else
{
lean_object* v_a_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_475_; 
lean_dec_ref(v___x_421_);
lean_dec_ref_known(v___x_413_, 1);
lean_dec_ref_known(v___x_412_, 1);
lean_dec(v_g_391_);
v_a_468_ = lean_ctor_get(v___x_424_, 0);
v_isSharedCheck_475_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_475_ == 0)
{
v___x_470_ = v___x_424_;
v_isShared_471_ = v_isSharedCheck_475_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_a_468_);
lean_dec(v___x_424_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_475_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
lean_object* v___x_473_; 
if (v_isShared_471_ == 0)
{
v___x_473_ = v___x_470_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v_a_468_);
v___x_473_ = v_reuseFailAlloc_474_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
return v___x_473_;
}
}
}
}
else
{
lean_object* v_a_476_; lean_object* v___x_478_; uint8_t v_isShared_479_; uint8_t v_isSharedCheck_483_; 
lean_dec_ref(v___x_415_);
lean_dec_ref_known(v___x_413_, 1);
lean_dec_ref_known(v___x_412_, 1);
lean_dec(v_g_391_);
lean_dec(v_hyp_390_);
v_a_476_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_483_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_483_ == 0)
{
v___x_478_ = v___x_418_;
v_isShared_479_ = v_isSharedCheck_483_;
goto v_resetjp_477_;
}
else
{
lean_inc(v_a_476_);
lean_dec(v___x_418_);
v___x_478_ = lean_box(0);
v_isShared_479_ = v_isSharedCheck_483_;
goto v_resetjp_477_;
}
v_resetjp_477_:
{
lean_object* v___x_481_; 
if (v_isShared_479_ == 0)
{
v___x_481_ = v___x_478_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_482_; 
v_reuseFailAlloc_482_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_482_, 0, v_a_476_);
v___x_481_ = v_reuseFailAlloc_482_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
return v___x_481_;
}
}
}
}
else
{
lean_object* v_a_484_; lean_object* v___x_486_; uint8_t v_isShared_487_; uint8_t v_isSharedCheck_491_; 
lean_dec(v_a_399_);
lean_dec(v_g_391_);
lean_dec(v_hyp_390_);
v_a_484_ = lean_ctor_get(v___x_409_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_409_);
if (v_isSharedCheck_491_ == 0)
{
v___x_486_ = v___x_409_;
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
else
{
lean_inc(v_a_484_);
lean_dec(v___x_409_);
v___x_486_ = lean_box(0);
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
v_resetjp_485_:
{
lean_object* v___x_489_; 
if (v_isShared_487_ == 0)
{
v___x_489_ = v___x_486_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v_a_484_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
}
}
}
}
else
{
lean_object* v_a_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_499_; 
lean_dec(v_a_399_);
lean_dec(v_g_391_);
lean_dec(v_hyp_390_);
v_a_492_ = lean_ctor_get(v___x_406_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_406_);
if (v_isSharedCheck_499_ == 0)
{
v___x_494_ = v___x_406_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_a_492_);
lean_dec(v___x_406_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_497_; 
if (v_isShared_495_ == 0)
{
v___x_497_ = v___x_494_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v_a_492_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
}
}
else
{
lean_object* v___x_500_; 
lean_dec_ref_known(v_a_401_, 1);
lean_dec(v_a_399_);
lean_inc(v_g_391_);
v___x_500_ = l_Lean_MVarId_getTag(v_g_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
if (lean_obj_tag(v___x_500_) == 0)
{
lean_object* v_a_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v_a_501_ = lean_ctor_get(v___x_500_, 0);
lean_inc(v_a_501_);
lean_dec_ref_known(v___x_500_, 1);
v___x_502_ = lean_unsigned_to_nat(0u);
v___x_503_ = lp_mathlib_Lean_Elab_Tactic_unfoldCases(v_g_391_, v_hyp_390_, v_a_501_, v___x_502_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
return v___x_503_;
}
else
{
lean_object* v_a_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_511_; 
lean_dec(v_g_391_);
lean_dec(v_hyp_390_);
v_a_504_ = lean_ctor_get(v___x_500_, 0);
v_isSharedCheck_511_ = !lean_is_exclusive(v___x_500_);
if (v_isSharedCheck_511_ == 0)
{
v___x_506_ = v___x_500_;
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_a_504_);
lean_dec(v___x_500_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_509_; 
if (v_isShared_507_ == 0)
{
v___x_509_ = v___x_506_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v_a_504_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
}
}
else
{
lean_object* v_a_512_; lean_object* v___x_514_; uint8_t v_isShared_515_; uint8_t v_isSharedCheck_519_; 
lean_dec(v_a_399_);
lean_dec(v_g_391_);
lean_dec(v_hyp_390_);
v_a_512_ = lean_ctor_get(v___x_400_, 0);
v_isSharedCheck_519_ = !lean_is_exclusive(v___x_400_);
if (v_isSharedCheck_519_ == 0)
{
v___x_514_ = v___x_400_;
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
else
{
lean_inc(v_a_512_);
lean_dec(v___x_400_);
v___x_514_ = lean_box(0);
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
v_resetjp_513_:
{
lean_object* v___x_517_; 
if (v_isShared_515_ == 0)
{
v___x_517_ = v___x_514_;
goto v_reusejp_516_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v_a_512_);
v___x_517_ = v_reuseFailAlloc_518_;
goto v_reusejp_516_;
}
v_reusejp_516_:
{
return v___x_517_;
}
}
}
}
else
{
lean_object* v_a_520_; lean_object* v___x_522_; uint8_t v_isShared_523_; uint8_t v_isSharedCheck_527_; 
lean_dec(v_g_391_);
lean_dec(v_hyp_390_);
v_a_520_ = lean_ctor_get(v___y_398_, 0);
v_isSharedCheck_527_ = !lean_is_exclusive(v___y_398_);
if (v_isSharedCheck_527_ == 0)
{
v___x_522_ = v___y_398_;
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
else
{
lean_inc(v_a_520_);
lean_dec(v___y_398_);
v___x_522_ = lean_box(0);
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
v_resetjp_521_:
{
lean_object* v___x_525_; 
if (v_isShared_523_ == 0)
{
v___x_525_ = v___x_522_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_526_; 
v_reuseFailAlloc_526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_526_, 0, v_a_520_);
v___x_525_ = v_reuseFailAlloc_526_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
return v___x_525_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___boxed(lean_object* v_hyp_531_, lean_object* v_g_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0(v_hyp_531_, v_g_532_, v___y_533_, v___y_534_, v___y_535_, v___y_536_);
lean_dec(v___y_536_);
lean_dec_ref(v___y_535_);
lean_dec(v___y_534_);
lean_dec_ref(v___y_533_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt(lean_object* v_g_539_, lean_object* v_hyp_540_, lean_object* v_a_541_, lean_object* v_a_542_, lean_object* v_a_543_, lean_object* v_a_544_){
_start:
{
lean_object* v___f_546_; lean_object* v___x_547_; 
lean_inc(v_g_539_);
v___f_546_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_finCasesAt___lam__0___boxed), 7, 2);
lean_closure_set(v___f_546_, 0, v_hyp_540_);
lean_closure_set(v___f_546_, 1, v_g_539_);
v___x_547_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_finCasesAt_spec__2___redArg(v_g_539_, v___f_546_, v_a_541_, v_a_542_, v_a_543_, v_a_544_);
return v___x_547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_finCasesAt___boxed(lean_object* v_g_548_, lean_object* v_hyp_549_, lean_object* v_a_550_, lean_object* v_a_551_, lean_object* v_a_552_, lean_object* v_a_553_, lean_object* v_a_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_mathlib_Lean_Elab_Tactic_finCasesAt(v_g_548_, v_hyp_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_);
lean_dec(v_a_553_);
lean_dec_ref(v_a_552_);
lean_dec(v_a_551_);
lean_dec_ref(v_a_550_);
return v_res_555_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v___x_631_ = lean_box(0);
v___x_632_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_633_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_633_, 0, v___x_632_);
lean_ctor_set(v___x_633_, 1, v___x_631_);
return v___x_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg(){
_start:
{
lean_object* v___x_635_; lean_object* v___x_636_; 
v___x_635_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___closed__0);
v___x_636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_636_, 0, v___x_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg___boxed(lean_object* v___y_637_){
_start:
{
lean_object* v_res_638_; 
v_res_638_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg();
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0(lean_object* v_00_u03b1_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg();
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___boxed(lean_object* v_00_u03b1_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_){
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0(v_00_u03b1_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec(v___y_656_);
lean_dec_ref(v___y_655_);
lean_dec(v___y_654_);
lean_dec_ref(v___y_653_);
lean_dec(v___y_652_);
lean_dec_ref(v___y_651_);
return v_res_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__0(lean_object* v_a_661_, lean_object* v___x_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_){
_start:
{
lean_object* v___x_672_; 
v___x_672_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_664_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
if (lean_obj_tag(v___x_672_) == 0)
{
lean_object* v_a_673_; lean_object* v___x_674_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
lean_dec_ref_known(v___x_672_, 1);
v___x_674_ = lp_mathlib_Lean_Elab_Tactic_finCasesAt(v_a_673_, v_a_661_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
if (lean_obj_tag(v___x_674_) == 0)
{
lean_object* v_a_675_; lean_object* v___x_676_; 
v_a_675_ = lean_ctor_get(v___x_674_, 0);
lean_inc(v_a_675_);
lean_dec_ref_known(v___x_674_, 1);
v___x_676_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_675_, v___y_664_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
if (lean_obj_tag(v___x_676_) == 0)
{
lean_object* v___x_678_; uint8_t v_isShared_679_; uint8_t v_isSharedCheck_683_; 
v_isSharedCheck_683_ = !lean_is_exclusive(v___x_676_);
if (v_isSharedCheck_683_ == 0)
{
lean_object* v_unused_684_; 
v_unused_684_ = lean_ctor_get(v___x_676_, 0);
lean_dec(v_unused_684_);
v___x_678_ = v___x_676_;
v_isShared_679_ = v_isSharedCheck_683_;
goto v_resetjp_677_;
}
else
{
lean_dec(v___x_676_);
v___x_678_ = lean_box(0);
v_isShared_679_ = v_isSharedCheck_683_;
goto v_resetjp_677_;
}
v_resetjp_677_:
{
lean_object* v___x_681_; 
if (v_isShared_679_ == 0)
{
lean_ctor_set(v___x_678_, 0, v___x_662_);
v___x_681_ = v___x_678_;
goto v_reusejp_680_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v___x_662_);
v___x_681_ = v_reuseFailAlloc_682_;
goto v_reusejp_680_;
}
v_reusejp_680_:
{
return v___x_681_;
}
}
}
else
{
return v___x_676_;
}
}
else
{
lean_object* v_a_685_; lean_object* v___x_687_; uint8_t v_isShared_688_; uint8_t v_isSharedCheck_692_; 
v_a_685_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_692_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_692_ == 0)
{
v___x_687_ = v___x_674_;
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
else
{
lean_inc(v_a_685_);
lean_dec(v___x_674_);
v___x_687_ = lean_box(0);
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
v_resetjp_686_:
{
lean_object* v___x_690_; 
if (v_isShared_688_ == 0)
{
v___x_690_ = v___x_687_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v_a_685_);
v___x_690_ = v_reuseFailAlloc_691_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
return v___x_690_;
}
}
}
}
else
{
lean_object* v_a_693_; lean_object* v___x_695_; uint8_t v_isShared_696_; uint8_t v_isSharedCheck_700_; 
lean_dec(v_a_661_);
v_a_693_ = lean_ctor_get(v___x_672_, 0);
v_isSharedCheck_700_ = !lean_is_exclusive(v___x_672_);
if (v_isSharedCheck_700_ == 0)
{
v___x_695_ = v___x_672_;
v_isShared_696_ = v_isSharedCheck_700_;
goto v_resetjp_694_;
}
else
{
lean_inc(v_a_693_);
lean_dec(v___x_672_);
v___x_695_ = lean_box(0);
v_isShared_696_ = v_isSharedCheck_700_;
goto v_resetjp_694_;
}
v_resetjp_694_:
{
lean_object* v___x_698_; 
if (v_isShared_696_ == 0)
{
v___x_698_ = v___x_695_;
goto v_reusejp_697_;
}
else
{
lean_object* v_reuseFailAlloc_699_; 
v_reuseFailAlloc_699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_699_, 0, v_a_693_);
v___x_698_ = v_reuseFailAlloc_699_;
goto v_reusejp_697_;
}
v_reusejp_697_:
{
return v___x_698_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__0___boxed(lean_object* v_a_701_, lean_object* v___x_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__0(v_a_701_, v___x_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
lean_dec(v___y_706_);
lean_dec_ref(v___y_705_);
lean_dec(v___y_704_);
lean_dec_ref(v___y_703_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__1(lean_object* v___f_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_){
_start:
{
lean_object* v___x_723_; 
v___x_723_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_713_, v___y_714_, v___y_715_, v___y_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__1___boxed(lean_object* v___f_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_){
_start:
{
lean_object* v_res_734_; 
v_res_734_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__1(v___f_724_, v___y_725_, v___y_726_, v___y_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_, v___y_732_);
lean_dec(v___y_732_);
lean_dec_ref(v___y_731_);
lean_dec(v___y_730_);
lean_dec_ref(v___y_729_);
lean_dec(v___y_728_);
lean_dec_ref(v___y_727_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
return v_res_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2(lean_object* v_as_735_, size_t v_sz_736_, size_t v_i_737_, lean_object* v_b_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_){
_start:
{
uint8_t v___x_748_; 
v___x_748_ = lean_usize_dec_lt(v_i_737_, v_sz_736_);
if (v___x_748_ == 0)
{
lean_object* v___x_749_; 
v___x_749_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_749_, 0, v_b_738_);
return v___x_749_;
}
else
{
lean_object* v_a_750_; lean_object* v___x_751_; 
v_a_750_ = lean_array_uget_borrowed(v_as_735_, v_i_737_);
lean_inc(v_a_750_);
v___x_751_ = l_Lean_Elab_Tactic_getFVarId(v_a_750_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
if (lean_obj_tag(v___x_751_) == 0)
{
lean_object* v_a_752_; lean_object* v___x_753_; lean_object* v___f_754_; lean_object* v___f_755_; lean_object* v___x_756_; 
v_a_752_ = lean_ctor_get(v___x_751_, 0);
lean_inc(v_a_752_);
lean_dec_ref_known(v___x_751_, 1);
v___x_753_ = lean_box(0);
v___f_754_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__0___boxed), 11, 2);
lean_closure_set(v___f_754_, 0, v_a_752_);
lean_closure_set(v___f_754_, 1, v___x_753_);
v___f_755_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___lam__1___boxed), 10, 1);
lean_closure_set(v___f_755_, 0, v___f_754_);
v___x_756_ = lp_mathlib_Lean_Elab_Tactic_allGoals(v___f_755_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
if (lean_obj_tag(v___x_756_) == 0)
{
size_t v___x_757_; size_t v___x_758_; 
lean_dec_ref_known(v___x_756_, 1);
v___x_757_ = ((size_t)1ULL);
v___x_758_ = lean_usize_add(v_i_737_, v___x_757_);
v_i_737_ = v___x_758_;
v_b_738_ = v___x_753_;
goto _start;
}
else
{
return v___x_756_;
}
}
else
{
lean_object* v_a_760_; lean_object* v___x_762_; uint8_t v_isShared_763_; uint8_t v_isSharedCheck_767_; 
v_a_760_ = lean_ctor_get(v___x_751_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_751_);
if (v_isSharedCheck_767_ == 0)
{
v___x_762_ = v___x_751_;
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_a_760_);
lean_dec(v___x_751_);
v___x_762_ = lean_box(0);
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
v_resetjp_761_:
{
lean_object* v___x_765_; 
if (v_isShared_763_ == 0)
{
v___x_765_ = v___x_762_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v_a_760_);
v___x_765_ = v_reuseFailAlloc_766_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
return v___x_765_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2___boxed(lean_object* v_as_768_, lean_object* v_sz_769_, lean_object* v_i_770_, lean_object* v_b_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
size_t v_sz_boxed_781_; size_t v_i_boxed_782_; lean_object* v_res_783_; 
v_sz_boxed_781_ = lean_unbox_usize(v_sz_769_);
lean_dec(v_sz_769_);
v_i_boxed_782_ = lean_unbox_usize(v_i_770_);
lean_dec(v_i_770_);
v_res_783_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2(v_as_768_, v_sz_boxed_781_, v_i_boxed_782_, v_b_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_);
lean_dec(v___y_779_);
lean_dec_ref(v___y_778_);
lean_dec(v___y_777_);
lean_dec_ref(v___y_776_);
lean_dec(v___y_775_);
lean_dec_ref(v___y_774_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec_ref(v_as_768_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___lam__0(lean_object* v_val_784_, size_t v_sz_785_, size_t v___x_786_, lean_object* v___x_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__2(v_val_784_, v_sz_785_, v___x_786_, v___x_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_, v___y_795_);
if (lean_obj_tag(v___x_797_) == 0)
{
lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_804_; 
v_isSharedCheck_804_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_804_ == 0)
{
lean_object* v_unused_805_; 
v_unused_805_ = lean_ctor_get(v___x_797_, 0);
lean_dec(v_unused_805_);
v___x_799_ = v___x_797_;
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
else
{
lean_dec(v___x_797_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_802_; 
if (v_isShared_800_ == 0)
{
lean_ctor_set(v___x_799_, 0, v___x_787_);
v___x_802_ = v___x_799_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v___x_787_);
v___x_802_ = v_reuseFailAlloc_803_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
return v___x_802_;
}
}
}
else
{
return v___x_797_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___lam__0___boxed(lean_object* v_val_806_, lean_object* v_sz_807_, lean_object* v___x_808_, lean_object* v___x_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_){
_start:
{
size_t v_sz_boxed_819_; size_t v___x_2814__boxed_820_; lean_object* v_res_821_; 
v_sz_boxed_819_ = lean_unbox_usize(v_sz_807_);
lean_dec(v_sz_807_);
v___x_2814__boxed_820_ = lean_unbox_usize(v___x_808_);
lean_dec(v___x_808_);
v_res_821_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___lam__0(v_val_806_, v_sz_boxed_819_, v___x_2814__boxed_820_, v___x_809_, v___y_810_, v___y_811_, v___y_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
lean_dec(v___y_817_);
lean_dec_ref(v___y_816_);
lean_dec(v___y_815_);
lean_dec_ref(v___y_814_);
lean_dec(v___y_813_);
lean_dec_ref(v___y_812_);
lean_dec(v___y_811_);
lean_dec_ref(v___y_810_);
lean_dec_ref(v_val_806_);
return v_res_821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1(size_t v_sz_825_, size_t v_i_826_, lean_object* v_bs_827_){
_start:
{
uint8_t v___x_828_; 
v___x_828_ = lean_usize_dec_lt(v_i_826_, v_sz_825_);
if (v___x_828_ == 0)
{
lean_object* v___x_829_; 
v___x_829_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_829_, 0, v_bs_827_);
return v___x_829_;
}
else
{
lean_object* v_v_830_; lean_object* v___x_831_; uint8_t v___x_832_; 
v_v_830_ = lean_array_uget(v_bs_827_, v_i_826_);
v___x_831_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___closed__1));
lean_inc(v_v_830_);
v___x_832_ = l_Lean_Syntax_isOfKind(v_v_830_, v___x_831_);
if (v___x_832_ == 0)
{
lean_object* v___x_833_; 
lean_dec(v_v_830_);
lean_dec_ref(v_bs_827_);
v___x_833_ = lean_box(0);
return v___x_833_;
}
else
{
lean_object* v___x_834_; lean_object* v_bs_x27_835_; size_t v___x_836_; size_t v___x_837_; lean_object* v___x_838_; 
v___x_834_ = lean_unsigned_to_nat(0u);
v_bs_x27_835_ = lean_array_uset(v_bs_827_, v_i_826_, v___x_834_);
v___x_836_ = ((size_t)1ULL);
v___x_837_ = lean_usize_add(v_i_826_, v___x_836_);
v___x_838_ = lean_array_uset(v_bs_x27_835_, v_i_826_, v_v_830_);
v_i_826_ = v___x_837_;
v_bs_827_ = v___x_838_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1___boxed(lean_object* v_sz_840_, lean_object* v_i_841_, lean_object* v_bs_842_){
_start:
{
size_t v_sz_boxed_843_; size_t v_i_boxed_844_; lean_object* v_res_845_; 
v_sz_boxed_843_ = lean_unbox_usize(v_sz_840_);
lean_dec(v_sz_840_);
v_i_boxed_844_ = lean_unbox_usize(v_i_841_);
lean_dec(v_i_841_);
v_res_845_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1(v_sz_boxed_843_, v_i_boxed_844_, v_bs_842_);
return v_res_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__3(uint8_t v___x_846_, lean_object* v_as_847_, size_t v_i_848_, size_t v_stop_849_, lean_object* v_b_850_){
_start:
{
lean_object* v___y_852_; uint8_t v___x_856_; 
v___x_856_ = lean_usize_dec_eq(v_i_848_, v_stop_849_);
if (v___x_856_ == 0)
{
lean_object* v_fst_857_; uint8_t v___x_858_; 
v_fst_857_ = lean_ctor_get(v_b_850_, 0);
v___x_858_ = lean_unbox(v_fst_857_);
if (v___x_858_ == 0)
{
lean_object* v_snd_859_; lean_object* v___x_861_; uint8_t v_isShared_862_; uint8_t v_isSharedCheck_867_; 
v_snd_859_ = lean_ctor_get(v_b_850_, 1);
v_isSharedCheck_867_ = !lean_is_exclusive(v_b_850_);
if (v_isSharedCheck_867_ == 0)
{
lean_object* v_unused_868_; 
v_unused_868_ = lean_ctor_get(v_b_850_, 0);
lean_dec(v_unused_868_);
v___x_861_ = v_b_850_;
v_isShared_862_ = v_isSharedCheck_867_;
goto v_resetjp_860_;
}
else
{
lean_inc(v_snd_859_);
lean_dec(v_b_850_);
v___x_861_ = lean_box(0);
v_isShared_862_ = v_isSharedCheck_867_;
goto v_resetjp_860_;
}
v_resetjp_860_:
{
lean_object* v___x_863_; lean_object* v___x_865_; 
v___x_863_ = lean_box(v___x_846_);
if (v_isShared_862_ == 0)
{
lean_ctor_set(v___x_861_, 0, v___x_863_);
v___x_865_ = v___x_861_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_866_; 
v_reuseFailAlloc_866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_866_, 0, v___x_863_);
lean_ctor_set(v_reuseFailAlloc_866_, 1, v_snd_859_);
v___x_865_ = v_reuseFailAlloc_866_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
v___y_852_ = v___x_865_;
goto v___jp_851_;
}
}
}
else
{
lean_object* v_snd_869_; lean_object* v___x_871_; uint8_t v_isShared_872_; uint8_t v_isSharedCheck_879_; 
v_snd_869_ = lean_ctor_get(v_b_850_, 1);
v_isSharedCheck_879_ = !lean_is_exclusive(v_b_850_);
if (v_isSharedCheck_879_ == 0)
{
lean_object* v_unused_880_; 
v_unused_880_ = lean_ctor_get(v_b_850_, 0);
lean_dec(v_unused_880_);
v___x_871_ = v_b_850_;
v_isShared_872_ = v_isSharedCheck_879_;
goto v_resetjp_870_;
}
else
{
lean_inc(v_snd_869_);
lean_dec(v_b_850_);
v___x_871_ = lean_box(0);
v_isShared_872_ = v_isSharedCheck_879_;
goto v_resetjp_870_;
}
v_resetjp_870_:
{
lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_877_; 
v___x_873_ = lean_array_uget_borrowed(v_as_847_, v_i_848_);
lean_inc(v___x_873_);
v___x_874_ = lean_array_push(v_snd_869_, v___x_873_);
v___x_875_ = lean_box(v___x_856_);
if (v_isShared_872_ == 0)
{
lean_ctor_set(v___x_871_, 1, v___x_874_);
lean_ctor_set(v___x_871_, 0, v___x_875_);
v___x_877_ = v___x_871_;
goto v_reusejp_876_;
}
else
{
lean_object* v_reuseFailAlloc_878_; 
v_reuseFailAlloc_878_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_878_, 0, v___x_875_);
lean_ctor_set(v_reuseFailAlloc_878_, 1, v___x_874_);
v___x_877_ = v_reuseFailAlloc_878_;
goto v_reusejp_876_;
}
v_reusejp_876_:
{
v___y_852_ = v___x_877_;
goto v___jp_851_;
}
}
}
}
else
{
return v_b_850_;
}
v___jp_851_:
{
size_t v___x_853_; size_t v___x_854_; 
v___x_853_ = ((size_t)1ULL);
v___x_854_ = lean_usize_add(v_i_848_, v___x_853_);
v_i_848_ = v___x_854_;
v_b_850_ = v___y_852_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__3___boxed(lean_object* v___x_881_, lean_object* v_as_882_, lean_object* v_i_883_, lean_object* v_stop_884_, lean_object* v_b_885_){
_start:
{
uint8_t v___x_2898__boxed_886_; size_t v_i_boxed_887_; size_t v_stop_boxed_888_; lean_object* v_res_889_; 
v___x_2898__boxed_886_ = lean_unbox(v___x_881_);
v_i_boxed_887_ = lean_unbox_usize(v_i_883_);
lean_dec(v_i_883_);
v_stop_boxed_888_ = lean_unbox_usize(v_stop_884_);
lean_dec(v_stop_884_);
v_res_889_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__3(v___x_2898__boxed_886_, v_as_882_, v_i_boxed_887_, v_stop_boxed_888_, v_b_885_);
lean_dec_ref(v_as_882_);
return v_res_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1(lean_object* v_x_894_, lean_object* v_a_895_, lean_object* v_a_896_, lean_object* v_a_897_, lean_object* v_a_898_, lean_object* v_a_899_, lean_object* v_a_900_, lean_object* v_a_901_, lean_object* v_a_902_){
_start:
{
lean_object* v___x_904_; uint8_t v___x_905_; 
v___x_904_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_finCases___closed__4));
lean_inc(v_x_894_);
v___x_905_ = l_Lean_Syntax_isOfKind(v_x_894_, v___x_904_);
if (v___x_905_ == 0)
{
lean_object* v___x_906_; 
lean_dec(v_x_894_);
v___x_906_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg();
return v___x_906_;
}
else
{
lean_object* v___x_907_; lean_object* v___y_909_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; uint8_t v___x_931_; 
v___x_907_ = lean_unsigned_to_nat(0u);
v___x_926_ = lean_unsigned_to_nat(1u);
v___x_927_ = l_Lean_Syntax_getArg(v_x_894_, v___x_926_);
v___x_928_ = l_Lean_Syntax_getArgs(v___x_927_);
lean_dec(v___x_927_);
v___x_929_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___closed__0));
v___x_930_ = lean_array_get_size(v___x_928_);
v___x_931_ = lean_nat_dec_lt(v___x_907_, v___x_930_);
if (v___x_931_ == 0)
{
lean_dec_ref(v___x_928_);
v___y_909_ = v___x_929_;
goto v___jp_908_;
}
else
{
lean_object* v___x_932_; lean_object* v___x_933_; uint8_t v___x_934_; 
v___x_932_ = lean_box(v___x_905_);
v___x_933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_933_, 0, v___x_932_);
lean_ctor_set(v___x_933_, 1, v___x_929_);
v___x_934_ = lean_nat_dec_le(v___x_930_, v___x_930_);
if (v___x_934_ == 0)
{
if (v___x_931_ == 0)
{
lean_dec_ref_known(v___x_933_, 2);
lean_dec_ref(v___x_928_);
v___y_909_ = v___x_929_;
goto v___jp_908_;
}
else
{
size_t v___x_935_; size_t v___x_936_; lean_object* v___x_937_; lean_object* v_snd_938_; 
v___x_935_ = ((size_t)0ULL);
v___x_936_ = lean_usize_of_nat(v___x_930_);
v___x_937_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__3(v___x_905_, v___x_928_, v___x_935_, v___x_936_, v___x_933_);
lean_dec_ref(v___x_928_);
v_snd_938_ = lean_ctor_get(v___x_937_, 1);
lean_inc(v_snd_938_);
lean_dec_ref(v___x_937_);
v___y_909_ = v_snd_938_;
goto v___jp_908_;
}
}
else
{
size_t v___x_939_; size_t v___x_940_; lean_object* v___x_941_; lean_object* v_snd_942_; 
v___x_939_ = ((size_t)0ULL);
v___x_940_ = lean_usize_of_nat(v___x_930_);
v___x_941_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__3(v___x_905_, v___x_928_, v___x_939_, v___x_940_, v___x_933_);
lean_dec_ref(v___x_928_);
v_snd_942_ = lean_ctor_get(v___x_941_, 1);
lean_inc(v_snd_942_);
lean_dec_ref(v___x_941_);
v___y_909_ = v_snd_942_;
goto v___jp_908_;
}
}
v___jp_908_:
{
size_t v_sz_910_; size_t v___x_911_; lean_object* v___x_912_; 
v_sz_910_ = lean_array_size(v___y_909_);
v___x_911_ = ((size_t)0ULL);
v___x_912_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__1(v_sz_910_, v___x_911_, v___y_909_);
if (lean_obj_tag(v___x_912_) == 0)
{
lean_object* v___x_913_; 
lean_dec(v_x_894_);
v___x_913_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg();
return v___x_913_;
}
else
{
lean_object* v_val_914_; lean_object* v___x_915_; lean_object* v___x_916_; uint8_t v___x_917_; 
v_val_914_ = lean_ctor_get(v___x_912_, 0);
lean_inc(v_val_914_);
lean_dec_ref_known(v___x_912_, 1);
v___x_915_ = lean_unsigned_to_nat(2u);
v___x_916_ = l_Lean_Syntax_getArg(v_x_894_, v___x_915_);
lean_dec(v_x_894_);
v___x_917_ = l_Lean_Syntax_matchesNull(v___x_916_, v___x_907_);
if (v___x_917_ == 0)
{
lean_object* v___x_918_; 
lean_dec(v_val_914_);
v___x_918_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1_spec__0___redArg();
return v___x_918_;
}
else
{
lean_object* v___x_919_; size_t v_sz_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___f_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
v___x_919_ = lean_box(0);
v_sz_920_ = lean_array_size(v_val_914_);
v___x_921_ = lean_box_usize(v_sz_920_);
v___x_922_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___boxed__const__1));
v___f_923_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_923_, 0, v_val_914_);
lean_closure_set(v___f_923_, 1, v___x_921_);
lean_closure_set(v___f_923_, 2, v___x_922_);
lean_closure_set(v___f_923_, 3, v___x_919_);
v___x_924_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_focus___boxed), 11, 2);
lean_closure_set(v___x_924_, 0, lean_box(0));
lean_closure_set(v___x_924_, 1, v___f_923_);
v___x_925_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___x_924_, v_a_895_, v_a_896_, v_a_897_, v_a_898_, v_a_899_, v_a_900_, v_a_901_, v_a_902_);
return v___x_925_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1___boxed(lean_object* v_x_943_, lean_object* v_a_944_, lean_object* v_a_945_, lean_object* v_a_946_, lean_object* v_a_947_, lean_object* v_a_948_, lean_object* v_a_949_, lean_object* v_a_950_, lean_object* v_a_951_, lean_object* v_a_952_){
_start:
{
lean_object* v_res_953_; 
v_res_953_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__FinCases______elabRules__Lean__Elab__Tactic__finCases__1(v_x_943_, v_a_944_, v_a_945_, v_a_946_, v_a_947_, v_a_948_, v_a_949_, v_a_950_, v_a_951_);
lean_dec(v_a_951_);
lean_dec_ref(v_a_950_);
lean_dec(v_a_949_);
lean_dec_ref(v_a_948_);
lean_dec(v_a_947_);
lean_dec_ref(v_a_946_);
lean_dec(v_a_945_);
lean_dec_ref(v_a_944_);
return v_res_953_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
}
#ifdef __cplusplus
}
#endif
