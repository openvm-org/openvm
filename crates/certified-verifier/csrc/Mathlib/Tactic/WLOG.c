// Lean compiler output
// Module: Mathlib.Tactic.WLOG
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Cases import all Lean.MetavarContext public import Mathlib.Tactic.Core public import Mathlib.Tactic.Push
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
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_isLetVar___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_push(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_binderIdent;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_byCases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_ensureHasNoMVars___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___at___00Lean_assignExp_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_assertHypotheses(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_tryClearMany(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_Meta_introNCore(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_getFVarIdsAt(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_MkBinding_collectForwardDeps(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_filterOutImplementationDetails(lean_object*, lean_object*);
lean_object* l___private_Lean_MetavarContext_0__Lean_MetavarContext_MkBinding_mkAuxMVarType(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_wlog_spec__5_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_wlog_spec__5_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_wlog___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_wlog___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Lean_MVarId_wlog___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "failed to create binder due to failure when reverting variable dependencies"};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_wlog___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__3;
static lean_once_cell_t lp_mathlib_Lean_MVarId_wlog___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__4;
static lean_once_cell_t lp_mathlib_Lean_MVarId_wlog___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__5;
static const lean_string_object lp_mathlib_Lean_MVarId_wlog___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_wlog___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Lean_MVarId_wlog___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "this"};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_wlog___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(38, 116, 214, 236, 212, 160, 188, 150)}};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___boxed(lean_object**);
static const lean_string_object lp_mathlib_Lean_MVarId_wlog___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "wlog"};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_wlog___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_wlog___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 135, 214, 229, 46, 151, 185, 171)}};
static const lean_object* lp_mathlib_Lean_MVarId_wlog___closed__1 = (const lean_object*)&lp_mathlib_Lean_MVarId_wlog___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlogCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlogCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlogCore___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlogCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlogCore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlogCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__3_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Lean_MVarId_wlog___closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 8, 157, 249, 128, 48, 59, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "wlog "};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__11_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__15_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = " generalizing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__19_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__21_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__24_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__27_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__32_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__33;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__34_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__35_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__36;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__37;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__38;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog___closed__39;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlog;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "wlog!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 96, 154, 167, 208, 225, 218, 237)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_wlog_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "wlog! "};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_wlog_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog_x21___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog_x21___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog_x21___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog_x21___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog_x21___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog_x21___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_wlog_x21___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlog_x21;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlogCore___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_wlog___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___redArg(v_e_30_, v___y_36_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___boxed(lean_object* v_e_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1(v_e_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
lean_dec(v___y_49_);
lean_dec_ref(v___y_48_);
lean_dec(v___y_47_);
lean_dec_ref(v___y_46_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___lam__0(lean_object* v_x_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_){
_start:
{
lean_object* v___x_62_; 
lean_inc(v___y_56_);
lean_inc_ref(v___y_55_);
lean_inc(v___y_54_);
lean_inc_ref(v___y_53_);
v___x_62_ = lean_apply_9(v_x_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_, v___y_59_, v___y_60_, lean_box(0));
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___lam__0___boxed(lean_object* v_x_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___lam__0(v_x_63_, v___y_64_, v___y_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_, v___y_70_, v___y_71_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg(lean_object* v_mvarId_74_, lean_object* v_x_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v___f_85_; lean_object* v___x_86_; 
lean_inc(v___y_79_);
lean_inc_ref(v___y_78_);
lean_inc(v___y_77_);
lean_inc_ref(v___y_76_);
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_85_, 0, v_x_75_);
lean_closure_set(v___f_85_, 1, v___y_76_);
lean_closure_set(v___f_85_, 2, v___y_77_);
lean_closure_set(v___f_85_, 3, v___y_78_);
lean_closure_set(v___f_85_, 4, v___y_79_);
v___x_86_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_74_, v___f_85_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
if (lean_obj_tag(v___x_86_) == 0)
{
return v___x_86_;
}
else
{
lean_object* v_a_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_94_; 
v_a_87_ = lean_ctor_get(v___x_86_, 0);
v_isSharedCheck_94_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_94_ == 0)
{
v___x_89_ = v___x_86_;
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_a_87_);
lean_dec(v___x_86_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_92_; 
if (v_isShared_90_ == 0)
{
v___x_92_ = v___x_89_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v_a_87_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg___boxed(lean_object* v_mvarId_95_, lean_object* v_x_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg(v_mvarId_95_, v_x_96_, v___y_97_, v___y_98_, v___y_99_, v___y_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
lean_dec(v___y_102_);
lean_dec_ref(v___y_101_);
lean_dec(v___y_100_);
lean_dec_ref(v___y_99_);
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4(lean_object* v_00_u03b1_107_, lean_object* v_mvarId_108_, lean_object* v_x_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg(v_mvarId_108_, v_x_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___boxed(lean_object* v_00_u03b1_120_, lean_object* v_mvarId_121_, lean_object* v_x_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4(v_00_u03b1_120_, v_mvarId_121_, v_x_122_, v___y_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
lean_dec(v___y_130_);
lean_dec_ref(v___y_129_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
lean_dec(v___y_124_);
lean_dec_ref(v___y_123_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0(size_t v_sz_133_, size_t v_i_134_, lean_object* v_bs_135_){
_start:
{
uint8_t v___x_136_; 
v___x_136_ = lean_usize_dec_lt(v_i_134_, v_sz_133_);
if (v___x_136_ == 0)
{
return v_bs_135_;
}
else
{
lean_object* v_v_137_; lean_object* v___x_138_; lean_object* v_bs_x27_139_; lean_object* v___x_140_; size_t v___x_141_; size_t v___x_142_; lean_object* v___x_143_; 
v_v_137_ = lean_array_uget(v_bs_135_, v_i_134_);
v___x_138_ = lean_unsigned_to_nat(0u);
v_bs_x27_139_ = lean_array_uset(v_bs_135_, v_i_134_, v___x_138_);
v___x_140_ = l_Lean_Expr_fvar___override(v_v_137_);
v___x_141_ = ((size_t)1ULL);
v___x_142_ = lean_usize_add(v_i_134_, v___x_141_);
v___x_143_ = lean_array_uset(v_bs_x27_139_, v_i_134_, v___x_140_);
v_i_134_ = v___x_142_;
v_bs_135_ = v___x_143_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0___boxed(lean_object* v_sz_145_, lean_object* v_i_146_, lean_object* v_bs_147_){
_start:
{
size_t v_sz_boxed_148_; size_t v_i_boxed_149_; lean_object* v_res_150_; 
v_sz_boxed_148_ = lean_unbox_usize(v_sz_145_);
lean_dec(v_sz_145_);
v_i_boxed_149_ = lean_unbox_usize(v_i_146_);
lean_dec(v_i_146_);
v_res_150_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0(v_sz_boxed_148_, v_i_boxed_149_, v_bs_147_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___redArg(lean_object* v_mvarId_151_, lean_object* v_val_152_, lean_object* v___y_153_){
_start:
{
lean_object* v___x_155_; lean_object* v_mctx_156_; lean_object* v_cache_157_; lean_object* v_zetaDeltaFVarIds_158_; lean_object* v_postponed_159_; lean_object* v_diag_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_188_; 
v___x_155_ = lean_st_ref_take(v___y_153_);
v_mctx_156_ = lean_ctor_get(v___x_155_, 0);
v_cache_157_ = lean_ctor_get(v___x_155_, 1);
v_zetaDeltaFVarIds_158_ = lean_ctor_get(v___x_155_, 2);
v_postponed_159_ = lean_ctor_get(v___x_155_, 3);
v_diag_160_ = lean_ctor_get(v___x_155_, 4);
v_isSharedCheck_188_ = !lean_is_exclusive(v___x_155_);
if (v_isSharedCheck_188_ == 0)
{
v___x_162_ = v___x_155_;
v_isShared_163_ = v_isSharedCheck_188_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_diag_160_);
lean_inc(v_postponed_159_);
lean_inc(v_zetaDeltaFVarIds_158_);
lean_inc(v_cache_157_);
lean_inc(v_mctx_156_);
lean_dec(v___x_155_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_188_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v_depth_164_; lean_object* v_levelAssignDepth_165_; lean_object* v_lmvarCounter_166_; lean_object* v_mvarCounter_167_; lean_object* v_lDecls_168_; lean_object* v_decls_169_; lean_object* v_userNames_170_; lean_object* v_lAssignment_171_; lean_object* v_eAssignment_172_; lean_object* v_dAssignment_173_; lean_object* v___x_175_; uint8_t v_isShared_176_; uint8_t v_isSharedCheck_187_; 
v_depth_164_ = lean_ctor_get(v_mctx_156_, 0);
v_levelAssignDepth_165_ = lean_ctor_get(v_mctx_156_, 1);
v_lmvarCounter_166_ = lean_ctor_get(v_mctx_156_, 2);
v_mvarCounter_167_ = lean_ctor_get(v_mctx_156_, 3);
v_lDecls_168_ = lean_ctor_get(v_mctx_156_, 4);
v_decls_169_ = lean_ctor_get(v_mctx_156_, 5);
v_userNames_170_ = lean_ctor_get(v_mctx_156_, 6);
v_lAssignment_171_ = lean_ctor_get(v_mctx_156_, 7);
v_eAssignment_172_ = lean_ctor_get(v_mctx_156_, 8);
v_dAssignment_173_ = lean_ctor_get(v_mctx_156_, 9);
v_isSharedCheck_187_ = !lean_is_exclusive(v_mctx_156_);
if (v_isSharedCheck_187_ == 0)
{
v___x_175_ = v_mctx_156_;
v_isShared_176_ = v_isSharedCheck_187_;
goto v_resetjp_174_;
}
else
{
lean_inc(v_dAssignment_173_);
lean_inc(v_eAssignment_172_);
lean_inc(v_lAssignment_171_);
lean_inc(v_userNames_170_);
lean_inc(v_decls_169_);
lean_inc(v_lDecls_168_);
lean_inc(v_mvarCounter_167_);
lean_inc(v_lmvarCounter_166_);
lean_inc(v_levelAssignDepth_165_);
lean_inc(v_depth_164_);
lean_dec(v_mctx_156_);
v___x_175_ = lean_box(0);
v_isShared_176_ = v_isSharedCheck_187_;
goto v_resetjp_174_;
}
v_resetjp_174_:
{
lean_object* v___x_177_; lean_object* v___x_179_; 
v___x_177_ = l_Lean_PersistentHashMap_insert___at___00Lean_assignExp_spec__0___redArg(v_eAssignment_172_, v_mvarId_151_, v_val_152_);
if (v_isShared_176_ == 0)
{
lean_ctor_set(v___x_175_, 8, v___x_177_);
v___x_179_ = v___x_175_;
goto v_reusejp_178_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v_depth_164_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_levelAssignDepth_165_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v_lmvarCounter_166_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v_mvarCounter_167_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v_lDecls_168_);
lean_ctor_set(v_reuseFailAlloc_186_, 5, v_decls_169_);
lean_ctor_set(v_reuseFailAlloc_186_, 6, v_userNames_170_);
lean_ctor_set(v_reuseFailAlloc_186_, 7, v_lAssignment_171_);
lean_ctor_set(v_reuseFailAlloc_186_, 8, v___x_177_);
lean_ctor_set(v_reuseFailAlloc_186_, 9, v_dAssignment_173_);
v___x_179_ = v_reuseFailAlloc_186_;
goto v_reusejp_178_;
}
v_reusejp_178_:
{
lean_object* v___x_181_; 
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 0, v___x_179_);
v___x_181_ = v___x_162_;
goto v_reusejp_180_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_179_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v_cache_157_);
lean_ctor_set(v_reuseFailAlloc_185_, 2, v_zetaDeltaFVarIds_158_);
lean_ctor_set(v_reuseFailAlloc_185_, 3, v_postponed_159_);
lean_ctor_set(v_reuseFailAlloc_185_, 4, v_diag_160_);
v___x_181_ = v_reuseFailAlloc_185_;
goto v_reusejp_180_;
}
v_reusejp_180_:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = lean_st_ref_set(v___y_153_, v___x_181_);
v___x_183_ = lean_box(0);
v___x_184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
return v___x_184_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___redArg___boxed(lean_object* v_mvarId_189_, lean_object* v_val_190_, lean_object* v___y_191_, lean_object* v___y_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___redArg(v_mvarId_189_, v_val_190_, v___y_191_);
lean_dec(v___y_191_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg(lean_object* v_as_194_, size_t v_i_195_, size_t v_stop_196_, lean_object* v_b_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_){
_start:
{
lean_object* v_a_203_; uint8_t v___x_207_; 
v___x_207_ = lean_usize_dec_eq(v_i_195_, v_stop_196_);
if (v___x_207_ == 0)
{
lean_object* v___x_208_; lean_object* v___x_211_; 
v___x_208_ = lean_array_uget_borrowed(v_as_194_, v_i_195_);
lean_inc(v___x_208_);
v___x_211_ = l_Lean_FVarId_isLetVar___redArg(v___x_208_, v___x_207_, v___y_198_, v___y_199_, v___y_200_);
if (lean_obj_tag(v___x_211_) == 0)
{
lean_object* v_a_212_; uint8_t v___x_213_; 
v_a_212_ = lean_ctor_get(v___x_211_, 0);
lean_inc(v_a_212_);
lean_dec_ref_known(v___x_211_, 1);
v___x_213_ = lean_unbox(v_a_212_);
lean_dec(v_a_212_);
if (v___x_213_ == 0)
{
goto v___jp_209_;
}
else
{
v_a_203_ = v_b_197_;
goto v___jp_202_;
}
}
else
{
if (lean_obj_tag(v___x_211_) == 0)
{
lean_object* v_a_214_; uint8_t v___x_215_; 
v_a_214_ = lean_ctor_get(v___x_211_, 0);
lean_inc(v_a_214_);
lean_dec_ref_known(v___x_211_, 1);
v___x_215_ = lean_unbox(v_a_214_);
lean_dec(v_a_214_);
if (v___x_215_ == 0)
{
v_a_203_ = v_b_197_;
goto v___jp_202_;
}
else
{
goto v___jp_209_;
}
}
else
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_223_; 
lean_dec_ref(v_b_197_);
v_a_216_ = lean_ctor_get(v___x_211_, 0);
v_isSharedCheck_223_ = !lean_is_exclusive(v___x_211_);
if (v_isSharedCheck_223_ == 0)
{
v___x_218_ = v___x_211_;
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_211_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_221_; 
if (v_isShared_219_ == 0)
{
v___x_221_ = v___x_218_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v_a_216_);
v___x_221_ = v_reuseFailAlloc_222_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
return v___x_221_;
}
}
}
}
v___jp_209_:
{
lean_object* v___x_210_; 
lean_inc(v___x_208_);
v___x_210_ = lean_array_push(v_b_197_, v___x_208_);
v_a_203_ = v___x_210_;
goto v___jp_202_;
}
}
else
{
lean_object* v___x_224_; 
v___x_224_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_224_, 0, v_b_197_);
return v___x_224_;
}
v___jp_202_:
{
size_t v___x_204_; size_t v___x_205_; 
v___x_204_ = ((size_t)1ULL);
v___x_205_ = lean_usize_add(v_i_195_, v___x_204_);
v_i_195_ = v___x_205_;
v_b_197_ = v_a_203_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg___boxed(lean_object* v_as_225_, lean_object* v_i_226_, lean_object* v_stop_227_, lean_object* v_b_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_){
_start:
{
size_t v_i_boxed_233_; size_t v_stop_boxed_234_; lean_object* v_res_235_; 
v_i_boxed_233_ = lean_unbox_usize(v_i_226_);
lean_dec(v_i_226_);
v_stop_boxed_234_ = lean_unbox_usize(v_stop_227_);
lean_dec(v_stop_227_);
v_res_235_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg(v_as_225_, v_i_boxed_233_, v_stop_boxed_234_, v_b_228_, v___y_229_, v___y_230_, v___y_231_);
lean_dec(v___y_231_);
lean_dec_ref(v___y_230_);
lean_dec_ref(v___y_229_);
lean_dec_ref(v_as_225_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__0(lean_object* v___x_236_, lean_object* v_fvarId_237_, lean_object* v_mvarId_238_, lean_object* v___x_239_, lean_object* v___x_240_, lean_object* v_fst_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_){
_start:
{
lean_object* v_a_252_; lean_object* v___y_265_; lean_object* v___x_275_; uint8_t v___x_276_; 
v___x_275_ = lean_mk_empty_array_with_capacity(v___x_239_);
v___x_276_ = lean_nat_dec_lt(v___x_239_, v___x_240_);
if (v___x_276_ == 0)
{
v_a_252_ = v___x_275_;
goto v___jp_251_;
}
else
{
uint8_t v___x_277_; 
v___x_277_ = lean_nat_dec_le(v___x_240_, v___x_240_);
if (v___x_277_ == 0)
{
if (v___x_276_ == 0)
{
v_a_252_ = v___x_275_;
goto v___jp_251_;
}
else
{
size_t v___x_278_; size_t v___x_279_; lean_object* v___x_280_; 
v___x_278_ = ((size_t)0ULL);
v___x_279_ = lean_usize_of_nat(v___x_240_);
v___x_280_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg(v_fst_241_, v___x_278_, v___x_279_, v___x_275_, v___y_246_, v___y_248_, v___y_249_);
v___y_265_ = v___x_280_;
goto v___jp_264_;
}
}
else
{
size_t v___x_281_; size_t v___x_282_; lean_object* v___x_283_; 
v___x_281_ = ((size_t)0ULL);
v___x_282_ = lean_usize_of_nat(v___x_240_);
v___x_283_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg(v_fst_241_, v___x_281_, v___x_282_, v___x_275_, v___y_246_, v___y_248_, v___y_249_);
v___y_265_ = v___x_283_;
goto v___jp_264_;
}
}
v___jp_251_:
{
lean_object* v___x_253_; size_t v_sz_254_; size_t v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v_a_261_; lean_object* v___x_262_; 
v___x_253_ = l_Lean_Expr_fvar___override(v___x_236_);
v_sz_254_ = lean_array_size(v_a_252_);
v___x_255_ = ((size_t)0ULL);
v___x_256_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0(v_sz_254_, v___x_255_, v_a_252_);
v___x_257_ = l_Lean_mkAppN(v___x_253_, v___x_256_);
lean_dec_ref(v___x_256_);
v___x_258_ = l_Lean_Expr_fvar___override(v_fvarId_237_);
v___x_259_ = l_Lean_Expr_app___override(v___x_257_, v___x_258_);
v___x_260_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_wlog_spec__1___redArg(v___x_259_, v___y_247_);
v_a_261_ = lean_ctor_get(v___x_260_, 0);
lean_inc_n(v_a_261_, 2);
lean_dec_ref(v___x_260_);
v___x_262_ = l_Lean_Elab_Tactic_ensureHasNoMVars___redArg(v_a_261_, v___y_244_, v___y_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
if (lean_obj_tag(v___x_262_) == 0)
{
lean_object* v___x_263_; 
lean_dec_ref_known(v___x_262_, 1);
v___x_263_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___redArg(v_mvarId_238_, v_a_261_, v___y_247_);
return v___x_263_;
}
else
{
lean_dec(v_a_261_);
lean_dec(v_mvarId_238_);
return v___x_262_;
}
}
v___jp_264_:
{
if (lean_obj_tag(v___y_265_) == 0)
{
lean_object* v_a_266_; 
v_a_266_ = lean_ctor_get(v___y_265_, 0);
lean_inc(v_a_266_);
lean_dec_ref_known(v___y_265_, 1);
v_a_252_ = v_a_266_;
goto v___jp_251_;
}
else
{
lean_object* v_a_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_274_; 
lean_dec(v_mvarId_238_);
lean_dec(v_fvarId_237_);
lean_dec(v___x_236_);
v_a_267_ = lean_ctor_get(v___y_265_, 0);
v_isSharedCheck_274_ = !lean_is_exclusive(v___y_265_);
if (v_isSharedCheck_274_ == 0)
{
v___x_269_ = v___y_265_;
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_a_267_);
lean_dec(v___y_265_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_272_; 
if (v_isShared_270_ == 0)
{
v___x_272_ = v___x_269_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v_a_267_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__0___boxed(lean_object* v___x_284_, lean_object* v_fvarId_285_, lean_object* v_mvarId_286_, lean_object* v___x_287_, lean_object* v___x_288_, lean_object* v_fst_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Lean_MVarId_wlog___lam__0(v___x_284_, v_fvarId_285_, v_mvarId_286_, v___x_287_, v___x_288_, v_fst_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_);
lean_dec(v___y_297_);
lean_dec_ref(v___y_296_);
lean_dec(v___y_295_);
lean_dec_ref(v___y_294_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
lean_dec(v___y_291_);
lean_dec_ref(v___y_290_);
lean_dec_ref(v_fst_289_);
lean_dec(v___x_288_);
lean_dec(v___x_287_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__6(size_t v_sz_300_, size_t v_i_301_, lean_object* v_bs_302_){
_start:
{
uint8_t v___x_303_; 
v___x_303_ = lean_usize_dec_lt(v_i_301_, v_sz_300_);
if (v___x_303_ == 0)
{
return v_bs_302_;
}
else
{
lean_object* v_v_304_; lean_object* v___x_305_; lean_object* v_bs_x27_306_; lean_object* v___x_307_; size_t v___x_308_; size_t v___x_309_; lean_object* v___x_310_; 
v_v_304_ = lean_array_uget(v_bs_302_, v_i_301_);
v___x_305_ = lean_unsigned_to_nat(0u);
v_bs_x27_306_ = lean_array_uset(v_bs_302_, v_i_301_, v___x_305_);
v___x_307_ = l_Lean_Expr_fvarId_x21(v_v_304_);
lean_dec(v_v_304_);
v___x_308_ = ((size_t)1ULL);
v___x_309_ = lean_usize_add(v_i_301_, v___x_308_);
v___x_310_ = lean_array_uset(v_bs_x27_306_, v_i_301_, v___x_307_);
v_i_301_ = v___x_309_;
v_bs_302_ = v___x_310_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__6___boxed(lean_object* v_sz_312_, lean_object* v_i_313_, lean_object* v_bs_314_){
_start:
{
size_t v_sz_boxed_315_; size_t v_i_boxed_316_; lean_object* v_res_317_; 
v_sz_boxed_315_ = lean_unbox_usize(v_sz_312_);
lean_dec(v_sz_312_);
v_i_boxed_316_ = lean_unbox_usize(v_i_313_);
lean_dec(v_i_313_);
v_res_317_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__6(v_sz_boxed_315_, v_i_boxed_316_, v_bs_314_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_wlog_spec__5_spec__5(lean_object* v_msgData_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
lean_object* v___x_324_; lean_object* v_env_325_; lean_object* v___x_326_; lean_object* v_mctx_327_; lean_object* v_lctx_328_; lean_object* v_options_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_324_ = lean_st_ref_get(v___y_322_);
v_env_325_ = lean_ctor_get(v___x_324_, 0);
lean_inc_ref(v_env_325_);
lean_dec(v___x_324_);
v___x_326_ = lean_st_ref_get(v___y_320_);
v_mctx_327_ = lean_ctor_get(v___x_326_, 0);
lean_inc_ref(v_mctx_327_);
lean_dec(v___x_326_);
v_lctx_328_ = lean_ctor_get(v___y_319_, 2);
v_options_329_ = lean_ctor_get(v___y_321_, 2);
lean_inc_ref(v_options_329_);
lean_inc_ref(v_lctx_328_);
v___x_330_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_330_, 0, v_env_325_);
lean_ctor_set(v___x_330_, 1, v_mctx_327_);
lean_ctor_set(v___x_330_, 2, v_lctx_328_);
lean_ctor_set(v___x_330_, 3, v_options_329_);
v___x_331_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
lean_ctor_set(v___x_331_, 1, v_msgData_318_);
v___x_332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_wlog_spec__5_spec__5___boxed(lean_object* v_msgData_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_wlog_spec__5_spec__5(v_msgData_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
lean_dec(v___y_337_);
lean_dec_ref(v___y_336_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___redArg(lean_object* v_msg_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v_ref_346_; lean_object* v___x_347_; lean_object* v_a_348_; lean_object* v___x_350_; uint8_t v_isShared_351_; uint8_t v_isSharedCheck_356_; 
v_ref_346_ = lean_ctor_get(v___y_343_, 5);
v___x_347_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_wlog_spec__5_spec__5(v_msg_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
v_a_348_ = lean_ctor_get(v___x_347_, 0);
v_isSharedCheck_356_ = !lean_is_exclusive(v___x_347_);
if (v_isSharedCheck_356_ == 0)
{
v___x_350_ = v___x_347_;
v_isShared_351_ = v_isSharedCheck_356_;
goto v_resetjp_349_;
}
else
{
lean_inc(v_a_348_);
lean_dec(v___x_347_);
v___x_350_ = lean_box(0);
v_isShared_351_ = v_isSharedCheck_356_;
goto v_resetjp_349_;
}
v_resetjp_349_:
{
lean_object* v___x_352_; lean_object* v___x_354_; 
lean_inc(v_ref_346_);
v___x_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_352_, 0, v_ref_346_);
lean_ctor_set(v___x_352_, 1, v_a_348_);
if (v_isShared_351_ == 0)
{
lean_ctor_set_tag(v___x_350_, 1);
lean_ctor_set(v___x_350_, 0, v___x_352_);
v___x_354_ = v___x_350_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v___x_352_);
v___x_354_ = v_reuseFailAlloc_355_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
return v___x_354_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___redArg___boxed(lean_object* v_msg_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___redArg(v_msg_357_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
lean_dec(v___y_361_);
lean_dec_ref(v___y_360_);
lean_dec(v___y_359_);
lean_dec_ref(v___y_358_);
return v_res_363_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_wlog___lam__1___closed__3(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_368_ = ((lean_object*)(lp_mathlib_Lean_MVarId_wlog___lam__1___closed__2));
v___x_369_ = l_Lean_stringToMessageData(v___x_368_);
return v___x_369_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_wlog___lam__1___closed__4(void){
_start:
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_370_ = lean_box(0);
v___x_371_ = lean_unsigned_to_nat(16u);
v___x_372_ = lean_mk_array(v___x_371_, v___x_370_);
return v___x_372_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_wlog___lam__1___closed__5(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_373_ = lean_obj_once(&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__4, &lp_mathlib_Lean_MVarId_wlog___lam__1___closed__4_once, _init_lp_mathlib_Lean_MVarId_wlog___lam__1___closed__4);
v___x_374_ = lean_unsigned_to_nat(0u);
v___x_375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v___x_373_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1(lean_object* v_goal_382_, lean_object* v___x_383_, lean_object* v___x_384_, lean_object* v_P_385_, lean_object* v___x_386_, lean_object* v_xs_387_, lean_object* v_h_388_, lean_object* v_H_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_){
_start:
{
lean_object* v___y_400_; lean_object* v___y_401_; lean_object* v___y_402_; lean_object* v___y_403_; lean_object* v___y_404_; lean_object* v___y_405_; lean_object* v___y_406_; lean_object* v___y_407_; lean_object* v___y_408_; lean_object* v___y_409_; lean_object* v___y_410_; lean_object* v___y_411_; lean_object* v___y_412_; lean_object* v___y_413_; lean_object* v___y_414_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_417_; lean_object* v___y_461_; lean_object* v___y_462_; lean_object* v___y_463_; lean_object* v___y_464_; uint8_t v___y_465_; lean_object* v___y_466_; lean_object* v___y_467_; lean_object* v___y_468_; lean_object* v_____x_469_; lean_object* v___y_470_; lean_object* v___y_471_; lean_object* v___y_472_; lean_object* v___y_473_; lean_object* v___y_474_; lean_object* v___y_475_; lean_object* v___y_476_; lean_object* v___y_477_; uint8_t v___y_484_; lean_object* v___y_485_; uint8_t v___y_486_; lean_object* v___y_487_; uint8_t v___y_488_; lean_object* v_fst_489_; lean_object* v_snd_490_; lean_object* v___y_567_; uint8_t v___y_568_; uint8_t v___y_569_; lean_object* v___y_570_; uint8_t v___y_571_; lean_object* v_a_572_; lean_object* v___y_619_; uint8_t v___y_620_; lean_object* v___y_621_; uint8_t v___y_622_; lean_object* v___y_623_; uint8_t v___y_624_; lean_object* v_a_625_; lean_object* v_a_626_; lean_object* v___y_663_; uint8_t v___y_664_; lean_object* v___y_665_; uint8_t v___y_666_; lean_object* v___y_667_; lean_object* v___y_668_; lean_object* v___x_744_; 
lean_inc(v_goal_382_);
v___x_744_ = l_Lean_MVarId_checkNotAssigned(v_goal_382_, v___x_383_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_744_) == 0)
{
uint8_t v___y_746_; lean_object* v___y_747_; lean_object* v___y_748_; lean_object* v___y_771_; uint8_t v___y_772_; lean_object* v___y_776_; 
lean_dec_ref_known(v___x_744_, 1);
if (lean_obj_tag(v_H_389_) == 0)
{
lean_object* v___x_779_; 
v___x_779_ = ((lean_object*)(lp_mathlib_Lean_MVarId_wlog___lam__1___closed__9));
v___y_776_ = v___x_779_;
goto v___jp_775_;
}
else
{
lean_object* v_val_780_; 
v_val_780_ = lean_ctor_get(v_H_389_, 0);
lean_inc(v_val_780_);
lean_dec_ref_known(v_H_389_, 1);
v___y_776_ = v_val_780_;
goto v___jp_775_;
}
v___jp_745_:
{
lean_object* v___x_749_; 
lean_inc(v_goal_382_);
v___x_749_ = l_Lean_MVarId_getType(v_goal_382_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_749_) == 0)
{
lean_object* v_a_750_; uint8_t v___x_751_; lean_object* v___x_752_; 
v_a_750_ = lean_ctor_get(v___x_749_, 0);
lean_inc(v_a_750_);
lean_dec_ref_known(v___x_749_, 1);
v___x_751_ = 0;
lean_inc_ref(v_P_385_);
lean_inc(v___y_748_);
v___x_752_ = l_Lean_Expr_forallE___override(v___y_748_, v_P_385_, v_a_750_, v___x_751_);
if (lean_obj_tag(v_xs_387_) == 0)
{
lean_object* v___x_753_; 
v___x_753_ = lean_box(0);
v___y_663_ = v___y_747_;
v___y_664_ = v___y_746_;
v___y_665_ = v___x_752_;
v___y_666_ = v___x_751_;
v___y_667_ = v___y_748_;
v___y_668_ = v___x_753_;
goto v___jp_662_;
}
else
{
lean_object* v_val_754_; lean_object* v___x_756_; uint8_t v_isShared_757_; uint8_t v_isSharedCheck_761_; 
v_val_754_ = lean_ctor_get(v_xs_387_, 0);
v_isSharedCheck_761_ = !lean_is_exclusive(v_xs_387_);
if (v_isSharedCheck_761_ == 0)
{
v___x_756_ = v_xs_387_;
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
else
{
lean_inc(v_val_754_);
lean_dec(v_xs_387_);
v___x_756_ = lean_box(0);
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
v_resetjp_755_:
{
lean_object* v___x_759_; 
if (v_isShared_757_ == 0)
{
v___x_759_ = v___x_756_;
goto v_reusejp_758_;
}
else
{
lean_object* v_reuseFailAlloc_760_; 
v_reuseFailAlloc_760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_760_, 0, v_val_754_);
v___x_759_ = v_reuseFailAlloc_760_;
goto v_reusejp_758_;
}
v_reusejp_758_:
{
v___y_663_ = v___y_747_;
v___y_664_ = v___y_746_;
v___y_665_ = v___x_752_;
v___y_666_ = v___x_751_;
v___y_667_ = v___y_748_;
v___y_668_ = v___x_759_;
goto v___jp_662_;
}
}
}
}
else
{
lean_object* v_a_762_; lean_object* v___x_764_; uint8_t v_isShared_765_; uint8_t v_isSharedCheck_769_; 
lean_dec(v___y_748_);
lean_dec(v___y_747_);
lean_dec(v_xs_387_);
lean_dec(v___x_386_);
lean_dec_ref(v_P_385_);
lean_dec(v_goal_382_);
v_a_762_ = lean_ctor_get(v___x_749_, 0);
v_isSharedCheck_769_ = !lean_is_exclusive(v___x_749_);
if (v_isSharedCheck_769_ == 0)
{
v___x_764_ = v___x_749_;
v_isShared_765_ = v_isSharedCheck_769_;
goto v_resetjp_763_;
}
else
{
lean_inc(v_a_762_);
lean_dec(v___x_749_);
v___x_764_ = lean_box(0);
v_isShared_765_ = v_isSharedCheck_769_;
goto v_resetjp_763_;
}
v_resetjp_763_:
{
lean_object* v___x_767_; 
if (v_isShared_765_ == 0)
{
v___x_767_ = v___x_764_;
goto v_reusejp_766_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v_a_762_);
v___x_767_ = v_reuseFailAlloc_768_;
goto v_reusejp_766_;
}
v_reusejp_766_:
{
return v___x_767_;
}
}
}
}
v___jp_770_:
{
if (lean_obj_tag(v_h_388_) == 0)
{
lean_object* v___x_773_; 
v___x_773_ = ((lean_object*)(lp_mathlib_Lean_MVarId_wlog___lam__1___closed__7));
v___y_746_ = v___y_772_;
v___y_747_ = v___y_771_;
v___y_748_ = v___x_773_;
goto v___jp_745_;
}
else
{
lean_object* v_val_774_; 
v_val_774_ = lean_ctor_get(v_h_388_, 0);
lean_inc(v_val_774_);
lean_dec_ref_known(v_h_388_, 1);
v___y_746_ = v___y_772_;
v___y_747_ = v___y_771_;
v___y_748_ = v_val_774_;
goto v___jp_745_;
}
}
v___jp_775_:
{
if (lean_obj_tag(v_h_388_) == 0)
{
uint8_t v___x_777_; 
v___x_777_ = 1;
v___y_771_ = v___y_776_;
v___y_772_ = v___x_777_;
goto v___jp_770_;
}
else
{
uint8_t v___x_778_; 
v___x_778_ = 0;
v___y_771_ = v___y_776_;
v___y_772_ = v___x_778_;
goto v___jp_770_;
}
}
}
else
{
lean_object* v_a_781_; lean_object* v___x_783_; uint8_t v_isShared_784_; uint8_t v_isSharedCheck_788_; 
lean_dec(v_H_389_);
lean_dec(v_h_388_);
lean_dec(v_xs_387_);
lean_dec(v___x_386_);
lean_dec_ref(v_P_385_);
lean_dec(v_goal_382_);
v_a_781_ = lean_ctor_get(v___x_744_, 0);
v_isSharedCheck_788_ = !lean_is_exclusive(v___x_744_);
if (v_isSharedCheck_788_ == 0)
{
v___x_783_ = v___x_744_;
v_isShared_784_ = v_isSharedCheck_788_;
goto v_resetjp_782_;
}
else
{
lean_inc(v_a_781_);
lean_dec(v___x_744_);
v___x_783_ = lean_box(0);
v_isShared_784_ = v_isSharedCheck_788_;
goto v_resetjp_782_;
}
v_resetjp_782_:
{
lean_object* v___x_786_; 
if (v_isShared_784_ == 0)
{
v___x_786_ = v___x_783_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_787_; 
v_reuseFailAlloc_787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_787_, 0, v_a_781_);
v___x_786_ = v_reuseFailAlloc_787_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
return v___x_786_;
}
}
}
v___jp_399_:
{
lean_object* v___x_418_; 
v___x_418_ = l_Lean_MVarId_byCases(v___y_406_, v_P_385_, v___y_417_, v___y_414_, v___y_412_, v___y_411_, v___y_409_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_object* v_a_419_; lean_object* v_fst_420_; lean_object* v_snd_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_451_; 
v_a_419_ = lean_ctor_get(v___x_418_, 0);
lean_inc(v_a_419_);
lean_dec_ref_known(v___x_418_, 1);
v_fst_420_ = lean_ctor_get(v_a_419_, 0);
v_snd_421_ = lean_ctor_get(v_a_419_, 1);
v_isSharedCheck_451_ = !lean_is_exclusive(v_a_419_);
if (v_isSharedCheck_451_ == 0)
{
v___x_423_ = v_a_419_;
v_isShared_424_ = v_isSharedCheck_451_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_snd_421_);
lean_inc(v_fst_420_);
lean_dec(v_a_419_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_451_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v_mvarId_425_; lean_object* v_fvarId_426_; lean_object* v_mvarId_427_; lean_object* v_fvarId_428_; lean_object* v___f_429_; lean_object* v___x_430_; 
v_mvarId_425_ = lean_ctor_get(v_fst_420_, 0);
lean_inc_n(v_mvarId_425_, 2);
v_fvarId_426_ = lean_ctor_get(v_fst_420_, 1);
lean_inc(v_fvarId_426_);
lean_dec(v_fst_420_);
v_mvarId_427_ = lean_ctor_get(v_snd_421_, 0);
lean_inc(v_mvarId_427_);
v_fvarId_428_ = lean_ctor_get(v_snd_421_, 1);
lean_inc(v_fvarId_428_);
lean_dec(v_snd_421_);
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_wlog___lam__0___boxed), 15, 6);
lean_closure_set(v___f_429_, 0, v___y_403_);
lean_closure_set(v___f_429_, 1, v_fvarId_426_);
lean_closure_set(v___f_429_, 2, v_mvarId_425_);
lean_closure_set(v___f_429_, 3, v___y_401_);
lean_closure_set(v___f_429_, 4, v___y_402_);
lean_closure_set(v___f_429_, 5, v___y_400_);
v___x_430_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg(v_mvarId_425_, v___f_429_, v___y_405_, v___y_407_, v___y_415_, v___y_416_, v___y_414_, v___y_412_, v___y_411_, v___y_409_);
if (lean_obj_tag(v___x_430_) == 0)
{
lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_441_; 
v_isSharedCheck_441_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_441_ == 0)
{
lean_object* v_unused_442_; 
v_unused_442_ = lean_ctor_get(v___x_430_, 0);
lean_dec(v_unused_442_);
v___x_432_ = v___x_430_;
v_isShared_433_ = v_isSharedCheck_441_;
goto v_resetjp_431_;
}
else
{
lean_dec(v___x_430_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_441_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_435_; 
if (v_isShared_424_ == 0)
{
lean_ctor_set(v___x_423_, 1, v_fvarId_428_);
lean_ctor_set(v___x_423_, 0, v___y_404_);
v___x_435_ = v___x_423_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v___y_404_);
lean_ctor_set(v_reuseFailAlloc_440_, 1, v_fvarId_428_);
v___x_435_ = v_reuseFailAlloc_440_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
lean_object* v___x_436_; lean_object* v___x_438_; 
v___x_436_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_436_, 0, v_mvarId_427_);
lean_ctor_set(v___x_436_, 1, v___x_435_);
lean_ctor_set(v___x_436_, 2, v___y_413_);
lean_ctor_set(v___x_436_, 3, v___y_410_);
lean_ctor_set(v___x_436_, 4, v___y_408_);
if (v_isShared_433_ == 0)
{
lean_ctor_set(v___x_432_, 0, v___x_436_);
v___x_438_ = v___x_432_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v___x_436_);
v___x_438_ = v_reuseFailAlloc_439_;
goto v_reusejp_437_;
}
v_reusejp_437_:
{
return v___x_438_;
}
}
}
}
else
{
lean_object* v_a_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_450_; 
lean_dec(v_fvarId_428_);
lean_dec(v_mvarId_427_);
lean_del_object(v___x_423_);
lean_dec(v___y_413_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_404_);
v_a_443_ = lean_ctor_get(v___x_430_, 0);
v_isSharedCheck_450_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_450_ == 0)
{
v___x_445_ = v___x_430_;
v_isShared_446_ = v_isSharedCheck_450_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_a_443_);
lean_dec(v___x_430_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_450_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v___x_448_; 
if (v_isShared_446_ == 0)
{
v___x_448_ = v___x_445_;
goto v_reusejp_447_;
}
else
{
lean_object* v_reuseFailAlloc_449_; 
v_reuseFailAlloc_449_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_449_, 0, v_a_443_);
v___x_448_ = v_reuseFailAlloc_449_;
goto v_reusejp_447_;
}
v_reusejp_447_:
{
return v___x_448_;
}
}
}
}
}
else
{
lean_object* v_a_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_459_; 
lean_dec(v___y_413_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_404_);
lean_dec(v___y_403_);
lean_dec(v___y_402_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
v_a_452_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_459_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_459_ == 0)
{
v___x_454_ = v___x_418_;
v_isShared_455_ = v_isSharedCheck_459_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_a_452_);
lean_dec(v___x_418_);
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
v___jp_460_:
{
if (v___y_465_ == 0)
{
lean_object* v_fst_478_; lean_object* v_snd_479_; 
v_fst_478_ = lean_ctor_get(v_____x_469_, 0);
lean_inc(v_fst_478_);
v_snd_479_ = lean_ctor_get(v_____x_469_, 1);
lean_inc(v_snd_479_);
lean_dec_ref(v_____x_469_);
lean_inc(v___y_464_);
v___y_400_ = v___y_461_;
v___y_401_ = v___y_462_;
v___y_402_ = v___y_463_;
v___y_403_ = v___y_464_;
v___y_404_ = v___y_464_;
v___y_405_ = v___y_470_;
v___y_406_ = v___y_466_;
v___y_407_ = v___y_471_;
v___y_408_ = v___y_467_;
v___y_409_ = v___y_477_;
v___y_410_ = v_fst_478_;
v___y_411_ = v___y_476_;
v___y_412_ = v___y_475_;
v___y_413_ = v_snd_479_;
v___y_414_ = v___y_474_;
v___y_415_ = v___y_472_;
v___y_416_ = v___y_473_;
v___y_417_ = v___y_468_;
goto v___jp_399_;
}
else
{
lean_object* v_fst_480_; lean_object* v_snd_481_; lean_object* v___x_482_; 
lean_dec(v___y_468_);
v_fst_480_ = lean_ctor_get(v_____x_469_, 0);
lean_inc(v_fst_480_);
v_snd_481_ = lean_ctor_get(v_____x_469_, 1);
lean_inc(v_snd_481_);
lean_dec_ref(v_____x_469_);
v___x_482_ = ((lean_object*)(lp_mathlib_Lean_MVarId_wlog___lam__1___closed__1));
lean_inc(v___y_464_);
v___y_400_ = v___y_461_;
v___y_401_ = v___y_462_;
v___y_402_ = v___y_463_;
v___y_403_ = v___y_464_;
v___y_404_ = v___y_464_;
v___y_405_ = v___y_470_;
v___y_406_ = v___y_466_;
v___y_407_ = v___y_471_;
v___y_408_ = v___y_467_;
v___y_409_ = v___y_477_;
v___y_410_ = v_fst_480_;
v___y_411_ = v___y_476_;
v___y_412_ = v___y_475_;
v___y_413_ = v_snd_481_;
v___y_414_ = v___y_474_;
v___y_415_ = v___y_472_;
v___y_416_ = v___y_473_;
v___y_417_ = v___x_482_;
goto v___jp_399_;
}
}
v___jp_483_:
{
lean_object* v___x_491_; lean_object* v___x_492_; 
v___x_491_ = lean_box(0);
lean_inc_ref(v_snd_490_);
v___x_492_ = l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(v_snd_490_, v___x_491_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_492_) == 0)
{
lean_object* v_a_493_; lean_object* v___x_494_; uint8_t v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; 
v_a_493_ = lean_ctor_get(v___x_492_, 0);
lean_inc(v_a_493_);
lean_dec_ref_known(v___x_492_, 1);
v___x_494_ = l_Lean_Expr_mvarId_x21(v_a_493_);
v___x_495_ = 0;
v___x_496_ = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(v___x_496_, 0, v___y_485_);
lean_ctor_set(v___x_496_, 1, v_snd_490_);
lean_ctor_set(v___x_496_, 2, v_a_493_);
lean_ctor_set_uint8(v___x_496_, sizeof(void*)*3, v___y_486_);
lean_ctor_set_uint8(v___x_496_, sizeof(void*)*3 + 1, v___x_495_);
v___x_497_ = lean_unsigned_to_nat(1u);
v___x_498_ = lean_mk_empty_array_with_capacity(v___x_497_);
v___x_499_ = lean_array_push(v___x_498_, v___x_496_);
v___x_500_ = l_Lean_MVarId_assertHypotheses(v_goal_382_, v___x_499_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_500_) == 0)
{
lean_object* v_a_501_; lean_object* v_fst_502_; lean_object* v_snd_503_; lean_object* v___x_504_; 
v_a_501_ = lean_ctor_get(v___x_500_, 0);
lean_inc(v_a_501_);
lean_dec_ref_known(v___x_500_, 1);
v_fst_502_ = lean_ctor_get(v_a_501_, 0);
lean_inc(v_fst_502_);
v_snd_503_ = lean_ctor_get(v_a_501_, 1);
lean_inc(v_snd_503_);
lean_dec(v_a_501_);
v___x_504_ = l_Lean_MVarId_tryClearMany(v___x_494_, v_fst_489_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_504_) == 0)
{
lean_object* v_a_505_; lean_object* v___x_506_; lean_object* v___x_507_; uint8_t v___x_508_; lean_object* v___x_509_; 
v_a_505_ = lean_ctor_get(v___x_504_, 0);
lean_inc(v_a_505_);
lean_dec_ref_known(v___x_504_, 1);
v___x_506_ = lean_array_get_size(v_fst_489_);
v___x_507_ = lean_box(0);
v___x_508_ = 1;
v___x_509_ = l_Lean_Meta_introNCore(v_a_505_, v___x_506_, v___x_507_, v___y_488_, v___x_508_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_509_) == 0)
{
lean_object* v_a_510_; lean_object* v_snd_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v_a_510_ = lean_ctor_get(v___x_509_, 0);
lean_inc(v_a_510_);
lean_dec_ref_known(v___x_509_, 1);
v_snd_511_ = lean_ctor_get(v_a_510_, 1);
lean_inc(v_snd_511_);
lean_dec(v_a_510_);
v___x_512_ = lean_unsigned_to_nat(0u);
v___x_513_ = lean_array_get(v___x_384_, v_fst_502_, v___x_512_);
lean_dec(v_fst_502_);
if (v___y_484_ == 0)
{
lean_object* v___x_514_; 
v___x_514_ = l_Lean_Meta_intro1Core(v_snd_511_, v___x_508_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_514_) == 0)
{
lean_object* v_a_515_; 
v_a_515_ = lean_ctor_get(v___x_514_, 0);
lean_inc(v_a_515_);
lean_dec_ref_known(v___x_514_, 1);
lean_inc_ref(v_fst_489_);
v___y_461_ = v_fst_489_;
v___y_462_ = v___x_512_;
v___y_463_ = v___x_506_;
v___y_464_ = v___x_513_;
v___y_465_ = v___y_484_;
v___y_466_ = v_snd_503_;
v___y_467_ = v_fst_489_;
v___y_468_ = v___y_487_;
v_____x_469_ = v_a_515_;
v___y_470_ = v___y_390_;
v___y_471_ = v___y_391_;
v___y_472_ = v___y_392_;
v___y_473_ = v___y_393_;
v___y_474_ = v___y_394_;
v___y_475_ = v___y_395_;
v___y_476_ = v___y_396_;
v___y_477_ = v___y_397_;
goto v___jp_460_;
}
else
{
lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_523_; 
lean_dec(v___x_513_);
lean_dec(v_snd_503_);
lean_dec_ref(v_fst_489_);
lean_dec(v___y_487_);
lean_dec_ref(v_P_385_);
v_a_516_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_523_ == 0)
{
v___x_518_ = v___x_514_;
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_514_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_521_; 
if (v_isShared_519_ == 0)
{
v___x_521_ = v___x_518_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_a_516_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
else
{
lean_object* v___x_524_; 
v___x_524_ = l_Lean_Meta_intro1Core(v_snd_511_, v___y_488_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_524_) == 0)
{
lean_object* v_a_525_; 
v_a_525_ = lean_ctor_get(v___x_524_, 0);
lean_inc(v_a_525_);
lean_dec_ref_known(v___x_524_, 1);
lean_inc_ref(v_fst_489_);
v___y_461_ = v_fst_489_;
v___y_462_ = v___x_512_;
v___y_463_ = v___x_506_;
v___y_464_ = v___x_513_;
v___y_465_ = v___y_484_;
v___y_466_ = v_snd_503_;
v___y_467_ = v_fst_489_;
v___y_468_ = v___y_487_;
v_____x_469_ = v_a_525_;
v___y_470_ = v___y_390_;
v___y_471_ = v___y_391_;
v___y_472_ = v___y_392_;
v___y_473_ = v___y_393_;
v___y_474_ = v___y_394_;
v___y_475_ = v___y_395_;
v___y_476_ = v___y_396_;
v___y_477_ = v___y_397_;
goto v___jp_460_;
}
else
{
lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_533_; 
lean_dec(v___x_513_);
lean_dec(v_snd_503_);
lean_dec_ref(v_fst_489_);
lean_dec(v___y_487_);
lean_dec_ref(v_P_385_);
v_a_526_ = lean_ctor_get(v___x_524_, 0);
v_isSharedCheck_533_ = !lean_is_exclusive(v___x_524_);
if (v_isSharedCheck_533_ == 0)
{
v___x_528_ = v___x_524_;
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_524_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v___x_531_; 
if (v_isShared_529_ == 0)
{
v___x_531_ = v___x_528_;
goto v_reusejp_530_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v_a_526_);
v___x_531_ = v_reuseFailAlloc_532_;
goto v_reusejp_530_;
}
v_reusejp_530_:
{
return v___x_531_;
}
}
}
}
}
else
{
lean_object* v_a_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_541_; 
lean_dec(v_snd_503_);
lean_dec(v_fst_502_);
lean_dec_ref(v_fst_489_);
lean_dec(v___y_487_);
lean_dec_ref(v_P_385_);
v_a_534_ = lean_ctor_get(v___x_509_, 0);
v_isSharedCheck_541_ = !lean_is_exclusive(v___x_509_);
if (v_isSharedCheck_541_ == 0)
{
v___x_536_ = v___x_509_;
v_isShared_537_ = v_isSharedCheck_541_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_a_534_);
lean_dec(v___x_509_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_541_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
lean_object* v___x_539_; 
if (v_isShared_537_ == 0)
{
v___x_539_ = v___x_536_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_540_; 
v_reuseFailAlloc_540_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_540_, 0, v_a_534_);
v___x_539_ = v_reuseFailAlloc_540_;
goto v_reusejp_538_;
}
v_reusejp_538_:
{
return v___x_539_;
}
}
}
}
else
{
lean_object* v_a_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_549_; 
lean_dec(v_snd_503_);
lean_dec(v_fst_502_);
lean_dec_ref(v_fst_489_);
lean_dec(v___y_487_);
lean_dec_ref(v_P_385_);
v_a_542_ = lean_ctor_get(v___x_504_, 0);
v_isSharedCheck_549_ = !lean_is_exclusive(v___x_504_);
if (v_isSharedCheck_549_ == 0)
{
v___x_544_ = v___x_504_;
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
else
{
lean_inc(v_a_542_);
lean_dec(v___x_504_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v___x_547_; 
if (v_isShared_545_ == 0)
{
v___x_547_ = v___x_544_;
goto v_reusejp_546_;
}
else
{
lean_object* v_reuseFailAlloc_548_; 
v_reuseFailAlloc_548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_548_, 0, v_a_542_);
v___x_547_ = v_reuseFailAlloc_548_;
goto v_reusejp_546_;
}
v_reusejp_546_:
{
return v___x_547_;
}
}
}
}
else
{
lean_object* v_a_550_; lean_object* v___x_552_; uint8_t v_isShared_553_; uint8_t v_isSharedCheck_557_; 
lean_dec(v___x_494_);
lean_dec_ref(v_fst_489_);
lean_dec(v___y_487_);
lean_dec_ref(v_P_385_);
v_a_550_ = lean_ctor_get(v___x_500_, 0);
v_isSharedCheck_557_ = !lean_is_exclusive(v___x_500_);
if (v_isSharedCheck_557_ == 0)
{
v___x_552_ = v___x_500_;
v_isShared_553_ = v_isSharedCheck_557_;
goto v_resetjp_551_;
}
else
{
lean_inc(v_a_550_);
lean_dec(v___x_500_);
v___x_552_ = lean_box(0);
v_isShared_553_ = v_isSharedCheck_557_;
goto v_resetjp_551_;
}
v_resetjp_551_:
{
lean_object* v___x_555_; 
if (v_isShared_553_ == 0)
{
v___x_555_ = v___x_552_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_556_; 
v_reuseFailAlloc_556_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_556_, 0, v_a_550_);
v___x_555_ = v_reuseFailAlloc_556_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
return v___x_555_;
}
}
}
}
else
{
lean_object* v_a_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_565_; 
lean_dec_ref(v_snd_490_);
lean_dec_ref(v_fst_489_);
lean_dec(v___y_487_);
lean_dec(v___y_485_);
lean_dec_ref(v_P_385_);
lean_dec(v_goal_382_);
v_a_558_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_565_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_565_ == 0)
{
v___x_560_ = v___x_492_;
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_a_558_);
lean_dec(v___x_492_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_563_; 
if (v_isShared_561_ == 0)
{
v___x_563_ = v___x_560_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_a_558_);
v___x_563_ = v_reuseFailAlloc_564_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
return v___x_563_;
}
}
}
}
v___jp_566_:
{
lean_object* v___x_573_; lean_object* v_mctx_574_; lean_object* v_nextMacroScope_575_; lean_object* v_ngen_576_; lean_object* v_cache_577_; lean_object* v_zetaDeltaFVarIds_578_; lean_object* v_postponed_579_; lean_object* v_diag_580_; lean_object* v___x_582_; uint8_t v_isShared_583_; uint8_t v_isSharedCheck_616_; 
lean_dec(v___y_570_);
lean_dec(v___y_567_);
v___x_573_ = lean_st_ref_take(v___y_395_);
v_mctx_574_ = lean_ctor_get(v_a_572_, 0);
lean_inc_ref(v_mctx_574_);
v_nextMacroScope_575_ = lean_ctor_get(v_a_572_, 1);
lean_inc(v_nextMacroScope_575_);
v_ngen_576_ = lean_ctor_get(v_a_572_, 2);
lean_inc_ref(v_ngen_576_);
lean_dec_ref(v_a_572_);
v_cache_577_ = lean_ctor_get(v___x_573_, 1);
v_zetaDeltaFVarIds_578_ = lean_ctor_get(v___x_573_, 2);
v_postponed_579_ = lean_ctor_get(v___x_573_, 3);
v_diag_580_ = lean_ctor_get(v___x_573_, 4);
v_isSharedCheck_616_ = !lean_is_exclusive(v___x_573_);
if (v_isSharedCheck_616_ == 0)
{
lean_object* v_unused_617_; 
v_unused_617_ = lean_ctor_get(v___x_573_, 0);
lean_dec(v_unused_617_);
v___x_582_ = v___x_573_;
v_isShared_583_ = v_isSharedCheck_616_;
goto v_resetjp_581_;
}
else
{
lean_inc(v_diag_580_);
lean_inc(v_postponed_579_);
lean_inc(v_zetaDeltaFVarIds_578_);
lean_inc(v_cache_577_);
lean_dec(v___x_573_);
v___x_582_ = lean_box(0);
v_isShared_583_ = v_isSharedCheck_616_;
goto v_resetjp_581_;
}
v_resetjp_581_:
{
lean_object* v___x_585_; 
if (v_isShared_583_ == 0)
{
lean_ctor_set(v___x_582_, 0, v_mctx_574_);
v___x_585_ = v___x_582_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v_mctx_574_);
lean_ctor_set(v_reuseFailAlloc_615_, 1, v_cache_577_);
lean_ctor_set(v_reuseFailAlloc_615_, 2, v_zetaDeltaFVarIds_578_);
lean_ctor_set(v_reuseFailAlloc_615_, 3, v_postponed_579_);
lean_ctor_set(v_reuseFailAlloc_615_, 4, v_diag_580_);
v___x_585_ = v_reuseFailAlloc_615_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v_env_588_; lean_object* v_auxDeclNGen_589_; lean_object* v_traceState_590_; lean_object* v_cache_591_; lean_object* v_messages_592_; lean_object* v_infoState_593_; lean_object* v_snapshotTasks_594_; lean_object* v___x_596_; uint8_t v_isShared_597_; uint8_t v_isSharedCheck_612_; 
v___x_586_ = lean_st_ref_set(v___y_395_, v___x_585_);
v___x_587_ = lean_st_ref_take(v___y_397_);
v_env_588_ = lean_ctor_get(v___x_587_, 0);
v_auxDeclNGen_589_ = lean_ctor_get(v___x_587_, 3);
v_traceState_590_ = lean_ctor_get(v___x_587_, 4);
v_cache_591_ = lean_ctor_get(v___x_587_, 5);
v_messages_592_ = lean_ctor_get(v___x_587_, 6);
v_infoState_593_ = lean_ctor_get(v___x_587_, 7);
v_snapshotTasks_594_ = lean_ctor_get(v___x_587_, 8);
v_isSharedCheck_612_ = !lean_is_exclusive(v___x_587_);
if (v_isSharedCheck_612_ == 0)
{
lean_object* v_unused_613_; lean_object* v_unused_614_; 
v_unused_613_ = lean_ctor_get(v___x_587_, 2);
lean_dec(v_unused_613_);
v_unused_614_ = lean_ctor_get(v___x_587_, 1);
lean_dec(v_unused_614_);
v___x_596_ = v___x_587_;
v_isShared_597_ = v_isSharedCheck_612_;
goto v_resetjp_595_;
}
else
{
lean_inc(v_snapshotTasks_594_);
lean_inc(v_infoState_593_);
lean_inc(v_messages_592_);
lean_inc(v_cache_591_);
lean_inc(v_traceState_590_);
lean_inc(v_auxDeclNGen_589_);
lean_inc(v_env_588_);
lean_dec(v___x_587_);
v___x_596_ = lean_box(0);
v_isShared_597_ = v_isSharedCheck_612_;
goto v_resetjp_595_;
}
v_resetjp_595_:
{
lean_object* v___x_599_; 
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 2, v_ngen_576_);
lean_ctor_set(v___x_596_, 1, v_nextMacroScope_575_);
v___x_599_ = v___x_596_;
goto v_reusejp_598_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_env_588_);
lean_ctor_set(v_reuseFailAlloc_611_, 1, v_nextMacroScope_575_);
lean_ctor_set(v_reuseFailAlloc_611_, 2, v_ngen_576_);
lean_ctor_set(v_reuseFailAlloc_611_, 3, v_auxDeclNGen_589_);
lean_ctor_set(v_reuseFailAlloc_611_, 4, v_traceState_590_);
lean_ctor_set(v_reuseFailAlloc_611_, 5, v_cache_591_);
lean_ctor_set(v_reuseFailAlloc_611_, 6, v_messages_592_);
lean_ctor_set(v_reuseFailAlloc_611_, 7, v_infoState_593_);
lean_ctor_set(v_reuseFailAlloc_611_, 8, v_snapshotTasks_594_);
v___x_599_ = v_reuseFailAlloc_611_;
goto v_reusejp_598_;
}
v_reusejp_598_:
{
lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
v___x_600_ = lean_st_ref_set(v___y_397_, v___x_599_);
v___x_601_ = lean_obj_once(&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__3, &lp_mathlib_Lean_MVarId_wlog___lam__1___closed__3_once, _init_lp_mathlib_Lean_MVarId_wlog___lam__1___closed__3);
v___x_602_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___redArg(v___x_601_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
v_a_603_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_602_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_602_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_608_; 
if (v_isShared_606_ == 0)
{
v___x_608_ = v___x_605_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v_a_603_);
v___x_608_ = v_reuseFailAlloc_609_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
return v___x_608_;
}
}
}
}
}
}
}
v___jp_618_:
{
lean_object* v___x_627_; lean_object* v_mctx_628_; lean_object* v_nextMacroScope_629_; lean_object* v_ngen_630_; lean_object* v_cache_631_; lean_object* v_zetaDeltaFVarIds_632_; lean_object* v_postponed_633_; lean_object* v_diag_634_; lean_object* v___x_636_; uint8_t v_isShared_637_; uint8_t v_isSharedCheck_660_; 
v___x_627_ = lean_st_ref_take(v___y_395_);
v_mctx_628_ = lean_ctor_get(v_a_626_, 0);
lean_inc_ref(v_mctx_628_);
v_nextMacroScope_629_ = lean_ctor_get(v_a_626_, 1);
lean_inc(v_nextMacroScope_629_);
v_ngen_630_ = lean_ctor_get(v_a_626_, 2);
lean_inc_ref(v_ngen_630_);
lean_dec_ref(v_a_626_);
v_cache_631_ = lean_ctor_get(v___x_627_, 1);
v_zetaDeltaFVarIds_632_ = lean_ctor_get(v___x_627_, 2);
v_postponed_633_ = lean_ctor_get(v___x_627_, 3);
v_diag_634_ = lean_ctor_get(v___x_627_, 4);
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_660_ == 0)
{
lean_object* v_unused_661_; 
v_unused_661_ = lean_ctor_get(v___x_627_, 0);
lean_dec(v_unused_661_);
v___x_636_ = v___x_627_;
v_isShared_637_ = v_isSharedCheck_660_;
goto v_resetjp_635_;
}
else
{
lean_inc(v_diag_634_);
lean_inc(v_postponed_633_);
lean_inc(v_zetaDeltaFVarIds_632_);
lean_inc(v_cache_631_);
lean_dec(v___x_627_);
v___x_636_ = lean_box(0);
v_isShared_637_ = v_isSharedCheck_660_;
goto v_resetjp_635_;
}
v_resetjp_635_:
{
lean_object* v___x_639_; 
if (v_isShared_637_ == 0)
{
lean_ctor_set(v___x_636_, 0, v_mctx_628_);
v___x_639_ = v___x_636_;
goto v_reusejp_638_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v_mctx_628_);
lean_ctor_set(v_reuseFailAlloc_659_, 1, v_cache_631_);
lean_ctor_set(v_reuseFailAlloc_659_, 2, v_zetaDeltaFVarIds_632_);
lean_ctor_set(v_reuseFailAlloc_659_, 3, v_postponed_633_);
lean_ctor_set(v_reuseFailAlloc_659_, 4, v_diag_634_);
v___x_639_ = v_reuseFailAlloc_659_;
goto v_reusejp_638_;
}
v_reusejp_638_:
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v_env_642_; lean_object* v_auxDeclNGen_643_; lean_object* v_traceState_644_; lean_object* v_cache_645_; lean_object* v_messages_646_; lean_object* v_infoState_647_; lean_object* v_snapshotTasks_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_656_; 
v___x_640_ = lean_st_ref_set(v___y_395_, v___x_639_);
v___x_641_ = lean_st_ref_take(v___y_397_);
v_env_642_ = lean_ctor_get(v___x_641_, 0);
v_auxDeclNGen_643_ = lean_ctor_get(v___x_641_, 3);
v_traceState_644_ = lean_ctor_get(v___x_641_, 4);
v_cache_645_ = lean_ctor_get(v___x_641_, 5);
v_messages_646_ = lean_ctor_get(v___x_641_, 6);
v_infoState_647_ = lean_ctor_get(v___x_641_, 7);
v_snapshotTasks_648_ = lean_ctor_get(v___x_641_, 8);
v_isSharedCheck_656_ = !lean_is_exclusive(v___x_641_);
if (v_isSharedCheck_656_ == 0)
{
lean_object* v_unused_657_; lean_object* v_unused_658_; 
v_unused_657_ = lean_ctor_get(v___x_641_, 2);
lean_dec(v_unused_657_);
v_unused_658_ = lean_ctor_get(v___x_641_, 1);
lean_dec(v_unused_658_);
v___x_650_ = v___x_641_;
v_isShared_651_ = v_isSharedCheck_656_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_snapshotTasks_648_);
lean_inc(v_infoState_647_);
lean_inc(v_messages_646_);
lean_inc(v_cache_645_);
lean_inc(v_traceState_644_);
lean_inc(v_auxDeclNGen_643_);
lean_inc(v_env_642_);
lean_dec(v___x_641_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_656_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
lean_ctor_set(v___x_650_, 2, v_ngen_630_);
lean_ctor_set(v___x_650_, 1, v_nextMacroScope_629_);
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_655_; 
v_reuseFailAlloc_655_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_655_, 0, v_env_642_);
lean_ctor_set(v_reuseFailAlloc_655_, 1, v_nextMacroScope_629_);
lean_ctor_set(v_reuseFailAlloc_655_, 2, v_ngen_630_);
lean_ctor_set(v_reuseFailAlloc_655_, 3, v_auxDeclNGen_643_);
lean_ctor_set(v_reuseFailAlloc_655_, 4, v_traceState_644_);
lean_ctor_set(v_reuseFailAlloc_655_, 5, v_cache_645_);
lean_ctor_set(v_reuseFailAlloc_655_, 6, v_messages_646_);
lean_ctor_set(v_reuseFailAlloc_655_, 7, v_infoState_647_);
lean_ctor_set(v_reuseFailAlloc_655_, 8, v_snapshotTasks_648_);
v___x_653_ = v_reuseFailAlloc_655_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
lean_object* v___x_654_; 
v___x_654_ = lean_st_ref_set(v___y_397_, v___x_653_);
v___y_484_ = v___y_620_;
v___y_485_ = v___y_619_;
v___y_486_ = v___y_622_;
v___y_487_ = v___y_623_;
v___y_488_ = v___y_624_;
v_fst_489_ = v___y_621_;
v_snd_490_ = v_a_625_;
goto v___jp_483_;
}
}
}
}
}
v___jp_662_:
{
uint8_t v___x_669_; lean_object* v___x_670_; 
v___x_669_ = 0;
lean_inc(v_goal_382_);
v___x_670_ = lp_mathlib_Lean_Elab_Tactic_getFVarIdsAt(v_goal_382_, v___y_668_, v___x_669_, v___y_390_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_670_) == 0)
{
lean_object* v_a_671_; lean_object* v___x_672_; 
v_a_671_ = lean_ctor_get(v___x_670_, 0);
lean_inc(v_a_671_);
lean_dec_ref_known(v___x_670_, 1);
lean_inc(v_goal_382_);
v___x_672_ = l_Lean_MVarId_getDecl(v_goal_382_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
if (lean_obj_tag(v___x_672_) == 0)
{
lean_object* v_a_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v_lctx_677_; lean_object* v_mctx_678_; lean_object* v_ngen_679_; lean_object* v_quotContext_680_; lean_object* v_nextMacroScope_681_; size_t v_sz_682_; size_t v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; uint8_t v___x_687_; uint8_t v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
lean_dec_ref_known(v___x_672_, 1);
v___x_674_ = lean_st_ref_get(v___y_395_);
v___x_675_ = lean_st_ref_get(v___y_397_);
v___x_676_ = lean_st_ref_get(v___y_397_);
v_lctx_677_ = lean_ctor_get(v_a_673_, 1);
lean_inc_ref_n(v_lctx_677_, 2);
lean_dec(v_a_673_);
v_mctx_678_ = lean_ctor_get(v___x_674_, 0);
lean_inc_ref(v_mctx_678_);
lean_dec(v___x_674_);
v_ngen_679_ = lean_ctor_get(v___x_675_, 2);
lean_inc_ref(v_ngen_679_);
lean_dec(v___x_675_);
v_quotContext_680_ = lean_ctor_get(v___y_396_, 10);
v_nextMacroScope_681_ = lean_ctor_get(v___x_676_, 1);
lean_inc(v_nextMacroScope_681_);
lean_dec(v___x_676_);
v_sz_682_ = lean_array_size(v_a_671_);
v___x_683_ = ((size_t)0ULL);
v___x_684_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0(v_sz_682_, v___x_683_, v_a_671_);
v___x_685_ = lean_obj_once(&lp_mathlib_Lean_MVarId_wlog___lam__1___closed__5, &lp_mathlib_Lean_MVarId_wlog___lam__1___closed__5_once, _init_lp_mathlib_Lean_MVarId_wlog___lam__1___closed__5);
v___x_686_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_686_, 0, v_mctx_678_);
lean_ctor_set(v___x_686_, 1, v_nextMacroScope_681_);
lean_ctor_set(v___x_686_, 2, v_ngen_679_);
lean_ctor_set(v___x_686_, 3, v___x_685_);
v___x_687_ = 1;
v___x_688_ = 1;
lean_inc(v_quotContext_680_);
v___x_689_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v___x_689_, 0, v_quotContext_680_);
lean_ctor_set(v___x_689_, 1, v___x_386_);
lean_ctor_set_uint8(v___x_689_, sizeof(void*)*2, v___x_669_);
lean_ctor_set_uint8(v___x_689_, sizeof(void*)*2 + 1, v___x_688_);
v___x_690_ = l_Lean_MetavarContext_MkBinding_collectForwardDeps(v_lctx_677_, v___x_684_, v___x_687_, v___x_689_, v___x_686_);
if (lean_obj_tag(v___x_690_) == 0)
{
lean_object* v_a_691_; lean_object* v_a_692_; lean_object* v_mctx_693_; lean_object* v_nextMacroScope_694_; lean_object* v_ngen_695_; lean_object* v_cache_696_; lean_object* v___x_698_; uint8_t v_isShared_699_; uint8_t v_isSharedCheck_726_; 
v_a_691_ = lean_ctor_get(v___x_690_, 1);
lean_inc(v_a_691_);
v_a_692_ = lean_ctor_get(v___x_690_, 0);
lean_inc(v_a_692_);
lean_dec_ref_known(v___x_690_, 2);
v_mctx_693_ = lean_ctor_get(v_a_691_, 0);
v_nextMacroScope_694_ = lean_ctor_get(v_a_691_, 1);
v_ngen_695_ = lean_ctor_get(v_a_691_, 2);
v_cache_696_ = lean_ctor_get(v_a_691_, 3);
v_isSharedCheck_726_ = !lean_is_exclusive(v_a_691_);
if (v_isSharedCheck_726_ == 0)
{
v___x_698_ = v_a_691_;
v_isShared_699_ = v_isSharedCheck_726_;
goto v_resetjp_697_;
}
else
{
lean_inc(v_cache_696_);
lean_inc(v_ngen_695_);
lean_inc(v_nextMacroScope_694_);
lean_inc(v_mctx_693_);
lean_dec(v_a_691_);
v___x_698_ = lean_box(0);
v_isShared_699_ = v_isSharedCheck_726_;
goto v_resetjp_697_;
}
v_resetjp_697_:
{
size_t v_sz_700_; lean_object* v___x_701_; lean_object* v___x_702_; size_t v_sz_703_; lean_object* v___x_704_; uint8_t v___x_705_; lean_object* v___x_707_; 
v_sz_700_ = lean_array_size(v_a_692_);
v___x_701_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__6(v_sz_700_, v___x_683_, v_a_692_);
v___x_702_ = lp_mathlib_Lean_Elab_Tactic_filterOutImplementationDetails(v_lctx_677_, v___x_701_);
lean_dec_ref(v___x_701_);
v_sz_703_ = lean_array_size(v___x_702_);
lean_inc_ref(v___x_702_);
v___x_704_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_wlog_spec__0(v_sz_703_, v___x_683_, v___x_702_);
v___x_705_ = 0;
if (v_isShared_699_ == 0)
{
lean_ctor_set(v___x_698_, 3, v___x_685_);
v___x_707_ = v___x_698_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_725_; 
v_reuseFailAlloc_725_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_725_, 0, v_mctx_693_);
lean_ctor_set(v_reuseFailAlloc_725_, 1, v_nextMacroScope_694_);
lean_ctor_set(v_reuseFailAlloc_725_, 2, v_ngen_695_);
lean_ctor_set(v_reuseFailAlloc_725_, 3, v___x_685_);
v___x_707_ = v_reuseFailAlloc_725_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
lean_object* v___x_708_; 
v___x_708_ = l___private_Lean_MetavarContext_0__Lean_MetavarContext_MkBinding_mkAuxMVarType(v_lctx_677_, v___x_704_, v___x_705_, v___y_665_, v___x_669_, v___x_689_, v___x_707_);
lean_dec_ref_known(v___x_689_, 2);
lean_dec_ref(v___x_704_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_object* v_a_709_; lean_object* v_a_710_; lean_object* v_mctx_711_; lean_object* v_nextMacroScope_712_; lean_object* v_ngen_713_; lean_object* v___x_715_; uint8_t v_isShared_716_; uint8_t v_isSharedCheck_720_; 
v_a_709_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_a_709_);
v_a_710_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_a_710_);
lean_dec_ref_known(v___x_708_, 2);
v_mctx_711_ = lean_ctor_get(v_a_709_, 0);
v_nextMacroScope_712_ = lean_ctor_get(v_a_709_, 1);
v_ngen_713_ = lean_ctor_get(v_a_709_, 2);
v_isSharedCheck_720_ = !lean_is_exclusive(v_a_709_);
if (v_isSharedCheck_720_ == 0)
{
lean_object* v_unused_721_; 
v_unused_721_ = lean_ctor_get(v_a_709_, 3);
lean_dec(v_unused_721_);
v___x_715_ = v_a_709_;
v_isShared_716_ = v_isSharedCheck_720_;
goto v_resetjp_714_;
}
else
{
lean_inc(v_ngen_713_);
lean_inc(v_nextMacroScope_712_);
lean_inc(v_mctx_711_);
lean_dec(v_a_709_);
v___x_715_ = lean_box(0);
v_isShared_716_ = v_isSharedCheck_720_;
goto v_resetjp_714_;
}
v_resetjp_714_:
{
lean_object* v___x_718_; 
if (v_isShared_716_ == 0)
{
lean_ctor_set(v___x_715_, 3, v_cache_696_);
v___x_718_ = v___x_715_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_719_; 
v_reuseFailAlloc_719_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_719_, 0, v_mctx_711_);
lean_ctor_set(v_reuseFailAlloc_719_, 1, v_nextMacroScope_712_);
lean_ctor_set(v_reuseFailAlloc_719_, 2, v_ngen_713_);
lean_ctor_set(v_reuseFailAlloc_719_, 3, v_cache_696_);
v___x_718_ = v_reuseFailAlloc_719_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
v___y_619_ = v___y_663_;
v___y_620_ = v___y_664_;
v___y_621_ = v___x_702_;
v___y_622_ = v___y_666_;
v___y_623_ = v___y_667_;
v___y_624_ = v___x_669_;
v_a_625_ = v_a_710_;
v_a_626_ = v___x_718_;
goto v___jp_618_;
}
}
}
else
{
lean_dec_ref(v_cache_696_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_object* v_a_722_; lean_object* v_a_723_; 
v_a_722_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_a_722_);
v_a_723_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_a_723_);
lean_dec_ref_known(v___x_708_, 2);
v___y_619_ = v___y_663_;
v___y_620_ = v___y_664_;
v___y_621_ = v___x_702_;
v___y_622_ = v___y_666_;
v___y_623_ = v___y_667_;
v___y_624_ = v___x_669_;
v_a_625_ = v_a_722_;
v_a_626_ = v_a_723_;
goto v___jp_618_;
}
else
{
lean_object* v_a_724_; 
lean_dec_ref(v___x_702_);
lean_dec_ref(v_P_385_);
lean_dec(v_goal_382_);
v_a_724_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_a_724_);
lean_dec_ref_known(v___x_708_, 2);
v___y_567_ = v___y_663_;
v___y_568_ = v___y_664_;
v___y_569_ = v___y_666_;
v___y_570_ = v___y_667_;
v___y_571_ = v___x_669_;
v_a_572_ = v_a_724_;
goto v___jp_566_;
}
}
}
}
}
else
{
lean_object* v_a_727_; 
lean_dec_ref_known(v___x_689_, 2);
lean_dec_ref(v_lctx_677_);
lean_dec_ref(v___y_665_);
lean_dec_ref(v_P_385_);
lean_dec(v_goal_382_);
v_a_727_ = lean_ctor_get(v___x_690_, 1);
lean_inc(v_a_727_);
lean_dec_ref_known(v___x_690_, 2);
v___y_567_ = v___y_663_;
v___y_568_ = v___y_664_;
v___y_569_ = v___y_666_;
v___y_570_ = v___y_667_;
v___y_571_ = v___x_669_;
v_a_572_ = v_a_727_;
goto v___jp_566_;
}
}
else
{
lean_object* v_a_728_; lean_object* v___x_730_; uint8_t v_isShared_731_; uint8_t v_isSharedCheck_735_; 
lean_dec(v_a_671_);
lean_dec(v___y_667_);
lean_dec_ref(v___y_665_);
lean_dec(v___y_663_);
lean_dec(v___x_386_);
lean_dec_ref(v_P_385_);
lean_dec(v_goal_382_);
v_a_728_ = lean_ctor_get(v___x_672_, 0);
v_isSharedCheck_735_ = !lean_is_exclusive(v___x_672_);
if (v_isSharedCheck_735_ == 0)
{
v___x_730_ = v___x_672_;
v_isShared_731_ = v_isSharedCheck_735_;
goto v_resetjp_729_;
}
else
{
lean_inc(v_a_728_);
lean_dec(v___x_672_);
v___x_730_ = lean_box(0);
v_isShared_731_ = v_isSharedCheck_735_;
goto v_resetjp_729_;
}
v_resetjp_729_:
{
lean_object* v___x_733_; 
if (v_isShared_731_ == 0)
{
v___x_733_ = v___x_730_;
goto v_reusejp_732_;
}
else
{
lean_object* v_reuseFailAlloc_734_; 
v_reuseFailAlloc_734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_734_, 0, v_a_728_);
v___x_733_ = v_reuseFailAlloc_734_;
goto v_reusejp_732_;
}
v_reusejp_732_:
{
return v___x_733_;
}
}
}
}
else
{
lean_object* v_a_736_; lean_object* v___x_738_; uint8_t v_isShared_739_; uint8_t v_isSharedCheck_743_; 
lean_dec(v___y_667_);
lean_dec_ref(v___y_665_);
lean_dec(v___y_663_);
lean_dec(v___x_386_);
lean_dec_ref(v_P_385_);
lean_dec(v_goal_382_);
v_a_736_ = lean_ctor_get(v___x_670_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_670_);
if (v_isSharedCheck_743_ == 0)
{
v___x_738_ = v___x_670_;
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_670_);
v___x_738_ = lean_box(0);
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
v_resetjp_737_:
{
lean_object* v___x_741_; 
if (v_isShared_739_ == 0)
{
v___x_741_ = v___x_738_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_a_736_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___lam__1___boxed(lean_object** _args){
lean_object* v_goal_789_ = _args[0];
lean_object* v___x_790_ = _args[1];
lean_object* v___x_791_ = _args[2];
lean_object* v_P_792_ = _args[3];
lean_object* v___x_793_ = _args[4];
lean_object* v_xs_794_ = _args[5];
lean_object* v_h_795_ = _args[6];
lean_object* v_H_796_ = _args[7];
lean_object* v___y_797_ = _args[8];
lean_object* v___y_798_ = _args[9];
lean_object* v___y_799_ = _args[10];
lean_object* v___y_800_ = _args[11];
lean_object* v___y_801_ = _args[12];
lean_object* v___y_802_ = _args[13];
lean_object* v___y_803_ = _args[14];
lean_object* v___y_804_ = _args[15];
lean_object* v___y_805_ = _args[16];
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_mathlib_Lean_MVarId_wlog___lam__1(v_goal_789_, v___x_790_, v___x_791_, v_P_792_, v___x_793_, v_xs_794_, v_h_795_, v_H_796_, v___y_797_, v___y_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_, v___y_804_);
lean_dec(v___y_804_);
lean_dec_ref(v___y_803_);
lean_dec(v___y_802_);
lean_dec_ref(v___y_801_);
lean_dec(v___y_800_);
lean_dec_ref(v___y_799_);
lean_dec(v___y_798_);
lean_dec_ref(v___y_797_);
lean_dec(v___x_791_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog(lean_object* v_goal_810_, lean_object* v_h_811_, lean_object* v_P_812_, lean_object* v_xs_813_, lean_object* v_H_814_, lean_object* v_a_815_, lean_object* v_a_816_, lean_object* v_a_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_, lean_object* v_a_822_){
_start:
{
lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___f_827_; lean_object* v___x_828_; 
v___x_824_ = lean_box(1);
v___x_825_ = lean_box(0);
v___x_826_ = ((lean_object*)(lp_mathlib_Lean_MVarId_wlog___closed__1));
lean_inc(v_goal_810_);
v___f_827_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_wlog___lam__1___boxed), 17, 8);
lean_closure_set(v___f_827_, 0, v_goal_810_);
lean_closure_set(v___f_827_, 1, v___x_826_);
lean_closure_set(v___f_827_, 2, v___x_825_);
lean_closure_set(v___f_827_, 3, v_P_812_);
lean_closure_set(v___f_827_, 4, v___x_824_);
lean_closure_set(v___f_827_, 5, v_xs_813_);
lean_closure_set(v___f_827_, 6, v_h_811_);
lean_closure_set(v___f_827_, 7, v_H_814_);
v___x_828_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg(v_goal_810_, v___f_827_, v_a_815_, v_a_816_, v_a_817_, v_a_818_, v_a_819_, v_a_820_, v_a_821_, v_a_822_);
return v___x_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_wlog___boxed(lean_object* v_goal_829_, lean_object* v_h_830_, lean_object* v_P_831_, lean_object* v_xs_832_, lean_object* v_H_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_a_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_a_841_, lean_object* v_a_842_){
_start:
{
lean_object* v_res_843_; 
v_res_843_ = lp_mathlib_Lean_MVarId_wlog(v_goal_829_, v_h_830_, v_P_831_, v_xs_832_, v_H_833_, v_a_834_, v_a_835_, v_a_836_, v_a_837_, v_a_838_, v_a_839_, v_a_840_, v_a_841_);
lean_dec(v_a_841_);
lean_dec_ref(v_a_840_);
lean_dec(v_a_839_);
lean_dec_ref(v_a_838_);
lean_dec(v_a_837_);
lean_dec_ref(v_a_836_);
lean_dec(v_a_835_);
lean_dec_ref(v_a_834_);
return v_res_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2(lean_object* v_mvarId_844_, lean_object* v_val_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v___x_855_; 
v___x_855_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___redArg(v_mvarId_844_, v_val_845_, v___y_851_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2___boxed(lean_object* v_mvarId_856_, lean_object* v_val_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_){
_start:
{
lean_object* v_res_867_; 
v_res_867_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_wlog_spec__2(v_mvarId_856_, v_val_857_, v___y_858_, v___y_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_, v___y_865_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
lean_dec(v___y_863_);
lean_dec_ref(v___y_862_);
lean_dec(v___y_861_);
lean_dec_ref(v___y_860_);
lean_dec(v___y_859_);
lean_dec_ref(v___y_858_);
return v_res_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3(lean_object* v_as_868_, size_t v_i_869_, size_t v_stop_870_, lean_object* v_b_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_){
_start:
{
lean_object* v___x_881_; 
v___x_881_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___redArg(v_as_868_, v_i_869_, v_stop_870_, v_b_871_, v___y_876_, v___y_878_, v___y_879_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3___boxed(lean_object* v_as_882_, lean_object* v_i_883_, lean_object* v_stop_884_, lean_object* v_b_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_){
_start:
{
size_t v_i_boxed_895_; size_t v_stop_boxed_896_; lean_object* v_res_897_; 
v_i_boxed_895_ = lean_unbox_usize(v_i_883_);
lean_dec(v_i_883_);
v_stop_boxed_896_ = lean_unbox_usize(v_stop_884_);
lean_dec(v_stop_884_);
v_res_897_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_MVarId_wlog_spec__3(v_as_882_, v_i_boxed_895_, v_stop_boxed_896_, v_b_885_, v___y_886_, v___y_887_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_);
lean_dec(v___y_893_);
lean_dec_ref(v___y_892_);
lean_dec(v___y_891_);
lean_dec_ref(v___y_890_);
lean_dec(v___y_889_);
lean_dec_ref(v___y_888_);
lean_dec(v___y_887_);
lean_dec_ref(v___y_886_);
lean_dec_ref(v_as_882_);
return v_res_897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5(lean_object* v_00_u03b1_898_, lean_object* v_msg_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_){
_start:
{
lean_object* v___x_905_; 
v___x_905_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___redArg(v_msg_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_);
return v___x_905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5___boxed(lean_object* v_00_u03b1_906_, lean_object* v_msg_907_, lean_object* v___y_908_, lean_object* v___y_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_){
_start:
{
lean_object* v_res_913_; 
v_res_913_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_wlog_spec__5(v_00_u03b1_906_, v_msg_907_, v___y_908_, v___y_909_, v___y_910_, v___y_911_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
lean_dec(v___y_909_);
lean_dec_ref(v___y_908_);
return v_res_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__0(lean_object* v_snd_919_, lean_object* v_val_920_, uint8_t v___x_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_){
_start:
{
lean_object* v___x_931_; 
v___x_931_ = l_Lean_FVarId_getUserName___redArg(v_snd_919_, v___y_926_, v___y_928_, v___y_929_);
if (lean_obj_tag(v___x_931_) == 0)
{
lean_object* v_a_932_; uint8_t v___x_933_; lean_object* v___x_934_; 
v_a_932_ = lean_ctor_get(v___x_931_, 0);
lean_inc(v_a_932_);
lean_dec_ref_known(v___x_931_, 1);
v___x_933_ = 0;
v___x_934_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v_val_920_, v___x_933_, v___x_921_, v___y_922_, v___y_928_, v___y_929_);
if (lean_obj_tag(v___x_934_) == 0)
{
lean_object* v_a_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; uint8_t v___x_943_; uint8_t v___x_944_; lean_object* v___x_945_; 
v_a_935_ = lean_ctor_get(v___x_934_, 0);
lean_inc(v_a_935_);
lean_dec_ref_known(v___x_934_, 1);
v___x_936_ = l_Lean_mkIdent(v_a_932_);
v___x_937_ = lean_box(0);
v___x_938_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___closed__2));
v___x_939_ = lean_unsigned_to_nat(1u);
v___x_940_ = lean_mk_empty_array_with_capacity(v___x_939_);
v___x_941_ = lean_array_push(v___x_940_, v___x_936_);
v___x_942_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_942_, 0, v___x_941_);
lean_ctor_set_uint8(v___x_942_, sizeof(void*)*1, v___x_933_);
v___x_943_ = 2;
v___x_944_ = lean_unbox(v_a_935_);
lean_dec(v_a_935_);
v___x_945_ = lp_mathlib_Mathlib_Tactic_Push_push(v___x_944_, v___x_937_, v___x_938_, v___x_942_, v___x_943_, v___y_922_, v___y_923_, v___y_924_, v___y_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_);
lean_dec_ref_known(v___x_942_, 1);
return v___x_945_;
}
else
{
lean_object* v_a_946_; lean_object* v___x_948_; uint8_t v_isShared_949_; uint8_t v_isSharedCheck_953_; 
lean_dec(v_a_932_);
v_a_946_ = lean_ctor_get(v___x_934_, 0);
v_isSharedCheck_953_ = !lean_is_exclusive(v___x_934_);
if (v_isSharedCheck_953_ == 0)
{
v___x_948_ = v___x_934_;
v_isShared_949_ = v_isSharedCheck_953_;
goto v_resetjp_947_;
}
else
{
lean_inc(v_a_946_);
lean_dec(v___x_934_);
v___x_948_ = lean_box(0);
v_isShared_949_ = v_isSharedCheck_953_;
goto v_resetjp_947_;
}
v_resetjp_947_:
{
lean_object* v___x_951_; 
if (v_isShared_949_ == 0)
{
v___x_951_ = v___x_948_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v_a_946_);
v___x_951_ = v_reuseFailAlloc_952_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
return v___x_951_;
}
}
}
}
else
{
lean_object* v_a_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_961_; 
lean_dec(v_val_920_);
v_a_954_ = lean_ctor_get(v___x_931_, 0);
v_isSharedCheck_961_ = !lean_is_exclusive(v___x_931_);
if (v_isSharedCheck_961_ == 0)
{
v___x_956_ = v___x_931_;
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_a_954_);
lean_dec(v___x_931_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v___x_959_; 
if (v_isShared_957_ == 0)
{
v___x_959_ = v___x_956_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v_a_954_);
v___x_959_ = v_reuseFailAlloc_960_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
return v___x_959_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___boxed(lean_object* v_snd_962_, lean_object* v_val_963_, lean_object* v___x_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_){
_start:
{
uint8_t v___x_1549__boxed_974_; lean_object* v_res_975_; 
v___x_1549__boxed_974_ = lean_unbox(v___x_964_);
v_res_975_ = lp_mathlib_Mathlib_Tactic_wlogCore___lam__0(v_snd_962_, v_val_963_, v___x_1549__boxed_974_, v___y_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
lean_dec(v___y_972_);
lean_dec_ref(v___y_971_);
lean_dec(v___y_970_);
lean_dec_ref(v___y_969_);
lean_dec(v___y_968_);
lean_dec_ref(v___y_967_);
lean_dec(v___y_966_);
lean_dec_ref(v___y_965_);
return v_res_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__1(lean_object* v_P_976_, lean_object* v___y_977_, lean_object* v_xs_978_, lean_object* v___y_979_, lean_object* v_pushConfig_980_, uint8_t v___x_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
lean_object* v___x_991_; 
v___x_991_ = l_Lean_Elab_Term_elabType(v_P_976_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
if (lean_obj_tag(v___x_991_) == 0)
{
lean_object* v_a_992_; lean_object* v___x_993_; 
v_a_992_ = lean_ctor_get(v___x_991_, 0);
lean_inc(v_a_992_);
lean_dec_ref_known(v___x_991_, 1);
v___x_993_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_983_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
if (lean_obj_tag(v___x_993_) == 0)
{
lean_object* v_a_994_; lean_object* v___x_995_; 
v_a_994_ = lean_ctor_get(v___x_993_, 0);
lean_inc(v_a_994_);
lean_dec_ref_known(v___x_993_, 1);
v___x_995_ = lp_mathlib_Lean_MVarId_wlog(v_a_994_, v___y_977_, v_a_992_, v_xs_978_, v___y_979_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
if (lean_obj_tag(v___x_995_) == 0)
{
lean_object* v_a_996_; lean_object* v_reductionGoal_997_; lean_object* v_reductionFVarIds_998_; lean_object* v_hypothesisGoal_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v_a_996_ = lean_ctor_get(v___x_995_, 0);
lean_inc(v_a_996_);
lean_dec_ref_known(v___x_995_, 1);
v_reductionGoal_997_ = lean_ctor_get(v_a_996_, 0);
lean_inc_n(v_reductionGoal_997_, 2);
v_reductionFVarIds_998_ = lean_ctor_get(v_a_996_, 1);
lean_inc_ref(v_reductionFVarIds_998_);
v_hypothesisGoal_999_ = lean_ctor_get(v_a_996_, 2);
lean_inc(v_hypothesisGoal_999_);
lean_dec(v_a_996_);
v___x_1000_ = lean_box(0);
v___x_1001_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1001_, 0, v_hypothesisGoal_999_);
lean_ctor_set(v___x_1001_, 1, v___x_1000_);
v___x_1002_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1002_, 0, v_reductionGoal_997_);
lean_ctor_set(v___x_1002_, 1, v___x_1001_);
v___x_1003_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1002_, v___y_983_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
if (lean_obj_tag(v___x_1003_) == 0)
{
lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1016_; 
v_isSharedCheck_1016_ = !lean_is_exclusive(v___x_1003_);
if (v_isSharedCheck_1016_ == 0)
{
lean_object* v_unused_1017_; 
v_unused_1017_ = lean_ctor_get(v___x_1003_, 0);
lean_dec(v_unused_1017_);
v___x_1005_ = v___x_1003_;
v_isShared_1006_ = v_isSharedCheck_1016_;
goto v_resetjp_1004_;
}
else
{
lean_dec(v___x_1003_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1016_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
if (lean_obj_tag(v_pushConfig_980_) == 1)
{
lean_object* v_val_1007_; lean_object* v_snd_1008_; lean_object* v___x_1009_; lean_object* v___f_1010_; lean_object* v___x_1011_; 
lean_del_object(v___x_1005_);
v_val_1007_ = lean_ctor_get(v_pushConfig_980_, 0);
lean_inc(v_val_1007_);
lean_dec_ref_known(v_pushConfig_980_, 1);
v_snd_1008_ = lean_ctor_get(v_reductionFVarIds_998_, 1);
lean_inc(v_snd_1008_);
lean_dec_ref(v_reductionFVarIds_998_);
v___x_1009_ = lean_box(v___x_981_);
v___f_1010_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_wlogCore___lam__0___boxed), 12, 3);
lean_closure_set(v___f_1010_, 0, v_snd_1008_);
lean_closure_set(v___f_1010_, 1, v_val_1007_);
lean_closure_set(v___f_1010_, 2, v___x_1009_);
v___x_1011_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_wlog_spec__4___redArg(v_reductionGoal_997_, v___f_1010_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
return v___x_1011_;
}
else
{
lean_object* v___x_1012_; lean_object* v___x_1014_; 
lean_dec_ref(v_reductionFVarIds_998_);
lean_dec(v_reductionGoal_997_);
lean_dec(v_pushConfig_980_);
v___x_1012_ = lean_box(0);
if (v_isShared_1006_ == 0)
{
lean_ctor_set(v___x_1005_, 0, v___x_1012_);
v___x_1014_ = v___x_1005_;
goto v_reusejp_1013_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1015_, 0, v___x_1012_);
v___x_1014_ = v_reuseFailAlloc_1015_;
goto v_reusejp_1013_;
}
v_reusejp_1013_:
{
return v___x_1014_;
}
}
}
}
else
{
lean_dec_ref(v_reductionFVarIds_998_);
lean_dec(v_reductionGoal_997_);
lean_dec(v_pushConfig_980_);
return v___x_1003_;
}
}
else
{
lean_object* v_a_1018_; lean_object* v___x_1020_; uint8_t v_isShared_1021_; uint8_t v_isSharedCheck_1025_; 
lean_dec(v_pushConfig_980_);
v_a_1018_ = lean_ctor_get(v___x_995_, 0);
v_isSharedCheck_1025_ = !lean_is_exclusive(v___x_995_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1020_ = v___x_995_;
v_isShared_1021_ = v_isSharedCheck_1025_;
goto v_resetjp_1019_;
}
else
{
lean_inc(v_a_1018_);
lean_dec(v___x_995_);
v___x_1020_ = lean_box(0);
v_isShared_1021_ = v_isSharedCheck_1025_;
goto v_resetjp_1019_;
}
v_resetjp_1019_:
{
lean_object* v___x_1023_; 
if (v_isShared_1021_ == 0)
{
v___x_1023_ = v___x_1020_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v_a_1018_);
v___x_1023_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
return v___x_1023_;
}
}
}
}
else
{
lean_object* v_a_1026_; lean_object* v___x_1028_; uint8_t v_isShared_1029_; uint8_t v_isSharedCheck_1033_; 
lean_dec(v_a_992_);
lean_dec(v_pushConfig_980_);
lean_dec(v___y_979_);
lean_dec(v_xs_978_);
lean_dec(v___y_977_);
v_a_1026_ = lean_ctor_get(v___x_993_, 0);
v_isSharedCheck_1033_ = !lean_is_exclusive(v___x_993_);
if (v_isSharedCheck_1033_ == 0)
{
v___x_1028_ = v___x_993_;
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
else
{
lean_inc(v_a_1026_);
lean_dec(v___x_993_);
v___x_1028_ = lean_box(0);
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
v_resetjp_1027_:
{
lean_object* v___x_1031_; 
if (v_isShared_1029_ == 0)
{
v___x_1031_ = v___x_1028_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v_a_1026_);
v___x_1031_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
return v___x_1031_;
}
}
}
}
else
{
lean_object* v_a_1034_; lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1041_; 
lean_dec(v_pushConfig_980_);
lean_dec(v___y_979_);
lean_dec(v_xs_978_);
lean_dec(v___y_977_);
v_a_1034_ = lean_ctor_get(v___x_991_, 0);
v_isSharedCheck_1041_ = !lean_is_exclusive(v___x_991_);
if (v_isSharedCheck_1041_ == 0)
{
v___x_1036_ = v___x_991_;
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
else
{
lean_inc(v_a_1034_);
lean_dec(v___x_991_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1041_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
lean_object* v___x_1039_; 
if (v_isShared_1037_ == 0)
{
v___x_1039_ = v___x_1036_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1040_; 
v_reuseFailAlloc_1040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1040_, 0, v_a_1034_);
v___x_1039_ = v_reuseFailAlloc_1040_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
return v___x_1039_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___lam__1___boxed(lean_object* v_P_1042_, lean_object* v___y_1043_, lean_object* v_xs_1044_, lean_object* v___y_1045_, lean_object* v_pushConfig_1046_, lean_object* v___x_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_){
_start:
{
uint8_t v___x_1650__boxed_1057_; lean_object* v_res_1058_; 
v___x_1650__boxed_1057_ = lean_unbox(v___x_1047_);
v_res_1058_ = lp_mathlib_Mathlib_Tactic_wlogCore___lam__1(v_P_1042_, v___y_1043_, v_xs_1044_, v___y_1045_, v_pushConfig_1046_, v___x_1650__boxed_1057_, v___y_1048_, v___y_1049_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_);
lean_dec(v___y_1055_);
lean_dec_ref(v___y_1054_);
lean_dec(v___y_1053_);
lean_dec_ref(v___y_1052_);
lean_dec(v___y_1051_);
lean_dec_ref(v___y_1050_);
lean_dec(v___y_1049_);
lean_dec_ref(v___y_1048_);
return v_res_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore(lean_object* v_h_1067_, lean_object* v_P_1068_, lean_object* v_xs_1069_, lean_object* v_H_1070_, lean_object* v_pushConfig_1071_, lean_object* v_a_1072_, lean_object* v_a_1073_, lean_object* v_a_1074_, lean_object* v_a_1075_, lean_object* v_a_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_){
_start:
{
uint8_t v___y_1082_; lean_object* v___y_1083_; lean_object* v___y_1084_; lean_object* v___y_1089_; 
if (lean_obj_tag(v_H_1070_) == 0)
{
lean_object* v___x_1101_; 
v___x_1101_ = lean_box(0);
v___y_1089_ = v___x_1101_;
goto v___jp_1088_;
}
else
{
lean_object* v_val_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1110_; 
v_val_1102_ = lean_ctor_get(v_H_1070_, 0);
v_isSharedCheck_1110_ = !lean_is_exclusive(v_H_1070_);
if (v_isSharedCheck_1110_ == 0)
{
v___x_1104_ = v_H_1070_;
v_isShared_1105_ = v_isSharedCheck_1110_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_val_1102_);
lean_dec(v_H_1070_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1110_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
lean_object* v___x_1106_; lean_object* v___x_1108_; 
v___x_1106_ = l_Lean_TSyntax_getId(v_val_1102_);
lean_dec(v_val_1102_);
if (v_isShared_1105_ == 0)
{
lean_ctor_set(v___x_1104_, 0, v___x_1106_);
v___x_1108_ = v___x_1104_;
goto v_reusejp_1107_;
}
else
{
lean_object* v_reuseFailAlloc_1109_; 
v_reuseFailAlloc_1109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1109_, 0, v___x_1106_);
v___x_1108_ = v_reuseFailAlloc_1109_;
goto v_reusejp_1107_;
}
v_reusejp_1107_:
{
v___y_1089_ = v___x_1108_;
goto v___jp_1088_;
}
}
}
v___jp_1081_:
{
lean_object* v___x_1085_; lean_object* v___f_1086_; lean_object* v___x_1087_; 
v___x_1085_ = lean_box(v___y_1082_);
v___f_1086_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_wlogCore___lam__1___boxed), 15, 6);
lean_closure_set(v___f_1086_, 0, v_P_1068_);
lean_closure_set(v___f_1086_, 1, v___y_1084_);
lean_closure_set(v___f_1086_, 2, v_xs_1069_);
lean_closure_set(v___f_1086_, 3, v___y_1083_);
lean_closure_set(v___f_1086_, 4, v_pushConfig_1071_);
lean_closure_set(v___f_1086_, 5, v___x_1085_);
v___x_1087_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1086_, v_a_1072_, v_a_1073_, v_a_1074_, v_a_1075_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
return v___x_1087_;
}
v___jp_1088_:
{
lean_object* v___x_1090_; uint8_t v___x_1091_; uint8_t v___x_1092_; 
v___x_1090_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlogCore___closed__2));
lean_inc(v_h_1067_);
v___x_1091_ = l_Lean_Syntax_isOfKind(v_h_1067_, v___x_1090_);
v___x_1092_ = 1;
if (v___x_1091_ == 0)
{
lean_object* v___x_1093_; 
lean_dec(v_h_1067_);
v___x_1093_ = lean_box(0);
v___y_1082_ = v___x_1092_;
v___y_1083_ = v___y_1089_;
v___y_1084_ = v___x_1093_;
goto v___jp_1081_;
}
else
{
lean_object* v___x_1094_; lean_object* v_h_1095_; lean_object* v___x_1096_; uint8_t v___x_1097_; 
v___x_1094_ = lean_unsigned_to_nat(0u);
v_h_1095_ = l_Lean_Syntax_getArg(v_h_1067_, v___x_1094_);
lean_dec(v_h_1067_);
v___x_1096_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlogCore___closed__4));
lean_inc(v_h_1095_);
v___x_1097_ = l_Lean_Syntax_isOfKind(v_h_1095_, v___x_1096_);
if (v___x_1097_ == 0)
{
lean_object* v___x_1098_; 
lean_dec(v_h_1095_);
v___x_1098_ = lean_box(0);
v___y_1082_ = v___x_1092_;
v___y_1083_ = v___y_1089_;
v___y_1084_ = v___x_1098_;
goto v___jp_1081_;
}
else
{
lean_object* v___x_1099_; lean_object* v___x_1100_; 
v___x_1099_ = l_Lean_TSyntax_getId(v_h_1095_);
lean_dec(v_h_1095_);
v___x_1100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1100_, 0, v___x_1099_);
v___y_1082_ = v___x_1092_;
v___y_1083_ = v___y_1089_;
v___y_1084_ = v___x_1100_;
goto v___jp_1081_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_wlogCore___boxed(lean_object* v_h_1111_, lean_object* v_P_1112_, lean_object* v_xs_1113_, lean_object* v_H_1114_, lean_object* v_pushConfig_1115_, lean_object* v_a_1116_, lean_object* v_a_1117_, lean_object* v_a_1118_, lean_object* v_a_1119_, lean_object* v_a_1120_, lean_object* v_a_1121_, lean_object* v_a_1122_, lean_object* v_a_1123_, lean_object* v_a_1124_){
_start:
{
lean_object* v_res_1125_; 
v_res_1125_ = lp_mathlib_Mathlib_Tactic_wlogCore(v_h_1111_, v_P_1112_, v_xs_1113_, v_H_1114_, v_pushConfig_1115_, v_a_1116_, v_a_1117_, v_a_1118_, v_a_1119_, v_a_1120_, v_a_1121_, v_a_1122_, v_a_1123_);
lean_dec(v_a_1123_);
lean_dec_ref(v_a_1122_);
lean_dec(v_a_1121_);
lean_dec_ref(v_a_1120_);
lean_dec(v_a_1119_);
lean_dec_ref(v_a_1118_);
lean_dec(v_a_1117_);
lean_dec_ref(v_a_1116_);
return v_res_1125_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__7(void){
_start:
{
lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; 
v___x_1139_ = l_Lean_binderIdent;
v___x_1140_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__6));
v___x_1141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1142_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1142_, 0, v___x_1141_);
lean_ctor_set(v___x_1142_, 1, v___x_1140_);
lean_ctor_set(v___x_1142_, 2, v___x_1139_);
return v___x_1142_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__10(void){
_start:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; 
v___x_1146_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__9));
v___x_1147_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__7, &lp_mathlib_Mathlib_Tactic_wlog___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__7);
v___x_1148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1149_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1149_, 0, v___x_1148_);
lean_ctor_set(v___x_1149_, 1, v___x_1147_);
lean_ctor_set(v___x_1149_, 2, v___x_1146_);
return v___x_1149_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__14(void){
_start:
{
lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; 
v___x_1156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__13));
v___x_1157_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__10, &lp_mathlib_Mathlib_Tactic_wlog___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__10);
v___x_1158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1159_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1159_, 0, v___x_1158_);
lean_ctor_set(v___x_1159_, 1, v___x_1157_);
lean_ctor_set(v___x_1159_, 2, v___x_1156_);
return v___x_1159_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__33(void){
_start:
{
lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1199_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__32));
v___x_1200_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__14, &lp_mathlib_Mathlib_Tactic_wlog___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__14);
v___x_1201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1202_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1201_);
lean_ctor_set(v___x_1202_, 1, v___x_1200_);
lean_ctor_set(v___x_1202_, 2, v___x_1199_);
return v___x_1202_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__36(void){
_start:
{
lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; 
v___x_1206_ = l_Lean_binderIdent;
v___x_1207_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__35));
v___x_1208_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1209_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1209_, 0, v___x_1208_);
lean_ctor_set(v___x_1209_, 1, v___x_1207_);
lean_ctor_set(v___x_1209_, 2, v___x_1206_);
return v___x_1209_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__37(void){
_start:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; 
v___x_1210_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__36, &lp_mathlib_Mathlib_Tactic_wlog___closed__36_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__36);
v___x_1211_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__16));
v___x_1212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1212_, 0, v___x_1211_);
lean_ctor_set(v___x_1212_, 1, v___x_1210_);
return v___x_1212_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__38(void){
_start:
{
lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; 
v___x_1213_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__37, &lp_mathlib_Mathlib_Tactic_wlog___closed__37_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__37);
v___x_1214_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__33, &lp_mathlib_Mathlib_Tactic_wlog___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__33);
v___x_1215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1216_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1216_, 0, v___x_1215_);
lean_ctor_set(v___x_1216_, 1, v___x_1214_);
lean_ctor_set(v___x_1216_, 2, v___x_1213_);
return v___x_1216_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog___closed__39(void){
_start:
{
lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1217_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__38, &lp_mathlib_Mathlib_Tactic_wlog___closed__38_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__38);
v___x_1218_ = lean_unsigned_to_nat(1022u);
v___x_1219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__2));
v___x_1220_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1220_, 0, v___x_1219_);
lean_ctor_set(v___x_1220_, 1, v___x_1218_);
lean_ctor_set(v___x_1220_, 2, v___x_1217_);
return v___x_1220_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog(void){
_start:
{
lean_object* v___x_1221_; 
v___x_1221_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__39, &lp_mathlib_Mathlib_Tactic_wlog___closed__39_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__39);
return v___x_1221_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; 
v___x_1222_ = lean_box(0);
v___x_1223_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1224_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1224_, 0, v___x_1223_);
lean_ctor_set(v___x_1224_, 1, v___x_1222_);
return v___x_1224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1226_; lean_object* v___x_1227_; 
v___x_1226_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___closed__0);
v___x_1227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1227_, 0, v___x_1226_);
return v___x_1227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg___boxed(lean_object* v___y_1228_){
_start:
{
lean_object* v_res_1229_; 
v_res_1229_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v_res_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0(lean_object* v_00_u03b1_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
lean_object* v___x_1240_; 
v___x_1240_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___boxed(lean_object* v_00_u03b1_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_){
_start:
{
lean_object* v_res_1251_; 
v_res_1251_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0(v_00_u03b1_1241_, v___y_1242_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_);
lean_dec(v___y_1249_);
lean_dec_ref(v___y_1248_);
lean_dec(v___y_1247_);
lean_dec_ref(v___y_1246_);
lean_dec(v___y_1245_);
lean_dec_ref(v___y_1244_);
lean_dec(v___y_1243_);
lean_dec_ref(v___y_1242_);
return v_res_1251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1(lean_object* v_x_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_, lean_object* v_a_1255_, lean_object* v_a_1256_, lean_object* v_a_1257_, lean_object* v_a_1258_, lean_object* v_a_1259_, lean_object* v_a_1260_){
_start:
{
lean_object* v___x_1262_; uint8_t v___x_1263_; 
v___x_1262_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__2));
lean_inc(v_x_1252_);
v___x_1263_ = l_Lean_Syntax_isOfKind(v_x_1252_, v___x_1262_);
if (v___x_1263_ == 0)
{
lean_object* v___x_1264_; 
lean_dec(v_x_1252_);
v___x_1264_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1264_;
}
else
{
lean_object* v___x_1265_; lean_object* v_h_1266_; lean_object* v___x_1267_; uint8_t v___x_1268_; 
v___x_1265_ = lean_unsigned_to_nat(1u);
v_h_1266_ = l_Lean_Syntax_getArg(v_x_1252_, v___x_1265_);
v___x_1267_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlogCore___closed__2));
lean_inc(v_h_1266_);
v___x_1268_ = l_Lean_Syntax_isOfKind(v_h_1266_, v___x_1267_);
if (v___x_1268_ == 0)
{
lean_object* v___x_1269_; 
lean_dec(v_h_1266_);
lean_dec(v_x_1252_);
v___x_1269_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1269_;
}
else
{
lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v_P_1273_; lean_object* v___y_1275_; lean_object* v_H_1276_; lean_object* v___y_1277_; lean_object* v___y_1278_; lean_object* v___y_1279_; lean_object* v___y_1280_; lean_object* v___y_1281_; lean_object* v___y_1282_; lean_object* v___y_1283_; lean_object* v___y_1284_; lean_object* v_xs_1288_; lean_object* v___y_1289_; lean_object* v___y_1290_; lean_object* v___y_1291_; lean_object* v___y_1292_; lean_object* v___y_1293_; lean_object* v___y_1294_; lean_object* v___y_1295_; lean_object* v___y_1296_; lean_object* v___x_1311_; lean_object* v___x_1312_; uint8_t v___x_1313_; 
v___x_1270_ = lean_unsigned_to_nat(0u);
v___x_1271_ = lean_unsigned_to_nat(2u);
v___x_1272_ = lean_unsigned_to_nat(3u);
v_P_1273_ = l_Lean_Syntax_getArg(v_x_1252_, v___x_1272_);
v___x_1311_ = lean_unsigned_to_nat(4u);
v___x_1312_ = l_Lean_Syntax_getArg(v_x_1252_, v___x_1311_);
v___x_1313_ = l_Lean_Syntax_isNone(v___x_1312_);
if (v___x_1313_ == 0)
{
uint8_t v___x_1314_; 
lean_inc(v___x_1312_);
v___x_1314_ = l_Lean_Syntax_matchesNull(v___x_1312_, v___x_1271_);
if (v___x_1314_ == 0)
{
lean_object* v___x_1315_; 
lean_dec(v___x_1312_);
lean_dec(v_P_1273_);
lean_dec(v_h_1266_);
lean_dec(v_x_1252_);
v___x_1315_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1315_;
}
else
{
lean_object* v___x_1316_; lean_object* v_xs_1317_; lean_object* v___x_1318_; 
v___x_1316_ = l_Lean_Syntax_getArg(v___x_1312_, v___x_1265_);
lean_dec(v___x_1312_);
v_xs_1317_ = l_Lean_Syntax_getArgs(v___x_1316_);
lean_dec(v___x_1316_);
v___x_1318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1318_, 0, v_xs_1317_);
v_xs_1288_ = v___x_1318_;
v___y_1289_ = v_a_1253_;
v___y_1290_ = v_a_1254_;
v___y_1291_ = v_a_1255_;
v___y_1292_ = v_a_1256_;
v___y_1293_ = v_a_1257_;
v___y_1294_ = v_a_1258_;
v___y_1295_ = v_a_1259_;
v___y_1296_ = v_a_1260_;
goto v___jp_1287_;
}
}
else
{
lean_object* v___x_1319_; 
lean_dec(v___x_1312_);
v___x_1319_ = lean_box(0);
v_xs_1288_ = v___x_1319_;
v___y_1289_ = v_a_1253_;
v___y_1290_ = v_a_1254_;
v___y_1291_ = v_a_1255_;
v___y_1292_ = v_a_1256_;
v___y_1293_ = v_a_1257_;
v___y_1294_ = v_a_1258_;
v___y_1295_ = v_a_1259_;
v___y_1296_ = v_a_1260_;
goto v___jp_1287_;
}
v___jp_1274_:
{
lean_object* v___x_1285_; lean_object* v___x_1286_; 
v___x_1285_ = lean_box(0);
v___x_1286_ = lp_mathlib_Mathlib_Tactic_wlogCore(v_h_1266_, v_P_1273_, v___y_1275_, v_H_1276_, v___x_1285_, v___y_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_);
return v___x_1286_;
}
v___jp_1287_:
{
lean_object* v___x_1297_; lean_object* v___x_1298_; uint8_t v___x_1299_; 
v___x_1297_ = lean_unsigned_to_nat(5u);
v___x_1298_ = l_Lean_Syntax_getArg(v_x_1252_, v___x_1297_);
lean_dec(v_x_1252_);
v___x_1299_ = l_Lean_Syntax_isNone(v___x_1298_);
if (v___x_1299_ == 0)
{
uint8_t v___x_1300_; 
lean_inc(v___x_1298_);
v___x_1300_ = l_Lean_Syntax_matchesNull(v___x_1298_, v___x_1271_);
if (v___x_1300_ == 0)
{
lean_object* v___x_1301_; 
lean_dec(v___x_1298_);
lean_dec(v_xs_1288_);
lean_dec(v_P_1273_);
lean_dec(v_h_1266_);
v___x_1301_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1301_;
}
else
{
lean_object* v___x_1302_; uint8_t v___x_1303_; 
v___x_1302_ = l_Lean_Syntax_getArg(v___x_1298_, v___x_1265_);
lean_dec(v___x_1298_);
lean_inc(v___x_1302_);
v___x_1303_ = l_Lean_Syntax_isOfKind(v___x_1302_, v___x_1267_);
if (v___x_1303_ == 0)
{
lean_object* v___x_1304_; 
lean_dec(v___x_1302_);
lean_dec(v_xs_1288_);
lean_dec(v_P_1273_);
lean_dec(v_h_1266_);
v___x_1304_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1304_;
}
else
{
lean_object* v_H_1305_; lean_object* v___x_1306_; uint8_t v___x_1307_; 
v_H_1305_ = l_Lean_Syntax_getArg(v___x_1302_, v___x_1270_);
lean_dec(v___x_1302_);
v___x_1306_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlogCore___closed__4));
lean_inc(v_H_1305_);
v___x_1307_ = l_Lean_Syntax_isOfKind(v_H_1305_, v___x_1306_);
if (v___x_1307_ == 0)
{
lean_object* v___x_1308_; 
lean_dec(v_H_1305_);
lean_dec(v_xs_1288_);
lean_dec(v_P_1273_);
lean_dec(v_h_1266_);
v___x_1308_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1308_;
}
else
{
lean_object* v___x_1309_; 
v___x_1309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1309_, 0, v_H_1305_);
v___y_1275_ = v_xs_1288_;
v_H_1276_ = v___x_1309_;
v___y_1277_ = v___y_1289_;
v___y_1278_ = v___y_1290_;
v___y_1279_ = v___y_1291_;
v___y_1280_ = v___y_1292_;
v___y_1281_ = v___y_1293_;
v___y_1282_ = v___y_1294_;
v___y_1283_ = v___y_1295_;
v___y_1284_ = v___y_1296_;
goto v___jp_1274_;
}
}
}
}
else
{
lean_object* v___x_1310_; 
lean_dec(v___x_1298_);
v___x_1310_ = lean_box(0);
v___y_1275_ = v_xs_1288_;
v_H_1276_ = v___x_1310_;
v___y_1277_ = v___y_1289_;
v___y_1278_ = v___y_1290_;
v___y_1279_ = v___y_1291_;
v___y_1280_ = v___y_1292_;
v___y_1281_ = v___y_1293_;
v___y_1282_ = v___y_1294_;
v___y_1283_ = v___y_1295_;
v___y_1284_ = v___y_1296_;
goto v___jp_1274_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1___boxed(lean_object* v_x_1320_, lean_object* v_a_1321_, lean_object* v_a_1322_, lean_object* v_a_1323_, lean_object* v_a_1324_, lean_object* v_a_1325_, lean_object* v_a_1326_, lean_object* v_a_1327_, lean_object* v_a_1328_, lean_object* v_a_1329_){
_start:
{
lean_object* v_res_1330_; 
v_res_1330_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1(v_x_1320_, v_a_1321_, v_a_1322_, v_a_1323_, v_a_1324_, v_a_1325_, v_a_1326_, v_a_1327_, v_a_1328_);
lean_dec(v_a_1328_);
lean_dec_ref(v_a_1327_);
lean_dec(v_a_1326_);
lean_dec_ref(v_a_1325_);
lean_dec(v_a_1324_);
lean_dec_ref(v_a_1323_);
lean_dec(v_a_1322_);
lean_dec_ref(v_a_1321_);
return v_res_1330_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__4(void){
_start:
{
lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; 
v___x_1340_ = l_Lean_Parser_Tactic_optConfig;
v___x_1341_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog_x21___closed__3));
v___x_1342_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1343_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1343_, 0, v___x_1342_);
lean_ctor_set(v___x_1343_, 1, v___x_1341_);
lean_ctor_set(v___x_1343_, 2, v___x_1340_);
return v___x_1343_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__5(void){
_start:
{
lean_object* v___x_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; 
v___x_1344_ = l_Lean_binderIdent;
v___x_1345_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__4, &lp_mathlib_Mathlib_Tactic_wlog_x21___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__4);
v___x_1346_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1347_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1347_, 0, v___x_1346_);
lean_ctor_set(v___x_1347_, 1, v___x_1345_);
lean_ctor_set(v___x_1347_, 2, v___x_1344_);
return v___x_1347_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__6(void){
_start:
{
lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; 
v___x_1348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__9));
v___x_1349_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__5, &lp_mathlib_Mathlib_Tactic_wlog_x21___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__5);
v___x_1350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1351_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1351_, 0, v___x_1350_);
lean_ctor_set(v___x_1351_, 1, v___x_1349_);
lean_ctor_set(v___x_1351_, 2, v___x_1348_);
return v___x_1351_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__7(void){
_start:
{
lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; 
v___x_1352_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__13));
v___x_1353_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__6, &lp_mathlib_Mathlib_Tactic_wlog_x21___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__6);
v___x_1354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1355_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1355_, 0, v___x_1354_);
lean_ctor_set(v___x_1355_, 1, v___x_1353_);
lean_ctor_set(v___x_1355_, 2, v___x_1352_);
return v___x_1355_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__8(void){
_start:
{
lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; 
v___x_1356_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__32));
v___x_1357_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__7, &lp_mathlib_Mathlib_Tactic_wlog_x21___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__7);
v___x_1358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1359_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1359_, 0, v___x_1358_);
lean_ctor_set(v___x_1359_, 1, v___x_1357_);
lean_ctor_set(v___x_1359_, 2, v___x_1356_);
return v___x_1359_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__9(void){
_start:
{
lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; 
v___x_1360_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog___closed__37, &lp_mathlib_Mathlib_Tactic_wlog___closed__37_once, _init_lp_mathlib_Mathlib_Tactic_wlog___closed__37);
v___x_1361_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__8, &lp_mathlib_Mathlib_Tactic_wlog_x21___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__8);
v___x_1362_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog___closed__4));
v___x_1363_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1363_, 0, v___x_1362_);
lean_ctor_set(v___x_1363_, 1, v___x_1361_);
lean_ctor_set(v___x_1363_, 2, v___x_1360_);
return v___x_1363_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__10(void){
_start:
{
lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1364_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__9, &lp_mathlib_Mathlib_Tactic_wlog_x21___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__9);
v___x_1365_ = lean_unsigned_to_nat(1022u);
v___x_1366_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1));
v___x_1367_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1367_, 0, v___x_1366_);
lean_ctor_set(v___x_1367_, 1, v___x_1365_);
lean_ctor_set(v___x_1367_, 2, v___x_1364_);
return v___x_1367_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_wlog_x21(void){
_start:
{
lean_object* v___x_1368_; 
v___x_1368_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_wlog_x21___closed__10, &lp_mathlib_Mathlib_Tactic_wlog_x21___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_wlog_x21___closed__10);
return v___x_1368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1(lean_object* v_x_1376_, lean_object* v_a_1377_, lean_object* v_a_1378_, lean_object* v_a_1379_, lean_object* v_a_1380_, lean_object* v_a_1381_, lean_object* v_a_1382_, lean_object* v_a_1383_, lean_object* v_a_1384_){
_start:
{
lean_object* v___x_1386_; uint8_t v___x_1387_; 
v___x_1386_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlog_x21___closed__1));
lean_inc(v_x_1376_);
v___x_1387_ = l_Lean_Syntax_isOfKind(v_x_1376_, v___x_1386_);
if (v___x_1387_ == 0)
{
lean_object* v___x_1388_; 
lean_dec(v_x_1376_);
v___x_1388_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1388_;
}
else
{
lean_object* v___x_1389_; lean_object* v_cfg_1390_; lean_object* v___x_1391_; uint8_t v___x_1392_; 
v___x_1389_ = lean_unsigned_to_nat(1u);
v_cfg_1390_ = l_Lean_Syntax_getArg(v_x_1376_, v___x_1389_);
v___x_1391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___closed__2));
lean_inc(v_cfg_1390_);
v___x_1392_ = l_Lean_Syntax_isOfKind(v_cfg_1390_, v___x_1391_);
if (v___x_1392_ == 0)
{
lean_object* v___x_1393_; 
lean_dec(v_cfg_1390_);
lean_dec(v_x_1376_);
v___x_1393_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1393_;
}
else
{
lean_object* v___x_1394_; lean_object* v_h_1395_; lean_object* v___x_1396_; uint8_t v___x_1397_; 
v___x_1394_ = lean_unsigned_to_nat(2u);
v_h_1395_ = l_Lean_Syntax_getArg(v_x_1376_, v___x_1394_);
v___x_1396_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlogCore___closed__2));
lean_inc(v_h_1395_);
v___x_1397_ = l_Lean_Syntax_isOfKind(v_h_1395_, v___x_1396_);
if (v___x_1397_ == 0)
{
lean_object* v___x_1398_; 
lean_dec(v_h_1395_);
lean_dec(v_cfg_1390_);
lean_dec(v_x_1376_);
v___x_1398_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1398_;
}
else
{
lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v_P_1401_; lean_object* v___y_1403_; lean_object* v_H_1404_; lean_object* v___y_1405_; lean_object* v___y_1406_; lean_object* v___y_1407_; lean_object* v___y_1408_; lean_object* v___y_1409_; lean_object* v___y_1410_; lean_object* v___y_1411_; lean_object* v___y_1412_; lean_object* v_xs_1416_; lean_object* v___y_1417_; lean_object* v___y_1418_; lean_object* v___y_1419_; lean_object* v___y_1420_; lean_object* v___y_1421_; lean_object* v___y_1422_; lean_object* v___y_1423_; lean_object* v___y_1424_; lean_object* v___x_1439_; lean_object* v___x_1440_; uint8_t v___x_1441_; 
v___x_1399_ = lean_unsigned_to_nat(0u);
v___x_1400_ = lean_unsigned_to_nat(4u);
v_P_1401_ = l_Lean_Syntax_getArg(v_x_1376_, v___x_1400_);
v___x_1439_ = lean_unsigned_to_nat(5u);
v___x_1440_ = l_Lean_Syntax_getArg(v_x_1376_, v___x_1439_);
v___x_1441_ = l_Lean_Syntax_isNone(v___x_1440_);
if (v___x_1441_ == 0)
{
uint8_t v___x_1442_; 
lean_inc(v___x_1440_);
v___x_1442_ = l_Lean_Syntax_matchesNull(v___x_1440_, v___x_1394_);
if (v___x_1442_ == 0)
{
lean_object* v___x_1443_; 
lean_dec(v___x_1440_);
lean_dec(v_P_1401_);
lean_dec(v_h_1395_);
lean_dec(v_cfg_1390_);
lean_dec(v_x_1376_);
v___x_1443_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1443_;
}
else
{
lean_object* v___x_1444_; lean_object* v_xs_1445_; lean_object* v___x_1446_; 
v___x_1444_ = l_Lean_Syntax_getArg(v___x_1440_, v___x_1389_);
lean_dec(v___x_1440_);
v_xs_1445_ = l_Lean_Syntax_getArgs(v___x_1444_);
lean_dec(v___x_1444_);
v___x_1446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1446_, 0, v_xs_1445_);
v_xs_1416_ = v___x_1446_;
v___y_1417_ = v_a_1377_;
v___y_1418_ = v_a_1378_;
v___y_1419_ = v_a_1379_;
v___y_1420_ = v_a_1380_;
v___y_1421_ = v_a_1381_;
v___y_1422_ = v_a_1382_;
v___y_1423_ = v_a_1383_;
v___y_1424_ = v_a_1384_;
goto v___jp_1415_;
}
}
else
{
lean_object* v___x_1447_; 
lean_dec(v___x_1440_);
v___x_1447_ = lean_box(0);
v_xs_1416_ = v___x_1447_;
v___y_1417_ = v_a_1377_;
v___y_1418_ = v_a_1378_;
v___y_1419_ = v_a_1379_;
v___y_1420_ = v_a_1380_;
v___y_1421_ = v_a_1381_;
v___y_1422_ = v_a_1382_;
v___y_1423_ = v_a_1383_;
v___y_1424_ = v_a_1384_;
goto v___jp_1415_;
}
v___jp_1402_:
{
lean_object* v___x_1413_; lean_object* v___x_1414_; 
v___x_1413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1413_, 0, v_cfg_1390_);
v___x_1414_ = lp_mathlib_Mathlib_Tactic_wlogCore(v_h_1395_, v_P_1401_, v___y_1403_, v_H_1404_, v___x_1413_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, v___y_1412_);
return v___x_1414_;
}
v___jp_1415_:
{
lean_object* v___x_1425_; lean_object* v___x_1426_; uint8_t v___x_1427_; 
v___x_1425_ = lean_unsigned_to_nat(6u);
v___x_1426_ = l_Lean_Syntax_getArg(v_x_1376_, v___x_1425_);
lean_dec(v_x_1376_);
v___x_1427_ = l_Lean_Syntax_isNone(v___x_1426_);
if (v___x_1427_ == 0)
{
uint8_t v___x_1428_; 
lean_inc(v___x_1426_);
v___x_1428_ = l_Lean_Syntax_matchesNull(v___x_1426_, v___x_1394_);
if (v___x_1428_ == 0)
{
lean_object* v___x_1429_; 
lean_dec(v___x_1426_);
lean_dec(v_xs_1416_);
lean_dec(v_P_1401_);
lean_dec(v_h_1395_);
lean_dec(v_cfg_1390_);
v___x_1429_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1429_;
}
else
{
lean_object* v___x_1430_; uint8_t v___x_1431_; 
v___x_1430_ = l_Lean_Syntax_getArg(v___x_1426_, v___x_1389_);
lean_dec(v___x_1426_);
lean_inc(v___x_1430_);
v___x_1431_ = l_Lean_Syntax_isOfKind(v___x_1430_, v___x_1396_);
if (v___x_1431_ == 0)
{
lean_object* v___x_1432_; 
lean_dec(v___x_1430_);
lean_dec(v_xs_1416_);
lean_dec(v_P_1401_);
lean_dec(v_h_1395_);
lean_dec(v_cfg_1390_);
v___x_1432_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1432_;
}
else
{
lean_object* v_H_1433_; lean_object* v___x_1434_; uint8_t v___x_1435_; 
v_H_1433_ = l_Lean_Syntax_getArg(v___x_1430_, v___x_1399_);
lean_dec(v___x_1430_);
v___x_1434_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_wlogCore___closed__4));
lean_inc(v_H_1433_);
v___x_1435_ = l_Lean_Syntax_isOfKind(v_H_1433_, v___x_1434_);
if (v___x_1435_ == 0)
{
lean_object* v___x_1436_; 
lean_dec(v_H_1433_);
lean_dec(v_xs_1416_);
lean_dec(v_P_1401_);
lean_dec(v_h_1395_);
lean_dec(v_cfg_1390_);
v___x_1436_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog__1_spec__0___redArg();
return v___x_1436_;
}
else
{
lean_object* v___x_1437_; 
v___x_1437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1437_, 0, v_H_1433_);
v___y_1403_ = v_xs_1416_;
v_H_1404_ = v___x_1437_;
v___y_1405_ = v___y_1417_;
v___y_1406_ = v___y_1418_;
v___y_1407_ = v___y_1419_;
v___y_1408_ = v___y_1420_;
v___y_1409_ = v___y_1421_;
v___y_1410_ = v___y_1422_;
v___y_1411_ = v___y_1423_;
v___y_1412_ = v___y_1424_;
goto v___jp_1402_;
}
}
}
}
else
{
lean_object* v___x_1438_; 
lean_dec(v___x_1426_);
v___x_1438_ = lean_box(0);
v___y_1403_ = v_xs_1416_;
v_H_1404_ = v___x_1438_;
v___y_1405_ = v___y_1417_;
v___y_1406_ = v___y_1418_;
v___y_1407_ = v___y_1419_;
v___y_1408_ = v___y_1420_;
v___y_1409_ = v___y_1421_;
v___y_1410_ = v___y_1422_;
v___y_1411_ = v___y_1423_;
v___y_1412_ = v___y_1424_;
goto v___jp_1402_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1___boxed(lean_object* v_x_1448_, lean_object* v_a_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_, lean_object* v_a_1455_, lean_object* v_a_1456_, lean_object* v_a_1457_){
_start:
{
lean_object* v_res_1458_; 
v_res_1458_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__WLOG______elabRules__Mathlib__Tactic__wlog_x21__1(v_x_1448_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_, v_a_1454_, v_a_1455_, v_a_1456_);
lean_dec(v_a_1456_);
lean_dec_ref(v_a_1455_);
lean_dec(v_a_1454_);
lean_dec_ref(v_a_1453_);
lean_dec(v_a_1452_);
lean_dec_ref(v_a_1451_);
lean_dec(v_a_1450_);
lean_dec_ref(v_a_1449_);
return v_res_1458_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_MetavarContext(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_WLOG(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_MetavarContext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Cases(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_WLOG(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_wlog = _init_lp_mathlib_Mathlib_Tactic_wlog();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_wlog);
lp_mathlib_Mathlib_Tactic_wlog_x21 = _init_lp_mathlib_Mathlib_Tactic_wlog_x21();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_wlog_x21);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Cases(uint8_t builtin);
lean_object* initialize_Lean_MetavarContext(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_WLOG(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_MetavarContext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_WLOG(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_WLOG(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_WLOG(builtin);
}
#ifdef __cplusplus
}
#endif
