// Lean compiler output
// Module: Mathlib.Tactic.Hint
// Imports: public import Init public meta import Init public meta import Batteries.Control.Nondet.Basic public import Batteries.Linter.UnreachableTactic public import Mathlib.Tactic.Basic
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
lean_object* lean_thunk_get_own(lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_thunk(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerSimplePersistentEnvExtension___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_admitGoal(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
extern lean_object* lp_batteries_Batteries_Linter_UnreachableTactic_ignoreTacticKindsRef;
lean_object* l_Lean_NameHashSet_insert(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* lp_mathlib_Lean_Elab_collectTryThisSuggestions(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_copyHeadTailInfoFrom(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_isNatLit_x3f(lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___lam__0_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___lam__1_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__0_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___lam__0_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__0_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__0_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__1_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___lam__1_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__1_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__1_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__2_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__2_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__2_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__3_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__3_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__3_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__4_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Hint"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__4_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__4_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__5_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "hintExtension"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__5_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__5_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__2_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__3_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__4_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(144, 22, 88, 160, 25, 48, 219, 242)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__5_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(2, 112, 178, 61, 54, 70, 73, 91)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__7_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__7_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__7_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__8_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__6_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__0_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__7_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__1_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__8_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__8_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hintExtension;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "registerHintStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__2_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__3_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__4_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(144, 22, 88, 160, 25, 48, 219, 242)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(54, 47, 26, 100, 196, 8, 163, 146)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "register_hint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__10_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__14_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Hint_registerHintStx = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "expected a numeric literal for priority"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_3357253919____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_3357253919____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = "\n⊢ "};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 3, .m_data = "🎉️ "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "\nRemaining subgoals:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_suggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_suggestion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Hint_hint___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Hint_hint_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Hint_hint_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Hint_hint_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_nil___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__9(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3___redArg(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Try these:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Hint_hint___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Hint_hint___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hint___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Hint_hint___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Hint_hint___lam__1___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hint___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "hintStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__2_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__3_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__4_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(144, 22, 88, 160, 25, 48, 219, 242)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 116, 192, 227, 251, 37, 40, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Hint_hintStx = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___lam__0_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_(lean_object* v_x_1_, lean_object* v_head_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3_, 0, v_head_2_);
lean_ctor_set(v___x_3_, 1, v_x_1_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___lam__1_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_(lean_object* v_es_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_array_mk(v_es_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_as_6_, size_t v_i_7_, size_t v_stop_8_, lean_object* v_b_9_){
_start:
{
uint8_t v___x_10_; 
v___x_10_ = lean_usize_dec_eq(v_i_7_, v_stop_8_);
if (v___x_10_ == 0)
{
lean_object* v___x_11_; lean_object* v___x_12_; size_t v___x_13_; size_t v___x_14_; 
v___x_11_ = lean_array_uget_borrowed(v_as_6_, v_i_7_);
lean_inc(v___x_11_);
v___x_12_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_12_, 0, v___x_11_);
lean_ctor_set(v___x_12_, 1, v_b_9_);
v___x_13_ = ((size_t)1ULL);
v___x_14_ = lean_usize_add(v_i_7_, v___x_13_);
v_i_7_ = v___x_14_;
v_b_9_ = v___x_12_;
goto _start;
}
else
{
return v_b_9_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_as_16_, lean_object* v_i_17_, lean_object* v_stop_18_, lean_object* v_b_19_){
_start:
{
size_t v_i_boxed_20_; size_t v_stop_boxed_21_; lean_object* v_res_22_; 
v_i_boxed_20_ = lean_unbox_usize(v_i_17_);
lean_dec(v_i_17_);
v_stop_boxed_21_ = lean_unbox_usize(v_stop_18_);
lean_dec(v_stop_18_);
v_res_22_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__0(v_as_16_, v_i_boxed_20_, v_stop_boxed_21_, v_b_19_);
lean_dec_ref(v_as_16_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_as_23_, size_t v_i_24_, size_t v_stop_25_, lean_object* v_b_26_){
_start:
{
lean_object* v___y_28_; uint8_t v___x_32_; 
v___x_32_ = lean_usize_dec_eq(v_i_24_, v_stop_25_);
if (v___x_32_ == 0)
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; uint8_t v___x_36_; 
v___x_33_ = lean_array_uget_borrowed(v_as_23_, v_i_24_);
v___x_34_ = lean_unsigned_to_nat(0u);
v___x_35_ = lean_array_get_size(v___x_33_);
v___x_36_ = lean_nat_dec_lt(v___x_34_, v___x_35_);
if (v___x_36_ == 0)
{
v___y_28_ = v_b_26_;
goto v___jp_27_;
}
else
{
uint8_t v___x_37_; 
v___x_37_ = lean_nat_dec_le(v___x_35_, v___x_35_);
if (v___x_37_ == 0)
{
if (v___x_36_ == 0)
{
v___y_28_ = v_b_26_;
goto v___jp_27_;
}
else
{
size_t v___x_38_; size_t v___x_39_; lean_object* v___x_40_; 
v___x_38_ = ((size_t)0ULL);
v___x_39_ = lean_usize_of_nat(v___x_35_);
v___x_40_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__0(v___x_33_, v___x_38_, v___x_39_, v_b_26_);
v___y_28_ = v___x_40_;
goto v___jp_27_;
}
}
else
{
size_t v___x_41_; size_t v___x_42_; lean_object* v___x_43_; 
v___x_41_ = ((size_t)0ULL);
v___x_42_ = lean_usize_of_nat(v___x_35_);
v___x_43_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__0(v___x_33_, v___x_41_, v___x_42_, v_b_26_);
v___y_28_ = v___x_43_;
goto v___jp_27_;
}
}
}
else
{
return v_b_26_;
}
v___jp_27_:
{
size_t v___x_29_; size_t v___x_30_; 
v___x_29_ = ((size_t)1ULL);
v___x_30_ = lean_usize_add(v_i_24_, v___x_29_);
v_i_24_ = v___x_30_;
v_b_26_ = v___y_28_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_as_44_, lean_object* v_i_45_, lean_object* v_stop_46_, lean_object* v_b_47_){
_start:
{
size_t v_i_boxed_48_; size_t v_stop_boxed_49_; lean_object* v_res_50_; 
v_i_boxed_48_ = lean_unbox_usize(v_i_45_);
lean_dec(v_i_45_);
v_stop_boxed_49_ = lean_unbox_usize(v_stop_46_);
lean_dec(v_stop_46_);
v_res_50_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__1(v_as_44_, v_i_boxed_48_, v_stop_boxed_49_, v_b_47_);
lean_dec_ref(v_as_44_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0(lean_object* v_initState_51_, lean_object* v_as_52_){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_53_ = lean_unsigned_to_nat(0u);
v___x_54_ = lean_array_get_size(v_as_52_);
v___x_55_ = lean_nat_dec_lt(v___x_53_, v___x_54_);
if (v___x_55_ == 0)
{
return v_initState_51_;
}
else
{
uint8_t v___x_56_; 
v___x_56_ = lean_nat_dec_le(v___x_54_, v___x_54_);
if (v___x_56_ == 0)
{
if (v___x_55_ == 0)
{
return v_initState_51_;
}
else
{
size_t v___x_57_; size_t v___x_58_; lean_object* v___x_59_; 
v___x_57_ = ((size_t)0ULL);
v___x_58_ = lean_usize_of_nat(v___x_54_);
v___x_59_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__1(v_as_52_, v___x_57_, v___x_58_, v_initState_51_);
return v___x_59_;
}
}
else
{
size_t v___x_60_; size_t v___x_61_; lean_object* v___x_62_; 
v___x_60_ = ((size_t)0ULL);
v___x_61_ = lean_usize_of_nat(v___x_54_);
v___x_62_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0_spec__1(v_as_52_, v___x_60_, v___x_61_, v_initState_51_);
return v___x_62_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0___boxed(lean_object* v_initState_63_, lean_object* v_as_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Lean_mkStateFromImportedEntries___at___00__private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2__spec__0(v_initState_63_, v_as_64_);
lean_dec_ref(v_as_64_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_87_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn___closed__8_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_));
v___x_88_ = l_Lean_registerSimplePersistentEnvExtension___redArg(v___x_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2____boxed(lean_object* v_a_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_();
return v_res_90_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__0(void){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__1(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_92_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__0);
v___x_93_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
return v___x_93_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__2(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__1);
v___x_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg(lean_object* v_prio_96_, lean_object* v_stx_97_, lean_object* v_a_98_){
_start:
{
lean_object* v___x_100_; lean_object* v_env_101_; lean_object* v_nextMacroScope_102_; lean_object* v_ngen_103_; lean_object* v_auxDeclNGen_104_; lean_object* v_traceState_105_; lean_object* v_messages_106_; lean_object* v_infoState_107_; lean_object* v_snapshotTasks_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_125_; 
v___x_100_ = lean_st_ref_take(v_a_98_);
v_env_101_ = lean_ctor_get(v___x_100_, 0);
v_nextMacroScope_102_ = lean_ctor_get(v___x_100_, 1);
v_ngen_103_ = lean_ctor_get(v___x_100_, 2);
v_auxDeclNGen_104_ = lean_ctor_get(v___x_100_, 3);
v_traceState_105_ = lean_ctor_get(v___x_100_, 4);
v_messages_106_ = lean_ctor_get(v___x_100_, 6);
v_infoState_107_ = lean_ctor_get(v___x_100_, 7);
v_snapshotTasks_108_ = lean_ctor_get(v___x_100_, 8);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_125_ == 0)
{
lean_object* v_unused_126_; 
v_unused_126_ = lean_ctor_get(v___x_100_, 5);
lean_dec(v_unused_126_);
v___x_110_ = v___x_100_;
v_isShared_111_ = v_isSharedCheck_125_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_snapshotTasks_108_);
lean_inc(v_infoState_107_);
lean_inc(v_messages_106_);
lean_inc(v_traceState_105_);
lean_inc(v_auxDeclNGen_104_);
lean_inc(v_ngen_103_);
lean_inc(v_nextMacroScope_102_);
lean_inc(v_env_101_);
lean_dec(v___x_100_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_125_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
lean_object* v___x_112_; lean_object* v_toEnvExtension_113_; lean_object* v_asyncMode_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_120_; 
v___x_112_ = lp_mathlib_Mathlib_Tactic_Hint_hintExtension;
v_toEnvExtension_113_ = lean_ctor_get(v___x_112_, 0);
v_asyncMode_114_ = lean_ctor_get(v_toEnvExtension_113_, 2);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v_prio_96_);
lean_ctor_set(v___x_115_, 1, v_stx_97_);
v___x_116_ = lean_box(0);
v___x_117_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_112_, v_env_101_, v___x_115_, v_asyncMode_114_, v___x_116_);
v___x_118_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___closed__2);
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 5, v___x_118_);
lean_ctor_set(v___x_110_, 0, v___x_117_);
v___x_120_ = v___x_110_;
goto v_reusejp_119_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v___x_117_);
lean_ctor_set(v_reuseFailAlloc_124_, 1, v_nextMacroScope_102_);
lean_ctor_set(v_reuseFailAlloc_124_, 2, v_ngen_103_);
lean_ctor_set(v_reuseFailAlloc_124_, 3, v_auxDeclNGen_104_);
lean_ctor_set(v_reuseFailAlloc_124_, 4, v_traceState_105_);
lean_ctor_set(v_reuseFailAlloc_124_, 5, v___x_118_);
lean_ctor_set(v_reuseFailAlloc_124_, 6, v_messages_106_);
lean_ctor_set(v_reuseFailAlloc_124_, 7, v_infoState_107_);
lean_ctor_set(v_reuseFailAlloc_124_, 8, v_snapshotTasks_108_);
v___x_120_ = v_reuseFailAlloc_124_;
goto v_reusejp_119_;
}
v_reusejp_119_:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_121_ = lean_st_ref_set(v_a_98_, v___x_120_);
v___x_122_ = lean_box(0);
v___x_123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
return v___x_123_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg___boxed(lean_object* v_prio_127_, lean_object* v_stx_128_, lean_object* v_a_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg(v_prio_127_, v_stx_128_, v_a_129_);
lean_dec(v_a_129_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint(lean_object* v_prio_132_, lean_object* v_stx_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg(v_prio_132_, v_stx_133_, v_a_135_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_addHint___boxed(lean_object* v_prio_138_, lean_object* v_stx_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_Mathlib_Tactic_Hint_addHint(v_prio_138_, v_stx_139_, v_a_140_, v_a_141_);
lean_dec(v_a_141_);
lean_dec_ref(v_a_140_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints___redArg(lean_object* v_a_144_){
_start:
{
lean_object* v___x_146_; lean_object* v_env_147_; lean_object* v___x_148_; lean_object* v_toEnvExtension_149_; lean_object* v_asyncMode_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_146_ = lean_st_ref_get(v_a_144_);
v_env_147_ = lean_ctor_get(v___x_146_, 0);
lean_inc_ref(v_env_147_);
lean_dec(v___x_146_);
v___x_148_ = lp_mathlib_Mathlib_Tactic_Hint_hintExtension;
v_toEnvExtension_149_ = lean_ctor_get(v___x_148_, 0);
v_asyncMode_150_ = lean_ctor_get(v_toEnvExtension_149_, 2);
v___x_151_ = lean_box(0);
v___x_152_ = lean_box(0);
v___x_153_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_151_, v___x_148_, v_env_147_, v_asyncMode_150_, v___x_152_);
v___x_154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints___redArg___boxed(lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_Mathlib_Tactic_Hint_getHints___redArg(v_a_155_);
lean_dec(v_a_155_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints(lean_object* v_a_158_, lean_object* v_a_159_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lp_mathlib_Mathlib_Tactic_Hint_getHints___redArg(v_a_159_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_getHints___boxed(lean_object* v_a_162_, lean_object* v_a_163_, lean_object* v_a_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_Mathlib_Tactic_Hint_getHints(v_a_162_, v_a_163_);
lean_dec(v_a_163_);
lean_dec_ref(v_a_162_);
return v_res_165_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_202_ = lean_box(0);
v___x_203_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_204_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
lean_ctor_set(v___x_204_, 1, v___x_202_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg(){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0);
v___x_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___boxed(lean_object* v___y_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg();
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0(lean_object* v_00_u03b1_210_, lean_object* v___y_211_, lean_object* v___y_212_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg();
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___boxed(lean_object* v_00_u03b1_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0(v_00_u03b1_215_, v___y_216_, v___y_217_);
lean_dec(v___y_217_);
lean_dec_ref(v___y_216_);
return v_res_219_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_220_ = lean_box(1);
v___x_221_ = l_Lean_MessageData_ofFormat(v___x_220_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__3(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_225_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__2));
v___x_226_ = l_Lean_MessageData_ofFormat(v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4(lean_object* v_x_227_, lean_object* v_x_228_){
_start:
{
if (lean_obj_tag(v_x_228_) == 0)
{
return v_x_227_;
}
else
{
lean_object* v_head_229_; lean_object* v_tail_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_252_; 
v_head_229_ = lean_ctor_get(v_x_228_, 0);
v_tail_230_ = lean_ctor_get(v_x_228_, 1);
v_isSharedCheck_252_ = !lean_is_exclusive(v_x_228_);
if (v_isSharedCheck_252_ == 0)
{
v___x_232_ = v_x_228_;
v_isShared_233_ = v_isSharedCheck_252_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_tail_230_);
lean_inc(v_head_229_);
lean_dec(v_x_228_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_252_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v_before_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_250_; 
v_before_234_ = lean_ctor_get(v_head_229_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v_head_229_);
if (v_isSharedCheck_250_ == 0)
{
lean_object* v_unused_251_; 
v_unused_251_ = lean_ctor_get(v_head_229_, 1);
lean_dec(v_unused_251_);
v___x_236_ = v_head_229_;
v_isShared_237_ = v_isSharedCheck_250_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_before_234_);
lean_dec(v_head_229_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_250_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v___x_238_; lean_object* v___x_240_; 
v___x_238_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0);
if (v_isShared_237_ == 0)
{
lean_ctor_set_tag(v___x_236_, 7);
lean_ctor_set(v___x_236_, 1, v___x_238_);
lean_ctor_set(v___x_236_, 0, v_x_227_);
v___x_240_ = v___x_236_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_x_227_);
lean_ctor_set(v_reuseFailAlloc_249_, 1, v___x_238_);
v___x_240_ = v_reuseFailAlloc_249_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
lean_object* v___x_241_; lean_object* v___x_243_; 
v___x_241_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__3);
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
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v___x_240_);
lean_ctor_set(v_reuseFailAlloc_248_, 1, v___x_241_);
v___x_243_ = v_reuseFailAlloc_248_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_244_ = l_Lean_MessageData_ofSyntax(v_before_234_);
v___x_245_ = l_Lean_indentD(v___x_244_);
v___x_246_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_243_);
lean_ctor_set(v___x_246_, 1, v___x_245_);
v_x_227_ = v___x_246_;
v_x_228_ = v_tail_230_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__3(lean_object* v_opts_253_, lean_object* v_opt_254_){
_start:
{
lean_object* v_name_255_; lean_object* v_defValue_256_; lean_object* v_map_257_; lean_object* v___x_258_; 
v_name_255_ = lean_ctor_get(v_opt_254_, 0);
v_defValue_256_ = lean_ctor_get(v_opt_254_, 1);
v_map_257_ = lean_ctor_get(v_opts_253_, 0);
v___x_258_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_257_, v_name_255_);
if (lean_obj_tag(v___x_258_) == 0)
{
uint8_t v___x_259_; 
v___x_259_ = lean_unbox(v_defValue_256_);
return v___x_259_;
}
else
{
lean_object* v_val_260_; 
v_val_260_ = lean_ctor_get(v___x_258_, 0);
lean_inc(v_val_260_);
lean_dec_ref_known(v___x_258_, 1);
if (lean_obj_tag(v_val_260_) == 1)
{
uint8_t v_v_261_; 
v_v_261_ = lean_ctor_get_uint8(v_val_260_, 0);
lean_dec_ref_known(v_val_260_, 0);
return v_v_261_;
}
else
{
uint8_t v___x_262_; 
lean_dec(v_val_260_);
v___x_262_ = lean_unbox(v_defValue_256_);
return v___x_262_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__3___boxed(lean_object* v_opts_263_, lean_object* v_opt_264_){
_start:
{
uint8_t v_res_265_; lean_object* v_r_266_; 
v_res_265_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__3(v_opts_263_, v_opt_264_);
lean_dec_ref(v_opt_264_);
lean_dec_ref(v_opts_263_);
v_r_266_ = lean_box(v_res_265_);
return v_r_266_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__1));
v___x_271_ = l_Lean_MessageData_ofFormat(v___x_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg(lean_object* v_msgData_272_, lean_object* v_macroStack_273_, lean_object* v___y_274_){
_start:
{
lean_object* v_options_276_; lean_object* v___x_277_; uint8_t v___x_278_; 
v_options_276_ = lean_ctor_get(v___y_274_, 2);
v___x_277_ = l_Lean_Elab_pp_macroStack;
v___x_278_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__3(v_options_276_, v___x_277_);
if (v___x_278_ == 0)
{
lean_object* v___x_279_; 
lean_dec(v_macroStack_273_);
v___x_279_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_279_, 0, v_msgData_272_);
return v___x_279_;
}
else
{
if (lean_obj_tag(v_macroStack_273_) == 0)
{
lean_object* v___x_280_; 
v___x_280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_280_, 0, v_msgData_272_);
return v___x_280_;
}
else
{
lean_object* v_head_281_; lean_object* v_after_282_; lean_object* v___x_284_; uint8_t v_isShared_285_; uint8_t v_isSharedCheck_297_; 
v_head_281_ = lean_ctor_get(v_macroStack_273_, 0);
lean_inc(v_head_281_);
v_after_282_ = lean_ctor_get(v_head_281_, 1);
v_isSharedCheck_297_ = !lean_is_exclusive(v_head_281_);
if (v_isSharedCheck_297_ == 0)
{
lean_object* v_unused_298_; 
v_unused_298_ = lean_ctor_get(v_head_281_, 0);
lean_dec(v_unused_298_);
v___x_284_ = v_head_281_;
v_isShared_285_ = v_isSharedCheck_297_;
goto v_resetjp_283_;
}
else
{
lean_inc(v_after_282_);
lean_dec(v_head_281_);
v___x_284_ = lean_box(0);
v_isShared_285_ = v_isSharedCheck_297_;
goto v_resetjp_283_;
}
v_resetjp_283_:
{
lean_object* v___x_286_; lean_object* v___x_288_; 
v___x_286_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4___closed__0);
if (v_isShared_285_ == 0)
{
lean_ctor_set_tag(v___x_284_, 7);
lean_ctor_set(v___x_284_, 1, v___x_286_);
lean_ctor_set(v___x_284_, 0, v_msgData_272_);
v___x_288_ = v___x_284_;
goto v_reusejp_287_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v_msgData_272_);
lean_ctor_set(v_reuseFailAlloc_296_, 1, v___x_286_);
v___x_288_ = v_reuseFailAlloc_296_;
goto v_reusejp_287_;
}
v_reusejp_287_:
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v_msgData_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_289_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___closed__2);
v___x_290_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_288_);
lean_ctor_set(v___x_290_, 1, v___x_289_);
v___x_291_ = l_Lean_MessageData_ofSyntax(v_after_282_);
v___x_292_ = l_Lean_indentD(v___x_291_);
v_msgData_293_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_293_, 0, v___x_290_);
lean_ctor_set(v_msgData_293_, 1, v___x_292_);
v___x_294_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2_spec__4(v_msgData_293_, v_macroStack_273_);
v___x_295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_295_, 0, v___x_294_);
return v___x_295_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg___boxed(lean_object* v_msgData_299_, lean_object* v_macroStack_300_, lean_object* v___y_301_, lean_object* v___y_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg(v_msgData_299_, v_macroStack_300_, v___y_301_);
lean_dec_ref(v___y_301_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__1(lean_object* v_msgData_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v___x_310_; lean_object* v_env_311_; lean_object* v___x_312_; lean_object* v_mctx_313_; lean_object* v_lctx_314_; lean_object* v_options_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_310_ = lean_st_ref_get(v___y_308_);
v_env_311_ = lean_ctor_get(v___x_310_, 0);
lean_inc_ref(v_env_311_);
lean_dec(v___x_310_);
v___x_312_ = lean_st_ref_get(v___y_306_);
v_mctx_313_ = lean_ctor_get(v___x_312_, 0);
lean_inc_ref(v_mctx_313_);
lean_dec(v___x_312_);
v_lctx_314_ = lean_ctor_get(v___y_305_, 2);
v_options_315_ = lean_ctor_get(v___y_307_, 2);
lean_inc_ref(v_options_315_);
lean_inc_ref(v_lctx_314_);
v___x_316_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_316_, 0, v_env_311_);
lean_ctor_set(v___x_316_, 1, v_mctx_313_);
lean_ctor_set(v___x_316_, 2, v_lctx_314_);
lean_ctor_set(v___x_316_, 3, v_options_315_);
v___x_317_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_317_, 0, v___x_316_);
lean_ctor_set(v___x_317_, 1, v_msgData_304_);
v___x_318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_318_, 0, v___x_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__1___boxed(lean_object* v_msgData_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__1(v_msgData_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___redArg(lean_object* v_msg_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_){
_start:
{
lean_object* v_ref_334_; lean_object* v___x_335_; lean_object* v_a_336_; lean_object* v_macroStack_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v_a_340_; lean_object* v___x_342_; uint8_t v_isShared_343_; uint8_t v_isSharedCheck_348_; 
v_ref_334_ = lean_ctor_get(v___y_331_, 5);
v___x_335_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__1(v_msg_326_, v___y_329_, v___y_330_, v___y_331_, v___y_332_);
v_a_336_ = lean_ctor_get(v___x_335_, 0);
lean_inc(v_a_336_);
lean_dec_ref(v___x_335_);
v_macroStack_337_ = lean_ctor_get(v___y_327_, 1);
v___x_338_ = l_Lean_Elab_getBetterRef(v_ref_334_, v_macroStack_337_);
lean_inc(v_macroStack_337_);
v___x_339_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg(v_a_336_, v_macroStack_337_, v___y_331_);
v_a_340_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_348_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_348_ == 0)
{
v___x_342_ = v___x_339_;
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
else
{
lean_inc(v_a_340_);
lean_dec(v___x_339_);
v___x_342_ = lean_box(0);
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
v_resetjp_341_:
{
lean_object* v___x_344_; lean_object* v___x_346_; 
v___x_344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_338_);
lean_ctor_set(v___x_344_, 1, v_a_340_);
if (v_isShared_343_ == 0)
{
lean_ctor_set_tag(v___x_342_, 1);
lean_ctor_set(v___x_342_, 0, v___x_344_);
v___x_346_ = v___x_342_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v___x_344_);
v___x_346_ = v_reuseFailAlloc_347_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
return v___x_346_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___redArg___boxed(lean_object* v_msg_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___redArg(v_msg_349_, v___y_350_, v___y_351_, v___y_352_, v___y_353_, v___y_354_, v___y_355_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
lean_dec(v___y_353_);
lean_dec_ref(v___y_352_);
lean_dec(v___y_351_);
lean_dec_ref(v___y_350_);
return v_res_357_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_359_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__0));
v___x_360_ = l_Lean_stringToMessageData(v___x_359_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0(lean_object* v___x_361_, lean_object* v_tac_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_){
_start:
{
if (lean_obj_tag(v___x_361_) == 1)
{
lean_object* v_val_370_; lean_object* v___x_371_; 
v_val_370_ = lean_ctor_get(v___x_361_, 0);
lean_inc(v_val_370_);
lean_dec_ref_known(v___x_361_, 1);
v___x_371_ = lp_mathlib_Mathlib_Tactic_Hint_addHint___redArg(v_val_370_, v_tac_362_, v___y_368_);
return v___x_371_;
}
else
{
lean_object* v___x_372_; lean_object* v___x_373_; 
lean_dec(v_tac_362_);
lean_dec(v___x_361_);
v___x_372_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___closed__1);
v___x_373_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___redArg(v___x_372_, v___y_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_, v___y_368_);
return v___x_373_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___boxed(lean_object* v___x_374_, lean_object* v_tac_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0(v___x_374_, v_tac_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
lean_dec(v___y_379_);
lean_dec_ref(v___y_378_);
lean_dec(v___y_377_);
lean_dec_ref(v___y_376_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1(lean_object* v_x_384_, lean_object* v_a_385_, lean_object* v_a_386_){
_start:
{
lean_object* v___x_388_; uint8_t v___x_389_; 
v___x_388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1));
lean_inc(v_x_384_);
v___x_389_ = l_Lean_Syntax_isOfKind(v_x_384_, v___x_388_);
if (v___x_389_ == 0)
{
lean_object* v___x_390_; 
lean_dec(v_x_384_);
v___x_390_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg();
return v___x_390_;
}
else
{
lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v_tac_396_; lean_object* v___x_397_; lean_object* v___y_398_; lean_object* v___x_399_; 
v___x_391_ = lean_unsigned_to_nat(1u);
v___x_392_ = l_Lean_Syntax_getArg(v_x_384_, v___x_391_);
v___x_393_ = lean_unsigned_to_nat(2u);
v___x_394_ = l_Lean_Syntax_getArg(v_x_384_, v___x_393_);
lean_dec(v_x_384_);
v___x_395_ = lean_box(0);
v_tac_396_ = l_Lean_Syntax_copyHeadTailInfoFrom(v___x_394_, v___x_395_);
v___x_397_ = l_Lean_Syntax_isNatLit_x3f(v___x_392_);
lean_dec(v___x_392_);
v___y_398_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___lam__0___boxed), 9, 2);
lean_closure_set(v___y_398_, 0, v___x_397_);
lean_closure_set(v___y_398_, 1, v_tac_396_);
v___x_399_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___y_398_, v_a_385_, v_a_386_);
return v___x_399_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1___boxed(lean_object* v_x_400_, lean_object* v_a_401_, lean_object* v_a_402_, lean_object* v_a_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1(v_x_400_, v_a_401_, v_a_402_);
lean_dec(v_a_402_);
lean_dec_ref(v_a_401_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1(lean_object* v_00_u03b1_405_, lean_object* v_msg_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___redArg(v_msg_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_, v___y_411_, v___y_412_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1___boxed(lean_object* v_00_u03b1_415_, lean_object* v_msg_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1(v_00_u03b1_415_, v_msg_416_, v___y_417_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
lean_dec(v___y_418_);
lean_dec_ref(v___y_417_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2(lean_object* v_msgData_425_, lean_object* v_macroStack_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___redArg(v_msgData_425_, v_macroStack_426_, v___y_431_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2___boxed(lean_object* v_msgData_435_, lean_object* v_macroStack_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__1_spec__2(v_msgData_435_, v_macroStack_436_, v___y_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_3357253919____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_446_ = lp_batteries_Batteries_Linter_UnreachableTactic_ignoreTacticKindsRef;
v___x_447_ = lean_st_ref_take(v___x_446_);
v___x_448_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__1));
v___x_449_ = l_Lean_NameHashSet_insert(v___x_447_, v___x_448_);
v___x_450_ = lean_st_ref_set(v___x_446_, v___x_449_);
v___x_451_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_451_, 0, v___x_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_3357253919____hygCtx___hyg_2____boxed(lean_object* v_a_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_3357253919____hygCtx___hyg_2_();
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___redArg(lean_object* v_e_454_, lean_object* v___y_455_){
_start:
{
uint8_t v___x_457_; 
v___x_457_ = l_Lean_Expr_hasMVar(v_e_454_);
if (v___x_457_ == 0)
{
lean_object* v___x_458_; 
v___x_458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_458_, 0, v_e_454_);
return v___x_458_;
}
else
{
lean_object* v___x_459_; lean_object* v_mctx_460_; lean_object* v___x_461_; lean_object* v_fst_462_; lean_object* v_snd_463_; lean_object* v___x_464_; lean_object* v_cache_465_; lean_object* v_zetaDeltaFVarIds_466_; lean_object* v_postponed_467_; lean_object* v_diag_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_477_; 
v___x_459_ = lean_st_ref_get(v___y_455_);
v_mctx_460_ = lean_ctor_get(v___x_459_, 0);
lean_inc_ref(v_mctx_460_);
lean_dec(v___x_459_);
v___x_461_ = l_Lean_instantiateMVarsCore(v_mctx_460_, v_e_454_);
v_fst_462_ = lean_ctor_get(v___x_461_, 0);
lean_inc(v_fst_462_);
v_snd_463_ = lean_ctor_get(v___x_461_, 1);
lean_inc(v_snd_463_);
lean_dec_ref(v___x_461_);
v___x_464_ = lean_st_ref_take(v___y_455_);
v_cache_465_ = lean_ctor_get(v___x_464_, 1);
v_zetaDeltaFVarIds_466_ = lean_ctor_get(v___x_464_, 2);
v_postponed_467_ = lean_ctor_get(v___x_464_, 3);
v_diag_468_ = lean_ctor_get(v___x_464_, 4);
v_isSharedCheck_477_ = !lean_is_exclusive(v___x_464_);
if (v_isSharedCheck_477_ == 0)
{
lean_object* v_unused_478_; 
v_unused_478_ = lean_ctor_get(v___x_464_, 0);
lean_dec(v_unused_478_);
v___x_470_ = v___x_464_;
v_isShared_471_ = v_isSharedCheck_477_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_diag_468_);
lean_inc(v_postponed_467_);
lean_inc(v_zetaDeltaFVarIds_466_);
lean_inc(v_cache_465_);
lean_dec(v___x_464_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_477_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
lean_object* v___x_473_; 
if (v_isShared_471_ == 0)
{
lean_ctor_set(v___x_470_, 0, v_snd_463_);
v___x_473_ = v___x_470_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_476_; 
v_reuseFailAlloc_476_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_476_, 0, v_snd_463_);
lean_ctor_set(v_reuseFailAlloc_476_, 1, v_cache_465_);
lean_ctor_set(v_reuseFailAlloc_476_, 2, v_zetaDeltaFVarIds_466_);
lean_ctor_set(v_reuseFailAlloc_476_, 3, v_postponed_467_);
lean_ctor_set(v_reuseFailAlloc_476_, 4, v_diag_468_);
v___x_473_ = v_reuseFailAlloc_476_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_474_ = lean_st_ref_set(v___y_455_, v___x_473_);
v___x_475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_475_, 0, v_fst_462_);
return v___x_475_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___redArg___boxed(lean_object* v_e_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___redArg(v_e_479_, v___y_480_);
lean_dec(v___y_480_);
return v_res_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0(lean_object* v_e_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___redArg(v_e_483_, v___y_489_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___boxed(lean_object* v_e_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_){
_start:
{
lean_object* v_res_504_; 
v_res_504_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0(v_e_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_, v___y_499_, v___y_500_, v___y_501_, v___y_502_);
lean_dec(v___y_502_);
lean_dec_ref(v___y_501_);
lean_dec(v___y_500_);
lean_dec_ref(v___y_499_);
lean_dec(v___y_498_);
lean_dec_ref(v___y_497_);
lean_dec(v___y_496_);
lean_dec_ref(v___y_495_);
return v_res_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg(lean_object* v_as_x27_508_, lean_object* v_b_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_){
_start:
{
if (lean_obj_tag(v_as_x27_508_) == 0)
{
lean_object* v___x_519_; 
v___x_519_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_519_, 0, v_b_509_);
return v___x_519_;
}
else
{
lean_object* v_head_520_; lean_object* v_tail_521_; lean_object* v___x_522_; 
v_head_520_ = lean_ctor_get(v_as_x27_508_, 0);
v_tail_521_ = lean_ctor_get(v_as_x27_508_, 1);
lean_inc(v_head_520_);
v___x_522_ = l_Lean_MVarId_getType(v_head_520_, v___y_514_, v___y_515_, v___y_516_, v___y_517_);
if (lean_obj_tag(v___x_522_) == 0)
{
lean_object* v_a_523_; lean_object* v___x_524_; lean_object* v_a_525_; lean_object* v___x_526_; 
v_a_523_ = lean_ctor_get(v___x_522_, 0);
lean_inc(v_a_523_);
lean_dec_ref_known(v___x_522_, 1);
v___x_524_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Hint_suggestion_spec__0___redArg(v_a_523_, v___y_515_);
v_a_525_ = lean_ctor_get(v___x_524_, 0);
lean_inc(v_a_525_);
lean_dec_ref(v___x_524_);
v___x_526_ = l_Lean_PrettyPrinter_ppExpr(v_a_525_, v___y_514_, v___y_515_, v___y_516_, v___y_517_);
if (lean_obj_tag(v___x_526_) == 0)
{
lean_object* v_a_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; 
v_a_527_ = lean_ctor_get(v___x_526_, 0);
lean_inc(v_a_527_);
lean_dec_ref_known(v___x_526_, 1);
v___x_528_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___closed__1));
v___x_529_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_529_, 0, v___x_528_);
lean_ctor_set(v___x_529_, 1, v_a_527_);
v___x_530_ = l_Std_Format_defWidth;
v___x_531_ = lean_unsigned_to_nat(0u);
v___x_532_ = l_Std_Format_pretty(v___x_529_, v___x_530_, v___x_531_, v___x_531_);
v___x_533_ = lean_string_append(v_b_509_, v___x_532_);
lean_dec_ref(v___x_532_);
v_as_x27_508_ = v_tail_521_;
v_b_509_ = v___x_533_;
goto _start;
}
else
{
lean_object* v_a_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_542_; 
lean_dec_ref(v_b_509_);
v_a_535_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_542_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_542_ == 0)
{
v___x_537_ = v___x_526_;
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_a_535_);
lean_dec(v___x_526_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_540_; 
if (v_isShared_538_ == 0)
{
v___x_540_ = v___x_537_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v_a_535_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
}
}
else
{
lean_object* v_a_543_; lean_object* v___x_545_; uint8_t v_isShared_546_; uint8_t v_isSharedCheck_550_; 
lean_dec_ref(v_b_509_);
v_a_543_ = lean_ctor_get(v___x_522_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_522_);
if (v_isSharedCheck_550_ == 0)
{
v___x_545_ = v___x_522_;
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
else
{
lean_inc(v_a_543_);
lean_dec(v___x_522_);
v___x_545_ = lean_box(0);
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
v_resetjp_544_:
{
lean_object* v___x_548_; 
if (v_isShared_546_ == 0)
{
v___x_548_ = v___x_545_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v_a_543_);
v___x_548_ = v_reuseFailAlloc_549_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
return v___x_548_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg___boxed(lean_object* v_as_x27_551_, lean_object* v_b_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_){
_start:
{
lean_object* v_res_562_; 
v_res_562_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg(v_as_x27_551_, v_b_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_, v___y_558_, v___y_559_, v___y_560_);
lean_dec(v___y_560_);
lean_dec_ref(v___y_559_);
lean_dec(v___y_558_);
lean_dec_ref(v___y_557_);
lean_dec(v___y_556_);
lean_dec_ref(v___y_555_);
lean_dec(v___y_554_);
lean_dec_ref(v___y_553_);
lean_dec(v_as_x27_551_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_suggestion(lean_object* v_tac_567_, lean_object* v_trees_568_, lean_object* v_a_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_, lean_object* v_a_574_, lean_object* v_a_575_, lean_object* v_a_576_){
_start:
{
lean_object* v___y_579_; lean_object* v___y_580_; lean_object* v___y_581_; lean_object* v___y_586_; lean_object* v___y_587_; lean_object* v___x_596_; 
v___x_596_ = l_Lean_Elab_Tactic_getGoals___redArg(v_a_570_);
if (lean_obj_tag(v___x_596_) == 0)
{
lean_object* v_a_597_; lean_object* v_postInfo_x3f_599_; uint8_t v___x_603_; 
v_a_597_ = lean_ctor_get(v___x_596_, 0);
lean_inc(v_a_597_);
lean_dec_ref_known(v___x_596_, 1);
v___x_603_ = l_List_isEmpty___redArg(v_a_597_);
if (v___x_603_ == 0)
{
lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__2));
v___x_605_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg(v_a_597_, v___x_604_, v_a_569_, v_a_570_, v_a_571_, v_a_572_, v_a_573_, v_a_574_, v_a_575_, v_a_576_);
if (lean_obj_tag(v___x_605_) == 0)
{
lean_object* v_a_606_; lean_object* v___x_607_; 
v_a_606_ = lean_ctor_get(v___x_605_, 0);
lean_inc(v_a_606_);
lean_dec_ref_known(v___x_605_, 1);
v___x_607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_607_, 0, v_a_606_);
v_postInfo_x3f_599_ = v___x_607_;
goto v___jp_598_;
}
else
{
lean_object* v_a_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_615_; 
lean_dec(v_a_597_);
lean_dec(v_tac_567_);
v_a_608_ = lean_ctor_get(v___x_605_, 0);
v_isSharedCheck_615_ = !lean_is_exclusive(v___x_605_);
if (v_isSharedCheck_615_ == 0)
{
v___x_610_ = v___x_605_;
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_a_608_);
lean_dec(v___x_605_);
v___x_610_ = lean_box(0);
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
v_resetjp_609_:
{
lean_object* v___x_613_; 
if (v_isShared_611_ == 0)
{
v___x_613_ = v___x_610_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v_a_608_);
v___x_613_ = v_reuseFailAlloc_614_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
return v___x_613_;
}
}
}
}
else
{
lean_object* v___x_616_; 
v___x_616_ = lean_box(0);
v_postInfo_x3f_599_ = v___x_616_;
goto v___jp_598_;
}
v___jp_598_:
{
uint8_t v___x_600_; 
v___x_600_ = l_List_isEmpty___redArg(v_a_597_);
lean_dec(v_a_597_);
if (v___x_600_ == 0)
{
lean_object* v___x_601_; 
v___x_601_ = lean_box(0);
v___y_586_ = v_postInfo_x3f_599_;
v___y_587_ = v___x_601_;
goto v___jp_585_;
}
else
{
lean_object* v___x_602_; 
v___x_602_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_suggestion___closed__1));
v___y_586_ = v_postInfo_x3f_599_;
v___y_587_ = v___x_602_;
goto v___jp_585_;
}
}
}
else
{
lean_object* v_a_617_; lean_object* v___x_619_; uint8_t v_isShared_620_; uint8_t v_isSharedCheck_624_; 
lean_dec(v_tac_567_);
v_a_617_ = lean_ctor_get(v___x_596_, 0);
v_isSharedCheck_624_ = !lean_is_exclusive(v___x_596_);
if (v_isSharedCheck_624_ == 0)
{
v___x_619_ = v___x_596_;
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
else
{
lean_inc(v_a_617_);
lean_dec(v___x_596_);
v___x_619_ = lean_box(0);
v_isShared_620_ = v_isSharedCheck_624_;
goto v_resetjp_618_;
}
v_resetjp_618_:
{
lean_object* v___x_622_; 
if (v_isShared_620_ == 0)
{
v___x_622_ = v___x_619_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v_a_617_);
v___x_622_ = v_reuseFailAlloc_623_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
return v___x_622_;
}
}
}
v___jp_578_:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_582_ = lean_box(0);
lean_inc(v___y_580_);
v___x_583_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_583_, 0, v___y_581_);
lean_ctor_set(v___x_583_, 1, v___y_580_);
lean_ctor_set(v___x_583_, 2, v___y_579_);
lean_ctor_set(v___x_583_, 3, v___x_582_);
lean_ctor_set(v___x_583_, 4, v___x_582_);
lean_ctor_set(v___x_583_, 5, v___x_582_);
v___x_584_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_584_, 0, v___x_583_);
return v___x_584_;
}
v___jp_585_:
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; uint8_t v___x_591_; 
v___x_588_ = lp_mathlib_Lean_Elab_collectTryThisSuggestions(v_trees_568_);
v___x_589_ = lean_unsigned_to_nat(0u);
v___x_590_ = lean_array_get_size(v___x_588_);
v___x_591_ = lean_nat_dec_lt(v___x_589_, v___x_590_);
if (v___x_591_ == 0)
{
lean_object* v___x_592_; lean_object* v___x_593_; 
lean_dec_ref(v___x_588_);
v___x_592_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_registerHintStx___closed__11));
v___x_593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
lean_ctor_set(v___x_593_, 1, v_tac_567_);
v___y_579_ = v___y_586_;
v___y_580_ = v___y_587_;
v___y_581_ = v___x_593_;
goto v___jp_578_;
}
else
{
lean_object* v___x_594_; lean_object* v_suggestion_595_; 
lean_dec(v_tac_567_);
v___x_594_ = lean_array_fget(v___x_588_, v___x_589_);
lean_dec_ref(v___x_588_);
v_suggestion_595_ = lean_ctor_get(v___x_594_, 0);
lean_inc_ref(v_suggestion_595_);
lean_dec(v___x_594_);
v___y_579_ = v___y_586_;
v___y_580_ = v___y_587_;
v___y_581_ = v_suggestion_595_;
goto v___jp_578_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_suggestion___boxed(lean_object* v_tac_625_, lean_object* v_trees_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_){
_start:
{
lean_object* v_res_636_; 
v_res_636_ = lp_mathlib_Mathlib_Tactic_Hint_suggestion(v_tac_625_, v_trees_626_, v_a_627_, v_a_628_, v_a_629_, v_a_630_, v_a_631_, v_a_632_, v_a_633_, v_a_634_);
lean_dec(v_a_634_);
lean_dec_ref(v_a_633_);
lean_dec(v_a_632_);
lean_dec_ref(v_a_631_);
lean_dec(v_a_630_);
lean_dec_ref(v_a_629_);
lean_dec(v_a_628_);
lean_dec_ref(v_a_627_);
lean_dec_ref(v_trees_626_);
return v_res_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1(lean_object* v_as_637_, lean_object* v_as_x27_638_, lean_object* v_b_639_, lean_object* v_a_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___redArg(v_as_x27_638_, v_b_639_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_, v___y_648_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1___boxed(lean_object* v_as_651_, lean_object* v_as_x27_652_, lean_object* v_b_653_, lean_object* v_a_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Hint_suggestion_spec__1(v_as_651_, v_as_x27_652_, v_b_653_, v_a_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_, v___y_662_);
lean_dec(v___y_662_);
lean_dec_ref(v___y_661_);
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec(v___y_656_);
lean_dec_ref(v___y_655_);
lean_dec(v_as_x27_652_);
lean_dec(v_as_651_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___redArg(lean_object* v_x_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_){
_start:
{
lean_object* v___x_675_; 
v___x_675_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_667_, v___y_669_, v___y_671_, v___y_673_);
if (lean_obj_tag(v___x_675_) == 0)
{
lean_object* v_a_676_; lean_object* v___x_677_; 
v_a_676_ = lean_ctor_get(v___x_675_, 0);
lean_inc(v_a_676_);
lean_dec_ref_known(v___x_675_, 1);
v___x_677_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_667_, v___y_669_, v___y_671_, v___y_673_);
if (lean_obj_tag(v___x_677_) == 0)
{
lean_object* v_a_678_; lean_object* v___x_679_; 
v_a_678_ = lean_ctor_get(v___x_677_, 0);
lean_inc(v_a_678_);
lean_dec_ref_known(v___x_677_, 1);
lean_inc(v___y_673_);
lean_inc_ref(v___y_672_);
lean_inc(v___y_671_);
lean_inc_ref(v___y_670_);
lean_inc(v___y_669_);
lean_inc_ref(v___y_668_);
lean_inc(v___y_667_);
lean_inc_ref(v___y_666_);
v___x_679_ = lean_apply_9(v_x_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_, lean_box(0));
if (lean_obj_tag(v___x_679_) == 0)
{
lean_object* v_a_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_688_; 
lean_dec(v_a_678_);
lean_dec(v_a_676_);
v_a_680_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_688_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_688_ == 0)
{
v___x_682_ = v___x_679_;
v_isShared_683_ = v_isSharedCheck_688_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_a_680_);
lean_dec(v___x_679_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_688_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___x_684_; lean_object* v___x_686_; 
v___x_684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_684_, 0, v_a_680_);
if (v_isShared_683_ == 0)
{
lean_ctor_set(v___x_682_, 0, v___x_684_);
v___x_686_ = v___x_682_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v___x_684_);
v___x_686_ = v_reuseFailAlloc_687_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
return v___x_686_;
}
}
}
else
{
lean_object* v_a_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_727_; 
v_a_689_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_727_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_727_ == 0)
{
v___x_691_ = v___x_679_;
v_isShared_692_ = v_isSharedCheck_727_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_a_689_);
lean_dec(v___x_679_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_727_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
uint8_t v___y_694_; uint8_t v___x_725_; 
v___x_725_ = l_Lean_Exception_isInterrupt(v_a_689_);
if (v___x_725_ == 0)
{
uint8_t v___x_726_; 
lean_inc(v_a_689_);
v___x_726_ = l_Lean_Exception_isRuntime(v_a_689_);
v___y_694_ = v___x_726_;
goto v___jp_693_;
}
else
{
v___y_694_ = v___x_725_;
goto v___jp_693_;
}
v___jp_693_:
{
if (v___y_694_ == 0)
{
lean_object* v___x_695_; 
lean_del_object(v___x_691_);
lean_dec(v_a_689_);
v___x_695_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_678_, v___y_694_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_);
if (lean_obj_tag(v___x_695_) == 0)
{
lean_object* v___x_696_; 
lean_dec_ref_known(v___x_695_, 1);
v___x_696_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_676_, v___y_694_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_);
if (lean_obj_tag(v___x_696_) == 0)
{
lean_object* v___x_698_; uint8_t v_isShared_699_; uint8_t v_isSharedCheck_704_; 
v_isSharedCheck_704_ = !lean_is_exclusive(v___x_696_);
if (v_isSharedCheck_704_ == 0)
{
lean_object* v_unused_705_; 
v_unused_705_ = lean_ctor_get(v___x_696_, 0);
lean_dec(v_unused_705_);
v___x_698_ = v___x_696_;
v_isShared_699_ = v_isSharedCheck_704_;
goto v_resetjp_697_;
}
else
{
lean_dec(v___x_696_);
v___x_698_ = lean_box(0);
v_isShared_699_ = v_isSharedCheck_704_;
goto v_resetjp_697_;
}
v_resetjp_697_:
{
lean_object* v___x_700_; lean_object* v___x_702_; 
v___x_700_ = lean_box(0);
if (v_isShared_699_ == 0)
{
lean_ctor_set(v___x_698_, 0, v___x_700_);
v___x_702_ = v___x_698_;
goto v_reusejp_701_;
}
else
{
lean_object* v_reuseFailAlloc_703_; 
v_reuseFailAlloc_703_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_703_, 0, v___x_700_);
v___x_702_ = v_reuseFailAlloc_703_;
goto v_reusejp_701_;
}
v_reusejp_701_:
{
return v___x_702_;
}
}
}
else
{
lean_object* v_a_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_713_; 
v_a_706_ = lean_ctor_get(v___x_696_, 0);
v_isSharedCheck_713_ = !lean_is_exclusive(v___x_696_);
if (v_isSharedCheck_713_ == 0)
{
v___x_708_ = v___x_696_;
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_a_706_);
lean_dec(v___x_696_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
lean_object* v___x_711_; 
if (v_isShared_709_ == 0)
{
v___x_711_ = v___x_708_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_712_; 
v_reuseFailAlloc_712_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_712_, 0, v_a_706_);
v___x_711_ = v_reuseFailAlloc_712_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
return v___x_711_;
}
}
}
}
else
{
lean_object* v_a_714_; lean_object* v___x_716_; uint8_t v_isShared_717_; uint8_t v_isSharedCheck_721_; 
lean_dec(v_a_676_);
v_a_714_ = lean_ctor_get(v___x_695_, 0);
v_isSharedCheck_721_ = !lean_is_exclusive(v___x_695_);
if (v_isSharedCheck_721_ == 0)
{
v___x_716_ = v___x_695_;
v_isShared_717_ = v_isSharedCheck_721_;
goto v_resetjp_715_;
}
else
{
lean_inc(v_a_714_);
lean_dec(v___x_695_);
v___x_716_ = lean_box(0);
v_isShared_717_ = v_isSharedCheck_721_;
goto v_resetjp_715_;
}
v_resetjp_715_:
{
lean_object* v___x_719_; 
if (v_isShared_717_ == 0)
{
v___x_719_ = v___x_716_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_720_; 
v_reuseFailAlloc_720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_720_, 0, v_a_714_);
v___x_719_ = v_reuseFailAlloc_720_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
return v___x_719_;
}
}
}
}
else
{
lean_object* v___x_723_; 
lean_dec(v_a_678_);
lean_dec(v_a_676_);
if (v_isShared_692_ == 0)
{
v___x_723_ = v___x_691_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v_a_689_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
}
}
}
}
}
}
else
{
lean_object* v_a_728_; lean_object* v___x_730_; uint8_t v_isShared_731_; uint8_t v_isSharedCheck_735_; 
lean_dec(v_a_676_);
lean_dec_ref(v_x_665_);
v_a_728_ = lean_ctor_get(v___x_677_, 0);
v_isSharedCheck_735_ = !lean_is_exclusive(v___x_677_);
if (v_isSharedCheck_735_ == 0)
{
v___x_730_ = v___x_677_;
v_isShared_731_ = v_isSharedCheck_735_;
goto v_resetjp_729_;
}
else
{
lean_inc(v_a_728_);
lean_dec(v___x_677_);
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
lean_dec_ref(v_x_665_);
v_a_736_ = lean_ctor_get(v___x_675_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_675_);
if (v_isSharedCheck_743_ == 0)
{
v___x_738_ = v___x_675_;
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_675_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___redArg___boxed(lean_object* v_x_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_){
_start:
{
lean_object* v_res_754_; 
v_res_754_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___redArg(v_x_744_, v___y_745_, v___y_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_748_);
lean_dec_ref(v___y_747_);
lean_dec(v___y_746_);
lean_dec_ref(v___y_745_);
return v_res_754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0(lean_object* v_00_u03b1_755_, lean_object* v_x_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_){
_start:
{
lean_object* v___x_766_; 
v___x_766_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___redArg(v_x_756_, v___y_757_, v___y_758_, v___y_759_, v___y_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___boxed(lean_object* v_00_u03b1_767_, lean_object* v_x_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_){
_start:
{
lean_object* v_res_778_; 
v_res_778_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0(v_00_u03b1_767_, v_x_768_, v___y_769_, v___y_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_, v___y_776_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
lean_dec(v___y_774_);
lean_dec_ref(v___y_773_);
lean_dec(v___y_772_);
lean_dec_ref(v___y_771_);
lean_dec(v___y_770_);
lean_dec_ref(v___y_769_);
return v_res_778_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Hint_hint___lam__0(lean_object* v_r_779_){
_start:
{
lean_object* v_fst_780_; lean_object* v_fst_781_; uint8_t v___x_782_; 
v_fst_780_ = lean_ctor_get(v_r_779_, 0);
v_fst_781_ = lean_ctor_get(v_fst_780_, 0);
v___x_782_ = l_List_isEmpty___redArg(v_fst_781_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__0___boxed(lean_object* v_r_783_){
_start:
{
uint8_t v_res_784_; lean_object* v_r_785_; 
v_res_784_ = lp_mathlib_Mathlib_Tactic_Hint_hint___lam__0(v_r_783_);
lean_dec_ref(v_r_783_);
v_r_785_ = lean_box(v_res_784_);
return v_r_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__1(lean_object* v_t_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_){
_start:
{
lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; 
lean_inc(v_t_786_);
v___x_796_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_796_, 0, v_t_786_);
v___x_797_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_withResetServerInfo___boxed), 11, 2);
lean_closure_set(v___x_797_, 0, lean_box(0));
lean_closure_set(v___x_797_, 1, v___x_796_);
v___x_798_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Hint_hint_spec__0___redArg(v___x_797_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
if (lean_obj_tag(v___x_798_) == 0)
{
lean_object* v_a_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_850_; 
v_a_799_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_850_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_850_ == 0)
{
v___x_801_ = v___x_798_;
v_isShared_802_ = v_isSharedCheck_850_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_a_799_);
lean_dec(v___x_798_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_850_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
if (lean_obj_tag(v_a_799_) == 1)
{
lean_object* v_val_803_; lean_object* v___x_805_; uint8_t v_isShared_806_; uint8_t v_isSharedCheck_845_; 
v_val_803_ = lean_ctor_get(v_a_799_, 0);
v_isSharedCheck_845_ = !lean_is_exclusive(v_a_799_);
if (v_isSharedCheck_845_ == 0)
{
v___x_805_ = v_a_799_;
v_isShared_806_ = v_isSharedCheck_845_;
goto v_resetjp_804_;
}
else
{
lean_inc(v_val_803_);
lean_dec(v_a_799_);
v___x_805_ = lean_box(0);
v_isShared_806_ = v_isSharedCheck_845_;
goto v_resetjp_804_;
}
v_resetjp_804_:
{
lean_object* v_msgs_807_; lean_object* v_trees_808_; uint8_t v___x_809_; 
v_msgs_807_ = lean_ctor_get(v_val_803_, 1);
lean_inc_ref(v_msgs_807_);
v_trees_808_ = lean_ctor_get(v_val_803_, 2);
lean_inc_ref(v_trees_808_);
lean_dec(v_val_803_);
v___x_809_ = l_Lean_MessageLog_hasErrors(v_msgs_807_);
lean_dec_ref(v_msgs_807_);
if (v___x_809_ == 0)
{
lean_object* v___x_810_; 
lean_del_object(v___x_801_);
v___x_810_ = l_Lean_Elab_Tactic_getGoals___redArg(v___y_788_);
if (lean_obj_tag(v___x_810_) == 0)
{
lean_object* v_a_811_; lean_object* v___x_812_; 
v_a_811_ = lean_ctor_get(v___x_810_, 0);
lean_inc(v_a_811_);
lean_dec_ref_known(v___x_810_, 1);
v___x_812_ = lp_mathlib_Mathlib_Tactic_Hint_suggestion(v_t_786_, v_trees_808_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
lean_dec_ref(v_trees_808_);
if (lean_obj_tag(v___x_812_) == 0)
{
lean_object* v_a_813_; lean_object* v___x_815_; uint8_t v_isShared_816_; uint8_t v_isSharedCheck_824_; 
v_a_813_ = lean_ctor_get(v___x_812_, 0);
v_isSharedCheck_824_ = !lean_is_exclusive(v___x_812_);
if (v_isSharedCheck_824_ == 0)
{
v___x_815_ = v___x_812_;
v_isShared_816_ = v_isSharedCheck_824_;
goto v_resetjp_814_;
}
else
{
lean_inc(v_a_813_);
lean_dec(v___x_812_);
v___x_815_ = lean_box(0);
v_isShared_816_ = v_isSharedCheck_824_;
goto v_resetjp_814_;
}
v_resetjp_814_:
{
lean_object* v___x_817_; lean_object* v___x_819_; 
v___x_817_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_817_, 0, v_a_811_);
lean_ctor_set(v___x_817_, 1, v_a_813_);
if (v_isShared_806_ == 0)
{
lean_ctor_set(v___x_805_, 0, v___x_817_);
v___x_819_ = v___x_805_;
goto v_reusejp_818_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v___x_817_);
v___x_819_ = v_reuseFailAlloc_823_;
goto v_reusejp_818_;
}
v_reusejp_818_:
{
lean_object* v___x_821_; 
if (v_isShared_816_ == 0)
{
lean_ctor_set(v___x_815_, 0, v___x_819_);
v___x_821_ = v___x_815_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_822_; 
v_reuseFailAlloc_822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_822_, 0, v___x_819_);
v___x_821_ = v_reuseFailAlloc_822_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
return v___x_821_;
}
}
}
}
else
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_832_; 
lean_dec(v_a_811_);
lean_del_object(v___x_805_);
v_a_825_ = lean_ctor_get(v___x_812_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_812_);
if (v_isSharedCheck_832_ == 0)
{
v___x_827_ = v___x_812_;
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_812_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
lean_object* v___x_830_; 
if (v_isShared_828_ == 0)
{
v___x_830_ = v___x_827_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v_a_825_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
}
else
{
lean_object* v_a_833_; lean_object* v___x_835_; uint8_t v_isShared_836_; uint8_t v_isSharedCheck_840_; 
lean_dec_ref(v_trees_808_);
lean_del_object(v___x_805_);
lean_dec(v_t_786_);
v_a_833_ = lean_ctor_get(v___x_810_, 0);
v_isSharedCheck_840_ = !lean_is_exclusive(v___x_810_);
if (v_isSharedCheck_840_ == 0)
{
v___x_835_ = v___x_810_;
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
else
{
lean_inc(v_a_833_);
lean_dec(v___x_810_);
v___x_835_ = lean_box(0);
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
v_resetjp_834_:
{
lean_object* v___x_838_; 
if (v_isShared_836_ == 0)
{
v___x_838_ = v___x_835_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v_a_833_);
v___x_838_ = v_reuseFailAlloc_839_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
return v___x_838_;
}
}
}
}
else
{
lean_object* v___x_841_; lean_object* v___x_843_; 
lean_dec_ref(v_trees_808_);
lean_del_object(v___x_805_);
lean_dec(v_t_786_);
v___x_841_ = lean_box(0);
if (v_isShared_802_ == 0)
{
lean_ctor_set(v___x_801_, 0, v___x_841_);
v___x_843_ = v___x_801_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v___x_841_);
v___x_843_ = v_reuseFailAlloc_844_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
return v___x_843_;
}
}
}
}
else
{
lean_object* v___x_846_; lean_object* v___x_848_; 
lean_dec(v_a_799_);
lean_dec(v_t_786_);
v___x_846_ = lean_box(0);
if (v_isShared_802_ == 0)
{
lean_ctor_set(v___x_801_, 0, v___x_846_);
v___x_848_ = v___x_801_;
goto v_reusejp_847_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v___x_846_);
v___x_848_ = v_reuseFailAlloc_849_;
goto v_reusejp_847_;
}
v_reusejp_847_:
{
return v___x_848_;
}
}
}
}
else
{
lean_object* v_a_851_; lean_object* v___x_853_; uint8_t v_isShared_854_; uint8_t v_isSharedCheck_858_; 
lean_dec(v_t_786_);
v_a_851_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_858_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_858_ == 0)
{
v___x_853_ = v___x_798_;
v_isShared_854_ = v_isSharedCheck_858_;
goto v_resetjp_852_;
}
else
{
lean_inc(v_a_851_);
lean_dec(v___x_798_);
v___x_853_ = lean_box(0);
v_isShared_854_ = v_isSharedCheck_858_;
goto v_resetjp_852_;
}
v_resetjp_852_:
{
lean_object* v___x_856_; 
if (v_isShared_854_ == 0)
{
v___x_856_ = v___x_853_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_857_; 
v_reuseFailAlloc_857_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_857_, 0, v_a_851_);
v___x_856_ = v_reuseFailAlloc_857_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
return v___x_856_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__1___boxed(lean_object* v_t_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_){
_start:
{
lean_object* v_res_869_; 
v_res_869_ = lp_mathlib_Mathlib_Tactic_Hint_hint___lam__1(v_t_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_, v___y_865_, v___y_866_, v___y_867_);
lean_dec(v___y_867_);
lean_dec_ref(v___y_866_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
lean_dec(v___y_863_);
lean_dec_ref(v___y_862_);
lean_dec(v___y_861_);
lean_dec_ref(v___y_860_);
return v_res_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Hint_hint_spec__1(lean_object* v_a_870_, lean_object* v_a_871_){
_start:
{
if (lean_obj_tag(v_a_870_) == 0)
{
lean_object* v___x_872_; 
v___x_872_ = l_List_reverse___redArg(v_a_871_);
return v___x_872_;
}
else
{
lean_object* v_head_873_; lean_object* v_tail_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_883_; 
v_head_873_ = lean_ctor_get(v_a_870_, 0);
v_tail_874_ = lean_ctor_get(v_a_870_, 1);
v_isSharedCheck_883_ = !lean_is_exclusive(v_a_870_);
if (v_isSharedCheck_883_ == 0)
{
v___x_876_ = v_a_870_;
v_isShared_877_ = v_isSharedCheck_883_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_tail_874_);
lean_inc(v_head_873_);
lean_dec(v_a_870_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_883_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
lean_object* v_snd_878_; lean_object* v___x_880_; 
v_snd_878_ = lean_ctor_get(v_head_873_, 1);
lean_inc(v_snd_878_);
lean_dec(v_head_873_);
if (v_isShared_877_ == 0)
{
lean_ctor_set(v___x_876_, 1, v_a_871_);
lean_ctor_set(v___x_876_, 0, v_snd_878_);
v___x_880_ = v___x_876_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_snd_878_);
lean_ctor_set(v_reuseFailAlloc_882_, 1, v_a_871_);
v___x_880_ = v_reuseFailAlloc_882_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
v_a_870_ = v_tail_874_;
v_a_871_ = v___x_880_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(lean_object* v_x_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_){
_start:
{
switch(lean_obj_tag(v_x_884_))
{
case 0:
{
lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_894_ = lean_box(0);
v___x_895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_895_, 0, v___x_894_);
return v___x_895_;
}
case 1:
{
lean_object* v_a_896_; lean_object* v_a_897_; lean_object* v___x_899_; uint8_t v_isShared_900_; uint8_t v_isSharedCheck_906_; 
v_a_896_ = lean_ctor_get(v_x_884_, 0);
v_a_897_ = lean_ctor_get(v_x_884_, 1);
v_isSharedCheck_906_ = !lean_is_exclusive(v_x_884_);
if (v_isSharedCheck_906_ == 0)
{
v___x_899_ = v_x_884_;
v_isShared_900_ = v_isSharedCheck_906_;
goto v_resetjp_898_;
}
else
{
lean_inc(v_a_897_);
lean_inc(v_a_896_);
lean_dec(v_x_884_);
v___x_899_ = lean_box(0);
v_isShared_900_ = v_isSharedCheck_906_;
goto v_resetjp_898_;
}
v_resetjp_898_:
{
lean_object* v___x_902_; 
if (v_isShared_900_ == 0)
{
lean_ctor_set_tag(v___x_899_, 0);
v___x_902_ = v___x_899_;
goto v_reusejp_901_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v_a_896_);
lean_ctor_set(v_reuseFailAlloc_905_, 1, v_a_897_);
v___x_902_ = v_reuseFailAlloc_905_;
goto v_reusejp_901_;
}
v_reusejp_901_:
{
lean_object* v___x_903_; lean_object* v___x_904_; 
v___x_903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_903_, 0, v___x_902_);
v___x_904_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_904_, 0, v___x_903_);
return v___x_904_;
}
}
}
case 2:
{
lean_object* v_a_907_; lean_object* v___x_908_; 
v_a_907_ = lean_ctor_get(v_x_884_, 0);
lean_inc_ref(v_a_907_);
lean_dec_ref_known(v_x_884_, 1);
v___x_908_ = lean_thunk_get_own(v_a_907_);
lean_dec_ref(v_a_907_);
v_x_884_ = v___x_908_;
goto _start;
}
default: 
{
lean_object* v_a_910_; lean_object* v___x_911_; lean_object* v___x_912_; 
v_a_910_ = lean_ctor_get(v_x_884_, 0);
lean_inc(v_a_910_);
lean_dec_ref_known(v_x_884_, 1);
v___x_911_ = lean_box(0);
lean_inc(v___y_892_);
lean_inc_ref(v___y_891_);
lean_inc(v___y_890_);
lean_inc_ref(v___y_889_);
lean_inc(v___y_888_);
lean_inc_ref(v___y_887_);
lean_inc(v___y_886_);
lean_inc_ref(v___y_885_);
v___x_912_ = lean_apply_10(v_a_910_, v___x_911_, v___y_885_, v___y_886_, v___y_887_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, lean_box(0));
if (lean_obj_tag(v___x_912_) == 0)
{
lean_object* v_a_913_; 
v_a_913_ = lean_ctor_get(v___x_912_, 0);
lean_inc(v_a_913_);
lean_dec_ref_known(v___x_912_, 1);
v_x_884_ = v_a_913_;
goto _start;
}
else
{
lean_object* v_a_915_; lean_object* v___x_917_; uint8_t v_isShared_918_; uint8_t v_isSharedCheck_922_; 
v_a_915_ = lean_ctor_get(v___x_912_, 0);
v_isSharedCheck_922_ = !lean_is_exclusive(v___x_912_);
if (v_isSharedCheck_922_ == 0)
{
v___x_917_ = v___x_912_;
v_isShared_918_ = v_isSharedCheck_922_;
goto v_resetjp_916_;
}
else
{
lean_inc(v_a_915_);
lean_dec(v___x_912_);
v___x_917_ = lean_box(0);
v_isShared_918_ = v_isSharedCheck_922_;
goto v_resetjp_916_;
}
v_resetjp_916_:
{
lean_object* v___x_920_; 
if (v_isShared_918_ == 0)
{
v___x_920_ = v___x_917_;
goto v_reusejp_919_;
}
else
{
lean_object* v_reuseFailAlloc_921_; 
v_reuseFailAlloc_921_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_921_, 0, v_a_915_);
v___x_920_ = v_reuseFailAlloc_921_;
goto v_reusejp_919_;
}
v_reusejp_919_:
{
return v___x_920_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg___boxed(lean_object* v_x_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
lean_object* v_res_933_; 
v_res_933_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(v_x_923_, v___y_924_, v___y_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_);
lean_dec(v___y_931_);
lean_dec_ref(v___y_930_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
lean_dec(v___y_927_);
lean_dec_ref(v___y_926_);
lean_dec(v___y_925_);
lean_dec_ref(v___y_924_);
return v_res_933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___redArg(lean_object* v_as_934_, lean_object* v_init_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_){
_start:
{
lean_object* v___x_945_; 
v___x_945_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(v_as_934_, v___y_936_, v___y_937_, v___y_938_, v___y_939_, v___y_940_, v___y_941_, v___y_942_, v___y_943_);
if (lean_obj_tag(v___x_945_) == 0)
{
lean_object* v_a_946_; lean_object* v___x_948_; uint8_t v_isShared_949_; uint8_t v_isSharedCheck_958_; 
v_a_946_ = lean_ctor_get(v___x_945_, 0);
v_isSharedCheck_958_ = !lean_is_exclusive(v___x_945_);
if (v_isSharedCheck_958_ == 0)
{
v___x_948_ = v___x_945_;
v_isShared_949_ = v_isSharedCheck_958_;
goto v_resetjp_947_;
}
else
{
lean_inc(v_a_946_);
lean_dec(v___x_945_);
v___x_948_ = lean_box(0);
v_isShared_949_ = v_isSharedCheck_958_;
goto v_resetjp_947_;
}
v_resetjp_947_:
{
if (lean_obj_tag(v_a_946_) == 0)
{
lean_object* v___x_951_; 
if (v_isShared_949_ == 0)
{
lean_ctor_set(v___x_948_, 0, v_init_935_);
v___x_951_ = v___x_948_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v_init_935_);
v___x_951_ = v_reuseFailAlloc_952_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
return v___x_951_;
}
}
else
{
lean_object* v_val_953_; lean_object* v_fst_954_; lean_object* v_snd_955_; lean_object* v_r_956_; 
lean_del_object(v___x_948_);
v_val_953_ = lean_ctor_get(v_a_946_, 0);
lean_inc(v_val_953_);
lean_dec_ref_known(v_a_946_, 1);
v_fst_954_ = lean_ctor_get(v_val_953_, 0);
lean_inc(v_fst_954_);
v_snd_955_ = lean_ctor_get(v_val_953_, 1);
lean_inc(v_snd_955_);
lean_dec(v_val_953_);
v_r_956_ = lean_array_push(v_init_935_, v_fst_954_);
v_as_934_ = v_snd_955_;
v_init_935_ = v_r_956_;
goto _start;
}
}
}
else
{
lean_object* v_a_959_; lean_object* v___x_961_; uint8_t v_isShared_962_; uint8_t v_isSharedCheck_966_; 
lean_dec_ref(v_init_935_);
v_a_959_ = lean_ctor_get(v___x_945_, 0);
v_isSharedCheck_966_ = !lean_is_exclusive(v___x_945_);
if (v_isSharedCheck_966_ == 0)
{
v___x_961_ = v___x_945_;
v_isShared_962_ = v_isSharedCheck_966_;
goto v_resetjp_960_;
}
else
{
lean_inc(v_a_959_);
lean_dec(v___x_945_);
v___x_961_ = lean_box(0);
v_isShared_962_ = v_isSharedCheck_966_;
goto v_resetjp_960_;
}
v_resetjp_960_:
{
lean_object* v___x_964_; 
if (v_isShared_962_ == 0)
{
v___x_964_ = v___x_961_;
goto v_reusejp_963_;
}
else
{
lean_object* v_reuseFailAlloc_965_; 
v_reuseFailAlloc_965_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_965_, 0, v_a_959_);
v___x_964_ = v_reuseFailAlloc_965_;
goto v_reusejp_963_;
}
v_reusejp_963_:
{
return v___x_964_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___redArg___boxed(lean_object* v_as_967_, lean_object* v_init_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___redArg(v_as_967_, v_init_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_);
lean_dec(v___y_976_);
lean_dec_ref(v___y_975_);
lean_dec(v___y_974_);
lean_dec_ref(v___y_973_);
lean_dec(v___y_972_);
lean_dec_ref(v___y_971_);
lean_dec(v___y_970_);
lean_dec_ref(v___y_969_);
return v_res_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg(lean_object* v_L_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
lean_object* v_r_991_; lean_object* v___x_992_; 
v_r_991_ = ((lean_object*)(lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg___closed__0));
v___x_992_ = lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___redArg(v_L_981_, v_r_991_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
return v___x_992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg___boxed(lean_object* v_L_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_){
_start:
{
lean_object* v_res_1003_; 
v_res_1003_ = lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg(v_L_993_, v___y_994_, v___y_995_, v___y_996_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_);
lean_dec(v___y_1001_);
lean_dec_ref(v___y_1000_);
lean_dec(v___y_999_);
lean_dec_ref(v___y_998_);
lean_dec(v___y_997_);
lean_dec_ref(v___y_996_);
lean_dec(v___y_995_);
lean_dec_ref(v___y_994_);
return v_res_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Hint_hint_spec__6(size_t v_sz_1004_, size_t v_i_1005_, lean_object* v_bs_1006_){
_start:
{
uint8_t v___x_1007_; 
v___x_1007_ = lean_usize_dec_lt(v_i_1005_, v_sz_1004_);
if (v___x_1007_ == 0)
{
return v_bs_1006_;
}
else
{
lean_object* v_v_1008_; lean_object* v_fst_1009_; lean_object* v_snd_1010_; lean_object* v___x_1011_; lean_object* v_bs_x27_1012_; size_t v___x_1013_; size_t v___x_1014_; lean_object* v___x_1015_; 
v_v_1008_ = lean_array_uget_borrowed(v_bs_1006_, v_i_1005_);
v_fst_1009_ = lean_ctor_get(v_v_1008_, 0);
v_snd_1010_ = lean_ctor_get(v_fst_1009_, 1);
lean_inc(v_snd_1010_);
v___x_1011_ = lean_unsigned_to_nat(0u);
v_bs_x27_1012_ = lean_array_uset(v_bs_1006_, v_i_1005_, v___x_1011_);
v___x_1013_ = ((size_t)1ULL);
v___x_1014_ = lean_usize_add(v_i_1005_, v___x_1013_);
v___x_1015_ = lean_array_uset(v_bs_x27_1012_, v_i_1005_, v_snd_1010_);
v_i_1005_ = v___x_1014_;
v_bs_1006_ = v___x_1015_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Hint_hint_spec__6___boxed(lean_object* v_sz_1017_, lean_object* v_i_1018_, lean_object* v_bs_1019_){
_start:
{
size_t v_sz_boxed_1020_; size_t v_i_boxed_1021_; lean_object* v_res_1022_; 
v_sz_boxed_1020_ = lean_unbox_usize(v_sz_1017_);
lean_dec(v_sz_1017_);
v_i_boxed_1021_ = lean_unbox_usize(v_i_1018_);
lean_dec(v_i_1018_);
v_res_1022_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Hint_hint_spec__6(v_sz_boxed_1020_, v_i_boxed_1021_, v_bs_1019_);
return v_res_1022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg___lam__0(lean_object* v_f_1023_, lean_object* v_a_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_){
_start:
{
lean_object* v___x_1034_; lean_object* v___x_1035_; 
v___x_1034_ = lean_apply_1(v_f_1023_, v_a_1024_);
v___x_1035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1035_, 0, v___x_1034_);
return v___x_1035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg___lam__0___boxed(lean_object* v_f_1036_, lean_object* v_a_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v_res_1047_; 
v_res_1047_ = lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg___lam__0(v_f_1036_, v_a_1037_, v___y_1038_, v___y_1039_, v___y_1040_, v___y_1041_, v___y_1042_, v___y_1043_, v___y_1044_, v___y_1045_);
lean_dec(v___y_1045_);
lean_dec_ref(v___y_1044_);
lean_dec(v___y_1043_);
lean_dec_ref(v___y_1042_);
lean_dec(v___y_1041_);
lean_dec_ref(v___y_1040_);
lean_dec(v___y_1039_);
lean_dec_ref(v___y_1038_);
return v_res_1047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg___lam__0(lean_object* v_xs_1048_, lean_object* v_f_1049_, lean_object* v_x_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_){
_start:
{
lean_object* v___x_1060_; 
v___x_1060_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(v_xs_1048_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_, v___y_1058_);
if (lean_obj_tag(v___x_1060_) == 0)
{
lean_object* v_a_1061_; lean_object* v___x_1063_; uint8_t v_isShared_1064_; uint8_t v_isSharedCheck_1101_; 
v_a_1061_ = lean_ctor_get(v___x_1060_, 0);
v_isSharedCheck_1101_ = !lean_is_exclusive(v___x_1060_);
if (v_isSharedCheck_1101_ == 0)
{
v___x_1063_ = v___x_1060_;
v_isShared_1064_ = v_isSharedCheck_1101_;
goto v_resetjp_1062_;
}
else
{
lean_inc(v_a_1061_);
lean_dec(v___x_1060_);
v___x_1063_ = lean_box(0);
v_isShared_1064_ = v_isSharedCheck_1101_;
goto v_resetjp_1062_;
}
v_resetjp_1062_:
{
if (lean_obj_tag(v_a_1061_) == 0)
{
lean_object* v___x_1065_; lean_object* v___x_1067_; 
lean_dec_ref(v_f_1049_);
v___x_1065_ = lean_box(0);
if (v_isShared_1064_ == 0)
{
lean_ctor_set(v___x_1063_, 0, v___x_1065_);
v___x_1067_ = v___x_1063_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v___x_1065_);
v___x_1067_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
return v___x_1067_;
}
}
else
{
lean_object* v_val_1069_; lean_object* v_fst_1070_; lean_object* v_snd_1071_; lean_object* v___x_1073_; uint8_t v_isShared_1074_; uint8_t v_isSharedCheck_1100_; 
lean_del_object(v___x_1063_);
v_val_1069_ = lean_ctor_get(v_a_1061_, 0);
lean_inc(v_val_1069_);
lean_dec_ref_known(v_a_1061_, 1);
v_fst_1070_ = lean_ctor_get(v_val_1069_, 0);
v_snd_1071_ = lean_ctor_get(v_val_1069_, 1);
v_isSharedCheck_1100_ = !lean_is_exclusive(v_val_1069_);
if (v_isSharedCheck_1100_ == 0)
{
v___x_1073_ = v_val_1069_;
v_isShared_1074_ = v_isSharedCheck_1100_;
goto v_resetjp_1072_;
}
else
{
lean_inc(v_snd_1071_);
lean_inc(v_fst_1070_);
lean_dec(v_val_1069_);
v___x_1073_ = lean_box(0);
v_isShared_1074_ = v_isSharedCheck_1100_;
goto v_resetjp_1072_;
}
v_resetjp_1072_:
{
lean_object* v___x_1075_; 
lean_inc_ref(v_f_1049_);
lean_inc(v___y_1058_);
lean_inc_ref(v___y_1057_);
lean_inc(v___y_1056_);
lean_inc_ref(v___y_1055_);
lean_inc(v___y_1054_);
lean_inc_ref(v___y_1053_);
lean_inc(v___y_1052_);
lean_inc_ref(v___y_1051_);
lean_inc(v_fst_1070_);
v___x_1075_ = lean_apply_10(v_f_1049_, v_fst_1070_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_, v___y_1058_, lean_box(0));
if (lean_obj_tag(v___x_1075_) == 0)
{
lean_object* v_a_1076_; lean_object* v___x_1078_; uint8_t v_isShared_1079_; uint8_t v_isSharedCheck_1091_; 
v_a_1076_ = lean_ctor_get(v___x_1075_, 0);
v_isSharedCheck_1091_ = !lean_is_exclusive(v___x_1075_);
if (v_isSharedCheck_1091_ == 0)
{
v___x_1078_ = v___x_1075_;
v_isShared_1079_ = v_isSharedCheck_1091_;
goto v_resetjp_1077_;
}
else
{
lean_inc(v_a_1076_);
lean_dec(v___x_1075_);
v___x_1078_ = lean_box(0);
v_isShared_1079_ = v_isSharedCheck_1091_;
goto v_resetjp_1077_;
}
v_resetjp_1077_:
{
lean_object* v___y_1081_; uint8_t v___x_1088_; 
v___x_1088_ = lean_unbox(v_a_1076_);
lean_dec(v_a_1076_);
if (v___x_1088_ == 0)
{
lean_object* v___x_1089_; 
v___x_1089_ = lp_mathlib_MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8___redArg(v_snd_1071_, v_f_1049_);
v___y_1081_ = v___x_1089_;
goto v___jp_1080_;
}
else
{
lean_object* v___x_1090_; 
lean_dec(v_snd_1071_);
lean_dec_ref(v_f_1049_);
v___x_1090_ = lean_box(0);
v___y_1081_ = v___x_1090_;
goto v___jp_1080_;
}
v___jp_1080_:
{
lean_object* v___x_1083_; 
if (v_isShared_1074_ == 0)
{
lean_ctor_set_tag(v___x_1073_, 1);
lean_ctor_set(v___x_1073_, 1, v___y_1081_);
v___x_1083_ = v___x_1073_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1087_; 
v_reuseFailAlloc_1087_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1087_, 0, v_fst_1070_);
lean_ctor_set(v_reuseFailAlloc_1087_, 1, v___y_1081_);
v___x_1083_ = v_reuseFailAlloc_1087_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
lean_object* v___x_1085_; 
if (v_isShared_1079_ == 0)
{
lean_ctor_set(v___x_1078_, 0, v___x_1083_);
v___x_1085_ = v___x_1078_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1086_; 
v_reuseFailAlloc_1086_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1086_, 0, v___x_1083_);
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
}
else
{
lean_object* v_a_1092_; lean_object* v___x_1094_; uint8_t v_isShared_1095_; uint8_t v_isSharedCheck_1099_; 
lean_del_object(v___x_1073_);
lean_dec(v_snd_1071_);
lean_dec(v_fst_1070_);
lean_dec_ref(v_f_1049_);
v_a_1092_ = lean_ctor_get(v___x_1075_, 0);
v_isSharedCheck_1099_ = !lean_is_exclusive(v___x_1075_);
if (v_isSharedCheck_1099_ == 0)
{
v___x_1094_ = v___x_1075_;
v_isShared_1095_ = v_isSharedCheck_1099_;
goto v_resetjp_1093_;
}
else
{
lean_inc(v_a_1092_);
lean_dec(v___x_1075_);
v___x_1094_ = lean_box(0);
v_isShared_1095_ = v_isSharedCheck_1099_;
goto v_resetjp_1093_;
}
v_resetjp_1093_:
{
lean_object* v___x_1097_; 
if (v_isShared_1095_ == 0)
{
v___x_1097_ = v___x_1094_;
goto v_reusejp_1096_;
}
else
{
lean_object* v_reuseFailAlloc_1098_; 
v_reuseFailAlloc_1098_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1098_, 0, v_a_1092_);
v___x_1097_ = v_reuseFailAlloc_1098_;
goto v_reusejp_1096_;
}
v_reusejp_1096_:
{
return v___x_1097_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1109_; 
lean_dec_ref(v_f_1049_);
v_a_1102_ = lean_ctor_get(v___x_1060_, 0);
v_isSharedCheck_1109_ = !lean_is_exclusive(v___x_1060_);
if (v_isSharedCheck_1109_ == 0)
{
v___x_1104_ = v___x_1060_;
v_isShared_1105_ = v_isSharedCheck_1109_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_a_1102_);
lean_dec(v___x_1060_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1109_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
lean_object* v___x_1107_; 
if (v_isShared_1105_ == 0)
{
v___x_1107_ = v___x_1104_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v_a_1102_);
v___x_1107_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
return v___x_1107_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg___lam__0___boxed(lean_object* v_xs_1110_, lean_object* v_f_1111_, lean_object* v_x_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg___lam__0(v_xs_1110_, v_f_1111_, v_x_1112_, v___y_1113_, v___y_1114_, v___y_1115_, v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_);
lean_dec(v___y_1120_);
lean_dec_ref(v___y_1119_);
lean_dec(v___y_1118_);
lean_dec_ref(v___y_1117_);
lean_dec(v___y_1116_);
lean_dec_ref(v___y_1115_);
lean_dec(v___y_1114_);
lean_dec_ref(v___y_1113_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg(lean_object* v_f_1123_, lean_object* v_xs_1124_){
_start:
{
lean_object* v___f_1125_; lean_object* v___x_1126_; 
v___f_1125_ = lean_alloc_closure((void*)(lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg___lam__0___boxed), 12, 2);
lean_closure_set(v___f_1125_, 0, v_xs_1124_);
lean_closure_set(v___f_1125_, 1, v_f_1123_);
v___x_1126_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1126_, 0, v___f_1125_);
return v___x_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8___redArg(lean_object* v_L_1127_, lean_object* v_f_1128_){
_start:
{
lean_object* v___x_1129_; 
v___x_1129_ = lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg(v_f_1128_, v_L_1127_);
return v___x_1129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg(lean_object* v_L_1130_, lean_object* v_f_1131_){
_start:
{
lean_object* v___f_1132_; lean_object* v___x_1133_; 
v___f_1132_ = lean_alloc_closure((void*)(lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg___lam__0___boxed), 11, 1);
lean_closure_set(v___f_1132_, 0, v_f_1131_);
v___x_1133_ = lp_mathlib_MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8___redArg(v_L_1130_, v___f_1132_);
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg___lam__0(lean_object* v_head_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_){
_start:
{
lean_object* v___x_1144_; 
v___x_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1144_, 0, v_head_1134_);
return v___x_1144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg___lam__0___boxed(lean_object* v_head_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_){
_start:
{
lean_object* v_res_1155_; 
v_res_1155_ = lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg___lam__0(v_head_1145_, v___y_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_);
lean_dec(v___y_1153_);
lean_dec_ref(v___y_1152_);
lean_dec(v___y_1151_);
lean_dec_ref(v___y_1150_);
lean_dec(v___y_1149_);
lean_dec_ref(v___y_1148_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
return v_res_1155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg(lean_object* v_a_1156_, lean_object* v_a_1157_){
_start:
{
if (lean_obj_tag(v_a_1156_) == 0)
{
lean_object* v___x_1158_; 
v___x_1158_ = l_List_reverse___redArg(v_a_1157_);
return v___x_1158_;
}
else
{
lean_object* v_head_1159_; lean_object* v_tail_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1169_; 
v_head_1159_ = lean_ctor_get(v_a_1156_, 0);
v_tail_1160_ = lean_ctor_get(v_a_1156_, 1);
v_isSharedCheck_1169_ = !lean_is_exclusive(v_a_1156_);
if (v_isSharedCheck_1169_ == 0)
{
v___x_1162_ = v_a_1156_;
v_isShared_1163_ = v_isSharedCheck_1169_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_tail_1160_);
lean_inc(v_head_1159_);
lean_dec(v_a_1156_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1169_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___f_1164_; lean_object* v___x_1166_; 
v___f_1164_ = lean_alloc_closure((void*)(lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1164_, 0, v_head_1159_);
if (v_isShared_1163_ == 0)
{
lean_ctor_set(v___x_1162_, 1, v_a_1157_);
lean_ctor_set(v___x_1162_, 0, v___f_1164_);
v___x_1166_ = v___x_1162_;
goto v_reusejp_1165_;
}
else
{
lean_object* v_reuseFailAlloc_1168_; 
v_reuseFailAlloc_1168_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1168_, 0, v___f_1164_);
lean_ctor_set(v_reuseFailAlloc_1168_, 1, v_a_1157_);
v___x_1166_ = v_reuseFailAlloc_1168_;
goto v_reusejp_1165_;
}
v_reusejp_1165_:
{
v_a_1156_ = v_tail_1160_;
v_a_1157_ = v___x_1166_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg___lam__0(lean_object* v_L_1170_, lean_object* v_x_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_){
_start:
{
lean_object* v___x_1181_; lean_object* v___x_1182_; 
v___x_1181_ = lean_box(0);
lean_inc(v___y_1179_);
lean_inc_ref(v___y_1178_);
lean_inc(v___y_1177_);
lean_inc_ref(v___y_1176_);
lean_inc(v___y_1175_);
lean_inc_ref(v___y_1174_);
lean_inc(v___y_1173_);
lean_inc_ref(v___y_1172_);
v___x_1182_ = lean_apply_10(v_L_1170_, v___x_1181_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, lean_box(0));
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_object* v_a_1183_; lean_object* v___x_1185_; uint8_t v_isShared_1186_; uint8_t v_isSharedCheck_1190_; 
v_a_1183_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1190_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1190_ == 0)
{
v___x_1185_ = v___x_1182_;
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
else
{
lean_inc(v_a_1183_);
lean_dec(v___x_1182_);
v___x_1185_ = lean_box(0);
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
v_resetjp_1184_:
{
lean_object* v___x_1188_; 
if (v_isShared_1186_ == 0)
{
v___x_1188_ = v___x_1185_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v_a_1183_);
v___x_1188_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
return v___x_1188_;
}
}
}
else
{
lean_object* v_a_1191_; lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1198_; 
v_a_1191_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1193_ = v___x_1182_;
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
else
{
lean_inc(v_a_1191_);
lean_dec(v___x_1182_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v___x_1196_; 
if (v_isShared_1194_ == 0)
{
v___x_1196_ = v___x_1193_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v_a_1191_);
v___x_1196_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
return v___x_1196_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg___lam__0___boxed(lean_object* v_L_1199_, lean_object* v_x_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_, lean_object* v___y_1209_){
_start:
{
lean_object* v_res_1210_; 
v_res_1210_ = lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg___lam__0(v_L_1199_, v_x_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_, v___y_1208_);
lean_dec(v___y_1208_);
lean_dec_ref(v___y_1207_);
lean_dec(v___y_1206_);
lean_dec_ref(v___y_1205_);
lean_dec(v___y_1204_);
lean_dec_ref(v___y_1203_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
return v_res_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg(lean_object* v_L_1211_){
_start:
{
lean_object* v___f_1212_; lean_object* v___x_1213_; 
v___f_1212_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg___lam__0___boxed), 11, 1);
lean_closure_set(v___f_1212_, 0, v_L_1211_);
v___x_1213_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1213_, 0, v___f_1212_);
return v___x_1213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg___lam__0(lean_object* v_head_1214_, lean_object* v_tail_1215_, lean_object* v_x_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_){
_start:
{
lean_object* v___x_1226_; 
lean_inc(v___y_1224_);
lean_inc_ref(v___y_1223_);
lean_inc(v___y_1222_);
lean_inc_ref(v___y_1221_);
lean_inc(v___y_1220_);
lean_inc_ref(v___y_1219_);
lean_inc(v___y_1218_);
lean_inc_ref(v___y_1217_);
v___x_1226_ = lean_apply_9(v_head_1214_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_, lean_box(0));
if (lean_obj_tag(v___x_1226_) == 0)
{
lean_object* v_a_1227_; lean_object* v___x_1229_; uint8_t v_isShared_1230_; uint8_t v_isSharedCheck_1236_; 
v_a_1227_ = lean_ctor_get(v___x_1226_, 0);
v_isSharedCheck_1236_ = !lean_is_exclusive(v___x_1226_);
if (v_isSharedCheck_1236_ == 0)
{
v___x_1229_ = v___x_1226_;
v_isShared_1230_ = v_isSharedCheck_1236_;
goto v_resetjp_1228_;
}
else
{
lean_inc(v_a_1227_);
lean_dec(v___x_1226_);
v___x_1229_ = lean_box(0);
v_isShared_1230_ = v_isSharedCheck_1236_;
goto v_resetjp_1228_;
}
v_resetjp_1228_:
{
lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1234_; 
v___x_1231_ = lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg(v_tail_1215_);
v___x_1232_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1232_, 0, v_a_1227_);
lean_ctor_set(v___x_1232_, 1, v___x_1231_);
if (v_isShared_1230_ == 0)
{
lean_ctor_set(v___x_1229_, 0, v___x_1232_);
v___x_1234_ = v___x_1229_;
goto v_reusejp_1233_;
}
else
{
lean_object* v_reuseFailAlloc_1235_; 
v_reuseFailAlloc_1235_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1235_, 0, v___x_1232_);
v___x_1234_ = v_reuseFailAlloc_1235_;
goto v_reusejp_1233_;
}
v_reusejp_1233_:
{
return v___x_1234_;
}
}
}
else
{
lean_object* v_a_1237_; lean_object* v___x_1239_; uint8_t v_isShared_1240_; uint8_t v_isSharedCheck_1244_; 
lean_dec(v_tail_1215_);
v_a_1237_ = lean_ctor_get(v___x_1226_, 0);
v_isSharedCheck_1244_ = !lean_is_exclusive(v___x_1226_);
if (v_isSharedCheck_1244_ == 0)
{
v___x_1239_ = v___x_1226_;
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
else
{
lean_inc(v_a_1237_);
lean_dec(v___x_1226_);
v___x_1239_ = lean_box(0);
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
v_resetjp_1238_:
{
lean_object* v___x_1242_; 
if (v_isShared_1240_ == 0)
{
v___x_1242_ = v___x_1239_;
goto v_reusejp_1241_;
}
else
{
lean_object* v_reuseFailAlloc_1243_; 
v_reuseFailAlloc_1243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1243_, 0, v_a_1237_);
v___x_1242_ = v_reuseFailAlloc_1243_;
goto v_reusejp_1241_;
}
v_reusejp_1241_:
{
return v___x_1242_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object* v_head_1245_, lean_object* v_tail_1246_, lean_object* v_x_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_){
_start:
{
lean_object* v_res_1257_; 
v_res_1257_ = lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg___lam__0(v_head_1245_, v_tail_1246_, v_x_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_, v___y_1252_, v___y_1253_, v___y_1254_, v___y_1255_);
lean_dec(v___y_1255_);
lean_dec_ref(v___y_1254_);
lean_dec(v___y_1253_);
lean_dec_ref(v___y_1252_);
lean_dec(v___y_1251_);
lean_dec_ref(v___y_1250_);
lean_dec(v___y_1249_);
lean_dec_ref(v___y_1248_);
return v_res_1257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg(lean_object* v_x_1258_){
_start:
{
if (lean_obj_tag(v_x_1258_) == 0)
{
lean_object* v___x_1259_; 
v___x_1259_ = lean_box(0);
return v___x_1259_;
}
else
{
lean_object* v_head_1260_; lean_object* v_tail_1261_; lean_object* v___f_1262_; lean_object* v___x_1263_; 
v_head_1260_ = lean_ctor_get(v_x_1258_, 0);
lean_inc(v_head_1260_);
v_tail_1261_ = lean_ctor_get(v_x_1258_, 1);
lean_inc(v_tail_1261_);
lean_dec_ref_known(v_x_1258_, 2);
v___f_1262_ = lean_alloc_closure((void*)(lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg___lam__0___boxed), 12, 2);
lean_closure_set(v___f_1262_, 0, v_head_1260_);
lean_closure_set(v___f_1262_, 1, v_tail_1261_);
v___x_1263_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1263_, 0, v___f_1262_);
return v___x_1263_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg___lam__0(lean_object* v_a_1264_, lean_object* v_head_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
uint8_t v___x_1275_; lean_object* v___x_1276_; 
v___x_1275_ = 0;
v___x_1276_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1264_, v___x_1275_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_, v___y_1272_, v___y_1273_);
if (lean_obj_tag(v___x_1276_) == 0)
{
lean_object* v___x_1277_; 
lean_dec_ref_known(v___x_1276_, 1);
lean_inc(v___y_1273_);
lean_inc(v___y_1271_);
lean_inc(v___y_1269_);
lean_inc(v___y_1267_);
v___x_1277_ = lean_apply_9(v_head_1265_, v___y_1266_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_, v___y_1272_, v___y_1273_, lean_box(0));
if (lean_obj_tag(v___x_1277_) == 0)
{
lean_object* v_a_1278_; lean_object* v___x_1279_; 
v_a_1278_ = lean_ctor_get(v___x_1277_, 0);
lean_inc(v_a_1278_);
lean_dec_ref_known(v___x_1277_, 1);
v___x_1279_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1267_, v___y_1269_, v___y_1271_, v___y_1273_);
lean_dec(v___y_1273_);
lean_dec(v___y_1271_);
lean_dec(v___y_1269_);
lean_dec(v___y_1267_);
if (lean_obj_tag(v___x_1279_) == 0)
{
lean_object* v_a_1280_; lean_object* v___x_1282_; uint8_t v_isShared_1283_; uint8_t v_isSharedCheck_1288_; 
v_a_1280_ = lean_ctor_get(v___x_1279_, 0);
v_isSharedCheck_1288_ = !lean_is_exclusive(v___x_1279_);
if (v_isSharedCheck_1288_ == 0)
{
v___x_1282_ = v___x_1279_;
v_isShared_1283_ = v_isSharedCheck_1288_;
goto v_resetjp_1281_;
}
else
{
lean_inc(v_a_1280_);
lean_dec(v___x_1279_);
v___x_1282_ = lean_box(0);
v_isShared_1283_ = v_isSharedCheck_1288_;
goto v_resetjp_1281_;
}
v_resetjp_1281_:
{
lean_object* v___x_1284_; lean_object* v___x_1286_; 
v___x_1284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1284_, 0, v_a_1278_);
lean_ctor_set(v___x_1284_, 1, v_a_1280_);
if (v_isShared_1283_ == 0)
{
lean_ctor_set(v___x_1282_, 0, v___x_1284_);
v___x_1286_ = v___x_1282_;
goto v_reusejp_1285_;
}
else
{
lean_object* v_reuseFailAlloc_1287_; 
v_reuseFailAlloc_1287_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1287_, 0, v___x_1284_);
v___x_1286_ = v_reuseFailAlloc_1287_;
goto v_reusejp_1285_;
}
v_reusejp_1285_:
{
return v___x_1286_;
}
}
}
else
{
lean_object* v_a_1289_; lean_object* v___x_1291_; uint8_t v_isShared_1292_; uint8_t v_isSharedCheck_1296_; 
lean_dec(v_a_1278_);
v_a_1289_ = lean_ctor_get(v___x_1279_, 0);
v_isSharedCheck_1296_ = !lean_is_exclusive(v___x_1279_);
if (v_isSharedCheck_1296_ == 0)
{
v___x_1291_ = v___x_1279_;
v_isShared_1292_ = v_isSharedCheck_1296_;
goto v_resetjp_1290_;
}
else
{
lean_inc(v_a_1289_);
lean_dec(v___x_1279_);
v___x_1291_ = lean_box(0);
v_isShared_1292_ = v_isSharedCheck_1296_;
goto v_resetjp_1290_;
}
v_resetjp_1290_:
{
lean_object* v___x_1294_; 
if (v_isShared_1292_ == 0)
{
v___x_1294_ = v___x_1291_;
goto v_reusejp_1293_;
}
else
{
lean_object* v_reuseFailAlloc_1295_; 
v_reuseFailAlloc_1295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1295_, 0, v_a_1289_);
v___x_1294_ = v_reuseFailAlloc_1295_;
goto v_reusejp_1293_;
}
v_reusejp_1293_:
{
return v___x_1294_;
}
}
}
}
else
{
lean_object* v_a_1297_; lean_object* v___x_1299_; uint8_t v_isShared_1300_; uint8_t v_isSharedCheck_1304_; 
lean_dec(v___y_1273_);
lean_dec(v___y_1271_);
lean_dec(v___y_1269_);
lean_dec(v___y_1267_);
v_a_1297_ = lean_ctor_get(v___x_1277_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1277_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1299_ = v___x_1277_;
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
else
{
lean_inc(v_a_1297_);
lean_dec(v___x_1277_);
v___x_1299_ = lean_box(0);
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
v_resetjp_1298_:
{
lean_object* v___x_1302_; 
if (v_isShared_1300_ == 0)
{
v___x_1302_ = v___x_1299_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_a_1297_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
}
}
else
{
lean_object* v_a_1305_; lean_object* v___x_1307_; uint8_t v_isShared_1308_; uint8_t v_isSharedCheck_1312_; 
lean_dec(v___y_1273_);
lean_dec_ref(v___y_1272_);
lean_dec(v___y_1271_);
lean_dec_ref(v___y_1270_);
lean_dec(v___y_1269_);
lean_dec_ref(v___y_1268_);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec_ref(v_head_1265_);
v_a_1305_ = lean_ctor_get(v___x_1276_, 0);
v_isSharedCheck_1312_ = !lean_is_exclusive(v___x_1276_);
if (v_isSharedCheck_1312_ == 0)
{
v___x_1307_ = v___x_1276_;
v_isShared_1308_ = v_isSharedCheck_1312_;
goto v_resetjp_1306_;
}
else
{
lean_inc(v_a_1305_);
lean_dec(v___x_1276_);
v___x_1307_ = lean_box(0);
v_isShared_1308_ = v_isSharedCheck_1312_;
goto v_resetjp_1306_;
}
v_resetjp_1306_:
{
lean_object* v___x_1310_; 
if (v_isShared_1308_ == 0)
{
v___x_1310_ = v___x_1307_;
goto v_reusejp_1309_;
}
else
{
lean_object* v_reuseFailAlloc_1311_; 
v_reuseFailAlloc_1311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1311_, 0, v_a_1305_);
v___x_1310_ = v_reuseFailAlloc_1311_;
goto v_reusejp_1309_;
}
v_reusejp_1309_:
{
return v___x_1310_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg___lam__0___boxed(lean_object* v_a_1313_, lean_object* v_head_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_){
_start:
{
lean_object* v_res_1324_; 
v_res_1324_ = lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg___lam__0(v_a_1313_, v_head_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
return v_res_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg(lean_object* v_a_1325_, lean_object* v_a_1326_, lean_object* v_a_1327_){
_start:
{
if (lean_obj_tag(v_a_1326_) == 0)
{
lean_object* v___x_1328_; 
lean_dec_ref(v_a_1325_);
v___x_1328_ = l_List_reverse___redArg(v_a_1327_);
return v___x_1328_;
}
else
{
lean_object* v_head_1329_; lean_object* v_tail_1330_; lean_object* v___x_1332_; uint8_t v_isShared_1333_; uint8_t v_isSharedCheck_1339_; 
v_head_1329_ = lean_ctor_get(v_a_1326_, 0);
v_tail_1330_ = lean_ctor_get(v_a_1326_, 1);
v_isSharedCheck_1339_ = !lean_is_exclusive(v_a_1326_);
if (v_isSharedCheck_1339_ == 0)
{
v___x_1332_ = v_a_1326_;
v_isShared_1333_ = v_isSharedCheck_1339_;
goto v_resetjp_1331_;
}
else
{
lean_inc(v_tail_1330_);
lean_inc(v_head_1329_);
lean_dec(v_a_1326_);
v___x_1332_ = lean_box(0);
v_isShared_1333_ = v_isSharedCheck_1339_;
goto v_resetjp_1331_;
}
v_resetjp_1331_:
{
lean_object* v___f_1334_; lean_object* v___x_1336_; 
lean_inc_ref(v_a_1325_);
v___f_1334_ = lean_alloc_closure((void*)(lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg___lam__0___boxed), 11, 2);
lean_closure_set(v___f_1334_, 0, v_a_1325_);
lean_closure_set(v___f_1334_, 1, v_head_1329_);
if (v_isShared_1333_ == 0)
{
lean_ctor_set(v___x_1332_, 1, v_a_1327_);
lean_ctor_set(v___x_1332_, 0, v___f_1334_);
v___x_1336_ = v___x_1332_;
goto v_reusejp_1335_;
}
else
{
lean_object* v_reuseFailAlloc_1338_; 
v_reuseFailAlloc_1338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1338_, 0, v___f_1334_);
lean_ctor_set(v_reuseFailAlloc_1338_, 1, v_a_1327_);
v___x_1336_ = v_reuseFailAlloc_1338_;
goto v_reusejp_1335_;
}
v_reusejp_1335_:
{
v_a_1326_ = v_tail_1330_;
v_a_1327_ = v___x_1336_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg___lam__0(lean_object* v_L_1340_, lean_object* v_x_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_){
_start:
{
lean_object* v___x_1351_; 
v___x_1351_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1343_, v___y_1345_, v___y_1347_, v___y_1349_);
if (lean_obj_tag(v___x_1351_) == 0)
{
lean_object* v_a_1352_; lean_object* v___x_1354_; uint8_t v_isShared_1355_; uint8_t v_isSharedCheck_1362_; 
v_a_1352_ = lean_ctor_get(v___x_1351_, 0);
v_isSharedCheck_1362_ = !lean_is_exclusive(v___x_1351_);
if (v_isSharedCheck_1362_ == 0)
{
v___x_1354_ = v___x_1351_;
v_isShared_1355_ = v_isSharedCheck_1362_;
goto v_resetjp_1353_;
}
else
{
lean_inc(v_a_1352_);
lean_dec(v___x_1351_);
v___x_1354_ = lean_box(0);
v_isShared_1355_ = v_isSharedCheck_1362_;
goto v_resetjp_1353_;
}
v_resetjp_1353_:
{
lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1360_; 
v___x_1356_ = lean_box(0);
v___x_1357_ = lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg(v_a_1352_, v_L_1340_, v___x_1356_);
v___x_1358_ = lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg(v___x_1357_);
if (v_isShared_1355_ == 0)
{
lean_ctor_set(v___x_1354_, 0, v___x_1358_);
v___x_1360_ = v___x_1354_;
goto v_reusejp_1359_;
}
else
{
lean_object* v_reuseFailAlloc_1361_; 
v_reuseFailAlloc_1361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1361_, 0, v___x_1358_);
v___x_1360_ = v_reuseFailAlloc_1361_;
goto v_reusejp_1359_;
}
v_reusejp_1359_:
{
return v___x_1360_;
}
}
}
else
{
lean_object* v_a_1363_; lean_object* v___x_1365_; uint8_t v_isShared_1366_; uint8_t v_isSharedCheck_1370_; 
lean_dec(v_L_1340_);
v_a_1363_ = lean_ctor_get(v___x_1351_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1351_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1365_ = v___x_1351_;
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
else
{
lean_inc(v_a_1363_);
lean_dec(v___x_1351_);
v___x_1365_ = lean_box(0);
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
v_resetjp_1364_:
{
lean_object* v___x_1368_; 
if (v_isShared_1366_ == 0)
{
v___x_1368_ = v___x_1365_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_a_1363_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg___lam__0___boxed(lean_object* v_L_1371_, lean_object* v_x_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_){
_start:
{
lean_object* v_res_1382_; 
v_res_1382_ = lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg___lam__0(v_L_1371_, v_x_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_, v___y_1379_, v___y_1380_);
lean_dec(v___y_1380_);
lean_dec_ref(v___y_1379_);
lean_dec(v___y_1378_);
lean_dec_ref(v___y_1377_);
lean_dec(v___y_1376_);
lean_dec_ref(v___y_1375_);
lean_dec(v___y_1374_);
lean_dec_ref(v___y_1373_);
return v_res_1382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg(lean_object* v_L_1383_){
_start:
{
lean_object* v___f_1384_; lean_object* v___x_1385_; 
v___f_1384_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg___lam__0___boxed), 11, 1);
lean_closure_set(v___f_1384_, 0, v_L_1383_);
v___x_1385_ = lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg(v___f_1384_);
return v___x_1385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2___redArg(lean_object* v_L_1386_){
_start:
{
lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1387_ = lean_box(0);
v___x_1388_ = lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg(v_L_1386_, v___x_1387_);
v___x_1389_ = lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg(v___x_1388_);
return v___x_1389_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0(lean_object* v_x1_1390_, lean_object* v_x2_1391_){
_start:
{
lean_object* v_fst_1392_; lean_object* v_fst_1393_; lean_object* v_fst_1394_; lean_object* v_fst_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; uint8_t v___x_1398_; 
v_fst_1392_ = lean_ctor_get(v_x1_1390_, 0);
v_fst_1393_ = lean_ctor_get(v_x2_1391_, 0);
v_fst_1394_ = lean_ctor_get(v_fst_1392_, 0);
v_fst_1395_ = lean_ctor_get(v_fst_1393_, 0);
v___x_1396_ = l_List_lengthTR___redArg(v_fst_1394_);
v___x_1397_ = l_List_lengthTR___redArg(v_fst_1395_);
v___x_1398_ = lean_nat_dec_lt(v___x_1396_, v___x_1397_);
lean_dec(v___x_1397_);
lean_dec(v___x_1396_);
return v___x_1398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0___boxed(lean_object* v_x1_1399_, lean_object* v_x2_1400_){
_start:
{
uint8_t v_res_1401_; lean_object* v_r_1402_; 
v_res_1401_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0(v_x1_1399_, v_x2_1400_);
lean_dec_ref(v_x2_1400_);
lean_dec_ref(v_x1_1399_);
v_r_1402_ = lean_box(v_res_1401_);
return v_r_1402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___redArg(lean_object* v_hi_1403_, lean_object* v_pivot_1404_, lean_object* v_as_1405_, lean_object* v_i_1406_, lean_object* v_k_1407_){
_start:
{
uint8_t v___x_1408_; 
v___x_1408_ = lean_nat_dec_lt(v_k_1407_, v_hi_1403_);
if (v___x_1408_ == 0)
{
lean_object* v___x_1409_; lean_object* v___x_1410_; 
lean_dec(v_k_1407_);
v___x_1409_ = lean_array_fswap(v_as_1405_, v_i_1406_, v_hi_1403_);
v___x_1410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1410_, 0, v_i_1406_);
lean_ctor_set(v___x_1410_, 1, v___x_1409_);
return v___x_1410_;
}
else
{
lean_object* v___x_1411_; lean_object* v_fst_1412_; lean_object* v_fst_1413_; lean_object* v_fst_1414_; lean_object* v_fst_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; uint8_t v___x_1418_; 
v___x_1411_ = lean_array_fget_borrowed(v_as_1405_, v_k_1407_);
v_fst_1412_ = lean_ctor_get(v___x_1411_, 0);
v_fst_1413_ = lean_ctor_get(v_pivot_1404_, 0);
v_fst_1414_ = lean_ctor_get(v_fst_1412_, 0);
v_fst_1415_ = lean_ctor_get(v_fst_1413_, 0);
v___x_1416_ = l_List_lengthTR___redArg(v_fst_1414_);
v___x_1417_ = l_List_lengthTR___redArg(v_fst_1415_);
v___x_1418_ = lean_nat_dec_lt(v___x_1416_, v___x_1417_);
lean_dec(v___x_1417_);
lean_dec(v___x_1416_);
if (v___x_1418_ == 0)
{
lean_object* v___x_1419_; lean_object* v___x_1420_; 
v___x_1419_ = lean_unsigned_to_nat(1u);
v___x_1420_ = lean_nat_add(v_k_1407_, v___x_1419_);
lean_dec(v_k_1407_);
v_k_1407_ = v___x_1420_;
goto _start;
}
else
{
lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; 
v___x_1422_ = lean_array_fswap(v_as_1405_, v_i_1406_, v_k_1407_);
v___x_1423_ = lean_unsigned_to_nat(1u);
v___x_1424_ = lean_nat_add(v_i_1406_, v___x_1423_);
lean_dec(v_i_1406_);
v___x_1425_ = lean_nat_add(v_k_1407_, v___x_1423_);
lean_dec(v_k_1407_);
v_as_1405_ = v___x_1422_;
v_i_1406_ = v___x_1424_;
v_k_1407_ = v___x_1425_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___redArg___boxed(lean_object* v_hi_1427_, lean_object* v_pivot_1428_, lean_object* v_as_1429_, lean_object* v_i_1430_, lean_object* v_k_1431_){
_start:
{
lean_object* v_res_1432_; 
v_res_1432_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___redArg(v_hi_1427_, v_pivot_1428_, v_as_1429_, v_i_1430_, v_k_1431_);
lean_dec_ref(v_pivot_1428_);
lean_dec(v_hi_1427_);
return v_res_1432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg(lean_object* v_n_1433_, lean_object* v_as_1434_, lean_object* v_lo_1435_, lean_object* v_hi_1436_){
_start:
{
lean_object* v___y_1438_; uint8_t v___x_1448_; 
v___x_1448_ = lean_nat_dec_lt(v_lo_1435_, v_hi_1436_);
if (v___x_1448_ == 0)
{
lean_dec(v_lo_1435_);
return v_as_1434_;
}
else
{
lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v_mid_1451_; lean_object* v___y_1453_; lean_object* v___y_1459_; lean_object* v___x_1464_; lean_object* v___x_1465_; uint8_t v___x_1466_; 
v___x_1449_ = lean_nat_add(v_lo_1435_, v_hi_1436_);
v___x_1450_ = lean_unsigned_to_nat(1u);
v_mid_1451_ = lean_nat_shiftr(v___x_1449_, v___x_1450_);
lean_dec(v___x_1449_);
v___x_1464_ = lean_array_fget_borrowed(v_as_1434_, v_mid_1451_);
v___x_1465_ = lean_array_fget_borrowed(v_as_1434_, v_lo_1435_);
v___x_1466_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0(v___x_1464_, v___x_1465_);
if (v___x_1466_ == 0)
{
v___y_1459_ = v_as_1434_;
goto v___jp_1458_;
}
else
{
lean_object* v___x_1467_; 
v___x_1467_ = lean_array_fswap(v_as_1434_, v_lo_1435_, v_mid_1451_);
v___y_1459_ = v___x_1467_;
goto v___jp_1458_;
}
v___jp_1452_:
{
lean_object* v___x_1454_; lean_object* v___x_1455_; uint8_t v___x_1456_; 
v___x_1454_ = lean_array_fget_borrowed(v___y_1453_, v_mid_1451_);
v___x_1455_ = lean_array_fget_borrowed(v___y_1453_, v_hi_1436_);
v___x_1456_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0(v___x_1454_, v___x_1455_);
if (v___x_1456_ == 0)
{
lean_dec(v_mid_1451_);
v___y_1438_ = v___y_1453_;
goto v___jp_1437_;
}
else
{
lean_object* v___x_1457_; 
v___x_1457_ = lean_array_fswap(v___y_1453_, v_mid_1451_, v_hi_1436_);
lean_dec(v_mid_1451_);
v___y_1438_ = v___x_1457_;
goto v___jp_1437_;
}
}
v___jp_1458_:
{
lean_object* v___x_1460_; lean_object* v___x_1461_; uint8_t v___x_1462_; 
v___x_1460_ = lean_array_fget_borrowed(v___y_1459_, v_hi_1436_);
v___x_1461_ = lean_array_fget_borrowed(v___y_1459_, v_lo_1435_);
v___x_1462_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___lam__0(v___x_1460_, v___x_1461_);
if (v___x_1462_ == 0)
{
v___y_1453_ = v___y_1459_;
goto v___jp_1452_;
}
else
{
lean_object* v___x_1463_; 
v___x_1463_ = lean_array_fswap(v___y_1459_, v_lo_1435_, v_hi_1436_);
v___y_1453_ = v___x_1463_;
goto v___jp_1452_;
}
}
}
v___jp_1437_:
{
lean_object* v_pivot_1439_; lean_object* v___x_1440_; lean_object* v_fst_1441_; lean_object* v_snd_1442_; uint8_t v___x_1443_; 
v_pivot_1439_ = lean_array_fget(v___y_1438_, v_hi_1436_);
lean_inc_n(v_lo_1435_, 2);
v___x_1440_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___redArg(v_hi_1436_, v_pivot_1439_, v___y_1438_, v_lo_1435_, v_lo_1435_);
lean_dec(v_pivot_1439_);
v_fst_1441_ = lean_ctor_get(v___x_1440_, 0);
lean_inc(v_fst_1441_);
v_snd_1442_ = lean_ctor_get(v___x_1440_, 1);
lean_inc(v_snd_1442_);
lean_dec_ref(v___x_1440_);
v___x_1443_ = lean_nat_dec_le(v_hi_1436_, v_fst_1441_);
if (v___x_1443_ == 0)
{
lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; 
v___x_1444_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg(v_n_1433_, v_snd_1442_, v_lo_1435_, v_fst_1441_);
v___x_1445_ = lean_unsigned_to_nat(1u);
v___x_1446_ = lean_nat_add(v_fst_1441_, v___x_1445_);
lean_dec(v_fst_1441_);
v_as_1434_ = v___x_1444_;
v_lo_1435_ = v___x_1446_;
goto _start;
}
else
{
lean_dec(v_fst_1441_);
lean_dec(v_lo_1435_);
return v_snd_1442_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg___boxed(lean_object* v_n_1468_, lean_object* v_as_1469_, lean_object* v_lo_1470_, lean_object* v_hi_1471_){
_start:
{
lean_object* v_res_1472_; 
v_res_1472_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg(v_n_1468_, v_as_1469_, v_lo_1470_, v_hi_1471_);
lean_dec(v_hi_1471_);
lean_dec(v_n_1468_);
return v_res_1472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13(lean_object* v_as_1476_, size_t v_sz_1477_, size_t v_i_1478_, lean_object* v_b_1479_){
_start:
{
uint8_t v___x_1480_; 
v___x_1480_ = lean_usize_dec_lt(v_i_1478_, v_sz_1477_);
if (v___x_1480_ == 0)
{
lean_inc_ref(v_b_1479_);
return v_b_1479_;
}
else
{
lean_object* v_a_1481_; lean_object* v_fst_1482_; lean_object* v_fst_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1498_; 
v_a_1481_ = lean_array_uget_borrowed(v_as_1476_, v_i_1478_);
v_fst_1482_ = lean_ctor_get(v_a_1481_, 0);
lean_inc(v_fst_1482_);
v_fst_1483_ = lean_ctor_get(v_fst_1482_, 0);
v_isSharedCheck_1498_ = !lean_is_exclusive(v_fst_1482_);
if (v_isSharedCheck_1498_ == 0)
{
lean_object* v_unused_1499_; 
v_unused_1499_ = lean_ctor_get(v_fst_1482_, 1);
lean_dec(v_unused_1499_);
v___x_1485_ = v_fst_1482_;
v_isShared_1486_ = v_isSharedCheck_1498_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_fst_1483_);
lean_dec(v_fst_1482_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1498_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v___x_1487_; uint8_t v___x_1488_; 
v___x_1487_ = lean_box(0);
v___x_1488_ = l_List_isEmpty___redArg(v_fst_1483_);
lean_dec(v_fst_1483_);
if (v___x_1488_ == 0)
{
lean_object* v___x_1489_; size_t v___x_1490_; size_t v___x_1491_; 
lean_del_object(v___x_1485_);
v___x_1489_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13___closed__0));
v___x_1490_ = ((size_t)1ULL);
v___x_1491_ = lean_usize_add(v_i_1478_, v___x_1490_);
v_i_1478_ = v___x_1491_;
v_b_1479_ = v___x_1489_;
goto _start;
}
else
{
lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1496_; 
lean_inc(v_a_1481_);
v___x_1493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1493_, 0, v_a_1481_);
v___x_1494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1494_, 0, v___x_1493_);
if (v_isShared_1486_ == 0)
{
lean_ctor_set(v___x_1485_, 1, v___x_1487_);
lean_ctor_set(v___x_1485_, 0, v___x_1494_);
v___x_1496_ = v___x_1485_;
goto v_reusejp_1495_;
}
else
{
lean_object* v_reuseFailAlloc_1497_; 
v_reuseFailAlloc_1497_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1497_, 0, v___x_1494_);
lean_ctor_set(v_reuseFailAlloc_1497_, 1, v___x_1487_);
v___x_1496_ = v_reuseFailAlloc_1497_;
goto v_reusejp_1495_;
}
v_reusejp_1495_:
{
return v___x_1496_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13___boxed(lean_object* v_as_1500_, lean_object* v_sz_1501_, lean_object* v_i_1502_, lean_object* v_b_1503_){
_start:
{
size_t v_sz_boxed_1504_; size_t v_i_boxed_1505_; lean_object* v_res_1506_; 
v_sz_boxed_1504_ = lean_unbox_usize(v_sz_1501_);
lean_dec(v_sz_1501_);
v_i_boxed_1505_ = lean_unbox_usize(v_i_1502_);
lean_dec(v_i_1502_);
v_res_1506_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13(v_as_1500_, v_sz_boxed_1504_, v_i_boxed_1505_, v_b_1503_);
lean_dec_ref(v_b_1503_);
lean_dec_ref(v_as_1500_);
return v_res_1506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7(lean_object* v_as_1507_, size_t v_sz_1508_, size_t v_i_1509_, lean_object* v_b_1510_){
_start:
{
uint8_t v___x_1511_; 
v___x_1511_ = lean_usize_dec_lt(v_i_1509_, v_sz_1508_);
if (v___x_1511_ == 0)
{
lean_inc_ref(v_b_1510_);
return v_b_1510_;
}
else
{
lean_object* v_a_1512_; lean_object* v_fst_1513_; lean_object* v_fst_1514_; lean_object* v___x_1516_; uint8_t v_isShared_1517_; uint8_t v_isSharedCheck_1529_; 
v_a_1512_ = lean_array_uget_borrowed(v_as_1507_, v_i_1509_);
v_fst_1513_ = lean_ctor_get(v_a_1512_, 0);
lean_inc(v_fst_1513_);
v_fst_1514_ = lean_ctor_get(v_fst_1513_, 0);
v_isSharedCheck_1529_ = !lean_is_exclusive(v_fst_1513_);
if (v_isSharedCheck_1529_ == 0)
{
lean_object* v_unused_1530_; 
v_unused_1530_ = lean_ctor_get(v_fst_1513_, 1);
lean_dec(v_unused_1530_);
v___x_1516_ = v_fst_1513_;
v_isShared_1517_ = v_isSharedCheck_1529_;
goto v_resetjp_1515_;
}
else
{
lean_inc(v_fst_1514_);
lean_dec(v_fst_1513_);
v___x_1516_ = lean_box(0);
v_isShared_1517_ = v_isSharedCheck_1529_;
goto v_resetjp_1515_;
}
v_resetjp_1515_:
{
lean_object* v___x_1518_; uint8_t v___x_1519_; 
v___x_1518_ = lean_box(0);
v___x_1519_ = l_List_isEmpty___redArg(v_fst_1514_);
lean_dec(v_fst_1514_);
if (v___x_1519_ == 0)
{
lean_object* v___x_1520_; size_t v___x_1521_; size_t v___x_1522_; lean_object* v___x_1523_; 
lean_del_object(v___x_1516_);
v___x_1520_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13___closed__0));
v___x_1521_ = ((size_t)1ULL);
v___x_1522_ = lean_usize_add(v_i_1509_, v___x_1521_);
v___x_1523_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7_spec__13(v_as_1507_, v_sz_1508_, v___x_1522_, v___x_1520_);
return v___x_1523_;
}
else
{
lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1527_; 
lean_inc(v_a_1512_);
v___x_1524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1524_, 0, v_a_1512_);
v___x_1525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1525_, 0, v___x_1524_);
if (v_isShared_1517_ == 0)
{
lean_ctor_set(v___x_1516_, 1, v___x_1518_);
lean_ctor_set(v___x_1516_, 0, v___x_1525_);
v___x_1527_ = v___x_1516_;
goto v_reusejp_1526_;
}
else
{
lean_object* v_reuseFailAlloc_1528_; 
v_reuseFailAlloc_1528_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1528_, 0, v___x_1525_);
lean_ctor_set(v_reuseFailAlloc_1528_, 1, v___x_1518_);
v___x_1527_ = v_reuseFailAlloc_1528_;
goto v_reusejp_1526_;
}
v_reusejp_1526_:
{
return v___x_1527_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7___boxed(lean_object* v_as_1531_, lean_object* v_sz_1532_, lean_object* v_i_1533_, lean_object* v_b_1534_){
_start:
{
size_t v_sz_boxed_1535_; size_t v_i_boxed_1536_; lean_object* v_res_1537_; 
v_sz_boxed_1535_ = lean_unbox_usize(v_sz_1532_);
lean_dec(v_sz_1532_);
v_i_boxed_1536_ = lean_unbox_usize(v_i_1533_);
lean_dec(v_i_1533_);
v_res_1537_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7(v_as_1531_, v_sz_boxed_1535_, v_i_boxed_1536_, v_b_1534_);
lean_dec_ref(v_b_1534_);
lean_dec_ref(v_as_1531_);
return v_res_1537_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0(lean_object* v_x1_1538_, lean_object* v_x2_1539_){
_start:
{
lean_object* v_fst_1540_; lean_object* v_fst_1541_; uint8_t v___x_1542_; 
v_fst_1540_ = lean_ctor_get(v_x2_1539_, 0);
v_fst_1541_ = lean_ctor_get(v_x1_1538_, 0);
v___x_1542_ = lean_nat_dec_lt(v_fst_1540_, v_fst_1541_);
return v___x_1542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0___boxed(lean_object* v_x1_1543_, lean_object* v_x2_1544_){
_start:
{
uint8_t v_res_1545_; lean_object* v_r_1546_; 
v_res_1545_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0(v_x1_1543_, v_x2_1544_);
lean_dec_ref(v_x2_1544_);
lean_dec_ref(v_x1_1543_);
v_r_1546_ = lean_box(v_res_1545_);
return v_r_1546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___redArg(lean_object* v_hi_1547_, lean_object* v_pivot_1548_, lean_object* v_as_1549_, lean_object* v_i_1550_, lean_object* v_k_1551_){
_start:
{
uint8_t v___x_1552_; 
v___x_1552_ = lean_nat_dec_lt(v_k_1551_, v_hi_1547_);
if (v___x_1552_ == 0)
{
lean_object* v___x_1553_; lean_object* v___x_1554_; 
lean_dec(v_k_1551_);
v___x_1553_ = lean_array_fswap(v_as_1549_, v_i_1550_, v_hi_1547_);
v___x_1554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1554_, 0, v_i_1550_);
lean_ctor_set(v___x_1554_, 1, v___x_1553_);
return v___x_1554_;
}
else
{
lean_object* v_fst_1555_; lean_object* v___x_1556_; lean_object* v_fst_1557_; uint8_t v___x_1558_; 
v_fst_1555_ = lean_ctor_get(v_pivot_1548_, 0);
v___x_1556_ = lean_array_fget_borrowed(v_as_1549_, v_k_1551_);
v_fst_1557_ = lean_ctor_get(v___x_1556_, 0);
v___x_1558_ = lean_nat_dec_lt(v_fst_1555_, v_fst_1557_);
if (v___x_1558_ == 0)
{
lean_object* v___x_1559_; lean_object* v___x_1560_; 
v___x_1559_ = lean_unsigned_to_nat(1u);
v___x_1560_ = lean_nat_add(v_k_1551_, v___x_1559_);
lean_dec(v_k_1551_);
v_k_1551_ = v___x_1560_;
goto _start;
}
else
{
lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; 
v___x_1562_ = lean_array_fswap(v_as_1549_, v_i_1550_, v_k_1551_);
v___x_1563_ = lean_unsigned_to_nat(1u);
v___x_1564_ = lean_nat_add(v_i_1550_, v___x_1563_);
lean_dec(v_i_1550_);
v___x_1565_ = lean_nat_add(v_k_1551_, v___x_1563_);
lean_dec(v_k_1551_);
v_as_1549_ = v___x_1562_;
v_i_1550_ = v___x_1564_;
v_k_1551_ = v___x_1565_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___redArg___boxed(lean_object* v_hi_1567_, lean_object* v_pivot_1568_, lean_object* v_as_1569_, lean_object* v_i_1570_, lean_object* v_k_1571_){
_start:
{
lean_object* v_res_1572_; 
v_res_1572_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___redArg(v_hi_1567_, v_pivot_1568_, v_as_1569_, v_i_1570_, v_k_1571_);
lean_dec_ref(v_pivot_1568_);
lean_dec(v_hi_1567_);
return v_res_1572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg(lean_object* v_n_1573_, lean_object* v_as_1574_, lean_object* v_lo_1575_, lean_object* v_hi_1576_){
_start:
{
lean_object* v___y_1578_; uint8_t v___x_1588_; 
v___x_1588_ = lean_nat_dec_lt(v_lo_1575_, v_hi_1576_);
if (v___x_1588_ == 0)
{
lean_dec(v_lo_1575_);
return v_as_1574_;
}
else
{
lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v_mid_1591_; lean_object* v___y_1593_; lean_object* v___y_1599_; lean_object* v___x_1604_; lean_object* v___x_1605_; uint8_t v___x_1606_; 
v___x_1589_ = lean_nat_add(v_lo_1575_, v_hi_1576_);
v___x_1590_ = lean_unsigned_to_nat(1u);
v_mid_1591_ = lean_nat_shiftr(v___x_1589_, v___x_1590_);
lean_dec(v___x_1589_);
v___x_1604_ = lean_array_fget_borrowed(v_as_1574_, v_mid_1591_);
v___x_1605_ = lean_array_fget_borrowed(v_as_1574_, v_lo_1575_);
v___x_1606_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0(v___x_1604_, v___x_1605_);
if (v___x_1606_ == 0)
{
v___y_1599_ = v_as_1574_;
goto v___jp_1598_;
}
else
{
lean_object* v___x_1607_; 
v___x_1607_ = lean_array_fswap(v_as_1574_, v_lo_1575_, v_mid_1591_);
v___y_1599_ = v___x_1607_;
goto v___jp_1598_;
}
v___jp_1592_:
{
lean_object* v___x_1594_; lean_object* v___x_1595_; uint8_t v___x_1596_; 
v___x_1594_ = lean_array_fget_borrowed(v___y_1593_, v_mid_1591_);
v___x_1595_ = lean_array_fget_borrowed(v___y_1593_, v_hi_1576_);
v___x_1596_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0(v___x_1594_, v___x_1595_);
if (v___x_1596_ == 0)
{
lean_dec(v_mid_1591_);
v___y_1578_ = v___y_1593_;
goto v___jp_1577_;
}
else
{
lean_object* v___x_1597_; 
v___x_1597_ = lean_array_fswap(v___y_1593_, v_mid_1591_, v_hi_1576_);
lean_dec(v_mid_1591_);
v___y_1578_ = v___x_1597_;
goto v___jp_1577_;
}
}
v___jp_1598_:
{
lean_object* v___x_1600_; lean_object* v___x_1601_; uint8_t v___x_1602_; 
v___x_1600_ = lean_array_fget_borrowed(v___y_1599_, v_hi_1576_);
v___x_1601_ = lean_array_fget_borrowed(v___y_1599_, v_lo_1575_);
v___x_1602_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___lam__0(v___x_1600_, v___x_1601_);
if (v___x_1602_ == 0)
{
v___y_1593_ = v___y_1599_;
goto v___jp_1592_;
}
else
{
lean_object* v___x_1603_; 
v___x_1603_ = lean_array_fswap(v___y_1599_, v_lo_1575_, v_hi_1576_);
v___y_1593_ = v___x_1603_;
goto v___jp_1592_;
}
}
}
v___jp_1577_:
{
lean_object* v_pivot_1579_; lean_object* v___x_1580_; lean_object* v_fst_1581_; lean_object* v_snd_1582_; uint8_t v___x_1583_; 
v_pivot_1579_ = lean_array_fget(v___y_1578_, v_hi_1576_);
lean_inc_n(v_lo_1575_, 2);
v___x_1580_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___redArg(v_hi_1576_, v_pivot_1579_, v___y_1578_, v_lo_1575_, v_lo_1575_);
lean_dec(v_pivot_1579_);
v_fst_1581_ = lean_ctor_get(v___x_1580_, 0);
lean_inc(v_fst_1581_);
v_snd_1582_ = lean_ctor_get(v___x_1580_, 1);
lean_inc(v_snd_1582_);
lean_dec_ref(v___x_1580_);
v___x_1583_ = lean_nat_dec_le(v_hi_1576_, v_fst_1581_);
if (v___x_1583_ == 0)
{
lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; 
v___x_1584_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg(v_n_1573_, v_snd_1582_, v_lo_1575_, v_fst_1581_);
v___x_1585_ = lean_unsigned_to_nat(1u);
v___x_1586_ = lean_nat_add(v_fst_1581_, v___x_1585_);
lean_dec(v_fst_1581_);
v_as_1574_ = v___x_1584_;
v_lo_1575_ = v___x_1586_;
goto _start;
}
else
{
lean_dec(v_fst_1581_);
lean_dec(v_lo_1575_);
return v_snd_1582_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg___boxed(lean_object* v_n_1608_, lean_object* v_as_1609_, lean_object* v_lo_1610_, lean_object* v_hi_1611_){
_start:
{
lean_object* v_res_1612_; 
v_res_1612_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg(v_n_1608_, v_as_1609_, v_lo_1610_, v_hi_1611_);
lean_dec(v_hi_1611_);
lean_dec(v_n_1608_);
return v_res_1612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg___lam__0(lean_object* v_xs_1613_, lean_object* v_ys_1614_, lean_object* v_x_1615_, lean_object* v___y_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v___x_1625_; 
v___x_1625_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(v_xs_1613_, v___y_1616_, v___y_1617_, v___y_1618_, v___y_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_);
if (lean_obj_tag(v___x_1625_) == 0)
{
lean_object* v_a_1626_; lean_object* v___x_1628_; uint8_t v_isShared_1629_; uint8_t v_isSharedCheck_1649_; 
v_a_1626_ = lean_ctor_get(v___x_1625_, 0);
v_isSharedCheck_1649_ = !lean_is_exclusive(v___x_1625_);
if (v_isSharedCheck_1649_ == 0)
{
v___x_1628_ = v___x_1625_;
v_isShared_1629_ = v_isSharedCheck_1649_;
goto v_resetjp_1627_;
}
else
{
lean_inc(v_a_1626_);
lean_dec(v___x_1625_);
v___x_1628_ = lean_box(0);
v_isShared_1629_ = v_isSharedCheck_1649_;
goto v_resetjp_1627_;
}
v_resetjp_1627_:
{
if (lean_obj_tag(v_a_1626_) == 0)
{
lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1633_; 
v___x_1630_ = lean_box(0);
v___x_1631_ = lean_apply_1(v_ys_1614_, v___x_1630_);
if (v_isShared_1629_ == 0)
{
lean_ctor_set(v___x_1628_, 0, v___x_1631_);
v___x_1633_ = v___x_1628_;
goto v_reusejp_1632_;
}
else
{
lean_object* v_reuseFailAlloc_1634_; 
v_reuseFailAlloc_1634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1634_, 0, v___x_1631_);
v___x_1633_ = v_reuseFailAlloc_1634_;
goto v_reusejp_1632_;
}
v_reusejp_1632_:
{
return v___x_1633_;
}
}
else
{
lean_object* v_val_1635_; lean_object* v_fst_1636_; lean_object* v_snd_1637_; lean_object* v___x_1639_; uint8_t v_isShared_1640_; uint8_t v_isSharedCheck_1648_; 
v_val_1635_ = lean_ctor_get(v_a_1626_, 0);
lean_inc(v_val_1635_);
lean_dec_ref_known(v_a_1626_, 1);
v_fst_1636_ = lean_ctor_get(v_val_1635_, 0);
v_snd_1637_ = lean_ctor_get(v_val_1635_, 1);
v_isSharedCheck_1648_ = !lean_is_exclusive(v_val_1635_);
if (v_isSharedCheck_1648_ == 0)
{
v___x_1639_ = v_val_1635_;
v_isShared_1640_ = v_isSharedCheck_1648_;
goto v_resetjp_1638_;
}
else
{
lean_inc(v_snd_1637_);
lean_inc(v_fst_1636_);
lean_dec(v_val_1635_);
v___x_1639_ = lean_box(0);
v_isShared_1640_ = v_isSharedCheck_1648_;
goto v_resetjp_1638_;
}
v_resetjp_1638_:
{
lean_object* v___x_1641_; lean_object* v___x_1643_; 
v___x_1641_ = lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13___redArg(v_snd_1637_, v_ys_1614_);
if (v_isShared_1640_ == 0)
{
lean_ctor_set_tag(v___x_1639_, 1);
lean_ctor_set(v___x_1639_, 1, v___x_1641_);
v___x_1643_ = v___x_1639_;
goto v_reusejp_1642_;
}
else
{
lean_object* v_reuseFailAlloc_1647_; 
v_reuseFailAlloc_1647_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1647_, 0, v_fst_1636_);
lean_ctor_set(v_reuseFailAlloc_1647_, 1, v___x_1641_);
v___x_1643_ = v_reuseFailAlloc_1647_;
goto v_reusejp_1642_;
}
v_reusejp_1642_:
{
lean_object* v___x_1645_; 
if (v_isShared_1629_ == 0)
{
lean_ctor_set(v___x_1628_, 0, v___x_1643_);
v___x_1645_ = v___x_1628_;
goto v_reusejp_1644_;
}
else
{
lean_object* v_reuseFailAlloc_1646_; 
v_reuseFailAlloc_1646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1646_, 0, v___x_1643_);
v___x_1645_ = v_reuseFailAlloc_1646_;
goto v_reusejp_1644_;
}
v_reusejp_1644_:
{
return v___x_1645_;
}
}
}
}
}
}
else
{
lean_object* v_a_1650_; lean_object* v___x_1652_; uint8_t v_isShared_1653_; uint8_t v_isSharedCheck_1657_; 
lean_dec(v_ys_1614_);
v_a_1650_ = lean_ctor_get(v___x_1625_, 0);
v_isSharedCheck_1657_ = !lean_is_exclusive(v___x_1625_);
if (v_isSharedCheck_1657_ == 0)
{
v___x_1652_ = v___x_1625_;
v_isShared_1653_ = v_isSharedCheck_1657_;
goto v_resetjp_1651_;
}
else
{
lean_inc(v_a_1650_);
lean_dec(v___x_1625_);
v___x_1652_ = lean_box(0);
v_isShared_1653_ = v_isSharedCheck_1657_;
goto v_resetjp_1651_;
}
v_resetjp_1651_:
{
lean_object* v___x_1655_; 
if (v_isShared_1653_ == 0)
{
v___x_1655_ = v___x_1652_;
goto v_reusejp_1654_;
}
else
{
lean_object* v_reuseFailAlloc_1656_; 
v_reuseFailAlloc_1656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1656_, 0, v_a_1650_);
v___x_1655_ = v_reuseFailAlloc_1656_;
goto v_reusejp_1654_;
}
v_reusejp_1654_:
{
return v___x_1655_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg___lam__0___boxed(lean_object* v_xs_1658_, lean_object* v_ys_1659_, lean_object* v_x_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_, lean_object* v___y_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_){
_start:
{
lean_object* v_res_1670_; 
v_res_1670_ = lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg___lam__0(v_xs_1658_, v_ys_1659_, v_x_1660_, v___y_1661_, v___y_1662_, v___y_1663_, v___y_1664_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_);
lean_dec(v___y_1668_);
lean_dec_ref(v___y_1667_);
lean_dec(v___y_1666_);
lean_dec_ref(v___y_1665_);
lean_dec(v___y_1664_);
lean_dec_ref(v___y_1663_);
lean_dec(v___y_1662_);
lean_dec_ref(v___y_1661_);
return v_res_1670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg(lean_object* v_ys_1671_, lean_object* v_xs_1672_){
_start:
{
lean_object* v___f_1673_; lean_object* v___x_1674_; 
v___f_1673_ = lean_alloc_closure((void*)(lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg___lam__0___boxed), 12, 2);
lean_closure_set(v___f_1673_, 0, v_xs_1672_);
lean_closure_set(v___f_1673_, 1, v_ys_1671_);
v___x_1674_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1674_, 0, v___f_1673_);
return v___x_1674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21___redArg___lam__0(lean_object* v_snd_1675_, lean_object* v_ys_1676_, lean_object* v_fst_1677_, lean_object* v_x_1678_){
_start:
{
lean_object* v___x_1679_; lean_object* v___x_1680_; 
v___x_1679_ = lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13___redArg(v_snd_1675_, v_ys_1676_);
v___x_1680_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1680_, 0, v_fst_1677_);
lean_ctor_set(v___x_1680_, 1, v___x_1679_);
return v___x_1680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21___redArg(lean_object* v_ys_1681_, lean_object* v_xs_1682_){
_start:
{
lean_object* v___x_1683_; 
lean_inc(v_xs_1682_);
v___x_1683_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl(lean_box(0), lean_box(0), v_xs_1682_);
if (lean_obj_tag(v___x_1683_) == 0)
{
lean_object* v___x_1684_; 
v___x_1684_ = lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg(v_ys_1681_, v_xs_1682_);
return v___x_1684_;
}
else
{
lean_object* v_val_1685_; lean_object* v___x_1687_; uint8_t v_isShared_1688_; uint8_t v_isSharedCheck_1705_; 
lean_dec(v_xs_1682_);
v_val_1685_ = lean_ctor_get(v___x_1683_, 0);
v_isSharedCheck_1705_ = !lean_is_exclusive(v___x_1683_);
if (v_isSharedCheck_1705_ == 0)
{
v___x_1687_ = v___x_1683_;
v_isShared_1688_ = v_isSharedCheck_1705_;
goto v_resetjp_1686_;
}
else
{
lean_inc(v_val_1685_);
lean_dec(v___x_1683_);
v___x_1687_ = lean_box(0);
v_isShared_1688_ = v_isSharedCheck_1705_;
goto v_resetjp_1686_;
}
v_resetjp_1686_:
{
if (lean_obj_tag(v_val_1685_) == 0)
{
lean_object* v___x_1689_; lean_object* v___x_1691_; 
v___x_1689_ = lean_mk_thunk(v_ys_1681_);
if (v_isShared_1688_ == 0)
{
lean_ctor_set_tag(v___x_1687_, 2);
lean_ctor_set(v___x_1687_, 0, v___x_1689_);
v___x_1691_ = v___x_1687_;
goto v_reusejp_1690_;
}
else
{
lean_object* v_reuseFailAlloc_1692_; 
v_reuseFailAlloc_1692_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1692_, 0, v___x_1689_);
v___x_1691_ = v_reuseFailAlloc_1692_;
goto v_reusejp_1690_;
}
v_reusejp_1690_:
{
return v___x_1691_;
}
}
else
{
lean_object* v_val_1693_; lean_object* v___x_1695_; uint8_t v_isShared_1696_; uint8_t v_isSharedCheck_1704_; 
lean_del_object(v___x_1687_);
v_val_1693_ = lean_ctor_get(v_val_1685_, 0);
v_isSharedCheck_1704_ = !lean_is_exclusive(v_val_1685_);
if (v_isSharedCheck_1704_ == 0)
{
v___x_1695_ = v_val_1685_;
v_isShared_1696_ = v_isSharedCheck_1704_;
goto v_resetjp_1694_;
}
else
{
lean_inc(v_val_1693_);
lean_dec(v_val_1685_);
v___x_1695_ = lean_box(0);
v_isShared_1696_ = v_isSharedCheck_1704_;
goto v_resetjp_1694_;
}
v_resetjp_1694_:
{
lean_object* v_fst_1697_; lean_object* v_snd_1698_; lean_object* v___f_1699_; lean_object* v___x_1700_; lean_object* v___x_1702_; 
v_fst_1697_ = lean_ctor_get(v_val_1693_, 0);
lean_inc(v_fst_1697_);
v_snd_1698_ = lean_ctor_get(v_val_1693_, 1);
lean_inc(v_snd_1698_);
lean_dec(v_val_1693_);
v___f_1699_ = lean_alloc_closure((void*)(lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1699_, 0, v_snd_1698_);
lean_closure_set(v___f_1699_, 1, v_ys_1681_);
lean_closure_set(v___f_1699_, 2, v_fst_1697_);
v___x_1700_ = lean_mk_thunk(v___f_1699_);
if (v_isShared_1696_ == 0)
{
lean_ctor_set_tag(v___x_1695_, 2);
lean_ctor_set(v___x_1695_, 0, v___x_1700_);
v___x_1702_ = v___x_1695_;
goto v_reusejp_1701_;
}
else
{
lean_object* v_reuseFailAlloc_1703_; 
v_reuseFailAlloc_1703_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1703_, 0, v___x_1700_);
v___x_1702_ = v_reuseFailAlloc_1703_;
goto v_reusejp_1701_;
}
v_reusejp_1701_:
{
return v___x_1702_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13___redArg(lean_object* v_xs_1706_, lean_object* v_ys_1707_){
_start:
{
lean_object* v___x_1708_; 
v___x_1708_ = lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21___redArg(v_ys_1707_, v_xs_1706_);
return v___x_1708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_nil___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__9(lean_object* v_00_u03b1_1709_){
_start:
{
lean_object* v___x_1710_; 
v___x_1710_ = lean_box(0);
return v___x_1710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__0(lean_object* v_r_1711_, lean_object* v_x_1712_){
_start:
{
lean_inc(v_r_1711_);
return v_r_1711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__0___boxed(lean_object* v_r_1713_, lean_object* v_x_1714_){
_start:
{
lean_object* v_res_1715_; 
v_res_1715_ = lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__0(v_r_1713_, v_x_1714_);
lean_dec(v_r_1713_);
return v_res_1715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__1(lean_object* v_L_1716_, lean_object* v_f_1717_, lean_object* v_x_1718_, lean_object* v___y_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_, lean_object* v___y_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_){
_start:
{
lean_object* v___x_1728_; 
v___x_1728_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(v_L_1716_, v___y_1719_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_, v___y_1725_, v___y_1726_);
if (lean_obj_tag(v___x_1728_) == 0)
{
lean_object* v_a_1729_; lean_object* v___x_1731_; uint8_t v_isShared_1732_; uint8_t v_isSharedCheck_1786_; 
v_a_1729_ = lean_ctor_get(v___x_1728_, 0);
v_isSharedCheck_1786_ = !lean_is_exclusive(v___x_1728_);
if (v_isSharedCheck_1786_ == 0)
{
v___x_1731_ = v___x_1728_;
v_isShared_1732_ = v_isSharedCheck_1786_;
goto v_resetjp_1730_;
}
else
{
lean_inc(v_a_1729_);
lean_dec(v___x_1728_);
v___x_1731_ = lean_box(0);
v_isShared_1732_ = v_isSharedCheck_1786_;
goto v_resetjp_1730_;
}
v_resetjp_1730_:
{
if (lean_obj_tag(v_a_1729_) == 0)
{
lean_object* v___x_1733_; lean_object* v___x_1735_; 
lean_dec(v_f_1717_);
v___x_1733_ = lean_box(0);
if (v_isShared_1732_ == 0)
{
lean_ctor_set(v___x_1731_, 0, v___x_1733_);
v___x_1735_ = v___x_1731_;
goto v_reusejp_1734_;
}
else
{
lean_object* v_reuseFailAlloc_1736_; 
v_reuseFailAlloc_1736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1736_, 0, v___x_1733_);
v___x_1735_ = v_reuseFailAlloc_1736_;
goto v_reusejp_1734_;
}
v_reusejp_1734_:
{
return v___x_1735_;
}
}
else
{
lean_object* v_val_1737_; lean_object* v_fst_1738_; lean_object* v_snd_1739_; lean_object* v_fst_1740_; lean_object* v_snd_1741_; uint8_t v___x_1742_; lean_object* v___x_1743_; 
lean_del_object(v___x_1731_);
v_val_1737_ = lean_ctor_get(v_a_1729_, 0);
lean_inc(v_val_1737_);
lean_dec_ref_known(v_a_1729_, 1);
v_fst_1738_ = lean_ctor_get(v_val_1737_, 0);
lean_inc(v_fst_1738_);
v_snd_1739_ = lean_ctor_get(v_val_1737_, 1);
lean_inc(v_snd_1739_);
lean_dec(v_val_1737_);
v_fst_1740_ = lean_ctor_get(v_fst_1738_, 0);
lean_inc(v_fst_1740_);
v_snd_1741_ = lean_ctor_get(v_fst_1738_, 1);
lean_inc(v_snd_1741_);
lean_dec(v_fst_1738_);
v___x_1742_ = 0;
v___x_1743_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_snd_1741_, v___x_1742_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_, v___y_1725_, v___y_1726_);
if (lean_obj_tag(v___x_1743_) == 0)
{
lean_object* v_r_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; 
lean_dec_ref_known(v___x_1743_, 1);
lean_inc(v_f_1717_);
v_r_1744_ = lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg(v_snd_1739_, v_f_1717_);
v___x_1745_ = lean_apply_1(v_f_1717_, v_fst_1740_);
v___x_1746_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(v___x_1745_, v___y_1719_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_, v___y_1725_, v___y_1726_);
if (lean_obj_tag(v___x_1746_) == 0)
{
lean_object* v_a_1747_; lean_object* v___x_1749_; uint8_t v_isShared_1750_; uint8_t v_isSharedCheck_1769_; 
v_a_1747_ = lean_ctor_get(v___x_1746_, 0);
v_isSharedCheck_1769_ = !lean_is_exclusive(v___x_1746_);
if (v_isSharedCheck_1769_ == 0)
{
v___x_1749_ = v___x_1746_;
v_isShared_1750_ = v_isSharedCheck_1769_;
goto v_resetjp_1748_;
}
else
{
lean_inc(v_a_1747_);
lean_dec(v___x_1746_);
v___x_1749_ = lean_box(0);
v_isShared_1750_ = v_isSharedCheck_1769_;
goto v_resetjp_1748_;
}
v_resetjp_1748_:
{
if (lean_obj_tag(v_a_1747_) == 0)
{
lean_object* v___x_1752_; 
if (v_isShared_1750_ == 0)
{
lean_ctor_set(v___x_1749_, 0, v_r_1744_);
v___x_1752_ = v___x_1749_;
goto v_reusejp_1751_;
}
else
{
lean_object* v_reuseFailAlloc_1753_; 
v_reuseFailAlloc_1753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1753_, 0, v_r_1744_);
v___x_1752_ = v_reuseFailAlloc_1753_;
goto v_reusejp_1751_;
}
v_reusejp_1751_:
{
return v___x_1752_;
}
}
else
{
lean_object* v_val_1754_; lean_object* v_fst_1755_; lean_object* v_snd_1756_; lean_object* v___x_1758_; uint8_t v_isShared_1759_; uint8_t v_isSharedCheck_1768_; 
v_val_1754_ = lean_ctor_get(v_a_1747_, 0);
lean_inc(v_val_1754_);
lean_dec_ref_known(v_a_1747_, 1);
v_fst_1755_ = lean_ctor_get(v_val_1754_, 0);
v_snd_1756_ = lean_ctor_get(v_val_1754_, 1);
v_isSharedCheck_1768_ = !lean_is_exclusive(v_val_1754_);
if (v_isSharedCheck_1768_ == 0)
{
v___x_1758_ = v_val_1754_;
v_isShared_1759_ = v_isSharedCheck_1768_;
goto v_resetjp_1757_;
}
else
{
lean_inc(v_snd_1756_);
lean_inc(v_fst_1755_);
lean_dec(v_val_1754_);
v___x_1758_ = lean_box(0);
v_isShared_1759_ = v_isSharedCheck_1768_;
goto v_resetjp_1757_;
}
v_resetjp_1757_:
{
lean_object* v___f_1760_; lean_object* v___x_1761_; lean_object* v___x_1763_; 
v___f_1760_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1760_, 0, v_r_1744_);
v___x_1761_ = lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13___redArg(v_snd_1756_, v___f_1760_);
if (v_isShared_1759_ == 0)
{
lean_ctor_set_tag(v___x_1758_, 1);
lean_ctor_set(v___x_1758_, 1, v___x_1761_);
v___x_1763_ = v___x_1758_;
goto v_reusejp_1762_;
}
else
{
lean_object* v_reuseFailAlloc_1767_; 
v_reuseFailAlloc_1767_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1767_, 0, v_fst_1755_);
lean_ctor_set(v_reuseFailAlloc_1767_, 1, v___x_1761_);
v___x_1763_ = v_reuseFailAlloc_1767_;
goto v_reusejp_1762_;
}
v_reusejp_1762_:
{
lean_object* v___x_1765_; 
if (v_isShared_1750_ == 0)
{
lean_ctor_set(v___x_1749_, 0, v___x_1763_);
v___x_1765_ = v___x_1749_;
goto v_reusejp_1764_;
}
else
{
lean_object* v_reuseFailAlloc_1766_; 
v_reuseFailAlloc_1766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1766_, 0, v___x_1763_);
v___x_1765_ = v_reuseFailAlloc_1766_;
goto v_reusejp_1764_;
}
v_reusejp_1764_:
{
return v___x_1765_;
}
}
}
}
}
}
else
{
lean_object* v_a_1770_; lean_object* v___x_1772_; uint8_t v_isShared_1773_; uint8_t v_isSharedCheck_1777_; 
lean_dec(v_r_1744_);
v_a_1770_ = lean_ctor_get(v___x_1746_, 0);
v_isSharedCheck_1777_ = !lean_is_exclusive(v___x_1746_);
if (v_isSharedCheck_1777_ == 0)
{
v___x_1772_ = v___x_1746_;
v_isShared_1773_ = v_isSharedCheck_1777_;
goto v_resetjp_1771_;
}
else
{
lean_inc(v_a_1770_);
lean_dec(v___x_1746_);
v___x_1772_ = lean_box(0);
v_isShared_1773_ = v_isSharedCheck_1777_;
goto v_resetjp_1771_;
}
v_resetjp_1771_:
{
lean_object* v___x_1775_; 
if (v_isShared_1773_ == 0)
{
v___x_1775_ = v___x_1772_;
goto v_reusejp_1774_;
}
else
{
lean_object* v_reuseFailAlloc_1776_; 
v_reuseFailAlloc_1776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1776_, 0, v_a_1770_);
v___x_1775_ = v_reuseFailAlloc_1776_;
goto v_reusejp_1774_;
}
v_reusejp_1774_:
{
return v___x_1775_;
}
}
}
}
else
{
lean_object* v_a_1778_; lean_object* v___x_1780_; uint8_t v_isShared_1781_; uint8_t v_isSharedCheck_1785_; 
lean_dec(v_fst_1740_);
lean_dec(v_snd_1739_);
lean_dec(v_f_1717_);
v_a_1778_ = lean_ctor_get(v___x_1743_, 0);
v_isSharedCheck_1785_ = !lean_is_exclusive(v___x_1743_);
if (v_isSharedCheck_1785_ == 0)
{
v___x_1780_ = v___x_1743_;
v_isShared_1781_ = v_isSharedCheck_1785_;
goto v_resetjp_1779_;
}
else
{
lean_inc(v_a_1778_);
lean_dec(v___x_1743_);
v___x_1780_ = lean_box(0);
v_isShared_1781_ = v_isSharedCheck_1785_;
goto v_resetjp_1779_;
}
v_resetjp_1779_:
{
lean_object* v___x_1783_; 
if (v_isShared_1781_ == 0)
{
v___x_1783_ = v___x_1780_;
goto v_reusejp_1782_;
}
else
{
lean_object* v_reuseFailAlloc_1784_; 
v_reuseFailAlloc_1784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1784_, 0, v_a_1778_);
v___x_1783_ = v_reuseFailAlloc_1784_;
goto v_reusejp_1782_;
}
v_reusejp_1782_:
{
return v___x_1783_;
}
}
}
}
}
}
else
{
lean_object* v_a_1787_; lean_object* v___x_1789_; uint8_t v_isShared_1790_; uint8_t v_isSharedCheck_1794_; 
lean_dec(v_f_1717_);
v_a_1787_ = lean_ctor_get(v___x_1728_, 0);
v_isSharedCheck_1794_ = !lean_is_exclusive(v___x_1728_);
if (v_isSharedCheck_1794_ == 0)
{
v___x_1789_ = v___x_1728_;
v_isShared_1790_ = v_isSharedCheck_1794_;
goto v_resetjp_1788_;
}
else
{
lean_inc(v_a_1787_);
lean_dec(v___x_1728_);
v___x_1789_ = lean_box(0);
v_isShared_1790_ = v_isSharedCheck_1794_;
goto v_resetjp_1788_;
}
v_resetjp_1788_:
{
lean_object* v___x_1792_; 
if (v_isShared_1790_ == 0)
{
v___x_1792_ = v___x_1789_;
goto v_reusejp_1791_;
}
else
{
lean_object* v_reuseFailAlloc_1793_; 
v_reuseFailAlloc_1793_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1793_, 0, v_a_1787_);
v___x_1792_ = v_reuseFailAlloc_1793_;
goto v_reusejp_1791_;
}
v_reusejp_1791_:
{
return v___x_1792_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__1___boxed(lean_object* v_L_1795_, lean_object* v_f_1796_, lean_object* v_x_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_){
_start:
{
lean_object* v_res_1807_; 
v_res_1807_ = lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__1(v_L_1795_, v_f_1796_, v_x_1797_, v___y_1798_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_, v___y_1803_, v___y_1804_, v___y_1805_);
lean_dec(v___y_1805_);
lean_dec_ref(v___y_1804_);
lean_dec(v___y_1803_);
lean_dec_ref(v___y_1802_);
lean_dec(v___y_1801_);
lean_dec_ref(v___y_1800_);
lean_dec(v___y_1799_);
lean_dec_ref(v___y_1798_);
return v_res_1807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg(lean_object* v_L_1808_, lean_object* v_f_1809_){
_start:
{
lean_object* v___f_1810_; lean_object* v___x_1811_; 
v___f_1810_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg___lam__1___boxed), 12, 2);
lean_closure_set(v___f_1810_, 0, v_L_1808_);
lean_closure_set(v___f_1810_, 1, v_f_1809_);
v___x_1811_ = lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg(v___f_1810_);
return v___x_1811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg___lam__0(lean_object* v_x_1812_, lean_object* v_x_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_){
_start:
{
lean_object* v___x_1823_; 
lean_inc(v___y_1821_);
lean_inc_ref(v___y_1820_);
lean_inc(v___y_1819_);
lean_inc_ref(v___y_1818_);
lean_inc(v___y_1817_);
lean_inc_ref(v___y_1816_);
lean_inc(v___y_1815_);
lean_inc_ref(v___y_1814_);
v___x_1823_ = lean_apply_9(v_x_1812_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_, v___y_1820_, v___y_1821_, lean_box(0));
if (lean_obj_tag(v___x_1823_) == 0)
{
lean_object* v_a_1824_; lean_object* v___x_1826_; uint8_t v_isShared_1827_; uint8_t v_isSharedCheck_1833_; 
v_a_1824_ = lean_ctor_get(v___x_1823_, 0);
v_isSharedCheck_1833_ = !lean_is_exclusive(v___x_1823_);
if (v_isSharedCheck_1833_ == 0)
{
v___x_1826_ = v___x_1823_;
v_isShared_1827_ = v_isSharedCheck_1833_;
goto v_resetjp_1825_;
}
else
{
lean_inc(v_a_1824_);
lean_dec(v___x_1823_);
v___x_1826_ = lean_box(0);
v_isShared_1827_ = v_isSharedCheck_1833_;
goto v_resetjp_1825_;
}
v_resetjp_1825_:
{
lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1831_; 
v___x_1828_ = lean_box(0);
v___x_1829_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1829_, 0, v_a_1824_);
lean_ctor_set(v___x_1829_, 1, v___x_1828_);
if (v_isShared_1827_ == 0)
{
lean_ctor_set(v___x_1826_, 0, v___x_1829_);
v___x_1831_ = v___x_1826_;
goto v_reusejp_1830_;
}
else
{
lean_object* v_reuseFailAlloc_1832_; 
v_reuseFailAlloc_1832_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1832_, 0, v___x_1829_);
v___x_1831_ = v_reuseFailAlloc_1832_;
goto v_reusejp_1830_;
}
v_reusejp_1830_:
{
return v___x_1831_;
}
}
}
else
{
lean_object* v_a_1834_; lean_object* v___x_1836_; uint8_t v_isShared_1837_; uint8_t v_isSharedCheck_1841_; 
v_a_1834_ = lean_ctor_get(v___x_1823_, 0);
v_isSharedCheck_1841_ = !lean_is_exclusive(v___x_1823_);
if (v_isSharedCheck_1841_ == 0)
{
v___x_1836_ = v___x_1823_;
v_isShared_1837_ = v_isSharedCheck_1841_;
goto v_resetjp_1835_;
}
else
{
lean_inc(v_a_1834_);
lean_dec(v___x_1823_);
v___x_1836_ = lean_box(0);
v_isShared_1837_ = v_isSharedCheck_1841_;
goto v_resetjp_1835_;
}
v_resetjp_1835_:
{
lean_object* v___x_1839_; 
if (v_isShared_1837_ == 0)
{
v___x_1839_ = v___x_1836_;
goto v_reusejp_1838_;
}
else
{
lean_object* v_reuseFailAlloc_1840_; 
v_reuseFailAlloc_1840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1840_, 0, v_a_1834_);
v___x_1839_ = v_reuseFailAlloc_1840_;
goto v_reusejp_1838_;
}
v_reusejp_1838_:
{
return v___x_1839_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg___lam__0___boxed(lean_object* v_x_1842_, lean_object* v_x_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_){
_start:
{
lean_object* v_res_1853_; 
v_res_1853_ = lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg___lam__0(v_x_1842_, v_x_1843_, v___y_1844_, v___y_1845_, v___y_1846_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_, v___y_1851_);
lean_dec(v___y_1851_);
lean_dec_ref(v___y_1850_);
lean_dec(v___y_1849_);
lean_dec_ref(v___y_1848_);
lean_dec(v___y_1847_);
lean_dec_ref(v___y_1846_);
lean_dec(v___y_1845_);
lean_dec_ref(v___y_1844_);
return v_res_1853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg(lean_object* v_x_1854_){
_start:
{
lean_object* v___f_1855_; lean_object* v___x_1856_; 
v___f_1855_ = lean_alloc_closure((void*)(lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg___lam__0___boxed), 11, 1);
lean_closure_set(v___f_1855_, 0, v_x_1854_);
v___x_1856_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1856_, 0, v___f_1855_);
return v___x_1856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg___lam__0(lean_object* v_x_1857_, lean_object* v___y_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_){
_start:
{
lean_object* v___x_1867_; 
lean_inc(v___y_1865_);
lean_inc(v___y_1863_);
lean_inc(v___y_1861_);
lean_inc(v___y_1859_);
v___x_1867_ = lean_apply_9(v_x_1857_, v___y_1858_, v___y_1859_, v___y_1860_, v___y_1861_, v___y_1862_, v___y_1863_, v___y_1864_, v___y_1865_, lean_box(0));
if (lean_obj_tag(v___x_1867_) == 0)
{
lean_object* v_a_1868_; lean_object* v___x_1869_; 
v_a_1868_ = lean_ctor_get(v___x_1867_, 0);
lean_inc(v_a_1868_);
lean_dec_ref_known(v___x_1867_, 1);
v___x_1869_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1859_, v___y_1861_, v___y_1863_, v___y_1865_);
lean_dec(v___y_1865_);
lean_dec(v___y_1863_);
lean_dec(v___y_1861_);
lean_dec(v___y_1859_);
if (lean_obj_tag(v___x_1869_) == 0)
{
lean_object* v_a_1870_; lean_object* v___x_1872_; uint8_t v_isShared_1873_; uint8_t v_isSharedCheck_1878_; 
v_a_1870_ = lean_ctor_get(v___x_1869_, 0);
v_isSharedCheck_1878_ = !lean_is_exclusive(v___x_1869_);
if (v_isSharedCheck_1878_ == 0)
{
v___x_1872_ = v___x_1869_;
v_isShared_1873_ = v_isSharedCheck_1878_;
goto v_resetjp_1871_;
}
else
{
lean_inc(v_a_1870_);
lean_dec(v___x_1869_);
v___x_1872_ = lean_box(0);
v_isShared_1873_ = v_isSharedCheck_1878_;
goto v_resetjp_1871_;
}
v_resetjp_1871_:
{
lean_object* v___x_1874_; lean_object* v___x_1876_; 
v___x_1874_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1874_, 0, v_a_1868_);
lean_ctor_set(v___x_1874_, 1, v_a_1870_);
if (v_isShared_1873_ == 0)
{
lean_ctor_set(v___x_1872_, 0, v___x_1874_);
v___x_1876_ = v___x_1872_;
goto v_reusejp_1875_;
}
else
{
lean_object* v_reuseFailAlloc_1877_; 
v_reuseFailAlloc_1877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1877_, 0, v___x_1874_);
v___x_1876_ = v_reuseFailAlloc_1877_;
goto v_reusejp_1875_;
}
v_reusejp_1875_:
{
return v___x_1876_;
}
}
}
else
{
lean_object* v_a_1879_; lean_object* v___x_1881_; uint8_t v_isShared_1882_; uint8_t v_isSharedCheck_1886_; 
lean_dec(v_a_1868_);
v_a_1879_ = lean_ctor_get(v___x_1869_, 0);
v_isSharedCheck_1886_ = !lean_is_exclusive(v___x_1869_);
if (v_isSharedCheck_1886_ == 0)
{
v___x_1881_ = v___x_1869_;
v_isShared_1882_ = v_isSharedCheck_1886_;
goto v_resetjp_1880_;
}
else
{
lean_inc(v_a_1879_);
lean_dec(v___x_1869_);
v___x_1881_ = lean_box(0);
v_isShared_1882_ = v_isSharedCheck_1886_;
goto v_resetjp_1880_;
}
v_resetjp_1880_:
{
lean_object* v___x_1884_; 
if (v_isShared_1882_ == 0)
{
v___x_1884_ = v___x_1881_;
goto v_reusejp_1883_;
}
else
{
lean_object* v_reuseFailAlloc_1885_; 
v_reuseFailAlloc_1885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1885_, 0, v_a_1879_);
v___x_1884_ = v_reuseFailAlloc_1885_;
goto v_reusejp_1883_;
}
v_reusejp_1883_:
{
return v___x_1884_;
}
}
}
}
else
{
lean_object* v_a_1887_; lean_object* v___x_1889_; uint8_t v_isShared_1890_; uint8_t v_isSharedCheck_1894_; 
lean_dec(v___y_1865_);
lean_dec(v___y_1863_);
lean_dec(v___y_1861_);
lean_dec(v___y_1859_);
v_a_1887_ = lean_ctor_get(v___x_1867_, 0);
v_isSharedCheck_1894_ = !lean_is_exclusive(v___x_1867_);
if (v_isSharedCheck_1894_ == 0)
{
v___x_1889_ = v___x_1867_;
v_isShared_1890_ = v_isSharedCheck_1894_;
goto v_resetjp_1888_;
}
else
{
lean_inc(v_a_1887_);
lean_dec(v___x_1867_);
v___x_1889_ = lean_box(0);
v_isShared_1890_ = v_isSharedCheck_1894_;
goto v_resetjp_1888_;
}
v_resetjp_1888_:
{
lean_object* v___x_1892_; 
if (v_isShared_1890_ == 0)
{
v___x_1892_ = v___x_1889_;
goto v_reusejp_1891_;
}
else
{
lean_object* v_reuseFailAlloc_1893_; 
v_reuseFailAlloc_1893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1893_, 0, v_a_1887_);
v___x_1892_ = v_reuseFailAlloc_1893_;
goto v_reusejp_1891_;
}
v_reusejp_1891_:
{
return v___x_1892_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg___lam__0___boxed(lean_object* v_x_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_){
_start:
{
lean_object* v_res_1905_; 
v_res_1905_ = lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg___lam__0(v_x_1895_, v___y_1896_, v___y_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_);
return v_res_1905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg(lean_object* v_x_1906_){
_start:
{
lean_object* v___f_1907_; lean_object* v___x_1908_; 
v___f_1907_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1907_, 0, v_x_1906_);
v___x_1908_ = lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg(v___f_1907_);
return v___x_1908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg___lam__0(lean_object* v_x_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
lean_object* v___x_1919_; 
v___x_1919_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1919_, 0, v_x_1909_);
return v___x_1919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg___lam__0___boxed(lean_object* v_x_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_){
_start:
{
lean_object* v_res_1930_; 
v_res_1930_ = lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg___lam__0(v_x_1920_, v___y_1921_, v___y_1922_, v___y_1923_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_);
lean_dec(v___y_1928_);
lean_dec_ref(v___y_1927_);
lean_dec(v___y_1926_);
lean_dec_ref(v___y_1925_);
lean_dec(v___y_1924_);
lean_dec_ref(v___y_1923_);
lean_dec(v___y_1922_);
lean_dec_ref(v___y_1921_);
return v_res_1930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg(lean_object* v_x_1931_){
_start:
{
lean_object* v___f_1932_; lean_object* v___x_1933_; 
v___f_1932_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1932_, 0, v_x_1931_);
v___x_1933_ = lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg(v___f_1932_);
return v___x_1933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg___lam__0(lean_object* v_x_1934_, lean_object* v_x_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_){
_start:
{
lean_object* v___x_1945_; 
lean_inc(v___y_1943_);
lean_inc_ref(v___y_1942_);
lean_inc(v___y_1941_);
lean_inc_ref(v___y_1940_);
lean_inc(v___y_1939_);
lean_inc_ref(v___y_1938_);
lean_inc(v___y_1937_);
lean_inc_ref(v___y_1936_);
v___x_1945_ = lean_apply_9(v_x_1934_, v___y_1936_, v___y_1937_, v___y_1938_, v___y_1939_, v___y_1940_, v___y_1941_, v___y_1942_, v___y_1943_, lean_box(0));
if (lean_obj_tag(v___x_1945_) == 0)
{
lean_object* v_a_1946_; lean_object* v___x_1948_; uint8_t v_isShared_1949_; uint8_t v_isSharedCheck_1959_; 
v_a_1946_ = lean_ctor_get(v___x_1945_, 0);
v_isSharedCheck_1959_ = !lean_is_exclusive(v___x_1945_);
if (v_isSharedCheck_1959_ == 0)
{
v___x_1948_ = v___x_1945_;
v_isShared_1949_ = v_isSharedCheck_1959_;
goto v_resetjp_1947_;
}
else
{
lean_inc(v_a_1946_);
lean_dec(v___x_1945_);
v___x_1948_ = lean_box(0);
v_isShared_1949_ = v_isSharedCheck_1959_;
goto v_resetjp_1947_;
}
v_resetjp_1947_:
{
if (lean_obj_tag(v_a_1946_) == 0)
{
lean_object* v___x_1950_; lean_object* v___x_1952_; 
v___x_1950_ = lean_box(0);
if (v_isShared_1949_ == 0)
{
lean_ctor_set(v___x_1948_, 0, v___x_1950_);
v___x_1952_ = v___x_1948_;
goto v_reusejp_1951_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v___x_1950_);
v___x_1952_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1951_;
}
v_reusejp_1951_:
{
return v___x_1952_;
}
}
else
{
lean_object* v_val_1954_; lean_object* v___x_1955_; lean_object* v___x_1957_; 
v_val_1954_ = lean_ctor_get(v_a_1946_, 0);
lean_inc(v_val_1954_);
lean_dec_ref_known(v_a_1946_, 1);
v___x_1955_ = lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg(v_val_1954_);
if (v_isShared_1949_ == 0)
{
lean_ctor_set(v___x_1948_, 0, v___x_1955_);
v___x_1957_ = v___x_1948_;
goto v_reusejp_1956_;
}
else
{
lean_object* v_reuseFailAlloc_1958_; 
v_reuseFailAlloc_1958_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1958_, 0, v___x_1955_);
v___x_1957_ = v_reuseFailAlloc_1958_;
goto v_reusejp_1956_;
}
v_reusejp_1956_:
{
return v___x_1957_;
}
}
}
}
else
{
lean_object* v_a_1960_; lean_object* v___x_1962_; uint8_t v_isShared_1963_; uint8_t v_isSharedCheck_1967_; 
v_a_1960_ = lean_ctor_get(v___x_1945_, 0);
v_isSharedCheck_1967_ = !lean_is_exclusive(v___x_1945_);
if (v_isSharedCheck_1967_ == 0)
{
v___x_1962_ = v___x_1945_;
v_isShared_1963_ = v_isSharedCheck_1967_;
goto v_resetjp_1961_;
}
else
{
lean_inc(v_a_1960_);
lean_dec(v___x_1945_);
v___x_1962_ = lean_box(0);
v_isShared_1963_ = v_isSharedCheck_1967_;
goto v_resetjp_1961_;
}
v_resetjp_1961_:
{
lean_object* v___x_1965_; 
if (v_isShared_1963_ == 0)
{
v___x_1965_ = v___x_1962_;
goto v_reusejp_1964_;
}
else
{
lean_object* v_reuseFailAlloc_1966_; 
v_reuseFailAlloc_1966_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1966_, 0, v_a_1960_);
v___x_1965_ = v_reuseFailAlloc_1966_;
goto v_reusejp_1964_;
}
v_reusejp_1964_:
{
return v___x_1965_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg___lam__0___boxed(lean_object* v_x_1968_, lean_object* v_x_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_){
_start:
{
lean_object* v_res_1979_; 
v_res_1979_ = lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg___lam__0(v_x_1968_, v_x_1969_, v___y_1970_, v___y_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_);
lean_dec(v___y_1977_);
lean_dec_ref(v___y_1976_);
lean_dec(v___y_1975_);
lean_dec_ref(v___y_1974_);
lean_dec(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec(v___y_1971_);
lean_dec_ref(v___y_1970_);
return v_res_1979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg(lean_object* v_x_1980_){
_start:
{
lean_object* v___f_1981_; lean_object* v___x_1982_; 
v___f_1981_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg___lam__0___boxed), 11, 1);
lean_closure_set(v___f_1981_, 0, v_x_1980_);
v___x_1982_ = lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg(v___f_1981_);
return v___x_1982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3___redArg___lam__0(lean_object* v_f_1983_, lean_object* v_a_1984_){
_start:
{
lean_object* v___x_1985_; lean_object* v___x_1986_; 
v___x_1985_ = lean_apply_1(v_f_1983_, v_a_1984_);
v___x_1986_ = lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg(v___x_1985_);
return v___x_1986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3___redArg(lean_object* v_f_1987_, lean_object* v_L_1988_){
_start:
{
lean_object* v___f_1989_; lean_object* v___x_1990_; 
v___f_1989_ = lean_alloc_closure((void*)(lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1989_, 0, v_f_1987_);
v___x_1990_ = lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg(v_L_1988_, v___f_1989_);
return v___x_1990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2(lean_object* v_stx_1995_, lean_object* v___f_1996_, lean_object* v___f_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_){
_start:
{
lean_object* v___y_2021_; lean_object* v___y_2064_; lean_object* v___y_2065_; lean_object* v___y_2066_; lean_object* v___y_2067_; lean_object* v___y_2070_; lean_object* v___y_2071_; lean_object* v___y_2072_; lean_object* v___y_2073_; lean_object* v___x_2075_; lean_object* v_a_2076_; lean_object* v___x_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___y_2081_; lean_object* v___x_2102_; lean_object* v___y_2104_; lean_object* v___y_2105_; uint8_t v___x_2107_; 
v___x_2075_ = lp_mathlib_Mathlib_Tactic_Hint_getHints___redArg(v___y_2005_);
v_a_2076_ = lean_ctor_get(v___x_2075_, 0);
lean_inc(v_a_2076_);
lean_dec_ref(v___x_2075_);
v___x_2077_ = lean_array_mk(v_a_2076_);
v___x_2078_ = lean_unsigned_to_nat(0u);
v___x_2079_ = lean_unsigned_to_nat(1u);
v___x_2102_ = lean_array_get_size(v___x_2077_);
v___x_2107_ = lean_nat_dec_eq(v___x_2102_, v___x_2078_);
if (v___x_2107_ == 0)
{
lean_object* v___x_2108_; lean_object* v___y_2110_; uint8_t v___x_2112_; 
v___x_2108_ = lean_nat_sub(v___x_2102_, v___x_2079_);
v___x_2112_ = lean_nat_dec_le(v___x_2078_, v___x_2108_);
if (v___x_2112_ == 0)
{
lean_inc(v___x_2108_);
v___y_2110_ = v___x_2108_;
goto v___jp_2109_;
}
else
{
v___y_2110_ = v___x_2078_;
goto v___jp_2109_;
}
v___jp_2109_:
{
uint8_t v___x_2111_; 
v___x_2111_ = lean_nat_dec_le(v___y_2110_, v___x_2108_);
if (v___x_2111_ == 0)
{
lean_dec(v___x_2108_);
lean_inc(v___y_2110_);
v___y_2104_ = v___y_2110_;
v___y_2105_ = v___y_2110_;
goto v___jp_2103_;
}
else
{
v___y_2104_ = v___y_2110_;
v___y_2105_ = v___x_2108_;
goto v___jp_2103_;
}
}
}
else
{
v___y_2081_ = v___x_2077_;
goto v___jp_2080_;
}
v___jp_2007_:
{
lean_object* v___x_2008_; 
v___x_2008_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1999_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_);
if (lean_obj_tag(v___x_2008_) == 0)
{
lean_object* v_a_2009_; uint8_t v___x_2010_; lean_object* v___x_2011_; 
v_a_2009_ = lean_ctor_get(v___x_2008_, 0);
lean_inc(v_a_2009_);
lean_dec_ref_known(v___x_2008_, 1);
v___x_2010_ = 1;
v___x_2011_ = l_Lean_Elab_admitGoal(v_a_2009_, v___x_2010_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_);
return v___x_2011_;
}
else
{
lean_object* v_a_2012_; lean_object* v___x_2014_; uint8_t v_isShared_2015_; uint8_t v_isSharedCheck_2019_; 
v_a_2012_ = lean_ctor_get(v___x_2008_, 0);
v_isSharedCheck_2019_ = !lean_is_exclusive(v___x_2008_);
if (v_isSharedCheck_2019_ == 0)
{
v___x_2014_ = v___x_2008_;
v_isShared_2015_ = v_isSharedCheck_2019_;
goto v_resetjp_2013_;
}
else
{
lean_inc(v_a_2012_);
lean_dec(v___x_2008_);
v___x_2014_ = lean_box(0);
v_isShared_2015_ = v_isSharedCheck_2019_;
goto v_resetjp_2013_;
}
v_resetjp_2013_:
{
lean_object* v___x_2017_; 
if (v_isShared_2015_ == 0)
{
v___x_2017_ = v___x_2014_;
goto v_reusejp_2016_;
}
else
{
lean_object* v_reuseFailAlloc_2018_; 
v_reuseFailAlloc_2018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2018_, 0, v_a_2012_);
v___x_2017_ = v_reuseFailAlloc_2018_;
goto v_reusejp_2016_;
}
v_reusejp_2016_:
{
return v___x_2017_;
}
}
}
}
v___jp_2020_:
{
size_t v_sz_2022_; size_t v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; lean_object* v___x_2026_; uint8_t v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; 
v_sz_2022_ = lean_array_size(v___y_2021_);
v___x_2023_ = ((size_t)0ULL);
lean_inc_ref(v___y_2021_);
v___x_2024_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Hint_hint_spec__6(v_sz_2022_, v___x_2023_, v___y_2021_);
v___x_2025_ = lean_box(0);
v___x_2026_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__0));
v___x_2027_ = 4;
v___x_2028_ = l_Lean_MessageData_nil;
v___x_2029_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_1995_, v___x_2024_, v___x_2025_, v___x_2026_, v___x_2025_, v___x_2027_, v___x_2028_, v___y_2004_, v___y_2005_);
if (lean_obj_tag(v___x_2029_) == 0)
{
lean_object* v___x_2031_; uint8_t v_isShared_2032_; uint8_t v_isSharedCheck_2061_; 
v_isSharedCheck_2061_ = !lean_is_exclusive(v___x_2029_);
if (v_isSharedCheck_2061_ == 0)
{
lean_object* v_unused_2062_; 
v_unused_2062_ = lean_ctor_get(v___x_2029_, 0);
lean_dec(v_unused_2062_);
v___x_2031_ = v___x_2029_;
v_isShared_2032_ = v_isSharedCheck_2061_;
goto v_resetjp_2030_;
}
else
{
lean_dec(v___x_2029_);
v___x_2031_ = lean_box(0);
v_isShared_2032_ = v_isSharedCheck_2061_;
goto v_resetjp_2030_;
}
v_resetjp_2030_:
{
lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v_fst_2036_; 
v___x_2033_ = lean_box(0);
v___x_2034_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___closed__1));
v___x_2035_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Hint_hint_spec__7(v___y_2021_, v_sz_2022_, v___x_2023_, v___x_2034_);
lean_dec_ref(v___y_2021_);
v_fst_2036_ = lean_ctor_get(v___x_2035_, 0);
lean_inc(v_fst_2036_);
lean_dec_ref(v___x_2035_);
if (lean_obj_tag(v_fst_2036_) == 0)
{
lean_del_object(v___x_2031_);
goto v___jp_2007_;
}
else
{
lean_object* v_val_2037_; 
v_val_2037_ = lean_ctor_get(v_fst_2036_, 0);
lean_inc(v_val_2037_);
lean_dec_ref_known(v_fst_2036_, 1);
if (lean_obj_tag(v_val_2037_) == 0)
{
lean_del_object(v___x_2031_);
goto v___jp_2007_;
}
else
{
lean_object* v_val_2038_; lean_object* v___x_2039_; lean_object* v_snd_2040_; lean_object* v_term_2041_; lean_object* v_meta_2042_; lean_object* v_meta_2043_; lean_object* v_mctx_2044_; lean_object* v_cache_2045_; lean_object* v_zetaDeltaFVarIds_2046_; lean_object* v_postponed_2047_; lean_object* v_diag_2048_; lean_object* v___x_2050_; uint8_t v_isShared_2051_; uint8_t v_isSharedCheck_2059_; 
v_val_2038_ = lean_ctor_get(v_val_2037_, 0);
lean_inc(v_val_2038_);
lean_dec_ref_known(v_val_2037_, 1);
v___x_2039_ = lean_st_ref_take(v___y_2003_);
v_snd_2040_ = lean_ctor_get(v_val_2038_, 1);
lean_inc(v_snd_2040_);
lean_dec(v_val_2038_);
v_term_2041_ = lean_ctor_get(v_snd_2040_, 0);
lean_inc_ref(v_term_2041_);
lean_dec(v_snd_2040_);
v_meta_2042_ = lean_ctor_get(v_term_2041_, 0);
lean_inc_ref(v_meta_2042_);
lean_dec_ref(v_term_2041_);
v_meta_2043_ = lean_ctor_get(v_meta_2042_, 1);
lean_inc_ref(v_meta_2043_);
lean_dec_ref(v_meta_2042_);
v_mctx_2044_ = lean_ctor_get(v_meta_2043_, 0);
lean_inc_ref(v_mctx_2044_);
lean_dec_ref(v_meta_2043_);
v_cache_2045_ = lean_ctor_get(v___x_2039_, 1);
v_zetaDeltaFVarIds_2046_ = lean_ctor_get(v___x_2039_, 2);
v_postponed_2047_ = lean_ctor_get(v___x_2039_, 3);
v_diag_2048_ = lean_ctor_get(v___x_2039_, 4);
v_isSharedCheck_2059_ = !lean_is_exclusive(v___x_2039_);
if (v_isSharedCheck_2059_ == 0)
{
lean_object* v_unused_2060_; 
v_unused_2060_ = lean_ctor_get(v___x_2039_, 0);
lean_dec(v_unused_2060_);
v___x_2050_ = v___x_2039_;
v_isShared_2051_ = v_isSharedCheck_2059_;
goto v_resetjp_2049_;
}
else
{
lean_inc(v_diag_2048_);
lean_inc(v_postponed_2047_);
lean_inc(v_zetaDeltaFVarIds_2046_);
lean_inc(v_cache_2045_);
lean_dec(v___x_2039_);
v___x_2050_ = lean_box(0);
v_isShared_2051_ = v_isSharedCheck_2059_;
goto v_resetjp_2049_;
}
v_resetjp_2049_:
{
lean_object* v___x_2053_; 
if (v_isShared_2051_ == 0)
{
lean_ctor_set(v___x_2050_, 0, v_mctx_2044_);
v___x_2053_ = v___x_2050_;
goto v_reusejp_2052_;
}
else
{
lean_object* v_reuseFailAlloc_2058_; 
v_reuseFailAlloc_2058_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2058_, 0, v_mctx_2044_);
lean_ctor_set(v_reuseFailAlloc_2058_, 1, v_cache_2045_);
lean_ctor_set(v_reuseFailAlloc_2058_, 2, v_zetaDeltaFVarIds_2046_);
lean_ctor_set(v_reuseFailAlloc_2058_, 3, v_postponed_2047_);
lean_ctor_set(v_reuseFailAlloc_2058_, 4, v_diag_2048_);
v___x_2053_ = v_reuseFailAlloc_2058_;
goto v_reusejp_2052_;
}
v_reusejp_2052_:
{
lean_object* v___x_2054_; lean_object* v___x_2056_; 
v___x_2054_ = lean_st_ref_set(v___y_2003_, v___x_2053_);
if (v_isShared_2032_ == 0)
{
lean_ctor_set(v___x_2031_, 0, v___x_2033_);
v___x_2056_ = v___x_2031_;
goto v_reusejp_2055_;
}
else
{
lean_object* v_reuseFailAlloc_2057_; 
v_reuseFailAlloc_2057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2057_, 0, v___x_2033_);
v___x_2056_ = v_reuseFailAlloc_2057_;
goto v_reusejp_2055_;
}
v_reusejp_2055_:
{
return v___x_2056_;
}
}
}
}
}
}
}
else
{
lean_dec_ref(v___y_2021_);
return v___x_2029_;
}
}
v___jp_2063_:
{
lean_object* v___x_2068_; 
v___x_2068_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg(v___y_2066_, v___y_2065_, v___y_2064_, v___y_2067_);
lean_dec(v___y_2067_);
lean_dec(v___y_2066_);
v___y_2021_ = v___x_2068_;
goto v___jp_2020_;
}
v___jp_2069_:
{
uint8_t v___x_2074_; 
v___x_2074_ = lean_nat_dec_le(v___y_2073_, v___y_2070_);
if (v___x_2074_ == 0)
{
lean_dec(v___y_2070_);
lean_inc(v___y_2073_);
v___y_2064_ = v___y_2073_;
v___y_2065_ = v___y_2071_;
v___y_2066_ = v___y_2072_;
v___y_2067_ = v___y_2073_;
goto v___jp_2063_;
}
else
{
v___y_2064_ = v___y_2073_;
v___y_2065_ = v___y_2071_;
v___y_2066_ = v___y_2072_;
v___y_2067_ = v___y_2070_;
goto v___jp_2063_;
}
}
v___jp_2080_:
{
lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; 
v___x_2082_ = lean_array_to_list(v___y_2081_);
v___x_2083_ = lean_box(0);
v___x_2084_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Hint_hint_spec__1(v___x_2082_, v___x_2083_);
v___x_2085_ = lp_mathlib_Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2___redArg(v___x_2084_);
v___x_2086_ = lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3___redArg(v___f_1996_, v___x_2085_);
v___x_2087_ = lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg(v___x_2086_, v___f_1997_);
v___x_2088_ = lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg(v___x_2087_, v___y_1998_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_);
if (lean_obj_tag(v___x_2088_) == 0)
{
lean_object* v_a_2089_; lean_object* v___x_2090_; uint8_t v___x_2091_; 
v_a_2089_ = lean_ctor_get(v___x_2088_, 0);
lean_inc(v_a_2089_);
lean_dec_ref_known(v___x_2088_, 1);
v___x_2090_ = lean_array_get_size(v_a_2089_);
v___x_2091_ = lean_nat_dec_eq(v___x_2090_, v___x_2078_);
if (v___x_2091_ == 0)
{
lean_object* v___x_2092_; uint8_t v___x_2093_; 
v___x_2092_ = lean_nat_sub(v___x_2090_, v___x_2079_);
v___x_2093_ = lean_nat_dec_le(v___x_2078_, v___x_2092_);
if (v___x_2093_ == 0)
{
lean_inc(v___x_2092_);
v___y_2070_ = v___x_2092_;
v___y_2071_ = v_a_2089_;
v___y_2072_ = v___x_2090_;
v___y_2073_ = v___x_2092_;
goto v___jp_2069_;
}
else
{
v___y_2070_ = v___x_2092_;
v___y_2071_ = v_a_2089_;
v___y_2072_ = v___x_2090_;
v___y_2073_ = v___x_2078_;
goto v___jp_2069_;
}
}
else
{
v___y_2021_ = v_a_2089_;
goto v___jp_2020_;
}
}
else
{
lean_object* v_a_2094_; lean_object* v___x_2096_; uint8_t v_isShared_2097_; uint8_t v_isSharedCheck_2101_; 
lean_dec(v_stx_1995_);
v_a_2094_ = lean_ctor_get(v___x_2088_, 0);
v_isSharedCheck_2101_ = !lean_is_exclusive(v___x_2088_);
if (v_isSharedCheck_2101_ == 0)
{
v___x_2096_ = v___x_2088_;
v_isShared_2097_ = v_isSharedCheck_2101_;
goto v_resetjp_2095_;
}
else
{
lean_inc(v_a_2094_);
lean_dec(v___x_2088_);
v___x_2096_ = lean_box(0);
v_isShared_2097_ = v_isSharedCheck_2101_;
goto v_resetjp_2095_;
}
v_resetjp_2095_:
{
lean_object* v___x_2099_; 
if (v_isShared_2097_ == 0)
{
v___x_2099_ = v___x_2096_;
goto v_reusejp_2098_;
}
else
{
lean_object* v_reuseFailAlloc_2100_; 
v_reuseFailAlloc_2100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2100_, 0, v_a_2094_);
v___x_2099_ = v_reuseFailAlloc_2100_;
goto v_reusejp_2098_;
}
v_reusejp_2098_:
{
return v___x_2099_;
}
}
}
}
v___jp_2103_:
{
lean_object* v___x_2106_; 
v___x_2106_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg(v___x_2102_, v___x_2077_, v___y_2104_, v___y_2105_);
lean_dec(v___y_2105_);
v___y_2081_ = v___x_2106_;
goto v___jp_2080_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___boxed(lean_object* v_stx_2113_, lean_object* v___f_2114_, lean_object* v___f_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_){
_start:
{
lean_object* v_res_2125_; 
v_res_2125_ = lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2(v_stx_2113_, v___f_2114_, v___f_2115_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
lean_dec(v___y_2123_);
lean_dec_ref(v___y_2122_);
lean_dec(v___y_2121_);
lean_dec_ref(v___y_2120_);
lean_dec(v___y_2119_);
lean_dec_ref(v___y_2118_);
lean_dec(v___y_2117_);
lean_dec_ref(v___y_2116_);
return v_res_2125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint(lean_object* v_stx_2128_, lean_object* v_a_2129_, lean_object* v_a_2130_, lean_object* v_a_2131_, lean_object* v_a_2132_, lean_object* v_a_2133_, lean_object* v_a_2134_, lean_object* v_a_2135_, lean_object* v_a_2136_){
_start:
{
lean_object* v___f_2138_; lean_object* v___f_2139_; lean_object* v___f_2140_; lean_object* v___x_2141_; 
v___f_2138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_hint___closed__0));
v___f_2139_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_hint___closed__1));
v___f_2140_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Hint_hint___lam__2___boxed), 12, 3);
lean_closure_set(v___f_2140_, 0, v_stx_2128_);
lean_closure_set(v___f_2140_, 1, v___f_2139_);
lean_closure_set(v___f_2140_, 2, v___f_2138_);
v___x_2141_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2140_, v_a_2129_, v_a_2130_, v_a_2131_, v_a_2132_, v_a_2133_, v_a_2134_, v_a_2135_, v_a_2136_);
return v___x_2141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint_hint___boxed(lean_object* v_stx_2142_, lean_object* v_a_2143_, lean_object* v_a_2144_, lean_object* v_a_2145_, lean_object* v_a_2146_, lean_object* v_a_2147_, lean_object* v_a_2148_, lean_object* v_a_2149_, lean_object* v_a_2150_, lean_object* v_a_2151_){
_start:
{
lean_object* v_res_2152_; 
v_res_2152_ = lp_mathlib_Mathlib_Tactic_Hint_hint(v_stx_2142_, v_a_2143_, v_a_2144_, v_a_2145_, v_a_2146_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_);
lean_dec(v_a_2150_);
lean_dec_ref(v_a_2149_);
lean_dec(v_a_2148_);
lean_dec_ref(v_a_2147_);
lean_dec(v_a_2146_);
lean_dec_ref(v_a_2145_);
lean_dec(v_a_2144_);
lean_dec_ref(v_a_2143_);
return v_res_2152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2(lean_object* v_00_u03b1_2153_, lean_object* v_L_2154_){
_start:
{
lean_object* v___x_2155_; 
v___x_2155_ = lp_mathlib_Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2___redArg(v_L_2154_);
return v___x_2155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3(lean_object* v_00_u03b1_2156_, lean_object* v_00_u03b2_2157_, lean_object* v_f_2158_, lean_object* v_L_2159_){
_start:
{
lean_object* v___x_2160_; 
v___x_2160_ = lp_mathlib_Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3___redArg(v_f_2158_, v_L_2159_);
return v___x_2160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4(lean_object* v_00_u03b1_2161_, lean_object* v_L_2162_, lean_object* v_f_2163_){
_start:
{
lean_object* v___x_2164_; 
v___x_2164_ = lp_mathlib_MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4___redArg(v_L_2162_, v_f_2163_);
return v___x_2164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5(lean_object* v_00_u03b1_2165_, lean_object* v_L_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_){
_start:
{
lean_object* v___x_2176_; 
v___x_2176_ = lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___redArg(v_L_2166_, v___y_2167_, v___y_2168_, v___y_2169_, v___y_2170_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_);
return v___x_2176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5___boxed(lean_object* v_00_u03b1_2177_, lean_object* v_L_2178_, lean_object* v___y_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_){
_start:
{
lean_object* v_res_2188_; 
v_res_2188_ = lp_mathlib_MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5(v_00_u03b1_2177_, v_L_2178_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_, v___y_2184_, v___y_2185_, v___y_2186_);
lean_dec(v___y_2186_);
lean_dec_ref(v___y_2185_);
lean_dec(v___y_2184_);
lean_dec_ref(v___y_2183_);
lean_dec(v___y_2182_);
lean_dec_ref(v___y_2181_);
lean_dec(v___y_2180_);
lean_dec_ref(v___y_2179_);
return v_res_2188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8(lean_object* v_n_2189_, lean_object* v_as_2190_, lean_object* v_lo_2191_, lean_object* v_hi_2192_, lean_object* v_w_2193_, lean_object* v_hlo_2194_, lean_object* v_hhi_2195_){
_start:
{
lean_object* v___x_2196_; 
v___x_2196_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___redArg(v_n_2189_, v_as_2190_, v_lo_2191_, v_hi_2192_);
return v___x_2196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8___boxed(lean_object* v_n_2197_, lean_object* v_as_2198_, lean_object* v_lo_2199_, lean_object* v_hi_2200_, lean_object* v_w_2201_, lean_object* v_hlo_2202_, lean_object* v_hhi_2203_){
_start:
{
lean_object* v_res_2204_; 
v_res_2204_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8(v_n_2197_, v_as_2198_, v_lo_2199_, v_hi_2200_, v_w_2201_, v_hlo_2202_, v_hhi_2203_);
lean_dec(v_hi_2200_);
lean_dec(v_n_2197_);
return v_res_2204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9(lean_object* v_n_2205_, lean_object* v_as_2206_, lean_object* v_lo_2207_, lean_object* v_hi_2208_, lean_object* v_w_2209_, lean_object* v_hlo_2210_, lean_object* v_hhi_2211_){
_start:
{
lean_object* v___x_2212_; 
v___x_2212_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___redArg(v_n_2205_, v_as_2206_, v_lo_2207_, v_hi_2208_);
return v___x_2212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9___boxed(lean_object* v_n_2213_, lean_object* v_as_2214_, lean_object* v_lo_2215_, lean_object* v_hi_2216_, lean_object* v_w_2217_, lean_object* v_hlo_2218_, lean_object* v_hhi_2219_){
_start:
{
lean_object* v_res_2220_; 
v_res_2220_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9(v_n_2213_, v_as_2214_, v_lo_2215_, v_hi_2216_, v_w_2217_, v_hlo_2218_, v_hhi_2219_);
lean_dec(v_hi_2216_);
lean_dec(v_n_2213_);
return v_res_2220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2(lean_object* v_00_u03b1_2221_, lean_object* v_a_2222_, lean_object* v_a_2223_){
_start:
{
lean_object* v___x_2224_; 
v___x_2224_ = lp_mathlib_List_mapTR_loop___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__2___redArg(v_a_2222_, v_a_2223_);
return v___x_2224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6(lean_object* v_00_u03b1_2225_, lean_object* v_L_2226_){
_start:
{
lean_object* v___x_2227_; 
v___x_2227_ = lp_mathlib_Nondet_squash___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__6___redArg(v_L_2226_);
return v___x_2227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3(lean_object* v_00_u03b1_2228_, lean_object* v_L_2229_){
_start:
{
lean_object* v___x_2230_; 
v___x_2230_ = lp_mathlib_Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3___redArg(v_L_2229_);
return v___x_2230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5(lean_object* v_00_u03b1_2231_, lean_object* v_x_2232_){
_start:
{
lean_object* v___x_2233_; 
v___x_2233_ = lp_mathlib_Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5___redArg(v_x_2232_);
return v___x_2233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6(lean_object* v_00_u03b1_2234_, lean_object* v_00_u03b2_2235_, lean_object* v_L_2236_, lean_object* v_f_2237_){
_start:
{
lean_object* v___x_2238_; 
v___x_2238_ = lp_mathlib_Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6___redArg(v_L_2236_, v_f_2237_);
return v___x_2238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8(lean_object* v_00_u03b1_2239_, lean_object* v_L_2240_, lean_object* v_f_2241_){
_start:
{
lean_object* v___x_2242_; 
v___x_2242_ = lp_mathlib_MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8___redArg(v_L_2240_, v_f_2241_);
return v___x_2242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10(lean_object* v_00_u03b1_2243_, lean_object* v_as_2244_, lean_object* v_init_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_, lean_object* v___y_2249_, lean_object* v___y_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_){
_start:
{
lean_object* v___x_2255_; 
v___x_2255_ = lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___redArg(v_as_2244_, v_init_2245_, v___y_2246_, v___y_2247_, v___y_2248_, v___y_2249_, v___y_2250_, v___y_2251_, v___y_2252_, v___y_2253_);
return v___x_2255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10___boxed(lean_object* v_00_u03b1_2256_, lean_object* v_as_2257_, lean_object* v_init_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_, lean_object* v___y_2267_){
_start:
{
lean_object* v_res_2268_; 
v_res_2268_ = lp_mathlib_MLList_forIn___at___00MLList_asArray___at___00Mathlib_Tactic_Hint_hint_spec__5_spec__10(v_00_u03b1_2256_, v_as_2257_, v_init_2258_, v___y_2259_, v___y_2260_, v___y_2261_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_, v___y_2266_);
lean_dec(v___y_2266_);
lean_dec_ref(v___y_2265_);
lean_dec(v___y_2264_);
lean_dec_ref(v___y_2263_);
lean_dec(v___y_2262_);
lean_dec_ref(v___y_2261_);
lean_dec(v___y_2260_);
lean_dec_ref(v___y_2259_);
return v_res_2268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15(lean_object* v_n_2269_, lean_object* v_lo_2270_, lean_object* v_hi_2271_, lean_object* v_hhi_2272_, lean_object* v_pivot_2273_, lean_object* v_as_2274_, lean_object* v_i_2275_, lean_object* v_k_2276_, lean_object* v_ilo_2277_, lean_object* v_ik_2278_, lean_object* v_w_2279_){
_start:
{
lean_object* v___x_2280_; 
v___x_2280_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___redArg(v_hi_2271_, v_pivot_2273_, v_as_2274_, v_i_2275_, v_k_2276_);
return v___x_2280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15___boxed(lean_object* v_n_2281_, lean_object* v_lo_2282_, lean_object* v_hi_2283_, lean_object* v_hhi_2284_, lean_object* v_pivot_2285_, lean_object* v_as_2286_, lean_object* v_i_2287_, lean_object* v_k_2288_, lean_object* v_ilo_2289_, lean_object* v_ik_2290_, lean_object* v_w_2291_){
_start:
{
lean_object* v_res_2292_; 
v_res_2292_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__8_spec__15(v_n_2281_, v_lo_2282_, v_hi_2283_, v_hhi_2284_, v_pivot_2285_, v_as_2286_, v_i_2287_, v_k_2288_, v_ilo_2289_, v_ik_2290_, v_w_2291_);
lean_dec_ref(v_pivot_2285_);
lean_dec(v_hi_2283_);
lean_dec(v_lo_2282_);
lean_dec(v_n_2281_);
return v_res_2292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17(lean_object* v_n_2293_, lean_object* v_lo_2294_, lean_object* v_hi_2295_, lean_object* v_hhi_2296_, lean_object* v_pivot_2297_, lean_object* v_as_2298_, lean_object* v_i_2299_, lean_object* v_k_2300_, lean_object* v_ilo_2301_, lean_object* v_ik_2302_, lean_object* v_w_2303_){
_start:
{
lean_object* v___x_2304_; 
v___x_2304_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___redArg(v_hi_2295_, v_pivot_2297_, v_as_2298_, v_i_2299_, v_k_2300_);
return v___x_2304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17___boxed(lean_object* v_n_2305_, lean_object* v_lo_2306_, lean_object* v_hi_2307_, lean_object* v_hhi_2308_, lean_object* v_pivot_2309_, lean_object* v_as_2310_, lean_object* v_i_2311_, lean_object* v_k_2312_, lean_object* v_ilo_2313_, lean_object* v_ik_2314_, lean_object* v_w_2315_){
_start:
{
lean_object* v_res_2316_; 
v_res_2316_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_Hint_hint_spec__9_spec__17(v_n_2305_, v_lo_2306_, v_hi_2307_, v_hhi_2308_, v_pivot_2309_, v_as_2310_, v_i_2311_, v_k_2312_, v_ilo_2313_, v_ik_2314_, v_w_2315_);
lean_dec_ref(v_pivot_2309_);
lean_dec(v_hi_2307_);
lean_dec(v_lo_2306_);
lean_dec(v_n_2305_);
return v_res_2316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4(lean_object* v_00_u03b1_2317_, lean_object* v_a_2318_, lean_object* v_a_2319_, lean_object* v_a_2320_){
_start:
{
lean_object* v___x_2321_; 
v___x_2321_ = lp_mathlib_List_mapTR_loop___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__4___redArg(v_a_2318_, v_a_2319_, v_a_2320_);
return v___x_2321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5(lean_object* v_00_u03b1_2322_, lean_object* v_x_2323_){
_start:
{
lean_object* v___x_2324_; 
v___x_2324_ = lp_mathlib_MLList_ofListM___at___00Nondet_ofListM___at___00Nondet_ofList___at___00Mathlib_Tactic_Hint_hint_spec__2_spec__3_spec__5___redArg(v_x_2323_);
return v___x_2324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10(lean_object* v_00_u03b1_2325_, lean_object* v_x_2326_){
_start:
{
lean_object* v___x_2327_; 
v___x_2327_ = lp_mathlib_Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10___redArg(v_x_2326_);
return v___x_2327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12(lean_object* v_00_u03b1_2328_, lean_object* v_x_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_){
_start:
{
lean_object* v___x_2339_; 
v___x_2339_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___redArg(v_x_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, v___y_2335_, v___y_2336_, v___y_2337_);
return v___x_2339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12___boxed(lean_object* v_00_u03b1_2340_, lean_object* v_x_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_, lean_object* v___y_2348_, lean_object* v___y_2349_, lean_object* v___y_2350_){
_start:
{
lean_object* v_res_2351_; 
v_res_2351_ = lp_mathlib___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__12(v_00_u03b1_2340_, v_x_2341_, v___y_2342_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_, v___y_2347_, v___y_2348_, v___y_2349_);
lean_dec(v___y_2349_);
lean_dec_ref(v___y_2348_);
lean_dec(v___y_2347_);
lean_dec_ref(v___y_2346_);
lean_dec(v___y_2345_);
lean_dec_ref(v___y_2344_);
lean_dec(v___y_2343_);
lean_dec_ref(v___y_2342_);
return v_res_2351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13(lean_object* v_00_u03b1_2352_, lean_object* v_xs_2353_, lean_object* v_ys_2354_){
_start:
{
lean_object* v___x_2355_; 
v___x_2355_ = lp_mathlib_MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13___redArg(v_xs_2353_, v_ys_2354_);
return v___x_2355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16(lean_object* v_00_u03b1_2356_, lean_object* v_f_2357_, lean_object* v_00_u03b2_2358_, lean_object* v_xs_2359_){
_start:
{
lean_object* v___x_2360_; 
v___x_2360_ = lp_mathlib_MLList_casesM___at___00MLList_takeUpToFirstM___at___00MLList_takeUpToFirst___at___00Mathlib_Tactic_Hint_hint_spec__4_spec__8_spec__16___redArg(v_f_2357_, v_xs_2359_);
return v___x_2360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24(lean_object* v_00_u03b1_2361_, lean_object* v_x_2362_){
_start:
{
lean_object* v___x_2363_; 
v___x_2363_ = lp_mathlib_MLList_singletonM___at___00Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17_spec__24___redArg(v_x_2362_);
return v___x_2363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17(lean_object* v_00_u03b1_2364_, lean_object* v_x_2365_){
_start:
{
lean_object* v___x_2366_; 
v___x_2366_ = lp_mathlib_Nondet_singletonM___at___00Nondet_singleton___at___00Nondet_ofOptionM___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__5_spec__10_spec__17___redArg(v_x_2365_);
return v___x_2366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21(lean_object* v_00_u03b1_2367_, lean_object* v_ys_2368_, lean_object* v_00_u03b2_2369_, lean_object* v_xs_2370_){
_start:
{
lean_object* v___x_2371_; 
v___x_2371_ = lp_mathlib_MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21___redArg(v_ys_2368_, v_xs_2370_);
return v___x_2371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28(lean_object* v_00_u03b1_2372_, lean_object* v_ys_2373_, lean_object* v_00_u03b2_2374_, lean_object* v_xs_2375_){
_start:
{
lean_object* v___x_2376_; 
v___x_2376_ = lp_mathlib_MLList_casesM___at___00MLList_cases___at___00MLList_append___at___00Nondet_bind___at___00Nondet_filterMapM___at___00Mathlib_Tactic_Hint_hint_spec__3_spec__6_spec__13_spec__21_spec__28___redArg(v_ys_2373_, v_xs_2375_);
return v___x_2376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2393_; lean_object* v___x_2394_; 
v___x_2393_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__registerHintStx__1_spec__0___redArg___closed__0);
v___x_2394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2394_, 0, v___x_2393_);
return v___x_2394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___redArg___boxed(lean_object* v___y_2395_){
_start:
{
lean_object* v_res_2396_; 
v_res_2396_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___redArg();
return v_res_2396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0(lean_object* v_00_u03b1_2397_, lean_object* v___y_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_, lean_object* v___y_2405_){
_start:
{
lean_object* v___x_2407_; 
v___x_2407_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___redArg();
return v___x_2407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___boxed(lean_object* v_00_u03b1_2408_, lean_object* v___y_2409_, lean_object* v___y_2410_, lean_object* v___y_2411_, lean_object* v___y_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_, lean_object* v___y_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_){
_start:
{
lean_object* v_res_2418_; 
v_res_2418_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0(v_00_u03b1_2408_, v___y_2409_, v___y_2410_, v___y_2411_, v___y_2412_, v___y_2413_, v___y_2414_, v___y_2415_, v___y_2416_);
lean_dec(v___y_2416_);
lean_dec_ref(v___y_2415_);
lean_dec(v___y_2414_);
lean_dec_ref(v___y_2413_);
lean_dec(v___y_2412_);
lean_dec_ref(v___y_2411_);
lean_dec(v___y_2410_);
lean_dec_ref(v___y_2409_);
return v_res_2418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1(lean_object* v_x_2419_, lean_object* v_a_2420_, lean_object* v_a_2421_, lean_object* v_a_2422_, lean_object* v_a_2423_, lean_object* v_a_2424_, lean_object* v_a_2425_, lean_object* v_a_2426_, lean_object* v_a_2427_){
_start:
{
lean_object* v___x_2429_; uint8_t v___x_2430_; 
v___x_2429_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Hint_hintStx___closed__1));
lean_inc(v_x_2419_);
v___x_2430_ = l_Lean_Syntax_isOfKind(v_x_2419_, v___x_2429_);
if (v___x_2430_ == 0)
{
lean_object* v___x_2431_; 
lean_dec(v_x_2419_);
v___x_2431_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1_spec__0___redArg();
return v___x_2431_;
}
else
{
lean_object* v___x_2432_; lean_object* v_tk_2433_; lean_object* v___x_2434_; 
v___x_2432_ = lean_unsigned_to_nat(0u);
v_tk_2433_ = l_Lean_Syntax_getArg(v_x_2419_, v___x_2432_);
lean_dec(v_x_2419_);
v___x_2434_ = lp_mathlib_Mathlib_Tactic_Hint_hint(v_tk_2433_, v_a_2420_, v_a_2421_, v_a_2422_, v_a_2423_, v_a_2424_, v_a_2425_, v_a_2426_, v_a_2427_);
return v___x_2434_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1___boxed(lean_object* v_x_2435_, lean_object* v_a_2436_, lean_object* v_a_2437_, lean_object* v_a_2438_, lean_object* v_a_2439_, lean_object* v_a_2440_, lean_object* v_a_2441_, lean_object* v_a_2442_, lean_object* v_a_2443_, lean_object* v_a_2444_){
_start:
{
lean_object* v_res_2445_; 
v_res_2445_ = lp_mathlib_Mathlib_Tactic_Hint___aux__Mathlib__Tactic__Hint______elabRules__Mathlib__Tactic__Hint__hintStx__1(v_x_2435_, v_a_2436_, v_a_2437_, v_a_2438_, v_a_2439_, v_a_2440_, v_a_2441_, v_a_2442_, v_a_2443_);
lean_dec(v_a_2443_);
lean_dec_ref(v_a_2442_);
lean_dec(v_a_2441_);
lean_dec_ref(v_a_2440_);
lean_dec(v_a_2439_);
lean_dec_ref(v_a_2438_);
lean_dec(v_a_2437_);
lean_dec_ref(v_a_2436_);
return v_res_2445_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Control_Nondet_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_Nondet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_2826036628____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Hint_hintExtension = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Hint_hintExtension);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Hint_0__Mathlib_Tactic_Hint_initFn_00___x40_Mathlib_Tactic_Hint_3357253919____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_Nondet_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Linter_UnreachableTactic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_Nondet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Linter_UnreachableTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Hint(builtin);
}
#ifdef __cplusplus
}
#endif
