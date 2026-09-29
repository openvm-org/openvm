// Lean compiler output
// Module: Mathlib.Tactic.ExtractGoal
// Imports: public import Init public meta import Init public meta import Lean.Elab.Term public meta import Lean.Elab.Tactic.ElabTerm public meta import Lean.Meta.Tactic.Cleanup public meta import Lean.PrettyPrinter public meta import Batteries.Lean.Meta.Inaccessible public import Lean.Elab.Command public import Mathlib.Tactic.MinImports
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_maxView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_minView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_Name_lt(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint64_t l_Lean_Level_hash(lean_object*);
uint8_t lean_level_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Term_saveState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_SavedState_restore(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_levelMVarToParam___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_getLevelNames___redArg(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_collectLevelParams(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Level_param___override(lean_object*);
lean_object* l_Lean_addAndCompile(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_signature(lean_object*);
extern lean_object* l_Lean_NameSet_empty;
extern lean_object* l_Lean_Options_empty;
extern lean_object* l_Lean_firstFrontendMacroScope;
lean_object* l_Lean_DeclNameGenerator_ofPrefix(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getVisited(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_nat_div(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Environment_getModuleIdx_x3f(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_MVarId_renameInaccessibleFVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_getFVarIds(lean_object*);
lean_object* l_Lean_MVarId_revert(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingName_x21(lean_object*);
uint8_t l_Lean_Name_isInternal(lean_object*);
uint8_t l_Lean_Expr_isForall(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_unlockAsync(lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_consumeMData(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Tactic_Cleanup_0__Lean_Meta_cleanupCore(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Tactic_getFVarIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_DeclNameGenerator_mkUniqueName(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "star"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ExtractGoal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__3_value),LEAN_SCALAR_PTR_LITERAL(174, 47, 2, 238, 136, 228, 187, 129)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__0_value),LEAN_SCALAR_PTR_LITERAL(222, 65, 232, 126, 125, 44, 101, 32)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_star = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__3_value),LEAN_SCALAR_PTR_LITERAL(174, 47, 2, 238, 136, 228, 187, 129)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 25, 129, 177, 184, 70, 65, 189)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__2_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__4_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__6_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__8_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__15_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__21_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_config = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "extractGoal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__3_value),LEAN_SCALAR_PTR_LITERAL(174, 47, 2, 238, 136, 228, 187, 129)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__0_value),LEAN_SCALAR_PTR_LITERAL(54, 216, 205, 156, 177, 41, 141, 21)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "extract_goal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__5_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__22(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__22___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__0 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__1 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__2 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14_spec__21___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "_uniq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(237, 141, 162, 170, 202, 74, 55, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__10_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__15;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Extracted goal has metavariables: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14_spec__21(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " := sorry"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "def"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "theorem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(227, 122, 176, 177, 50, 175, 152, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extracted"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(98, 82, 122, 210, 120, 192, 16, 199)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg(lean_object* v_e_104_, lean_object* v___y_105_){
_start:
{
uint8_t v___x_107_; 
v___x_107_ = l_Lean_Expr_hasMVar(v_e_104_);
if (v___x_107_ == 0)
{
lean_object* v___x_108_; 
v___x_108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_108_, 0, v_e_104_);
return v___x_108_;
}
else
{
lean_object* v___x_109_; lean_object* v_mctx_110_; lean_object* v___x_111_; lean_object* v_fst_112_; lean_object* v_snd_113_; lean_object* v___x_114_; lean_object* v_cache_115_; lean_object* v_zetaDeltaFVarIds_116_; lean_object* v_postponed_117_; lean_object* v_diag_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_127_; 
v___x_109_ = lean_st_ref_get(v___y_105_);
v_mctx_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc_ref(v_mctx_110_);
lean_dec(v___x_109_);
v___x_111_ = l_Lean_instantiateMVarsCore(v_mctx_110_, v_e_104_);
v_fst_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc(v_fst_112_);
v_snd_113_ = lean_ctor_get(v___x_111_, 1);
lean_inc(v_snd_113_);
lean_dec_ref(v___x_111_);
v___x_114_ = lean_st_ref_take(v___y_105_);
v_cache_115_ = lean_ctor_get(v___x_114_, 1);
v_zetaDeltaFVarIds_116_ = lean_ctor_get(v___x_114_, 2);
v_postponed_117_ = lean_ctor_get(v___x_114_, 3);
v_diag_118_ = lean_ctor_get(v___x_114_, 4);
v_isSharedCheck_127_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_127_ == 0)
{
lean_object* v_unused_128_; 
v_unused_128_ = lean_ctor_get(v___x_114_, 0);
lean_dec(v_unused_128_);
v___x_120_ = v___x_114_;
v_isShared_121_ = v_isSharedCheck_127_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_diag_118_);
lean_inc(v_postponed_117_);
lean_inc(v_zetaDeltaFVarIds_116_);
lean_inc(v_cache_115_);
lean_dec(v___x_114_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_127_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
lean_ctor_set(v___x_120_, 0, v_snd_113_);
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v_snd_113_);
lean_ctor_set(v_reuseFailAlloc_126_, 1, v_cache_115_);
lean_ctor_set(v_reuseFailAlloc_126_, 2, v_zetaDeltaFVarIds_116_);
lean_ctor_set(v_reuseFailAlloc_126_, 3, v_postponed_117_);
lean_ctor_set(v_reuseFailAlloc_126_, 4, v_diag_118_);
v___x_123_ = v_reuseFailAlloc_126_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = lean_st_ref_set(v___y_105_, v___x_123_);
v___x_125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_125_, 0, v_fst_112_);
return v___x_125_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg___boxed(lean_object* v_e_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg(v_e_129_, v___y_130_);
lean_dec(v___y_130_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0(lean_object* v_e_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg(v_e_133_, v___y_137_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___boxed(lean_object* v_e_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0(v_e_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_, v___y_147_, v___y_148_);
lean_dec(v___y_148_);
lean_dec_ref(v___y_147_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3(lean_object* v_msgData_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
lean_object* v___x_157_; lean_object* v_env_158_; lean_object* v___x_159_; lean_object* v_mctx_160_; lean_object* v_lctx_161_; lean_object* v_options_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_157_ = lean_st_ref_get(v___y_155_);
v_env_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc_ref(v_env_158_);
lean_dec(v___x_157_);
v___x_159_ = lean_st_ref_get(v___y_153_);
v_mctx_160_ = lean_ctor_get(v___x_159_, 0);
lean_inc_ref(v_mctx_160_);
lean_dec(v___x_159_);
v_lctx_161_ = lean_ctor_get(v___y_152_, 2);
v_options_162_ = lean_ctor_get(v___y_154_, 2);
lean_inc_ref(v_options_162_);
lean_inc_ref(v_lctx_161_);
v___x_163_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_163_, 0, v_env_158_);
lean_ctor_set(v___x_163_, 1, v_mctx_160_);
lean_ctor_set(v___x_163_, 2, v_lctx_161_);
lean_ctor_set(v___x_163_, 3, v_options_162_);
v___x_164_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v_msgData_151_);
v___x_165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3___boxed(lean_object* v_msgData_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3(v_msgData_166_, v___y_167_, v___y_168_, v___y_169_, v___y_170_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__10(lean_object* v_msg_173_){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_174_ = lean_box(0);
v___x_175_ = lean_panic_fn_borrowed(v___x_174_, v_msg_173_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___lam__0(lean_object* v_a_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v_a_x3f_183_){
_start:
{
uint8_t v___x_185_; lean_object* v___x_186_; 
v___x_185_ = 0;
v___x_186_ = l_Lean_Elab_Term_SavedState_restore(v_a_176_, v___x_185_, v___y_177_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___lam__0___boxed(lean_object* v_a_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v_a_x3f_194_, lean_object* v___y_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___lam__0(v_a_187_, v___y_188_, v___y_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v_a_x3f_194_);
lean_dec(v_a_x3f_194_);
lean_dec(v___y_193_);
lean_dec_ref(v___y_192_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
lean_dec(v___y_189_);
lean_dec_ref(v___y_188_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg(lean_object* v_x_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = l_Lean_Elab_Term_saveState___redArg(v___y_199_, v___y_201_, v___y_203_);
if (lean_obj_tag(v___x_205_) == 0)
{
lean_object* v_a_206_; lean_object* v_r_207_; 
v_a_206_ = lean_ctor_get(v___x_205_, 0);
lean_inc(v_a_206_);
lean_dec_ref_known(v___x_205_, 1);
lean_inc(v___y_203_);
lean_inc_ref(v___y_202_);
lean_inc(v___y_201_);
lean_inc_ref(v___y_200_);
lean_inc(v___y_199_);
lean_inc_ref(v___y_198_);
v_r_207_ = lean_apply_7(v_x_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_, lean_box(0));
if (lean_obj_tag(v_r_207_) == 0)
{
lean_object* v_a_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_232_; 
v_a_208_ = lean_ctor_get(v_r_207_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v_r_207_);
if (v_isSharedCheck_232_ == 0)
{
v___x_210_ = v_r_207_;
v_isShared_211_ = v_isSharedCheck_232_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_a_208_);
lean_dec(v_r_207_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_232_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_213_; 
lean_inc(v_a_208_);
if (v_isShared_211_ == 0)
{
lean_ctor_set_tag(v___x_210_, 1);
v___x_213_ = v___x_210_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v_a_208_);
v___x_213_ = v_reuseFailAlloc_231_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___lam__0(v_a_206_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_, v___x_213_);
lean_dec_ref(v___x_213_);
if (lean_obj_tag(v___x_214_) == 0)
{
lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_221_; 
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_221_ == 0)
{
lean_object* v_unused_222_; 
v_unused_222_ = lean_ctor_get(v___x_214_, 0);
lean_dec(v_unused_222_);
v___x_216_ = v___x_214_;
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
else
{
lean_dec(v___x_214_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_221_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_219_; 
if (v_isShared_217_ == 0)
{
lean_ctor_set(v___x_216_, 0, v_a_208_);
v___x_219_ = v___x_216_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_a_208_);
v___x_219_ = v_reuseFailAlloc_220_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
return v___x_219_;
}
}
}
else
{
lean_object* v_a_223_; lean_object* v___x_225_; uint8_t v_isShared_226_; uint8_t v_isSharedCheck_230_; 
lean_dec(v_a_208_);
v_a_223_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_230_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_230_ == 0)
{
v___x_225_ = v___x_214_;
v_isShared_226_ = v_isSharedCheck_230_;
goto v_resetjp_224_;
}
else
{
lean_inc(v_a_223_);
lean_dec(v___x_214_);
v___x_225_ = lean_box(0);
v_isShared_226_ = v_isSharedCheck_230_;
goto v_resetjp_224_;
}
v_resetjp_224_:
{
lean_object* v___x_228_; 
if (v_isShared_226_ == 0)
{
v___x_228_ = v___x_225_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v_a_223_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
return v___x_228_;
}
}
}
}
}
}
else
{
lean_object* v_a_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v_a_233_ = lean_ctor_get(v_r_207_, 0);
lean_inc(v_a_233_);
lean_dec_ref_known(v_r_207_, 1);
v___x_234_ = lean_box(0);
v___x_235_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___lam__0(v_a_206_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_, v___x_234_);
if (lean_obj_tag(v___x_235_) == 0)
{
lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_242_; 
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_235_);
if (v_isSharedCheck_242_ == 0)
{
lean_object* v_unused_243_; 
v_unused_243_ = lean_ctor_get(v___x_235_, 0);
lean_dec(v_unused_243_);
v___x_237_ = v___x_235_;
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
else
{
lean_dec(v___x_235_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_240_; 
if (v_isShared_238_ == 0)
{
lean_ctor_set_tag(v___x_237_, 1);
lean_ctor_set(v___x_237_, 0, v_a_233_);
v___x_240_ = v___x_237_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v_a_233_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
}
else
{
lean_object* v_a_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_251_; 
lean_dec(v_a_233_);
v_a_244_ = lean_ctor_get(v___x_235_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_235_);
if (v_isSharedCheck_251_ == 0)
{
v___x_246_ = v___x_235_;
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_a_244_);
lean_dec(v___x_235_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___x_249_; 
if (v_isShared_247_ == 0)
{
v___x_249_ = v___x_246_;
goto v_reusejp_248_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_a_244_);
v___x_249_ = v_reuseFailAlloc_250_;
goto v_reusejp_248_;
}
v_reusejp_248_:
{
return v___x_249_;
}
}
}
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
lean_dec_ref(v_x_197_);
v_a_252_ = lean_ctor_get(v___x_205_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_205_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_205_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_205_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg___boxed(lean_object* v_x_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg(v_x_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13(lean_object* v_00_u03b1_269_, lean_object* v_x_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___redArg(v_x_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_, v___y_275_, v___y_276_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___boxed(lean_object* v_00_u03b1_279_, lean_object* v_x_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13(v_00_u03b1_279_, v_x_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec(v___y_282_);
lean_dec_ref(v___y_281_);
return v_res_288_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__0(uint8_t v___x_289_, lean_object* v_x_290_){
_start:
{
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__0___boxed(lean_object* v___x_291_, lean_object* v_x_292_){
_start:
{
uint8_t v___x_24041__boxed_293_; uint8_t v_res_294_; lean_object* v_r_295_; 
v___x_24041__boxed_293_ = lean_unbox(v___x_291_);
v_res_294_ = lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__0(v___x_24041__boxed_293_, v_x_292_);
lean_dec(v_x_292_);
v_r_295_ = lean_box(v_res_294_);
return v_r_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___redArg(lean_object* v_hi_296_, lean_object* v_pivot_297_, lean_object* v_as_298_, lean_object* v_i_299_, lean_object* v_k_300_){
_start:
{
uint8_t v___x_301_; 
v___x_301_ = lean_nat_dec_lt(v_k_300_, v_hi_296_);
if (v___x_301_ == 0)
{
lean_object* v___x_302_; lean_object* v___x_303_; 
lean_dec(v_k_300_);
v___x_302_ = lean_array_fswap(v_as_298_, v_i_299_, v_hi_296_);
v___x_303_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_303_, 0, v_i_299_);
lean_ctor_set(v___x_303_, 1, v___x_302_);
return v___x_303_;
}
else
{
lean_object* v___x_304_; uint8_t v___x_305_; 
v___x_304_ = lean_array_fget_borrowed(v_as_298_, v_k_300_);
v___x_305_ = l_Lean_Name_lt(v___x_304_, v_pivot_297_);
if (v___x_305_ == 0)
{
lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_306_ = lean_unsigned_to_nat(1u);
v___x_307_ = lean_nat_add(v_k_300_, v___x_306_);
lean_dec(v_k_300_);
v_k_300_ = v___x_307_;
goto _start;
}
else
{
lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_309_ = lean_array_fswap(v_as_298_, v_i_299_, v_k_300_);
v___x_310_ = lean_unsigned_to_nat(1u);
v___x_311_ = lean_nat_add(v_i_299_, v___x_310_);
lean_dec(v_i_299_);
v___x_312_ = lean_nat_add(v_k_300_, v___x_310_);
lean_dec(v_k_300_);
v_as_298_ = v___x_309_;
v_i_299_ = v___x_311_;
v_k_300_ = v___x_312_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___redArg___boxed(lean_object* v_hi_314_, lean_object* v_pivot_315_, lean_object* v_as_316_, lean_object* v_i_317_, lean_object* v_k_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___redArg(v_hi_314_, v_pivot_315_, v_as_316_, v_i_317_, v_k_318_);
lean_dec(v_pivot_315_);
lean_dec(v_hi_314_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg(lean_object* v_n_320_, lean_object* v_as_321_, lean_object* v_lo_322_, lean_object* v_hi_323_){
_start:
{
lean_object* v___y_325_; uint8_t v___x_335_; 
v___x_335_ = lean_nat_dec_lt(v_lo_322_, v_hi_323_);
if (v___x_335_ == 0)
{
lean_dec(v_lo_322_);
return v_as_321_;
}
else
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v_mid_338_; lean_object* v___y_340_; lean_object* v___y_346_; lean_object* v___x_351_; lean_object* v___x_352_; uint8_t v___x_353_; 
v___x_336_ = lean_nat_add(v_lo_322_, v_hi_323_);
v___x_337_ = lean_unsigned_to_nat(1u);
v_mid_338_ = lean_nat_shiftr(v___x_336_, v___x_337_);
lean_dec(v___x_336_);
v___x_351_ = lean_array_fget_borrowed(v_as_321_, v_mid_338_);
v___x_352_ = lean_array_fget_borrowed(v_as_321_, v_lo_322_);
v___x_353_ = l_Lean_Name_lt(v___x_351_, v___x_352_);
if (v___x_353_ == 0)
{
v___y_346_ = v_as_321_;
goto v___jp_345_;
}
else
{
lean_object* v___x_354_; 
v___x_354_ = lean_array_fswap(v_as_321_, v_lo_322_, v_mid_338_);
v___y_346_ = v___x_354_;
goto v___jp_345_;
}
v___jp_339_:
{
lean_object* v___x_341_; lean_object* v___x_342_; uint8_t v___x_343_; 
v___x_341_ = lean_array_fget_borrowed(v___y_340_, v_mid_338_);
v___x_342_ = lean_array_fget_borrowed(v___y_340_, v_hi_323_);
v___x_343_ = l_Lean_Name_lt(v___x_341_, v___x_342_);
if (v___x_343_ == 0)
{
lean_dec(v_mid_338_);
v___y_325_ = v___y_340_;
goto v___jp_324_;
}
else
{
lean_object* v___x_344_; 
v___x_344_ = lean_array_fswap(v___y_340_, v_mid_338_, v_hi_323_);
lean_dec(v_mid_338_);
v___y_325_ = v___x_344_;
goto v___jp_324_;
}
}
v___jp_345_:
{
lean_object* v___x_347_; lean_object* v___x_348_; uint8_t v___x_349_; 
v___x_347_ = lean_array_fget_borrowed(v___y_346_, v_hi_323_);
v___x_348_ = lean_array_fget_borrowed(v___y_346_, v_lo_322_);
v___x_349_ = l_Lean_Name_lt(v___x_347_, v___x_348_);
if (v___x_349_ == 0)
{
v___y_340_ = v___y_346_;
goto v___jp_339_;
}
else
{
lean_object* v___x_350_; 
v___x_350_ = lean_array_fswap(v___y_346_, v_lo_322_, v_hi_323_);
v___y_340_ = v___x_350_;
goto v___jp_339_;
}
}
}
v___jp_324_:
{
lean_object* v_pivot_326_; lean_object* v___x_327_; lean_object* v_fst_328_; lean_object* v_snd_329_; uint8_t v___x_330_; 
v_pivot_326_ = lean_array_fget(v___y_325_, v_hi_323_);
lean_inc_n(v_lo_322_, 2);
v___x_327_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___redArg(v_hi_323_, v_pivot_326_, v___y_325_, v_lo_322_, v_lo_322_);
lean_dec(v_pivot_326_);
v_fst_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_fst_328_);
v_snd_329_ = lean_ctor_get(v___x_327_, 1);
lean_inc(v_snd_329_);
lean_dec_ref(v___x_327_);
v___x_330_ = lean_nat_dec_le(v_hi_323_, v_fst_328_);
if (v___x_330_ == 0)
{
lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_331_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg(v_n_320_, v_snd_329_, v_lo_322_, v_fst_328_);
v___x_332_ = lean_unsigned_to_nat(1u);
v___x_333_ = lean_nat_add(v_fst_328_, v___x_332_);
lean_dec(v_fst_328_);
v_as_321_ = v___x_331_;
v_lo_322_ = v___x_333_;
goto _start;
}
else
{
lean_dec(v_fst_328_);
lean_dec(v_lo_322_);
return v_snd_329_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg___boxed(lean_object* v_n_355_, lean_object* v_as_356_, lean_object* v_lo_357_, lean_object* v_hi_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg(v_n_355_, v_as_356_, v_lo_357_, v_hi_358_);
lean_dec(v_hi_358_);
lean_dec(v_n_355_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__5_spec__6(lean_object* v_init_360_, lean_object* v_x_361_){
_start:
{
if (lean_obj_tag(v_x_361_) == 0)
{
lean_object* v_k_362_; lean_object* v_l_363_; lean_object* v_r_364_; lean_object* v___x_365_; lean_object* v___x_366_; 
v_k_362_ = lean_ctor_get(v_x_361_, 1);
lean_inc(v_k_362_);
v_l_363_ = lean_ctor_get(v_x_361_, 3);
lean_inc(v_l_363_);
v_r_364_ = lean_ctor_get(v_x_361_, 4);
lean_inc(v_r_364_);
lean_dec_ref_known(v_x_361_, 5);
v___x_365_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__5_spec__6(v_init_360_, v_l_363_);
v___x_366_ = lean_array_push(v___x_365_, v_k_362_);
v_init_360_ = v___x_366_;
v_x_361_ = v_r_364_;
goto _start;
}
else
{
return v_init_360_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg(lean_object* v_k_368_, lean_object* v_t_369_){
_start:
{
if (lean_obj_tag(v_t_369_) == 0)
{
lean_object* v_k_370_; lean_object* v_v_371_; lean_object* v_l_372_; lean_object* v_r_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_1027_; 
v_k_370_ = lean_ctor_get(v_t_369_, 1);
v_v_371_ = lean_ctor_get(v_t_369_, 2);
v_l_372_ = lean_ctor_get(v_t_369_, 3);
v_r_373_ = lean_ctor_get(v_t_369_, 4);
v_isSharedCheck_1027_ = !lean_is_exclusive(v_t_369_);
if (v_isSharedCheck_1027_ == 0)
{
lean_object* v_unused_1028_; 
v_unused_1028_ = lean_ctor_get(v_t_369_, 0);
lean_dec(v_unused_1028_);
v___x_375_ = v_t_369_;
v_isShared_376_ = v_isSharedCheck_1027_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_r_373_);
lean_inc(v_l_372_);
lean_inc(v_v_371_);
lean_inc(v_k_370_);
lean_dec(v_t_369_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_1027_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
uint8_t v___x_377_; 
v___x_377_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_368_, v_k_370_);
switch(v___x_377_)
{
case 0:
{
lean_object* v_impl_378_; lean_object* v___x_379_; 
v_impl_378_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg(v_k_368_, v_l_372_);
v___x_379_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_378_) == 0)
{
if (lean_obj_tag(v_r_373_) == 0)
{
lean_object* v_size_380_; lean_object* v_size_381_; lean_object* v_k_382_; lean_object* v_v_383_; lean_object* v_l_384_; lean_object* v_r_385_; lean_object* v___x_386_; lean_object* v___x_387_; uint8_t v___x_388_; 
v_size_380_ = lean_ctor_get(v_impl_378_, 0);
lean_inc(v_size_380_);
v_size_381_ = lean_ctor_get(v_r_373_, 0);
v_k_382_ = lean_ctor_get(v_r_373_, 1);
v_v_383_ = lean_ctor_get(v_r_373_, 2);
v_l_384_ = lean_ctor_get(v_r_373_, 3);
lean_inc(v_l_384_);
v_r_385_ = lean_ctor_get(v_r_373_, 4);
v___x_386_ = lean_unsigned_to_nat(3u);
v___x_387_ = lean_nat_mul(v___x_386_, v_size_380_);
v___x_388_ = lean_nat_dec_lt(v___x_387_, v_size_381_);
lean_dec(v___x_387_);
if (v___x_388_ == 0)
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_392_; 
lean_dec(v_l_384_);
v___x_389_ = lean_nat_add(v___x_379_, v_size_380_);
lean_dec(v_size_380_);
v___x_390_ = lean_nat_add(v___x_389_, v_size_381_);
lean_dec(v___x_389_);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 3, v_impl_378_);
lean_ctor_set(v___x_375_, 0, v___x_390_);
v___x_392_ = v___x_375_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_390_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_393_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_393_, 3, v_impl_378_);
lean_ctor_set(v_reuseFailAlloc_393_, 4, v_r_373_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
else
{
lean_object* v___x_395_; uint8_t v_isShared_396_; uint8_t v_isSharedCheck_457_; 
lean_inc(v_r_385_);
lean_inc(v_v_383_);
lean_inc(v_k_382_);
lean_inc(v_size_381_);
v_isSharedCheck_457_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_457_ == 0)
{
lean_object* v_unused_458_; lean_object* v_unused_459_; lean_object* v_unused_460_; lean_object* v_unused_461_; lean_object* v_unused_462_; 
v_unused_458_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_458_);
v_unused_459_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_459_);
v_unused_460_ = lean_ctor_get(v_r_373_, 2);
lean_dec(v_unused_460_);
v_unused_461_ = lean_ctor_get(v_r_373_, 1);
lean_dec(v_unused_461_);
v_unused_462_ = lean_ctor_get(v_r_373_, 0);
lean_dec(v_unused_462_);
v___x_395_ = v_r_373_;
v_isShared_396_ = v_isSharedCheck_457_;
goto v_resetjp_394_;
}
else
{
lean_dec(v_r_373_);
v___x_395_ = lean_box(0);
v_isShared_396_ = v_isSharedCheck_457_;
goto v_resetjp_394_;
}
v_resetjp_394_:
{
lean_object* v_size_397_; lean_object* v_k_398_; lean_object* v_v_399_; lean_object* v_l_400_; lean_object* v_r_401_; lean_object* v_size_402_; lean_object* v___x_403_; lean_object* v___x_404_; uint8_t v___x_405_; 
v_size_397_ = lean_ctor_get(v_l_384_, 0);
v_k_398_ = lean_ctor_get(v_l_384_, 1);
v_v_399_ = lean_ctor_get(v_l_384_, 2);
v_l_400_ = lean_ctor_get(v_l_384_, 3);
v_r_401_ = lean_ctor_get(v_l_384_, 4);
v_size_402_ = lean_ctor_get(v_r_385_, 0);
v___x_403_ = lean_unsigned_to_nat(2u);
v___x_404_ = lean_nat_mul(v___x_403_, v_size_402_);
v___x_405_ = lean_nat_dec_lt(v_size_397_, v___x_404_);
lean_dec(v___x_404_);
if (v___x_405_ == 0)
{
lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_433_; 
lean_inc(v_r_401_);
lean_inc(v_l_400_);
lean_inc(v_v_399_);
lean_inc(v_k_398_);
v_isSharedCheck_433_ = !lean_is_exclusive(v_l_384_);
if (v_isSharedCheck_433_ == 0)
{
lean_object* v_unused_434_; lean_object* v_unused_435_; lean_object* v_unused_436_; lean_object* v_unused_437_; lean_object* v_unused_438_; 
v_unused_434_ = lean_ctor_get(v_l_384_, 4);
lean_dec(v_unused_434_);
v_unused_435_ = lean_ctor_get(v_l_384_, 3);
lean_dec(v_unused_435_);
v_unused_436_ = lean_ctor_get(v_l_384_, 2);
lean_dec(v_unused_436_);
v_unused_437_ = lean_ctor_get(v_l_384_, 1);
lean_dec(v_unused_437_);
v_unused_438_ = lean_ctor_get(v_l_384_, 0);
lean_dec(v_unused_438_);
v___x_407_ = v_l_384_;
v_isShared_408_ = v_isSharedCheck_433_;
goto v_resetjp_406_;
}
else
{
lean_dec(v_l_384_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_433_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___y_412_; lean_object* v___y_413_; lean_object* v___y_414_; lean_object* v___y_423_; 
v___x_409_ = lean_nat_add(v___x_379_, v_size_380_);
lean_dec(v_size_380_);
v___x_410_ = lean_nat_add(v___x_409_, v_size_381_);
lean_dec(v_size_381_);
if (lean_obj_tag(v_l_400_) == 0)
{
lean_object* v_size_431_; 
v_size_431_ = lean_ctor_get(v_l_400_, 0);
lean_inc(v_size_431_);
v___y_423_ = v_size_431_;
goto v___jp_422_;
}
else
{
lean_object* v___x_432_; 
v___x_432_ = lean_unsigned_to_nat(0u);
v___y_423_ = v___x_432_;
goto v___jp_422_;
}
v___jp_411_:
{
lean_object* v___x_415_; lean_object* v___x_417_; 
v___x_415_ = lean_nat_add(v___y_412_, v___y_414_);
lean_dec(v___y_414_);
lean_dec(v___y_412_);
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 4, v_r_385_);
lean_ctor_set(v___x_407_, 3, v_r_401_);
lean_ctor_set(v___x_407_, 2, v_v_383_);
lean_ctor_set(v___x_407_, 1, v_k_382_);
lean_ctor_set(v___x_407_, 0, v___x_415_);
v___x_417_ = v___x_407_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v___x_415_);
lean_ctor_set(v_reuseFailAlloc_421_, 1, v_k_382_);
lean_ctor_set(v_reuseFailAlloc_421_, 2, v_v_383_);
lean_ctor_set(v_reuseFailAlloc_421_, 3, v_r_401_);
lean_ctor_set(v_reuseFailAlloc_421_, 4, v_r_385_);
v___x_417_ = v_reuseFailAlloc_421_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
lean_object* v___x_419_; 
if (v_isShared_396_ == 0)
{
lean_ctor_set(v___x_395_, 4, v___x_417_);
lean_ctor_set(v___x_395_, 3, v___y_413_);
lean_ctor_set(v___x_395_, 2, v_v_399_);
lean_ctor_set(v___x_395_, 1, v_k_398_);
lean_ctor_set(v___x_395_, 0, v___x_410_);
v___x_419_ = v___x_395_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_420_; 
v_reuseFailAlloc_420_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_420_, 0, v___x_410_);
lean_ctor_set(v_reuseFailAlloc_420_, 1, v_k_398_);
lean_ctor_set(v_reuseFailAlloc_420_, 2, v_v_399_);
lean_ctor_set(v_reuseFailAlloc_420_, 3, v___y_413_);
lean_ctor_set(v_reuseFailAlloc_420_, 4, v___x_417_);
v___x_419_ = v_reuseFailAlloc_420_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
return v___x_419_;
}
}
}
v___jp_422_:
{
lean_object* v___x_424_; lean_object* v___x_426_; 
v___x_424_ = lean_nat_add(v___x_409_, v___y_423_);
lean_dec(v___y_423_);
lean_dec(v___x_409_);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_l_400_);
lean_ctor_set(v___x_375_, 3, v_impl_378_);
lean_ctor_set(v___x_375_, 0, v___x_424_);
v___x_426_ = v___x_375_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_424_);
lean_ctor_set(v_reuseFailAlloc_430_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_430_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_430_, 3, v_impl_378_);
lean_ctor_set(v_reuseFailAlloc_430_, 4, v_l_400_);
v___x_426_ = v_reuseFailAlloc_430_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
lean_object* v___x_427_; 
v___x_427_ = lean_nat_add(v___x_379_, v_size_402_);
if (lean_obj_tag(v_r_401_) == 0)
{
lean_object* v_size_428_; 
v_size_428_ = lean_ctor_get(v_r_401_, 0);
lean_inc(v_size_428_);
v___y_412_ = v___x_427_;
v___y_413_ = v___x_426_;
v___y_414_ = v_size_428_;
goto v___jp_411_;
}
else
{
lean_object* v___x_429_; 
v___x_429_ = lean_unsigned_to_nat(0u);
v___y_412_ = v___x_427_;
v___y_413_ = v___x_426_;
v___y_414_ = v___x_429_;
goto v___jp_411_;
}
}
}
}
}
else
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_443_; 
lean_del_object(v___x_375_);
v___x_439_ = lean_nat_add(v___x_379_, v_size_380_);
lean_dec(v_size_380_);
v___x_440_ = lean_nat_add(v___x_439_, v_size_381_);
lean_dec(v_size_381_);
v___x_441_ = lean_nat_add(v___x_439_, v_size_397_);
lean_dec(v___x_439_);
lean_inc_ref(v_impl_378_);
if (v_isShared_396_ == 0)
{
lean_ctor_set(v___x_395_, 4, v_l_384_);
lean_ctor_set(v___x_395_, 3, v_impl_378_);
lean_ctor_set(v___x_395_, 2, v_v_371_);
lean_ctor_set(v___x_395_, 1, v_k_370_);
lean_ctor_set(v___x_395_, 0, v___x_441_);
v___x_443_ = v___x_395_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_456_; 
v_reuseFailAlloc_456_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_456_, 0, v___x_441_);
lean_ctor_set(v_reuseFailAlloc_456_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_456_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_456_, 3, v_impl_378_);
lean_ctor_set(v_reuseFailAlloc_456_, 4, v_l_384_);
v___x_443_ = v_reuseFailAlloc_456_;
goto v_reusejp_442_;
}
v_reusejp_442_:
{
lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_450_; 
v_isSharedCheck_450_ = !lean_is_exclusive(v_impl_378_);
if (v_isSharedCheck_450_ == 0)
{
lean_object* v_unused_451_; lean_object* v_unused_452_; lean_object* v_unused_453_; lean_object* v_unused_454_; lean_object* v_unused_455_; 
v_unused_451_ = lean_ctor_get(v_impl_378_, 4);
lean_dec(v_unused_451_);
v_unused_452_ = lean_ctor_get(v_impl_378_, 3);
lean_dec(v_unused_452_);
v_unused_453_ = lean_ctor_get(v_impl_378_, 2);
lean_dec(v_unused_453_);
v_unused_454_ = lean_ctor_get(v_impl_378_, 1);
lean_dec(v_unused_454_);
v_unused_455_ = lean_ctor_get(v_impl_378_, 0);
lean_dec(v_unused_455_);
v___x_445_ = v_impl_378_;
v_isShared_446_ = v_isSharedCheck_450_;
goto v_resetjp_444_;
}
else
{
lean_dec(v_impl_378_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_450_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v___x_448_; 
if (v_isShared_446_ == 0)
{
lean_ctor_set(v___x_445_, 4, v_r_385_);
lean_ctor_set(v___x_445_, 3, v___x_443_);
lean_ctor_set(v___x_445_, 2, v_v_383_);
lean_ctor_set(v___x_445_, 1, v_k_382_);
lean_ctor_set(v___x_445_, 0, v___x_440_);
v___x_448_ = v___x_445_;
goto v_reusejp_447_;
}
else
{
lean_object* v_reuseFailAlloc_449_; 
v_reuseFailAlloc_449_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_449_, 0, v___x_440_);
lean_ctor_set(v_reuseFailAlloc_449_, 1, v_k_382_);
lean_ctor_set(v_reuseFailAlloc_449_, 2, v_v_383_);
lean_ctor_set(v_reuseFailAlloc_449_, 3, v___x_443_);
lean_ctor_set(v_reuseFailAlloc_449_, 4, v_r_385_);
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
}
}
else
{
lean_object* v_size_463_; lean_object* v___x_464_; lean_object* v___x_466_; 
v_size_463_ = lean_ctor_get(v_impl_378_, 0);
lean_inc(v_size_463_);
v___x_464_ = lean_nat_add(v___x_379_, v_size_463_);
lean_dec(v_size_463_);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 3, v_impl_378_);
lean_ctor_set(v___x_375_, 0, v___x_464_);
v___x_466_ = v___x_375_;
goto v_reusejp_465_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v___x_464_);
lean_ctor_set(v_reuseFailAlloc_467_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_467_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_467_, 3, v_impl_378_);
lean_ctor_set(v_reuseFailAlloc_467_, 4, v_r_373_);
v___x_466_ = v_reuseFailAlloc_467_;
goto v_reusejp_465_;
}
v_reusejp_465_:
{
return v___x_466_;
}
}
}
else
{
if (lean_obj_tag(v_r_373_) == 0)
{
lean_object* v_l_468_; 
v_l_468_ = lean_ctor_get(v_r_373_, 3);
lean_inc(v_l_468_);
if (lean_obj_tag(v_l_468_) == 0)
{
lean_object* v_r_469_; 
v_r_469_ = lean_ctor_get(v_r_373_, 4);
lean_inc(v_r_469_);
if (lean_obj_tag(v_r_469_) == 0)
{
lean_object* v_size_470_; lean_object* v_k_471_; lean_object* v_v_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_485_; 
v_size_470_ = lean_ctor_get(v_r_373_, 0);
v_k_471_ = lean_ctor_get(v_r_373_, 1);
v_v_472_ = lean_ctor_get(v_r_373_, 2);
v_isSharedCheck_485_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_485_ == 0)
{
lean_object* v_unused_486_; lean_object* v_unused_487_; 
v_unused_486_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_486_);
v_unused_487_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_487_);
v___x_474_ = v_r_373_;
v_isShared_475_ = v_isSharedCheck_485_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_v_472_);
lean_inc(v_k_471_);
lean_inc(v_size_470_);
lean_dec(v_r_373_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_485_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v_size_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_480_; 
v_size_476_ = lean_ctor_get(v_l_468_, 0);
v___x_477_ = lean_nat_add(v___x_379_, v_size_470_);
lean_dec(v_size_470_);
v___x_478_ = lean_nat_add(v___x_379_, v_size_476_);
if (v_isShared_475_ == 0)
{
lean_ctor_set(v___x_474_, 4, v_l_468_);
lean_ctor_set(v___x_474_, 3, v_impl_378_);
lean_ctor_set(v___x_474_, 2, v_v_371_);
lean_ctor_set(v___x_474_, 1, v_k_370_);
lean_ctor_set(v___x_474_, 0, v___x_478_);
v___x_480_ = v___x_474_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v___x_478_);
lean_ctor_set(v_reuseFailAlloc_484_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_484_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_484_, 3, v_impl_378_);
lean_ctor_set(v_reuseFailAlloc_484_, 4, v_l_468_);
v___x_480_ = v_reuseFailAlloc_484_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
lean_object* v___x_482_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_r_469_);
lean_ctor_set(v___x_375_, 3, v___x_480_);
lean_ctor_set(v___x_375_, 2, v_v_472_);
lean_ctor_set(v___x_375_, 1, v_k_471_);
lean_ctor_set(v___x_375_, 0, v___x_477_);
v___x_482_ = v___x_375_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v___x_477_);
lean_ctor_set(v_reuseFailAlloc_483_, 1, v_k_471_);
lean_ctor_set(v_reuseFailAlloc_483_, 2, v_v_472_);
lean_ctor_set(v_reuseFailAlloc_483_, 3, v___x_480_);
lean_ctor_set(v_reuseFailAlloc_483_, 4, v_r_469_);
v___x_482_ = v_reuseFailAlloc_483_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
return v___x_482_;
}
}
}
}
else
{
lean_object* v_k_488_; lean_object* v_v_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_512_; 
v_k_488_ = lean_ctor_get(v_r_373_, 1);
v_v_489_ = lean_ctor_get(v_r_373_, 2);
v_isSharedCheck_512_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_512_ == 0)
{
lean_object* v_unused_513_; lean_object* v_unused_514_; lean_object* v_unused_515_; 
v_unused_513_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_513_);
v_unused_514_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_514_);
v_unused_515_ = lean_ctor_get(v_r_373_, 0);
lean_dec(v_unused_515_);
v___x_491_ = v_r_373_;
v_isShared_492_ = v_isSharedCheck_512_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_v_489_);
lean_inc(v_k_488_);
lean_dec(v_r_373_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_512_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v_k_493_; lean_object* v_v_494_; lean_object* v___x_496_; uint8_t v_isShared_497_; uint8_t v_isSharedCheck_508_; 
v_k_493_ = lean_ctor_get(v_l_468_, 1);
v_v_494_ = lean_ctor_get(v_l_468_, 2);
v_isSharedCheck_508_ = !lean_is_exclusive(v_l_468_);
if (v_isSharedCheck_508_ == 0)
{
lean_object* v_unused_509_; lean_object* v_unused_510_; lean_object* v_unused_511_; 
v_unused_509_ = lean_ctor_get(v_l_468_, 4);
lean_dec(v_unused_509_);
v_unused_510_ = lean_ctor_get(v_l_468_, 3);
lean_dec(v_unused_510_);
v_unused_511_ = lean_ctor_get(v_l_468_, 0);
lean_dec(v_unused_511_);
v___x_496_ = v_l_468_;
v_isShared_497_ = v_isSharedCheck_508_;
goto v_resetjp_495_;
}
else
{
lean_inc(v_v_494_);
lean_inc(v_k_493_);
lean_dec(v_l_468_);
v___x_496_ = lean_box(0);
v_isShared_497_ = v_isSharedCheck_508_;
goto v_resetjp_495_;
}
v_resetjp_495_:
{
lean_object* v___x_498_; lean_object* v___x_500_; 
v___x_498_ = lean_unsigned_to_nat(3u);
if (v_isShared_497_ == 0)
{
lean_ctor_set(v___x_496_, 4, v_r_469_);
lean_ctor_set(v___x_496_, 3, v_r_469_);
lean_ctor_set(v___x_496_, 2, v_v_371_);
lean_ctor_set(v___x_496_, 1, v_k_370_);
lean_ctor_set(v___x_496_, 0, v___x_379_);
v___x_500_ = v___x_496_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_507_; 
v_reuseFailAlloc_507_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_507_, 0, v___x_379_);
lean_ctor_set(v_reuseFailAlloc_507_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_507_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_507_, 3, v_r_469_);
lean_ctor_set(v_reuseFailAlloc_507_, 4, v_r_469_);
v___x_500_ = v_reuseFailAlloc_507_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
lean_object* v___x_502_; 
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 3, v_r_469_);
lean_ctor_set(v___x_491_, 0, v___x_379_);
v___x_502_ = v___x_491_;
goto v_reusejp_501_;
}
else
{
lean_object* v_reuseFailAlloc_506_; 
v_reuseFailAlloc_506_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_506_, 0, v___x_379_);
lean_ctor_set(v_reuseFailAlloc_506_, 1, v_k_488_);
lean_ctor_set(v_reuseFailAlloc_506_, 2, v_v_489_);
lean_ctor_set(v_reuseFailAlloc_506_, 3, v_r_469_);
lean_ctor_set(v_reuseFailAlloc_506_, 4, v_r_469_);
v___x_502_ = v_reuseFailAlloc_506_;
goto v_reusejp_501_;
}
v_reusejp_501_:
{
lean_object* v___x_504_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v___x_502_);
lean_ctor_set(v___x_375_, 3, v___x_500_);
lean_ctor_set(v___x_375_, 2, v_v_494_);
lean_ctor_set(v___x_375_, 1, v_k_493_);
lean_ctor_set(v___x_375_, 0, v___x_498_);
v___x_504_ = v___x_375_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v___x_498_);
lean_ctor_set(v_reuseFailAlloc_505_, 1, v_k_493_);
lean_ctor_set(v_reuseFailAlloc_505_, 2, v_v_494_);
lean_ctor_set(v_reuseFailAlloc_505_, 3, v___x_500_);
lean_ctor_set(v_reuseFailAlloc_505_, 4, v___x_502_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
return v___x_504_;
}
}
}
}
}
}
}
else
{
lean_object* v_r_516_; 
v_r_516_ = lean_ctor_get(v_r_373_, 4);
lean_inc(v_r_516_);
if (lean_obj_tag(v_r_516_) == 0)
{
lean_object* v_k_517_; lean_object* v_v_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_529_; 
v_k_517_ = lean_ctor_get(v_r_373_, 1);
v_v_518_ = lean_ctor_get(v_r_373_, 2);
v_isSharedCheck_529_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_529_ == 0)
{
lean_object* v_unused_530_; lean_object* v_unused_531_; lean_object* v_unused_532_; 
v_unused_530_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_530_);
v_unused_531_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_531_);
v_unused_532_ = lean_ctor_get(v_r_373_, 0);
lean_dec(v_unused_532_);
v___x_520_ = v_r_373_;
v_isShared_521_ = v_isSharedCheck_529_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_v_518_);
lean_inc(v_k_517_);
lean_dec(v_r_373_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_529_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_522_; lean_object* v___x_524_; 
v___x_522_ = lean_unsigned_to_nat(3u);
if (v_isShared_521_ == 0)
{
lean_ctor_set(v___x_520_, 4, v_l_468_);
lean_ctor_set(v___x_520_, 2, v_v_371_);
lean_ctor_set(v___x_520_, 1, v_k_370_);
lean_ctor_set(v___x_520_, 0, v___x_379_);
v___x_524_ = v___x_520_;
goto v_reusejp_523_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v___x_379_);
lean_ctor_set(v_reuseFailAlloc_528_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_528_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_528_, 3, v_l_468_);
lean_ctor_set(v_reuseFailAlloc_528_, 4, v_l_468_);
v___x_524_ = v_reuseFailAlloc_528_;
goto v_reusejp_523_;
}
v_reusejp_523_:
{
lean_object* v___x_526_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_r_516_);
lean_ctor_set(v___x_375_, 3, v___x_524_);
lean_ctor_set(v___x_375_, 2, v_v_518_);
lean_ctor_set(v___x_375_, 1, v_k_517_);
lean_ctor_set(v___x_375_, 0, v___x_522_);
v___x_526_ = v___x_375_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v___x_522_);
lean_ctor_set(v_reuseFailAlloc_527_, 1, v_k_517_);
lean_ctor_set(v_reuseFailAlloc_527_, 2, v_v_518_);
lean_ctor_set(v_reuseFailAlloc_527_, 3, v___x_524_);
lean_ctor_set(v_reuseFailAlloc_527_, 4, v_r_516_);
v___x_526_ = v_reuseFailAlloc_527_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
return v___x_526_;
}
}
}
}
else
{
lean_object* v_size_533_; lean_object* v_k_534_; lean_object* v_v_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_546_; 
v_size_533_ = lean_ctor_get(v_r_373_, 0);
v_k_534_ = lean_ctor_get(v_r_373_, 1);
v_v_535_ = lean_ctor_get(v_r_373_, 2);
v_isSharedCheck_546_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_546_ == 0)
{
lean_object* v_unused_547_; lean_object* v_unused_548_; 
v_unused_547_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_547_);
v_unused_548_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_548_);
v___x_537_ = v_r_373_;
v_isShared_538_ = v_isSharedCheck_546_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_v_535_);
lean_inc(v_k_534_);
lean_inc(v_size_533_);
lean_dec(v_r_373_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_546_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_540_; 
if (v_isShared_538_ == 0)
{
lean_ctor_set(v___x_537_, 3, v_r_516_);
v___x_540_ = v___x_537_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v_size_533_);
lean_ctor_set(v_reuseFailAlloc_545_, 1, v_k_534_);
lean_ctor_set(v_reuseFailAlloc_545_, 2, v_v_535_);
lean_ctor_set(v_reuseFailAlloc_545_, 3, v_r_516_);
lean_ctor_set(v_reuseFailAlloc_545_, 4, v_r_516_);
v___x_540_ = v_reuseFailAlloc_545_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
lean_object* v___x_541_; lean_object* v___x_543_; 
v___x_541_ = lean_unsigned_to_nat(2u);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v___x_540_);
lean_ctor_set(v___x_375_, 3, v_r_516_);
lean_ctor_set(v___x_375_, 0, v___x_541_);
v___x_543_ = v___x_375_;
goto v_reusejp_542_;
}
else
{
lean_object* v_reuseFailAlloc_544_; 
v_reuseFailAlloc_544_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_544_, 0, v___x_541_);
lean_ctor_set(v_reuseFailAlloc_544_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_544_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_544_, 3, v_r_516_);
lean_ctor_set(v_reuseFailAlloc_544_, 4, v___x_540_);
v___x_543_ = v_reuseFailAlloc_544_;
goto v_reusejp_542_;
}
v_reusejp_542_:
{
return v___x_543_;
}
}
}
}
}
}
else
{
lean_object* v___x_550_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 3, v_r_373_);
lean_ctor_set(v___x_375_, 0, v___x_379_);
v___x_550_ = v___x_375_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v___x_379_);
lean_ctor_set(v_reuseFailAlloc_551_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_551_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_551_, 3, v_r_373_);
lean_ctor_set(v_reuseFailAlloc_551_, 4, v_r_373_);
v___x_550_ = v_reuseFailAlloc_551_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
return v___x_550_;
}
}
}
}
case 1:
{
lean_del_object(v___x_375_);
lean_dec(v_v_371_);
lean_dec(v_k_370_);
if (lean_obj_tag(v_l_372_) == 0)
{
if (lean_obj_tag(v_r_373_) == 0)
{
lean_object* v_size_552_; lean_object* v_k_553_; lean_object* v_v_554_; lean_object* v_l_555_; lean_object* v_r_556_; lean_object* v_size_557_; lean_object* v_k_558_; lean_object* v_v_559_; lean_object* v_l_560_; lean_object* v_r_561_; lean_object* v___x_562_; uint8_t v___x_563_; 
v_size_552_ = lean_ctor_get(v_l_372_, 0);
v_k_553_ = lean_ctor_get(v_l_372_, 1);
v_v_554_ = lean_ctor_get(v_l_372_, 2);
v_l_555_ = lean_ctor_get(v_l_372_, 3);
v_r_556_ = lean_ctor_get(v_l_372_, 4);
lean_inc(v_r_556_);
v_size_557_ = lean_ctor_get(v_r_373_, 0);
v_k_558_ = lean_ctor_get(v_r_373_, 1);
v_v_559_ = lean_ctor_get(v_r_373_, 2);
v_l_560_ = lean_ctor_get(v_r_373_, 3);
lean_inc(v_l_560_);
v_r_561_ = lean_ctor_get(v_r_373_, 4);
v___x_562_ = lean_unsigned_to_nat(1u);
v___x_563_ = lean_nat_dec_lt(v_size_552_, v_size_557_);
if (v___x_563_ == 0)
{
lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_699_; 
lean_inc(v_l_555_);
lean_inc(v_v_554_);
lean_inc(v_k_553_);
v_isSharedCheck_699_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_699_ == 0)
{
lean_object* v_unused_700_; lean_object* v_unused_701_; lean_object* v_unused_702_; lean_object* v_unused_703_; lean_object* v_unused_704_; 
v_unused_700_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_700_);
v_unused_701_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_701_);
v_unused_702_ = lean_ctor_get(v_l_372_, 2);
lean_dec(v_unused_702_);
v_unused_703_ = lean_ctor_get(v_l_372_, 1);
lean_dec(v_unused_703_);
v_unused_704_ = lean_ctor_get(v_l_372_, 0);
lean_dec(v_unused_704_);
v___x_565_ = v_l_372_;
v_isShared_566_ = v_isSharedCheck_699_;
goto v_resetjp_564_;
}
else
{
lean_dec(v_l_372_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_699_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
lean_object* v___x_567_; lean_object* v_tree_568_; 
v___x_567_ = l_Std_DTreeMap_Internal_Impl_maxView___redArg(v_k_553_, v_v_554_, v_l_555_, v_r_556_);
v_tree_568_ = lean_ctor_get(v___x_567_, 2);
lean_inc(v_tree_568_);
if (lean_obj_tag(v_tree_568_) == 0)
{
lean_object* v_k_569_; lean_object* v_v_570_; lean_object* v_size_571_; lean_object* v___x_572_; lean_object* v___x_573_; uint8_t v___x_574_; 
v_k_569_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_k_569_);
v_v_570_ = lean_ctor_get(v___x_567_, 1);
lean_inc(v_v_570_);
lean_dec_ref(v___x_567_);
v_size_571_ = lean_ctor_get(v_tree_568_, 0);
v___x_572_ = lean_unsigned_to_nat(3u);
v___x_573_ = lean_nat_mul(v___x_572_, v_size_571_);
v___x_574_ = lean_nat_dec_lt(v___x_573_, v_size_557_);
lean_dec(v___x_573_);
if (v___x_574_ == 0)
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_578_; 
lean_dec(v_l_560_);
v___x_575_ = lean_nat_add(v___x_562_, v_size_571_);
v___x_576_ = lean_nat_add(v___x_575_, v_size_557_);
lean_dec(v___x_575_);
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v_r_373_);
lean_ctor_set(v___x_565_, 3, v_tree_568_);
lean_ctor_set(v___x_565_, 2, v_v_570_);
lean_ctor_set(v___x_565_, 1, v_k_569_);
lean_ctor_set(v___x_565_, 0, v___x_576_);
v___x_578_ = v___x_565_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v___x_576_);
lean_ctor_set(v_reuseFailAlloc_579_, 1, v_k_569_);
lean_ctor_set(v_reuseFailAlloc_579_, 2, v_v_570_);
lean_ctor_set(v_reuseFailAlloc_579_, 3, v_tree_568_);
lean_ctor_set(v_reuseFailAlloc_579_, 4, v_r_373_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
else
{
lean_object* v___x_581_; uint8_t v_isShared_582_; uint8_t v_isSharedCheck_634_; 
lean_inc(v_r_561_);
lean_inc(v_v_559_);
lean_inc(v_k_558_);
lean_inc(v_size_557_);
v_isSharedCheck_634_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_634_ == 0)
{
lean_object* v_unused_635_; lean_object* v_unused_636_; lean_object* v_unused_637_; lean_object* v_unused_638_; lean_object* v_unused_639_; 
v_unused_635_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_635_);
v_unused_636_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_636_);
v_unused_637_ = lean_ctor_get(v_r_373_, 2);
lean_dec(v_unused_637_);
v_unused_638_ = lean_ctor_get(v_r_373_, 1);
lean_dec(v_unused_638_);
v_unused_639_ = lean_ctor_get(v_r_373_, 0);
lean_dec(v_unused_639_);
v___x_581_ = v_r_373_;
v_isShared_582_ = v_isSharedCheck_634_;
goto v_resetjp_580_;
}
else
{
lean_dec(v_r_373_);
v___x_581_ = lean_box(0);
v_isShared_582_ = v_isSharedCheck_634_;
goto v_resetjp_580_;
}
v_resetjp_580_:
{
lean_object* v_size_583_; lean_object* v_k_584_; lean_object* v_v_585_; lean_object* v_l_586_; lean_object* v_r_587_; lean_object* v_size_588_; lean_object* v___x_589_; lean_object* v___x_590_; uint8_t v___x_591_; 
v_size_583_ = lean_ctor_get(v_l_560_, 0);
v_k_584_ = lean_ctor_get(v_l_560_, 1);
v_v_585_ = lean_ctor_get(v_l_560_, 2);
v_l_586_ = lean_ctor_get(v_l_560_, 3);
v_r_587_ = lean_ctor_get(v_l_560_, 4);
v_size_588_ = lean_ctor_get(v_r_561_, 0);
v___x_589_ = lean_unsigned_to_nat(2u);
v___x_590_ = lean_nat_mul(v___x_589_, v_size_588_);
v___x_591_ = lean_nat_dec_lt(v_size_583_, v___x_590_);
lean_dec(v___x_590_);
if (v___x_591_ == 0)
{
lean_object* v___x_593_; uint8_t v_isShared_594_; uint8_t v_isSharedCheck_619_; 
lean_inc(v_r_587_);
lean_inc(v_l_586_);
lean_inc(v_v_585_);
lean_inc(v_k_584_);
v_isSharedCheck_619_ = !lean_is_exclusive(v_l_560_);
if (v_isSharedCheck_619_ == 0)
{
lean_object* v_unused_620_; lean_object* v_unused_621_; lean_object* v_unused_622_; lean_object* v_unused_623_; lean_object* v_unused_624_; 
v_unused_620_ = lean_ctor_get(v_l_560_, 4);
lean_dec(v_unused_620_);
v_unused_621_ = lean_ctor_get(v_l_560_, 3);
lean_dec(v_unused_621_);
v_unused_622_ = lean_ctor_get(v_l_560_, 2);
lean_dec(v_unused_622_);
v_unused_623_ = lean_ctor_get(v_l_560_, 1);
lean_dec(v_unused_623_);
v_unused_624_ = lean_ctor_get(v_l_560_, 0);
lean_dec(v_unused_624_);
v___x_593_ = v_l_560_;
v_isShared_594_ = v_isSharedCheck_619_;
goto v_resetjp_592_;
}
else
{
lean_dec(v_l_560_);
v___x_593_ = lean_box(0);
v_isShared_594_ = v_isSharedCheck_619_;
goto v_resetjp_592_;
}
v_resetjp_592_:
{
lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___y_598_; lean_object* v___y_599_; lean_object* v___y_600_; lean_object* v___y_609_; 
v___x_595_ = lean_nat_add(v___x_562_, v_size_571_);
v___x_596_ = lean_nat_add(v___x_595_, v_size_557_);
lean_dec(v_size_557_);
if (lean_obj_tag(v_l_586_) == 0)
{
lean_object* v_size_617_; 
v_size_617_ = lean_ctor_get(v_l_586_, 0);
lean_inc(v_size_617_);
v___y_609_ = v_size_617_;
goto v___jp_608_;
}
else
{
lean_object* v___x_618_; 
v___x_618_ = lean_unsigned_to_nat(0u);
v___y_609_ = v___x_618_;
goto v___jp_608_;
}
v___jp_597_:
{
lean_object* v___x_601_; lean_object* v___x_603_; 
v___x_601_ = lean_nat_add(v___y_598_, v___y_600_);
lean_dec(v___y_600_);
lean_dec(v___y_598_);
if (v_isShared_594_ == 0)
{
lean_ctor_set(v___x_593_, 4, v_r_561_);
lean_ctor_set(v___x_593_, 3, v_r_587_);
lean_ctor_set(v___x_593_, 2, v_v_559_);
lean_ctor_set(v___x_593_, 1, v_k_558_);
lean_ctor_set(v___x_593_, 0, v___x_601_);
v___x_603_ = v___x_593_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_601_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v_k_558_);
lean_ctor_set(v_reuseFailAlloc_607_, 2, v_v_559_);
lean_ctor_set(v_reuseFailAlloc_607_, 3, v_r_587_);
lean_ctor_set(v_reuseFailAlloc_607_, 4, v_r_561_);
v___x_603_ = v_reuseFailAlloc_607_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
lean_object* v___x_605_; 
if (v_isShared_582_ == 0)
{
lean_ctor_set(v___x_581_, 4, v___x_603_);
lean_ctor_set(v___x_581_, 3, v___y_599_);
lean_ctor_set(v___x_581_, 2, v_v_585_);
lean_ctor_set(v___x_581_, 1, v_k_584_);
lean_ctor_set(v___x_581_, 0, v___x_596_);
v___x_605_ = v___x_581_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v___x_596_);
lean_ctor_set(v_reuseFailAlloc_606_, 1, v_k_584_);
lean_ctor_set(v_reuseFailAlloc_606_, 2, v_v_585_);
lean_ctor_set(v_reuseFailAlloc_606_, 3, v___y_599_);
lean_ctor_set(v_reuseFailAlloc_606_, 4, v___x_603_);
v___x_605_ = v_reuseFailAlloc_606_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
return v___x_605_;
}
}
}
v___jp_608_:
{
lean_object* v___x_610_; lean_object* v___x_612_; 
v___x_610_ = lean_nat_add(v___x_595_, v___y_609_);
lean_dec(v___y_609_);
lean_dec(v___x_595_);
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v_l_586_);
lean_ctor_set(v___x_565_, 3, v_tree_568_);
lean_ctor_set(v___x_565_, 2, v_v_570_);
lean_ctor_set(v___x_565_, 1, v_k_569_);
lean_ctor_set(v___x_565_, 0, v___x_610_);
v___x_612_ = v___x_565_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v___x_610_);
lean_ctor_set(v_reuseFailAlloc_616_, 1, v_k_569_);
lean_ctor_set(v_reuseFailAlloc_616_, 2, v_v_570_);
lean_ctor_set(v_reuseFailAlloc_616_, 3, v_tree_568_);
lean_ctor_set(v_reuseFailAlloc_616_, 4, v_l_586_);
v___x_612_ = v_reuseFailAlloc_616_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
lean_object* v___x_613_; 
v___x_613_ = lean_nat_add(v___x_562_, v_size_588_);
if (lean_obj_tag(v_r_587_) == 0)
{
lean_object* v_size_614_; 
v_size_614_ = lean_ctor_get(v_r_587_, 0);
lean_inc(v_size_614_);
v___y_598_ = v___x_613_;
v___y_599_ = v___x_612_;
v___y_600_ = v_size_614_;
goto v___jp_597_;
}
else
{
lean_object* v___x_615_; 
v___x_615_ = lean_unsigned_to_nat(0u);
v___y_598_ = v___x_613_;
v___y_599_ = v___x_612_;
v___y_600_ = v___x_615_;
goto v___jp_597_;
}
}
}
}
}
else
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_629_; 
v___x_625_ = lean_nat_add(v___x_562_, v_size_571_);
v___x_626_ = lean_nat_add(v___x_625_, v_size_557_);
lean_dec(v_size_557_);
v___x_627_ = lean_nat_add(v___x_625_, v_size_583_);
lean_dec(v___x_625_);
if (v_isShared_582_ == 0)
{
lean_ctor_set(v___x_581_, 4, v_l_560_);
lean_ctor_set(v___x_581_, 3, v_tree_568_);
lean_ctor_set(v___x_581_, 2, v_v_570_);
lean_ctor_set(v___x_581_, 1, v_k_569_);
lean_ctor_set(v___x_581_, 0, v___x_627_);
v___x_629_ = v___x_581_;
goto v_reusejp_628_;
}
else
{
lean_object* v_reuseFailAlloc_633_; 
v_reuseFailAlloc_633_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_633_, 0, v___x_627_);
lean_ctor_set(v_reuseFailAlloc_633_, 1, v_k_569_);
lean_ctor_set(v_reuseFailAlloc_633_, 2, v_v_570_);
lean_ctor_set(v_reuseFailAlloc_633_, 3, v_tree_568_);
lean_ctor_set(v_reuseFailAlloc_633_, 4, v_l_560_);
v___x_629_ = v_reuseFailAlloc_633_;
goto v_reusejp_628_;
}
v_reusejp_628_:
{
lean_object* v___x_631_; 
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v_r_561_);
lean_ctor_set(v___x_565_, 3, v___x_629_);
lean_ctor_set(v___x_565_, 2, v_v_559_);
lean_ctor_set(v___x_565_, 1, v_k_558_);
lean_ctor_set(v___x_565_, 0, v___x_626_);
v___x_631_ = v___x_565_;
goto v_reusejp_630_;
}
else
{
lean_object* v_reuseFailAlloc_632_; 
v_reuseFailAlloc_632_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_632_, 0, v___x_626_);
lean_ctor_set(v_reuseFailAlloc_632_, 1, v_k_558_);
lean_ctor_set(v_reuseFailAlloc_632_, 2, v_v_559_);
lean_ctor_set(v_reuseFailAlloc_632_, 3, v___x_629_);
lean_ctor_set(v_reuseFailAlloc_632_, 4, v_r_561_);
v___x_631_ = v_reuseFailAlloc_632_;
goto v_reusejp_630_;
}
v_reusejp_630_:
{
return v___x_631_;
}
}
}
}
}
}
else
{
lean_object* v___x_641_; uint8_t v_isShared_642_; uint8_t v_isSharedCheck_693_; 
lean_inc(v_r_561_);
lean_inc(v_v_559_);
lean_inc(v_k_558_);
lean_inc(v_size_557_);
v_isSharedCheck_693_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_693_ == 0)
{
lean_object* v_unused_694_; lean_object* v_unused_695_; lean_object* v_unused_696_; lean_object* v_unused_697_; lean_object* v_unused_698_; 
v_unused_694_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_694_);
v_unused_695_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_695_);
v_unused_696_ = lean_ctor_get(v_r_373_, 2);
lean_dec(v_unused_696_);
v_unused_697_ = lean_ctor_get(v_r_373_, 1);
lean_dec(v_unused_697_);
v_unused_698_ = lean_ctor_get(v_r_373_, 0);
lean_dec(v_unused_698_);
v___x_641_ = v_r_373_;
v_isShared_642_ = v_isSharedCheck_693_;
goto v_resetjp_640_;
}
else
{
lean_dec(v_r_373_);
v___x_641_ = lean_box(0);
v_isShared_642_ = v_isSharedCheck_693_;
goto v_resetjp_640_;
}
v_resetjp_640_:
{
if (lean_obj_tag(v_l_560_) == 0)
{
if (lean_obj_tag(v_r_561_) == 0)
{
lean_object* v_k_643_; lean_object* v_v_644_; lean_object* v_size_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_649_; 
v_k_643_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_k_643_);
v_v_644_ = lean_ctor_get(v___x_567_, 1);
lean_inc(v_v_644_);
lean_dec_ref(v___x_567_);
v_size_645_ = lean_ctor_get(v_l_560_, 0);
v___x_646_ = lean_nat_add(v___x_562_, v_size_557_);
lean_dec(v_size_557_);
v___x_647_ = lean_nat_add(v___x_562_, v_size_645_);
if (v_isShared_642_ == 0)
{
lean_ctor_set(v___x_641_, 4, v_l_560_);
lean_ctor_set(v___x_641_, 3, v_tree_568_);
lean_ctor_set(v___x_641_, 2, v_v_644_);
lean_ctor_set(v___x_641_, 1, v_k_643_);
lean_ctor_set(v___x_641_, 0, v___x_647_);
v___x_649_ = v___x_641_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_653_; 
v_reuseFailAlloc_653_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_653_, 0, v___x_647_);
lean_ctor_set(v_reuseFailAlloc_653_, 1, v_k_643_);
lean_ctor_set(v_reuseFailAlloc_653_, 2, v_v_644_);
lean_ctor_set(v_reuseFailAlloc_653_, 3, v_tree_568_);
lean_ctor_set(v_reuseFailAlloc_653_, 4, v_l_560_);
v___x_649_ = v_reuseFailAlloc_653_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
lean_object* v___x_651_; 
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v_r_561_);
lean_ctor_set(v___x_565_, 3, v___x_649_);
lean_ctor_set(v___x_565_, 2, v_v_559_);
lean_ctor_set(v___x_565_, 1, v_k_558_);
lean_ctor_set(v___x_565_, 0, v___x_646_);
v___x_651_ = v___x_565_;
goto v_reusejp_650_;
}
else
{
lean_object* v_reuseFailAlloc_652_; 
v_reuseFailAlloc_652_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_652_, 0, v___x_646_);
lean_ctor_set(v_reuseFailAlloc_652_, 1, v_k_558_);
lean_ctor_set(v_reuseFailAlloc_652_, 2, v_v_559_);
lean_ctor_set(v_reuseFailAlloc_652_, 3, v___x_649_);
lean_ctor_set(v_reuseFailAlloc_652_, 4, v_r_561_);
v___x_651_ = v_reuseFailAlloc_652_;
goto v_reusejp_650_;
}
v_reusejp_650_:
{
return v___x_651_;
}
}
}
else
{
lean_object* v_k_654_; lean_object* v_v_655_; lean_object* v_k_656_; lean_object* v_v_657_; lean_object* v___x_659_; uint8_t v_isShared_660_; uint8_t v_isSharedCheck_671_; 
lean_dec(v_size_557_);
v_k_654_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_k_654_);
v_v_655_ = lean_ctor_get(v___x_567_, 1);
lean_inc(v_v_655_);
lean_dec_ref(v___x_567_);
v_k_656_ = lean_ctor_get(v_l_560_, 1);
v_v_657_ = lean_ctor_get(v_l_560_, 2);
v_isSharedCheck_671_ = !lean_is_exclusive(v_l_560_);
if (v_isSharedCheck_671_ == 0)
{
lean_object* v_unused_672_; lean_object* v_unused_673_; lean_object* v_unused_674_; 
v_unused_672_ = lean_ctor_get(v_l_560_, 4);
lean_dec(v_unused_672_);
v_unused_673_ = lean_ctor_get(v_l_560_, 3);
lean_dec(v_unused_673_);
v_unused_674_ = lean_ctor_get(v_l_560_, 0);
lean_dec(v_unused_674_);
v___x_659_ = v_l_560_;
v_isShared_660_ = v_isSharedCheck_671_;
goto v_resetjp_658_;
}
else
{
lean_inc(v_v_657_);
lean_inc(v_k_656_);
lean_dec(v_l_560_);
v___x_659_ = lean_box(0);
v_isShared_660_ = v_isSharedCheck_671_;
goto v_resetjp_658_;
}
v_resetjp_658_:
{
lean_object* v___x_661_; lean_object* v___x_663_; 
v___x_661_ = lean_unsigned_to_nat(3u);
if (v_isShared_660_ == 0)
{
lean_ctor_set(v___x_659_, 4, v_r_561_);
lean_ctor_set(v___x_659_, 3, v_r_561_);
lean_ctor_set(v___x_659_, 2, v_v_655_);
lean_ctor_set(v___x_659_, 1, v_k_654_);
lean_ctor_set(v___x_659_, 0, v___x_562_);
v___x_663_ = v___x_659_;
goto v_reusejp_662_;
}
else
{
lean_object* v_reuseFailAlloc_670_; 
v_reuseFailAlloc_670_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_670_, 0, v___x_562_);
lean_ctor_set(v_reuseFailAlloc_670_, 1, v_k_654_);
lean_ctor_set(v_reuseFailAlloc_670_, 2, v_v_655_);
lean_ctor_set(v_reuseFailAlloc_670_, 3, v_r_561_);
lean_ctor_set(v_reuseFailAlloc_670_, 4, v_r_561_);
v___x_663_ = v_reuseFailAlloc_670_;
goto v_reusejp_662_;
}
v_reusejp_662_:
{
lean_object* v___x_665_; 
if (v_isShared_642_ == 0)
{
lean_ctor_set(v___x_641_, 3, v_r_561_);
lean_ctor_set(v___x_641_, 0, v___x_562_);
v___x_665_ = v___x_641_;
goto v_reusejp_664_;
}
else
{
lean_object* v_reuseFailAlloc_669_; 
v_reuseFailAlloc_669_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_669_, 0, v___x_562_);
lean_ctor_set(v_reuseFailAlloc_669_, 1, v_k_558_);
lean_ctor_set(v_reuseFailAlloc_669_, 2, v_v_559_);
lean_ctor_set(v_reuseFailAlloc_669_, 3, v_r_561_);
lean_ctor_set(v_reuseFailAlloc_669_, 4, v_r_561_);
v___x_665_ = v_reuseFailAlloc_669_;
goto v_reusejp_664_;
}
v_reusejp_664_:
{
lean_object* v___x_667_; 
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v___x_665_);
lean_ctor_set(v___x_565_, 3, v___x_663_);
lean_ctor_set(v___x_565_, 2, v_v_657_);
lean_ctor_set(v___x_565_, 1, v_k_656_);
lean_ctor_set(v___x_565_, 0, v___x_661_);
v___x_667_ = v___x_565_;
goto v_reusejp_666_;
}
else
{
lean_object* v_reuseFailAlloc_668_; 
v_reuseFailAlloc_668_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_668_, 0, v___x_661_);
lean_ctor_set(v_reuseFailAlloc_668_, 1, v_k_656_);
lean_ctor_set(v_reuseFailAlloc_668_, 2, v_v_657_);
lean_ctor_set(v_reuseFailAlloc_668_, 3, v___x_663_);
lean_ctor_set(v_reuseFailAlloc_668_, 4, v___x_665_);
v___x_667_ = v_reuseFailAlloc_668_;
goto v_reusejp_666_;
}
v_reusejp_666_:
{
return v___x_667_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_561_) == 0)
{
lean_object* v_k_675_; lean_object* v_v_676_; lean_object* v___x_677_; lean_object* v___x_679_; 
lean_dec(v_size_557_);
v_k_675_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_k_675_);
v_v_676_ = lean_ctor_get(v___x_567_, 1);
lean_inc(v_v_676_);
lean_dec_ref(v___x_567_);
v___x_677_ = lean_unsigned_to_nat(3u);
if (v_isShared_642_ == 0)
{
lean_ctor_set(v___x_641_, 4, v_l_560_);
lean_ctor_set(v___x_641_, 2, v_v_676_);
lean_ctor_set(v___x_641_, 1, v_k_675_);
lean_ctor_set(v___x_641_, 0, v___x_562_);
v___x_679_ = v___x_641_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_683_; 
v_reuseFailAlloc_683_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_683_, 0, v___x_562_);
lean_ctor_set(v_reuseFailAlloc_683_, 1, v_k_675_);
lean_ctor_set(v_reuseFailAlloc_683_, 2, v_v_676_);
lean_ctor_set(v_reuseFailAlloc_683_, 3, v_l_560_);
lean_ctor_set(v_reuseFailAlloc_683_, 4, v_l_560_);
v___x_679_ = v_reuseFailAlloc_683_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
lean_object* v___x_681_; 
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v_r_561_);
lean_ctor_set(v___x_565_, 3, v___x_679_);
lean_ctor_set(v___x_565_, 2, v_v_559_);
lean_ctor_set(v___x_565_, 1, v_k_558_);
lean_ctor_set(v___x_565_, 0, v___x_677_);
v___x_681_ = v___x_565_;
goto v_reusejp_680_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v___x_677_);
lean_ctor_set(v_reuseFailAlloc_682_, 1, v_k_558_);
lean_ctor_set(v_reuseFailAlloc_682_, 2, v_v_559_);
lean_ctor_set(v_reuseFailAlloc_682_, 3, v___x_679_);
lean_ctor_set(v_reuseFailAlloc_682_, 4, v_r_561_);
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
lean_object* v_k_684_; lean_object* v_v_685_; lean_object* v___x_687_; 
v_k_684_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_k_684_);
v_v_685_ = lean_ctor_get(v___x_567_, 1);
lean_inc(v_v_685_);
lean_dec_ref(v___x_567_);
if (v_isShared_642_ == 0)
{
lean_ctor_set(v___x_641_, 3, v_r_561_);
v___x_687_ = v___x_641_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v_size_557_);
lean_ctor_set(v_reuseFailAlloc_692_, 1, v_k_558_);
lean_ctor_set(v_reuseFailAlloc_692_, 2, v_v_559_);
lean_ctor_set(v_reuseFailAlloc_692_, 3, v_r_561_);
lean_ctor_set(v_reuseFailAlloc_692_, 4, v_r_561_);
v___x_687_ = v_reuseFailAlloc_692_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
lean_object* v___x_688_; lean_object* v___x_690_; 
v___x_688_ = lean_unsigned_to_nat(2u);
if (v_isShared_566_ == 0)
{
lean_ctor_set(v___x_565_, 4, v___x_687_);
lean_ctor_set(v___x_565_, 3, v_r_561_);
lean_ctor_set(v___x_565_, 2, v_v_685_);
lean_ctor_set(v___x_565_, 1, v_k_684_);
lean_ctor_set(v___x_565_, 0, v___x_688_);
v___x_690_ = v___x_565_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v___x_688_);
lean_ctor_set(v_reuseFailAlloc_691_, 1, v_k_684_);
lean_ctor_set(v_reuseFailAlloc_691_, 2, v_v_685_);
lean_ctor_set(v_reuseFailAlloc_691_, 3, v_r_561_);
lean_ctor_set(v_reuseFailAlloc_691_, 4, v___x_687_);
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
}
}
}
}
else
{
lean_object* v___x_706_; uint8_t v_isShared_707_; uint8_t v_isSharedCheck_857_; 
lean_inc(v_r_561_);
lean_inc(v_v_559_);
lean_inc(v_k_558_);
v_isSharedCheck_857_ = !lean_is_exclusive(v_r_373_);
if (v_isSharedCheck_857_ == 0)
{
lean_object* v_unused_858_; lean_object* v_unused_859_; lean_object* v_unused_860_; lean_object* v_unused_861_; lean_object* v_unused_862_; 
v_unused_858_ = lean_ctor_get(v_r_373_, 4);
lean_dec(v_unused_858_);
v_unused_859_ = lean_ctor_get(v_r_373_, 3);
lean_dec(v_unused_859_);
v_unused_860_ = lean_ctor_get(v_r_373_, 2);
lean_dec(v_unused_860_);
v_unused_861_ = lean_ctor_get(v_r_373_, 1);
lean_dec(v_unused_861_);
v_unused_862_ = lean_ctor_get(v_r_373_, 0);
lean_dec(v_unused_862_);
v___x_706_ = v_r_373_;
v_isShared_707_ = v_isSharedCheck_857_;
goto v_resetjp_705_;
}
else
{
lean_dec(v_r_373_);
v___x_706_ = lean_box(0);
v_isShared_707_ = v_isSharedCheck_857_;
goto v_resetjp_705_;
}
v_resetjp_705_:
{
lean_object* v___x_708_; lean_object* v_tree_709_; 
v___x_708_ = l_Std_DTreeMap_Internal_Impl_minView___redArg(v_k_558_, v_v_559_, v_l_560_, v_r_561_);
v_tree_709_ = lean_ctor_get(v___x_708_, 2);
lean_inc(v_tree_709_);
if (lean_obj_tag(v_tree_709_) == 0)
{
lean_object* v_k_710_; lean_object* v_v_711_; lean_object* v_size_712_; lean_object* v___x_713_; lean_object* v___x_714_; uint8_t v___x_715_; 
v_k_710_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_k_710_);
v_v_711_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_v_711_);
lean_dec_ref(v___x_708_);
v_size_712_ = lean_ctor_get(v_tree_709_, 0);
v___x_713_ = lean_unsigned_to_nat(3u);
v___x_714_ = lean_nat_mul(v___x_713_, v_size_712_);
v___x_715_ = lean_nat_dec_lt(v___x_714_, v_size_552_);
lean_dec(v___x_714_);
if (v___x_715_ == 0)
{
lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_719_; 
lean_dec(v_r_556_);
v___x_716_ = lean_nat_add(v___x_562_, v_size_552_);
v___x_717_ = lean_nat_add(v___x_716_, v_size_712_);
lean_dec(v___x_716_);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 4, v_tree_709_);
lean_ctor_set(v___x_706_, 3, v_l_372_);
lean_ctor_set(v___x_706_, 2, v_v_711_);
lean_ctor_set(v___x_706_, 1, v_k_710_);
lean_ctor_set(v___x_706_, 0, v___x_717_);
v___x_719_ = v___x_706_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_720_; 
v_reuseFailAlloc_720_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_720_, 0, v___x_717_);
lean_ctor_set(v_reuseFailAlloc_720_, 1, v_k_710_);
lean_ctor_set(v_reuseFailAlloc_720_, 2, v_v_711_);
lean_ctor_set(v_reuseFailAlloc_720_, 3, v_l_372_);
lean_ctor_set(v_reuseFailAlloc_720_, 4, v_tree_709_);
v___x_719_ = v_reuseFailAlloc_720_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
return v___x_719_;
}
}
else
{
lean_object* v___x_722_; uint8_t v_isShared_723_; uint8_t v_isSharedCheck_786_; 
lean_inc(v_l_555_);
lean_inc(v_v_554_);
lean_inc(v_k_553_);
lean_inc(v_size_552_);
v_isSharedCheck_786_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_786_ == 0)
{
lean_object* v_unused_787_; lean_object* v_unused_788_; lean_object* v_unused_789_; lean_object* v_unused_790_; lean_object* v_unused_791_; 
v_unused_787_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_787_);
v_unused_788_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_788_);
v_unused_789_ = lean_ctor_get(v_l_372_, 2);
lean_dec(v_unused_789_);
v_unused_790_ = lean_ctor_get(v_l_372_, 1);
lean_dec(v_unused_790_);
v_unused_791_ = lean_ctor_get(v_l_372_, 0);
lean_dec(v_unused_791_);
v___x_722_ = v_l_372_;
v_isShared_723_ = v_isSharedCheck_786_;
goto v_resetjp_721_;
}
else
{
lean_dec(v_l_372_);
v___x_722_ = lean_box(0);
v_isShared_723_ = v_isSharedCheck_786_;
goto v_resetjp_721_;
}
v_resetjp_721_:
{
lean_object* v_size_724_; lean_object* v_size_725_; lean_object* v_k_726_; lean_object* v_v_727_; lean_object* v_l_728_; lean_object* v_r_729_; lean_object* v___x_730_; lean_object* v___x_731_; uint8_t v___x_732_; 
v_size_724_ = lean_ctor_get(v_l_555_, 0);
v_size_725_ = lean_ctor_get(v_r_556_, 0);
v_k_726_ = lean_ctor_get(v_r_556_, 1);
v_v_727_ = lean_ctor_get(v_r_556_, 2);
v_l_728_ = lean_ctor_get(v_r_556_, 3);
v_r_729_ = lean_ctor_get(v_r_556_, 4);
v___x_730_ = lean_unsigned_to_nat(2u);
v___x_731_ = lean_nat_mul(v___x_730_, v_size_724_);
v___x_732_ = lean_nat_dec_lt(v_size_725_, v___x_731_);
lean_dec(v___x_731_);
if (v___x_732_ == 0)
{
lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_770_; 
lean_inc(v_r_729_);
lean_inc(v_l_728_);
lean_inc(v_v_727_);
lean_inc(v_k_726_);
lean_del_object(v___x_722_);
v_isSharedCheck_770_ = !lean_is_exclusive(v_r_556_);
if (v_isSharedCheck_770_ == 0)
{
lean_object* v_unused_771_; lean_object* v_unused_772_; lean_object* v_unused_773_; lean_object* v_unused_774_; lean_object* v_unused_775_; 
v_unused_771_ = lean_ctor_get(v_r_556_, 4);
lean_dec(v_unused_771_);
v_unused_772_ = lean_ctor_get(v_r_556_, 3);
lean_dec(v_unused_772_);
v_unused_773_ = lean_ctor_get(v_r_556_, 2);
lean_dec(v_unused_773_);
v_unused_774_ = lean_ctor_get(v_r_556_, 1);
lean_dec(v_unused_774_);
v_unused_775_ = lean_ctor_get(v_r_556_, 0);
lean_dec(v_unused_775_);
v___x_734_ = v_r_556_;
v_isShared_735_ = v_isSharedCheck_770_;
goto v_resetjp_733_;
}
else
{
lean_dec(v_r_556_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_770_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___y_739_; lean_object* v___y_740_; lean_object* v___y_741_; lean_object* v___x_758_; lean_object* v___y_760_; 
v___x_736_ = lean_nat_add(v___x_562_, v_size_552_);
lean_dec(v_size_552_);
v___x_737_ = lean_nat_add(v___x_736_, v_size_712_);
lean_dec(v___x_736_);
v___x_758_ = lean_nat_add(v___x_562_, v_size_724_);
if (lean_obj_tag(v_l_728_) == 0)
{
lean_object* v_size_768_; 
v_size_768_ = lean_ctor_get(v_l_728_, 0);
lean_inc(v_size_768_);
v___y_760_ = v_size_768_;
goto v___jp_759_;
}
else
{
lean_object* v___x_769_; 
v___x_769_ = lean_unsigned_to_nat(0u);
v___y_760_ = v___x_769_;
goto v___jp_759_;
}
v___jp_738_:
{
lean_object* v___x_742_; lean_object* v___x_744_; 
v___x_742_ = lean_nat_add(v___y_740_, v___y_741_);
lean_dec(v___y_741_);
lean_dec(v___y_740_);
lean_inc_ref(v_tree_709_);
if (v_isShared_735_ == 0)
{
lean_ctor_set(v___x_734_, 4, v_tree_709_);
lean_ctor_set(v___x_734_, 3, v_r_729_);
lean_ctor_set(v___x_734_, 2, v_v_711_);
lean_ctor_set(v___x_734_, 1, v_k_710_);
lean_ctor_set(v___x_734_, 0, v___x_742_);
v___x_744_ = v___x_734_;
goto v_reusejp_743_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v___x_742_);
lean_ctor_set(v_reuseFailAlloc_757_, 1, v_k_710_);
lean_ctor_set(v_reuseFailAlloc_757_, 2, v_v_711_);
lean_ctor_set(v_reuseFailAlloc_757_, 3, v_r_729_);
lean_ctor_set(v_reuseFailAlloc_757_, 4, v_tree_709_);
v___x_744_ = v_reuseFailAlloc_757_;
goto v_reusejp_743_;
}
v_reusejp_743_:
{
lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_751_; 
v_isSharedCheck_751_ = !lean_is_exclusive(v_tree_709_);
if (v_isSharedCheck_751_ == 0)
{
lean_object* v_unused_752_; lean_object* v_unused_753_; lean_object* v_unused_754_; lean_object* v_unused_755_; lean_object* v_unused_756_; 
v_unused_752_ = lean_ctor_get(v_tree_709_, 4);
lean_dec(v_unused_752_);
v_unused_753_ = lean_ctor_get(v_tree_709_, 3);
lean_dec(v_unused_753_);
v_unused_754_ = lean_ctor_get(v_tree_709_, 2);
lean_dec(v_unused_754_);
v_unused_755_ = lean_ctor_get(v_tree_709_, 1);
lean_dec(v_unused_755_);
v_unused_756_ = lean_ctor_get(v_tree_709_, 0);
lean_dec(v_unused_756_);
v___x_746_ = v_tree_709_;
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
else
{
lean_dec(v_tree_709_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
lean_object* v___x_749_; 
if (v_isShared_747_ == 0)
{
lean_ctor_set(v___x_746_, 4, v___x_744_);
lean_ctor_set(v___x_746_, 3, v___y_739_);
lean_ctor_set(v___x_746_, 2, v_v_727_);
lean_ctor_set(v___x_746_, 1, v_k_726_);
lean_ctor_set(v___x_746_, 0, v___x_737_);
v___x_749_ = v___x_746_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v___x_737_);
lean_ctor_set(v_reuseFailAlloc_750_, 1, v_k_726_);
lean_ctor_set(v_reuseFailAlloc_750_, 2, v_v_727_);
lean_ctor_set(v_reuseFailAlloc_750_, 3, v___y_739_);
lean_ctor_set(v_reuseFailAlloc_750_, 4, v___x_744_);
v___x_749_ = v_reuseFailAlloc_750_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
return v___x_749_;
}
}
}
}
v___jp_759_:
{
lean_object* v___x_761_; lean_object* v___x_763_; 
v___x_761_ = lean_nat_add(v___x_758_, v___y_760_);
lean_dec(v___y_760_);
lean_dec(v___x_758_);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 4, v_l_728_);
lean_ctor_set(v___x_706_, 3, v_l_555_);
lean_ctor_set(v___x_706_, 2, v_v_554_);
lean_ctor_set(v___x_706_, 1, v_k_553_);
lean_ctor_set(v___x_706_, 0, v___x_761_);
v___x_763_ = v___x_706_;
goto v_reusejp_762_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v___x_761_);
lean_ctor_set(v_reuseFailAlloc_767_, 1, v_k_553_);
lean_ctor_set(v_reuseFailAlloc_767_, 2, v_v_554_);
lean_ctor_set(v_reuseFailAlloc_767_, 3, v_l_555_);
lean_ctor_set(v_reuseFailAlloc_767_, 4, v_l_728_);
v___x_763_ = v_reuseFailAlloc_767_;
goto v_reusejp_762_;
}
v_reusejp_762_:
{
lean_object* v___x_764_; 
v___x_764_ = lean_nat_add(v___x_562_, v_size_712_);
if (lean_obj_tag(v_r_729_) == 0)
{
lean_object* v_size_765_; 
v_size_765_ = lean_ctor_get(v_r_729_, 0);
lean_inc(v_size_765_);
v___y_739_ = v___x_763_;
v___y_740_ = v___x_764_;
v___y_741_ = v_size_765_;
goto v___jp_738_;
}
else
{
lean_object* v___x_766_; 
v___x_766_ = lean_unsigned_to_nat(0u);
v___y_739_ = v___x_763_;
v___y_740_ = v___x_764_;
v___y_741_ = v___x_766_;
goto v___jp_738_;
}
}
}
}
}
else
{
lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_781_; 
v___x_776_ = lean_nat_add(v___x_562_, v_size_552_);
lean_dec(v_size_552_);
v___x_777_ = lean_nat_add(v___x_776_, v_size_712_);
lean_dec(v___x_776_);
v___x_778_ = lean_nat_add(v___x_562_, v_size_712_);
v___x_779_ = lean_nat_add(v___x_778_, v_size_725_);
lean_dec(v___x_778_);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 4, v_tree_709_);
lean_ctor_set(v___x_706_, 3, v_r_556_);
lean_ctor_set(v___x_706_, 2, v_v_711_);
lean_ctor_set(v___x_706_, 1, v_k_710_);
lean_ctor_set(v___x_706_, 0, v___x_779_);
v___x_781_ = v___x_706_;
goto v_reusejp_780_;
}
else
{
lean_object* v_reuseFailAlloc_785_; 
v_reuseFailAlloc_785_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_785_, 0, v___x_779_);
lean_ctor_set(v_reuseFailAlloc_785_, 1, v_k_710_);
lean_ctor_set(v_reuseFailAlloc_785_, 2, v_v_711_);
lean_ctor_set(v_reuseFailAlloc_785_, 3, v_r_556_);
lean_ctor_set(v_reuseFailAlloc_785_, 4, v_tree_709_);
v___x_781_ = v_reuseFailAlloc_785_;
goto v_reusejp_780_;
}
v_reusejp_780_:
{
lean_object* v___x_783_; 
if (v_isShared_723_ == 0)
{
lean_ctor_set(v___x_722_, 4, v___x_781_);
lean_ctor_set(v___x_722_, 0, v___x_777_);
v___x_783_ = v___x_722_;
goto v_reusejp_782_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v___x_777_);
lean_ctor_set(v_reuseFailAlloc_784_, 1, v_k_553_);
lean_ctor_set(v_reuseFailAlloc_784_, 2, v_v_554_);
lean_ctor_set(v_reuseFailAlloc_784_, 3, v_l_555_);
lean_ctor_set(v_reuseFailAlloc_784_, 4, v___x_781_);
v___x_783_ = v_reuseFailAlloc_784_;
goto v_reusejp_782_;
}
v_reusejp_782_:
{
return v___x_783_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_l_555_) == 0)
{
lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_815_; 
lean_inc_ref(v_l_555_);
lean_inc(v_v_554_);
lean_inc(v_k_553_);
lean_inc(v_size_552_);
v_isSharedCheck_815_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_815_ == 0)
{
lean_object* v_unused_816_; lean_object* v_unused_817_; lean_object* v_unused_818_; lean_object* v_unused_819_; lean_object* v_unused_820_; 
v_unused_816_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_816_);
v_unused_817_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_817_);
v_unused_818_ = lean_ctor_get(v_l_372_, 2);
lean_dec(v_unused_818_);
v_unused_819_ = lean_ctor_get(v_l_372_, 1);
lean_dec(v_unused_819_);
v_unused_820_ = lean_ctor_get(v_l_372_, 0);
lean_dec(v_unused_820_);
v___x_793_ = v_l_372_;
v_isShared_794_ = v_isSharedCheck_815_;
goto v_resetjp_792_;
}
else
{
lean_dec(v_l_372_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_815_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
if (lean_obj_tag(v_r_556_) == 0)
{
lean_object* v_k_795_; lean_object* v_v_796_; lean_object* v_size_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_801_; 
v_k_795_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_k_795_);
v_v_796_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_v_796_);
lean_dec_ref(v___x_708_);
v_size_797_ = lean_ctor_get(v_r_556_, 0);
v___x_798_ = lean_nat_add(v___x_562_, v_size_552_);
lean_dec(v_size_552_);
v___x_799_ = lean_nat_add(v___x_562_, v_size_797_);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 4, v_tree_709_);
lean_ctor_set(v___x_706_, 3, v_r_556_);
lean_ctor_set(v___x_706_, 2, v_v_796_);
lean_ctor_set(v___x_706_, 1, v_k_795_);
lean_ctor_set(v___x_706_, 0, v___x_799_);
v___x_801_ = v___x_706_;
goto v_reusejp_800_;
}
else
{
lean_object* v_reuseFailAlloc_805_; 
v_reuseFailAlloc_805_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_805_, 0, v___x_799_);
lean_ctor_set(v_reuseFailAlloc_805_, 1, v_k_795_);
lean_ctor_set(v_reuseFailAlloc_805_, 2, v_v_796_);
lean_ctor_set(v_reuseFailAlloc_805_, 3, v_r_556_);
lean_ctor_set(v_reuseFailAlloc_805_, 4, v_tree_709_);
v___x_801_ = v_reuseFailAlloc_805_;
goto v_reusejp_800_;
}
v_reusejp_800_:
{
lean_object* v___x_803_; 
if (v_isShared_794_ == 0)
{
lean_ctor_set(v___x_793_, 4, v___x_801_);
lean_ctor_set(v___x_793_, 0, v___x_798_);
v___x_803_ = v___x_793_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v___x_798_);
lean_ctor_set(v_reuseFailAlloc_804_, 1, v_k_553_);
lean_ctor_set(v_reuseFailAlloc_804_, 2, v_v_554_);
lean_ctor_set(v_reuseFailAlloc_804_, 3, v_l_555_);
lean_ctor_set(v_reuseFailAlloc_804_, 4, v___x_801_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
else
{
lean_object* v_k_806_; lean_object* v_v_807_; lean_object* v___x_808_; lean_object* v___x_810_; 
lean_dec(v_size_552_);
v_k_806_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_k_806_);
v_v_807_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_v_807_);
lean_dec_ref(v___x_708_);
v___x_808_ = lean_unsigned_to_nat(3u);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 4, v_r_556_);
lean_ctor_set(v___x_706_, 3, v_r_556_);
lean_ctor_set(v___x_706_, 2, v_v_807_);
lean_ctor_set(v___x_706_, 1, v_k_806_);
lean_ctor_set(v___x_706_, 0, v___x_562_);
v___x_810_ = v___x_706_;
goto v_reusejp_809_;
}
else
{
lean_object* v_reuseFailAlloc_814_; 
v_reuseFailAlloc_814_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_814_, 0, v___x_562_);
lean_ctor_set(v_reuseFailAlloc_814_, 1, v_k_806_);
lean_ctor_set(v_reuseFailAlloc_814_, 2, v_v_807_);
lean_ctor_set(v_reuseFailAlloc_814_, 3, v_r_556_);
lean_ctor_set(v_reuseFailAlloc_814_, 4, v_r_556_);
v___x_810_ = v_reuseFailAlloc_814_;
goto v_reusejp_809_;
}
v_reusejp_809_:
{
lean_object* v___x_812_; 
if (v_isShared_794_ == 0)
{
lean_ctor_set(v___x_793_, 4, v___x_810_);
lean_ctor_set(v___x_793_, 0, v___x_808_);
v___x_812_ = v___x_793_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v___x_808_);
lean_ctor_set(v_reuseFailAlloc_813_, 1, v_k_553_);
lean_ctor_set(v_reuseFailAlloc_813_, 2, v_v_554_);
lean_ctor_set(v_reuseFailAlloc_813_, 3, v_l_555_);
lean_ctor_set(v_reuseFailAlloc_813_, 4, v___x_810_);
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
}
else
{
if (lean_obj_tag(v_r_556_) == 0)
{
lean_object* v___x_822_; uint8_t v_isShared_823_; uint8_t v_isSharedCheck_845_; 
lean_inc(v_l_555_);
lean_inc(v_v_554_);
lean_inc(v_k_553_);
v_isSharedCheck_845_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_845_ == 0)
{
lean_object* v_unused_846_; lean_object* v_unused_847_; lean_object* v_unused_848_; lean_object* v_unused_849_; lean_object* v_unused_850_; 
v_unused_846_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_846_);
v_unused_847_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_847_);
v_unused_848_ = lean_ctor_get(v_l_372_, 2);
lean_dec(v_unused_848_);
v_unused_849_ = lean_ctor_get(v_l_372_, 1);
lean_dec(v_unused_849_);
v_unused_850_ = lean_ctor_get(v_l_372_, 0);
lean_dec(v_unused_850_);
v___x_822_ = v_l_372_;
v_isShared_823_ = v_isSharedCheck_845_;
goto v_resetjp_821_;
}
else
{
lean_dec(v_l_372_);
v___x_822_ = lean_box(0);
v_isShared_823_ = v_isSharedCheck_845_;
goto v_resetjp_821_;
}
v_resetjp_821_:
{
lean_object* v_k_824_; lean_object* v_v_825_; lean_object* v_k_826_; lean_object* v_v_827_; lean_object* v___x_829_; uint8_t v_isShared_830_; uint8_t v_isSharedCheck_841_; 
v_k_824_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_k_824_);
v_v_825_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_v_825_);
lean_dec_ref(v___x_708_);
v_k_826_ = lean_ctor_get(v_r_556_, 1);
v_v_827_ = lean_ctor_get(v_r_556_, 2);
v_isSharedCheck_841_ = !lean_is_exclusive(v_r_556_);
if (v_isSharedCheck_841_ == 0)
{
lean_object* v_unused_842_; lean_object* v_unused_843_; lean_object* v_unused_844_; 
v_unused_842_ = lean_ctor_get(v_r_556_, 4);
lean_dec(v_unused_842_);
v_unused_843_ = lean_ctor_get(v_r_556_, 3);
lean_dec(v_unused_843_);
v_unused_844_ = lean_ctor_get(v_r_556_, 0);
lean_dec(v_unused_844_);
v___x_829_ = v_r_556_;
v_isShared_830_ = v_isSharedCheck_841_;
goto v_resetjp_828_;
}
else
{
lean_inc(v_v_827_);
lean_inc(v_k_826_);
lean_dec(v_r_556_);
v___x_829_ = lean_box(0);
v_isShared_830_ = v_isSharedCheck_841_;
goto v_resetjp_828_;
}
v_resetjp_828_:
{
lean_object* v___x_831_; lean_object* v___x_833_; 
v___x_831_ = lean_unsigned_to_nat(3u);
if (v_isShared_830_ == 0)
{
lean_ctor_set(v___x_829_, 4, v_l_555_);
lean_ctor_set(v___x_829_, 3, v_l_555_);
lean_ctor_set(v___x_829_, 2, v_v_554_);
lean_ctor_set(v___x_829_, 1, v_k_553_);
lean_ctor_set(v___x_829_, 0, v___x_562_);
v___x_833_ = v___x_829_;
goto v_reusejp_832_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v___x_562_);
lean_ctor_set(v_reuseFailAlloc_840_, 1, v_k_553_);
lean_ctor_set(v_reuseFailAlloc_840_, 2, v_v_554_);
lean_ctor_set(v_reuseFailAlloc_840_, 3, v_l_555_);
lean_ctor_set(v_reuseFailAlloc_840_, 4, v_l_555_);
v___x_833_ = v_reuseFailAlloc_840_;
goto v_reusejp_832_;
}
v_reusejp_832_:
{
lean_object* v___x_835_; 
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 4, v_l_555_);
lean_ctor_set(v___x_706_, 3, v_l_555_);
lean_ctor_set(v___x_706_, 2, v_v_825_);
lean_ctor_set(v___x_706_, 1, v_k_824_);
lean_ctor_set(v___x_706_, 0, v___x_562_);
v___x_835_ = v___x_706_;
goto v_reusejp_834_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v___x_562_);
lean_ctor_set(v_reuseFailAlloc_839_, 1, v_k_824_);
lean_ctor_set(v_reuseFailAlloc_839_, 2, v_v_825_);
lean_ctor_set(v_reuseFailAlloc_839_, 3, v_l_555_);
lean_ctor_set(v_reuseFailAlloc_839_, 4, v_l_555_);
v___x_835_ = v_reuseFailAlloc_839_;
goto v_reusejp_834_;
}
v_reusejp_834_:
{
lean_object* v___x_837_; 
if (v_isShared_823_ == 0)
{
lean_ctor_set(v___x_822_, 4, v___x_835_);
lean_ctor_set(v___x_822_, 3, v___x_833_);
lean_ctor_set(v___x_822_, 2, v_v_827_);
lean_ctor_set(v___x_822_, 1, v_k_826_);
lean_ctor_set(v___x_822_, 0, v___x_831_);
v___x_837_ = v___x_822_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_838_; 
v_reuseFailAlloc_838_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_838_, 0, v___x_831_);
lean_ctor_set(v_reuseFailAlloc_838_, 1, v_k_826_);
lean_ctor_set(v_reuseFailAlloc_838_, 2, v_v_827_);
lean_ctor_set(v_reuseFailAlloc_838_, 3, v___x_833_);
lean_ctor_set(v_reuseFailAlloc_838_, 4, v___x_835_);
v___x_837_ = v_reuseFailAlloc_838_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
return v___x_837_;
}
}
}
}
}
}
else
{
lean_object* v_k_851_; lean_object* v_v_852_; lean_object* v___x_853_; lean_object* v___x_855_; 
v_k_851_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_k_851_);
v_v_852_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_v_852_);
lean_dec_ref(v___x_708_);
v___x_853_ = lean_unsigned_to_nat(2u);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 4, v_r_556_);
lean_ctor_set(v___x_706_, 3, v_l_372_);
lean_ctor_set(v___x_706_, 2, v_v_852_);
lean_ctor_set(v___x_706_, 1, v_k_851_);
lean_ctor_set(v___x_706_, 0, v___x_853_);
v___x_855_ = v___x_706_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v___x_853_);
lean_ctor_set(v_reuseFailAlloc_856_, 1, v_k_851_);
lean_ctor_set(v_reuseFailAlloc_856_, 2, v_v_852_);
lean_ctor_set(v_reuseFailAlloc_856_, 3, v_l_372_);
lean_ctor_set(v_reuseFailAlloc_856_, 4, v_r_556_);
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
}
}
else
{
return v_l_372_;
}
}
else
{
return v_r_373_;
}
}
default: 
{
lean_object* v_impl_863_; lean_object* v___x_864_; 
v_impl_863_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg(v_k_368_, v_r_373_);
v___x_864_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_863_) == 0)
{
if (lean_obj_tag(v_l_372_) == 0)
{
lean_object* v_size_865_; lean_object* v_size_866_; lean_object* v_k_867_; lean_object* v_v_868_; lean_object* v_l_869_; lean_object* v_r_870_; lean_object* v___x_871_; lean_object* v___x_872_; uint8_t v___x_873_; 
v_size_865_ = lean_ctor_get(v_impl_863_, 0);
lean_inc(v_size_865_);
v_size_866_ = lean_ctor_get(v_l_372_, 0);
v_k_867_ = lean_ctor_get(v_l_372_, 1);
v_v_868_ = lean_ctor_get(v_l_372_, 2);
v_l_869_ = lean_ctor_get(v_l_372_, 3);
v_r_870_ = lean_ctor_get(v_l_372_, 4);
lean_inc(v_r_870_);
v___x_871_ = lean_unsigned_to_nat(3u);
v___x_872_ = lean_nat_mul(v___x_871_, v_size_865_);
v___x_873_ = lean_nat_dec_lt(v___x_872_, v_size_866_);
lean_dec(v___x_872_);
if (v___x_873_ == 0)
{
lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_877_; 
lean_dec(v_r_870_);
v___x_874_ = lean_nat_add(v___x_864_, v_size_866_);
v___x_875_ = lean_nat_add(v___x_874_, v_size_865_);
lean_dec(v_size_865_);
lean_dec(v___x_874_);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_impl_863_);
lean_ctor_set(v___x_375_, 0, v___x_875_);
v___x_877_ = v___x_375_;
goto v_reusejp_876_;
}
else
{
lean_object* v_reuseFailAlloc_878_; 
v_reuseFailAlloc_878_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_878_, 0, v___x_875_);
lean_ctor_set(v_reuseFailAlloc_878_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_878_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_878_, 3, v_l_372_);
lean_ctor_set(v_reuseFailAlloc_878_, 4, v_impl_863_);
v___x_877_ = v_reuseFailAlloc_878_;
goto v_reusejp_876_;
}
v_reusejp_876_:
{
return v___x_877_;
}
}
else
{
lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_944_; 
lean_inc(v_l_869_);
lean_inc(v_v_868_);
lean_inc(v_k_867_);
lean_inc(v_size_866_);
v_isSharedCheck_944_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_944_ == 0)
{
lean_object* v_unused_945_; lean_object* v_unused_946_; lean_object* v_unused_947_; lean_object* v_unused_948_; lean_object* v_unused_949_; 
v_unused_945_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_945_);
v_unused_946_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_946_);
v_unused_947_ = lean_ctor_get(v_l_372_, 2);
lean_dec(v_unused_947_);
v_unused_948_ = lean_ctor_get(v_l_372_, 1);
lean_dec(v_unused_948_);
v_unused_949_ = lean_ctor_get(v_l_372_, 0);
lean_dec(v_unused_949_);
v___x_880_ = v_l_372_;
v_isShared_881_ = v_isSharedCheck_944_;
goto v_resetjp_879_;
}
else
{
lean_dec(v_l_372_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_944_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v_size_882_; lean_object* v_size_883_; lean_object* v_k_884_; lean_object* v_v_885_; lean_object* v_l_886_; lean_object* v_r_887_; lean_object* v___x_888_; lean_object* v___x_889_; uint8_t v___x_890_; 
v_size_882_ = lean_ctor_get(v_l_869_, 0);
v_size_883_ = lean_ctor_get(v_r_870_, 0);
v_k_884_ = lean_ctor_get(v_r_870_, 1);
v_v_885_ = lean_ctor_get(v_r_870_, 2);
v_l_886_ = lean_ctor_get(v_r_870_, 3);
v_r_887_ = lean_ctor_get(v_r_870_, 4);
v___x_888_ = lean_unsigned_to_nat(2u);
v___x_889_ = lean_nat_mul(v___x_888_, v_size_882_);
v___x_890_ = lean_nat_dec_lt(v_size_883_, v___x_889_);
lean_dec(v___x_889_);
if (v___x_890_ == 0)
{
lean_object* v___x_892_; uint8_t v_isShared_893_; uint8_t v_isSharedCheck_919_; 
lean_inc(v_r_887_);
lean_inc(v_l_886_);
lean_inc(v_v_885_);
lean_inc(v_k_884_);
v_isSharedCheck_919_ = !lean_is_exclusive(v_r_870_);
if (v_isSharedCheck_919_ == 0)
{
lean_object* v_unused_920_; lean_object* v_unused_921_; lean_object* v_unused_922_; lean_object* v_unused_923_; lean_object* v_unused_924_; 
v_unused_920_ = lean_ctor_get(v_r_870_, 4);
lean_dec(v_unused_920_);
v_unused_921_ = lean_ctor_get(v_r_870_, 3);
lean_dec(v_unused_921_);
v_unused_922_ = lean_ctor_get(v_r_870_, 2);
lean_dec(v_unused_922_);
v_unused_923_ = lean_ctor_get(v_r_870_, 1);
lean_dec(v_unused_923_);
v_unused_924_ = lean_ctor_get(v_r_870_, 0);
lean_dec(v_unused_924_);
v___x_892_ = v_r_870_;
v_isShared_893_ = v_isSharedCheck_919_;
goto v_resetjp_891_;
}
else
{
lean_dec(v_r_870_);
v___x_892_ = lean_box(0);
v_isShared_893_ = v_isSharedCheck_919_;
goto v_resetjp_891_;
}
v_resetjp_891_:
{
lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___y_897_; lean_object* v___y_898_; lean_object* v___y_899_; lean_object* v___x_907_; lean_object* v___y_909_; 
v___x_894_ = lean_nat_add(v___x_864_, v_size_866_);
lean_dec(v_size_866_);
v___x_895_ = lean_nat_add(v___x_894_, v_size_865_);
lean_dec(v___x_894_);
v___x_907_ = lean_nat_add(v___x_864_, v_size_882_);
if (lean_obj_tag(v_l_886_) == 0)
{
lean_object* v_size_917_; 
v_size_917_ = lean_ctor_get(v_l_886_, 0);
lean_inc(v_size_917_);
v___y_909_ = v_size_917_;
goto v___jp_908_;
}
else
{
lean_object* v___x_918_; 
v___x_918_ = lean_unsigned_to_nat(0u);
v___y_909_ = v___x_918_;
goto v___jp_908_;
}
v___jp_896_:
{
lean_object* v___x_900_; lean_object* v___x_902_; 
v___x_900_ = lean_nat_add(v___y_897_, v___y_899_);
lean_dec(v___y_899_);
lean_dec(v___y_897_);
if (v_isShared_893_ == 0)
{
lean_ctor_set(v___x_892_, 4, v_impl_863_);
lean_ctor_set(v___x_892_, 3, v_r_887_);
lean_ctor_set(v___x_892_, 2, v_v_371_);
lean_ctor_set(v___x_892_, 1, v_k_370_);
lean_ctor_set(v___x_892_, 0, v___x_900_);
v___x_902_ = v___x_892_;
goto v_reusejp_901_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v___x_900_);
lean_ctor_set(v_reuseFailAlloc_906_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_906_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_906_, 3, v_r_887_);
lean_ctor_set(v_reuseFailAlloc_906_, 4, v_impl_863_);
v___x_902_ = v_reuseFailAlloc_906_;
goto v_reusejp_901_;
}
v_reusejp_901_:
{
lean_object* v___x_904_; 
if (v_isShared_881_ == 0)
{
lean_ctor_set(v___x_880_, 4, v___x_902_);
lean_ctor_set(v___x_880_, 3, v___y_898_);
lean_ctor_set(v___x_880_, 2, v_v_885_);
lean_ctor_set(v___x_880_, 1, v_k_884_);
lean_ctor_set(v___x_880_, 0, v___x_895_);
v___x_904_ = v___x_880_;
goto v_reusejp_903_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v___x_895_);
lean_ctor_set(v_reuseFailAlloc_905_, 1, v_k_884_);
lean_ctor_set(v_reuseFailAlloc_905_, 2, v_v_885_);
lean_ctor_set(v_reuseFailAlloc_905_, 3, v___y_898_);
lean_ctor_set(v_reuseFailAlloc_905_, 4, v___x_902_);
v___x_904_ = v_reuseFailAlloc_905_;
goto v_reusejp_903_;
}
v_reusejp_903_:
{
return v___x_904_;
}
}
}
v___jp_908_:
{
lean_object* v___x_910_; lean_object* v___x_912_; 
v___x_910_ = lean_nat_add(v___x_907_, v___y_909_);
lean_dec(v___y_909_);
lean_dec(v___x_907_);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_l_886_);
lean_ctor_set(v___x_375_, 3, v_l_869_);
lean_ctor_set(v___x_375_, 2, v_v_868_);
lean_ctor_set(v___x_375_, 1, v_k_867_);
lean_ctor_set(v___x_375_, 0, v___x_910_);
v___x_912_ = v___x_375_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_916_; 
v_reuseFailAlloc_916_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_916_, 0, v___x_910_);
lean_ctor_set(v_reuseFailAlloc_916_, 1, v_k_867_);
lean_ctor_set(v_reuseFailAlloc_916_, 2, v_v_868_);
lean_ctor_set(v_reuseFailAlloc_916_, 3, v_l_869_);
lean_ctor_set(v_reuseFailAlloc_916_, 4, v_l_886_);
v___x_912_ = v_reuseFailAlloc_916_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
lean_object* v___x_913_; 
v___x_913_ = lean_nat_add(v___x_864_, v_size_865_);
lean_dec(v_size_865_);
if (lean_obj_tag(v_r_887_) == 0)
{
lean_object* v_size_914_; 
v_size_914_ = lean_ctor_get(v_r_887_, 0);
lean_inc(v_size_914_);
v___y_897_ = v___x_913_;
v___y_898_ = v___x_912_;
v___y_899_ = v_size_914_;
goto v___jp_896_;
}
else
{
lean_object* v___x_915_; 
v___x_915_ = lean_unsigned_to_nat(0u);
v___y_897_ = v___x_913_;
v___y_898_ = v___x_912_;
v___y_899_ = v___x_915_;
goto v___jp_896_;
}
}
}
}
}
else
{
lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_930_; 
lean_del_object(v___x_375_);
v___x_925_ = lean_nat_add(v___x_864_, v_size_866_);
lean_dec(v_size_866_);
v___x_926_ = lean_nat_add(v___x_925_, v_size_865_);
lean_dec(v___x_925_);
v___x_927_ = lean_nat_add(v___x_864_, v_size_865_);
lean_dec(v_size_865_);
v___x_928_ = lean_nat_add(v___x_927_, v_size_883_);
lean_dec(v___x_927_);
lean_inc_ref(v_impl_863_);
if (v_isShared_881_ == 0)
{
lean_ctor_set(v___x_880_, 4, v_impl_863_);
lean_ctor_set(v___x_880_, 3, v_r_870_);
lean_ctor_set(v___x_880_, 2, v_v_371_);
lean_ctor_set(v___x_880_, 1, v_k_370_);
lean_ctor_set(v___x_880_, 0, v___x_928_);
v___x_930_ = v___x_880_;
goto v_reusejp_929_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v___x_928_);
lean_ctor_set(v_reuseFailAlloc_943_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_943_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_943_, 3, v_r_870_);
lean_ctor_set(v_reuseFailAlloc_943_, 4, v_impl_863_);
v___x_930_ = v_reuseFailAlloc_943_;
goto v_reusejp_929_;
}
v_reusejp_929_:
{
lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_937_; 
v_isSharedCheck_937_ = !lean_is_exclusive(v_impl_863_);
if (v_isSharedCheck_937_ == 0)
{
lean_object* v_unused_938_; lean_object* v_unused_939_; lean_object* v_unused_940_; lean_object* v_unused_941_; lean_object* v_unused_942_; 
v_unused_938_ = lean_ctor_get(v_impl_863_, 4);
lean_dec(v_unused_938_);
v_unused_939_ = lean_ctor_get(v_impl_863_, 3);
lean_dec(v_unused_939_);
v_unused_940_ = lean_ctor_get(v_impl_863_, 2);
lean_dec(v_unused_940_);
v_unused_941_ = lean_ctor_get(v_impl_863_, 1);
lean_dec(v_unused_941_);
v_unused_942_ = lean_ctor_get(v_impl_863_, 0);
lean_dec(v_unused_942_);
v___x_932_ = v_impl_863_;
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
else
{
lean_dec(v_impl_863_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_935_; 
if (v_isShared_933_ == 0)
{
lean_ctor_set(v___x_932_, 4, v___x_930_);
lean_ctor_set(v___x_932_, 3, v_l_869_);
lean_ctor_set(v___x_932_, 2, v_v_868_);
lean_ctor_set(v___x_932_, 1, v_k_867_);
lean_ctor_set(v___x_932_, 0, v___x_926_);
v___x_935_ = v___x_932_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v___x_926_);
lean_ctor_set(v_reuseFailAlloc_936_, 1, v_k_867_);
lean_ctor_set(v_reuseFailAlloc_936_, 2, v_v_868_);
lean_ctor_set(v_reuseFailAlloc_936_, 3, v_l_869_);
lean_ctor_set(v_reuseFailAlloc_936_, 4, v___x_930_);
v___x_935_ = v_reuseFailAlloc_936_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
return v___x_935_;
}
}
}
}
}
}
}
else
{
lean_object* v_size_950_; lean_object* v___x_951_; lean_object* v___x_953_; 
v_size_950_ = lean_ctor_get(v_impl_863_, 0);
lean_inc(v_size_950_);
v___x_951_ = lean_nat_add(v___x_864_, v_size_950_);
lean_dec(v_size_950_);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_impl_863_);
lean_ctor_set(v___x_375_, 0, v___x_951_);
v___x_953_ = v___x_375_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v___x_951_);
lean_ctor_set(v_reuseFailAlloc_954_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_954_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_954_, 3, v_l_372_);
lean_ctor_set(v_reuseFailAlloc_954_, 4, v_impl_863_);
v___x_953_ = v_reuseFailAlloc_954_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
return v___x_953_;
}
}
}
else
{
if (lean_obj_tag(v_l_372_) == 0)
{
lean_object* v_l_955_; 
v_l_955_ = lean_ctor_get(v_l_372_, 3);
if (lean_obj_tag(v_l_955_) == 0)
{
lean_object* v_r_956_; 
lean_inc_ref(v_l_955_);
v_r_956_ = lean_ctor_get(v_l_372_, 4);
lean_inc(v_r_956_);
if (lean_obj_tag(v_r_956_) == 0)
{
lean_object* v_size_957_; lean_object* v_k_958_; lean_object* v_v_959_; lean_object* v___x_961_; uint8_t v_isShared_962_; uint8_t v_isSharedCheck_972_; 
v_size_957_ = lean_ctor_get(v_l_372_, 0);
v_k_958_ = lean_ctor_get(v_l_372_, 1);
v_v_959_ = lean_ctor_get(v_l_372_, 2);
v_isSharedCheck_972_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_972_ == 0)
{
lean_object* v_unused_973_; lean_object* v_unused_974_; 
v_unused_973_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_973_);
v_unused_974_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_974_);
v___x_961_ = v_l_372_;
v_isShared_962_ = v_isSharedCheck_972_;
goto v_resetjp_960_;
}
else
{
lean_inc(v_v_959_);
lean_inc(v_k_958_);
lean_inc(v_size_957_);
lean_dec(v_l_372_);
v___x_961_ = lean_box(0);
v_isShared_962_ = v_isSharedCheck_972_;
goto v_resetjp_960_;
}
v_resetjp_960_:
{
lean_object* v_size_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_967_; 
v_size_963_ = lean_ctor_get(v_r_956_, 0);
v___x_964_ = lean_nat_add(v___x_864_, v_size_957_);
lean_dec(v_size_957_);
v___x_965_ = lean_nat_add(v___x_864_, v_size_963_);
if (v_isShared_962_ == 0)
{
lean_ctor_set(v___x_961_, 4, v_impl_863_);
lean_ctor_set(v___x_961_, 3, v_r_956_);
lean_ctor_set(v___x_961_, 2, v_v_371_);
lean_ctor_set(v___x_961_, 1, v_k_370_);
lean_ctor_set(v___x_961_, 0, v___x_965_);
v___x_967_ = v___x_961_;
goto v_reusejp_966_;
}
else
{
lean_object* v_reuseFailAlloc_971_; 
v_reuseFailAlloc_971_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_971_, 0, v___x_965_);
lean_ctor_set(v_reuseFailAlloc_971_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_971_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_971_, 3, v_r_956_);
lean_ctor_set(v_reuseFailAlloc_971_, 4, v_impl_863_);
v___x_967_ = v_reuseFailAlloc_971_;
goto v_reusejp_966_;
}
v_reusejp_966_:
{
lean_object* v___x_969_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v___x_967_);
lean_ctor_set(v___x_375_, 3, v_l_955_);
lean_ctor_set(v___x_375_, 2, v_v_959_);
lean_ctor_set(v___x_375_, 1, v_k_958_);
lean_ctor_set(v___x_375_, 0, v___x_964_);
v___x_969_ = v___x_375_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v___x_964_);
lean_ctor_set(v_reuseFailAlloc_970_, 1, v_k_958_);
lean_ctor_set(v_reuseFailAlloc_970_, 2, v_v_959_);
lean_ctor_set(v_reuseFailAlloc_970_, 3, v_l_955_);
lean_ctor_set(v_reuseFailAlloc_970_, 4, v___x_967_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
}
}
else
{
lean_object* v_k_975_; lean_object* v_v_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_987_; 
v_k_975_ = lean_ctor_get(v_l_372_, 1);
v_v_976_ = lean_ctor_get(v_l_372_, 2);
v_isSharedCheck_987_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_987_ == 0)
{
lean_object* v_unused_988_; lean_object* v_unused_989_; lean_object* v_unused_990_; 
v_unused_988_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_988_);
v_unused_989_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_989_);
v_unused_990_ = lean_ctor_get(v_l_372_, 0);
lean_dec(v_unused_990_);
v___x_978_ = v_l_372_;
v_isShared_979_ = v_isSharedCheck_987_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_v_976_);
lean_inc(v_k_975_);
lean_dec(v_l_372_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_987_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_980_; lean_object* v___x_982_; 
v___x_980_ = lean_unsigned_to_nat(3u);
if (v_isShared_979_ == 0)
{
lean_ctor_set(v___x_978_, 3, v_r_956_);
lean_ctor_set(v___x_978_, 2, v_v_371_);
lean_ctor_set(v___x_978_, 1, v_k_370_);
lean_ctor_set(v___x_978_, 0, v___x_864_);
v___x_982_ = v___x_978_;
goto v_reusejp_981_;
}
else
{
lean_object* v_reuseFailAlloc_986_; 
v_reuseFailAlloc_986_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_986_, 0, v___x_864_);
lean_ctor_set(v_reuseFailAlloc_986_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_986_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_986_, 3, v_r_956_);
lean_ctor_set(v_reuseFailAlloc_986_, 4, v_r_956_);
v___x_982_ = v_reuseFailAlloc_986_;
goto v_reusejp_981_;
}
v_reusejp_981_:
{
lean_object* v___x_984_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v___x_982_);
lean_ctor_set(v___x_375_, 3, v_l_955_);
lean_ctor_set(v___x_375_, 2, v_v_976_);
lean_ctor_set(v___x_375_, 1, v_k_975_);
lean_ctor_set(v___x_375_, 0, v___x_980_);
v___x_984_ = v___x_375_;
goto v_reusejp_983_;
}
else
{
lean_object* v_reuseFailAlloc_985_; 
v_reuseFailAlloc_985_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_985_, 0, v___x_980_);
lean_ctor_set(v_reuseFailAlloc_985_, 1, v_k_975_);
lean_ctor_set(v_reuseFailAlloc_985_, 2, v_v_976_);
lean_ctor_set(v_reuseFailAlloc_985_, 3, v_l_955_);
lean_ctor_set(v_reuseFailAlloc_985_, 4, v___x_982_);
v___x_984_ = v_reuseFailAlloc_985_;
goto v_reusejp_983_;
}
v_reusejp_983_:
{
return v___x_984_;
}
}
}
}
}
else
{
lean_object* v_r_991_; 
v_r_991_ = lean_ctor_get(v_l_372_, 4);
lean_inc(v_r_991_);
if (lean_obj_tag(v_r_991_) == 0)
{
lean_object* v_k_992_; lean_object* v_v_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1016_; 
lean_inc(v_l_955_);
v_k_992_ = lean_ctor_get(v_l_372_, 1);
v_v_993_ = lean_ctor_get(v_l_372_, 2);
v_isSharedCheck_1016_ = !lean_is_exclusive(v_l_372_);
if (v_isSharedCheck_1016_ == 0)
{
lean_object* v_unused_1017_; lean_object* v_unused_1018_; lean_object* v_unused_1019_; 
v_unused_1017_ = lean_ctor_get(v_l_372_, 4);
lean_dec(v_unused_1017_);
v_unused_1018_ = lean_ctor_get(v_l_372_, 3);
lean_dec(v_unused_1018_);
v_unused_1019_ = lean_ctor_get(v_l_372_, 0);
lean_dec(v_unused_1019_);
v___x_995_ = v_l_372_;
v_isShared_996_ = v_isSharedCheck_1016_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_v_993_);
lean_inc(v_k_992_);
lean_dec(v_l_372_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1016_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v_k_997_; lean_object* v_v_998_; lean_object* v___x_1000_; uint8_t v_isShared_1001_; uint8_t v_isSharedCheck_1012_; 
v_k_997_ = lean_ctor_get(v_r_991_, 1);
v_v_998_ = lean_ctor_get(v_r_991_, 2);
v_isSharedCheck_1012_ = !lean_is_exclusive(v_r_991_);
if (v_isSharedCheck_1012_ == 0)
{
lean_object* v_unused_1013_; lean_object* v_unused_1014_; lean_object* v_unused_1015_; 
v_unused_1013_ = lean_ctor_get(v_r_991_, 4);
lean_dec(v_unused_1013_);
v_unused_1014_ = lean_ctor_get(v_r_991_, 3);
lean_dec(v_unused_1014_);
v_unused_1015_ = lean_ctor_get(v_r_991_, 0);
lean_dec(v_unused_1015_);
v___x_1000_ = v_r_991_;
v_isShared_1001_ = v_isSharedCheck_1012_;
goto v_resetjp_999_;
}
else
{
lean_inc(v_v_998_);
lean_inc(v_k_997_);
lean_dec(v_r_991_);
v___x_1000_ = lean_box(0);
v_isShared_1001_ = v_isSharedCheck_1012_;
goto v_resetjp_999_;
}
v_resetjp_999_:
{
lean_object* v___x_1002_; lean_object* v___x_1004_; 
v___x_1002_ = lean_unsigned_to_nat(3u);
if (v_isShared_1001_ == 0)
{
lean_ctor_set(v___x_1000_, 4, v_l_955_);
lean_ctor_set(v___x_1000_, 3, v_l_955_);
lean_ctor_set(v___x_1000_, 2, v_v_993_);
lean_ctor_set(v___x_1000_, 1, v_k_992_);
lean_ctor_set(v___x_1000_, 0, v___x_864_);
v___x_1004_ = v___x_1000_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v___x_864_);
lean_ctor_set(v_reuseFailAlloc_1011_, 1, v_k_992_);
lean_ctor_set(v_reuseFailAlloc_1011_, 2, v_v_993_);
lean_ctor_set(v_reuseFailAlloc_1011_, 3, v_l_955_);
lean_ctor_set(v_reuseFailAlloc_1011_, 4, v_l_955_);
v___x_1004_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
lean_object* v___x_1006_; 
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 4, v_l_955_);
lean_ctor_set(v___x_995_, 2, v_v_371_);
lean_ctor_set(v___x_995_, 1, v_k_370_);
lean_ctor_set(v___x_995_, 0, v___x_864_);
v___x_1006_ = v___x_995_;
goto v_reusejp_1005_;
}
else
{
lean_object* v_reuseFailAlloc_1010_; 
v_reuseFailAlloc_1010_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1010_, 0, v___x_864_);
lean_ctor_set(v_reuseFailAlloc_1010_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_1010_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_1010_, 3, v_l_955_);
lean_ctor_set(v_reuseFailAlloc_1010_, 4, v_l_955_);
v___x_1006_ = v_reuseFailAlloc_1010_;
goto v_reusejp_1005_;
}
v_reusejp_1005_:
{
lean_object* v___x_1008_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v___x_1006_);
lean_ctor_set(v___x_375_, 3, v___x_1004_);
lean_ctor_set(v___x_375_, 2, v_v_998_);
lean_ctor_set(v___x_375_, 1, v_k_997_);
lean_ctor_set(v___x_375_, 0, v___x_1002_);
v___x_1008_ = v___x_375_;
goto v_reusejp_1007_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v___x_1002_);
lean_ctor_set(v_reuseFailAlloc_1009_, 1, v_k_997_);
lean_ctor_set(v_reuseFailAlloc_1009_, 2, v_v_998_);
lean_ctor_set(v_reuseFailAlloc_1009_, 3, v___x_1004_);
lean_ctor_set(v_reuseFailAlloc_1009_, 4, v___x_1006_);
v___x_1008_ = v_reuseFailAlloc_1009_;
goto v_reusejp_1007_;
}
v_reusejp_1007_:
{
return v___x_1008_;
}
}
}
}
}
}
else
{
lean_object* v___x_1020_; lean_object* v___x_1022_; 
v___x_1020_ = lean_unsigned_to_nat(2u);
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_r_991_);
lean_ctor_set(v___x_375_, 0, v___x_1020_);
v___x_1022_ = v___x_375_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v___x_1020_);
lean_ctor_set(v_reuseFailAlloc_1023_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_1023_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_1023_, 3, v_l_372_);
lean_ctor_set(v_reuseFailAlloc_1023_, 4, v_r_991_);
v___x_1022_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
return v___x_1022_;
}
}
}
}
else
{
lean_object* v___x_1025_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 4, v_l_372_);
lean_ctor_set(v___x_375_, 0, v___x_864_);
v___x_1025_ = v___x_375_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1026_; 
v_reuseFailAlloc_1026_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1026_, 0, v___x_864_);
lean_ctor_set(v_reuseFailAlloc_1026_, 1, v_k_370_);
lean_ctor_set(v_reuseFailAlloc_1026_, 2, v_v_371_);
lean_ctor_set(v_reuseFailAlloc_1026_, 3, v_l_372_);
lean_ctor_set(v_reuseFailAlloc_1026_, 4, v_l_372_);
v___x_1025_ = v_reuseFailAlloc_1026_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
return v___x_1025_;
}
}
}
}
}
}
}
else
{
return v_t_369_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg___boxed(lean_object* v_k_1029_, lean_object* v_t_1030_){
_start:
{
lean_object* v_res_1031_; 
v_res_1031_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg(v_k_1029_, v_t_1030_);
lean_dec(v_k_1029_);
return v_res_1031_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__22(lean_object* v_opts_1032_, lean_object* v_opt_1033_){
_start:
{
lean_object* v_name_1034_; lean_object* v_defValue_1035_; lean_object* v_map_1036_; lean_object* v___x_1037_; 
v_name_1034_ = lean_ctor_get(v_opt_1033_, 0);
v_defValue_1035_ = lean_ctor_get(v_opt_1033_, 1);
v_map_1036_ = lean_ctor_get(v_opts_1032_, 0);
v___x_1037_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1036_, v_name_1034_);
if (lean_obj_tag(v___x_1037_) == 0)
{
uint8_t v___x_1038_; 
v___x_1038_ = lean_unbox(v_defValue_1035_);
return v___x_1038_;
}
else
{
lean_object* v_val_1039_; 
v_val_1039_ = lean_ctor_get(v___x_1037_, 0);
lean_inc(v_val_1039_);
lean_dec_ref_known(v___x_1037_, 1);
if (lean_obj_tag(v_val_1039_) == 1)
{
uint8_t v_v_1040_; 
v_v_1040_ = lean_ctor_get_uint8(v_val_1039_, 0);
lean_dec_ref_known(v_val_1039_, 0);
return v_v_1040_;
}
else
{
uint8_t v___x_1041_; 
lean_dec(v_val_1039_);
v___x_1041_ = lean_unbox(v_defValue_1035_);
return v___x_1041_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__22___boxed(lean_object* v_opts_1042_, lean_object* v_opt_1043_){
_start:
{
uint8_t v_res_1044_; lean_object* v_r_1045_; 
v_res_1044_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__22(v_opts_1042_, v_opt_1043_);
lean_dec_ref(v_opt_1043_);
lean_dec_ref(v_opts_1042_);
v_r_1045_ = lean_box(v_res_1044_);
return v_r_1045_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0(void){
_start:
{
lean_object* v___x_1046_; lean_object* v___x_1047_; 
v___x_1046_ = lean_box(1);
v___x_1047_ = l_Lean_MessageData_ofFormat(v___x_1046_);
return v___x_1047_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__3(void){
_start:
{
lean_object* v___x_1051_; lean_object* v___x_1052_; 
v___x_1051_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__2));
v___x_1052_ = l_Lean_MessageData_ofFormat(v___x_1051_);
return v___x_1052_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23(lean_object* v_x_1053_, lean_object* v_x_1054_){
_start:
{
if (lean_obj_tag(v_x_1054_) == 0)
{
return v_x_1053_;
}
else
{
lean_object* v_head_1055_; lean_object* v_tail_1056_; lean_object* v___x_1058_; uint8_t v_isShared_1059_; uint8_t v_isSharedCheck_1078_; 
v_head_1055_ = lean_ctor_get(v_x_1054_, 0);
v_tail_1056_ = lean_ctor_get(v_x_1054_, 1);
v_isSharedCheck_1078_ = !lean_is_exclusive(v_x_1054_);
if (v_isSharedCheck_1078_ == 0)
{
v___x_1058_ = v_x_1054_;
v_isShared_1059_ = v_isSharedCheck_1078_;
goto v_resetjp_1057_;
}
else
{
lean_inc(v_tail_1056_);
lean_inc(v_head_1055_);
lean_dec(v_x_1054_);
v___x_1058_ = lean_box(0);
v_isShared_1059_ = v_isSharedCheck_1078_;
goto v_resetjp_1057_;
}
v_resetjp_1057_:
{
lean_object* v_before_1060_; lean_object* v___x_1062_; uint8_t v_isShared_1063_; uint8_t v_isSharedCheck_1076_; 
v_before_1060_ = lean_ctor_get(v_head_1055_, 0);
v_isSharedCheck_1076_ = !lean_is_exclusive(v_head_1055_);
if (v_isSharedCheck_1076_ == 0)
{
lean_object* v_unused_1077_; 
v_unused_1077_ = lean_ctor_get(v_head_1055_, 1);
lean_dec(v_unused_1077_);
v___x_1062_ = v_head_1055_;
v_isShared_1063_ = v_isSharedCheck_1076_;
goto v_resetjp_1061_;
}
else
{
lean_inc(v_before_1060_);
lean_dec(v_head_1055_);
v___x_1062_ = lean_box(0);
v_isShared_1063_ = v_isSharedCheck_1076_;
goto v_resetjp_1061_;
}
v_resetjp_1061_:
{
lean_object* v___x_1064_; lean_object* v___x_1066_; 
v___x_1064_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0);
if (v_isShared_1063_ == 0)
{
lean_ctor_set_tag(v___x_1062_, 7);
lean_ctor_set(v___x_1062_, 1, v___x_1064_);
lean_ctor_set(v___x_1062_, 0, v_x_1053_);
v___x_1066_ = v___x_1062_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v_x_1053_);
lean_ctor_set(v_reuseFailAlloc_1075_, 1, v___x_1064_);
v___x_1066_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
lean_object* v___x_1067_; lean_object* v___x_1069_; 
v___x_1067_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__3);
if (v_isShared_1059_ == 0)
{
lean_ctor_set_tag(v___x_1058_, 7);
lean_ctor_set(v___x_1058_, 1, v___x_1067_);
lean_ctor_set(v___x_1058_, 0, v___x_1066_);
v___x_1069_ = v___x_1058_;
goto v_reusejp_1068_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v___x_1066_);
lean_ctor_set(v_reuseFailAlloc_1074_, 1, v___x_1067_);
v___x_1069_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1068_;
}
v_reusejp_1068_:
{
lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; 
v___x_1070_ = l_Lean_MessageData_ofSyntax(v_before_1060_);
v___x_1071_ = l_Lean_indentD(v___x_1070_);
v___x_1072_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1072_, 0, v___x_1069_);
lean_ctor_set(v___x_1072_, 1, v___x_1071_);
v_x_1053_ = v___x_1072_;
v_x_1054_ = v_tail_1056_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__2(void){
_start:
{
lean_object* v___x_1082_; lean_object* v___x_1083_; 
v___x_1082_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__1));
v___x_1083_ = l_Lean_MessageData_ofFormat(v___x_1082_);
return v___x_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg(lean_object* v_msgData_1084_, lean_object* v_macroStack_1085_, lean_object* v___y_1086_){
_start:
{
lean_object* v_options_1088_; lean_object* v___x_1089_; uint8_t v___x_1090_; 
v_options_1088_ = lean_ctor_get(v___y_1086_, 2);
v___x_1089_ = l_Lean_Elab_pp_macroStack;
v___x_1090_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__22(v_options_1088_, v___x_1089_);
if (v___x_1090_ == 0)
{
lean_object* v___x_1091_; 
lean_dec(v_macroStack_1085_);
v___x_1091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1091_, 0, v_msgData_1084_);
return v___x_1091_;
}
else
{
if (lean_obj_tag(v_macroStack_1085_) == 0)
{
lean_object* v___x_1092_; 
v___x_1092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1092_, 0, v_msgData_1084_);
return v___x_1092_;
}
else
{
lean_object* v_head_1093_; lean_object* v_after_1094_; lean_object* v___x_1096_; uint8_t v_isShared_1097_; uint8_t v_isSharedCheck_1109_; 
v_head_1093_ = lean_ctor_get(v_macroStack_1085_, 0);
lean_inc(v_head_1093_);
v_after_1094_ = lean_ctor_get(v_head_1093_, 1);
v_isSharedCheck_1109_ = !lean_is_exclusive(v_head_1093_);
if (v_isSharedCheck_1109_ == 0)
{
lean_object* v_unused_1110_; 
v_unused_1110_ = lean_ctor_get(v_head_1093_, 0);
lean_dec(v_unused_1110_);
v___x_1096_ = v_head_1093_;
v_isShared_1097_ = v_isSharedCheck_1109_;
goto v_resetjp_1095_;
}
else
{
lean_inc(v_after_1094_);
lean_dec(v_head_1093_);
v___x_1096_ = lean_box(0);
v_isShared_1097_ = v_isSharedCheck_1109_;
goto v_resetjp_1095_;
}
v_resetjp_1095_:
{
lean_object* v___x_1098_; lean_object* v___x_1100_; 
v___x_1098_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23___closed__0);
if (v_isShared_1097_ == 0)
{
lean_ctor_set_tag(v___x_1096_, 7);
lean_ctor_set(v___x_1096_, 1, v___x_1098_);
lean_ctor_set(v___x_1096_, 0, v_msgData_1084_);
v___x_1100_ = v___x_1096_;
goto v_reusejp_1099_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v_msgData_1084_);
lean_ctor_set(v_reuseFailAlloc_1108_, 1, v___x_1098_);
v___x_1100_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1099_;
}
v_reusejp_1099_:
{
lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v_msgData_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; 
v___x_1101_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___closed__2);
v___x_1102_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1102_, 0, v___x_1100_);
lean_ctor_set(v___x_1102_, 1, v___x_1101_);
v___x_1103_ = l_Lean_MessageData_ofSyntax(v_after_1094_);
v___x_1104_ = l_Lean_indentD(v___x_1103_);
v_msgData_1105_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1105_, 0, v___x_1102_);
lean_ctor_set(v_msgData_1105_, 1, v___x_1104_);
v___x_1106_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__23(v_msgData_1105_, v_macroStack_1085_);
v___x_1107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1107_, 0, v___x_1106_);
return v___x_1107_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg___boxed(lean_object* v_msgData_1111_, lean_object* v_macroStack_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_){
_start:
{
lean_object* v_res_1115_; 
v_res_1115_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg(v_msgData_1111_, v_macroStack_1112_, v___y_1113_);
lean_dec_ref(v___y_1113_);
return v_res_1115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___redArg(lean_object* v_msg_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_){
_start:
{
lean_object* v_ref_1124_; lean_object* v___x_1125_; lean_object* v_a_1126_; lean_object* v_macroStack_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v_a_1130_; lean_object* v___x_1132_; uint8_t v_isShared_1133_; uint8_t v_isSharedCheck_1138_; 
v_ref_1124_ = lean_ctor_get(v___y_1121_, 5);
v___x_1125_ = lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3(v_msg_1116_, v___y_1119_, v___y_1120_, v___y_1121_, v___y_1122_);
v_a_1126_ = lean_ctor_get(v___x_1125_, 0);
lean_inc(v_a_1126_);
lean_dec_ref(v___x_1125_);
v_macroStack_1127_ = lean_ctor_get(v___y_1117_, 1);
v___x_1128_ = l_Lean_Elab_getBetterRef(v_ref_1124_, v_macroStack_1127_);
lean_inc(v_macroStack_1127_);
v___x_1129_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg(v_a_1126_, v_macroStack_1127_, v___y_1121_);
v_a_1130_ = lean_ctor_get(v___x_1129_, 0);
v_isSharedCheck_1138_ = !lean_is_exclusive(v___x_1129_);
if (v_isSharedCheck_1138_ == 0)
{
v___x_1132_ = v___x_1129_;
v_isShared_1133_ = v_isSharedCheck_1138_;
goto v_resetjp_1131_;
}
else
{
lean_inc(v_a_1130_);
lean_dec(v___x_1129_);
v___x_1132_ = lean_box(0);
v_isShared_1133_ = v_isSharedCheck_1138_;
goto v_resetjp_1131_;
}
v_resetjp_1131_:
{
lean_object* v___x_1134_; lean_object* v___x_1136_; 
v___x_1134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1134_, 0, v___x_1128_);
lean_ctor_set(v___x_1134_, 1, v_a_1130_);
if (v_isShared_1133_ == 0)
{
lean_ctor_set_tag(v___x_1132_, 1);
lean_ctor_set(v___x_1132_, 0, v___x_1134_);
v___x_1136_ = v___x_1132_;
goto v_reusejp_1135_;
}
else
{
lean_object* v_reuseFailAlloc_1137_; 
v_reuseFailAlloc_1137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1137_, 0, v___x_1134_);
v___x_1136_ = v_reuseFailAlloc_1137_;
goto v_reusejp_1135_;
}
v_reusejp_1135_:
{
return v___x_1136_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___redArg___boxed(lean_object* v_msg_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v_res_1147_; 
v_res_1147_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___redArg(v_msg_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_);
lean_dec(v___y_1145_);
lean_dec_ref(v___y_1144_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
return v_res_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___redArg(lean_object* v_a_1148_, lean_object* v_x_1149_){
_start:
{
if (lean_obj_tag(v_x_1149_) == 0)
{
lean_object* v___x_1150_; 
v___x_1150_ = lean_box(0);
return v___x_1150_;
}
else
{
lean_object* v_key_1151_; lean_object* v_value_1152_; lean_object* v_tail_1153_; uint8_t v___x_1154_; 
v_key_1151_ = lean_ctor_get(v_x_1149_, 0);
v_value_1152_ = lean_ctor_get(v_x_1149_, 1);
v_tail_1153_ = lean_ctor_get(v_x_1149_, 2);
v___x_1154_ = lean_nat_dec_eq(v_key_1151_, v_a_1148_);
if (v___x_1154_ == 0)
{
v_x_1149_ = v_tail_1153_;
goto _start;
}
else
{
lean_object* v___x_1156_; 
lean_inc(v_value_1152_);
v___x_1156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1156_, 0, v_value_1152_);
return v___x_1156_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___redArg___boxed(lean_object* v_a_1157_, lean_object* v_x_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___redArg(v_a_1157_, v_x_1158_);
lean_dec(v_x_1158_);
lean_dec(v_a_1157_);
return v_res_1159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___redArg(lean_object* v_m_1160_, lean_object* v_a_1161_){
_start:
{
lean_object* v_buckets_1162_; lean_object* v___x_1163_; uint64_t v___x_1164_; uint64_t v___x_1165_; uint64_t v___x_1166_; uint64_t v_fold_1167_; uint64_t v___x_1168_; uint64_t v___x_1169_; uint64_t v___x_1170_; size_t v___x_1171_; size_t v___x_1172_; size_t v___x_1173_; size_t v___x_1174_; size_t v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; 
v_buckets_1162_ = lean_ctor_get(v_m_1160_, 1);
v___x_1163_ = lean_array_get_size(v_buckets_1162_);
v___x_1164_ = lean_uint64_of_nat(v_a_1161_);
v___x_1165_ = 32ULL;
v___x_1166_ = lean_uint64_shift_right(v___x_1164_, v___x_1165_);
v_fold_1167_ = lean_uint64_xor(v___x_1164_, v___x_1166_);
v___x_1168_ = 16ULL;
v___x_1169_ = lean_uint64_shift_right(v_fold_1167_, v___x_1168_);
v___x_1170_ = lean_uint64_xor(v_fold_1167_, v___x_1169_);
v___x_1171_ = lean_uint64_to_usize(v___x_1170_);
v___x_1172_ = lean_usize_of_nat(v___x_1163_);
v___x_1173_ = ((size_t)1ULL);
v___x_1174_ = lean_usize_sub(v___x_1172_, v___x_1173_);
v___x_1175_ = lean_usize_land(v___x_1171_, v___x_1174_);
v___x_1176_ = lean_array_uget_borrowed(v_buckets_1162_, v___x_1175_);
v___x_1177_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___redArg(v_a_1161_, v___x_1176_);
return v___x_1177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___redArg___boxed(lean_object* v_m_1178_, lean_object* v_a_1179_){
_start:
{
lean_object* v_res_1180_; 
v_res_1180_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___redArg(v_m_1178_, v_a_1179_);
lean_dec(v_a_1179_);
lean_dec_ref(v_m_1178_);
return v_res_1180_;
}
}
static lean_object* _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__3(void){
_start:
{
lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; 
v___x_1184_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__2));
v___x_1185_ = lean_unsigned_to_nat(14u);
v___x_1186_ = lean_unsigned_to_nat(22u);
v___x_1187_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__1));
v___x_1188_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__0));
v___x_1189_ = l_mkPanicMessageWithDecl(v___x_1188_, v___x_1187_, v___x_1186_, v___x_1185_, v___x_1184_);
return v___x_1189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg(lean_object* v___x_1190_, lean_object* v_a_1191_, lean_object* v_init_1192_, lean_object* v_x_1193_){
_start:
{
lean_object* v_d_1196_; 
if (lean_obj_tag(v_x_1193_) == 0)
{
lean_object* v_k_1199_; lean_object* v_l_1200_; lean_object* v_r_1201_; lean_object* v___x_1202_; lean_object* v_a_1203_; 
v_k_1199_ = lean_ctor_get(v_x_1193_, 1);
v_l_1200_ = lean_ctor_get(v_x_1193_, 3);
v_r_1201_ = lean_ctor_get(v_x_1193_, 4);
v___x_1202_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg(v___x_1190_, v_a_1191_, v_init_1192_, v_l_1200_);
v_a_1203_ = lean_ctor_get(v___x_1202_, 0);
lean_inc(v_a_1203_);
if (lean_obj_tag(v_a_1203_) == 0)
{
lean_object* v_a_1204_; 
lean_dec_ref(v___x_1202_);
v_a_1204_ = lean_ctor_get(v_a_1203_, 0);
lean_inc(v_a_1204_);
lean_dec_ref_known(v_a_1203_, 1);
v_d_1196_ = v_a_1204_;
goto v___jp_1195_;
}
else
{
lean_object* v_a_1205_; lean_object* v___y_1207_; lean_object* v___x_1215_; 
v_a_1205_ = lean_ctor_get(v_a_1203_, 0);
lean_inc(v_a_1205_);
lean_dec_ref_known(v_a_1203_, 1);
v___x_1215_ = l_Lean_Environment_getModuleIdxFor_x3f(v___x_1190_, v_k_1199_);
if (lean_obj_tag(v___x_1215_) == 0)
{
lean_object* v___x_1216_; 
v___x_1216_ = lean_box(0);
v___y_1207_ = v___x_1216_;
goto v___jp_1206_;
}
else
{
lean_object* v_val_1217_; lean_object* v___x_1218_; 
v_val_1217_ = lean_ctor_get(v___x_1215_, 0);
lean_inc(v_val_1217_);
lean_dec_ref_known(v___x_1215_, 1);
v___x_1218_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___redArg(v_a_1191_, v_val_1217_);
lean_dec(v_val_1217_);
if (lean_obj_tag(v___x_1218_) == 0)
{
lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1219_ = lean_obj_once(&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__3, &lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__3_once, _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___closed__3);
v___x_1220_ = lp_mathlib_panic___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__10(v___x_1219_);
v___y_1207_ = v___x_1220_;
goto v___jp_1206_;
}
else
{
lean_object* v_val_1221_; 
v_val_1221_ = lean_ctor_get(v___x_1218_, 0);
lean_inc(v_val_1221_);
lean_dec_ref_known(v___x_1218_, 1);
v___y_1207_ = v_val_1221_;
goto v___jp_1206_;
}
}
v___jp_1206_:
{
uint8_t v___x_1208_; 
v___x_1208_ = l_Lean_NameSet_contains(v_a_1205_, v___y_1207_);
if (v___x_1208_ == 0)
{
lean_object* v___x_1209_; 
lean_dec_ref(v___x_1202_);
v___x_1209_ = l_Lean_NameSet_insert(v_a_1205_, v___y_1207_);
v_init_1192_ = v___x_1209_;
v_x_1193_ = v_r_1201_;
goto _start;
}
else
{
lean_object* v_a_1211_; 
lean_dec(v___y_1207_);
lean_dec(v_a_1205_);
v_a_1211_ = lean_ctor_get(v___x_1202_, 0);
lean_inc(v_a_1211_);
lean_dec_ref(v___x_1202_);
if (lean_obj_tag(v_a_1211_) == 0)
{
lean_object* v_a_1212_; 
v_a_1212_ = lean_ctor_get(v_a_1211_, 0);
lean_inc(v_a_1212_);
lean_dec_ref_known(v_a_1211_, 1);
v_d_1196_ = v_a_1212_;
goto v___jp_1195_;
}
else
{
lean_object* v_a_1213_; 
v_a_1213_ = lean_ctor_get(v_a_1211_, 0);
lean_inc(v_a_1213_);
lean_dec_ref_known(v_a_1211_, 1);
v_init_1192_ = v_a_1213_;
v_x_1193_ = v_r_1201_;
goto _start;
}
}
}
}
}
else
{
lean_object* v___x_1222_; lean_object* v___x_1223_; 
v___x_1222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1222_, 0, v_init_1192_);
v___x_1223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1223_, 0, v___x_1222_);
return v___x_1223_;
}
v___jp_1195_:
{
lean_object* v___x_1197_; lean_object* v___x_1198_; 
v___x_1197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1197_, 0, v_d_1196_);
v___x_1198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1198_, 0, v___x_1197_);
return v___x_1198_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg___boxed(lean_object* v___x_1224_, lean_object* v_a_1225_, lean_object* v_init_1226_, lean_object* v_x_1227_, lean_object* v___y_1228_){
_start:
{
lean_object* v_res_1229_; 
v_res_1229_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg(v___x_1224_, v_a_1225_, v_init_1226_, v_x_1227_);
lean_dec(v_x_1227_);
lean_dec_ref(v_a_1225_);
lean_dec_ref(v___x_1224_);
return v_res_1229_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___redArg(lean_object* v_a_1230_, lean_object* v_x_1231_){
_start:
{
if (lean_obj_tag(v_x_1231_) == 0)
{
uint8_t v___x_1232_; 
v___x_1232_ = 0;
return v___x_1232_;
}
else
{
lean_object* v_key_1233_; lean_object* v_tail_1234_; uint8_t v___x_1235_; 
v_key_1233_ = lean_ctor_get(v_x_1231_, 0);
v_tail_1234_ = lean_ctor_get(v_x_1231_, 2);
v___x_1235_ = lean_level_eq(v_key_1233_, v_a_1230_);
if (v___x_1235_ == 0)
{
v_x_1231_ = v_tail_1234_;
goto _start;
}
else
{
return v___x_1235_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___redArg___boxed(lean_object* v_a_1237_, lean_object* v_x_1238_){
_start:
{
uint8_t v_res_1239_; lean_object* v_r_1240_; 
v_res_1239_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___redArg(v_a_1237_, v_x_1238_);
lean_dec(v_x_1238_);
lean_dec(v_a_1237_);
v_r_1240_ = lean_box(v_res_1239_);
return v_r_1240_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___redArg(lean_object* v_m_1241_, lean_object* v_a_1242_){
_start:
{
lean_object* v_buckets_1243_; lean_object* v___x_1244_; uint64_t v___x_1245_; uint64_t v___x_1246_; uint64_t v___x_1247_; uint64_t v_fold_1248_; uint64_t v___x_1249_; uint64_t v___x_1250_; uint64_t v___x_1251_; size_t v___x_1252_; size_t v___x_1253_; size_t v___x_1254_; size_t v___x_1255_; size_t v___x_1256_; lean_object* v___x_1257_; uint8_t v___x_1258_; 
v_buckets_1243_ = lean_ctor_get(v_m_1241_, 1);
v___x_1244_ = lean_array_get_size(v_buckets_1243_);
v___x_1245_ = l_Lean_Level_hash(v_a_1242_);
v___x_1246_ = 32ULL;
v___x_1247_ = lean_uint64_shift_right(v___x_1245_, v___x_1246_);
v_fold_1248_ = lean_uint64_xor(v___x_1245_, v___x_1247_);
v___x_1249_ = 16ULL;
v___x_1250_ = lean_uint64_shift_right(v_fold_1248_, v___x_1249_);
v___x_1251_ = lean_uint64_xor(v_fold_1248_, v___x_1250_);
v___x_1252_ = lean_uint64_to_usize(v___x_1251_);
v___x_1253_ = lean_usize_of_nat(v___x_1244_);
v___x_1254_ = ((size_t)1ULL);
v___x_1255_ = lean_usize_sub(v___x_1253_, v___x_1254_);
v___x_1256_ = lean_usize_land(v___x_1252_, v___x_1255_);
v___x_1257_ = lean_array_uget_borrowed(v_buckets_1243_, v___x_1256_);
v___x_1258_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___redArg(v_a_1242_, v___x_1257_);
return v___x_1258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___redArg___boxed(lean_object* v_m_1259_, lean_object* v_a_1260_){
_start:
{
uint8_t v_res_1261_; lean_object* v_r_1262_; 
v_res_1261_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___redArg(v_m_1259_, v_a_1260_);
lean_dec(v_a_1260_);
lean_dec_ref(v_m_1259_);
v_r_1262_ = lean_box(v_res_1261_);
return v_r_1262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__2(lean_object* v___x_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_){
_start:
{
if (lean_obj_tag(v_a_1264_) == 0)
{
lean_object* v___x_1266_; 
v___x_1266_ = l_List_reverse___redArg(v_a_1265_);
return v___x_1266_;
}
else
{
lean_object* v_head_1267_; lean_object* v_tail_1268_; lean_object* v___x_1270_; uint8_t v_isShared_1271_; uint8_t v_isSharedCheck_1280_; 
v_head_1267_ = lean_ctor_get(v_a_1264_, 0);
v_tail_1268_ = lean_ctor_get(v_a_1264_, 1);
v_isSharedCheck_1280_ = !lean_is_exclusive(v_a_1264_);
if (v_isSharedCheck_1280_ == 0)
{
v___x_1270_ = v_a_1264_;
v_isShared_1271_ = v_isSharedCheck_1280_;
goto v_resetjp_1269_;
}
else
{
lean_inc(v_tail_1268_);
lean_inc(v_head_1267_);
lean_dec(v_a_1264_);
v___x_1270_ = lean_box(0);
v_isShared_1271_ = v_isSharedCheck_1280_;
goto v_resetjp_1269_;
}
v_resetjp_1269_:
{
lean_object* v_visitedLevel_1272_; lean_object* v___x_1273_; uint8_t v___x_1274_; 
v_visitedLevel_1272_ = lean_ctor_get(v___x_1263_, 0);
lean_inc(v_head_1267_);
v___x_1273_ = l_Lean_Level_param___override(v_head_1267_);
v___x_1274_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___redArg(v_visitedLevel_1272_, v___x_1273_);
lean_dec(v___x_1273_);
if (v___x_1274_ == 0)
{
lean_del_object(v___x_1270_);
lean_dec(v_head_1267_);
v_a_1264_ = v_tail_1268_;
goto _start;
}
else
{
lean_object* v___x_1277_; 
if (v_isShared_1271_ == 0)
{
lean_ctor_set(v___x_1270_, 1, v_a_1265_);
v___x_1277_ = v___x_1270_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1279_; 
v_reuseFailAlloc_1279_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1279_, 0, v_head_1267_);
lean_ctor_set(v_reuseFailAlloc_1279_, 1, v_a_1265_);
v___x_1277_ = v_reuseFailAlloc_1279_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
v_a_1264_ = v_tail_1268_;
v_a_1265_ = v___x_1277_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__2___boxed(lean_object* v___x_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_){
_start:
{
lean_object* v_res_1284_; 
v_res_1284_ = lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__2(v___x_1281_, v_a_1282_, v_a_1283_);
lean_dec_ref(v___x_1281_);
return v_res_1284_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___redArg(lean_object* v_a_1285_, lean_object* v_x_1286_){
_start:
{
if (lean_obj_tag(v_x_1286_) == 0)
{
uint8_t v___x_1287_; 
v___x_1287_ = 0;
return v___x_1287_;
}
else
{
lean_object* v_key_1288_; lean_object* v_tail_1289_; uint8_t v___x_1290_; 
v_key_1288_ = lean_ctor_get(v_x_1286_, 0);
v_tail_1289_ = lean_ctor_get(v_x_1286_, 2);
v___x_1290_ = lean_nat_dec_eq(v_key_1288_, v_a_1285_);
if (v___x_1290_ == 0)
{
v_x_1286_ = v_tail_1289_;
goto _start;
}
else
{
return v___x_1290_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___redArg___boxed(lean_object* v_a_1292_, lean_object* v_x_1293_){
_start:
{
uint8_t v_res_1294_; lean_object* v_r_1295_; 
v_res_1294_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___redArg(v_a_1292_, v_x_1293_);
lean_dec(v_x_1293_);
lean_dec(v_a_1292_);
v_r_1295_ = lean_box(v_res_1294_);
return v_r_1295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__12___redArg(lean_object* v_a_1296_, lean_object* v_b_1297_, lean_object* v_x_1298_){
_start:
{
if (lean_obj_tag(v_x_1298_) == 0)
{
lean_dec(v_b_1297_);
lean_dec(v_a_1296_);
return v_x_1298_;
}
else
{
lean_object* v_key_1299_; lean_object* v_value_1300_; lean_object* v_tail_1301_; lean_object* v___x_1303_; uint8_t v_isShared_1304_; uint8_t v_isSharedCheck_1313_; 
v_key_1299_ = lean_ctor_get(v_x_1298_, 0);
v_value_1300_ = lean_ctor_get(v_x_1298_, 1);
v_tail_1301_ = lean_ctor_get(v_x_1298_, 2);
v_isSharedCheck_1313_ = !lean_is_exclusive(v_x_1298_);
if (v_isSharedCheck_1313_ == 0)
{
v___x_1303_ = v_x_1298_;
v_isShared_1304_ = v_isSharedCheck_1313_;
goto v_resetjp_1302_;
}
else
{
lean_inc(v_tail_1301_);
lean_inc(v_value_1300_);
lean_inc(v_key_1299_);
lean_dec(v_x_1298_);
v___x_1303_ = lean_box(0);
v_isShared_1304_ = v_isSharedCheck_1313_;
goto v_resetjp_1302_;
}
v_resetjp_1302_:
{
uint8_t v___x_1305_; 
v___x_1305_ = lean_nat_dec_eq(v_key_1299_, v_a_1296_);
if (v___x_1305_ == 0)
{
lean_object* v___x_1306_; lean_object* v___x_1308_; 
v___x_1306_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__12___redArg(v_a_1296_, v_b_1297_, v_tail_1301_);
if (v_isShared_1304_ == 0)
{
lean_ctor_set(v___x_1303_, 2, v___x_1306_);
v___x_1308_ = v___x_1303_;
goto v_reusejp_1307_;
}
else
{
lean_object* v_reuseFailAlloc_1309_; 
v_reuseFailAlloc_1309_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1309_, 0, v_key_1299_);
lean_ctor_set(v_reuseFailAlloc_1309_, 1, v_value_1300_);
lean_ctor_set(v_reuseFailAlloc_1309_, 2, v___x_1306_);
v___x_1308_ = v_reuseFailAlloc_1309_;
goto v_reusejp_1307_;
}
v_reusejp_1307_:
{
return v___x_1308_;
}
}
else
{
lean_object* v___x_1311_; 
lean_dec(v_value_1300_);
lean_dec(v_key_1299_);
if (v_isShared_1304_ == 0)
{
lean_ctor_set(v___x_1303_, 1, v_b_1297_);
lean_ctor_set(v___x_1303_, 0, v_a_1296_);
v___x_1311_ = v___x_1303_;
goto v_reusejp_1310_;
}
else
{
lean_object* v_reuseFailAlloc_1312_; 
v_reuseFailAlloc_1312_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1312_, 0, v_a_1296_);
lean_ctor_set(v_reuseFailAlloc_1312_, 1, v_b_1297_);
lean_ctor_set(v_reuseFailAlloc_1312_, 2, v_tail_1301_);
v___x_1311_ = v_reuseFailAlloc_1312_;
goto v_reusejp_1310_;
}
v_reusejp_1310_:
{
return v___x_1311_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14_spec__21___redArg(lean_object* v_x_1314_, lean_object* v_x_1315_){
_start:
{
if (lean_obj_tag(v_x_1315_) == 0)
{
return v_x_1314_;
}
else
{
lean_object* v_key_1316_; lean_object* v_value_1317_; lean_object* v_tail_1318_; lean_object* v___x_1320_; uint8_t v_isShared_1321_; uint8_t v_isSharedCheck_1341_; 
v_key_1316_ = lean_ctor_get(v_x_1315_, 0);
v_value_1317_ = lean_ctor_get(v_x_1315_, 1);
v_tail_1318_ = lean_ctor_get(v_x_1315_, 2);
v_isSharedCheck_1341_ = !lean_is_exclusive(v_x_1315_);
if (v_isSharedCheck_1341_ == 0)
{
v___x_1320_ = v_x_1315_;
v_isShared_1321_ = v_isSharedCheck_1341_;
goto v_resetjp_1319_;
}
else
{
lean_inc(v_tail_1318_);
lean_inc(v_value_1317_);
lean_inc(v_key_1316_);
lean_dec(v_x_1315_);
v___x_1320_ = lean_box(0);
v_isShared_1321_ = v_isSharedCheck_1341_;
goto v_resetjp_1319_;
}
v_resetjp_1319_:
{
lean_object* v___x_1322_; uint64_t v___x_1323_; uint64_t v___x_1324_; uint64_t v___x_1325_; uint64_t v_fold_1326_; uint64_t v___x_1327_; uint64_t v___x_1328_; uint64_t v___x_1329_; size_t v___x_1330_; size_t v___x_1331_; size_t v___x_1332_; size_t v___x_1333_; size_t v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1337_; 
v___x_1322_ = lean_array_get_size(v_x_1314_);
v___x_1323_ = lean_uint64_of_nat(v_key_1316_);
v___x_1324_ = 32ULL;
v___x_1325_ = lean_uint64_shift_right(v___x_1323_, v___x_1324_);
v_fold_1326_ = lean_uint64_xor(v___x_1323_, v___x_1325_);
v___x_1327_ = 16ULL;
v___x_1328_ = lean_uint64_shift_right(v_fold_1326_, v___x_1327_);
v___x_1329_ = lean_uint64_xor(v_fold_1326_, v___x_1328_);
v___x_1330_ = lean_uint64_to_usize(v___x_1329_);
v___x_1331_ = lean_usize_of_nat(v___x_1322_);
v___x_1332_ = ((size_t)1ULL);
v___x_1333_ = lean_usize_sub(v___x_1331_, v___x_1332_);
v___x_1334_ = lean_usize_land(v___x_1330_, v___x_1333_);
v___x_1335_ = lean_array_uget_borrowed(v_x_1314_, v___x_1334_);
lean_inc(v___x_1335_);
if (v_isShared_1321_ == 0)
{
lean_ctor_set(v___x_1320_, 2, v___x_1335_);
v___x_1337_ = v___x_1320_;
goto v_reusejp_1336_;
}
else
{
lean_object* v_reuseFailAlloc_1340_; 
v_reuseFailAlloc_1340_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1340_, 0, v_key_1316_);
lean_ctor_set(v_reuseFailAlloc_1340_, 1, v_value_1317_);
lean_ctor_set(v_reuseFailAlloc_1340_, 2, v___x_1335_);
v___x_1337_ = v_reuseFailAlloc_1340_;
goto v_reusejp_1336_;
}
v_reusejp_1336_:
{
lean_object* v___x_1338_; 
v___x_1338_ = lean_array_uset(v_x_1314_, v___x_1334_, v___x_1337_);
v_x_1314_ = v___x_1338_;
v_x_1315_ = v_tail_1318_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14___redArg(lean_object* v_i_1342_, lean_object* v_source_1343_, lean_object* v_target_1344_){
_start:
{
lean_object* v___x_1345_; uint8_t v___x_1346_; 
v___x_1345_ = lean_array_get_size(v_source_1343_);
v___x_1346_ = lean_nat_dec_lt(v_i_1342_, v___x_1345_);
if (v___x_1346_ == 0)
{
lean_dec_ref(v_source_1343_);
lean_dec(v_i_1342_);
return v_target_1344_;
}
else
{
lean_object* v_es_1347_; lean_object* v___x_1348_; lean_object* v_source_1349_; lean_object* v_target_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; 
v_es_1347_ = lean_array_fget(v_source_1343_, v_i_1342_);
v___x_1348_ = lean_box(0);
v_source_1349_ = lean_array_fset(v_source_1343_, v_i_1342_, v___x_1348_);
v_target_1350_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14_spec__21___redArg(v_target_1344_, v_es_1347_);
v___x_1351_ = lean_unsigned_to_nat(1u);
v___x_1352_ = lean_nat_add(v_i_1342_, v___x_1351_);
lean_dec(v_i_1342_);
v_i_1342_ = v___x_1352_;
v_source_1343_ = v_source_1349_;
v_target_1344_ = v_target_1350_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11___redArg(lean_object* v_data_1354_){
_start:
{
lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v_nbuckets_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; 
v___x_1355_ = lean_array_get_size(v_data_1354_);
v___x_1356_ = lean_unsigned_to_nat(2u);
v_nbuckets_1357_ = lean_nat_mul(v___x_1355_, v___x_1356_);
v___x_1358_ = lean_unsigned_to_nat(0u);
v___x_1359_ = lean_box(0);
v___x_1360_ = lean_mk_array(v_nbuckets_1357_, v___x_1359_);
v___x_1361_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14___redArg(v___x_1358_, v_data_1354_, v___x_1360_);
return v___x_1361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7___redArg(lean_object* v_m_1362_, lean_object* v_a_1363_, lean_object* v_b_1364_){
_start:
{
lean_object* v_size_1365_; lean_object* v_buckets_1366_; lean_object* v___x_1368_; uint8_t v_isShared_1369_; uint8_t v_isSharedCheck_1409_; 
v_size_1365_ = lean_ctor_get(v_m_1362_, 0);
v_buckets_1366_ = lean_ctor_get(v_m_1362_, 1);
v_isSharedCheck_1409_ = !lean_is_exclusive(v_m_1362_);
if (v_isSharedCheck_1409_ == 0)
{
v___x_1368_ = v_m_1362_;
v_isShared_1369_ = v_isSharedCheck_1409_;
goto v_resetjp_1367_;
}
else
{
lean_inc(v_buckets_1366_);
lean_inc(v_size_1365_);
lean_dec(v_m_1362_);
v___x_1368_ = lean_box(0);
v_isShared_1369_ = v_isSharedCheck_1409_;
goto v_resetjp_1367_;
}
v_resetjp_1367_:
{
lean_object* v___x_1370_; uint64_t v___x_1371_; uint64_t v___x_1372_; uint64_t v___x_1373_; uint64_t v_fold_1374_; uint64_t v___x_1375_; uint64_t v___x_1376_; uint64_t v___x_1377_; size_t v___x_1378_; size_t v___x_1379_; size_t v___x_1380_; size_t v___x_1381_; size_t v___x_1382_; lean_object* v_bkt_1383_; uint8_t v___x_1384_; 
v___x_1370_ = lean_array_get_size(v_buckets_1366_);
v___x_1371_ = lean_uint64_of_nat(v_a_1363_);
v___x_1372_ = 32ULL;
v___x_1373_ = lean_uint64_shift_right(v___x_1371_, v___x_1372_);
v_fold_1374_ = lean_uint64_xor(v___x_1371_, v___x_1373_);
v___x_1375_ = 16ULL;
v___x_1376_ = lean_uint64_shift_right(v_fold_1374_, v___x_1375_);
v___x_1377_ = lean_uint64_xor(v_fold_1374_, v___x_1376_);
v___x_1378_ = lean_uint64_to_usize(v___x_1377_);
v___x_1379_ = lean_usize_of_nat(v___x_1370_);
v___x_1380_ = ((size_t)1ULL);
v___x_1381_ = lean_usize_sub(v___x_1379_, v___x_1380_);
v___x_1382_ = lean_usize_land(v___x_1378_, v___x_1381_);
v_bkt_1383_ = lean_array_uget_borrowed(v_buckets_1366_, v___x_1382_);
v___x_1384_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___redArg(v_a_1363_, v_bkt_1383_);
if (v___x_1384_ == 0)
{
lean_object* v___x_1385_; lean_object* v_size_x27_1386_; lean_object* v___x_1387_; lean_object* v_buckets_x27_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; uint8_t v___x_1394_; 
v___x_1385_ = lean_unsigned_to_nat(1u);
v_size_x27_1386_ = lean_nat_add(v_size_1365_, v___x_1385_);
lean_dec(v_size_1365_);
lean_inc(v_bkt_1383_);
v___x_1387_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1387_, 0, v_a_1363_);
lean_ctor_set(v___x_1387_, 1, v_b_1364_);
lean_ctor_set(v___x_1387_, 2, v_bkt_1383_);
v_buckets_x27_1388_ = lean_array_uset(v_buckets_1366_, v___x_1382_, v___x_1387_);
v___x_1389_ = lean_unsigned_to_nat(4u);
v___x_1390_ = lean_nat_mul(v_size_x27_1386_, v___x_1389_);
v___x_1391_ = lean_unsigned_to_nat(3u);
v___x_1392_ = lean_nat_div(v___x_1390_, v___x_1391_);
lean_dec(v___x_1390_);
v___x_1393_ = lean_array_get_size(v_buckets_x27_1388_);
v___x_1394_ = lean_nat_dec_le(v___x_1392_, v___x_1393_);
lean_dec(v___x_1392_);
if (v___x_1394_ == 0)
{
lean_object* v_val_1395_; lean_object* v___x_1397_; 
v_val_1395_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11___redArg(v_buckets_x27_1388_);
if (v_isShared_1369_ == 0)
{
lean_ctor_set(v___x_1368_, 1, v_val_1395_);
lean_ctor_set(v___x_1368_, 0, v_size_x27_1386_);
v___x_1397_ = v___x_1368_;
goto v_reusejp_1396_;
}
else
{
lean_object* v_reuseFailAlloc_1398_; 
v_reuseFailAlloc_1398_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1398_, 0, v_size_x27_1386_);
lean_ctor_set(v_reuseFailAlloc_1398_, 1, v_val_1395_);
v___x_1397_ = v_reuseFailAlloc_1398_;
goto v_reusejp_1396_;
}
v_reusejp_1396_:
{
return v___x_1397_;
}
}
else
{
lean_object* v___x_1400_; 
if (v_isShared_1369_ == 0)
{
lean_ctor_set(v___x_1368_, 1, v_buckets_x27_1388_);
lean_ctor_set(v___x_1368_, 0, v_size_x27_1386_);
v___x_1400_ = v___x_1368_;
goto v_reusejp_1399_;
}
else
{
lean_object* v_reuseFailAlloc_1401_; 
v_reuseFailAlloc_1401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1401_, 0, v_size_x27_1386_);
lean_ctor_set(v_reuseFailAlloc_1401_, 1, v_buckets_x27_1388_);
v___x_1400_ = v_reuseFailAlloc_1401_;
goto v_reusejp_1399_;
}
v_reusejp_1399_:
{
return v___x_1400_;
}
}
}
else
{
lean_object* v___x_1402_; lean_object* v_buckets_x27_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1407_; 
lean_inc(v_bkt_1383_);
v___x_1402_ = lean_box(0);
v_buckets_x27_1403_ = lean_array_uset(v_buckets_1366_, v___x_1382_, v___x_1402_);
v___x_1404_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__12___redArg(v_a_1363_, v_b_1364_, v_bkt_1383_);
v___x_1405_ = lean_array_uset(v_buckets_x27_1403_, v___x_1382_, v___x_1404_);
if (v_isShared_1369_ == 0)
{
lean_ctor_set(v___x_1368_, 1, v___x_1405_);
v___x_1407_ = v___x_1368_;
goto v_reusejp_1406_;
}
else
{
lean_object* v_reuseFailAlloc_1408_; 
v_reuseFailAlloc_1408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1408_, 0, v_size_1365_);
lean_ctor_set(v_reuseFailAlloc_1408_, 1, v___x_1405_);
v___x_1407_ = v_reuseFailAlloc_1408_;
goto v_reusejp_1406_;
}
v_reusejp_1406_:
{
return v___x_1407_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___redArg(lean_object* v___x_1410_, lean_object* v_as_1411_, size_t v_sz_1412_, size_t v_i_1413_, lean_object* v_b_1414_){
_start:
{
uint8_t v___x_1416_; 
v___x_1416_ = lean_usize_dec_lt(v_i_1413_, v_sz_1412_);
if (v___x_1416_ == 0)
{
lean_object* v___x_1417_; 
v___x_1417_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1417_, 0, v_b_1414_);
return v___x_1417_;
}
else
{
lean_object* v_a_1418_; lean_object* v___y_1420_; lean_object* v___x_1425_; 
v_a_1418_ = lean_array_uget_borrowed(v_as_1411_, v_i_1413_);
v___x_1425_ = l_Lean_Environment_getModuleIdx_x3f(v___x_1410_, v_a_1418_);
if (lean_obj_tag(v___x_1425_) == 0)
{
lean_object* v___x_1426_; 
v___x_1426_ = lean_unsigned_to_nat(0u);
v___y_1420_ = v___x_1426_;
goto v___jp_1419_;
}
else
{
lean_object* v_val_1427_; 
v_val_1427_ = lean_ctor_get(v___x_1425_, 0);
lean_inc(v_val_1427_);
lean_dec_ref_known(v___x_1425_, 1);
v___y_1420_ = v_val_1427_;
goto v___jp_1419_;
}
v___jp_1419_:
{
lean_object* v___x_1421_; size_t v___x_1422_; size_t v___x_1423_; 
lean_inc(v_a_1418_);
v___x_1421_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7___redArg(v_b_1414_, v___y_1420_, v_a_1418_);
v___x_1422_ = ((size_t)1ULL);
v___x_1423_ = lean_usize_add(v_i_1413_, v___x_1422_);
v_i_1413_ = v___x_1423_;
v_b_1414_ = v___x_1421_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___redArg___boxed(lean_object* v___x_1428_, lean_object* v_as_1429_, lean_object* v_sz_1430_, lean_object* v_i_1431_, lean_object* v_b_1432_, lean_object* v___y_1433_){
_start:
{
size_t v_sz_boxed_1434_; size_t v_i_boxed_1435_; lean_object* v_res_1436_; 
v_sz_boxed_1434_ = lean_unbox_usize(v_sz_1430_);
lean_dec(v_sz_1430_);
v_i_boxed_1435_ = lean_unbox_usize(v_i_1431_);
lean_dec(v_i_1431_);
v_res_1436_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___redArg(v___x_1428_, v_as_1429_, v_sz_boxed_1434_, v_i_boxed_1435_, v_b_1432_);
lean_dec_ref(v_as_1429_);
lean_dec_ref(v___x_1428_);
return v_res_1436_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; 
v___x_1437_ = lean_box(0);
v___x_1438_ = lean_unsigned_to_nat(16u);
v___x_1439_ = lean_mk_array(v___x_1438_, v___x_1437_);
return v___x_1439_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; 
v___x_1440_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__0, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__0);
v___x_1441_ = lean_unsigned_to_nat(0u);
v___x_1442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1442_, 0, v___x_1441_);
lean_ctor_set(v___x_1442_, 1, v___x_1440_);
return v___x_1442_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; 
v___x_1445_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__2));
v___x_1446_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1);
v___x_1447_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1447_, 0, v___x_1446_);
lean_ctor_set(v___x_1447_, 1, v___x_1446_);
lean_ctor_set(v___x_1447_, 2, v___x_1445_);
return v___x_1447_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__4(void){
_start:
{
lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; 
v___x_1448_ = lean_unsigned_to_nat(32u);
v___x_1449_ = lean_mk_empty_array_with_capacity(v___x_1448_);
v___x_1450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1450_, 0, v___x_1449_);
return v___x_1450_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5(void){
_start:
{
size_t v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; 
v___x_1451_ = ((size_t)5ULL);
v___x_1452_ = lean_unsigned_to_nat(0u);
v___x_1453_ = lean_unsigned_to_nat(32u);
v___x_1454_ = lean_mk_empty_array_with_capacity(v___x_1453_);
v___x_1455_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__4, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__4);
v___x_1456_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1456_, 0, v___x_1455_);
lean_ctor_set(v___x_1456_, 1, v___x_1454_);
lean_ctor_set(v___x_1456_, 2, v___x_1452_);
lean_ctor_set(v___x_1456_, 3, v___x_1452_);
lean_ctor_set_usize(v___x_1456_, 4, v___x_1451_);
return v___x_1456_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__6(void){
_start:
{
lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; 
v___x_1457_ = l_Lean_NameSet_empty;
v___x_1458_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5);
v___x_1459_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1459_, 0, v___x_1458_);
lean_ctor_set(v___x_1459_, 1, v___x_1458_);
lean_ctor_set(v___x_1459_, 2, v___x_1457_);
return v___x_1459_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__8(void){
_start:
{
lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; 
v___x_1461_ = lean_unsigned_to_nat(1u);
v___x_1462_ = l_Lean_firstFrontendMacroScope;
v___x_1463_ = lean_nat_add(v___x_1462_, v___x_1461_);
return v___x_1463_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__12(void){
_start:
{
lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___x_1470_ = lean_box(0);
v___x_1471_ = l_Lean_DeclNameGenerator_ofPrefix(v___x_1470_);
return v___x_1471_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__13(void){
_start:
{
lean_object* v___x_1472_; 
v___x_1472_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1472_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__14(void){
_start:
{
lean_object* v___x_1473_; lean_object* v___x_1474_; 
v___x_1473_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__13, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__13);
v___x_1474_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1474_, 0, v___x_1473_);
return v___x_1474_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__15(void){
_start:
{
lean_object* v___x_1475_; uint64_t v___x_1476_; lean_object* v___x_1477_; 
v___x_1475_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5);
v___x_1476_ = 0ULL;
v___x_1477_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1477_, 0, v___x_1475_);
lean_ctor_set_uint64(v___x_1477_, sizeof(void*)*1, v___x_1476_);
return v___x_1477_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__18(void){
_start:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; 
v___x_1482_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__17));
v___x_1483_ = l_Lean_stringToMessageData(v___x_1482_);
return v___x_1483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1(lean_object* v_g_1484_, lean_object* v_name_1485_, lean_object* v___y_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_){
_start:
{
lean_object* v___y_1494_; uint8_t v___y_1495_; lean_object* v___y_1496_; lean_object* v___y_1497_; lean_object* v___y_1504_; lean_object* v___y_1505_; lean_object* v___y_1506_; uint8_t v___y_1507_; lean_object* v___y_1508_; lean_object* v___y_1509_; lean_object* v___y_1510_; lean_object* v___y_1513_; lean_object* v___y_1514_; uint8_t v___y_1515_; lean_object* v___y_1516_; lean_object* v___y_1517_; lean_object* v___y_1518_; lean_object* v___y_1519_; lean_object* v___y_1522_; lean_object* v___y_1523_; uint8_t v___y_1524_; lean_object* v___y_1525_; lean_object* v___y_1526_; lean_object* v___y_1527_; lean_object* v___y_1528_; lean_object* v___y_1536_; lean_object* v___y_1537_; lean_object* v___y_1538_; uint8_t v___y_1539_; lean_object* v___y_1540_; lean_object* v___y_1541_; lean_object* v___y_1542_; lean_object* v_a_1543_; uint8_t v___y_1550_; lean_object* v___y_1551_; lean_object* v___y_1552_; uint8_t v___y_1553_; uint8_t v___y_1554_; lean_object* v___y_1555_; lean_object* v___y_1556_; lean_object* v___y_1557_; lean_object* v___y_1558_; lean_object* v___y_1559_; lean_object* v___y_1560_; lean_object* v___x_1661_; 
v___x_1661_ = lp_batteries_Lean_MVarId_renameInaccessibleFVars(v_g_1484_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_);
if (lean_obj_tag(v___x_1661_) == 0)
{
lean_object* v_a_1662_; lean_object* v_fst_1663_; lean_object* v___x_1664_; 
v_a_1662_ = lean_ctor_get(v___x_1661_, 0);
lean_inc(v_a_1662_);
lean_dec_ref_known(v___x_1661_, 1);
v_fst_1663_ = lean_ctor_get(v_a_1662_, 0);
lean_inc_n(v_fst_1663_, 2);
lean_dec(v_a_1662_);
v___x_1664_ = l_Lean_MVarId_getType(v_fst_1663_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_);
if (lean_obj_tag(v___x_1664_) == 0)
{
lean_object* v_a_1665_; lean_object* v___x_1666_; lean_object* v_a_1667_; uint8_t v___y_1669_; uint8_t v___y_1729_; uint8_t v___x_1733_; 
v_a_1665_ = lean_ctor_get(v___x_1664_, 0);
lean_inc(v_a_1665_);
lean_dec_ref_known(v___x_1664_, 1);
v___x_1666_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg(v_a_1665_, v___y_1489_);
v_a_1667_ = lean_ctor_get(v___x_1666_, 0);
lean_inc(v_a_1667_);
lean_dec_ref(v___x_1666_);
v___x_1733_ = l_Lean_Expr_isForall(v_a_1667_);
if (v___x_1733_ == 0)
{
v___y_1729_ = v___x_1733_;
goto v___jp_1728_;
}
else
{
lean_object* v___x_1734_; uint8_t v___x_1735_; 
v___x_1734_ = l_Lean_Expr_bindingName_x21(v_a_1667_);
v___x_1735_ = l_Lean_Name_isAnonymous(v___x_1734_);
lean_dec(v___x_1734_);
if (v___x_1735_ == 0)
{
v___y_1729_ = v___x_1733_;
goto v___jp_1728_;
}
else
{
uint8_t v___x_1736_; 
lean_dec(v_a_1667_);
v___x_1736_ = 0;
v___y_1669_ = v___x_1736_;
goto v___jp_1668_;
}
}
v___jp_1668_:
{
lean_object* v___x_1670_; 
lean_inc(v_fst_1663_);
v___x_1670_ = l_Lean_MVarId_getDecl(v_fst_1663_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_);
if (lean_obj_tag(v___x_1670_) == 0)
{
lean_object* v_a_1671_; lean_object* v_lctx_1672_; lean_object* v___x_1673_; uint8_t v___x_1674_; uint8_t v___x_1675_; lean_object* v___x_1676_; 
v_a_1671_ = lean_ctor_get(v___x_1670_, 0);
lean_inc(v_a_1671_);
lean_dec_ref_known(v___x_1670_, 1);
v_lctx_1672_ = lean_ctor_get(v_a_1671_, 1);
lean_inc_ref(v_lctx_1672_);
lean_dec(v_a_1671_);
v___x_1673_ = l_Lean_LocalContext_getFVarIds(v_lctx_1672_);
lean_dec_ref(v_lctx_1672_);
v___x_1674_ = 0;
v___x_1675_ = 1;
v___x_1676_ = l_Lean_MVarId_revert(v_fst_1663_, v___x_1673_, v___x_1674_, v___x_1675_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_);
if (lean_obj_tag(v___x_1676_) == 0)
{
lean_object* v_a_1677_; lean_object* v_snd_1678_; lean_object* v___x_1680_; uint8_t v_isShared_1681_; uint8_t v_isSharedCheck_1710_; 
v_a_1677_ = lean_ctor_get(v___x_1676_, 0);
lean_inc(v_a_1677_);
lean_dec_ref_known(v___x_1676_, 1);
v_snd_1678_ = lean_ctor_get(v_a_1677_, 1);
v_isSharedCheck_1710_ = !lean_is_exclusive(v_a_1677_);
if (v_isSharedCheck_1710_ == 0)
{
lean_object* v_unused_1711_; 
v_unused_1711_ = lean_ctor_get(v_a_1677_, 0);
lean_dec(v_unused_1711_);
v___x_1680_ = v_a_1677_;
v_isShared_1681_ = v_isSharedCheck_1710_;
goto v_resetjp_1679_;
}
else
{
lean_inc(v_snd_1678_);
lean_dec(v_a_1677_);
v___x_1680_ = lean_box(0);
v_isShared_1681_ = v_isSharedCheck_1710_;
goto v_resetjp_1679_;
}
v_resetjp_1679_:
{
lean_object* v___x_1682_; 
v___x_1682_ = l_Lean_MVarId_getType(v_snd_1678_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_);
if (lean_obj_tag(v___x_1682_) == 0)
{
lean_object* v_a_1683_; lean_object* v___x_1684_; lean_object* v_a_1685_; lean_object* v___f_1686_; uint8_t v___x_1687_; 
v_a_1683_ = lean_ctor_get(v___x_1682_, 0);
lean_inc(v_a_1683_);
lean_dec_ref_known(v___x_1682_, 1);
v___x_1684_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__0___redArg(v_a_1683_, v___y_1489_);
v_a_1685_ = lean_ctor_get(v___x_1684_, 0);
lean_inc(v_a_1685_);
lean_dec_ref(v___x_1684_);
v___f_1686_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__16));
v___x_1687_ = l_Lean_Expr_hasExprMVar(v_a_1685_);
if (v___x_1687_ == 0)
{
lean_del_object(v___x_1680_);
v___y_1550_ = v___y_1669_;
v___y_1551_ = v_a_1685_;
v___y_1552_ = v___f_1686_;
v___y_1553_ = v___x_1675_;
v___y_1554_ = v___x_1674_;
v___y_1555_ = v___y_1486_;
v___y_1556_ = v___y_1487_;
v___y_1557_ = v___y_1488_;
v___y_1558_ = v___y_1489_;
v___y_1559_ = v___y_1490_;
v___y_1560_ = v___y_1491_;
goto v___jp_1549_;
}
else
{
lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v___x_1691_; 
lean_dec(v_name_1485_);
v___x_1688_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__18, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__18);
v___x_1689_ = l_Lean_MessageData_ofExpr(v_a_1685_);
if (v_isShared_1681_ == 0)
{
lean_ctor_set_tag(v___x_1680_, 7);
lean_ctor_set(v___x_1680_, 1, v___x_1689_);
lean_ctor_set(v___x_1680_, 0, v___x_1688_);
v___x_1691_ = v___x_1680_;
goto v_reusejp_1690_;
}
else
{
lean_object* v_reuseFailAlloc_1701_; 
v_reuseFailAlloc_1701_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1701_, 0, v___x_1688_);
lean_ctor_set(v_reuseFailAlloc_1701_, 1, v___x_1689_);
v___x_1691_ = v_reuseFailAlloc_1701_;
goto v_reusejp_1690_;
}
v_reusejp_1690_:
{
lean_object* v___x_1692_; lean_object* v_a_1693_; lean_object* v___x_1695_; uint8_t v_isShared_1696_; uint8_t v_isSharedCheck_1700_; 
v___x_1692_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___redArg(v___x_1691_, v___y_1486_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_);
v_a_1693_ = lean_ctor_get(v___x_1692_, 0);
v_isSharedCheck_1700_ = !lean_is_exclusive(v___x_1692_);
if (v_isSharedCheck_1700_ == 0)
{
v___x_1695_ = v___x_1692_;
v_isShared_1696_ = v_isSharedCheck_1700_;
goto v_resetjp_1694_;
}
else
{
lean_inc(v_a_1693_);
lean_dec(v___x_1692_);
v___x_1695_ = lean_box(0);
v_isShared_1696_ = v_isSharedCheck_1700_;
goto v_resetjp_1694_;
}
v_resetjp_1694_:
{
lean_object* v___x_1698_; 
if (v_isShared_1696_ == 0)
{
v___x_1698_ = v___x_1695_;
goto v_reusejp_1697_;
}
else
{
lean_object* v_reuseFailAlloc_1699_; 
v_reuseFailAlloc_1699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1699_, 0, v_a_1693_);
v___x_1698_ = v_reuseFailAlloc_1699_;
goto v_reusejp_1697_;
}
v_reusejp_1697_:
{
return v___x_1698_;
}
}
}
}
}
else
{
lean_object* v_a_1702_; lean_object* v___x_1704_; uint8_t v_isShared_1705_; uint8_t v_isSharedCheck_1709_; 
lean_del_object(v___x_1680_);
lean_dec(v_name_1485_);
v_a_1702_ = lean_ctor_get(v___x_1682_, 0);
v_isSharedCheck_1709_ = !lean_is_exclusive(v___x_1682_);
if (v_isSharedCheck_1709_ == 0)
{
v___x_1704_ = v___x_1682_;
v_isShared_1705_ = v_isSharedCheck_1709_;
goto v_resetjp_1703_;
}
else
{
lean_inc(v_a_1702_);
lean_dec(v___x_1682_);
v___x_1704_ = lean_box(0);
v_isShared_1705_ = v_isSharedCheck_1709_;
goto v_resetjp_1703_;
}
v_resetjp_1703_:
{
lean_object* v___x_1707_; 
if (v_isShared_1705_ == 0)
{
v___x_1707_ = v___x_1704_;
goto v_reusejp_1706_;
}
else
{
lean_object* v_reuseFailAlloc_1708_; 
v_reuseFailAlloc_1708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1708_, 0, v_a_1702_);
v___x_1707_ = v_reuseFailAlloc_1708_;
goto v_reusejp_1706_;
}
v_reusejp_1706_:
{
return v___x_1707_;
}
}
}
}
}
else
{
lean_object* v_a_1712_; lean_object* v___x_1714_; uint8_t v_isShared_1715_; uint8_t v_isSharedCheck_1719_; 
lean_dec(v_name_1485_);
v_a_1712_ = lean_ctor_get(v___x_1676_, 0);
v_isSharedCheck_1719_ = !lean_is_exclusive(v___x_1676_);
if (v_isSharedCheck_1719_ == 0)
{
v___x_1714_ = v___x_1676_;
v_isShared_1715_ = v_isSharedCheck_1719_;
goto v_resetjp_1713_;
}
else
{
lean_inc(v_a_1712_);
lean_dec(v___x_1676_);
v___x_1714_ = lean_box(0);
v_isShared_1715_ = v_isSharedCheck_1719_;
goto v_resetjp_1713_;
}
v_resetjp_1713_:
{
lean_object* v___x_1717_; 
if (v_isShared_1715_ == 0)
{
v___x_1717_ = v___x_1714_;
goto v_reusejp_1716_;
}
else
{
lean_object* v_reuseFailAlloc_1718_; 
v_reuseFailAlloc_1718_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1718_, 0, v_a_1712_);
v___x_1717_ = v_reuseFailAlloc_1718_;
goto v_reusejp_1716_;
}
v_reusejp_1716_:
{
return v___x_1717_;
}
}
}
}
else
{
lean_object* v_a_1720_; lean_object* v___x_1722_; uint8_t v_isShared_1723_; uint8_t v_isSharedCheck_1727_; 
lean_dec(v_fst_1663_);
lean_dec(v_name_1485_);
v_a_1720_ = lean_ctor_get(v___x_1670_, 0);
v_isSharedCheck_1727_ = !lean_is_exclusive(v___x_1670_);
if (v_isSharedCheck_1727_ == 0)
{
v___x_1722_ = v___x_1670_;
v_isShared_1723_ = v_isSharedCheck_1727_;
goto v_resetjp_1721_;
}
else
{
lean_inc(v_a_1720_);
lean_dec(v___x_1670_);
v___x_1722_ = lean_box(0);
v_isShared_1723_ = v_isSharedCheck_1727_;
goto v_resetjp_1721_;
}
v_resetjp_1721_:
{
lean_object* v___x_1725_; 
if (v_isShared_1723_ == 0)
{
v___x_1725_ = v___x_1722_;
goto v_reusejp_1724_;
}
else
{
lean_object* v_reuseFailAlloc_1726_; 
v_reuseFailAlloc_1726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1726_, 0, v_a_1720_);
v___x_1725_ = v_reuseFailAlloc_1726_;
goto v_reusejp_1724_;
}
v_reusejp_1724_:
{
return v___x_1725_;
}
}
}
}
v___jp_1728_:
{
if (v___y_1729_ == 0)
{
lean_dec(v_a_1667_);
v___y_1669_ = v___y_1729_;
goto v___jp_1668_;
}
else
{
lean_object* v___x_1730_; uint8_t v___x_1731_; 
v___x_1730_ = l_Lean_Expr_bindingName_x21(v_a_1667_);
lean_dec(v_a_1667_);
v___x_1731_ = l_Lean_Name_isInternal(v___x_1730_);
lean_dec(v___x_1730_);
if (v___x_1731_ == 0)
{
v___y_1669_ = v___y_1729_;
goto v___jp_1668_;
}
else
{
uint8_t v___x_1732_; 
v___x_1732_ = 0;
v___y_1669_ = v___x_1732_;
goto v___jp_1668_;
}
}
}
}
else
{
lean_object* v_a_1737_; lean_object* v___x_1739_; uint8_t v_isShared_1740_; uint8_t v_isSharedCheck_1744_; 
lean_dec(v_fst_1663_);
lean_dec(v_name_1485_);
v_a_1737_ = lean_ctor_get(v___x_1664_, 0);
v_isSharedCheck_1744_ = !lean_is_exclusive(v___x_1664_);
if (v_isSharedCheck_1744_ == 0)
{
v___x_1739_ = v___x_1664_;
v_isShared_1740_ = v_isSharedCheck_1744_;
goto v_resetjp_1738_;
}
else
{
lean_inc(v_a_1737_);
lean_dec(v___x_1664_);
v___x_1739_ = lean_box(0);
v_isShared_1740_ = v_isSharedCheck_1744_;
goto v_resetjp_1738_;
}
v_resetjp_1738_:
{
lean_object* v___x_1742_; 
if (v_isShared_1740_ == 0)
{
v___x_1742_ = v___x_1739_;
goto v_reusejp_1741_;
}
else
{
lean_object* v_reuseFailAlloc_1743_; 
v_reuseFailAlloc_1743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1743_, 0, v_a_1737_);
v___x_1742_ = v_reuseFailAlloc_1743_;
goto v_reusejp_1741_;
}
v_reusejp_1741_:
{
return v___x_1742_;
}
}
}
}
else
{
lean_object* v_a_1745_; lean_object* v___x_1747_; uint8_t v_isShared_1748_; uint8_t v_isSharedCheck_1752_; 
lean_dec(v_name_1485_);
v_a_1745_ = lean_ctor_get(v___x_1661_, 0);
v_isSharedCheck_1752_ = !lean_is_exclusive(v___x_1661_);
if (v_isSharedCheck_1752_ == 0)
{
v___x_1747_ = v___x_1661_;
v_isShared_1748_ = v_isSharedCheck_1752_;
goto v_resetjp_1746_;
}
else
{
lean_inc(v_a_1745_);
lean_dec(v___x_1661_);
v___x_1747_ = lean_box(0);
v_isShared_1748_ = v_isSharedCheck_1752_;
goto v_resetjp_1746_;
}
v_resetjp_1746_:
{
lean_object* v___x_1750_; 
if (v_isShared_1748_ == 0)
{
v___x_1750_ = v___x_1747_;
goto v_reusejp_1749_;
}
else
{
lean_object* v_reuseFailAlloc_1751_; 
v_reuseFailAlloc_1751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1751_, 0, v_a_1745_);
v___x_1750_ = v_reuseFailAlloc_1751_;
goto v_reusejp_1749_;
}
v_reusejp_1749_:
{
return v___x_1750_;
}
}
}
v___jp_1493_:
{
lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; 
v___x_1498_ = lean_box(v___y_1495_);
v___x_1499_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1499_, 0, v___y_1497_);
lean_ctor_set(v___x_1499_, 1, v___x_1498_);
v___x_1500_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1500_, 0, v___y_1494_);
lean_ctor_set(v___x_1500_, 1, v___x_1499_);
v___x_1501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1501_, 0, v___y_1496_);
lean_ctor_set(v___x_1501_, 1, v___x_1500_);
v___x_1502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1502_, 0, v___x_1501_);
return v___x_1502_;
}
v___jp_1503_:
{
lean_object* v___x_1511_; 
v___x_1511_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg(v___y_1506_, v___y_1509_, v___y_1505_, v___y_1510_);
lean_dec(v___y_1510_);
lean_dec(v___y_1506_);
v___y_1494_ = v___y_1504_;
v___y_1495_ = v___y_1507_;
v___y_1496_ = v___y_1508_;
v___y_1497_ = v___x_1511_;
goto v___jp_1493_;
}
v___jp_1512_:
{
uint8_t v___x_1520_; 
v___x_1520_ = lean_nat_dec_le(v___y_1519_, v___y_1516_);
if (v___x_1520_ == 0)
{
lean_dec(v___y_1516_);
lean_inc(v___y_1519_);
v___y_1504_ = v___y_1513_;
v___y_1505_ = v___y_1519_;
v___y_1506_ = v___y_1514_;
v___y_1507_ = v___y_1515_;
v___y_1508_ = v___y_1517_;
v___y_1509_ = v___y_1518_;
v___y_1510_ = v___y_1519_;
goto v___jp_1503_;
}
else
{
v___y_1504_ = v___y_1513_;
v___y_1505_ = v___y_1519_;
v___y_1506_ = v___y_1514_;
v___y_1507_ = v___y_1515_;
v___y_1508_ = v___y_1517_;
v___y_1509_ = v___y_1518_;
v___y_1510_ = v___y_1516_;
goto v___jp_1503_;
}
}
v___jp_1521_:
{
lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; uint8_t v___x_1532_; 
v___x_1529_ = lean_mk_empty_array_with_capacity(v___y_1528_);
lean_dec(v___y_1528_);
v___x_1530_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__5_spec__6(v___x_1529_, v___y_1525_);
v___x_1531_ = lean_array_get_size(v___x_1530_);
v___x_1532_ = lean_nat_dec_eq(v___x_1531_, v___y_1523_);
if (v___x_1532_ == 0)
{
lean_object* v___x_1533_; uint8_t v___x_1534_; 
v___x_1533_ = lean_nat_sub(v___x_1531_, v___y_1527_);
v___x_1534_ = lean_nat_dec_le(v___y_1523_, v___x_1533_);
if (v___x_1534_ == 0)
{
lean_dec(v___y_1523_);
lean_inc(v___x_1533_);
v___y_1513_ = v___y_1522_;
v___y_1514_ = v___x_1531_;
v___y_1515_ = v___y_1524_;
v___y_1516_ = v___x_1533_;
v___y_1517_ = v___y_1526_;
v___y_1518_ = v___x_1530_;
v___y_1519_ = v___x_1533_;
goto v___jp_1512_;
}
else
{
v___y_1513_ = v___y_1522_;
v___y_1514_ = v___x_1531_;
v___y_1515_ = v___y_1524_;
v___y_1516_ = v___x_1533_;
v___y_1517_ = v___y_1526_;
v___y_1518_ = v___x_1530_;
v___y_1519_ = v___y_1523_;
goto v___jp_1512_;
}
}
else
{
lean_dec(v___y_1523_);
v___y_1494_ = v___y_1522_;
v___y_1495_ = v___y_1524_;
v___y_1496_ = v___y_1526_;
v___y_1497_ = v___x_1530_;
goto v___jp_1493_;
}
}
v___jp_1535_:
{
lean_object* v___x_1544_; lean_object* v_env_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; 
v___x_1544_ = lean_st_ref_get(v___y_1540_);
v_env_1545_ = lean_ctor_get(v___x_1544_, 0);
lean_inc_ref(v_env_1545_);
lean_dec(v___x_1544_);
v___x_1546_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg(v___y_1538_, v_a_1543_);
v___x_1547_ = lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(v_env_1545_, v___x_1546_);
lean_dec_ref(v_env_1545_);
if (lean_obj_tag(v___x_1547_) == 0)
{
lean_object* v_size_1548_; 
v_size_1548_ = lean_ctor_get(v___x_1547_, 0);
lean_inc(v_size_1548_);
v___y_1522_ = v___y_1536_;
v___y_1523_ = v___y_1537_;
v___y_1524_ = v___y_1539_;
v___y_1525_ = v___x_1547_;
v___y_1526_ = v___y_1541_;
v___y_1527_ = v___y_1542_;
v___y_1528_ = v_size_1548_;
goto v___jp_1521_;
}
else
{
lean_inc(v___y_1537_);
v___y_1522_ = v___y_1536_;
v___y_1523_ = v___y_1537_;
v___y_1524_ = v___y_1539_;
v___y_1525_ = v___x_1547_;
v___y_1526_ = v___y_1541_;
v___y_1527_ = v___y_1542_;
v___y_1528_ = v___y_1537_;
goto v___jp_1521_;
}
}
v___jp_1549_:
{
lean_object* v___x_1561_; 
lean_inc_ref(v___y_1552_);
v___x_1561_ = l_Lean_Elab_Term_levelMVarToParam___redArg(v___y_1551_, v___y_1552_, v___y_1556_, v___y_1558_);
if (lean_obj_tag(v___x_1561_) == 0)
{
lean_object* v_a_1562_; lean_object* v___x_1563_; 
v_a_1562_ = lean_ctor_get(v___x_1561_, 0);
lean_inc(v_a_1562_);
lean_dec_ref_known(v___x_1561_, 1);
v___x_1563_ = l_Lean_Elab_Term_getLevelNames___redArg(v___y_1556_);
if (lean_obj_tag(v___x_1563_) == 0)
{
lean_object* v_a_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; 
v_a_1564_ = lean_ctor_get(v___x_1563_, 0);
lean_inc(v_a_1564_);
lean_dec_ref_known(v___x_1563_, 1);
v___x_1565_ = lean_unsigned_to_nat(0u);
v___x_1566_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__1);
v___x_1567_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__2));
v___x_1568_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__3, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__3);
lean_inc_n(v_a_1562_, 2);
v___x_1569_ = l_Lean_collectLevelParams(v___x_1568_, v_a_1562_);
v___x_1570_ = lean_box(0);
v___x_1571_ = lp_mathlib_List_filterTR_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__2(v___x_1569_, v_a_1564_, v___x_1570_);
lean_dec_ref(v___x_1569_);
lean_inc(v_name_1485_);
v___x_1572_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1572_, 0, v_name_1485_);
lean_ctor_set(v___x_1572_, 1, v___x_1571_);
lean_ctor_set(v___x_1572_, 2, v_a_1562_);
v___x_1573_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1573_, 0, v___x_1572_);
lean_ctor_set_uint8(v___x_1573_, sizeof(void*)*1, v___y_1554_);
v___x_1574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1574_, 0, v___x_1573_);
v___x_1575_ = l_Lean_addAndCompile(v___x_1574_, v___y_1553_, v___y_1554_, v___y_1559_, v___y_1560_);
if (lean_obj_tag(v___x_1575_) == 0)
{
lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v_a_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v_env_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v_fileName_1589_; lean_object* v_fileMap_1590_; lean_object* v_currRecDepth_1591_; lean_object* v_maxRecDepth_1592_; lean_object* v_ref_1593_; lean_object* v_currMacroScope_1594_; lean_object* v_cancelTk_x3f_1595_; uint8_t v_suppressElabErrors_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; 
lean_dec_ref_known(v___x_1575_, 1);
lean_inc(v_name_1485_);
v___x_1576_ = l_Lean_MessageData_signature(v_name_1485_);
v___x_1577_ = lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3(v___x_1576_, v___y_1557_, v___y_1558_, v___y_1559_, v___y_1560_);
v_a_1578_ = lean_ctor_get(v___x_1577_, 0);
lean_inc(v_a_1578_);
lean_dec_ref(v___x_1577_);
v___x_1579_ = lean_st_ref_get(v___y_1556_);
lean_dec(v___x_1579_);
v___x_1580_ = lean_st_ref_get(v___y_1560_);
v_env_1581_ = lean_ctor_get(v___x_1580_, 0);
lean_inc_ref_n(v_env_1581_, 2);
lean_dec(v___x_1580_);
v___x_1582_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__5);
v___x_1583_ = l_Lean_NameSet_empty;
v___x_1584_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__6, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__6);
v___x_1585_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__7));
v___x_1586_ = l_Lean_Options_empty;
v___x_1587_ = lean_box(0);
v___x_1588_ = lean_alloc_ctor(0, 10, 3);
lean_ctor_set(v___x_1588_, 0, v___x_1585_);
lean_ctor_set(v___x_1588_, 1, v___x_1586_);
lean_ctor_set(v___x_1588_, 2, v___x_1587_);
lean_ctor_set(v___x_1588_, 3, v___x_1570_);
lean_ctor_set(v___x_1588_, 4, v___x_1570_);
lean_ctor_set(v___x_1588_, 5, v___x_1567_);
lean_ctor_set(v___x_1588_, 6, v___x_1567_);
lean_ctor_set(v___x_1588_, 7, v___x_1570_);
lean_ctor_set(v___x_1588_, 8, v___x_1570_);
lean_ctor_set(v___x_1588_, 9, v___x_1570_);
lean_ctor_set_uint8(v___x_1588_, sizeof(void*)*10, v___y_1554_);
lean_ctor_set_uint8(v___x_1588_, sizeof(void*)*10 + 1, v___y_1554_);
lean_ctor_set_uint8(v___x_1588_, sizeof(void*)*10 + 2, v___y_1554_);
v_fileName_1589_ = lean_ctor_get(v___y_1559_, 0);
v_fileMap_1590_ = lean_ctor_get(v___y_1559_, 1);
v_currRecDepth_1591_ = lean_ctor_get(v___y_1559_, 3);
v_maxRecDepth_1592_ = lean_ctor_get(v___y_1559_, 4);
v_ref_1593_ = lean_ctor_get(v___y_1559_, 5);
v_currMacroScope_1594_ = lean_ctor_get(v___y_1559_, 11);
v_cancelTk_x3f_1595_ = lean_ctor_get(v___y_1559_, 12);
v_suppressElabErrors_1596_ = lean_ctor_get_uint8(v___y_1559_, sizeof(void*)*14 + 1);
v___x_1597_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1597_, 0, v___x_1588_);
lean_ctor_set(v___x_1597_, 1, v___x_1570_);
v___x_1598_ = lean_unsigned_to_nat(1u);
v___x_1599_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__8, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__8);
v___x_1600_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__11));
v___x_1601_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__12, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__12);
v___x_1602_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__14, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__14);
v___x_1603_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_1603_, 0, v___x_1602_);
lean_ctor_set(v___x_1603_, 1, v___x_1602_);
lean_ctor_set(v___x_1603_, 2, v___x_1582_);
lean_ctor_set_uint8(v___x_1603_, sizeof(void*)*3, v___y_1553_);
v___x_1604_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__15, &lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__15);
v___x_1605_ = lean_box(0);
lean_inc(v_maxRecDepth_1592_);
v___x_1606_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v___x_1606_, 0, v_env_1581_);
lean_ctor_set(v___x_1606_, 1, v___x_1584_);
lean_ctor_set(v___x_1606_, 2, v___x_1597_);
lean_ctor_set(v___x_1606_, 3, v___x_1583_);
lean_ctor_set(v___x_1606_, 4, v___x_1599_);
lean_ctor_set(v___x_1606_, 5, v_maxRecDepth_1592_);
lean_ctor_set(v___x_1606_, 6, v___x_1600_);
lean_ctor_set(v___x_1606_, 7, v___x_1601_);
lean_ctor_set(v___x_1606_, 8, v___x_1603_);
lean_ctor_set(v___x_1606_, 9, v___x_1604_);
lean_ctor_set(v___x_1606_, 10, v___x_1567_);
lean_ctor_set(v___x_1606_, 11, v___x_1605_);
v___x_1607_ = lean_st_mk_ref(v___x_1606_);
lean_inc(v_cancelTk_x3f_1595_);
lean_inc(v_ref_1593_);
lean_inc(v_currMacroScope_1594_);
lean_inc(v_currRecDepth_1591_);
lean_inc_ref(v_fileMap_1590_);
lean_inc_ref(v_fileName_1589_);
v___x_1608_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_1608_, 0, v_fileName_1589_);
lean_ctor_set(v___x_1608_, 1, v_fileMap_1590_);
lean_ctor_set(v___x_1608_, 2, v_currRecDepth_1591_);
lean_ctor_set(v___x_1608_, 3, v___x_1565_);
lean_ctor_set(v___x_1608_, 4, v___x_1570_);
lean_ctor_set(v___x_1608_, 5, v___x_1605_);
lean_ctor_set(v___x_1608_, 6, v_currMacroScope_1594_);
lean_ctor_set(v___x_1608_, 7, v_ref_1593_);
lean_ctor_set(v___x_1608_, 8, v___x_1605_);
lean_ctor_set(v___x_1608_, 9, v_cancelTk_x3f_1595_);
lean_ctor_set_uint8(v___x_1608_, sizeof(void*)*10, v_suppressElabErrors_1596_);
v___x_1609_ = lp_mathlib_Mathlib_Command_MinImports_getVisited(v_name_1485_, v___x_1608_, v___x_1607_);
lean_dec_ref_known(v___x_1608_, 10);
if (lean_obj_tag(v___x_1609_) == 0)
{
lean_object* v_a_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; size_t v_sz_1614_; size_t v___x_1615_; lean_object* v___x_1616_; 
v_a_1610_ = lean_ctor_get(v___x_1609_, 0);
lean_inc(v_a_1610_);
lean_dec_ref_known(v___x_1609_, 1);
v___x_1611_ = lean_st_ref_get(v___x_1607_);
lean_dec(v___x_1607_);
lean_dec(v___x_1611_);
v___x_1612_ = l_Lean_Environment_header(v_env_1581_);
v___x_1613_ = l_Lean_EnvironmentHeader_moduleNames(v___x_1612_);
v_sz_1614_ = lean_array_size(v___x_1613_);
v___x_1615_ = ((size_t)0ULL);
v___x_1616_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___redArg(v_env_1581_, v___x_1613_, v_sz_1614_, v___x_1615_, v___x_1566_);
lean_dec_ref(v___x_1613_);
if (lean_obj_tag(v___x_1616_) == 0)
{
lean_object* v_a_1617_; lean_object* v___x_1618_; lean_object* v_a_1619_; lean_object* v_a_1620_; 
v_a_1617_ = lean_ctor_get(v___x_1616_, 0);
lean_inc(v_a_1617_);
lean_dec_ref_known(v___x_1616_, 1);
v___x_1618_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg(v_env_1581_, v_a_1617_, v___x_1583_, v_a_1610_);
lean_dec(v_a_1610_);
lean_dec(v_a_1617_);
lean_dec_ref(v_env_1581_);
v_a_1619_ = lean_ctor_get(v___x_1618_, 0);
lean_inc(v_a_1619_);
lean_dec_ref(v___x_1618_);
v_a_1620_ = lean_ctor_get(v_a_1619_, 0);
lean_inc(v_a_1620_);
lean_dec(v_a_1619_);
v___y_1536_ = v_a_1562_;
v___y_1537_ = v___x_1565_;
v___y_1538_ = v___x_1587_;
v___y_1539_ = v___y_1550_;
v___y_1540_ = v___y_1560_;
v___y_1541_ = v_a_1578_;
v___y_1542_ = v___x_1598_;
v_a_1543_ = v_a_1620_;
goto v___jp_1535_;
}
else
{
lean_object* v_a_1621_; lean_object* v___x_1623_; uint8_t v_isShared_1624_; uint8_t v_isSharedCheck_1628_; 
lean_dec(v_a_1610_);
lean_dec_ref(v_env_1581_);
lean_dec(v_a_1578_);
lean_dec(v_a_1562_);
v_a_1621_ = lean_ctor_get(v___x_1616_, 0);
v_isSharedCheck_1628_ = !lean_is_exclusive(v___x_1616_);
if (v_isSharedCheck_1628_ == 0)
{
v___x_1623_ = v___x_1616_;
v_isShared_1624_ = v_isSharedCheck_1628_;
goto v_resetjp_1622_;
}
else
{
lean_inc(v_a_1621_);
lean_dec(v___x_1616_);
v___x_1623_ = lean_box(0);
v_isShared_1624_ = v_isSharedCheck_1628_;
goto v_resetjp_1622_;
}
v_resetjp_1622_:
{
lean_object* v___x_1626_; 
if (v_isShared_1624_ == 0)
{
v___x_1626_ = v___x_1623_;
goto v_reusejp_1625_;
}
else
{
lean_object* v_reuseFailAlloc_1627_; 
v_reuseFailAlloc_1627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1627_, 0, v_a_1621_);
v___x_1626_ = v_reuseFailAlloc_1627_;
goto v_reusejp_1625_;
}
v_reusejp_1625_:
{
return v___x_1626_;
}
}
}
}
else
{
lean_object* v_a_1629_; lean_object* v___x_1631_; uint8_t v_isShared_1632_; uint8_t v_isSharedCheck_1636_; 
lean_dec(v___x_1607_);
lean_dec_ref(v_env_1581_);
lean_dec(v_a_1578_);
lean_dec(v_a_1562_);
v_a_1629_ = lean_ctor_get(v___x_1609_, 0);
v_isSharedCheck_1636_ = !lean_is_exclusive(v___x_1609_);
if (v_isSharedCheck_1636_ == 0)
{
v___x_1631_ = v___x_1609_;
v_isShared_1632_ = v_isSharedCheck_1636_;
goto v_resetjp_1630_;
}
else
{
lean_inc(v_a_1629_);
lean_dec(v___x_1609_);
v___x_1631_ = lean_box(0);
v_isShared_1632_ = v_isSharedCheck_1636_;
goto v_resetjp_1630_;
}
v_resetjp_1630_:
{
lean_object* v___x_1634_; 
if (v_isShared_1632_ == 0)
{
v___x_1634_ = v___x_1631_;
goto v_reusejp_1633_;
}
else
{
lean_object* v_reuseFailAlloc_1635_; 
v_reuseFailAlloc_1635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1635_, 0, v_a_1629_);
v___x_1634_ = v_reuseFailAlloc_1635_;
goto v_reusejp_1633_;
}
v_reusejp_1633_:
{
return v___x_1634_;
}
}
}
}
else
{
lean_object* v_a_1637_; lean_object* v___x_1639_; uint8_t v_isShared_1640_; uint8_t v_isSharedCheck_1644_; 
lean_dec(v_a_1562_);
lean_dec(v_name_1485_);
v_a_1637_ = lean_ctor_get(v___x_1575_, 0);
v_isSharedCheck_1644_ = !lean_is_exclusive(v___x_1575_);
if (v_isSharedCheck_1644_ == 0)
{
v___x_1639_ = v___x_1575_;
v_isShared_1640_ = v_isSharedCheck_1644_;
goto v_resetjp_1638_;
}
else
{
lean_inc(v_a_1637_);
lean_dec(v___x_1575_);
v___x_1639_ = lean_box(0);
v_isShared_1640_ = v_isSharedCheck_1644_;
goto v_resetjp_1638_;
}
v_resetjp_1638_:
{
lean_object* v___x_1642_; 
if (v_isShared_1640_ == 0)
{
v___x_1642_ = v___x_1639_;
goto v_reusejp_1641_;
}
else
{
lean_object* v_reuseFailAlloc_1643_; 
v_reuseFailAlloc_1643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1643_, 0, v_a_1637_);
v___x_1642_ = v_reuseFailAlloc_1643_;
goto v_reusejp_1641_;
}
v_reusejp_1641_:
{
return v___x_1642_;
}
}
}
}
else
{
lean_object* v_a_1645_; lean_object* v___x_1647_; uint8_t v_isShared_1648_; uint8_t v_isSharedCheck_1652_; 
lean_dec(v_a_1562_);
lean_dec(v_name_1485_);
v_a_1645_ = lean_ctor_get(v___x_1563_, 0);
v_isSharedCheck_1652_ = !lean_is_exclusive(v___x_1563_);
if (v_isSharedCheck_1652_ == 0)
{
v___x_1647_ = v___x_1563_;
v_isShared_1648_ = v_isSharedCheck_1652_;
goto v_resetjp_1646_;
}
else
{
lean_inc(v_a_1645_);
lean_dec(v___x_1563_);
v___x_1647_ = lean_box(0);
v_isShared_1648_ = v_isSharedCheck_1652_;
goto v_resetjp_1646_;
}
v_resetjp_1646_:
{
lean_object* v___x_1650_; 
if (v_isShared_1648_ == 0)
{
v___x_1650_ = v___x_1647_;
goto v_reusejp_1649_;
}
else
{
lean_object* v_reuseFailAlloc_1651_; 
v_reuseFailAlloc_1651_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1651_, 0, v_a_1645_);
v___x_1650_ = v_reuseFailAlloc_1651_;
goto v_reusejp_1649_;
}
v_reusejp_1649_:
{
return v___x_1650_;
}
}
}
}
else
{
lean_object* v_a_1653_; lean_object* v___x_1655_; uint8_t v_isShared_1656_; uint8_t v_isSharedCheck_1660_; 
lean_dec(v_name_1485_);
v_a_1653_ = lean_ctor_get(v___x_1561_, 0);
v_isSharedCheck_1660_ = !lean_is_exclusive(v___x_1561_);
if (v_isSharedCheck_1660_ == 0)
{
v___x_1655_ = v___x_1561_;
v_isShared_1656_ = v_isSharedCheck_1660_;
goto v_resetjp_1654_;
}
else
{
lean_inc(v_a_1653_);
lean_dec(v___x_1561_);
v___x_1655_ = lean_box(0);
v_isShared_1656_ = v_isSharedCheck_1660_;
goto v_resetjp_1654_;
}
v_resetjp_1654_:
{
lean_object* v___x_1658_; 
if (v_isShared_1656_ == 0)
{
v___x_1658_ = v___x_1655_;
goto v_reusejp_1657_;
}
else
{
lean_object* v_reuseFailAlloc_1659_; 
v_reuseFailAlloc_1659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1659_, 0, v_a_1653_);
v___x_1658_ = v_reuseFailAlloc_1659_;
goto v_reusejp_1657_;
}
v_reusejp_1657_:
{
return v___x_1658_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___boxed(lean_object* v_g_1753_, lean_object* v_name_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_){
_start:
{
lean_object* v_res_1762_; 
v_res_1762_ = lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1(v_g_1753_, v_name_1754_, v___y_1755_, v___y_1756_, v___y_1757_, v___y_1758_, v___y_1759_, v___y_1760_);
lean_dec(v___y_1760_);
lean_dec_ref(v___y_1759_);
lean_dec(v___y_1758_);
lean_dec_ref(v___y_1757_);
lean_dec(v___y_1756_);
lean_dec_ref(v___y_1755_);
return v_res_1762_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__0(void){
_start:
{
lean_object* v___x_1763_; 
v___x_1763_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1763_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1(void){
_start:
{
lean_object* v___x_1764_; lean_object* v___x_1765_; 
v___x_1764_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__0, &lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__0_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__0);
v___x_1765_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1765_, 0, v___x_1764_);
return v___x_1765_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2(void){
_start:
{
lean_object* v___x_1766_; lean_object* v___x_1767_; 
v___x_1766_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1);
v___x_1767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1767_, 0, v___x_1766_);
lean_ctor_set(v___x_1767_, 1, v___x_1766_);
return v___x_1767_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3(void){
_start:
{
lean_object* v___x_1768_; lean_object* v___x_1769_; 
v___x_1768_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__1);
v___x_1769_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1769_, 0, v___x_1768_);
lean_ctor_set(v___x_1769_, 1, v___x_1768_);
lean_ctor_set(v___x_1769_, 2, v___x_1768_);
lean_ctor_set(v___x_1769_, 3, v___x_1768_);
lean_ctor_set(v___x_1769_, 4, v___x_1768_);
lean_ctor_set(v___x_1769_, 5, v___x_1768_);
return v___x_1769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg(lean_object* v_env_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_){
_start:
{
lean_object* v___x_1774_; lean_object* v_nextMacroScope_1775_; lean_object* v_ngen_1776_; lean_object* v_auxDeclNGen_1777_; lean_object* v_traceState_1778_; lean_object* v_messages_1779_; lean_object* v_infoState_1780_; lean_object* v_snapshotTasks_1781_; lean_object* v___x_1783_; uint8_t v_isShared_1784_; uint8_t v_isSharedCheck_1807_; 
v___x_1774_ = lean_st_ref_take(v___y_1772_);
v_nextMacroScope_1775_ = lean_ctor_get(v___x_1774_, 1);
v_ngen_1776_ = lean_ctor_get(v___x_1774_, 2);
v_auxDeclNGen_1777_ = lean_ctor_get(v___x_1774_, 3);
v_traceState_1778_ = lean_ctor_get(v___x_1774_, 4);
v_messages_1779_ = lean_ctor_get(v___x_1774_, 6);
v_infoState_1780_ = lean_ctor_get(v___x_1774_, 7);
v_snapshotTasks_1781_ = lean_ctor_get(v___x_1774_, 8);
v_isSharedCheck_1807_ = !lean_is_exclusive(v___x_1774_);
if (v_isSharedCheck_1807_ == 0)
{
lean_object* v_unused_1808_; lean_object* v_unused_1809_; 
v_unused_1808_ = lean_ctor_get(v___x_1774_, 5);
lean_dec(v_unused_1808_);
v_unused_1809_ = lean_ctor_get(v___x_1774_, 0);
lean_dec(v_unused_1809_);
v___x_1783_ = v___x_1774_;
v_isShared_1784_ = v_isSharedCheck_1807_;
goto v_resetjp_1782_;
}
else
{
lean_inc(v_snapshotTasks_1781_);
lean_inc(v_infoState_1780_);
lean_inc(v_messages_1779_);
lean_inc(v_traceState_1778_);
lean_inc(v_auxDeclNGen_1777_);
lean_inc(v_ngen_1776_);
lean_inc(v_nextMacroScope_1775_);
lean_dec(v___x_1774_);
v___x_1783_ = lean_box(0);
v_isShared_1784_ = v_isSharedCheck_1807_;
goto v_resetjp_1782_;
}
v_resetjp_1782_:
{
lean_object* v___x_1785_; lean_object* v___x_1787_; 
v___x_1785_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2);
if (v_isShared_1784_ == 0)
{
lean_ctor_set(v___x_1783_, 5, v___x_1785_);
lean_ctor_set(v___x_1783_, 0, v_env_1770_);
v___x_1787_ = v___x_1783_;
goto v_reusejp_1786_;
}
else
{
lean_object* v_reuseFailAlloc_1806_; 
v_reuseFailAlloc_1806_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1806_, 0, v_env_1770_);
lean_ctor_set(v_reuseFailAlloc_1806_, 1, v_nextMacroScope_1775_);
lean_ctor_set(v_reuseFailAlloc_1806_, 2, v_ngen_1776_);
lean_ctor_set(v_reuseFailAlloc_1806_, 3, v_auxDeclNGen_1777_);
lean_ctor_set(v_reuseFailAlloc_1806_, 4, v_traceState_1778_);
lean_ctor_set(v_reuseFailAlloc_1806_, 5, v___x_1785_);
lean_ctor_set(v_reuseFailAlloc_1806_, 6, v_messages_1779_);
lean_ctor_set(v_reuseFailAlloc_1806_, 7, v_infoState_1780_);
lean_ctor_set(v_reuseFailAlloc_1806_, 8, v_snapshotTasks_1781_);
v___x_1787_ = v_reuseFailAlloc_1806_;
goto v_reusejp_1786_;
}
v_reusejp_1786_:
{
lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v_mctx_1790_; lean_object* v_zetaDeltaFVarIds_1791_; lean_object* v_postponed_1792_; lean_object* v_diag_1793_; lean_object* v___x_1795_; uint8_t v_isShared_1796_; uint8_t v_isSharedCheck_1804_; 
v___x_1788_ = lean_st_ref_set(v___y_1772_, v___x_1787_);
v___x_1789_ = lean_st_ref_take(v___y_1771_);
v_mctx_1790_ = lean_ctor_get(v___x_1789_, 0);
v_zetaDeltaFVarIds_1791_ = lean_ctor_get(v___x_1789_, 2);
v_postponed_1792_ = lean_ctor_get(v___x_1789_, 3);
v_diag_1793_ = lean_ctor_get(v___x_1789_, 4);
v_isSharedCheck_1804_ = !lean_is_exclusive(v___x_1789_);
if (v_isSharedCheck_1804_ == 0)
{
lean_object* v_unused_1805_; 
v_unused_1805_ = lean_ctor_get(v___x_1789_, 1);
lean_dec(v_unused_1805_);
v___x_1795_ = v___x_1789_;
v_isShared_1796_ = v_isSharedCheck_1804_;
goto v_resetjp_1794_;
}
else
{
lean_inc(v_diag_1793_);
lean_inc(v_postponed_1792_);
lean_inc(v_zetaDeltaFVarIds_1791_);
lean_inc(v_mctx_1790_);
lean_dec(v___x_1789_);
v___x_1795_ = lean_box(0);
v_isShared_1796_ = v_isSharedCheck_1804_;
goto v_resetjp_1794_;
}
v_resetjp_1794_:
{
lean_object* v___x_1797_; lean_object* v___x_1799_; 
v___x_1797_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3, &lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3);
if (v_isShared_1796_ == 0)
{
lean_ctor_set(v___x_1795_, 1, v___x_1797_);
v___x_1799_ = v___x_1795_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1803_; 
v_reuseFailAlloc_1803_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1803_, 0, v_mctx_1790_);
lean_ctor_set(v_reuseFailAlloc_1803_, 1, v___x_1797_);
lean_ctor_set(v_reuseFailAlloc_1803_, 2, v_zetaDeltaFVarIds_1791_);
lean_ctor_set(v_reuseFailAlloc_1803_, 3, v_postponed_1792_);
lean_ctor_set(v_reuseFailAlloc_1803_, 4, v_diag_1793_);
v___x_1799_ = v_reuseFailAlloc_1803_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; 
v___x_1800_ = lean_st_ref_set(v___y_1771_, v___x_1799_);
v___x_1801_ = lean_box(0);
v___x_1802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1802_, 0, v___x_1801_);
return v___x_1802_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___boxed(lean_object* v_env_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_){
_start:
{
lean_object* v_res_1814_; 
v_res_1814_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg(v_env_1810_, v___y_1811_, v___y_1812_);
lean_dec(v___y_1812_);
lean_dec(v___y_1811_);
return v_res_1814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___redArg(lean_object* v_env_1815_, lean_object* v_x_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_){
_start:
{
lean_object* v___x_1824_; lean_object* v_env_1825_; lean_object* v_a_1827_; lean_object* v___x_1837_; lean_object* v___x_1838_; 
v___x_1824_ = lean_st_ref_get(v___y_1822_);
v_env_1825_ = lean_ctor_get(v___x_1824_, 0);
lean_inc_ref(v_env_1825_);
lean_dec(v___x_1824_);
v___x_1837_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg(v_env_1815_, v___y_1820_, v___y_1822_);
lean_dec_ref(v___x_1837_);
lean_inc(v___y_1822_);
lean_inc_ref(v___y_1821_);
lean_inc(v___y_1820_);
lean_inc_ref(v___y_1819_);
lean_inc(v___y_1818_);
lean_inc_ref(v___y_1817_);
v___x_1838_ = lean_apply_7(v_x_1816_, v___y_1817_, v___y_1818_, v___y_1819_, v___y_1820_, v___y_1821_, v___y_1822_, lean_box(0));
if (lean_obj_tag(v___x_1838_) == 0)
{
lean_object* v_a_1839_; lean_object* v___x_1840_; lean_object* v___x_1842_; uint8_t v_isShared_1843_; uint8_t v_isSharedCheck_1847_; 
v_a_1839_ = lean_ctor_get(v___x_1838_, 0);
lean_inc(v_a_1839_);
lean_dec_ref_known(v___x_1838_, 1);
v___x_1840_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg(v_env_1825_, v___y_1820_, v___y_1822_);
v_isSharedCheck_1847_ = !lean_is_exclusive(v___x_1840_);
if (v_isSharedCheck_1847_ == 0)
{
lean_object* v_unused_1848_; 
v_unused_1848_ = lean_ctor_get(v___x_1840_, 0);
lean_dec(v_unused_1848_);
v___x_1842_ = v___x_1840_;
v_isShared_1843_ = v_isSharedCheck_1847_;
goto v_resetjp_1841_;
}
else
{
lean_dec(v___x_1840_);
v___x_1842_ = lean_box(0);
v_isShared_1843_ = v_isSharedCheck_1847_;
goto v_resetjp_1841_;
}
v_resetjp_1841_:
{
lean_object* v___x_1845_; 
if (v_isShared_1843_ == 0)
{
lean_ctor_set(v___x_1842_, 0, v_a_1839_);
v___x_1845_ = v___x_1842_;
goto v_reusejp_1844_;
}
else
{
lean_object* v_reuseFailAlloc_1846_; 
v_reuseFailAlloc_1846_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1846_, 0, v_a_1839_);
v___x_1845_ = v_reuseFailAlloc_1846_;
goto v_reusejp_1844_;
}
v_reusejp_1844_:
{
return v___x_1845_;
}
}
}
else
{
lean_object* v_a_1849_; 
v_a_1849_ = lean_ctor_get(v___x_1838_, 0);
lean_inc(v_a_1849_);
lean_dec_ref_known(v___x_1838_, 1);
v_a_1827_ = v_a_1849_;
goto v___jp_1826_;
}
v___jp_1826_:
{
lean_object* v___x_1828_; lean_object* v___x_1830_; uint8_t v_isShared_1831_; uint8_t v_isSharedCheck_1835_; 
v___x_1828_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg(v_env_1825_, v___y_1820_, v___y_1822_);
v_isSharedCheck_1835_ = !lean_is_exclusive(v___x_1828_);
if (v_isSharedCheck_1835_ == 0)
{
lean_object* v_unused_1836_; 
v_unused_1836_ = lean_ctor_get(v___x_1828_, 0);
lean_dec(v_unused_1836_);
v___x_1830_ = v___x_1828_;
v_isShared_1831_ = v_isSharedCheck_1835_;
goto v_resetjp_1829_;
}
else
{
lean_dec(v___x_1828_);
v___x_1830_ = lean_box(0);
v_isShared_1831_ = v_isSharedCheck_1835_;
goto v_resetjp_1829_;
}
v_resetjp_1829_:
{
lean_object* v___x_1833_; 
if (v_isShared_1831_ == 0)
{
lean_ctor_set_tag(v___x_1830_, 1);
lean_ctor_set(v___x_1830_, 0, v_a_1827_);
v___x_1833_ = v___x_1830_;
goto v_reusejp_1832_;
}
else
{
lean_object* v_reuseFailAlloc_1834_; 
v_reuseFailAlloc_1834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1834_, 0, v_a_1827_);
v___x_1833_ = v_reuseFailAlloc_1834_;
goto v_reusejp_1832_;
}
v_reusejp_1832_:
{
return v___x_1833_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___redArg___boxed(lean_object* v_env_1850_, lean_object* v_x_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_){
_start:
{
lean_object* v_res_1859_; 
v_res_1859_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___redArg(v_env_1850_, v_x_1851_, v___y_1852_, v___y_1853_, v___y_1854_, v___y_1855_, v___y_1856_, v___y_1857_);
lean_dec(v___y_1857_);
lean_dec_ref(v___y_1856_);
lean_dec(v___y_1855_);
lean_dec_ref(v___y_1854_);
lean_dec(v___y_1853_);
lean_dec_ref(v___y_1852_);
return v_res_1859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature(lean_object* v_name_1860_, lean_object* v_g_1861_, lean_object* v_a_1862_, lean_object* v_a_1863_, lean_object* v_a_1864_, lean_object* v_a_1865_, lean_object* v_a_1866_, lean_object* v_a_1867_){
_start:
{
lean_object* v___x_1869_; lean_object* v_env_1870_; lean_object* v___f_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; 
v___x_1869_ = lean_st_ref_get(v_a_1867_);
v_env_1870_ = lean_ctor_get(v___x_1869_, 0);
lean_inc_ref(v_env_1870_);
lean_dec(v___x_1869_);
v___f_1871_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___boxed), 9, 2);
lean_closure_set(v___f_1871_, 0, v_g_1861_);
lean_closure_set(v___f_1871_, 1, v_name_1860_);
v___x_1872_ = lean_alloc_closure((void*)(lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__13___boxed), 9, 2);
lean_closure_set(v___x_1872_, 0, lean_box(0));
lean_closure_set(v___x_1872_, 1, v___f_1871_);
v___x_1873_ = l_Lean_Environment_unlockAsync(v_env_1870_);
v___x_1874_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___redArg(v___x_1873_, v___x_1872_, v_a_1862_, v_a_1863_, v_a_1864_, v_a_1865_, v_a_1866_, v_a_1867_);
return v___x_1874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___boxed(lean_object* v_name_1875_, lean_object* v_g_1876_, lean_object* v_a_1877_, lean_object* v_a_1878_, lean_object* v_a_1879_, lean_object* v_a_1880_, lean_object* v_a_1881_, lean_object* v_a_1882_, lean_object* v_a_1883_){
_start:
{
lean_object* v_res_1884_; 
v_res_1884_ = lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature(v_name_1875_, v_g_1876_, v_a_1877_, v_a_1878_, v_a_1879_, v_a_1880_, v_a_1881_, v_a_1882_);
lean_dec(v_a_1882_);
lean_dec_ref(v_a_1881_);
lean_dec(v_a_1880_);
lean_dec_ref(v_a_1879_);
lean_dec(v_a_1878_);
lean_dec_ref(v_a_1877_);
return v_res_1884_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1(lean_object* v_00_u03b2_1885_, lean_object* v_m_1886_, lean_object* v_a_1887_){
_start:
{
uint8_t v___x_1888_; 
v___x_1888_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___redArg(v_m_1886_, v_a_1887_);
return v___x_1888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1___boxed(lean_object* v_00_u03b2_1889_, lean_object* v_m_1890_, lean_object* v_a_1891_){
_start:
{
uint8_t v_res_1892_; lean_object* v_r_1893_; 
v_res_1892_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1(v_00_u03b2_1889_, v_m_1890_, v_a_1891_);
lean_dec(v_a_1891_);
lean_dec_ref(v_m_1890_);
v_r_1893_ = lean_box(v_res_1892_);
return v_r_1893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4(lean_object* v_00_u03b2_1894_, lean_object* v_k_1895_, lean_object* v_t_1896_, lean_object* v_h_1897_){
_start:
{
lean_object* v___x_1898_; 
v___x_1898_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___redArg(v_k_1895_, v_t_1896_);
return v___x_1898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4___boxed(lean_object* v_00_u03b2_1899_, lean_object* v_k_1900_, lean_object* v_t_1901_, lean_object* v_h_1902_){
_start:
{
lean_object* v_res_1903_; 
v_res_1903_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__4(v_00_u03b2_1899_, v_k_1900_, v_t_1901_, v_h_1902_);
lean_dec(v_k_1900_);
return v_res_1903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__5(lean_object* v_init_1904_, lean_object* v_t_1905_){
_start:
{
lean_object* v___x_1906_; 
v___x_1906_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__5_spec__6(v_init_1904_, v_t_1905_);
return v___x_1906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6(lean_object* v_n_1907_, lean_object* v_as_1908_, lean_object* v_lo_1909_, lean_object* v_hi_1910_, lean_object* v_w_1911_, lean_object* v_hlo_1912_, lean_object* v_hhi_1913_){
_start:
{
lean_object* v___x_1914_; 
v___x_1914_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___redArg(v_n_1907_, v_as_1908_, v_lo_1909_, v_hi_1910_);
return v___x_1914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6___boxed(lean_object* v_n_1915_, lean_object* v_as_1916_, lean_object* v_lo_1917_, lean_object* v_hi_1918_, lean_object* v_w_1919_, lean_object* v_hlo_1920_, lean_object* v_hhi_1921_){
_start:
{
lean_object* v_res_1922_; 
v_res_1922_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6(v_n_1915_, v_as_1916_, v_lo_1917_, v_hi_1918_, v_w_1919_, v_hlo_1920_, v_hhi_1921_);
lean_dec(v_hi_1918_);
lean_dec(v_n_1915_);
return v_res_1922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7(lean_object* v_00_u03b2_1923_, lean_object* v_m_1924_, lean_object* v_a_1925_, lean_object* v_b_1926_){
_start:
{
lean_object* v___x_1927_; 
v___x_1927_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7___redArg(v_m_1924_, v_a_1925_, v_b_1926_);
return v___x_1927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8(lean_object* v___x_1928_, lean_object* v_as_1929_, size_t v_sz_1930_, size_t v_i_1931_, lean_object* v_b_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_){
_start:
{
lean_object* v___x_1940_; 
v___x_1940_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___redArg(v___x_1928_, v_as_1929_, v_sz_1930_, v_i_1931_, v_b_1932_);
return v___x_1940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8___boxed(lean_object* v___x_1941_, lean_object* v_as_1942_, lean_object* v_sz_1943_, lean_object* v_i_1944_, lean_object* v_b_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_){
_start:
{
size_t v_sz_boxed_1953_; size_t v_i_boxed_1954_; lean_object* v_res_1955_; 
v_sz_boxed_1953_ = lean_unbox_usize(v_sz_1943_);
lean_dec(v_sz_1943_);
v_i_boxed_1954_ = lean_unbox_usize(v_i_1944_);
lean_dec(v_i_1944_);
v_res_1955_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__8(v___x_1941_, v_as_1942_, v_sz_boxed_1953_, v_i_boxed_1954_, v_b_1945_, v___y_1946_, v___y_1947_, v___y_1948_, v___y_1949_, v___y_1950_, v___y_1951_);
lean_dec(v___y_1951_);
lean_dec_ref(v___y_1950_);
lean_dec(v___y_1949_);
lean_dec_ref(v___y_1948_);
lean_dec(v___y_1947_);
lean_dec_ref(v___y_1946_);
lean_dec_ref(v_as_1942_);
lean_dec_ref(v___x_1941_);
return v_res_1955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9(lean_object* v_00_u03b2_1956_, lean_object* v_m_1957_, lean_object* v_a_1958_){
_start:
{
lean_object* v___x_1959_; 
v___x_1959_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___redArg(v_m_1957_, v_a_1958_);
return v___x_1959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9___boxed(lean_object* v_00_u03b2_1960_, lean_object* v_m_1961_, lean_object* v_a_1962_){
_start:
{
lean_object* v_res_1963_; 
v_res_1963_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9(v_00_u03b2_1960_, v_m_1961_, v_a_1962_);
lean_dec(v_a_1962_);
lean_dec_ref(v_m_1961_);
return v_res_1963_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11(lean_object* v___x_1964_, lean_object* v_a_1965_, lean_object* v_init_1966_, lean_object* v_x_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_){
_start:
{
lean_object* v___x_1975_; 
v___x_1975_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___redArg(v___x_1964_, v_a_1965_, v_init_1966_, v_x_1967_);
return v___x_1975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11___boxed(lean_object* v___x_1976_, lean_object* v_a_1977_, lean_object* v_init_1978_, lean_object* v_x_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_){
_start:
{
lean_object* v_res_1987_; 
v_res_1987_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__11(v___x_1976_, v_a_1977_, v_init_1978_, v_x_1979_, v___y_1980_, v___y_1981_, v___y_1982_, v___y_1983_, v___y_1984_, v___y_1985_);
lean_dec(v___y_1985_);
lean_dec_ref(v___y_1984_);
lean_dec(v___y_1983_);
lean_dec_ref(v___y_1982_);
lean_dec(v___y_1981_);
lean_dec_ref(v___y_1980_);
lean_dec(v_x_1979_);
lean_dec_ref(v_a_1977_);
lean_dec_ref(v___x_1976_);
return v_res_1987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12(lean_object* v_00_u03b1_1988_, lean_object* v_msg_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_){
_start:
{
lean_object* v___x_1997_; 
v___x_1997_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___redArg(v_msg_1989_, v___y_1990_, v___y_1991_, v___y_1992_, v___y_1993_, v___y_1994_, v___y_1995_);
return v___x_1997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12___boxed(lean_object* v_00_u03b1_1998_, lean_object* v_msg_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_){
_start:
{
lean_object* v_res_2007_; 
v_res_2007_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12(v_00_u03b1_1998_, v_msg_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_);
lean_dec(v___y_2005_);
lean_dec_ref(v___y_2004_);
lean_dec(v___y_2003_);
lean_dec_ref(v___y_2002_);
lean_dec(v___y_2001_);
lean_dec_ref(v___y_2000_);
return v_res_2007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22(lean_object* v_env_2008_, lean_object* v___y_2009_, lean_object* v___y_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_){
_start:
{
lean_object* v___x_2016_; 
v___x_2016_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg(v_env_2008_, v___y_2012_, v___y_2014_);
return v___x_2016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___boxed(lean_object* v_env_2017_, lean_object* v___y_2018_, lean_object* v___y_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_, lean_object* v___y_2023_, lean_object* v___y_2024_){
_start:
{
lean_object* v_res_2025_; 
v_res_2025_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22(v_env_2017_, v___y_2018_, v___y_2019_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_);
lean_dec(v___y_2023_);
lean_dec_ref(v___y_2022_);
lean_dec(v___y_2021_);
lean_dec_ref(v___y_2020_);
lean_dec(v___y_2019_);
lean_dec_ref(v___y_2018_);
return v_res_2025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14(lean_object* v_00_u03b1_2026_, lean_object* v_env_2027_, lean_object* v_x_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_, lean_object* v___y_2034_){
_start:
{
lean_object* v___x_2036_; 
v___x_2036_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___redArg(v_env_2027_, v_x_2028_, v___y_2029_, v___y_2030_, v___y_2031_, v___y_2032_, v___y_2033_, v___y_2034_);
return v___x_2036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14___boxed(lean_object* v_00_u03b1_2037_, lean_object* v_env_2038_, lean_object* v_x_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_){
_start:
{
lean_object* v_res_2047_; 
v_res_2047_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14(v_00_u03b1_2037_, v_env_2038_, v_x_2039_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_, v___y_2044_, v___y_2045_);
lean_dec(v___y_2045_);
lean_dec_ref(v___y_2044_);
lean_dec(v___y_2043_);
lean_dec_ref(v___y_2042_);
lean_dec(v___y_2041_);
lean_dec_ref(v___y_2040_);
return v_res_2047_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1(lean_object* v_00_u03b2_2048_, lean_object* v_a_2049_, lean_object* v_x_2050_){
_start:
{
uint8_t v___x_2051_; 
v___x_2051_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___redArg(v_a_2049_, v_x_2050_);
return v___x_2051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1___boxed(lean_object* v_00_u03b2_2052_, lean_object* v_a_2053_, lean_object* v_x_2054_){
_start:
{
uint8_t v_res_2055_; lean_object* v_r_2056_; 
v_res_2055_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__1_spec__1(v_00_u03b2_2052_, v_a_2053_, v_x_2054_);
lean_dec(v_x_2054_);
lean_dec(v_a_2053_);
v_r_2056_ = lean_box(v_res_2055_);
return v_r_2056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8(lean_object* v_n_2057_, lean_object* v_lo_2058_, lean_object* v_hi_2059_, lean_object* v_hhi_2060_, lean_object* v_pivot_2061_, lean_object* v_as_2062_, lean_object* v_i_2063_, lean_object* v_k_2064_, lean_object* v_ilo_2065_, lean_object* v_ik_2066_, lean_object* v_w_2067_){
_start:
{
lean_object* v___x_2068_; 
v___x_2068_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___redArg(v_hi_2059_, v_pivot_2061_, v_as_2062_, v_i_2063_, v_k_2064_);
return v___x_2068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8___boxed(lean_object* v_n_2069_, lean_object* v_lo_2070_, lean_object* v_hi_2071_, lean_object* v_hhi_2072_, lean_object* v_pivot_2073_, lean_object* v_as_2074_, lean_object* v_i_2075_, lean_object* v_k_2076_, lean_object* v_ilo_2077_, lean_object* v_ik_2078_, lean_object* v_w_2079_){
_start:
{
lean_object* v_res_2080_; 
v_res_2080_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__6_spec__8(v_n_2069_, v_lo_2070_, v_hi_2071_, v_hhi_2072_, v_pivot_2073_, v_as_2074_, v_i_2075_, v_k_2076_, v_ilo_2077_, v_ik_2078_, v_w_2079_);
lean_dec(v_pivot_2073_);
lean_dec(v_hi_2071_);
lean_dec(v_lo_2070_);
lean_dec(v_n_2069_);
return v_res_2080_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10(lean_object* v_00_u03b2_2081_, lean_object* v_a_2082_, lean_object* v_x_2083_){
_start:
{
uint8_t v___x_2084_; 
v___x_2084_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___redArg(v_a_2082_, v_x_2083_);
return v___x_2084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10___boxed(lean_object* v_00_u03b2_2085_, lean_object* v_a_2086_, lean_object* v_x_2087_){
_start:
{
uint8_t v_res_2088_; lean_object* v_r_2089_; 
v_res_2088_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__10(v_00_u03b2_2085_, v_a_2086_, v_x_2087_);
lean_dec(v_x_2087_);
lean_dec(v_a_2086_);
v_r_2089_ = lean_box(v_res_2088_);
return v_r_2089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11(lean_object* v_00_u03b2_2090_, lean_object* v_data_2091_){
_start:
{
lean_object* v___x_2092_; 
v___x_2092_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11___redArg(v_data_2091_);
return v___x_2092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__12(lean_object* v_00_u03b2_2093_, lean_object* v_a_2094_, lean_object* v_b_2095_, lean_object* v_x_2096_){
_start:
{
lean_object* v___x_2097_; 
v___x_2097_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__12___redArg(v_a_2094_, v_b_2095_, v_x_2096_);
return v___x_2097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15(lean_object* v_00_u03b2_2098_, lean_object* v_a_2099_, lean_object* v_x_2100_){
_start:
{
lean_object* v___x_2101_; 
v___x_2101_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___redArg(v_a_2099_, v_x_2100_);
return v___x_2101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15___boxed(lean_object* v_00_u03b2_2102_, lean_object* v_a_2103_, lean_object* v_x_2104_){
_start:
{
lean_object* v_res_2105_; 
v_res_2105_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__9_spec__15(v_00_u03b2_2102_, v_a_2103_, v_x_2104_);
lean_dec(v_x_2104_);
lean_dec(v_a_2103_);
return v_res_2105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19(lean_object* v_msgData_2106_, lean_object* v_macroStack_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_){
_start:
{
lean_object* v___x_2115_; 
v___x_2115_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___redArg(v_msgData_2106_, v_macroStack_2107_, v___y_2112_);
return v___x_2115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19___boxed(lean_object* v_msgData_2116_, lean_object* v_macroStack_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_){
_start:
{
lean_object* v_res_2125_; 
v_res_2125_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19(v_msgData_2116_, v_macroStack_2117_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
lean_dec(v___y_2123_);
lean_dec_ref(v___y_2122_);
lean_dec(v___y_2121_);
lean_dec_ref(v___y_2120_);
lean_dec(v___y_2119_);
lean_dec_ref(v___y_2118_);
return v_res_2125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14(lean_object* v_00_u03b2_2126_, lean_object* v_i_2127_, lean_object* v_source_2128_, lean_object* v_target_2129_){
_start:
{
lean_object* v___x_2130_; 
v___x_2130_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14___redArg(v_i_2127_, v_source_2128_, v_target_2129_);
return v___x_2130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14_spec__21(lean_object* v_00_u03b2_2131_, lean_object* v_x_2132_, lean_object* v_x_2133_){
_start:
{
lean_object* v___x_2134_; 
v___x_2134_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__7_spec__11_spec__14_spec__21___redArg(v_x_2132_, v_x_2133_);
return v___x_2134_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; 
v___x_2135_ = lean_box(0);
v___x_2136_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2137_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2137_, 0, v___x_2136_);
lean_ctor_set(v___x_2137_, 1, v___x_2135_);
return v___x_2137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2139_; lean_object* v___x_2140_; 
v___x_2139_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___closed__0);
v___x_2140_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2140_, 0, v___x_2139_);
return v___x_2140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg___boxed(lean_object* v___y_2141_){
_start:
{
lean_object* v_res_2142_; 
v_res_2142_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg();
return v_res_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0(lean_object* v_00_u03b1_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_, lean_object* v___y_2150_, lean_object* v___y_2151_){
_start:
{
lean_object* v___x_2153_; 
v___x_2153_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg();
return v___x_2153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___boxed(lean_object* v_00_u03b1_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_, lean_object* v___y_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_){
_start:
{
lean_object* v_res_2164_; 
v_res_2164_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0(v_00_u03b1_2154_, v___y_2155_, v___y_2156_, v___y_2157_, v___y_2158_, v___y_2159_, v___y_2160_, v___y_2161_, v___y_2162_);
lean_dec(v___y_2162_);
lean_dec_ref(v___y_2161_);
lean_dec(v___y_2160_);
lean_dec_ref(v___y_2159_);
lean_dec(v___y_2158_);
lean_dec_ref(v___y_2157_);
lean_dec(v___y_2156_);
lean_dec_ref(v___y_2155_);
return v_res_2164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___redArg(lean_object* v_e_2165_, lean_object* v___y_2166_){
_start:
{
uint8_t v___x_2168_; 
v___x_2168_ = l_Lean_Expr_hasMVar(v_e_2165_);
if (v___x_2168_ == 0)
{
lean_object* v___x_2169_; 
v___x_2169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2169_, 0, v_e_2165_);
return v___x_2169_;
}
else
{
lean_object* v___x_2170_; lean_object* v_mctx_2171_; lean_object* v___x_2172_; lean_object* v_fst_2173_; lean_object* v_snd_2174_; lean_object* v___x_2175_; lean_object* v_cache_2176_; lean_object* v_zetaDeltaFVarIds_2177_; lean_object* v_postponed_2178_; lean_object* v_diag_2179_; lean_object* v___x_2181_; uint8_t v_isShared_2182_; uint8_t v_isSharedCheck_2188_; 
v___x_2170_ = lean_st_ref_get(v___y_2166_);
v_mctx_2171_ = lean_ctor_get(v___x_2170_, 0);
lean_inc_ref(v_mctx_2171_);
lean_dec(v___x_2170_);
v___x_2172_ = l_Lean_instantiateMVarsCore(v_mctx_2171_, v_e_2165_);
v_fst_2173_ = lean_ctor_get(v___x_2172_, 0);
lean_inc(v_fst_2173_);
v_snd_2174_ = lean_ctor_get(v___x_2172_, 1);
lean_inc(v_snd_2174_);
lean_dec_ref(v___x_2172_);
v___x_2175_ = lean_st_ref_take(v___y_2166_);
v_cache_2176_ = lean_ctor_get(v___x_2175_, 1);
v_zetaDeltaFVarIds_2177_ = lean_ctor_get(v___x_2175_, 2);
v_postponed_2178_ = lean_ctor_get(v___x_2175_, 3);
v_diag_2179_ = lean_ctor_get(v___x_2175_, 4);
v_isSharedCheck_2188_ = !lean_is_exclusive(v___x_2175_);
if (v_isSharedCheck_2188_ == 0)
{
lean_object* v_unused_2189_; 
v_unused_2189_ = lean_ctor_get(v___x_2175_, 0);
lean_dec(v_unused_2189_);
v___x_2181_ = v___x_2175_;
v_isShared_2182_ = v_isSharedCheck_2188_;
goto v_resetjp_2180_;
}
else
{
lean_inc(v_diag_2179_);
lean_inc(v_postponed_2178_);
lean_inc(v_zetaDeltaFVarIds_2177_);
lean_inc(v_cache_2176_);
lean_dec(v___x_2175_);
v___x_2181_ = lean_box(0);
v_isShared_2182_ = v_isSharedCheck_2188_;
goto v_resetjp_2180_;
}
v_resetjp_2180_:
{
lean_object* v___x_2184_; 
if (v_isShared_2182_ == 0)
{
lean_ctor_set(v___x_2181_, 0, v_snd_2174_);
v___x_2184_ = v___x_2181_;
goto v_reusejp_2183_;
}
else
{
lean_object* v_reuseFailAlloc_2187_; 
v_reuseFailAlloc_2187_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2187_, 0, v_snd_2174_);
lean_ctor_set(v_reuseFailAlloc_2187_, 1, v_cache_2176_);
lean_ctor_set(v_reuseFailAlloc_2187_, 2, v_zetaDeltaFVarIds_2177_);
lean_ctor_set(v_reuseFailAlloc_2187_, 3, v_postponed_2178_);
lean_ctor_set(v_reuseFailAlloc_2187_, 4, v_diag_2179_);
v___x_2184_ = v_reuseFailAlloc_2187_;
goto v_reusejp_2183_;
}
v_reusejp_2183_:
{
lean_object* v___x_2185_; lean_object* v___x_2186_; 
v___x_2185_ = lean_st_ref_set(v___y_2166_, v___x_2184_);
v___x_2186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2186_, 0, v_fst_2173_);
return v___x_2186_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___redArg___boxed(lean_object* v_e_2190_, lean_object* v___y_2191_, lean_object* v___y_2192_){
_start:
{
lean_object* v_res_2193_; 
v_res_2193_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___redArg(v_e_2190_, v___y_2191_);
lean_dec(v___y_2191_);
return v_res_2193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1(lean_object* v_e_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_){
_start:
{
lean_object* v___x_2204_; 
v___x_2204_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___redArg(v_e_2194_, v___y_2200_);
return v___x_2204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___boxed(lean_object* v_e_2205_, lean_object* v___y_2206_, lean_object* v___y_2207_, lean_object* v___y_2208_, lean_object* v___y_2209_, lean_object* v___y_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_, lean_object* v___y_2213_, lean_object* v___y_2214_){
_start:
{
lean_object* v_res_2215_; 
v_res_2215_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1(v_e_2205_, v___y_2206_, v___y_2207_, v___y_2208_, v___y_2209_, v___y_2210_, v___y_2211_, v___y_2212_, v___y_2213_);
lean_dec(v___y_2213_);
lean_dec_ref(v___y_2212_);
lean_dec(v___y_2211_);
lean_dec_ref(v___y_2210_);
lean_dec(v___y_2209_);
lean_dec_ref(v___y_2208_);
lean_dec(v___y_2207_);
lean_dec_ref(v___y_2206_);
return v_res_2215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___lam__0(lean_object* v_a_2216_, lean_object* v___y_2217_, lean_object* v___y_2218_, lean_object* v___y_2219_, lean_object* v___y_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v_a_x3f_2224_){
_start:
{
uint8_t v___x_2226_; lean_object* v___x_2227_; 
v___x_2226_ = 0;
v___x_2227_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_2216_, v___x_2226_, v___y_2217_, v___y_2218_, v___y_2219_, v___y_2220_, v___y_2221_, v___y_2222_, v___y_2223_);
return v___x_2227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___lam__0___boxed(lean_object* v_a_2228_, lean_object* v___y_2229_, lean_object* v___y_2230_, lean_object* v___y_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_, lean_object* v___y_2235_, lean_object* v_a_x3f_2236_, lean_object* v___y_2237_){
_start:
{
lean_object* v_res_2238_; 
v_res_2238_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___lam__0(v_a_2228_, v___y_2229_, v___y_2230_, v___y_2231_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_, v_a_x3f_2236_);
lean_dec(v_a_x3f_2236_);
lean_dec(v___y_2235_);
lean_dec_ref(v___y_2234_);
lean_dec(v___y_2233_);
lean_dec_ref(v___y_2232_);
lean_dec(v___y_2231_);
lean_dec_ref(v___y_2230_);
lean_dec(v___y_2229_);
return v_res_2238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg(lean_object* v_x_2239_, lean_object* v___y_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_, lean_object* v___y_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_){
_start:
{
lean_object* v___x_2249_; 
v___x_2249_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_2241_, v___y_2243_, v___y_2245_, v___y_2247_);
if (lean_obj_tag(v___x_2249_) == 0)
{
lean_object* v_a_2250_; lean_object* v_r_2251_; 
v_a_2250_ = lean_ctor_get(v___x_2249_, 0);
lean_inc(v_a_2250_);
lean_dec_ref_known(v___x_2249_, 1);
lean_inc(v___y_2247_);
lean_inc_ref(v___y_2246_);
lean_inc(v___y_2245_);
lean_inc_ref(v___y_2244_);
lean_inc(v___y_2243_);
lean_inc_ref(v___y_2242_);
lean_inc(v___y_2241_);
lean_inc_ref(v___y_2240_);
v_r_2251_ = lean_apply_9(v_x_2239_, v___y_2240_, v___y_2241_, v___y_2242_, v___y_2243_, v___y_2244_, v___y_2245_, v___y_2246_, v___y_2247_, lean_box(0));
if (lean_obj_tag(v_r_2251_) == 0)
{
lean_object* v_a_2252_; lean_object* v___x_2254_; uint8_t v_isShared_2255_; uint8_t v_isSharedCheck_2276_; 
v_a_2252_ = lean_ctor_get(v_r_2251_, 0);
v_isSharedCheck_2276_ = !lean_is_exclusive(v_r_2251_);
if (v_isSharedCheck_2276_ == 0)
{
v___x_2254_ = v_r_2251_;
v_isShared_2255_ = v_isSharedCheck_2276_;
goto v_resetjp_2253_;
}
else
{
lean_inc(v_a_2252_);
lean_dec(v_r_2251_);
v___x_2254_ = lean_box(0);
v_isShared_2255_ = v_isSharedCheck_2276_;
goto v_resetjp_2253_;
}
v_resetjp_2253_:
{
lean_object* v___x_2257_; 
lean_inc(v_a_2252_);
if (v_isShared_2255_ == 0)
{
lean_ctor_set_tag(v___x_2254_, 1);
v___x_2257_ = v___x_2254_;
goto v_reusejp_2256_;
}
else
{
lean_object* v_reuseFailAlloc_2275_; 
v_reuseFailAlloc_2275_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2275_, 0, v_a_2252_);
v___x_2257_ = v_reuseFailAlloc_2275_;
goto v_reusejp_2256_;
}
v_reusejp_2256_:
{
lean_object* v___x_2258_; 
v___x_2258_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___lam__0(v_a_2250_, v___y_2241_, v___y_2242_, v___y_2243_, v___y_2244_, v___y_2245_, v___y_2246_, v___y_2247_, v___x_2257_);
lean_dec_ref(v___x_2257_);
if (lean_obj_tag(v___x_2258_) == 0)
{
lean_object* v___x_2260_; uint8_t v_isShared_2261_; uint8_t v_isSharedCheck_2265_; 
v_isSharedCheck_2265_ = !lean_is_exclusive(v___x_2258_);
if (v_isSharedCheck_2265_ == 0)
{
lean_object* v_unused_2266_; 
v_unused_2266_ = lean_ctor_get(v___x_2258_, 0);
lean_dec(v_unused_2266_);
v___x_2260_ = v___x_2258_;
v_isShared_2261_ = v_isSharedCheck_2265_;
goto v_resetjp_2259_;
}
else
{
lean_dec(v___x_2258_);
v___x_2260_ = lean_box(0);
v_isShared_2261_ = v_isSharedCheck_2265_;
goto v_resetjp_2259_;
}
v_resetjp_2259_:
{
lean_object* v___x_2263_; 
if (v_isShared_2261_ == 0)
{
lean_ctor_set(v___x_2260_, 0, v_a_2252_);
v___x_2263_ = v___x_2260_;
goto v_reusejp_2262_;
}
else
{
lean_object* v_reuseFailAlloc_2264_; 
v_reuseFailAlloc_2264_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2264_, 0, v_a_2252_);
v___x_2263_ = v_reuseFailAlloc_2264_;
goto v_reusejp_2262_;
}
v_reusejp_2262_:
{
return v___x_2263_;
}
}
}
else
{
lean_object* v_a_2267_; lean_object* v___x_2269_; uint8_t v_isShared_2270_; uint8_t v_isSharedCheck_2274_; 
lean_dec(v_a_2252_);
v_a_2267_ = lean_ctor_get(v___x_2258_, 0);
v_isSharedCheck_2274_ = !lean_is_exclusive(v___x_2258_);
if (v_isSharedCheck_2274_ == 0)
{
v___x_2269_ = v___x_2258_;
v_isShared_2270_ = v_isSharedCheck_2274_;
goto v_resetjp_2268_;
}
else
{
lean_inc(v_a_2267_);
lean_dec(v___x_2258_);
v___x_2269_ = lean_box(0);
v_isShared_2270_ = v_isSharedCheck_2274_;
goto v_resetjp_2268_;
}
v_resetjp_2268_:
{
lean_object* v___x_2272_; 
if (v_isShared_2270_ == 0)
{
v___x_2272_ = v___x_2269_;
goto v_reusejp_2271_;
}
else
{
lean_object* v_reuseFailAlloc_2273_; 
v_reuseFailAlloc_2273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2273_, 0, v_a_2267_);
v___x_2272_ = v_reuseFailAlloc_2273_;
goto v_reusejp_2271_;
}
v_reusejp_2271_:
{
return v___x_2272_;
}
}
}
}
}
}
else
{
lean_object* v_a_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; 
v_a_2277_ = lean_ctor_get(v_r_2251_, 0);
lean_inc(v_a_2277_);
lean_dec_ref_known(v_r_2251_, 1);
v___x_2278_ = lean_box(0);
v___x_2279_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___lam__0(v_a_2250_, v___y_2241_, v___y_2242_, v___y_2243_, v___y_2244_, v___y_2245_, v___y_2246_, v___y_2247_, v___x_2278_);
if (lean_obj_tag(v___x_2279_) == 0)
{
lean_object* v___x_2281_; uint8_t v_isShared_2282_; uint8_t v_isSharedCheck_2286_; 
v_isSharedCheck_2286_ = !lean_is_exclusive(v___x_2279_);
if (v_isSharedCheck_2286_ == 0)
{
lean_object* v_unused_2287_; 
v_unused_2287_ = lean_ctor_get(v___x_2279_, 0);
lean_dec(v_unused_2287_);
v___x_2281_ = v___x_2279_;
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
else
{
lean_dec(v___x_2279_);
v___x_2281_ = lean_box(0);
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
v_resetjp_2280_:
{
lean_object* v___x_2284_; 
if (v_isShared_2282_ == 0)
{
lean_ctor_set_tag(v___x_2281_, 1);
lean_ctor_set(v___x_2281_, 0, v_a_2277_);
v___x_2284_ = v___x_2281_;
goto v_reusejp_2283_;
}
else
{
lean_object* v_reuseFailAlloc_2285_; 
v_reuseFailAlloc_2285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2285_, 0, v_a_2277_);
v___x_2284_ = v_reuseFailAlloc_2285_;
goto v_reusejp_2283_;
}
v_reusejp_2283_:
{
return v___x_2284_;
}
}
}
else
{
lean_object* v_a_2288_; lean_object* v___x_2290_; uint8_t v_isShared_2291_; uint8_t v_isSharedCheck_2295_; 
lean_dec(v_a_2277_);
v_a_2288_ = lean_ctor_get(v___x_2279_, 0);
v_isSharedCheck_2295_ = !lean_is_exclusive(v___x_2279_);
if (v_isSharedCheck_2295_ == 0)
{
v___x_2290_ = v___x_2279_;
v_isShared_2291_ = v_isSharedCheck_2295_;
goto v_resetjp_2289_;
}
else
{
lean_inc(v_a_2288_);
lean_dec(v___x_2279_);
v___x_2290_ = lean_box(0);
v_isShared_2291_ = v_isSharedCheck_2295_;
goto v_resetjp_2289_;
}
v_resetjp_2289_:
{
lean_object* v___x_2293_; 
if (v_isShared_2291_ == 0)
{
v___x_2293_ = v___x_2290_;
goto v_reusejp_2292_;
}
else
{
lean_object* v_reuseFailAlloc_2294_; 
v_reuseFailAlloc_2294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2294_, 0, v_a_2288_);
v___x_2293_ = v_reuseFailAlloc_2294_;
goto v_reusejp_2292_;
}
v_reusejp_2292_:
{
return v___x_2293_;
}
}
}
}
}
else
{
lean_object* v_a_2296_; lean_object* v___x_2298_; uint8_t v_isShared_2299_; uint8_t v_isSharedCheck_2303_; 
lean_dec_ref(v_x_2239_);
v_a_2296_ = lean_ctor_get(v___x_2249_, 0);
v_isSharedCheck_2303_ = !lean_is_exclusive(v___x_2249_);
if (v_isSharedCheck_2303_ == 0)
{
v___x_2298_ = v___x_2249_;
v_isShared_2299_ = v_isSharedCheck_2303_;
goto v_resetjp_2297_;
}
else
{
lean_inc(v_a_2296_);
lean_dec(v___x_2249_);
v___x_2298_ = lean_box(0);
v_isShared_2299_ = v_isSharedCheck_2303_;
goto v_resetjp_2297_;
}
v_resetjp_2297_:
{
lean_object* v___x_2301_; 
if (v_isShared_2299_ == 0)
{
v___x_2301_ = v___x_2298_;
goto v_reusejp_2300_;
}
else
{
lean_object* v_reuseFailAlloc_2302_; 
v_reuseFailAlloc_2302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2302_, 0, v_a_2296_);
v___x_2301_ = v_reuseFailAlloc_2302_;
goto v_reusejp_2300_;
}
v_reusejp_2300_:
{
return v___x_2301_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg___boxed(lean_object* v_x_2304_, lean_object* v___y_2305_, lean_object* v___y_2306_, lean_object* v___y_2307_, lean_object* v___y_2308_, lean_object* v___y_2309_, lean_object* v___y_2310_, lean_object* v___y_2311_, lean_object* v___y_2312_, lean_object* v___y_2313_){
_start:
{
lean_object* v_res_2314_; 
v_res_2314_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg(v_x_2304_, v___y_2305_, v___y_2306_, v___y_2307_, v___y_2308_, v___y_2309_, v___y_2310_, v___y_2311_, v___y_2312_);
lean_dec(v___y_2312_);
lean_dec_ref(v___y_2311_);
lean_dec(v___y_2310_);
lean_dec_ref(v___y_2309_);
lean_dec(v___y_2308_);
lean_dec_ref(v___y_2307_);
lean_dec(v___y_2306_);
lean_dec_ref(v___y_2305_);
return v_res_2314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2(lean_object* v_00_u03b1_2315_, lean_object* v_x_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_){
_start:
{
lean_object* v___x_2326_; 
v___x_2326_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___redArg(v_x_2316_, v___y_2317_, v___y_2318_, v___y_2319_, v___y_2320_, v___y_2321_, v___y_2322_, v___y_2323_, v___y_2324_);
return v___x_2326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___boxed(lean_object* v_00_u03b1_2327_, lean_object* v_x_2328_, lean_object* v___y_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_){
_start:
{
lean_object* v_res_2338_; 
v_res_2338_ = lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2(v_00_u03b1_2327_, v_x_2328_, v___y_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, v___y_2335_, v___y_2336_);
lean_dec(v___y_2336_);
lean_dec_ref(v___y_2335_);
lean_dec(v___y_2334_);
lean_dec_ref(v___y_2333_);
lean_dec(v___y_2332_);
lean_dec_ref(v___y_2331_);
lean_dec(v___y_2330_);
lean_dec_ref(v___y_2329_);
return v_res_2338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___redArg(lean_object* v_kind_2339_, lean_object* v___y_2340_){
_start:
{
lean_object* v___x_2342_; lean_object* v_auxDeclNGen_2343_; lean_object* v___x_2344_; lean_object* v_env_2345_; lean_object* v___x_2346_; lean_object* v_fst_2347_; lean_object* v_snd_2348_; lean_object* v___x_2349_; lean_object* v_env_2350_; lean_object* v_nextMacroScope_2351_; lean_object* v_ngen_2352_; lean_object* v_traceState_2353_; lean_object* v_cache_2354_; lean_object* v_messages_2355_; lean_object* v_infoState_2356_; lean_object* v_snapshotTasks_2357_; lean_object* v___x_2359_; uint8_t v_isShared_2360_; uint8_t v_isSharedCheck_2366_; 
v___x_2342_ = lean_st_ref_get(v___y_2340_);
v_auxDeclNGen_2343_ = lean_ctor_get(v___x_2342_, 3);
lean_inc_ref(v_auxDeclNGen_2343_);
lean_dec(v___x_2342_);
v___x_2344_ = lean_st_ref_get(v___y_2340_);
v_env_2345_ = lean_ctor_get(v___x_2344_, 0);
lean_inc_ref(v_env_2345_);
lean_dec(v___x_2344_);
v___x_2346_ = l_Lean_DeclNameGenerator_mkUniqueName(v_env_2345_, v_auxDeclNGen_2343_, v_kind_2339_);
v_fst_2347_ = lean_ctor_get(v___x_2346_, 0);
lean_inc(v_fst_2347_);
v_snd_2348_ = lean_ctor_get(v___x_2346_, 1);
lean_inc(v_snd_2348_);
lean_dec_ref(v___x_2346_);
v___x_2349_ = lean_st_ref_take(v___y_2340_);
v_env_2350_ = lean_ctor_get(v___x_2349_, 0);
v_nextMacroScope_2351_ = lean_ctor_get(v___x_2349_, 1);
v_ngen_2352_ = lean_ctor_get(v___x_2349_, 2);
v_traceState_2353_ = lean_ctor_get(v___x_2349_, 4);
v_cache_2354_ = lean_ctor_get(v___x_2349_, 5);
v_messages_2355_ = lean_ctor_get(v___x_2349_, 6);
v_infoState_2356_ = lean_ctor_get(v___x_2349_, 7);
v_snapshotTasks_2357_ = lean_ctor_get(v___x_2349_, 8);
v_isSharedCheck_2366_ = !lean_is_exclusive(v___x_2349_);
if (v_isSharedCheck_2366_ == 0)
{
lean_object* v_unused_2367_; 
v_unused_2367_ = lean_ctor_get(v___x_2349_, 3);
lean_dec(v_unused_2367_);
v___x_2359_ = v___x_2349_;
v_isShared_2360_ = v_isSharedCheck_2366_;
goto v_resetjp_2358_;
}
else
{
lean_inc(v_snapshotTasks_2357_);
lean_inc(v_infoState_2356_);
lean_inc(v_messages_2355_);
lean_inc(v_cache_2354_);
lean_inc(v_traceState_2353_);
lean_inc(v_ngen_2352_);
lean_inc(v_nextMacroScope_2351_);
lean_inc(v_env_2350_);
lean_dec(v___x_2349_);
v___x_2359_ = lean_box(0);
v_isShared_2360_ = v_isSharedCheck_2366_;
goto v_resetjp_2358_;
}
v_resetjp_2358_:
{
lean_object* v___x_2362_; 
if (v_isShared_2360_ == 0)
{
lean_ctor_set(v___x_2359_, 3, v_snd_2348_);
v___x_2362_ = v___x_2359_;
goto v_reusejp_2361_;
}
else
{
lean_object* v_reuseFailAlloc_2365_; 
v_reuseFailAlloc_2365_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2365_, 0, v_env_2350_);
lean_ctor_set(v_reuseFailAlloc_2365_, 1, v_nextMacroScope_2351_);
lean_ctor_set(v_reuseFailAlloc_2365_, 2, v_ngen_2352_);
lean_ctor_set(v_reuseFailAlloc_2365_, 3, v_snd_2348_);
lean_ctor_set(v_reuseFailAlloc_2365_, 4, v_traceState_2353_);
lean_ctor_set(v_reuseFailAlloc_2365_, 5, v_cache_2354_);
lean_ctor_set(v_reuseFailAlloc_2365_, 6, v_messages_2355_);
lean_ctor_set(v_reuseFailAlloc_2365_, 7, v_infoState_2356_);
lean_ctor_set(v_reuseFailAlloc_2365_, 8, v_snapshotTasks_2357_);
v___x_2362_ = v_reuseFailAlloc_2365_;
goto v_reusejp_2361_;
}
v_reusejp_2361_:
{
lean_object* v___x_2363_; lean_object* v___x_2364_; 
v___x_2363_ = lean_st_ref_set(v___y_2340_, v___x_2362_);
v___x_2364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2364_, 0, v_fst_2347_);
return v___x_2364_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___redArg___boxed(lean_object* v_kind_2368_, lean_object* v___y_2369_, lean_object* v___y_2370_){
_start:
{
lean_object* v_res_2371_; 
v_res_2371_ = lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___redArg(v_kind_2368_, v___y_2369_);
lean_dec(v___y_2369_);
return v_res_2371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5(lean_object* v_kind_2372_, lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_){
_start:
{
lean_object* v___x_2382_; 
v___x_2382_ = lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___redArg(v_kind_2372_, v___y_2380_);
return v___x_2382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___boxed(lean_object* v_kind_2383_, lean_object* v___y_2384_, lean_object* v___y_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_, lean_object* v___y_2392_){
_start:
{
lean_object* v_res_2393_; 
v_res_2393_ = lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5(v_kind_2383_, v___y_2384_, v___y_2385_, v___y_2386_, v___y_2387_, v___y_2388_, v___y_2389_, v___y_2390_, v___y_2391_);
lean_dec(v___y_2391_);
lean_dec_ref(v___y_2390_);
lean_dec(v___y_2389_);
lean_dec_ref(v___y_2388_);
lean_dec(v___y_2387_);
lean_dec_ref(v___y_2386_);
lean_dec(v___y_2385_);
lean_dec_ref(v___y_2384_);
return v_res_2393_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2395_; lean_object* v___x_2396_; 
v___x_2395_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__0));
v___x_2396_ = l_Lean_stringToMessageData(v___x_2395_);
return v___x_2396_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3(void){
_start:
{
lean_object* v___x_2398_; lean_object* v___x_2399_; 
v___x_2398_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__2));
v___x_2399_ = l_Lean_stringToMessageData(v___x_2398_);
return v___x_2399_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__5(void){
_start:
{
lean_object* v___x_2401_; lean_object* v___x_2402_; 
v___x_2401_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__4));
v___x_2402_ = l_Lean_stringToMessageData(v___x_2401_);
return v___x_2402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0(lean_object* v_name_2408_, uint8_t v___x_2409_, lean_object* v___x_2410_, lean_object* v___x_2411_, lean_object* v___x_2412_, lean_object* v___x_2413_, lean_object* v___x_2414_, lean_object* v___y_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_, lean_object* v___y_2418_, lean_object* v___y_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_){
_start:
{
lean_object* v___y_2425_; lean_object* v___y_2426_; uint8_t v___y_2427_; lean_object* v___y_2428_; lean_object* v_g_2449_; lean_object* v___y_2450_; lean_object* v___y_2451_; lean_object* v___y_2452_; lean_object* v___y_2453_; lean_object* v___y_2454_; lean_object* v___y_2455_; lean_object* v___x_2486_; 
v___x_2486_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2416_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_);
if (lean_obj_tag(v___x_2486_) == 0)
{
lean_object* v_a_2487_; lean_object* v___y_2489_; 
v_a_2487_ = lean_ctor_get(v___x_2486_, 0);
lean_inc(v_a_2487_);
lean_dec_ref_known(v___x_2486_, 1);
if (v___x_2409_ == 0)
{
lean_object* v___x_2513_; lean_object* v_a_2514_; lean_object* v___x_2516_; uint8_t v_isShared_2517_; uint8_t v_isSharedCheck_2521_; 
lean_dec(v_a_2487_);
lean_dec_ref(v___x_2414_);
lean_dec_ref(v___x_2413_);
lean_dec_ref(v___x_2412_);
lean_dec(v_name_2408_);
v___x_2513_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg();
v_a_2514_ = lean_ctor_get(v___x_2513_, 0);
v_isSharedCheck_2521_ = !lean_is_exclusive(v___x_2513_);
if (v_isSharedCheck_2521_ == 0)
{
v___x_2516_ = v___x_2513_;
v_isShared_2517_ = v_isSharedCheck_2521_;
goto v_resetjp_2515_;
}
else
{
lean_inc(v_a_2514_);
lean_dec(v___x_2513_);
v___x_2516_ = lean_box(0);
v_isShared_2517_ = v_isSharedCheck_2521_;
goto v_resetjp_2515_;
}
v_resetjp_2515_:
{
lean_object* v___x_2519_; 
if (v_isShared_2517_ == 0)
{
v___x_2519_ = v___x_2516_;
goto v_reusejp_2518_;
}
else
{
lean_object* v_reuseFailAlloc_2520_; 
v_reuseFailAlloc_2520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2520_, 0, v_a_2514_);
v___x_2519_ = v_reuseFailAlloc_2520_;
goto v_reusejp_2518_;
}
v_reusejp_2518_:
{
return v___x_2519_;
}
}
}
else
{
lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; uint8_t v___x_2525_; 
v___x_2522_ = l_Lean_Syntax_getArg(v___x_2410_, v___x_2411_);
v___x_2523_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__0));
v___x_2524_ = l_Lean_Name_mkStr4(v___x_2412_, v___x_2413_, v___x_2414_, v___x_2523_);
lean_inc(v___x_2522_);
v___x_2525_ = l_Lean_Syntax_isOfKind(v___x_2522_, v___x_2524_);
lean_dec(v___x_2524_);
if (v___x_2525_ == 0)
{
uint8_t v___x_2526_; 
lean_inc(v___x_2522_);
v___x_2526_ = l_Lean_Syntax_matchesNull(v___x_2522_, v___x_2411_);
if (v___x_2526_ == 0)
{
lean_object* v___x_2527_; lean_object* v___x_2528_; 
v___x_2527_ = l_Lean_Syntax_getArgs(v___x_2522_);
lean_dec(v___x_2522_);
v___x_2528_ = l_Lean_Elab_Tactic_getFVarIds(v___x_2527_, v___y_2415_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_);
if (lean_obj_tag(v___x_2528_) == 0)
{
lean_object* v_a_2529_; lean_object* v___x_2530_; 
v_a_2529_ = lean_ctor_get(v___x_2528_, 0);
lean_inc(v_a_2529_);
lean_dec_ref_known(v___x_2528_, 1);
v___x_2530_ = l___private_Lean_Meta_Tactic_Cleanup_0__Lean_Meta_cleanupCore(v_a_2487_, v_a_2529_, v___x_2526_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_);
if (lean_obj_tag(v___x_2530_) == 0)
{
lean_object* v_a_2531_; 
v_a_2531_ = lean_ctor_get(v___x_2530_, 0);
lean_inc(v_a_2531_);
lean_dec_ref_known(v___x_2530_, 1);
v_g_2449_ = v_a_2531_;
v___y_2450_ = v___y_2417_;
v___y_2451_ = v___y_2418_;
v___y_2452_ = v___y_2419_;
v___y_2453_ = v___y_2420_;
v___y_2454_ = v___y_2421_;
v___y_2455_ = v___y_2422_;
goto v___jp_2448_;
}
else
{
lean_object* v_a_2532_; lean_object* v___x_2534_; uint8_t v_isShared_2535_; uint8_t v_isSharedCheck_2539_; 
lean_dec(v_name_2408_);
v_a_2532_ = lean_ctor_get(v___x_2530_, 0);
v_isSharedCheck_2539_ = !lean_is_exclusive(v___x_2530_);
if (v_isSharedCheck_2539_ == 0)
{
v___x_2534_ = v___x_2530_;
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
else
{
lean_inc(v_a_2532_);
lean_dec(v___x_2530_);
v___x_2534_ = lean_box(0);
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
v_resetjp_2533_:
{
lean_object* v___x_2537_; 
if (v_isShared_2535_ == 0)
{
v___x_2537_ = v___x_2534_;
goto v_reusejp_2536_;
}
else
{
lean_object* v_reuseFailAlloc_2538_; 
v_reuseFailAlloc_2538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2538_, 0, v_a_2532_);
v___x_2537_ = v_reuseFailAlloc_2538_;
goto v_reusejp_2536_;
}
v_reusejp_2536_:
{
return v___x_2537_;
}
}
}
}
else
{
lean_object* v_a_2540_; lean_object* v___x_2542_; uint8_t v_isShared_2543_; uint8_t v_isSharedCheck_2547_; 
lean_dec(v_a_2487_);
lean_dec(v_name_2408_);
v_a_2540_ = lean_ctor_get(v___x_2528_, 0);
v_isSharedCheck_2547_ = !lean_is_exclusive(v___x_2528_);
if (v_isSharedCheck_2547_ == 0)
{
v___x_2542_ = v___x_2528_;
v_isShared_2543_ = v_isSharedCheck_2547_;
goto v_resetjp_2541_;
}
else
{
lean_inc(v_a_2540_);
lean_dec(v___x_2528_);
v___x_2542_ = lean_box(0);
v_isShared_2543_ = v_isSharedCheck_2547_;
goto v_resetjp_2541_;
}
v_resetjp_2541_:
{
lean_object* v___x_2545_; 
if (v_isShared_2543_ == 0)
{
v___x_2545_ = v___x_2542_;
goto v_reusejp_2544_;
}
else
{
lean_object* v_reuseFailAlloc_2546_; 
v_reuseFailAlloc_2546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2546_, 0, v_a_2540_);
v___x_2545_ = v_reuseFailAlloc_2546_;
goto v_reusejp_2544_;
}
v_reusejp_2544_:
{
return v___x_2545_;
}
}
}
}
else
{
lean_object* v___x_2548_; 
lean_dec(v___x_2522_);
lean_inc(v_a_2487_);
v___x_2548_ = l_Lean_MVarId_getType(v_a_2487_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_);
if (lean_obj_tag(v___x_2548_) == 0)
{
lean_object* v_a_2549_; lean_object* v___x_2550_; 
v_a_2549_ = lean_ctor_get(v___x_2548_, 0);
lean_inc(v_a_2549_);
lean_dec_ref_known(v___x_2548_, 1);
v___x_2550_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__1___redArg(v_a_2549_, v___y_2420_);
v___y_2489_ = v___x_2550_;
goto v___jp_2488_;
}
else
{
v___y_2489_ = v___x_2548_;
goto v___jp_2488_;
}
}
}
else
{
lean_dec(v___x_2522_);
v_g_2449_ = v_a_2487_;
v___y_2450_ = v___y_2417_;
v___y_2451_ = v___y_2418_;
v___y_2452_ = v___y_2419_;
v___y_2453_ = v___y_2420_;
v___y_2454_ = v___y_2421_;
v___y_2455_ = v___y_2422_;
goto v___jp_2448_;
}
}
v___jp_2488_:
{
if (lean_obj_tag(v___y_2489_) == 0)
{
lean_object* v_a_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; uint8_t v___x_2493_; 
v_a_2490_ = lean_ctor_get(v___y_2489_, 0);
lean_inc(v_a_2490_);
lean_dec_ref_known(v___y_2489_, 1);
v___x_2491_ = l_Lean_Expr_consumeMData(v_a_2490_);
lean_dec(v_a_2490_);
v___x_2492_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__9));
v___x_2493_ = l_Lean_Expr_isConstOf(v___x_2491_, v___x_2492_);
lean_dec_ref(v___x_2491_);
if (v___x_2493_ == 0)
{
lean_object* v___x_2494_; lean_object* v___x_2495_; 
v___x_2494_ = lean_mk_empty_array_with_capacity(v___x_2411_);
v___x_2495_ = l___private_Lean_Meta_Tactic_Cleanup_0__Lean_Meta_cleanupCore(v_a_2487_, v___x_2494_, v___x_2409_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_);
if (lean_obj_tag(v___x_2495_) == 0)
{
lean_object* v_a_2496_; 
v_a_2496_ = lean_ctor_get(v___x_2495_, 0);
lean_inc(v_a_2496_);
lean_dec_ref_known(v___x_2495_, 1);
v_g_2449_ = v_a_2496_;
v___y_2450_ = v___y_2417_;
v___y_2451_ = v___y_2418_;
v___y_2452_ = v___y_2419_;
v___y_2453_ = v___y_2420_;
v___y_2454_ = v___y_2421_;
v___y_2455_ = v___y_2422_;
goto v___jp_2448_;
}
else
{
lean_object* v_a_2497_; lean_object* v___x_2499_; uint8_t v_isShared_2500_; uint8_t v_isSharedCheck_2504_; 
lean_dec(v_name_2408_);
v_a_2497_ = lean_ctor_get(v___x_2495_, 0);
v_isSharedCheck_2504_ = !lean_is_exclusive(v___x_2495_);
if (v_isSharedCheck_2504_ == 0)
{
v___x_2499_ = v___x_2495_;
v_isShared_2500_ = v_isSharedCheck_2504_;
goto v_resetjp_2498_;
}
else
{
lean_inc(v_a_2497_);
lean_dec(v___x_2495_);
v___x_2499_ = lean_box(0);
v_isShared_2500_ = v_isSharedCheck_2504_;
goto v_resetjp_2498_;
}
v_resetjp_2498_:
{
lean_object* v___x_2502_; 
if (v_isShared_2500_ == 0)
{
v___x_2502_ = v___x_2499_;
goto v_reusejp_2501_;
}
else
{
lean_object* v_reuseFailAlloc_2503_; 
v_reuseFailAlloc_2503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2503_, 0, v_a_2497_);
v___x_2502_ = v_reuseFailAlloc_2503_;
goto v_reusejp_2501_;
}
v_reusejp_2501_:
{
return v___x_2502_;
}
}
}
}
else
{
v_g_2449_ = v_a_2487_;
v___y_2450_ = v___y_2417_;
v___y_2451_ = v___y_2418_;
v___y_2452_ = v___y_2419_;
v___y_2453_ = v___y_2420_;
v___y_2454_ = v___y_2421_;
v___y_2455_ = v___y_2422_;
goto v___jp_2448_;
}
}
else
{
lean_object* v_a_2505_; lean_object* v___x_2507_; uint8_t v_isShared_2508_; uint8_t v_isSharedCheck_2512_; 
lean_dec(v_a_2487_);
lean_dec(v_name_2408_);
v_a_2505_ = lean_ctor_get(v___y_2489_, 0);
v_isSharedCheck_2512_ = !lean_is_exclusive(v___y_2489_);
if (v_isSharedCheck_2512_ == 0)
{
v___x_2507_ = v___y_2489_;
v_isShared_2508_ = v_isSharedCheck_2512_;
goto v_resetjp_2506_;
}
else
{
lean_inc(v_a_2505_);
lean_dec(v___y_2489_);
v___x_2507_ = lean_box(0);
v_isShared_2508_ = v_isSharedCheck_2512_;
goto v_resetjp_2506_;
}
v_resetjp_2506_:
{
lean_object* v___x_2510_; 
if (v_isShared_2508_ == 0)
{
v___x_2510_ = v___x_2507_;
goto v_reusejp_2509_;
}
else
{
lean_object* v_reuseFailAlloc_2511_; 
v_reuseFailAlloc_2511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2511_, 0, v_a_2505_);
v___x_2510_ = v_reuseFailAlloc_2511_;
goto v_reusejp_2509_;
}
v_reusejp_2509_:
{
return v___x_2510_;
}
}
}
}
}
else
{
lean_object* v_a_2551_; lean_object* v___x_2553_; uint8_t v_isShared_2554_; uint8_t v_isSharedCheck_2558_; 
lean_dec_ref(v___x_2414_);
lean_dec_ref(v___x_2413_);
lean_dec_ref(v___x_2412_);
lean_dec(v_name_2408_);
v_a_2551_ = lean_ctor_get(v___x_2486_, 0);
v_isSharedCheck_2558_ = !lean_is_exclusive(v___x_2486_);
if (v_isSharedCheck_2558_ == 0)
{
v___x_2553_ = v___x_2486_;
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
else
{
lean_inc(v_a_2551_);
lean_dec(v___x_2486_);
v___x_2553_ = lean_box(0);
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
v_resetjp_2552_:
{
lean_object* v___x_2556_; 
if (v_isShared_2554_ == 0)
{
v___x_2556_ = v___x_2553_;
goto v_reusejp_2555_;
}
else
{
lean_object* v_reuseFailAlloc_2557_; 
v_reuseFailAlloc_2557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2557_, 0, v_a_2551_);
v___x_2556_ = v_reuseFailAlloc_2557_;
goto v_reusejp_2555_;
}
v_reusejp_2555_:
{
return v___x_2556_;
}
}
}
v___jp_2424_:
{
if (v___y_2427_ == 0)
{
lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; 
lean_dec_ref(v___y_2425_);
lean_dec(v_name_2408_);
lean_inc_ref(v___y_2428_);
v___x_2429_ = l_Lean_stringToMessageData(v___y_2428_);
v___x_2430_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1);
v___x_2431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2429_);
lean_ctor_set(v___x_2431_, 1, v___x_2430_);
v___x_2432_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2432_, 0, v___x_2431_);
lean_ctor_set(v___x_2432_, 1, v___y_2426_);
v___x_2433_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3);
v___x_2434_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2434_, 0, v___x_2432_);
lean_ctor_set(v___x_2434_, 1, v___x_2433_);
v___x_2435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2435_, 0, v___x_2434_);
return v___x_2435_;
}
else
{
lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; 
lean_dec_ref(v___y_2426_);
lean_inc_ref(v___y_2428_);
v___x_2436_ = l_Lean_stringToMessageData(v___y_2428_);
v___x_2437_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__1);
v___x_2438_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2438_, 0, v___x_2436_);
lean_ctor_set(v___x_2438_, 1, v___x_2437_);
v___x_2439_ = l_Lean_MessageData_ofName(v_name_2408_);
v___x_2440_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2440_, 0, v___x_2438_);
lean_ctor_set(v___x_2440_, 1, v___x_2439_);
v___x_2441_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__5);
v___x_2442_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2442_, 0, v___x_2440_);
lean_ctor_set(v___x_2442_, 1, v___x_2441_);
v___x_2443_ = l_Lean_MessageData_ofExpr(v___y_2425_);
v___x_2444_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2444_, 0, v___x_2442_);
lean_ctor_set(v___x_2444_, 1, v___x_2443_);
v___x_2445_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__3);
v___x_2446_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2446_, 0, v___x_2444_);
lean_ctor_set(v___x_2446_, 1, v___x_2445_);
v___x_2447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2447_, 0, v___x_2446_);
return v___x_2447_;
}
}
v___jp_2448_:
{
lean_object* v___x_2456_; 
lean_inc(v_name_2408_);
v___x_2456_ = lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature(v_name_2408_, v_g_2449_, v___y_2450_, v___y_2451_, v___y_2452_, v___y_2453_, v___y_2454_, v___y_2455_);
if (lean_obj_tag(v___x_2456_) == 0)
{
lean_object* v_a_2457_; lean_object* v_snd_2458_; lean_object* v_snd_2459_; lean_object* v_fst_2460_; lean_object* v_fst_2461_; lean_object* v_snd_2462_; lean_object* v___x_2463_; 
v_a_2457_ = lean_ctor_get(v___x_2456_, 0);
lean_inc(v_a_2457_);
lean_dec_ref_known(v___x_2456_, 1);
v_snd_2458_ = lean_ctor_get(v_a_2457_, 1);
lean_inc(v_snd_2458_);
v_snd_2459_ = lean_ctor_get(v_snd_2458_, 1);
lean_inc(v_snd_2459_);
v_fst_2460_ = lean_ctor_get(v_a_2457_, 0);
lean_inc(v_fst_2460_);
lean_dec(v_a_2457_);
v_fst_2461_ = lean_ctor_get(v_snd_2458_, 0);
lean_inc_n(v_fst_2461_, 2);
lean_dec(v_snd_2458_);
v_snd_2462_ = lean_ctor_get(v_snd_2459_, 1);
lean_inc(v_snd_2462_);
lean_dec(v_snd_2459_);
v___x_2463_ = l_Lean_Meta_isProp(v_fst_2461_, v___y_2452_, v___y_2453_, v___y_2454_, v___y_2455_);
if (lean_obj_tag(v___x_2463_) == 0)
{
lean_object* v_a_2464_; uint8_t v___x_2465_; 
v_a_2464_ = lean_ctor_get(v___x_2463_, 0);
lean_inc(v_a_2464_);
lean_dec_ref_known(v___x_2463_, 1);
v___x_2465_ = lean_unbox(v_a_2464_);
lean_dec(v_a_2464_);
if (v___x_2465_ == 0)
{
lean_object* v___x_2466_; uint8_t v___x_2467_; 
v___x_2466_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__6));
v___x_2467_ = lean_unbox(v_snd_2462_);
lean_dec(v_snd_2462_);
v___y_2425_ = v_fst_2461_;
v___y_2426_ = v_fst_2460_;
v___y_2427_ = v___x_2467_;
v___y_2428_ = v___x_2466_;
goto v___jp_2424_;
}
else
{
lean_object* v___x_2468_; uint8_t v___x_2469_; 
v___x_2468_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___closed__7));
v___x_2469_ = lean_unbox(v_snd_2462_);
lean_dec(v_snd_2462_);
v___y_2425_ = v_fst_2461_;
v___y_2426_ = v_fst_2460_;
v___y_2427_ = v___x_2469_;
v___y_2428_ = v___x_2468_;
goto v___jp_2424_;
}
}
else
{
lean_object* v_a_2470_; lean_object* v___x_2472_; uint8_t v_isShared_2473_; uint8_t v_isSharedCheck_2477_; 
lean_dec(v_snd_2462_);
lean_dec(v_fst_2461_);
lean_dec(v_fst_2460_);
lean_dec(v_name_2408_);
v_a_2470_ = lean_ctor_get(v___x_2463_, 0);
v_isSharedCheck_2477_ = !lean_is_exclusive(v___x_2463_);
if (v_isSharedCheck_2477_ == 0)
{
v___x_2472_ = v___x_2463_;
v_isShared_2473_ = v_isSharedCheck_2477_;
goto v_resetjp_2471_;
}
else
{
lean_inc(v_a_2470_);
lean_dec(v___x_2463_);
v___x_2472_ = lean_box(0);
v_isShared_2473_ = v_isSharedCheck_2477_;
goto v_resetjp_2471_;
}
v_resetjp_2471_:
{
lean_object* v___x_2475_; 
if (v_isShared_2473_ == 0)
{
v___x_2475_ = v___x_2472_;
goto v_reusejp_2474_;
}
else
{
lean_object* v_reuseFailAlloc_2476_; 
v_reuseFailAlloc_2476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2476_, 0, v_a_2470_);
v___x_2475_ = v_reuseFailAlloc_2476_;
goto v_reusejp_2474_;
}
v_reusejp_2474_:
{
return v___x_2475_;
}
}
}
}
else
{
lean_object* v_a_2478_; lean_object* v___x_2480_; uint8_t v_isShared_2481_; uint8_t v_isSharedCheck_2485_; 
lean_dec(v_name_2408_);
v_a_2478_ = lean_ctor_get(v___x_2456_, 0);
v_isSharedCheck_2485_ = !lean_is_exclusive(v___x_2456_);
if (v_isSharedCheck_2485_ == 0)
{
v___x_2480_ = v___x_2456_;
v_isShared_2481_ = v_isSharedCheck_2485_;
goto v_resetjp_2479_;
}
else
{
lean_inc(v_a_2478_);
lean_dec(v___x_2456_);
v___x_2480_ = lean_box(0);
v_isShared_2481_ = v_isSharedCheck_2485_;
goto v_resetjp_2479_;
}
v_resetjp_2479_:
{
lean_object* v___x_2483_; 
if (v_isShared_2481_ == 0)
{
v___x_2483_ = v___x_2480_;
goto v_reusejp_2482_;
}
else
{
lean_object* v_reuseFailAlloc_2484_; 
v_reuseFailAlloc_2484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2484_, 0, v_a_2478_);
v___x_2483_ = v_reuseFailAlloc_2484_;
goto v_reusejp_2482_;
}
v_reusejp_2482_:
{
return v___x_2483_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___boxed(lean_object* v_name_2559_, lean_object* v___x_2560_, lean_object* v___x_2561_, lean_object* v___x_2562_, lean_object* v___x_2563_, lean_object* v___x_2564_, lean_object* v___x_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_, lean_object* v___y_2568_, lean_object* v___y_2569_, lean_object* v___y_2570_, lean_object* v___y_2571_, lean_object* v___y_2572_, lean_object* v___y_2573_, lean_object* v___y_2574_){
_start:
{
uint8_t v___x_16536__boxed_2575_; lean_object* v_res_2576_; 
v___x_16536__boxed_2575_ = lean_unbox(v___x_2560_);
v_res_2576_ = lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0(v_name_2559_, v___x_16536__boxed_2575_, v___x_2561_, v___x_2562_, v___x_2563_, v___x_2564_, v___x_2565_, v___y_2566_, v___y_2567_, v___y_2568_, v___y_2569_, v___y_2570_, v___y_2571_, v___y_2572_, v___y_2573_);
lean_dec(v___y_2573_);
lean_dec_ref(v___y_2572_);
lean_dec(v___y_2571_);
lean_dec_ref(v___y_2570_);
lean_dec(v___y_2569_);
lean_dec_ref(v___y_2568_);
lean_dec(v___y_2567_);
lean_dec_ref(v___y_2566_);
lean_dec(v___x_2562_);
lean_dec(v___x_2561_);
return v_res_2576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg(lean_object* v_env_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_){
_start:
{
lean_object* v___x_2581_; lean_object* v_nextMacroScope_2582_; lean_object* v_ngen_2583_; lean_object* v_auxDeclNGen_2584_; lean_object* v_traceState_2585_; lean_object* v_messages_2586_; lean_object* v_infoState_2587_; lean_object* v_snapshotTasks_2588_; lean_object* v___x_2590_; uint8_t v_isShared_2591_; uint8_t v_isSharedCheck_2614_; 
v___x_2581_ = lean_st_ref_take(v___y_2579_);
v_nextMacroScope_2582_ = lean_ctor_get(v___x_2581_, 1);
v_ngen_2583_ = lean_ctor_get(v___x_2581_, 2);
v_auxDeclNGen_2584_ = lean_ctor_get(v___x_2581_, 3);
v_traceState_2585_ = lean_ctor_get(v___x_2581_, 4);
v_messages_2586_ = lean_ctor_get(v___x_2581_, 6);
v_infoState_2587_ = lean_ctor_get(v___x_2581_, 7);
v_snapshotTasks_2588_ = lean_ctor_get(v___x_2581_, 8);
v_isSharedCheck_2614_ = !lean_is_exclusive(v___x_2581_);
if (v_isSharedCheck_2614_ == 0)
{
lean_object* v_unused_2615_; lean_object* v_unused_2616_; 
v_unused_2615_ = lean_ctor_get(v___x_2581_, 5);
lean_dec(v_unused_2615_);
v_unused_2616_ = lean_ctor_get(v___x_2581_, 0);
lean_dec(v_unused_2616_);
v___x_2590_ = v___x_2581_;
v_isShared_2591_ = v_isSharedCheck_2614_;
goto v_resetjp_2589_;
}
else
{
lean_inc(v_snapshotTasks_2588_);
lean_inc(v_infoState_2587_);
lean_inc(v_messages_2586_);
lean_inc(v_traceState_2585_);
lean_inc(v_auxDeclNGen_2584_);
lean_inc(v_ngen_2583_);
lean_inc(v_nextMacroScope_2582_);
lean_dec(v___x_2581_);
v___x_2590_ = lean_box(0);
v_isShared_2591_ = v_isSharedCheck_2614_;
goto v_resetjp_2589_;
}
v_resetjp_2589_:
{
lean_object* v___x_2592_; lean_object* v___x_2594_; 
v___x_2592_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__2);
if (v_isShared_2591_ == 0)
{
lean_ctor_set(v___x_2590_, 5, v___x_2592_);
lean_ctor_set(v___x_2590_, 0, v_env_2577_);
v___x_2594_ = v___x_2590_;
goto v_reusejp_2593_;
}
else
{
lean_object* v_reuseFailAlloc_2613_; 
v_reuseFailAlloc_2613_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2613_, 0, v_env_2577_);
lean_ctor_set(v_reuseFailAlloc_2613_, 1, v_nextMacroScope_2582_);
lean_ctor_set(v_reuseFailAlloc_2613_, 2, v_ngen_2583_);
lean_ctor_set(v_reuseFailAlloc_2613_, 3, v_auxDeclNGen_2584_);
lean_ctor_set(v_reuseFailAlloc_2613_, 4, v_traceState_2585_);
lean_ctor_set(v_reuseFailAlloc_2613_, 5, v___x_2592_);
lean_ctor_set(v_reuseFailAlloc_2613_, 6, v_messages_2586_);
lean_ctor_set(v_reuseFailAlloc_2613_, 7, v_infoState_2587_);
lean_ctor_set(v_reuseFailAlloc_2613_, 8, v_snapshotTasks_2588_);
v___x_2594_ = v_reuseFailAlloc_2613_;
goto v_reusejp_2593_;
}
v_reusejp_2593_:
{
lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v_mctx_2597_; lean_object* v_zetaDeltaFVarIds_2598_; lean_object* v_postponed_2599_; lean_object* v_diag_2600_; lean_object* v___x_2602_; uint8_t v_isShared_2603_; uint8_t v_isSharedCheck_2611_; 
v___x_2595_ = lean_st_ref_set(v___y_2579_, v___x_2594_);
v___x_2596_ = lean_st_ref_take(v___y_2578_);
v_mctx_2597_ = lean_ctor_get(v___x_2596_, 0);
v_zetaDeltaFVarIds_2598_ = lean_ctor_get(v___x_2596_, 2);
v_postponed_2599_ = lean_ctor_get(v___x_2596_, 3);
v_diag_2600_ = lean_ctor_get(v___x_2596_, 4);
v_isSharedCheck_2611_ = !lean_is_exclusive(v___x_2596_);
if (v_isSharedCheck_2611_ == 0)
{
lean_object* v_unused_2612_; 
v_unused_2612_ = lean_ctor_get(v___x_2596_, 1);
lean_dec(v_unused_2612_);
v___x_2602_ = v___x_2596_;
v_isShared_2603_ = v_isSharedCheck_2611_;
goto v_resetjp_2601_;
}
else
{
lean_inc(v_diag_2600_);
lean_inc(v_postponed_2599_);
lean_inc(v_zetaDeltaFVarIds_2598_);
lean_inc(v_mctx_2597_);
lean_dec(v___x_2596_);
v___x_2602_ = lean_box(0);
v_isShared_2603_ = v_isSharedCheck_2611_;
goto v_resetjp_2601_;
}
v_resetjp_2601_:
{
lean_object* v___x_2604_; lean_object* v___x_2606_; 
v___x_2604_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3, &lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__14_spec__22___redArg___closed__3);
if (v_isShared_2603_ == 0)
{
lean_ctor_set(v___x_2602_, 1, v___x_2604_);
v___x_2606_ = v___x_2602_;
goto v_reusejp_2605_;
}
else
{
lean_object* v_reuseFailAlloc_2610_; 
v_reuseFailAlloc_2610_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2610_, 0, v_mctx_2597_);
lean_ctor_set(v_reuseFailAlloc_2610_, 1, v___x_2604_);
lean_ctor_set(v_reuseFailAlloc_2610_, 2, v_zetaDeltaFVarIds_2598_);
lean_ctor_set(v_reuseFailAlloc_2610_, 3, v_postponed_2599_);
lean_ctor_set(v_reuseFailAlloc_2610_, 4, v_diag_2600_);
v___x_2606_ = v_reuseFailAlloc_2610_;
goto v_reusejp_2605_;
}
v_reusejp_2605_:
{
lean_object* v___x_2607_; lean_object* v___x_2608_; lean_object* v___x_2609_; 
v___x_2607_ = lean_st_ref_set(v___y_2578_, v___x_2606_);
v___x_2608_ = lean_box(0);
v___x_2609_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2609_, 0, v___x_2608_);
return v___x_2609_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg___boxed(lean_object* v_env_2617_, lean_object* v___y_2618_, lean_object* v___y_2619_, lean_object* v___y_2620_){
_start:
{
lean_object* v_res_2621_; 
v_res_2621_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg(v_env_2617_, v___y_2618_, v___y_2619_);
lean_dec(v___y_2619_);
lean_dec(v___y_2618_);
return v_res_2621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___redArg(lean_object* v_env_2622_, lean_object* v_x_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_, lean_object* v___y_2626_, lean_object* v___y_2627_, lean_object* v___y_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_){
_start:
{
lean_object* v___x_2633_; lean_object* v_env_2634_; lean_object* v_a_2636_; lean_object* v___x_2646_; lean_object* v___x_2647_; 
v___x_2633_ = lean_st_ref_get(v___y_2631_);
v_env_2634_ = lean_ctor_get(v___x_2633_, 0);
lean_inc_ref(v_env_2634_);
lean_dec(v___x_2633_);
v___x_2646_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg(v_env_2622_, v___y_2629_, v___y_2631_);
lean_dec_ref(v___x_2646_);
lean_inc(v___y_2631_);
lean_inc_ref(v___y_2630_);
lean_inc(v___y_2629_);
lean_inc_ref(v___y_2628_);
lean_inc(v___y_2627_);
lean_inc_ref(v___y_2626_);
lean_inc(v___y_2625_);
lean_inc_ref(v___y_2624_);
v___x_2647_ = lean_apply_9(v_x_2623_, v___y_2624_, v___y_2625_, v___y_2626_, v___y_2627_, v___y_2628_, v___y_2629_, v___y_2630_, v___y_2631_, lean_box(0));
if (lean_obj_tag(v___x_2647_) == 0)
{
lean_object* v_a_2648_; lean_object* v___x_2649_; lean_object* v___x_2651_; uint8_t v_isShared_2652_; uint8_t v_isSharedCheck_2656_; 
v_a_2648_ = lean_ctor_get(v___x_2647_, 0);
lean_inc(v_a_2648_);
lean_dec_ref_known(v___x_2647_, 1);
v___x_2649_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg(v_env_2634_, v___y_2629_, v___y_2631_);
v_isSharedCheck_2656_ = !lean_is_exclusive(v___x_2649_);
if (v_isSharedCheck_2656_ == 0)
{
lean_object* v_unused_2657_; 
v_unused_2657_ = lean_ctor_get(v___x_2649_, 0);
lean_dec(v_unused_2657_);
v___x_2651_ = v___x_2649_;
v_isShared_2652_ = v_isSharedCheck_2656_;
goto v_resetjp_2650_;
}
else
{
lean_dec(v___x_2649_);
v___x_2651_ = lean_box(0);
v_isShared_2652_ = v_isSharedCheck_2656_;
goto v_resetjp_2650_;
}
v_resetjp_2650_:
{
lean_object* v___x_2654_; 
if (v_isShared_2652_ == 0)
{
lean_ctor_set(v___x_2651_, 0, v_a_2648_);
v___x_2654_ = v___x_2651_;
goto v_reusejp_2653_;
}
else
{
lean_object* v_reuseFailAlloc_2655_; 
v_reuseFailAlloc_2655_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2655_, 0, v_a_2648_);
v___x_2654_ = v_reuseFailAlloc_2655_;
goto v_reusejp_2653_;
}
v_reusejp_2653_:
{
return v___x_2654_;
}
}
}
else
{
lean_object* v_a_2658_; 
v_a_2658_ = lean_ctor_get(v___x_2647_, 0);
lean_inc(v_a_2658_);
lean_dec_ref_known(v___x_2647_, 1);
v_a_2636_ = v_a_2658_;
goto v___jp_2635_;
}
v___jp_2635_:
{
lean_object* v___x_2637_; lean_object* v___x_2639_; uint8_t v_isShared_2640_; uint8_t v_isSharedCheck_2644_; 
v___x_2637_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg(v_env_2634_, v___y_2629_, v___y_2631_);
v_isSharedCheck_2644_ = !lean_is_exclusive(v___x_2637_);
if (v_isSharedCheck_2644_ == 0)
{
lean_object* v_unused_2645_; 
v_unused_2645_ = lean_ctor_get(v___x_2637_, 0);
lean_dec(v_unused_2645_);
v___x_2639_ = v___x_2637_;
v_isShared_2640_ = v_isSharedCheck_2644_;
goto v_resetjp_2638_;
}
else
{
lean_dec(v___x_2637_);
v___x_2639_ = lean_box(0);
v_isShared_2640_ = v_isSharedCheck_2644_;
goto v_resetjp_2638_;
}
v_resetjp_2638_:
{
lean_object* v___x_2642_; 
if (v_isShared_2640_ == 0)
{
lean_ctor_set_tag(v___x_2639_, 1);
lean_ctor_set(v___x_2639_, 0, v_a_2636_);
v___x_2642_ = v___x_2639_;
goto v_reusejp_2641_;
}
else
{
lean_object* v_reuseFailAlloc_2643_; 
v_reuseFailAlloc_2643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2643_, 0, v_a_2636_);
v___x_2642_ = v_reuseFailAlloc_2643_;
goto v_reusejp_2641_;
}
v_reusejp_2641_:
{
return v___x_2642_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___redArg___boxed(lean_object* v_env_2659_, lean_object* v_x_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_, lean_object* v___y_2669_){
_start:
{
lean_object* v_res_2670_; 
v_res_2670_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___redArg(v_env_2659_, v_x_2660_, v___y_2661_, v___y_2662_, v___y_2663_, v___y_2664_, v___y_2665_, v___y_2666_, v___y_2667_, v___y_2668_);
lean_dec(v___y_2668_);
lean_dec_ref(v___y_2667_);
lean_dec(v___y_2666_);
lean_dec_ref(v___y_2665_);
lean_dec(v___y_2664_);
lean_dec_ref(v___y_2663_);
lean_dec(v___y_2662_);
lean_dec_ref(v___y_2661_);
return v_res_2670_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0(uint8_t v___y_2678_, uint8_t v_suppressElabErrors_2679_, lean_object* v_x_2680_){
_start:
{
if (lean_obj_tag(v_x_2680_) == 1)
{
lean_object* v_pre_2681_; 
v_pre_2681_ = lean_ctor_get(v_x_2680_, 0);
switch(lean_obj_tag(v_pre_2681_))
{
case 1:
{
lean_object* v_pre_2682_; 
v_pre_2682_ = lean_ctor_get(v_pre_2681_, 0);
switch(lean_obj_tag(v_pre_2682_))
{
case 0:
{
lean_object* v_str_2683_; lean_object* v_str_2684_; lean_object* v___x_2685_; uint8_t v___x_2686_; 
v_str_2683_ = lean_ctor_get(v_x_2680_, 1);
v_str_2684_ = lean_ctor_get(v_pre_2681_, 1);
v___x_2685_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__0));
v___x_2686_ = lean_string_dec_eq(v_str_2684_, v___x_2685_);
if (v___x_2686_ == 0)
{
lean_object* v___x_2687_; uint8_t v___x_2688_; 
v___x_2687_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2));
v___x_2688_ = lean_string_dec_eq(v_str_2684_, v___x_2687_);
if (v___x_2688_ == 0)
{
return v___y_2678_;
}
else
{
lean_object* v___x_2689_; uint8_t v___x_2690_; 
v___x_2689_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__1));
v___x_2690_ = lean_string_dec_eq(v_str_2683_, v___x_2689_);
if (v___x_2690_ == 0)
{
return v___y_2678_;
}
else
{
return v_suppressElabErrors_2679_;
}
}
}
else
{
lean_object* v___x_2691_; uint8_t v___x_2692_; 
v___x_2691_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__2));
v___x_2692_ = lean_string_dec_eq(v_str_2683_, v___x_2691_);
if (v___x_2692_ == 0)
{
return v___y_2678_;
}
else
{
return v_suppressElabErrors_2679_;
}
}
}
case 1:
{
lean_object* v_pre_2693_; 
v_pre_2693_ = lean_ctor_get(v_pre_2682_, 0);
if (lean_obj_tag(v_pre_2693_) == 0)
{
lean_object* v_str_2694_; lean_object* v_str_2695_; lean_object* v_str_2696_; lean_object* v___x_2697_; uint8_t v___x_2698_; 
v_str_2694_ = lean_ctor_get(v_x_2680_, 1);
v_str_2695_ = lean_ctor_get(v_pre_2681_, 1);
v_str_2696_ = lean_ctor_get(v_pre_2682_, 1);
v___x_2697_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__3));
v___x_2698_ = lean_string_dec_eq(v_str_2696_, v___x_2697_);
if (v___x_2698_ == 0)
{
return v___y_2678_;
}
else
{
lean_object* v___x_2699_; uint8_t v___x_2700_; 
v___x_2699_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__4));
v___x_2700_ = lean_string_dec_eq(v_str_2695_, v___x_2699_);
if (v___x_2700_ == 0)
{
return v___y_2678_;
}
else
{
lean_object* v___x_2701_; uint8_t v___x_2702_; 
v___x_2701_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__5));
v___x_2702_ = lean_string_dec_eq(v_str_2694_, v___x_2701_);
if (v___x_2702_ == 0)
{
return v___y_2678_;
}
else
{
return v_suppressElabErrors_2679_;
}
}
}
}
else
{
return v___y_2678_;
}
}
default: 
{
return v___y_2678_;
}
}
}
case 0:
{
lean_object* v_str_2703_; lean_object* v___x_2704_; uint8_t v___x_2705_; 
v_str_2703_ = lean_ctor_get(v_x_2680_, 1);
v___x_2704_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___closed__6));
v___x_2705_ = lean_string_dec_eq(v_str_2703_, v___x_2704_);
if (v___x_2705_ == 0)
{
return v___y_2678_;
}
else
{
return v_suppressElabErrors_2679_;
}
}
default: 
{
return v___y_2678_;
}
}
}
else
{
return v___y_2678_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___boxed(lean_object* v___y_2706_, lean_object* v_suppressElabErrors_2707_, lean_object* v_x_2708_){
_start:
{
uint8_t v___y_17038__boxed_2709_; uint8_t v_suppressElabErrors_boxed_2710_; uint8_t v_res_2711_; lean_object* v_r_2712_; 
v___y_17038__boxed_2709_ = lean_unbox(v___y_2706_);
v_suppressElabErrors_boxed_2710_ = lean_unbox(v_suppressElabErrors_2707_);
v_res_2711_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0(v___y_17038__boxed_2709_, v_suppressElabErrors_boxed_2710_, v_x_2708_);
lean_dec(v_x_2708_);
v_r_2712_ = lean_box(v_res_2711_);
return v_r_2712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg(lean_object* v_ref_2713_, lean_object* v_msgData_2714_, uint8_t v_severity_2715_, uint8_t v_isSilent_2716_, lean_object* v___y_2717_, lean_object* v___y_2718_, lean_object* v___y_2719_, lean_object* v___y_2720_){
_start:
{
lean_object* v___y_2723_; uint8_t v___y_2724_; lean_object* v___y_2725_; lean_object* v___y_2726_; lean_object* v___y_2727_; uint8_t v___y_2728_; lean_object* v___y_2729_; lean_object* v___y_2730_; lean_object* v___y_2731_; lean_object* v___y_2759_; lean_object* v___y_2760_; lean_object* v___y_2761_; uint8_t v___y_2762_; uint8_t v___y_2763_; lean_object* v___y_2764_; uint8_t v___y_2765_; lean_object* v___y_2766_; lean_object* v___y_2784_; lean_object* v___y_2785_; uint8_t v___y_2786_; lean_object* v___y_2787_; uint8_t v___y_2788_; lean_object* v___y_2789_; uint8_t v___y_2790_; lean_object* v___y_2791_; lean_object* v___y_2795_; lean_object* v___y_2796_; uint8_t v___y_2797_; lean_object* v___y_2798_; lean_object* v___y_2799_; uint8_t v___y_2800_; uint8_t v___y_2801_; uint8_t v___x_2806_; lean_object* v___y_2808_; uint8_t v___y_2809_; lean_object* v___y_2810_; lean_object* v___y_2811_; lean_object* v___y_2812_; uint8_t v___y_2813_; uint8_t v___y_2814_; uint8_t v___y_2816_; uint8_t v___x_2831_; 
v___x_2806_ = 2;
v___x_2831_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2715_, v___x_2806_);
if (v___x_2831_ == 0)
{
v___y_2816_ = v___x_2831_;
goto v___jp_2815_;
}
else
{
uint8_t v___x_2832_; 
lean_inc_ref(v_msgData_2714_);
v___x_2832_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2714_);
v___y_2816_ = v___x_2832_;
goto v___jp_2815_;
}
v___jp_2722_:
{
lean_object* v___x_2732_; lean_object* v_currNamespace_2733_; lean_object* v_openDecls_2734_; lean_object* v_env_2735_; lean_object* v_nextMacroScope_2736_; lean_object* v_ngen_2737_; lean_object* v_auxDeclNGen_2738_; lean_object* v_traceState_2739_; lean_object* v_cache_2740_; lean_object* v_messages_2741_; lean_object* v_infoState_2742_; lean_object* v_snapshotTasks_2743_; lean_object* v___x_2745_; uint8_t v_isShared_2746_; uint8_t v_isSharedCheck_2757_; 
v___x_2732_ = lean_st_ref_take(v___y_2731_);
v_currNamespace_2733_ = lean_ctor_get(v___y_2730_, 6);
v_openDecls_2734_ = lean_ctor_get(v___y_2730_, 7);
v_env_2735_ = lean_ctor_get(v___x_2732_, 0);
v_nextMacroScope_2736_ = lean_ctor_get(v___x_2732_, 1);
v_ngen_2737_ = lean_ctor_get(v___x_2732_, 2);
v_auxDeclNGen_2738_ = lean_ctor_get(v___x_2732_, 3);
v_traceState_2739_ = lean_ctor_get(v___x_2732_, 4);
v_cache_2740_ = lean_ctor_get(v___x_2732_, 5);
v_messages_2741_ = lean_ctor_get(v___x_2732_, 6);
v_infoState_2742_ = lean_ctor_get(v___x_2732_, 7);
v_snapshotTasks_2743_ = lean_ctor_get(v___x_2732_, 8);
v_isSharedCheck_2757_ = !lean_is_exclusive(v___x_2732_);
if (v_isSharedCheck_2757_ == 0)
{
v___x_2745_ = v___x_2732_;
v_isShared_2746_ = v_isSharedCheck_2757_;
goto v_resetjp_2744_;
}
else
{
lean_inc(v_snapshotTasks_2743_);
lean_inc(v_infoState_2742_);
lean_inc(v_messages_2741_);
lean_inc(v_cache_2740_);
lean_inc(v_traceState_2739_);
lean_inc(v_auxDeclNGen_2738_);
lean_inc(v_ngen_2737_);
lean_inc(v_nextMacroScope_2736_);
lean_inc(v_env_2735_);
lean_dec(v___x_2732_);
v___x_2745_ = lean_box(0);
v_isShared_2746_ = v_isSharedCheck_2757_;
goto v_resetjp_2744_;
}
v_resetjp_2744_:
{
lean_object* v___x_2747_; lean_object* v___x_2748_; lean_object* v___x_2749_; lean_object* v___x_2750_; lean_object* v___x_2752_; 
lean_inc(v_openDecls_2734_);
lean_inc(v_currNamespace_2733_);
v___x_2747_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2747_, 0, v_currNamespace_2733_);
lean_ctor_set(v___x_2747_, 1, v_openDecls_2734_);
v___x_2748_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2748_, 0, v___x_2747_);
lean_ctor_set(v___x_2748_, 1, v___y_2729_);
lean_inc_ref(v___y_2723_);
lean_inc_ref(v___y_2726_);
v___x_2749_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2749_, 0, v___y_2726_);
lean_ctor_set(v___x_2749_, 1, v___y_2725_);
lean_ctor_set(v___x_2749_, 2, v___y_2727_);
lean_ctor_set(v___x_2749_, 3, v___y_2723_);
lean_ctor_set(v___x_2749_, 4, v___x_2748_);
lean_ctor_set_uint8(v___x_2749_, sizeof(void*)*5, v___y_2728_);
lean_ctor_set_uint8(v___x_2749_, sizeof(void*)*5 + 1, v___y_2724_);
lean_ctor_set_uint8(v___x_2749_, sizeof(void*)*5 + 2, v_isSilent_2716_);
v___x_2750_ = l_Lean_MessageLog_add(v___x_2749_, v_messages_2741_);
if (v_isShared_2746_ == 0)
{
lean_ctor_set(v___x_2745_, 6, v___x_2750_);
v___x_2752_ = v___x_2745_;
goto v_reusejp_2751_;
}
else
{
lean_object* v_reuseFailAlloc_2756_; 
v_reuseFailAlloc_2756_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2756_, 0, v_env_2735_);
lean_ctor_set(v_reuseFailAlloc_2756_, 1, v_nextMacroScope_2736_);
lean_ctor_set(v_reuseFailAlloc_2756_, 2, v_ngen_2737_);
lean_ctor_set(v_reuseFailAlloc_2756_, 3, v_auxDeclNGen_2738_);
lean_ctor_set(v_reuseFailAlloc_2756_, 4, v_traceState_2739_);
lean_ctor_set(v_reuseFailAlloc_2756_, 5, v_cache_2740_);
lean_ctor_set(v_reuseFailAlloc_2756_, 6, v___x_2750_);
lean_ctor_set(v_reuseFailAlloc_2756_, 7, v_infoState_2742_);
lean_ctor_set(v_reuseFailAlloc_2756_, 8, v_snapshotTasks_2743_);
v___x_2752_ = v_reuseFailAlloc_2756_;
goto v_reusejp_2751_;
}
v_reusejp_2751_:
{
lean_object* v___x_2753_; lean_object* v___x_2754_; lean_object* v___x_2755_; 
v___x_2753_ = lean_st_ref_set(v___y_2731_, v___x_2752_);
v___x_2754_ = lean_box(0);
v___x_2755_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2755_, 0, v___x_2754_);
return v___x_2755_;
}
}
}
v___jp_2758_:
{
lean_object* v___x_2767_; lean_object* v___x_2768_; lean_object* v_a_2769_; lean_object* v___x_2771_; uint8_t v_isShared_2772_; uint8_t v_isSharedCheck_2782_; 
v___x_2767_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2714_);
v___x_2768_ = lp_mathlib_Lean_addMessageContextFull___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__3(v___x_2767_, v___y_2717_, v___y_2718_, v___y_2719_, v___y_2720_);
v_a_2769_ = lean_ctor_get(v___x_2768_, 0);
v_isSharedCheck_2782_ = !lean_is_exclusive(v___x_2768_);
if (v_isSharedCheck_2782_ == 0)
{
v___x_2771_ = v___x_2768_;
v_isShared_2772_ = v_isSharedCheck_2782_;
goto v_resetjp_2770_;
}
else
{
lean_inc(v_a_2769_);
lean_dec(v___x_2768_);
v___x_2771_ = lean_box(0);
v_isShared_2772_ = v_isSharedCheck_2782_;
goto v_resetjp_2770_;
}
v_resetjp_2770_:
{
lean_object* v___x_2773_; lean_object* v___x_2774_; lean_object* v___x_2775_; lean_object* v___x_2776_; 
lean_inc_ref_n(v___y_2761_, 2);
v___x_2773_ = l_Lean_FileMap_toPosition(v___y_2761_, v___y_2760_);
lean_dec(v___y_2760_);
v___x_2774_ = l_Lean_FileMap_toPosition(v___y_2761_, v___y_2766_);
lean_dec(v___y_2766_);
v___x_2775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2775_, 0, v___x_2774_);
v___x_2776_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_goalSignature___lam__1___closed__7));
if (v___y_2762_ == 0)
{
lean_del_object(v___x_2771_);
lean_dec_ref(v___y_2759_);
v___y_2723_ = v___x_2776_;
v___y_2724_ = v___y_2763_;
v___y_2725_ = v___x_2773_;
v___y_2726_ = v___y_2764_;
v___y_2727_ = v___x_2775_;
v___y_2728_ = v___y_2765_;
v___y_2729_ = v_a_2769_;
v___y_2730_ = v___y_2719_;
v___y_2731_ = v___y_2720_;
goto v___jp_2722_;
}
else
{
uint8_t v___x_2777_; 
lean_inc(v_a_2769_);
v___x_2777_ = l_Lean_MessageData_hasTag(v___y_2759_, v_a_2769_);
if (v___x_2777_ == 0)
{
lean_object* v___x_2778_; lean_object* v___x_2780_; 
lean_dec_ref_known(v___x_2775_, 1);
lean_dec_ref(v___x_2773_);
lean_dec(v_a_2769_);
v___x_2778_ = lean_box(0);
if (v_isShared_2772_ == 0)
{
lean_ctor_set(v___x_2771_, 0, v___x_2778_);
v___x_2780_ = v___x_2771_;
goto v_reusejp_2779_;
}
else
{
lean_object* v_reuseFailAlloc_2781_; 
v_reuseFailAlloc_2781_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2781_, 0, v___x_2778_);
v___x_2780_ = v_reuseFailAlloc_2781_;
goto v_reusejp_2779_;
}
v_reusejp_2779_:
{
return v___x_2780_;
}
}
else
{
lean_del_object(v___x_2771_);
v___y_2723_ = v___x_2776_;
v___y_2724_ = v___y_2763_;
v___y_2725_ = v___x_2773_;
v___y_2726_ = v___y_2764_;
v___y_2727_ = v___x_2775_;
v___y_2728_ = v___y_2765_;
v___y_2729_ = v_a_2769_;
v___y_2730_ = v___y_2719_;
v___y_2731_ = v___y_2720_;
goto v___jp_2722_;
}
}
}
}
v___jp_2783_:
{
lean_object* v___x_2792_; 
v___x_2792_ = l_Lean_Syntax_getTailPos_x3f(v___y_2787_, v___y_2790_);
lean_dec(v___y_2787_);
if (lean_obj_tag(v___x_2792_) == 0)
{
lean_inc(v___y_2791_);
v___y_2759_ = v___y_2784_;
v___y_2760_ = v___y_2791_;
v___y_2761_ = v___y_2785_;
v___y_2762_ = v___y_2786_;
v___y_2763_ = v___y_2788_;
v___y_2764_ = v___y_2789_;
v___y_2765_ = v___y_2790_;
v___y_2766_ = v___y_2791_;
goto v___jp_2758_;
}
else
{
lean_object* v_val_2793_; 
v_val_2793_ = lean_ctor_get(v___x_2792_, 0);
lean_inc(v_val_2793_);
lean_dec_ref_known(v___x_2792_, 1);
v___y_2759_ = v___y_2784_;
v___y_2760_ = v___y_2791_;
v___y_2761_ = v___y_2785_;
v___y_2762_ = v___y_2786_;
v___y_2763_ = v___y_2788_;
v___y_2764_ = v___y_2789_;
v___y_2765_ = v___y_2790_;
v___y_2766_ = v_val_2793_;
goto v___jp_2758_;
}
}
v___jp_2794_:
{
lean_object* v_ref_2802_; lean_object* v___x_2803_; 
v_ref_2802_ = l_Lean_replaceRef(v_ref_2713_, v___y_2799_);
v___x_2803_ = l_Lean_Syntax_getPos_x3f(v_ref_2802_, v___y_2800_);
if (lean_obj_tag(v___x_2803_) == 0)
{
lean_object* v___x_2804_; 
v___x_2804_ = lean_unsigned_to_nat(0u);
v___y_2784_ = v___y_2795_;
v___y_2785_ = v___y_2796_;
v___y_2786_ = v___y_2797_;
v___y_2787_ = v_ref_2802_;
v___y_2788_ = v___y_2801_;
v___y_2789_ = v___y_2798_;
v___y_2790_ = v___y_2800_;
v___y_2791_ = v___x_2804_;
goto v___jp_2783_;
}
else
{
lean_object* v_val_2805_; 
v_val_2805_ = lean_ctor_get(v___x_2803_, 0);
lean_inc(v_val_2805_);
lean_dec_ref_known(v___x_2803_, 1);
v___y_2784_ = v___y_2795_;
v___y_2785_ = v___y_2796_;
v___y_2786_ = v___y_2797_;
v___y_2787_ = v_ref_2802_;
v___y_2788_ = v___y_2801_;
v___y_2789_ = v___y_2798_;
v___y_2790_ = v___y_2800_;
v___y_2791_ = v_val_2805_;
goto v___jp_2783_;
}
}
v___jp_2807_:
{
if (v___y_2814_ == 0)
{
v___y_2795_ = v___y_2810_;
v___y_2796_ = v___y_2808_;
v___y_2797_ = v___y_2809_;
v___y_2798_ = v___y_2811_;
v___y_2799_ = v___y_2812_;
v___y_2800_ = v___y_2813_;
v___y_2801_ = v_severity_2715_;
goto v___jp_2794_;
}
else
{
v___y_2795_ = v___y_2810_;
v___y_2796_ = v___y_2808_;
v___y_2797_ = v___y_2809_;
v___y_2798_ = v___y_2811_;
v___y_2799_ = v___y_2812_;
v___y_2800_ = v___y_2813_;
v___y_2801_ = v___x_2806_;
goto v___jp_2794_;
}
}
v___jp_2815_:
{
if (v___y_2816_ == 0)
{
lean_object* v_fileName_2817_; lean_object* v_fileMap_2818_; lean_object* v_options_2819_; lean_object* v_ref_2820_; uint8_t v_suppressElabErrors_2821_; lean_object* v___x_2822_; lean_object* v___x_2823_; lean_object* v___f_2824_; uint8_t v___x_2825_; uint8_t v___x_2826_; 
v_fileName_2817_ = lean_ctor_get(v___y_2719_, 0);
v_fileMap_2818_ = lean_ctor_get(v___y_2719_, 1);
v_options_2819_ = lean_ctor_get(v___y_2719_, 2);
v_ref_2820_ = lean_ctor_get(v___y_2719_, 5);
v_suppressElabErrors_2821_ = lean_ctor_get_uint8(v___y_2719_, sizeof(void*)*14 + 1);
v___x_2822_ = lean_box(v___y_2816_);
v___x_2823_ = lean_box(v_suppressElabErrors_2821_);
v___f_2824_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2824_, 0, v___x_2822_);
lean_closure_set(v___f_2824_, 1, v___x_2823_);
v___x_2825_ = 1;
v___x_2826_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2715_, v___x_2825_);
if (v___x_2826_ == 0)
{
v___y_2808_ = v_fileMap_2818_;
v___y_2809_ = v_suppressElabErrors_2821_;
v___y_2810_ = v___f_2824_;
v___y_2811_ = v_fileName_2817_;
v___y_2812_ = v_ref_2820_;
v___y_2813_ = v___y_2816_;
v___y_2814_ = v___x_2826_;
goto v___jp_2807_;
}
else
{
lean_object* v___x_2827_; uint8_t v___x_2828_; 
v___x_2827_ = l_Lean_warningAsError;
v___x_2828_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtractGoal_goalSignature_spec__12_spec__19_spec__22(v_options_2819_, v___x_2827_);
v___y_2808_ = v_fileMap_2818_;
v___y_2809_ = v_suppressElabErrors_2821_;
v___y_2810_ = v___f_2824_;
v___y_2811_ = v_fileName_2817_;
v___y_2812_ = v_ref_2820_;
v___y_2813_ = v___y_2816_;
v___y_2814_ = v___x_2828_;
goto v___jp_2807_;
}
}
else
{
lean_object* v___x_2829_; lean_object* v___x_2830_; 
lean_dec_ref(v_msgData_2714_);
v___x_2829_ = lean_box(0);
v___x_2830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2830_, 0, v___x_2829_);
return v___x_2830_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg___boxed(lean_object* v_ref_2833_, lean_object* v_msgData_2834_, lean_object* v_severity_2835_, lean_object* v_isSilent_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_, lean_object* v___y_2839_, lean_object* v___y_2840_, lean_object* v___y_2841_){
_start:
{
uint8_t v_severity_boxed_2842_; uint8_t v_isSilent_boxed_2843_; lean_object* v_res_2844_; 
v_severity_boxed_2842_ = lean_unbox(v_severity_2835_);
v_isSilent_boxed_2843_ = lean_unbox(v_isSilent_2836_);
v_res_2844_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg(v_ref_2833_, v_msgData_2834_, v_severity_boxed_2842_, v_isSilent_boxed_2843_, v___y_2837_, v___y_2838_, v___y_2839_, v___y_2840_);
lean_dec(v___y_2840_);
lean_dec_ref(v___y_2839_);
lean_dec(v___y_2838_);
lean_dec_ref(v___y_2837_);
lean_dec(v_ref_2833_);
return v_res_2844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5(lean_object* v_msgData_2845_, uint8_t v_severity_2846_, uint8_t v_isSilent_2847_, lean_object* v___y_2848_, lean_object* v___y_2849_, lean_object* v___y_2850_, lean_object* v___y_2851_, lean_object* v___y_2852_, lean_object* v___y_2853_, lean_object* v___y_2854_, lean_object* v___y_2855_){
_start:
{
lean_object* v_ref_2857_; lean_object* v___x_2858_; 
v_ref_2857_ = lean_ctor_get(v___y_2854_, 5);
v___x_2858_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg(v_ref_2857_, v_msgData_2845_, v_severity_2846_, v_isSilent_2847_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
return v___x_2858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5___boxed(lean_object* v_msgData_2859_, lean_object* v_severity_2860_, lean_object* v_isSilent_2861_, lean_object* v___y_2862_, lean_object* v___y_2863_, lean_object* v___y_2864_, lean_object* v___y_2865_, lean_object* v___y_2866_, lean_object* v___y_2867_, lean_object* v___y_2868_, lean_object* v___y_2869_, lean_object* v___y_2870_){
_start:
{
uint8_t v_severity_boxed_2871_; uint8_t v_isSilent_boxed_2872_; lean_object* v_res_2873_; 
v_severity_boxed_2871_ = lean_unbox(v_severity_2860_);
v_isSilent_boxed_2872_ = lean_unbox(v_isSilent_2861_);
v_res_2873_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5(v_msgData_2859_, v_severity_boxed_2871_, v_isSilent_boxed_2872_, v___y_2862_, v___y_2863_, v___y_2864_, v___y_2865_, v___y_2866_, v___y_2867_, v___y_2868_, v___y_2869_);
lean_dec(v___y_2869_);
lean_dec_ref(v___y_2868_);
lean_dec(v___y_2867_);
lean_dec_ref(v___y_2866_);
lean_dec(v___y_2865_);
lean_dec_ref(v___y_2864_);
lean_dec(v___y_2863_);
lean_dec_ref(v___y_2862_);
return v_res_2873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4(lean_object* v_msgData_2874_, lean_object* v___y_2875_, lean_object* v___y_2876_, lean_object* v___y_2877_, lean_object* v___y_2878_, lean_object* v___y_2879_, lean_object* v___y_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_){
_start:
{
uint8_t v___x_2884_; uint8_t v___x_2885_; lean_object* v___x_2886_; 
v___x_2884_ = 0;
v___x_2885_ = 0;
v___x_2886_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5(v_msgData_2874_, v___x_2884_, v___x_2885_, v___y_2875_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_, v___y_2880_, v___y_2881_, v___y_2882_);
return v___x_2886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4___boxed(lean_object* v_msgData_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_){
_start:
{
lean_object* v_res_2897_; 
v_res_2897_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4(v_msgData_2887_, v___y_2888_, v___y_2889_, v___y_2890_, v___y_2891_, v___y_2892_, v___y_2893_, v___y_2894_, v___y_2895_);
lean_dec(v___y_2895_);
lean_dec_ref(v___y_2894_);
lean_dec(v___y_2893_);
lean_dec_ref(v___y_2892_);
lean_dec(v___y_2891_);
lean_dec_ref(v___y_2890_);
lean_dec(v___y_2889_);
lean_dec_ref(v___y_2888_);
return v_res_2897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1(lean_object* v_x_2901_, lean_object* v_a_2902_, lean_object* v_a_2903_, lean_object* v_a_2904_, lean_object* v_a_2905_, lean_object* v_a_2906_, lean_object* v_a_2907_, lean_object* v_a_2908_, lean_object* v_a_2909_){
_start:
{
lean_object* v___x_2911_; lean_object* v___x_2912_; lean_object* v___x_2913_; lean_object* v___x_2914_; uint8_t v___x_2915_; 
v___x_2911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__1));
v___x_2912_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__2));
v___x_2913_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_star___closed__3));
v___x_2914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_extractGoal___closed__1));
lean_inc(v_x_2901_);
v___x_2915_ = l_Lean_Syntax_isOfKind(v_x_2901_, v___x_2914_);
if (v___x_2915_ == 0)
{
lean_object* v___x_2916_; 
lean_dec(v_x_2901_);
v___x_2916_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg();
return v___x_2916_;
}
else
{
lean_object* v___x_2917_; lean_object* v___x_2918_; lean_object* v___x_2919_; uint8_t v___x_2920_; 
v___x_2917_ = lean_unsigned_to_nat(1u);
v___x_2918_ = l_Lean_Syntax_getArg(v_x_2901_, v___x_2917_);
v___x_2919_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal_config___closed__1));
lean_inc(v___x_2918_);
v___x_2920_ = l_Lean_Syntax_isOfKind(v___x_2918_, v___x_2919_);
if (v___x_2920_ == 0)
{
lean_object* v___x_2921_; 
lean_dec(v___x_2918_);
lean_dec(v_x_2901_);
v___x_2921_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg();
return v___x_2921_;
}
else
{
lean_object* v___x_2922_; lean_object* v_name_2924_; lean_object* v___y_2925_; lean_object* v___y_2926_; lean_object* v___y_2927_; lean_object* v___y_2928_; lean_object* v___y_2929_; lean_object* v___y_2930_; lean_object* v___y_2931_; lean_object* v___y_2932_; lean_object* v___x_2950_; lean_object* v___x_2951_; uint8_t v___x_2952_; 
v___x_2922_ = lean_unsigned_to_nat(0u);
v___x_2950_ = lean_unsigned_to_nat(2u);
v___x_2951_ = l_Lean_Syntax_getArg(v_x_2901_, v___x_2950_);
lean_dec(v_x_2901_);
v___x_2952_ = l_Lean_Syntax_isNone(v___x_2951_);
if (v___x_2952_ == 0)
{
uint8_t v___x_2953_; 
lean_inc(v___x_2951_);
v___x_2953_ = l_Lean_Syntax_matchesNull(v___x_2951_, v___x_2950_);
if (v___x_2953_ == 0)
{
lean_object* v___x_2954_; 
lean_dec(v___x_2951_);
lean_dec(v___x_2918_);
v___x_2954_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__0___redArg();
return v___x_2954_;
}
else
{
lean_object* v_name_x3f_2955_; lean_object* v___x_2956_; 
v_name_x3f_2955_ = l_Lean_Syntax_getArg(v___x_2951_, v___x_2917_);
lean_dec(v___x_2951_);
v___x_2956_ = l_Lean_TSyntax_getId(v_name_x3f_2955_);
lean_dec(v_name_x3f_2955_);
v_name_2924_ = v___x_2956_;
v___y_2925_ = v_a_2902_;
v___y_2926_ = v_a_2903_;
v___y_2927_ = v_a_2904_;
v___y_2928_ = v_a_2905_;
v___y_2929_ = v_a_2906_;
v___y_2930_ = v_a_2907_;
v___y_2931_ = v_a_2908_;
v___y_2932_ = v_a_2909_;
goto v___jp_2923_;
}
}
else
{
lean_object* v___x_2957_; lean_object* v___x_2958_; lean_object* v_a_2959_; 
lean_dec(v___x_2951_);
v___x_2957_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___closed__1));
v___x_2958_ = lp_mathlib_Lean_mkAuxDeclName___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__5___redArg(v___x_2957_, v_a_2909_);
v_a_2959_ = lean_ctor_get(v___x_2958_, 0);
lean_inc(v_a_2959_);
lean_dec_ref(v___x_2958_);
v_name_2924_ = v_a_2959_;
v___y_2925_ = v_a_2902_;
v___y_2926_ = v_a_2903_;
v___y_2927_ = v_a_2904_;
v___y_2928_ = v_a_2905_;
v___y_2929_ = v_a_2906_;
v___y_2930_ = v_a_2907_;
v___y_2931_ = v_a_2908_;
v___y_2932_ = v_a_2909_;
goto v___jp_2923_;
}
v___jp_2923_:
{
lean_object* v___x_2933_; lean_object* v_env_2934_; lean_object* v___x_2935_; lean_object* v___f_2936_; lean_object* v___x_2937_; lean_object* v___x_2938_; lean_object* v___x_2939_; 
v___x_2933_ = lean_st_ref_get(v___y_2932_);
v_env_2934_ = lean_ctor_get(v___x_2933_, 0);
lean_inc_ref(v_env_2934_);
lean_dec(v___x_2933_);
v___x_2935_ = lean_box(v___x_2920_);
v___f_2936_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___lam__0___boxed), 16, 7);
lean_closure_set(v___f_2936_, 0, v_name_2924_);
lean_closure_set(v___f_2936_, 1, v___x_2935_);
lean_closure_set(v___f_2936_, 2, v___x_2918_);
lean_closure_set(v___f_2936_, 3, v___x_2922_);
lean_closure_set(v___f_2936_, 4, v___x_2911_);
lean_closure_set(v___f_2936_, 5, v___x_2912_);
lean_closure_set(v___f_2936_, 6, v___x_2913_);
v___x_2937_ = lean_alloc_closure((void*)(lp_mathlib_Lean_withoutModifyingState___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__2___boxed), 11, 2);
lean_closure_set(v___x_2937_, 0, lean_box(0));
lean_closure_set(v___x_2937_, 1, v___f_2936_);
v___x_2938_ = l_Lean_Environment_unlockAsync(v_env_2934_);
v___x_2939_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___redArg(v___x_2938_, v___x_2937_, v___y_2925_, v___y_2926_, v___y_2927_, v___y_2928_, v___y_2929_, v___y_2930_, v___y_2931_, v___y_2932_);
if (lean_obj_tag(v___x_2939_) == 0)
{
lean_object* v_a_2940_; lean_object* v___x_2941_; 
v_a_2940_ = lean_ctor_get(v___x_2939_, 0);
lean_inc(v_a_2940_);
lean_dec_ref_known(v___x_2939_, 1);
v___x_2941_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4(v_a_2940_, v___y_2925_, v___y_2926_, v___y_2927_, v___y_2928_, v___y_2929_, v___y_2930_, v___y_2931_, v___y_2932_);
return v___x_2941_;
}
else
{
lean_object* v_a_2942_; lean_object* v___x_2944_; uint8_t v_isShared_2945_; uint8_t v_isSharedCheck_2949_; 
v_a_2942_ = lean_ctor_get(v___x_2939_, 0);
v_isSharedCheck_2949_ = !lean_is_exclusive(v___x_2939_);
if (v_isSharedCheck_2949_ == 0)
{
v___x_2944_ = v___x_2939_;
v_isShared_2945_ = v_isSharedCheck_2949_;
goto v_resetjp_2943_;
}
else
{
lean_inc(v_a_2942_);
lean_dec(v___x_2939_);
v___x_2944_ = lean_box(0);
v_isShared_2945_ = v_isSharedCheck_2949_;
goto v_resetjp_2943_;
}
v_resetjp_2943_:
{
lean_object* v___x_2947_; 
if (v_isShared_2945_ == 0)
{
v___x_2947_ = v___x_2944_;
goto v_reusejp_2946_;
}
else
{
lean_object* v_reuseFailAlloc_2948_; 
v_reuseFailAlloc_2948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2948_, 0, v_a_2942_);
v___x_2947_ = v_reuseFailAlloc_2948_;
goto v_reusejp_2946_;
}
v_reusejp_2946_:
{
return v___x_2947_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1___boxed(lean_object* v_x_2960_, lean_object* v_a_2961_, lean_object* v_a_2962_, lean_object* v_a_2963_, lean_object* v_a_2964_, lean_object* v_a_2965_, lean_object* v_a_2966_, lean_object* v_a_2967_, lean_object* v_a_2968_, lean_object* v_a_2969_){
_start:
{
lean_object* v_res_2970_; 
v_res_2970_ = lp_mathlib_Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1(v_x_2960_, v_a_2961_, v_a_2962_, v_a_2963_, v_a_2964_, v_a_2965_, v_a_2966_, v_a_2967_, v_a_2968_);
lean_dec(v_a_2968_);
lean_dec_ref(v_a_2967_);
lean_dec(v_a_2966_);
lean_dec_ref(v_a_2965_);
lean_dec(v_a_2964_);
lean_dec_ref(v_a_2963_);
lean_dec(v_a_2962_);
lean_dec_ref(v_a_2961_);
return v_res_2970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3(lean_object* v_env_2971_, lean_object* v___y_2972_, lean_object* v___y_2973_, lean_object* v___y_2974_, lean_object* v___y_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_){
_start:
{
lean_object* v___x_2981_; 
v___x_2981_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___redArg(v_env_2971_, v___y_2977_, v___y_2979_);
return v___x_2981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3___boxed(lean_object* v_env_2982_, lean_object* v___y_2983_, lean_object* v___y_2984_, lean_object* v___y_2985_, lean_object* v___y_2986_, lean_object* v___y_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_){
_start:
{
lean_object* v_res_2992_; 
v_res_2992_ = lp_mathlib_Lean_setEnv___at___00Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3_spec__3(v_env_2982_, v___y_2983_, v___y_2984_, v___y_2985_, v___y_2986_, v___y_2987_, v___y_2988_, v___y_2989_, v___y_2990_);
lean_dec(v___y_2990_);
lean_dec_ref(v___y_2989_);
lean_dec(v___y_2988_);
lean_dec_ref(v___y_2987_);
lean_dec(v___y_2986_);
lean_dec_ref(v___y_2985_);
lean_dec(v___y_2984_);
lean_dec_ref(v___y_2983_);
return v_res_2992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3(lean_object* v_00_u03b1_2993_, lean_object* v_env_2994_, lean_object* v_x_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_, lean_object* v___y_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_){
_start:
{
lean_object* v___x_3005_; 
v___x_3005_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___redArg(v_env_2994_, v_x_2995_, v___y_2996_, v___y_2997_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_, v___y_3002_, v___y_3003_);
return v___x_3005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3___boxed(lean_object* v_00_u03b1_3006_, lean_object* v_env_3007_, lean_object* v_x_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_){
_start:
{
lean_object* v_res_3018_; 
v_res_3018_ = lp_mathlib_Lean_withEnv___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__3(v_00_u03b1_3006_, v_env_3007_, v_x_3008_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_);
lean_dec(v___y_3016_);
lean_dec_ref(v___y_3015_);
lean_dec(v___y_3014_);
lean_dec_ref(v___y_3013_);
lean_dec(v___y_3012_);
lean_dec_ref(v___y_3011_);
lean_dec(v___y_3010_);
lean_dec_ref(v___y_3009_);
return v_res_3018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7(lean_object* v_ref_3019_, lean_object* v_msgData_3020_, uint8_t v_severity_3021_, uint8_t v_isSilent_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_){
_start:
{
lean_object* v___x_3032_; 
v___x_3032_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___redArg(v_ref_3019_, v_msgData_3020_, v_severity_3021_, v_isSilent_3022_, v___y_3027_, v___y_3028_, v___y_3029_, v___y_3030_);
return v___x_3032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7___boxed(lean_object* v_ref_3033_, lean_object* v_msgData_3034_, lean_object* v_severity_3035_, lean_object* v_isSilent_3036_, lean_object* v___y_3037_, lean_object* v___y_3038_, lean_object* v___y_3039_, lean_object* v___y_3040_, lean_object* v___y_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_, lean_object* v___y_3044_, lean_object* v___y_3045_){
_start:
{
uint8_t v_severity_boxed_3046_; uint8_t v_isSilent_boxed_3047_; lean_object* v_res_3048_; 
v_severity_boxed_3046_ = lean_unbox(v_severity_3035_);
v_isSilent_boxed_3047_ = lean_unbox(v_isSilent_3036_);
v_res_3048_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Tactic_ExtractGoal___aux__Mathlib__Tactic__ExtractGoal______elabRules__Mathlib__Tactic__ExtractGoal__extractGoal__1_spec__4_spec__5_spec__7(v_ref_3033_, v_msgData_3034_, v_severity_boxed_3046_, v_isSilent_boxed_3047_, v___y_3037_, v___y_3038_, v___y_3039_, v___y_3040_, v___y_3041_, v___y_3042_, v___y_3043_, v___y_3044_);
lean_dec(v___y_3044_);
lean_dec_ref(v___y_3043_);
lean_dec(v___y_3042_);
lean_dec_ref(v___y_3041_);
lean_dec(v___y_3040_);
lean_dec_ref(v___y_3039_);
lean_dec(v___y_3038_);
lean_dec_ref(v___y_3037_);
lean_dec(v_ref_3033_);
return v_res_3048_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ExtractGoal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Term(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Cleanup(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Inaccessible(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ExtractGoal(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Cleanup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Inaccessible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Term(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Cleanup(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_Inaccessible(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ExtractGoal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Cleanup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_PrettyPrinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Inaccessible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ExtractGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ExtractGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ExtractGoal(builtin);
}
#ifdef __cplusplus
}
#endif
