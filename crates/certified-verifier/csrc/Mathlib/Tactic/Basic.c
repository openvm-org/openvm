// Lean compiler output
// Module: Mathlib.Tactic.Basic
// Imports: public import Init public meta import Init public meta import Lean.Elab.BuiltinCommand public import Mathlib.Tactic.PPWithUniv public import Mathlib.Tactic.ExtendDoc public import Mathlib.Tactic.Linter.OldObtain public import Batteries.Util.LibraryNote
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
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_instInhabitedFileMap_default;
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_LocalDecl_isAuxDecl(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_MVarId_tryClear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_get_x21(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_Elab_pushInfoLeaf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Core_getMessageLog___redArg(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Elab_InfoTree_substitute(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabVariable(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_binderIdent;
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getNumArgs(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Array_extract___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_variables___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_variables___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_variables___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "variables"};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 148, 129, 172, 39, 121, 191, 39)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_variables___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_variables___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__7_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_variables___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__9_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_variables___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "bracketedBinder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__12_value),LEAN_SCALAR_PTR_LITERAL(126, 188, 9, 177, 18, 110, 216, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_variables___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_variables___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__18_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_variables = (const lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "'variables' has been replaced by 'variable' in lean 4"};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_elabVariables___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "variable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__5_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__6_value),LEAN_SCALAR_PTR_LITERAL(250, 93, 226, 106, 76, 14, 69, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_elabVariables___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_elabVariables___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabVariables(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_introv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "introv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_introv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_introv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_introv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(68, 207, 118, 66, 144, 156, 96, 220)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_introv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_introv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_introv___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__3_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_introv___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_introv___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_introv___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_introv___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_introv___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_introv___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_introv___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_introv___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_introv;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_introsDep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_introsDep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__2_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "seq1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__7_value),LEAN_SCALAR_PTR_LITERAL(242, 140, 137, 56, 141, 11, 143, 117)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__10_value),LEAN_SCALAR_PTR_LITERAL(41, 145, 9, 18, 75, 146, 159, 78)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_evalIntrov___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_evalIntrov___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticAssumption'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(107, 131, 17, 59, 169, 254, 12, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "assumption'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tacticAssumption_x27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "anyGoals"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 19, 163, 3, 232, 106, 175, 32)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "any_goals"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_elabVariables___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "clearAuxDecl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_variables___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__0_value),LEAN_SCALAR_PTR_LITERAL(180, 178, 107, 211, 121, 199, 195, 50)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "clear_aux_decl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_clearAuxDecl = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = lean_box(0);
v___x_43_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_44_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v___x_42_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg(){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0);
v___x_47_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___boxed(lean_object* v___y_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg();
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0(lean_object* v_00_u03b1_50_, lean_object* v___y_51_, lean_object* v___y_52_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg();
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___boxed(lean_object* v_00_u03b1_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0(v_00_u03b1_55_, v___y_56_, v___y_57_);
lean_dec(v___y_57_);
lean_dec_ref(v___y_56_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___redArg(lean_object* v___y_60_){
_start:
{
lean_object* v___x_62_; lean_object* v_env_63_; lean_object* v___x_64_; lean_object* v_mainModule_65_; lean_object* v___x_66_; 
v___x_62_ = lean_st_ref_get(v___y_60_);
v_env_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc_ref(v_env_63_);
lean_dec(v___x_62_);
v___x_64_ = l_Lean_Environment_header(v_env_63_);
lean_dec_ref(v_env_63_);
v_mainModule_65_ = lean_ctor_get(v___x_64_, 0);
lean_inc(v_mainModule_65_);
lean_dec_ref(v___x_64_);
v___x_66_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_66_, 0, v_mainModule_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___redArg___boxed(lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___redArg(v___y_67_);
lean_dec(v___y_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2(lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___redArg(v___y_71_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___boxed(lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2(v___y_74_, v___y_75_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_77_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0(uint8_t v___y_79_, uint8_t v_suppressElabErrors_80_, lean_object* v_x_81_){
_start:
{
if (lean_obj_tag(v_x_81_) == 1)
{
lean_object* v_pre_82_; 
v_pre_82_ = lean_ctor_get(v_x_81_, 0);
if (lean_obj_tag(v_pre_82_) == 0)
{
lean_object* v_str_83_; lean_object* v___x_84_; uint8_t v___x_85_; 
v_str_83_ = lean_ctor_get(v_x_81_, 1);
v___x_84_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0___closed__0));
v___x_85_ = lean_string_dec_eq(v_str_83_, v___x_84_);
if (v___x_85_ == 0)
{
return v___y_79_;
}
else
{
return v_suppressElabErrors_80_;
}
}
else
{
return v___y_79_;
}
}
else
{
return v___y_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0___boxed(lean_object* v___y_86_, lean_object* v_suppressElabErrors_87_, lean_object* v_x_88_){
_start:
{
uint8_t v___y_3692__boxed_89_; uint8_t v_suppressElabErrors_boxed_90_; uint8_t v_res_91_; lean_object* v_r_92_; 
v___y_3692__boxed_89_ = lean_unbox(v___y_86_);
v_suppressElabErrors_boxed_90_ = lean_unbox(v_suppressElabErrors_87_);
v_res_91_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0(v___y_3692__boxed_89_, v_suppressElabErrors_boxed_90_, v_x_88_);
lean_dec(v_x_88_);
v_r_92_ = lean_box(v_res_91_);
return v_r_92_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_93_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__0);
v___x_95_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
return v___x_95_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_96_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1);
v___x_97_ = lean_unsigned_to_nat(0u);
v___x_98_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
lean_ctor_set(v___x_98_, 2, v___x_97_);
lean_ctor_set(v___x_98_, 3, v___x_97_);
lean_ctor_set(v___x_98_, 4, v___x_96_);
lean_ctor_set(v___x_98_, 5, v___x_96_);
lean_ctor_set(v___x_98_, 6, v___x_96_);
lean_ctor_set(v___x_98_, 7, v___x_96_);
lean_ctor_set(v___x_98_, 8, v___x_96_);
lean_ctor_set(v___x_98_, 9, v___x_96_);
return v___x_98_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_99_ = lean_unsigned_to_nat(32u);
v___x_100_ = lean_mk_empty_array_with_capacity(v___x_99_);
v___x_101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__4(void){
_start:
{
size_t v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_102_ = ((size_t)5ULL);
v___x_103_ = lean_unsigned_to_nat(0u);
v___x_104_ = lean_unsigned_to_nat(32u);
v___x_105_ = lean_mk_empty_array_with_capacity(v___x_104_);
v___x_106_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__3);
v___x_107_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v___x_105_);
lean_ctor_set(v___x_107_, 2, v___x_103_);
lean_ctor_set(v___x_107_, 3, v___x_103_);
lean_ctor_set_usize(v___x_107_, 4, v___x_102_);
return v___x_107_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_108_ = lean_box(1);
v___x_109_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__4);
v___x_110_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__1);
v___x_111_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
lean_ctor_set(v___x_111_, 1, v___x_109_);
lean_ctor_set(v___x_111_, 2, v___x_108_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg(lean_object* v_msgData_112_, lean_object* v___y_113_){
_start:
{
lean_object* v___x_115_; lean_object* v_env_116_; lean_object* v___x_117_; lean_object* v_scopes_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v_opts_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_115_ = lean_st_ref_get(v___y_113_);
v_env_116_ = lean_ctor_get(v___x_115_, 0);
lean_inc_ref(v_env_116_);
lean_dec(v___x_115_);
v___x_117_ = lean_st_ref_get(v___y_113_);
v_scopes_118_ = lean_ctor_get(v___x_117_, 2);
lean_inc(v_scopes_118_);
lean_dec(v___x_117_);
v___x_119_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_120_ = l_List_head_x21___redArg(v___x_119_, v_scopes_118_);
lean_dec(v_scopes_118_);
v_opts_121_ = lean_ctor_get(v___x_120_, 1);
lean_inc_ref(v_opts_121_);
lean_dec(v___x_120_);
v___x_122_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__2);
v___x_123_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___closed__5);
v___x_124_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_124_, 0, v_env_116_);
lean_ctor_set(v___x_124_, 1, v___x_122_);
lean_ctor_set(v___x_124_, 2, v___x_123_);
lean_ctor_set(v___x_124_, 3, v_opts_121_);
v___x_125_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_msgData_112_);
v___x_126_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_126_, 0, v___x_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_msgData_127_, lean_object* v___y_128_, lean_object* v___y_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg(v_msgData_127_, v___y_128_);
lean_dec(v___y_128_);
return v_res_130_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__4(lean_object* v_opts_131_, lean_object* v_opt_132_){
_start:
{
lean_object* v_name_133_; lean_object* v_defValue_134_; lean_object* v_map_135_; lean_object* v___x_136_; 
v_name_133_ = lean_ctor_get(v_opt_132_, 0);
v_defValue_134_ = lean_ctor_get(v_opt_132_, 1);
v_map_135_ = lean_ctor_get(v_opts_131_, 0);
v___x_136_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_135_, v_name_133_);
if (lean_obj_tag(v___x_136_) == 0)
{
uint8_t v___x_137_; 
v___x_137_ = lean_unbox(v_defValue_134_);
return v___x_137_;
}
else
{
lean_object* v_val_138_; 
v_val_138_ = lean_ctor_get(v___x_136_, 0);
lean_inc(v_val_138_);
lean_dec_ref_known(v___x_136_, 1);
if (lean_obj_tag(v_val_138_) == 1)
{
uint8_t v_v_139_; 
v_v_139_ = lean_ctor_get_uint8(v_val_138_, 0);
lean_dec_ref_known(v_val_138_, 0);
return v_v_139_;
}
else
{
uint8_t v___x_140_; 
lean_dec(v_val_138_);
v___x_140_ = lean_unbox(v_defValue_134_);
return v___x_140_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__4___boxed(lean_object* v_opts_141_, lean_object* v_opt_142_){
_start:
{
uint8_t v_res_143_; lean_object* v_r_144_; 
v_res_143_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__4(v_opts_141_, v_opt_142_);
lean_dec_ref(v_opt_142_);
lean_dec_ref(v_opts_141_);
v_r_144_ = lean_box(v_res_143_);
return v_r_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1(lean_object* v_ref_146_, lean_object* v_msgData_147_, uint8_t v_severity_148_, uint8_t v_isSilent_149_, lean_object* v___y_150_, lean_object* v___y_151_){
_start:
{
lean_object* v___y_154_; lean_object* v___y_155_; lean_object* v___y_156_; uint8_t v___y_157_; uint8_t v___y_158_; lean_object* v___y_159_; lean_object* v___y_160_; lean_object* v___y_161_; uint8_t v___y_218_; lean_object* v___y_219_; uint8_t v___y_220_; uint8_t v___y_221_; lean_object* v___y_222_; uint8_t v___y_246_; lean_object* v___y_247_; uint8_t v___y_248_; uint8_t v___y_249_; lean_object* v___y_250_; uint8_t v___y_254_; uint8_t v___y_255_; uint8_t v___y_256_; uint8_t v___x_271_; uint8_t v___y_273_; uint8_t v___y_274_; uint8_t v___y_275_; uint8_t v___y_277_; uint8_t v___x_289_; 
v___x_271_ = 2;
v___x_289_ = l_Lean_instBEqMessageSeverity_beq(v_severity_148_, v___x_271_);
if (v___x_289_ == 0)
{
v___y_277_ = v___x_289_;
goto v___jp_276_;
}
else
{
uint8_t v___x_290_; 
lean_inc_ref(v_msgData_147_);
v___x_290_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_147_);
v___y_277_ = v___x_290_;
goto v___jp_276_;
}
v___jp_153_:
{
lean_object* v___x_162_; 
v___x_162_ = l_Lean_Elab_Command_getScope___redArg(v___y_161_);
if (lean_obj_tag(v___x_162_) == 0)
{
lean_object* v_a_163_; lean_object* v___x_164_; 
v_a_163_ = lean_ctor_get(v___x_162_, 0);
lean_inc(v_a_163_);
lean_dec_ref_known(v___x_162_, 1);
v___x_164_ = l_Lean_Elab_Command_getScope___redArg(v___y_161_);
if (lean_obj_tag(v___x_164_) == 0)
{
lean_object* v_a_165_; lean_object* v___x_167_; uint8_t v_isShared_168_; uint8_t v_isSharedCheck_200_; 
v_a_165_ = lean_ctor_get(v___x_164_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_164_);
if (v_isSharedCheck_200_ == 0)
{
v___x_167_ = v___x_164_;
v_isShared_168_ = v_isSharedCheck_200_;
goto v_resetjp_166_;
}
else
{
lean_inc(v_a_165_);
lean_dec(v___x_164_);
v___x_167_ = lean_box(0);
v_isShared_168_ = v_isSharedCheck_200_;
goto v_resetjp_166_;
}
v_resetjp_166_:
{
lean_object* v___x_169_; lean_object* v_currNamespace_170_; lean_object* v_openDecls_171_; lean_object* v_env_172_; lean_object* v_messages_173_; lean_object* v_scopes_174_; lean_object* v_usedQuotCtxts_175_; lean_object* v_nextMacroScope_176_; lean_object* v_maxRecDepth_177_; lean_object* v_ngen_178_; lean_object* v_auxDeclNGen_179_; lean_object* v_infoState_180_; lean_object* v_traceState_181_; lean_object* v_snapshotTasks_182_; lean_object* v_prevLinterStates_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_199_; 
v___x_169_ = lean_st_ref_take(v___y_161_);
v_currNamespace_170_ = lean_ctor_get(v_a_163_, 2);
lean_inc(v_currNamespace_170_);
lean_dec(v_a_163_);
v_openDecls_171_ = lean_ctor_get(v_a_165_, 3);
lean_inc(v_openDecls_171_);
lean_dec(v_a_165_);
v_env_172_ = lean_ctor_get(v___x_169_, 0);
v_messages_173_ = lean_ctor_get(v___x_169_, 1);
v_scopes_174_ = lean_ctor_get(v___x_169_, 2);
v_usedQuotCtxts_175_ = lean_ctor_get(v___x_169_, 3);
v_nextMacroScope_176_ = lean_ctor_get(v___x_169_, 4);
v_maxRecDepth_177_ = lean_ctor_get(v___x_169_, 5);
v_ngen_178_ = lean_ctor_get(v___x_169_, 6);
v_auxDeclNGen_179_ = lean_ctor_get(v___x_169_, 7);
v_infoState_180_ = lean_ctor_get(v___x_169_, 8);
v_traceState_181_ = lean_ctor_get(v___x_169_, 9);
v_snapshotTasks_182_ = lean_ctor_get(v___x_169_, 10);
v_prevLinterStates_183_ = lean_ctor_get(v___x_169_, 11);
v_isSharedCheck_199_ = !lean_is_exclusive(v___x_169_);
if (v_isSharedCheck_199_ == 0)
{
v___x_185_ = v___x_169_;
v_isShared_186_ = v_isSharedCheck_199_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_prevLinterStates_183_);
lean_inc(v_snapshotTasks_182_);
lean_inc(v_traceState_181_);
lean_inc(v_infoState_180_);
lean_inc(v_auxDeclNGen_179_);
lean_inc(v_ngen_178_);
lean_inc(v_maxRecDepth_177_);
lean_inc(v_nextMacroScope_176_);
lean_inc(v_usedQuotCtxts_175_);
lean_inc(v_scopes_174_);
lean_inc(v_messages_173_);
lean_inc(v_env_172_);
lean_dec(v___x_169_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_199_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_192_; 
v___x_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_187_, 0, v_currNamespace_170_);
lean_ctor_set(v___x_187_, 1, v_openDecls_171_);
v___x_188_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
lean_ctor_set(v___x_188_, 1, v___y_154_);
lean_inc_ref(v___y_155_);
lean_inc_ref(v___y_160_);
v___x_189_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_189_, 0, v___y_160_);
lean_ctor_set(v___x_189_, 1, v___y_156_);
lean_ctor_set(v___x_189_, 2, v___y_159_);
lean_ctor_set(v___x_189_, 3, v___y_155_);
lean_ctor_set(v___x_189_, 4, v___x_188_);
lean_ctor_set_uint8(v___x_189_, sizeof(void*)*5, v___y_157_);
lean_ctor_set_uint8(v___x_189_, sizeof(void*)*5 + 1, v___y_158_);
lean_ctor_set_uint8(v___x_189_, sizeof(void*)*5 + 2, v_isSilent_149_);
v___x_190_ = l_Lean_MessageLog_add(v___x_189_, v_messages_173_);
if (v_isShared_186_ == 0)
{
lean_ctor_set(v___x_185_, 1, v___x_190_);
v___x_192_ = v___x_185_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v_env_172_);
lean_ctor_set(v_reuseFailAlloc_198_, 1, v___x_190_);
lean_ctor_set(v_reuseFailAlloc_198_, 2, v_scopes_174_);
lean_ctor_set(v_reuseFailAlloc_198_, 3, v_usedQuotCtxts_175_);
lean_ctor_set(v_reuseFailAlloc_198_, 4, v_nextMacroScope_176_);
lean_ctor_set(v_reuseFailAlloc_198_, 5, v_maxRecDepth_177_);
lean_ctor_set(v_reuseFailAlloc_198_, 6, v_ngen_178_);
lean_ctor_set(v_reuseFailAlloc_198_, 7, v_auxDeclNGen_179_);
lean_ctor_set(v_reuseFailAlloc_198_, 8, v_infoState_180_);
lean_ctor_set(v_reuseFailAlloc_198_, 9, v_traceState_181_);
lean_ctor_set(v_reuseFailAlloc_198_, 10, v_snapshotTasks_182_);
lean_ctor_set(v_reuseFailAlloc_198_, 11, v_prevLinterStates_183_);
v___x_192_ = v_reuseFailAlloc_198_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_196_; 
v___x_193_ = lean_st_ref_set(v___y_161_, v___x_192_);
v___x_194_ = lean_box(0);
if (v_isShared_168_ == 0)
{
lean_ctor_set(v___x_167_, 0, v___x_194_);
v___x_196_ = v___x_167_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v___x_194_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
}
}
else
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
lean_dec(v_a_163_);
lean_dec(v___y_159_);
lean_dec_ref(v___y_156_);
lean_dec_ref(v___y_154_);
v_a_201_ = lean_ctor_get(v___x_164_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_164_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_164_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_164_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
else
{
lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
lean_dec(v___y_159_);
lean_dec_ref(v___y_156_);
lean_dec_ref(v___y_154_);
v_a_209_ = lean_ctor_get(v___x_162_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_162_);
if (v_isSharedCheck_216_ == 0)
{
v___x_211_ = v___x_162_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_162_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_a_209_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
v___jp_217_:
{
lean_object* v_fileName_223_; lean_object* v_fileMap_224_; uint8_t v_suppressElabErrors_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_244_; 
v_fileName_223_ = lean_ctor_get(v___y_150_, 0);
v_fileMap_224_ = lean_ctor_get(v___y_150_, 1);
v_suppressElabErrors_225_ = lean_ctor_get_uint8(v___y_150_, sizeof(void*)*10);
v___x_226_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_147_);
v___x_227_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg(v___x_226_, v___y_151_);
v_a_228_ = lean_ctor_get(v___x_227_, 0);
v_isSharedCheck_244_ = !lean_is_exclusive(v___x_227_);
if (v_isSharedCheck_244_ == 0)
{
v___x_230_ = v___x_227_;
v_isShared_231_ = v_isSharedCheck_244_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_227_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_244_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
lean_inc_ref_n(v_fileMap_224_, 2);
v___x_232_ = l_Lean_FileMap_toPosition(v_fileMap_224_, v___y_219_);
lean_dec(v___y_219_);
v___x_233_ = l_Lean_FileMap_toPosition(v_fileMap_224_, v___y_222_);
lean_dec(v___y_222_);
v___x_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
v___x_235_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___closed__0));
if (v_suppressElabErrors_225_ == 0)
{
lean_del_object(v___x_230_);
v___y_154_ = v_a_228_;
v___y_155_ = v___x_235_;
v___y_156_ = v___x_232_;
v___y_157_ = v___y_220_;
v___y_158_ = v___y_221_;
v___y_159_ = v___x_234_;
v___y_160_ = v_fileName_223_;
v___y_161_ = v___y_151_;
goto v___jp_153_;
}
else
{
lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___f_238_; uint8_t v___x_239_; 
v___x_236_ = lean_box(v___y_218_);
v___x_237_ = lean_box(v_suppressElabErrors_225_);
v___f_238_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___lam__0___boxed), 3, 2);
lean_closure_set(v___f_238_, 0, v___x_236_);
lean_closure_set(v___f_238_, 1, v___x_237_);
lean_inc(v_a_228_);
v___x_239_ = l_Lean_MessageData_hasTag(v___f_238_, v_a_228_);
if (v___x_239_ == 0)
{
lean_object* v___x_240_; lean_object* v___x_242_; 
lean_dec_ref_known(v___x_234_, 1);
lean_dec_ref(v___x_232_);
lean_dec(v_a_228_);
v___x_240_ = lean_box(0);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 0, v___x_240_);
v___x_242_ = v___x_230_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v___x_240_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
else
{
lean_del_object(v___x_230_);
v___y_154_ = v_a_228_;
v___y_155_ = v___x_235_;
v___y_156_ = v___x_232_;
v___y_157_ = v___y_220_;
v___y_158_ = v___y_221_;
v___y_159_ = v___x_234_;
v___y_160_ = v_fileName_223_;
v___y_161_ = v___y_151_;
goto v___jp_153_;
}
}
}
}
v___jp_245_:
{
lean_object* v___x_251_; 
v___x_251_ = l_Lean_Syntax_getTailPos_x3f(v___y_247_, v___y_248_);
lean_dec(v___y_247_);
if (lean_obj_tag(v___x_251_) == 0)
{
lean_inc(v___y_250_);
v___y_218_ = v___y_246_;
v___y_219_ = v___y_250_;
v___y_220_ = v___y_248_;
v___y_221_ = v___y_249_;
v___y_222_ = v___y_250_;
goto v___jp_217_;
}
else
{
lean_object* v_val_252_; 
v_val_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc(v_val_252_);
lean_dec_ref_known(v___x_251_, 1);
v___y_218_ = v___y_246_;
v___y_219_ = v___y_250_;
v___y_220_ = v___y_248_;
v___y_221_ = v___y_249_;
v___y_222_ = v_val_252_;
goto v___jp_217_;
}
}
v___jp_253_:
{
lean_object* v___x_257_; 
v___x_257_ = l_Lean_Elab_Command_getRef___redArg(v___y_150_);
if (lean_obj_tag(v___x_257_) == 0)
{
lean_object* v_a_258_; lean_object* v_ref_259_; lean_object* v___x_260_; 
v_a_258_ = lean_ctor_get(v___x_257_, 0);
lean_inc(v_a_258_);
lean_dec_ref_known(v___x_257_, 1);
v_ref_259_ = l_Lean_replaceRef(v_ref_146_, v_a_258_);
lean_dec(v_a_258_);
v___x_260_ = l_Lean_Syntax_getPos_x3f(v_ref_259_, v___y_255_);
if (lean_obj_tag(v___x_260_) == 0)
{
lean_object* v___x_261_; 
v___x_261_ = lean_unsigned_to_nat(0u);
v___y_246_ = v___y_254_;
v___y_247_ = v_ref_259_;
v___y_248_ = v___y_255_;
v___y_249_ = v___y_256_;
v___y_250_ = v___x_261_;
goto v___jp_245_;
}
else
{
lean_object* v_val_262_; 
v_val_262_ = lean_ctor_get(v___x_260_, 0);
lean_inc(v_val_262_);
lean_dec_ref_known(v___x_260_, 1);
v___y_246_ = v___y_254_;
v___y_247_ = v_ref_259_;
v___y_248_ = v___y_255_;
v___y_249_ = v___y_256_;
v___y_250_ = v_val_262_;
goto v___jp_245_;
}
}
else
{
lean_object* v_a_263_; lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_270_; 
lean_dec_ref(v_msgData_147_);
v_a_263_ = lean_ctor_get(v___x_257_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_270_ == 0)
{
v___x_265_ = v___x_257_;
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
else
{
lean_inc(v_a_263_);
lean_dec(v___x_257_);
v___x_265_ = lean_box(0);
v_isShared_266_ = v_isSharedCheck_270_;
goto v_resetjp_264_;
}
v_resetjp_264_:
{
lean_object* v___x_268_; 
if (v_isShared_266_ == 0)
{
v___x_268_ = v___x_265_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v_a_263_);
v___x_268_ = v_reuseFailAlloc_269_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
return v___x_268_;
}
}
}
}
v___jp_272_:
{
if (v___y_275_ == 0)
{
v___y_254_ = v___y_273_;
v___y_255_ = v___y_274_;
v___y_256_ = v_severity_148_;
goto v___jp_253_;
}
else
{
v___y_254_ = v___y_273_;
v___y_255_ = v___y_274_;
v___y_256_ = v___x_271_;
goto v___jp_253_;
}
}
v___jp_276_:
{
if (v___y_277_ == 0)
{
lean_object* v___x_278_; lean_object* v_scopes_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v_opts_282_; uint8_t v___x_283_; uint8_t v___x_284_; 
v___x_278_ = lean_st_ref_get(v___y_151_);
v_scopes_279_ = lean_ctor_get(v___x_278_, 2);
lean_inc(v_scopes_279_);
lean_dec(v___x_278_);
v___x_280_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_281_ = l_List_head_x21___redArg(v___x_280_, v_scopes_279_);
lean_dec(v_scopes_279_);
v_opts_282_ = lean_ctor_get(v___x_281_, 1);
lean_inc_ref(v_opts_282_);
lean_dec(v___x_281_);
v___x_283_ = 1;
v___x_284_ = l_Lean_instBEqMessageSeverity_beq(v_severity_148_, v___x_283_);
if (v___x_284_ == 0)
{
lean_dec_ref(v_opts_282_);
v___y_273_ = v___y_277_;
v___y_274_ = v___y_277_;
v___y_275_ = v___x_284_;
goto v___jp_272_;
}
else
{
lean_object* v___x_285_; uint8_t v___x_286_; 
v___x_285_ = l_Lean_warningAsError;
v___x_286_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__4(v_opts_282_, v___x_285_);
lean_dec_ref(v_opts_282_);
v___y_273_ = v___y_277_;
v___y_274_ = v___y_277_;
v___y_275_ = v___x_286_;
goto v___jp_272_;
}
}
else
{
lean_object* v___x_287_; lean_object* v___x_288_; 
lean_dec_ref(v_msgData_147_);
v___x_287_ = lean_box(0);
v___x_288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
return v___x_288_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1___boxed(lean_object* v_ref_291_, lean_object* v_msgData_292_, lean_object* v_severity_293_, lean_object* v_isSilent_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_){
_start:
{
uint8_t v_severity_boxed_298_; uint8_t v_isSilent_boxed_299_; lean_object* v_res_300_; 
v_severity_boxed_298_ = lean_unbox(v_severity_293_);
v_isSilent_boxed_299_ = lean_unbox(v_isSilent_294_);
v_res_300_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1(v_ref_291_, v_msgData_292_, v_severity_boxed_298_, v_isSilent_boxed_299_, v___y_295_, v___y_296_);
lean_dec(v___y_296_);
lean_dec_ref(v___y_295_);
lean_dec(v_ref_291_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1(lean_object* v_ref_301_, lean_object* v_msgData_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
uint8_t v___x_306_; uint8_t v___x_307_; lean_object* v___x_308_; 
v___x_306_ = 1;
v___x_307_ = 0;
v___x_308_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1(v_ref_301_, v_msgData_302_, v___x_306_, v___x_307_, v___y_303_, v___y_304_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1___boxed(lean_object* v_ref_309_, lean_object* v_msgData_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1(v_ref_309_, v_msgData_310_, v___y_311_, v___y_312_);
lean_dec(v___y_312_);
lean_dec_ref(v___y_311_);
lean_dec(v_ref_309_);
return v_res_314_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_elabVariables___closed__2(void){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabVariables___closed__1));
v___x_319_ = l_Lean_MessageData_ofFormat(v___x_318_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_elabVariables___closed__10(void){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = l_Array_mkArray0(lean_box(0));
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabVariables(lean_object* v_x_333_, lean_object* v_a_334_, lean_object* v_a_335_){
_start:
{
lean_object* v___x_337_; uint8_t v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_variables___closed__3));
lean_inc(v_x_333_);
v___x_338_ = l_Lean_Syntax_isOfKind(v_x_333_, v___x_337_);
if (v___x_338_ == 0)
{
lean_object* v___x_339_; 
lean_dec(v_x_333_);
v___x_339_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg();
return v___x_339_;
}
else
{
lean_object* v___x_340_; lean_object* v_pos_341_; lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_340_ = lean_unsigned_to_nat(0u);
v_pos_341_ = l_Lean_Syntax_getArg(v_x_333_, v___x_340_);
v___x_342_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_elabVariables___closed__2, &lp_mathlib_Mathlib_Tactic_elabVariables___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_elabVariables___closed__2);
v___x_343_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1(v_pos_341_, v___x_342_, v_a_334_, v_a_335_);
if (lean_obj_tag(v___x_343_) == 0)
{
lean_object* v___x_344_; 
lean_dec_ref_known(v___x_343_, 1);
v___x_344_ = l_Lean_Elab_Command_getRef___redArg(v_a_334_);
if (lean_obj_tag(v___x_344_) == 0)
{
lean_object* v_a_345_; lean_object* v___x_346_; 
v_a_345_ = lean_ctor_get(v___x_344_, 0);
lean_inc(v_a_345_);
lean_dec_ref_known(v___x_344_, 1);
v___x_346_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_334_);
if (lean_obj_tag(v___x_346_) == 0)
{
lean_object* v_quotContext_x3f_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v_binders_350_; uint8_t v___x_351_; lean_object* v___x_352_; 
lean_dec_ref_known(v___x_346_, 1);
v_quotContext_x3f_347_ = lean_ctor_get(v_a_334_, 5);
v___x_348_ = lean_unsigned_to_nat(1u);
v___x_349_ = l_Lean_Syntax_getArg(v_x_333_, v___x_348_);
lean_dec(v_x_333_);
v_binders_350_ = l_Lean_Syntax_getArgs(v___x_349_);
lean_dec(v___x_349_);
v___x_351_ = 0;
v___x_352_ = l_Lean_SourceInfo_fromRef(v_a_345_, v___x_351_);
lean_dec(v_a_345_);
if (lean_obj_tag(v_quotContext_x3f_347_) == 0)
{
lean_object* v___x_364_; 
v___x_364_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_elabVariables_spec__2___redArg(v_a_335_);
lean_dec_ref(v___x_364_);
goto v___jp_353_;
}
else
{
goto v___jp_353_;
}
v___jp_353_:
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabVariables___closed__6));
v___x_355_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabVariables___closed__7));
v___x_356_ = l_Lean_SourceInfo_fromRef(v_pos_341_, v___x_338_);
lean_dec(v_pos_341_);
v___x_357_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
lean_ctor_set(v___x_357_, 1, v___x_354_);
v___x_358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabVariables___closed__9));
v___x_359_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_elabVariables___closed__10, &lp_mathlib_Mathlib_Tactic_elabVariables___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_elabVariables___closed__10);
v___x_360_ = l_Array_append___redArg(v___x_359_, v_binders_350_);
lean_dec_ref(v_binders_350_);
lean_inc(v___x_352_);
v___x_361_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_361_, 0, v___x_352_);
lean_ctor_set(v___x_361_, 1, v___x_358_);
lean_ctor_set(v___x_361_, 2, v___x_360_);
v___x_362_ = l_Lean_Syntax_node2(v___x_352_, v___x_355_, v___x_357_, v___x_361_);
v___x_363_ = l_Lean_Elab_Command_elabVariable(v___x_362_, v_a_334_, v_a_335_);
return v___x_363_;
}
}
else
{
lean_object* v_a_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_372_; 
lean_dec(v_a_345_);
lean_dec(v_pos_341_);
lean_dec(v_x_333_);
v_a_365_ = lean_ctor_get(v___x_346_, 0);
v_isSharedCheck_372_ = !lean_is_exclusive(v___x_346_);
if (v_isSharedCheck_372_ == 0)
{
v___x_367_ = v___x_346_;
v_isShared_368_ = v_isSharedCheck_372_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_a_365_);
lean_dec(v___x_346_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_372_;
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
lean_object* v_reuseFailAlloc_371_; 
v_reuseFailAlloc_371_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_371_, 0, v_a_365_);
v___x_370_ = v_reuseFailAlloc_371_;
goto v_reusejp_369_;
}
v_reusejp_369_:
{
return v___x_370_;
}
}
}
}
else
{
lean_object* v_a_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_380_; 
lean_dec(v_pos_341_);
lean_dec(v_x_333_);
v_a_373_ = lean_ctor_get(v___x_344_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v___x_344_);
if (v_isSharedCheck_380_ == 0)
{
v___x_375_ = v___x_344_;
v_isShared_376_ = v_isSharedCheck_380_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_a_373_);
lean_dec(v___x_344_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_380_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
lean_object* v___x_378_; 
if (v_isShared_376_ == 0)
{
v___x_378_ = v___x_375_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v_a_373_);
v___x_378_ = v_reuseFailAlloc_379_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
return v___x_378_;
}
}
}
}
else
{
lean_dec(v_pos_341_);
lean_dec(v_x_333_);
return v___x_343_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabVariables___boxed(lean_object* v_x_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Mathlib_Tactic_elabVariables(v_x_381_, v_a_382_, v_a_383_);
lean_dec(v_a_383_);
lean_dec_ref(v_a_382_);
return v_res_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3(lean_object* v_msgData_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
lean_object* v___x_390_; 
v___x_390_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___redArg(v_msgData_386_, v___y_388_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_elabVariables_spec__1_spec__1_spec__3(v_msgData_391_, v___y_392_, v___y_393_);
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
return v_res_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__0(lean_object* v___x_396_, lean_object* v_toPure_397_, lean_object* v_r_398_){
_start:
{
lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_399_, 0, v___x_396_);
v___x_400_ = lean_apply_2(v_toPure_397_, lean_box(0), v___x_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__1(lean_object* v_toPure_401_, lean_object* v_newLCtx_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_toBind_405_, lean_object* v_a_406_, lean_object* v_x_407_, lean_object* v___y_408_){
_start:
{
lean_object* v_array_409_; lean_object* v_start_410_; lean_object* v_stop_411_; uint8_t v___x_412_; 
v_array_409_ = lean_ctor_get(v___y_408_, 0);
v_start_410_ = lean_ctor_get(v___y_408_, 1);
v_stop_411_ = lean_ctor_get(v___y_408_, 2);
v___x_412_ = lean_nat_dec_lt(v_start_410_, v_stop_411_);
if (v___x_412_ == 0)
{
lean_object* v___x_413_; lean_object* v___x_414_; 
lean_dec(v_a_406_);
lean_dec(v_toBind_405_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_newLCtx_402_);
v___x_413_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_413_, 0, v___y_408_);
v___x_414_ = lean_apply_2(v_toPure_401_, lean_box(0), v___x_413_);
return v___x_414_;
}
else
{
lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_434_; 
lean_inc(v_stop_411_);
lean_inc(v_start_410_);
lean_inc_ref(v_array_409_);
v_isSharedCheck_434_ = !lean_is_exclusive(v___y_408_);
if (v_isSharedCheck_434_ == 0)
{
lean_object* v_unused_435_; lean_object* v_unused_436_; lean_object* v_unused_437_; 
v_unused_435_ = lean_ctor_get(v___y_408_, 2);
lean_dec(v_unused_435_);
v_unused_436_ = lean_ctor_get(v___y_408_, 1);
lean_dec(v_unused_436_);
v_unused_437_ = lean_ctor_get(v___y_408_, 0);
lean_dec(v_unused_437_);
v___x_416_ = v___y_408_;
v_isShared_417_ = v_isSharedCheck_434_;
goto v_resetjp_415_;
}
else
{
lean_dec(v___y_408_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_434_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_422_; 
v___x_418_ = lean_array_fget(v_array_409_, v_start_410_);
v___x_419_ = lean_unsigned_to_nat(1u);
v___x_420_ = lean_nat_add(v_start_410_, v___x_419_);
lean_dec(v_start_410_);
if (v_isShared_417_ == 0)
{
lean_ctor_set(v___x_416_, 1, v___x_420_);
v___x_422_ = v___x_416_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v_array_409_);
lean_ctor_set(v_reuseFailAlloc_433_, 1, v___x_420_);
lean_ctor_set(v_reuseFailAlloc_433_, 2, v_stop_411_);
v___x_422_ = v_reuseFailAlloc_433_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
uint8_t v___x_423_; 
v___x_423_ = l_Lean_instBEqFVarId_beq(v_a_406_, v___x_418_);
if (v___x_423_ == 0)
{
lean_object* v___f_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; 
v___f_424_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__0), 3, 2);
lean_closure_set(v___f_424_, 0, v___x_422_);
lean_closure_set(v___f_424_, 1, v_toPure_401_);
lean_inc(v___x_418_);
v___x_425_ = l_Lean_LocalContext_get_x21(v_newLCtx_402_, v___x_418_);
v___x_426_ = l_Lean_LocalDecl_userName(v___x_425_);
lean_dec_ref(v___x_425_);
v___x_427_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_427_, 0, v___x_426_);
lean_ctor_set(v___x_427_, 1, v___x_418_);
lean_ctor_set(v___x_427_, 2, v_a_406_);
v___x_428_ = lean_alloc_ctor(11, 1, 0);
lean_ctor_set(v___x_428_, 0, v___x_427_);
v___x_429_ = l_Lean_Elab_pushInfoLeaf___redArg(v_inst_403_, v_inst_404_, v___x_428_);
v___x_430_ = lean_apply_4(v_toBind_405_, lean_box(0), lean_box(0), v___x_429_, v___f_424_);
return v___x_430_;
}
else
{
lean_object* v___x_431_; lean_object* v___x_432_; 
lean_dec(v___x_418_);
lean_dec(v_a_406_);
lean_dec(v_toBind_405_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_newLCtx_402_);
v___x_431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_431_, 0, v___x_422_);
v___x_432_ = lean_apply_2(v_toPure_401_, lean_box(0), v___x_431_);
return v___x_432_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__2(lean_object* v_toPure_438_, lean_object* v_____s_439_){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_440_ = lean_box(0);
v___x_441_ = lean_apply_2(v_toPure_438_, lean_box(0), v___x_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__2___boxed(lean_object* v_toPure_442_, lean_object* v_____s_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__2(v_toPure_442_, v_____s_443_);
lean_dec_ref(v_____s_443_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg(lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_oldFVars_447_, lean_object* v_newFVars_448_, lean_object* v_newLCtx_449_){
_start:
{
lean_object* v_toApplicative_450_; lean_object* v_toBind_451_; lean_object* v_toPure_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___f_456_; lean_object* v___f_457_; size_t v_sz_458_; size_t v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
v_toApplicative_450_ = lean_ctor_get(v_inst_445_, 0);
v_toBind_451_ = lean_ctor_get(v_inst_445_, 1);
lean_inc_n(v_toBind_451_, 2);
v_toPure_452_ = lean_ctor_get(v_toApplicative_450_, 1);
v___x_453_ = lean_array_get_size(v_newFVars_448_);
v___x_454_ = lean_unsigned_to_nat(0u);
v___x_455_ = l_Array_toSubarray___redArg(v_newFVars_448_, v___x_454_, v___x_453_);
lean_inc_ref(v_inst_445_);
lean_inc_n(v_toPure_452_, 2);
v___f_456_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__1), 8, 5);
lean_closure_set(v___f_456_, 0, v_toPure_452_);
lean_closure_set(v___f_456_, 1, v_newLCtx_449_);
lean_closure_set(v___f_456_, 2, v_inst_445_);
lean_closure_set(v___f_456_, 3, v_inst_446_);
lean_closure_set(v___f_456_, 4, v_toBind_451_);
v___f_457_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_457_, 0, v_toPure_452_);
v_sz_458_ = lean_array_size(v_oldFVars_447_);
v___x_459_ = ((size_t)0ULL);
v___x_460_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_445_, v_oldFVars_447_, v___f_456_, v_sz_458_, v___x_459_, v___x_455_);
v___x_461_ = lean_apply_4(v_toBind_451_, lean_box(0), lean_box(0), v___x_460_, v___f_457_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo(lean_object* v_m_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_oldFVars_465_, lean_object* v_newFVars_466_, lean_object* v_newLCtx_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lp_mathlib_Mathlib_Tactic_pushFVarAliasInfo___redArg(v_inst_463_, v_inst_464_, v_oldFVars_465_, v_newFVars_466_, v_newLCtx_467_);
return v___x_468_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_introv___closed__7(void){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; 
v___x_486_ = l_Lean_binderIdent;
v___x_487_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_introv___closed__6));
v___x_488_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_variables___closed__5));
v___x_489_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_489_, 0, v___x_488_);
lean_ctor_set(v___x_489_, 1, v___x_487_);
lean_ctor_set(v___x_489_, 2, v___x_486_);
return v___x_489_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_introv___closed__8(void){
_start:
{
lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
v___x_490_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_introv___closed__7, &lp_mathlib_Mathlib_Tactic_introv___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_introv___closed__7);
v___x_491_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_variables___closed__8));
v___x_492_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_491_);
lean_ctor_set(v___x_492_, 1, v___x_490_);
return v___x_492_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_introv___closed__9(void){
_start:
{
lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_493_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_introv___closed__8, &lp_mathlib_Mathlib_Tactic_introv___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_introv___closed__8);
v___x_494_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_introv___closed__2));
v___x_495_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_variables___closed__5));
v___x_496_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_496_, 0, v___x_495_);
lean_ctor_set(v___x_496_, 1, v___x_494_);
lean_ctor_set(v___x_496_, 2, v___x_493_);
return v___x_496_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_introv___closed__10(void){
_start:
{
lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_497_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_introv___closed__9, &lp_mathlib_Mathlib_Tactic_introv___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_introv___closed__9);
v___x_498_ = lean_unsigned_to_nat(1022u);
v___x_499_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_introv___closed__1));
v___x_500_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_500_, 0, v___x_499_);
lean_ctor_set(v___x_500_, 1, v___x_498_);
lean_ctor_set(v___x_500_, 2, v___x_497_);
return v___x_500_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_introv(void){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_introv___closed__10, &lp_mathlib_Mathlib_Tactic_introv___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_introv___closed__10);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___lam__0(lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_503_, v___y_506_, v___y_507_, v___y_508_, v___y_509_);
if (lean_obj_tag(v___x_511_) == 0)
{
lean_object* v_a_512_; uint8_t v___x_513_; lean_object* v___x_514_; 
v_a_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc(v_a_512_);
lean_dec_ref_known(v___x_511_, 1);
v___x_513_ = 1;
v___x_514_ = l_Lean_Meta_intro1Core(v_a_512_, v___x_513_, v___y_506_, v___y_507_, v___y_508_, v___y_509_);
if (lean_obj_tag(v___x_514_) == 0)
{
lean_object* v_a_515_; lean_object* v_snd_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_534_; 
v_a_515_ = lean_ctor_get(v___x_514_, 0);
lean_inc(v_a_515_);
lean_dec_ref_known(v___x_514_, 1);
v_snd_516_ = lean_ctor_get(v_a_515_, 1);
v_isSharedCheck_534_ = !lean_is_exclusive(v_a_515_);
if (v_isSharedCheck_534_ == 0)
{
lean_object* v_unused_535_; 
v_unused_535_ = lean_ctor_get(v_a_515_, 0);
lean_dec(v_unused_535_);
v___x_518_ = v_a_515_;
v_isShared_519_ = v_isSharedCheck_534_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_snd_516_);
lean_dec(v_a_515_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_534_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_520_; lean_object* v___x_522_; 
v___x_520_ = lean_box(0);
if (v_isShared_519_ == 0)
{
lean_ctor_set_tag(v___x_518_, 1);
lean_ctor_set(v___x_518_, 1, v___x_520_);
lean_ctor_set(v___x_518_, 0, v_snd_516_);
v___x_522_ = v___x_518_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_533_, 0, v_snd_516_);
lean_ctor_set(v_reuseFailAlloc_533_, 1, v___x_520_);
v___x_522_ = v_reuseFailAlloc_533_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
lean_object* v___x_523_; 
v___x_523_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_522_, v___y_503_, v___y_506_, v___y_507_, v___y_508_, v___y_509_);
if (lean_obj_tag(v___x_523_) == 0)
{
lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_531_; 
v_isSharedCheck_531_ = !lean_is_exclusive(v___x_523_);
if (v_isSharedCheck_531_ == 0)
{
lean_object* v_unused_532_; 
v_unused_532_ = lean_ctor_get(v___x_523_, 0);
lean_dec(v_unused_532_);
v___x_525_ = v___x_523_;
v_isShared_526_ = v_isSharedCheck_531_;
goto v_resetjp_524_;
}
else
{
lean_dec(v___x_523_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_531_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_527_; lean_object* v___x_529_; 
v___x_527_ = lean_box(0);
if (v_isShared_526_ == 0)
{
lean_ctor_set(v___x_525_, 0, v___x_527_);
v___x_529_ = v___x_525_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v___x_527_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
else
{
return v___x_523_;
}
}
}
}
else
{
lean_object* v_a_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_543_; 
v_a_536_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_543_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_543_ == 0)
{
v___x_538_ = v___x_514_;
v_isShared_539_ = v_isSharedCheck_543_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_a_536_);
lean_dec(v___x_514_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_543_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_541_; 
if (v_isShared_539_ == 0)
{
v___x_541_ = v___x_538_;
goto v_reusejp_540_;
}
else
{
lean_object* v_reuseFailAlloc_542_; 
v_reuseFailAlloc_542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_542_, 0, v_a_536_);
v___x_541_ = v_reuseFailAlloc_542_;
goto v_reusejp_540_;
}
v_reusejp_540_:
{
return v___x_541_;
}
}
}
}
else
{
lean_object* v_a_544_; lean_object* v___x_546_; uint8_t v_isShared_547_; uint8_t v_isSharedCheck_551_; 
v_a_544_ = lean_ctor_get(v___x_511_, 0);
v_isSharedCheck_551_ = !lean_is_exclusive(v___x_511_);
if (v_isSharedCheck_551_ == 0)
{
v___x_546_ = v___x_511_;
v_isShared_547_ = v_isSharedCheck_551_;
goto v_resetjp_545_;
}
else
{
lean_inc(v_a_544_);
lean_dec(v___x_511_);
v___x_546_ = lean_box(0);
v_isShared_547_ = v_isSharedCheck_551_;
goto v_resetjp_545_;
}
v_resetjp_545_:
{
lean_object* v___x_549_; 
if (v_isShared_547_ == 0)
{
v___x_549_ = v___x_546_;
goto v_reusejp_548_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v_a_544_);
v___x_549_ = v_reuseFailAlloc_550_;
goto v_reusejp_548_;
}
v_reusejp_548_:
{
return v___x_549_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___lam__0___boxed(lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___lam__0(v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_, v___y_558_, v___y_559_);
lean_dec(v___y_559_);
lean_dec_ref(v___y_558_);
lean_dec(v___y_557_);
lean_dec_ref(v___y_556_);
lean_dec(v___y_555_);
lean_dec_ref(v___y_554_);
lean_dec(v___y_553_);
lean_dec_ref(v___y_552_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep(lean_object* v_a_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_, lean_object* v_a_568_, lean_object* v_a_569_, lean_object* v_a_570_){
_start:
{
lean_object* v___f_572_; lean_object* v___x_573_; 
v___f_572_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___closed__0));
v___x_573_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_572_, v_a_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_, v_a_569_, v_a_570_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep___boxed(lean_object* v_a_574_, lean_object* v_a_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep(v_a_574_, v_a_575_, v_a_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_);
lean_dec(v_a_581_);
lean_dec_ref(v_a_580_);
lean_dec(v_a_579_);
lean_dec_ref(v_a_578_);
lean_dec(v_a_577_);
lean_dec_ref(v_a_576_);
lean_dec(v_a_575_);
lean_dec_ref(v_a_574_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_introsDep(lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_){
_start:
{
lean_object* v___x_593_; 
v___x_593_ = l_Lean_Elab_Tactic_getMainTarget(v_a_584_, v_a_585_, v_a_586_, v_a_587_, v_a_588_, v_a_589_, v_a_590_, v_a_591_);
if (lean_obj_tag(v___x_593_) == 0)
{
lean_object* v_a_594_; lean_object* v___x_596_; uint8_t v_isShared_597_; uint8_t v_isSharedCheck_610_; 
v_a_594_ = lean_ctor_get(v___x_593_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_610_ == 0)
{
v___x_596_ = v___x_593_;
v_isShared_597_ = v_isSharedCheck_610_;
goto v_resetjp_595_;
}
else
{
lean_inc(v_a_594_);
lean_dec(v___x_593_);
v___x_596_ = lean_box(0);
v_isShared_597_ = v_isSharedCheck_610_;
goto v_resetjp_595_;
}
v_resetjp_595_:
{
if (lean_obj_tag(v_a_594_) == 7)
{
lean_object* v_body_598_; uint8_t v___x_599_; 
v_body_598_ = lean_ctor_get(v_a_594_, 2);
lean_inc_ref(v_body_598_);
lean_dec_ref_known(v_a_594_, 3);
v___x_599_ = l_Lean_Expr_hasLooseBVars(v_body_598_);
lean_dec_ref(v_body_598_);
if (v___x_599_ == 0)
{
lean_object* v___x_600_; lean_object* v___x_602_; 
v___x_600_ = lean_box(0);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 0, v___x_600_);
v___x_602_ = v___x_596_;
goto v_reusejp_601_;
}
else
{
lean_object* v_reuseFailAlloc_603_; 
v_reuseFailAlloc_603_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_603_, 0, v___x_600_);
v___x_602_ = v_reuseFailAlloc_603_;
goto v_reusejp_601_;
}
v_reusejp_601_:
{
return v___x_602_;
}
}
else
{
lean_object* v___x_604_; 
lean_del_object(v___x_596_);
v___x_604_ = lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_intro1PStep(v_a_584_, v_a_585_, v_a_586_, v_a_587_, v_a_588_, v_a_589_, v_a_590_, v_a_591_);
if (lean_obj_tag(v___x_604_) == 0)
{
lean_dec_ref_known(v___x_604_, 1);
goto _start;
}
else
{
return v___x_604_;
}
}
}
else
{
lean_object* v___x_606_; lean_object* v___x_608_; 
lean_dec(v_a_594_);
v___x_606_ = lean_box(0);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 0, v___x_606_);
v___x_608_ = v___x_596_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v___x_606_);
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
else
{
lean_object* v_a_611_; lean_object* v___x_613_; uint8_t v_isShared_614_; uint8_t v_isSharedCheck_618_; 
v_a_611_ = lean_ctor_get(v___x_593_, 0);
v_isSharedCheck_618_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_618_ == 0)
{
v___x_613_ = v___x_593_;
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
else
{
lean_inc(v_a_611_);
lean_dec(v___x_593_);
v___x_613_ = lean_box(0);
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
v_resetjp_612_:
{
lean_object* v___x_616_; 
if (v_isShared_614_ == 0)
{
v___x_616_ = v___x_613_;
goto v_reusejp_615_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v_a_611_);
v___x_616_ = v_reuseFailAlloc_617_;
goto v_reusejp_615_;
}
v_reusejp_615_:
{
return v___x_616_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_introsDep___boxed(lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_){
_start:
{
lean_object* v_res_628_; 
v_res_628_ = lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_introsDep(v_a_619_, v_a_620_, v_a_621_, v_a_622_, v_a_623_, v_a_624_, v_a_625_, v_a_626_);
lean_dec(v_a_626_);
lean_dec_ref(v_a_625_);
lean_dec(v_a_624_);
lean_dec_ref(v_a_623_);
lean_dec(v_a_622_);
lean_dec_ref(v_a_621_);
lean_dec(v_a_620_);
lean_dec_ref(v_a_619_);
return v_res_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg(){
_start:
{
lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_630_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_elabVariables_spec__0___redArg___closed__0);
v___x_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_631_, 0, v___x_630_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg___boxed(lean_object* v___y_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0(lean_object* v_00_u03b1_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___boxed(lean_object* v_00_u03b1_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_){
_start:
{
lean_object* v_res_655_; 
v_res_655_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0(v_00_u03b1_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_, v___y_652_, v___y_653_);
lean_dec(v___y_653_);
lean_dec_ref(v___y_652_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
return v_res_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov(lean_object* v_stx_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_){
_start:
{
lean_object* v___x_694_; lean_object* v___x_695_; uint8_t v___x_696_; 
v___x_694_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_introv___closed__0));
v___x_695_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_introv___closed__1));
lean_inc(v_stx_684_);
v___x_696_ = l_Lean_Syntax_isOfKind(v_stx_684_, v___x_695_);
if (v___x_696_ == 0)
{
lean_object* v___x_697_; 
lean_dec(v_stx_684_);
v___x_697_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
return v___x_697_;
}
else
{
lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; uint8_t v___x_701_; 
v___x_698_ = lean_unsigned_to_nat(0u);
v___x_699_ = lean_unsigned_to_nat(1u);
v___x_700_ = l_Lean_Syntax_getArg(v_stx_684_, v___x_699_);
lean_dec(v_stx_684_);
lean_inc(v___x_700_);
v___x_701_ = l_Lean_Syntax_matchesNull(v___x_700_, v___x_698_);
if (v___x_701_ == 0)
{
lean_object* v___x_702_; uint8_t v___x_703_; 
v___x_702_ = l_Lean_Syntax_getNumArgs(v___x_700_);
v___x_703_ = lean_nat_dec_le(v___x_699_, v___x_702_);
if (v___x_703_ == 0)
{
lean_object* v___x_704_; 
lean_dec(v___x_702_);
lean_dec(v___x_700_);
v___x_704_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
return v___x_704_;
}
else
{
lean_object* v___x_705_; lean_object* v___x_706_; uint8_t v___x_707_; 
v___x_705_ = l_Lean_Syntax_getArg(v___x_700_, v___x_698_);
v___x_706_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__1));
lean_inc(v___x_705_);
v___x_707_ = l_Lean_Syntax_isOfKind(v___x_705_, v___x_706_);
if (v___x_707_ == 0)
{
lean_object* v___x_708_; 
lean_dec(v___x_705_);
lean_dec(v___x_702_);
lean_dec(v___x_700_);
v___x_708_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
return v___x_708_;
}
else
{
lean_object* v___x_709_; lean_object* v___x_710_; uint8_t v___x_711_; 
v___x_709_ = l_Lean_Syntax_getArg(v___x_705_, v___x_698_);
lean_dec(v___x_705_);
v___x_710_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__3));
lean_inc(v___x_709_);
v___x_711_ = l_Lean_Syntax_isOfKind(v___x_709_, v___x_710_);
if (v___x_711_ == 0)
{
lean_object* v___x_712_; uint8_t v___x_713_; 
v___x_712_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__6));
lean_inc(v___x_709_);
v___x_713_ = l_Lean_Syntax_isOfKind(v___x_709_, v___x_712_);
if (v___x_713_ == 0)
{
lean_object* v___x_714_; 
lean_dec(v___x_709_);
lean_dec(v___x_702_);
lean_dec(v___x_700_);
v___x_714_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
return v___x_714_;
}
else
{
lean_object* v_ref_715_; lean_object* v_tk_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v_hs_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; 
v_ref_715_ = lean_ctor_get(v_a_691_, 5);
v_tk_716_ = l_Lean_Syntax_getArg(v___x_709_, v___x_698_);
lean_dec(v___x_709_);
v___x_717_ = l_Lean_Syntax_getArgs(v___x_700_);
lean_dec(v___x_700_);
v___x_718_ = l_Array_extract___redArg(v___x_717_, v___x_699_, v___x_702_);
lean_dec_ref(v___x_717_);
v___x_719_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabVariables___closed__9));
v___x_720_ = lean_box(2);
v___x_721_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_721_, 0, v___x_720_);
lean_ctor_set(v___x_721_, 1, v___x_719_);
lean_ctor_set(v___x_721_, 2, v___x_718_);
v_hs_722_ = l_Lean_Syntax_getArgs(v___x_721_);
lean_dec_ref_known(v___x_721_, 3);
v___x_723_ = l_Lean_SourceInfo_fromRef(v_ref_715_, v___x_711_);
v___x_724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8));
lean_inc_n(v___x_723_, 11);
v___x_725_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_725_, 0, v___x_723_);
lean_ctor_set(v___x_725_, 1, v___x_694_);
v___x_726_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_elabVariables___closed__10, &lp_mathlib_Mathlib_Tactic_elabVariables___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_elabVariables___closed__10);
v___x_727_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_727_, 0, v___x_723_);
lean_ctor_set(v___x_727_, 1, v___x_719_);
lean_ctor_set(v___x_727_, 2, v___x_726_);
lean_inc_ref(v___x_725_);
v___x_728_ = l_Lean_Syntax_node2(v___x_723_, v___x_695_, v___x_725_, v___x_727_);
v___x_729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__9));
v___x_730_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_723_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__10));
v___x_732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11));
v___x_733_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_733_, 0, v___x_723_);
lean_ctor_set(v___x_733_, 1, v___x_731_);
v___x_734_ = l_Lean_SourceInfo_fromRef(v_tk_716_, v___x_713_);
lean_dec(v_tk_716_);
v___x_735_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__12));
v___x_736_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_736_, 0, v___x_734_);
lean_ctor_set(v___x_736_, 1, v___x_735_);
v___x_737_ = l_Lean_Syntax_node1(v___x_723_, v___x_712_, v___x_736_);
v___x_738_ = l_Lean_Syntax_node1(v___x_723_, v___x_719_, v___x_737_);
v___x_739_ = l_Lean_Syntax_node2(v___x_723_, v___x_732_, v___x_733_, v___x_738_);
v___x_740_ = l_Array_append___redArg(v___x_726_, v_hs_722_);
lean_dec_ref(v_hs_722_);
v___x_741_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_741_, 0, v___x_723_);
lean_ctor_set(v___x_741_, 1, v___x_719_);
lean_ctor_set(v___x_741_, 2, v___x_740_);
v___x_742_ = l_Lean_Syntax_node2(v___x_723_, v___x_695_, v___x_725_, v___x_741_);
lean_inc_ref(v___x_730_);
v___x_743_ = l_Lean_Syntax_node5(v___x_723_, v___x_719_, v___x_728_, v___x_730_, v___x_739_, v___x_730_, v___x_742_);
v___x_744_ = l_Lean_Syntax_node1(v___x_723_, v___x_724_, v___x_743_);
v___x_745_ = l_Lean_Elab_Tactic_evalTactic(v___x_744_, v_a_685_, v_a_686_, v_a_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_);
return v___x_745_;
}
}
else
{
lean_object* v_ref_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v_hs_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; 
v_ref_746_ = lean_ctor_get(v_a_691_, 5);
v___x_747_ = l_Lean_Syntax_getArgs(v___x_700_);
lean_dec(v___x_700_);
v___x_748_ = l_Array_extract___redArg(v___x_747_, v___x_699_, v___x_702_);
lean_dec_ref(v___x_747_);
v___x_749_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabVariables___closed__9));
v___x_750_ = lean_box(2);
v___x_751_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_751_, 0, v___x_750_);
lean_ctor_set(v___x_751_, 1, v___x_749_);
lean_ctor_set(v___x_751_, 2, v___x_748_);
v_hs_752_ = l_Lean_Syntax_getArgs(v___x_751_);
lean_dec_ref_known(v___x_751_, 3);
v___x_753_ = l_Lean_SourceInfo_fromRef(v_ref_746_, v___x_701_);
v___x_754_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__8));
lean_inc_n(v___x_753_, 10);
v___x_755_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_755_, 0, v___x_753_);
lean_ctor_set(v___x_755_, 1, v___x_694_);
v___x_756_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_elabVariables___closed__10, &lp_mathlib_Mathlib_Tactic_elabVariables___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_elabVariables___closed__10);
v___x_757_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_757_, 0, v___x_753_);
lean_ctor_set(v___x_757_, 1, v___x_749_);
lean_ctor_set(v___x_757_, 2, v___x_756_);
lean_inc_ref(v___x_755_);
v___x_758_ = l_Lean_Syntax_node2(v___x_753_, v___x_695_, v___x_755_, v___x_757_);
v___x_759_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__9));
v___x_760_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_760_, 0, v___x_753_);
lean_ctor_set(v___x_760_, 1, v___x_759_);
v___x_761_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__10));
v___x_762_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_evalIntrov___closed__11));
v___x_763_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_763_, 0, v___x_753_);
lean_ctor_set(v___x_763_, 1, v___x_761_);
v___x_764_ = l_Lean_Syntax_node1(v___x_753_, v___x_749_, v___x_709_);
v___x_765_ = l_Lean_Syntax_node2(v___x_753_, v___x_762_, v___x_763_, v___x_764_);
v___x_766_ = l_Array_append___redArg(v___x_756_, v_hs_752_);
lean_dec_ref(v_hs_752_);
v___x_767_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_767_, 0, v___x_753_);
lean_ctor_set(v___x_767_, 1, v___x_749_);
lean_ctor_set(v___x_767_, 2, v___x_766_);
v___x_768_ = l_Lean_Syntax_node2(v___x_753_, v___x_695_, v___x_755_, v___x_767_);
lean_inc_ref(v___x_760_);
v___x_769_ = l_Lean_Syntax_node5(v___x_753_, v___x_749_, v___x_758_, v___x_760_, v___x_765_, v___x_760_, v___x_768_);
v___x_770_ = l_Lean_Syntax_node1(v___x_753_, v___x_754_, v___x_769_);
v___x_771_ = l_Lean_Elab_Tactic_evalTactic(v___x_770_, v_a_685_, v_a_686_, v_a_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_);
return v___x_771_;
}
}
}
}
else
{
lean_object* v___x_772_; 
lean_dec(v___x_700_);
v___x_772_ = lp_mathlib___private_Mathlib_Tactic_Basic_0__Mathlib_Tactic_evalIntrov_introsDep(v_a_685_, v_a_686_, v_a_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_);
return v___x_772_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_evalIntrov___boxed(lean_object* v_stx_773_, lean_object* v_a_774_, lean_object* v_a_775_, lean_object* v_a_776_, lean_object* v_a_777_, lean_object* v_a_778_, lean_object* v_a_779_, lean_object* v_a_780_, lean_object* v_a_781_, lean_object* v_a_782_){
_start:
{
lean_object* v_res_783_; 
v_res_783_ = lp_mathlib_Mathlib_Tactic_evalIntrov(v_stx_773_, v_a_774_, v_a_775_, v_a_776_, v_a_777_, v_a_778_, v_a_779_, v_a_780_, v_a_781_);
lean_dec(v_a_781_);
lean_dec_ref(v_a_780_);
lean_dec(v_a_779_);
lean_dec_ref(v_a_778_);
lean_dec(v_a_777_);
lean_dec_ref(v_a_776_);
lean_dec(v_a_775_);
lean_dec_ref(v_a_774_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1(lean_object* v_x_823_, lean_object* v_a_824_, lean_object* v_a_825_){
_start:
{
lean_object* v___x_826_; uint8_t v___x_827_; 
v___x_826_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticAssumption_x27___closed__1));
v___x_827_ = l_Lean_Syntax_isOfKind(v_x_823_, v___x_826_);
if (v___x_827_ == 0)
{
lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_828_ = lean_box(1);
v___x_829_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_829_, 0, v___x_828_);
lean_ctor_set(v___x_829_, 1, v_a_825_);
return v___x_829_;
}
else
{
lean_object* v_ref_830_; uint8_t v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; 
v_ref_830_ = lean_ctor_get(v_a_824_, 5);
v___x_831_ = 0;
v___x_832_ = l_Lean_SourceInfo_fromRef(v_ref_830_, v___x_831_);
v___x_833_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__1));
v___x_834_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__2));
lean_inc_n(v___x_832_, 6);
v___x_835_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_835_, 0, v___x_832_);
lean_ctor_set(v___x_835_, 1, v___x_834_);
v___x_836_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__4));
v___x_837_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__6));
v___x_838_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabVariables___closed__9));
v___x_839_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__7));
v___x_840_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___closed__8));
v___x_841_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_841_, 0, v___x_832_);
lean_ctor_set(v___x_841_, 1, v___x_839_);
v___x_842_ = l_Lean_Syntax_node1(v___x_832_, v___x_840_, v___x_841_);
v___x_843_ = l_Lean_Syntax_node1(v___x_832_, v___x_838_, v___x_842_);
v___x_844_ = l_Lean_Syntax_node1(v___x_832_, v___x_837_, v___x_843_);
v___x_845_ = l_Lean_Syntax_node1(v___x_832_, v___x_836_, v___x_844_);
v___x_846_ = l_Lean_Syntax_node2(v___x_832_, v___x_833_, v___x_835_, v___x_845_);
v___x_847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_847_, 0, v___x_846_);
lean_ctor_set(v___x_847_, 1, v_a_825_);
return v___x_847_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1___boxed(lean_object* v_x_848_, lean_object* v_a_849_, lean_object* v_a_850_){
_start:
{
lean_object* v_res_851_; 
v_res_851_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______macroRules__Mathlib__Tactic__tacticAssumption_x27__1(v_x_848_, v_a_849_, v_a_850_);
lean_dec_ref(v_a_849_);
return v_res_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_as_866_, size_t v_sz_867_, size_t v_i_868_, lean_object* v_b_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_){
_start:
{
uint8_t v___x_875_; 
v___x_875_ = lean_usize_dec_lt(v_i_868_, v_sz_867_);
if (v___x_875_ == 0)
{
lean_object* v___x_876_; 
v___x_876_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_876_, 0, v_b_869_);
return v___x_876_;
}
else
{
lean_object* v_snd_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_904_; 
v_snd_877_ = lean_ctor_get(v_b_869_, 1);
v_isSharedCheck_904_ = !lean_is_exclusive(v_b_869_);
if (v_isSharedCheck_904_ == 0)
{
lean_object* v_unused_905_; 
v_unused_905_ = lean_ctor_get(v_b_869_, 0);
lean_dec(v_unused_905_);
v___x_879_ = v_b_869_;
v_isShared_880_ = v_isSharedCheck_904_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_snd_877_);
lean_dec(v_b_869_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_904_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
lean_object* v___x_881_; lean_object* v_a_883_; lean_object* v_a_890_; 
v___x_881_ = lean_box(0);
v_a_890_ = lean_array_uget_borrowed(v_as_866_, v_i_868_);
if (lean_obj_tag(v_a_890_) == 0)
{
v_a_883_ = v_snd_877_;
goto v___jp_882_;
}
else
{
lean_object* v_val_891_; uint8_t v___x_892_; 
v_val_891_ = lean_ctor_get(v_a_890_, 0);
v___x_892_ = l_Lean_LocalDecl_isAuxDecl(v_val_891_);
if (v___x_892_ == 0)
{
v_a_883_ = v_snd_877_;
goto v___jp_882_;
}
else
{
lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_893_ = l_Lean_LocalDecl_fvarId(v_val_891_);
v___x_894_ = l_Lean_MVarId_tryClear(v_snd_877_, v___x_893_, v___y_870_, v___y_871_, v___y_872_, v___y_873_);
if (lean_obj_tag(v___x_894_) == 0)
{
lean_object* v_a_895_; 
v_a_895_ = lean_ctor_get(v___x_894_, 0);
lean_inc(v_a_895_);
lean_dec_ref_known(v___x_894_, 1);
v_a_883_ = v_a_895_;
goto v___jp_882_;
}
else
{
lean_object* v_a_896_; lean_object* v___x_898_; uint8_t v_isShared_899_; uint8_t v_isSharedCheck_903_; 
lean_del_object(v___x_879_);
v_a_896_ = lean_ctor_get(v___x_894_, 0);
v_isSharedCheck_903_ = !lean_is_exclusive(v___x_894_);
if (v_isSharedCheck_903_ == 0)
{
v___x_898_ = v___x_894_;
v_isShared_899_ = v_isSharedCheck_903_;
goto v_resetjp_897_;
}
else
{
lean_inc(v_a_896_);
lean_dec(v___x_894_);
v___x_898_ = lean_box(0);
v_isShared_899_ = v_isSharedCheck_903_;
goto v_resetjp_897_;
}
v_resetjp_897_:
{
lean_object* v___x_901_; 
if (v_isShared_899_ == 0)
{
v___x_901_ = v___x_898_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_902_; 
v_reuseFailAlloc_902_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_902_, 0, v_a_896_);
v___x_901_ = v_reuseFailAlloc_902_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
return v___x_901_;
}
}
}
}
}
v___jp_882_:
{
lean_object* v___x_885_; 
if (v_isShared_880_ == 0)
{
lean_ctor_set(v___x_879_, 1, v_a_883_);
lean_ctor_set(v___x_879_, 0, v___x_881_);
v___x_885_ = v___x_879_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_889_; 
v_reuseFailAlloc_889_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_889_, 0, v___x_881_);
lean_ctor_set(v_reuseFailAlloc_889_, 1, v_a_883_);
v___x_885_ = v_reuseFailAlloc_889_;
goto v_reusejp_884_;
}
v_reusejp_884_:
{
size_t v___x_886_; size_t v___x_887_; 
v___x_886_ = ((size_t)1ULL);
v___x_887_ = lean_usize_add(v_i_868_, v___x_886_);
v_i_868_ = v___x_887_;
v_b_869_ = v___x_885_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___redArg___boxed(lean_object* v_as_906_, lean_object* v_sz_907_, lean_object* v_i_908_, lean_object* v_b_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
size_t v_sz_boxed_915_; size_t v_i_boxed_916_; lean_object* v_res_917_; 
v_sz_boxed_915_ = lean_unbox_usize(v_sz_907_);
lean_dec(v_sz_907_);
v_i_boxed_916_ = lean_unbox_usize(v_i_908_);
lean_dec(v_i_908_);
v_res_917_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___redArg(v_as_906_, v_sz_boxed_915_, v_i_boxed_916_, v_b_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
lean_dec_ref(v_as_906_);
return v_res_917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2(lean_object* v_as_918_, size_t v_sz_919_, size_t v_i_920_, lean_object* v_b_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_){
_start:
{
uint8_t v___x_931_; 
v___x_931_ = lean_usize_dec_lt(v_i_920_, v_sz_919_);
if (v___x_931_ == 0)
{
lean_object* v___x_932_; 
v___x_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_932_, 0, v_b_921_);
return v___x_932_;
}
else
{
lean_object* v_snd_933_; lean_object* v___x_935_; uint8_t v_isShared_936_; uint8_t v_isSharedCheck_960_; 
v_snd_933_ = lean_ctor_get(v_b_921_, 1);
v_isSharedCheck_960_ = !lean_is_exclusive(v_b_921_);
if (v_isSharedCheck_960_ == 0)
{
lean_object* v_unused_961_; 
v_unused_961_ = lean_ctor_get(v_b_921_, 0);
lean_dec(v_unused_961_);
v___x_935_ = v_b_921_;
v_isShared_936_ = v_isSharedCheck_960_;
goto v_resetjp_934_;
}
else
{
lean_inc(v_snd_933_);
lean_dec(v_b_921_);
v___x_935_ = lean_box(0);
v_isShared_936_ = v_isSharedCheck_960_;
goto v_resetjp_934_;
}
v_resetjp_934_:
{
lean_object* v___x_937_; lean_object* v_a_939_; lean_object* v_a_946_; 
v___x_937_ = lean_box(0);
v_a_946_ = lean_array_uget_borrowed(v_as_918_, v_i_920_);
if (lean_obj_tag(v_a_946_) == 0)
{
v_a_939_ = v_snd_933_;
goto v___jp_938_;
}
else
{
lean_object* v_val_947_; uint8_t v___x_948_; 
v_val_947_ = lean_ctor_get(v_a_946_, 0);
v___x_948_ = l_Lean_LocalDecl_isAuxDecl(v_val_947_);
if (v___x_948_ == 0)
{
v_a_939_ = v_snd_933_;
goto v___jp_938_;
}
else
{
lean_object* v___x_949_; lean_object* v___x_950_; 
v___x_949_ = l_Lean_LocalDecl_fvarId(v_val_947_);
v___x_950_ = l_Lean_MVarId_tryClear(v_snd_933_, v___x_949_, v___y_926_, v___y_927_, v___y_928_, v___y_929_);
if (lean_obj_tag(v___x_950_) == 0)
{
lean_object* v_a_951_; 
v_a_951_ = lean_ctor_get(v___x_950_, 0);
lean_inc(v_a_951_);
lean_dec_ref_known(v___x_950_, 1);
v_a_939_ = v_a_951_;
goto v___jp_938_;
}
else
{
lean_object* v_a_952_; lean_object* v___x_954_; uint8_t v_isShared_955_; uint8_t v_isSharedCheck_959_; 
lean_del_object(v___x_935_);
v_a_952_ = lean_ctor_get(v___x_950_, 0);
v_isSharedCheck_959_ = !lean_is_exclusive(v___x_950_);
if (v_isSharedCheck_959_ == 0)
{
v___x_954_ = v___x_950_;
v_isShared_955_ = v_isSharedCheck_959_;
goto v_resetjp_953_;
}
else
{
lean_inc(v_a_952_);
lean_dec(v___x_950_);
v___x_954_ = lean_box(0);
v_isShared_955_ = v_isSharedCheck_959_;
goto v_resetjp_953_;
}
v_resetjp_953_:
{
lean_object* v___x_957_; 
if (v_isShared_955_ == 0)
{
v___x_957_ = v___x_954_;
goto v_reusejp_956_;
}
else
{
lean_object* v_reuseFailAlloc_958_; 
v_reuseFailAlloc_958_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_958_, 0, v_a_952_);
v___x_957_ = v_reuseFailAlloc_958_;
goto v_reusejp_956_;
}
v_reusejp_956_:
{
return v___x_957_;
}
}
}
}
}
v___jp_938_:
{
lean_object* v___x_941_; 
if (v_isShared_936_ == 0)
{
lean_ctor_set(v___x_935_, 1, v_a_939_);
lean_ctor_set(v___x_935_, 0, v___x_937_);
v___x_941_ = v___x_935_;
goto v_reusejp_940_;
}
else
{
lean_object* v_reuseFailAlloc_945_; 
v_reuseFailAlloc_945_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_945_, 0, v___x_937_);
lean_ctor_set(v_reuseFailAlloc_945_, 1, v_a_939_);
v___x_941_ = v_reuseFailAlloc_945_;
goto v_reusejp_940_;
}
v_reusejp_940_:
{
size_t v___x_942_; size_t v___x_943_; lean_object* v___x_944_; 
v___x_942_ = ((size_t)1ULL);
v___x_943_ = lean_usize_add(v_i_920_, v___x_942_);
v___x_944_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___redArg(v_as_918_, v_sz_919_, v___x_943_, v___x_941_, v___y_926_, v___y_927_, v___y_928_, v___y_929_);
return v___x_944_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2___boxed(lean_object* v_as_962_, lean_object* v_sz_963_, lean_object* v_i_964_, lean_object* v_b_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_){
_start:
{
size_t v_sz_boxed_975_; size_t v_i_boxed_976_; lean_object* v_res_977_; 
v_sz_boxed_975_ = lean_unbox_usize(v_sz_963_);
lean_dec(v_sz_963_);
v_i_boxed_976_ = lean_unbox_usize(v_i_964_);
lean_dec(v_i_964_);
v_res_977_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2(v_as_962_, v_sz_boxed_975_, v_i_boxed_976_, v_b_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
lean_dec(v___y_969_);
lean_dec_ref(v___y_968_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
lean_dec_ref(v_as_962_);
return v_res_977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0(lean_object* v_init_978_, lean_object* v_n_979_, lean_object* v_b_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_){
_start:
{
if (lean_obj_tag(v_n_979_) == 0)
{
lean_object* v_cs_990_; lean_object* v___x_991_; lean_object* v___x_992_; size_t v_sz_993_; size_t v___x_994_; lean_object* v___x_995_; 
v_cs_990_ = lean_ctor_get(v_n_979_, 0);
v___x_991_ = lean_box(0);
v___x_992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_992_, 0, v___x_991_);
lean_ctor_set(v___x_992_, 1, v_b_980_);
v_sz_993_ = lean_array_size(v_cs_990_);
v___x_994_ = ((size_t)0ULL);
v___x_995_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__1(v_init_978_, v_cs_990_, v_sz_993_, v___x_994_, v___x_992_, v___y_981_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_);
if (lean_obj_tag(v___x_995_) == 0)
{
lean_object* v_a_996_; lean_object* v___x_998_; uint8_t v_isShared_999_; uint8_t v_isSharedCheck_1010_; 
v_a_996_ = lean_ctor_get(v___x_995_, 0);
v_isSharedCheck_1010_ = !lean_is_exclusive(v___x_995_);
if (v_isSharedCheck_1010_ == 0)
{
v___x_998_ = v___x_995_;
v_isShared_999_ = v_isSharedCheck_1010_;
goto v_resetjp_997_;
}
else
{
lean_inc(v_a_996_);
lean_dec(v___x_995_);
v___x_998_ = lean_box(0);
v_isShared_999_ = v_isSharedCheck_1010_;
goto v_resetjp_997_;
}
v_resetjp_997_:
{
lean_object* v_fst_1000_; 
v_fst_1000_ = lean_ctor_get(v_a_996_, 0);
if (lean_obj_tag(v_fst_1000_) == 0)
{
lean_object* v_snd_1001_; lean_object* v___x_1002_; lean_object* v___x_1004_; 
v_snd_1001_ = lean_ctor_get(v_a_996_, 1);
lean_inc(v_snd_1001_);
lean_dec(v_a_996_);
v___x_1002_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1002_, 0, v_snd_1001_);
if (v_isShared_999_ == 0)
{
lean_ctor_set(v___x_998_, 0, v___x_1002_);
v___x_1004_ = v___x_998_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1005_; 
v_reuseFailAlloc_1005_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1005_, 0, v___x_1002_);
v___x_1004_ = v_reuseFailAlloc_1005_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
return v___x_1004_;
}
}
else
{
lean_object* v_val_1006_; lean_object* v___x_1008_; 
lean_inc_ref(v_fst_1000_);
lean_dec(v_a_996_);
v_val_1006_ = lean_ctor_get(v_fst_1000_, 0);
lean_inc(v_val_1006_);
lean_dec_ref_known(v_fst_1000_, 1);
if (v_isShared_999_ == 0)
{
lean_ctor_set(v___x_998_, 0, v_val_1006_);
v___x_1008_ = v___x_998_;
goto v_reusejp_1007_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v_val_1006_);
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
else
{
lean_object* v_a_1011_; lean_object* v___x_1013_; uint8_t v_isShared_1014_; uint8_t v_isSharedCheck_1018_; 
v_a_1011_ = lean_ctor_get(v___x_995_, 0);
v_isSharedCheck_1018_ = !lean_is_exclusive(v___x_995_);
if (v_isSharedCheck_1018_ == 0)
{
v___x_1013_ = v___x_995_;
v_isShared_1014_ = v_isSharedCheck_1018_;
goto v_resetjp_1012_;
}
else
{
lean_inc(v_a_1011_);
lean_dec(v___x_995_);
v___x_1013_ = lean_box(0);
v_isShared_1014_ = v_isSharedCheck_1018_;
goto v_resetjp_1012_;
}
v_resetjp_1012_:
{
lean_object* v___x_1016_; 
if (v_isShared_1014_ == 0)
{
v___x_1016_ = v___x_1013_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1017_; 
v_reuseFailAlloc_1017_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1017_, 0, v_a_1011_);
v___x_1016_ = v_reuseFailAlloc_1017_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
return v___x_1016_;
}
}
}
}
else
{
lean_object* v_vs_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; size_t v_sz_1022_; size_t v___x_1023_; lean_object* v___x_1024_; 
v_vs_1019_ = lean_ctor_get(v_n_979_, 0);
v___x_1020_ = lean_box(0);
v___x_1021_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1021_, 0, v___x_1020_);
lean_ctor_set(v___x_1021_, 1, v_b_980_);
v_sz_1022_ = lean_array_size(v_vs_1019_);
v___x_1023_ = ((size_t)0ULL);
v___x_1024_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2(v_vs_1019_, v_sz_1022_, v___x_1023_, v___x_1021_, v___y_981_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_);
if (lean_obj_tag(v___x_1024_) == 0)
{
lean_object* v_a_1025_; lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1039_; 
v_a_1025_ = lean_ctor_get(v___x_1024_, 0);
v_isSharedCheck_1039_ = !lean_is_exclusive(v___x_1024_);
if (v_isSharedCheck_1039_ == 0)
{
v___x_1027_ = v___x_1024_;
v_isShared_1028_ = v_isSharedCheck_1039_;
goto v_resetjp_1026_;
}
else
{
lean_inc(v_a_1025_);
lean_dec(v___x_1024_);
v___x_1027_ = lean_box(0);
v_isShared_1028_ = v_isSharedCheck_1039_;
goto v_resetjp_1026_;
}
v_resetjp_1026_:
{
lean_object* v_fst_1029_; 
v_fst_1029_ = lean_ctor_get(v_a_1025_, 0);
if (lean_obj_tag(v_fst_1029_) == 0)
{
lean_object* v_snd_1030_; lean_object* v___x_1031_; lean_object* v___x_1033_; 
v_snd_1030_ = lean_ctor_get(v_a_1025_, 1);
lean_inc(v_snd_1030_);
lean_dec(v_a_1025_);
v___x_1031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1031_, 0, v_snd_1030_);
if (v_isShared_1028_ == 0)
{
lean_ctor_set(v___x_1027_, 0, v___x_1031_);
v___x_1033_ = v___x_1027_;
goto v_reusejp_1032_;
}
else
{
lean_object* v_reuseFailAlloc_1034_; 
v_reuseFailAlloc_1034_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1034_, 0, v___x_1031_);
v___x_1033_ = v_reuseFailAlloc_1034_;
goto v_reusejp_1032_;
}
v_reusejp_1032_:
{
return v___x_1033_;
}
}
else
{
lean_object* v_val_1035_; lean_object* v___x_1037_; 
lean_inc_ref(v_fst_1029_);
lean_dec(v_a_1025_);
v_val_1035_ = lean_ctor_get(v_fst_1029_, 0);
lean_inc(v_val_1035_);
lean_dec_ref_known(v_fst_1029_, 1);
if (v_isShared_1028_ == 0)
{
lean_ctor_set(v___x_1027_, 0, v_val_1035_);
v___x_1037_ = v___x_1027_;
goto v_reusejp_1036_;
}
else
{
lean_object* v_reuseFailAlloc_1038_; 
v_reuseFailAlloc_1038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1038_, 0, v_val_1035_);
v___x_1037_ = v_reuseFailAlloc_1038_;
goto v_reusejp_1036_;
}
v_reusejp_1036_:
{
return v___x_1037_;
}
}
}
}
else
{
lean_object* v_a_1040_; lean_object* v___x_1042_; uint8_t v_isShared_1043_; uint8_t v_isSharedCheck_1047_; 
v_a_1040_ = lean_ctor_get(v___x_1024_, 0);
v_isSharedCheck_1047_ = !lean_is_exclusive(v___x_1024_);
if (v_isSharedCheck_1047_ == 0)
{
v___x_1042_ = v___x_1024_;
v_isShared_1043_ = v_isSharedCheck_1047_;
goto v_resetjp_1041_;
}
else
{
lean_inc(v_a_1040_);
lean_dec(v___x_1024_);
v___x_1042_ = lean_box(0);
v_isShared_1043_ = v_isSharedCheck_1047_;
goto v_resetjp_1041_;
}
v_resetjp_1041_:
{
lean_object* v___x_1045_; 
if (v_isShared_1043_ == 0)
{
v___x_1045_ = v___x_1042_;
goto v_reusejp_1044_;
}
else
{
lean_object* v_reuseFailAlloc_1046_; 
v_reuseFailAlloc_1046_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1046_, 0, v_a_1040_);
v___x_1045_ = v_reuseFailAlloc_1046_;
goto v_reusejp_1044_;
}
v_reusejp_1044_:
{
return v___x_1045_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__1(lean_object* v_init_1048_, lean_object* v_as_1049_, size_t v_sz_1050_, size_t v_i_1051_, lean_object* v_b_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_){
_start:
{
uint8_t v___x_1062_; 
v___x_1062_ = lean_usize_dec_lt(v_i_1051_, v_sz_1050_);
if (v___x_1062_ == 0)
{
lean_object* v___x_1063_; 
v___x_1063_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1063_, 0, v_b_1052_);
return v___x_1063_;
}
else
{
lean_object* v_snd_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1098_; 
v_snd_1064_ = lean_ctor_get(v_b_1052_, 1);
v_isSharedCheck_1098_ = !lean_is_exclusive(v_b_1052_);
if (v_isSharedCheck_1098_ == 0)
{
lean_object* v_unused_1099_; 
v_unused_1099_ = lean_ctor_get(v_b_1052_, 0);
lean_dec(v_unused_1099_);
v___x_1066_ = v_b_1052_;
v_isShared_1067_ = v_isSharedCheck_1098_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_snd_1064_);
lean_dec(v_b_1052_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1098_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v_a_1068_; lean_object* v___x_1069_; 
v_a_1068_ = lean_array_uget_borrowed(v_as_1049_, v_i_1051_);
lean_inc(v_snd_1064_);
v___x_1069_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0(v_init_1048_, v_a_1068_, v_snd_1064_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_);
if (lean_obj_tag(v___x_1069_) == 0)
{
lean_object* v_a_1070_; lean_object* v___x_1072_; uint8_t v_isShared_1073_; uint8_t v_isSharedCheck_1089_; 
v_a_1070_ = lean_ctor_get(v___x_1069_, 0);
v_isSharedCheck_1089_ = !lean_is_exclusive(v___x_1069_);
if (v_isSharedCheck_1089_ == 0)
{
v___x_1072_ = v___x_1069_;
v_isShared_1073_ = v_isSharedCheck_1089_;
goto v_resetjp_1071_;
}
else
{
lean_inc(v_a_1070_);
lean_dec(v___x_1069_);
v___x_1072_ = lean_box(0);
v_isShared_1073_ = v_isSharedCheck_1089_;
goto v_resetjp_1071_;
}
v_resetjp_1071_:
{
if (lean_obj_tag(v_a_1070_) == 0)
{
lean_object* v___x_1074_; lean_object* v___x_1076_; 
v___x_1074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1074_, 0, v_a_1070_);
if (v_isShared_1067_ == 0)
{
lean_ctor_set(v___x_1066_, 0, v___x_1074_);
v___x_1076_ = v___x_1066_;
goto v_reusejp_1075_;
}
else
{
lean_object* v_reuseFailAlloc_1080_; 
v_reuseFailAlloc_1080_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1080_, 0, v___x_1074_);
lean_ctor_set(v_reuseFailAlloc_1080_, 1, v_snd_1064_);
v___x_1076_ = v_reuseFailAlloc_1080_;
goto v_reusejp_1075_;
}
v_reusejp_1075_:
{
lean_object* v___x_1078_; 
if (v_isShared_1073_ == 0)
{
lean_ctor_set(v___x_1072_, 0, v___x_1076_);
v___x_1078_ = v___x_1072_;
goto v_reusejp_1077_;
}
else
{
lean_object* v_reuseFailAlloc_1079_; 
v_reuseFailAlloc_1079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1079_, 0, v___x_1076_);
v___x_1078_ = v_reuseFailAlloc_1079_;
goto v_reusejp_1077_;
}
v_reusejp_1077_:
{
return v___x_1078_;
}
}
}
else
{
lean_object* v_a_1081_; lean_object* v___x_1082_; lean_object* v___x_1084_; 
lean_del_object(v___x_1072_);
lean_dec(v_snd_1064_);
v_a_1081_ = lean_ctor_get(v_a_1070_, 0);
lean_inc(v_a_1081_);
lean_dec_ref_known(v_a_1070_, 1);
v___x_1082_ = lean_box(0);
if (v_isShared_1067_ == 0)
{
lean_ctor_set(v___x_1066_, 1, v_a_1081_);
lean_ctor_set(v___x_1066_, 0, v___x_1082_);
v___x_1084_ = v___x_1066_;
goto v_reusejp_1083_;
}
else
{
lean_object* v_reuseFailAlloc_1088_; 
v_reuseFailAlloc_1088_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1088_, 0, v___x_1082_);
lean_ctor_set(v_reuseFailAlloc_1088_, 1, v_a_1081_);
v___x_1084_ = v_reuseFailAlloc_1088_;
goto v_reusejp_1083_;
}
v_reusejp_1083_:
{
size_t v___x_1085_; size_t v___x_1086_; 
v___x_1085_ = ((size_t)1ULL);
v___x_1086_ = lean_usize_add(v_i_1051_, v___x_1085_);
v_i_1051_ = v___x_1086_;
v_b_1052_ = v___x_1084_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1090_; lean_object* v___x_1092_; uint8_t v_isShared_1093_; uint8_t v_isSharedCheck_1097_; 
lean_del_object(v___x_1066_);
lean_dec(v_snd_1064_);
v_a_1090_ = lean_ctor_get(v___x_1069_, 0);
v_isSharedCheck_1097_ = !lean_is_exclusive(v___x_1069_);
if (v_isSharedCheck_1097_ == 0)
{
v___x_1092_ = v___x_1069_;
v_isShared_1093_ = v_isSharedCheck_1097_;
goto v_resetjp_1091_;
}
else
{
lean_inc(v_a_1090_);
lean_dec(v___x_1069_);
v___x_1092_ = lean_box(0);
v_isShared_1093_ = v_isSharedCheck_1097_;
goto v_resetjp_1091_;
}
v_resetjp_1091_:
{
lean_object* v___x_1095_; 
if (v_isShared_1093_ == 0)
{
v___x_1095_ = v___x_1092_;
goto v_reusejp_1094_;
}
else
{
lean_object* v_reuseFailAlloc_1096_; 
v_reuseFailAlloc_1096_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1096_, 0, v_a_1090_);
v___x_1095_ = v_reuseFailAlloc_1096_;
goto v_reusejp_1094_;
}
v_reusejp_1094_:
{
return v___x_1095_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__1___boxed(lean_object* v_init_1100_, lean_object* v_as_1101_, lean_object* v_sz_1102_, lean_object* v_i_1103_, lean_object* v_b_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
size_t v_sz_boxed_1114_; size_t v_i_boxed_1115_; lean_object* v_res_1116_; 
v_sz_boxed_1114_ = lean_unbox_usize(v_sz_1102_);
lean_dec(v_sz_1102_);
v_i_boxed_1115_ = lean_unbox_usize(v_i_1103_);
lean_dec(v_i_1103_);
v_res_1116_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__1(v_init_1100_, v_as_1101_, v_sz_boxed_1114_, v_i_boxed_1115_, v_b_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_);
lean_dec(v___y_1112_);
lean_dec_ref(v___y_1111_);
lean_dec(v___y_1110_);
lean_dec_ref(v___y_1109_);
lean_dec(v___y_1108_);
lean_dec_ref(v___y_1107_);
lean_dec(v___y_1106_);
lean_dec_ref(v___y_1105_);
lean_dec_ref(v_as_1101_);
lean_dec(v_init_1100_);
return v_res_1116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0___boxed(lean_object* v_init_1117_, lean_object* v_n_1118_, lean_object* v_b_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_){
_start:
{
lean_object* v_res_1129_; 
v_res_1129_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0(v_init_1117_, v_n_1118_, v_b_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_, v___y_1126_, v___y_1127_);
lean_dec(v___y_1127_);
lean_dec_ref(v___y_1126_);
lean_dec(v___y_1125_);
lean_dec_ref(v___y_1124_);
lean_dec(v___y_1123_);
lean_dec_ref(v___y_1122_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
lean_dec_ref(v_n_1118_);
lean_dec(v_init_1117_);
return v_res_1129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___redArg(lean_object* v_as_1130_, size_t v_sz_1131_, size_t v_i_1132_, lean_object* v_b_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_){
_start:
{
uint8_t v___x_1139_; 
v___x_1139_ = lean_usize_dec_lt(v_i_1132_, v_sz_1131_);
if (v___x_1139_ == 0)
{
lean_object* v___x_1140_; 
v___x_1140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1140_, 0, v_b_1133_);
return v___x_1140_;
}
else
{
lean_object* v_snd_1141_; lean_object* v___x_1143_; uint8_t v_isShared_1144_; uint8_t v_isSharedCheck_1168_; 
v_snd_1141_ = lean_ctor_get(v_b_1133_, 1);
v_isSharedCheck_1168_ = !lean_is_exclusive(v_b_1133_);
if (v_isSharedCheck_1168_ == 0)
{
lean_object* v_unused_1169_; 
v_unused_1169_ = lean_ctor_get(v_b_1133_, 0);
lean_dec(v_unused_1169_);
v___x_1143_ = v_b_1133_;
v_isShared_1144_ = v_isSharedCheck_1168_;
goto v_resetjp_1142_;
}
else
{
lean_inc(v_snd_1141_);
lean_dec(v_b_1133_);
v___x_1143_ = lean_box(0);
v_isShared_1144_ = v_isSharedCheck_1168_;
goto v_resetjp_1142_;
}
v_resetjp_1142_:
{
lean_object* v___x_1145_; lean_object* v_a_1147_; lean_object* v_a_1154_; 
v___x_1145_ = lean_box(0);
v_a_1154_ = lean_array_uget_borrowed(v_as_1130_, v_i_1132_);
if (lean_obj_tag(v_a_1154_) == 0)
{
v_a_1147_ = v_snd_1141_;
goto v___jp_1146_;
}
else
{
lean_object* v_val_1155_; uint8_t v___x_1156_; 
v_val_1155_ = lean_ctor_get(v_a_1154_, 0);
v___x_1156_ = l_Lean_LocalDecl_isAuxDecl(v_val_1155_);
if (v___x_1156_ == 0)
{
v_a_1147_ = v_snd_1141_;
goto v___jp_1146_;
}
else
{
lean_object* v___x_1157_; lean_object* v___x_1158_; 
v___x_1157_ = l_Lean_LocalDecl_fvarId(v_val_1155_);
v___x_1158_ = l_Lean_MVarId_tryClear(v_snd_1141_, v___x_1157_, v___y_1134_, v___y_1135_, v___y_1136_, v___y_1137_);
if (lean_obj_tag(v___x_1158_) == 0)
{
lean_object* v_a_1159_; 
v_a_1159_ = lean_ctor_get(v___x_1158_, 0);
lean_inc(v_a_1159_);
lean_dec_ref_known(v___x_1158_, 1);
v_a_1147_ = v_a_1159_;
goto v___jp_1146_;
}
else
{
lean_object* v_a_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1167_; 
lean_del_object(v___x_1143_);
v_a_1160_ = lean_ctor_get(v___x_1158_, 0);
v_isSharedCheck_1167_ = !lean_is_exclusive(v___x_1158_);
if (v_isSharedCheck_1167_ == 0)
{
v___x_1162_ = v___x_1158_;
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_a_1160_);
lean_dec(v___x_1158_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___x_1165_; 
if (v_isShared_1163_ == 0)
{
v___x_1165_ = v___x_1162_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v_a_1160_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
}
}
}
v___jp_1146_:
{
lean_object* v___x_1149_; 
if (v_isShared_1144_ == 0)
{
lean_ctor_set(v___x_1143_, 1, v_a_1147_);
lean_ctor_set(v___x_1143_, 0, v___x_1145_);
v___x_1149_ = v___x_1143_;
goto v_reusejp_1148_;
}
else
{
lean_object* v_reuseFailAlloc_1153_; 
v_reuseFailAlloc_1153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1153_, 0, v___x_1145_);
lean_ctor_set(v_reuseFailAlloc_1153_, 1, v_a_1147_);
v___x_1149_ = v_reuseFailAlloc_1153_;
goto v_reusejp_1148_;
}
v_reusejp_1148_:
{
size_t v___x_1150_; size_t v___x_1151_; 
v___x_1150_ = ((size_t)1ULL);
v___x_1151_ = lean_usize_add(v_i_1132_, v___x_1150_);
v_i_1132_ = v___x_1151_;
v_b_1133_ = v___x_1149_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_as_1170_, lean_object* v_sz_1171_, lean_object* v_i_1172_, lean_object* v_b_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_){
_start:
{
size_t v_sz_boxed_1179_; size_t v_i_boxed_1180_; lean_object* v_res_1181_; 
v_sz_boxed_1179_ = lean_unbox_usize(v_sz_1171_);
lean_dec(v_sz_1171_);
v_i_boxed_1180_ = lean_unbox_usize(v_i_1172_);
lean_dec(v_i_1172_);
v_res_1181_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___redArg(v_as_1170_, v_sz_boxed_1179_, v_i_boxed_1180_, v_b_1173_, v___y_1174_, v___y_1175_, v___y_1176_, v___y_1177_);
lean_dec(v___y_1177_);
lean_dec_ref(v___y_1176_);
lean_dec(v___y_1175_);
lean_dec_ref(v___y_1174_);
lean_dec_ref(v_as_1170_);
return v_res_1181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1(lean_object* v_as_1182_, size_t v_sz_1183_, size_t v_i_1184_, lean_object* v_b_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_){
_start:
{
uint8_t v___x_1195_; 
v___x_1195_ = lean_usize_dec_lt(v_i_1184_, v_sz_1183_);
if (v___x_1195_ == 0)
{
lean_object* v___x_1196_; 
v___x_1196_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1196_, 0, v_b_1185_);
return v___x_1196_;
}
else
{
lean_object* v_snd_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1224_; 
v_snd_1197_ = lean_ctor_get(v_b_1185_, 1);
v_isSharedCheck_1224_ = !lean_is_exclusive(v_b_1185_);
if (v_isSharedCheck_1224_ == 0)
{
lean_object* v_unused_1225_; 
v_unused_1225_ = lean_ctor_get(v_b_1185_, 0);
lean_dec(v_unused_1225_);
v___x_1199_ = v_b_1185_;
v_isShared_1200_ = v_isSharedCheck_1224_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_snd_1197_);
lean_dec(v_b_1185_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1224_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___x_1201_; lean_object* v_a_1203_; lean_object* v_a_1210_; 
v___x_1201_ = lean_box(0);
v_a_1210_ = lean_array_uget_borrowed(v_as_1182_, v_i_1184_);
if (lean_obj_tag(v_a_1210_) == 0)
{
v_a_1203_ = v_snd_1197_;
goto v___jp_1202_;
}
else
{
lean_object* v_val_1211_; uint8_t v___x_1212_; 
v_val_1211_ = lean_ctor_get(v_a_1210_, 0);
v___x_1212_ = l_Lean_LocalDecl_isAuxDecl(v_val_1211_);
if (v___x_1212_ == 0)
{
v_a_1203_ = v_snd_1197_;
goto v___jp_1202_;
}
else
{
lean_object* v___x_1213_; lean_object* v___x_1214_; 
v___x_1213_ = l_Lean_LocalDecl_fvarId(v_val_1211_);
v___x_1214_ = l_Lean_MVarId_tryClear(v_snd_1197_, v___x_1213_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_);
if (lean_obj_tag(v___x_1214_) == 0)
{
lean_object* v_a_1215_; 
v_a_1215_ = lean_ctor_get(v___x_1214_, 0);
lean_inc(v_a_1215_);
lean_dec_ref_known(v___x_1214_, 1);
v_a_1203_ = v_a_1215_;
goto v___jp_1202_;
}
else
{
lean_object* v_a_1216_; lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1223_; 
lean_del_object(v___x_1199_);
v_a_1216_ = lean_ctor_get(v___x_1214_, 0);
v_isSharedCheck_1223_ = !lean_is_exclusive(v___x_1214_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1218_ = v___x_1214_;
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
else
{
lean_inc(v_a_1216_);
lean_dec(v___x_1214_);
v___x_1218_ = lean_box(0);
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
v_resetjp_1217_:
{
lean_object* v___x_1221_; 
if (v_isShared_1219_ == 0)
{
v___x_1221_ = v___x_1218_;
goto v_reusejp_1220_;
}
else
{
lean_object* v_reuseFailAlloc_1222_; 
v_reuseFailAlloc_1222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1222_, 0, v_a_1216_);
v___x_1221_ = v_reuseFailAlloc_1222_;
goto v_reusejp_1220_;
}
v_reusejp_1220_:
{
return v___x_1221_;
}
}
}
}
}
v___jp_1202_:
{
lean_object* v___x_1205_; 
if (v_isShared_1200_ == 0)
{
lean_ctor_set(v___x_1199_, 1, v_a_1203_);
lean_ctor_set(v___x_1199_, 0, v___x_1201_);
v___x_1205_ = v___x_1199_;
goto v_reusejp_1204_;
}
else
{
lean_object* v_reuseFailAlloc_1209_; 
v_reuseFailAlloc_1209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1209_, 0, v___x_1201_);
lean_ctor_set(v_reuseFailAlloc_1209_, 1, v_a_1203_);
v___x_1205_ = v_reuseFailAlloc_1209_;
goto v_reusejp_1204_;
}
v_reusejp_1204_:
{
size_t v___x_1206_; size_t v___x_1207_; lean_object* v___x_1208_; 
v___x_1206_ = ((size_t)1ULL);
v___x_1207_ = lean_usize_add(v_i_1184_, v___x_1206_);
v___x_1208_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___redArg(v_as_1182_, v_sz_1183_, v___x_1207_, v___x_1205_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_);
return v___x_1208_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1___boxed(lean_object* v_as_1226_, lean_object* v_sz_1227_, lean_object* v_i_1228_, lean_object* v_b_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
size_t v_sz_boxed_1239_; size_t v_i_boxed_1240_; lean_object* v_res_1241_; 
v_sz_boxed_1239_ = lean_unbox_usize(v_sz_1227_);
lean_dec(v_sz_1227_);
v_i_boxed_1240_ = lean_unbox_usize(v_i_1228_);
lean_dec(v_i_1228_);
v_res_1241_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1(v_as_1226_, v_sz_boxed_1239_, v_i_boxed_1240_, v_b_1229_, v___y_1230_, v___y_1231_, v___y_1232_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_);
lean_dec(v___y_1237_);
lean_dec_ref(v___y_1236_);
lean_dec(v___y_1235_);
lean_dec_ref(v___y_1234_);
lean_dec(v___y_1233_);
lean_dec_ref(v___y_1232_);
lean_dec(v___y_1231_);
lean_dec_ref(v___y_1230_);
lean_dec_ref(v_as_1226_);
return v_res_1241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0(lean_object* v_t_1242_, lean_object* v_init_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_){
_start:
{
lean_object* v_root_1253_; lean_object* v_tail_1254_; lean_object* v___x_1255_; 
v_root_1253_ = lean_ctor_get(v_t_1242_, 0);
v_tail_1254_ = lean_ctor_get(v_t_1242_, 1);
lean_inc(v_init_1243_);
v___x_1255_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0(v_init_1243_, v_root_1253_, v_init_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_);
lean_dec(v_init_1243_);
if (lean_obj_tag(v___x_1255_) == 0)
{
lean_object* v_a_1256_; lean_object* v___x_1258_; uint8_t v_isShared_1259_; uint8_t v_isSharedCheck_1292_; 
v_a_1256_ = lean_ctor_get(v___x_1255_, 0);
v_isSharedCheck_1292_ = !lean_is_exclusive(v___x_1255_);
if (v_isSharedCheck_1292_ == 0)
{
v___x_1258_ = v___x_1255_;
v_isShared_1259_ = v_isSharedCheck_1292_;
goto v_resetjp_1257_;
}
else
{
lean_inc(v_a_1256_);
lean_dec(v___x_1255_);
v___x_1258_ = lean_box(0);
v_isShared_1259_ = v_isSharedCheck_1292_;
goto v_resetjp_1257_;
}
v_resetjp_1257_:
{
if (lean_obj_tag(v_a_1256_) == 0)
{
lean_object* v_a_1260_; lean_object* v___x_1262_; 
v_a_1260_ = lean_ctor_get(v_a_1256_, 0);
lean_inc(v_a_1260_);
lean_dec_ref_known(v_a_1256_, 1);
if (v_isShared_1259_ == 0)
{
lean_ctor_set(v___x_1258_, 0, v_a_1260_);
v___x_1262_ = v___x_1258_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v_a_1260_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
return v___x_1262_;
}
}
else
{
lean_object* v_a_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; size_t v_sz_1267_; size_t v___x_1268_; lean_object* v___x_1269_; 
lean_del_object(v___x_1258_);
v_a_1264_ = lean_ctor_get(v_a_1256_, 0);
lean_inc(v_a_1264_);
lean_dec_ref_known(v_a_1256_, 1);
v___x_1265_ = lean_box(0);
v___x_1266_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1266_, 0, v___x_1265_);
lean_ctor_set(v___x_1266_, 1, v_a_1264_);
v_sz_1267_ = lean_array_size(v_tail_1254_);
v___x_1268_ = ((size_t)0ULL);
v___x_1269_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1(v_tail_1254_, v_sz_1267_, v___x_1268_, v___x_1266_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_);
if (lean_obj_tag(v___x_1269_) == 0)
{
lean_object* v_a_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1283_; 
v_a_1270_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1283_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1283_ == 0)
{
v___x_1272_ = v___x_1269_;
v_isShared_1273_ = v_isSharedCheck_1283_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_a_1270_);
lean_dec(v___x_1269_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1283_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v_fst_1274_; 
v_fst_1274_ = lean_ctor_get(v_a_1270_, 0);
if (lean_obj_tag(v_fst_1274_) == 0)
{
lean_object* v_snd_1275_; lean_object* v___x_1277_; 
v_snd_1275_ = lean_ctor_get(v_a_1270_, 1);
lean_inc(v_snd_1275_);
lean_dec(v_a_1270_);
if (v_isShared_1273_ == 0)
{
lean_ctor_set(v___x_1272_, 0, v_snd_1275_);
v___x_1277_ = v___x_1272_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1278_; 
v_reuseFailAlloc_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1278_, 0, v_snd_1275_);
v___x_1277_ = v_reuseFailAlloc_1278_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
return v___x_1277_;
}
}
else
{
lean_object* v_val_1279_; lean_object* v___x_1281_; 
lean_inc_ref(v_fst_1274_);
lean_dec(v_a_1270_);
v_val_1279_ = lean_ctor_get(v_fst_1274_, 0);
lean_inc(v_val_1279_);
lean_dec_ref_known(v_fst_1274_, 1);
if (v_isShared_1273_ == 0)
{
lean_ctor_set(v___x_1272_, 0, v_val_1279_);
v___x_1281_ = v___x_1272_;
goto v_reusejp_1280_;
}
else
{
lean_object* v_reuseFailAlloc_1282_; 
v_reuseFailAlloc_1282_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1282_, 0, v_val_1279_);
v___x_1281_ = v_reuseFailAlloc_1282_;
goto v_reusejp_1280_;
}
v_reusejp_1280_:
{
return v___x_1281_;
}
}
}
}
else
{
lean_object* v_a_1284_; lean_object* v___x_1286_; uint8_t v_isShared_1287_; uint8_t v_isSharedCheck_1291_; 
v_a_1284_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1291_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1291_ == 0)
{
v___x_1286_ = v___x_1269_;
v_isShared_1287_ = v_isSharedCheck_1291_;
goto v_resetjp_1285_;
}
else
{
lean_inc(v_a_1284_);
lean_dec(v___x_1269_);
v___x_1286_ = lean_box(0);
v_isShared_1287_ = v_isSharedCheck_1291_;
goto v_resetjp_1285_;
}
v_resetjp_1285_:
{
lean_object* v___x_1289_; 
if (v_isShared_1287_ == 0)
{
v___x_1289_ = v___x_1286_;
goto v_reusejp_1288_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v_a_1284_);
v___x_1289_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1288_;
}
v_reusejp_1288_:
{
return v___x_1289_;
}
}
}
}
}
}
else
{
lean_object* v_a_1293_; lean_object* v___x_1295_; uint8_t v_isShared_1296_; uint8_t v_isSharedCheck_1300_; 
v_a_1293_ = lean_ctor_get(v___x_1255_, 0);
v_isSharedCheck_1300_ = !lean_is_exclusive(v___x_1255_);
if (v_isSharedCheck_1300_ == 0)
{
v___x_1295_ = v___x_1255_;
v_isShared_1296_ = v_isSharedCheck_1300_;
goto v_resetjp_1294_;
}
else
{
lean_inc(v_a_1293_);
lean_dec(v___x_1255_);
v___x_1295_ = lean_box(0);
v_isShared_1296_ = v_isSharedCheck_1300_;
goto v_resetjp_1294_;
}
v_resetjp_1294_:
{
lean_object* v___x_1298_; 
if (v_isShared_1296_ == 0)
{
v___x_1298_ = v___x_1295_;
goto v_reusejp_1297_;
}
else
{
lean_object* v_reuseFailAlloc_1299_; 
v_reuseFailAlloc_1299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1299_, 0, v_a_1293_);
v___x_1298_ = v_reuseFailAlloc_1299_;
goto v_reusejp_1297_;
}
v_reusejp_1297_:
{
return v___x_1298_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0___boxed(lean_object* v_t_1301_, lean_object* v_init_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0(v_t_1301_, v_init_1302_, v___y_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_, v___y_1310_);
lean_dec(v___y_1310_);
lean_dec_ref(v___y_1309_);
lean_dec(v___y_1308_);
lean_dec_ref(v___y_1307_);
lean_dec(v___y_1306_);
lean_dec_ref(v___y_1305_);
lean_dec(v___y_1304_);
lean_dec_ref(v___y_1303_);
lean_dec_ref(v_t_1301_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___lam__0(lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1314_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
if (lean_obj_tag(v___x_1322_) == 0)
{
lean_object* v_lctx_1323_; lean_object* v_a_1324_; lean_object* v_decls_1325_; lean_object* v___x_1326_; 
v_lctx_1323_ = lean_ctor_get(v___y_1317_, 2);
v_a_1324_ = lean_ctor_get(v___x_1322_, 0);
lean_inc(v_a_1324_);
lean_dec_ref_known(v___x_1322_, 1);
v_decls_1325_ = lean_ctor_get(v_lctx_1323_, 1);
v___x_1326_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0(v_decls_1325_, v_a_1324_, v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
if (lean_obj_tag(v___x_1326_) == 0)
{
lean_object* v_a_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; 
v_a_1327_ = lean_ctor_get(v___x_1326_, 0);
lean_inc(v_a_1327_);
lean_dec_ref_known(v___x_1326_, 1);
v___x_1328_ = lean_box(0);
v___x_1329_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1329_, 0, v_a_1327_);
lean_ctor_set(v___x_1329_, 1, v___x_1328_);
v___x_1330_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1329_, v___y_1314_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
return v___x_1330_;
}
else
{
lean_object* v_a_1331_; lean_object* v___x_1333_; uint8_t v_isShared_1334_; uint8_t v_isSharedCheck_1338_; 
v_a_1331_ = lean_ctor_get(v___x_1326_, 0);
v_isSharedCheck_1338_ = !lean_is_exclusive(v___x_1326_);
if (v_isSharedCheck_1338_ == 0)
{
v___x_1333_ = v___x_1326_;
v_isShared_1334_ = v_isSharedCheck_1338_;
goto v_resetjp_1332_;
}
else
{
lean_inc(v_a_1331_);
lean_dec(v___x_1326_);
v___x_1333_ = lean_box(0);
v_isShared_1334_ = v_isSharedCheck_1338_;
goto v_resetjp_1332_;
}
v_resetjp_1332_:
{
lean_object* v___x_1336_; 
if (v_isShared_1334_ == 0)
{
v___x_1336_ = v___x_1333_;
goto v_reusejp_1335_;
}
else
{
lean_object* v_reuseFailAlloc_1337_; 
v_reuseFailAlloc_1337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1337_, 0, v_a_1331_);
v___x_1336_ = v_reuseFailAlloc_1337_;
goto v_reusejp_1335_;
}
v_reusejp_1335_:
{
return v___x_1336_;
}
}
}
}
else
{
lean_object* v_a_1339_; lean_object* v___x_1341_; uint8_t v_isShared_1342_; uint8_t v_isSharedCheck_1346_; 
v_a_1339_ = lean_ctor_get(v___x_1322_, 0);
v_isSharedCheck_1346_ = !lean_is_exclusive(v___x_1322_);
if (v_isSharedCheck_1346_ == 0)
{
v___x_1341_ = v___x_1322_;
v_isShared_1342_ = v_isSharedCheck_1346_;
goto v_resetjp_1340_;
}
else
{
lean_inc(v_a_1339_);
lean_dec(v___x_1322_);
v___x_1341_ = lean_box(0);
v_isShared_1342_ = v_isSharedCheck_1346_;
goto v_resetjp_1340_;
}
v_resetjp_1340_:
{
lean_object* v___x_1344_; 
if (v_isShared_1342_ == 0)
{
v___x_1344_ = v___x_1341_;
goto v_reusejp_1343_;
}
else
{
lean_object* v_reuseFailAlloc_1345_; 
v_reuseFailAlloc_1345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1345_, 0, v_a_1339_);
v___x_1344_ = v_reuseFailAlloc_1345_;
goto v_reusejp_1343_;
}
v_reusejp_1343_:
{
return v___x_1344_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___lam__0___boxed(lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_){
_start:
{
lean_object* v_res_1356_; 
v_res_1356_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___lam__0(v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
lean_dec(v___y_1354_);
lean_dec_ref(v___y_1353_);
lean_dec(v___y_1352_);
lean_dec_ref(v___y_1351_);
lean_dec(v___y_1350_);
lean_dec_ref(v___y_1349_);
lean_dec(v___y_1348_);
lean_dec_ref(v___y_1347_);
return v_res_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1(lean_object* v_x_1358_, lean_object* v_a_1359_, lean_object* v_a_1360_, lean_object* v_a_1361_, lean_object* v_a_1362_, lean_object* v_a_1363_, lean_object* v_a_1364_, lean_object* v_a_1365_, lean_object* v_a_1366_){
_start:
{
lean_object* v___x_1368_; uint8_t v___x_1369_; 
v___x_1368_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_clearAuxDecl___closed__1));
v___x_1369_ = l_Lean_Syntax_isOfKind(v_x_1358_, v___x_1368_);
if (v___x_1369_ == 0)
{
lean_object* v___x_1370_; 
v___x_1370_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_evalIntrov_spec__0___redArg();
return v___x_1370_;
}
else
{
lean_object* v___f_1371_; lean_object* v___x_1372_; 
v___f_1371_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___closed__0));
v___x_1372_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1371_, v_a_1359_, v_a_1360_, v_a_1361_, v_a_1362_, v_a_1363_, v_a_1364_, v_a_1365_, v_a_1366_);
return v___x_1372_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1___boxed(lean_object* v_x_1373_, lean_object* v_a_1374_, lean_object* v_a_1375_, lean_object* v_a_1376_, lean_object* v_a_1377_, lean_object* v_a_1378_, lean_object* v_a_1379_, lean_object* v_a_1380_, lean_object* v_a_1381_, lean_object* v_a_1382_){
_start:
{
lean_object* v_res_1383_; 
v_res_1383_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1(v_x_1373_, v_a_1374_, v_a_1375_, v_a_1376_, v_a_1377_, v_a_1378_, v_a_1379_, v_a_1380_, v_a_1381_);
lean_dec(v_a_1381_);
lean_dec_ref(v_a_1380_);
lean_dec(v_a_1379_);
lean_dec_ref(v_a_1378_);
lean_dec(v_a_1377_);
lean_dec_ref(v_a_1376_);
lean_dec(v_a_1375_);
lean_dec_ref(v_a_1374_);
return v_res_1383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4(lean_object* v_as_1384_, size_t v_sz_1385_, size_t v_i_1386_, lean_object* v_b_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_){
_start:
{
lean_object* v___x_1397_; 
v___x_1397_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___redArg(v_as_1384_, v_sz_1385_, v_i_1386_, v_b_1387_, v___y_1392_, v___y_1393_, v___y_1394_, v___y_1395_);
return v___x_1397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4___boxed(lean_object* v_as_1398_, lean_object* v_sz_1399_, lean_object* v_i_1400_, lean_object* v_b_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_){
_start:
{
size_t v_sz_boxed_1411_; size_t v_i_boxed_1412_; lean_object* v_res_1413_; 
v_sz_boxed_1411_ = lean_unbox_usize(v_sz_1399_);
lean_dec(v_sz_1399_);
v_i_boxed_1412_ = lean_unbox_usize(v_i_1400_);
lean_dec(v_i_1400_);
v_res_1413_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__1_spec__4(v_as_1398_, v_sz_boxed_1411_, v_i_boxed_1412_, v_b_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_);
lean_dec(v___y_1409_);
lean_dec_ref(v___y_1408_);
lean_dec(v___y_1407_);
lean_dec_ref(v___y_1406_);
lean_dec(v___y_1405_);
lean_dec_ref(v___y_1404_);
lean_dec(v___y_1403_);
lean_dec_ref(v___y_1402_);
lean_dec_ref(v_as_1398_);
return v_res_1413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3(lean_object* v_as_1414_, size_t v_sz_1415_, size_t v_i_1416_, lean_object* v_b_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_){
_start:
{
lean_object* v___x_1427_; 
v___x_1427_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___redArg(v_as_1414_, v_sz_1415_, v_i_1416_, v_b_1417_, v___y_1422_, v___y_1423_, v___y_1424_, v___y_1425_);
return v___x_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_as_1428_, lean_object* v_sz_1429_, lean_object* v_i_1430_, lean_object* v_b_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_){
_start:
{
size_t v_sz_boxed_1441_; size_t v_i_boxed_1442_; lean_object* v_res_1443_; 
v_sz_boxed_1441_ = lean_unbox_usize(v_sz_1429_);
lean_dec(v_sz_1429_);
v_i_boxed_1442_ = lean_unbox_usize(v_i_1430_);
lean_dec(v_i_1430_);
v_res_1443_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Basic______elabRules__Mathlib__Tactic__clearAuxDecl__1_spec__0_spec__0_spec__2_spec__3(v_as_1428_, v_sz_boxed_1441_, v_i_boxed_1442_, v_b_1431_, v___y_1432_, v___y_1433_, v___y_1434_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_, v___y_1439_);
lean_dec(v___y_1439_);
lean_dec_ref(v___y_1438_);
lean_dec(v___y_1437_);
lean_dec_ref(v___y_1436_);
lean_dec(v___y_1435_);
lean_dec_ref(v___y_1434_);
lean_dec(v___y_1433_);
lean_dec_ref(v___y_1432_);
lean_dec_ref(v_as_1428_);
return v_res_1443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___redArg(lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_){
_start:
{
lean_object* v___x_1448_; lean_object* v_env_1449_; lean_object* v___x_1450_; lean_object* v_mctx_1451_; lean_object* v_options_1452_; lean_object* v_currNamespace_1453_; lean_object* v_openDecls_1454_; lean_object* v___x_1455_; lean_object* v_ngen_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; 
v___x_1448_ = lean_st_ref_get(v___y_1446_);
v_env_1449_ = lean_ctor_get(v___x_1448_, 0);
lean_inc_ref(v_env_1449_);
lean_dec(v___x_1448_);
v___x_1450_ = lean_st_ref_get(v___y_1444_);
v_mctx_1451_ = lean_ctor_get(v___x_1450_, 0);
lean_inc_ref(v_mctx_1451_);
lean_dec(v___x_1450_);
v_options_1452_ = lean_ctor_get(v___y_1445_, 2);
v_currNamespace_1453_ = lean_ctor_get(v___y_1445_, 6);
v_openDecls_1454_ = lean_ctor_get(v___y_1445_, 7);
v___x_1455_ = lean_st_ref_get(v___y_1446_);
v_ngen_1456_ = lean_ctor_get(v___x_1455_, 2);
lean_inc_ref(v_ngen_1456_);
lean_dec(v___x_1455_);
v___x_1457_ = lean_box(0);
v___x_1458_ = l_Lean_instInhabitedFileMap_default;
lean_inc(v_openDecls_1454_);
lean_inc(v_currNamespace_1453_);
lean_inc_ref(v_options_1452_);
v___x_1459_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_1459_, 0, v_env_1449_);
lean_ctor_set(v___x_1459_, 1, v___x_1457_);
lean_ctor_set(v___x_1459_, 2, v___x_1458_);
lean_ctor_set(v___x_1459_, 3, v_mctx_1451_);
lean_ctor_set(v___x_1459_, 4, v_options_1452_);
lean_ctor_set(v___x_1459_, 5, v_currNamespace_1453_);
lean_ctor_set(v___x_1459_, 6, v_openDecls_1454_);
lean_ctor_set(v___x_1459_, 7, v_ngen_1456_);
v___x_1460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1460_, 0, v___x_1459_);
return v___x_1460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___redArg___boxed(lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_){
_start:
{
lean_object* v_res_1465_; 
v_res_1465_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___redArg(v___y_1461_, v___y_1462_, v___y_1463_);
lean_dec(v___y_1463_);
lean_dec_ref(v___y_1462_);
lean_dec(v___y_1461_);
return v_res_1465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0(lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_){
_start:
{
lean_object* v___x_1475_; lean_object* v_a_1476_; lean_object* v___x_1478_; uint8_t v_isShared_1479_; uint8_t v_isSharedCheck_1500_; 
v___x_1475_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___redArg(v___y_1471_, v___y_1472_, v___y_1473_);
v_a_1476_ = lean_ctor_get(v___x_1475_, 0);
v_isSharedCheck_1500_ = !lean_is_exclusive(v___x_1475_);
if (v_isSharedCheck_1500_ == 0)
{
v___x_1478_ = v___x_1475_;
v_isShared_1479_ = v_isSharedCheck_1500_;
goto v_resetjp_1477_;
}
else
{
lean_inc(v_a_1476_);
lean_dec(v___x_1475_);
v___x_1478_ = lean_box(0);
v_isShared_1479_ = v_isSharedCheck_1500_;
goto v_resetjp_1477_;
}
v_resetjp_1477_:
{
lean_object* v_fileMap_1480_; lean_object* v_env_1481_; lean_object* v_mctx_1482_; lean_object* v_options_1483_; lean_object* v_currNamespace_1484_; lean_object* v_openDecls_1485_; lean_object* v_ngen_1486_; lean_object* v___x_1488_; uint8_t v_isShared_1489_; uint8_t v_isSharedCheck_1497_; 
v_fileMap_1480_ = lean_ctor_get(v___y_1472_, 1);
v_env_1481_ = lean_ctor_get(v_a_1476_, 0);
v_mctx_1482_ = lean_ctor_get(v_a_1476_, 3);
v_options_1483_ = lean_ctor_get(v_a_1476_, 4);
v_currNamespace_1484_ = lean_ctor_get(v_a_1476_, 5);
v_openDecls_1485_ = lean_ctor_get(v_a_1476_, 6);
v_ngen_1486_ = lean_ctor_get(v_a_1476_, 7);
v_isSharedCheck_1497_ = !lean_is_exclusive(v_a_1476_);
if (v_isSharedCheck_1497_ == 0)
{
lean_object* v_unused_1498_; lean_object* v_unused_1499_; 
v_unused_1498_ = lean_ctor_get(v_a_1476_, 2);
lean_dec(v_unused_1498_);
v_unused_1499_ = lean_ctor_get(v_a_1476_, 1);
lean_dec(v_unused_1499_);
v___x_1488_ = v_a_1476_;
v_isShared_1489_ = v_isSharedCheck_1497_;
goto v_resetjp_1487_;
}
else
{
lean_inc(v_ngen_1486_);
lean_inc(v_openDecls_1485_);
lean_inc(v_currNamespace_1484_);
lean_inc(v_options_1483_);
lean_inc(v_mctx_1482_);
lean_inc(v_env_1481_);
lean_dec(v_a_1476_);
v___x_1488_ = lean_box(0);
v_isShared_1489_ = v_isSharedCheck_1497_;
goto v_resetjp_1487_;
}
v_resetjp_1487_:
{
lean_object* v___x_1490_; lean_object* v___x_1492_; 
v___x_1490_ = lean_box(0);
lean_inc_ref(v_fileMap_1480_);
if (v_isShared_1489_ == 0)
{
lean_ctor_set(v___x_1488_, 2, v_fileMap_1480_);
lean_ctor_set(v___x_1488_, 1, v___x_1490_);
v___x_1492_ = v___x_1488_;
goto v_reusejp_1491_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v_env_1481_);
lean_ctor_set(v_reuseFailAlloc_1496_, 1, v___x_1490_);
lean_ctor_set(v_reuseFailAlloc_1496_, 2, v_fileMap_1480_);
lean_ctor_set(v_reuseFailAlloc_1496_, 3, v_mctx_1482_);
lean_ctor_set(v_reuseFailAlloc_1496_, 4, v_options_1483_);
lean_ctor_set(v_reuseFailAlloc_1496_, 5, v_currNamespace_1484_);
lean_ctor_set(v_reuseFailAlloc_1496_, 6, v_openDecls_1485_);
lean_ctor_set(v_reuseFailAlloc_1496_, 7, v_ngen_1486_);
v___x_1492_ = v_reuseFailAlloc_1496_;
goto v_reusejp_1491_;
}
v_reusejp_1491_:
{
lean_object* v___x_1494_; 
if (v_isShared_1479_ == 0)
{
lean_ctor_set(v___x_1478_, 0, v___x_1492_);
v___x_1494_ = v___x_1478_;
goto v_reusejp_1493_;
}
else
{
lean_object* v_reuseFailAlloc_1495_; 
v_reuseFailAlloc_1495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1495_, 0, v___x_1492_);
v___x_1494_ = v_reuseFailAlloc_1495_;
goto v_reusejp_1493_;
}
v_reusejp_1493_:
{
return v___x_1494_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0___boxed(lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_){
_start:
{
lean_object* v_res_1510_; 
v_res_1510_ = lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0(v___y_1501_, v___y_1502_, v___y_1503_, v___y_1504_, v___y_1505_, v___y_1506_, v___y_1507_, v___y_1508_);
lean_dec(v___y_1508_);
lean_dec_ref(v___y_1507_);
lean_dec(v___y_1506_);
lean_dec_ref(v___y_1505_);
lean_dec(v___y_1504_);
lean_dec_ref(v___y_1503_);
lean_dec(v___y_1502_);
lean_dec_ref(v___y_1501_);
return v_res_1510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__3(lean_object* v___x_1511_, size_t v_sz_1512_, size_t v_i_1513_, lean_object* v_bs_1514_, lean_object* v___y_1515_, lean_object* v___y_1516_, lean_object* v___y_1517_, lean_object* v___y_1518_, lean_object* v___y_1519_, lean_object* v___y_1520_, lean_object* v___y_1521_, lean_object* v___y_1522_){
_start:
{
uint8_t v___x_1524_; 
v___x_1524_ = lean_usize_dec_lt(v_i_1513_, v_sz_1512_);
if (v___x_1524_ == 0)
{
lean_object* v___x_1525_; 
v___x_1525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1525_, 0, v_bs_1514_);
return v___x_1525_;
}
else
{
lean_object* v___x_1526_; 
v___x_1526_ = lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0(v___y_1515_, v___y_1516_, v___y_1517_, v___y_1518_, v___y_1519_, v___y_1520_, v___y_1521_, v___y_1522_);
if (lean_obj_tag(v___x_1526_) == 0)
{
lean_object* v_a_1527_; lean_object* v_assignment_1528_; lean_object* v_v_1529_; lean_object* v___x_1530_; lean_object* v_bs_x27_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; size_t v___x_1535_; size_t v___x_1536_; lean_object* v___x_1537_; 
v_a_1527_ = lean_ctor_get(v___x_1526_, 0);
lean_inc(v_a_1527_);
lean_dec_ref_known(v___x_1526_, 1);
v_assignment_1528_ = lean_ctor_get(v___x_1511_, 0);
v_v_1529_ = lean_array_uget(v_bs_1514_, v_i_1513_);
v___x_1530_ = lean_unsigned_to_nat(0u);
v_bs_x27_1531_ = lean_array_uset(v_bs_1514_, v_i_1513_, v___x_1530_);
v___x_1532_ = l_Lean_Elab_InfoTree_substitute(v_v_1529_, v_assignment_1528_);
v___x_1533_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1533_, 0, v_a_1527_);
v___x_1534_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1534_, 0, v___x_1533_);
lean_ctor_set(v___x_1534_, 1, v___x_1532_);
v___x_1535_ = ((size_t)1ULL);
v___x_1536_ = lean_usize_add(v_i_1513_, v___x_1535_);
v___x_1537_ = lean_array_uset(v_bs_x27_1531_, v_i_1513_, v___x_1534_);
v_i_1513_ = v___x_1536_;
v_bs_1514_ = v___x_1537_;
goto _start;
}
else
{
lean_object* v_a_1539_; lean_object* v___x_1541_; uint8_t v_isShared_1542_; uint8_t v_isSharedCheck_1546_; 
lean_dec_ref(v_bs_1514_);
v_a_1539_ = lean_ctor_get(v___x_1526_, 0);
v_isSharedCheck_1546_ = !lean_is_exclusive(v___x_1526_);
if (v_isSharedCheck_1546_ == 0)
{
v___x_1541_ = v___x_1526_;
v_isShared_1542_ = v_isSharedCheck_1546_;
goto v_resetjp_1540_;
}
else
{
lean_inc(v_a_1539_);
lean_dec(v___x_1526_);
v___x_1541_ = lean_box(0);
v_isShared_1542_ = v_isSharedCheck_1546_;
goto v_resetjp_1540_;
}
v_resetjp_1540_:
{
lean_object* v___x_1544_; 
if (v_isShared_1542_ == 0)
{
v___x_1544_ = v___x_1541_;
goto v_reusejp_1543_;
}
else
{
lean_object* v_reuseFailAlloc_1545_; 
v_reuseFailAlloc_1545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1545_, 0, v_a_1539_);
v___x_1544_ = v_reuseFailAlloc_1545_;
goto v_reusejp_1543_;
}
v_reusejp_1543_:
{
return v___x_1544_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__3___boxed(lean_object* v___x_1547_, lean_object* v_sz_1548_, lean_object* v_i_1549_, lean_object* v_bs_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_){
_start:
{
size_t v_sz_boxed_1560_; size_t v_i_boxed_1561_; lean_object* v_res_1562_; 
v_sz_boxed_1560_ = lean_unbox_usize(v_sz_1548_);
lean_dec(v_sz_1548_);
v_i_boxed_1561_ = lean_unbox_usize(v_i_1549_);
lean_dec(v_i_1549_);
v_res_1562_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__3(v___x_1547_, v_sz_boxed_1560_, v_i_boxed_1561_, v_bs_1550_, v___y_1551_, v___y_1552_, v___y_1553_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_, v___y_1558_);
lean_dec(v___y_1558_);
lean_dec_ref(v___y_1557_);
lean_dec(v___y_1556_);
lean_dec_ref(v___y_1555_);
lean_dec(v___y_1554_);
lean_dec_ref(v___y_1553_);
lean_dec(v___y_1552_);
lean_dec_ref(v___y_1551_);
lean_dec_ref(v___x_1547_);
return v_res_1562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2(lean_object* v___x_1563_, lean_object* v_x_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_, lean_object* v___y_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_){
_start:
{
if (lean_obj_tag(v_x_1564_) == 0)
{
lean_object* v_cs_1574_; lean_object* v___x_1576_; uint8_t v_isShared_1577_; uint8_t v_isSharedCheck_1600_; 
v_cs_1574_ = lean_ctor_get(v_x_1564_, 0);
v_isSharedCheck_1600_ = !lean_is_exclusive(v_x_1564_);
if (v_isSharedCheck_1600_ == 0)
{
v___x_1576_ = v_x_1564_;
v_isShared_1577_ = v_isSharedCheck_1600_;
goto v_resetjp_1575_;
}
else
{
lean_inc(v_cs_1574_);
lean_dec(v_x_1564_);
v___x_1576_ = lean_box(0);
v_isShared_1577_ = v_isSharedCheck_1600_;
goto v_resetjp_1575_;
}
v_resetjp_1575_:
{
size_t v_sz_1578_; size_t v___x_1579_; lean_object* v___x_1580_; 
v_sz_1578_ = lean_array_size(v_cs_1574_);
v___x_1579_ = ((size_t)0ULL);
v___x_1580_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2_spec__3(v___x_1563_, v_sz_1578_, v___x_1579_, v_cs_1574_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_, v___y_1572_);
if (lean_obj_tag(v___x_1580_) == 0)
{
lean_object* v_a_1581_; lean_object* v___x_1583_; uint8_t v_isShared_1584_; uint8_t v_isSharedCheck_1591_; 
v_a_1581_ = lean_ctor_get(v___x_1580_, 0);
v_isSharedCheck_1591_ = !lean_is_exclusive(v___x_1580_);
if (v_isSharedCheck_1591_ == 0)
{
v___x_1583_ = v___x_1580_;
v_isShared_1584_ = v_isSharedCheck_1591_;
goto v_resetjp_1582_;
}
else
{
lean_inc(v_a_1581_);
lean_dec(v___x_1580_);
v___x_1583_ = lean_box(0);
v_isShared_1584_ = v_isSharedCheck_1591_;
goto v_resetjp_1582_;
}
v_resetjp_1582_:
{
lean_object* v___x_1586_; 
if (v_isShared_1577_ == 0)
{
lean_ctor_set(v___x_1576_, 0, v_a_1581_);
v___x_1586_ = v___x_1576_;
goto v_reusejp_1585_;
}
else
{
lean_object* v_reuseFailAlloc_1590_; 
v_reuseFailAlloc_1590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1590_, 0, v_a_1581_);
v___x_1586_ = v_reuseFailAlloc_1590_;
goto v_reusejp_1585_;
}
v_reusejp_1585_:
{
lean_object* v___x_1588_; 
if (v_isShared_1584_ == 0)
{
lean_ctor_set(v___x_1583_, 0, v___x_1586_);
v___x_1588_ = v___x_1583_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v___x_1586_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
}
}
else
{
lean_object* v_a_1592_; lean_object* v___x_1594_; uint8_t v_isShared_1595_; uint8_t v_isSharedCheck_1599_; 
lean_del_object(v___x_1576_);
v_a_1592_ = lean_ctor_get(v___x_1580_, 0);
v_isSharedCheck_1599_ = !lean_is_exclusive(v___x_1580_);
if (v_isSharedCheck_1599_ == 0)
{
v___x_1594_ = v___x_1580_;
v_isShared_1595_ = v_isSharedCheck_1599_;
goto v_resetjp_1593_;
}
else
{
lean_inc(v_a_1592_);
lean_dec(v___x_1580_);
v___x_1594_ = lean_box(0);
v_isShared_1595_ = v_isSharedCheck_1599_;
goto v_resetjp_1593_;
}
v_resetjp_1593_:
{
lean_object* v___x_1597_; 
if (v_isShared_1595_ == 0)
{
v___x_1597_ = v___x_1594_;
goto v_reusejp_1596_;
}
else
{
lean_object* v_reuseFailAlloc_1598_; 
v_reuseFailAlloc_1598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1598_, 0, v_a_1592_);
v___x_1597_ = v_reuseFailAlloc_1598_;
goto v_reusejp_1596_;
}
v_reusejp_1596_:
{
return v___x_1597_;
}
}
}
}
}
else
{
lean_object* v_vs_1601_; lean_object* v___x_1603_; uint8_t v_isShared_1604_; uint8_t v_isSharedCheck_1627_; 
v_vs_1601_ = lean_ctor_get(v_x_1564_, 0);
v_isSharedCheck_1627_ = !lean_is_exclusive(v_x_1564_);
if (v_isSharedCheck_1627_ == 0)
{
v___x_1603_ = v_x_1564_;
v_isShared_1604_ = v_isSharedCheck_1627_;
goto v_resetjp_1602_;
}
else
{
lean_inc(v_vs_1601_);
lean_dec(v_x_1564_);
v___x_1603_ = lean_box(0);
v_isShared_1604_ = v_isSharedCheck_1627_;
goto v_resetjp_1602_;
}
v_resetjp_1602_:
{
size_t v_sz_1605_; size_t v___x_1606_; lean_object* v___x_1607_; 
v_sz_1605_ = lean_array_size(v_vs_1601_);
v___x_1606_ = ((size_t)0ULL);
v___x_1607_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__3(v___x_1563_, v_sz_1605_, v___x_1606_, v_vs_1601_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_, v___y_1572_);
if (lean_obj_tag(v___x_1607_) == 0)
{
lean_object* v_a_1608_; lean_object* v___x_1610_; uint8_t v_isShared_1611_; uint8_t v_isSharedCheck_1618_; 
v_a_1608_ = lean_ctor_get(v___x_1607_, 0);
v_isSharedCheck_1618_ = !lean_is_exclusive(v___x_1607_);
if (v_isSharedCheck_1618_ == 0)
{
v___x_1610_ = v___x_1607_;
v_isShared_1611_ = v_isSharedCheck_1618_;
goto v_resetjp_1609_;
}
else
{
lean_inc(v_a_1608_);
lean_dec(v___x_1607_);
v___x_1610_ = lean_box(0);
v_isShared_1611_ = v_isSharedCheck_1618_;
goto v_resetjp_1609_;
}
v_resetjp_1609_:
{
lean_object* v___x_1613_; 
if (v_isShared_1604_ == 0)
{
lean_ctor_set(v___x_1603_, 0, v_a_1608_);
v___x_1613_ = v___x_1603_;
goto v_reusejp_1612_;
}
else
{
lean_object* v_reuseFailAlloc_1617_; 
v_reuseFailAlloc_1617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1617_, 0, v_a_1608_);
v___x_1613_ = v_reuseFailAlloc_1617_;
goto v_reusejp_1612_;
}
v_reusejp_1612_:
{
lean_object* v___x_1615_; 
if (v_isShared_1611_ == 0)
{
lean_ctor_set(v___x_1610_, 0, v___x_1613_);
v___x_1615_ = v___x_1610_;
goto v_reusejp_1614_;
}
else
{
lean_object* v_reuseFailAlloc_1616_; 
v_reuseFailAlloc_1616_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1616_, 0, v___x_1613_);
v___x_1615_ = v_reuseFailAlloc_1616_;
goto v_reusejp_1614_;
}
v_reusejp_1614_:
{
return v___x_1615_;
}
}
}
}
else
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1626_; 
lean_del_object(v___x_1603_);
v_a_1619_ = lean_ctor_get(v___x_1607_, 0);
v_isSharedCheck_1626_ = !lean_is_exclusive(v___x_1607_);
if (v_isSharedCheck_1626_ == 0)
{
v___x_1621_ = v___x_1607_;
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1607_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1624_; 
if (v_isShared_1622_ == 0)
{
v___x_1624_ = v___x_1621_;
goto v_reusejp_1623_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_a_1619_);
v___x_1624_ = v_reuseFailAlloc_1625_;
goto v_reusejp_1623_;
}
v_reusejp_1623_:
{
return v___x_1624_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2_spec__3(lean_object* v___x_1628_, size_t v_sz_1629_, size_t v_i_1630_, lean_object* v_bs_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_){
_start:
{
uint8_t v___x_1641_; 
v___x_1641_ = lean_usize_dec_lt(v_i_1630_, v_sz_1629_);
if (v___x_1641_ == 0)
{
lean_object* v___x_1642_; 
v___x_1642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1642_, 0, v_bs_1631_);
return v___x_1642_;
}
else
{
lean_object* v_v_1643_; lean_object* v___x_1644_; 
v_v_1643_ = lean_array_uget_borrowed(v_bs_1631_, v_i_1630_);
lean_inc(v_v_1643_);
v___x_1644_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2(v___x_1628_, v_v_1643_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_);
if (lean_obj_tag(v___x_1644_) == 0)
{
lean_object* v_a_1645_; lean_object* v___x_1646_; lean_object* v_bs_x27_1647_; size_t v___x_1648_; size_t v___x_1649_; lean_object* v___x_1650_; 
v_a_1645_ = lean_ctor_get(v___x_1644_, 0);
lean_inc(v_a_1645_);
lean_dec_ref_known(v___x_1644_, 1);
v___x_1646_ = lean_unsigned_to_nat(0u);
v_bs_x27_1647_ = lean_array_uset(v_bs_1631_, v_i_1630_, v___x_1646_);
v___x_1648_ = ((size_t)1ULL);
v___x_1649_ = lean_usize_add(v_i_1630_, v___x_1648_);
v___x_1650_ = lean_array_uset(v_bs_x27_1647_, v_i_1630_, v_a_1645_);
v_i_1630_ = v___x_1649_;
v_bs_1631_ = v___x_1650_;
goto _start;
}
else
{
lean_object* v_a_1652_; lean_object* v___x_1654_; uint8_t v_isShared_1655_; uint8_t v_isSharedCheck_1659_; 
lean_dec_ref(v_bs_1631_);
v_a_1652_ = lean_ctor_get(v___x_1644_, 0);
v_isSharedCheck_1659_ = !lean_is_exclusive(v___x_1644_);
if (v_isSharedCheck_1659_ == 0)
{
v___x_1654_ = v___x_1644_;
v_isShared_1655_ = v_isSharedCheck_1659_;
goto v_resetjp_1653_;
}
else
{
lean_inc(v_a_1652_);
lean_dec(v___x_1644_);
v___x_1654_ = lean_box(0);
v_isShared_1655_ = v_isSharedCheck_1659_;
goto v_resetjp_1653_;
}
v_resetjp_1653_:
{
lean_object* v___x_1657_; 
if (v_isShared_1655_ == 0)
{
v___x_1657_ = v___x_1654_;
goto v_reusejp_1656_;
}
else
{
lean_object* v_reuseFailAlloc_1658_; 
v_reuseFailAlloc_1658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1658_, 0, v_a_1652_);
v___x_1657_ = v_reuseFailAlloc_1658_;
goto v_reusejp_1656_;
}
v_reusejp_1656_:
{
return v___x_1657_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2_spec__3___boxed(lean_object* v___x_1660_, lean_object* v_sz_1661_, lean_object* v_i_1662_, lean_object* v_bs_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_){
_start:
{
size_t v_sz_boxed_1673_; size_t v_i_boxed_1674_; lean_object* v_res_1675_; 
v_sz_boxed_1673_ = lean_unbox_usize(v_sz_1661_);
lean_dec(v_sz_1661_);
v_i_boxed_1674_ = lean_unbox_usize(v_i_1662_);
lean_dec(v_i_1662_);
v_res_1675_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2_spec__3(v___x_1660_, v_sz_boxed_1673_, v_i_boxed_1674_, v_bs_1663_, v___y_1664_, v___y_1665_, v___y_1666_, v___y_1667_, v___y_1668_, v___y_1669_, v___y_1670_, v___y_1671_);
lean_dec(v___y_1671_);
lean_dec_ref(v___y_1670_);
lean_dec(v___y_1669_);
lean_dec_ref(v___y_1668_);
lean_dec(v___y_1667_);
lean_dec_ref(v___y_1666_);
lean_dec(v___y_1665_);
lean_dec_ref(v___y_1664_);
lean_dec_ref(v___x_1660_);
return v_res_1675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2___boxed(lean_object* v___x_1676_, lean_object* v_x_1677_, lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_){
_start:
{
lean_object* v_res_1687_; 
v_res_1687_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2(v___x_1676_, v_x_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_, v___y_1682_, v___y_1683_, v___y_1684_, v___y_1685_);
lean_dec(v___y_1685_);
lean_dec_ref(v___y_1684_);
lean_dec(v___y_1683_);
lean_dec_ref(v___y_1682_);
lean_dec(v___y_1681_);
lean_dec_ref(v___y_1680_);
lean_dec(v___y_1679_);
lean_dec_ref(v___y_1678_);
lean_dec_ref(v___x_1676_);
return v_res_1687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1(lean_object* v___x_1688_, lean_object* v_t_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_){
_start:
{
lean_object* v_root_1699_; lean_object* v_tail_1700_; lean_object* v_size_1701_; size_t v_shift_1702_; lean_object* v_tailOff_1703_; lean_object* v___x_1705_; uint8_t v_isShared_1706_; uint8_t v_isSharedCheck_1739_; 
v_root_1699_ = lean_ctor_get(v_t_1689_, 0);
v_tail_1700_ = lean_ctor_get(v_t_1689_, 1);
v_size_1701_ = lean_ctor_get(v_t_1689_, 2);
v_shift_1702_ = lean_ctor_get_usize(v_t_1689_, 4);
v_tailOff_1703_ = lean_ctor_get(v_t_1689_, 3);
v_isSharedCheck_1739_ = !lean_is_exclusive(v_t_1689_);
if (v_isSharedCheck_1739_ == 0)
{
v___x_1705_ = v_t_1689_;
v_isShared_1706_ = v_isSharedCheck_1739_;
goto v_resetjp_1704_;
}
else
{
lean_inc(v_tailOff_1703_);
lean_inc(v_size_1701_);
lean_inc(v_tail_1700_);
lean_inc(v_root_1699_);
lean_dec(v_t_1689_);
v___x_1705_ = lean_box(0);
v_isShared_1706_ = v_isSharedCheck_1739_;
goto v_resetjp_1704_;
}
v_resetjp_1704_:
{
lean_object* v___x_1707_; 
v___x_1707_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__2(v___x_1688_, v_root_1699_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_, v___y_1695_, v___y_1696_, v___y_1697_);
if (lean_obj_tag(v___x_1707_) == 0)
{
lean_object* v_a_1708_; size_t v_sz_1709_; size_t v___x_1710_; lean_object* v___x_1711_; 
v_a_1708_ = lean_ctor_get(v___x_1707_, 0);
lean_inc(v_a_1708_);
lean_dec_ref_known(v___x_1707_, 1);
v_sz_1709_ = lean_array_size(v_tail_1700_);
v___x_1710_ = ((size_t)0ULL);
v___x_1711_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1_spec__3(v___x_1688_, v_sz_1709_, v___x_1710_, v_tail_1700_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_, v___y_1695_, v___y_1696_, v___y_1697_);
if (lean_obj_tag(v___x_1711_) == 0)
{
lean_object* v_a_1712_; lean_object* v___x_1714_; uint8_t v_isShared_1715_; uint8_t v_isSharedCheck_1722_; 
v_a_1712_ = lean_ctor_get(v___x_1711_, 0);
v_isSharedCheck_1722_ = !lean_is_exclusive(v___x_1711_);
if (v_isSharedCheck_1722_ == 0)
{
v___x_1714_ = v___x_1711_;
v_isShared_1715_ = v_isSharedCheck_1722_;
goto v_resetjp_1713_;
}
else
{
lean_inc(v_a_1712_);
lean_dec(v___x_1711_);
v___x_1714_ = lean_box(0);
v_isShared_1715_ = v_isSharedCheck_1722_;
goto v_resetjp_1713_;
}
v_resetjp_1713_:
{
lean_object* v___x_1717_; 
if (v_isShared_1706_ == 0)
{
lean_ctor_set(v___x_1705_, 1, v_a_1712_);
lean_ctor_set(v___x_1705_, 0, v_a_1708_);
v___x_1717_ = v___x_1705_;
goto v_reusejp_1716_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v_a_1708_);
lean_ctor_set(v_reuseFailAlloc_1721_, 1, v_a_1712_);
lean_ctor_set(v_reuseFailAlloc_1721_, 2, v_size_1701_);
lean_ctor_set(v_reuseFailAlloc_1721_, 3, v_tailOff_1703_);
lean_ctor_set_usize(v_reuseFailAlloc_1721_, 4, v_shift_1702_);
v___x_1717_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1716_;
}
v_reusejp_1716_:
{
lean_object* v___x_1719_; 
if (v_isShared_1715_ == 0)
{
lean_ctor_set(v___x_1714_, 0, v___x_1717_);
v___x_1719_ = v___x_1714_;
goto v_reusejp_1718_;
}
else
{
lean_object* v_reuseFailAlloc_1720_; 
v_reuseFailAlloc_1720_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1720_, 0, v___x_1717_);
v___x_1719_ = v_reuseFailAlloc_1720_;
goto v_reusejp_1718_;
}
v_reusejp_1718_:
{
return v___x_1719_;
}
}
}
}
else
{
lean_object* v_a_1723_; lean_object* v___x_1725_; uint8_t v_isShared_1726_; uint8_t v_isSharedCheck_1730_; 
lean_dec(v_a_1708_);
lean_del_object(v___x_1705_);
lean_dec(v_tailOff_1703_);
lean_dec(v_size_1701_);
v_a_1723_ = lean_ctor_get(v___x_1711_, 0);
v_isSharedCheck_1730_ = !lean_is_exclusive(v___x_1711_);
if (v_isSharedCheck_1730_ == 0)
{
v___x_1725_ = v___x_1711_;
v_isShared_1726_ = v_isSharedCheck_1730_;
goto v_resetjp_1724_;
}
else
{
lean_inc(v_a_1723_);
lean_dec(v___x_1711_);
v___x_1725_ = lean_box(0);
v_isShared_1726_ = v_isSharedCheck_1730_;
goto v_resetjp_1724_;
}
v_resetjp_1724_:
{
lean_object* v___x_1728_; 
if (v_isShared_1726_ == 0)
{
v___x_1728_ = v___x_1725_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v_a_1723_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
}
}
else
{
lean_object* v_a_1731_; lean_object* v___x_1733_; uint8_t v_isShared_1734_; uint8_t v_isSharedCheck_1738_; 
lean_del_object(v___x_1705_);
lean_dec(v_tailOff_1703_);
lean_dec(v_size_1701_);
lean_dec_ref(v_tail_1700_);
v_a_1731_ = lean_ctor_get(v___x_1707_, 0);
v_isSharedCheck_1738_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1738_ == 0)
{
v___x_1733_ = v___x_1707_;
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
else
{
lean_inc(v_a_1731_);
lean_dec(v___x_1707_);
v___x_1733_ = lean_box(0);
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
v_resetjp_1732_:
{
lean_object* v___x_1736_; 
if (v_isShared_1734_ == 0)
{
v___x_1736_ = v___x_1733_;
goto v_reusejp_1735_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v_a_1731_);
v___x_1736_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1735_;
}
v_reusejp_1735_:
{
return v___x_1736_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1___boxed(lean_object* v___x_1740_, lean_object* v_t_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_){
_start:
{
lean_object* v_res_1751_; 
v_res_1751_ = lp_mathlib_Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1(v___x_1740_, v_t_1741_, v___y_1742_, v___y_1743_, v___y_1744_, v___y_1745_, v___y_1746_, v___y_1747_, v___y_1748_, v___y_1749_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v___y_1747_);
lean_dec_ref(v___y_1746_);
lean_dec(v___y_1745_);
lean_dec_ref(v___y_1744_);
lean_dec(v___y_1743_);
lean_dec_ref(v___y_1742_);
lean_dec_ref(v___x_1740_);
return v_res_1751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___lam__0(lean_object* v_a_1752_, lean_object* v_a_1753_, lean_object* v_a_1754_, lean_object* v_a_1755_, lean_object* v_a_1756_, lean_object* v_a_1757_, lean_object* v_a_1758_, lean_object* v_a_1759_, lean_object* v_trees_1760_, lean_object* v_messages_1761_, lean_object* v_a_x3f_1762_){
_start:
{
lean_object* v___x_1764_; 
v___x_1764_ = l_Lean_Core_getMessageLog___redArg(v_a_1752_);
if (lean_obj_tag(v___x_1764_) == 0)
{
lean_object* v_a_1765_; lean_object* v___x_1766_; lean_object* v_infoState_1767_; lean_object* v_trees_1768_; lean_object* v___x_1769_; 
v_a_1765_ = lean_ctor_get(v___x_1764_, 0);
lean_inc(v_a_1765_);
lean_dec_ref_known(v___x_1764_, 1);
v___x_1766_ = lean_st_ref_get(v_a_1752_);
v_infoState_1767_ = lean_ctor_get(v___x_1766_, 7);
lean_inc_ref(v_infoState_1767_);
lean_dec(v___x_1766_);
v_trees_1768_ = lean_ctor_get(v_infoState_1767_, 2);
lean_inc_ref(v_trees_1768_);
v___x_1769_ = lp_mathlib_Lean_PersistentArray_mapM___at___00Mathlib_Tactic_withResetServerInfo_spec__1(v_infoState_1767_, v_trees_1768_, v_a_1753_, v_a_1754_, v_a_1755_, v_a_1756_, v_a_1757_, v_a_1758_, v_a_1759_, v_a_1752_);
lean_dec_ref(v_infoState_1767_);
if (lean_obj_tag(v___x_1769_) == 0)
{
lean_object* v_a_1770_; lean_object* v___x_1772_; uint8_t v_isShared_1773_; uint8_t v_isSharedCheck_1807_; 
v_a_1770_ = lean_ctor_get(v___x_1769_, 0);
v_isSharedCheck_1807_ = !lean_is_exclusive(v___x_1769_);
if (v_isSharedCheck_1807_ == 0)
{
v___x_1772_ = v___x_1769_;
v_isShared_1773_ = v_isSharedCheck_1807_;
goto v_resetjp_1771_;
}
else
{
lean_inc(v_a_1770_);
lean_dec(v___x_1769_);
v___x_1772_ = lean_box(0);
v_isShared_1773_ = v_isSharedCheck_1807_;
goto v_resetjp_1771_;
}
v_resetjp_1771_:
{
lean_object* v___x_1774_; lean_object* v_infoState_1775_; lean_object* v_env_1776_; lean_object* v_nextMacroScope_1777_; lean_object* v_ngen_1778_; lean_object* v_auxDeclNGen_1779_; lean_object* v_traceState_1780_; lean_object* v_cache_1781_; lean_object* v_snapshotTasks_1782_; lean_object* v___x_1784_; uint8_t v_isShared_1785_; uint8_t v_isSharedCheck_1805_; 
v___x_1774_ = lean_st_ref_take(v_a_1752_);
v_infoState_1775_ = lean_ctor_get(v___x_1774_, 7);
v_env_1776_ = lean_ctor_get(v___x_1774_, 0);
v_nextMacroScope_1777_ = lean_ctor_get(v___x_1774_, 1);
v_ngen_1778_ = lean_ctor_get(v___x_1774_, 2);
v_auxDeclNGen_1779_ = lean_ctor_get(v___x_1774_, 3);
v_traceState_1780_ = lean_ctor_get(v___x_1774_, 4);
v_cache_1781_ = lean_ctor_get(v___x_1774_, 5);
v_snapshotTasks_1782_ = lean_ctor_get(v___x_1774_, 8);
v_isSharedCheck_1805_ = !lean_is_exclusive(v___x_1774_);
if (v_isSharedCheck_1805_ == 0)
{
lean_object* v_unused_1806_; 
v_unused_1806_ = lean_ctor_get(v___x_1774_, 6);
lean_dec(v_unused_1806_);
v___x_1784_ = v___x_1774_;
v_isShared_1785_ = v_isSharedCheck_1805_;
goto v_resetjp_1783_;
}
else
{
lean_inc(v_snapshotTasks_1782_);
lean_inc(v_infoState_1775_);
lean_inc(v_cache_1781_);
lean_inc(v_traceState_1780_);
lean_inc(v_auxDeclNGen_1779_);
lean_inc(v_ngen_1778_);
lean_inc(v_nextMacroScope_1777_);
lean_inc(v_env_1776_);
lean_dec(v___x_1774_);
v___x_1784_ = lean_box(0);
v_isShared_1785_ = v_isSharedCheck_1805_;
goto v_resetjp_1783_;
}
v_resetjp_1783_:
{
uint8_t v_enabled_1786_; lean_object* v_assignment_1787_; lean_object* v_lazyAssignment_1788_; lean_object* v___x_1790_; uint8_t v_isShared_1791_; uint8_t v_isSharedCheck_1803_; 
v_enabled_1786_ = lean_ctor_get_uint8(v_infoState_1775_, sizeof(void*)*3);
v_assignment_1787_ = lean_ctor_get(v_infoState_1775_, 0);
v_lazyAssignment_1788_ = lean_ctor_get(v_infoState_1775_, 1);
v_isSharedCheck_1803_ = !lean_is_exclusive(v_infoState_1775_);
if (v_isSharedCheck_1803_ == 0)
{
lean_object* v_unused_1804_; 
v_unused_1804_ = lean_ctor_get(v_infoState_1775_, 2);
lean_dec(v_unused_1804_);
v___x_1790_ = v_infoState_1775_;
v_isShared_1791_ = v_isSharedCheck_1803_;
goto v_resetjp_1789_;
}
else
{
lean_inc(v_lazyAssignment_1788_);
lean_inc(v_assignment_1787_);
lean_dec(v_infoState_1775_);
v___x_1790_ = lean_box(0);
v_isShared_1791_ = v_isSharedCheck_1803_;
goto v_resetjp_1789_;
}
v_resetjp_1789_:
{
lean_object* v___x_1793_; 
if (v_isShared_1791_ == 0)
{
lean_ctor_set(v___x_1790_, 2, v_trees_1760_);
v___x_1793_ = v___x_1790_;
goto v_reusejp_1792_;
}
else
{
lean_object* v_reuseFailAlloc_1802_; 
v_reuseFailAlloc_1802_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_1802_, 0, v_assignment_1787_);
lean_ctor_set(v_reuseFailAlloc_1802_, 1, v_lazyAssignment_1788_);
lean_ctor_set(v_reuseFailAlloc_1802_, 2, v_trees_1760_);
lean_ctor_set_uint8(v_reuseFailAlloc_1802_, sizeof(void*)*3, v_enabled_1786_);
v___x_1793_ = v_reuseFailAlloc_1802_;
goto v_reusejp_1792_;
}
v_reusejp_1792_:
{
lean_object* v___x_1795_; 
if (v_isShared_1785_ == 0)
{
lean_ctor_set(v___x_1784_, 7, v___x_1793_);
lean_ctor_set(v___x_1784_, 6, v_messages_1761_);
v___x_1795_ = v___x_1784_;
goto v_reusejp_1794_;
}
else
{
lean_object* v_reuseFailAlloc_1801_; 
v_reuseFailAlloc_1801_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1801_, 0, v_env_1776_);
lean_ctor_set(v_reuseFailAlloc_1801_, 1, v_nextMacroScope_1777_);
lean_ctor_set(v_reuseFailAlloc_1801_, 2, v_ngen_1778_);
lean_ctor_set(v_reuseFailAlloc_1801_, 3, v_auxDeclNGen_1779_);
lean_ctor_set(v_reuseFailAlloc_1801_, 4, v_traceState_1780_);
lean_ctor_set(v_reuseFailAlloc_1801_, 5, v_cache_1781_);
lean_ctor_set(v_reuseFailAlloc_1801_, 6, v_messages_1761_);
lean_ctor_set(v_reuseFailAlloc_1801_, 7, v___x_1793_);
lean_ctor_set(v_reuseFailAlloc_1801_, 8, v_snapshotTasks_1782_);
v___x_1795_ = v_reuseFailAlloc_1801_;
goto v_reusejp_1794_;
}
v_reusejp_1794_:
{
lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1799_; 
v___x_1796_ = lean_st_ref_set(v_a_1752_, v___x_1795_);
v___x_1797_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1797_, 0, v_a_x3f_1762_);
lean_ctor_set(v___x_1797_, 1, v_a_1765_);
lean_ctor_set(v___x_1797_, 2, v_a_1770_);
if (v_isShared_1773_ == 0)
{
lean_ctor_set(v___x_1772_, 0, v___x_1797_);
v___x_1799_ = v___x_1772_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1800_; 
v_reuseFailAlloc_1800_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1800_, 0, v___x_1797_);
v___x_1799_ = v_reuseFailAlloc_1800_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
return v___x_1799_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1808_; lean_object* v___x_1810_; uint8_t v_isShared_1811_; uint8_t v_isSharedCheck_1815_; 
lean_dec(v_a_1765_);
lean_dec(v_a_x3f_1762_);
lean_dec_ref(v_messages_1761_);
lean_dec_ref(v_trees_1760_);
v_a_1808_ = lean_ctor_get(v___x_1769_, 0);
v_isSharedCheck_1815_ = !lean_is_exclusive(v___x_1769_);
if (v_isSharedCheck_1815_ == 0)
{
v___x_1810_ = v___x_1769_;
v_isShared_1811_ = v_isSharedCheck_1815_;
goto v_resetjp_1809_;
}
else
{
lean_inc(v_a_1808_);
lean_dec(v___x_1769_);
v___x_1810_ = lean_box(0);
v_isShared_1811_ = v_isSharedCheck_1815_;
goto v_resetjp_1809_;
}
v_resetjp_1809_:
{
lean_object* v___x_1813_; 
if (v_isShared_1811_ == 0)
{
v___x_1813_ = v___x_1810_;
goto v_reusejp_1812_;
}
else
{
lean_object* v_reuseFailAlloc_1814_; 
v_reuseFailAlloc_1814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1814_, 0, v_a_1808_);
v___x_1813_ = v_reuseFailAlloc_1814_;
goto v_reusejp_1812_;
}
v_reusejp_1812_:
{
return v___x_1813_;
}
}
}
}
else
{
lean_object* v_a_1816_; lean_object* v___x_1818_; uint8_t v_isShared_1819_; uint8_t v_isSharedCheck_1823_; 
lean_dec(v_a_x3f_1762_);
lean_dec_ref(v_messages_1761_);
lean_dec_ref(v_trees_1760_);
v_a_1816_ = lean_ctor_get(v___x_1764_, 0);
v_isSharedCheck_1823_ = !lean_is_exclusive(v___x_1764_);
if (v_isSharedCheck_1823_ == 0)
{
v___x_1818_ = v___x_1764_;
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
else
{
lean_inc(v_a_1816_);
lean_dec(v___x_1764_);
v___x_1818_ = lean_box(0);
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
v_resetjp_1817_:
{
lean_object* v___x_1821_; 
if (v_isShared_1819_ == 0)
{
v___x_1821_ = v___x_1818_;
goto v_reusejp_1820_;
}
else
{
lean_object* v_reuseFailAlloc_1822_; 
v_reuseFailAlloc_1822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1822_, 0, v_a_1816_);
v___x_1821_ = v_reuseFailAlloc_1822_;
goto v_reusejp_1820_;
}
v_reusejp_1820_:
{
return v___x_1821_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___lam__0___boxed(lean_object* v_a_1824_, lean_object* v_a_1825_, lean_object* v_a_1826_, lean_object* v_a_1827_, lean_object* v_a_1828_, lean_object* v_a_1829_, lean_object* v_a_1830_, lean_object* v_a_1831_, lean_object* v_trees_1832_, lean_object* v_messages_1833_, lean_object* v_a_x3f_1834_, lean_object* v___y_1835_){
_start:
{
lean_object* v_res_1836_; 
v_res_1836_ = lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___lam__0(v_a_1824_, v_a_1825_, v_a_1826_, v_a_1827_, v_a_1828_, v_a_1829_, v_a_1830_, v_a_1831_, v_trees_1832_, v_messages_1833_, v_a_x3f_1834_);
lean_dec_ref(v_a_1831_);
lean_dec(v_a_1830_);
lean_dec_ref(v_a_1829_);
lean_dec(v_a_1828_);
lean_dec_ref(v_a_1827_);
lean_dec(v_a_1826_);
lean_dec_ref(v_a_1825_);
lean_dec(v_a_1824_);
return v_res_1836_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__0(void){
_start:
{
lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; 
v___x_1837_ = lean_unsigned_to_nat(32u);
v___x_1838_ = lean_mk_empty_array_with_capacity(v___x_1837_);
v___x_1839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1839_, 0, v___x_1838_);
return v___x_1839_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1(void){
_start:
{
size_t v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; 
v___x_1840_ = ((size_t)5ULL);
v___x_1841_ = lean_unsigned_to_nat(0u);
v___x_1842_ = lean_unsigned_to_nat(32u);
v___x_1843_ = lean_mk_empty_array_with_capacity(v___x_1842_);
v___x_1844_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__0, &lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__0);
v___x_1845_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1845_, 0, v___x_1844_);
lean_ctor_set(v___x_1845_, 1, v___x_1843_);
lean_ctor_set(v___x_1845_, 2, v___x_1841_);
lean_ctor_set(v___x_1845_, 3, v___x_1841_);
lean_ctor_set_usize(v___x_1845_, 4, v___x_1840_);
return v___x_1845_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__2(void){
_start:
{
lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; 
v___x_1846_ = l_Lean_NameSet_empty;
v___x_1847_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1);
v___x_1848_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1848_, 0, v___x_1847_);
lean_ctor_set(v___x_1848_, 1, v___x_1847_);
lean_ctor_set(v___x_1848_, 2, v___x_1846_);
return v___x_1848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg(lean_object* v_t_1849_, lean_object* v_a_1850_, lean_object* v_a_1851_, lean_object* v_a_1852_, lean_object* v_a_1853_, lean_object* v_a_1854_, lean_object* v_a_1855_, lean_object* v_a_1856_, lean_object* v_a_1857_){
_start:
{
lean_object* v___x_1859_; lean_object* v_env_1860_; lean_object* v_nextMacroScope_1861_; lean_object* v_ngen_1862_; lean_object* v_auxDeclNGen_1863_; lean_object* v_traceState_1864_; lean_object* v_cache_1865_; lean_object* v_messages_1866_; lean_object* v_infoState_1867_; lean_object* v_snapshotTasks_1868_; lean_object* v___x_1870_; uint8_t v_isShared_1871_; uint8_t v_isSharedCheck_1910_; 
v___x_1859_ = lean_st_ref_take(v_a_1857_);
v_env_1860_ = lean_ctor_get(v___x_1859_, 0);
v_nextMacroScope_1861_ = lean_ctor_get(v___x_1859_, 1);
v_ngen_1862_ = lean_ctor_get(v___x_1859_, 2);
v_auxDeclNGen_1863_ = lean_ctor_get(v___x_1859_, 3);
v_traceState_1864_ = lean_ctor_get(v___x_1859_, 4);
v_cache_1865_ = lean_ctor_get(v___x_1859_, 5);
v_messages_1866_ = lean_ctor_get(v___x_1859_, 6);
v_infoState_1867_ = lean_ctor_get(v___x_1859_, 7);
v_snapshotTasks_1868_ = lean_ctor_get(v___x_1859_, 8);
v_isSharedCheck_1910_ = !lean_is_exclusive(v___x_1859_);
if (v_isSharedCheck_1910_ == 0)
{
v___x_1870_ = v___x_1859_;
v_isShared_1871_ = v_isSharedCheck_1910_;
goto v_resetjp_1869_;
}
else
{
lean_inc(v_snapshotTasks_1868_);
lean_inc(v_infoState_1867_);
lean_inc(v_messages_1866_);
lean_inc(v_cache_1865_);
lean_inc(v_traceState_1864_);
lean_inc(v_auxDeclNGen_1863_);
lean_inc(v_ngen_1862_);
lean_inc(v_nextMacroScope_1861_);
lean_inc(v_env_1860_);
lean_dec(v___x_1859_);
v___x_1870_ = lean_box(0);
v_isShared_1871_ = v_isSharedCheck_1910_;
goto v_resetjp_1869_;
}
v_resetjp_1869_:
{
lean_object* v___x_1872_; lean_object* v___x_1873_; uint8_t v_enabled_1874_; lean_object* v_assignment_1875_; lean_object* v_lazyAssignment_1876_; lean_object* v_trees_1877_; lean_object* v___x_1879_; uint8_t v_isShared_1880_; uint8_t v_isSharedCheck_1909_; 
v___x_1872_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__1);
v___x_1873_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__2, &lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___closed__2);
v_enabled_1874_ = lean_ctor_get_uint8(v_infoState_1867_, sizeof(void*)*3);
v_assignment_1875_ = lean_ctor_get(v_infoState_1867_, 0);
v_lazyAssignment_1876_ = lean_ctor_get(v_infoState_1867_, 1);
v_trees_1877_ = lean_ctor_get(v_infoState_1867_, 2);
v_isSharedCheck_1909_ = !lean_is_exclusive(v_infoState_1867_);
if (v_isSharedCheck_1909_ == 0)
{
v___x_1879_ = v_infoState_1867_;
v_isShared_1880_ = v_isSharedCheck_1909_;
goto v_resetjp_1878_;
}
else
{
lean_inc(v_trees_1877_);
lean_inc(v_lazyAssignment_1876_);
lean_inc(v_assignment_1875_);
lean_dec(v_infoState_1867_);
v___x_1879_ = lean_box(0);
v_isShared_1880_ = v_isSharedCheck_1909_;
goto v_resetjp_1878_;
}
v_resetjp_1878_:
{
lean_object* v___x_1882_; 
if (v_isShared_1880_ == 0)
{
lean_ctor_set(v___x_1879_, 2, v___x_1872_);
v___x_1882_ = v___x_1879_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1908_; 
v_reuseFailAlloc_1908_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_1908_, 0, v_assignment_1875_);
lean_ctor_set(v_reuseFailAlloc_1908_, 1, v_lazyAssignment_1876_);
lean_ctor_set(v_reuseFailAlloc_1908_, 2, v___x_1872_);
lean_ctor_set_uint8(v_reuseFailAlloc_1908_, sizeof(void*)*3, v_enabled_1874_);
v___x_1882_ = v_reuseFailAlloc_1908_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
lean_object* v___x_1884_; 
if (v_isShared_1871_ == 0)
{
lean_ctor_set(v___x_1870_, 7, v___x_1882_);
lean_ctor_set(v___x_1870_, 6, v___x_1873_);
v___x_1884_ = v___x_1870_;
goto v_reusejp_1883_;
}
else
{
lean_object* v_reuseFailAlloc_1907_; 
v_reuseFailAlloc_1907_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1907_, 0, v_env_1860_);
lean_ctor_set(v_reuseFailAlloc_1907_, 1, v_nextMacroScope_1861_);
lean_ctor_set(v_reuseFailAlloc_1907_, 2, v_ngen_1862_);
lean_ctor_set(v_reuseFailAlloc_1907_, 3, v_auxDeclNGen_1863_);
lean_ctor_set(v_reuseFailAlloc_1907_, 4, v_traceState_1864_);
lean_ctor_set(v_reuseFailAlloc_1907_, 5, v_cache_1865_);
lean_ctor_set(v_reuseFailAlloc_1907_, 6, v___x_1873_);
lean_ctor_set(v_reuseFailAlloc_1907_, 7, v___x_1882_);
lean_ctor_set(v_reuseFailAlloc_1907_, 8, v_snapshotTasks_1868_);
v___x_1884_ = v_reuseFailAlloc_1907_;
goto v_reusejp_1883_;
}
v_reusejp_1883_:
{
lean_object* v___x_1885_; lean_object* v_r_1886_; 
v___x_1885_ = lean_st_ref_set(v_a_1857_, v___x_1884_);
lean_inc(v_a_1857_);
lean_inc_ref(v_a_1856_);
lean_inc(v_a_1855_);
lean_inc_ref(v_a_1854_);
lean_inc(v_a_1853_);
lean_inc_ref(v_a_1852_);
lean_inc(v_a_1851_);
lean_inc_ref(v_a_1850_);
v_r_1886_ = lean_apply_9(v_t_1849_, v_a_1850_, v_a_1851_, v_a_1852_, v_a_1853_, v_a_1854_, v_a_1855_, v_a_1856_, v_a_1857_, lean_box(0));
if (lean_obj_tag(v_r_1886_) == 0)
{
lean_object* v_a_1887_; lean_object* v___x_1889_; uint8_t v_isShared_1890_; uint8_t v_isSharedCheck_1895_; 
v_a_1887_ = lean_ctor_get(v_r_1886_, 0);
v_isSharedCheck_1895_ = !lean_is_exclusive(v_r_1886_);
if (v_isSharedCheck_1895_ == 0)
{
v___x_1889_ = v_r_1886_;
v_isShared_1890_ = v_isSharedCheck_1895_;
goto v_resetjp_1888_;
}
else
{
lean_inc(v_a_1887_);
lean_dec(v_r_1886_);
v___x_1889_ = lean_box(0);
v_isShared_1890_ = v_isSharedCheck_1895_;
goto v_resetjp_1888_;
}
v_resetjp_1888_:
{
lean_object* v___x_1892_; 
if (v_isShared_1890_ == 0)
{
lean_ctor_set_tag(v___x_1889_, 1);
v___x_1892_ = v___x_1889_;
goto v_reusejp_1891_;
}
else
{
lean_object* v_reuseFailAlloc_1894_; 
v_reuseFailAlloc_1894_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1894_, 0, v_a_1887_);
v___x_1892_ = v_reuseFailAlloc_1894_;
goto v_reusejp_1891_;
}
v_reusejp_1891_:
{
lean_object* v___x_1893_; 
v___x_1893_ = lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___lam__0(v_a_1857_, v_a_1850_, v_a_1851_, v_a_1852_, v_a_1853_, v_a_1854_, v_a_1855_, v_a_1856_, v_trees_1877_, v_messages_1866_, v___x_1892_);
return v___x_1893_;
}
}
}
else
{
lean_object* v_a_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; 
v_a_1896_ = lean_ctor_get(v_r_1886_, 0);
lean_inc(v_a_1896_);
lean_dec_ref_known(v_r_1886_, 1);
v___x_1897_ = lean_box(0);
v___x_1898_ = lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___lam__0(v_a_1857_, v_a_1850_, v_a_1851_, v_a_1852_, v_a_1853_, v_a_1854_, v_a_1855_, v_a_1856_, v_trees_1877_, v_messages_1866_, v___x_1897_);
if (lean_obj_tag(v___x_1898_) == 0)
{
lean_object* v___x_1900_; uint8_t v_isShared_1901_; uint8_t v_isSharedCheck_1905_; 
v_isSharedCheck_1905_ = !lean_is_exclusive(v___x_1898_);
if (v_isSharedCheck_1905_ == 0)
{
lean_object* v_unused_1906_; 
v_unused_1906_ = lean_ctor_get(v___x_1898_, 0);
lean_dec(v_unused_1906_);
v___x_1900_ = v___x_1898_;
v_isShared_1901_ = v_isSharedCheck_1905_;
goto v_resetjp_1899_;
}
else
{
lean_dec(v___x_1898_);
v___x_1900_ = lean_box(0);
v_isShared_1901_ = v_isSharedCheck_1905_;
goto v_resetjp_1899_;
}
v_resetjp_1899_:
{
lean_object* v___x_1903_; 
if (v_isShared_1901_ == 0)
{
lean_ctor_set_tag(v___x_1900_, 1);
lean_ctor_set(v___x_1900_, 0, v_a_1896_);
v___x_1903_ = v___x_1900_;
goto v_reusejp_1902_;
}
else
{
lean_object* v_reuseFailAlloc_1904_; 
v_reuseFailAlloc_1904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1904_, 0, v_a_1896_);
v___x_1903_ = v_reuseFailAlloc_1904_;
goto v_reusejp_1902_;
}
v_reusejp_1902_:
{
return v___x_1903_;
}
}
}
else
{
lean_dec(v_a_1896_);
return v___x_1898_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg___boxed(lean_object* v_t_1911_, lean_object* v_a_1912_, lean_object* v_a_1913_, lean_object* v_a_1914_, lean_object* v_a_1915_, lean_object* v_a_1916_, lean_object* v_a_1917_, lean_object* v_a_1918_, lean_object* v_a_1919_, lean_object* v_a_1920_){
_start:
{
lean_object* v_res_1921_; 
v_res_1921_ = lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg(v_t_1911_, v_a_1912_, v_a_1913_, v_a_1914_, v_a_1915_, v_a_1916_, v_a_1917_, v_a_1918_, v_a_1919_);
lean_dec(v_a_1919_);
lean_dec_ref(v_a_1918_);
lean_dec(v_a_1917_);
lean_dec_ref(v_a_1916_);
lean_dec(v_a_1915_);
lean_dec_ref(v_a_1914_);
lean_dec(v_a_1913_);
lean_dec_ref(v_a_1912_);
return v_res_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo(lean_object* v_00_u03b1_1922_, lean_object* v_t_1923_, lean_object* v_a_1924_, lean_object* v_a_1925_, lean_object* v_a_1926_, lean_object* v_a_1927_, lean_object* v_a_1928_, lean_object* v_a_1929_, lean_object* v_a_1930_, lean_object* v_a_1931_){
_start:
{
lean_object* v___x_1933_; 
v___x_1933_ = lp_mathlib_Mathlib_Tactic_withResetServerInfo___redArg(v_t_1923_, v_a_1924_, v_a_1925_, v_a_1926_, v_a_1927_, v_a_1928_, v_a_1929_, v_a_1930_, v_a_1931_);
return v___x_1933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_withResetServerInfo___boxed(lean_object* v_00_u03b1_1934_, lean_object* v_t_1935_, lean_object* v_a_1936_, lean_object* v_a_1937_, lean_object* v_a_1938_, lean_object* v_a_1939_, lean_object* v_a_1940_, lean_object* v_a_1941_, lean_object* v_a_1942_, lean_object* v_a_1943_, lean_object* v_a_1944_){
_start:
{
lean_object* v_res_1945_; 
v_res_1945_ = lp_mathlib_Mathlib_Tactic_withResetServerInfo(v_00_u03b1_1934_, v_t_1935_, v_a_1936_, v_a_1937_, v_a_1938_, v_a_1939_, v_a_1940_, v_a_1941_, v_a_1942_, v_a_1943_);
lean_dec(v_a_1943_);
lean_dec_ref(v_a_1942_);
lean_dec(v_a_1941_);
lean_dec_ref(v_a_1940_);
lean_dec(v_a_1939_);
lean_dec_ref(v_a_1938_);
lean_dec(v_a_1937_);
lean_dec_ref(v_a_1936_);
return v_res_1945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0(lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_){
_start:
{
lean_object* v___x_1955_; 
v___x_1955_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___redArg(v___y_1951_, v___y_1952_, v___y_1953_);
return v___x_1955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0___boxed(lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_){
_start:
{
lean_object* v_res_1965_; 
v_res_1965_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Mathlib_Tactic_withResetServerInfo_spec__0_spec__0(v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_, v___y_1961_, v___y_1962_, v___y_1963_);
lean_dec(v___y_1963_);
lean_dec_ref(v___y_1962_);
lean_dec(v___y_1961_);
lean_dec_ref(v___y_1960_);
lean_dec(v___y_1959_);
lean_dec_ref(v___y_1958_);
lean_dec(v___y_1957_);
lean_dec_ref(v___y_1956_);
return v_res_1965_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ExtendDoc(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ExtendDoc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_BuiltinCommand(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_BuiltinCommand(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_introv = _init_lp_mathlib_Mathlib_Tactic_introv();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_introv);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_BuiltinCommand(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ExtendDoc(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_BuiltinCommand(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ExtendDoc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
