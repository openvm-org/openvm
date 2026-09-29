// Lean compiler output
// Module: Aesop.Script.Main
// Imports: public import Init public meta import Init public import Aesop.Script.Check public import Aesop.Stats.Basic public import Aesop.Options.Internal import Batteries.Lean.Meta.SavedState import Aesop.Script.OptimizeSyntax import Aesop.Script.StructureDynamic import Aesop.Script.StructureStatic
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
lean_object* l_Lean_logWarning___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Check_name(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_aesop_Aesop_Script_Step_sTactic_x3f(lean_object*);
lean_object* lp_aesop_Aesop_Script_SScript_takeNConsecutiveFocusAndSolve_x3f(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lp_aesop_Aesop_Script_mkOnGoal(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_Step_uTactic(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Syntax_mkNumLit(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_mkSepArray(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_mkInitial___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_UScript_toSScriptStatic(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_UScript_toSScriptDynamic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_dynamicStructuring;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_optimizedDynamicStructuring;
extern lean_object* lp_aesop_Aesop_Check_script;
lean_object* lp_aesop_Aesop_Check_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_checkRenderedScriptIfEnabled___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_addTryThisTacticSeqSuggestion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_recordScriptGenerated___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_UScript_renderTacticSeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureStatic(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureStatic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureDynamic(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureDynamic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Script_UScript_optimize_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Script_UScript_optimize_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__1_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__2_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__2_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__3_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(238, 151, 138, 49, 249, 18, 254, 242)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__1 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__1_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__2 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(117, 126, 44, 217, 38, 3, 69, 145)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__3 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__3_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__6 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__6_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__6_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__8 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__8_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeqBracketed"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__10 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(142, 80, 121, 250, 245, 54, 71, 145)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "renameI"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__12 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__12_value),LEAN_SCALAR_PTR_LITERAL(20, 41, 101, 89, 107, 117, 242, 244)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticNext_=>_"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__15 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__15_value),LEAN_SCALAR_PTR_LITERAL(90, 21, 53, 2, 17, 158, 67, 66)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "next"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__17 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__17_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__18 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__18_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19_value;
static lean_once_cell_t lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__21 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__21_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__6(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__10___redArg(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__0;
static lean_once_cell_t lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__1;
static const lean_array_object lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__2 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__2_value;
static const lean_string_object lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rename_i"};
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__3 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticOn_goal-_=>_"};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(243, 56, 227, 189, 147, 207, 104, 76)}};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "on_goal"};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__4_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__6_value;
static const lean_array_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__7_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19_value),((lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__7_value)}};
static const lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__8_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_optimize(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_optimize___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = ": structuring the script failed. Reporting unstructured script."};
static const lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__0 = (const lean_object*)&lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__1;
static const lean_string_object lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = ": structuring the script failed"};
static const lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__2 = (const lean_object*)&lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__5(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureStatic(lean_object* v_uscript_1_, uint8_t v_proofHasMVar_2_, lean_object* v_preState_3_, lean_object* v_goal_4_, lean_object* v_a_5_, lean_object* v_a_6_, lean_object* v_a_7_, lean_object* v_a_8_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_mkInitial___boxed), 6, 1);
lean_closure_set(v___x_10_, 0, v_goal_4_);
v___x_11_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_preState_3_, v___x_10_, v_a_5_, v_a_6_, v_a_7_, v_a_8_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v_a_12_; lean_object* v___x_13_; 
v_a_12_ = lean_ctor_get(v___x_11_, 0);
lean_inc(v_a_12_);
lean_dec_ref_known(v___x_11_, 1);
v___x_13_ = lp_aesop_Aesop_Script_UScript_toSScriptStatic(v_a_12_, v_uscript_1_, v_a_7_, v_a_8_);
if (lean_obj_tag(v___x_13_) == 0)
{
lean_object* v_a_14_; lean_object* v___x_16_; uint8_t v_isShared_17_; uint8_t v_isSharedCheck_34_; 
v_a_14_ = lean_ctor_get(v___x_13_, 0);
v_isSharedCheck_34_ = !lean_is_exclusive(v___x_13_);
if (v_isSharedCheck_34_ == 0)
{
v___x_16_ = v___x_13_;
v_isShared_17_ = v_isSharedCheck_34_;
goto v_resetjp_15_;
}
else
{
lean_inc(v_a_14_);
lean_dec(v___x_13_);
v___x_16_ = lean_box(0);
v_isShared_17_ = v_isSharedCheck_34_;
goto v_resetjp_15_;
}
v_resetjp_15_:
{
lean_object* v_fst_18_; lean_object* v_snd_19_; lean_object* v___x_21_; uint8_t v_isShared_22_; uint8_t v_isSharedCheck_33_; 
v_fst_18_ = lean_ctor_get(v_a_14_, 0);
v_snd_19_ = lean_ctor_get(v_a_14_, 1);
v_isSharedCheck_33_ = !lean_is_exclusive(v_a_14_);
if (v_isSharedCheck_33_ == 0)
{
v___x_21_ = v_a_14_;
v_isShared_22_ = v_isSharedCheck_33_;
goto v_resetjp_20_;
}
else
{
lean_inc(v_snd_19_);
lean_inc(v_fst_18_);
lean_dec(v_a_14_);
v___x_21_ = lean_box(0);
v_isShared_22_ = v_isSharedCheck_33_;
goto v_resetjp_20_;
}
v_resetjp_20_:
{
uint8_t v___x_23_; lean_object* v___x_24_; uint8_t v___x_25_; lean_object* v___x_27_; 
v___x_23_ = 0;
v___x_24_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v___x_24_, 0, v___x_23_);
v___x_25_ = lean_unbox(v_snd_19_);
lean_dec(v_snd_19_);
lean_ctor_set_uint8(v___x_24_, 1, v___x_25_);
lean_ctor_set_uint8(v___x_24_, 2, v_proofHasMVar_2_);
if (v_isShared_22_ == 0)
{
lean_ctor_set(v___x_21_, 1, v___x_24_);
v___x_27_ = v___x_21_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_32_; 
v_reuseFailAlloc_32_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_32_, 0, v_fst_18_);
lean_ctor_set(v_reuseFailAlloc_32_, 1, v___x_24_);
v___x_27_ = v_reuseFailAlloc_32_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
lean_object* v___x_28_; lean_object* v___x_30_; 
v___x_28_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_28_, 0, v___x_27_);
if (v_isShared_17_ == 0)
{
lean_ctor_set(v___x_16_, 0, v___x_28_);
v___x_30_ = v___x_16_;
goto v_reusejp_29_;
}
else
{
lean_object* v_reuseFailAlloc_31_; 
v_reuseFailAlloc_31_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_31_, 0, v___x_28_);
v___x_30_ = v_reuseFailAlloc_31_;
goto v_reusejp_29_;
}
v_reusejp_29_:
{
return v___x_30_;
}
}
}
}
}
else
{
lean_object* v_a_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_42_; 
v_a_35_ = lean_ctor_get(v___x_13_, 0);
v_isSharedCheck_42_ = !lean_is_exclusive(v___x_13_);
if (v_isSharedCheck_42_ == 0)
{
v___x_37_ = v___x_13_;
v_isShared_38_ = v_isSharedCheck_42_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_a_35_);
lean_dec(v___x_13_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_42_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_40_; 
if (v_isShared_38_ == 0)
{
v___x_40_ = v___x_37_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v_a_35_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
}
}
else
{
lean_object* v_a_43_; lean_object* v___x_45_; uint8_t v_isShared_46_; uint8_t v_isSharedCheck_50_; 
lean_dec_ref(v_uscript_1_);
v_a_43_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_50_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_50_ == 0)
{
v___x_45_ = v___x_11_;
v_isShared_46_ = v_isSharedCheck_50_;
goto v_resetjp_44_;
}
else
{
lean_inc(v_a_43_);
lean_dec(v___x_11_);
v___x_45_ = lean_box(0);
v_isShared_46_ = v_isSharedCheck_50_;
goto v_resetjp_44_;
}
v_resetjp_44_:
{
lean_object* v___x_48_; 
if (v_isShared_46_ == 0)
{
v___x_48_ = v___x_45_;
goto v_reusejp_47_;
}
else
{
lean_object* v_reuseFailAlloc_49_; 
v_reuseFailAlloc_49_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_49_, 0, v_a_43_);
v___x_48_ = v_reuseFailAlloc_49_;
goto v_reusejp_47_;
}
v_reusejp_47_:
{
return v___x_48_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureStatic___boxed(lean_object* v_uscript_51_, lean_object* v_proofHasMVar_52_, lean_object* v_preState_53_, lean_object* v_goal_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
uint8_t v_proofHasMVar_boxed_60_; lean_object* v_res_61_; 
v_proofHasMVar_boxed_60_ = lean_unbox(v_proofHasMVar_52_);
v_res_61_ = lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureStatic(v_uscript_51_, v_proofHasMVar_boxed_60_, v_preState_53_, v_goal_54_, v_a_55_, v_a_56_, v_a_57_, v_a_58_);
lean_dec(v_a_58_);
lean_dec_ref(v_a_57_);
lean_dec(v_a_56_);
lean_dec_ref(v_a_55_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureDynamic(lean_object* v_uscript_62_, uint8_t v_proofHasMVar_63_, lean_object* v_preState_64_, lean_object* v_goal_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_aesop_Aesop_Script_UScript_toSScriptDynamic(v_preState_64_, v_goal_65_, v_uscript_62_, v_a_66_, v_a_67_, v_a_68_, v_a_69_);
if (lean_obj_tag(v___x_71_) == 0)
{
lean_object* v_a_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_103_; 
v_a_72_ = lean_ctor_get(v___x_71_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v___x_71_);
if (v_isSharedCheck_103_ == 0)
{
v___x_74_ = v___x_71_;
v_isShared_75_ = v_isSharedCheck_103_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_a_72_);
lean_dec(v___x_71_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_103_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
if (lean_obj_tag(v_a_72_) == 1)
{
lean_object* v_val_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_98_; 
v_val_76_ = lean_ctor_get(v_a_72_, 0);
v_isSharedCheck_98_ = !lean_is_exclusive(v_a_72_);
if (v_isSharedCheck_98_ == 0)
{
v___x_78_ = v_a_72_;
v_isShared_79_ = v_isSharedCheck_98_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_val_76_);
lean_dec(v_a_72_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_98_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v_fst_80_; lean_object* v_snd_81_; lean_object* v___x_83_; uint8_t v_isShared_84_; uint8_t v_isSharedCheck_97_; 
v_fst_80_ = lean_ctor_get(v_val_76_, 0);
v_snd_81_ = lean_ctor_get(v_val_76_, 1);
v_isSharedCheck_97_ = !lean_is_exclusive(v_val_76_);
if (v_isSharedCheck_97_ == 0)
{
v___x_83_ = v_val_76_;
v_isShared_84_ = v_isSharedCheck_97_;
goto v_resetjp_82_;
}
else
{
lean_inc(v_snd_81_);
lean_inc(v_fst_80_);
lean_dec(v_val_76_);
v___x_83_ = lean_box(0);
v_isShared_84_ = v_isSharedCheck_97_;
goto v_resetjp_82_;
}
v_resetjp_82_:
{
uint8_t v___x_85_; lean_object* v___x_86_; uint8_t v___x_87_; lean_object* v___x_89_; 
v___x_85_ = 1;
v___x_86_ = lean_alloc_ctor(0, 0, 3);
lean_ctor_set_uint8(v___x_86_, 0, v___x_85_);
v___x_87_ = lean_unbox(v_snd_81_);
lean_dec(v_snd_81_);
lean_ctor_set_uint8(v___x_86_, 1, v___x_87_);
lean_ctor_set_uint8(v___x_86_, 2, v_proofHasMVar_63_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 1, v___x_86_);
v___x_89_ = v___x_83_;
goto v_reusejp_88_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_fst_80_);
lean_ctor_set(v_reuseFailAlloc_96_, 1, v___x_86_);
v___x_89_ = v_reuseFailAlloc_96_;
goto v_reusejp_88_;
}
v_reusejp_88_:
{
lean_object* v___x_91_; 
if (v_isShared_79_ == 0)
{
lean_ctor_set(v___x_78_, 0, v___x_89_);
v___x_91_ = v___x_78_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v___x_89_);
v___x_91_ = v_reuseFailAlloc_95_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
lean_object* v___x_93_; 
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v___x_91_);
v___x_93_ = v___x_74_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v___x_91_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
}
}
}
else
{
lean_object* v___x_99_; lean_object* v___x_101_; 
lean_dec(v_a_72_);
v___x_99_ = lean_box(0);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v___x_99_);
v___x_101_ = v___x_74_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v___x_99_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
return v___x_101_;
}
}
}
}
else
{
lean_object* v_a_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_111_; 
v_a_104_ = lean_ctor_get(v___x_71_, 0);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_71_);
if (v_isSharedCheck_111_ == 0)
{
v___x_106_ = v___x_71_;
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_a_104_);
lean_dec(v___x_71_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_109_; 
if (v_isShared_107_ == 0)
{
v___x_109_ = v___x_106_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v_a_104_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureDynamic___boxed(lean_object* v_uscript_112_, lean_object* v_proofHasMVar_113_, lean_object* v_preState_114_, lean_object* v_goal_115_, lean_object* v_a_116_, lean_object* v_a_117_, lean_object* v_a_118_, lean_object* v_a_119_, lean_object* v_a_120_){
_start:
{
uint8_t v_proofHasMVar_boxed_121_; lean_object* v_res_122_; 
v_proofHasMVar_boxed_121_ = lean_unbox(v_proofHasMVar_113_);
v_res_122_ = lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureDynamic(v_uscript_112_, v_proofHasMVar_boxed_121_, v_preState_114_, v_goal_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_);
lean_dec(v_a_119_);
lean_dec_ref(v_a_118_);
lean_dec(v_a_117_);
lean_dec_ref(v_a_116_);
return v_res_122_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Script_UScript_optimize_spec__2(lean_object* v_opts_123_, lean_object* v_opt_124_){
_start:
{
lean_object* v_name_125_; lean_object* v_defValue_126_; lean_object* v_map_127_; lean_object* v___x_128_; 
v_name_125_ = lean_ctor_get(v_opt_124_, 0);
v_defValue_126_ = lean_ctor_get(v_opt_124_, 1);
v_map_127_ = lean_ctor_get(v_opts_123_, 0);
v___x_128_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_127_, v_name_125_);
if (lean_obj_tag(v___x_128_) == 0)
{
uint8_t v___x_129_; 
v___x_129_ = lean_unbox(v_defValue_126_);
return v___x_129_;
}
else
{
lean_object* v_val_130_; 
v_val_130_ = lean_ctor_get(v___x_128_, 0);
lean_inc(v_val_130_);
lean_dec_ref_known(v___x_128_, 1);
if (lean_obj_tag(v_val_130_) == 1)
{
uint8_t v_v_131_; 
v_v_131_ = lean_ctor_get_uint8(v_val_130_, 0);
lean_dec_ref_known(v_val_130_, 0);
return v_v_131_;
}
else
{
uint8_t v___x_132_; 
lean_dec(v_val_130_);
v___x_132_ = lean_unbox(v_defValue_126_);
return v___x_132_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Script_UScript_optimize_spec__2___boxed(lean_object* v_opts_133_, lean_object* v_opt_134_){
_start:
{
uint8_t v_res_135_; lean_object* v_r_136_; 
v_res_135_ = lp_aesop_Lean_Option_get___at___00Aesop_Script_UScript_optimize_spec__2(v_opts_133_, v_opt_134_);
lean_dec_ref(v_opt_134_);
lean_dec_ref(v_opts_133_);
v_r_136_ = lean_box(v_res_135_);
return v_r_136_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7(size_t v_sz_145_, size_t v_i_146_, lean_object* v_bs_147_){
_start:
{
uint8_t v___x_148_; 
v___x_148_ = lean_usize_dec_lt(v_i_146_, v_sz_145_);
if (v___x_148_ == 0)
{
lean_object* v___x_149_; 
v___x_149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_149_, 0, v_bs_147_);
return v___x_149_;
}
else
{
lean_object* v_v_150_; lean_object* v___x_151_; uint8_t v___x_152_; 
v_v_150_ = lean_array_uget_borrowed(v_bs_147_, v_i_146_);
v___x_151_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__2));
lean_inc(v_v_150_);
v___x_152_ = l_Lean_Syntax_isOfKind(v_v_150_, v___x_151_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; 
lean_dec_ref(v_bs_147_);
v___x_153_ = lean_box(0);
return v___x_153_;
}
else
{
lean_object* v___x_154_; lean_object* v_ns_155_; lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_154_ = lean_unsigned_to_nat(0u);
v_ns_155_ = l_Lean_Syntax_getArg(v_v_150_, v___x_154_);
v___x_156_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__4));
lean_inc(v_ns_155_);
v___x_157_ = l_Lean_Syntax_isOfKind(v_ns_155_, v___x_156_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; 
lean_dec(v_ns_155_);
lean_dec_ref(v_bs_147_);
v___x_158_ = lean_box(0);
return v___x_158_;
}
else
{
lean_object* v_bs_x27_159_; size_t v___x_160_; size_t v___x_161_; lean_object* v___x_162_; 
v_bs_x27_159_ = lean_array_uset(v_bs_147_, v_i_146_, v___x_154_);
v___x_160_ = ((size_t)1ULL);
v___x_161_ = lean_usize_add(v_i_146_, v___x_160_);
v___x_162_ = lean_array_uset(v_bs_x27_159_, v_i_146_, v_ns_155_);
v_i_146_ = v___x_161_;
v_bs_147_ = v___x_162_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___boxed(lean_object* v_sz_164_, lean_object* v_i_165_, lean_object* v_bs_166_){
_start:
{
size_t v_sz_boxed_167_; size_t v_i_boxed_168_; lean_object* v_res_169_; 
v_sz_boxed_167_ = lean_unbox_usize(v_sz_164_);
lean_dec(v_sz_164_);
v_i_boxed_168_ = lean_unbox_usize(v_i_165_);
lean_dec(v_i_165_);
v_res_169_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7(v_sz_boxed_167_, v_i_boxed_168_, v_bs_166_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8(lean_object* v___x_170_, size_t v_sz_171_, size_t v_i_172_, lean_object* v_bs_173_){
_start:
{
uint8_t v___x_174_; 
v___x_174_ = lean_usize_dec_lt(v_i_172_, v_sz_171_);
if (v___x_174_ == 0)
{
lean_dec(v___x_170_);
return v_bs_173_;
}
else
{
lean_object* v_v_175_; lean_object* v___x_176_; lean_object* v_bs_x27_177_; lean_object* v___x_178_; lean_object* v___x_179_; size_t v___x_180_; size_t v___x_181_; lean_object* v___x_182_; 
v_v_175_ = lean_array_uget(v_bs_173_, v_i_172_);
v___x_176_ = lean_unsigned_to_nat(0u);
v_bs_x27_177_ = lean_array_uset(v_bs_173_, v_i_172_, v___x_176_);
v___x_178_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7___closed__2));
lean_inc(v___x_170_);
v___x_179_ = l_Lean_Syntax_node1(v___x_170_, v___x_178_, v_v_175_);
v___x_180_ = ((size_t)1ULL);
v___x_181_ = lean_usize_add(v_i_172_, v___x_180_);
v___x_182_ = lean_array_uset(v_bs_x27_177_, v_i_172_, v___x_179_);
v_i_172_ = v___x_181_;
v_bs_173_ = v___x_182_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8___boxed(lean_object* v___x_184_, lean_object* v_sz_185_, lean_object* v_i_186_, lean_object* v_bs_187_){
_start:
{
size_t v_sz_boxed_188_; size_t v_i_boxed_189_; lean_object* v_res_190_; 
v_sz_boxed_188_ = lean_unbox_usize(v_sz_185_);
lean_dec(v_sz_185_);
v_i_boxed_189_ = lean_unbox_usize(v_i_186_);
lean_dec(v_i_186_);
v_res_190_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8(v___x_184_, v_sz_boxed_188_, v_i_boxed_189_, v_bs_187_);
return v_res_190_;
}
}
static lean_object* _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20(void){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = l_Array_mkArray0(lean_box(0));
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2(lean_object* v_x_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_){
_start:
{
if (lean_obj_tag(v_x_238_) == 1)
{
lean_object* v_info_244_; lean_object* v_kind_245_; lean_object* v_args_246_; lean_object* v___x_248_; uint8_t v_isShared_249_; uint8_t v_isSharedCheck_391_; 
v_info_244_ = lean_ctor_get(v_x_238_, 0);
v_kind_245_ = lean_ctor_get(v_x_238_, 1);
v_args_246_ = lean_ctor_get(v_x_238_, 2);
v_isSharedCheck_391_ = !lean_is_exclusive(v_x_238_);
if (v_isSharedCheck_391_ == 0)
{
v___x_248_ = v_x_238_;
v_isShared_249_ = v_isSharedCheck_391_;
goto v_resetjp_247_;
}
else
{
lean_inc(v_args_246_);
lean_inc(v_kind_245_);
lean_inc(v_info_244_);
lean_dec(v_x_238_);
v___x_248_ = lean_box(0);
v_isShared_249_ = v_isSharedCheck_391_;
goto v_resetjp_247_;
}
v_resetjp_247_:
{
size_t v_sz_250_; size_t v___x_251_; lean_object* v___x_252_; 
v_sz_250_ = lean_array_size(v_args_246_);
v___x_251_ = ((size_t)0ULL);
v___x_252_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__6(v_sz_250_, v___x_251_, v_args_246_, v___y_239_, v___y_240_, v___y_241_, v___y_242_);
if (lean_obj_tag(v___x_252_) == 0)
{
lean_object* v_a_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_382_; 
v_a_253_ = lean_ctor_get(v___x_252_, 0);
v_isSharedCheck_382_ = !lean_is_exclusive(v___x_252_);
if (v_isSharedCheck_382_ == 0)
{
v___x_255_ = v___x_252_;
v_isShared_256_ = v_isSharedCheck_382_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_a_253_);
lean_dec(v___x_252_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_382_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v_stx_258_; 
if (v_isShared_249_ == 0)
{
lean_ctor_set(v___x_248_, 2, v_a_253_);
v_stx_258_ = v___x_248_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v_info_244_);
lean_ctor_set(v_reuseFailAlloc_381_, 1, v_kind_245_);
lean_ctor_set(v_reuseFailAlloc_381_, 2, v_a_253_);
v_stx_258_ = v_reuseFailAlloc_381_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
lean_object* v___x_259_; uint8_t v___x_260_; 
v___x_259_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__1));
lean_inc_ref(v_stx_258_);
v___x_260_ = l_Lean_Syntax_isOfKind(v_stx_258_, v___x_259_);
if (v___x_260_ == 0)
{
lean_object* v___x_262_; 
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_262_ = v___x_255_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_stx_258_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
else
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; uint8_t v___x_267_; 
v___x_264_ = lean_unsigned_to_nat(0u);
v___x_265_ = l_Lean_Syntax_getArg(v_stx_258_, v___x_264_);
v___x_266_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__3));
v___x_267_ = l_Lean_Syntax_isOfKind(v___x_265_, v___x_266_);
if (v___x_267_ == 0)
{
lean_object* v___x_269_; 
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_269_ = v___x_255_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v_stx_258_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
return v___x_269_;
}
}
else
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; uint8_t v___x_274_; 
v___x_271_ = lean_unsigned_to_nat(1u);
v___x_272_ = l_Lean_Syntax_getArg(v_stx_258_, v___x_271_);
v___x_273_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7));
lean_inc(v___x_272_);
v___x_274_ = l_Lean_Syntax_isOfKind(v___x_272_, v___x_273_);
if (v___x_274_ == 0)
{
lean_object* v___x_276_; 
lean_dec(v___x_272_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_276_ = v___x_255_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v_stx_258_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
else
{
lean_object* v___x_278_; lean_object* v___x_279_; uint8_t v___x_280_; 
v___x_278_ = l_Lean_Syntax_getArg(v___x_272_, v___x_264_);
lean_dec(v___x_272_);
v___x_279_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9));
lean_inc(v___x_278_);
v___x_280_ = l_Lean_Syntax_isOfKind(v___x_278_, v___x_279_);
if (v___x_280_ == 0)
{
lean_object* v___x_281_; uint8_t v___x_282_; 
v___x_281_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11));
lean_inc(v___x_278_);
v___x_282_ = l_Lean_Syntax_isOfKind(v___x_278_, v___x_281_);
if (v___x_282_ == 0)
{
lean_object* v___x_284_; 
lean_dec(v___x_278_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_284_ = v___x_255_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_stx_258_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
return v___x_284_;
}
}
else
{
lean_object* v___x_286_; lean_object* v_tacs_287_; lean_object* v_a_288_; lean_object* v___x_289_; uint8_t v___x_290_; 
v___x_286_ = l_Lean_Syntax_getArg(v___x_278_, v___x_271_);
lean_dec(v___x_278_);
v_tacs_287_ = l_Lean_Syntax_getArgs(v___x_286_);
lean_dec(v___x_286_);
v_a_288_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_tacs_287_);
lean_dec_ref(v_tacs_287_);
v___x_289_ = lean_array_get_size(v_a_288_);
v___x_290_ = lean_nat_dec_lt(v___x_264_, v___x_289_);
if (v___x_290_ == 0)
{
lean_object* v___x_292_; 
lean_dec_ref(v_a_288_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_292_ = v___x_255_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v_stx_258_);
v___x_292_ = v_reuseFailAlloc_293_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
return v___x_292_;
}
}
else
{
lean_object* v___x_294_; lean_object* v___x_295_; uint8_t v___x_296_; 
v___x_294_ = lean_array_fget(v_a_288_, v___x_264_);
v___x_295_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13));
lean_inc(v___x_294_);
v___x_296_ = l_Lean_Syntax_isOfKind(v___x_294_, v___x_295_);
if (v___x_296_ == 0)
{
lean_object* v___x_298_; 
lean_dec(v___x_294_);
lean_dec_ref(v_a_288_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_298_ = v___x_255_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_stx_258_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
else
{
lean_object* v___x_300_; lean_object* v___x_301_; size_t v_sz_302_; lean_object* v___x_303_; 
v___x_300_ = l_Lean_Syntax_getArg(v___x_294_, v___x_271_);
lean_dec(v___x_294_);
v___x_301_ = l_Lean_Syntax_getArgs(v___x_300_);
lean_dec(v___x_300_);
v_sz_302_ = lean_array_size(v___x_301_);
v___x_303_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7(v_sz_302_, v___x_251_, v___x_301_);
if (lean_obj_tag(v___x_303_) == 0)
{
lean_object* v___x_305_; 
lean_dec_ref(v_a_288_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_305_ = v___x_255_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v_stx_258_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
else
{
lean_object* v_val_307_; lean_object* v_ref_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; size_t v_sz_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_331_; 
lean_dec_ref(v_stx_258_);
v_val_307_ = lean_ctor_get(v___x_303_, 0);
lean_inc(v_val_307_);
lean_dec_ref_known(v___x_303_, 1);
v_ref_308_ = lean_ctor_get(v___y_241_, 5);
v___x_309_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14));
v___x_310_ = l_Lean_SourceInfo_fromRef(v_ref_308_, v___x_280_);
v___x_311_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16));
v___x_312_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__17));
lean_inc_n(v___x_310_, 7);
v___x_313_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_310_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
v___x_314_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19));
v___x_315_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20, &lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20);
v_sz_316_ = lean_array_size(v_val_307_);
v___x_317_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8(v___x_310_, v_sz_316_, v___x_251_, v_val_307_);
v___x_318_ = l_Array_append___redArg(v___x_315_, v___x_317_);
lean_dec_ref(v___x_317_);
v___x_319_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_319_, 0, v___x_310_);
lean_ctor_set(v___x_319_, 1, v___x_314_);
lean_ctor_set(v___x_319_, 2, v___x_318_);
v___x_320_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__21));
v___x_321_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_321_, 0, v___x_310_);
lean_ctor_set(v___x_321_, 1, v___x_320_);
v___x_322_ = l_Array_toSubarray___redArg(v_a_288_, v___x_271_, v___x_289_);
v___x_323_ = l_Subarray_copy___redArg(v___x_322_);
v___x_324_ = l_Lean_Syntax_SepArray_ofElems(v___x_309_, v___x_323_);
lean_dec_ref(v___x_323_);
v___x_325_ = l_Array_append___redArg(v___x_315_, v___x_324_);
lean_dec_ref(v___x_324_);
v___x_326_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_326_, 0, v___x_310_);
lean_ctor_set(v___x_326_, 1, v___x_314_);
lean_ctor_set(v___x_326_, 2, v___x_325_);
v___x_327_ = l_Lean_Syntax_node1(v___x_310_, v___x_279_, v___x_326_);
v___x_328_ = l_Lean_Syntax_node1(v___x_310_, v___x_273_, v___x_327_);
v___x_329_ = l_Lean_Syntax_node4(v___x_310_, v___x_311_, v___x_313_, v___x_319_, v___x_321_, v___x_328_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v___x_329_);
v___x_331_ = v___x_255_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v___x_329_);
v___x_331_ = v_reuseFailAlloc_332_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
return v___x_331_;
}
}
}
}
}
}
else
{
lean_object* v___x_333_; lean_object* v_tacs_334_; lean_object* v_a_335_; lean_object* v___x_336_; uint8_t v___x_337_; 
v___x_333_ = l_Lean_Syntax_getArg(v___x_278_, v___x_264_);
lean_dec(v___x_278_);
v_tacs_334_ = l_Lean_Syntax_getArgs(v___x_333_);
lean_dec(v___x_333_);
v_a_335_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_tacs_334_);
lean_dec_ref(v_tacs_334_);
v___x_336_ = lean_array_get_size(v_a_335_);
v___x_337_ = lean_nat_dec_lt(v___x_264_, v___x_336_);
if (v___x_337_ == 0)
{
lean_object* v___x_339_; 
lean_dec_ref(v_a_335_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_339_ = v___x_255_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v_stx_258_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
else
{
lean_object* v___x_341_; lean_object* v___x_342_; uint8_t v___x_343_; 
v___x_341_ = lean_array_fget(v_a_335_, v___x_264_);
v___x_342_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13));
lean_inc(v___x_341_);
v___x_343_ = l_Lean_Syntax_isOfKind(v___x_341_, v___x_342_);
if (v___x_343_ == 0)
{
lean_object* v___x_345_; 
lean_dec(v___x_341_);
lean_dec_ref(v_a_335_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_345_ = v___x_255_;
goto v_reusejp_344_;
}
else
{
lean_object* v_reuseFailAlloc_346_; 
v_reuseFailAlloc_346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_346_, 0, v_stx_258_);
v___x_345_ = v_reuseFailAlloc_346_;
goto v_reusejp_344_;
}
v_reusejp_344_:
{
return v___x_345_;
}
}
else
{
lean_object* v___x_347_; lean_object* v___x_348_; size_t v_sz_349_; lean_object* v___x_350_; 
v___x_347_ = l_Lean_Syntax_getArg(v___x_341_, v___x_271_);
lean_dec(v___x_341_);
v___x_348_ = l_Lean_Syntax_getArgs(v___x_347_);
lean_dec(v___x_347_);
v_sz_349_ = lean_array_size(v___x_348_);
v___x_350_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7(v_sz_349_, v___x_251_, v___x_348_);
if (lean_obj_tag(v___x_350_) == 0)
{
lean_object* v___x_352_; 
lean_dec_ref(v_a_335_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v_stx_258_);
v___x_352_ = v___x_255_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v_stx_258_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
else
{
lean_object* v_val_354_; lean_object* v_ref_355_; lean_object* v___x_356_; uint8_t v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; size_t v_sz_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_379_; 
lean_dec_ref(v_stx_258_);
v_val_354_ = lean_ctor_get(v___x_350_, 0);
lean_inc(v_val_354_);
lean_dec_ref_known(v___x_350_, 1);
v_ref_355_ = lean_ctor_get(v___y_241_, 5);
v___x_356_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14));
v___x_357_ = 0;
v___x_358_ = l_Lean_SourceInfo_fromRef(v_ref_355_, v___x_357_);
v___x_359_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__16));
v___x_360_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__17));
lean_inc_n(v___x_358_, 7);
v___x_361_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_358_);
lean_ctor_set(v___x_361_, 1, v___x_360_);
v___x_362_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19));
v___x_363_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20, &lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20);
v_sz_364_ = lean_array_size(v_val_354_);
v___x_365_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8(v___x_358_, v_sz_364_, v___x_251_, v_val_354_);
v___x_366_ = l_Array_append___redArg(v___x_363_, v___x_365_);
lean_dec_ref(v___x_365_);
v___x_367_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_367_, 0, v___x_358_);
lean_ctor_set(v___x_367_, 1, v___x_362_);
lean_ctor_set(v___x_367_, 2, v___x_366_);
v___x_368_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__21));
v___x_369_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_358_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
v___x_370_ = l_Array_toSubarray___redArg(v_a_335_, v___x_271_, v___x_336_);
v___x_371_ = l_Subarray_copy___redArg(v___x_370_);
v___x_372_ = l_Lean_Syntax_SepArray_ofElems(v___x_356_, v___x_371_);
lean_dec_ref(v___x_371_);
v___x_373_ = l_Array_append___redArg(v___x_363_, v___x_372_);
lean_dec_ref(v___x_372_);
v___x_374_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_374_, 0, v___x_358_);
lean_ctor_set(v___x_374_, 1, v___x_362_);
lean_ctor_set(v___x_374_, 2, v___x_373_);
v___x_375_ = l_Lean_Syntax_node1(v___x_358_, v___x_279_, v___x_374_);
v___x_376_ = l_Lean_Syntax_node1(v___x_358_, v___x_273_, v___x_375_);
v___x_377_ = l_Lean_Syntax_node4(v___x_358_, v___x_359_, v___x_361_, v___x_367_, v___x_369_, v___x_376_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 0, v___x_377_);
v___x_379_ = v___x_255_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v___x_377_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
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
}
else
{
lean_object* v_a_383_; lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_390_; 
lean_del_object(v___x_248_);
lean_dec(v_kind_245_);
lean_dec(v_info_244_);
v_a_383_ = lean_ctor_get(v___x_252_, 0);
v_isSharedCheck_390_ = !lean_is_exclusive(v___x_252_);
if (v_isSharedCheck_390_ == 0)
{
v___x_385_ = v___x_252_;
v_isShared_386_ = v_isSharedCheck_390_;
goto v_resetjp_384_;
}
else
{
lean_inc(v_a_383_);
lean_dec(v___x_252_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_390_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_388_; 
if (v_isShared_386_ == 0)
{
v___x_388_ = v___x_385_;
goto v_reusejp_387_;
}
else
{
lean_object* v_reuseFailAlloc_389_; 
v_reuseFailAlloc_389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_389_, 0, v_a_383_);
v___x_388_ = v_reuseFailAlloc_389_;
goto v_reusejp_387_;
}
v_reusejp_387_:
{
return v___x_388_;
}
}
}
}
}
else
{
lean_object* v___x_392_; 
v___x_392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_392_, 0, v_x_238_);
return v___x_392_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__6(size_t v_sz_393_, size_t v_i_394_, lean_object* v_bs_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_){
_start:
{
uint8_t v___x_401_; 
v___x_401_ = lean_usize_dec_lt(v_i_394_, v_sz_393_);
if (v___x_401_ == 0)
{
lean_object* v___x_402_; 
v___x_402_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_402_, 0, v_bs_395_);
return v___x_402_;
}
else
{
lean_object* v_v_403_; lean_object* v___x_404_; 
v_v_403_ = lean_array_uget_borrowed(v_bs_395_, v_i_394_);
lean_inc(v_v_403_);
v___x_404_ = lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2(v_v_403_, v___y_396_, v___y_397_, v___y_398_, v___y_399_);
if (lean_obj_tag(v___x_404_) == 0)
{
lean_object* v_a_405_; lean_object* v___x_406_; lean_object* v_bs_x27_407_; size_t v___x_408_; size_t v___x_409_; lean_object* v___x_410_; 
v_a_405_ = lean_ctor_get(v___x_404_, 0);
lean_inc(v_a_405_);
lean_dec_ref_known(v___x_404_, 1);
v___x_406_ = lean_unsigned_to_nat(0u);
v_bs_x27_407_ = lean_array_uset(v_bs_395_, v_i_394_, v___x_406_);
v___x_408_ = ((size_t)1ULL);
v___x_409_ = lean_usize_add(v_i_394_, v___x_408_);
v___x_410_ = lean_array_uset(v_bs_x27_407_, v_i_394_, v_a_405_);
v_i_394_ = v___x_409_;
v_bs_395_ = v___x_410_;
goto _start;
}
else
{
lean_object* v_a_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_419_; 
lean_dec_ref(v_bs_395_);
v_a_412_ = lean_ctor_get(v___x_404_, 0);
v_isSharedCheck_419_ = !lean_is_exclusive(v___x_404_);
if (v_isSharedCheck_419_ == 0)
{
v___x_414_ = v___x_404_;
v_isShared_415_ = v_isSharedCheck_419_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_a_412_);
lean_dec(v___x_404_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_419_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_417_; 
if (v_isShared_415_ == 0)
{
v___x_417_ = v___x_414_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v_a_412_);
v___x_417_ = v_reuseFailAlloc_418_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
return v___x_417_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__6___boxed(lean_object* v_sz_420_, lean_object* v_i_421_, lean_object* v_bs_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
size_t v_sz_boxed_428_; size_t v_i_boxed_429_; lean_object* v_res_430_; 
v_sz_boxed_428_ = lean_unbox_usize(v_sz_420_);
lean_dec(v_sz_420_);
v_i_boxed_429_ = lean_unbox_usize(v_i_421_);
lean_dec(v_i_421_);
v_res_430_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__6(v_sz_boxed_428_, v_i_boxed_429_, v_bs_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___boxed(lean_object* v_x_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2(v_x_431_, v___y_432_, v___y_433_, v___y_434_, v___y_435_);
lean_dec(v___y_435_);
lean_dec_ref(v___y_434_);
lean_dec(v___y_433_);
lean_dec_ref(v___y_432_);
return v_res_437_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___redArg(lean_object* v_a_438_, lean_object* v_x_439_){
_start:
{
if (lean_obj_tag(v_x_439_) == 0)
{
uint8_t v___x_440_; 
v___x_440_ = 0;
return v___x_440_;
}
else
{
lean_object* v_key_441_; lean_object* v_tail_442_; uint8_t v___x_443_; 
v_key_441_ = lean_ctor_get(v_x_439_, 0);
v_tail_442_ = lean_ctor_get(v_x_439_, 2);
v___x_443_ = lean_name_eq(v_key_441_, v_a_438_);
if (v___x_443_ == 0)
{
v_x_439_ = v_tail_442_;
goto _start;
}
else
{
return v___x_443_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___redArg___boxed(lean_object* v_a_445_, lean_object* v_x_446_){
_start:
{
uint8_t v_res_447_; lean_object* v_r_448_; 
v_res_447_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___redArg(v_a_445_, v_x_446_);
lean_dec(v_x_446_);
lean_dec(v_a_445_);
v_r_448_ = lean_box(v_res_447_);
return v_r_448_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___redArg(lean_object* v_m_449_, lean_object* v_a_450_){
_start:
{
lean_object* v_buckets_451_; lean_object* v___x_452_; uint64_t v___y_454_; 
v_buckets_451_ = lean_ctor_get(v_m_449_, 1);
v___x_452_ = lean_array_get_size(v_buckets_451_);
if (lean_obj_tag(v_a_450_) == 0)
{
uint64_t v___x_468_; 
v___x_468_ = 1723ULL;
v___y_454_ = v___x_468_;
goto v___jp_453_;
}
else
{
uint64_t v_hash_469_; 
v_hash_469_ = lean_ctor_get_uint64(v_a_450_, sizeof(void*)*2);
v___y_454_ = v_hash_469_;
goto v___jp_453_;
}
v___jp_453_:
{
uint64_t v___x_455_; uint64_t v___x_456_; uint64_t v_fold_457_; uint64_t v___x_458_; uint64_t v___x_459_; uint64_t v___x_460_; size_t v___x_461_; size_t v___x_462_; size_t v___x_463_; size_t v___x_464_; size_t v___x_465_; lean_object* v___x_466_; uint8_t v___x_467_; 
v___x_455_ = 32ULL;
v___x_456_ = lean_uint64_shift_right(v___y_454_, v___x_455_);
v_fold_457_ = lean_uint64_xor(v___y_454_, v___x_456_);
v___x_458_ = 16ULL;
v___x_459_ = lean_uint64_shift_right(v_fold_457_, v___x_458_);
v___x_460_ = lean_uint64_xor(v_fold_457_, v___x_459_);
v___x_461_ = lean_uint64_to_usize(v___x_460_);
v___x_462_ = lean_usize_of_nat(v___x_452_);
v___x_463_ = ((size_t)1ULL);
v___x_464_ = lean_usize_sub(v___x_462_, v___x_463_);
v___x_465_ = lean_usize_land(v___x_461_, v___x_464_);
v___x_466_ = lean_array_uget_borrowed(v_buckets_451_, v___x_465_);
v___x_467_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___redArg(v_a_450_, v___x_466_);
return v___x_467_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___redArg___boxed(lean_object* v_m_470_, lean_object* v_a_471_){
_start:
{
uint8_t v_res_472_; lean_object* v_r_473_; 
v_res_472_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___redArg(v_m_470_, v_a_471_);
lean_dec(v_a_471_);
lean_dec_ref(v_m_470_);
v_r_473_ = lean_box(v_res_472_);
return v_r_473_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___redArg(lean_object* v_usedNames_474_, lean_object* v_as_475_, size_t v_sz_476_, size_t v_i_477_, lean_object* v_b_478_){
_start:
{
uint8_t v___x_480_; 
v___x_480_ = lean_usize_dec_lt(v_i_477_, v_sz_476_);
if (v___x_480_ == 0)
{
lean_object* v___x_481_; 
v___x_481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_481_, 0, v_b_478_);
return v___x_481_;
}
else
{
lean_object* v_a_482_; lean_object* v___x_483_; uint8_t v___x_484_; 
v_a_482_ = lean_array_uget_borrowed(v_as_475_, v_i_477_);
v___x_483_ = l_Lean_TSyntax_getId(v_a_482_);
v___x_484_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___redArg(v_usedNames_474_, v___x_483_);
lean_dec(v___x_483_);
if (v___x_484_ == 0)
{
lean_object* v___x_485_; lean_object* v___x_486_; size_t v___x_487_; size_t v___x_488_; 
v___x_485_ = lean_unsigned_to_nat(1u);
v___x_486_ = lean_nat_add(v_b_478_, v___x_485_);
lean_dec(v_b_478_);
v___x_487_ = ((size_t)1ULL);
v___x_488_ = lean_usize_add(v_i_477_, v___x_487_);
v_i_477_ = v___x_488_;
v_b_478_ = v___x_486_;
goto _start;
}
else
{
lean_object* v___x_490_; 
v___x_490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_490_, 0, v_b_478_);
return v___x_490_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___redArg___boxed(lean_object* v_usedNames_491_, lean_object* v_as_492_, lean_object* v_sz_493_, lean_object* v_i_494_, lean_object* v_b_495_, lean_object* v___y_496_){
_start:
{
size_t v_sz_boxed_497_; size_t v_i_boxed_498_; lean_object* v_res_499_; 
v_sz_boxed_497_ = lean_unbox_usize(v_sz_493_);
lean_dec(v_sz_493_);
v_i_boxed_498_ = lean_unbox_usize(v_i_494_);
lean_dec(v_i_494_);
v_res_499_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___redArg(v_usedNames_491_, v_as_492_, v_sz_boxed_497_, v_i_boxed_498_, v_b_495_);
lean_dec_ref(v_as_492_);
lean_dec_ref(v_usedNames_491_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg(lean_object* v_tacs_500_, lean_object* v___y_501_){
_start:
{
lean_object* v_ref_503_; uint8_t v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; 
v_ref_503_ = lean_ctor_get(v___y_501_, 5);
v___x_504_ = 0;
v___x_505_ = l_Lean_SourceInfo_fromRef(v_ref_503_, v___x_504_);
v___x_506_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7));
v___x_507_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9));
v___x_508_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19));
v___x_509_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20, &lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20);
v___x_510_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14));
v___x_511_ = l_Lean_Syntax_SepArray_ofElems(v___x_510_, v_tacs_500_);
v___x_512_ = l_Array_append___redArg(v___x_509_, v___x_511_);
lean_dec_ref(v___x_511_);
lean_inc_n(v___x_505_, 2);
v___x_513_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_513_, 0, v___x_505_);
lean_ctor_set(v___x_513_, 1, v___x_508_);
lean_ctor_set(v___x_513_, 2, v___x_512_);
v___x_514_ = l_Lean_Syntax_node1(v___x_505_, v___x_507_, v___x_513_);
v___x_515_ = l_Lean_Syntax_node1(v___x_505_, v___x_506_, v___x_514_);
v___x_516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_516_, 0, v___x_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg___boxed(lean_object* v_tacs_517_, lean_object* v___y_518_, lean_object* v___y_519_){
_start:
{
lean_object* v_res_520_; 
v_res_520_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg(v_tacs_517_, v___y_518_);
lean_dec_ref(v___y_518_);
lean_dec_ref(v_tacs_517_);
return v_res_520_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__13___redArg(lean_object* v_a_521_, lean_object* v_b_522_){
_start:
{
lean_object* v_array_523_; lean_object* v_start_524_; lean_object* v_stop_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_538_; 
v_array_523_ = lean_ctor_get(v_a_521_, 0);
v_start_524_ = lean_ctor_get(v_a_521_, 1);
v_stop_525_ = lean_ctor_get(v_a_521_, 2);
v_isSharedCheck_538_ = !lean_is_exclusive(v_a_521_);
if (v_isSharedCheck_538_ == 0)
{
v___x_527_ = v_a_521_;
v_isShared_528_ = v_isSharedCheck_538_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_stop_525_);
lean_inc(v_start_524_);
lean_inc(v_array_523_);
lean_dec(v_a_521_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_538_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
uint8_t v___x_529_; 
v___x_529_ = lean_nat_dec_lt(v_start_524_, v_stop_525_);
if (v___x_529_ == 0)
{
lean_del_object(v___x_527_);
lean_dec(v_stop_525_);
lean_dec(v_start_524_);
lean_dec_ref(v_array_523_);
return v_b_522_;
}
else
{
lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_533_; 
v___x_530_ = lean_unsigned_to_nat(1u);
v___x_531_ = lean_nat_add(v_start_524_, v___x_530_);
lean_inc_ref(v_array_523_);
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 1, v___x_531_);
v___x_533_ = v___x_527_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v_array_523_);
lean_ctor_set(v_reuseFailAlloc_537_, 1, v___x_531_);
lean_ctor_set(v_reuseFailAlloc_537_, 2, v_stop_525_);
v___x_533_ = v_reuseFailAlloc_537_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_534_ = lean_array_fget(v_array_523_, v_start_524_);
lean_dec(v_start_524_);
lean_dec_ref(v_array_523_);
v___x_535_ = lean_array_push(v_b_522_, v___x_534_);
v_a_521_ = v___x_533_;
v_b_522_ = v___x_535_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__10___redArg(lean_object* v_a_539_, lean_object* v_b_540_){
_start:
{
lean_object* v_array_541_; lean_object* v_start_542_; lean_object* v_stop_543_; lean_object* v___x_545_; uint8_t v_isShared_546_; uint8_t v_isSharedCheck_556_; 
v_array_541_ = lean_ctor_get(v_a_539_, 0);
v_start_542_ = lean_ctor_get(v_a_539_, 1);
v_stop_543_ = lean_ctor_get(v_a_539_, 2);
v_isSharedCheck_556_ = !lean_is_exclusive(v_a_539_);
if (v_isSharedCheck_556_ == 0)
{
v___x_545_ = v_a_539_;
v_isShared_546_ = v_isSharedCheck_556_;
goto v_resetjp_544_;
}
else
{
lean_inc(v_stop_543_);
lean_inc(v_start_542_);
lean_inc(v_array_541_);
lean_dec(v_a_539_);
v___x_545_ = lean_box(0);
v_isShared_546_ = v_isSharedCheck_556_;
goto v_resetjp_544_;
}
v_resetjp_544_:
{
uint8_t v___x_547_; 
v___x_547_ = lean_nat_dec_lt(v_start_542_, v_stop_543_);
if (v___x_547_ == 0)
{
lean_del_object(v___x_545_);
lean_dec(v_stop_543_);
lean_dec(v_start_542_);
lean_dec_ref(v_array_541_);
return v_b_540_;
}
else
{
lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_551_; 
v___x_548_ = lean_unsigned_to_nat(1u);
v___x_549_ = lean_nat_add(v_start_542_, v___x_548_);
lean_inc_ref(v_array_541_);
if (v_isShared_546_ == 0)
{
lean_ctor_set(v___x_545_, 1, v___x_549_);
v___x_551_ = v___x_545_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_array_541_);
lean_ctor_set(v_reuseFailAlloc_555_, 1, v___x_549_);
lean_ctor_set(v_reuseFailAlloc_555_, 2, v_stop_543_);
v___x_551_ = v_reuseFailAlloc_555_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_552_ = lean_array_fget(v_array_541_, v_start_542_);
lean_dec(v_start_542_);
lean_dec_ref(v_array_541_);
v___x_553_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents(v_b_540_, v___x_552_);
v_a_539_ = v___x_551_;
v_b_540_ = v___x_553_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_557_ = lean_box(0);
v___x_558_ = lean_unsigned_to_nat(16u);
v___x_559_ = lean_mk_array(v___x_558_, v___x_557_);
return v___x_559_;
}
}
static lean_object* _init_lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__1(void){
_start:
{
lean_object* v___x_560_; lean_object* v_dropUntil_561_; lean_object* v___x_562_; 
v___x_560_ = lean_obj_once(&lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__0, &lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__0_once, _init_lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__0);
v_dropUntil_561_ = lean_unsigned_to_nat(0u);
v___x_562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_562_, 0, v_dropUntil_561_);
lean_ctor_set(v___x_562_, 1, v___x_560_);
return v___x_562_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3(lean_object* v_x_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_){
_start:
{
lean_object* v___x_572_; uint8_t v___x_573_; 
v___x_572_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7));
lean_inc(v_x_566_);
v___x_573_ = l_Lean_Syntax_isOfKind(v_x_566_, v___x_572_);
if (v___x_573_ == 0)
{
lean_object* v___x_574_; 
v___x_574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_574_, 0, v_x_566_);
return v___x_574_;
}
else
{
lean_object* v_dropUntil_575_; lean_object* v___x_576_; lean_object* v___x_577_; uint8_t v___x_578_; 
v_dropUntil_575_ = lean_unsigned_to_nat(0u);
v___x_576_ = l_Lean_Syntax_getArg(v_x_566_, v_dropUntil_575_);
v___x_577_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9));
lean_inc(v___x_576_);
v___x_578_ = l_Lean_Syntax_isOfKind(v___x_576_, v___x_577_);
if (v___x_578_ == 0)
{
lean_object* v___x_579_; 
lean_dec(v___x_576_);
v___x_579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_579_, 0, v_x_566_);
return v___x_579_;
}
else
{
lean_object* v___x_580_; lean_object* v_tacs_581_; lean_object* v_a_582_; lean_object* v___x_583_; uint8_t v___x_584_; 
v___x_580_ = l_Lean_Syntax_getArg(v___x_576_, v_dropUntil_575_);
lean_dec(v___x_576_);
v_tacs_581_ = l_Lean_Syntax_getArgs(v___x_580_);
lean_dec(v___x_580_);
v_a_582_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_tacs_581_);
lean_dec_ref(v_tacs_581_);
v___x_583_ = lean_array_get_size(v_a_582_);
v___x_584_ = lean_nat_dec_lt(v_dropUntil_575_, v___x_583_);
if (v___x_584_ == 0)
{
lean_object* v___x_585_; 
lean_dec_ref(v_a_582_);
v___x_585_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_585_, 0, v_x_566_);
return v___x_585_;
}
else
{
lean_object* v___x_586_; lean_object* v___x_587_; uint8_t v___x_588_; 
v___x_586_ = lean_array_fget(v_a_582_, v_dropUntil_575_);
v___x_587_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__13));
lean_inc(v___x_586_);
v___x_588_ = l_Lean_Syntax_isOfKind(v___x_586_, v___x_587_);
if (v___x_588_ == 0)
{
lean_object* v___x_589_; 
lean_dec(v___x_586_);
lean_dec_ref(v_a_582_);
v___x_589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_589_, 0, v_x_566_);
return v___x_589_;
}
else
{
lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; size_t v_sz_593_; size_t v___x_594_; lean_object* v___x_595_; 
v___x_590_ = lean_unsigned_to_nat(1u);
v___x_591_ = l_Lean_Syntax_getArg(v___x_586_, v___x_590_);
lean_dec(v___x_586_);
v___x_592_ = l_Lean_Syntax_getArgs(v___x_591_);
lean_dec(v___x_591_);
v_sz_593_ = lean_array_size(v___x_592_);
v___x_594_ = ((size_t)0ULL);
v___x_595_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__7(v_sz_593_, v___x_594_, v___x_592_);
if (lean_obj_tag(v___x_595_) == 0)
{
lean_object* v___x_596_; 
lean_dec_ref(v_a_582_);
v___x_596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_596_, 0, v_x_566_);
return v___x_596_;
}
else
{
lean_object* v_val_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v_usedNames_600_; size_t v_sz_601_; lean_object* v___x_602_; 
v_val_597_ = lean_ctor_get(v___x_595_, 0);
lean_inc(v_val_597_);
lean_dec_ref_known(v___x_595_, 1);
v___x_598_ = lean_obj_once(&lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__1, &lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__1_once, _init_lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__1);
v___x_599_ = l_Array_toSubarray___redArg(v_a_582_, v___x_590_, v___x_583_);
lean_inc_ref(v___x_599_);
v_usedNames_600_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__10___redArg(v___x_599_, v___x_598_);
v_sz_601_ = lean_array_size(v_val_597_);
v___x_602_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___redArg(v_usedNames_600_, v_val_597_, v_sz_601_, v___x_594_, v_dropUntil_575_);
lean_dec_ref(v_usedNames_600_);
if (lean_obj_tag(v___x_602_) == 0)
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_650_; 
v_a_603_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_650_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_650_ == 0)
{
v___x_605_ = v___x_602_;
v_isShared_606_ = v_isSharedCheck_650_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_602_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_650_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
uint8_t v___x_607_; 
v___x_607_ = lean_nat_dec_eq(v_a_603_, v_dropUntil_575_);
if (v___x_607_ == 0)
{
lean_object* v___x_608_; uint8_t v___x_609_; 
lean_del_object(v___x_605_);
lean_dec(v_x_566_);
v___x_608_ = lean_array_get_size(v_val_597_);
v___x_609_ = lean_nat_dec_eq(v_a_603_, v___x_608_);
if (v___x_609_ == 0)
{
lean_object* v_ref_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v_ns_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; size_t v_sz_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v_result_624_; lean_object* v_result_625_; lean_object* v___x_626_; lean_object* v_result_627_; lean_object* v___x_628_; lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
v_ref_610_ = lean_ctor_get(v___y_569_, 5);
v___x_611_ = l_Array_toSubarray___redArg(v_val_597_, v_a_603_, v___x_608_);
v___x_612_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__2));
v_ns_613_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__13___redArg(v___x_611_, v___x_612_);
v___x_614_ = l_Lean_SourceInfo_fromRef(v_ref_610_, v___x_609_);
v___x_615_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__3));
lean_inc_n(v___x_614_, 3);
v___x_616_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_614_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
v___x_617_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19));
v___x_618_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20, &lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20);
v_sz_619_ = lean_array_size(v_ns_613_);
v___x_620_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2_spec__8(v___x_614_, v_sz_619_, v___x_594_, v_ns_613_);
v___x_621_ = l_Array_append___redArg(v___x_618_, v___x_620_);
lean_dec_ref(v___x_620_);
v___x_622_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_622_, 0, v___x_614_);
lean_ctor_set(v___x_622_, 1, v___x_617_);
lean_ctor_set(v___x_622_, 2, v___x_621_);
v___x_623_ = l_Lean_Syntax_node2(v___x_614_, v___x_587_, v___x_616_, v___x_622_);
v_result_624_ = lean_mk_empty_array_with_capacity(v___x_583_);
v_result_625_ = lean_array_push(v_result_624_, v___x_623_);
v___x_626_ = l_Subarray_copy___redArg(v___x_599_);
v_result_627_ = l_Array_append___redArg(v_result_625_, v___x_626_);
lean_dec_ref(v___x_626_);
v___x_628_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg(v_result_627_, v___y_569_);
lean_dec_ref(v_result_627_);
v_a_629_ = lean_ctor_get(v___x_628_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_628_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_628_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_628_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_a_629_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
else
{
lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v_a_639_; lean_object* v___x_641_; uint8_t v_isShared_642_; uint8_t v_isSharedCheck_646_; 
lean_dec(v_a_603_);
lean_dec(v_val_597_);
v___x_637_ = l_Subarray_copy___redArg(v___x_599_);
v___x_638_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg(v___x_637_, v___y_569_);
lean_dec_ref(v___x_637_);
v_a_639_ = lean_ctor_get(v___x_638_, 0);
v_isSharedCheck_646_ = !lean_is_exclusive(v___x_638_);
if (v_isSharedCheck_646_ == 0)
{
v___x_641_ = v___x_638_;
v_isShared_642_ = v_isSharedCheck_646_;
goto v_resetjp_640_;
}
else
{
lean_inc(v_a_639_);
lean_dec(v___x_638_);
v___x_641_ = lean_box(0);
v_isShared_642_ = v_isSharedCheck_646_;
goto v_resetjp_640_;
}
v_resetjp_640_:
{
lean_object* v___x_644_; 
if (v_isShared_642_ == 0)
{
v___x_644_ = v___x_641_;
goto v_reusejp_643_;
}
else
{
lean_object* v_reuseFailAlloc_645_; 
v_reuseFailAlloc_645_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_645_, 0, v_a_639_);
v___x_644_ = v_reuseFailAlloc_645_;
goto v_reusejp_643_;
}
v_reusejp_643_:
{
return v___x_644_;
}
}
}
}
else
{
lean_object* v___x_648_; 
lean_dec(v_a_603_);
lean_dec_ref(v___x_599_);
lean_dec(v_val_597_);
if (v_isShared_606_ == 0)
{
lean_ctor_set(v___x_605_, 0, v_x_566_);
v___x_648_ = v___x_605_;
goto v_reusejp_647_;
}
else
{
lean_object* v_reuseFailAlloc_649_; 
v_reuseFailAlloc_649_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_649_, 0, v_x_566_);
v___x_648_ = v_reuseFailAlloc_649_;
goto v_reusejp_647_;
}
v_reusejp_647_:
{
return v___x_648_;
}
}
}
}
else
{
lean_object* v_a_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_658_; 
lean_dec_ref(v___x_599_);
lean_dec(v_val_597_);
lean_dec(v_x_566_);
v_a_651_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_658_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_658_ == 0)
{
v___x_653_ = v___x_602_;
v_isShared_654_ = v_isSharedCheck_658_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_a_651_);
lean_dec(v___x_602_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_658_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
lean_object* v___x_656_; 
if (v_isShared_654_ == 0)
{
v___x_656_ = v___x_653_;
goto v_reusejp_655_;
}
else
{
lean_object* v_reuseFailAlloc_657_; 
v_reuseFailAlloc_657_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_657_, 0, v_a_651_);
v___x_656_ = v_reuseFailAlloc_657_;
goto v_reusejp_655_;
}
v_reusejp_655_:
{
return v___x_656_;
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___boxed(lean_object* v_x_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3(v_x_659_, v___y_660_, v___y_661_, v___y_662_, v___y_663_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
lean_dec(v___y_661_);
lean_dec_ref(v___y_660_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___redArg(lean_object* v_stx_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_){
_start:
{
lean_object* v___x_672_; 
v___x_672_ = lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2(v_stx_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
if (lean_obj_tag(v___x_672_) == 0)
{
lean_object* v_a_673_; lean_object* v___x_674_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
lean_dec_ref_known(v___x_672_, 1);
v___x_674_ = lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3(v_a_673_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
if (lean_obj_tag(v___x_674_) == 0)
{
lean_object* v_a_675_; lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_682_; 
v_a_675_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_682_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_682_ == 0)
{
v___x_677_ = v___x_674_;
v_isShared_678_ = v_isSharedCheck_682_;
goto v_resetjp_676_;
}
else
{
lean_inc(v_a_675_);
lean_dec(v___x_674_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_682_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
lean_object* v___x_680_; 
if (v_isShared_678_ == 0)
{
v___x_680_ = v___x_677_;
goto v_reusejp_679_;
}
else
{
lean_object* v_reuseFailAlloc_681_; 
v_reuseFailAlloc_681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_681_, 0, v_a_675_);
v___x_680_ = v_reuseFailAlloc_681_;
goto v_reusejp_679_;
}
v_reusejp_679_:
{
return v___x_680_;
}
}
}
else
{
lean_object* v_a_683_; lean_object* v___x_685_; uint8_t v_isShared_686_; uint8_t v_isSharedCheck_690_; 
v_a_683_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_690_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_690_ == 0)
{
v___x_685_ = v___x_674_;
v_isShared_686_ = v_isSharedCheck_690_;
goto v_resetjp_684_;
}
else
{
lean_inc(v_a_683_);
lean_dec(v___x_674_);
v___x_685_ = lean_box(0);
v_isShared_686_ = v_isSharedCheck_690_;
goto v_resetjp_684_;
}
v_resetjp_684_:
{
lean_object* v___x_688_; 
if (v_isShared_686_ == 0)
{
v___x_688_ = v___x_685_;
goto v_reusejp_687_;
}
else
{
lean_object* v_reuseFailAlloc_689_; 
v_reuseFailAlloc_689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_689_, 0, v_a_683_);
v___x_688_ = v_reuseFailAlloc_689_;
goto v_reusejp_687_;
}
v_reusejp_687_:
{
return v___x_688_;
}
}
}
}
else
{
lean_object* v_a_691_; lean_object* v___x_693_; uint8_t v_isShared_694_; uint8_t v_isSharedCheck_698_; 
v_a_691_ = lean_ctor_get(v___x_672_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v___x_672_);
if (v_isSharedCheck_698_ == 0)
{
v___x_693_ = v___x_672_;
v_isShared_694_ = v_isSharedCheck_698_;
goto v_resetjp_692_;
}
else
{
lean_inc(v_a_691_);
lean_dec(v___x_672_);
v___x_693_ = lean_box(0);
v_isShared_694_ = v_isSharedCheck_698_;
goto v_resetjp_692_;
}
v_resetjp_692_:
{
lean_object* v___x_696_; 
if (v_isShared_694_ == 0)
{
v___x_696_ = v___x_693_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v_a_691_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___redArg___boxed(lean_object* v_stx_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_){
_start:
{
lean_object* v_res_705_; 
v_res_705_ = lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___redArg(v_stx_699_, v___y_700_, v___y_701_, v___y_702_, v___y_703_);
lean_dec(v___y_703_);
lean_dec_ref(v___y_702_);
lean_dec(v___y_701_);
lean_dec_ref(v___y_700_);
return v_res_705_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__3(size_t v_sz_706_, size_t v_i_707_, lean_object* v_bs_708_){
_start:
{
uint8_t v___x_709_; 
v___x_709_ = lean_usize_dec_lt(v_i_707_, v_sz_706_);
if (v___x_709_ == 0)
{
return v_bs_708_;
}
else
{
lean_object* v_v_710_; lean_object* v___x_711_; lean_object* v_bs_x27_712_; size_t v___x_713_; size_t v___x_714_; lean_object* v___x_715_; 
v_v_710_ = lean_array_uget(v_bs_708_, v_i_707_);
v___x_711_ = lean_unsigned_to_nat(0u);
v_bs_x27_712_ = lean_array_uset(v_bs_708_, v_i_707_, v___x_711_);
v___x_713_ = ((size_t)1ULL);
v___x_714_ = lean_usize_add(v_i_707_, v___x_713_);
v___x_715_ = lean_array_uset(v_bs_x27_712_, v_i_707_, v_v_710_);
v_i_707_ = v___x_714_;
v_bs_708_ = v___x_715_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__3___boxed(lean_object* v_sz_717_, lean_object* v_i_718_, lean_object* v_bs_719_){
_start:
{
size_t v_sz_boxed_720_; size_t v_i_boxed_721_; lean_object* v_res_722_; 
v_sz_boxed_720_ = lean_unbox_usize(v_sz_717_);
lean_dec(v_sz_717_);
v_i_boxed_721_ = lean_unbox_usize(v_i_718_);
lean_dec(v_i_718_);
v_res_722_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__3(v_sz_boxed_720_, v_i_boxed_721_, v_bs_719_);
return v_res_722_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2(lean_object* v_goalPos_725_, lean_object* v_step_726_, lean_object* v_tail_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lp_aesop_Aesop_Script_Step_sTactic_x3f(v_step_726_);
if (lean_obj_tag(v___x_733_) == 1)
{
lean_object* v_val_734_; lean_object* v___x_736_; uint8_t v_isShared_737_; uint8_t v_isSharedCheck_784_; 
v_val_734_ = lean_ctor_get(v___x_733_, 0);
v_isSharedCheck_784_ = !lean_is_exclusive(v___x_733_);
if (v_isSharedCheck_784_ == 0)
{
v___x_736_ = v___x_733_;
v_isShared_737_ = v_isSharedCheck_784_;
goto v_resetjp_735_;
}
else
{
lean_inc(v_val_734_);
lean_dec(v___x_733_);
v___x_736_ = lean_box(0);
v_isShared_737_ = v_isSharedCheck_784_;
goto v_resetjp_735_;
}
v_resetjp_735_:
{
lean_object* v_numSubgoals_738_; lean_object* v_run_739_; lean_object* v___x_740_; lean_object* v___x_741_; 
v_numSubgoals_738_ = lean_ctor_get(v_val_734_, 0);
lean_inc(v_numSubgoals_738_);
v_run_739_ = lean_ctor_get(v_val_734_, 1);
lean_inc_ref(v_run_739_);
lean_dec(v_val_734_);
v___x_740_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2___closed__0));
v___x_741_ = lp_aesop_Aesop_Script_SScript_takeNConsecutiveFocusAndSolve_x3f(v___x_740_, v_numSubgoals_738_, v_tail_727_);
if (lean_obj_tag(v___x_741_) == 1)
{
lean_object* v_val_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_779_; 
lean_del_object(v___x_736_);
v_val_742_ = lean_ctor_get(v___x_741_, 0);
v_isSharedCheck_779_ = !lean_is_exclusive(v___x_741_);
if (v_isSharedCheck_779_ == 0)
{
v___x_744_ = v___x_741_;
v_isShared_745_ = v_isSharedCheck_779_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_val_742_);
lean_dec(v___x_741_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_779_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v_fst_746_; lean_object* v_snd_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_778_; 
v_fst_746_ = lean_ctor_get(v_val_742_, 0);
v_snd_747_ = lean_ctor_get(v_val_742_, 1);
v_isSharedCheck_778_ = !lean_is_exclusive(v_val_742_);
if (v_isSharedCheck_778_ == 0)
{
v___x_749_ = v_val_742_;
v_isShared_750_ = v_isSharedCheck_778_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_snd_747_);
lean_inc(v_fst_746_);
lean_dec(v_val_742_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_778_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
size_t v_sz_751_; size_t v___x_752_; lean_object* v___x_753_; 
v_sz_751_ = lean_array_size(v_fst_746_);
v___x_752_ = ((size_t)0ULL);
v___x_753_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2_spec__4(v_sz_751_, v___x_752_, v_fst_746_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
if (lean_obj_tag(v___x_753_) == 0)
{
lean_object* v_a_754_; lean_object* v___x_756_; uint8_t v_isShared_757_; uint8_t v_isSharedCheck_769_; 
v_a_754_ = lean_ctor_get(v___x_753_, 0);
v_isSharedCheck_769_ = !lean_is_exclusive(v___x_753_);
if (v_isSharedCheck_769_ == 0)
{
v___x_756_ = v___x_753_;
v_isShared_757_ = v_isSharedCheck_769_;
goto v_resetjp_755_;
}
else
{
lean_inc(v_a_754_);
lean_dec(v___x_753_);
v___x_756_ = lean_box(0);
v_isShared_757_ = v_isSharedCheck_769_;
goto v_resetjp_755_;
}
v_resetjp_755_:
{
lean_object* v___x_758_; lean_object* v_tactic_759_; lean_object* v___x_761_; 
v___x_758_ = lean_apply_1(v_run_739_, v_a_754_);
v_tactic_759_ = lp_aesop_Aesop_Script_mkOnGoal(v_goalPos_725_, v___x_758_);
if (v_isShared_750_ == 0)
{
lean_ctor_set(v___x_749_, 0, v_tactic_759_);
v___x_761_ = v___x_749_;
goto v_reusejp_760_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v_tactic_759_);
lean_ctor_set(v_reuseFailAlloc_768_, 1, v_snd_747_);
v___x_761_ = v_reuseFailAlloc_768_;
goto v_reusejp_760_;
}
v_reusejp_760_:
{
lean_object* v___x_763_; 
if (v_isShared_745_ == 0)
{
lean_ctor_set(v___x_744_, 0, v___x_761_);
v___x_763_ = v___x_744_;
goto v_reusejp_762_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v___x_761_);
v___x_763_ = v_reuseFailAlloc_767_;
goto v_reusejp_762_;
}
v_reusejp_762_:
{
lean_object* v___x_765_; 
if (v_isShared_757_ == 0)
{
lean_ctor_set(v___x_756_, 0, v___x_763_);
v___x_765_ = v___x_756_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v___x_763_);
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
else
{
lean_object* v_a_770_; lean_object* v___x_772_; uint8_t v_isShared_773_; uint8_t v_isSharedCheck_777_; 
lean_del_object(v___x_749_);
lean_dec(v_snd_747_);
lean_del_object(v___x_744_);
lean_dec_ref(v_run_739_);
v_a_770_ = lean_ctor_get(v___x_753_, 0);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_753_);
if (v_isSharedCheck_777_ == 0)
{
v___x_772_ = v___x_753_;
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
else
{
lean_inc(v_a_770_);
lean_dec(v___x_753_);
v___x_772_ = lean_box(0);
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
v_resetjp_771_:
{
lean_object* v___x_775_; 
if (v_isShared_773_ == 0)
{
v___x_775_ = v___x_772_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_a_770_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
}
}
}
}
else
{
lean_object* v___x_780_; lean_object* v___x_782_; 
lean_dec(v___x_741_);
lean_dec_ref(v_run_739_);
v___x_780_ = lean_box(0);
if (v_isShared_737_ == 0)
{
lean_ctor_set_tag(v___x_736_, 0);
lean_ctor_set(v___x_736_, 0, v___x_780_);
v___x_782_ = v___x_736_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_783_; 
v_reuseFailAlloc_783_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_783_, 0, v___x_780_);
v___x_782_ = v_reuseFailAlloc_783_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
return v___x_782_;
}
}
}
}
else
{
lean_object* v___x_785_; lean_object* v___x_786_; 
lean_dec(v___x_733_);
lean_dec(v_tail_727_);
v___x_785_ = lean_box(0);
v___x_786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_786_, 0, v___x_785_);
return v___x_786_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0(lean_object* v_acc_803_, lean_object* v_a_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_){
_start:
{
switch(lean_obj_tag(v_a_804_))
{
case 0:
{
lean_object* v___x_810_; 
v___x_810_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_810_, 0, v_acc_803_);
return v___x_810_;
}
case 1:
{
lean_object* v_goalPos_811_; lean_object* v_step_812_; lean_object* v_tail_813_; lean_object* v___x_814_; 
v_goalPos_811_ = lean_ctor_get(v_a_804_, 0);
lean_inc(v_goalPos_811_);
v_step_812_ = lean_ctor_get(v_a_804_, 1);
lean_inc_ref(v_step_812_);
v_tail_813_ = lean_ctor_get(v_a_804_, 2);
lean_inc_n(v_tail_813_, 2);
lean_dec_ref_known(v_a_804_, 3);
v___x_814_ = lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2(v_goalPos_811_, v_step_812_, v_tail_813_, v___y_805_, v___y_806_, v___y_807_, v___y_808_);
if (lean_obj_tag(v___x_814_) == 0)
{
lean_object* v_a_815_; 
v_a_815_ = lean_ctor_get(v___x_814_, 0);
lean_inc(v_a_815_);
lean_dec_ref_known(v___x_814_, 1);
if (lean_obj_tag(v_a_815_) == 1)
{
lean_object* v_val_816_; lean_object* v_fst_817_; lean_object* v_snd_818_; lean_object* v_script_819_; 
lean_dec(v_tail_813_);
lean_dec_ref(v_step_812_);
lean_dec(v_goalPos_811_);
v_val_816_ = lean_ctor_get(v_a_815_, 0);
lean_inc(v_val_816_);
lean_dec_ref_known(v_a_815_, 1);
v_fst_817_ = lean_ctor_get(v_val_816_, 0);
lean_inc(v_fst_817_);
v_snd_818_ = lean_ctor_get(v_val_816_, 1);
lean_inc(v_snd_818_);
lean_dec(v_val_816_);
v_script_819_ = lean_array_push(v_acc_803_, v_fst_817_);
v_acc_803_ = v_script_819_;
v_a_804_ = v_snd_818_;
goto _start;
}
else
{
lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v_script_823_; 
lean_dec(v_a_815_);
v___x_821_ = lp_aesop_Aesop_Script_Step_uTactic(v_step_812_);
lean_dec_ref(v_step_812_);
v___x_822_ = lp_aesop_Aesop_Script_mkOnGoal(v_goalPos_811_, v___x_821_);
lean_dec(v_goalPos_811_);
v_script_823_ = lean_array_push(v_acc_803_, v___x_822_);
v_acc_803_ = v_script_823_;
v_a_804_ = v_tail_813_;
goto _start;
}
}
else
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_832_; 
lean_dec(v_tail_813_);
lean_dec_ref(v_step_812_);
lean_dec(v_goalPos_811_);
lean_dec_ref(v_acc_803_);
v_a_825_ = lean_ctor_get(v___x_814_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_814_);
if (v_isSharedCheck_832_ == 0)
{
v___x_827_ = v___x_814_;
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_814_);
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
default: 
{
lean_object* v_goalPos_833_; lean_object* v_here_834_; lean_object* v_tail_835_; lean_object* v___x_837_; uint8_t v_isShared_838_; uint8_t v_isSharedCheck_906_; 
v_goalPos_833_ = lean_ctor_get(v_a_804_, 0);
v_here_834_ = lean_ctor_get(v_a_804_, 1);
v_tail_835_ = lean_ctor_get(v_a_804_, 2);
v_isSharedCheck_906_ = !lean_is_exclusive(v_a_804_);
if (v_isSharedCheck_906_ == 0)
{
v___x_837_ = v_a_804_;
v_isShared_838_ = v_isSharedCheck_906_;
goto v_resetjp_836_;
}
else
{
lean_inc(v_tail_835_);
lean_inc(v_here_834_);
lean_inc(v_goalPos_833_);
lean_dec(v_a_804_);
v___x_837_ = lean_box(0);
v_isShared_838_ = v_isSharedCheck_906_;
goto v_resetjp_836_;
}
v_resetjp_836_:
{
lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; 
v___x_839_ = lean_unsigned_to_nat(0u);
v___x_840_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__2));
v___x_841_ = lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0(v___x_840_, v_here_834_, v___y_805_, v___y_806_, v___y_807_, v___y_808_);
if (lean_obj_tag(v___x_841_) == 0)
{
lean_object* v_a_842_; lean_object* v_t_844_; lean_object* v___y_845_; lean_object* v___y_846_; lean_object* v___y_847_; lean_object* v___y_848_; uint8_t v___x_851_; 
v_a_842_ = lean_ctor_get(v___x_841_, 0);
lean_inc(v_a_842_);
lean_dec_ref_known(v___x_841_, 1);
v___x_851_ = lean_nat_dec_eq(v_goalPos_833_, v___x_839_);
if (v___x_851_ == 0)
{
lean_object* v_ref_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v_posLit_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_865_; 
v_ref_852_ = lean_ctor_get(v___y_807_, 5);
v___x_853_ = lean_unsigned_to_nat(1u);
v___x_854_ = lean_nat_add(v_goalPos_833_, v___x_853_);
lean_dec(v_goalPos_833_);
v___x_855_ = l_Nat_reprFast(v___x_854_);
v___x_856_ = lean_box(2);
v_posLit_857_ = l_Lean_Syntax_mkNumLit(v___x_855_, v___x_856_);
v___x_858_ = l_Lean_SourceInfo_fromRef(v_ref_852_, v___x_851_);
v___x_859_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__2));
v___x_860_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__3));
lean_inc_n(v___x_858_, 2);
v___x_861_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_861_, 0, v___x_858_);
lean_ctor_set(v___x_861_, 1, v___x_860_);
v___x_862_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19));
v___x_863_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20, &lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20);
if (v_isShared_838_ == 0)
{
lean_ctor_set_tag(v___x_837_, 1);
lean_ctor_set(v___x_837_, 2, v___x_863_);
lean_ctor_set(v___x_837_, 1, v___x_862_);
lean_ctor_set(v___x_837_, 0, v___x_858_);
v___x_865_ = v___x_837_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v___x_858_);
lean_ctor_set(v_reuseFailAlloc_881_, 1, v___x_862_);
lean_ctor_set(v_reuseFailAlloc_881_, 2, v___x_863_);
v___x_865_ = v_reuseFailAlloc_881_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; 
v___x_866_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__21));
lean_inc_n(v___x_858_, 6);
v___x_867_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_867_, 0, v___x_858_);
lean_ctor_set(v___x_867_, 1, v___x_866_);
v___x_868_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7));
v___x_869_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__11));
v___x_870_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__4));
v___x_871_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_871_, 0, v___x_858_);
lean_ctor_set(v___x_871_, 1, v___x_870_);
v___x_872_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14));
v___x_873_ = l_Lean_Syntax_SepArray_ofElems(v___x_872_, v_a_842_);
lean_dec(v_a_842_);
v___x_874_ = l_Array_append___redArg(v___x_863_, v___x_873_);
lean_dec_ref(v___x_873_);
v___x_875_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_875_, 0, v___x_858_);
lean_ctor_set(v___x_875_, 1, v___x_862_);
lean_ctor_set(v___x_875_, 2, v___x_874_);
v___x_876_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__5));
v___x_877_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_877_, 0, v___x_858_);
lean_ctor_set(v___x_877_, 1, v___x_876_);
v___x_878_ = l_Lean_Syntax_node3(v___x_858_, v___x_869_, v___x_871_, v___x_875_, v___x_877_);
v___x_879_ = l_Lean_Syntax_node1(v___x_858_, v___x_868_, v___x_878_);
v___x_880_ = l_Lean_Syntax_node5(v___x_858_, v___x_859_, v___x_861_, v___x_865_, v_posLit_857_, v___x_867_, v___x_879_);
v_t_844_ = v___x_880_;
v___y_845_ = v___y_805_;
v___y_846_ = v___y_806_;
v___y_847_ = v___y_807_;
v___y_848_ = v___y_808_;
goto v___jp_843_;
}
}
else
{
lean_object* v_ref_882_; uint8_t v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; size_t v_sz_894_; size_t v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_901_; 
lean_dec(v_goalPos_833_);
v_ref_882_ = lean_ctor_get(v___y_807_, 5);
v___x_883_ = 0;
v___x_884_ = l_Lean_SourceInfo_fromRef(v_ref_882_, v___x_883_);
v___x_885_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__1));
v___x_886_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__3));
v___x_887_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__6));
lean_inc_n(v___x_884_, 3);
v___x_888_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_888_, 0, v___x_884_);
lean_ctor_set(v___x_888_, 1, v___x_887_);
v___x_889_ = l_Lean_Syntax_node1(v___x_884_, v___x_886_, v___x_888_);
v___x_890_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7));
v___x_891_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9));
v___x_892_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19));
v___x_893_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20, &lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20);
v_sz_894_ = lean_array_size(v_a_842_);
v___x_895_ = ((size_t)0ULL);
v___x_896_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__3(v_sz_894_, v___x_895_, v_a_842_);
v___x_897_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___closed__8));
v___x_898_ = l_Lean_mkSepArray(v___x_896_, v___x_897_);
lean_dec_ref(v___x_896_);
v___x_899_ = l_Array_append___redArg(v___x_893_, v___x_898_);
lean_dec_ref(v___x_898_);
if (v_isShared_838_ == 0)
{
lean_ctor_set_tag(v___x_837_, 1);
lean_ctor_set(v___x_837_, 2, v___x_899_);
lean_ctor_set(v___x_837_, 1, v___x_892_);
lean_ctor_set(v___x_837_, 0, v___x_884_);
v___x_901_ = v___x_837_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v___x_884_);
lean_ctor_set(v_reuseFailAlloc_905_, 1, v___x_892_);
lean_ctor_set(v_reuseFailAlloc_905_, 2, v___x_899_);
v___x_901_ = v_reuseFailAlloc_905_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; 
lean_inc_n(v___x_884_, 2);
v___x_902_ = l_Lean_Syntax_node1(v___x_884_, v___x_891_, v___x_901_);
v___x_903_ = l_Lean_Syntax_node1(v___x_884_, v___x_890_, v___x_902_);
v___x_904_ = l_Lean_Syntax_node2(v___x_884_, v___x_885_, v___x_889_, v___x_903_);
v_t_844_ = v___x_904_;
v___y_845_ = v___y_805_;
v___y_846_ = v___y_806_;
v___y_847_ = v___y_807_;
v___y_848_ = v___y_808_;
goto v___jp_843_;
}
}
v___jp_843_:
{
lean_object* v___x_849_; 
v___x_849_ = lean_array_push(v_acc_803_, v_t_844_);
v_acc_803_ = v___x_849_;
v_a_804_ = v_tail_835_;
v___y_805_ = v___y_845_;
v___y_806_ = v___y_846_;
v___y_807_ = v___y_847_;
v___y_808_ = v___y_848_;
goto _start;
}
}
else
{
lean_del_object(v___x_837_);
lean_dec(v_tail_835_);
lean_dec(v_goalPos_833_);
lean_dec_ref(v_acc_803_);
return v___x_841_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2_spec__4(size_t v_sz_907_, size_t v_i_908_, lean_object* v_bs_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_){
_start:
{
uint8_t v___x_915_; 
v___x_915_ = lean_usize_dec_lt(v_i_908_, v_sz_907_);
if (v___x_915_ == 0)
{
lean_object* v___x_916_; 
v___x_916_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_916_, 0, v_bs_909_);
return v___x_916_;
}
else
{
lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v_v_919_; lean_object* v___x_920_; 
v___x_917_ = lean_unsigned_to_nat(0u);
v___x_918_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__2));
v_v_919_ = lean_array_uget_borrowed(v_bs_909_, v_i_908_);
lean_inc(v_v_919_);
v___x_920_ = lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0(v___x_918_, v_v_919_, v___y_910_, v___y_911_, v___y_912_, v___y_913_);
if (lean_obj_tag(v___x_920_) == 0)
{
lean_object* v_a_921_; lean_object* v_bs_x27_922_; size_t v___x_923_; size_t v___x_924_; lean_object* v___x_925_; 
v_a_921_ = lean_ctor_get(v___x_920_, 0);
lean_inc(v_a_921_);
lean_dec_ref_known(v___x_920_, 1);
v_bs_x27_922_ = lean_array_uset(v_bs_909_, v_i_908_, v___x_917_);
v___x_923_ = ((size_t)1ULL);
v___x_924_ = lean_usize_add(v_i_908_, v___x_923_);
v___x_925_ = lean_array_uset(v_bs_x27_922_, v_i_908_, v_a_921_);
v_i_908_ = v___x_924_;
v_bs_909_ = v___x_925_;
goto _start;
}
else
{
lean_object* v_a_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_934_; 
lean_dec_ref(v_bs_909_);
v_a_927_ = lean_ctor_get(v___x_920_, 0);
v_isSharedCheck_934_ = !lean_is_exclusive(v___x_920_);
if (v_isSharedCheck_934_ == 0)
{
v___x_929_ = v___x_920_;
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_a_927_);
lean_dec(v___x_920_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_932_; 
if (v_isShared_930_ == 0)
{
v___x_932_ = v___x_929_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_a_927_);
v___x_932_ = v_reuseFailAlloc_933_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
return v___x_932_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_sz_935_, lean_object* v_i_936_, lean_object* v_bs_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_){
_start:
{
size_t v_sz_boxed_943_; size_t v_i_boxed_944_; lean_object* v_res_945_; 
v_sz_boxed_943_ = lean_unbox_usize(v_sz_935_);
lean_dec(v_sz_935_);
v_i_boxed_944_ = lean_unbox_usize(v_i_936_);
lean_dec(v_i_936_);
v_res_945_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2_spec__4(v_sz_boxed_943_, v_i_boxed_944_, v_bs_937_, v___y_938_, v___y_939_, v___y_940_, v___y_941_);
lean_dec(v___y_941_);
lean_dec_ref(v___y_940_);
lean_dec(v___y_939_);
lean_dec_ref(v___y_938_);
return v_res_945_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2___boxed(lean_object* v_goalPos_946_, lean_object* v_step_947_, lean_object* v_tail_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
lean_object* v_res_954_; 
v_res_954_ = lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_renderSTactic_x3f___at___00__private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0_spec__2(v_goalPos_946_, v_step_947_, v_tail_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
lean_dec(v___y_952_);
lean_dec_ref(v___y_951_);
lean_dec(v___y_950_);
lean_dec_ref(v___y_949_);
lean_dec_ref(v_step_947_);
lean_dec(v_goalPos_946_);
return v_res_954_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0___boxed(lean_object* v_acc_955_, lean_object* v_a_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_){
_start:
{
lean_object* v_res_962_; 
v_res_962_ = lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0(v_acc_955_, v_a_956_, v___y_957_, v___y_958_, v___y_959_, v___y_960_);
lean_dec(v___y_960_);
lean_dec_ref(v___y_959_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
return v_res_962_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0(lean_object* v_script_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_){
_start:
{
lean_object* v___x_969_; lean_object* v___x_970_; 
v___x_969_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3___closed__2));
v___x_970_ = lp_aesop___private_Aesop_Script_SScript_0__Aesop_Script_SScript_render_go___at___00Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0_spec__0(v___x_969_, v_script_963_, v___y_964_, v___y_965_, v___y_966_, v___y_967_);
return v___x_970_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0___boxed(lean_object* v_script_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_){
_start:
{
lean_object* v_res_977_; 
v_res_977_ = lp_aesop_Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0(v_script_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_);
lean_dec(v___y_975_);
lean_dec_ref(v___y_974_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
return v_res_977_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_optimize(lean_object* v_uscript_978_, uint8_t v_proofHasMVar_979_, lean_object* v_preState_980_, lean_object* v_goal_981_, lean_object* v_a_982_, lean_object* v_a_983_, lean_object* v_a_984_, lean_object* v_a_985_){
_start:
{
lean_object* v_structureResult_x3f_988_; lean_object* v___y_989_; lean_object* v___y_990_; lean_object* v___y_991_; lean_object* v___y_992_; uint8_t v___y_1064_; lean_object* v_options_1075_; lean_object* v___x_1076_; uint8_t v___x_1077_; uint8_t v___y_1079_; 
v_options_1075_ = lean_ctor_get(v_a_984_, 2);
v___x_1076_ = lp_aesop_Aesop_aesop_dev_dynamicStructuring;
v___x_1077_ = lp_aesop_Lean_Option_get___at___00Aesop_Script_UScript_optimize_spec__2(v_options_1075_, v___x_1076_);
if (v___x_1077_ == 0)
{
v___y_1064_ = v___x_1077_;
goto v___jp_1063_;
}
else
{
lean_object* v___x_1080_; uint8_t v___x_1081_; 
v___x_1080_ = lp_aesop_Aesop_aesop_dev_optimizedDynamicStructuring;
v___x_1081_ = lp_aesop_Lean_Option_get___at___00Aesop_Script_UScript_optimize_spec__2(v_options_1075_, v___x_1080_);
if (v___x_1081_ == 0)
{
v___y_1079_ = v___x_1081_;
goto v___jp_1078_;
}
else
{
if (v_proofHasMVar_979_ == 0)
{
v___y_1079_ = v___x_1081_;
goto v___jp_1078_;
}
else
{
v___y_1064_ = v___x_1077_;
goto v___jp_1063_;
}
}
}
v___jp_987_:
{
if (lean_obj_tag(v_structureResult_x3f_988_) == 1)
{
lean_object* v_val_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1049_; 
v_val_993_ = lean_ctor_get(v_structureResult_x3f_988_, 0);
v_isSharedCheck_1049_ = !lean_is_exclusive(v_structureResult_x3f_988_);
if (v_isSharedCheck_1049_ == 0)
{
v___x_995_ = v_structureResult_x3f_988_;
v_isShared_996_ = v_isSharedCheck_1049_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_val_993_);
lean_dec(v_structureResult_x3f_988_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1049_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v_fst_997_; lean_object* v_snd_998_; lean_object* v___x_1000_; uint8_t v_isShared_1001_; uint8_t v_isSharedCheck_1048_; 
v_fst_997_ = lean_ctor_get(v_val_993_, 0);
v_snd_998_ = lean_ctor_get(v_val_993_, 1);
v_isSharedCheck_1048_ = !lean_is_exclusive(v_val_993_);
if (v_isSharedCheck_1048_ == 0)
{
v___x_1000_ = v_val_993_;
v_isShared_1001_ = v_isSharedCheck_1048_;
goto v_resetjp_999_;
}
else
{
lean_inc(v_snd_998_);
lean_inc(v_fst_997_);
lean_dec(v_val_993_);
v___x_1000_ = lean_box(0);
v_isShared_1001_ = v_isSharedCheck_1048_;
goto v_resetjp_999_;
}
v_resetjp_999_:
{
lean_object* v___x_1002_; 
v___x_1002_ = lp_aesop_Aesop_Script_SScript_render___at___00Aesop_Script_UScript_optimize_spec__0(v_fst_997_, v___y_989_, v___y_990_, v___y_991_, v___y_992_);
if (lean_obj_tag(v___x_1002_) == 0)
{
lean_object* v_a_1003_; lean_object* v_ref_1004_; uint8_t v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; 
v_a_1003_ = lean_ctor_get(v___x_1002_, 0);
lean_inc(v_a_1003_);
lean_dec_ref_known(v___x_1002_, 1);
v_ref_1004_ = lean_ctor_get(v___y_991_, 5);
v___x_1005_ = 0;
v___x_1006_ = l_Lean_SourceInfo_fromRef(v_ref_1004_, v___x_1005_);
v___x_1007_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__7));
v___x_1008_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__9));
v___x_1009_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__19));
v___x_1010_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20, &lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__20);
v___x_1011_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__2___closed__14));
v___x_1012_ = l_Lean_Syntax_SepArray_ofElems(v___x_1011_, v_a_1003_);
lean_dec(v_a_1003_);
v___x_1013_ = l_Array_append___redArg(v___x_1010_, v___x_1012_);
lean_dec_ref(v___x_1012_);
lean_inc_n(v___x_1006_, 2);
v___x_1014_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1014_, 0, v___x_1006_);
lean_ctor_set(v___x_1014_, 1, v___x_1009_);
lean_ctor_set(v___x_1014_, 2, v___x_1013_);
v___x_1015_ = l_Lean_Syntax_node1(v___x_1006_, v___x_1008_, v___x_1014_);
v___x_1016_ = l_Lean_Syntax_node1(v___x_1006_, v___x_1007_, v___x_1015_);
v___x_1017_ = lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___redArg(v___x_1016_, v___y_989_, v___y_990_, v___y_991_, v___y_992_);
if (lean_obj_tag(v___x_1017_) == 0)
{
lean_object* v_a_1018_; lean_object* v___x_1020_; uint8_t v_isShared_1021_; uint8_t v_isSharedCheck_1031_; 
v_a_1018_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1031_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1031_ == 0)
{
v___x_1020_ = v___x_1017_;
v_isShared_1021_ = v_isSharedCheck_1031_;
goto v_resetjp_1019_;
}
else
{
lean_inc(v_a_1018_);
lean_dec(v___x_1017_);
v___x_1020_ = lean_box(0);
v_isShared_1021_ = v_isSharedCheck_1031_;
goto v_resetjp_1019_;
}
v_resetjp_1019_:
{
lean_object* v___x_1023_; 
if (v_isShared_1001_ == 0)
{
lean_ctor_set(v___x_1000_, 0, v_a_1018_);
v___x_1023_ = v___x_1000_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1030_; 
v_reuseFailAlloc_1030_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1030_, 0, v_a_1018_);
lean_ctor_set(v_reuseFailAlloc_1030_, 1, v_snd_998_);
v___x_1023_ = v_reuseFailAlloc_1030_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
lean_object* v___x_1025_; 
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 0, v___x_1023_);
v___x_1025_ = v___x_995_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1029_; 
v_reuseFailAlloc_1029_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1029_, 0, v___x_1023_);
v___x_1025_ = v_reuseFailAlloc_1029_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
lean_object* v___x_1027_; 
if (v_isShared_1021_ == 0)
{
lean_ctor_set(v___x_1020_, 0, v___x_1025_);
v___x_1027_ = v___x_1020_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v___x_1025_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
}
}
}
else
{
lean_object* v_a_1032_; lean_object* v___x_1034_; uint8_t v_isShared_1035_; uint8_t v_isSharedCheck_1039_; 
lean_del_object(v___x_1000_);
lean_dec(v_snd_998_);
lean_del_object(v___x_995_);
v_a_1032_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1039_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1039_ == 0)
{
v___x_1034_ = v___x_1017_;
v_isShared_1035_ = v_isSharedCheck_1039_;
goto v_resetjp_1033_;
}
else
{
lean_inc(v_a_1032_);
lean_dec(v___x_1017_);
v___x_1034_ = lean_box(0);
v_isShared_1035_ = v_isSharedCheck_1039_;
goto v_resetjp_1033_;
}
v_resetjp_1033_:
{
lean_object* v___x_1037_; 
if (v_isShared_1035_ == 0)
{
v___x_1037_ = v___x_1034_;
goto v_reusejp_1036_;
}
else
{
lean_object* v_reuseFailAlloc_1038_; 
v_reuseFailAlloc_1038_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1038_, 0, v_a_1032_);
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
lean_del_object(v___x_1000_);
lean_dec(v_snd_998_);
lean_del_object(v___x_995_);
v_a_1040_ = lean_ctor_get(v___x_1002_, 0);
v_isSharedCheck_1047_ = !lean_is_exclusive(v___x_1002_);
if (v_isSharedCheck_1047_ == 0)
{
v___x_1042_ = v___x_1002_;
v_isShared_1043_ = v_isSharedCheck_1047_;
goto v_resetjp_1041_;
}
else
{
lean_inc(v_a_1040_);
lean_dec(v___x_1002_);
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
else
{
lean_object* v___x_1050_; lean_object* v___x_1051_; 
lean_dec(v_structureResult_x3f_988_);
v___x_1050_ = lean_box(0);
v___x_1051_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1051_, 0, v___x_1050_);
return v___x_1051_;
}
}
v___jp_1052_:
{
lean_object* v___x_1053_; 
v___x_1053_ = lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureStatic(v_uscript_978_, v_proofHasMVar_979_, v_preState_980_, v_goal_981_, v_a_982_, v_a_983_, v_a_984_, v_a_985_);
if (lean_obj_tag(v___x_1053_) == 0)
{
lean_object* v_a_1054_; 
v_a_1054_ = lean_ctor_get(v___x_1053_, 0);
lean_inc(v_a_1054_);
lean_dec_ref_known(v___x_1053_, 1);
v_structureResult_x3f_988_ = v_a_1054_;
v___y_989_ = v_a_982_;
v___y_990_ = v_a_983_;
v___y_991_ = v_a_984_;
v___y_992_ = v_a_985_;
goto v___jp_987_;
}
else
{
lean_object* v_a_1055_; lean_object* v___x_1057_; uint8_t v_isShared_1058_; uint8_t v_isSharedCheck_1062_; 
v_a_1055_ = lean_ctor_get(v___x_1053_, 0);
v_isSharedCheck_1062_ = !lean_is_exclusive(v___x_1053_);
if (v_isSharedCheck_1062_ == 0)
{
v___x_1057_ = v___x_1053_;
v_isShared_1058_ = v_isSharedCheck_1062_;
goto v_resetjp_1056_;
}
else
{
lean_inc(v_a_1055_);
lean_dec(v___x_1053_);
v___x_1057_ = lean_box(0);
v_isShared_1058_ = v_isSharedCheck_1062_;
goto v_resetjp_1056_;
}
v_resetjp_1056_:
{
lean_object* v___x_1060_; 
if (v_isShared_1058_ == 0)
{
v___x_1060_ = v___x_1057_;
goto v_reusejp_1059_;
}
else
{
lean_object* v_reuseFailAlloc_1061_; 
v_reuseFailAlloc_1061_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1061_, 0, v_a_1055_);
v___x_1060_ = v_reuseFailAlloc_1061_;
goto v_reusejp_1059_;
}
v_reusejp_1059_:
{
return v___x_1060_;
}
}
}
}
v___jp_1063_:
{
if (v___y_1064_ == 0)
{
goto v___jp_1052_;
}
else
{
lean_object* v___x_1065_; 
v___x_1065_ = lp_aesop___private_Aesop_Script_Main_0__Aesop_Script_UScript_optimize_structureDynamic(v_uscript_978_, v_proofHasMVar_979_, v_preState_980_, v_goal_981_, v_a_982_, v_a_983_, v_a_984_, v_a_985_);
if (lean_obj_tag(v___x_1065_) == 0)
{
lean_object* v_a_1066_; 
v_a_1066_ = lean_ctor_get(v___x_1065_, 0);
lean_inc(v_a_1066_);
lean_dec_ref_known(v___x_1065_, 1);
v_structureResult_x3f_988_ = v_a_1066_;
v___y_989_ = v_a_982_;
v___y_990_ = v_a_983_;
v___y_991_ = v_a_984_;
v___y_992_ = v_a_985_;
goto v___jp_987_;
}
else
{
lean_object* v_a_1067_; lean_object* v___x_1069_; uint8_t v_isShared_1070_; uint8_t v_isSharedCheck_1074_; 
v_a_1067_ = lean_ctor_get(v___x_1065_, 0);
v_isSharedCheck_1074_ = !lean_is_exclusive(v___x_1065_);
if (v_isSharedCheck_1074_ == 0)
{
v___x_1069_ = v___x_1065_;
v_isShared_1070_ = v_isSharedCheck_1074_;
goto v_resetjp_1068_;
}
else
{
lean_inc(v_a_1067_);
lean_dec(v___x_1065_);
v___x_1069_ = lean_box(0);
v_isShared_1070_ = v_isSharedCheck_1074_;
goto v_resetjp_1068_;
}
v_resetjp_1068_:
{
lean_object* v___x_1072_; 
if (v_isShared_1070_ == 0)
{
v___x_1072_ = v___x_1069_;
goto v_reusejp_1071_;
}
else
{
lean_object* v_reuseFailAlloc_1073_; 
v_reuseFailAlloc_1073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1073_, 0, v_a_1067_);
v___x_1072_ = v_reuseFailAlloc_1073_;
goto v_reusejp_1071_;
}
v_reusejp_1071_:
{
return v___x_1072_;
}
}
}
}
}
v___jp_1078_:
{
if (v___y_1079_ == 0)
{
v___y_1064_ = v___x_1077_;
goto v___jp_1063_;
}
else
{
goto v___jp_1052_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_optimize___boxed(lean_object* v_uscript_1082_, lean_object* v_proofHasMVar_1083_, lean_object* v_preState_1084_, lean_object* v_goal_1085_, lean_object* v_a_1086_, lean_object* v_a_1087_, lean_object* v_a_1088_, lean_object* v_a_1089_, lean_object* v_a_1090_){
_start:
{
uint8_t v_proofHasMVar_boxed_1091_; lean_object* v_res_1092_; 
v_proofHasMVar_boxed_1091_ = lean_unbox(v_proofHasMVar_1083_);
v_res_1092_ = lp_aesop_Aesop_Script_UScript_optimize(v_uscript_1082_, v_proofHasMVar_boxed_1091_, v_preState_1084_, v_goal_1085_, v_a_1086_, v_a_1087_, v_a_1088_, v_a_1089_);
lean_dec(v_a_1089_);
lean_dec_ref(v_a_1088_);
lean_dec(v_a_1087_);
lean_dec_ref(v_a_1086_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1(lean_object* v_kind_1093_, lean_object* v_stx_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_){
_start:
{
lean_object* v___x_1100_; 
v___x_1100_ = lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___redArg(v_stx_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_);
return v___x_1100_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1___boxed(lean_object* v_kind_1101_, lean_object* v_stx_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_){
_start:
{
lean_object* v_res_1108_; 
v_res_1108_ = lp_aesop_Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1(v_kind_1101_, v_stx_1102_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_);
lean_dec(v___y_1106_);
lean_dec_ref(v___y_1105_);
lean_dec(v___y_1104_);
lean_dec_ref(v___y_1103_);
lean_dec(v_kind_1101_);
return v_res_1108_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14(lean_object* v_tacs_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
lean_object* v___x_1115_; 
v___x_1115_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___redArg(v_tacs_1109_, v___y_1112_);
return v___x_1115_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14___boxed(lean_object* v_tacs_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__14(v_tacs_1116_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_);
lean_dec(v___y_1120_);
lean_dec_ref(v___y_1119_);
lean_dec(v___y_1118_);
lean_dec_ref(v___y_1117_);
lean_dec_ref(v_tacs_1116_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__10(lean_object* v_inst_1123_, lean_object* v_R_1124_, lean_object* v_a_1125_, lean_object* v_b_1126_, lean_object* v_c_1127_){
_start:
{
lean_object* v___x_1128_; 
v___x_1128_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__10___redArg(v_a_1125_, v_b_1126_);
return v___x_1128_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11(lean_object* v_00_u03b2_1129_, lean_object* v_m_1130_, lean_object* v_a_1131_){
_start:
{
uint8_t v___x_1132_; 
v___x_1132_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___redArg(v_m_1130_, v_a_1131_);
return v___x_1132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11___boxed(lean_object* v_00_u03b2_1133_, lean_object* v_m_1134_, lean_object* v_a_1135_){
_start:
{
uint8_t v_res_1136_; lean_object* v_r_1137_; 
v_res_1136_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11(v_00_u03b2_1133_, v_m_1134_, v_a_1135_);
lean_dec(v_a_1135_);
lean_dec_ref(v_m_1134_);
v_r_1137_ = lean_box(v_res_1136_);
return v_r_1137_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12(lean_object* v_usedNames_1138_, lean_object* v_as_1139_, size_t v_sz_1140_, size_t v_i_1141_, lean_object* v_b_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v___x_1148_; 
v___x_1148_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___redArg(v_usedNames_1138_, v_as_1139_, v_sz_1140_, v_i_1141_, v_b_1142_);
return v___x_1148_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12___boxed(lean_object* v_usedNames_1149_, lean_object* v_as_1150_, lean_object* v_sz_1151_, lean_object* v_i_1152_, lean_object* v_b_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_){
_start:
{
size_t v_sz_boxed_1159_; size_t v_i_boxed_1160_; lean_object* v_res_1161_; 
v_sz_boxed_1159_ = lean_unbox_usize(v_sz_1151_);
lean_dec(v_sz_1151_);
v_i_boxed_1160_ = lean_unbox_usize(v_i_1152_);
lean_dec(v_i_1152_);
v_res_1161_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__12(v_usedNames_1149_, v_as_1150_, v_sz_boxed_1159_, v_i_boxed_1160_, v_b_1153_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_);
lean_dec(v___y_1157_);
lean_dec_ref(v___y_1156_);
lean_dec(v___y_1155_);
lean_dec_ref(v___y_1154_);
lean_dec_ref(v_as_1150_);
lean_dec_ref(v_usedNames_1149_);
return v_res_1161_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__13(lean_object* v_inst_1162_, lean_object* v_R_1163_, lean_object* v_a_1164_, lean_object* v_b_1165_){
_start:
{
lean_object* v___x_1166_; 
v___x_1166_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__13___redArg(v_a_1164_, v_b_1165_);
return v___x_1166_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13(lean_object* v_00_u03b2_1167_, lean_object* v_a_1168_, lean_object* v_x_1169_){
_start:
{
uint8_t v___x_1170_; 
v___x_1170_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___redArg(v_a_1168_, v_x_1169_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13___boxed(lean_object* v_00_u03b2_1171_, lean_object* v_a_1172_, lean_object* v_x_1173_){
_start:
{
uint8_t v_res_1174_; lean_object* v_r_1175_; 
v_res_1174_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_optimizeInitialRenameI___at___00Aesop_optimizeSyntax___at___00Aesop_Script_UScript_optimize_spec__1_spec__3_spec__11_spec__13(v_00_u03b2_1171_, v_a_1172_, v_x_1173_);
lean_dec(v_x_1173_);
lean_dec(v_a_1172_);
v_r_1175_ = lean_box(v_res_1174_);
return v_r_1175_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__0(lean_object* v_fst_1176_, lean_object* v_preState_1177_, lean_object* v_goal_1178_, uint8_t v_expectCompleteProof_1179_, lean_object* v_inst_1180_, lean_object* v_____r_1181_){
_start:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; 
v___x_1182_ = lean_box(v_expectCompleteProof_1179_);
v___x_1183_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkRenderedScriptIfEnabled___boxed), 9, 4);
lean_closure_set(v___x_1183_, 0, v_fst_1176_);
lean_closure_set(v___x_1183_, 1, v_preState_1177_);
lean_closure_set(v___x_1183_, 2, v_goal_1178_);
lean_closure_set(v___x_1183_, 3, v___x_1182_);
v___x_1184_ = lean_apply_2(v_inst_1180_, lean_box(0), v___x_1183_);
return v___x_1184_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__0___boxed(lean_object* v_fst_1185_, lean_object* v_preState_1186_, lean_object* v_goal_1187_, lean_object* v_expectCompleteProof_1188_, lean_object* v_inst_1189_, lean_object* v_____r_1190_){
_start:
{
uint8_t v_expectCompleteProof_boxed_1191_; lean_object* v_res_1192_; 
v_expectCompleteProof_boxed_1191_ = lean_unbox(v_expectCompleteProof_1188_);
v_res_1192_ = lp_aesop_Aesop_checkAndTraceScript___redArg___lam__0(v_fst_1185_, v_preState_1186_, v_goal_1187_, v_expectCompleteProof_boxed_1191_, v_inst_1189_, v_____r_1190_);
return v_res_1192_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__1(lean_object* v___f_1193_, lean_object* v_____r_1194_){
_start:
{
lean_object* v___x_1195_; 
v___x_1195_ = lean_apply_1(v___f_1193_, v_____r_1194_);
return v___x_1195_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__2(lean_object* v_fst_1196_, lean_object* v_inst_1197_, lean_object* v_toBind_1198_, lean_object* v___f_1199_, lean_object* v_____do__lift_1200_){
_start:
{
lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; 
v___x_1201_ = lean_box(0);
v___x_1202_ = lean_alloc_closure((void*)(lp_aesop_Aesop_addTryThisTacticSeqSuggestion___boxed), 8, 3);
lean_closure_set(v___x_1202_, 0, v_____do__lift_1200_);
lean_closure_set(v___x_1202_, 1, v_fst_1196_);
lean_closure_set(v___x_1202_, 2, v___x_1201_);
v___x_1203_ = lean_apply_2(v_inst_1197_, lean_box(0), v___x_1202_);
v___x_1204_ = lean_apply_4(v_toBind_1198_, lean_box(0), lean_box(0), v___x_1203_, v___f_1199_);
return v___x_1204_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__3(lean_object* v_options_1205_, lean_object* v___f_1206_, lean_object* v_inst_1207_, lean_object* v_toBind_1208_, lean_object* v___f_1209_, lean_object* v_____r_1210_){
_start:
{
lean_object* v_toOptions_1211_; uint8_t v_traceScript_1212_; 
v_toOptions_1211_ = lean_ctor_get(v_options_1205_, 0);
v_traceScript_1212_ = lean_ctor_get_uint8(v_toOptions_1211_, sizeof(void*)*6 + 6);
if (v_traceScript_1212_ == 0)
{
lean_object* v___x_1213_; lean_object* v___x_1214_; 
lean_dec(v___f_1209_);
lean_dec(v_toBind_1208_);
lean_dec_ref(v_inst_1207_);
v___x_1213_ = lean_box(0);
v___x_1214_ = lean_apply_1(v___f_1206_, v___x_1213_);
return v___x_1214_;
}
else
{
lean_object* v_getRef_1215_; lean_object* v___x_1216_; 
lean_dec(v___f_1206_);
v_getRef_1215_ = lean_ctor_get(v_inst_1207_, 0);
lean_inc(v_getRef_1215_);
lean_dec_ref(v_inst_1207_);
v___x_1216_ = lean_apply_4(v_toBind_1208_, lean_box(0), lean_box(0), v_getRef_1215_, v___f_1209_);
return v___x_1216_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__3___boxed(lean_object* v_options_1217_, lean_object* v___f_1218_, lean_object* v_inst_1219_, lean_object* v_toBind_1220_, lean_object* v___f_1221_, lean_object* v_____r_1222_){
_start:
{
lean_object* v_res_1223_; 
v_res_1223_ = lp_aesop_Aesop_checkAndTraceScript___redArg___lam__3(v_options_1217_, v___f_1218_, v_inst_1219_, v_toBind_1220_, v___f_1221_, v_____r_1222_);
lean_dec_ref(v_options_1217_);
return v_res_1223_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__1(void){
_start:
{
lean_object* v___x_1225_; lean_object* v___x_1226_; 
v___x_1225_ = ((lean_object*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__0));
v___x_1226_ = l_Lean_stringToMessageData(v___x_1225_);
return v___x_1226_;
}
}
static lean_object* _init_lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__3(void){
_start:
{
lean_object* v___x_1228_; lean_object* v___x_1229_; 
v___x_1228_ = ((lean_object*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__2));
v___x_1229_ = l_Lean_stringToMessageData(v___x_1228_);
return v___x_1229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4(uint8_t v_traceScript_1230_, lean_object* v_toApplicative_1231_, lean_object* v_tacticName_1232_, lean_object* v_inst_1233_, lean_object* v_inst_1234_, lean_object* v_inst_1235_, lean_object* v_toMonadOptions_1236_, lean_object* v___x_1237_, lean_object* v_inst_1238_, uint8_t v_____do__lift_1239_){
_start:
{
if (v_____do__lift_1239_ == 0)
{
lean_dec_ref(v_inst_1238_);
if (v_traceScript_1230_ == 0)
{
lean_object* v_toPure_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; 
lean_dec(v_toMonadOptions_1236_);
lean_dec(v_inst_1235_);
lean_dec_ref(v_inst_1234_);
lean_dec_ref(v_inst_1233_);
lean_dec_ref(v_tacticName_1232_);
v_toPure_1240_ = lean_ctor_get(v_toApplicative_1231_, 1);
lean_inc(v_toPure_1240_);
lean_dec_ref(v_toApplicative_1231_);
v___x_1241_ = lean_box(0);
v___x_1242_ = lean_apply_2(v_toPure_1240_, lean_box(0), v___x_1241_);
return v___x_1242_;
}
else
{
lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; 
lean_dec_ref(v_toApplicative_1231_);
v___x_1243_ = l_Lean_stringToMessageData(v_tacticName_1232_);
v___x_1244_ = lean_obj_once(&lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__1, &lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__1_once, _init_lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__1);
v___x_1245_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1245_, 0, v___x_1243_);
lean_ctor_set(v___x_1245_, 1, v___x_1244_);
v___x_1246_ = l_Lean_logWarning___redArg(v_inst_1233_, v_inst_1234_, v_inst_1235_, v_toMonadOptions_1236_, v___x_1245_);
return v___x_1246_;
}
}
else
{
lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; 
lean_dec(v_toMonadOptions_1236_);
lean_dec(v_inst_1235_);
lean_dec_ref(v_inst_1234_);
lean_dec_ref(v_tacticName_1232_);
lean_dec_ref(v_toApplicative_1231_);
v___x_1247_ = lp_aesop_Aesop_Check_name(v___x_1237_);
v___x_1248_ = l_Lean_MessageData_ofName(v___x_1247_);
v___x_1249_ = lean_obj_once(&lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__3, &lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__3_once, _init_lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___closed__3);
v___x_1250_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1250_, 0, v___x_1248_);
lean_ctor_set(v___x_1250_, 1, v___x_1249_);
v___x_1251_ = l_Lean_throwError___redArg(v_inst_1233_, v_inst_1238_, v___x_1250_);
return v___x_1251_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___boxed(lean_object* v_traceScript_1252_, lean_object* v_toApplicative_1253_, lean_object* v_tacticName_1254_, lean_object* v_inst_1255_, lean_object* v_inst_1256_, lean_object* v_inst_1257_, lean_object* v_toMonadOptions_1258_, lean_object* v___x_1259_, lean_object* v_inst_1260_, lean_object* v_____do__lift_1261_){
_start:
{
uint8_t v_traceScript_boxed_1262_; uint8_t v_____do__lift_601__boxed_1263_; lean_object* v_res_1264_; 
v_traceScript_boxed_1262_ = lean_unbox(v_traceScript_1252_);
v_____do__lift_601__boxed_1263_ = lean_unbox(v_____do__lift_1261_);
v_res_1264_ = lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4(v_traceScript_boxed_1262_, v_toApplicative_1253_, v_tacticName_1254_, v_inst_1255_, v_inst_1256_, v_inst_1257_, v_toMonadOptions_1258_, v___x_1259_, v_inst_1260_, v_____do__lift_601__boxed_1263_);
lean_dec_ref(v___x_1259_);
return v_res_1264_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__5(uint8_t v_traceScript_1265_, lean_object* v_toApplicative_1266_, lean_object* v_tacticName_1267_, lean_object* v_inst_1268_, lean_object* v_inst_1269_, lean_object* v_inst_1270_, lean_object* v_toMonadOptions_1271_, lean_object* v_inst_1272_, lean_object* v_toBind_1273_, lean_object* v_____r_1274_){
_start:
{
lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___f_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; 
v___x_1275_ = lp_aesop_Aesop_Check_script;
v___x_1276_ = lean_box(v_traceScript_1265_);
lean_inc(v_toMonadOptions_1271_);
lean_inc_ref(v_inst_1268_);
v___f_1277_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__4___boxed), 10, 9);
lean_closure_set(v___f_1277_, 0, v___x_1276_);
lean_closure_set(v___f_1277_, 1, v_toApplicative_1266_);
lean_closure_set(v___f_1277_, 2, v_tacticName_1267_);
lean_closure_set(v___f_1277_, 3, v_inst_1268_);
lean_closure_set(v___f_1277_, 4, v_inst_1269_);
lean_closure_set(v___f_1277_, 5, v_inst_1270_);
lean_closure_set(v___f_1277_, 6, v_toMonadOptions_1271_);
lean_closure_set(v___f_1277_, 7, v___x_1275_);
lean_closure_set(v___f_1277_, 8, v_inst_1272_);
v___x_1278_ = lp_aesop_Aesop_Check_isEnabled___redArg(v_inst_1268_, v_toMonadOptions_1271_, v___x_1275_);
v___x_1279_ = lean_apply_4(v_toBind_1273_, lean_box(0), lean_box(0), v___x_1278_, v___f_1277_);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__5___boxed(lean_object* v_traceScript_1280_, lean_object* v_toApplicative_1281_, lean_object* v_tacticName_1282_, lean_object* v_inst_1283_, lean_object* v_inst_1284_, lean_object* v_inst_1285_, lean_object* v_toMonadOptions_1286_, lean_object* v_inst_1287_, lean_object* v_toBind_1288_, lean_object* v_____r_1289_){
_start:
{
uint8_t v_traceScript_boxed_1290_; lean_object* v_res_1291_; 
v_traceScript_boxed_1290_ = lean_unbox(v_traceScript_1280_);
v_res_1291_ = lp_aesop_Aesop_checkAndTraceScript___redArg___lam__5(v_traceScript_boxed_1290_, v_toApplicative_1281_, v_tacticName_1282_, v_inst_1283_, v_inst_1284_, v_inst_1285_, v_toMonadOptions_1286_, v_inst_1287_, v_toBind_1288_, v_____r_1289_);
return v_res_1291_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__7(lean_object* v_tacticSeq_1292_, lean_object* v_inst_1293_, lean_object* v_toBind_1294_, lean_object* v___f_1295_, lean_object* v_____do__lift_1296_){
_start:
{
lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; 
v___x_1297_ = lean_box(0);
v___x_1298_ = lean_alloc_closure((void*)(lp_aesop_Aesop_addTryThisTacticSeqSuggestion___boxed), 8, 3);
lean_closure_set(v___x_1298_, 0, v_____do__lift_1296_);
lean_closure_set(v___x_1298_, 1, v_tacticSeq_1292_);
lean_closure_set(v___x_1298_, 2, v___x_1297_);
v___x_1299_ = lean_apply_2(v_inst_1293_, lean_box(0), v___x_1298_);
v___x_1300_ = lean_apply_4(v_toBind_1294_, lean_box(0), lean_box(0), v___x_1299_, v___f_1295_);
return v___x_1300_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___lam__6(lean_object* v_inst_1301_, lean_object* v_inst_1302_, lean_object* v_toBind_1303_, lean_object* v___f_1304_, lean_object* v_tacticSeq_1305_){
_start:
{
lean_object* v_getRef_1306_; lean_object* v___f_1307_; lean_object* v___x_1308_; 
v_getRef_1306_ = lean_ctor_get(v_inst_1301_, 0);
lean_inc(v_getRef_1306_);
lean_dec_ref(v_inst_1301_);
lean_inc(v_toBind_1303_);
v___f_1307_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__7), 5, 4);
lean_closure_set(v___f_1307_, 0, v_tacticSeq_1305_);
lean_closure_set(v___f_1307_, 1, v_inst_1302_);
lean_closure_set(v___f_1307_, 2, v_toBind_1303_);
lean_closure_set(v___f_1307_, 3, v___f_1304_);
v___x_1308_ = lean_apply_4(v_toBind_1303_, lean_box(0), lean_box(0), v_getRef_1306_, v___f_1307_);
return v___x_1308_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg(lean_object* v_inst_1309_, lean_object* v_inst_1310_, lean_object* v_inst_1311_, lean_object* v_inst_1312_, lean_object* v_inst_1313_, lean_object* v_inst_1314_, lean_object* v_inst_1315_, lean_object* v_uscript_1316_, lean_object* v_sscript_x3f_1317_, lean_object* v_preState_1318_, lean_object* v_goal_1319_, lean_object* v_options_1320_, uint8_t v_expectCompleteProof_1321_, lean_object* v_tacticName_1322_){
_start:
{
if (lean_obj_tag(v_sscript_x3f_1317_) == 1)
{
lean_object* v_val_1323_; lean_object* v_toBind_1324_; lean_object* v_fst_1325_; lean_object* v_snd_1326_; lean_object* v___x_1327_; lean_object* v___f_1328_; lean_object* v___f_1329_; lean_object* v___f_1330_; lean_object* v___f_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; 
lean_dec_ref(v_tacticName_1322_);
lean_dec_ref(v_uscript_1316_);
lean_dec(v_inst_1313_);
lean_dec_ref(v_inst_1312_);
lean_dec_ref(v_inst_1310_);
v_val_1323_ = lean_ctor_get(v_sscript_x3f_1317_, 0);
lean_inc(v_val_1323_);
lean_dec_ref_known(v_sscript_x3f_1317_, 1);
v_toBind_1324_ = lean_ctor_get(v_inst_1309_, 1);
lean_inc_n(v_toBind_1324_, 3);
v_fst_1325_ = lean_ctor_get(v_val_1323_, 0);
lean_inc_n(v_fst_1325_, 2);
v_snd_1326_ = lean_ctor_get(v_val_1323_, 1);
lean_inc(v_snd_1326_);
lean_dec(v_val_1323_);
v___x_1327_ = lean_box(v_expectCompleteProof_1321_);
lean_inc(v_inst_1315_);
v___f_1328_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_1328_, 0, v_fst_1325_);
lean_closure_set(v___f_1328_, 1, v_preState_1318_);
lean_closure_set(v___f_1328_, 2, v_goal_1319_);
lean_closure_set(v___f_1328_, 3, v___x_1327_);
lean_closure_set(v___f_1328_, 4, v_inst_1315_);
lean_inc_ref(v___f_1328_);
v___f_1329_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1329_, 0, v___f_1328_);
v___f_1330_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__2), 5, 4);
lean_closure_set(v___f_1330_, 0, v_fst_1325_);
lean_closure_set(v___f_1330_, 1, v_inst_1315_);
lean_closure_set(v___f_1330_, 2, v_toBind_1324_);
lean_closure_set(v___f_1330_, 3, v___f_1329_);
v___f_1331_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__3___boxed), 6, 5);
lean_closure_set(v___f_1331_, 0, v_options_1320_);
lean_closure_set(v___f_1331_, 1, v___f_1328_);
lean_closure_set(v___f_1331_, 2, v_inst_1311_);
lean_closure_set(v___f_1331_, 3, v_toBind_1324_);
lean_closure_set(v___f_1331_, 4, v___f_1330_);
v___x_1332_ = lp_aesop_Aesop_recordScriptGenerated___redArg(v_inst_1309_, v_inst_1314_, v_snd_1326_);
v___x_1333_ = lean_apply_4(v_toBind_1324_, lean_box(0), lean_box(0), v___x_1332_, v___f_1331_);
return v___x_1333_;
}
else
{
lean_object* v_toOptions_1334_; lean_object* v_toApplicative_1335_; lean_object* v_toBind_1336_; lean_object* v_toMonadOptions_1337_; uint8_t v_traceScript_1338_; lean_object* v___x_1339_; lean_object* v___f_1340_; 
lean_dec(v_sscript_x3f_1317_);
v_toOptions_1334_ = lean_ctor_get(v_options_1320_, 0);
lean_inc_ref(v_toOptions_1334_);
lean_dec_ref(v_options_1320_);
v_toApplicative_1335_ = lean_ctor_get(v_inst_1309_, 0);
lean_inc_ref_n(v_toApplicative_1335_, 2);
v_toBind_1336_ = lean_ctor_get(v_inst_1309_, 1);
lean_inc_n(v_toBind_1336_, 2);
v_toMonadOptions_1337_ = lean_ctor_get(v_inst_1314_, 0);
lean_inc_n(v_toMonadOptions_1337_, 2);
lean_dec_ref(v_inst_1314_);
v_traceScript_1338_ = lean_ctor_get_uint8(v_toOptions_1334_, sizeof(void*)*6 + 6);
lean_dec_ref(v_toOptions_1334_);
v___x_1339_ = lean_box(v_traceScript_1338_);
lean_inc_ref(v_inst_1312_);
lean_inc(v_inst_1313_);
lean_inc_ref(v_inst_1310_);
lean_inc_ref(v_inst_1309_);
lean_inc_ref(v_tacticName_1322_);
v___f_1340_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__5___boxed), 10, 9);
lean_closure_set(v___f_1340_, 0, v___x_1339_);
lean_closure_set(v___f_1340_, 1, v_toApplicative_1335_);
lean_closure_set(v___f_1340_, 2, v_tacticName_1322_);
lean_closure_set(v___f_1340_, 3, v_inst_1309_);
lean_closure_set(v___f_1340_, 4, v_inst_1310_);
lean_closure_set(v___f_1340_, 5, v_inst_1313_);
lean_closure_set(v___f_1340_, 6, v_toMonadOptions_1337_);
lean_closure_set(v___f_1340_, 7, v_inst_1312_);
lean_closure_set(v___f_1340_, 8, v_toBind_1336_);
if (v_traceScript_1338_ == 0)
{
lean_object* v___x_1341_; lean_object* v___x_1342_; 
lean_dec_ref(v___f_1340_);
lean_dec(v_goal_1319_);
lean_dec_ref(v_preState_1318_);
lean_dec_ref(v_uscript_1316_);
lean_dec(v_inst_1315_);
lean_dec_ref(v_inst_1311_);
v___x_1341_ = lean_box(0);
v___x_1342_ = lp_aesop_Aesop_checkAndTraceScript___redArg___lam__5(v_traceScript_1338_, v_toApplicative_1335_, v_tacticName_1322_, v_inst_1309_, v_inst_1310_, v_inst_1313_, v_toMonadOptions_1337_, v_inst_1312_, v_toBind_1336_, v___x_1341_);
return v___x_1342_;
}
else
{
lean_object* v___f_1343_; lean_object* v___f_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; 
lean_dec(v_toMonadOptions_1337_);
lean_dec_ref(v_toApplicative_1335_);
lean_dec_ref(v_tacticName_1322_);
lean_dec(v_inst_1313_);
lean_dec_ref(v_inst_1312_);
lean_dec_ref(v_inst_1310_);
lean_dec_ref(v_inst_1309_);
v___f_1343_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1343_, 0, v___f_1340_);
lean_inc(v_toBind_1336_);
lean_inc(v_inst_1315_);
v___f_1344_ = lean_alloc_closure((void*)(lp_aesop_Aesop_checkAndTraceScript___redArg___lam__6), 5, 4);
lean_closure_set(v___f_1344_, 0, v_inst_1311_);
lean_closure_set(v___f_1344_, 1, v_inst_1315_);
lean_closure_set(v___f_1344_, 2, v_toBind_1336_);
lean_closure_set(v___f_1344_, 3, v___f_1343_);
v___x_1345_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_UScript_renderTacticSeq___boxed), 8, 3);
lean_closure_set(v___x_1345_, 0, v_uscript_1316_);
lean_closure_set(v___x_1345_, 1, v_preState_1318_);
lean_closure_set(v___x_1345_, 2, v_goal_1319_);
v___x_1346_ = lean_apply_2(v_inst_1315_, lean_box(0), v___x_1345_);
v___x_1347_ = lean_apply_4(v_toBind_1336_, lean_box(0), lean_box(0), v___x_1346_, v___f_1344_);
return v___x_1347_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___redArg___boxed(lean_object* v_inst_1348_, lean_object* v_inst_1349_, lean_object* v_inst_1350_, lean_object* v_inst_1351_, lean_object* v_inst_1352_, lean_object* v_inst_1353_, lean_object* v_inst_1354_, lean_object* v_uscript_1355_, lean_object* v_sscript_x3f_1356_, lean_object* v_preState_1357_, lean_object* v_goal_1358_, lean_object* v_options_1359_, lean_object* v_expectCompleteProof_1360_, lean_object* v_tacticName_1361_){
_start:
{
uint8_t v_expectCompleteProof_boxed_1362_; lean_object* v_res_1363_; 
v_expectCompleteProof_boxed_1362_ = lean_unbox(v_expectCompleteProof_1360_);
v_res_1363_ = lp_aesop_Aesop_checkAndTraceScript___redArg(v_inst_1348_, v_inst_1349_, v_inst_1350_, v_inst_1351_, v_inst_1352_, v_inst_1353_, v_inst_1354_, v_uscript_1355_, v_sscript_x3f_1356_, v_preState_1357_, v_goal_1358_, v_options_1359_, v_expectCompleteProof_boxed_1362_, v_tacticName_1361_);
return v_res_1363_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript(lean_object* v_m_1364_, lean_object* v_inst_1365_, lean_object* v_inst_1366_, lean_object* v_inst_1367_, lean_object* v_inst_1368_, lean_object* v_inst_1369_, lean_object* v_inst_1370_, lean_object* v_inst_1371_, lean_object* v_uscript_1372_, lean_object* v_sscript_x3f_1373_, lean_object* v_preState_1374_, lean_object* v_goal_1375_, lean_object* v_options_1376_, uint8_t v_expectCompleteProof_1377_, lean_object* v_tacticName_1378_){
_start:
{
lean_object* v___x_1379_; 
v___x_1379_ = lp_aesop_Aesop_checkAndTraceScript___redArg(v_inst_1365_, v_inst_1366_, v_inst_1367_, v_inst_1368_, v_inst_1369_, v_inst_1370_, v_inst_1371_, v_uscript_1372_, v_sscript_x3f_1373_, v_preState_1374_, v_goal_1375_, v_options_1376_, v_expectCompleteProof_1377_, v_tacticName_1378_);
return v___x_1379_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_checkAndTraceScript___boxed(lean_object* v_m_1380_, lean_object* v_inst_1381_, lean_object* v_inst_1382_, lean_object* v_inst_1383_, lean_object* v_inst_1384_, lean_object* v_inst_1385_, lean_object* v_inst_1386_, lean_object* v_inst_1387_, lean_object* v_uscript_1388_, lean_object* v_sscript_x3f_1389_, lean_object* v_preState_1390_, lean_object* v_goal_1391_, lean_object* v_options_1392_, lean_object* v_expectCompleteProof_1393_, lean_object* v_tacticName_1394_){
_start:
{
uint8_t v_expectCompleteProof_boxed_1395_; lean_object* v_res_1396_; 
v_expectCompleteProof_boxed_1395_ = lean_unbox(v_expectCompleteProof_1393_);
v_res_1396_ = lp_aesop_Aesop_checkAndTraceScript(v_m_1380_, v_inst_1381_, v_inst_1382_, v_inst_1383_, v_inst_1384_, v_inst_1385_, v_inst_1386_, v_inst_1387_, v_uscript_1388_, v_sscript_x3f_1389_, v_preState_1390_, v_goal_1391_, v_options_1392_, v_expectCompleteProof_boxed_1395_, v_tacticName_1394_);
return v_res_1396_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_Check(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Options_Internal(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_OptimizeSyntax(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_StructureDynamic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_StructureStatic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_Main(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Options_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_OptimizeSyntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_StructureDynamic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_StructureStatic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_Main(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Script_Check(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Options_Internal(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_OptimizeSyntax(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_StructureDynamic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_StructureStatic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_Main(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Options_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_OptimizeSyntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_StructureDynamic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_StructureStatic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_Main(builtin);
}
#ifdef __cplusplus
}
#endif
