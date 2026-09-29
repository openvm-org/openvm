// Lean compiler output
// Module: Mathlib.Util.CountHeartbeats
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Util.Heartbeats public meta import Lean.Meta.Tactic.TryThis
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_getMaxHeartbeats___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Message_toString(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
uint8_t lean_uint32_dec_le(uint32_t, uint32_t);
lean_object* lean_string_utf8_set(lean_object*, lean_object*, uint32_t);
uint32_t lean_uint32_add(uint32_t, uint32_t);
lean_object* l_Lean_Syntax_getKind(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
double lean_float_sub(double, double);
double lean_float_of_nat(lean_object*);
double pow(double, double);
uint64_t lean_uint64_of_nat(lean_object*);
double lean_uint64_to_float(uint64_t);
double lean_float_add(double, double);
lean_object* l_List_lengthTR___redArg(lean_object*);
double lean_float_div(double, double);
double sqrt(double);
uint64_t lean_float_to_uint64(double);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* l_Lean_logInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* l_Lean_Syntax_mkNumLit(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "async"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__2_value),LEAN_SCALAR_PTR_LITERAL(163, 142, 149, 180, 91, 16, 128, 108)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3___boxed(lean_object*);
LEAN_EXPORT double lp_mathlib_List_foldl___at___00Mathlib_CountHeartbeats_variation_spec__1(double, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_CountHeartbeats_variation_spec__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2(double, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats_variation___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Mathlib_CountHeartbeats_variation___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_variation(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Min: "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " Max: "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " StdDev: "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "%"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "CountHeartbeats"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "tactic#count_heartbeats_"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(236, 172, 170, 21, 94, 194, 51, 232)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(30, 182, 54, 245, 255, 63, 203, 244)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "#count_heartbeats "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats__ = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "tactic#count_heartbeats!_In__"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(236, 172, 170, 21, 94, 194, 51, 232)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 155, 56, 152, 129, 36, 15, 142)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "#count_heartbeats! "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "in"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ppLine"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(117, 61, 38, 245, 158, 59, 171, 58)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__15_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__13_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__20 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__20_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__21 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__22 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In____ = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_roundDownIf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "approximately "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_roundDownIf___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_roundDownIf___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_roundDownIf(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_roundDownIf___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "command#count_heartbeatsApproximatelyIn__"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(236, 172, 170, 21, 94, 194, 51, 232)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 115, 34, 78, 65, 28, 131, 214)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_roundDownIf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn____ = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "set_option"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "maxHeartbeats"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(163, 202, 216, 251, 148, 187, 135, 206)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getMaxHeartbeats___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Used "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 58, .m_data = " heartbeats, which is greater than the current maximum of "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__4;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = " heartbeats, which is less than the current maximum of "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(65, 79, 35, 19, 21, 38, 89, 10)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(216, 223, 149, 245, 150, 86, 134, 198)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Elab.async"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 84, 199, 228, 250, 36, 60, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__2_value),LEAN_SCALAR_PTR_LITERAL(6, 0, 36, 68, 138, 2, 151, 20)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(242, 134, 20, 158, 154, 92, 25, 244)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "commandGuard_min_heartbeatsApproximately_In__"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(236, 172, 170, 21, 94, 194, 51, 232)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(36, 52, 14, 90, 35, 241, 108, 102)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "guard_min_heartbeats "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In____ = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = " heartbeats, which is less than the minimum of "};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_elabForHeartbeats(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_elabForHeartbeats___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "command#count_heartbeats!_In__"};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(236, 172, 170, 21, 94, 194, 51, 232)}};
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(48, 175, 200, 69, 110, 248, 169, 214)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In____ = (const lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "countHeartbeats"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(15, 248, 232, 255, 190, 169, 76, 116)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "enable the countHeartbeats linter"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "2026-07-30"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 115, .m_capacity = 115, .m_length = 114, .m_data = "use `#count_heartbeats in` or `set_option trace.profiler true` with `set_option trace.profiler.useHeartbeats true`"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__9_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__9_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__9_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(188, 211, 125, 188, 150, 106, 82, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_countHeartbeats;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "countHeartbeatsApprox"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(70, 138, 25, 77, 205, 141, 95, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 102, .m_capacity = 102, .m_length = 101, .m_data = "if set to `true`, then the countHeartbeats linter rounds down to the nearest 1000 the heartbeat count"};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(197, 38, 178, 43, 125, 125, 44, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_countHeartbeatsApprox;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__2(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "' "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lemma"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(117, 34, 246, 137, 114, 183, 220, 217)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__5_value;
static const lean_array_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "#count_heartbeats"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "approximately"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "countHeartbeatsLinter"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(26, 255, 47, 156, 32, 212, 229, 214)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(204, 108, 204, 93, 37, 193, 193, 138)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_CountHeartbeats_initFn_00___x40_Mathlib_Util_CountHeartbeats_347709687____hygCtx___hyg_8_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_CountHeartbeats_initFn_00___x40_Mathlib_Util_CountHeartbeats_347709687____hygCtx___hyg_8____boxed(lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(26, 255, 47, 156, 32, 212, 229, 214)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(47, 126, 27, 0, 211, 123, 160, 8)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " approximately"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "linter.countHeartbeats"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "linter.countHeartbeatsApprox"};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0(lean_object* v_o_4_, lean_object* v_k_5_, uint8_t v_v_6_){
_start:
{
lean_object* v_map_7_; uint8_t v_hasTrace_8_; lean_object* v___x_10_; uint8_t v_isShared_11_; uint8_t v_isSharedCheck_22_; 
v_map_7_ = lean_ctor_get(v_o_4_, 0);
v_hasTrace_8_ = lean_ctor_get_uint8(v_o_4_, sizeof(void*)*1);
v_isSharedCheck_22_ = !lean_is_exclusive(v_o_4_);
if (v_isSharedCheck_22_ == 0)
{
v___x_10_ = v_o_4_;
v_isShared_11_ = v_isSharedCheck_22_;
goto v_resetjp_9_;
}
else
{
lean_inc(v_map_7_);
lean_dec(v_o_4_);
v___x_10_ = lean_box(0);
v_isShared_11_ = v_isSharedCheck_22_;
goto v_resetjp_9_;
}
v_resetjp_9_:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_12_, 0, v_v_6_);
lean_inc(v_k_5_);
v___x_13_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_5_, v___x_12_, v_map_7_);
if (v_hasTrace_8_ == 0)
{
lean_object* v___x_14_; uint8_t v___x_15_; lean_object* v___x_17_; 
v___x_14_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__1));
v___x_15_ = l_Lean_Name_isPrefixOf(v___x_14_, v_k_5_);
lean_dec(v_k_5_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 0, v___x_13_);
v___x_17_ = v___x_10_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___x_13_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
lean_ctor_set_uint8(v___x_17_, sizeof(void*)*1, v___x_15_);
return v___x_17_;
}
}
else
{
lean_object* v___x_20_; 
lean_dec(v_k_5_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 0, v___x_13_);
v___x_20_ = v___x_10_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v___x_13_);
lean_ctor_set_uint8(v_reuseFailAlloc_21_, sizeof(void*)*1, v_hasTrace_8_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___boxed(lean_object* v_o_23_, lean_object* v_k_24_, lean_object* v_v_25_){
_start:
{
uint8_t v_v_boxed_26_; lean_object* v_res_27_; 
v_v_boxed_26_ = lean_unbox(v_v_25_);
v_res_27_ = lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0(v_o_23_, v_k_24_, v_v_boxed_26_);
return v_res_27_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1(lean_object* v_opts_28_, lean_object* v_opt_29_){
_start:
{
lean_object* v_name_30_; lean_object* v_defValue_31_; lean_object* v_map_32_; lean_object* v___x_33_; 
v_name_30_ = lean_ctor_get(v_opt_29_, 0);
v_defValue_31_ = lean_ctor_get(v_opt_29_, 1);
v_map_32_ = lean_ctor_get(v_opts_28_, 0);
v___x_33_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_32_, v_name_30_);
if (lean_obj_tag(v___x_33_) == 0)
{
uint8_t v___x_34_; 
v___x_34_ = lean_unbox(v_defValue_31_);
return v___x_34_;
}
else
{
lean_object* v_val_35_; 
v_val_35_ = lean_ctor_get(v___x_33_, 0);
lean_inc(v_val_35_);
lean_dec_ref_known(v___x_33_, 1);
if (lean_obj_tag(v_val_35_) == 1)
{
uint8_t v_v_36_; 
v_v_36_ = lean_ctor_get_uint8(v_val_35_, 0);
lean_dec_ref_known(v_val_35_, 0);
return v_v_36_;
}
else
{
uint8_t v___x_37_; 
lean_dec(v_val_35_);
v___x_37_ = lean_unbox(v_defValue_31_);
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1___boxed(lean_object* v_opts_38_, lean_object* v_opt_39_){
_start:
{
uint8_t v_res_40_; lean_object* v_r_41_; 
v_res_40_ = lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1(v_opts_38_, v_opt_39_);
lean_dec_ref(v_opt_39_);
lean_dec_ref(v_opts_38_);
v_r_41_ = lean_box(v_res_40_);
return v_r_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__2(lean_object* v_opts_42_, lean_object* v_opt_43_){
_start:
{
lean_object* v_name_44_; lean_object* v_defValue_45_; lean_object* v_map_46_; lean_object* v___x_47_; 
v_name_44_ = lean_ctor_get(v_opt_43_, 0);
v_defValue_45_ = lean_ctor_get(v_opt_43_, 1);
v_map_46_ = lean_ctor_get(v_opts_42_, 0);
v___x_47_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_46_, v_name_44_);
if (lean_obj_tag(v___x_47_) == 0)
{
lean_inc(v_defValue_45_);
return v_defValue_45_;
}
else
{
lean_object* v_val_48_; 
v_val_48_ = lean_ctor_get(v___x_47_, 0);
lean_inc(v_val_48_);
lean_dec_ref_known(v___x_47_, 1);
if (lean_obj_tag(v_val_48_) == 3)
{
lean_object* v_v_49_; 
v_v_49_ = lean_ctor_get(v_val_48_, 0);
lean_inc(v_v_49_);
lean_dec_ref_known(v_val_48_, 1);
return v_v_49_;
}
else
{
lean_dec(v_val_48_);
lean_inc(v_defValue_45_);
return v_defValue_45_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__2___boxed(lean_object* v_opts_50_, lean_object* v_opt_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__2(v_opts_50_, v_opt_51_);
lean_dec_ref(v_opt_51_);
lean_dec_ref(v_opts_50_);
return v_res_52_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__4(void){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__5(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_61_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__4, &lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__4_once, _init_lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__4);
v___x_62_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
return v___x_62_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__6(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__5, &lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__5_once, _init_lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__5);
v___x_64_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_64_, 0, v___x_63_);
lean_ctor_set(v___x_64_, 1, v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats(lean_object* v_tac_65_, uint8_t v_revert_66_, lean_object* v_a_67_, lean_object* v_a_68_, lean_object* v_a_69_, lean_object* v_a_70_, lean_object* v_a_71_, lean_object* v_a_72_, lean_object* v_a_73_, lean_object* v_a_74_){
_start:
{
lean_object* v___x_76_; lean_object* v___x_81_; 
v___x_76_ = lean_io_get_num_heartbeats();
v___x_81_ = l_Lean_Elab_Tactic_saveState___redArg(v_a_68_, v_a_70_, v_a_72_, v_a_74_);
if (lean_obj_tag(v___x_81_) == 0)
{
lean_object* v_a_82_; lean_object* v___x_83_; lean_object* v_fileName_84_; lean_object* v_fileMap_85_; lean_object* v_options_86_; lean_object* v_currRecDepth_87_; lean_object* v_ref_88_; lean_object* v_currNamespace_89_; lean_object* v_openDecls_90_; lean_object* v_initHeartbeats_91_; lean_object* v_maxHeartbeats_92_; lean_object* v_quotContext_93_; lean_object* v_currMacroScope_94_; lean_object* v_cancelTk_x3f_95_; uint8_t v_suppressElabErrors_96_; lean_object* v_inheritedTraceOptions_97_; lean_object* v_env_98_; lean_object* v___x_99_; uint8_t v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; lean_object* v_fileName_105_; lean_object* v_fileMap_106_; lean_object* v_currRecDepth_107_; lean_object* v_ref_108_; lean_object* v_currNamespace_109_; lean_object* v_openDecls_110_; lean_object* v_initHeartbeats_111_; lean_object* v_maxHeartbeats_112_; lean_object* v_quotContext_113_; lean_object* v_currMacroScope_114_; lean_object* v_cancelTk_x3f_115_; uint8_t v_suppressElabErrors_116_; lean_object* v_inheritedTraceOptions_117_; lean_object* v___y_118_; uint8_t v___y_141_; uint8_t v___x_162_; 
v_a_82_ = lean_ctor_get(v___x_81_, 0);
lean_inc(v_a_82_);
lean_dec_ref_known(v___x_81_, 1);
v___x_83_ = lean_st_ref_get(v_a_74_);
v_fileName_84_ = lean_ctor_get(v_a_73_, 0);
v_fileMap_85_ = lean_ctor_get(v_a_73_, 1);
v_options_86_ = lean_ctor_get(v_a_73_, 2);
v_currRecDepth_87_ = lean_ctor_get(v_a_73_, 3);
v_ref_88_ = lean_ctor_get(v_a_73_, 5);
v_currNamespace_89_ = lean_ctor_get(v_a_73_, 6);
v_openDecls_90_ = lean_ctor_get(v_a_73_, 7);
v_initHeartbeats_91_ = lean_ctor_get(v_a_73_, 8);
v_maxHeartbeats_92_ = lean_ctor_get(v_a_73_, 9);
v_quotContext_93_ = lean_ctor_get(v_a_73_, 10);
v_currMacroScope_94_ = lean_ctor_get(v_a_73_, 11);
v_cancelTk_x3f_95_ = lean_ctor_get(v_a_73_, 12);
v_suppressElabErrors_96_ = lean_ctor_get_uint8(v_a_73_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_97_ = lean_ctor_get(v_a_73_, 13);
v_env_98_ = lean_ctor_get(v___x_83_, 0);
lean_inc_ref(v_env_98_);
lean_dec(v___x_83_);
v___x_99_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__3));
v___x_100_ = 0;
lean_inc_ref(v_options_86_);
v___x_101_ = lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0(v_options_86_, v___x_99_, v___x_100_);
v___x_102_ = l_Lean_diagnostics;
v___x_103_ = lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1(v___x_101_, v___x_102_);
v___x_162_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_98_);
lean_dec_ref(v_env_98_);
if (v___x_162_ == 0)
{
if (v___x_103_ == 0)
{
v_fileName_105_ = v_fileName_84_;
v_fileMap_106_ = v_fileMap_85_;
v_currRecDepth_107_ = v_currRecDepth_87_;
v_ref_108_ = v_ref_88_;
v_currNamespace_109_ = v_currNamespace_89_;
v_openDecls_110_ = v_openDecls_90_;
v_initHeartbeats_111_ = v_initHeartbeats_91_;
v_maxHeartbeats_112_ = v_maxHeartbeats_92_;
v_quotContext_113_ = v_quotContext_93_;
v_currMacroScope_114_ = v_currMacroScope_94_;
v_cancelTk_x3f_115_ = v_cancelTk_x3f_95_;
v_suppressElabErrors_116_ = v_suppressElabErrors_96_;
v_inheritedTraceOptions_117_ = v_inheritedTraceOptions_97_;
v___y_118_ = v_a_74_;
goto v___jp_104_;
}
else
{
v___y_141_ = v___x_162_;
goto v___jp_140_;
}
}
else
{
v___y_141_ = v___x_103_;
goto v___jp_140_;
}
v___jp_104_:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_119_ = l_Lean_maxRecDepth;
v___x_120_ = lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__2(v___x_101_, v___x_119_);
lean_inc_ref(v_inheritedTraceOptions_117_);
lean_inc(v_cancelTk_x3f_115_);
lean_inc(v_currMacroScope_114_);
lean_inc(v_quotContext_113_);
lean_inc(v_maxHeartbeats_112_);
lean_inc(v_initHeartbeats_111_);
lean_inc(v_openDecls_110_);
lean_inc(v_currNamespace_109_);
lean_inc(v_ref_108_);
lean_inc(v_currRecDepth_107_);
lean_inc_ref(v_fileMap_106_);
lean_inc_ref(v_fileName_105_);
v___x_121_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_121_, 0, v_fileName_105_);
lean_ctor_set(v___x_121_, 1, v_fileMap_106_);
lean_ctor_set(v___x_121_, 2, v___x_101_);
lean_ctor_set(v___x_121_, 3, v_currRecDepth_107_);
lean_ctor_set(v___x_121_, 4, v___x_120_);
lean_ctor_set(v___x_121_, 5, v_ref_108_);
lean_ctor_set(v___x_121_, 6, v_currNamespace_109_);
lean_ctor_set(v___x_121_, 7, v_openDecls_110_);
lean_ctor_set(v___x_121_, 8, v_initHeartbeats_111_);
lean_ctor_set(v___x_121_, 9, v_maxHeartbeats_112_);
lean_ctor_set(v___x_121_, 10, v_quotContext_113_);
lean_ctor_set(v___x_121_, 11, v_currMacroScope_114_);
lean_ctor_set(v___x_121_, 12, v_cancelTk_x3f_115_);
lean_ctor_set(v___x_121_, 13, v_inheritedTraceOptions_117_);
lean_ctor_set_uint8(v___x_121_, sizeof(void*)*14, v___x_103_);
lean_ctor_set_uint8(v___x_121_, sizeof(void*)*14 + 1, v_suppressElabErrors_116_);
v___x_122_ = l_Lean_Elab_Tactic_evalTactic(v_tac_65_, v_a_67_, v_a_68_, v_a_69_, v_a_70_, v_a_71_, v_a_72_, v___x_121_, v___y_118_);
lean_dec_ref_known(v___x_121_, 14);
if (lean_obj_tag(v___x_122_) == 0)
{
lean_dec_ref_known(v___x_122_, 1);
if (v_revert_66_ == 0)
{
lean_dec(v_a_82_);
goto v___jp_77_;
}
else
{
lean_object* v___x_123_; 
v___x_123_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_82_, v___x_100_, v_a_68_, v_a_69_, v_a_70_, v_a_71_, v_a_72_, v_a_73_, v_a_74_);
if (lean_obj_tag(v___x_123_) == 0)
{
lean_dec_ref_known(v___x_123_, 1);
goto v___jp_77_;
}
else
{
lean_object* v_a_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_131_; 
lean_dec(v___x_76_);
v_a_124_ = lean_ctor_get(v___x_123_, 0);
v_isSharedCheck_131_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_131_ == 0)
{
v___x_126_ = v___x_123_;
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_a_124_);
lean_dec(v___x_123_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v___x_129_; 
if (v_isShared_127_ == 0)
{
v___x_129_ = v___x_126_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v_a_124_);
v___x_129_ = v_reuseFailAlloc_130_;
goto v_reusejp_128_;
}
v_reusejp_128_:
{
return v___x_129_;
}
}
}
}
}
else
{
lean_object* v_a_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_139_; 
lean_dec(v_a_82_);
lean_dec(v___x_76_);
v_a_132_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_139_ == 0)
{
v___x_134_ = v___x_122_;
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_a_132_);
lean_dec(v___x_122_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_137_; 
if (v_isShared_135_ == 0)
{
v___x_137_ = v___x_134_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v_a_132_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
v___jp_140_:
{
if (v___y_141_ == 0)
{
lean_object* v___x_142_; lean_object* v_env_143_; lean_object* v_nextMacroScope_144_; lean_object* v_ngen_145_; lean_object* v_auxDeclNGen_146_; lean_object* v_traceState_147_; lean_object* v_messages_148_; lean_object* v_infoState_149_; lean_object* v_snapshotTasks_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_160_; 
v___x_142_ = lean_st_ref_take(v_a_74_);
v_env_143_ = lean_ctor_get(v___x_142_, 0);
v_nextMacroScope_144_ = lean_ctor_get(v___x_142_, 1);
v_ngen_145_ = lean_ctor_get(v___x_142_, 2);
v_auxDeclNGen_146_ = lean_ctor_get(v___x_142_, 3);
v_traceState_147_ = lean_ctor_get(v___x_142_, 4);
v_messages_148_ = lean_ctor_get(v___x_142_, 6);
v_infoState_149_ = lean_ctor_get(v___x_142_, 7);
v_snapshotTasks_150_ = lean_ctor_get(v___x_142_, 8);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_160_ == 0)
{
lean_object* v_unused_161_; 
v_unused_161_ = lean_ctor_get(v___x_142_, 5);
lean_dec(v_unused_161_);
v___x_152_ = v___x_142_;
v_isShared_153_ = v_isSharedCheck_160_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_snapshotTasks_150_);
lean_inc(v_infoState_149_);
lean_inc(v_messages_148_);
lean_inc(v_traceState_147_);
lean_inc(v_auxDeclNGen_146_);
lean_inc(v_ngen_145_);
lean_inc(v_nextMacroScope_144_);
lean_inc(v_env_143_);
lean_dec(v___x_142_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_160_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_157_; 
v___x_154_ = l_Lean_Kernel_enableDiag(v_env_143_, v___x_103_);
v___x_155_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__6, &lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__6_once, _init_lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__6);
if (v_isShared_153_ == 0)
{
lean_ctor_set(v___x_152_, 5, v___x_155_);
lean_ctor_set(v___x_152_, 0, v___x_154_);
v___x_157_ = v___x_152_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v___x_154_);
lean_ctor_set(v_reuseFailAlloc_159_, 1, v_nextMacroScope_144_);
lean_ctor_set(v_reuseFailAlloc_159_, 2, v_ngen_145_);
lean_ctor_set(v_reuseFailAlloc_159_, 3, v_auxDeclNGen_146_);
lean_ctor_set(v_reuseFailAlloc_159_, 4, v_traceState_147_);
lean_ctor_set(v_reuseFailAlloc_159_, 5, v___x_155_);
lean_ctor_set(v_reuseFailAlloc_159_, 6, v_messages_148_);
lean_ctor_set(v_reuseFailAlloc_159_, 7, v_infoState_149_);
lean_ctor_set(v_reuseFailAlloc_159_, 8, v_snapshotTasks_150_);
v___x_157_ = v_reuseFailAlloc_159_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
lean_object* v___x_158_; 
v___x_158_ = lean_st_ref_set(v_a_74_, v___x_157_);
v_fileName_105_ = v_fileName_84_;
v_fileMap_106_ = v_fileMap_85_;
v_currRecDepth_107_ = v_currRecDepth_87_;
v_ref_108_ = v_ref_88_;
v_currNamespace_109_ = v_currNamespace_89_;
v_openDecls_110_ = v_openDecls_90_;
v_initHeartbeats_111_ = v_initHeartbeats_91_;
v_maxHeartbeats_112_ = v_maxHeartbeats_92_;
v_quotContext_113_ = v_quotContext_93_;
v_currMacroScope_114_ = v_currMacroScope_94_;
v_cancelTk_x3f_115_ = v_cancelTk_x3f_95_;
v_suppressElabErrors_116_ = v_suppressElabErrors_96_;
v_inheritedTraceOptions_117_ = v_inheritedTraceOptions_97_;
v___y_118_ = v_a_74_;
goto v___jp_104_;
}
}
}
else
{
v_fileName_105_ = v_fileName_84_;
v_fileMap_106_ = v_fileMap_85_;
v_currRecDepth_107_ = v_currRecDepth_87_;
v_ref_108_ = v_ref_88_;
v_currNamespace_109_ = v_currNamespace_89_;
v_openDecls_110_ = v_openDecls_90_;
v_initHeartbeats_111_ = v_initHeartbeats_91_;
v_maxHeartbeats_112_ = v_maxHeartbeats_92_;
v_quotContext_113_ = v_quotContext_93_;
v_currMacroScope_114_ = v_currMacroScope_94_;
v_cancelTk_x3f_115_ = v_cancelTk_x3f_95_;
v_suppressElabErrors_116_ = v_suppressElabErrors_96_;
v_inheritedTraceOptions_117_ = v_inheritedTraceOptions_97_;
v___y_118_ = v_a_74_;
goto v___jp_104_;
}
}
}
else
{
lean_object* v_a_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_170_; 
lean_dec(v___x_76_);
lean_dec(v_tac_65_);
v_a_163_ = lean_ctor_get(v___x_81_, 0);
v_isSharedCheck_170_ = !lean_is_exclusive(v___x_81_);
if (v_isSharedCheck_170_ == 0)
{
v___x_165_ = v___x_81_;
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_a_163_);
lean_dec(v___x_81_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___x_168_; 
if (v_isShared_166_ == 0)
{
v___x_168_ = v___x_165_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_a_163_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
}
v___jp_77_:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_78_ = lean_io_get_num_heartbeats();
v___x_79_ = lean_nat_sub(v___x_78_, v___x_76_);
lean_dec(v___x_76_);
lean_dec(v___x_78_);
v___x_80_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
return v___x_80_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___boxed(lean_object* v_tac_171_, lean_object* v_revert_172_, lean_object* v_a_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
uint8_t v_revert_boxed_182_; lean_object* v_res_183_; 
v_revert_boxed_182_ = lean_unbox(v_revert_172_);
v_res_183_ = lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats(v_tac_171_, v_revert_boxed_182_, v_a_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_, v_a_178_, v_a_179_, v_a_180_);
lean_dec(v_a_180_);
lean_dec_ref(v_a_179_);
lean_dec(v_a_178_);
lean_dec_ref(v_a_177_);
lean_dec(v_a_176_);
lean_dec_ref(v_a_175_);
lean_dec(v_a_174_);
lean_dec_ref(v_a_173_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4_spec__5(lean_object* v_x_184_, lean_object* v_x_185_){
_start:
{
if (lean_obj_tag(v_x_185_) == 0)
{
lean_inc(v_x_184_);
return v_x_184_;
}
else
{
lean_object* v_head_186_; lean_object* v_tail_187_; uint8_t v___x_188_; 
v_head_186_ = lean_ctor_get(v_x_185_, 0);
v_tail_187_ = lean_ctor_get(v_x_185_, 1);
v___x_188_ = lean_nat_dec_le(v_x_184_, v_head_186_);
if (v___x_188_ == 0)
{
v_x_184_ = v_head_186_;
v_x_185_ = v_tail_187_;
goto _start;
}
else
{
v_x_185_ = v_tail_187_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4_spec__5___boxed(lean_object* v_x_191_, lean_object* v_x_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_List_foldl___at___00List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4_spec__5(v_x_191_, v_x_192_);
lean_dec(v_x_192_);
lean_dec(v_x_191_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4(lean_object* v_x_194_){
_start:
{
if (lean_obj_tag(v_x_194_) == 0)
{
lean_object* v___x_195_; 
v___x_195_ = lean_box(0);
return v___x_195_;
}
else
{
lean_object* v_head_196_; lean_object* v_tail_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v_head_196_ = lean_ctor_get(v_x_194_, 0);
v_tail_197_ = lean_ctor_get(v_x_194_, 1);
v___x_198_ = lp_mathlib_List_foldl___at___00List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4_spec__5(v_head_196_, v_tail_197_);
v___x_199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
return v___x_199_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4___boxed(lean_object* v_x_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4(v_x_200_);
lean_dec(v_x_200_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__0(lean_object* v_a_202_, lean_object* v_a_203_){
_start:
{
if (lean_obj_tag(v_a_202_) == 0)
{
lean_object* v___x_204_; 
v___x_204_ = l_List_reverse___redArg(v_a_203_);
return v___x_204_;
}
else
{
lean_object* v_head_205_; lean_object* v_tail_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_217_; 
v_head_205_ = lean_ctor_get(v_a_202_, 0);
v_tail_206_ = lean_ctor_get(v_a_202_, 1);
v_isSharedCheck_217_ = !lean_is_exclusive(v_a_202_);
if (v_isSharedCheck_217_ == 0)
{
v___x_208_ = v_a_202_;
v_isShared_209_ = v_isSharedCheck_217_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_tail_206_);
lean_inc(v_head_205_);
lean_dec(v_a_202_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_217_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
uint64_t v___x_210_; double v___x_211_; lean_object* v___x_212_; lean_object* v___x_214_; 
v___x_210_ = lean_uint64_of_nat(v_head_205_);
lean_dec(v_head_205_);
v___x_211_ = lean_uint64_to_float(v___x_210_);
v___x_212_ = lean_box_float(v___x_211_);
if (v_isShared_209_ == 0)
{
lean_ctor_set(v___x_208_, 1, v_a_203_);
lean_ctor_set(v___x_208_, 0, v___x_212_);
v___x_214_ = v___x_208_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v___x_212_);
lean_ctor_set(v_reuseFailAlloc_216_, 1, v_a_203_);
v___x_214_ = v_reuseFailAlloc_216_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
v_a_202_ = v_tail_206_;
v_a_203_ = v___x_214_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3(lean_object* v_x_218_, lean_object* v_x_219_){
_start:
{
if (lean_obj_tag(v_x_219_) == 0)
{
lean_inc(v_x_218_);
return v_x_218_;
}
else
{
lean_object* v_head_220_; lean_object* v_tail_221_; uint8_t v___x_222_; 
v_head_220_ = lean_ctor_get(v_x_219_, 0);
v_tail_221_ = lean_ctor_get(v_x_219_, 1);
v___x_222_ = lean_nat_dec_le(v_x_218_, v_head_220_);
if (v___x_222_ == 0)
{
v_x_219_ = v_tail_221_;
goto _start;
}
else
{
v_x_218_ = v_head_220_;
v_x_219_ = v_tail_221_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3___boxed(lean_object* v_x_225_, lean_object* v_x_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3(v_x_225_, v_x_226_);
lean_dec(v_x_226_);
lean_dec(v_x_225_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3(lean_object* v_x_228_){
_start:
{
if (lean_obj_tag(v_x_228_) == 0)
{
lean_object* v___x_229_; 
v___x_229_ = lean_box(0);
return v___x_229_;
}
else
{
lean_object* v_head_230_; lean_object* v_tail_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v_head_230_ = lean_ctor_get(v_x_228_, 0);
v_tail_231_ = lean_ctor_get(v_x_228_, 1);
v___x_232_ = lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3(v_head_230_, v_tail_231_);
v___x_233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
return v___x_233_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3___boxed(lean_object* v_x_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib_List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3(v_x_234_);
lean_dec(v_x_234_);
return v_res_235_;
}
}
LEAN_EXPORT double lp_mathlib_List_foldl___at___00Mathlib_CountHeartbeats_variation_spec__1(double v_x_236_, lean_object* v_x_237_){
_start:
{
if (lean_obj_tag(v_x_237_) == 0)
{
return v_x_236_;
}
else
{
lean_object* v_head_238_; lean_object* v_tail_239_; double v___x_240_; double v___x_241_; 
v_head_238_ = lean_ctor_get(v_x_237_, 0);
v_tail_239_ = lean_ctor_get(v_x_237_, 1);
v___x_240_ = lean_unbox_float(v_head_238_);
v___x_241_ = lean_float_add(v_x_236_, v___x_240_);
v_x_236_ = v___x_241_;
v_x_237_ = v_tail_239_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_CountHeartbeats_variation_spec__1___boxed(lean_object* v_x_243_, lean_object* v_x_244_){
_start:
{
double v_x_390__boxed_245_; double v_res_246_; lean_object* v_r_247_; 
v_x_390__boxed_245_ = lean_unbox_float(v_x_243_);
lean_dec_ref(v_x_243_);
v_res_246_ = lp_mathlib_List_foldl___at___00Mathlib_CountHeartbeats_variation_spec__1(v_x_390__boxed_245_, v_x_244_);
lean_dec(v_x_244_);
v_r_247_ = lean_box_float(v_res_246_);
return v_r_247_;
}
}
static double _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___closed__0(void){
_start:
{
lean_object* v___x_248_; double v___x_249_; 
v___x_248_ = lean_unsigned_to_nat(2u);
v___x_249_ = lean_float_of_nat(v___x_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2(double v_00_u03bc_250_, lean_object* v_a_251_, lean_object* v_a_252_){
_start:
{
if (lean_obj_tag(v_a_251_) == 0)
{
lean_object* v___x_253_; 
v___x_253_ = l_List_reverse___redArg(v_a_252_);
return v___x_253_;
}
else
{
lean_object* v_head_254_; lean_object* v_tail_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_268_; 
v_head_254_ = lean_ctor_get(v_a_251_, 0);
v_tail_255_ = lean_ctor_get(v_a_251_, 1);
v_isSharedCheck_268_ = !lean_is_exclusive(v_a_251_);
if (v_isSharedCheck_268_ == 0)
{
v___x_257_ = v_a_251_;
v_isShared_258_ = v_isSharedCheck_268_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_tail_255_);
lean_inc(v_head_254_);
lean_dec(v_a_251_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_268_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
double v___x_259_; double v___x_260_; double v___x_261_; double v___x_262_; lean_object* v___x_263_; lean_object* v___x_265_; 
v___x_259_ = lean_unbox_float(v_head_254_);
lean_dec(v_head_254_);
v___x_260_ = lean_float_sub(v___x_259_, v_00_u03bc_250_);
v___x_261_ = lean_float_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___closed__0, &lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___closed__0);
v___x_262_ = pow(v___x_260_, v___x_261_);
v___x_263_ = lean_box_float(v___x_262_);
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 1, v_a_252_);
lean_ctor_set(v___x_257_, 0, v___x_263_);
v___x_265_ = v___x_257_;
goto v_reusejp_264_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v___x_263_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v_a_252_);
v___x_265_ = v_reuseFailAlloc_267_;
goto v_reusejp_264_;
}
v_reusejp_264_:
{
v_a_251_ = v_tail_255_;
v_a_252_ = v___x_265_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2___boxed(lean_object* v_00_u03bc_269_, lean_object* v_a_270_, lean_object* v_a_271_){
_start:
{
double v_00_u03bc_boxed_272_; lean_object* v_res_273_; 
v_00_u03bc_boxed_272_ = lean_unbox_float(v_00_u03bc_269_);
lean_dec_ref(v_00_u03bc_269_);
v_res_273_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2(v_00_u03bc_boxed_272_, v_a_270_, v_a_271_);
return v_res_273_;
}
}
static double _init_lp_mathlib_Mathlib_CountHeartbeats_variation___closed__0(void){
_start:
{
lean_object* v___x_274_; double v___x_275_; 
v___x_274_ = lean_unsigned_to_nat(0u);
v___x_275_ = lean_float_of_nat(v___x_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_variation(lean_object* v_counts_276_){
_start:
{
lean_object* v___y_278_; lean_object* v___y_279_; lean_object* v___y_298_; lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_List_min_x3f___at___00Mathlib_CountHeartbeats_variation_spec__4(v_counts_276_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v___x_303_; 
v___x_303_ = lean_unsigned_to_nat(0u);
v___y_298_ = v___x_303_;
goto v___jp_297_;
}
else
{
lean_object* v_val_304_; 
v_val_304_ = lean_ctor_get(v___x_302_, 0);
lean_inc(v_val_304_);
lean_dec_ref_known(v___x_302_, 1);
v___y_298_ = v_val_304_;
goto v___jp_297_;
}
v___jp_277_:
{
lean_object* v___x_280_; lean_object* v_counts_x27_281_; double v___x_282_; double v___x_283_; lean_object* v___x_284_; uint64_t v___x_285_; double v___x_286_; double v_00_u03bc_287_; lean_object* v___x_288_; double v___x_289_; double v___x_290_; double v_stddev_291_; uint64_t v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_280_ = lean_box(0);
lean_inc(v_counts_276_);
v_counts_x27_281_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__0(v_counts_276_, v___x_280_);
v___x_282_ = lean_float_once(&lp_mathlib_Mathlib_CountHeartbeats_variation___closed__0, &lp_mathlib_Mathlib_CountHeartbeats_variation___closed__0_once, _init_lp_mathlib_Mathlib_CountHeartbeats_variation___closed__0);
v___x_283_ = lp_mathlib_List_foldl___at___00Mathlib_CountHeartbeats_variation_spec__1(v___x_282_, v_counts_x27_281_);
v___x_284_ = l_List_lengthTR___redArg(v_counts_276_);
lean_dec(v_counts_276_);
v___x_285_ = lean_uint64_of_nat(v___x_284_);
lean_dec(v___x_284_);
v___x_286_ = lean_uint64_to_float(v___x_285_);
v_00_u03bc_287_ = lean_float_div(v___x_283_, v___x_286_);
v___x_288_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_CountHeartbeats_variation_spec__2(v_00_u03bc_287_, v_counts_x27_281_, v___x_280_);
v___x_289_ = lp_mathlib_List_foldl___at___00Mathlib_CountHeartbeats_variation_spec__1(v___x_282_, v___x_288_);
lean_dec(v___x_288_);
v___x_290_ = lean_float_div(v___x_289_, v___x_286_);
v_stddev_291_ = sqrt(v___x_290_);
v___x_292_ = lean_float_to_uint64(v_stddev_291_);
v___x_293_ = lean_uint64_to_nat(v___x_292_);
v___x_294_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
lean_ctor_set(v___x_294_, 1, v___x_280_);
v___x_295_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_295_, 0, v___y_279_);
lean_ctor_set(v___x_295_, 1, v___x_294_);
v___x_296_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_296_, 0, v___y_278_);
lean_ctor_set(v___x_296_, 1, v___x_295_);
return v___x_296_;
}
v___jp_297_:
{
lean_object* v___x_299_; 
v___x_299_ = lp_mathlib_List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3(v_counts_276_);
if (lean_obj_tag(v___x_299_) == 0)
{
lean_object* v___x_300_; 
v___x_300_ = lean_unsigned_to_nat(0u);
v___y_278_ = v___y_298_;
v___y_279_ = v___x_300_;
goto v___jp_277_;
}
else
{
lean_object* v_val_301_; 
v_val_301_ = lean_ctor_get(v___x_299_, 0);
lean_inc(v_val_301_);
lean_dec_ref_known(v___x_299_, 1);
v___y_278_ = v___y_298_;
v___y_279_ = v_val_301_;
goto v___jp_277_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg(lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_counts_313_){
_start:
{
lean_object* v_toApplicative_314_; lean_object* v_toPure_315_; lean_object* v___x_319_; 
v_toApplicative_314_ = lean_ctor_get(v_inst_309_, 0);
v_toPure_315_ = lean_ctor_get(v_toApplicative_314_, 1);
v___x_319_ = lp_mathlib_Mathlib_CountHeartbeats_variation(v_counts_313_);
if (lean_obj_tag(v___x_319_) == 1)
{
lean_object* v_tail_320_; 
v_tail_320_ = lean_ctor_get(v___x_319_, 1);
lean_inc(v_tail_320_);
if (lean_obj_tag(v_tail_320_) == 1)
{
lean_object* v_tail_321_; 
v_tail_321_ = lean_ctor_get(v_tail_320_, 1);
lean_inc(v_tail_321_);
if (lean_obj_tag(v_tail_321_) == 1)
{
lean_object* v_tail_322_; 
v_tail_322_ = lean_ctor_get(v_tail_321_, 1);
if (lean_obj_tag(v_tail_322_) == 0)
{
lean_object* v_head_323_; lean_object* v_head_324_; lean_object* v_head_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v_head_323_ = lean_ctor_get(v___x_319_, 0);
lean_inc(v_head_323_);
lean_dec_ref_known(v___x_319_, 2);
v_head_324_ = lean_ctor_get(v_tail_320_, 0);
lean_inc(v_head_324_);
lean_dec_ref_known(v_tail_320_, 2);
v_head_325_ = lean_ctor_get(v_tail_321_, 0);
lean_inc(v_head_325_);
lean_dec_ref_known(v_tail_321_, 2);
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__0));
v___x_327_ = lean_unsigned_to_nat(1000u);
v___x_328_ = lean_nat_div(v_head_323_, v___x_327_);
lean_dec(v_head_323_);
v___x_329_ = l_Nat_reprFast(v___x_328_);
v___x_330_ = lean_string_append(v___x_326_, v___x_329_);
lean_dec_ref(v___x_329_);
v___x_331_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__1));
v___x_332_ = lean_string_append(v___x_330_, v___x_331_);
v___x_333_ = lean_nat_div(v_head_324_, v___x_327_);
lean_dec(v_head_324_);
v___x_334_ = l_Nat_reprFast(v___x_333_);
v___x_335_ = lean_string_append(v___x_332_, v___x_334_);
lean_dec_ref(v___x_334_);
v___x_336_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__2));
v___x_337_ = lean_string_append(v___x_335_, v___x_336_);
v___x_338_ = lean_unsigned_to_nat(10u);
v___x_339_ = lean_nat_div(v_head_325_, v___x_338_);
lean_dec(v_head_325_);
v___x_340_ = l_Nat_reprFast(v___x_339_);
v___x_341_ = lean_string_append(v___x_337_, v___x_340_);
lean_dec_ref(v___x_340_);
v___x_342_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__3));
v___x_343_ = lean_string_append(v___x_341_, v___x_342_);
v___x_344_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_344_, 0, v___x_343_);
v___x_345_ = l_Lean_MessageData_ofFormat(v___x_344_);
v___x_346_ = l_Lean_logInfo___redArg(v_inst_309_, v_inst_310_, v_inst_311_, v_inst_312_, v___x_345_);
return v___x_346_;
}
else
{
lean_inc(v_toPure_315_);
lean_dec_ref_known(v_tail_321_, 2);
lean_dec_ref_known(v_tail_320_, 2);
lean_dec_ref_known(v___x_319_, 2);
lean_dec(v_inst_312_);
lean_dec(v_inst_311_);
lean_dec_ref(v_inst_310_);
lean_dec_ref(v_inst_309_);
goto v___jp_316_;
}
}
else
{
lean_inc(v_toPure_315_);
lean_dec(v_tail_321_);
lean_dec_ref_known(v_tail_320_, 2);
lean_dec_ref_known(v___x_319_, 2);
lean_dec(v_inst_312_);
lean_dec(v_inst_311_);
lean_dec_ref(v_inst_310_);
lean_dec_ref(v_inst_309_);
goto v___jp_316_;
}
}
else
{
lean_inc(v_toPure_315_);
lean_dec_ref_known(v___x_319_, 2);
lean_dec(v_tail_320_);
lean_dec(v_inst_312_);
lean_dec(v_inst_311_);
lean_dec_ref(v_inst_310_);
lean_dec_ref(v_inst_309_);
goto v___jp_316_;
}
}
else
{
lean_inc(v_toPure_315_);
lean_dec(v___x_319_);
lean_dec(v_inst_312_);
lean_dec(v_inst_311_);
lean_dec_ref(v_inst_310_);
lean_dec_ref(v_inst_309_);
goto v___jp_316_;
}
v___jp_316_:
{
lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_317_ = lean_box(0);
v___x_318_ = lean_apply_2(v_toPure_315_, lean_box(0), v___x_317_);
return v___x_318_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation(lean_object* v_m_347_, lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_counts_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg(v_inst_348_, v_inst_349_, v_inst_350_, v_inst_351_, v_counts_352_);
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_382_ = lean_box(0);
v___x_383_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_384_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_384_, 0, v___x_383_);
lean_ctor_set(v___x_384_, 1, v___x_382_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg(){
_start:
{
lean_object* v___x_386_; lean_object* v___x_387_; 
v___x_386_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0);
v___x_387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_387_, 0, v___x_386_);
return v___x_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___boxed(lean_object* v___y_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg();
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0(lean_object* v_00_u03b1_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg();
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___boxed(lean_object* v_00_u03b1_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0(v_00_u03b1_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_);
lean_dec(v___y_409_);
lean_dec_ref(v___y_408_);
lean_dec(v___y_407_);
lean_dec_ref(v___y_406_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2_spec__3(lean_object* v_msgData_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
lean_object* v___x_418_; lean_object* v_env_419_; lean_object* v___x_420_; lean_object* v_mctx_421_; lean_object* v_lctx_422_; lean_object* v_options_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v___x_418_ = lean_st_ref_get(v___y_416_);
v_env_419_ = lean_ctor_get(v___x_418_, 0);
lean_inc_ref(v_env_419_);
lean_dec(v___x_418_);
v___x_420_ = lean_st_ref_get(v___y_414_);
v_mctx_421_ = lean_ctor_get(v___x_420_, 0);
lean_inc_ref(v_mctx_421_);
lean_dec(v___x_420_);
v_lctx_422_ = lean_ctor_get(v___y_413_, 2);
v_options_423_ = lean_ctor_get(v___y_415_, 2);
lean_inc_ref(v_options_423_);
lean_inc_ref(v_lctx_422_);
v___x_424_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_424_, 0, v_env_419_);
lean_ctor_set(v___x_424_, 1, v_mctx_421_);
lean_ctor_set(v___x_424_, 2, v_lctx_422_);
lean_ctor_set(v___x_424_, 3, v_options_423_);
v___x_425_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_425_, 0, v___x_424_);
lean_ctor_set(v___x_425_, 1, v_msgData_412_);
v___x_426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_426_, 0, v___x_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2_spec__3___boxed(lean_object* v_msgData_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_){
_start:
{
lean_object* v_res_433_; 
v_res_433_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2_spec__3(v_msgData_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_);
lean_dec(v___y_431_);
lean_dec_ref(v___y_430_);
lean_dec(v___y_429_);
lean_dec_ref(v___y_428_);
return v_res_433_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0(uint8_t v___y_440_, uint8_t v_suppressElabErrors_441_, lean_object* v_x_442_){
_start:
{
if (lean_obj_tag(v_x_442_) == 1)
{
lean_object* v_pre_443_; 
v_pre_443_ = lean_ctor_get(v_x_442_, 0);
switch(lean_obj_tag(v_pre_443_))
{
case 1:
{
lean_object* v_pre_444_; 
v_pre_444_ = lean_ctor_get(v_pre_443_, 0);
switch(lean_obj_tag(v_pre_444_))
{
case 0:
{
lean_object* v_str_445_; lean_object* v_str_446_; lean_object* v___x_447_; uint8_t v___x_448_; 
v_str_445_ = lean_ctor_get(v_x_442_, 1);
v_str_446_ = lean_ctor_get(v_pre_443_, 1);
v___x_447_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__1));
v___x_448_ = lean_string_dec_eq(v_str_446_, v___x_447_);
if (v___x_448_ == 0)
{
lean_object* v___x_449_; uint8_t v___x_450_; 
v___x_449_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__0));
v___x_450_ = lean_string_dec_eq(v_str_446_, v___x_449_);
if (v___x_450_ == 0)
{
return v___y_440_;
}
else
{
lean_object* v___x_451_; uint8_t v___x_452_; 
v___x_451_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__1));
v___x_452_ = lean_string_dec_eq(v_str_445_, v___x_451_);
if (v___x_452_ == 0)
{
return v___y_440_;
}
else
{
return v_suppressElabErrors_441_;
}
}
}
else
{
lean_object* v___x_453_; uint8_t v___x_454_; 
v___x_453_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__2));
v___x_454_ = lean_string_dec_eq(v_str_445_, v___x_453_);
if (v___x_454_ == 0)
{
return v___y_440_;
}
else
{
return v_suppressElabErrors_441_;
}
}
}
case 1:
{
lean_object* v_pre_455_; 
v_pre_455_ = lean_ctor_get(v_pre_444_, 0);
if (lean_obj_tag(v_pre_455_) == 0)
{
lean_object* v_str_456_; lean_object* v_str_457_; lean_object* v_str_458_; lean_object* v___x_459_; uint8_t v___x_460_; 
v_str_456_ = lean_ctor_get(v_x_442_, 1);
v_str_457_ = lean_ctor_get(v_pre_443_, 1);
v_str_458_ = lean_ctor_get(v_pre_444_, 1);
v___x_459_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__3));
v___x_460_ = lean_string_dec_eq(v_str_458_, v___x_459_);
if (v___x_460_ == 0)
{
return v___y_440_;
}
else
{
lean_object* v___x_461_; uint8_t v___x_462_; 
v___x_461_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__4));
v___x_462_ = lean_string_dec_eq(v_str_457_, v___x_461_);
if (v___x_462_ == 0)
{
return v___y_440_;
}
else
{
lean_object* v___x_463_; uint8_t v___x_464_; 
v___x_463_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___closed__5));
v___x_464_ = lean_string_dec_eq(v_str_456_, v___x_463_);
if (v___x_464_ == 0)
{
return v___y_440_;
}
else
{
return v_suppressElabErrors_441_;
}
}
}
}
else
{
return v___y_440_;
}
}
default: 
{
return v___y_440_;
}
}
}
case 0:
{
lean_object* v_str_465_; lean_object* v___x_466_; uint8_t v___x_467_; 
v_str_465_ = lean_ctor_get(v_x_442_, 1);
v___x_466_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__0));
v___x_467_ = lean_string_dec_eq(v_str_465_, v___x_466_);
if (v___x_467_ == 0)
{
return v___y_440_;
}
else
{
return v_suppressElabErrors_441_;
}
}
default: 
{
return v___y_440_;
}
}
}
else
{
return v___y_440_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___boxed(lean_object* v___y_468_, lean_object* v_suppressElabErrors_469_, lean_object* v_x_470_){
_start:
{
uint8_t v___y_4903__boxed_471_; uint8_t v_suppressElabErrors_boxed_472_; uint8_t v_res_473_; lean_object* v_r_474_; 
v___y_4903__boxed_471_ = lean_unbox(v___y_468_);
v_suppressElabErrors_boxed_472_ = lean_unbox(v_suppressElabErrors_469_);
v_res_473_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0(v___y_4903__boxed_471_, v_suppressElabErrors_boxed_472_, v_x_470_);
lean_dec(v_x_470_);
v_r_474_ = lean_box(v_res_473_);
return v_r_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg(lean_object* v_ref_476_, lean_object* v_msgData_477_, uint8_t v_severity_478_, uint8_t v_isSilent_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_){
_start:
{
lean_object* v___y_486_; lean_object* v___y_487_; lean_object* v___y_488_; uint8_t v___y_489_; lean_object* v___y_490_; uint8_t v___y_491_; lean_object* v___y_492_; lean_object* v___y_493_; lean_object* v___y_494_; lean_object* v___y_522_; uint8_t v___y_523_; lean_object* v___y_524_; uint8_t v___y_525_; lean_object* v___y_526_; uint8_t v___y_527_; lean_object* v___y_528_; lean_object* v___y_529_; lean_object* v___y_547_; uint8_t v___y_548_; lean_object* v___y_549_; lean_object* v___y_550_; uint8_t v___y_551_; uint8_t v___y_552_; lean_object* v___y_553_; lean_object* v___y_554_; lean_object* v___y_558_; lean_object* v___y_559_; uint8_t v___y_560_; lean_object* v___y_561_; uint8_t v___y_562_; lean_object* v___y_563_; uint8_t v___y_564_; uint8_t v___x_569_; lean_object* v___y_571_; lean_object* v___y_572_; lean_object* v___y_573_; uint8_t v___y_574_; lean_object* v___y_575_; uint8_t v___y_576_; uint8_t v___y_577_; uint8_t v___y_579_; uint8_t v___x_594_; 
v___x_569_ = 2;
v___x_594_ = l_Lean_instBEqMessageSeverity_beq(v_severity_478_, v___x_569_);
if (v___x_594_ == 0)
{
v___y_579_ = v___x_594_;
goto v___jp_578_;
}
else
{
uint8_t v___x_595_; 
lean_inc_ref(v_msgData_477_);
v___x_595_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_477_);
v___y_579_ = v___x_595_;
goto v___jp_578_;
}
v___jp_485_:
{
lean_object* v___x_495_; lean_object* v_currNamespace_496_; lean_object* v_openDecls_497_; lean_object* v_env_498_; lean_object* v_nextMacroScope_499_; lean_object* v_ngen_500_; lean_object* v_auxDeclNGen_501_; lean_object* v_traceState_502_; lean_object* v_cache_503_; lean_object* v_messages_504_; lean_object* v_infoState_505_; lean_object* v_snapshotTasks_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_520_; 
v___x_495_ = lean_st_ref_take(v___y_494_);
v_currNamespace_496_ = lean_ctor_get(v___y_493_, 6);
v_openDecls_497_ = lean_ctor_get(v___y_493_, 7);
v_env_498_ = lean_ctor_get(v___x_495_, 0);
v_nextMacroScope_499_ = lean_ctor_get(v___x_495_, 1);
v_ngen_500_ = lean_ctor_get(v___x_495_, 2);
v_auxDeclNGen_501_ = lean_ctor_get(v___x_495_, 3);
v_traceState_502_ = lean_ctor_get(v___x_495_, 4);
v_cache_503_ = lean_ctor_get(v___x_495_, 5);
v_messages_504_ = lean_ctor_get(v___x_495_, 6);
v_infoState_505_ = lean_ctor_get(v___x_495_, 7);
v_snapshotTasks_506_ = lean_ctor_get(v___x_495_, 8);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_520_ == 0)
{
v___x_508_ = v___x_495_;
v_isShared_509_ = v_isSharedCheck_520_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_snapshotTasks_506_);
lean_inc(v_infoState_505_);
lean_inc(v_messages_504_);
lean_inc(v_cache_503_);
lean_inc(v_traceState_502_);
lean_inc(v_auxDeclNGen_501_);
lean_inc(v_ngen_500_);
lean_inc(v_nextMacroScope_499_);
lean_inc(v_env_498_);
lean_dec(v___x_495_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_520_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_515_; 
lean_inc(v_openDecls_497_);
lean_inc(v_currNamespace_496_);
v___x_510_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_510_, 0, v_currNamespace_496_);
lean_ctor_set(v___x_510_, 1, v_openDecls_497_);
v___x_511_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_511_, 0, v___x_510_);
lean_ctor_set(v___x_511_, 1, v___y_487_);
lean_inc_ref(v___y_488_);
lean_inc_ref(v___y_490_);
v___x_512_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_512_, 0, v___y_490_);
lean_ctor_set(v___x_512_, 1, v___y_492_);
lean_ctor_set(v___x_512_, 2, v___y_486_);
lean_ctor_set(v___x_512_, 3, v___y_488_);
lean_ctor_set(v___x_512_, 4, v___x_511_);
lean_ctor_set_uint8(v___x_512_, sizeof(void*)*5, v___y_491_);
lean_ctor_set_uint8(v___x_512_, sizeof(void*)*5 + 1, v___y_489_);
lean_ctor_set_uint8(v___x_512_, sizeof(void*)*5 + 2, v_isSilent_479_);
v___x_513_ = l_Lean_MessageLog_add(v___x_512_, v_messages_504_);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 6, v___x_513_);
v___x_515_ = v___x_508_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_env_498_);
lean_ctor_set(v_reuseFailAlloc_519_, 1, v_nextMacroScope_499_);
lean_ctor_set(v_reuseFailAlloc_519_, 2, v_ngen_500_);
lean_ctor_set(v_reuseFailAlloc_519_, 3, v_auxDeclNGen_501_);
lean_ctor_set(v_reuseFailAlloc_519_, 4, v_traceState_502_);
lean_ctor_set(v_reuseFailAlloc_519_, 5, v_cache_503_);
lean_ctor_set(v_reuseFailAlloc_519_, 6, v___x_513_);
lean_ctor_set(v_reuseFailAlloc_519_, 7, v_infoState_505_);
lean_ctor_set(v_reuseFailAlloc_519_, 8, v_snapshotTasks_506_);
v___x_515_ = v_reuseFailAlloc_519_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; 
v___x_516_ = lean_st_ref_set(v___y_494_, v___x_515_);
v___x_517_ = lean_box(0);
v___x_518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_518_, 0, v___x_517_);
return v___x_518_;
}
}
}
v___jp_521_:
{
lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v_a_532_; lean_object* v___x_534_; uint8_t v_isShared_535_; uint8_t v_isSharedCheck_545_; 
v___x_530_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_477_);
v___x_531_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2_spec__3(v___x_530_, v___y_480_, v___y_481_, v___y_482_, v___y_483_);
v_a_532_ = lean_ctor_get(v___x_531_, 0);
v_isSharedCheck_545_ = !lean_is_exclusive(v___x_531_);
if (v_isSharedCheck_545_ == 0)
{
v___x_534_ = v___x_531_;
v_isShared_535_ = v_isSharedCheck_545_;
goto v_resetjp_533_;
}
else
{
lean_inc(v_a_532_);
lean_dec(v___x_531_);
v___x_534_ = lean_box(0);
v_isShared_535_ = v_isSharedCheck_545_;
goto v_resetjp_533_;
}
v_resetjp_533_:
{
lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
lean_inc_ref_n(v___y_528_, 2);
v___x_536_ = l_Lean_FileMap_toPosition(v___y_528_, v___y_526_);
lean_dec(v___y_526_);
v___x_537_ = l_Lean_FileMap_toPosition(v___y_528_, v___y_529_);
lean_dec(v___y_529_);
v___x_538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_538_, 0, v___x_537_);
v___x_539_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___closed__0));
if (v___y_527_ == 0)
{
lean_del_object(v___x_534_);
lean_dec_ref(v___y_522_);
v___y_486_ = v___x_538_;
v___y_487_ = v_a_532_;
v___y_488_ = v___x_539_;
v___y_489_ = v___y_523_;
v___y_490_ = v___y_524_;
v___y_491_ = v___y_525_;
v___y_492_ = v___x_536_;
v___y_493_ = v___y_482_;
v___y_494_ = v___y_483_;
goto v___jp_485_;
}
else
{
uint8_t v___x_540_; 
lean_inc(v_a_532_);
v___x_540_ = l_Lean_MessageData_hasTag(v___y_522_, v_a_532_);
if (v___x_540_ == 0)
{
lean_object* v___x_541_; lean_object* v___x_543_; 
lean_dec_ref_known(v___x_538_, 1);
lean_dec_ref(v___x_536_);
lean_dec(v_a_532_);
v___x_541_ = lean_box(0);
if (v_isShared_535_ == 0)
{
lean_ctor_set(v___x_534_, 0, v___x_541_);
v___x_543_ = v___x_534_;
goto v_reusejp_542_;
}
else
{
lean_object* v_reuseFailAlloc_544_; 
v_reuseFailAlloc_544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_544_, 0, v___x_541_);
v___x_543_ = v_reuseFailAlloc_544_;
goto v_reusejp_542_;
}
v_reusejp_542_:
{
return v___x_543_;
}
}
else
{
lean_del_object(v___x_534_);
v___y_486_ = v___x_538_;
v___y_487_ = v_a_532_;
v___y_488_ = v___x_539_;
v___y_489_ = v___y_523_;
v___y_490_ = v___y_524_;
v___y_491_ = v___y_525_;
v___y_492_ = v___x_536_;
v___y_493_ = v___y_482_;
v___y_494_ = v___y_483_;
goto v___jp_485_;
}
}
}
}
v___jp_546_:
{
lean_object* v___x_555_; 
v___x_555_ = l_Lean_Syntax_getTailPos_x3f(v___y_549_, v___y_551_);
lean_dec(v___y_549_);
if (lean_obj_tag(v___x_555_) == 0)
{
lean_inc(v___y_554_);
v___y_522_ = v___y_547_;
v___y_523_ = v___y_548_;
v___y_524_ = v___y_550_;
v___y_525_ = v___y_551_;
v___y_526_ = v___y_554_;
v___y_527_ = v___y_552_;
v___y_528_ = v___y_553_;
v___y_529_ = v___y_554_;
goto v___jp_521_;
}
else
{
lean_object* v_val_556_; 
v_val_556_ = lean_ctor_get(v___x_555_, 0);
lean_inc(v_val_556_);
lean_dec_ref_known(v___x_555_, 1);
v___y_522_ = v___y_547_;
v___y_523_ = v___y_548_;
v___y_524_ = v___y_550_;
v___y_525_ = v___y_551_;
v___y_526_ = v___y_554_;
v___y_527_ = v___y_552_;
v___y_528_ = v___y_553_;
v___y_529_ = v_val_556_;
goto v___jp_521_;
}
}
v___jp_557_:
{
lean_object* v_ref_565_; lean_object* v___x_566_; 
v_ref_565_ = l_Lean_replaceRef(v_ref_476_, v___y_561_);
v___x_566_ = l_Lean_Syntax_getPos_x3f(v_ref_565_, v___y_560_);
if (lean_obj_tag(v___x_566_) == 0)
{
lean_object* v___x_567_; 
v___x_567_ = lean_unsigned_to_nat(0u);
v___y_547_ = v___y_558_;
v___y_548_ = v___y_564_;
v___y_549_ = v_ref_565_;
v___y_550_ = v___y_559_;
v___y_551_ = v___y_560_;
v___y_552_ = v___y_562_;
v___y_553_ = v___y_563_;
v___y_554_ = v___x_567_;
goto v___jp_546_;
}
else
{
lean_object* v_val_568_; 
v_val_568_ = lean_ctor_get(v___x_566_, 0);
lean_inc(v_val_568_);
lean_dec_ref_known(v___x_566_, 1);
v___y_547_ = v___y_558_;
v___y_548_ = v___y_564_;
v___y_549_ = v_ref_565_;
v___y_550_ = v___y_559_;
v___y_551_ = v___y_560_;
v___y_552_ = v___y_562_;
v___y_553_ = v___y_563_;
v___y_554_ = v_val_568_;
goto v___jp_546_;
}
}
v___jp_570_:
{
if (v___y_577_ == 0)
{
v___y_558_ = v___y_572_;
v___y_559_ = v___y_571_;
v___y_560_ = v___y_576_;
v___y_561_ = v___y_573_;
v___y_562_ = v___y_574_;
v___y_563_ = v___y_575_;
v___y_564_ = v_severity_478_;
goto v___jp_557_;
}
else
{
v___y_558_ = v___y_572_;
v___y_559_ = v___y_571_;
v___y_560_ = v___y_576_;
v___y_561_ = v___y_573_;
v___y_562_ = v___y_574_;
v___y_563_ = v___y_575_;
v___y_564_ = v___x_569_;
goto v___jp_557_;
}
}
v___jp_578_:
{
if (v___y_579_ == 0)
{
lean_object* v_fileName_580_; lean_object* v_fileMap_581_; lean_object* v_options_582_; lean_object* v_ref_583_; uint8_t v_suppressElabErrors_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___f_587_; uint8_t v___x_588_; uint8_t v___x_589_; 
v_fileName_580_ = lean_ctor_get(v___y_482_, 0);
v_fileMap_581_ = lean_ctor_get(v___y_482_, 1);
v_options_582_ = lean_ctor_get(v___y_482_, 2);
v_ref_583_ = lean_ctor_get(v___y_482_, 5);
v_suppressElabErrors_584_ = lean_ctor_get_uint8(v___y_482_, sizeof(void*)*14 + 1);
v___x_585_ = lean_box(v___y_579_);
v___x_586_ = lean_box(v_suppressElabErrors_584_);
v___f_587_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_587_, 0, v___x_585_);
lean_closure_set(v___f_587_, 1, v___x_586_);
v___x_588_ = 1;
v___x_589_ = l_Lean_instBEqMessageSeverity_beq(v_severity_478_, v___x_588_);
if (v___x_589_ == 0)
{
v___y_571_ = v_fileName_580_;
v___y_572_ = v___f_587_;
v___y_573_ = v_ref_583_;
v___y_574_ = v_suppressElabErrors_584_;
v___y_575_ = v_fileMap_581_;
v___y_576_ = v___y_579_;
v___y_577_ = v___x_589_;
goto v___jp_570_;
}
else
{
lean_object* v___x_590_; uint8_t v___x_591_; 
v___x_590_ = l_Lean_warningAsError;
v___x_591_ = lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1(v_options_582_, v___x_590_);
v___y_571_ = v_fileName_580_;
v___y_572_ = v___f_587_;
v___y_573_ = v_ref_583_;
v___y_574_ = v_suppressElabErrors_584_;
v___y_575_ = v_fileMap_581_;
v___y_576_ = v___y_579_;
v___y_577_ = v___x_591_;
goto v___jp_570_;
}
}
else
{
lean_object* v___x_592_; lean_object* v___x_593_; 
lean_dec_ref(v_msgData_477_);
v___x_592_ = lean_box(0);
v___x_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
return v___x_593_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_ref_596_, lean_object* v_msgData_597_, lean_object* v_severity_598_, lean_object* v_isSilent_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_){
_start:
{
uint8_t v_severity_boxed_605_; uint8_t v_isSilent_boxed_606_; lean_object* v_res_607_; 
v_severity_boxed_605_ = lean_unbox(v_severity_598_);
v_isSilent_boxed_606_ = lean_unbox(v_isSilent_599_);
v_res_607_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg(v_ref_596_, v_msgData_597_, v_severity_boxed_605_, v_isSilent_boxed_606_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
lean_dec(v___y_603_);
lean_dec_ref(v___y_602_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
lean_dec(v_ref_596_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1(lean_object* v_msgData_608_, uint8_t v_severity_609_, uint8_t v_isSilent_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_){
_start:
{
lean_object* v_ref_620_; lean_object* v___x_621_; 
v_ref_620_ = lean_ctor_get(v___y_617_, 5);
v___x_621_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg(v_ref_620_, v_msgData_608_, v_severity_609_, v_isSilent_610_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
return v___x_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1___boxed(lean_object* v_msgData_622_, lean_object* v_severity_623_, lean_object* v_isSilent_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_){
_start:
{
uint8_t v_severity_boxed_634_; uint8_t v_isSilent_boxed_635_; lean_object* v_res_636_; 
v_severity_boxed_634_ = lean_unbox(v_severity_623_);
v_isSilent_boxed_635_ = lean_unbox(v_isSilent_624_);
v_res_636_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1(v_msgData_622_, v_severity_boxed_634_, v_isSilent_boxed_635_, v___y_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_);
lean_dec(v___y_632_);
lean_dec_ref(v___y_631_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
lean_dec(v___y_628_);
lean_dec_ref(v___y_627_);
lean_dec(v___y_626_);
lean_dec_ref(v___y_625_);
return v_res_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1(lean_object* v_msgData_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
uint8_t v___x_647_; uint8_t v___x_648_; lean_object* v___x_649_; 
v___x_647_ = 0;
v___x_648_ = 0;
v___x_649_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1(v_msgData_637_, v___x_647_, v___x_648_, v___y_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1___boxed(lean_object* v_msgData_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_){
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1(v_msgData_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1(lean_object* v_x_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_, lean_object* v_a_668_, lean_object* v_a_669_){
_start:
{
lean_object* v___x_671_; uint8_t v___x_672_; 
v___x_671_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__3));
lean_inc(v_x_661_);
v___x_672_ = l_Lean_Syntax_isOfKind(v_x_661_, v___x_671_);
if (v___x_672_ == 0)
{
lean_object* v___x_673_; 
lean_dec(v_x_661_);
v___x_673_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg();
return v___x_673_;
}
else
{
lean_object* v___x_674_; lean_object* v_tac_675_; uint8_t v___x_676_; lean_object* v___x_677_; 
v___x_674_ = lean_unsigned_to_nat(1u);
v_tac_675_ = l_Lean_Syntax_getArg(v_x_661_, v___x_674_);
lean_dec(v_x_661_);
v___x_676_ = 0;
v___x_677_ = lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats(v_tac_675_, v___x_676_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_, v_a_668_, v_a_669_);
if (lean_obj_tag(v___x_677_) == 0)
{
lean_object* v_a_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v_a_678_ = lean_ctor_get(v___x_677_, 0);
lean_inc(v_a_678_);
lean_dec_ref_known(v___x_677_, 1);
v___x_679_ = l_Nat_reprFast(v_a_678_);
v___x_680_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_680_, 0, v___x_679_);
v___x_681_ = l_Lean_MessageData_ofFormat(v___x_680_);
v___x_682_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1(v___x_681_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_, v_a_668_, v_a_669_);
return v___x_682_;
}
else
{
lean_object* v_a_683_; lean_object* v___x_685_; uint8_t v_isShared_686_; uint8_t v_isSharedCheck_690_; 
v_a_683_ = lean_ctor_get(v___x_677_, 0);
v_isSharedCheck_690_ = !lean_is_exclusive(v___x_677_);
if (v_isSharedCheck_690_ == 0)
{
v___x_685_ = v___x_677_;
v_isShared_686_ = v_isSharedCheck_690_;
goto v_resetjp_684_;
}
else
{
lean_inc(v_a_683_);
lean_dec(v___x_677_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1___boxed(lean_object* v_x_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_, lean_object* v_a_695_, lean_object* v_a_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_){
_start:
{
lean_object* v_res_701_; 
v_res_701_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1(v_x_691_, v_a_692_, v_a_693_, v_a_694_, v_a_695_, v_a_696_, v_a_697_, v_a_698_, v_a_699_);
lean_dec(v_a_699_);
lean_dec_ref(v_a_698_);
lean_dec(v_a_697_);
lean_dec_ref(v_a_696_);
lean_dec(v_a_695_);
lean_dec_ref(v_a_694_);
lean_dec(v_a_693_);
lean_dec_ref(v_a_692_);
return v_res_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2(lean_object* v_ref_702_, lean_object* v_msgData_703_, uint8_t v_severity_704_, uint8_t v_isSilent_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_){
_start:
{
lean_object* v___x_715_; 
v___x_715_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg(v_ref_702_, v_msgData_703_, v_severity_704_, v_isSilent_705_, v___y_710_, v___y_711_, v___y_712_, v___y_713_);
return v___x_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_716_, lean_object* v_msgData_717_, lean_object* v_severity_718_, lean_object* v_isSilent_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_){
_start:
{
uint8_t v_severity_boxed_729_; uint8_t v_isSilent_boxed_730_; lean_object* v_res_731_; 
v_severity_boxed_729_ = lean_unbox(v_severity_718_);
v_isSilent_boxed_730_ = lean_unbox(v_isSilent_719_);
v_res_731_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2(v_ref_716_, v_msgData_717_, v_severity_boxed_729_, v_isSilent_boxed_730_, v___y_720_, v___y_721_, v___y_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_, v___y_727_);
lean_dec(v___y_727_);
lean_dec_ref(v___y_726_);
lean_dec(v___y_725_);
lean_dec_ref(v___y_724_);
lean_dec(v___y_723_);
lean_dec_ref(v___y_722_);
lean_dec(v___y_721_);
lean_dec_ref(v___y_720_);
lean_dec(v_ref_716_);
return v_res_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__1(lean_object* v_counts_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_){
_start:
{
lean_object* v___x_800_; 
v___x_800_ = lp_mathlib_Mathlib_CountHeartbeats_variation(v_counts_787_);
if (lean_obj_tag(v___x_800_) == 1)
{
lean_object* v_tail_801_; 
v_tail_801_ = lean_ctor_get(v___x_800_, 1);
lean_inc(v_tail_801_);
if (lean_obj_tag(v_tail_801_) == 1)
{
lean_object* v_tail_802_; 
v_tail_802_ = lean_ctor_get(v_tail_801_, 1);
lean_inc(v_tail_802_);
if (lean_obj_tag(v_tail_802_) == 1)
{
lean_object* v_tail_803_; 
v_tail_803_ = lean_ctor_get(v_tail_802_, 1);
if (lean_obj_tag(v_tail_803_) == 0)
{
lean_object* v_head_804_; lean_object* v_head_805_; lean_object* v_head_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; 
v_head_804_ = lean_ctor_get(v___x_800_, 0);
lean_inc(v_head_804_);
lean_dec_ref_known(v___x_800_, 2);
v_head_805_ = lean_ctor_get(v_tail_801_, 0);
lean_inc(v_head_805_);
lean_dec_ref_known(v_tail_801_, 2);
v_head_806_ = lean_ctor_get(v_tail_802_, 0);
lean_inc(v_head_806_);
lean_dec_ref_known(v_tail_802_, 2);
v___x_807_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__0));
v___x_808_ = lean_unsigned_to_nat(1000u);
v___x_809_ = lean_nat_div(v_head_804_, v___x_808_);
lean_dec(v_head_804_);
v___x_810_ = l_Nat_reprFast(v___x_809_);
v___x_811_ = lean_string_append(v___x_807_, v___x_810_);
lean_dec_ref(v___x_810_);
v___x_812_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__1));
v___x_813_ = lean_string_append(v___x_811_, v___x_812_);
v___x_814_ = lean_nat_div(v_head_805_, v___x_808_);
lean_dec(v_head_805_);
v___x_815_ = l_Nat_reprFast(v___x_814_);
v___x_816_ = lean_string_append(v___x_813_, v___x_815_);
lean_dec_ref(v___x_815_);
v___x_817_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__2));
v___x_818_ = lean_string_append(v___x_816_, v___x_817_);
v___x_819_ = lean_unsigned_to_nat(10u);
v___x_820_ = lean_nat_div(v_head_806_, v___x_819_);
lean_dec(v_head_806_);
v___x_821_ = l_Nat_reprFast(v___x_820_);
v___x_822_ = lean_string_append(v___x_818_, v___x_821_);
lean_dec_ref(v___x_821_);
v___x_823_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__3));
v___x_824_ = lean_string_append(v___x_822_, v___x_823_);
v___x_825_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_825_, 0, v___x_824_);
v___x_826_ = l_Lean_MessageData_ofFormat(v___x_825_);
v___x_827_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1(v___x_826_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_, v___y_795_);
return v___x_827_;
}
else
{
lean_dec_ref_known(v_tail_802_, 2);
lean_dec_ref_known(v_tail_801_, 2);
lean_dec_ref_known(v___x_800_, 2);
goto v___jp_797_;
}
}
else
{
lean_dec_ref_known(v_tail_801_, 2);
lean_dec(v_tail_802_);
lean_dec_ref_known(v___x_800_, 2);
goto v___jp_797_;
}
}
else
{
lean_dec(v_tail_801_);
lean_dec_ref_known(v___x_800_, 2);
goto v___jp_797_;
}
}
else
{
lean_dec(v___x_800_);
goto v___jp_797_;
}
v___jp_797_:
{
lean_object* v___x_798_; lean_object* v___x_799_; 
v___x_798_ = lean_box(0);
v___x_799_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_799_, 0, v___x_798_);
return v___x_799_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__1___boxed(lean_object* v_counts_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_){
_start:
{
lean_object* v_res_838_; 
v_res_838_ = lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__1(v_counts_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_, v___y_836_);
lean_dec(v___y_836_);
lean_dec_ref(v___y_835_);
lean_dec(v___y_834_);
lean_dec_ref(v___y_833_);
lean_dec(v___y_832_);
lean_dec_ref(v___y_831_);
lean_dec(v___y_830_);
lean_dec_ref(v___y_829_);
return v_res_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__0(lean_object* v_tac_839_, uint8_t v___x_840_, lean_object* v_x_841_, lean_object* v_x_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_){
_start:
{
if (lean_obj_tag(v_x_841_) == 0)
{
lean_object* v___x_852_; lean_object* v___x_853_; 
lean_dec(v_tac_839_);
v___x_852_ = l_List_reverse___redArg(v_x_842_);
v___x_853_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_853_, 0, v___x_852_);
return v___x_853_;
}
else
{
lean_object* v_tail_854_; lean_object* v___x_856_; uint8_t v_isShared_857_; uint8_t v_isSharedCheck_872_; 
v_tail_854_ = lean_ctor_get(v_x_841_, 1);
v_isSharedCheck_872_ = !lean_is_exclusive(v_x_841_);
if (v_isSharedCheck_872_ == 0)
{
lean_object* v_unused_873_; 
v_unused_873_ = lean_ctor_get(v_x_841_, 0);
lean_dec(v_unused_873_);
v___x_856_ = v_x_841_;
v_isShared_857_ = v_isSharedCheck_872_;
goto v_resetjp_855_;
}
else
{
lean_inc(v_tail_854_);
lean_dec(v_x_841_);
v___x_856_ = lean_box(0);
v_isShared_857_ = v_isSharedCheck_872_;
goto v_resetjp_855_;
}
v_resetjp_855_:
{
lean_object* v___x_858_; 
lean_inc(v_tac_839_);
v___x_858_ = lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats(v_tac_839_, v___x_840_, v___y_843_, v___y_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
if (lean_obj_tag(v___x_858_) == 0)
{
lean_object* v_a_859_; lean_object* v___x_861_; 
v_a_859_ = lean_ctor_get(v___x_858_, 0);
lean_inc(v_a_859_);
lean_dec_ref_known(v___x_858_, 1);
if (v_isShared_857_ == 0)
{
lean_ctor_set(v___x_856_, 1, v_x_842_);
lean_ctor_set(v___x_856_, 0, v_a_859_);
v___x_861_ = v___x_856_;
goto v_reusejp_860_;
}
else
{
lean_object* v_reuseFailAlloc_863_; 
v_reuseFailAlloc_863_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_863_, 0, v_a_859_);
lean_ctor_set(v_reuseFailAlloc_863_, 1, v_x_842_);
v___x_861_ = v_reuseFailAlloc_863_;
goto v_reusejp_860_;
}
v_reusejp_860_:
{
v_x_841_ = v_tail_854_;
v_x_842_ = v___x_861_;
goto _start;
}
}
else
{
lean_object* v_a_864_; lean_object* v___x_866_; uint8_t v_isShared_867_; uint8_t v_isSharedCheck_871_; 
lean_del_object(v___x_856_);
lean_dec(v_tail_854_);
lean_dec(v_x_842_);
lean_dec(v_tac_839_);
v_a_864_ = lean_ctor_get(v___x_858_, 0);
v_isSharedCheck_871_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_871_ == 0)
{
v___x_866_ = v___x_858_;
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
else
{
lean_inc(v_a_864_);
lean_dec(v___x_858_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__0___boxed(lean_object* v_tac_874_, lean_object* v___x_875_, lean_object* v_x_876_, lean_object* v_x_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_){
_start:
{
uint8_t v___x_1239__boxed_887_; lean_object* v_res_888_; 
v___x_1239__boxed_887_ = lean_unbox(v___x_875_);
v_res_888_ = lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__0(v_tac_874_, v___x_1239__boxed_887_, v_x_876_, v_x_877_, v___y_878_, v___y_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_, v___y_885_);
lean_dec(v___y_885_);
lean_dec_ref(v___y_884_);
lean_dec(v___y_883_);
lean_dec_ref(v___y_882_);
lean_dec(v___y_881_);
lean_dec_ref(v___y_880_);
lean_dec(v___y_879_);
lean_dec_ref(v___y_878_);
return v_res_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1(lean_object* v_x_889_, lean_object* v_a_890_, lean_object* v_a_891_, lean_object* v_a_892_, lean_object* v_a_893_, lean_object* v_a_894_, lean_object* v_a_895_, lean_object* v_a_896_, lean_object* v_a_897_){
_start:
{
lean_object* v___x_899_; uint8_t v___x_900_; 
v___x_899_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__1));
lean_inc(v_x_889_);
v___x_900_ = l_Lean_Syntax_isOfKind(v_x_889_, v___x_899_);
if (v___x_900_ == 0)
{
lean_object* v___x_901_; 
lean_dec(v_x_889_);
v___x_901_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg();
return v___x_901_;
}
else
{
lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v_tac_905_; lean_object* v___y_907_; lean_object* v___x_934_; 
v___x_902_ = lean_unsigned_to_nat(1u);
v___x_903_ = l_Lean_Syntax_getArg(v_x_889_, v___x_902_);
v___x_904_ = lean_unsigned_to_nat(4u);
v_tac_905_ = l_Lean_Syntax_getArg(v_x_889_, v___x_904_);
lean_dec(v_x_889_);
v___x_934_ = l_Lean_Syntax_getOptional_x3f(v___x_903_);
lean_dec(v___x_903_);
if (lean_obj_tag(v___x_934_) == 0)
{
lean_object* v___x_935_; 
v___x_935_ = lean_unsigned_to_nat(10u);
v___y_907_ = v___x_935_;
goto v___jp_906_;
}
else
{
lean_object* v_val_936_; lean_object* v___x_937_; 
v_val_936_ = lean_ctor_get(v___x_934_, 0);
lean_inc(v_val_936_);
lean_dec_ref_known(v___x_934_, 1);
v___x_937_ = l_Lean_TSyntax_getNat(v_val_936_);
lean_dec(v_val_936_);
v___y_907_ = v___x_937_;
goto v___jp_906_;
}
v___jp_906_:
{
lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; 
v___x_908_ = lean_nat_sub(v___y_907_, v___x_902_);
lean_dec(v___y_907_);
v___x_909_ = l_List_range(v___x_908_);
v___x_910_ = lean_box(0);
lean_inc(v_tac_905_);
v___x_911_ = lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__0(v_tac_905_, v___x_900_, v___x_909_, v___x_910_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, v_a_895_, v_a_896_, v_a_897_);
if (lean_obj_tag(v___x_911_) == 0)
{
lean_object* v_a_912_; uint8_t v___x_913_; lean_object* v___x_914_; 
v_a_912_ = lean_ctor_get(v___x_911_, 0);
lean_inc(v_a_912_);
lean_dec_ref_known(v___x_911_, 1);
v___x_913_ = 0;
v___x_914_ = lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats(v_tac_905_, v___x_913_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, v_a_895_, v_a_896_, v_a_897_);
if (lean_obj_tag(v___x_914_) == 0)
{
lean_object* v_a_915_; lean_object* v___x_916_; lean_object* v___x_917_; 
v_a_915_ = lean_ctor_get(v___x_914_, 0);
lean_inc(v_a_915_);
lean_dec_ref_known(v___x_914_, 1);
v___x_916_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_916_, 0, v_a_915_);
lean_ctor_set(v___x_916_, 1, v_a_912_);
v___x_917_ = lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1_spec__1(v___x_916_, v_a_890_, v_a_891_, v_a_892_, v_a_893_, v_a_894_, v_a_895_, v_a_896_, v_a_897_);
return v___x_917_;
}
else
{
lean_object* v_a_918_; lean_object* v___x_920_; uint8_t v_isShared_921_; uint8_t v_isSharedCheck_925_; 
lean_dec(v_a_912_);
v_a_918_ = lean_ctor_get(v___x_914_, 0);
v_isSharedCheck_925_ = !lean_is_exclusive(v___x_914_);
if (v_isSharedCheck_925_ == 0)
{
v___x_920_ = v___x_914_;
v_isShared_921_ = v_isSharedCheck_925_;
goto v_resetjp_919_;
}
else
{
lean_inc(v_a_918_);
lean_dec(v___x_914_);
v___x_920_ = lean_box(0);
v_isShared_921_ = v_isSharedCheck_925_;
goto v_resetjp_919_;
}
v_resetjp_919_:
{
lean_object* v___x_923_; 
if (v_isShared_921_ == 0)
{
v___x_923_ = v___x_920_;
goto v_reusejp_922_;
}
else
{
lean_object* v_reuseFailAlloc_924_; 
v_reuseFailAlloc_924_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_924_, 0, v_a_918_);
v___x_923_ = v_reuseFailAlloc_924_;
goto v_reusejp_922_;
}
v_reusejp_922_:
{
return v___x_923_;
}
}
}
}
else
{
lean_object* v_a_926_; lean_object* v___x_928_; uint8_t v_isShared_929_; uint8_t v_isSharedCheck_933_; 
lean_dec(v_tac_905_);
v_a_926_ = lean_ctor_get(v___x_911_, 0);
v_isSharedCheck_933_ = !lean_is_exclusive(v___x_911_);
if (v_isSharedCheck_933_ == 0)
{
v___x_928_ = v___x_911_;
v_isShared_929_ = v_isSharedCheck_933_;
goto v_resetjp_927_;
}
else
{
lean_inc(v_a_926_);
lean_dec(v___x_911_);
v___x_928_ = lean_box(0);
v_isShared_929_ = v_isSharedCheck_933_;
goto v_resetjp_927_;
}
v_resetjp_927_:
{
lean_object* v___x_931_; 
if (v_isShared_929_ == 0)
{
v___x_931_ = v___x_928_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_932_; 
v_reuseFailAlloc_932_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_932_, 0, v_a_926_);
v___x_931_ = v_reuseFailAlloc_932_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
return v___x_931_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1___boxed(lean_object* v_x_938_, lean_object* v_a_939_, lean_object* v_a_940_, lean_object* v_a_941_, lean_object* v_a_942_, lean_object* v_a_943_, lean_object* v_a_944_, lean_object* v_a_945_, lean_object* v_a_946_, lean_object* v_a_947_){
_start:
{
lean_object* v_res_948_; 
v_res_948_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats_x21__In______1(v_x_938_, v_a_939_, v_a_940_, v_a_941_, v_a_942_, v_a_943_, v_a_944_, v_a_945_, v_a_946_);
lean_dec(v_a_946_);
lean_dec_ref(v_a_945_);
lean_dec(v_a_944_);
lean_dec_ref(v_a_943_);
lean_dec(v_a_942_);
lean_dec_ref(v_a_941_);
lean_dec(v_a_940_);
lean_dec_ref(v_a_939_);
return v_res_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_roundDownIf(lean_object* v_n_950_, uint8_t v_approx_951_){
_start:
{
if (v_approx_951_ == 0)
{
lean_object* v___x_952_; 
v___x_952_ = l_Nat_reprFast(v_n_950_);
return v___x_952_;
}
else
{
lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; 
v___x_953_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_roundDownIf___closed__0));
v___x_954_ = lean_unsigned_to_nat(1000u);
v___x_955_ = lean_nat_div(v_n_950_, v___x_954_);
lean_dec(v_n_950_);
v___x_956_ = lean_nat_mul(v___x_955_, v___x_954_);
lean_dec(v___x_955_);
v___x_957_ = l_Nat_reprFast(v___x_956_);
v___x_958_ = lean_string_append(v___x_953_, v___x_957_);
lean_dec_ref(v___x_957_);
return v___x_958_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_roundDownIf___boxed(lean_object* v_n_959_, lean_object* v_approx_960_){
_start:
{
uint8_t v_approx_boxed_961_; lean_object* v_res_962_; 
v_approx_boxed_961_ = lean_unbox(v_approx_960_);
v_res_962_ = lp_mathlib_Mathlib_CountHeartbeats_roundDownIf(v_n_959_, v_approx_boxed_961_);
return v_res_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg(){
_start:
{
lean_object* v___x_1004_; lean_object* v___x_1005_; 
v___x_1004_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__0___redArg___closed__0);
v___x_1005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1005_, 0, v___x_1004_);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg___boxed(lean_object* v___y_1006_){
_start:
{
lean_object* v_res_1007_; 
v_res_1007_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg();
return v_res_1007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0(lean_object* v_00_u03b1_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
lean_object* v___x_1012_; 
v___x_1012_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg();
return v___x_1012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___boxed(lean_object* v_00_u03b1_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_){
_start:
{
lean_object* v_res_1017_; 
v_res_1017_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0(v_00_u03b1_1013_, v___y_1014_, v___y_1015_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
return v_res_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(lean_object* v___y_1018_){
_start:
{
lean_object* v___x_1020_; lean_object* v_env_1021_; lean_object* v___x_1022_; lean_object* v_mainModule_1023_; lean_object* v___x_1024_; 
v___x_1020_ = lean_st_ref_get(v___y_1018_);
v_env_1021_ = lean_ctor_get(v___x_1020_, 0);
lean_inc_ref(v_env_1021_);
lean_dec(v___x_1020_);
v___x_1022_ = l_Lean_Environment_header(v_env_1021_);
lean_dec_ref(v_env_1021_);
v_mainModule_1023_ = lean_ctor_get(v___x_1022_, 0);
lean_inc(v_mainModule_1023_);
lean_dec_ref(v___x_1022_);
v___x_1024_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1024_, 0, v_mainModule_1023_);
return v___x_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg___boxed(lean_object* v___y_1025_, lean_object* v___y_1026_){
_start:
{
lean_object* v_res_1027_; 
v_res_1027_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(v___y_1025_);
lean_dec(v___y_1025_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3(lean_object* v___y_1028_, lean_object* v___y_1029_){
_start:
{
lean_object* v___x_1031_; 
v___x_1031_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(v___y_1029_);
return v___x_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___boxed(lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_){
_start:
{
lean_object* v_res_1035_; 
v_res_1035_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3(v___y_1032_, v___y_1033_);
lean_dec(v___y_1033_);
lean_dec_ref(v___y_1032_);
return v_res_1035_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1039_; lean_object* v___x_1040_; 
v___x_1039_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__2));
v___x_1040_ = l_String_toRawSubstring_x27(v___x_1039_);
return v___x_1040_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7(void){
_start:
{
lean_object* v___x_1046_; 
v___x_1046_ = l_Array_mkArray0(lean_box(0));
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0(lean_object* v___x_1048_, uint8_t v___x_1049_, lean_object* v___x_1050_, lean_object* v___x_1051_, lean_object* v___x_1052_, lean_object* v___x_1053_, lean_object* v___x_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_){
_start:
{
lean_object* v___x_1058_; lean_object* v_ref_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; uint8_t v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; 
v___x_1058_ = lean_st_mk_ref(v___x_1048_);
v_ref_1059_ = lean_ctor_get(v___y_1055_, 5);
lean_inc(v_ref_1059_);
v___x_1060_ = l_Lean_SourceInfo_fromRef(v_ref_1059_, v___x_1049_);
v___x_1061_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__0));
v___x_1062_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11));
lean_inc_ref(v___x_1051_);
lean_inc_ref(v___x_1050_);
v___x_1063_ = l_Lean_Name_mkStr4(v___x_1050_, v___x_1051_, v___x_1061_, v___x_1062_);
v___x_1064_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1));
v___x_1065_ = l_Lean_Name_mkStr4(v___x_1050_, v___x_1051_, v___x_1061_, v___x_1064_);
lean_inc_n(v___x_1060_, 5);
v___x_1066_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1066_, 0, v___x_1060_);
lean_ctor_set(v___x_1066_, 1, v___x_1064_);
v___x_1067_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3);
v___x_1068_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__4));
v___x_1069_ = lean_box(0);
v___x_1070_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1070_, 0, v___x_1060_);
lean_ctor_set(v___x_1070_, 1, v___x_1067_);
lean_ctor_set(v___x_1070_, 2, v___x_1068_);
lean_ctor_set(v___x_1070_, 3, v___x_1069_);
v___x_1071_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_1072_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_1073_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1073_, 0, v___x_1060_);
lean_ctor_set(v___x_1073_, 1, v___x_1071_);
lean_ctor_set(v___x_1073_, 2, v___x_1072_);
v___x_1074_ = l_Lean_Syntax_node4(v___x_1060_, v___x_1065_, v___x_1066_, v___x_1070_, v___x_1073_, v___x_1052_);
v___x_1075_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1075_, 0, v___x_1060_);
lean_ctor_set(v___x_1075_, 1, v___x_1062_);
v___x_1076_ = l_Lean_Syntax_node3(v___x_1060_, v___x_1063_, v___x_1074_, v___x_1075_, v___x_1053_);
v___x_1077_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1077_, 0, v___x_1054_);
lean_ctor_set(v___x_1077_, 1, v___x_1076_);
v___x_1078_ = lean_box(0);
v___x_1079_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1079_, 0, v___x_1077_);
lean_ctor_set(v___x_1079_, 1, v___x_1078_);
lean_ctor_set(v___x_1079_, 2, v___x_1078_);
lean_ctor_set(v___x_1079_, 3, v___x_1078_);
lean_ctor_set(v___x_1079_, 4, v___x_1078_);
lean_ctor_set(v___x_1079_, 5, v___x_1078_);
v___x_1080_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__8));
v___x_1081_ = 4;
v___x_1082_ = l_Lean_MessageData_nil;
v___x_1083_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_ref_1059_, v___x_1079_, v___x_1078_, v___x_1080_, v___x_1078_, v___x_1081_, v___x_1082_, v___y_1055_, v___y_1056_);
lean_dec_ref(v___y_1055_);
if (lean_obj_tag(v___x_1083_) == 0)
{
lean_object* v_a_1084_; lean_object* v___x_1086_; uint8_t v_isShared_1087_; uint8_t v_isSharedCheck_1092_; 
v_a_1084_ = lean_ctor_get(v___x_1083_, 0);
v_isSharedCheck_1092_ = !lean_is_exclusive(v___x_1083_);
if (v_isSharedCheck_1092_ == 0)
{
v___x_1086_ = v___x_1083_;
v_isShared_1087_ = v_isSharedCheck_1092_;
goto v_resetjp_1085_;
}
else
{
lean_inc(v_a_1084_);
lean_dec(v___x_1083_);
v___x_1086_ = lean_box(0);
v_isShared_1087_ = v_isSharedCheck_1092_;
goto v_resetjp_1085_;
}
v_resetjp_1085_:
{
lean_object* v___x_1088_; lean_object* v___x_1090_; 
v___x_1088_ = lean_st_ref_get(v___x_1058_);
lean_dec(v___x_1058_);
lean_dec(v___x_1088_);
if (v_isShared_1087_ == 0)
{
v___x_1090_ = v___x_1086_;
goto v_reusejp_1089_;
}
else
{
lean_object* v_reuseFailAlloc_1091_; 
v_reuseFailAlloc_1091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1091_, 0, v_a_1084_);
v___x_1090_ = v_reuseFailAlloc_1091_;
goto v_reusejp_1089_;
}
v_reusejp_1089_:
{
return v___x_1090_;
}
}
}
else
{
lean_dec(v___x_1058_);
return v___x_1083_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___boxed(lean_object* v___x_1093_, lean_object* v___x_1094_, lean_object* v___x_1095_, lean_object* v___x_1096_, lean_object* v___x_1097_, lean_object* v___x_1098_, lean_object* v___x_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_){
_start:
{
uint8_t v___x_13685__boxed_1103_; lean_object* v_res_1104_; 
v___x_13685__boxed_1103_ = lean_unbox(v___x_1094_);
v_res_1104_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0(v___x_1093_, v___x_13685__boxed_1103_, v___x_1095_, v___x_1096_, v___x_1097_, v___x_1098_, v___x_1099_, v___y_1100_, v___y_1101_);
lean_dec(v___y_1101_);
return v_res_1104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___redArg(lean_object* v___x_1105_, lean_object* v_a_1106_){
_start:
{
uint8_t v___x_1108_; 
v___x_1108_ = lean_nat_dec_lt(v_a_1106_, v___x_1105_);
if (v___x_1108_ == 0)
{
lean_object* v___x_1109_; 
v___x_1109_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1109_, 0, v_a_1106_);
return v___x_1109_;
}
else
{
lean_object* v___x_1110_; lean_object* v___x_1111_; 
v___x_1110_ = lean_unsigned_to_nat(2u);
v___x_1111_ = lean_nat_mul(v___x_1110_, v_a_1106_);
lean_dec(v_a_1106_);
v_a_1106_ = v___x_1111_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___redArg___boxed(lean_object* v___x_1113_, lean_object* v_a_1114_, lean_object* v___y_1115_){
_start:
{
lean_object* v_res_1116_; 
v_res_1116_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___redArg(v___x_1113_, v_a_1114_);
lean_dec(v___x_1113_);
return v_res_1116_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___lam__0(uint8_t v___y_1117_, uint8_t v_suppressElabErrors_1118_, lean_object* v_x_1119_){
_start:
{
if (lean_obj_tag(v_x_1119_) == 1)
{
lean_object* v_pre_1120_; 
v_pre_1120_ = lean_ctor_get(v_x_1119_, 0);
if (lean_obj_tag(v_pre_1120_) == 0)
{
lean_object* v_str_1121_; lean_object* v___x_1122_; uint8_t v___x_1123_; 
v_str_1121_ = lean_ctor_get(v_x_1119_, 1);
v___x_1122_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__0___closed__0));
v___x_1123_ = lean_string_dec_eq(v_str_1121_, v___x_1122_);
if (v___x_1123_ == 0)
{
return v___y_1117_;
}
else
{
return v_suppressElabErrors_1118_;
}
}
else
{
return v___y_1117_;
}
}
else
{
return v___y_1117_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___lam__0___boxed(lean_object* v___y_1124_, lean_object* v_suppressElabErrors_1125_, lean_object* v_x_1126_){
_start:
{
uint8_t v___y_13812__boxed_1127_; uint8_t v_suppressElabErrors_boxed_1128_; uint8_t v_res_1129_; lean_object* v_r_1130_; 
v___y_13812__boxed_1127_ = lean_unbox(v___y_1124_);
v_suppressElabErrors_boxed_1128_ = lean_unbox(v_suppressElabErrors_1125_);
v_res_1129_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___lam__0(v___y_13812__boxed_1127_, v_suppressElabErrors_boxed_1128_, v_x_1126_);
lean_dec(v_x_1126_);
v_r_1130_ = lean_box(v_res_1129_);
return v_r_1130_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_1131_; 
v___x_1131_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1131_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_1132_; lean_object* v___x_1133_; 
v___x_1132_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__0);
v___x_1133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1132_);
return v___x_1133_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; 
v___x_1134_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1);
v___x_1135_ = lean_unsigned_to_nat(0u);
v___x_1136_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1136_, 0, v___x_1135_);
lean_ctor_set(v___x_1136_, 1, v___x_1135_);
lean_ctor_set(v___x_1136_, 2, v___x_1135_);
lean_ctor_set(v___x_1136_, 3, v___x_1135_);
lean_ctor_set(v___x_1136_, 4, v___x_1134_);
lean_ctor_set(v___x_1136_, 5, v___x_1134_);
lean_ctor_set(v___x_1136_, 6, v___x_1134_);
lean_ctor_set(v___x_1136_, 7, v___x_1134_);
lean_ctor_set(v___x_1136_, 8, v___x_1134_);
lean_ctor_set(v___x_1136_, 9, v___x_1134_);
return v___x_1136_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; 
v___x_1137_ = lean_unsigned_to_nat(32u);
v___x_1138_ = lean_mk_empty_array_with_capacity(v___x_1137_);
v___x_1139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1139_, 0, v___x_1138_);
return v___x_1139_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__4(void){
_start:
{
size_t v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; 
v___x_1140_ = ((size_t)5ULL);
v___x_1141_ = lean_unsigned_to_nat(0u);
v___x_1142_ = lean_unsigned_to_nat(32u);
v___x_1143_ = lean_mk_empty_array_with_capacity(v___x_1142_);
v___x_1144_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__3);
v___x_1145_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1145_, 0, v___x_1144_);
lean_ctor_set(v___x_1145_, 1, v___x_1143_);
lean_ctor_set(v___x_1145_, 2, v___x_1141_);
lean_ctor_set(v___x_1145_, 3, v___x_1141_);
lean_ctor_set_usize(v___x_1145_, 4, v___x_1140_);
return v___x_1145_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; 
v___x_1146_ = lean_box(1);
v___x_1147_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__4);
v___x_1148_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1);
v___x_1149_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1149_, 0, v___x_1148_);
lean_ctor_set(v___x_1149_, 1, v___x_1147_);
lean_ctor_set(v___x_1149_, 2, v___x_1146_);
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg(lean_object* v_msgData_1150_, lean_object* v___y_1151_){
_start:
{
lean_object* v___x_1153_; lean_object* v_env_1154_; lean_object* v___x_1155_; lean_object* v_scopes_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v_opts_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; 
v___x_1153_ = lean_st_ref_get(v___y_1151_);
v_env_1154_ = lean_ctor_get(v___x_1153_, 0);
lean_inc_ref(v_env_1154_);
lean_dec(v___x_1153_);
v___x_1155_ = lean_st_ref_get(v___y_1151_);
v_scopes_1156_ = lean_ctor_get(v___x_1155_, 2);
lean_inc(v_scopes_1156_);
lean_dec(v___x_1155_);
v___x_1157_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1158_ = l_List_head_x21___redArg(v___x_1157_, v_scopes_1156_);
lean_dec(v_scopes_1156_);
v_opts_1159_ = lean_ctor_get(v___x_1158_, 1);
lean_inc_ref(v_opts_1159_);
lean_dec(v___x_1158_);
v___x_1160_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__2);
v___x_1161_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__5);
v___x_1162_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1162_, 0, v_env_1154_);
lean_ctor_set(v___x_1162_, 1, v___x_1160_);
lean_ctor_set(v___x_1162_, 2, v___x_1161_);
lean_ctor_set(v___x_1162_, 3, v_opts_1159_);
v___x_1163_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1163_, 0, v___x_1162_);
lean_ctor_set(v___x_1163_, 1, v_msgData_1150_);
v___x_1164_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1164_, 0, v___x_1163_);
return v___x_1164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___boxed(lean_object* v_msgData_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_){
_start:
{
lean_object* v_res_1168_; 
v_res_1168_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg(v_msgData_1165_, v___y_1166_);
lean_dec(v___y_1166_);
return v_res_1168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4(lean_object* v_ref_1169_, lean_object* v_msgData_1170_, uint8_t v_severity_1171_, uint8_t v_isSilent_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_){
_start:
{
lean_object* v___y_1177_; uint8_t v___y_1178_; lean_object* v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; uint8_t v___y_1182_; lean_object* v___y_1183_; lean_object* v___y_1184_; uint8_t v___y_1241_; uint8_t v___y_1242_; lean_object* v___y_1243_; uint8_t v___y_1244_; lean_object* v___y_1245_; uint8_t v___y_1269_; uint8_t v___y_1270_; lean_object* v___y_1271_; uint8_t v___y_1272_; lean_object* v___y_1273_; uint8_t v___y_1277_; uint8_t v___y_1278_; uint8_t v___y_1279_; uint8_t v___x_1294_; uint8_t v___y_1296_; uint8_t v___y_1297_; uint8_t v___y_1298_; uint8_t v___y_1300_; uint8_t v___x_1312_; 
v___x_1294_ = 2;
v___x_1312_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1171_, v___x_1294_);
if (v___x_1312_ == 0)
{
v___y_1300_ = v___x_1312_;
goto v___jp_1299_;
}
else
{
uint8_t v___x_1313_; 
lean_inc_ref(v_msgData_1170_);
v___x_1313_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1170_);
v___y_1300_ = v___x_1313_;
goto v___jp_1299_;
}
v___jp_1176_:
{
lean_object* v___x_1185_; 
v___x_1185_ = l_Lean_Elab_Command_getScope___redArg(v___y_1184_);
if (lean_obj_tag(v___x_1185_) == 0)
{
lean_object* v_a_1186_; lean_object* v___x_1187_; 
v_a_1186_ = lean_ctor_get(v___x_1185_, 0);
lean_inc(v_a_1186_);
lean_dec_ref_known(v___x_1185_, 1);
v___x_1187_ = l_Lean_Elab_Command_getScope___redArg(v___y_1184_);
if (lean_obj_tag(v___x_1187_) == 0)
{
lean_object* v_a_1188_; lean_object* v___x_1190_; uint8_t v_isShared_1191_; uint8_t v_isSharedCheck_1223_; 
v_a_1188_ = lean_ctor_get(v___x_1187_, 0);
v_isSharedCheck_1223_ = !lean_is_exclusive(v___x_1187_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1190_ = v___x_1187_;
v_isShared_1191_ = v_isSharedCheck_1223_;
goto v_resetjp_1189_;
}
else
{
lean_inc(v_a_1188_);
lean_dec(v___x_1187_);
v___x_1190_ = lean_box(0);
v_isShared_1191_ = v_isSharedCheck_1223_;
goto v_resetjp_1189_;
}
v_resetjp_1189_:
{
lean_object* v___x_1192_; lean_object* v_currNamespace_1193_; lean_object* v_openDecls_1194_; lean_object* v_env_1195_; lean_object* v_messages_1196_; lean_object* v_scopes_1197_; lean_object* v_usedQuotCtxts_1198_; lean_object* v_nextMacroScope_1199_; lean_object* v_maxRecDepth_1200_; lean_object* v_ngen_1201_; lean_object* v_auxDeclNGen_1202_; lean_object* v_infoState_1203_; lean_object* v_traceState_1204_; lean_object* v_snapshotTasks_1205_; lean_object* v_prevLinterStates_1206_; lean_object* v___x_1208_; uint8_t v_isShared_1209_; uint8_t v_isSharedCheck_1222_; 
v___x_1192_ = lean_st_ref_take(v___y_1184_);
v_currNamespace_1193_ = lean_ctor_get(v_a_1186_, 2);
lean_inc(v_currNamespace_1193_);
lean_dec(v_a_1186_);
v_openDecls_1194_ = lean_ctor_get(v_a_1188_, 3);
lean_inc(v_openDecls_1194_);
lean_dec(v_a_1188_);
v_env_1195_ = lean_ctor_get(v___x_1192_, 0);
v_messages_1196_ = lean_ctor_get(v___x_1192_, 1);
v_scopes_1197_ = lean_ctor_get(v___x_1192_, 2);
v_usedQuotCtxts_1198_ = lean_ctor_get(v___x_1192_, 3);
v_nextMacroScope_1199_ = lean_ctor_get(v___x_1192_, 4);
v_maxRecDepth_1200_ = lean_ctor_get(v___x_1192_, 5);
v_ngen_1201_ = lean_ctor_get(v___x_1192_, 6);
v_auxDeclNGen_1202_ = lean_ctor_get(v___x_1192_, 7);
v_infoState_1203_ = lean_ctor_get(v___x_1192_, 8);
v_traceState_1204_ = lean_ctor_get(v___x_1192_, 9);
v_snapshotTasks_1205_ = lean_ctor_get(v___x_1192_, 10);
v_prevLinterStates_1206_ = lean_ctor_get(v___x_1192_, 11);
v_isSharedCheck_1222_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1222_ == 0)
{
v___x_1208_ = v___x_1192_;
v_isShared_1209_ = v_isSharedCheck_1222_;
goto v_resetjp_1207_;
}
else
{
lean_inc(v_prevLinterStates_1206_);
lean_inc(v_snapshotTasks_1205_);
lean_inc(v_traceState_1204_);
lean_inc(v_infoState_1203_);
lean_inc(v_auxDeclNGen_1202_);
lean_inc(v_ngen_1201_);
lean_inc(v_maxRecDepth_1200_);
lean_inc(v_nextMacroScope_1199_);
lean_inc(v_usedQuotCtxts_1198_);
lean_inc(v_scopes_1197_);
lean_inc(v_messages_1196_);
lean_inc(v_env_1195_);
lean_dec(v___x_1192_);
v___x_1208_ = lean_box(0);
v_isShared_1209_ = v_isSharedCheck_1222_;
goto v_resetjp_1207_;
}
v_resetjp_1207_:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1215_; 
v___x_1210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1210_, 0, v_currNamespace_1193_);
lean_ctor_set(v___x_1210_, 1, v_openDecls_1194_);
v___x_1211_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1211_, 0, v___x_1210_);
lean_ctor_set(v___x_1211_, 1, v___y_1179_);
lean_inc_ref(v___y_1183_);
lean_inc_ref(v___y_1181_);
v___x_1212_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1212_, 0, v___y_1181_);
lean_ctor_set(v___x_1212_, 1, v___y_1177_);
lean_ctor_set(v___x_1212_, 2, v___y_1180_);
lean_ctor_set(v___x_1212_, 3, v___y_1183_);
lean_ctor_set(v___x_1212_, 4, v___x_1211_);
lean_ctor_set_uint8(v___x_1212_, sizeof(void*)*5, v___y_1182_);
lean_ctor_set_uint8(v___x_1212_, sizeof(void*)*5 + 1, v___y_1178_);
lean_ctor_set_uint8(v___x_1212_, sizeof(void*)*5 + 2, v_isSilent_1172_);
v___x_1213_ = l_Lean_MessageLog_add(v___x_1212_, v_messages_1196_);
if (v_isShared_1209_ == 0)
{
lean_ctor_set(v___x_1208_, 1, v___x_1213_);
v___x_1215_ = v___x_1208_;
goto v_reusejp_1214_;
}
else
{
lean_object* v_reuseFailAlloc_1221_; 
v_reuseFailAlloc_1221_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1221_, 0, v_env_1195_);
lean_ctor_set(v_reuseFailAlloc_1221_, 1, v___x_1213_);
lean_ctor_set(v_reuseFailAlloc_1221_, 2, v_scopes_1197_);
lean_ctor_set(v_reuseFailAlloc_1221_, 3, v_usedQuotCtxts_1198_);
lean_ctor_set(v_reuseFailAlloc_1221_, 4, v_nextMacroScope_1199_);
lean_ctor_set(v_reuseFailAlloc_1221_, 5, v_maxRecDepth_1200_);
lean_ctor_set(v_reuseFailAlloc_1221_, 6, v_ngen_1201_);
lean_ctor_set(v_reuseFailAlloc_1221_, 7, v_auxDeclNGen_1202_);
lean_ctor_set(v_reuseFailAlloc_1221_, 8, v_infoState_1203_);
lean_ctor_set(v_reuseFailAlloc_1221_, 9, v_traceState_1204_);
lean_ctor_set(v_reuseFailAlloc_1221_, 10, v_snapshotTasks_1205_);
lean_ctor_set(v_reuseFailAlloc_1221_, 11, v_prevLinterStates_1206_);
v___x_1215_ = v_reuseFailAlloc_1221_;
goto v_reusejp_1214_;
}
v_reusejp_1214_:
{
lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1219_; 
v___x_1216_ = lean_st_ref_set(v___y_1184_, v___x_1215_);
v___x_1217_ = lean_box(0);
if (v_isShared_1191_ == 0)
{
lean_ctor_set(v___x_1190_, 0, v___x_1217_);
v___x_1219_ = v___x_1190_;
goto v_reusejp_1218_;
}
else
{
lean_object* v_reuseFailAlloc_1220_; 
v_reuseFailAlloc_1220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1220_, 0, v___x_1217_);
v___x_1219_ = v_reuseFailAlloc_1220_;
goto v_reusejp_1218_;
}
v_reusejp_1218_:
{
return v___x_1219_;
}
}
}
}
}
else
{
lean_object* v_a_1224_; lean_object* v___x_1226_; uint8_t v_isShared_1227_; uint8_t v_isSharedCheck_1231_; 
lean_dec(v_a_1186_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
lean_dec_ref(v___y_1177_);
v_a_1224_ = lean_ctor_get(v___x_1187_, 0);
v_isSharedCheck_1231_ = !lean_is_exclusive(v___x_1187_);
if (v_isSharedCheck_1231_ == 0)
{
v___x_1226_ = v___x_1187_;
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
else
{
lean_inc(v_a_1224_);
lean_dec(v___x_1187_);
v___x_1226_ = lean_box(0);
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
v_resetjp_1225_:
{
lean_object* v___x_1229_; 
if (v_isShared_1227_ == 0)
{
v___x_1229_ = v___x_1226_;
goto v_reusejp_1228_;
}
else
{
lean_object* v_reuseFailAlloc_1230_; 
v_reuseFailAlloc_1230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1230_, 0, v_a_1224_);
v___x_1229_ = v_reuseFailAlloc_1230_;
goto v_reusejp_1228_;
}
v_reusejp_1228_:
{
return v___x_1229_;
}
}
}
}
else
{
lean_object* v_a_1232_; lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1239_; 
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
lean_dec_ref(v___y_1177_);
v_a_1232_ = lean_ctor_get(v___x_1185_, 0);
v_isSharedCheck_1239_ = !lean_is_exclusive(v___x_1185_);
if (v_isSharedCheck_1239_ == 0)
{
v___x_1234_ = v___x_1185_;
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
else
{
lean_inc(v_a_1232_);
lean_dec(v___x_1185_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v___x_1237_; 
if (v_isShared_1235_ == 0)
{
v___x_1237_ = v___x_1234_;
goto v_reusejp_1236_;
}
else
{
lean_object* v_reuseFailAlloc_1238_; 
v_reuseFailAlloc_1238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1238_, 0, v_a_1232_);
v___x_1237_ = v_reuseFailAlloc_1238_;
goto v_reusejp_1236_;
}
v_reusejp_1236_:
{
return v___x_1237_;
}
}
}
}
v___jp_1240_:
{
lean_object* v_fileName_1246_; lean_object* v_fileMap_1247_; uint8_t v_suppressElabErrors_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v_a_1251_; lean_object* v___x_1253_; uint8_t v_isShared_1254_; uint8_t v_isSharedCheck_1267_; 
v_fileName_1246_ = lean_ctor_get(v___y_1173_, 0);
v_fileMap_1247_ = lean_ctor_get(v___y_1173_, 1);
v_suppressElabErrors_1248_ = lean_ctor_get_uint8(v___y_1173_, sizeof(void*)*10);
v___x_1249_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1170_);
v___x_1250_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg(v___x_1249_, v___y_1174_);
v_a_1251_ = lean_ctor_get(v___x_1250_, 0);
v_isSharedCheck_1267_ = !lean_is_exclusive(v___x_1250_);
if (v_isSharedCheck_1267_ == 0)
{
v___x_1253_ = v___x_1250_;
v_isShared_1254_ = v_isSharedCheck_1267_;
goto v_resetjp_1252_;
}
else
{
lean_inc(v_a_1251_);
lean_dec(v___x_1250_);
v___x_1253_ = lean_box(0);
v_isShared_1254_ = v_isSharedCheck_1267_;
goto v_resetjp_1252_;
}
v_resetjp_1252_:
{
lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; 
lean_inc_ref_n(v_fileMap_1247_, 2);
v___x_1255_ = l_Lean_FileMap_toPosition(v_fileMap_1247_, v___y_1243_);
lean_dec(v___y_1243_);
v___x_1256_ = l_Lean_FileMap_toPosition(v_fileMap_1247_, v___y_1245_);
lean_dec(v___y_1245_);
v___x_1257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1257_, 0, v___x_1256_);
v___x_1258_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__tactic_x23count__heartbeats____1_spec__1_spec__1_spec__2___redArg___closed__0));
if (v_suppressElabErrors_1248_ == 0)
{
lean_del_object(v___x_1253_);
v___y_1177_ = v___x_1255_;
v___y_1178_ = v___y_1242_;
v___y_1179_ = v_a_1251_;
v___y_1180_ = v___x_1257_;
v___y_1181_ = v_fileName_1246_;
v___y_1182_ = v___y_1244_;
v___y_1183_ = v___x_1258_;
v___y_1184_ = v___y_1174_;
goto v___jp_1176_;
}
else
{
lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___f_1261_; uint8_t v___x_1262_; 
v___x_1259_ = lean_box(v___y_1241_);
v___x_1260_ = lean_box(v_suppressElabErrors_1248_);
v___f_1261_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1261_, 0, v___x_1259_);
lean_closure_set(v___f_1261_, 1, v___x_1260_);
lean_inc(v_a_1251_);
v___x_1262_ = l_Lean_MessageData_hasTag(v___f_1261_, v_a_1251_);
if (v___x_1262_ == 0)
{
lean_object* v___x_1263_; lean_object* v___x_1265_; 
lean_dec_ref_known(v___x_1257_, 1);
lean_dec_ref(v___x_1255_);
lean_dec(v_a_1251_);
v___x_1263_ = lean_box(0);
if (v_isShared_1254_ == 0)
{
lean_ctor_set(v___x_1253_, 0, v___x_1263_);
v___x_1265_ = v___x_1253_;
goto v_reusejp_1264_;
}
else
{
lean_object* v_reuseFailAlloc_1266_; 
v_reuseFailAlloc_1266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1266_, 0, v___x_1263_);
v___x_1265_ = v_reuseFailAlloc_1266_;
goto v_reusejp_1264_;
}
v_reusejp_1264_:
{
return v___x_1265_;
}
}
else
{
lean_del_object(v___x_1253_);
v___y_1177_ = v___x_1255_;
v___y_1178_ = v___y_1242_;
v___y_1179_ = v_a_1251_;
v___y_1180_ = v___x_1257_;
v___y_1181_ = v_fileName_1246_;
v___y_1182_ = v___y_1244_;
v___y_1183_ = v___x_1258_;
v___y_1184_ = v___y_1174_;
goto v___jp_1176_;
}
}
}
}
v___jp_1268_:
{
lean_object* v___x_1274_; 
v___x_1274_ = l_Lean_Syntax_getTailPos_x3f(v___y_1271_, v___y_1272_);
lean_dec(v___y_1271_);
if (lean_obj_tag(v___x_1274_) == 0)
{
lean_inc(v___y_1273_);
v___y_1241_ = v___y_1269_;
v___y_1242_ = v___y_1270_;
v___y_1243_ = v___y_1273_;
v___y_1244_ = v___y_1272_;
v___y_1245_ = v___y_1273_;
goto v___jp_1240_;
}
else
{
lean_object* v_val_1275_; 
v_val_1275_ = lean_ctor_get(v___x_1274_, 0);
lean_inc(v_val_1275_);
lean_dec_ref_known(v___x_1274_, 1);
v___y_1241_ = v___y_1269_;
v___y_1242_ = v___y_1270_;
v___y_1243_ = v___y_1273_;
v___y_1244_ = v___y_1272_;
v___y_1245_ = v_val_1275_;
goto v___jp_1240_;
}
}
v___jp_1276_:
{
lean_object* v___x_1280_; 
v___x_1280_ = l_Lean_Elab_Command_getRef___redArg(v___y_1173_);
if (lean_obj_tag(v___x_1280_) == 0)
{
lean_object* v_a_1281_; lean_object* v_ref_1282_; lean_object* v___x_1283_; 
v_a_1281_ = lean_ctor_get(v___x_1280_, 0);
lean_inc(v_a_1281_);
lean_dec_ref_known(v___x_1280_, 1);
v_ref_1282_ = l_Lean_replaceRef(v_ref_1169_, v_a_1281_);
lean_dec(v_a_1281_);
v___x_1283_ = l_Lean_Syntax_getPos_x3f(v_ref_1282_, v___y_1278_);
if (lean_obj_tag(v___x_1283_) == 0)
{
lean_object* v___x_1284_; 
v___x_1284_ = lean_unsigned_to_nat(0u);
v___y_1269_ = v___y_1277_;
v___y_1270_ = v___y_1279_;
v___y_1271_ = v_ref_1282_;
v___y_1272_ = v___y_1278_;
v___y_1273_ = v___x_1284_;
goto v___jp_1268_;
}
else
{
lean_object* v_val_1285_; 
v_val_1285_ = lean_ctor_get(v___x_1283_, 0);
lean_inc(v_val_1285_);
lean_dec_ref_known(v___x_1283_, 1);
v___y_1269_ = v___y_1277_;
v___y_1270_ = v___y_1279_;
v___y_1271_ = v_ref_1282_;
v___y_1272_ = v___y_1278_;
v___y_1273_ = v_val_1285_;
goto v___jp_1268_;
}
}
else
{
lean_object* v_a_1286_; lean_object* v___x_1288_; uint8_t v_isShared_1289_; uint8_t v_isSharedCheck_1293_; 
lean_dec_ref(v_msgData_1170_);
v_a_1286_ = lean_ctor_get(v___x_1280_, 0);
v_isSharedCheck_1293_ = !lean_is_exclusive(v___x_1280_);
if (v_isSharedCheck_1293_ == 0)
{
v___x_1288_ = v___x_1280_;
v_isShared_1289_ = v_isSharedCheck_1293_;
goto v_resetjp_1287_;
}
else
{
lean_inc(v_a_1286_);
lean_dec(v___x_1280_);
v___x_1288_ = lean_box(0);
v_isShared_1289_ = v_isSharedCheck_1293_;
goto v_resetjp_1287_;
}
v_resetjp_1287_:
{
lean_object* v___x_1291_; 
if (v_isShared_1289_ == 0)
{
v___x_1291_ = v___x_1288_;
goto v_reusejp_1290_;
}
else
{
lean_object* v_reuseFailAlloc_1292_; 
v_reuseFailAlloc_1292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1292_, 0, v_a_1286_);
v___x_1291_ = v_reuseFailAlloc_1292_;
goto v_reusejp_1290_;
}
v_reusejp_1290_:
{
return v___x_1291_;
}
}
}
}
v___jp_1295_:
{
if (v___y_1298_ == 0)
{
v___y_1277_ = v___y_1296_;
v___y_1278_ = v___y_1297_;
v___y_1279_ = v_severity_1171_;
goto v___jp_1276_;
}
else
{
v___y_1277_ = v___y_1296_;
v___y_1278_ = v___y_1297_;
v___y_1279_ = v___x_1294_;
goto v___jp_1276_;
}
}
v___jp_1299_:
{
if (v___y_1300_ == 0)
{
lean_object* v___x_1301_; lean_object* v_scopes_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v_opts_1305_; uint8_t v___x_1306_; uint8_t v___x_1307_; 
v___x_1301_ = lean_st_ref_get(v___y_1174_);
v_scopes_1302_ = lean_ctor_get(v___x_1301_, 2);
lean_inc(v_scopes_1302_);
lean_dec(v___x_1301_);
v___x_1303_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1304_ = l_List_head_x21___redArg(v___x_1303_, v_scopes_1302_);
lean_dec(v_scopes_1302_);
v_opts_1305_ = lean_ctor_get(v___x_1304_, 1);
lean_inc_ref(v_opts_1305_);
lean_dec(v___x_1304_);
v___x_1306_ = 1;
v___x_1307_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1171_, v___x_1306_);
if (v___x_1307_ == 0)
{
lean_dec_ref(v_opts_1305_);
v___y_1296_ = v___y_1300_;
v___y_1297_ = v___y_1300_;
v___y_1298_ = v___x_1307_;
goto v___jp_1295_;
}
else
{
lean_object* v___x_1308_; uint8_t v___x_1309_; 
v___x_1308_ = l_Lean_warningAsError;
v___x_1309_ = lp_mathlib_Lean_Option_get___at___00Mathlib_CountHeartbeats_runTacForHeartbeats_spec__1(v_opts_1305_, v___x_1308_);
lean_dec_ref(v_opts_1305_);
v___y_1296_ = v___y_1300_;
v___y_1297_ = v___y_1300_;
v___y_1298_ = v___x_1309_;
goto v___jp_1295_;
}
}
else
{
lean_object* v___x_1310_; lean_object* v___x_1311_; 
lean_dec_ref(v_msgData_1170_);
v___x_1310_ = lean_box(0);
v___x_1311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1311_, 0, v___x_1310_);
return v___x_1311_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4___boxed(lean_object* v_ref_1314_, lean_object* v_msgData_1315_, lean_object* v_severity_1316_, lean_object* v_isSilent_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_){
_start:
{
uint8_t v_severity_boxed_1321_; uint8_t v_isSilent_boxed_1322_; lean_object* v_res_1323_; 
v_severity_boxed_1321_ = lean_unbox(v_severity_1316_);
v_isSilent_boxed_1322_ = lean_unbox(v_isSilent_1317_);
v_res_1323_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4(v_ref_1314_, v_msgData_1315_, v_severity_boxed_1321_, v_isSilent_boxed_1322_, v___y_1318_, v___y_1319_);
lean_dec(v___y_1319_);
lean_dec_ref(v___y_1318_);
lean_dec(v_ref_1314_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2(lean_object* v_msgData_1324_, uint8_t v_severity_1325_, uint8_t v_isSilent_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_){
_start:
{
lean_object* v___x_1330_; 
v___x_1330_ = l_Lean_Elab_Command_getRef___redArg(v___y_1327_);
if (lean_obj_tag(v___x_1330_) == 0)
{
lean_object* v_a_1331_; lean_object* v___x_1332_; 
v_a_1331_ = lean_ctor_get(v___x_1330_, 0);
lean_inc(v_a_1331_);
lean_dec_ref_known(v___x_1330_, 1);
v___x_1332_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4(v_a_1331_, v_msgData_1324_, v_severity_1325_, v_isSilent_1326_, v___y_1327_, v___y_1328_);
lean_dec(v_a_1331_);
return v___x_1332_;
}
else
{
lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1340_; 
lean_dec_ref(v_msgData_1324_);
v_a_1333_ = lean_ctor_get(v___x_1330_, 0);
v_isSharedCheck_1340_ = !lean_is_exclusive(v___x_1330_);
if (v_isSharedCheck_1340_ == 0)
{
v___x_1335_ = v___x_1330_;
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1330_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1338_; 
if (v_isShared_1336_ == 0)
{
v___x_1338_ = v___x_1335_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v_a_1333_);
v___x_1338_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
return v___x_1338_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2___boxed(lean_object* v_msgData_1341_, lean_object* v_severity_1342_, lean_object* v_isSilent_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_){
_start:
{
uint8_t v_severity_boxed_1347_; uint8_t v_isSilent_boxed_1348_; lean_object* v_res_1349_; 
v_severity_boxed_1347_ = lean_unbox(v_severity_1342_);
v_isSilent_boxed_1348_ = lean_unbox(v_isSilent_1343_);
v_res_1349_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2(v_msgData_1341_, v_severity_boxed_1347_, v_isSilent_boxed_1348_, v___y_1344_, v___y_1345_);
lean_dec(v___y_1345_);
lean_dec_ref(v___y_1344_);
return v_res_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2(lean_object* v_msgData_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_){
_start:
{
uint8_t v___x_1354_; uint8_t v___x_1355_; lean_object* v___x_1356_; 
v___x_1354_ = 0;
v___x_1355_ = 0;
v___x_1356_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2(v_msgData_1350_, v___x_1354_, v___x_1355_, v___y_1351_, v___y_1352_);
return v___x_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2___boxed(lean_object* v_msgData_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_){
_start:
{
lean_object* v_res_1361_; 
v_res_1361_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2(v_msgData_1357_, v___y_1358_, v___y_1359_);
lean_dec(v___y_1359_);
lean_dec_ref(v___y_1358_);
return v_res_1361_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2(void){
_start:
{
lean_object* v___x_1364_; lean_object* v___x_1365_; 
v___x_1364_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__1));
v___x_1365_ = l_Lean_stringToMessageData(v___x_1364_);
return v___x_1365_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__4(void){
_start:
{
lean_object* v___x_1367_; lean_object* v___x_1368_; 
v___x_1367_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__3));
v___x_1368_ = l_Lean_stringToMessageData(v___x_1367_);
return v___x_1368_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6(void){
_start:
{
lean_object* v___x_1370_; lean_object* v___x_1371_; 
v___x_1370_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__5));
v___x_1371_ = l_Lean_stringToMessageData(v___x_1370_);
return v___x_1371_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__7(void){
_start:
{
lean_object* v___x_1372_; lean_object* v___x_1373_; 
v___x_1372_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1);
v___x_1373_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1372_);
lean_ctor_set(v___x_1373_, 1, v___x_1372_);
lean_ctor_set(v___x_1373_, 2, v___x_1372_);
lean_ctor_set(v___x_1373_, 3, v___x_1372_);
lean_ctor_set(v___x_1373_, 4, v___x_1372_);
lean_ctor_set(v___x_1373_, 5, v___x_1372_);
return v___x_1373_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__8(void){
_start:
{
lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; 
v___x_1374_ = lean_unsigned_to_nat(32u);
v___x_1375_ = lean_mk_empty_array_with_capacity(v___x_1374_);
v___x_1376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1376_, 0, v___x_1375_);
return v___x_1376_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__9(void){
_start:
{
lean_object* v___x_1377_; lean_object* v___x_1378_; 
v___x_1377_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1);
v___x_1378_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1378_, 0, v___x_1377_);
lean_ctor_set(v___x_1378_, 1, v___x_1377_);
lean_ctor_set(v___x_1378_, 2, v___x_1377_);
lean_ctor_set(v___x_1378_, 3, v___x_1377_);
lean_ctor_set(v___x_1378_, 4, v___x_1377_);
return v___x_1378_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__11(void){
_start:
{
lean_object* v___x_1380_; lean_object* v___x_1381_; 
v___x_1380_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__10));
v___x_1381_ = l_Lean_stringToMessageData(v___x_1380_);
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1(lean_object* v_val_1382_, lean_object* v_a_1383_, lean_object* v_a_1384_, lean_object* v___x_1385_, lean_object* v___x_1386_, lean_object* v___x_1387_, lean_object* v___x_1388_, lean_object* v___x_1389_, lean_object* v___y_1390_, uint8_t v___x_1391_, lean_object* v_a_x3f_1392_){
_start:
{
lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; uint8_t v___y_1399_; 
v___x_1394_ = lean_io_get_num_heartbeats();
v___x_1395_ = lean_nat_sub(v___x_1394_, v_val_1382_);
lean_dec(v___x_1394_);
v___x_1396_ = lean_unsigned_to_nat(1000u);
v___x_1397_ = lean_nat_div(v___x_1395_, v___x_1396_);
lean_dec(v___x_1395_);
if (lean_obj_tag(v___y_1390_) == 0)
{
uint8_t v___x_1465_; 
v___x_1465_ = 0;
v___y_1399_ = v___x_1465_;
goto v___jp_1398_;
}
else
{
v___y_1399_ = v___x_1391_;
goto v___jp_1398_;
}
v___jp_1398_:
{
lean_object* v___x_1400_; lean_object* v___x_1401_; 
v___x_1400_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__0));
v___x_1401_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_1400_, v_a_1383_, v_a_1384_);
if (lean_obj_tag(v___x_1401_) == 0)
{
lean_object* v_a_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; uint8_t v___x_1405_; 
v_a_1402_ = lean_ctor_get(v___x_1401_, 0);
lean_inc(v_a_1402_);
lean_dec_ref_known(v___x_1401_, 1);
lean_inc(v___x_1397_);
v___x_1403_ = lp_mathlib_Mathlib_CountHeartbeats_roundDownIf(v___x_1397_, v___y_1399_);
v___x_1404_ = lean_nat_div(v_a_1402_, v___x_1396_);
lean_dec(v_a_1402_);
v___x_1405_ = lean_nat_dec_lt(v___x_1397_, v___x_1404_);
if (v___x_1405_ == 0)
{
lean_object* v___x_1406_; 
lean_inc(v___x_1404_);
v___x_1406_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___redArg(v___x_1397_, v___x_1404_);
lean_dec(v___x_1397_);
if (lean_obj_tag(v___x_1406_) == 0)
{
lean_object* v_a_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; 
v_a_1407_ = lean_ctor_get(v___x_1406_, 0);
lean_inc(v_a_1407_);
lean_dec_ref_known(v___x_1406_, 1);
v___x_1408_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2);
v___x_1409_ = l_Lean_stringToMessageData(v___x_1403_);
v___x_1410_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1410_, 0, v___x_1408_);
lean_ctor_set(v___x_1410_, 1, v___x_1409_);
v___x_1411_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__4, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__4_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__4);
v___x_1412_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1412_, 0, v___x_1410_);
lean_ctor_set(v___x_1412_, 1, v___x_1411_);
v___x_1413_ = l_Nat_reprFast(v___x_1404_);
v___x_1414_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1414_, 0, v___x_1413_);
v___x_1415_ = l_Lean_MessageData_ofFormat(v___x_1414_);
v___x_1416_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1416_, 0, v___x_1412_);
lean_ctor_set(v___x_1416_, 1, v___x_1415_);
v___x_1417_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6);
v___x_1418_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1418_, 0, v___x_1416_);
lean_ctor_set(v___x_1418_, 1, v___x_1417_);
v___x_1419_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2(v___x_1418_, v_a_1383_, v_a_1384_);
if (lean_obj_tag(v___x_1419_) == 0)
{
lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; size_t v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___f_1435_; lean_object* v___x_1436_; 
lean_dec_ref_known(v___x_1419_, 1);
v___x_1420_ = l_Nat_reprFast(v_a_1407_);
v___x_1421_ = lean_box(2);
v___x_1422_ = l_Lean_Syntax_mkNumLit(v___x_1420_, v___x_1421_);
v___x_1423_ = lean_box(1);
v___x_1424_ = lean_unsigned_to_nat(32u);
v___x_1425_ = ((size_t)5ULL);
v___x_1426_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg___closed__1);
lean_inc_n(v___x_1385_, 5);
v___x_1427_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1427_, 0, v___x_1385_);
lean_ctor_set(v___x_1427_, 1, v___x_1385_);
lean_ctor_set(v___x_1427_, 2, v___x_1385_);
lean_ctor_set(v___x_1427_, 3, v___x_1385_);
lean_ctor_set(v___x_1427_, 4, v___x_1426_);
lean_ctor_set(v___x_1427_, 5, v___x_1426_);
lean_ctor_set(v___x_1427_, 6, v___x_1426_);
lean_ctor_set(v___x_1427_, 7, v___x_1426_);
lean_ctor_set(v___x_1427_, 8, v___x_1426_);
lean_ctor_set(v___x_1427_, 9, v___x_1426_);
v___x_1428_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__7);
v___x_1429_ = lean_mk_empty_array_with_capacity(v___x_1424_);
v___x_1430_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__8, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__8);
v___x_1431_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1431_, 0, v___x_1430_);
lean_ctor_set(v___x_1431_, 1, v___x_1429_);
lean_ctor_set(v___x_1431_, 2, v___x_1385_);
lean_ctor_set(v___x_1431_, 3, v___x_1385_);
lean_ctor_set_usize(v___x_1431_, 4, v___x_1425_);
v___x_1432_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__9, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__9);
v___x_1433_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1433_, 0, v___x_1427_);
lean_ctor_set(v___x_1433_, 1, v___x_1428_);
lean_ctor_set(v___x_1433_, 2, v___x_1423_);
lean_ctor_set(v___x_1433_, 3, v___x_1431_);
lean_ctor_set(v___x_1433_, 4, v___x_1432_);
v___x_1434_ = lean_box(v___x_1405_);
v___f_1435_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___boxed), 10, 7);
lean_closure_set(v___f_1435_, 0, v___x_1433_);
lean_closure_set(v___f_1435_, 1, v___x_1434_);
lean_closure_set(v___f_1435_, 2, v___x_1386_);
lean_closure_set(v___f_1435_, 3, v___x_1387_);
lean_closure_set(v___f_1435_, 4, v___x_1422_);
lean_closure_set(v___f_1435_, 5, v___x_1388_);
lean_closure_set(v___f_1435_, 6, v___x_1389_);
v___x_1436_ = l_Lean_Elab_Command_liftCoreM___redArg(v___f_1435_, v_a_1383_, v_a_1384_);
return v___x_1436_;
}
else
{
lean_dec(v_a_1407_);
lean_dec(v___x_1389_);
lean_dec(v___x_1388_);
lean_dec_ref(v___x_1387_);
lean_dec_ref(v___x_1386_);
lean_dec(v___x_1385_);
return v___x_1419_;
}
}
else
{
lean_object* v_a_1437_; lean_object* v___x_1439_; uint8_t v_isShared_1440_; uint8_t v_isSharedCheck_1444_; 
lean_dec(v___x_1404_);
lean_dec_ref(v___x_1403_);
lean_dec(v___x_1389_);
lean_dec(v___x_1388_);
lean_dec_ref(v___x_1387_);
lean_dec_ref(v___x_1386_);
lean_dec(v___x_1385_);
v_a_1437_ = lean_ctor_get(v___x_1406_, 0);
v_isSharedCheck_1444_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1444_ == 0)
{
v___x_1439_ = v___x_1406_;
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
else
{
lean_inc(v_a_1437_);
lean_dec(v___x_1406_);
v___x_1439_ = lean_box(0);
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
v_resetjp_1438_:
{
lean_object* v___x_1442_; 
if (v_isShared_1440_ == 0)
{
v___x_1442_ = v___x_1439_;
goto v_reusejp_1441_;
}
else
{
lean_object* v_reuseFailAlloc_1443_; 
v_reuseFailAlloc_1443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1443_, 0, v_a_1437_);
v___x_1442_ = v_reuseFailAlloc_1443_;
goto v_reusejp_1441_;
}
v_reusejp_1441_:
{
return v___x_1442_;
}
}
}
}
else
{
lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; 
lean_dec(v___x_1397_);
lean_dec(v___x_1389_);
lean_dec(v___x_1388_);
lean_dec_ref(v___x_1387_);
lean_dec_ref(v___x_1386_);
lean_dec(v___x_1385_);
v___x_1445_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2);
v___x_1446_ = l_Lean_stringToMessageData(v___x_1403_);
v___x_1447_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1447_, 0, v___x_1445_);
lean_ctor_set(v___x_1447_, 1, v___x_1446_);
v___x_1448_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__11, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__11_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__11);
v___x_1449_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1449_, 0, v___x_1447_);
lean_ctor_set(v___x_1449_, 1, v___x_1448_);
v___x_1450_ = l_Nat_reprFast(v___x_1404_);
v___x_1451_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1451_, 0, v___x_1450_);
v___x_1452_ = l_Lean_MessageData_ofFormat(v___x_1451_);
v___x_1453_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1453_, 0, v___x_1449_);
lean_ctor_set(v___x_1453_, 1, v___x_1452_);
v___x_1454_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6);
v___x_1455_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1455_, 0, v___x_1453_);
lean_ctor_set(v___x_1455_, 1, v___x_1454_);
v___x_1456_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2(v___x_1455_, v_a_1383_, v_a_1384_);
return v___x_1456_;
}
}
else
{
lean_object* v_a_1457_; lean_object* v___x_1459_; uint8_t v_isShared_1460_; uint8_t v_isSharedCheck_1464_; 
lean_dec(v___x_1397_);
lean_dec(v___x_1389_);
lean_dec(v___x_1388_);
lean_dec_ref(v___x_1387_);
lean_dec_ref(v___x_1386_);
lean_dec(v___x_1385_);
v_a_1457_ = lean_ctor_get(v___x_1401_, 0);
v_isSharedCheck_1464_ = !lean_is_exclusive(v___x_1401_);
if (v_isSharedCheck_1464_ == 0)
{
v___x_1459_ = v___x_1401_;
v_isShared_1460_ = v_isSharedCheck_1464_;
goto v_resetjp_1458_;
}
else
{
lean_inc(v_a_1457_);
lean_dec(v___x_1401_);
v___x_1459_ = lean_box(0);
v_isShared_1460_ = v_isSharedCheck_1464_;
goto v_resetjp_1458_;
}
v_resetjp_1458_:
{
lean_object* v___x_1462_; 
if (v_isShared_1460_ == 0)
{
v___x_1462_ = v___x_1459_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1463_; 
v_reuseFailAlloc_1463_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1463_, 0, v_a_1457_);
v___x_1462_ = v_reuseFailAlloc_1463_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
return v___x_1462_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___boxed(lean_object* v_val_1466_, lean_object* v_a_1467_, lean_object* v_a_1468_, lean_object* v___x_1469_, lean_object* v___x_1470_, lean_object* v___x_1471_, lean_object* v___x_1472_, lean_object* v___x_1473_, lean_object* v___y_1474_, lean_object* v___x_1475_, lean_object* v_a_x3f_1476_, lean_object* v___y_1477_){
_start:
{
uint8_t v___x_14296__boxed_1478_; lean_object* v_res_1479_; 
v___x_14296__boxed_1478_ = lean_unbox(v___x_1475_);
v_res_1479_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1(v_val_1466_, v_a_1467_, v_a_1468_, v___x_1469_, v___x_1470_, v___x_1471_, v___x_1472_, v___x_1473_, v___y_1474_, v___x_14296__boxed_1478_, v_a_x3f_1476_);
lean_dec(v_a_x3f_1476_);
lean_dec(v___y_1474_);
lean_dec(v_a_1468_);
lean_dec_ref(v_a_1467_);
lean_dec(v_val_1466_);
return v_res_1479_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4(void){
_start:
{
lean_object* v___x_1492_; lean_object* v___x_1493_; 
v___x_1492_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__3));
v___x_1493_ = l_String_toRawSubstring_x27(v___x_1492_);
return v___x_1493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1(lean_object* v_x_1514_, lean_object* v_a_1515_, lean_object* v_a_1516_){
_start:
{
lean_object* v___y_1519_; lean_object* v_a_1520_; lean_object* v___x_1531_; uint8_t v___x_1532_; 
v___x_1531_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1));
lean_inc(v_x_1514_);
v___x_1532_ = l_Lean_Syntax_isOfKind(v_x_1514_, v___x_1531_);
if (v___x_1532_ == 0)
{
lean_object* v___x_1533_; 
lean_dec(v_x_1514_);
v___x_1533_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg();
return v___x_1533_;
}
else
{
lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___y_1543_; lean_object* v___y_1544_; lean_object* v___y_1545_; lean_object* v_a_1546_; lean_object* v___y_1596_; lean_object* v___x_1612_; 
v___x_1534_ = lean_unsigned_to_nat(0u);
v___x_1535_ = lean_unsigned_to_nat(1u);
v___x_1536_ = l_Lean_Syntax_getArg(v_x_1514_, v___x_1535_);
v___x_1537_ = lean_unsigned_to_nat(4u);
v___x_1538_ = l_Lean_Syntax_getArg(v_x_1514_, v___x_1537_);
lean_dec(v_x_1514_);
v___x_1539_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__9));
v___x_1540_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_runTacForHeartbeats___closed__0));
v___x_1541_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__0));
v___x_1612_ = l_Lean_Syntax_getOptional_x3f(v___x_1536_);
lean_dec(v___x_1536_);
if (lean_obj_tag(v___x_1612_) == 0)
{
lean_object* v___x_1613_; 
v___x_1613_ = lean_box(0);
v___y_1596_ = v___x_1613_;
goto v___jp_1595_;
}
else
{
lean_object* v_val_1614_; lean_object* v___x_1616_; uint8_t v_isShared_1617_; uint8_t v_isSharedCheck_1621_; 
v_val_1614_ = lean_ctor_get(v___x_1612_, 0);
v_isSharedCheck_1621_ = !lean_is_exclusive(v___x_1612_);
if (v_isSharedCheck_1621_ == 0)
{
v___x_1616_ = v___x_1612_;
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
else
{
lean_inc(v_val_1614_);
lean_dec(v___x_1612_);
v___x_1616_ = lean_box(0);
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
v_resetjp_1615_:
{
lean_object* v___x_1619_; 
if (v_isShared_1617_ == 0)
{
v___x_1619_ = v___x_1616_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v_val_1614_);
v___x_1619_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
v___y_1596_ = v___x_1619_;
goto v___jp_1595_;
}
}
}
v___jp_1542_:
{
lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; 
v___x_1547_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11));
v___x_1548_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1));
v___x_1549_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1));
v___x_1550_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2));
lean_inc_n(v___y_1543_, 11);
v___x_1551_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1551_, 0, v___y_1543_);
lean_ctor_set(v___x_1551_, 1, v___x_1549_);
v___x_1552_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4);
v___x_1553_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5));
lean_inc(v___y_1544_);
lean_inc(v_a_1546_);
v___x_1554_ = l_Lean_addMacroScope(v_a_1546_, v___x_1553_, v___y_1544_);
v___x_1555_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__7));
v___x_1556_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1556_, 0, v___y_1543_);
lean_ctor_set(v___x_1556_, 1, v___x_1552_);
lean_ctor_set(v___x_1556_, 2, v___x_1554_);
lean_ctor_set(v___x_1556_, 3, v___x_1555_);
v___x_1557_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_1558_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_1559_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1559_, 0, v___y_1543_);
lean_ctor_set(v___x_1559_, 1, v___x_1557_);
lean_ctor_set(v___x_1559_, 2, v___x_1558_);
v___x_1560_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__8));
v___x_1561_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1561_, 0, v___y_1543_);
lean_ctor_set(v___x_1561_, 1, v___x_1560_);
lean_inc_ref(v___x_1559_);
lean_inc_ref(v___x_1551_);
v___x_1562_ = l_Lean_Syntax_node4(v___y_1543_, v___x_1550_, v___x_1551_, v___x_1556_, v___x_1559_, v___x_1561_);
v___x_1563_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1563_, 0, v___y_1543_);
lean_ctor_set(v___x_1563_, 1, v___x_1547_);
v___x_1564_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3);
v___x_1565_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__4));
v___x_1566_ = l_Lean_addMacroScope(v_a_1546_, v___x_1565_, v___y_1544_);
v___x_1567_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__11));
v___x_1568_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1568_, 0, v___y_1543_);
lean_ctor_set(v___x_1568_, 1, v___x_1564_);
lean_ctor_set(v___x_1568_, 2, v___x_1566_);
lean_ctor_set(v___x_1568_, 3, v___x_1567_);
v___x_1569_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__7));
v___x_1570_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__12));
v___x_1571_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1571_, 0, v___y_1543_);
lean_ctor_set(v___x_1571_, 1, v___x_1570_);
v___x_1572_ = l_Lean_Syntax_node1(v___y_1543_, v___x_1569_, v___x_1571_);
v___x_1573_ = l_Lean_Syntax_node4(v___y_1543_, v___x_1550_, v___x_1551_, v___x_1568_, v___x_1559_, v___x_1572_);
lean_inc_ref(v___x_1563_);
v___x_1574_ = l_Lean_Syntax_node3(v___y_1543_, v___x_1548_, v___x_1573_, v___x_1563_, v___x_1538_);
v___x_1575_ = l_Lean_Syntax_node3(v___y_1543_, v___x_1548_, v___x_1562_, v___x_1563_, v___x_1574_);
v___x_1576_ = l_Lean_Elab_Command_elabCommand(v___x_1575_, v_a_1515_, v_a_1516_);
if (lean_obj_tag(v___x_1576_) == 0)
{
lean_object* v_a_1577_; lean_object* v___x_1579_; uint8_t v_isShared_1580_; uint8_t v_isSharedCheck_1593_; 
v_a_1577_ = lean_ctor_get(v___x_1576_, 0);
v_isSharedCheck_1593_ = !lean_is_exclusive(v___x_1576_);
if (v_isSharedCheck_1593_ == 0)
{
v___x_1579_ = v___x_1576_;
v_isShared_1580_ = v_isSharedCheck_1593_;
goto v_resetjp_1578_;
}
else
{
lean_inc(v_a_1577_);
lean_dec(v___x_1576_);
v___x_1579_ = lean_box(0);
v_isShared_1580_ = v_isSharedCheck_1593_;
goto v_resetjp_1578_;
}
v_resetjp_1578_:
{
lean_object* v___x_1582_; 
lean_inc(v_a_1577_);
if (v_isShared_1580_ == 0)
{
lean_ctor_set_tag(v___x_1579_, 1);
v___x_1582_ = v___x_1579_;
goto v_reusejp_1581_;
}
else
{
lean_object* v_reuseFailAlloc_1592_; 
v_reuseFailAlloc_1592_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1592_, 0, v_a_1577_);
v___x_1582_ = v_reuseFailAlloc_1592_;
goto v_reusejp_1581_;
}
v_reusejp_1581_:
{
lean_object* v___x_1583_; 
v___x_1583_ = lean_apply_2(v___y_1545_, v___x_1582_, lean_box(0));
if (lean_obj_tag(v___x_1583_) == 0)
{
lean_object* v___x_1585_; uint8_t v_isShared_1586_; uint8_t v_isSharedCheck_1590_; 
v_isSharedCheck_1590_ = !lean_is_exclusive(v___x_1583_);
if (v_isSharedCheck_1590_ == 0)
{
lean_object* v_unused_1591_; 
v_unused_1591_ = lean_ctor_get(v___x_1583_, 0);
lean_dec(v_unused_1591_);
v___x_1585_ = v___x_1583_;
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
else
{
lean_dec(v___x_1583_);
v___x_1585_ = lean_box(0);
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
v_resetjp_1584_:
{
lean_object* v___x_1588_; 
if (v_isShared_1586_ == 0)
{
lean_ctor_set(v___x_1585_, 0, v_a_1577_);
v___x_1588_ = v___x_1585_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v_a_1577_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
}
else
{
lean_dec(v_a_1577_);
return v___x_1583_;
}
}
}
}
else
{
lean_object* v_a_1594_; 
v_a_1594_ = lean_ctor_get(v___x_1576_, 0);
lean_inc(v_a_1594_);
lean_dec_ref_known(v___x_1576_, 1);
v___y_1519_ = v___y_1545_;
v_a_1520_ = v_a_1594_;
goto v___jp_1518_;
}
}
v___jp_1595_:
{
lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___f_1599_; lean_object* v___x_1600_; 
v___x_1597_ = lean_io_get_num_heartbeats();
v___x_1598_ = lean_box(v___x_1532_);
lean_inc(v___x_1538_);
lean_inc(v_a_1516_);
lean_inc_ref(v_a_1515_);
v___f_1599_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___boxed), 12, 10);
lean_closure_set(v___f_1599_, 0, v___x_1597_);
lean_closure_set(v___f_1599_, 1, v_a_1515_);
lean_closure_set(v___f_1599_, 2, v_a_1516_);
lean_closure_set(v___f_1599_, 3, v___x_1534_);
lean_closure_set(v___f_1599_, 4, v___x_1540_);
lean_closure_set(v___f_1599_, 5, v___x_1541_);
lean_closure_set(v___f_1599_, 6, v___x_1538_);
lean_closure_set(v___f_1599_, 7, v___x_1539_);
lean_closure_set(v___f_1599_, 8, v___y_1596_);
lean_closure_set(v___f_1599_, 9, v___x_1598_);
v___x_1600_ = l_Lean_Elab_Command_getRef___redArg(v_a_1515_);
if (lean_obj_tag(v___x_1600_) == 0)
{
lean_object* v_a_1601_; lean_object* v___x_1602_; 
v_a_1601_ = lean_ctor_get(v___x_1600_, 0);
lean_inc(v_a_1601_);
lean_dec_ref_known(v___x_1600_, 1);
v___x_1602_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_1515_);
if (lean_obj_tag(v___x_1602_) == 0)
{
lean_object* v_a_1603_; lean_object* v_quotContext_x3f_1604_; uint8_t v___x_1605_; lean_object* v___x_1606_; 
v_a_1603_ = lean_ctor_get(v___x_1602_, 0);
lean_inc(v_a_1603_);
lean_dec_ref_known(v___x_1602_, 1);
v_quotContext_x3f_1604_ = lean_ctor_get(v_a_1515_, 5);
v___x_1605_ = 0;
v___x_1606_ = l_Lean_SourceInfo_fromRef(v_a_1601_, v___x_1605_);
lean_dec(v_a_1601_);
if (lean_obj_tag(v_quotContext_x3f_1604_) == 0)
{
lean_object* v___x_1607_; lean_object* v_a_1608_; 
v___x_1607_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(v_a_1516_);
v_a_1608_ = lean_ctor_get(v___x_1607_, 0);
lean_inc(v_a_1608_);
lean_dec_ref(v___x_1607_);
v___y_1543_ = v___x_1606_;
v___y_1544_ = v_a_1603_;
v___y_1545_ = v___f_1599_;
v_a_1546_ = v_a_1608_;
goto v___jp_1542_;
}
else
{
lean_object* v_val_1609_; 
v_val_1609_ = lean_ctor_get(v_quotContext_x3f_1604_, 0);
lean_inc(v_val_1609_);
v___y_1543_ = v___x_1606_;
v___y_1544_ = v_a_1603_;
v___y_1545_ = v___f_1599_;
v_a_1546_ = v_val_1609_;
goto v___jp_1542_;
}
}
else
{
lean_object* v_a_1610_; 
lean_dec(v_a_1601_);
lean_dec(v___x_1538_);
v_a_1610_ = lean_ctor_get(v___x_1602_, 0);
lean_inc(v_a_1610_);
lean_dec_ref_known(v___x_1602_, 1);
v___y_1519_ = v___f_1599_;
v_a_1520_ = v_a_1610_;
goto v___jp_1518_;
}
}
else
{
lean_object* v_a_1611_; 
lean_dec(v___x_1538_);
v_a_1611_ = lean_ctor_get(v___x_1600_, 0);
lean_inc(v_a_1611_);
lean_dec_ref_known(v___x_1600_, 1);
v___y_1519_ = v___f_1599_;
v_a_1520_ = v_a_1611_;
goto v___jp_1518_;
}
}
}
v___jp_1518_:
{
lean_object* v___x_1521_; lean_object* v___x_1522_; 
v___x_1521_ = lean_box(0);
v___x_1522_ = lean_apply_2(v___y_1519_, v___x_1521_, lean_box(0));
if (lean_obj_tag(v___x_1522_) == 0)
{
lean_object* v___x_1524_; uint8_t v_isShared_1525_; uint8_t v_isSharedCheck_1529_; 
v_isSharedCheck_1529_ = !lean_is_exclusive(v___x_1522_);
if (v_isSharedCheck_1529_ == 0)
{
lean_object* v_unused_1530_; 
v_unused_1530_ = lean_ctor_get(v___x_1522_, 0);
lean_dec(v_unused_1530_);
v___x_1524_ = v___x_1522_;
v_isShared_1525_ = v_isSharedCheck_1529_;
goto v_resetjp_1523_;
}
else
{
lean_dec(v___x_1522_);
v___x_1524_ = lean_box(0);
v_isShared_1525_ = v_isSharedCheck_1529_;
goto v_resetjp_1523_;
}
v_resetjp_1523_:
{
lean_object* v___x_1527_; 
if (v_isShared_1525_ == 0)
{
lean_ctor_set_tag(v___x_1524_, 1);
lean_ctor_set(v___x_1524_, 0, v_a_1520_);
v___x_1527_ = v___x_1524_;
goto v_reusejp_1526_;
}
else
{
lean_object* v_reuseFailAlloc_1528_; 
v_reuseFailAlloc_1528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1528_, 0, v_a_1520_);
v___x_1527_ = v_reuseFailAlloc_1528_;
goto v_reusejp_1526_;
}
v_reusejp_1526_:
{
return v___x_1527_;
}
}
}
else
{
lean_dec_ref(v_a_1520_);
return v___x_1522_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___boxed(lean_object* v_x_1622_, lean_object* v_a_1623_, lean_object* v_a_1624_, lean_object* v_a_1625_){
_start:
{
lean_object* v_res_1626_; 
v_res_1626_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1(v_x_1622_, v_a_1623_, v_a_1624_);
lean_dec(v_a_1624_);
lean_dec_ref(v_a_1623_);
return v_res_1626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1(lean_object* v___x_1627_, lean_object* v_inst_1628_, lean_object* v_a_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_){
_start:
{
lean_object* v___x_1633_; 
v___x_1633_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___redArg(v___x_1627_, v_a_1629_);
return v___x_1633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1___boxed(lean_object* v___x_1634_, lean_object* v_inst_1635_, lean_object* v_a_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_){
_start:
{
lean_object* v_res_1640_; 
v_res_1640_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__1(v___x_1634_, v_inst_1635_, v_a_1636_, v___y_1637_, v___y_1638_);
lean_dec(v___y_1638_);
lean_dec_ref(v___y_1637_);
lean_dec(v___x_1634_);
return v_res_1640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5(lean_object* v_msgData_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_){
_start:
{
lean_object* v___x_1645_; 
v___x_1645_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___redArg(v_msgData_1641_, v___y_1643_);
return v___x_1645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5___boxed(lean_object* v_msgData_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_){
_start:
{
lean_object* v_res_1650_; 
v_res_1650_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4_spec__5(v_msgData_1646_, v___y_1647_, v___y_1648_);
lean_dec(v___y_1648_);
lean_dec_ref(v___y_1647_);
return v_res_1650_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1685_; lean_object* v___x_1686_; 
v___x_1685_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__0));
v___x_1686_ = l_Lean_stringToMessageData(v___x_1685_);
return v___x_1686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0(lean_object* v_val_1687_, lean_object* v___x_1688_, lean_object* v___y_1689_, lean_object* v_a_1690_, lean_object* v_a_1691_, lean_object* v___y_1692_, uint8_t v___x_1693_, lean_object* v_a_x3f_1694_){
_start:
{
lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; uint8_t v___x_1699_; 
v___x_1696_ = lean_io_get_num_heartbeats();
v___x_1697_ = lean_nat_sub(v___x_1696_, v_val_1687_);
lean_dec(v___x_1696_);
v___x_1698_ = lean_nat_div(v___x_1697_, v___x_1688_);
lean_dec(v___x_1697_);
v___x_1699_ = lean_nat_dec_lt(v___x_1698_, v___y_1689_);
if (v___x_1699_ == 0)
{
lean_object* v___x_1700_; lean_object* v___x_1701_; 
lean_dec(v___x_1698_);
lean_dec(v___y_1689_);
v___x_1700_ = lean_box(0);
v___x_1701_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1701_, 0, v___x_1700_);
return v___x_1701_;
}
else
{
lean_object* v___x_1702_; uint8_t v___y_1704_; 
v___x_1702_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__2);
if (lean_obj_tag(v___y_1692_) == 0)
{
uint8_t v___x_1717_; 
v___x_1717_ = 0;
v___y_1704_ = v___x_1717_;
goto v___jp_1703_;
}
else
{
v___y_1704_ = v___x_1693_;
goto v___jp_1703_;
}
v___jp_1703_:
{
lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; 
v___x_1705_ = lp_mathlib_Mathlib_CountHeartbeats_roundDownIf(v___x_1698_, v___y_1704_);
v___x_1706_ = l_Lean_stringToMessageData(v___x_1705_);
v___x_1707_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1707_, 0, v___x_1702_);
lean_ctor_set(v___x_1707_, 1, v___x_1706_);
v___x_1708_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__1, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___closed__1);
v___x_1709_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1709_, 0, v___x_1707_);
lean_ctor_set(v___x_1709_, 1, v___x_1708_);
v___x_1710_ = l_Nat_reprFast(v___y_1689_);
v___x_1711_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1711_, 0, v___x_1710_);
v___x_1712_ = l_Lean_MessageData_ofFormat(v___x_1711_);
v___x_1713_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1713_, 0, v___x_1709_);
lean_ctor_set(v___x_1713_, 1, v___x_1712_);
v___x_1714_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__6);
v___x_1715_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1715_, 0, v___x_1713_);
lean_ctor_set(v___x_1715_, 1, v___x_1714_);
v___x_1716_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2(v___x_1715_, v_a_1690_, v_a_1691_);
return v___x_1716_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___boxed(lean_object* v_val_1718_, lean_object* v___x_1719_, lean_object* v___y_1720_, lean_object* v_a_1721_, lean_object* v_a_1722_, lean_object* v___y_1723_, lean_object* v___x_1724_, lean_object* v_a_x3f_1725_, lean_object* v___y_1726_){
_start:
{
uint8_t v___x_4620__boxed_1727_; lean_object* v_res_1728_; 
v___x_4620__boxed_1727_ = lean_unbox(v___x_1724_);
v_res_1728_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0(v_val_1718_, v___x_1719_, v___y_1720_, v_a_1721_, v_a_1722_, v___y_1723_, v___x_4620__boxed_1727_, v_a_x3f_1725_);
lean_dec(v_a_x3f_1725_);
lean_dec(v___y_1723_);
lean_dec(v_a_1722_);
lean_dec_ref(v_a_1721_);
lean_dec(v___x_1719_);
lean_dec(v_val_1718_);
return v_res_1728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1(lean_object* v_x_1729_, lean_object* v_a_1730_, lean_object* v_a_1731_){
_start:
{
lean_object* v___y_1734_; lean_object* v_a_1735_; lean_object* v___x_1746_; uint8_t v___x_1747_; 
v___x_1746_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_commandGuard__min__heartbeatsApproximately__In_____00__closed__1));
lean_inc(v_x_1729_);
v___x_1747_ = l_Lean_Syntax_isOfKind(v_x_1729_, v___x_1746_);
if (v___x_1747_ == 0)
{
lean_object* v___x_1748_; 
lean_dec(v_x_1729_);
v___x_1748_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg();
return v___x_1748_;
}
else
{
lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___y_1756_; lean_object* v___y_1757_; lean_object* v___y_1758_; lean_object* v_a_1759_; lean_object* v___y_1809_; lean_object* v___y_1810_; lean_object* v___y_1811_; lean_object* v___y_1828_; lean_object* v___y_1829_; lean_object* v___y_1846_; lean_object* v___x_1857_; 
v___x_1749_ = lean_unsigned_to_nat(1u);
v___x_1750_ = l_Lean_Syntax_getArg(v_x_1729_, v___x_1749_);
v___x_1751_ = lean_unsigned_to_nat(2u);
v___x_1752_ = l_Lean_Syntax_getArg(v_x_1729_, v___x_1751_);
v___x_1753_ = lean_unsigned_to_nat(5u);
v___x_1754_ = l_Lean_Syntax_getArg(v_x_1729_, v___x_1753_);
lean_dec(v_x_1729_);
v___x_1857_ = l_Lean_Syntax_getOptional_x3f(v___x_1752_);
lean_dec(v___x_1752_);
if (lean_obj_tag(v___x_1857_) == 0)
{
lean_object* v___x_1858_; 
v___x_1858_ = lean_box(0);
v___y_1846_ = v___x_1858_;
goto v___jp_1845_;
}
else
{
lean_object* v_val_1859_; lean_object* v___x_1861_; uint8_t v_isShared_1862_; uint8_t v_isSharedCheck_1866_; 
v_val_1859_ = lean_ctor_get(v___x_1857_, 0);
v_isSharedCheck_1866_ = !lean_is_exclusive(v___x_1857_);
if (v_isSharedCheck_1866_ == 0)
{
v___x_1861_ = v___x_1857_;
v_isShared_1862_ = v_isSharedCheck_1866_;
goto v_resetjp_1860_;
}
else
{
lean_inc(v_val_1859_);
lean_dec(v___x_1857_);
v___x_1861_ = lean_box(0);
v_isShared_1862_ = v_isSharedCheck_1866_;
goto v_resetjp_1860_;
}
v_resetjp_1860_:
{
lean_object* v___x_1864_; 
if (v_isShared_1862_ == 0)
{
v___x_1864_ = v___x_1861_;
goto v_reusejp_1863_;
}
else
{
lean_object* v_reuseFailAlloc_1865_; 
v_reuseFailAlloc_1865_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1865_, 0, v_val_1859_);
v___x_1864_ = v_reuseFailAlloc_1865_;
goto v_reusejp_1863_;
}
v_reusejp_1863_:
{
v___y_1846_ = v___x_1864_;
goto v___jp_1845_;
}
}
}
v___jp_1755_:
{
lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; 
v___x_1760_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11));
v___x_1761_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1));
v___x_1762_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1));
v___x_1763_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2));
lean_inc_n(v___y_1758_, 11);
v___x_1764_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1764_, 0, v___y_1758_);
lean_ctor_set(v___x_1764_, 1, v___x_1762_);
v___x_1765_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4);
v___x_1766_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5));
lean_inc(v___y_1756_);
lean_inc(v_a_1759_);
v___x_1767_ = l_Lean_addMacroScope(v_a_1759_, v___x_1766_, v___y_1756_);
v___x_1768_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__7));
v___x_1769_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1769_, 0, v___y_1758_);
lean_ctor_set(v___x_1769_, 1, v___x_1765_);
lean_ctor_set(v___x_1769_, 2, v___x_1767_);
lean_ctor_set(v___x_1769_, 3, v___x_1768_);
v___x_1770_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_1771_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_1772_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1772_, 0, v___y_1758_);
lean_ctor_set(v___x_1772_, 1, v___x_1770_);
lean_ctor_set(v___x_1772_, 2, v___x_1771_);
v___x_1773_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__8));
v___x_1774_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1774_, 0, v___y_1758_);
lean_ctor_set(v___x_1774_, 1, v___x_1773_);
lean_inc_ref(v___x_1772_);
lean_inc_ref(v___x_1764_);
v___x_1775_ = l_Lean_Syntax_node4(v___y_1758_, v___x_1763_, v___x_1764_, v___x_1769_, v___x_1772_, v___x_1774_);
v___x_1776_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1776_, 0, v___y_1758_);
lean_ctor_set(v___x_1776_, 1, v___x_1760_);
v___x_1777_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3);
v___x_1778_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__4));
v___x_1779_ = l_Lean_addMacroScope(v_a_1759_, v___x_1778_, v___y_1756_);
v___x_1780_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__11));
v___x_1781_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1781_, 0, v___y_1758_);
lean_ctor_set(v___x_1781_, 1, v___x_1777_);
lean_ctor_set(v___x_1781_, 2, v___x_1779_);
lean_ctor_set(v___x_1781_, 3, v___x_1780_);
v___x_1782_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__7));
v___x_1783_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__12));
v___x_1784_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1784_, 0, v___y_1758_);
lean_ctor_set(v___x_1784_, 1, v___x_1783_);
v___x_1785_ = l_Lean_Syntax_node1(v___y_1758_, v___x_1782_, v___x_1784_);
v___x_1786_ = l_Lean_Syntax_node4(v___y_1758_, v___x_1763_, v___x_1764_, v___x_1781_, v___x_1772_, v___x_1785_);
lean_inc_ref(v___x_1776_);
v___x_1787_ = l_Lean_Syntax_node3(v___y_1758_, v___x_1761_, v___x_1786_, v___x_1776_, v___x_1754_);
v___x_1788_ = l_Lean_Syntax_node3(v___y_1758_, v___x_1761_, v___x_1775_, v___x_1776_, v___x_1787_);
v___x_1789_ = l_Lean_Elab_Command_elabCommand(v___x_1788_, v_a_1730_, v_a_1731_);
if (lean_obj_tag(v___x_1789_) == 0)
{
lean_object* v_a_1790_; lean_object* v___x_1792_; uint8_t v_isShared_1793_; uint8_t v_isSharedCheck_1806_; 
v_a_1790_ = lean_ctor_get(v___x_1789_, 0);
v_isSharedCheck_1806_ = !lean_is_exclusive(v___x_1789_);
if (v_isSharedCheck_1806_ == 0)
{
v___x_1792_ = v___x_1789_;
v_isShared_1793_ = v_isSharedCheck_1806_;
goto v_resetjp_1791_;
}
else
{
lean_inc(v_a_1790_);
lean_dec(v___x_1789_);
v___x_1792_ = lean_box(0);
v_isShared_1793_ = v_isSharedCheck_1806_;
goto v_resetjp_1791_;
}
v_resetjp_1791_:
{
lean_object* v___x_1795_; 
lean_inc(v_a_1790_);
if (v_isShared_1793_ == 0)
{
lean_ctor_set_tag(v___x_1792_, 1);
v___x_1795_ = v___x_1792_;
goto v_reusejp_1794_;
}
else
{
lean_object* v_reuseFailAlloc_1805_; 
v_reuseFailAlloc_1805_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1805_, 0, v_a_1790_);
v___x_1795_ = v_reuseFailAlloc_1805_;
goto v_reusejp_1794_;
}
v_reusejp_1794_:
{
lean_object* v___x_1796_; 
v___x_1796_ = lean_apply_2(v___y_1757_, v___x_1795_, lean_box(0));
if (lean_obj_tag(v___x_1796_) == 0)
{
lean_object* v___x_1798_; uint8_t v_isShared_1799_; uint8_t v_isSharedCheck_1803_; 
v_isSharedCheck_1803_ = !lean_is_exclusive(v___x_1796_);
if (v_isSharedCheck_1803_ == 0)
{
lean_object* v_unused_1804_; 
v_unused_1804_ = lean_ctor_get(v___x_1796_, 0);
lean_dec(v_unused_1804_);
v___x_1798_ = v___x_1796_;
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
else
{
lean_dec(v___x_1796_);
v___x_1798_ = lean_box(0);
v_isShared_1799_ = v_isSharedCheck_1803_;
goto v_resetjp_1797_;
}
v_resetjp_1797_:
{
lean_object* v___x_1801_; 
if (v_isShared_1799_ == 0)
{
lean_ctor_set(v___x_1798_, 0, v_a_1790_);
v___x_1801_ = v___x_1798_;
goto v_reusejp_1800_;
}
else
{
lean_object* v_reuseFailAlloc_1802_; 
v_reuseFailAlloc_1802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1802_, 0, v_a_1790_);
v___x_1801_ = v_reuseFailAlloc_1802_;
goto v_reusejp_1800_;
}
v_reusejp_1800_:
{
return v___x_1801_;
}
}
}
else
{
lean_dec(v_a_1790_);
return v___x_1796_;
}
}
}
}
else
{
lean_object* v_a_1807_; 
v_a_1807_ = lean_ctor_get(v___x_1789_, 0);
lean_inc(v_a_1807_);
lean_dec_ref_known(v___x_1789_, 1);
v___y_1734_ = v___y_1757_;
v_a_1735_ = v_a_1807_;
goto v___jp_1733_;
}
}
v___jp_1808_:
{
lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___f_1814_; lean_object* v___x_1815_; 
v___x_1812_ = lean_io_get_num_heartbeats();
v___x_1813_ = lean_box(v___x_1747_);
lean_inc(v_a_1731_);
lean_inc_ref(v_a_1730_);
v___f_1814_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___lam__0___boxed), 9, 7);
lean_closure_set(v___f_1814_, 0, v___x_1812_);
lean_closure_set(v___f_1814_, 1, v___y_1810_);
lean_closure_set(v___f_1814_, 2, v___y_1811_);
lean_closure_set(v___f_1814_, 3, v_a_1730_);
lean_closure_set(v___f_1814_, 4, v_a_1731_);
lean_closure_set(v___f_1814_, 5, v___y_1809_);
lean_closure_set(v___f_1814_, 6, v___x_1813_);
v___x_1815_ = l_Lean_Elab_Command_getRef___redArg(v_a_1730_);
if (lean_obj_tag(v___x_1815_) == 0)
{
lean_object* v_a_1816_; lean_object* v___x_1817_; 
v_a_1816_ = lean_ctor_get(v___x_1815_, 0);
lean_inc(v_a_1816_);
lean_dec_ref_known(v___x_1815_, 1);
v___x_1817_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_1730_);
if (lean_obj_tag(v___x_1817_) == 0)
{
lean_object* v_a_1818_; lean_object* v_quotContext_x3f_1819_; uint8_t v___x_1820_; lean_object* v___x_1821_; 
v_a_1818_ = lean_ctor_get(v___x_1817_, 0);
lean_inc(v_a_1818_);
lean_dec_ref_known(v___x_1817_, 1);
v_quotContext_x3f_1819_ = lean_ctor_get(v_a_1730_, 5);
v___x_1820_ = 0;
v___x_1821_ = l_Lean_SourceInfo_fromRef(v_a_1816_, v___x_1820_);
lean_dec(v_a_1816_);
if (lean_obj_tag(v_quotContext_x3f_1819_) == 0)
{
lean_object* v___x_1822_; lean_object* v_a_1823_; 
v___x_1822_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(v_a_1731_);
v_a_1823_ = lean_ctor_get(v___x_1822_, 0);
lean_inc(v_a_1823_);
lean_dec_ref(v___x_1822_);
v___y_1756_ = v_a_1818_;
v___y_1757_ = v___f_1814_;
v___y_1758_ = v___x_1821_;
v_a_1759_ = v_a_1823_;
goto v___jp_1755_;
}
else
{
lean_object* v_val_1824_; 
v_val_1824_ = lean_ctor_get(v_quotContext_x3f_1819_, 0);
lean_inc(v_val_1824_);
v___y_1756_ = v_a_1818_;
v___y_1757_ = v___f_1814_;
v___y_1758_ = v___x_1821_;
v_a_1759_ = v_val_1824_;
goto v___jp_1755_;
}
}
else
{
lean_object* v_a_1825_; 
lean_dec(v_a_1816_);
lean_dec(v___x_1754_);
v_a_1825_ = lean_ctor_get(v___x_1817_, 0);
lean_inc(v_a_1825_);
lean_dec_ref_known(v___x_1817_, 1);
v___y_1734_ = v___f_1814_;
v_a_1735_ = v_a_1825_;
goto v___jp_1733_;
}
}
else
{
lean_object* v_a_1826_; 
lean_dec(v___x_1754_);
v_a_1826_ = lean_ctor_get(v___x_1815_, 0);
lean_inc(v_a_1826_);
lean_dec_ref_known(v___x_1815_, 1);
v___y_1734_ = v___f_1814_;
v_a_1735_ = v_a_1826_;
goto v___jp_1733_;
}
}
v___jp_1827_:
{
lean_object* v___x_1830_; lean_object* v___x_1831_; 
v___x_1830_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__1___closed__0));
v___x_1831_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_1830_, v_a_1730_, v_a_1731_);
if (lean_obj_tag(v___x_1831_) == 0)
{
lean_object* v_a_1832_; lean_object* v___x_1833_; 
v_a_1832_ = lean_ctor_get(v___x_1831_, 0);
lean_inc(v_a_1832_);
lean_dec_ref_known(v___x_1831_, 1);
v___x_1833_ = lean_unsigned_to_nat(1000u);
if (lean_obj_tag(v___y_1828_) == 0)
{
lean_object* v___x_1834_; 
v___x_1834_ = lean_nat_div(v_a_1832_, v___x_1833_);
lean_dec(v_a_1832_);
v___y_1809_ = v___y_1829_;
v___y_1810_ = v___x_1833_;
v___y_1811_ = v___x_1834_;
goto v___jp_1808_;
}
else
{
lean_object* v_val_1835_; lean_object* v___x_1836_; 
lean_dec(v_a_1832_);
v_val_1835_ = lean_ctor_get(v___y_1828_, 0);
lean_inc(v_val_1835_);
lean_dec_ref_known(v___y_1828_, 1);
v___x_1836_ = l_Lean_TSyntax_getNat(v_val_1835_);
lean_dec(v_val_1835_);
v___y_1809_ = v___y_1829_;
v___y_1810_ = v___x_1833_;
v___y_1811_ = v___x_1836_;
goto v___jp_1808_;
}
}
else
{
lean_object* v_a_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1844_; 
lean_dec(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec(v___x_1754_);
v_a_1837_ = lean_ctor_get(v___x_1831_, 0);
v_isSharedCheck_1844_ = !lean_is_exclusive(v___x_1831_);
if (v_isSharedCheck_1844_ == 0)
{
v___x_1839_ = v___x_1831_;
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_a_1837_);
lean_dec(v___x_1831_);
v___x_1839_ = lean_box(0);
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
v_resetjp_1838_:
{
lean_object* v___x_1842_; 
if (v_isShared_1840_ == 0)
{
v___x_1842_ = v___x_1839_;
goto v_reusejp_1841_;
}
else
{
lean_object* v_reuseFailAlloc_1843_; 
v_reuseFailAlloc_1843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1843_, 0, v_a_1837_);
v___x_1842_ = v_reuseFailAlloc_1843_;
goto v_reusejp_1841_;
}
v_reusejp_1841_:
{
return v___x_1842_;
}
}
}
}
v___jp_1845_:
{
lean_object* v___x_1847_; 
v___x_1847_ = l_Lean_Syntax_getOptional_x3f(v___x_1750_);
lean_dec(v___x_1750_);
if (lean_obj_tag(v___x_1847_) == 0)
{
lean_object* v___x_1848_; 
v___x_1848_ = lean_box(0);
v___y_1828_ = v___y_1846_;
v___y_1829_ = v___x_1848_;
goto v___jp_1827_;
}
else
{
lean_object* v_val_1849_; lean_object* v___x_1851_; uint8_t v_isShared_1852_; uint8_t v_isSharedCheck_1856_; 
v_val_1849_ = lean_ctor_get(v___x_1847_, 0);
v_isSharedCheck_1856_ = !lean_is_exclusive(v___x_1847_);
if (v_isSharedCheck_1856_ == 0)
{
v___x_1851_ = v___x_1847_;
v_isShared_1852_ = v_isSharedCheck_1856_;
goto v_resetjp_1850_;
}
else
{
lean_inc(v_val_1849_);
lean_dec(v___x_1847_);
v___x_1851_ = lean_box(0);
v_isShared_1852_ = v_isSharedCheck_1856_;
goto v_resetjp_1850_;
}
v_resetjp_1850_:
{
lean_object* v___x_1854_; 
if (v_isShared_1852_ == 0)
{
v___x_1854_ = v___x_1851_;
goto v_reusejp_1853_;
}
else
{
lean_object* v_reuseFailAlloc_1855_; 
v_reuseFailAlloc_1855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1855_, 0, v_val_1849_);
v___x_1854_ = v_reuseFailAlloc_1855_;
goto v_reusejp_1853_;
}
v_reusejp_1853_:
{
v___y_1828_ = v___y_1846_;
v___y_1829_ = v___x_1854_;
goto v___jp_1827_;
}
}
}
}
}
v___jp_1733_:
{
lean_object* v___x_1736_; lean_object* v___x_1737_; 
v___x_1736_ = lean_box(0);
v___x_1737_ = lean_apply_2(v___y_1734_, v___x_1736_, lean_box(0));
if (lean_obj_tag(v___x_1737_) == 0)
{
lean_object* v___x_1739_; uint8_t v_isShared_1740_; uint8_t v_isSharedCheck_1744_; 
v_isSharedCheck_1744_ = !lean_is_exclusive(v___x_1737_);
if (v_isSharedCheck_1744_ == 0)
{
lean_object* v_unused_1745_; 
v_unused_1745_ = lean_ctor_get(v___x_1737_, 0);
lean_dec(v_unused_1745_);
v___x_1739_ = v___x_1737_;
v_isShared_1740_ = v_isSharedCheck_1744_;
goto v_resetjp_1738_;
}
else
{
lean_dec(v___x_1737_);
v___x_1739_ = lean_box(0);
v_isShared_1740_ = v_isSharedCheck_1744_;
goto v_resetjp_1738_;
}
v_resetjp_1738_:
{
lean_object* v___x_1742_; 
if (v_isShared_1740_ == 0)
{
lean_ctor_set_tag(v___x_1739_, 1);
lean_ctor_set(v___x_1739_, 0, v_a_1735_);
v___x_1742_ = v___x_1739_;
goto v_reusejp_1741_;
}
else
{
lean_object* v_reuseFailAlloc_1743_; 
v_reuseFailAlloc_1743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1743_, 0, v_a_1735_);
v___x_1742_ = v_reuseFailAlloc_1743_;
goto v_reusejp_1741_;
}
v_reusejp_1741_:
{
return v___x_1742_;
}
}
}
else
{
lean_dec_ref(v_a_1735_);
return v___x_1737_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1___boxed(lean_object* v_x_1867_, lean_object* v_a_1868_, lean_object* v_a_1869_, lean_object* v_a_1870_){
_start:
{
lean_object* v_res_1871_; 
v_res_1871_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__commandGuard__min__heartbeatsApproximately__In______1(v_x_1867_, v_a_1868_, v_a_1869_);
lean_dec(v_a_1869_);
lean_dec_ref(v_a_1868_);
return v_res_1871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_elabForHeartbeats(lean_object* v_cmd_1872_, uint8_t v_revert_1873_, lean_object* v_a_1874_, lean_object* v_a_1875_){
_start:
{
lean_object* v___x_1877_; lean_object* v___x_1882_; lean_object* v___x_1883_; 
v___x_1877_ = lean_io_get_num_heartbeats();
v___x_1882_ = lean_st_ref_get(v_a_1875_);
v___x_1883_ = l_Lean_Elab_Command_getRef___redArg(v_a_1874_);
if (lean_obj_tag(v___x_1883_) == 0)
{
lean_object* v_a_1884_; lean_object* v___x_1885_; 
v_a_1884_ = lean_ctor_get(v___x_1883_, 0);
lean_inc(v_a_1884_);
lean_dec_ref_known(v___x_1883_, 1);
v___x_1885_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_1874_);
if (lean_obj_tag(v___x_1885_) == 0)
{
lean_object* v_a_1886_; lean_object* v_quotContext_x3f_1887_; uint8_t v___x_1888_; lean_object* v___x_1889_; lean_object* v_a_1891_; 
v_a_1886_ = lean_ctor_get(v___x_1885_, 0);
lean_inc(v_a_1886_);
lean_dec_ref_known(v___x_1885_, 1);
v_quotContext_x3f_1887_ = lean_ctor_get(v_a_1874_, 5);
v___x_1888_ = 0;
v___x_1889_ = l_Lean_SourceInfo_fromRef(v_a_1884_, v___x_1888_);
lean_dec(v_a_1884_);
if (lean_obj_tag(v_quotContext_x3f_1887_) == 0)
{
lean_object* v___x_1931_; lean_object* v_a_1932_; 
v___x_1931_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(v_a_1875_);
v_a_1932_ = lean_ctor_get(v___x_1931_, 0);
lean_inc(v_a_1932_);
lean_dec_ref(v___x_1931_);
v_a_1891_ = v_a_1932_;
goto v___jp_1890_;
}
else
{
lean_object* v_val_1933_; 
v_val_1933_ = lean_ctor_get(v_quotContext_x3f_1887_, 0);
lean_inc(v_val_1933_);
v_a_1891_ = v_val_1933_;
goto v___jp_1890_;
}
v___jp_1890_:
{
lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; lean_object* v___x_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; 
v___x_1892_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11));
v___x_1893_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__1));
v___x_1894_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1));
v___x_1895_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2));
lean_inc_n(v___x_1889_, 11);
v___x_1896_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1896_, 0, v___x_1889_);
lean_ctor_set(v___x_1896_, 1, v___x_1894_);
v___x_1897_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__4);
v___x_1898_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__5));
lean_inc(v_a_1886_);
lean_inc(v_a_1891_);
v___x_1899_ = l_Lean_addMacroScope(v_a_1891_, v___x_1898_, v_a_1886_);
v___x_1900_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__7));
v___x_1901_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1901_, 0, v___x_1889_);
lean_ctor_set(v___x_1901_, 1, v___x_1897_);
lean_ctor_set(v___x_1901_, 2, v___x_1899_);
lean_ctor_set(v___x_1901_, 3, v___x_1900_);
v___x_1902_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_1903_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_1904_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1904_, 0, v___x_1889_);
lean_ctor_set(v___x_1904_, 1, v___x_1902_);
lean_ctor_set(v___x_1904_, 2, v___x_1903_);
v___x_1905_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__8));
v___x_1906_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1906_, 0, v___x_1889_);
lean_ctor_set(v___x_1906_, 1, v___x_1905_);
lean_inc_ref(v___x_1904_);
lean_inc_ref(v___x_1896_);
v___x_1907_ = l_Lean_Syntax_node4(v___x_1889_, v___x_1895_, v___x_1896_, v___x_1901_, v___x_1904_, v___x_1906_);
v___x_1908_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1908_, 0, v___x_1889_);
lean_ctor_set(v___x_1908_, 1, v___x_1892_);
v___x_1909_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__3);
v___x_1910_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__4));
v___x_1911_ = l_Lean_addMacroScope(v_a_1891_, v___x_1910_, v_a_1886_);
v___x_1912_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__11));
v___x_1913_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1913_, 0, v___x_1889_);
lean_ctor_set(v___x_1913_, 1, v___x_1909_);
lean_ctor_set(v___x_1913_, 2, v___x_1911_);
lean_ctor_set(v___x_1913_, 3, v___x_1912_);
v___x_1914_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__7));
v___x_1915_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__12));
v___x_1916_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1916_, 0, v___x_1889_);
lean_ctor_set(v___x_1916_, 1, v___x_1915_);
v___x_1917_ = l_Lean_Syntax_node1(v___x_1889_, v___x_1914_, v___x_1916_);
v___x_1918_ = l_Lean_Syntax_node4(v___x_1889_, v___x_1895_, v___x_1896_, v___x_1913_, v___x_1904_, v___x_1917_);
lean_inc_ref(v___x_1908_);
v___x_1919_ = l_Lean_Syntax_node3(v___x_1889_, v___x_1893_, v___x_1918_, v___x_1908_, v_cmd_1872_);
v___x_1920_ = l_Lean_Syntax_node3(v___x_1889_, v___x_1893_, v___x_1907_, v___x_1908_, v___x_1919_);
v___x_1921_ = l_Lean_Elab_Command_elabCommand(v___x_1920_, v_a_1874_, v_a_1875_);
if (lean_obj_tag(v___x_1921_) == 0)
{
lean_dec_ref_known(v___x_1921_, 1);
if (v_revert_1873_ == 0)
{
lean_dec(v___x_1882_);
goto v___jp_1878_;
}
else
{
lean_object* v___x_1922_; 
v___x_1922_ = lean_st_ref_set(v_a_1875_, v___x_1882_);
goto v___jp_1878_;
}
}
else
{
lean_object* v_a_1923_; lean_object* v___x_1925_; uint8_t v_isShared_1926_; uint8_t v_isSharedCheck_1930_; 
lean_dec(v___x_1882_);
lean_dec(v___x_1877_);
v_a_1923_ = lean_ctor_get(v___x_1921_, 0);
v_isSharedCheck_1930_ = !lean_is_exclusive(v___x_1921_);
if (v_isSharedCheck_1930_ == 0)
{
v___x_1925_ = v___x_1921_;
v_isShared_1926_ = v_isSharedCheck_1930_;
goto v_resetjp_1924_;
}
else
{
lean_inc(v_a_1923_);
lean_dec(v___x_1921_);
v___x_1925_ = lean_box(0);
v_isShared_1926_ = v_isSharedCheck_1930_;
goto v_resetjp_1924_;
}
v_resetjp_1924_:
{
lean_object* v___x_1928_; 
if (v_isShared_1926_ == 0)
{
v___x_1928_ = v___x_1925_;
goto v_reusejp_1927_;
}
else
{
lean_object* v_reuseFailAlloc_1929_; 
v_reuseFailAlloc_1929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1929_, 0, v_a_1923_);
v___x_1928_ = v_reuseFailAlloc_1929_;
goto v_reusejp_1927_;
}
v_reusejp_1927_:
{
return v___x_1928_;
}
}
}
}
}
else
{
lean_dec(v_a_1884_);
lean_dec(v___x_1882_);
lean_dec(v___x_1877_);
lean_dec(v_cmd_1872_);
return v___x_1885_;
}
}
else
{
lean_object* v_a_1934_; lean_object* v___x_1936_; uint8_t v_isShared_1937_; uint8_t v_isSharedCheck_1941_; 
lean_dec(v___x_1882_);
lean_dec(v___x_1877_);
lean_dec(v_cmd_1872_);
v_a_1934_ = lean_ctor_get(v___x_1883_, 0);
v_isSharedCheck_1941_ = !lean_is_exclusive(v___x_1883_);
if (v_isSharedCheck_1941_ == 0)
{
v___x_1936_ = v___x_1883_;
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
else
{
lean_inc(v_a_1934_);
lean_dec(v___x_1883_);
v___x_1936_ = lean_box(0);
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
v_resetjp_1935_:
{
lean_object* v___x_1939_; 
if (v_isShared_1937_ == 0)
{
v___x_1939_ = v___x_1936_;
goto v_reusejp_1938_;
}
else
{
lean_object* v_reuseFailAlloc_1940_; 
v_reuseFailAlloc_1940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1940_, 0, v_a_1934_);
v___x_1939_ = v_reuseFailAlloc_1940_;
goto v_reusejp_1938_;
}
v_reusejp_1938_:
{
return v___x_1939_;
}
}
}
v___jp_1878_:
{
lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
v___x_1879_ = lean_io_get_num_heartbeats();
v___x_1880_ = lean_nat_sub(v___x_1879_, v___x_1877_);
lean_dec(v___x_1877_);
lean_dec(v___x_1879_);
v___x_1881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1881_, 0, v___x_1880_);
return v___x_1881_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_elabForHeartbeats___boxed(lean_object* v_cmd_1942_, lean_object* v_revert_1943_, lean_object* v_a_1944_, lean_object* v_a_1945_, lean_object* v_a_1946_){
_start:
{
uint8_t v_revert_boxed_1947_; lean_object* v_res_1948_; 
v_revert_boxed_1947_ = lean_unbox(v_revert_1943_);
v_res_1948_ = lp_mathlib_Mathlib_CountHeartbeats_elabForHeartbeats(v_cmd_1942_, v_revert_boxed_1947_, v_a_1944_, v_a_1945_);
lean_dec(v_a_1945_);
lean_dec_ref(v_a_1944_);
return v_res_1948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__0(lean_object* v_cmd_1977_, uint8_t v___x_1978_, lean_object* v_x_1979_, lean_object* v_x_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_){
_start:
{
if (lean_obj_tag(v_x_1979_) == 0)
{
lean_object* v___x_1984_; lean_object* v___x_1985_; 
lean_dec(v_cmd_1977_);
v___x_1984_ = l_List_reverse___redArg(v_x_1980_);
v___x_1985_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1985_, 0, v___x_1984_);
return v___x_1985_;
}
else
{
lean_object* v_tail_1986_; lean_object* v___x_1988_; uint8_t v_isShared_1989_; uint8_t v_isSharedCheck_2004_; 
v_tail_1986_ = lean_ctor_get(v_x_1979_, 1);
v_isSharedCheck_2004_ = !lean_is_exclusive(v_x_1979_);
if (v_isSharedCheck_2004_ == 0)
{
lean_object* v_unused_2005_; 
v_unused_2005_ = lean_ctor_get(v_x_1979_, 0);
lean_dec(v_unused_2005_);
v___x_1988_ = v_x_1979_;
v_isShared_1989_ = v_isSharedCheck_2004_;
goto v_resetjp_1987_;
}
else
{
lean_inc(v_tail_1986_);
lean_dec(v_x_1979_);
v___x_1988_ = lean_box(0);
v_isShared_1989_ = v_isSharedCheck_2004_;
goto v_resetjp_1987_;
}
v_resetjp_1987_:
{
lean_object* v___x_1990_; 
lean_inc(v_cmd_1977_);
v___x_1990_ = lp_mathlib_Mathlib_CountHeartbeats_elabForHeartbeats(v_cmd_1977_, v___x_1978_, v___y_1981_, v___y_1982_);
if (lean_obj_tag(v___x_1990_) == 0)
{
lean_object* v_a_1991_; lean_object* v___x_1993_; 
v_a_1991_ = lean_ctor_get(v___x_1990_, 0);
lean_inc(v_a_1991_);
lean_dec_ref_known(v___x_1990_, 1);
if (v_isShared_1989_ == 0)
{
lean_ctor_set(v___x_1988_, 1, v_x_1980_);
lean_ctor_set(v___x_1988_, 0, v_a_1991_);
v___x_1993_ = v___x_1988_;
goto v_reusejp_1992_;
}
else
{
lean_object* v_reuseFailAlloc_1995_; 
v_reuseFailAlloc_1995_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1995_, 0, v_a_1991_);
lean_ctor_set(v_reuseFailAlloc_1995_, 1, v_x_1980_);
v___x_1993_ = v_reuseFailAlloc_1995_;
goto v_reusejp_1992_;
}
v_reusejp_1992_:
{
v_x_1979_ = v_tail_1986_;
v_x_1980_ = v___x_1993_;
goto _start;
}
}
else
{
lean_object* v_a_1996_; lean_object* v___x_1998_; uint8_t v_isShared_1999_; uint8_t v_isSharedCheck_2003_; 
lean_del_object(v___x_1988_);
lean_dec(v_tail_1986_);
lean_dec(v_x_1980_);
lean_dec(v_cmd_1977_);
v_a_1996_ = lean_ctor_get(v___x_1990_, 0);
v_isSharedCheck_2003_ = !lean_is_exclusive(v___x_1990_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1998_ = v___x_1990_;
v_isShared_1999_ = v_isSharedCheck_2003_;
goto v_resetjp_1997_;
}
else
{
lean_inc(v_a_1996_);
lean_dec(v___x_1990_);
v___x_1998_ = lean_box(0);
v_isShared_1999_ = v_isSharedCheck_2003_;
goto v_resetjp_1997_;
}
v_resetjp_1997_:
{
lean_object* v___x_2001_; 
if (v_isShared_1999_ == 0)
{
v___x_2001_ = v___x_1998_;
goto v_reusejp_2000_;
}
else
{
lean_object* v_reuseFailAlloc_2002_; 
v_reuseFailAlloc_2002_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2002_, 0, v_a_1996_);
v___x_2001_ = v_reuseFailAlloc_2002_;
goto v_reusejp_2000_;
}
v_reusejp_2000_:
{
return v___x_2001_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__0___boxed(lean_object* v_cmd_2006_, lean_object* v___x_2007_, lean_object* v_x_2008_, lean_object* v_x_2009_, lean_object* v___y_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_){
_start:
{
uint8_t v___x_760__boxed_2013_; lean_object* v_res_2014_; 
v___x_760__boxed_2013_ = lean_unbox(v___x_2007_);
v_res_2014_ = lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__0(v_cmd_2006_, v___x_760__boxed_2013_, v_x_2008_, v_x_2009_, v___y_2010_, v___y_2011_);
lean_dec(v___y_2011_);
lean_dec_ref(v___y_2010_);
return v_res_2014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__1(lean_object* v_counts_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_){
_start:
{
lean_object* v___x_2022_; 
v___x_2022_ = lp_mathlib_Mathlib_CountHeartbeats_variation(v_counts_2015_);
if (lean_obj_tag(v___x_2022_) == 1)
{
lean_object* v_tail_2023_; 
v_tail_2023_ = lean_ctor_get(v___x_2022_, 1);
lean_inc(v_tail_2023_);
if (lean_obj_tag(v_tail_2023_) == 1)
{
lean_object* v_tail_2024_; 
v_tail_2024_ = lean_ctor_get(v_tail_2023_, 1);
lean_inc(v_tail_2024_);
if (lean_obj_tag(v_tail_2024_) == 1)
{
lean_object* v_tail_2025_; 
v_tail_2025_ = lean_ctor_get(v_tail_2024_, 1);
if (lean_obj_tag(v_tail_2025_) == 0)
{
lean_object* v_head_2026_; lean_object* v_head_2027_; lean_object* v_head_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; 
v_head_2026_ = lean_ctor_get(v___x_2022_, 0);
lean_inc(v_head_2026_);
lean_dec_ref_known(v___x_2022_, 2);
v_head_2027_ = lean_ctor_get(v_tail_2023_, 0);
lean_inc(v_head_2027_);
lean_dec_ref_known(v_tail_2023_, 2);
v_head_2028_ = lean_ctor_get(v_tail_2024_, 0);
lean_inc(v_head_2028_);
lean_dec_ref_known(v_tail_2024_, 2);
v___x_2029_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__0));
v___x_2030_ = lean_unsigned_to_nat(1000u);
v___x_2031_ = lean_nat_div(v_head_2026_, v___x_2030_);
lean_dec(v_head_2026_);
v___x_2032_ = l_Nat_reprFast(v___x_2031_);
v___x_2033_ = lean_string_append(v___x_2029_, v___x_2032_);
lean_dec_ref(v___x_2032_);
v___x_2034_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__1));
v___x_2035_ = lean_string_append(v___x_2033_, v___x_2034_);
v___x_2036_ = lean_nat_div(v_head_2027_, v___x_2030_);
lean_dec(v_head_2027_);
v___x_2037_ = l_Nat_reprFast(v___x_2036_);
v___x_2038_ = lean_string_append(v___x_2035_, v___x_2037_);
lean_dec_ref(v___x_2037_);
v___x_2039_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__2));
v___x_2040_ = lean_string_append(v___x_2038_, v___x_2039_);
v___x_2041_ = lean_unsigned_to_nat(10u);
v___x_2042_ = lean_nat_div(v_head_2028_, v___x_2041_);
lean_dec(v_head_2028_);
v___x_2043_ = l_Nat_reprFast(v___x_2042_);
v___x_2044_ = lean_string_append(v___x_2040_, v___x_2043_);
lean_dec_ref(v___x_2043_);
v___x_2045_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_logVariation___redArg___closed__3));
v___x_2046_ = lean_string_append(v___x_2044_, v___x_2045_);
v___x_2047_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2047_, 0, v___x_2046_);
v___x_2048_ = l_Lean_MessageData_ofFormat(v___x_2047_);
v___x_2049_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2(v___x_2048_, v___y_2016_, v___y_2017_);
return v___x_2049_;
}
else
{
lean_dec_ref_known(v_tail_2024_, 2);
lean_dec_ref_known(v_tail_2023_, 2);
lean_dec_ref_known(v___x_2022_, 2);
goto v___jp_2019_;
}
}
else
{
lean_dec(v_tail_2024_);
lean_dec_ref_known(v_tail_2023_, 2);
lean_dec_ref_known(v___x_2022_, 2);
goto v___jp_2019_;
}
}
else
{
lean_dec_ref_known(v___x_2022_, 2);
lean_dec(v_tail_2023_);
goto v___jp_2019_;
}
}
else
{
lean_dec(v___x_2022_);
goto v___jp_2019_;
}
v___jp_2019_:
{
lean_object* v___x_2020_; lean_object* v___x_2021_; 
v___x_2020_ = lean_box(0);
v___x_2021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2021_, 0, v___x_2020_);
return v___x_2021_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__1___boxed(lean_object* v_counts_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_){
_start:
{
lean_object* v_res_2054_; 
v_res_2054_ = lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__1(v_counts_2050_, v___y_2051_, v___y_2052_);
lean_dec(v___y_2052_);
lean_dec_ref(v___y_2051_);
return v_res_2054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1(lean_object* v_x_2055_, lean_object* v_a_2056_, lean_object* v_a_2057_){
_start:
{
lean_object* v___x_2059_; uint8_t v___x_2060_; 
v___x_2059_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeats_x21__In_____00__closed__1));
lean_inc(v_x_2055_);
v___x_2060_ = l_Lean_Syntax_isOfKind(v_x_2055_, v___x_2059_);
if (v___x_2060_ == 0)
{
lean_object* v___x_2061_; 
lean_dec(v_x_2055_);
v___x_2061_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__0___redArg();
return v___x_2061_;
}
else
{
lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v_cmd_2065_; lean_object* v___y_2067_; lean_object* v___x_2094_; 
v___x_2062_ = lean_unsigned_to_nat(1u);
v___x_2063_ = l_Lean_Syntax_getArg(v_x_2055_, v___x_2062_);
v___x_2064_ = lean_unsigned_to_nat(4u);
v_cmd_2065_ = l_Lean_Syntax_getArg(v_x_2055_, v___x_2064_);
lean_dec(v_x_2055_);
v___x_2094_ = l_Lean_Syntax_getOptional_x3f(v___x_2063_);
lean_dec(v___x_2063_);
if (lean_obj_tag(v___x_2094_) == 0)
{
lean_object* v___x_2095_; 
v___x_2095_ = lean_unsigned_to_nat(10u);
v___y_2067_ = v___x_2095_;
goto v___jp_2066_;
}
else
{
lean_object* v_val_2096_; lean_object* v___x_2097_; 
v_val_2096_ = lean_ctor_get(v___x_2094_, 0);
lean_inc(v_val_2096_);
lean_dec_ref_known(v___x_2094_, 1);
v___x_2097_ = l_Lean_TSyntax_getNat(v_val_2096_);
lean_dec(v_val_2096_);
v___y_2067_ = v___x_2097_;
goto v___jp_2066_;
}
v___jp_2066_:
{
lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; 
v___x_2068_ = lean_nat_sub(v___y_2067_, v___x_2062_);
lean_dec(v___y_2067_);
v___x_2069_ = l_List_range(v___x_2068_);
v___x_2070_ = lean_box(0);
lean_inc(v_cmd_2065_);
v___x_2071_ = lp_mathlib_List_mapM_loop___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__0(v_cmd_2065_, v___x_2060_, v___x_2069_, v___x_2070_, v_a_2056_, v_a_2057_);
if (lean_obj_tag(v___x_2071_) == 0)
{
lean_object* v_a_2072_; uint8_t v___x_2073_; lean_object* v___x_2074_; 
v_a_2072_ = lean_ctor_get(v___x_2071_, 0);
lean_inc(v_a_2072_);
lean_dec_ref_known(v___x_2071_, 1);
v___x_2073_ = 0;
v___x_2074_ = lp_mathlib_Mathlib_CountHeartbeats_elabForHeartbeats(v_cmd_2065_, v___x_2073_, v_a_2056_, v_a_2057_);
if (lean_obj_tag(v___x_2074_) == 0)
{
lean_object* v_a_2075_; lean_object* v___x_2076_; lean_object* v___x_2077_; 
v_a_2075_ = lean_ctor_get(v___x_2074_, 0);
lean_inc(v_a_2075_);
lean_dec_ref_known(v___x_2074_, 1);
v___x_2076_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2076_, 0, v_a_2075_);
lean_ctor_set(v___x_2076_, 1, v_a_2072_);
v___x_2077_ = lp_mathlib_Mathlib_CountHeartbeats_logVariation___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1_spec__1(v___x_2076_, v_a_2056_, v_a_2057_);
return v___x_2077_;
}
else
{
lean_object* v_a_2078_; lean_object* v___x_2080_; uint8_t v_isShared_2081_; uint8_t v_isSharedCheck_2085_; 
lean_dec(v_a_2072_);
v_a_2078_ = lean_ctor_get(v___x_2074_, 0);
v_isSharedCheck_2085_ = !lean_is_exclusive(v___x_2074_);
if (v_isSharedCheck_2085_ == 0)
{
v___x_2080_ = v___x_2074_;
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
else
{
lean_inc(v_a_2078_);
lean_dec(v___x_2074_);
v___x_2080_ = lean_box(0);
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
v_resetjp_2079_:
{
lean_object* v___x_2083_; 
if (v_isShared_2081_ == 0)
{
v___x_2083_ = v___x_2080_;
goto v_reusejp_2082_;
}
else
{
lean_object* v_reuseFailAlloc_2084_; 
v_reuseFailAlloc_2084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2084_, 0, v_a_2078_);
v___x_2083_ = v_reuseFailAlloc_2084_;
goto v_reusejp_2082_;
}
v_reusejp_2082_:
{
return v___x_2083_;
}
}
}
}
else
{
lean_object* v_a_2086_; lean_object* v___x_2088_; uint8_t v_isShared_2089_; uint8_t v_isSharedCheck_2093_; 
lean_dec(v_cmd_2065_);
v_a_2086_ = lean_ctor_get(v___x_2071_, 0);
v_isSharedCheck_2093_ = !lean_is_exclusive(v___x_2071_);
if (v_isSharedCheck_2093_ == 0)
{
v___x_2088_ = v___x_2071_;
v_isShared_2089_ = v_isSharedCheck_2093_;
goto v_resetjp_2087_;
}
else
{
lean_inc(v_a_2086_);
lean_dec(v___x_2071_);
v___x_2088_ = lean_box(0);
v_isShared_2089_ = v_isSharedCheck_2093_;
goto v_resetjp_2087_;
}
v_resetjp_2087_:
{
lean_object* v___x_2091_; 
if (v_isShared_2089_ == 0)
{
v___x_2091_ = v___x_2088_;
goto v_reusejp_2090_;
}
else
{
lean_object* v_reuseFailAlloc_2092_; 
v_reuseFailAlloc_2092_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2092_, 0, v_a_2086_);
v___x_2091_ = v_reuseFailAlloc_2092_;
goto v_reusejp_2090_;
}
v_reusejp_2090_:
{
return v___x_2091_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1___boxed(lean_object* v_x_2098_, lean_object* v_a_2099_, lean_object* v_a_2100_, lean_object* v_a_2101_){
_start:
{
lean_object* v_res_2102_; 
v_res_2102_ = lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeats_x21__In______1(v_x_2098_, v_a_2099_, v_a_2100_);
lean_dec(v_a_2100_);
lean_dec_ref(v_a_2099_);
return v_res_2102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__spec__0(lean_object* v_name_2103_, lean_object* v_decl_2104_, lean_object* v_ref_2105_){
_start:
{
lean_object* v_defValue_2107_; lean_object* v_descr_2108_; lean_object* v_deprecation_x3f_2109_; lean_object* v___x_2110_; uint8_t v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; 
v_defValue_2107_ = lean_ctor_get(v_decl_2104_, 0);
v_descr_2108_ = lean_ctor_get(v_decl_2104_, 1);
v_deprecation_x3f_2109_ = lean_ctor_get(v_decl_2104_, 2);
v___x_2110_ = lean_alloc_ctor(1, 0, 1);
v___x_2111_ = lean_unbox(v_defValue_2107_);
lean_ctor_set_uint8(v___x_2110_, 0, v___x_2111_);
lean_inc(v_deprecation_x3f_2109_);
lean_inc_ref(v_descr_2108_);
lean_inc_n(v_name_2103_, 2);
v___x_2112_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2112_, 0, v_name_2103_);
lean_ctor_set(v___x_2112_, 1, v_ref_2105_);
lean_ctor_set(v___x_2112_, 2, v___x_2110_);
lean_ctor_set(v___x_2112_, 3, v_descr_2108_);
lean_ctor_set(v___x_2112_, 4, v_deprecation_x3f_2109_);
v___x_2113_ = lean_register_option(v_name_2103_, v___x_2112_);
if (lean_obj_tag(v___x_2113_) == 0)
{
lean_object* v___x_2115_; uint8_t v_isShared_2116_; uint8_t v_isSharedCheck_2121_; 
v_isSharedCheck_2121_ = !lean_is_exclusive(v___x_2113_);
if (v_isSharedCheck_2121_ == 0)
{
lean_object* v_unused_2122_; 
v_unused_2122_ = lean_ctor_get(v___x_2113_, 0);
lean_dec(v_unused_2122_);
v___x_2115_ = v___x_2113_;
v_isShared_2116_ = v_isSharedCheck_2121_;
goto v_resetjp_2114_;
}
else
{
lean_dec(v___x_2113_);
v___x_2115_ = lean_box(0);
v_isShared_2116_ = v_isSharedCheck_2121_;
goto v_resetjp_2114_;
}
v_resetjp_2114_:
{
lean_object* v___x_2117_; lean_object* v___x_2119_; 
lean_inc(v_defValue_2107_);
v___x_2117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2117_, 0, v_name_2103_);
lean_ctor_set(v___x_2117_, 1, v_defValue_2107_);
if (v_isShared_2116_ == 0)
{
lean_ctor_set(v___x_2115_, 0, v___x_2117_);
v___x_2119_ = v___x_2115_;
goto v_reusejp_2118_;
}
else
{
lean_object* v_reuseFailAlloc_2120_; 
v_reuseFailAlloc_2120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2120_, 0, v___x_2117_);
v___x_2119_ = v_reuseFailAlloc_2120_;
goto v_reusejp_2118_;
}
v_reusejp_2118_:
{
return v___x_2119_;
}
}
}
else
{
lean_object* v_a_2123_; lean_object* v___x_2125_; uint8_t v_isShared_2126_; uint8_t v_isSharedCheck_2130_; 
lean_dec(v_name_2103_);
v_a_2123_ = lean_ctor_get(v___x_2113_, 0);
v_isSharedCheck_2130_ = !lean_is_exclusive(v___x_2113_);
if (v_isSharedCheck_2130_ == 0)
{
v___x_2125_ = v___x_2113_;
v_isShared_2126_ = v_isSharedCheck_2130_;
goto v_resetjp_2124_;
}
else
{
lean_inc(v_a_2123_);
lean_dec(v___x_2113_);
v___x_2125_ = lean_box(0);
v_isShared_2126_ = v_isSharedCheck_2130_;
goto v_resetjp_2124_;
}
v_resetjp_2124_:
{
lean_object* v___x_2128_; 
if (v_isShared_2126_ == 0)
{
v___x_2128_ = v___x_2125_;
goto v_reusejp_2127_;
}
else
{
lean_object* v_reuseFailAlloc_2129_; 
v_reuseFailAlloc_2129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2129_, 0, v_a_2123_);
v___x_2128_ = v_reuseFailAlloc_2129_;
goto v_reusejp_2127_;
}
v_reusejp_2127_:
{
return v___x_2128_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_2131_, lean_object* v_decl_2132_, lean_object* v_ref_2133_, lean_object* v_a_2134_){
_start:
{
lean_object* v_res_2135_; 
v_res_2135_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__spec__0(v_name_2131_, v_decl_2132_, v_ref_2133_);
lean_dec_ref(v_decl_2132_);
return v_res_2135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; 
v___x_2164_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_));
v___x_2165_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__9_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_));
v___x_2166_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__11_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_));
v___x_2167_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__spec__0(v___x_2164_, v___x_2165_, v___x_2166_);
return v___x_2167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4____boxed(lean_object* v_a_2168_){
_start:
{
lean_object* v_res_2169_; 
v_res_2169_ = lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_();
return v_res_2169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; 
v___x_2186_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_));
v___x_2187_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_));
v___x_2188_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_));
v___x_2189_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4__spec__0(v___x_2186_, v___x_2187_, v___x_2188_);
return v___x_2189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4____boxed(lean_object* v_a_2190_){
_start:
{
lean_object* v_res_2191_; 
v_res_2191_ = lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_();
return v_res_2191_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0(lean_object* v_x_2198_){
_start:
{
lean_object* v___x_2199_; uint8_t v___x_2200_; 
v___x_2199_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___closed__1));
v___x_2200_ = l_Lean_Syntax_isOfKind(v_x_2198_, v___x_2199_);
return v___x_2200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0___boxed(lean_object* v_x_2201_){
_start:
{
uint8_t v_res_2202_; lean_object* v_r_2203_; 
v_res_2202_ = lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__0(v_x_2201_);
v_r_2203_ = lean_box(v_res_2202_);
return v_r_2203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__1(lean_object* v_ref_2204_, lean_object* v_msgData_2205_, lean_object* v___y_2206_, lean_object* v___y_2207_){
_start:
{
uint8_t v___x_2209_; uint8_t v___x_2210_; lean_object* v___x_2211_; 
v___x_2209_ = 0;
v___x_2210_ = 0;
v___x_2211_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__2_spec__2_spec__4(v_ref_2204_, v_msgData_2205_, v___x_2209_, v___x_2210_, v___y_2206_, v___y_2207_);
return v___x_2211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__1___boxed(lean_object* v_ref_2212_, lean_object* v_msgData_2213_, lean_object* v___y_2214_, lean_object* v___y_2215_, lean_object* v___y_2216_){
_start:
{
lean_object* v_res_2217_; 
v_res_2217_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__1(v_ref_2212_, v_msgData_2213_, v___y_2214_, v___y_2215_);
lean_dec(v___y_2215_);
lean_dec_ref(v___y_2214_);
lean_dec(v_ref_2212_);
return v_res_2217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__2(uint8_t v___x_2218_, lean_object* v_stx_2219_, lean_object* v_as_2220_, size_t v_sz_2221_, size_t v_i_2222_, lean_object* v_b_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_){
_start:
{
uint8_t v___x_2227_; 
v___x_2227_ = lean_usize_dec_lt(v_i_2222_, v_sz_2221_);
if (v___x_2227_ == 0)
{
lean_object* v___x_2228_; 
v___x_2228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2228_, 0, v_b_2223_);
return v___x_2228_;
}
else
{
lean_object* v_a_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; 
v_a_2229_ = lean_array_uget_borrowed(v_as_2220_, v_i_2222_);
lean_inc(v_a_2229_);
v___x_2230_ = l_Lean_Message_toString(v_a_2229_, v___x_2218_);
v___x_2231_ = l_Lean_stringToMessageData(v___x_2230_);
v___x_2232_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__1(v_stx_2219_, v___x_2231_, v___y_2224_, v___y_2225_);
if (lean_obj_tag(v___x_2232_) == 0)
{
lean_object* v___x_2233_; size_t v___x_2234_; size_t v___x_2235_; 
lean_dec_ref_known(v___x_2232_, 1);
v___x_2233_ = lean_box(0);
v___x_2234_ = ((size_t)1ULL);
v___x_2235_ = lean_usize_add(v_i_2222_, v___x_2234_);
v_i_2222_ = v___x_2235_;
v_b_2223_ = v___x_2233_;
goto _start;
}
else
{
return v___x_2232_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__2___boxed(lean_object* v___x_2237_, lean_object* v_stx_2238_, lean_object* v_as_2239_, lean_object* v_sz_2240_, lean_object* v_i_2241_, lean_object* v_b_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_, lean_object* v___y_2245_){
_start:
{
uint8_t v___x_9998__boxed_2246_; size_t v_sz_boxed_2247_; size_t v_i_boxed_2248_; lean_object* v_res_2249_; 
v___x_9998__boxed_2246_ = lean_unbox(v___x_2237_);
v_sz_boxed_2247_ = lean_unbox_usize(v_sz_2240_);
lean_dec(v_sz_2240_);
v_i_boxed_2248_ = lean_unbox_usize(v_i_2241_);
lean_dec(v_i_2241_);
v_res_2249_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__2(v___x_9998__boxed_2246_, v_stx_2238_, v_as_2239_, v_sz_boxed_2247_, v_i_boxed_2248_, v_b_2242_, v___y_2243_, v___y_2244_);
lean_dec(v___y_2244_);
lean_dec_ref(v___y_2243_);
lean_dec_ref(v_as_2239_);
lean_dec(v_stx_2238_);
return v_res_2249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___redArg(lean_object* v_o_2250_, lean_object* v___y_2251_){
_start:
{
lean_object* v___x_2253_; lean_object* v_env_2254_; lean_object* v___x_2255_; lean_object* v_toEnvExtension_2256_; lean_object* v_asyncMode_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v_merged_2261_; lean_object* v___x_2263_; uint8_t v_isShared_2264_; uint8_t v_isSharedCheck_2269_; 
v___x_2253_ = lean_st_ref_get(v___y_2251_);
v_env_2254_ = lean_ctor_get(v___x_2253_, 0);
lean_inc_ref(v_env_2254_);
lean_dec(v___x_2253_);
v___x_2255_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_2256_ = lean_ctor_get(v___x_2255_, 0);
v_asyncMode_2257_ = lean_ctor_get(v_toEnvExtension_2256_, 2);
v___x_2258_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_2259_ = lean_box(0);
v___x_2260_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_2258_, v___x_2255_, v_env_2254_, v_asyncMode_2257_, v___x_2259_);
v_merged_2261_ = lean_ctor_get(v___x_2260_, 0);
v_isSharedCheck_2269_ = !lean_is_exclusive(v___x_2260_);
if (v_isSharedCheck_2269_ == 0)
{
lean_object* v_unused_2270_; 
v_unused_2270_ = lean_ctor_get(v___x_2260_, 1);
lean_dec(v_unused_2270_);
v___x_2263_ = v___x_2260_;
v_isShared_2264_ = v_isSharedCheck_2269_;
goto v_resetjp_2262_;
}
else
{
lean_inc(v_merged_2261_);
lean_dec(v___x_2260_);
v___x_2263_ = lean_box(0);
v_isShared_2264_ = v_isSharedCheck_2269_;
goto v_resetjp_2262_;
}
v_resetjp_2262_:
{
lean_object* v___x_2266_; 
if (v_isShared_2264_ == 0)
{
lean_ctor_set(v___x_2263_, 1, v_merged_2261_);
lean_ctor_set(v___x_2263_, 0, v_o_2250_);
v___x_2266_ = v___x_2263_;
goto v_reusejp_2265_;
}
else
{
lean_object* v_reuseFailAlloc_2268_; 
v_reuseFailAlloc_2268_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2268_, 0, v_o_2250_);
lean_ctor_set(v_reuseFailAlloc_2268_, 1, v_merged_2261_);
v___x_2266_ = v_reuseFailAlloc_2268_;
goto v_reusejp_2265_;
}
v_reusejp_2265_:
{
lean_object* v___x_2267_; 
v___x_2267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2267_, 0, v___x_2266_);
return v___x_2267_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_){
_start:
{
lean_object* v_res_2274_; 
v_res_2274_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___redArg(v_o_2271_, v___y_2272_);
lean_dec(v___y_2272_);
return v_res_2274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0(lean_object* v___y_2275_, lean_object* v___y_2276_){
_start:
{
lean_object* v___x_2278_; lean_object* v_scopes_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v_opts_2282_; lean_object* v___x_2283_; 
v___x_2278_ = lean_st_ref_get(v___y_2276_);
v_scopes_2279_ = lean_ctor_get(v___x_2278_, 2);
lean_inc(v_scopes_2279_);
lean_dec(v___x_2278_);
v___x_2280_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2281_ = l_List_head_x21___redArg(v___x_2280_, v_scopes_2279_);
lean_dec(v_scopes_2279_);
v_opts_2282_ = lean_ctor_get(v___x_2281_, 1);
lean_inc_ref(v_opts_2282_);
lean_dec(v___x_2281_);
v___x_2283_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___redArg(v_opts_2282_, v___y_2276_);
return v___x_2283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0___boxed(lean_object* v___y_2284_, lean_object* v___y_2285_, lean_object* v___y_2286_){
_start:
{
lean_object* v_res_2287_; 
v_res_2287_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0(v___y_2284_, v___y_2285_);
lean_dec(v___y_2285_);
lean_dec_ref(v___y_2284_);
return v_res_2287_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__4(lean_object* v_a_2288_, lean_object* v_x_2289_){
_start:
{
if (lean_obj_tag(v_x_2289_) == 0)
{
uint8_t v___x_2290_; 
v___x_2290_ = 0;
return v___x_2290_;
}
else
{
lean_object* v_head_2291_; lean_object* v_tail_2292_; uint8_t v___x_2293_; 
v_head_2291_ = lean_ctor_get(v_x_2289_, 0);
v_tail_2292_ = lean_ctor_get(v_x_2289_, 1);
v___x_2293_ = lean_name_eq(v_a_2288_, v_head_2291_);
if (v___x_2293_ == 0)
{
v_x_2289_ = v_tail_2292_;
goto _start;
}
else
{
return v___x_2293_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__4___boxed(lean_object* v_a_2295_, lean_object* v_x_2296_){
_start:
{
uint8_t v_res_2297_; lean_object* v_r_2298_; 
v_res_2297_ = lp_mathlib_List_elem___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__4(v_a_2295_, v_x_2296_);
lean_dec(v_x_2296_);
lean_dec(v_a_2295_);
v_r_2298_ = lean_box(v_res_2297_);
return v_r_2298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__5(lean_object* v_as_2299_, size_t v_i_2300_, size_t v_stop_2301_, lean_object* v_b_2302_){
_start:
{
lean_object* v___y_2304_; uint8_t v___x_2308_; 
v___x_2308_ = lean_usize_dec_eq(v_i_2300_, v_stop_2301_);
if (v___x_2308_ == 0)
{
lean_object* v___x_2309_; uint8_t v_severity_2310_; uint8_t v___x_2311_; uint8_t v___x_2312_; 
v___x_2309_ = lean_array_uget_borrowed(v_as_2299_, v_i_2300_);
v_severity_2310_ = lean_ctor_get_uint8(v___x_2309_, sizeof(void*)*5 + 1);
v___x_2311_ = 2;
v___x_2312_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2310_, v___x_2311_);
if (v___x_2312_ == 0)
{
lean_object* v___x_2313_; 
lean_inc(v___x_2309_);
v___x_2313_ = lean_array_push(v_b_2302_, v___x_2309_);
v___y_2304_ = v___x_2313_;
goto v___jp_2303_;
}
else
{
v___y_2304_ = v_b_2302_;
goto v___jp_2303_;
}
}
else
{
return v_b_2302_;
}
v___jp_2303_:
{
size_t v___x_2305_; size_t v___x_2306_; 
v___x_2305_ = ((size_t)1ULL);
v___x_2306_ = lean_usize_add(v_i_2300_, v___x_2305_);
v_i_2300_ = v___x_2306_;
v_b_2302_ = v___y_2304_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__5___boxed(lean_object* v_as_2314_, lean_object* v_i_2315_, lean_object* v_stop_2316_, lean_object* v_b_2317_){
_start:
{
size_t v_i_boxed_2318_; size_t v_stop_boxed_2319_; lean_object* v_res_2320_; 
v_i_boxed_2318_ = lean_unbox_usize(v_i_2315_);
lean_dec(v_i_2315_);
v_stop_boxed_2319_ = lean_unbox_usize(v_stop_2316_);
lean_dec(v_stop_2316_);
v_res_2320_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__5(v_as_2314_, v_i_boxed_2318_, v_stop_boxed_2319_, v_b_2317_);
lean_dec_ref(v_as_2314_);
return v_res_2320_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__1(void){
_start:
{
lean_object* v___x_2322_; lean_object* v___x_2323_; 
v___x_2322_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__0));
v___x_2323_ = l_Lean_stringToMessageData(v___x_2322_);
return v___x_2323_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__3(void){
_start:
{
lean_object* v___x_2325_; lean_object* v___x_2326_; 
v___x_2325_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__2));
v___x_2326_ = l_Lean_stringToMessageData(v___x_2325_);
return v___x_2326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3(uint8_t v___x_2327_, lean_object* v_val_2328_, lean_object* v_as_2329_, size_t v_sz_2330_, size_t v_i_2331_, lean_object* v_b_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_){
_start:
{
uint8_t v___x_2336_; 
v___x_2336_ = lean_usize_dec_lt(v_i_2331_, v_sz_2330_);
if (v___x_2336_ == 0)
{
lean_object* v___x_2337_; 
v___x_2337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2337_, 0, v_b_2332_);
return v___x_2337_;
}
else
{
lean_object* v_a_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___y_2350_; uint32_t v___x_2357_; uint32_t v___x_2358_; uint8_t v___x_2359_; 
v_a_2338_ = lean_array_uget_borrowed(v_as_2329_, v_i_2331_);
lean_inc(v_a_2338_);
v___x_2339_ = l_Lean_Message_toString(v_a_2338_, v___x_2327_);
v___x_2340_ = lean_box(0);
v___x_2341_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__1);
v___x_2342_ = lean_unsigned_to_nat(0u);
v___x_2343_ = l_Lean_Syntax_getArg(v_val_2328_, v___x_2342_);
v___x_2344_ = l_Lean_Syntax_getId(v___x_2343_);
lean_dec(v___x_2343_);
v___x_2345_ = l_Lean_MessageData_ofName(v___x_2344_);
v___x_2346_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2346_, 0, v___x_2341_);
lean_ctor_set(v___x_2346_, 1, v___x_2345_);
v___x_2347_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___closed__3);
v___x_2348_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2348_, 0, v___x_2346_);
lean_ctor_set(v___x_2348_, 1, v___x_2347_);
v___x_2357_ = lean_string_utf8_get(v___x_2339_, v___x_2342_);
v___x_2358_ = 65;
v___x_2359_ = lean_uint32_dec_le(v___x_2358_, v___x_2357_);
if (v___x_2359_ == 0)
{
lean_object* v___x_2360_; 
v___x_2360_ = lean_string_utf8_set(v___x_2339_, v___x_2342_, v___x_2357_);
v___y_2350_ = v___x_2360_;
goto v___jp_2349_;
}
else
{
uint32_t v___x_2361_; uint8_t v___x_2362_; 
v___x_2361_ = 90;
v___x_2362_ = lean_uint32_dec_le(v___x_2357_, v___x_2361_);
if (v___x_2362_ == 0)
{
lean_object* v___x_2363_; 
v___x_2363_ = lean_string_utf8_set(v___x_2339_, v___x_2342_, v___x_2357_);
v___y_2350_ = v___x_2363_;
goto v___jp_2349_;
}
else
{
uint32_t v___x_2364_; uint32_t v___x_2365_; lean_object* v___x_2366_; 
v___x_2364_ = 32;
v___x_2365_ = lean_uint32_add(v___x_2357_, v___x_2364_);
v___x_2366_ = lean_string_utf8_set(v___x_2339_, v___x_2342_, v___x_2365_);
v___y_2350_ = v___x_2366_;
goto v___jp_2349_;
}
}
v___jp_2349_:
{
lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; 
v___x_2351_ = l_Lean_stringToMessageData(v___y_2350_);
v___x_2352_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2352_, 0, v___x_2348_);
lean_ctor_set(v___x_2352_, 1, v___x_2351_);
v___x_2353_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__1(v_val_2328_, v___x_2352_, v___y_2333_, v___y_2334_);
if (lean_obj_tag(v___x_2353_) == 0)
{
size_t v___x_2354_; size_t v___x_2355_; 
lean_dec_ref_known(v___x_2353_, 1);
v___x_2354_ = ((size_t)1ULL);
v___x_2355_ = lean_usize_add(v_i_2331_, v___x_2354_);
v_i_2331_ = v___x_2355_;
v_b_2332_ = v___x_2340_;
goto _start;
}
else
{
return v___x_2353_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3___boxed(lean_object* v___x_2367_, lean_object* v_val_2368_, lean_object* v_as_2369_, lean_object* v_sz_2370_, lean_object* v_i_2371_, lean_object* v_b_2372_, lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_){
_start:
{
uint8_t v___x_10126__boxed_2376_; size_t v_sz_boxed_2377_; size_t v_i_boxed_2378_; lean_object* v_res_2379_; 
v___x_10126__boxed_2376_ = lean_unbox(v___x_2367_);
v_sz_boxed_2377_ = lean_unbox_usize(v_sz_2370_);
lean_dec(v_sz_2370_);
v_i_boxed_2378_ = lean_unbox_usize(v_i_2371_);
lean_dec(v_i_2371_);
v_res_2379_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3(v___x_10126__boxed_2376_, v_val_2368_, v_as_2369_, v_sz_boxed_2377_, v_i_boxed_2378_, v_b_2372_, v___y_2373_, v___y_2374_);
lean_dec(v___y_2374_);
lean_dec_ref(v___y_2373_);
lean_dec_ref(v_as_2369_);
lean_dec(v_val_2368_);
return v_res_2379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1(lean_object* v___f_2399_, lean_object* v_stx_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_){
_start:
{
lean_object* v___x_2404_; lean_object* v_a_2405_; lean_object* v___x_2407_; uint8_t v_isShared_2408_; uint8_t v_isSharedCheck_2556_; 
v___x_2404_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0(v___y_2401_, v___y_2402_);
v_a_2405_ = lean_ctor_get(v___x_2404_, 0);
v_isSharedCheck_2556_ = !lean_is_exclusive(v___x_2404_);
if (v_isSharedCheck_2556_ == 0)
{
v___x_2407_ = v___x_2404_;
v_isShared_2408_ = v_isSharedCheck_2556_;
goto v_resetjp_2406_;
}
else
{
lean_inc(v_a_2405_);
lean_dec(v___x_2404_);
v___x_2407_ = lean_box(0);
v_isShared_2408_ = v_isSharedCheck_2556_;
goto v_resetjp_2406_;
}
v_resetjp_2406_:
{
lean_object* v___x_2409_; uint8_t v___x_2410_; 
v___x_2409_ = lp_mathlib_Mathlib_Linter_linter_countHeartbeats;
v___x_2410_ = l_Lean_Linter_getLinterValue(v___x_2409_, v_a_2405_);
lean_dec(v_a_2405_);
if (v___x_2410_ == 0)
{
lean_object* v___x_2411_; lean_object* v___x_2413_; 
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
v___x_2411_ = lean_box(0);
if (v_isShared_2408_ == 0)
{
lean_ctor_set(v___x_2407_, 0, v___x_2411_);
v___x_2413_ = v___x_2407_;
goto v_reusejp_2412_;
}
else
{
lean_object* v_reuseFailAlloc_2414_; 
v_reuseFailAlloc_2414_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2414_, 0, v___x_2411_);
v___x_2413_ = v_reuseFailAlloc_2414_;
goto v_reusejp_2412_;
}
v_reusejp_2412_:
{
return v___x_2413_;
}
}
else
{
lean_object* v___x_2415_; lean_object* v_messages_2416_; uint8_t v___x_2417_; lean_object* v_msgs_2419_; lean_object* v___y_2420_; lean_object* v___y_2421_; 
v___x_2415_ = lean_st_ref_get(v___y_2402_);
v_messages_2416_ = lean_ctor_get(v___x_2415_, 1);
lean_inc_ref(v_messages_2416_);
lean_dec(v___x_2415_);
v___x_2417_ = l_Lean_MessageLog_hasErrors(v_messages_2416_);
lean_dec_ref(v_messages_2416_);
if (v___x_2417_ == 0)
{
lean_object* v___x_2448_; lean_object* v___x_2449_; uint8_t v___x_2450_; 
lean_del_object(v___x_2407_);
v___x_2448_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__5));
lean_inc(v_stx_2400_);
v___x_2449_ = l_Lean_Syntax_getKind(v_stx_2400_);
v___x_2450_ = lp_mathlib_List_elem___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__4(v___x_2449_, v___x_2448_);
lean_dec(v___x_2449_);
if (v___x_2450_ == 0)
{
lean_object* v___x_2451_; 
v___x_2451_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__6));
v_msgs_2419_ = v___x_2451_;
v___y_2420_ = v___y_2401_;
v___y_2421_ = v___y_2402_;
goto v___jp_2418_;
}
else
{
lean_object* v___x_2452_; lean_object* v___y_2454_; lean_object* v___y_2455_; lean_object* v___y_2456_; lean_object* v___y_2459_; lean_object* v___y_2460_; lean_object* v___x_2476_; lean_object* v_a_2477_; lean_object* v___x_2478_; uint8_t v___x_2479_; 
v___x_2452_ = lean_st_ref_get(v___y_2402_);
v___x_2476_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0(v___y_2401_, v___y_2402_);
v_a_2477_ = lean_ctor_get(v___x_2476_, 0);
lean_inc(v_a_2477_);
lean_dec_ref(v___x_2476_);
v___x_2478_ = lp_mathlib_Mathlib_Linter_linter_countHeartbeatsApprox;
v___x_2479_ = l_Lean_Linter_getLinterValue(v___x_2478_, v_a_2477_);
lean_dec(v_a_2477_);
if (v___x_2479_ == 0)
{
lean_object* v___x_2480_; 
v___x_2480_ = l_Lean_Elab_Command_getRef___redArg(v___y_2401_);
if (lean_obj_tag(v___x_2480_) == 0)
{
lean_object* v_a_2481_; lean_object* v___x_2482_; 
v_a_2481_ = lean_ctor_get(v___x_2480_, 0);
lean_inc(v_a_2481_);
lean_dec_ref_known(v___x_2480_, 1);
v___x_2482_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_2401_);
if (lean_obj_tag(v___x_2482_) == 0)
{
lean_object* v_quotContext_x3f_2483_; lean_object* v___x_2484_; 
lean_dec_ref_known(v___x_2482_, 1);
v_quotContext_x3f_2483_ = lean_ctor_get(v___y_2401_, 5);
v___x_2484_ = l_Lean_SourceInfo_fromRef(v_a_2481_, v___x_2479_);
lean_dec(v_a_2481_);
if (lean_obj_tag(v_quotContext_x3f_2483_) == 0)
{
lean_object* v___x_2498_; 
v___x_2498_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(v___y_2402_);
lean_dec_ref(v___x_2498_);
goto v___jp_2485_;
}
else
{
goto v___jp_2485_;
}
v___jp_2485_:
{
lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; 
v___x_2486_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1));
v___x_2487_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__7));
lean_inc_n(v___x_2484_, 4);
v___x_2488_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2488_, 0, v___x_2484_);
lean_ctor_set(v___x_2488_, 1, v___x_2487_);
v___x_2489_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_2490_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_2491_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2491_, 0, v___x_2484_);
lean_ctor_set(v___x_2491_, 1, v___x_2489_);
lean_ctor_set(v___x_2491_, 2, v___x_2490_);
v___x_2492_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11));
v___x_2493_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2493_, 0, v___x_2484_);
lean_ctor_set(v___x_2493_, 1, v___x_2492_);
v___x_2494_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__15));
v___x_2495_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2495_, 0, v___x_2484_);
lean_ctor_set(v___x_2495_, 1, v___x_2494_);
lean_ctor_set(v___x_2495_, 2, v___x_2490_);
lean_inc(v_stx_2400_);
v___x_2496_ = l_Lean_Syntax_node5(v___x_2484_, v___x_2486_, v___x_2488_, v___x_2491_, v___x_2493_, v___x_2495_, v_stx_2400_);
v___x_2497_ = l_Lean_Elab_Command_elabCommand(v___x_2496_, v___y_2401_, v___y_2402_);
if (lean_obj_tag(v___x_2497_) == 0)
{
lean_dec_ref_known(v___x_2497_, 1);
v___y_2459_ = v___y_2401_;
v___y_2460_ = v___y_2402_;
goto v___jp_2458_;
}
else
{
lean_dec(v___x_2452_);
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
return v___x_2497_;
}
}
}
else
{
lean_object* v_a_2499_; lean_object* v___x_2501_; uint8_t v_isShared_2502_; uint8_t v_isSharedCheck_2506_; 
lean_dec(v_a_2481_);
lean_dec(v___x_2452_);
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
v_a_2499_ = lean_ctor_get(v___x_2482_, 0);
v_isSharedCheck_2506_ = !lean_is_exclusive(v___x_2482_);
if (v_isSharedCheck_2506_ == 0)
{
v___x_2501_ = v___x_2482_;
v_isShared_2502_ = v_isSharedCheck_2506_;
goto v_resetjp_2500_;
}
else
{
lean_inc(v_a_2499_);
lean_dec(v___x_2482_);
v___x_2501_ = lean_box(0);
v_isShared_2502_ = v_isSharedCheck_2506_;
goto v_resetjp_2500_;
}
v_resetjp_2500_:
{
lean_object* v___x_2504_; 
if (v_isShared_2502_ == 0)
{
v___x_2504_ = v___x_2501_;
goto v_reusejp_2503_;
}
else
{
lean_object* v_reuseFailAlloc_2505_; 
v_reuseFailAlloc_2505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2505_, 0, v_a_2499_);
v___x_2504_ = v_reuseFailAlloc_2505_;
goto v_reusejp_2503_;
}
v_reusejp_2503_:
{
return v___x_2504_;
}
}
}
}
else
{
lean_object* v_a_2507_; lean_object* v___x_2509_; uint8_t v_isShared_2510_; uint8_t v_isSharedCheck_2514_; 
lean_dec(v___x_2452_);
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
v_a_2507_ = lean_ctor_get(v___x_2480_, 0);
v_isSharedCheck_2514_ = !lean_is_exclusive(v___x_2480_);
if (v_isSharedCheck_2514_ == 0)
{
v___x_2509_ = v___x_2480_;
v_isShared_2510_ = v_isSharedCheck_2514_;
goto v_resetjp_2508_;
}
else
{
lean_inc(v_a_2507_);
lean_dec(v___x_2480_);
v___x_2509_ = lean_box(0);
v_isShared_2510_ = v_isSharedCheck_2514_;
goto v_resetjp_2508_;
}
v_resetjp_2508_:
{
lean_object* v___x_2512_; 
if (v_isShared_2510_ == 0)
{
v___x_2512_ = v___x_2509_;
goto v_reusejp_2511_;
}
else
{
lean_object* v_reuseFailAlloc_2513_; 
v_reuseFailAlloc_2513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2513_, 0, v_a_2507_);
v___x_2512_ = v_reuseFailAlloc_2513_;
goto v_reusejp_2511_;
}
v_reusejp_2511_:
{
return v___x_2512_;
}
}
}
}
else
{
lean_object* v___x_2515_; 
v___x_2515_ = l_Lean_Elab_Command_getRef___redArg(v___y_2401_);
if (lean_obj_tag(v___x_2515_) == 0)
{
lean_object* v_a_2516_; lean_object* v___x_2517_; 
v_a_2516_ = lean_ctor_get(v___x_2515_, 0);
lean_inc(v_a_2516_);
lean_dec_ref_known(v___x_2515_, 1);
v___x_2517_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_2401_);
if (lean_obj_tag(v___x_2517_) == 0)
{
lean_object* v_quotContext_x3f_2518_; lean_object* v___x_2519_; 
lean_dec_ref_known(v___x_2517_, 1);
v_quotContext_x3f_2518_ = lean_ctor_get(v___y_2401_, 5);
v___x_2519_ = l_Lean_SourceInfo_fromRef(v_a_2516_, v___x_2417_);
lean_dec(v_a_2516_);
if (lean_obj_tag(v_quotContext_x3f_2518_) == 0)
{
lean_object* v___x_2535_; 
v___x_2535_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1_spec__3___redArg(v___y_2402_);
lean_dec_ref(v___x_2535_);
goto v___jp_2520_;
}
else
{
goto v___jp_2520_;
}
v___jp_2520_:
{
lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; lean_object* v___x_2534_; 
v___x_2521_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_command_x23count__heartbeatsApproximatelyIn_____00__closed__1));
v___x_2522_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__7));
lean_inc_n(v___x_2519_, 5);
v___x_2523_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2523_, 0, v___x_2519_);
lean_ctor_set(v___x_2523_, 1, v___x_2522_);
v___x_2524_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_2525_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__8));
v___x_2526_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2526_, 0, v___x_2519_);
lean_ctor_set(v___x_2526_, 1, v___x_2525_);
v___x_2527_ = l_Lean_Syntax_node1(v___x_2519_, v___x_2524_, v___x_2526_);
v___x_2528_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__11));
v___x_2529_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2529_, 0, v___x_2519_);
lean_ctor_set(v___x_2529_, 1, v___x_2528_);
v___x_2530_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats_x21__In_____00__closed__15));
v___x_2531_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_2532_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2532_, 0, v___x_2519_);
lean_ctor_set(v___x_2532_, 1, v___x_2530_);
lean_ctor_set(v___x_2532_, 2, v___x_2531_);
lean_inc(v_stx_2400_);
v___x_2533_ = l_Lean_Syntax_node5(v___x_2519_, v___x_2521_, v___x_2523_, v___x_2527_, v___x_2529_, v___x_2532_, v_stx_2400_);
v___x_2534_ = l_Lean_Elab_Command_elabCommand(v___x_2533_, v___y_2401_, v___y_2402_);
if (lean_obj_tag(v___x_2534_) == 0)
{
lean_dec_ref_known(v___x_2534_, 1);
v___y_2459_ = v___y_2401_;
v___y_2460_ = v___y_2402_;
goto v___jp_2458_;
}
else
{
lean_dec(v___x_2452_);
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
return v___x_2534_;
}
}
}
else
{
lean_object* v_a_2536_; lean_object* v___x_2538_; uint8_t v_isShared_2539_; uint8_t v_isSharedCheck_2543_; 
lean_dec(v_a_2516_);
lean_dec(v___x_2452_);
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
v_a_2536_ = lean_ctor_get(v___x_2517_, 0);
v_isSharedCheck_2543_ = !lean_is_exclusive(v___x_2517_);
if (v_isSharedCheck_2543_ == 0)
{
v___x_2538_ = v___x_2517_;
v_isShared_2539_ = v_isSharedCheck_2543_;
goto v_resetjp_2537_;
}
else
{
lean_inc(v_a_2536_);
lean_dec(v___x_2517_);
v___x_2538_ = lean_box(0);
v_isShared_2539_ = v_isSharedCheck_2543_;
goto v_resetjp_2537_;
}
v_resetjp_2537_:
{
lean_object* v___x_2541_; 
if (v_isShared_2539_ == 0)
{
v___x_2541_ = v___x_2538_;
goto v_reusejp_2540_;
}
else
{
lean_object* v_reuseFailAlloc_2542_; 
v_reuseFailAlloc_2542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2542_, 0, v_a_2536_);
v___x_2541_ = v_reuseFailAlloc_2542_;
goto v_reusejp_2540_;
}
v_reusejp_2540_:
{
return v___x_2541_;
}
}
}
}
else
{
lean_object* v_a_2544_; lean_object* v___x_2546_; uint8_t v_isShared_2547_; uint8_t v_isSharedCheck_2551_; 
lean_dec(v___x_2452_);
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
v_a_2544_ = lean_ctor_get(v___x_2515_, 0);
v_isSharedCheck_2551_ = !lean_is_exclusive(v___x_2515_);
if (v_isSharedCheck_2551_ == 0)
{
v___x_2546_ = v___x_2515_;
v_isShared_2547_ = v_isSharedCheck_2551_;
goto v_resetjp_2545_;
}
else
{
lean_inc(v_a_2544_);
lean_dec(v___x_2515_);
v___x_2546_ = lean_box(0);
v_isShared_2547_ = v_isSharedCheck_2551_;
goto v_resetjp_2545_;
}
v_resetjp_2545_:
{
lean_object* v___x_2549_; 
if (v_isShared_2547_ == 0)
{
v___x_2549_ = v___x_2546_;
goto v_reusejp_2548_;
}
else
{
lean_object* v_reuseFailAlloc_2550_; 
v_reuseFailAlloc_2550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2550_, 0, v_a_2544_);
v___x_2549_ = v_reuseFailAlloc_2550_;
goto v_reusejp_2548_;
}
v_reusejp_2548_:
{
return v___x_2549_;
}
}
}
}
v___jp_2453_:
{
lean_object* v___x_2457_; 
v___x_2457_ = lean_st_ref_set(v___y_2455_, v___x_2452_);
v_msgs_2419_ = v___y_2456_;
v___y_2420_ = v___y_2454_;
v___y_2421_ = v___y_2455_;
goto v___jp_2418_;
}
v___jp_2458_:
{
lean_object* v___x_2461_; lean_object* v_messages_2462_; lean_object* v_unreported_2463_; lean_object* v___x_2464_; lean_object* v___x_2465_; lean_object* v___x_2466_; lean_object* v___x_2467_; uint8_t v___x_2468_; 
v___x_2461_ = lean_st_ref_get(v___y_2460_);
v_messages_2462_ = lean_ctor_get(v___x_2461_, 1);
lean_inc_ref(v_messages_2462_);
lean_dec(v___x_2461_);
v_unreported_2463_ = lean_ctor_get(v_messages_2462_, 1);
lean_inc_ref(v_unreported_2463_);
lean_dec_ref(v_messages_2462_);
v___x_2464_ = l_Lean_PersistentArray_toArray___redArg(v_unreported_2463_);
lean_dec_ref(v_unreported_2463_);
v___x_2465_ = lean_unsigned_to_nat(0u);
v___x_2466_ = lean_array_get_size(v___x_2464_);
v___x_2467_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___closed__6));
v___x_2468_ = lean_nat_dec_lt(v___x_2465_, v___x_2466_);
if (v___x_2468_ == 0)
{
lean_dec_ref(v___x_2464_);
v___y_2454_ = v___y_2459_;
v___y_2455_ = v___y_2460_;
v___y_2456_ = v___x_2467_;
goto v___jp_2453_;
}
else
{
uint8_t v___x_2469_; 
v___x_2469_ = lean_nat_dec_le(v___x_2466_, v___x_2466_);
if (v___x_2469_ == 0)
{
if (v___x_2468_ == 0)
{
lean_dec_ref(v___x_2464_);
v___y_2454_ = v___y_2459_;
v___y_2455_ = v___y_2460_;
v___y_2456_ = v___x_2467_;
goto v___jp_2453_;
}
else
{
size_t v___x_2470_; size_t v___x_2471_; lean_object* v___x_2472_; 
v___x_2470_ = ((size_t)0ULL);
v___x_2471_ = lean_usize_of_nat(v___x_2466_);
v___x_2472_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__5(v___x_2464_, v___x_2470_, v___x_2471_, v___x_2467_);
lean_dec_ref(v___x_2464_);
v___y_2454_ = v___y_2459_;
v___y_2455_ = v___y_2460_;
v___y_2456_ = v___x_2472_;
goto v___jp_2453_;
}
}
else
{
size_t v___x_2473_; size_t v___x_2474_; lean_object* v___x_2475_; 
v___x_2473_ = ((size_t)0ULL);
v___x_2474_ = lean_usize_of_nat(v___x_2466_);
v___x_2475_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__5(v___x_2464_, v___x_2473_, v___x_2474_, v___x_2467_);
lean_dec_ref(v___x_2464_);
v___y_2454_ = v___y_2459_;
v___y_2455_ = v___y_2460_;
v___y_2456_ = v___x_2475_;
goto v___jp_2453_;
}
}
}
}
}
else
{
lean_object* v___x_2552_; lean_object* v___x_2554_; 
lean_dec(v_stx_2400_);
lean_dec_ref(v___f_2399_);
v___x_2552_ = lean_box(0);
if (v_isShared_2408_ == 0)
{
lean_ctor_set(v___x_2407_, 0, v___x_2552_);
v___x_2554_ = v___x_2407_;
goto v_reusejp_2553_;
}
else
{
lean_object* v_reuseFailAlloc_2555_; 
v_reuseFailAlloc_2555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2555_, 0, v___x_2552_);
v___x_2554_ = v_reuseFailAlloc_2555_;
goto v_reusejp_2553_;
}
v_reusejp_2553_:
{
return v___x_2554_;
}
}
v___jp_2418_:
{
lean_object* v___x_2422_; 
lean_inc(v_stx_2400_);
v___x_2422_ = l_Lean_Syntax_find_x3f(v_stx_2400_, v___f_2399_);
if (lean_obj_tag(v___x_2422_) == 0)
{
lean_object* v___x_2423_; size_t v_sz_2424_; size_t v___x_2425_; lean_object* v___x_2426_; 
v___x_2423_ = lean_box(0);
v_sz_2424_ = lean_array_size(v_msgs_2419_);
v___x_2425_ = ((size_t)0ULL);
v___x_2426_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__2(v___x_2417_, v_stx_2400_, v_msgs_2419_, v_sz_2424_, v___x_2425_, v___x_2423_, v___y_2420_, v___y_2421_);
lean_dec_ref(v_msgs_2419_);
lean_dec(v_stx_2400_);
if (lean_obj_tag(v___x_2426_) == 0)
{
lean_object* v___x_2428_; uint8_t v_isShared_2429_; uint8_t v_isSharedCheck_2433_; 
v_isSharedCheck_2433_ = !lean_is_exclusive(v___x_2426_);
if (v_isSharedCheck_2433_ == 0)
{
lean_object* v_unused_2434_; 
v_unused_2434_ = lean_ctor_get(v___x_2426_, 0);
lean_dec(v_unused_2434_);
v___x_2428_ = v___x_2426_;
v_isShared_2429_ = v_isSharedCheck_2433_;
goto v_resetjp_2427_;
}
else
{
lean_dec(v___x_2426_);
v___x_2428_ = lean_box(0);
v_isShared_2429_ = v_isSharedCheck_2433_;
goto v_resetjp_2427_;
}
v_resetjp_2427_:
{
lean_object* v___x_2431_; 
if (v_isShared_2429_ == 0)
{
lean_ctor_set(v___x_2428_, 0, v___x_2423_);
v___x_2431_ = v___x_2428_;
goto v_reusejp_2430_;
}
else
{
lean_object* v_reuseFailAlloc_2432_; 
v_reuseFailAlloc_2432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2432_, 0, v___x_2423_);
v___x_2431_ = v_reuseFailAlloc_2432_;
goto v_reusejp_2430_;
}
v_reusejp_2430_:
{
return v___x_2431_;
}
}
}
else
{
return v___x_2426_;
}
}
else
{
lean_object* v_val_2435_; lean_object* v___x_2436_; size_t v_sz_2437_; size_t v___x_2438_; lean_object* v___x_2439_; 
lean_dec(v_stx_2400_);
v_val_2435_ = lean_ctor_get(v___x_2422_, 0);
lean_inc(v_val_2435_);
lean_dec_ref_known(v___x_2422_, 1);
v___x_2436_ = lean_box(0);
v_sz_2437_ = lean_array_size(v_msgs_2419_);
v___x_2438_ = ((size_t)0ULL);
v___x_2439_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__3(v___x_2417_, v_val_2435_, v_msgs_2419_, v_sz_2437_, v___x_2438_, v___x_2436_, v___y_2420_, v___y_2421_);
lean_dec_ref(v_msgs_2419_);
lean_dec(v_val_2435_);
if (lean_obj_tag(v___x_2439_) == 0)
{
lean_object* v___x_2441_; uint8_t v_isShared_2442_; uint8_t v_isSharedCheck_2446_; 
v_isSharedCheck_2446_ = !lean_is_exclusive(v___x_2439_);
if (v_isSharedCheck_2446_ == 0)
{
lean_object* v_unused_2447_; 
v_unused_2447_ = lean_ctor_get(v___x_2439_, 0);
lean_dec(v_unused_2447_);
v___x_2441_ = v___x_2439_;
v_isShared_2442_ = v_isSharedCheck_2446_;
goto v_resetjp_2440_;
}
else
{
lean_dec(v___x_2439_);
v___x_2441_ = lean_box(0);
v_isShared_2442_ = v_isSharedCheck_2446_;
goto v_resetjp_2440_;
}
v_resetjp_2440_:
{
lean_object* v___x_2444_; 
if (v_isShared_2442_ == 0)
{
lean_ctor_set(v___x_2441_, 0, v___x_2436_);
v___x_2444_ = v___x_2441_;
goto v_reusejp_2443_;
}
else
{
lean_object* v_reuseFailAlloc_2445_; 
v_reuseFailAlloc_2445_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2445_, 0, v___x_2436_);
v___x_2444_ = v_reuseFailAlloc_2445_;
goto v_reusejp_2443_;
}
v_reusejp_2443_:
{
return v___x_2444_;
}
}
}
else
{
return v___x_2439_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1___boxed(lean_object* v___f_2557_, lean_object* v_stx_2558_, lean_object* v___y_2559_, lean_object* v___y_2560_, lean_object* v___y_2561_){
_start:
{
lean_object* v_res_2562_; 
v_res_2562_ = lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter___lam__1(v___f_2557_, v_stx_2558_, v___y_2559_, v___y_2560_);
lean_dec(v___y_2560_);
lean_dec_ref(v___y_2559_);
return v_res_2562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0(lean_object* v_o_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_){
_start:
{
lean_object* v___x_2582_; 
v___x_2582_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___redArg(v_o_2578_, v___y_2580_);
return v___x_2582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0___boxed(lean_object* v_o_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_){
_start:
{
lean_object* v_res_2587_; 
v_res_2587_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter_spec__0_spec__0(v_o_2583_, v___y_2584_, v___y_2585_);
lean_dec(v___y_2585_);
lean_dec_ref(v___y_2584_);
return v_res_2587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_CountHeartbeats_initFn_00___x40_Mathlib_Util_CountHeartbeats_347709687____hygCtx___hyg_8_(){
_start:
{
lean_object* v___x_2589_; lean_object* v___x_2590_; 
v___x_2589_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeatsLinter));
v___x_2590_ = l_Lean_Elab_Command_addLinter(v___x_2589_);
return v___x_2590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_CountHeartbeats_initFn_00___x40_Mathlib_Util_CountHeartbeats_347709687____hygCtx___hyg_8____boxed(lean_object* v_a_2591_){
_start:
{
lean_object* v_res_2592_; 
v_res_2592_ = lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_CountHeartbeats_initFn_00___x40_Mathlib_Util_CountHeartbeats_347709687____hygCtx___hyg_8_();
return v_res_2592_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2617_; lean_object* v___x_2618_; 
v___x_2617_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__0));
v___x_2618_ = l_String_toRawSubstring_x27(v___x_2617_);
return v___x_2618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0(lean_object* v___x_2620_, lean_object* v___x_2621_, lean_object* v___x_2622_, lean_object* v_approx_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_){
_start:
{
lean_object* v_quotContext_2626_; lean_object* v_currMacroScope_2627_; lean_object* v_ref_2628_; uint8_t v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; lean_object* v___x_2641_; lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2645_; lean_object* v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; 
v_quotContext_2626_ = lean_ctor_get(v___y_2624_, 1);
v_currMacroScope_2627_ = lean_ctor_get(v___y_2624_, 2);
v_ref_2628_ = lean_ctor_get(v___y_2624_, 5);
v___x_2629_ = 0;
v___x_2630_ = l_Lean_SourceInfo_fromRef(v_ref_2628_, v___x_2629_);
v___x_2631_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1));
v___x_2632_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2));
lean_inc_n(v___x_2630_, 4);
v___x_2633_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2633_, 0, v___x_2630_);
lean_ctor_set(v___x_2633_, 1, v___x_2631_);
v___x_2634_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__1, &lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__1);
v___x_2635_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_));
lean_inc_ref(v___x_2620_);
v___x_2636_ = l_Lean_Name_mkStr2(v___x_2635_, v___x_2620_);
lean_inc(v_currMacroScope_2627_);
lean_inc(v_quotContext_2626_);
v___x_2637_ = l_Lean_addMacroScope(v_quotContext_2626_, v___x_2636_, v_currMacroScope_2627_);
v___x_2638_ = l_Lean_Name_mkStr4(v___x_2621_, v___x_2622_, v___x_2635_, v___x_2620_);
v___x_2639_ = lean_box(0);
v___x_2640_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2640_, 0, v___x_2638_);
lean_ctor_set(v___x_2640_, 1, v___x_2639_);
v___x_2641_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2641_, 0, v___x_2640_);
lean_ctor_set(v___x_2641_, 1, v___x_2639_);
v___x_2642_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2642_, 0, v___x_2630_);
lean_ctor_set(v___x_2642_, 1, v___x_2634_);
lean_ctor_set(v___x_2642_, 2, v___x_2637_);
lean_ctor_set(v___x_2642_, 3, v___x_2641_);
v___x_2643_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_2644_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_2645_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2645_, 0, v___x_2630_);
lean_ctor_set(v___x_2645_, 1, v___x_2643_);
lean_ctor_set(v___x_2645_, 2, v___x_2644_);
v___x_2646_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__2));
v___x_2647_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2647_, 0, v___x_2630_);
lean_ctor_set(v___x_2647_, 1, v___x_2646_);
v___x_2648_ = l_Lean_Syntax_node4(v___x_2630_, v___x_2632_, v___x_2633_, v___x_2642_, v___x_2645_, v___x_2647_);
v___x_2649_ = lean_unsigned_to_nat(2u);
v___x_2650_ = lean_mk_empty_array_with_capacity(v___x_2649_);
v___x_2651_ = lean_array_push(v___x_2650_, v___x_2648_);
v___x_2652_ = lean_array_push(v___x_2651_, v_approx_2623_);
v___x_2653_ = lean_box(2);
v___x_2654_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2654_, 0, v___x_2653_);
lean_ctor_set(v___x_2654_, 1, v___x_2643_);
lean_ctor_set(v___x_2654_, 2, v___x_2652_);
v___x_2655_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2655_, 0, v___x_2654_);
lean_ctor_set(v___x_2655_, 1, v___y_2625_);
return v___x_2655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___boxed(lean_object* v___x_2656_, lean_object* v___x_2657_, lean_object* v___x_2658_, lean_object* v_approx_2659_, lean_object* v___y_2660_, lean_object* v___y_2661_){
_start:
{
lean_object* v_res_2662_; 
v_res_2662_ = lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0(v___x_2656_, v___x_2657_, v___x_2658_, v_approx_2659_, v___y_2660_, v___y_2661_);
lean_dec_ref(v___y_2660_);
return v_res_2662_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1(void){
_start:
{
lean_object* v___x_2664_; lean_object* v___x_2665_; 
v___x_2664_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__0));
v___x_2665_ = l_String_toRawSubstring_x27(v___x_2664_);
return v___x_2665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1(lean_object* v_x_2672_, lean_object* v_a_2673_, lean_object* v_a_2674_){
_start:
{
lean_object* v___y_2676_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; uint8_t v___x_2690_; 
v___x_2686_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats_tactic_x23count__heartbeats___00__closed__0));
v___x_2687_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__10_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_));
v___x_2688_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_));
v___x_2689_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats_countHeartbeats___closed__0));
lean_inc(v_x_2672_);
v___x_2690_ = l_Lean_Syntax_isOfKind(v_x_2672_, v___x_2689_);
if (v___x_2690_ == 0)
{
lean_object* v___x_2691_; lean_object* v___x_2692_; 
lean_dec(v_x_2672_);
v___x_2691_ = lean_box(1);
v___x_2692_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2692_, 0, v___x_2691_);
lean_ctor_set(v___x_2692_, 1, v_a_2674_);
return v___x_2692_;
}
else
{
lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; 
v___x_2714_ = lean_unsigned_to_nat(1u);
v___x_2715_ = l_Lean_Syntax_getArg(v_x_2672_, v___x_2714_);
lean_dec(v_x_2672_);
v___x_2716_ = l_Lean_Syntax_getOptional_x3f(v___x_2715_);
lean_dec(v___x_2715_);
if (lean_obj_tag(v___x_2716_) == 0)
{
goto v___jp_2693_;
}
else
{
lean_dec_ref_known(v___x_2716_, 1);
if (v___x_2690_ == 0)
{
goto v___jp_2693_;
}
else
{
lean_object* v_quotContext_2717_; lean_object* v_currMacroScope_2718_; lean_object* v_ref_2719_; uint8_t v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___x_2725_; lean_object* v___x_2726_; lean_object* v___x_2727_; lean_object* v___x_2728_; lean_object* v___x_2729_; lean_object* v___x_2730_; lean_object* v___x_2731_; lean_object* v___x_2732_; lean_object* v___x_2733_; lean_object* v___x_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; 
v_quotContext_2717_ = lean_ctor_get(v_a_2673_, 1);
v_currMacroScope_2718_ = lean_ctor_get(v_a_2673_, 2);
v_ref_2719_ = lean_ctor_get(v_a_2673_, 5);
v___x_2720_ = 0;
v___x_2721_ = l_Lean_SourceInfo_fromRef(v_ref_2719_, v___x_2720_);
v___x_2722_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1));
v___x_2723_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2));
lean_inc_n(v___x_2721_, 4);
v___x_2724_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2724_, 0, v___x_2721_);
lean_ctor_set(v___x_2724_, 1, v___x_2722_);
v___x_2725_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1, &lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1_once, _init_lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1);
v___x_2726_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_));
lean_inc(v_currMacroScope_2718_);
lean_inc(v_quotContext_2717_);
v___x_2727_ = l_Lean_addMacroScope(v_quotContext_2717_, v___x_2726_, v_currMacroScope_2718_);
v___x_2728_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__3));
v___x_2729_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2729_, 0, v___x_2721_);
lean_ctor_set(v___x_2729_, 1, v___x_2725_);
lean_ctor_set(v___x_2729_, 2, v___x_2727_);
lean_ctor_set(v___x_2729_, 3, v___x_2728_);
v___x_2730_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_2731_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_2732_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2732_, 0, v___x_2721_);
lean_ctor_set(v___x_2732_, 1, v___x_2730_);
lean_ctor_set(v___x_2732_, 2, v___x_2731_);
v___x_2733_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0___closed__2));
v___x_2734_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2734_, 0, v___x_2721_);
lean_ctor_set(v___x_2734_, 1, v___x_2733_);
v___x_2735_ = l_Lean_Syntax_node4(v___x_2721_, v___x_2723_, v___x_2724_, v___x_2729_, v___x_2732_, v___x_2734_);
v___x_2736_ = lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0(v___x_2688_, v___x_2686_, v___x_2687_, v___x_2735_, v_a_2673_, v_a_2674_);
v___y_2676_ = v___x_2736_;
goto v___jp_2675_;
}
}
v___jp_2693_:
{
lean_object* v_quotContext_2694_; lean_object* v_currMacroScope_2695_; lean_object* v_ref_2696_; uint8_t v___x_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; 
v_quotContext_2694_ = lean_ctor_get(v_a_2673_, 1);
v_currMacroScope_2695_ = lean_ctor_get(v_a_2673_, 2);
v_ref_2696_ = lean_ctor_get(v_a_2673_, 5);
v___x_2697_ = 0;
v___x_2698_ = l_Lean_SourceInfo_fromRef(v_ref_2696_, v___x_2697_);
v___x_2699_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__1));
v___x_2700_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__2));
lean_inc_n(v___x_2698_, 4);
v___x_2701_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2701_, 0, v___x_2698_);
lean_ctor_set(v___x_2701_, 1, v___x_2699_);
v___x_2702_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1, &lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1_once, _init_lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__1);
v___x_2703_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_));
lean_inc(v_currMacroScope_2695_);
lean_inc(v_quotContext_2694_);
v___x_2704_ = l_Lean_addMacroScope(v_quotContext_2694_, v___x_2703_, v_currMacroScope_2695_);
v___x_2705_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___closed__3));
v___x_2706_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2706_, 0, v___x_2698_);
lean_ctor_set(v___x_2706_, 1, v___x_2702_);
lean_ctor_set(v___x_2706_, 2, v___x_2704_);
lean_ctor_set(v___x_2706_, 3, v___x_2705_);
v___x_2707_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__6));
v___x_2708_ = lean_obj_once(&lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7, &lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___lam__0___closed__7);
v___x_2709_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2709_, 0, v___x_2698_);
lean_ctor_set(v___x_2709_, 1, v___x_2707_);
lean_ctor_set(v___x_2709_, 2, v___x_2708_);
v___x_2710_ = ((lean_object*)(lp_mathlib_Mathlib_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______elabRules__Mathlib__CountHeartbeats__command_x23count__heartbeatsApproximatelyIn______1___closed__8));
v___x_2711_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2711_, 0, v___x_2698_);
lean_ctor_set(v___x_2711_, 1, v___x_2710_);
v___x_2712_ = l_Lean_Syntax_node4(v___x_2698_, v___x_2700_, v___x_2701_, v___x_2706_, v___x_2709_, v___x_2711_);
v___x_2713_ = lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___lam__0(v___x_2688_, v___x_2686_, v___x_2687_, v___x_2712_, v_a_2673_, v_a_2674_);
v___y_2676_ = v___x_2713_;
goto v___jp_2675_;
}
}
v___jp_2675_:
{
lean_object* v_a_2677_; lean_object* v_a_2678_; lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_2685_; 
v_a_2677_ = lean_ctor_get(v___y_2676_, 0);
v_a_2678_ = lean_ctor_get(v___y_2676_, 1);
v_isSharedCheck_2685_ = !lean_is_exclusive(v___y_2676_);
if (v_isSharedCheck_2685_ == 0)
{
v___x_2680_ = v___y_2676_;
v_isShared_2681_ = v_isSharedCheck_2685_;
goto v_resetjp_2679_;
}
else
{
lean_inc(v_a_2678_);
lean_inc(v_a_2677_);
lean_dec(v___y_2676_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_2685_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v___x_2683_; 
if (v_isShared_2681_ == 0)
{
v___x_2683_ = v___x_2680_;
goto v_reusejp_2682_;
}
else
{
lean_object* v_reuseFailAlloc_2684_; 
v_reuseFailAlloc_2684_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2684_, 0, v_a_2677_);
lean_ctor_set(v_reuseFailAlloc_2684_, 1, v_a_2678_);
v___x_2683_ = v_reuseFailAlloc_2684_;
goto v_reusejp_2682_;
}
v_reusejp_2682_:
{
return v___x_2683_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1___boxed(lean_object* v_x_2737_, lean_object* v_a_2738_, lean_object* v_a_2739_){
_start:
{
lean_object* v_res_2740_; 
v_res_2740_ = lp_mathlib_Mathlib_Linter_CountHeartbeats___aux__Mathlib__Util__CountHeartbeats______macroRules__Mathlib__Linter__CountHeartbeats__countHeartbeats__1(v_x_2737_, v_a_2738_, v_a_2739_);
lean_dec_ref(v_a_2738_);
return v_res_2740_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_CountHeartbeats(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Util_Heartbeats(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_CountHeartbeats(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_Heartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_3857189103____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_countHeartbeats = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_countHeartbeats);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_initFn_00___x40_Mathlib_Util_CountHeartbeats_772767431____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_countHeartbeatsApprox = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_countHeartbeatsApprox);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Util_CountHeartbeats_0__Mathlib_Linter_CountHeartbeats_initFn_00___x40_Mathlib_Util_CountHeartbeats_347709687____hygCtx___hyg_8_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Util_Heartbeats(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_CountHeartbeats(uint8_t builtin) {
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
res = initialize_Lean_Util_Heartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CountHeartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_CountHeartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_CountHeartbeats(builtin);
}
#ifdef __cplusplus
}
#endif
