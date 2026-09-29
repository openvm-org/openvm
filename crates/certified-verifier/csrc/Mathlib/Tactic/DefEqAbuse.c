// Lean compiler output
// Module: Mathlib.Tactic.DefEqAbuse
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Mathlib.Lean.MessageData.Trace
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
uint8_t l_Lean_instBEqTraceResult_beq(uint8_t, uint8_t);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_toString(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_string_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TraceResult_toEmoji(uint8_t);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_String_Slice_subslice_x21(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Array_append___redArg___boxed(lean_object*, lean_object*);
uint8_t lean_string_get_byte_fast(lean_object*, lean_object*);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_String_Slice_posGE___redArg(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lp_mathlib_Lean_MessageData_extractInstName(lean_object*);
lean_object* lp_mathlib_Lean_MessageData_dedupByString(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
extern lean_object* l_Lean_diagnostics;
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_MessageLog_toList(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_withScope___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_logWarning___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_descend_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_descend_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ascend_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ascend_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__12;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "choice"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(59, 66, 148, 42, 181, 100, 85, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "term{}"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(44, 141, 217, 101, 193, 131, 35, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__19;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__21_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__23;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__24;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__25;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "structInst"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(50, 43, 73, 62, 118, 124, 31, 28)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__29_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__30;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "structInstFields"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(0, 82, 141, 43, 62, 171, 163, 69)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__34;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__35;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "optEllipsis"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__36 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__36_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(13, 1, 242, 203, 207, 188, 181, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__38;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__39;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__40;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__41;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__42;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__43;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__44;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__45;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__46;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__47;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__48;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__49;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__50;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__51;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__7;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__9_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__11_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__14;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__15;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__16_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "[anonymous]"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__19;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__20;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__21;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__22;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__24;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__25;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_++_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__27_value),LEAN_SCALAR_PTR_LITERAL(90, 69, 86, 178, 149, 48, 216, 23)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__28_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__29_value),LEAN_SCALAR_PTR_LITERAL(215, 94, 65, 66, 49, 100, 151, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__31_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__32;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__33;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__34;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "++"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__37_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__38;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__39;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__40;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__41;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__42;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__43_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__45;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__46;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__47;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__48;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__49;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__50;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__51;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__52;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__53;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__54;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__55;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__56;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__57;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∪_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__58 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__58_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__58_value),LEAN_SCALAR_PTR_LITERAL(202, 164, 141, 67, 105, 98, 49, 125)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__59 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__59_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∪"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__60 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__60_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__61;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__62;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__63;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__64;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__65;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__66;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__67;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__68_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__68;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__69;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__70;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__71_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__71;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__72_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__72;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__73;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__74_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__74;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__75;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__76;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__77_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__77;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__78_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__78;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__79_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__79;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__80_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__80;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__81_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__81;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__82_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__82;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__83_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__83;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__84_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__84;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__85_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__85;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__86_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__86;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___lam__0(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "isDefEq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "onFailure"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 173, 228, 229, 125, 117, 225, 10)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(167, 117, 242, 50, 155, 140, 245, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 173, 228, 229, 125, 117, 225, 10)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_append___redArg___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__0_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "apply "};
static const lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__5;
static const lean_ctor_object lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__6 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___boxed(lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "synthInstance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(183, 134, 217, 158, 109, 129, 245, 185)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_append___redArg___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__0_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__5;
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___boxed(lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__1(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__4(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__0;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3___closed__0 = (const lean_object*)&lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " =\?= "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "universes"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(79, 49, 200, 238, 5, 247, 132, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 109, 223, 122, 147, 21, 229, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___closed__3_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "#defeq_abuse: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 147, .m_capacity = 147, .m_length = 146, .m_data = " fails with `backward.isDefEq.respectTransparency true` but succeeds with `false`.\nThe following synthesis applications fail due to transparency:\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "    "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "  "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__3(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 149, .m_capacity = 149, .m_length = 148, .m_data = " fails with `backward.isDefEq.respectTransparency true` but succeeds with `false`.\nThe following isDefEq checks are the root causes of the failure:\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 147, .m_capacity = 147, .m_length = 146, .m_data = " fails with `backward.isDefEq.respectTransparency true` but succeeds with `false`.\nCould not identify specific failing isDefEq checks from traces."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "DefEqAbuse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "defeqAbuse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__1_value),LEAN_SCALAR_PTR_LITERAL(165, 139, 89, 183, 212, 92, 213, 237)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__2_value),LEAN_SCALAR_PTR_LITERAL(13, 59, 143, 10, 25, 130, 240, 26)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "#defeq_abuse "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "in "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__11_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__15_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "backward"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "respectTransparency"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(77, 196, 98, 49, 58, 220, 29, 220)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(36, 118, 4, 150, 194, 42, 143, 196)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(186, 186, 50, 40, 52, 56, 153, 40)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(200, 160, 146, 56, 76, 73, 209, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(229, 58, 150, 125, 47, 35, 93, 14)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__1;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__0;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 89, .m_capacity = 89, .m_length = 88, .m_data = "#defeq_abuse: tactic fails regardless of `backward.isDefEq.respectTransparency` setting."};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 99, .m_capacity = 99, .m_length = 98, .m_data = "#defeq_abuse: tactic succeeds with `backward.isDefEq.respectTransparency true`. No abuse detected."};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "defeqAbuseCmd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__1_value),LEAN_SCALAR_PTR_LITERAL(165, 139, 89, 183, 212, 92, 213, 237)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(21, 92, 102, 121, 223, 229, 241, 192)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "in"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__6_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "async"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(13, 84, 199, 228, 250, 36, 60, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 0, 36, 68, 138, 2, 151, 20)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(200, 160, 146, 56, 76, 73, 209, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(0, 98, 228, 43, 93, 17, 45, 215)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "command produced errors"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__3(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__3___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 89, .m_data = "#defeq_abuse: command fails regardless of `backward.isDefEq.respectTransparency` setting."};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__2;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__3___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 100, .m_capacity = 100, .m_length = 99, .m_data = "#defeq_abuse: command succeeds with `backward.isDefEq.respectTransparency true`. No abuse detected."};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___redArg(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___redArg___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___redArg(v_x_4_);
lean_dec_ref(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx(lean_object* v_00_u03b1_6_, lean_object* v_x_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___redArg(v_x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx___boxed(lean_object* v_00_u03b1_9_, lean_object* v_x_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorIdx(v_00_u03b1_9_, v_x_10_);
lean_dec_ref(v_x_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___redArg(lean_object* v_t_12_, lean_object* v_k_13_){
_start:
{
lean_object* v_butFirst_14_; lean_object* v___x_15_; 
v_butFirst_14_ = lean_ctor_get(v_t_12_, 0);
lean_inc(v_butFirst_14_);
lean_dec_ref(v_t_12_);
v___x_15_ = lean_apply_1(v_k_13_, v_butFirst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim(lean_object* v_00_u03b1_16_, lean_object* v_motive_17_, lean_object* v_ctorIdx_18_, lean_object* v_t_19_, lean_object* v_h_20_, lean_object* v_k_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___redArg(v_t_19_, v_k_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___boxed(lean_object* v_00_u03b1_23_, lean_object* v_motive_24_, lean_object* v_ctorIdx_25_, lean_object* v_t_26_, lean_object* v_h_27_, lean_object* v_k_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim(v_00_u03b1_23_, v_motive_24_, v_ctorIdx_25_, v_t_26_, v_h_27_, v_k_28_);
lean_dec(v_ctorIdx_25_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_descend_elim___redArg(lean_object* v_t_30_, lean_object* v_descend_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___redArg(v_t_30_, v_descend_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_descend_elim(lean_object* v_00_u03b1_33_, lean_object* v_motive_34_, lean_object* v_t_35_, lean_object* v_h_36_, lean_object* v_descend_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___redArg(v_t_35_, v_descend_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ascend_elim___redArg(lean_object* v_t_39_, lean_object* v_ascend_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___redArg(v_t_39_, v_ascend_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ascend_elim(lean_object* v_00_u03b1_42_, lean_object* v_motive_43_, lean_object* v_t_44_, lean_object* v_h_45_, lean_object* v_ascend_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_VisitStep_ctorElim___redArg(v_t_44_, v_ascend_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__12(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_74_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__10));
v___x_75_ = l_Lean_mkAtom(v___x_74_);
return v___x_75_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_76_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__12, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__12);
v___x_77_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_78_ = lean_array_push(v___x_77_, v___x_76_);
return v___x_78_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__19(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__18));
v___x_87_ = l_Lean_mkAtom(v___x_86_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_88_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__19, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__19_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__19);
v___x_89_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_90_ = lean_array_push(v___x_89_, v___x_88_);
return v___x_90_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__21));
v___x_93_ = l_Lean_mkAtom(v___x_92_);
return v___x_93_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__23(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_94_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22);
v___x_95_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20);
v___x_96_ = lean_array_push(v___x_95_, v___x_94_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__24(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_97_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__23, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__23_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__23);
v___x_98_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__17));
v___x_99_ = lean_box(2);
v___x_100_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v___x_98_);
lean_ctor_set(v___x_100_, 2, v___x_97_);
return v___x_100_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__25(void){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_101_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__24, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__24_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__24);
v___x_102_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_103_ = lean_array_push(v___x_102_, v___x_101_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__30(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_115_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__29));
v___x_116_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__20);
v___x_117_ = lean_array_push(v___x_116_, v___x_115_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__29));
v___x_125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_126_ = lean_array_push(v___x_125_, v___x_124_);
return v___x_126_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__34(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_127_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33);
v___x_128_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__32));
v___x_129_ = lean_box(2);
v___x_130_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v___x_128_);
lean_ctor_set(v___x_130_, 2, v___x_127_);
return v___x_130_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__35(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__34, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__34_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__34);
v___x_132_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__30, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__30_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__30);
v___x_133_ = lean_array_push(v___x_132_, v___x_131_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__38(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_140_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__33);
v___x_141_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__37));
v___x_142_ = lean_box(2);
v___x_143_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
lean_ctor_set(v___x_143_, 1, v___x_141_);
lean_ctor_set(v___x_143_, 2, v___x_140_);
return v___x_143_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__39(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_144_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__38, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__38_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__38);
v___x_145_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__35, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__35_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__35);
v___x_146_ = lean_array_push(v___x_145_, v___x_144_);
return v___x_146_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__40(void){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_147_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__29));
v___x_148_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__39, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__39_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__39);
v___x_149_ = lean_array_push(v___x_148_, v___x_147_);
return v___x_149_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__41(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_150_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__22);
v___x_151_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__40, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__40_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__40);
v___x_152_ = lean_array_push(v___x_151_, v___x_150_);
return v___x_152_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__42(void){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_153_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__41, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__41_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__41);
v___x_154_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__28));
v___x_155_ = lean_box(2);
v___x_156_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
lean_ctor_set(v___x_156_, 1, v___x_154_);
lean_ctor_set(v___x_156_, 2, v___x_153_);
return v___x_156_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__43(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_157_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__42, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__42_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__42);
v___x_158_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__25, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__25_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__25);
v___x_159_ = lean_array_push(v___x_158_, v___x_157_);
return v___x_159_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__44(void){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_160_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__43, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__43_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__43);
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__15));
v___x_162_ = lean_box(2);
v___x_163_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v___x_161_);
lean_ctor_set(v___x_163_, 2, v___x_160_);
return v___x_163_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__45(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_164_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__44, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__44_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__44);
v___x_165_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13);
v___x_166_ = lean_array_push(v___x_165_, v___x_164_);
return v___x_166_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__46(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_167_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__45, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__45_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__45);
v___x_168_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11));
v___x_169_ = lean_box(2);
v___x_170_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v___x_168_);
lean_ctor_set(v___x_170_, 2, v___x_167_);
return v___x_170_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__47(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_171_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__46, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__46_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__46);
v___x_172_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_173_ = lean_array_push(v___x_172_, v___x_171_);
return v___x_173_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__48(void){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_174_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__47, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__47_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__47);
v___x_175_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9));
v___x_176_ = lean_box(2);
v___x_177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v___x_175_);
lean_ctor_set(v___x_177_, 2, v___x_174_);
return v___x_177_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__49(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_178_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__48, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__48_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__48);
v___x_179_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_180_ = lean_array_push(v___x_179_, v___x_178_);
return v___x_180_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__50(void){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_181_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__49, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__49_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__49);
v___x_182_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7));
v___x_183_ = lean_box(2);
v___x_184_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v___x_182_);
lean_ctor_set(v___x_184_, 2, v___x_181_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__51(void){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_185_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__50, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__50_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__50);
v___x_186_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_187_ = lean_array_push(v___x_186_, v___x_185_);
return v___x_187_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_188_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__51, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__51_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__51);
v___x_189_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4));
v___x_190_ = lean_box(2);
v___x_191_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v___x_189_);
lean_ctor_set(v___x_191_, 2, v___x_188_);
return v___x_191_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1(void){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52);
return v___x_192_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__2(void){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__0));
v___x_200_ = l_Lean_mkAtom(v___x_199_);
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__3(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_201_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__2);
v___x_202_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_203_ = lean_array_push(v___x_202_, v___x_201_);
return v___x_203_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__7(void){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_208_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__6));
v___x_209_ = l_Lean_mkAtom(v___x_208_);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_210_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__7, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__7);
v___x_211_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_212_ = lean_array_push(v___x_211_, v___x_210_);
return v___x_212_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__14(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_226_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__13));
v___x_227_ = l_Lean_mkAtom(v___x_226_);
return v___x_227_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__15(void){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_228_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__14, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__14);
v___x_229_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_230_ = lean_array_push(v___x_229_, v___x_228_);
return v___x_230_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__19(void){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_235_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__18));
v___x_236_ = lean_string_utf8_byte_size(v___x_235_);
return v___x_236_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__20(void){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_237_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__19, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__19_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__19);
v___x_238_ = lean_unsigned_to_nat(0u);
v___x_239_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__18));
v___x_240_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
lean_ctor_set(v___x_240_, 1, v___x_238_);
lean_ctor_set(v___x_240_, 2, v___x_237_);
return v___x_240_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__21(void){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_241_ = lean_box(0);
v___x_242_ = lean_box(0);
v___x_243_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__20, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__20);
v___x_244_ = lean_box(2);
v___x_245_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
lean_ctor_set(v___x_245_, 1, v___x_243_);
lean_ctor_set(v___x_245_, 2, v___x_242_);
lean_ctor_set(v___x_245_, 3, v___x_241_);
return v___x_245_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__22(void){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_246_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__21, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__21_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__21);
v___x_247_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_248_ = lean_array_push(v___x_247_, v___x_246_);
return v___x_248_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23(void){
_start:
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_249_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__22, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__22_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__22);
v___x_250_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__17));
v___x_251_ = lean_box(2);
v___x_252_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v___x_250_);
lean_ctor_set(v___x_252_, 2, v___x_249_);
return v___x_252_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__24(void){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_253_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23);
v___x_254_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__15, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__15);
v___x_255_ = lean_array_push(v___x_254_, v___x_253_);
return v___x_255_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__25(void){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_256_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__24, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__24_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__24);
v___x_257_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__12));
v___x_258_ = lean_box(2);
v___x_259_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_259_, 0, v___x_258_);
lean_ctor_set(v___x_259_, 1, v___x_257_);
lean_ctor_set(v___x_259_, 2, v___x_256_);
return v___x_259_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_260_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__25, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__25_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__25);
v___x_261_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_262_ = lean_array_push(v___x_261_, v___x_260_);
return v___x_262_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__32(void){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__31));
v___x_274_ = l_Lean_mkAtom(v___x_273_);
return v___x_274_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__33(void){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_275_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__32, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__32_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__32);
v___x_276_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_277_ = lean_array_push(v___x_276_, v___x_275_);
return v___x_277_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__34(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_278_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__23);
v___x_279_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__33, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__33_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__33);
v___x_280_ = lean_array_push(v___x_279_, v___x_278_);
return v___x_280_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35(void){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_281_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__34, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__34_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__34);
v___x_282_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__30));
v___x_283_ = lean_box(2);
v___x_284_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
lean_ctor_set(v___x_284_, 1, v___x_282_);
lean_ctor_set(v___x_284_, 2, v___x_281_);
return v___x_284_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_285_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35);
v___x_286_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_287_ = lean_array_push(v___x_286_, v___x_285_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__38(void){
_start:
{
lean_object* v___x_289_; lean_object* v___x_290_; 
v___x_289_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__37));
v___x_290_ = l_Lean_mkAtom(v___x_289_);
return v___x_290_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__39(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_291_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__38, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__38_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__38);
v___x_292_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36);
v___x_293_ = lean_array_push(v___x_292_, v___x_291_);
return v___x_293_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__40(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_294_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35);
v___x_295_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__39, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__39_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__39);
v___x_296_ = lean_array_push(v___x_295_, v___x_294_);
return v___x_296_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__41(void){
_start:
{
lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_297_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__40, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__40_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__40);
v___x_298_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__28));
v___x_299_ = lean_box(2);
v___x_300_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
lean_ctor_set(v___x_300_, 1, v___x_298_);
lean_ctor_set(v___x_300_, 2, v___x_297_);
return v___x_300_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__42(void){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
v___x_301_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__41, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__41_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__41);
v___x_302_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26);
v___x_303_ = lean_array_push(v___x_302_, v___x_301_);
return v___x_303_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44(void){
_start:
{
lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_305_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__43));
v___x_306_ = l_Lean_mkAtom(v___x_305_);
return v___x_306_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__45(void){
_start:
{
lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_307_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44);
v___x_308_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__42, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__42_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__42);
v___x_309_ = lean_array_push(v___x_308_, v___x_307_);
return v___x_309_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__46(void){
_start:
{
lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_310_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__45, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__45_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__45);
v___x_311_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10));
v___x_312_ = lean_box(2);
v___x_313_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
lean_ctor_set(v___x_313_, 1, v___x_311_);
lean_ctor_set(v___x_313_, 2, v___x_310_);
return v___x_313_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__47(void){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_314_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__46, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__46_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__46);
v___x_315_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13);
v___x_316_ = lean_array_push(v___x_315_, v___x_314_);
return v___x_316_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__48(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; 
v___x_317_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__47, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__47_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__47);
v___x_318_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11));
v___x_319_ = lean_box(2);
v___x_320_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
lean_ctor_set(v___x_320_, 1, v___x_318_);
lean_ctor_set(v___x_320_, 2, v___x_317_);
return v___x_320_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__49(void){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
v___x_321_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__48, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__48_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__48);
v___x_322_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_323_ = lean_array_push(v___x_322_, v___x_321_);
return v___x_323_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__50(void){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_324_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__49, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__49_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__49);
v___x_325_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9));
v___x_326_ = lean_box(2);
v___x_327_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_327_, 0, v___x_326_);
lean_ctor_set(v___x_327_, 1, v___x_325_);
lean_ctor_set(v___x_327_, 2, v___x_324_);
return v___x_327_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__51(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__50, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__50_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__50);
v___x_329_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_330_ = lean_array_push(v___x_329_, v___x_328_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__52(void){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; 
v___x_331_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__51, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__51_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__51);
v___x_332_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7));
v___x_333_ = lean_box(2);
v___x_334_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_334_, 0, v___x_333_);
lean_ctor_set(v___x_334_, 1, v___x_332_);
lean_ctor_set(v___x_334_, 2, v___x_331_);
return v___x_334_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__53(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_335_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__52, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__52_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__52);
v___x_336_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_337_ = lean_array_push(v___x_336_, v___x_335_);
return v___x_337_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__54(void){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; 
v___x_338_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__53, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__53_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__53);
v___x_339_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4));
v___x_340_ = lean_box(2);
v___x_341_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_341_, 0, v___x_340_);
lean_ctor_set(v___x_341_, 1, v___x_339_);
lean_ctor_set(v___x_341_, 2, v___x_338_);
return v___x_341_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__55(void){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_342_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__54, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__54_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__54);
v___x_343_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8);
v___x_344_ = lean_array_push(v___x_343_, v___x_342_);
return v___x_344_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__56(void){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_345_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__55, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__55_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__55);
v___x_346_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__5));
v___x_347_ = lean_box(2);
v___x_348_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v___x_346_);
lean_ctor_set(v___x_348_, 2, v___x_345_);
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__57(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_349_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__56, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__56_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__56);
v___x_350_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_351_ = lean_array_push(v___x_350_, v___x_349_);
return v___x_351_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__61(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_356_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__60));
v___x_357_ = l_Lean_mkAtom(v___x_356_);
return v___x_357_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__62(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_358_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__61, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__61_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__61);
v___x_359_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__36);
v___x_360_ = lean_array_push(v___x_359_, v___x_358_);
return v___x_360_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__63(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_361_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__35);
v___x_362_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__62, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__62_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__62);
v___x_363_ = lean_array_push(v___x_362_, v___x_361_);
return v___x_363_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__64(void){
_start:
{
lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_364_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__63, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__63_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__63);
v___x_365_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__59));
v___x_366_ = lean_box(2);
v___x_367_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_367_, 0, v___x_366_);
lean_ctor_set(v___x_367_, 1, v___x_365_);
lean_ctor_set(v___x_367_, 2, v___x_364_);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__65(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_368_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__64, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__64_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__64);
v___x_369_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__26);
v___x_370_ = lean_array_push(v___x_369_, v___x_368_);
return v___x_370_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__66(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_371_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__44);
v___x_372_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__65, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__65_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__65);
v___x_373_ = lean_array_push(v___x_372_, v___x_371_);
return v___x_373_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__67(void){
_start:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_374_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__66, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__66_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__66);
v___x_375_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__10));
v___x_376_ = lean_box(2);
v___x_377_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
lean_ctor_set(v___x_377_, 1, v___x_375_);
lean_ctor_set(v___x_377_, 2, v___x_374_);
return v___x_377_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__68(void){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_378_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__67, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__67_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__67);
v___x_379_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__13);
v___x_380_ = lean_array_push(v___x_379_, v___x_378_);
return v___x_380_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__69(void){
_start:
{
lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_381_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__68, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__68_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__68);
v___x_382_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__11));
v___x_383_ = lean_box(2);
v___x_384_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_384_, 0, v___x_383_);
lean_ctor_set(v___x_384_, 1, v___x_382_);
lean_ctor_set(v___x_384_, 2, v___x_381_);
return v___x_384_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__70(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v___x_385_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__69, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__69_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__69);
v___x_386_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_387_ = lean_array_push(v___x_386_, v___x_385_);
return v___x_387_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__71(void){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_388_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__70, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__70_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__70);
v___x_389_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9));
v___x_390_ = lean_box(2);
v___x_391_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_391_, 0, v___x_390_);
lean_ctor_set(v___x_391_, 1, v___x_389_);
lean_ctor_set(v___x_391_, 2, v___x_388_);
return v___x_391_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__72(void){
_start:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_392_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__71, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__71_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__71);
v___x_393_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_394_ = lean_array_push(v___x_393_, v___x_392_);
return v___x_394_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__73(void){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
v___x_395_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__72, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__72_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__72);
v___x_396_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7));
v___x_397_ = lean_box(2);
v___x_398_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
lean_ctor_set(v___x_398_, 1, v___x_396_);
lean_ctor_set(v___x_398_, 2, v___x_395_);
return v___x_398_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__74(void){
_start:
{
lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; 
v___x_399_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__73, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__73_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__73);
v___x_400_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_401_ = lean_array_push(v___x_400_, v___x_399_);
return v___x_401_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__75(void){
_start:
{
lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; 
v___x_402_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__74, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__74_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__74);
v___x_403_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4));
v___x_404_ = lean_box(2);
v___x_405_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_405_, 0, v___x_404_);
lean_ctor_set(v___x_405_, 1, v___x_403_);
lean_ctor_set(v___x_405_, 2, v___x_402_);
return v___x_405_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__76(void){
_start:
{
lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; 
v___x_406_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__75, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__75_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__75);
v___x_407_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__8);
v___x_408_ = lean_array_push(v___x_407_, v___x_406_);
return v___x_408_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__77(void){
_start:
{
lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_409_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__76, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__76_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__76);
v___x_410_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__5));
v___x_411_ = lean_box(2);
v___x_412_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_412_, 0, v___x_411_);
lean_ctor_set(v___x_412_, 1, v___x_410_);
lean_ctor_set(v___x_412_, 2, v___x_409_);
return v___x_412_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__78(void){
_start:
{
lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_413_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__77, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__77_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__77);
v___x_414_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__57, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__57_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__57);
v___x_415_ = lean_array_push(v___x_414_, v___x_413_);
return v___x_415_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__79(void){
_start:
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_416_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__78, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__78_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__78);
v___x_417_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9));
v___x_418_ = lean_box(2);
v___x_419_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
lean_ctor_set(v___x_419_, 1, v___x_417_);
lean_ctor_set(v___x_419_, 2, v___x_416_);
return v___x_419_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__80(void){
_start:
{
lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_420_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__79, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__79_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__79);
v___x_421_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__3, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__3);
v___x_422_ = lean_array_push(v___x_421_, v___x_420_);
return v___x_422_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__81(void){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v___x_423_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__80, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__80_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__80);
v___x_424_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__1));
v___x_425_ = lean_box(2);
v___x_426_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_426_, 0, v___x_425_);
lean_ctor_set(v___x_426_, 1, v___x_424_);
lean_ctor_set(v___x_426_, 2, v___x_423_);
return v___x_426_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__82(void){
_start:
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; 
v___x_427_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__81, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__81_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__81);
v___x_428_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_429_ = lean_array_push(v___x_428_, v___x_427_);
return v___x_429_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__83(void){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_430_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__82, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__82_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__82);
v___x_431_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__9));
v___x_432_ = lean_box(2);
v___x_433_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
lean_ctor_set(v___x_433_, 1, v___x_431_);
lean_ctor_set(v___x_433_, 2, v___x_430_);
return v___x_433_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__84(void){
_start:
{
lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; 
v___x_434_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__83, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__83_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__83);
v___x_435_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_436_ = lean_array_push(v___x_435_, v___x_434_);
return v___x_436_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__85(void){
_start:
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v___x_437_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__84, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__84_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__84);
v___x_438_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__7));
v___x_439_ = lean_box(2);
v___x_440_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_440_, 0, v___x_439_);
lean_ctor_set(v___x_440_, 1, v___x_438_);
lean_ctor_set(v___x_440_, 2, v___x_437_);
return v___x_440_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__86(void){
_start:
{
lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_441_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__85, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__85_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__85);
v___x_442_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__5));
v___x_443_ = lean_array_push(v___x_442_, v___x_441_);
return v___x_443_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87(void){
_start:
{
lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_444_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__86, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__86_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__86);
v___x_445_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__4));
v___x_446_ = lean_box(2);
v___x_447_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_447_, 0, v___x_446_);
lean_ctor_set(v___x_447_, 1, v___x_445_);
lean_ctor_set(v___x_447_, 2, v___x_444_);
return v___x_447_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3(void){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__0(lean_object* v_combine_449_, lean_object* v_____do__lift_450_, lean_object* v_toPure_451_, lean_object* v_____do__lift_452_){
_start:
{
lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_453_ = lean_apply_2(v_combine_449_, v_____do__lift_450_, v_____do__lift_452_);
v___x_454_ = lean_apply_2(v_toPure_451_, lean_box(0), v___x_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__3(lean_object* v_combine_455_, lean_object* v___y_456_, lean_object* v_toPure_457_, lean_object* v_____do__lift_458_){
_start:
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
v___x_459_ = lean_apply_2(v_combine_455_, v___y_456_, v_____do__lift_458_);
v___x_460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
v___x_461_ = lean_apply_2(v_toPure_457_, lean_box(0), v___x_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__2(lean_object* v_toPure_462_, lean_object* v_____s_463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lean_apply_2(v_toPure_462_, lean_box(0), v_____s_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__5(lean_object* v_children_465_, lean_object* v_inst_466_, lean_object* v___f_467_, lean_object* v_toBind_468_, lean_object* v___f_469_, lean_object* v_empty_470_, lean_object* v_toPure_471_, lean_object* v_____do__lift_472_){
_start:
{
lean_object* v___y_474_; 
if (lean_obj_tag(v_____do__lift_472_) == 0)
{
lean_object* v_butFirst_479_; 
lean_dec(v_toPure_471_);
v_butFirst_479_ = lean_ctor_get(v_____do__lift_472_, 0);
lean_inc(v_butFirst_479_);
lean_dec_ref_known(v_____do__lift_472_, 1);
if (lean_obj_tag(v_butFirst_479_) == 0)
{
v___y_474_ = v_empty_470_;
goto v___jp_473_;
}
else
{
lean_object* v_val_480_; 
lean_dec(v_empty_470_);
v_val_480_ = lean_ctor_get(v_butFirst_479_, 0);
lean_inc(v_val_480_);
lean_dec_ref_known(v_butFirst_479_, 1);
v___y_474_ = v_val_480_;
goto v___jp_473_;
}
}
else
{
lean_object* v_returning_481_; 
lean_dec(v___f_469_);
lean_dec(v_toBind_468_);
lean_dec(v___f_467_);
lean_dec_ref(v_inst_466_);
lean_dec_ref(v_children_465_);
v_returning_481_ = lean_ctor_get(v_____do__lift_472_, 0);
lean_inc(v_returning_481_);
lean_dec_ref_known(v_____do__lift_472_, 1);
if (lean_obj_tag(v_returning_481_) == 0)
{
lean_object* v___x_482_; 
v___x_482_ = lean_apply_2(v_toPure_471_, lean_box(0), v_empty_470_);
return v___x_482_;
}
else
{
lean_object* v_val_483_; lean_object* v___x_484_; 
lean_dec(v_empty_470_);
v_val_483_ = lean_ctor_get(v_returning_481_, 0);
lean_inc(v_val_483_);
lean_dec_ref_known(v_returning_481_, 1);
v___x_484_ = lean_apply_2(v_toPure_471_, lean_box(0), v_val_483_);
return v___x_484_;
}
}
v___jp_473_:
{
size_t v_sz_475_; size_t v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; 
v_sz_475_ = lean_array_size(v_children_465_);
v___x_476_ = ((size_t)0ULL);
v___x_477_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_466_, v_children_465_, v___f_467_, v_sz_475_, v___x_476_, v___y_474_);
v___x_478_ = lean_apply_4(v_toBind_468_, lean_box(0), lean_box(0), v___x_477_, v___f_469_);
return v___x_478_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__4(lean_object* v_combine_485_, lean_object* v_toPure_486_, lean_object* v_inst_487_, lean_object* v_onTrace_488_, lean_object* v_empty_489_, lean_object* v_toBind_490_, lean_object* v_a_491_, lean_object* v_x_492_, lean_object* v___y_493_){
_start:
{
lean_object* v___f_494_; lean_object* v___x_495_; lean_object* v___x_496_; 
lean_inc(v_combine_485_);
v___f_494_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__3), 4, 3);
lean_closure_set(v___f_494_, 0, v_combine_485_);
lean_closure_set(v___f_494_, 1, v___y_493_);
lean_closure_set(v___f_494_, 2, v_toPure_486_);
v___x_495_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(v_inst_487_, v_onTrace_488_, v_empty_489_, v_combine_485_, v_a_491_);
v___x_496_ = lean_apply_4(v_toBind_490_, lean_box(0), lean_box(0), v___x_495_, v___f_494_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(lean_object* v_inst_497_, lean_object* v_onTrace_498_, lean_object* v_empty_499_, lean_object* v_combine_500_, lean_object* v_a_501_){
_start:
{
switch(lean_obj_tag(v_a_501_))
{
case 3:
{
lean_object* v_a_502_; 
v_a_502_ = lean_ctor_get(v_a_501_, 1);
lean_inc_ref(v_a_502_);
lean_dec_ref_known(v_a_501_, 2);
v_a_501_ = v_a_502_;
goto _start;
}
case 4:
{
lean_object* v_a_504_; 
v_a_504_ = lean_ctor_get(v_a_501_, 1);
lean_inc_ref(v_a_504_);
lean_dec_ref_known(v_a_501_, 2);
v_a_501_ = v_a_504_;
goto _start;
}
case 5:
{
lean_object* v_a_506_; 
v_a_506_ = lean_ctor_get(v_a_501_, 1);
lean_inc_ref(v_a_506_);
lean_dec_ref_known(v_a_501_, 2);
v_a_501_ = v_a_506_;
goto _start;
}
case 6:
{
lean_object* v_a_508_; 
v_a_508_ = lean_ctor_get(v_a_501_, 0);
lean_inc_ref(v_a_508_);
lean_dec_ref_known(v_a_501_, 1);
v_a_501_ = v_a_508_;
goto _start;
}
case 7:
{
lean_object* v_toApplicative_510_; lean_object* v_toBind_511_; lean_object* v_toPure_512_; lean_object* v_a_513_; lean_object* v_a_514_; lean_object* v___f_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v_toApplicative_510_ = lean_ctor_get(v_inst_497_, 0);
v_toBind_511_ = lean_ctor_get(v_inst_497_, 1);
lean_inc_n(v_toBind_511_, 2);
v_toPure_512_ = lean_ctor_get(v_toApplicative_510_, 1);
v_a_513_ = lean_ctor_get(v_a_501_, 0);
lean_inc_ref(v_a_513_);
v_a_514_ = lean_ctor_get(v_a_501_, 1);
lean_inc_ref(v_a_514_);
lean_dec_ref_known(v_a_501_, 2);
lean_inc(v_empty_499_);
lean_inc(v_onTrace_498_);
lean_inc_ref(v_inst_497_);
lean_inc(v_toPure_512_);
lean_inc(v_combine_500_);
v___f_515_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__1), 8, 7);
lean_closure_set(v___f_515_, 0, v_combine_500_);
lean_closure_set(v___f_515_, 1, v_toPure_512_);
lean_closure_set(v___f_515_, 2, v_inst_497_);
lean_closure_set(v___f_515_, 3, v_onTrace_498_);
lean_closure_set(v___f_515_, 4, v_empty_499_);
lean_closure_set(v___f_515_, 5, v_a_514_);
lean_closure_set(v___f_515_, 6, v_toBind_511_);
v___x_516_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(v_inst_497_, v_onTrace_498_, v_empty_499_, v_combine_500_, v_a_513_);
v___x_517_ = lean_apply_4(v_toBind_511_, lean_box(0), lean_box(0), v___x_516_, v___f_515_);
return v___x_517_;
}
case 8:
{
lean_object* v_a_518_; 
v_a_518_ = lean_ctor_get(v_a_501_, 1);
lean_inc_ref(v_a_518_);
lean_dec_ref_known(v_a_501_, 2);
v_a_501_ = v_a_518_;
goto _start;
}
case 9:
{
lean_object* v_toApplicative_520_; lean_object* v_toBind_521_; lean_object* v_toPure_522_; lean_object* v_data_523_; lean_object* v_msg_524_; lean_object* v_children_525_; lean_object* v___f_526_; lean_object* v___f_527_; lean_object* v___f_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v_toApplicative_520_ = lean_ctor_get(v_inst_497_, 0);
v_toBind_521_ = lean_ctor_get(v_inst_497_, 1);
lean_inc_n(v_toBind_521_, 3);
v_toPure_522_ = lean_ctor_get(v_toApplicative_520_, 1);
lean_inc_n(v_toPure_522_, 3);
v_data_523_ = lean_ctor_get(v_a_501_, 0);
lean_inc_ref(v_data_523_);
v_msg_524_ = lean_ctor_get(v_a_501_, 1);
lean_inc_ref(v_msg_524_);
v_children_525_ = lean_ctor_get(v_a_501_, 2);
lean_inc_ref_n(v_children_525_, 2);
lean_dec_ref_known(v_a_501_, 3);
v___f_526_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__2), 2, 1);
lean_closure_set(v___f_526_, 0, v_toPure_522_);
lean_inc(v_empty_499_);
lean_inc(v_onTrace_498_);
lean_inc_ref(v_inst_497_);
v___f_527_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__4), 9, 6);
lean_closure_set(v___f_527_, 0, v_combine_500_);
lean_closure_set(v___f_527_, 1, v_toPure_522_);
lean_closure_set(v___f_527_, 2, v_inst_497_);
lean_closure_set(v___f_527_, 3, v_onTrace_498_);
lean_closure_set(v___f_527_, 4, v_empty_499_);
lean_closure_set(v___f_527_, 5, v_toBind_521_);
v___f_528_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__5), 8, 7);
lean_closure_set(v___f_528_, 0, v_children_525_);
lean_closure_set(v___f_528_, 1, v_inst_497_);
lean_closure_set(v___f_528_, 2, v___f_527_);
lean_closure_set(v___f_528_, 3, v_toBind_521_);
lean_closure_set(v___f_528_, 4, v___f_526_);
lean_closure_set(v___f_528_, 5, v_empty_499_);
lean_closure_set(v___f_528_, 6, v_toPure_522_);
v___x_529_ = lean_apply_3(v_onTrace_498_, v_data_523_, v_msg_524_, v_children_525_);
v___x_530_ = lean_apply_4(v_toBind_521_, lean_box(0), lean_box(0), v___x_529_, v___f_528_);
return v___x_530_;
}
case 11:
{
lean_object* v_a_531_; 
v_a_531_ = lean_ctor_get(v_a_501_, 1);
lean_inc_ref(v_a_531_);
lean_dec_ref_known(v_a_501_, 2);
v_a_501_ = v_a_531_;
goto _start;
}
default: 
{
lean_object* v_toApplicative_533_; lean_object* v_toPure_534_; lean_object* v___x_535_; 
v_toApplicative_533_ = lean_ctor_get(v_inst_497_, 0);
lean_inc_ref(v_toApplicative_533_);
lean_dec_ref(v_a_501_);
lean_dec(v_combine_500_);
lean_dec(v_onTrace_498_);
lean_dec_ref(v_inst_497_);
v_toPure_534_ = lean_ctor_get(v_toApplicative_533_, 1);
lean_inc(v_toPure_534_);
lean_dec_ref(v_toApplicative_533_);
v___x_535_ = lean_apply_2(v_toPure_534_, lean_box(0), v_empty_499_);
return v___x_535_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__1(lean_object* v_combine_536_, lean_object* v_toPure_537_, lean_object* v_inst_538_, lean_object* v_onTrace_539_, lean_object* v_empty_540_, lean_object* v_a_541_, lean_object* v_toBind_542_, lean_object* v_____do__lift_543_){
_start:
{
lean_object* v___f_544_; lean_object* v___x_545_; lean_object* v___x_546_; 
lean_inc(v_combine_536_);
v___f_544_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg___lam__0), 4, 3);
lean_closure_set(v___f_544_, 0, v_combine_536_);
lean_closure_set(v___f_544_, 1, v_____do__lift_543_);
lean_closure_set(v___f_544_, 2, v_toPure_537_);
v___x_545_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(v_inst_538_, v_onTrace_539_, v_empty_540_, v_combine_536_, v_a_541_);
v___x_546_ = lean_apply_4(v_toBind_542_, lean_box(0), lean_box(0), v___x_545_, v___f_544_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go(lean_object* v_m_547_, lean_object* v_inst_548_, lean_object* v_00_u03b1_549_, lean_object* v_onTrace_550_, lean_object* v_empty_551_, lean_object* v_combine_552_, lean_object* v_a_553_){
_start:
{
lean_object* v___x_554_; 
v___x_554_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(v_inst_548_, v_onTrace_550_, v_empty_551_, v_combine_552_, v_a_553_);
return v___x_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___redArg(lean_object* v_inst_555_, lean_object* v_msg_556_, lean_object* v_onTrace_557_, lean_object* v_empty_558_, lean_object* v_combine_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(v_inst_555_, v_onTrace_557_, v_empty_558_, v_combine_559_, v_msg_556_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM(lean_object* v_m_561_, lean_object* v_inst_562_, lean_object* v_00_u03b1_563_, lean_object* v_msg_564_, lean_object* v_onTrace_565_, lean_object* v_empty_566_, lean_object* v_combine_567_){
_start:
{
lean_object* v___x_568_; 
v___x_568_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___redArg(v_inst_562_, v_onTrace_565_, v_empty_566_, v_combine_567_, v_msg_564_);
return v___x_568_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__1(void){
_start:
{
lean_object* v___x_569_; 
v___x_569_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52);
return v___x_569_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__3(void){
_start:
{
lean_object* v___x_570_; 
v___x_570_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__0(lean_object* v_combine_571_, lean_object* v_acc_572_, lean_object* v_toPure_573_, lean_object* v_____do__lift_574_){
_start:
{
lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_575_ = lean_apply_2(v_combine_571_, v_acc_572_, v_____do__lift_574_);
v___x_576_ = lean_apply_2(v_toPure_573_, lean_box(0), v___x_575_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__1(lean_object* v_combine_577_, lean_object* v_toPure_578_, lean_object* v_visitM_579_, lean_object* v_toBind_580_, lean_object* v_acc_581_, lean_object* v_msg_582_){
_start:
{
lean_object* v___f_583_; lean_object* v___x_584_; lean_object* v___x_585_; 
v___f_583_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_583_, 0, v_combine_577_);
lean_closure_set(v___f_583_, 1, v_acc_581_);
lean_closure_set(v___f_583_, 2, v_toPure_578_);
v___x_584_ = lean_apply_1(v_visitM_579_, v_msg_582_);
v___x_585_ = lean_apply_4(v_toBind_580_, lean_box(0), lean_box(0), v___x_584_, v___f_583_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg(lean_object* v_inst_586_, lean_object* v_arr_587_, lean_object* v_visitM_588_, lean_object* v_empty_589_, lean_object* v_combine_590_){
_start:
{
lean_object* v_toApplicative_591_; lean_object* v_toBind_592_; lean_object* v_toPure_593_; lean_object* v___x_594_; lean_object* v___x_595_; uint8_t v___x_596_; 
v_toApplicative_591_ = lean_ctor_get(v_inst_586_, 0);
v_toBind_592_ = lean_ctor_get(v_inst_586_, 1);
v_toPure_593_ = lean_ctor_get(v_toApplicative_591_, 1);
v___x_594_ = lean_unsigned_to_nat(0u);
v___x_595_ = lean_array_get_size(v_arr_587_);
v___x_596_ = lean_nat_dec_lt(v___x_594_, v___x_595_);
if (v___x_596_ == 0)
{
lean_object* v___x_597_; 
lean_inc(v_toPure_593_);
lean_dec(v_combine_590_);
lean_dec(v_visitM_588_);
lean_dec_ref(v_arr_587_);
lean_dec_ref(v_inst_586_);
v___x_597_ = lean_apply_2(v_toPure_593_, lean_box(0), v_empty_589_);
return v___x_597_;
}
else
{
lean_object* v___f_598_; uint8_t v___x_599_; 
lean_inc(v_toBind_592_);
lean_inc(v_toPure_593_);
v___f_598_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_598_, 0, v_combine_590_);
lean_closure_set(v___f_598_, 1, v_toPure_593_);
lean_closure_set(v___f_598_, 2, v_visitM_588_);
lean_closure_set(v___f_598_, 3, v_toBind_592_);
v___x_599_ = lean_nat_dec_le(v___x_595_, v___x_595_);
if (v___x_599_ == 0)
{
if (v___x_596_ == 0)
{
lean_object* v___x_600_; 
lean_inc(v_toPure_593_);
lean_dec_ref(v___f_598_);
lean_dec_ref(v_arr_587_);
lean_dec_ref(v_inst_586_);
v___x_600_ = lean_apply_2(v_toPure_593_, lean_box(0), v_empty_589_);
return v___x_600_;
}
else
{
size_t v___x_601_; size_t v___x_602_; lean_object* v___x_603_; 
v___x_601_ = ((size_t)0ULL);
v___x_602_ = lean_usize_of_nat(v___x_595_);
v___x_603_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_586_, v___f_598_, v_arr_587_, v___x_601_, v___x_602_, v_empty_589_);
return v___x_603_;
}
}
else
{
size_t v___x_604_; size_t v___x_605_; lean_object* v___x_606_; 
v___x_604_ = ((size_t)0ULL);
v___x_605_ = lean_usize_of_nat(v___x_595_);
v___x_606_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_586_, v___f_598_, v_arr_587_, v___x_604_, v___x_605_, v_empty_589_);
return v___x_606_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM(lean_object* v_m_607_, lean_object* v_inst_608_, lean_object* v_00_u03b1_609_, lean_object* v_00_u03b2_610_, lean_object* v_arr_611_, lean_object* v_visitM_612_, lean_object* v_empty_613_, lean_object* v_combine_614_){
_start:
{
lean_object* v_toApplicative_615_; lean_object* v_toBind_616_; lean_object* v_toPure_617_; lean_object* v___x_618_; lean_object* v___x_619_; uint8_t v___x_620_; 
v_toApplicative_615_ = lean_ctor_get(v_inst_608_, 0);
v_toBind_616_ = lean_ctor_get(v_inst_608_, 1);
v_toPure_617_ = lean_ctor_get(v_toApplicative_615_, 1);
v___x_618_ = lean_unsigned_to_nat(0u);
v___x_619_ = lean_array_get_size(v_arr_611_);
v___x_620_ = lean_nat_dec_lt(v___x_618_, v___x_619_);
if (v___x_620_ == 0)
{
lean_object* v___x_621_; 
lean_inc(v_toPure_617_);
lean_dec(v_combine_614_);
lean_dec(v_visitM_612_);
lean_dec_ref(v_arr_611_);
lean_dec_ref(v_inst_608_);
v___x_621_ = lean_apply_2(v_toPure_617_, lean_box(0), v_empty_613_);
return v___x_621_;
}
else
{
lean_object* v___f_622_; uint8_t v___x_623_; 
lean_inc(v_toBind_616_);
lean_inc(v_toPure_617_);
v___f_622_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_622_, 0, v_combine_614_);
lean_closure_set(v___f_622_, 1, v_toPure_617_);
lean_closure_set(v___f_622_, 2, v_visitM_612_);
lean_closure_set(v___f_622_, 3, v_toBind_616_);
v___x_623_ = lean_nat_dec_le(v___x_619_, v___x_619_);
if (v___x_623_ == 0)
{
if (v___x_620_ == 0)
{
lean_object* v___x_624_; 
lean_inc(v_toPure_617_);
lean_dec_ref(v___f_622_);
lean_dec_ref(v_arr_611_);
lean_dec_ref(v_inst_608_);
v___x_624_ = lean_apply_2(v_toPure_617_, lean_box(0), v_empty_613_);
return v___x_624_;
}
else
{
size_t v___x_625_; size_t v___x_626_; lean_object* v___x_627_; 
v___x_625_ = ((size_t)0ULL);
v___x_626_ = lean_usize_of_nat(v___x_619_);
v___x_627_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_608_, v___f_622_, v_arr_611_, v___x_625_, v___x_626_, v_empty_613_);
return v___x_627_;
}
}
else
{
size_t v___x_628_; size_t v___x_629_; lean_object* v___x_630_; 
v___x_628_ = ((size_t)0ULL);
v___x_629_ = lean_usize_of_nat(v___x_619_);
v___x_630_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_608_, v___f_622_, v_arr_611_, v___x_628_, v___x_629_, v_empty_613_);
return v___x_630_;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__1(void){
_start:
{
lean_object* v___x_631_; 
v___x_631_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__52);
return v___x_631_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__3(void){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3___closed__87);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___lam__0(lean_object* v_toPure_633_, lean_object* v_____do__lift_634_){
_start:
{
lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_635_, 0, v_____do__lift_634_);
v___x_636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_636_, 0, v___x_635_);
v___x_637_ = lean_apply_2(v_toPure_633_, lean_box(0), v___x_636_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg(lean_object* v_inst_640_, lean_object* v_arr_641_, lean_object* v_visitM_642_, lean_object* v_empty_643_, lean_object* v_combine_644_){
_start:
{
lean_object* v___x_645_; lean_object* v___x_646_; uint8_t v___x_647_; 
v___x_645_ = lean_array_get_size(v_arr_641_);
v___x_646_ = lean_unsigned_to_nat(0u);
v___x_647_ = lean_nat_dec_eq(v___x_645_, v___x_646_);
if (v___x_647_ == 0)
{
lean_object* v_toApplicative_648_; lean_object* v_toBind_649_; lean_object* v_toPure_650_; lean_object* v___f_651_; uint8_t v___x_652_; 
v_toApplicative_648_ = lean_ctor_get(v_inst_640_, 0);
v_toBind_649_ = lean_ctor_get(v_inst_640_, 1);
lean_inc(v_toBind_649_);
v_toPure_650_ = lean_ctor_get(v_toApplicative_648_, 1);
lean_inc(v_toPure_650_);
v___f_651_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___lam__0), 2, 1);
lean_closure_set(v___f_651_, 0, v_toPure_650_);
v___x_652_ = lean_nat_dec_lt(v___x_646_, v___x_645_);
if (v___x_652_ == 0)
{
lean_object* v___x_653_; lean_object* v___x_654_; 
lean_inc(v_toPure_650_);
lean_dec(v_combine_644_);
lean_dec(v_visitM_642_);
lean_dec_ref(v_arr_641_);
lean_dec_ref(v_inst_640_);
v___x_653_ = lean_apply_2(v_toPure_650_, lean_box(0), v_empty_643_);
v___x_654_ = lean_apply_4(v_toBind_649_, lean_box(0), lean_box(0), v___x_653_, v___f_651_);
return v___x_654_;
}
else
{
lean_object* v___f_655_; uint8_t v___x_656_; 
lean_inc(v_toBind_649_);
lean_inc(v_toPure_650_);
v___f_655_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_655_, 0, v_combine_644_);
lean_closure_set(v___f_655_, 1, v_toPure_650_);
lean_closure_set(v___f_655_, 2, v_visitM_642_);
lean_closure_set(v___f_655_, 3, v_toBind_649_);
v___x_656_ = lean_nat_dec_le(v___x_645_, v___x_645_);
if (v___x_656_ == 0)
{
if (v___x_652_ == 0)
{
lean_object* v___x_657_; lean_object* v___x_658_; 
lean_inc(v_toPure_650_);
lean_dec_ref(v___f_655_);
lean_dec_ref(v_arr_641_);
lean_dec_ref(v_inst_640_);
v___x_657_ = lean_apply_2(v_toPure_650_, lean_box(0), v_empty_643_);
v___x_658_ = lean_apply_4(v_toBind_649_, lean_box(0), lean_box(0), v___x_657_, v___f_651_);
return v___x_658_;
}
else
{
size_t v___x_659_; size_t v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
v___x_659_ = ((size_t)0ULL);
v___x_660_ = lean_usize_of_nat(v___x_645_);
v___x_661_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_640_, v___f_655_, v_arr_641_, v___x_659_, v___x_660_, v_empty_643_);
v___x_662_ = lean_apply_4(v_toBind_649_, lean_box(0), lean_box(0), v___x_661_, v___f_651_);
return v___x_662_;
}
}
else
{
size_t v___x_663_; size_t v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; 
v___x_663_ = ((size_t)0ULL);
v___x_664_ = lean_usize_of_nat(v___x_645_);
v___x_665_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_640_, v___f_655_, v_arr_641_, v___x_663_, v___x_664_, v_empty_643_);
v___x_666_ = lean_apply_4(v_toBind_649_, lean_box(0), lean_box(0), v___x_665_, v___f_651_);
return v___x_666_;
}
}
}
else
{
lean_object* v_toApplicative_667_; lean_object* v_toPure_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
lean_dec(v_combine_644_);
lean_dec(v_empty_643_);
lean_dec(v_visitM_642_);
lean_dec_ref(v_arr_641_);
v_toApplicative_667_ = lean_ctor_get(v_inst_640_, 0);
lean_inc_ref(v_toApplicative_667_);
lean_dec_ref(v_inst_640_);
v_toPure_668_ = lean_ctor_get(v_toApplicative_667_, 1);
lean_inc(v_toPure_668_);
lean_dec_ref(v_toApplicative_667_);
v___x_669_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___closed__0));
v___x_670_ = lean_apply_2(v_toPure_668_, lean_box(0), v___x_669_);
return v___x_670_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM(lean_object* v_m_671_, lean_object* v_inst_672_, lean_object* v_00_u03b1_673_, lean_object* v_00_u03b2_674_, lean_object* v_arr_675_, lean_object* v_visitM_676_, lean_object* v_empty_677_, lean_object* v_combine_678_){
_start:
{
lean_object* v___x_679_; lean_object* v___x_680_; uint8_t v___x_681_; 
v___x_679_ = lean_array_get_size(v_arr_675_);
v___x_680_ = lean_unsigned_to_nat(0u);
v___x_681_ = lean_nat_dec_eq(v___x_679_, v___x_680_);
if (v___x_681_ == 0)
{
lean_object* v_toApplicative_682_; lean_object* v_toBind_683_; lean_object* v_toPure_684_; lean_object* v___f_685_; uint8_t v___x_686_; 
v_toApplicative_682_ = lean_ctor_get(v_inst_672_, 0);
v_toBind_683_ = lean_ctor_get(v_inst_672_, 1);
lean_inc(v_toBind_683_);
v_toPure_684_ = lean_ctor_get(v_toApplicative_682_, 1);
lean_inc(v_toPure_684_);
v___f_685_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___lam__0), 2, 1);
lean_closure_set(v___f_685_, 0, v_toPure_684_);
v___x_686_ = lean_nat_dec_lt(v___x_680_, v___x_679_);
if (v___x_686_ == 0)
{
lean_object* v___x_687_; lean_object* v___x_688_; 
lean_inc(v_toPure_684_);
lean_dec(v_combine_678_);
lean_dec(v_visitM_676_);
lean_dec_ref(v_arr_675_);
lean_dec_ref(v_inst_672_);
v___x_687_ = lean_apply_2(v_toPure_684_, lean_box(0), v_empty_677_);
v___x_688_ = lean_apply_4(v_toBind_683_, lean_box(0), lean_box(0), v___x_687_, v___f_685_);
return v___x_688_;
}
else
{
lean_object* v___f_689_; uint8_t v___x_690_; 
lean_inc(v_toBind_683_);
lean_inc(v_toPure_684_);
v___f_689_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_689_, 0, v_combine_678_);
lean_closure_set(v___f_689_, 1, v_toPure_684_);
lean_closure_set(v___f_689_, 2, v_visitM_676_);
lean_closure_set(v___f_689_, 3, v_toBind_683_);
v___x_690_ = lean_nat_dec_le(v___x_679_, v___x_679_);
if (v___x_690_ == 0)
{
if (v___x_686_ == 0)
{
lean_object* v___x_691_; lean_object* v___x_692_; 
lean_inc(v_toPure_684_);
lean_dec_ref(v___f_689_);
lean_dec_ref(v_arr_675_);
lean_dec_ref(v_inst_672_);
v___x_691_ = lean_apply_2(v_toPure_684_, lean_box(0), v_empty_677_);
v___x_692_ = lean_apply_4(v_toBind_683_, lean_box(0), lean_box(0), v___x_691_, v___f_685_);
return v___x_692_;
}
else
{
size_t v___x_693_; size_t v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; 
v___x_693_ = ((size_t)0ULL);
v___x_694_ = lean_usize_of_nat(v___x_679_);
v___x_695_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_672_, v___f_689_, v_arr_675_, v___x_693_, v___x_694_, v_empty_677_);
v___x_696_ = lean_apply_4(v_toBind_683_, lean_box(0), lean_box(0), v___x_695_, v___f_685_);
return v___x_696_;
}
}
else
{
size_t v___x_697_; size_t v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; 
v___x_697_ = ((size_t)0ULL);
v___x_698_ = lean_usize_of_nat(v___x_679_);
v___x_699_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_672_, v___f_689_, v_arr_675_, v___x_697_, v___x_698_, v_empty_677_);
v___x_700_ = lean_apply_4(v_toBind_683_, lean_box(0), lean_box(0), v___x_699_, v___f_685_);
return v___x_700_;
}
}
}
else
{
lean_object* v_toApplicative_701_; lean_object* v_toPure_702_; lean_object* v___x_703_; lean_object* v___x_704_; 
lean_dec(v_combine_678_);
lean_dec(v_empty_677_);
lean_dec(v_visitM_676_);
lean_dec_ref(v_arr_675_);
v_toApplicative_701_ = lean_ctor_get(v_inst_672_, 0);
lean_inc_ref(v_toApplicative_701_);
lean_dec_ref(v_inst_672_);
v_toPure_702_ = lean_ctor_get(v_toApplicative_701_, 1);
lean_inc(v_toPure_702_);
lean_dec_ref(v_toApplicative_701_);
v___x_703_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___closed__0));
v___x_704_ = lean_apply_2(v_toPure_702_, lean_box(0), v___x_703_);
return v___x_704_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(lean_object* v_msg_705_, lean_object* v_modify_706_){
_start:
{
switch(lean_obj_tag(v_msg_705_))
{
case 2:
{
lean_object* v_a_707_; lean_object* v_a_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_716_; 
v_a_707_ = lean_ctor_get(v_msg_705_, 0);
v_a_708_ = lean_ctor_get(v_msg_705_, 1);
v_isSharedCheck_716_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_716_ == 0)
{
v___x_710_ = v_msg_705_;
v_isShared_711_ = v_isSharedCheck_716_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_a_708_);
lean_inc(v_a_707_);
lean_dec(v_msg_705_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_716_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_712_; lean_object* v___x_714_; 
v___x_712_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_708_, v_modify_706_);
if (v_isShared_711_ == 0)
{
lean_ctor_set(v___x_710_, 1, v___x_712_);
v___x_714_ = v___x_710_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_a_707_);
lean_ctor_set(v_reuseFailAlloc_715_, 1, v___x_712_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
}
case 3:
{
lean_object* v_a_717_; lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_738_; 
v_a_717_ = lean_ctor_get(v_msg_705_, 0);
v_a_718_ = lean_ctor_get(v_msg_705_, 1);
v_isSharedCheck_738_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_738_ == 0)
{
v___x_720_ = v_msg_705_;
v_isShared_721_ = v_isSharedCheck_738_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_inc(v_a_717_);
lean_dec(v_msg_705_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_738_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v_env_722_; lean_object* v_mctx_723_; lean_object* v_lctx_724_; lean_object* v_opts_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_737_; 
v_env_722_ = lean_ctor_get(v_a_717_, 0);
v_mctx_723_ = lean_ctor_get(v_a_717_, 1);
v_lctx_724_ = lean_ctor_get(v_a_717_, 2);
v_opts_725_ = lean_ctor_get(v_a_717_, 3);
v_isSharedCheck_737_ = !lean_is_exclusive(v_a_717_);
if (v_isSharedCheck_737_ == 0)
{
v___x_727_ = v_a_717_;
v_isShared_728_ = v_isSharedCheck_737_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_opts_725_);
lean_inc(v_lctx_724_);
lean_inc(v_mctx_723_);
lean_inc(v_env_722_);
lean_dec(v_a_717_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_737_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v___x_729_; lean_object* v___x_731_; 
lean_inc_ref(v_modify_706_);
v___x_729_ = lean_apply_1(v_modify_706_, v_opts_725_);
if (v_isShared_728_ == 0)
{
lean_ctor_set(v___x_727_, 3, v___x_729_);
v___x_731_ = v___x_727_;
goto v_reusejp_730_;
}
else
{
lean_object* v_reuseFailAlloc_736_; 
v_reuseFailAlloc_736_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_736_, 0, v_env_722_);
lean_ctor_set(v_reuseFailAlloc_736_, 1, v_mctx_723_);
lean_ctor_set(v_reuseFailAlloc_736_, 2, v_lctx_724_);
lean_ctor_set(v_reuseFailAlloc_736_, 3, v___x_729_);
v___x_731_ = v_reuseFailAlloc_736_;
goto v_reusejp_730_;
}
v_reusejp_730_:
{
lean_object* v___x_732_; lean_object* v___x_734_; 
v___x_732_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_718_, v_modify_706_);
if (v_isShared_721_ == 0)
{
lean_ctor_set(v___x_720_, 1, v___x_732_);
lean_ctor_set(v___x_720_, 0, v___x_731_);
v___x_734_ = v___x_720_;
goto v_reusejp_733_;
}
else
{
lean_object* v_reuseFailAlloc_735_; 
v_reuseFailAlloc_735_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_735_, 0, v___x_731_);
lean_ctor_set(v_reuseFailAlloc_735_, 1, v___x_732_);
v___x_734_ = v_reuseFailAlloc_735_;
goto v_reusejp_733_;
}
v_reusejp_733_:
{
return v___x_734_;
}
}
}
}
}
case 4:
{
lean_object* v_a_739_; lean_object* v_a_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_748_; 
v_a_739_ = lean_ctor_get(v_msg_705_, 0);
v_a_740_ = lean_ctor_get(v_msg_705_, 1);
v_isSharedCheck_748_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_748_ == 0)
{
v___x_742_ = v_msg_705_;
v_isShared_743_ = v_isSharedCheck_748_;
goto v_resetjp_741_;
}
else
{
lean_inc(v_a_740_);
lean_inc(v_a_739_);
lean_dec(v_msg_705_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_748_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
lean_object* v___x_744_; lean_object* v___x_746_; 
v___x_744_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_740_, v_modify_706_);
if (v_isShared_743_ == 0)
{
lean_ctor_set(v___x_742_, 1, v___x_744_);
v___x_746_ = v___x_742_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v_a_739_);
lean_ctor_set(v_reuseFailAlloc_747_, 1, v___x_744_);
v___x_746_ = v_reuseFailAlloc_747_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
return v___x_746_;
}
}
}
case 5:
{
lean_object* v_a_749_; lean_object* v_a_750_; lean_object* v___x_752_; uint8_t v_isShared_753_; uint8_t v_isSharedCheck_758_; 
v_a_749_ = lean_ctor_get(v_msg_705_, 0);
v_a_750_ = lean_ctor_get(v_msg_705_, 1);
v_isSharedCheck_758_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_758_ == 0)
{
v___x_752_ = v_msg_705_;
v_isShared_753_ = v_isSharedCheck_758_;
goto v_resetjp_751_;
}
else
{
lean_inc(v_a_750_);
lean_inc(v_a_749_);
lean_dec(v_msg_705_);
v___x_752_ = lean_box(0);
v_isShared_753_ = v_isSharedCheck_758_;
goto v_resetjp_751_;
}
v_resetjp_751_:
{
lean_object* v___x_754_; lean_object* v___x_756_; 
v___x_754_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_750_, v_modify_706_);
if (v_isShared_753_ == 0)
{
lean_ctor_set(v___x_752_, 1, v___x_754_);
v___x_756_ = v___x_752_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v_a_749_);
lean_ctor_set(v_reuseFailAlloc_757_, 1, v___x_754_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
}
case 6:
{
lean_object* v_a_759_; lean_object* v___x_761_; uint8_t v_isShared_762_; uint8_t v_isSharedCheck_767_; 
v_a_759_ = lean_ctor_get(v_msg_705_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_767_ == 0)
{
v___x_761_ = v_msg_705_;
v_isShared_762_ = v_isSharedCheck_767_;
goto v_resetjp_760_;
}
else
{
lean_inc(v_a_759_);
lean_dec(v_msg_705_);
v___x_761_ = lean_box(0);
v_isShared_762_ = v_isSharedCheck_767_;
goto v_resetjp_760_;
}
v_resetjp_760_:
{
lean_object* v___x_763_; lean_object* v___x_765_; 
v___x_763_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_759_, v_modify_706_);
if (v_isShared_762_ == 0)
{
lean_ctor_set(v___x_761_, 0, v___x_763_);
v___x_765_ = v___x_761_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(6, 1, 0);
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
case 7:
{
lean_object* v_a_768_; lean_object* v_a_769_; lean_object* v___x_771_; uint8_t v_isShared_772_; uint8_t v_isSharedCheck_778_; 
v_a_768_ = lean_ctor_get(v_msg_705_, 0);
v_a_769_ = lean_ctor_get(v_msg_705_, 1);
v_isSharedCheck_778_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_778_ == 0)
{
v___x_771_ = v_msg_705_;
v_isShared_772_ = v_isSharedCheck_778_;
goto v_resetjp_770_;
}
else
{
lean_inc(v_a_769_);
lean_inc(v_a_768_);
lean_dec(v_msg_705_);
v___x_771_ = lean_box(0);
v_isShared_772_ = v_isSharedCheck_778_;
goto v_resetjp_770_;
}
v_resetjp_770_:
{
lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_776_; 
lean_inc_ref(v_modify_706_);
v___x_773_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_768_, v_modify_706_);
v___x_774_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_769_, v_modify_706_);
if (v_isShared_772_ == 0)
{
lean_ctor_set(v___x_771_, 1, v___x_774_);
lean_ctor_set(v___x_771_, 0, v___x_773_);
v___x_776_ = v___x_771_;
goto v_reusejp_775_;
}
else
{
lean_object* v_reuseFailAlloc_777_; 
v_reuseFailAlloc_777_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_777_, 0, v___x_773_);
lean_ctor_set(v_reuseFailAlloc_777_, 1, v___x_774_);
v___x_776_ = v_reuseFailAlloc_777_;
goto v_reusejp_775_;
}
v_reusejp_775_:
{
return v___x_776_;
}
}
}
case 8:
{
lean_object* v_a_779_; lean_object* v_a_780_; lean_object* v___x_782_; uint8_t v_isShared_783_; uint8_t v_isSharedCheck_788_; 
v_a_779_ = lean_ctor_get(v_msg_705_, 0);
v_a_780_ = lean_ctor_get(v_msg_705_, 1);
v_isSharedCheck_788_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_788_ == 0)
{
v___x_782_ = v_msg_705_;
v_isShared_783_ = v_isSharedCheck_788_;
goto v_resetjp_781_;
}
else
{
lean_inc(v_a_780_);
lean_inc(v_a_779_);
lean_dec(v_msg_705_);
v___x_782_ = lean_box(0);
v_isShared_783_ = v_isSharedCheck_788_;
goto v_resetjp_781_;
}
v_resetjp_781_:
{
lean_object* v___x_784_; lean_object* v___x_786_; 
v___x_784_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_780_, v_modify_706_);
if (v_isShared_783_ == 0)
{
lean_ctor_set(v___x_782_, 1, v___x_784_);
v___x_786_ = v___x_782_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_787_; 
v_reuseFailAlloc_787_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v_reuseFailAlloc_787_, 0, v_a_779_);
lean_ctor_set(v_reuseFailAlloc_787_, 1, v___x_784_);
v___x_786_ = v_reuseFailAlloc_787_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
return v___x_786_;
}
}
}
case 9:
{
lean_object* v_data_789_; lean_object* v_msg_790_; lean_object* v_children_791_; lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_802_; 
v_data_789_ = lean_ctor_get(v_msg_705_, 0);
v_msg_790_ = lean_ctor_get(v_msg_705_, 1);
v_children_791_ = lean_ctor_get(v_msg_705_, 2);
v_isSharedCheck_802_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_802_ == 0)
{
v___x_793_ = v_msg_705_;
v_isShared_794_ = v_isSharedCheck_802_;
goto v_resetjp_792_;
}
else
{
lean_inc(v_children_791_);
lean_inc(v_msg_790_);
lean_inc(v_data_789_);
lean_dec(v_msg_705_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_802_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v___x_795_; size_t v_sz_796_; size_t v___x_797_; lean_object* v___x_798_; lean_object* v___x_800_; 
lean_inc_ref(v_modify_706_);
v___x_795_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_msg_790_, v_modify_706_);
v_sz_796_ = lean_array_size(v_children_791_);
v___x_797_ = ((size_t)0ULL);
v___x_798_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions_spec__0(v_modify_706_, v_sz_796_, v___x_797_, v_children_791_);
if (v_isShared_794_ == 0)
{
lean_ctor_set(v___x_793_, 2, v___x_798_);
lean_ctor_set(v___x_793_, 1, v___x_795_);
v___x_800_ = v___x_793_;
goto v_reusejp_799_;
}
else
{
lean_object* v_reuseFailAlloc_801_; 
v_reuseFailAlloc_801_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_reuseFailAlloc_801_, 0, v_data_789_);
lean_ctor_set(v_reuseFailAlloc_801_, 1, v___x_795_);
lean_ctor_set(v_reuseFailAlloc_801_, 2, v___x_798_);
v___x_800_ = v_reuseFailAlloc_801_;
goto v_reusejp_799_;
}
v_reusejp_799_:
{
return v___x_800_;
}
}
}
case 11:
{
lean_object* v_a_803_; lean_object* v_a_804_; lean_object* v___x_806_; uint8_t v_isShared_807_; uint8_t v_isSharedCheck_812_; 
v_a_803_ = lean_ctor_get(v_msg_705_, 0);
v_a_804_ = lean_ctor_get(v_msg_705_, 1);
v_isSharedCheck_812_ = !lean_is_exclusive(v_msg_705_);
if (v_isSharedCheck_812_ == 0)
{
v___x_806_ = v_msg_705_;
v_isShared_807_ = v_isSharedCheck_812_;
goto v_resetjp_805_;
}
else
{
lean_inc(v_a_804_);
lean_inc(v_a_803_);
lean_dec(v_msg_705_);
v___x_806_ = lean_box(0);
v_isShared_807_ = v_isSharedCheck_812_;
goto v_resetjp_805_;
}
v_resetjp_805_:
{
lean_object* v___x_808_; lean_object* v___x_810_; 
v___x_808_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_a_804_, v_modify_706_);
if (v_isShared_807_ == 0)
{
lean_ctor_set(v___x_806_, 1, v___x_808_);
v___x_810_ = v___x_806_;
goto v_reusejp_809_;
}
else
{
lean_object* v_reuseFailAlloc_811_; 
v_reuseFailAlloc_811_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v_reuseFailAlloc_811_, 0, v_a_803_);
lean_ctor_set(v_reuseFailAlloc_811_, 1, v___x_808_);
v___x_810_ = v_reuseFailAlloc_811_;
goto v_reusejp_809_;
}
v_reusejp_809_:
{
return v___x_810_;
}
}
}
default: 
{
lean_dec_ref(v_modify_706_);
return v_msg_705_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions_spec__0(lean_object* v_modify_813_, size_t v_sz_814_, size_t v_i_815_, lean_object* v_bs_816_){
_start:
{
uint8_t v___x_817_; 
v___x_817_ = lean_usize_dec_lt(v_i_815_, v_sz_814_);
if (v___x_817_ == 0)
{
lean_dec_ref(v_modify_813_);
return v_bs_816_;
}
else
{
lean_object* v_v_818_; lean_object* v___x_819_; lean_object* v_bs_x27_820_; lean_object* v___x_821_; size_t v___x_822_; size_t v___x_823_; lean_object* v___x_824_; 
v_v_818_ = lean_array_uget(v_bs_816_, v_i_815_);
v___x_819_ = lean_unsigned_to_nat(0u);
v_bs_x27_820_ = lean_array_uset(v_bs_816_, v_i_815_, v___x_819_);
lean_inc_ref(v_modify_813_);
v___x_821_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_v_818_, v_modify_813_);
v___x_822_ = ((size_t)1ULL);
v___x_823_ = lean_usize_add(v_i_815_, v___x_822_);
v___x_824_ = lean_array_uset(v_bs_x27_820_, v_i_815_, v___x_821_);
v_i_815_ = v___x_823_;
v_bs_816_ = v___x_824_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions_spec__0___boxed(lean_object* v_modify_826_, lean_object* v_sz_827_, lean_object* v_i_828_, lean_object* v_bs_829_){
_start:
{
size_t v_sz_boxed_830_; size_t v_i_boxed_831_; lean_object* v_res_832_; 
v_sz_boxed_830_ = lean_unbox_usize(v_sz_827_);
lean_dec(v_sz_827_);
v_i_boxed_831_ = lean_unbox_usize(v_i_828_);
lean_dec(v_i_828_);
v_res_832_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions_spec__0(v_modify_826_, v_sz_boxed_830_, v_i_boxed_831_, v_bs_829_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg(lean_object* v_inst_845_, lean_object* v_f_846_, lean_object* v_td_847_, lean_object* v_header_848_, lean_object* v_children_849_){
_start:
{
lean_object* v_cls_850_; lean_object* v___x_851_; uint8_t v___x_852_; 
v_cls_850_ = lean_ctor_get(v_td_847_, 0);
v___x_851_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3));
v___x_852_ = lean_name_eq(v_cls_850_, v___x_851_);
if (v___x_852_ == 0)
{
lean_object* v___x_853_; uint8_t v___x_854_; 
v___x_853_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4));
v___x_854_ = l_Lean_Name_isPrefixOf(v___x_853_, v_cls_850_);
if (v___x_854_ == 0)
{
lean_object* v_toApplicative_855_; lean_object* v_toPure_856_; lean_object* v___x_857_; lean_object* v___x_858_; 
lean_dec_ref(v_children_849_);
lean_dec_ref(v_header_848_);
lean_dec_ref(v_td_847_);
lean_dec(v_f_846_);
v_toApplicative_855_ = lean_ctor_get(v_inst_845_, 0);
lean_inc_ref(v_toApplicative_855_);
lean_dec_ref(v_inst_845_);
v_toPure_856_ = lean_ctor_get(v_toApplicative_855_, 1);
lean_inc(v_toPure_856_);
lean_dec_ref(v_toApplicative_855_);
v___x_857_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__5));
v___x_858_ = lean_apply_2(v_toPure_856_, lean_box(0), v___x_857_);
return v___x_858_;
}
else
{
lean_object* v___x_859_; 
lean_dec_ref(v_inst_845_);
v___x_859_ = lean_apply_3(v_f_846_, v_td_847_, v_header_848_, v_children_849_);
return v___x_859_;
}
}
else
{
lean_object* v_toApplicative_860_; lean_object* v_toPure_861_; lean_object* v___x_862_; lean_object* v___x_863_; 
lean_dec_ref(v_children_849_);
lean_dec_ref(v_header_848_);
lean_dec_ref(v_td_847_);
lean_dec(v_f_846_);
v_toApplicative_860_ = lean_ctor_get(v_inst_845_, 0);
lean_inc_ref(v_toApplicative_860_);
lean_dec_ref(v_inst_845_);
v_toPure_861_ = lean_ctor_get(v_toApplicative_860_, 1);
lean_inc(v_toPure_861_);
lean_dec_ref(v_toApplicative_860_);
v___x_862_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___closed__0));
v___x_863_ = lean_apply_2(v_toPure_861_, lean_box(0), v___x_862_);
return v___x_863_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes(lean_object* v_m_864_, lean_object* v_inst_865_, lean_object* v_00_u03b1_866_, lean_object* v_f_867_, lean_object* v_td_868_, lean_object* v_header_869_, lean_object* v_children_870_){
_start:
{
lean_object* v_cls_871_; lean_object* v___x_872_; uint8_t v___x_873_; 
v_cls_871_ = lean_ctor_get(v_td_868_, 0);
v___x_872_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3));
v___x_873_ = lean_name_eq(v_cls_871_, v___x_872_);
if (v___x_873_ == 0)
{
lean_object* v___x_874_; uint8_t v___x_875_; 
v___x_874_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4));
v___x_875_ = l_Lean_Name_isPrefixOf(v___x_874_, v_cls_871_);
if (v___x_875_ == 0)
{
lean_object* v_toApplicative_876_; lean_object* v_toPure_877_; lean_object* v___x_878_; lean_object* v___x_879_; 
lean_dec_ref(v_children_870_);
lean_dec_ref(v_header_869_);
lean_dec_ref(v_td_868_);
lean_dec(v_f_867_);
v_toApplicative_876_ = lean_ctor_get(v_inst_865_, 0);
lean_inc_ref(v_toApplicative_876_);
lean_dec_ref(v_inst_865_);
v_toPure_877_ = lean_ctor_get(v_toApplicative_876_, 1);
lean_inc(v_toPure_877_);
lean_dec_ref(v_toApplicative_876_);
v___x_878_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__5));
v___x_879_ = lean_apply_2(v_toPure_877_, lean_box(0), v___x_878_);
return v___x_879_;
}
else
{
lean_object* v___x_880_; 
lean_dec_ref(v_inst_865_);
v___x_880_ = lean_apply_3(v_f_867_, v_td_868_, v_header_869_, v_children_870_);
return v___x_880_;
}
}
else
{
lean_object* v_toApplicative_881_; lean_object* v_toPure_882_; lean_object* v___x_883_; lean_object* v___x_884_; 
lean_dec_ref(v_children_870_);
lean_dec_ref(v_header_869_);
lean_dec_ref(v_td_868_);
lean_dec(v_f_867_);
v_toApplicative_881_ = lean_ctor_get(v_inst_865_, 0);
lean_inc_ref(v_toApplicative_881_);
lean_dec_ref(v_inst_865_);
v_toPure_882_ = lean_ctor_get(v_toApplicative_881_, 1);
lean_inc(v_toPure_882_);
lean_dec_ref(v_toApplicative_881_);
v___x_883_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___redArg___closed__0));
v___x_884_ = lean_apply_2(v_toPure_882_, lean_box(0), v___x_883_);
return v___x_884_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(lean_object* v_onTrace_885_, lean_object* v_empty_886_, lean_object* v_combine_887_, lean_object* v_a_888_){
_start:
{
switch(lean_obj_tag(v_a_888_))
{
case 3:
{
lean_object* v_a_890_; 
v_a_890_ = lean_ctor_get(v_a_888_, 1);
lean_inc_ref(v_a_890_);
lean_dec_ref_known(v_a_888_, 2);
v_a_888_ = v_a_890_;
goto _start;
}
case 4:
{
lean_object* v_a_892_; 
v_a_892_ = lean_ctor_get(v_a_888_, 1);
lean_inc_ref(v_a_892_);
lean_dec_ref_known(v_a_888_, 2);
v_a_888_ = v_a_892_;
goto _start;
}
case 5:
{
lean_object* v_a_894_; 
v_a_894_ = lean_ctor_get(v_a_888_, 1);
lean_inc_ref(v_a_894_);
lean_dec_ref_known(v_a_888_, 2);
v_a_888_ = v_a_894_;
goto _start;
}
case 6:
{
lean_object* v_a_896_; 
v_a_896_ = lean_ctor_get(v_a_888_, 0);
lean_inc_ref(v_a_896_);
lean_dec_ref_known(v_a_888_, 1);
v_a_888_ = v_a_896_;
goto _start;
}
case 7:
{
lean_object* v_a_898_; lean_object* v_a_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
v_a_898_ = lean_ctor_get(v_a_888_, 0);
lean_inc_ref(v_a_898_);
v_a_899_ = lean_ctor_get(v_a_888_, 1);
lean_inc_ref(v_a_899_);
lean_dec_ref_known(v_a_888_, 2);
lean_inc_n(v_combine_887_, 2);
lean_inc(v_empty_886_);
lean_inc_ref(v_onTrace_885_);
v___x_900_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v_onTrace_885_, v_empty_886_, v_combine_887_, v_a_898_);
v___x_901_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v_onTrace_885_, v_empty_886_, v_combine_887_, v_a_899_);
v___x_902_ = lean_apply_2(v_combine_887_, v___x_900_, v___x_901_);
return v___x_902_;
}
case 8:
{
lean_object* v_a_903_; 
v_a_903_ = lean_ctor_get(v_a_888_, 1);
lean_inc_ref(v_a_903_);
lean_dec_ref_known(v_a_888_, 2);
v_a_888_ = v_a_903_;
goto _start;
}
case 9:
{
lean_object* v_data_905_; lean_object* v_msg_906_; lean_object* v_children_907_; lean_object* v___x_908_; lean_object* v___y_910_; 
v_data_905_ = lean_ctor_get(v_a_888_, 0);
lean_inc_ref(v_data_905_);
v_msg_906_ = lean_ctor_get(v_a_888_, 1);
lean_inc_ref(v_msg_906_);
v_children_907_ = lean_ctor_get(v_a_888_, 2);
lean_inc_ref_n(v_children_907_, 2);
lean_dec_ref_known(v_a_888_, 3);
lean_inc_ref(v_onTrace_885_);
v___x_908_ = lean_apply_4(v_onTrace_885_, v_data_905_, v_msg_906_, v_children_907_, lean_box(0));
if (lean_obj_tag(v___x_908_) == 0)
{
lean_object* v_butFirst_914_; 
v_butFirst_914_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_butFirst_914_);
lean_dec_ref_known(v___x_908_, 1);
if (lean_obj_tag(v_butFirst_914_) == 0)
{
lean_inc(v_empty_886_);
v___y_910_ = v_empty_886_;
goto v___jp_909_;
}
else
{
lean_object* v_val_915_; 
v_val_915_ = lean_ctor_get(v_butFirst_914_, 0);
lean_inc(v_val_915_);
lean_dec_ref_known(v_butFirst_914_, 1);
v___y_910_ = v_val_915_;
goto v___jp_909_;
}
}
else
{
lean_object* v_returning_916_; 
lean_dec_ref(v_children_907_);
lean_dec(v_combine_887_);
lean_dec_ref(v_onTrace_885_);
v_returning_916_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_returning_916_);
lean_dec_ref_known(v___x_908_, 1);
if (lean_obj_tag(v_returning_916_) == 0)
{
return v_empty_886_;
}
else
{
lean_object* v_val_917_; 
lean_dec(v_empty_886_);
v_val_917_ = lean_ctor_get(v_returning_916_, 0);
lean_inc(v_val_917_);
lean_dec_ref_known(v_returning_916_, 1);
return v_val_917_;
}
}
v___jp_909_:
{
size_t v_sz_911_; size_t v___x_912_; lean_object* v___x_913_; 
v_sz_911_ = lean_array_size(v_children_907_);
v___x_912_ = ((size_t)0ULL);
v___x_913_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___redArg(v_onTrace_885_, v_empty_886_, v_combine_887_, v_children_907_, v_sz_911_, v___x_912_, v___y_910_);
lean_dec_ref(v_children_907_);
return v___x_913_;
}
}
case 11:
{
lean_object* v_a_918_; 
v_a_918_ = lean_ctor_get(v_a_888_, 1);
lean_inc_ref(v_a_918_);
lean_dec_ref_known(v_a_888_, 2);
v_a_888_ = v_a_918_;
goto _start;
}
default: 
{
lean_dec_ref(v_a_888_);
lean_dec(v_combine_887_);
lean_dec_ref(v_onTrace_885_);
return v_empty_886_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___redArg(lean_object* v_onTrace_920_, lean_object* v_empty_921_, lean_object* v_combine_922_, lean_object* v_as_923_, size_t v_sz_924_, size_t v_i_925_, lean_object* v_b_926_){
_start:
{
uint8_t v___x_928_; 
v___x_928_ = lean_usize_dec_lt(v_i_925_, v_sz_924_);
if (v___x_928_ == 0)
{
lean_dec(v_combine_922_);
lean_dec(v_empty_921_);
lean_dec_ref(v_onTrace_920_);
return v_b_926_;
}
else
{
lean_object* v_a_929_; lean_object* v___x_930_; lean_object* v___x_931_; size_t v___x_932_; size_t v___x_933_; 
v_a_929_ = lean_array_uget_borrowed(v_as_923_, v_i_925_);
lean_inc(v_a_929_);
lean_inc_n(v_combine_922_, 2);
lean_inc(v_empty_921_);
lean_inc_ref(v_onTrace_920_);
v___x_930_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v_onTrace_920_, v_empty_921_, v_combine_922_, v_a_929_);
v___x_931_ = lean_apply_2(v_combine_922_, v_b_926_, v___x_930_);
v___x_932_ = ((size_t)1ULL);
v___x_933_ = lean_usize_add(v_i_925_, v___x_932_);
v_i_925_ = v___x_933_;
v_b_926_ = v___x_931_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___redArg___boxed(lean_object* v_onTrace_935_, lean_object* v_empty_936_, lean_object* v_combine_937_, lean_object* v_as_938_, lean_object* v_sz_939_, lean_object* v_i_940_, lean_object* v_b_941_, lean_object* v___y_942_){
_start:
{
size_t v_sz_boxed_943_; size_t v_i_boxed_944_; lean_object* v_res_945_; 
v_sz_boxed_943_ = lean_unbox_usize(v_sz_939_);
lean_dec(v_sz_939_);
v_i_boxed_944_ = lean_unbox_usize(v_i_940_);
lean_dec(v_i_940_);
v_res_945_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___redArg(v_onTrace_935_, v_empty_936_, v_combine_937_, v_as_938_, v_sz_boxed_943_, v_i_boxed_944_, v_b_941_);
lean_dec_ref(v_as_938_);
return v_res_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg___boxed(lean_object* v_onTrace_946_, lean_object* v_empty_947_, lean_object* v_combine_948_, lean_object* v_a_949_, lean_object* v___y_950_){
_start:
{
lean_object* v_res_951_; 
v_res_951_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v_onTrace_946_, v_empty_947_, v_combine_948_, v_a_949_);
return v_res_951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__0(lean_object* v_as_959_, size_t v_i_960_, size_t v_stop_961_, lean_object* v_b_962_){
_start:
{
uint8_t v___x_964_; 
v___x_964_ = lean_usize_dec_eq(v_i_960_, v_stop_961_);
if (v___x_964_ == 0)
{
lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; size_t v___x_968_; size_t v___x_969_; 
v___x_965_ = lean_array_uget_borrowed(v_as_959_, v_i_960_);
lean_inc(v___x_965_);
v___x_966_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures(v___x_965_);
v___x_967_ = l_Array_append___redArg(v_b_962_, v___x_966_);
lean_dec_ref(v___x_966_);
v___x_968_ = ((size_t)1ULL);
v___x_969_ = lean_usize_add(v_i_960_, v___x_968_);
v_i_960_ = v___x_969_;
v_b_962_ = v___x_967_;
goto _start;
}
else
{
return v_b_962_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0(lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_){
_start:
{
lean_object* v___y_978_; lean_object* v_val_981_; lean_object* v___y_991_; lean_object* v_cls_992_; lean_object* v_result_x3f_993_; lean_object* v___x_994_; uint8_t v___x_995_; 
v_cls_992_ = lean_ctor_get(v___y_971_, 0);
v_result_x3f_993_ = lean_ctor_get(v___y_971_, 1);
v___x_994_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3));
v___x_995_ = lean_name_eq(v_cls_992_, v___x_994_);
if (v___x_995_ == 0)
{
lean_object* v___x_996_; uint8_t v___x_997_; 
v___x_996_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4));
v___x_997_ = l_Lean_Name_isPrefixOf(v___x_996_, v_cls_992_);
if (v___x_997_ == 0)
{
lean_object* v___x_998_; 
lean_dec_ref(v___y_972_);
v___x_998_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__1));
return v___x_998_;
}
else
{
if (lean_obj_tag(v_result_x3f_993_) == 1)
{
lean_object* v_val_999_; uint8_t v___x_1000_; 
v_val_999_ = lean_ctor_get(v_result_x3f_993_, 0);
v___x_1000_ = lean_unbox(v_val_999_);
if (v___x_1000_ == 1)
{
lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; uint8_t v___x_1004_; 
v___x_1001_ = lean_unsigned_to_nat(0u);
v___x_1002_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__1));
v___x_1003_ = lean_array_get_size(v___y_973_);
v___x_1004_ = lean_nat_dec_lt(v___x_1001_, v___x_1003_);
if (v___x_1004_ == 0)
{
v_val_981_ = v___x_1002_;
goto v___jp_980_;
}
else
{
uint8_t v___x_1005_; 
v___x_1005_ = lean_nat_dec_le(v___x_1003_, v___x_1003_);
if (v___x_1005_ == 0)
{
if (v___x_1004_ == 0)
{
v_val_981_ = v___x_1002_;
goto v___jp_980_;
}
else
{
size_t v___x_1006_; size_t v___x_1007_; lean_object* v___x_1008_; 
v___x_1006_ = ((size_t)0ULL);
v___x_1007_ = lean_usize_of_nat(v___x_1003_);
v___x_1008_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__0(v___y_973_, v___x_1006_, v___x_1007_, v___x_1002_);
v___y_991_ = v___x_1008_;
goto v___jp_990_;
}
}
else
{
size_t v___x_1009_; size_t v___x_1010_; lean_object* v___x_1011_; 
v___x_1009_ = ((size_t)0ULL);
v___x_1010_ = lean_usize_of_nat(v___x_1003_);
v___x_1011_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__0(v___y_973_, v___x_1009_, v___x_1010_, v___x_1002_);
v___y_991_ = v___x_1011_;
goto v___jp_990_;
}
}
}
else
{
lean_dec_ref(v___y_972_);
goto v___jp_975_;
}
}
else
{
lean_dec_ref(v___y_972_);
goto v___jp_975_;
}
}
}
else
{
lean_object* v___x_1012_; 
lean_dec_ref(v___y_972_);
v___x_1012_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__0));
return v___x_1012_;
}
v___jp_975_:
{
lean_object* v___x_976_; 
v___x_976_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__0));
return v___x_976_;
}
v___jp_977_:
{
lean_object* v___x_979_; 
v___x_979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_979_, 0, v___y_978_);
return v___x_979_;
}
v___jp_980_:
{
lean_object* v___x_982_; lean_object* v___x_983_; uint8_t v___x_984_; 
v___x_982_ = lean_array_get_size(v_val_981_);
v___x_983_ = lean_unsigned_to_nat(0u);
v___x_984_ = lean_nat_dec_eq(v___x_982_, v___x_983_);
if (v___x_984_ == 0)
{
lean_object* v___x_985_; 
lean_dec_ref(v___y_972_);
v___x_985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_985_, 0, v_val_981_);
v___y_978_ = v___x_985_;
goto v___jp_977_;
}
else
{
lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; 
lean_dec_ref(v_val_981_);
v___x_986_ = lean_unsigned_to_nat(1u);
v___x_987_ = lean_mk_empty_array_with_capacity(v___x_986_);
v___x_988_ = lean_array_push(v___x_987_, v___y_972_);
v___x_989_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_989_, 0, v___x_988_);
v___y_978_ = v___x_989_;
goto v___jp_977_;
}
}
v___jp_990_:
{
v_val_981_ = v___y_991_;
goto v___jp_980_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___boxed(lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_){
_start:
{
lean_object* v_res_1017_; 
v_res_1017_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0(v___y_1013_, v___y_1014_, v___y_1015_);
lean_dec_ref(v___y_1015_);
lean_dec_ref(v___y_1013_);
return v_res_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures(lean_object* v_msg_1018_){
_start:
{
lean_object* v___f_1020_; lean_object* v___f_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; 
v___f_1020_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__0));
v___f_1021_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___boxed), 4, 0);
v___x_1022_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__1));
v___x_1023_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v___f_1021_, v___x_1022_, v___f_1020_, v_msg_1018_);
return v___x_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___boxed(lean_object* v_msg_1024_, lean_object* v_a_1025_){
_start:
{
lean_object* v_res_1026_; 
v_res_1026_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures(v_msg_1024_);
return v_res_1026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__0___boxed(lean_object* v_as_1027_, lean_object* v_i_1028_, lean_object* v_stop_1029_, lean_object* v_b_1030_, lean_object* v___y_1031_){
_start:
{
size_t v_i_boxed_1032_; size_t v_stop_boxed_1033_; lean_object* v_res_1034_; 
v_i_boxed_1032_ = lean_unbox_usize(v_i_1028_);
lean_dec(v_i_1028_);
v_stop_boxed_1033_ = lean_unbox_usize(v_stop_1029_);
lean_dec(v_stop_1029_);
v_res_1034_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__0(v_as_1027_, v_i_boxed_1032_, v_stop_boxed_1033_, v_b_1030_);
lean_dec_ref(v_as_1027_);
return v_res_1034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1(lean_object* v_00_u03b1_1035_, lean_object* v_onTrace_1036_, lean_object* v_empty_1037_, lean_object* v_combine_1038_, lean_object* v_a_1039_){
_start:
{
lean_object* v___x_1041_; 
v___x_1041_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v_onTrace_1036_, v_empty_1037_, v_combine_1038_, v_a_1039_);
return v___x_1041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___boxed(lean_object* v_00_u03b1_1042_, lean_object* v_onTrace_1043_, lean_object* v_empty_1044_, lean_object* v_combine_1045_, lean_object* v_a_1046_, lean_object* v___y_1047_){
_start:
{
lean_object* v_res_1048_; 
v_res_1048_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1(v_00_u03b1_1042_, v_onTrace_1043_, v_empty_1044_, v_combine_1045_, v_a_1046_);
return v_res_1048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1(lean_object* v_00_u03b1_1049_, lean_object* v_onTrace_1050_, lean_object* v_empty_1051_, lean_object* v_combine_1052_, lean_object* v_as_1053_, size_t v_sz_1054_, size_t v_i_1055_, lean_object* v_b_1056_){
_start:
{
lean_object* v___x_1058_; 
v___x_1058_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___redArg(v_onTrace_1050_, v_empty_1051_, v_combine_1052_, v_as_1053_, v_sz_1054_, v_i_1055_, v_b_1056_);
return v___x_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1059_, lean_object* v_onTrace_1060_, lean_object* v_empty_1061_, lean_object* v_combine_1062_, lean_object* v_as_1063_, lean_object* v_sz_1064_, lean_object* v_i_1065_, lean_object* v_b_1066_, lean_object* v___y_1067_){
_start:
{
size_t v_sz_boxed_1068_; size_t v_i_boxed_1069_; lean_object* v_res_1070_; 
v_sz_boxed_1068_ = lean_unbox_usize(v_sz_1064_);
lean_dec(v_sz_1064_);
v_i_boxed_1069_ = lean_unbox_usize(v_i_1065_);
lean_dec(v_i_1065_);
v_res_1070_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1_spec__1(v_00_u03b1_1059_, v_onTrace_1060_, v_empty_1061_, v_combine_1062_, v_as_1063_, v_sz_boxed_1068_, v_i_boxed_1069_, v_b_1066_);
lean_dec_ref(v_as_1063_);
return v_res_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2_spec__6___redArg(lean_object* v_x_1071_, lean_object* v_x_1072_){
_start:
{
if (lean_obj_tag(v_x_1072_) == 0)
{
return v_x_1071_;
}
else
{
lean_object* v_key_1073_; lean_object* v_value_1074_; lean_object* v_tail_1075_; lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1098_; 
v_key_1073_ = lean_ctor_get(v_x_1072_, 0);
v_value_1074_ = lean_ctor_get(v_x_1072_, 1);
v_tail_1075_ = lean_ctor_get(v_x_1072_, 2);
v_isSharedCheck_1098_ = !lean_is_exclusive(v_x_1072_);
if (v_isSharedCheck_1098_ == 0)
{
v___x_1077_ = v_x_1072_;
v_isShared_1078_ = v_isSharedCheck_1098_;
goto v_resetjp_1076_;
}
else
{
lean_inc(v_tail_1075_);
lean_inc(v_value_1074_);
lean_inc(v_key_1073_);
lean_dec(v_x_1072_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1098_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
lean_object* v___x_1079_; uint64_t v___x_1080_; uint64_t v___x_1081_; uint64_t v___x_1082_; uint64_t v_fold_1083_; uint64_t v___x_1084_; uint64_t v___x_1085_; uint64_t v___x_1086_; size_t v___x_1087_; size_t v___x_1088_; size_t v___x_1089_; size_t v___x_1090_; size_t v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1094_; 
v___x_1079_ = lean_array_get_size(v_x_1071_);
v___x_1080_ = lean_string_hash(v_key_1073_);
v___x_1081_ = 32ULL;
v___x_1082_ = lean_uint64_shift_right(v___x_1080_, v___x_1081_);
v_fold_1083_ = lean_uint64_xor(v___x_1080_, v___x_1082_);
v___x_1084_ = 16ULL;
v___x_1085_ = lean_uint64_shift_right(v_fold_1083_, v___x_1084_);
v___x_1086_ = lean_uint64_xor(v_fold_1083_, v___x_1085_);
v___x_1087_ = lean_uint64_to_usize(v___x_1086_);
v___x_1088_ = lean_usize_of_nat(v___x_1079_);
v___x_1089_ = ((size_t)1ULL);
v___x_1090_ = lean_usize_sub(v___x_1088_, v___x_1089_);
v___x_1091_ = lean_usize_land(v___x_1087_, v___x_1090_);
v___x_1092_ = lean_array_uget_borrowed(v_x_1071_, v___x_1091_);
lean_inc(v___x_1092_);
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 2, v___x_1092_);
v___x_1094_ = v___x_1077_;
goto v_reusejp_1093_;
}
else
{
lean_object* v_reuseFailAlloc_1097_; 
v_reuseFailAlloc_1097_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1097_, 0, v_key_1073_);
lean_ctor_set(v_reuseFailAlloc_1097_, 1, v_value_1074_);
lean_ctor_set(v_reuseFailAlloc_1097_, 2, v___x_1092_);
v___x_1094_ = v_reuseFailAlloc_1097_;
goto v_reusejp_1093_;
}
v_reusejp_1093_:
{
lean_object* v___x_1095_; 
v___x_1095_ = lean_array_uset(v_x_1071_, v___x_1091_, v___x_1094_);
v_x_1071_ = v___x_1095_;
v_x_1072_ = v_tail_1075_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2___redArg(lean_object* v_i_1099_, lean_object* v_source_1100_, lean_object* v_target_1101_){
_start:
{
lean_object* v___x_1102_; uint8_t v___x_1103_; 
v___x_1102_ = lean_array_get_size(v_source_1100_);
v___x_1103_ = lean_nat_dec_lt(v_i_1099_, v___x_1102_);
if (v___x_1103_ == 0)
{
lean_dec_ref(v_source_1100_);
lean_dec(v_i_1099_);
return v_target_1101_;
}
else
{
lean_object* v_es_1104_; lean_object* v___x_1105_; lean_object* v_source_1106_; lean_object* v_target_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; 
v_es_1104_ = lean_array_fget(v_source_1100_, v_i_1099_);
v___x_1105_ = lean_box(0);
v_source_1106_ = lean_array_fset(v_source_1100_, v_i_1099_, v___x_1105_);
v_target_1107_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2_spec__6___redArg(v_target_1101_, v_es_1104_);
v___x_1108_ = lean_unsigned_to_nat(1u);
v___x_1109_ = lean_nat_add(v_i_1099_, v___x_1108_);
lean_dec(v_i_1099_);
v_i_1099_ = v___x_1109_;
v_source_1100_ = v_source_1106_;
v_target_1101_ = v_target_1107_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1___redArg(lean_object* v_data_1111_){
_start:
{
lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v_nbuckets_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; 
v___x_1112_ = lean_array_get_size(v_data_1111_);
v___x_1113_ = lean_unsigned_to_nat(2u);
v_nbuckets_1114_ = lean_nat_mul(v___x_1112_, v___x_1113_);
v___x_1115_ = lean_unsigned_to_nat(0u);
v___x_1116_ = lean_box(0);
v___x_1117_ = lean_mk_array(v_nbuckets_1114_, v___x_1116_);
v___x_1118_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2___redArg(v___x_1115_, v_data_1111_, v___x_1117_);
return v___x_1118_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg(lean_object* v_a_1119_, lean_object* v_x_1120_){
_start:
{
if (lean_obj_tag(v_x_1120_) == 0)
{
uint8_t v___x_1121_; 
v___x_1121_ = 0;
return v___x_1121_;
}
else
{
lean_object* v_key_1122_; lean_object* v_tail_1123_; uint8_t v___x_1124_; 
v_key_1122_ = lean_ctor_get(v_x_1120_, 0);
v_tail_1123_ = lean_ctor_get(v_x_1120_, 2);
v___x_1124_ = lean_string_dec_eq(v_key_1122_, v_a_1119_);
if (v___x_1124_ == 0)
{
v_x_1120_ = v_tail_1123_;
goto _start;
}
else
{
return v___x_1124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg___boxed(lean_object* v_a_1126_, lean_object* v_x_1127_){
_start:
{
uint8_t v_res_1128_; lean_object* v_r_1129_; 
v_res_1128_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg(v_a_1126_, v_x_1127_);
lean_dec(v_x_1127_);
lean_dec_ref(v_a_1126_);
v_r_1129_ = lean_box(v_res_1128_);
return v_r_1129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0___redArg(lean_object* v_m_1130_, lean_object* v_a_1131_, lean_object* v_b_1132_){
_start:
{
lean_object* v_size_1133_; lean_object* v_buckets_1134_; lean_object* v___x_1135_; uint64_t v___x_1136_; uint64_t v___x_1137_; uint64_t v___x_1138_; uint64_t v_fold_1139_; uint64_t v___x_1140_; uint64_t v___x_1141_; uint64_t v___x_1142_; size_t v___x_1143_; size_t v___x_1144_; size_t v___x_1145_; size_t v___x_1146_; size_t v___x_1147_; lean_object* v_bkt_1148_; uint8_t v___x_1149_; 
v_size_1133_ = lean_ctor_get(v_m_1130_, 0);
v_buckets_1134_ = lean_ctor_get(v_m_1130_, 1);
v___x_1135_ = lean_array_get_size(v_buckets_1134_);
v___x_1136_ = lean_string_hash(v_a_1131_);
v___x_1137_ = 32ULL;
v___x_1138_ = lean_uint64_shift_right(v___x_1136_, v___x_1137_);
v_fold_1139_ = lean_uint64_xor(v___x_1136_, v___x_1138_);
v___x_1140_ = 16ULL;
v___x_1141_ = lean_uint64_shift_right(v_fold_1139_, v___x_1140_);
v___x_1142_ = lean_uint64_xor(v_fold_1139_, v___x_1141_);
v___x_1143_ = lean_uint64_to_usize(v___x_1142_);
v___x_1144_ = lean_usize_of_nat(v___x_1135_);
v___x_1145_ = ((size_t)1ULL);
v___x_1146_ = lean_usize_sub(v___x_1144_, v___x_1145_);
v___x_1147_ = lean_usize_land(v___x_1143_, v___x_1146_);
v_bkt_1148_ = lean_array_uget_borrowed(v_buckets_1134_, v___x_1147_);
v___x_1149_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg(v_a_1131_, v_bkt_1148_);
if (v___x_1149_ == 0)
{
lean_object* v___x_1151_; uint8_t v_isShared_1152_; uint8_t v_isSharedCheck_1170_; 
lean_inc_ref(v_buckets_1134_);
lean_inc(v_size_1133_);
v_isSharedCheck_1170_ = !lean_is_exclusive(v_m_1130_);
if (v_isSharedCheck_1170_ == 0)
{
lean_object* v_unused_1171_; lean_object* v_unused_1172_; 
v_unused_1171_ = lean_ctor_get(v_m_1130_, 1);
lean_dec(v_unused_1171_);
v_unused_1172_ = lean_ctor_get(v_m_1130_, 0);
lean_dec(v_unused_1172_);
v___x_1151_ = v_m_1130_;
v_isShared_1152_ = v_isSharedCheck_1170_;
goto v_resetjp_1150_;
}
else
{
lean_dec(v_m_1130_);
v___x_1151_ = lean_box(0);
v_isShared_1152_ = v_isSharedCheck_1170_;
goto v_resetjp_1150_;
}
v_resetjp_1150_:
{
lean_object* v___x_1153_; lean_object* v_size_x27_1154_; lean_object* v___x_1155_; lean_object* v_buckets_x27_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; uint8_t v___x_1162_; 
v___x_1153_ = lean_unsigned_to_nat(1u);
v_size_x27_1154_ = lean_nat_add(v_size_1133_, v___x_1153_);
lean_dec(v_size_1133_);
lean_inc(v_bkt_1148_);
v___x_1155_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1155_, 0, v_a_1131_);
lean_ctor_set(v___x_1155_, 1, v_b_1132_);
lean_ctor_set(v___x_1155_, 2, v_bkt_1148_);
v_buckets_x27_1156_ = lean_array_uset(v_buckets_1134_, v___x_1147_, v___x_1155_);
v___x_1157_ = lean_unsigned_to_nat(4u);
v___x_1158_ = lean_nat_mul(v_size_x27_1154_, v___x_1157_);
v___x_1159_ = lean_unsigned_to_nat(3u);
v___x_1160_ = lean_nat_div(v___x_1158_, v___x_1159_);
lean_dec(v___x_1158_);
v___x_1161_ = lean_array_get_size(v_buckets_x27_1156_);
v___x_1162_ = lean_nat_dec_le(v___x_1160_, v___x_1161_);
lean_dec(v___x_1160_);
if (v___x_1162_ == 0)
{
lean_object* v_val_1163_; lean_object* v___x_1165_; 
v_val_1163_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1___redArg(v_buckets_x27_1156_);
if (v_isShared_1152_ == 0)
{
lean_ctor_set(v___x_1151_, 1, v_val_1163_);
lean_ctor_set(v___x_1151_, 0, v_size_x27_1154_);
v___x_1165_ = v___x_1151_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v_size_x27_1154_);
lean_ctor_set(v_reuseFailAlloc_1166_, 1, v_val_1163_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
else
{
lean_object* v___x_1168_; 
if (v_isShared_1152_ == 0)
{
lean_ctor_set(v___x_1151_, 1, v_buckets_x27_1156_);
lean_ctor_set(v___x_1151_, 0, v_size_x27_1154_);
v___x_1168_ = v___x_1151_;
goto v_reusejp_1167_;
}
else
{
lean_object* v_reuseFailAlloc_1169_; 
v_reuseFailAlloc_1169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1169_, 0, v_size_x27_1154_);
lean_ctor_set(v_reuseFailAlloc_1169_, 1, v_buckets_x27_1156_);
v___x_1168_ = v_reuseFailAlloc_1169_;
goto v_reusejp_1167_;
}
v_reusejp_1167_:
{
return v___x_1168_;
}
}
}
}
else
{
lean_dec(v_b_1132_);
lean_dec_ref(v_a_1131_);
return v_m_1130_;
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; 
v___x_1175_ = lean_box(0);
v___x_1176_ = lean_unsigned_to_nat(16u);
v___x_1177_ = lean_mk_array(v___x_1176_, v___x_1175_);
return v___x_1177_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2(void){
_start:
{
lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; 
v___x_1178_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__1);
v___x_1179_ = lean_unsigned_to_nat(0u);
v___x_1180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1180_, 0, v___x_1179_);
lean_ctor_set(v___x_1180_, 1, v___x_1178_);
return v___x_1180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0(lean_object* v_pred_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_){
_start:
{
lean_object* v_cls_1190_; lean_object* v_result_x3f_1191_; lean_object* v___x_1192_; uint8_t v___x_1193_; 
v_cls_1190_ = lean_ctor_get(v___y_1184_, 0);
lean_inc(v_cls_1190_);
v_result_x3f_1191_ = lean_ctor_get(v___y_1184_, 1);
lean_inc(v_result_x3f_1191_);
lean_dec_ref(v___y_1184_);
v___x_1192_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3));
v___x_1193_ = lean_name_eq(v_cls_1190_, v___x_1192_);
if (v___x_1193_ == 0)
{
lean_object* v___x_1194_; uint8_t v___x_1195_; 
v___x_1194_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4));
v___x_1195_ = l_Lean_Name_isPrefixOf(v___x_1194_, v_cls_1190_);
lean_dec(v_cls_1190_);
if (v___x_1195_ == 0)
{
lean_object* v___x_1196_; 
lean_dec(v_result_x3f_1191_);
lean_dec_ref(v___y_1185_);
lean_dec_ref(v_pred_1183_);
v___x_1196_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__0));
return v___x_1196_;
}
else
{
if (lean_obj_tag(v_result_x3f_1191_) == 1)
{
lean_object* v_val_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1211_; 
v_val_1197_ = lean_ctor_get(v_result_x3f_1191_, 0);
v_isSharedCheck_1211_ = !lean_is_exclusive(v_result_x3f_1191_);
if (v_isSharedCheck_1211_ == 0)
{
v___x_1199_ = v_result_x3f_1191_;
v_isShared_1200_ = v_isSharedCheck_1211_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_val_1197_);
lean_dec(v_result_x3f_1191_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1211_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___x_1201_; uint8_t v___x_1202_; 
v___x_1201_ = lean_apply_1(v_pred_1183_, v_val_1197_);
v___x_1202_ = lean_unbox(v___x_1201_);
if (v___x_1202_ == 0)
{
lean_del_object(v___x_1199_);
lean_dec_ref(v___y_1185_);
goto v___jp_1188_;
}
else
{
lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1208_; 
v___x_1203_ = l_Lean_MessageData_toString(v___y_1185_);
v___x_1204_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2);
v___x_1205_ = lean_box(0);
v___x_1206_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0___redArg(v___x_1204_, v___x_1203_, v___x_1205_);
if (v_isShared_1200_ == 0)
{
lean_ctor_set(v___x_1199_, 0, v___x_1206_);
v___x_1208_ = v___x_1199_;
goto v_reusejp_1207_;
}
else
{
lean_object* v_reuseFailAlloc_1210_; 
v_reuseFailAlloc_1210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1210_, 0, v___x_1206_);
v___x_1208_ = v_reuseFailAlloc_1210_;
goto v_reusejp_1207_;
}
v_reusejp_1207_:
{
lean_object* v___x_1209_; 
v___x_1209_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1209_, 0, v___x_1208_);
return v___x_1209_;
}
}
}
}
else
{
lean_dec(v_result_x3f_1191_);
lean_dec_ref(v___y_1185_);
lean_dec_ref(v_pred_1183_);
goto v___jp_1188_;
}
}
}
else
{
lean_object* v___x_1212_; 
lean_dec(v_result_x3f_1191_);
lean_dec(v_cls_1190_);
lean_dec_ref(v___y_1185_);
lean_dec_ref(v_pred_1183_);
v___x_1212_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__3));
return v___x_1212_;
}
v___jp_1188_:
{
lean_object* v___x_1189_; 
v___x_1189_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__0));
return v___x_1189_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___boxed(lean_object* v_pred_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_){
_start:
{
lean_object* v_res_1218_; 
v_res_1218_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0(v_pred_1213_, v___y_1214_, v___y_1215_, v___y_1216_);
lean_dec_ref(v___y_1216_);
return v_res_1218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__2(lean_object* v_a_1219_, lean_object* v_a_1220_){
_start:
{
if (lean_obj_tag(v_a_1219_) == 0)
{
lean_object* v___x_1221_; 
v___x_1221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1221_, 0, v_a_1220_);
return v___x_1221_;
}
else
{
lean_object* v_key_1222_; lean_object* v_value_1223_; lean_object* v_tail_1224_; lean_object* v_r_1225_; 
v_key_1222_ = lean_ctor_get(v_a_1219_, 0);
lean_inc(v_key_1222_);
v_value_1223_ = lean_ctor_get(v_a_1219_, 1);
lean_inc(v_value_1223_);
v_tail_1224_ = lean_ctor_get(v_a_1219_, 2);
lean_inc(v_tail_1224_);
lean_dec_ref_known(v_a_1219_, 3);
v_r_1225_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0___redArg(v_a_1220_, v_key_1222_, v_value_1223_);
v_a_1219_ = v_tail_1224_;
v_a_1220_ = v_r_1225_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3(lean_object* v_as_1227_, size_t v_sz_1228_, size_t v_i_1229_, lean_object* v_b_1230_){
_start:
{
uint8_t v___x_1231_; 
v___x_1231_ = lean_usize_dec_lt(v_i_1229_, v_sz_1228_);
if (v___x_1231_ == 0)
{
return v_b_1230_;
}
else
{
lean_object* v_a_1232_; lean_object* v___x_1233_; 
v_a_1232_ = lean_array_uget_borrowed(v_as_1227_, v_i_1229_);
lean_inc(v_a_1232_);
v___x_1233_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__2(v_a_1232_, v_b_1230_);
if (lean_obj_tag(v___x_1233_) == 0)
{
lean_object* v_a_1234_; 
v_a_1234_ = lean_ctor_get(v___x_1233_, 0);
lean_inc(v_a_1234_);
lean_dec_ref_known(v___x_1233_, 1);
return v_a_1234_;
}
else
{
lean_object* v_a_1235_; size_t v___x_1236_; size_t v___x_1237_; 
v_a_1235_ = lean_ctor_get(v___x_1233_, 0);
lean_inc(v_a_1235_);
lean_dec_ref_known(v___x_1233_, 1);
v___x_1236_ = ((size_t)1ULL);
v___x_1237_ = lean_usize_add(v_i_1229_, v___x_1236_);
v_i_1229_ = v___x_1237_;
v_b_1230_ = v_a_1235_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3___boxed(lean_object* v_as_1239_, lean_object* v_sz_1240_, lean_object* v_i_1241_, lean_object* v_b_1242_){
_start:
{
size_t v_sz_boxed_1243_; size_t v_i_boxed_1244_; lean_object* v_res_1245_; 
v_sz_boxed_1243_ = lean_unbox_usize(v_sz_1240_);
lean_dec(v_sz_1240_);
v_i_boxed_1244_ = lean_unbox_usize(v_i_1241_);
lean_dec(v_i_1241_);
v_res_1245_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3(v_as_1239_, v_sz_boxed_1243_, v_i_boxed_1244_, v_b_1242_);
lean_dec_ref(v_as_1239_);
return v_res_1245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3_spec__5___redArg(lean_object* v_a_1246_, lean_object* v_b_1247_, lean_object* v_x_1248_){
_start:
{
if (lean_obj_tag(v_x_1248_) == 0)
{
lean_dec(v_b_1247_);
lean_dec_ref(v_a_1246_);
return v_x_1248_;
}
else
{
lean_object* v_key_1249_; lean_object* v_value_1250_; lean_object* v_tail_1251_; lean_object* v___x_1253_; uint8_t v_isShared_1254_; uint8_t v_isSharedCheck_1263_; 
v_key_1249_ = lean_ctor_get(v_x_1248_, 0);
v_value_1250_ = lean_ctor_get(v_x_1248_, 1);
v_tail_1251_ = lean_ctor_get(v_x_1248_, 2);
v_isSharedCheck_1263_ = !lean_is_exclusive(v_x_1248_);
if (v_isSharedCheck_1263_ == 0)
{
v___x_1253_ = v_x_1248_;
v_isShared_1254_ = v_isSharedCheck_1263_;
goto v_resetjp_1252_;
}
else
{
lean_inc(v_tail_1251_);
lean_inc(v_value_1250_);
lean_inc(v_key_1249_);
lean_dec(v_x_1248_);
v___x_1253_ = lean_box(0);
v_isShared_1254_ = v_isSharedCheck_1263_;
goto v_resetjp_1252_;
}
v_resetjp_1252_:
{
uint8_t v___x_1255_; 
v___x_1255_ = lean_string_dec_eq(v_key_1249_, v_a_1246_);
if (v___x_1255_ == 0)
{
lean_object* v___x_1256_; lean_object* v___x_1258_; 
v___x_1256_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3_spec__5___redArg(v_a_1246_, v_b_1247_, v_tail_1251_);
if (v_isShared_1254_ == 0)
{
lean_ctor_set(v___x_1253_, 2, v___x_1256_);
v___x_1258_ = v___x_1253_;
goto v_reusejp_1257_;
}
else
{
lean_object* v_reuseFailAlloc_1259_; 
v_reuseFailAlloc_1259_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1259_, 0, v_key_1249_);
lean_ctor_set(v_reuseFailAlloc_1259_, 1, v_value_1250_);
lean_ctor_set(v_reuseFailAlloc_1259_, 2, v___x_1256_);
v___x_1258_ = v_reuseFailAlloc_1259_;
goto v_reusejp_1257_;
}
v_reusejp_1257_:
{
return v___x_1258_;
}
}
else
{
lean_object* v___x_1261_; 
lean_dec(v_value_1250_);
lean_dec(v_key_1249_);
if (v_isShared_1254_ == 0)
{
lean_ctor_set(v___x_1253_, 1, v_b_1247_);
lean_ctor_set(v___x_1253_, 0, v_a_1246_);
v___x_1261_ = v___x_1253_;
goto v_reusejp_1260_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v_a_1246_);
lean_ctor_set(v_reuseFailAlloc_1262_, 1, v_b_1247_);
lean_ctor_set(v_reuseFailAlloc_1262_, 2, v_tail_1251_);
v___x_1261_ = v_reuseFailAlloc_1262_;
goto v_reusejp_1260_;
}
v_reusejp_1260_:
{
return v___x_1261_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3___redArg(lean_object* v_m_1264_, lean_object* v_a_1265_, lean_object* v_b_1266_){
_start:
{
lean_object* v_size_1267_; lean_object* v_buckets_1268_; lean_object* v___x_1270_; uint8_t v_isShared_1271_; uint8_t v_isSharedCheck_1311_; 
v_size_1267_ = lean_ctor_get(v_m_1264_, 0);
v_buckets_1268_ = lean_ctor_get(v_m_1264_, 1);
v_isSharedCheck_1311_ = !lean_is_exclusive(v_m_1264_);
if (v_isSharedCheck_1311_ == 0)
{
v___x_1270_ = v_m_1264_;
v_isShared_1271_ = v_isSharedCheck_1311_;
goto v_resetjp_1269_;
}
else
{
lean_inc(v_buckets_1268_);
lean_inc(v_size_1267_);
lean_dec(v_m_1264_);
v___x_1270_ = lean_box(0);
v_isShared_1271_ = v_isSharedCheck_1311_;
goto v_resetjp_1269_;
}
v_resetjp_1269_:
{
lean_object* v___x_1272_; uint64_t v___x_1273_; uint64_t v___x_1274_; uint64_t v___x_1275_; uint64_t v_fold_1276_; uint64_t v___x_1277_; uint64_t v___x_1278_; uint64_t v___x_1279_; size_t v___x_1280_; size_t v___x_1281_; size_t v___x_1282_; size_t v___x_1283_; size_t v___x_1284_; lean_object* v_bkt_1285_; uint8_t v___x_1286_; 
v___x_1272_ = lean_array_get_size(v_buckets_1268_);
v___x_1273_ = lean_string_hash(v_a_1265_);
v___x_1274_ = 32ULL;
v___x_1275_ = lean_uint64_shift_right(v___x_1273_, v___x_1274_);
v_fold_1276_ = lean_uint64_xor(v___x_1273_, v___x_1275_);
v___x_1277_ = 16ULL;
v___x_1278_ = lean_uint64_shift_right(v_fold_1276_, v___x_1277_);
v___x_1279_ = lean_uint64_xor(v_fold_1276_, v___x_1278_);
v___x_1280_ = lean_uint64_to_usize(v___x_1279_);
v___x_1281_ = lean_usize_of_nat(v___x_1272_);
v___x_1282_ = ((size_t)1ULL);
v___x_1283_ = lean_usize_sub(v___x_1281_, v___x_1282_);
v___x_1284_ = lean_usize_land(v___x_1280_, v___x_1283_);
v_bkt_1285_ = lean_array_uget_borrowed(v_buckets_1268_, v___x_1284_);
v___x_1286_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg(v_a_1265_, v_bkt_1285_);
if (v___x_1286_ == 0)
{
lean_object* v___x_1287_; lean_object* v_size_x27_1288_; lean_object* v___x_1289_; lean_object* v_buckets_x27_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; uint8_t v___x_1296_; 
v___x_1287_ = lean_unsigned_to_nat(1u);
v_size_x27_1288_ = lean_nat_add(v_size_1267_, v___x_1287_);
lean_dec(v_size_1267_);
lean_inc(v_bkt_1285_);
v___x_1289_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1289_, 0, v_a_1265_);
lean_ctor_set(v___x_1289_, 1, v_b_1266_);
lean_ctor_set(v___x_1289_, 2, v_bkt_1285_);
v_buckets_x27_1290_ = lean_array_uset(v_buckets_1268_, v___x_1284_, v___x_1289_);
v___x_1291_ = lean_unsigned_to_nat(4u);
v___x_1292_ = lean_nat_mul(v_size_x27_1288_, v___x_1291_);
v___x_1293_ = lean_unsigned_to_nat(3u);
v___x_1294_ = lean_nat_div(v___x_1292_, v___x_1293_);
lean_dec(v___x_1292_);
v___x_1295_ = lean_array_get_size(v_buckets_x27_1290_);
v___x_1296_ = lean_nat_dec_le(v___x_1294_, v___x_1295_);
lean_dec(v___x_1294_);
if (v___x_1296_ == 0)
{
lean_object* v_val_1297_; lean_object* v___x_1299_; 
v_val_1297_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1___redArg(v_buckets_x27_1290_);
if (v_isShared_1271_ == 0)
{
lean_ctor_set(v___x_1270_, 1, v_val_1297_);
lean_ctor_set(v___x_1270_, 0, v_size_x27_1288_);
v___x_1299_ = v___x_1270_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v_size_x27_1288_);
lean_ctor_set(v_reuseFailAlloc_1300_, 1, v_val_1297_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
else
{
lean_object* v___x_1302_; 
if (v_isShared_1271_ == 0)
{
lean_ctor_set(v___x_1270_, 1, v_buckets_x27_1290_);
lean_ctor_set(v___x_1270_, 0, v_size_x27_1288_);
v___x_1302_ = v___x_1270_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_size_x27_1288_);
lean_ctor_set(v_reuseFailAlloc_1303_, 1, v_buckets_x27_1290_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
}
else
{
lean_object* v___x_1304_; lean_object* v_buckets_x27_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1309_; 
lean_inc(v_bkt_1285_);
v___x_1304_ = lean_box(0);
v_buckets_x27_1305_ = lean_array_uset(v_buckets_1268_, v___x_1284_, v___x_1304_);
v___x_1306_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3_spec__5___redArg(v_a_1265_, v_b_1266_, v_bkt_1285_);
v___x_1307_ = lean_array_uset(v_buckets_x27_1305_, v___x_1284_, v___x_1306_);
if (v_isShared_1271_ == 0)
{
lean_ctor_set(v___x_1270_, 1, v___x_1307_);
v___x_1309_ = v___x_1270_;
goto v_reusejp_1308_;
}
else
{
lean_object* v_reuseFailAlloc_1310_; 
v_reuseFailAlloc_1310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1310_, 0, v_size_1267_);
lean_ctor_set(v_reuseFailAlloc_1310_, 1, v___x_1307_);
v___x_1309_ = v_reuseFailAlloc_1310_;
goto v_reusejp_1308_;
}
v_reusejp_1308_:
{
return v___x_1309_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__4(lean_object* v_a_1312_, lean_object* v_a_1313_){
_start:
{
if (lean_obj_tag(v_a_1312_) == 0)
{
lean_object* v___x_1314_; 
v___x_1314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1314_, 0, v_a_1313_);
return v___x_1314_;
}
else
{
lean_object* v_key_1315_; lean_object* v_value_1316_; lean_object* v_tail_1317_; lean_object* v_r_1318_; 
v_key_1315_ = lean_ctor_get(v_a_1312_, 0);
lean_inc(v_key_1315_);
v_value_1316_ = lean_ctor_get(v_a_1312_, 1);
lean_inc(v_value_1316_);
v_tail_1317_ = lean_ctor_get(v_a_1312_, 2);
lean_inc(v_tail_1317_);
lean_dec_ref_known(v_a_1312_, 3);
v_r_1318_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3___redArg(v_a_1313_, v_key_1315_, v_value_1316_);
v_a_1312_ = v_tail_1317_;
v_a_1313_ = v_r_1318_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__5(lean_object* v_as_1320_, size_t v_sz_1321_, size_t v_i_1322_, lean_object* v_b_1323_){
_start:
{
uint8_t v___x_1324_; 
v___x_1324_ = lean_usize_dec_lt(v_i_1322_, v_sz_1321_);
if (v___x_1324_ == 0)
{
return v_b_1323_;
}
else
{
lean_object* v_a_1325_; lean_object* v___x_1326_; 
v_a_1325_ = lean_array_uget_borrowed(v_as_1320_, v_i_1322_);
lean_inc(v_a_1325_);
v___x_1326_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__4(v_a_1325_, v_b_1323_);
if (lean_obj_tag(v___x_1326_) == 0)
{
lean_object* v_a_1327_; 
v_a_1327_ = lean_ctor_get(v___x_1326_, 0);
lean_inc(v_a_1327_);
lean_dec_ref_known(v___x_1326_, 1);
return v_a_1327_;
}
else
{
lean_object* v_a_1328_; size_t v___x_1329_; size_t v___x_1330_; 
v_a_1328_ = lean_ctor_get(v___x_1326_, 0);
lean_inc(v_a_1328_);
lean_dec_ref_known(v___x_1326_, 1);
v___x_1329_ = ((size_t)1ULL);
v___x_1330_ = lean_usize_add(v_i_1322_, v___x_1329_);
v_i_1322_ = v___x_1330_;
v_b_1323_ = v_a_1328_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__5___boxed(lean_object* v_as_1332_, lean_object* v_sz_1333_, lean_object* v_i_1334_, lean_object* v_b_1335_){
_start:
{
size_t v_sz_boxed_1336_; size_t v_i_boxed_1337_; lean_object* v_res_1338_; 
v_sz_boxed_1336_ = lean_unbox_usize(v_sz_1333_);
lean_dec(v_sz_1333_);
v_i_boxed_1337_ = lean_unbox_usize(v_i_1334_);
lean_dec(v_i_1334_);
v_res_1338_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__5(v_as_1332_, v_sz_boxed_1336_, v_i_boxed_1337_, v_b_1335_);
lean_dec_ref(v_as_1332_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1(lean_object* v_m_1339_, lean_object* v_l_1340_){
_start:
{
lean_object* v_buckets_1341_; size_t v_sz_1342_; size_t v___x_1343_; lean_object* v___x_1344_; 
v_buckets_1341_ = lean_ctor_get(v_l_1340_, 1);
v_sz_1342_ = lean_array_size(v_buckets_1341_);
v___x_1343_ = ((size_t)0ULL);
v___x_1344_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__5(v_buckets_1341_, v_sz_1342_, v___x_1343_, v_m_1339_);
return v___x_1344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1___boxed(lean_object* v_m_1345_, lean_object* v_l_1346_){
_start:
{
lean_object* v_res_1347_; 
v_res_1347_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1(v_m_1345_, v_l_1346_);
lean_dec_ref(v_l_1346_);
return v_res_1347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__1(lean_object* v_x1_1348_, lean_object* v_x2_1349_){
_start:
{
lean_object* v_size_1350_; lean_object* v_buckets_1351_; lean_object* v_size_1352_; uint8_t v___x_1353_; 
v_size_1350_ = lean_ctor_get(v_x1_1348_, 0);
v_buckets_1351_ = lean_ctor_get(v_x1_1348_, 1);
v_size_1352_ = lean_ctor_get(v_x2_1349_, 0);
v___x_1353_ = lean_nat_dec_le(v_size_1350_, v_size_1352_);
if (v___x_1353_ == 0)
{
lean_object* v___x_1354_; 
v___x_1354_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1(v_x1_1348_, v_x2_1349_);
lean_dec_ref(v_x2_1349_);
return v___x_1354_;
}
else
{
size_t v_sz_1355_; size_t v___x_1356_; lean_object* v___x_1357_; 
lean_inc_ref(v_buckets_1351_);
lean_dec_ref(v_x1_1348_);
v_sz_1355_ = lean_array_size(v_buckets_1351_);
v___x_1356_ = ((size_t)0ULL);
v___x_1357_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3(v_buckets_1351_, v_sz_1355_, v___x_1356_, v_x2_1349_);
lean_dec_ref(v_buckets_1351_);
return v___x_1357_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks(lean_object* v_pred_1359_, lean_object* v_msg_1360_){
_start:
{
lean_object* v___f_1362_; lean_object* v___f_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; 
v___f_1362_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___boxed), 5, 1);
lean_closure_set(v___f_1362_, 0, v_pred_1359_);
v___f_1363_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___closed__0));
v___x_1364_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2);
v___x_1365_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v___f_1362_, v___x_1364_, v___f_1363_, v_msg_1360_);
return v___x_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___boxed(lean_object* v_pred_1366_, lean_object* v_msg_1367_, lean_object* v_a_1368_){
_start:
{
lean_object* v_res_1369_; 
v_res_1369_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks(v_pred_1366_, v_msg_1367_);
return v_res_1369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0(lean_object* v_00_u03b2_1370_, lean_object* v_m_1371_, lean_object* v_a_1372_, lean_object* v_b_1373_){
_start:
{
lean_object* v___x_1374_; 
v___x_1374_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0___redArg(v_m_1371_, v_a_1372_, v_b_1373_);
return v___x_1374_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0(lean_object* v_00_u03b2_1375_, lean_object* v_a_1376_, lean_object* v_x_1377_){
_start:
{
uint8_t v___x_1378_; 
v___x_1378_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg(v_a_1376_, v_x_1377_);
return v___x_1378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1379_, lean_object* v_a_1380_, lean_object* v_x_1381_){
_start:
{
uint8_t v_res_1382_; lean_object* v_r_1383_; 
v_res_1382_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0(v_00_u03b2_1379_, v_a_1380_, v_x_1381_);
lean_dec(v_x_1381_);
lean_dec_ref(v_a_1380_);
v_r_1383_ = lean_box(v_res_1382_);
return v_r_1383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1(lean_object* v_00_u03b2_1384_, lean_object* v_data_1385_){
_start:
{
lean_object* v___x_1386_; 
v___x_1386_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1___redArg(v_data_1385_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3(lean_object* v_00_u03b2_1387_, lean_object* v_m_1388_, lean_object* v_a_1389_, lean_object* v_b_1390_){
_start:
{
lean_object* v___x_1391_; 
v___x_1391_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3___redArg(v_m_1388_, v_a_1389_, v_b_1390_);
return v___x_1391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_1392_, lean_object* v_i_1393_, lean_object* v_source_1394_, lean_object* v_target_1395_){
_start:
{
lean_object* v___x_1396_; 
v___x_1396_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2___redArg(v_i_1393_, v_source_1394_, v_target_1395_);
return v___x_1396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3_spec__5(lean_object* v_00_u03b2_1397_, lean_object* v_a_1398_, lean_object* v_b_1399_, lean_object* v_x_1400_){
_start:
{
lean_object* v___x_1401_; 
v___x_1401_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1_spec__3_spec__5___redArg(v_a_1398_, v_b_1399_, v_x_1400_);
return v___x_1401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2_spec__6(lean_object* v_00_u03b2_1402_, lean_object* v_x_1403_, lean_object* v_x_1404_){
_start:
{
lean_object* v___x_1405_; 
v___x_1405_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__1_spec__2_spec__6___redArg(v_x_1403_, v_x_1404_);
return v___x_1405_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg(lean_object* v_m_1406_, lean_object* v_a_1407_){
_start:
{
lean_object* v_buckets_1408_; lean_object* v___x_1409_; uint64_t v___x_1410_; uint64_t v___x_1411_; uint64_t v___x_1412_; uint64_t v_fold_1413_; uint64_t v___x_1414_; uint64_t v___x_1415_; uint64_t v___x_1416_; size_t v___x_1417_; size_t v___x_1418_; size_t v___x_1419_; size_t v___x_1420_; size_t v___x_1421_; lean_object* v___x_1422_; uint8_t v___x_1423_; 
v_buckets_1408_ = lean_ctor_get(v_m_1406_, 1);
v___x_1409_ = lean_array_get_size(v_buckets_1408_);
v___x_1410_ = lean_string_hash(v_a_1407_);
v___x_1411_ = 32ULL;
v___x_1412_ = lean_uint64_shift_right(v___x_1410_, v___x_1411_);
v_fold_1413_ = lean_uint64_xor(v___x_1410_, v___x_1412_);
v___x_1414_ = 16ULL;
v___x_1415_ = lean_uint64_shift_right(v_fold_1413_, v___x_1414_);
v___x_1416_ = lean_uint64_xor(v_fold_1413_, v___x_1415_);
v___x_1417_ = lean_uint64_to_usize(v___x_1416_);
v___x_1418_ = lean_usize_of_nat(v___x_1409_);
v___x_1419_ = ((size_t)1ULL);
v___x_1420_ = lean_usize_sub(v___x_1418_, v___x_1419_);
v___x_1421_ = lean_usize_land(v___x_1417_, v___x_1420_);
v___x_1422_ = lean_array_uget_borrowed(v_buckets_1408_, v___x_1421_);
v___x_1423_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0_spec__0___redArg(v_a_1407_, v___x_1422_);
return v___x_1423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg___boxed(lean_object* v_m_1424_, lean_object* v_a_1425_){
_start:
{
uint8_t v_res_1426_; lean_object* v_r_1427_; 
v_res_1426_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg(v_m_1424_, v_a_1425_);
lean_dec_ref(v_a_1425_);
lean_dec_ref(v_m_1424_);
v_r_1427_ = lean_box(v_res_1426_);
return v_r_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0(lean_object* v_permSuccesses_1428_, lean_object* v_permFailures_1429_, lean_object* v_as_1430_, size_t v_i_1431_, size_t v_stop_1432_, lean_object* v_b_1433_){
_start:
{
uint8_t v___x_1435_; 
v___x_1435_ = lean_usize_dec_eq(v_i_1431_, v_stop_1432_);
if (v___x_1435_ == 0)
{
lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; size_t v___x_1439_; size_t v___x_1440_; 
v___x_1436_ = lean_array_uget_borrowed(v_as_1430_, v_i_1431_);
lean_inc(v___x_1436_);
lean_inc_ref(v_permFailures_1429_);
lean_inc_ref(v_permSuccesses_1428_);
v___x_1437_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures(v_permSuccesses_1428_, v_permFailures_1429_, v___x_1436_);
v___x_1438_ = l_Array_append___redArg(v_b_1433_, v___x_1437_);
lean_dec_ref(v___x_1437_);
v___x_1439_ = ((size_t)1ULL);
v___x_1440_ = lean_usize_add(v_i_1431_, v___x_1439_);
v_i_1431_ = v___x_1440_;
v_b_1433_ = v___x_1438_;
goto _start;
}
else
{
lean_dec_ref(v_permFailures_1429_);
lean_dec_ref(v_permSuccesses_1428_);
return v_b_1433_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___lam__0(lean_object* v___x_1442_, lean_object* v_permSuccesses_1443_, lean_object* v_permFailures_1444_, uint8_t v___x_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_){
_start:
{
lean_object* v___y_1451_; lean_object* v_val_1454_; lean_object* v___y_1463_; uint8_t v___y_1465_; uint8_t v___y_1482_; lean_object* v_cls_1484_; lean_object* v_result_x3f_1485_; lean_object* v___x_1486_; uint8_t v___x_1487_; 
v_cls_1484_ = lean_ctor_get(v___y_1446_, 0);
v_result_x3f_1485_ = lean_ctor_get(v___y_1446_, 1);
v___x_1486_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3));
v___x_1487_ = lean_name_eq(v_cls_1484_, v___x_1486_);
if (v___x_1487_ == 0)
{
lean_object* v___x_1488_; uint8_t v___x_1489_; 
v___x_1488_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4));
v___x_1489_ = l_Lean_Name_isPrefixOf(v___x_1488_, v_cls_1484_);
if (v___x_1489_ == 0)
{
lean_object* v___x_1490_; 
lean_dec_ref(v___y_1447_);
lean_dec_ref(v_permFailures_1444_);
lean_dec_ref(v_permSuccesses_1443_);
v___x_1490_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__1));
return v___x_1490_;
}
else
{
if (lean_obj_tag(v_result_x3f_1485_) == 1)
{
lean_object* v_val_1491_; uint8_t v___x_1492_; 
v_val_1491_ = lean_ctor_get(v_result_x3f_1485_, 0);
v___x_1492_ = lean_unbox(v_val_1491_);
if (v___x_1492_ == 1)
{
goto v___jp_1477_;
}
else
{
v___y_1482_ = v___x_1445_;
goto v___jp_1481_;
}
}
else
{
v___y_1482_ = v___x_1445_;
goto v___jp_1481_;
}
}
}
else
{
lean_object* v___x_1493_; 
lean_dec_ref(v___y_1447_);
lean_dec_ref(v_permFailures_1444_);
lean_dec_ref(v_permSuccesses_1443_);
v___x_1493_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__0));
return v___x_1493_;
}
v___jp_1450_:
{
lean_object* v___x_1452_; 
v___x_1452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1452_, 0, v___y_1451_);
return v___x_1452_;
}
v___jp_1453_:
{
lean_object* v___x_1455_; uint8_t v___x_1456_; 
v___x_1455_ = lean_array_get_size(v_val_1454_);
v___x_1456_ = lean_nat_dec_eq(v___x_1455_, v___x_1442_);
if (v___x_1456_ == 0)
{
lean_object* v___x_1457_; 
lean_dec_ref(v___y_1447_);
v___x_1457_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1457_, 0, v_val_1454_);
v___y_1451_ = v___x_1457_;
goto v___jp_1450_;
}
else
{
lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; 
lean_dec_ref(v_val_1454_);
v___x_1458_ = lean_unsigned_to_nat(1u);
v___x_1459_ = lean_mk_empty_array_with_capacity(v___x_1458_);
v___x_1460_ = lean_array_push(v___x_1459_, v___y_1447_);
v___x_1461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1461_, 0, v___x_1460_);
v___y_1451_ = v___x_1461_;
goto v___jp_1450_;
}
}
v___jp_1462_:
{
v_val_1454_ = v___y_1463_;
goto v___jp_1453_;
}
v___jp_1464_:
{
if (v___y_1465_ == 0)
{
lean_object* v___x_1466_; 
lean_dec_ref(v___y_1447_);
lean_dec_ref(v_permFailures_1444_);
lean_dec_ref(v_permSuccesses_1443_);
v___x_1466_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__1));
return v___x_1466_;
}
else
{
lean_object* v___x_1467_; lean_object* v___x_1468_; uint8_t v___x_1469_; 
v___x_1467_ = lean_mk_empty_array_with_capacity(v___x_1442_);
v___x_1468_ = lean_array_get_size(v___y_1448_);
v___x_1469_ = lean_nat_dec_lt(v___x_1442_, v___x_1468_);
if (v___x_1469_ == 0)
{
lean_dec_ref(v_permFailures_1444_);
lean_dec_ref(v_permSuccesses_1443_);
v_val_1454_ = v___x_1467_;
goto v___jp_1453_;
}
else
{
uint8_t v___x_1470_; 
v___x_1470_ = lean_nat_dec_le(v___x_1468_, v___x_1468_);
if (v___x_1470_ == 0)
{
if (v___x_1469_ == 0)
{
lean_dec_ref(v_permFailures_1444_);
lean_dec_ref(v_permSuccesses_1443_);
v_val_1454_ = v___x_1467_;
goto v___jp_1453_;
}
else
{
size_t v___x_1471_; size_t v___x_1472_; lean_object* v___x_1473_; 
v___x_1471_ = ((size_t)0ULL);
v___x_1472_ = lean_usize_of_nat(v___x_1468_);
v___x_1473_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0(v_permSuccesses_1443_, v_permFailures_1444_, v___y_1448_, v___x_1471_, v___x_1472_, v___x_1467_);
v___y_1463_ = v___x_1473_;
goto v___jp_1462_;
}
}
else
{
size_t v___x_1474_; size_t v___x_1475_; lean_object* v___x_1476_; 
v___x_1474_ = ((size_t)0ULL);
v___x_1475_ = lean_usize_of_nat(v___x_1468_);
v___x_1476_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0(v_permSuccesses_1443_, v_permFailures_1444_, v___y_1448_, v___x_1474_, v___x_1475_, v___x_1467_);
v___y_1463_ = v___x_1476_;
goto v___jp_1462_;
}
}
}
}
v___jp_1477_:
{
lean_object* v___x_1478_; uint8_t v___x_1479_; 
lean_inc_ref(v___y_1447_);
v___x_1478_ = l_Lean_MessageData_toString(v___y_1447_);
v___x_1479_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg(v_permSuccesses_1443_, v___x_1478_);
if (v___x_1479_ == 0)
{
lean_dec_ref(v___x_1478_);
v___y_1465_ = v___x_1479_;
goto v___jp_1464_;
}
else
{
uint8_t v___x_1480_; 
v___x_1480_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg(v_permFailures_1444_, v___x_1478_);
lean_dec_ref(v___x_1478_);
if (v___x_1480_ == 0)
{
v___y_1465_ = v___x_1479_;
goto v___jp_1464_;
}
else
{
v___y_1465_ = v___x_1445_;
goto v___jp_1464_;
}
}
}
v___jp_1481_:
{
if (v___y_1482_ == 0)
{
lean_object* v___x_1483_; 
lean_dec_ref(v___y_1447_);
lean_dec_ref(v_permFailures_1444_);
lean_dec_ref(v_permSuccesses_1443_);
v___x_1483_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___lam__0___closed__1));
return v___x_1483_;
}
else
{
goto v___jp_1477_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___lam__0___boxed(lean_object* v___x_1494_, lean_object* v_permSuccesses_1495_, lean_object* v_permFailures_1496_, lean_object* v___x_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_){
_start:
{
uint8_t v___x_1821__boxed_1502_; lean_object* v_res_1503_; 
v___x_1821__boxed_1502_ = lean_unbox(v___x_1497_);
v_res_1503_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___lam__0(v___x_1494_, v_permSuccesses_1495_, v_permFailures_1496_, v___x_1821__boxed_1502_, v___y_1498_, v___y_1499_, v___y_1500_);
lean_dec_ref(v___y_1500_);
lean_dec_ref(v___y_1498_);
lean_dec(v___x_1494_);
return v_res_1503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures(lean_object* v_permSuccesses_1504_, lean_object* v_permFailures_1505_, lean_object* v_msg_1506_){
_start:
{
lean_object* v_size_1508_; lean_object* v___x_1509_; uint8_t v___x_1510_; 
v_size_1508_ = lean_ctor_get(v_permSuccesses_1504_, 0);
v___x_1509_ = lean_unsigned_to_nat(0u);
v___x_1510_ = lean_nat_dec_eq(v_size_1508_, v___x_1509_);
if (v___x_1510_ == 0)
{
lean_object* v___f_1511_; lean_object* v___x_1512_; lean_object* v___f_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; 
v___f_1511_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__0));
v___x_1512_ = lean_box(v___x_1510_);
v___f_1513_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___lam__0___boxed), 8, 4);
lean_closure_set(v___f_1513_, 0, v___x_1509_);
lean_closure_set(v___f_1513_, 1, v_permSuccesses_1504_);
lean_closure_set(v___f_1513_, 2, v_permFailures_1505_);
lean_closure_set(v___f_1513_, 3, v___x_1512_);
v___x_1514_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__1));
v___x_1515_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v___f_1513_, v___x_1514_, v___f_1511_, v_msg_1506_);
return v___x_1515_;
}
else
{
lean_object* v___x_1516_; 
lean_dec_ref(v_permFailures_1505_);
lean_dec_ref(v_permSuccesses_1504_);
v___x_1516_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures(v_msg_1506_);
return v___x_1516_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures___boxed(lean_object* v_permSuccesses_1517_, lean_object* v_permFailures_1518_, lean_object* v_msg_1519_, lean_object* v_a_1520_){
_start:
{
lean_object* v_res_1521_; 
v_res_1521_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures(v_permSuccesses_1517_, v_permFailures_1518_, v_msg_1519_);
return v_res_1521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0___boxed(lean_object* v_permSuccesses_1522_, lean_object* v_permFailures_1523_, lean_object* v_as_1524_, lean_object* v_i_1525_, lean_object* v_stop_1526_, lean_object* v_b_1527_, lean_object* v___y_1528_){
_start:
{
size_t v_i_boxed_1529_; size_t v_stop_boxed_1530_; lean_object* v_res_1531_; 
v_i_boxed_1529_ = lean_unbox_usize(v_i_1525_);
lean_dec(v_i_1525_);
v_stop_boxed_1530_ = lean_unbox_usize(v_stop_1526_);
lean_dec(v_stop_1526_);
v_res_1531_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0(v_permSuccesses_1522_, v_permFailures_1523_, v_as_1524_, v_i_boxed_1529_, v_stop_boxed_1530_, v_b_1527_);
lean_dec_ref(v_as_1524_);
return v_res_1531_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1(lean_object* v_00_u03b2_1532_, lean_object* v_m_1533_, lean_object* v_a_1534_){
_start:
{
uint8_t v___x_1535_; 
v___x_1535_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg(v_m_1533_, v_a_1534_);
return v___x_1535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___boxed(lean_object* v_00_u03b2_1536_, lean_object* v_m_1537_, lean_object* v_a_1538_){
_start:
{
uint8_t v_res_1539_; lean_object* v_r_1540_; 
v_res_1539_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1(v_00_u03b2_1536_, v_m_1537_, v_a_1538_);
lean_dec_ref(v_a_1538_);
lean_dec_ref(v_m_1537_);
v_r_1540_ = lean_box(v_res_1539_);
return v_r_1540_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg(lean_object* v_s_1541_, lean_object* v_a_1542_, uint8_t v_b_1543_){
_start:
{
uint8_t v___x_1544_; 
v___x_1544_ = 0;
switch(lean_obj_tag(v_a_1542_))
{
case 0:
{
uint8_t v___x_1545_; 
lean_dec_ref_known(v_a_1542_, 1);
v___x_1545_ = 1;
return v___x_1545_;
}
case 1:
{
lean_object* v_pos_1546_; lean_object* v___x_1548_; uint8_t v_isShared_1549_; uint8_t v_isSharedCheck_1559_; 
v_pos_1546_ = lean_ctor_get(v_a_1542_, 0);
v_isSharedCheck_1559_ = !lean_is_exclusive(v_a_1542_);
if (v_isSharedCheck_1559_ == 0)
{
v___x_1548_ = v_a_1542_;
v_isShared_1549_ = v_isSharedCheck_1559_;
goto v_resetjp_1547_;
}
else
{
lean_inc(v_pos_1546_);
lean_dec(v_a_1542_);
v___x_1548_ = lean_box(0);
v_isShared_1549_ = v_isSharedCheck_1559_;
goto v_resetjp_1547_;
}
v_resetjp_1547_:
{
lean_object* v_str_1550_; lean_object* v_startInclusive_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1556_; 
v_str_1550_ = lean_ctor_get(v_s_1541_, 0);
v_startInclusive_1551_ = lean_ctor_get(v_s_1541_, 1);
v___x_1552_ = lean_nat_add(v_startInclusive_1551_, v_pos_1546_);
lean_dec(v_pos_1546_);
v___x_1553_ = lean_string_utf8_next_fast(v_str_1550_, v___x_1552_);
lean_dec(v___x_1552_);
v___x_1554_ = lean_nat_sub(v___x_1553_, v_startInclusive_1551_);
if (v_isShared_1549_ == 0)
{
lean_ctor_set_tag(v___x_1548_, 0);
lean_ctor_set(v___x_1548_, 0, v___x_1554_);
v___x_1556_ = v___x_1548_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v___x_1554_);
v___x_1556_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
v_a_1542_ = v___x_1556_;
v_b_1543_ = v___x_1544_;
goto _start;
}
}
}
case 2:
{
lean_object* v_needle_1560_; lean_object* v_table_1561_; lean_object* v_stackPos_1562_; lean_object* v_needlePos_1563_; lean_object* v___x_1565_; uint8_t v_isShared_1566_; uint8_t v_isSharedCheck_1616_; 
v_needle_1560_ = lean_ctor_get(v_a_1542_, 0);
v_table_1561_ = lean_ctor_get(v_a_1542_, 1);
v_stackPos_1562_ = lean_ctor_get(v_a_1542_, 2);
v_needlePos_1563_ = lean_ctor_get(v_a_1542_, 3);
v_isSharedCheck_1616_ = !lean_is_exclusive(v_a_1542_);
if (v_isSharedCheck_1616_ == 0)
{
v___x_1565_ = v_a_1542_;
v_isShared_1566_ = v_isSharedCheck_1616_;
goto v_resetjp_1564_;
}
else
{
lean_inc(v_needlePos_1563_);
lean_inc(v_stackPos_1562_);
lean_inc(v_table_1561_);
lean_inc(v_needle_1560_);
lean_dec(v_a_1542_);
v___x_1565_ = lean_box(0);
v_isShared_1566_ = v_isSharedCheck_1616_;
goto v_resetjp_1564_;
}
v_resetjp_1564_:
{
lean_object* v_str_1567_; lean_object* v_startInclusive_1568_; lean_object* v_endExclusive_1569_; lean_object* v_str_1570_; lean_object* v_startInclusive_1571_; lean_object* v_endExclusive_1572_; lean_object* v_basePos_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; uint8_t v___x_1577_; 
v_str_1567_ = lean_ctor_get(v_needle_1560_, 0);
v_startInclusive_1568_ = lean_ctor_get(v_needle_1560_, 1);
v_endExclusive_1569_ = lean_ctor_get(v_needle_1560_, 2);
v_str_1570_ = lean_ctor_get(v_s_1541_, 0);
v_startInclusive_1571_ = lean_ctor_get(v_s_1541_, 1);
v_endExclusive_1572_ = lean_ctor_get(v_s_1541_, 2);
v_basePos_1573_ = lean_nat_sub(v_stackPos_1562_, v_needlePos_1563_);
v___x_1574_ = lean_nat_sub(v_endExclusive_1569_, v_startInclusive_1568_);
v___x_1575_ = lean_nat_add(v_basePos_1573_, v___x_1574_);
v___x_1576_ = lean_nat_sub(v_endExclusive_1572_, v_startInclusive_1571_);
v___x_1577_ = lean_nat_dec_le(v___x_1575_, v___x_1576_);
lean_dec(v___x_1575_);
if (v___x_1577_ == 0)
{
uint8_t v___x_1578_; 
lean_dec(v___x_1574_);
lean_del_object(v___x_1565_);
lean_dec(v_needlePos_1563_);
lean_dec(v_stackPos_1562_);
lean_dec_ref(v_table_1561_);
lean_dec_ref(v_needle_1560_);
v___x_1578_ = lean_nat_dec_lt(v_basePos_1573_, v___x_1576_);
lean_dec(v___x_1576_);
lean_dec(v_basePos_1573_);
if (v___x_1578_ == 0)
{
return v_b_1543_;
}
else
{
lean_object* v___x_1579_; 
v___x_1579_ = lean_box(3);
v_a_1542_ = v___x_1579_;
v_b_1543_ = v___x_1544_;
goto _start;
}
}
else
{
lean_object* v___x_1581_; uint8_t v_stackByte_1582_; lean_object* v___x_1583_; uint8_t v_patByte_1584_; uint8_t v___x_1585_; 
lean_dec(v___x_1576_);
lean_dec(v_basePos_1573_);
v___x_1581_ = lean_nat_add(v_startInclusive_1571_, v_stackPos_1562_);
v_stackByte_1582_ = lean_string_get_byte_fast(v_str_1570_, v___x_1581_);
v___x_1583_ = lean_nat_add(v_startInclusive_1568_, v_needlePos_1563_);
v_patByte_1584_ = lean_string_get_byte_fast(v_str_1567_, v___x_1583_);
v___x_1585_ = lean_uint8_dec_eq(v_stackByte_1582_, v_patByte_1584_);
if (v___x_1585_ == 0)
{
lean_object* v___x_1586_; uint8_t v___x_1587_; 
lean_dec(v___x_1574_);
v___x_1586_ = lean_unsigned_to_nat(0u);
v___x_1587_ = lean_nat_dec_eq(v_needlePos_1563_, v___x_1586_);
if (v___x_1587_ == 0)
{
lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v_newNeedlePos_1590_; uint8_t v___x_1591_; 
v___x_1588_ = lean_unsigned_to_nat(1u);
v___x_1589_ = lean_nat_sub(v_needlePos_1563_, v___x_1588_);
lean_dec(v_needlePos_1563_);
v_newNeedlePos_1590_ = lean_array_fget_borrowed(v_table_1561_, v___x_1589_);
lean_dec(v___x_1589_);
v___x_1591_ = lean_nat_dec_eq(v_newNeedlePos_1590_, v___x_1586_);
if (v___x_1591_ == 0)
{
lean_object* v___x_1593_; 
lean_inc(v_newNeedlePos_1590_);
if (v_isShared_1566_ == 0)
{
lean_ctor_set(v___x_1565_, 3, v_newNeedlePos_1590_);
v___x_1593_ = v___x_1565_;
goto v_reusejp_1592_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v_needle_1560_);
lean_ctor_set(v_reuseFailAlloc_1595_, 1, v_table_1561_);
lean_ctor_set(v_reuseFailAlloc_1595_, 2, v_stackPos_1562_);
lean_ctor_set(v_reuseFailAlloc_1595_, 3, v_newNeedlePos_1590_);
v___x_1593_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1592_;
}
v_reusejp_1592_:
{
v_a_1542_ = v___x_1593_;
v_b_1543_ = v___x_1544_;
goto _start;
}
}
else
{
lean_object* v_nextStackPos_1596_; lean_object* v___x_1598_; 
v_nextStackPos_1596_ = l_String_Slice_posGE___redArg(v_s_1541_, v_stackPos_1562_);
if (v_isShared_1566_ == 0)
{
lean_ctor_set(v___x_1565_, 3, v___x_1586_);
lean_ctor_set(v___x_1565_, 2, v_nextStackPos_1596_);
v___x_1598_ = v___x_1565_;
goto v_reusejp_1597_;
}
else
{
lean_object* v_reuseFailAlloc_1600_; 
v_reuseFailAlloc_1600_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1600_, 0, v_needle_1560_);
lean_ctor_set(v_reuseFailAlloc_1600_, 1, v_table_1561_);
lean_ctor_set(v_reuseFailAlloc_1600_, 2, v_nextStackPos_1596_);
lean_ctor_set(v_reuseFailAlloc_1600_, 3, v___x_1586_);
v___x_1598_ = v_reuseFailAlloc_1600_;
goto v_reusejp_1597_;
}
v_reusejp_1597_:
{
v_a_1542_ = v___x_1598_;
v_b_1543_ = v___x_1544_;
goto _start;
}
}
}
else
{
lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v_nextStackPos_1603_; lean_object* v___x_1605_; 
lean_dec(v_needlePos_1563_);
v___x_1601_ = lean_unsigned_to_nat(1u);
v___x_1602_ = lean_nat_add(v_stackPos_1562_, v___x_1601_);
lean_dec(v_stackPos_1562_);
v_nextStackPos_1603_ = l_String_Slice_posGE___redArg(v_s_1541_, v___x_1602_);
if (v_isShared_1566_ == 0)
{
lean_ctor_set(v___x_1565_, 3, v___x_1586_);
lean_ctor_set(v___x_1565_, 2, v_nextStackPos_1603_);
v___x_1605_ = v___x_1565_;
goto v_reusejp_1604_;
}
else
{
lean_object* v_reuseFailAlloc_1607_; 
v_reuseFailAlloc_1607_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1607_, 0, v_needle_1560_);
lean_ctor_set(v_reuseFailAlloc_1607_, 1, v_table_1561_);
lean_ctor_set(v_reuseFailAlloc_1607_, 2, v_nextStackPos_1603_);
lean_ctor_set(v_reuseFailAlloc_1607_, 3, v___x_1586_);
v___x_1605_ = v_reuseFailAlloc_1607_;
goto v_reusejp_1604_;
}
v_reusejp_1604_:
{
v_a_1542_ = v___x_1605_;
v_b_1543_ = v___x_1544_;
goto _start;
}
}
}
else
{
lean_object* v___x_1608_; lean_object* v_nextNeedlePos_1609_; uint8_t v___x_1610_; 
v___x_1608_ = lean_unsigned_to_nat(1u);
v_nextNeedlePos_1609_ = lean_nat_add(v_needlePos_1563_, v___x_1608_);
lean_dec(v_needlePos_1563_);
v___x_1610_ = lean_nat_dec_eq(v_nextNeedlePos_1609_, v___x_1574_);
lean_dec(v___x_1574_);
if (v___x_1610_ == 0)
{
lean_object* v_nextStackPos_1611_; lean_object* v___x_1613_; 
v_nextStackPos_1611_ = lean_nat_add(v_stackPos_1562_, v___x_1608_);
lean_dec(v_stackPos_1562_);
if (v_isShared_1566_ == 0)
{
lean_ctor_set(v___x_1565_, 3, v_nextNeedlePos_1609_);
lean_ctor_set(v___x_1565_, 2, v_nextStackPos_1611_);
v___x_1613_ = v___x_1565_;
goto v_reusejp_1612_;
}
else
{
lean_object* v_reuseFailAlloc_1615_; 
v_reuseFailAlloc_1615_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1615_, 0, v_needle_1560_);
lean_ctor_set(v_reuseFailAlloc_1615_, 1, v_table_1561_);
lean_ctor_set(v_reuseFailAlloc_1615_, 2, v_nextStackPos_1611_);
lean_ctor_set(v_reuseFailAlloc_1615_, 3, v_nextNeedlePos_1609_);
v___x_1613_ = v_reuseFailAlloc_1615_;
goto v_reusejp_1612_;
}
v_reusejp_1612_:
{
v_a_1542_ = v___x_1613_;
goto _start;
}
}
else
{
lean_dec(v_nextNeedlePos_1609_);
lean_del_object(v___x_1565_);
lean_dec(v_stackPos_1562_);
lean_dec_ref(v_table_1561_);
lean_dec_ref(v_needle_1560_);
return v___x_1610_;
}
}
}
}
}
default: 
{
return v_b_1543_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg___boxed(lean_object* v_s_1617_, lean_object* v_a_1618_, lean_object* v_b_1619_){
_start:
{
uint8_t v_b_boxed_1620_; uint8_t v_res_1621_; lean_object* v_r_1622_; 
v_b_boxed_1620_ = lean_unbox(v_b_1619_);
v_res_1621_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg(v_s_1617_, v_a_1618_, v_b_boxed_1620_);
lean_dec_ref(v_s_1617_);
v_r_1622_ = lean_box(v_res_1621_);
return v_r_1622_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1624_; lean_object* v___x_1625_; 
v___x_1624_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__0));
v___x_1625_ = lean_string_utf8_byte_size(v___x_1624_);
return v___x_1625_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__2(void){
_start:
{
lean_object* v___x_1626_; lean_object* v___x_1627_; uint8_t v___x_1628_; 
v___x_1626_ = lean_unsigned_to_nat(0u);
v___x_1627_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1);
v___x_1628_ = lean_nat_dec_eq(v___x_1627_, v___x_1626_);
return v___x_1628_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3(void){
_start:
{
lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; 
v___x_1629_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__1);
v___x_1630_ = lean_unsigned_to_nat(0u);
v___x_1631_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__0));
v___x_1632_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1632_, 0, v___x_1631_);
lean_ctor_set(v___x_1632_, 1, v___x_1630_);
lean_ctor_set(v___x_1632_, 2, v___x_1629_);
return v___x_1632_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__4(void){
_start:
{
lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1633_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3);
v___x_1634_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_1633_);
return v___x_1634_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__5(void){
_start:
{
lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; 
v___x_1635_ = lean_unsigned_to_nat(0u);
v___x_1636_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__4, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__4_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__4);
v___x_1637_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__3);
v___x_1638_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_1638_, 0, v___x_1637_);
lean_ctor_set(v___x_1638_, 1, v___x_1636_);
lean_ctor_set(v___x_1638_, 2, v___x_1635_);
lean_ctor_set(v___x_1638_, 3, v___x_1635_);
return v___x_1638_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0(lean_object* v_s_1641_){
_start:
{
lean_object* v___y_1643_; uint8_t v___x_1646_; 
v___x_1646_ = lean_uint8_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__2, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__2_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__2);
if (v___x_1646_ == 0)
{
lean_object* v___x_1647_; 
v___x_1647_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__5, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__5_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__5);
v___y_1643_ = v___x_1647_;
goto v___jp_1642_;
}
else
{
lean_object* v___x_1648_; 
v___x_1648_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__6));
v___y_1643_ = v___x_1648_;
goto v___jp_1642_;
}
v___jp_1642_:
{
uint8_t v___x_1644_; uint8_t v___x_1645_; 
v___x_1644_ = 0;
lean_inc(v___y_1643_);
v___x_1645_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg(v_s_1641_, v___y_1643_, v___x_1644_);
return v___x_1645_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___boxed(lean_object* v_s_1649_){
_start:
{
uint8_t v_res_1650_; lean_object* v_r_1651_; 
v_res_1650_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0(v_s_1649_);
lean_dec_ref(v_s_1649_);
v_r_1651_ = lean_box(v_res_1650_);
return v_r_1651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0(lean_object* v_permSuccesses_1660_, lean_object* v_permFailures_1661_, lean_object* v_td_1662_, lean_object* v_header_1663_, lean_object* v_children_1664_){
_start:
{
lean_object* v___y_1667_; lean_object* v_cls_1676_; lean_object* v_result_x3f_1677_; lean_object* v___x_1678_; uint8_t v___x_1679_; lean_object* v_val_1681_; lean_object* v___y_1686_; 
v_cls_1676_ = lean_ctor_get(v_td_1662_, 0);
v_result_x3f_1677_ = lean_ctor_get(v_td_1662_, 1);
v___x_1678_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3));
v___x_1679_ = lean_name_eq(v_cls_1676_, v___x_1678_);
if (v___x_1679_ == 0)
{
lean_object* v___x_1687_; uint8_t v___x_1688_; 
v___x_1687_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2));
v___x_1688_ = lean_name_eq(v_cls_1676_, v___x_1687_);
if (v___x_1688_ == 0)
{
lean_dec_ref(v_header_1663_);
lean_dec_ref(v_permFailures_1661_);
lean_dec_ref(v_permSuccesses_1660_);
goto v___jp_1674_;
}
else
{
lean_object* v___x_1689_; uint8_t v___y_1691_; 
lean_inc_ref(v_header_1663_);
v___x_1689_ = l_Lean_MessageData_toString(v_header_1663_);
if (lean_obj_tag(v_result_x3f_1677_) == 1)
{
lean_object* v_val_1706_; uint8_t v___x_1707_; 
v_val_1706_ = lean_ctor_get(v_result_x3f_1677_, 0);
v___x_1707_ = lean_unbox(v_val_1706_);
if (v___x_1707_ == 1)
{
v___y_1691_ = v___x_1688_;
goto v___jp_1690_;
}
else
{
v___y_1691_ = v___x_1679_;
goto v___jp_1690_;
}
}
else
{
v___y_1691_ = v___x_1679_;
goto v___jp_1690_;
}
v___jp_1690_:
{
if (v___y_1691_ == 0)
{
lean_dec_ref(v___x_1689_);
lean_dec_ref(v_header_1663_);
lean_dec_ref(v_permFailures_1661_);
lean_dec_ref(v_permSuccesses_1660_);
goto v___jp_1674_;
}
else
{
lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; uint8_t v___x_1695_; 
v___x_1692_ = lean_unsigned_to_nat(0u);
v___x_1693_ = lean_string_utf8_byte_size(v___x_1689_);
v___x_1694_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1694_, 0, v___x_1689_);
lean_ctor_set(v___x_1694_, 1, v___x_1692_);
lean_ctor_set(v___x_1694_, 2, v___x_1693_);
v___x_1695_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0(v___x_1694_);
lean_dec_ref_known(v___x_1694_, 3);
if (v___x_1695_ == 0)
{
lean_dec_ref(v_header_1663_);
lean_dec_ref(v_permFailures_1661_);
lean_dec_ref(v_permSuccesses_1660_);
goto v___jp_1674_;
}
else
{
lean_object* v___x_1696_; lean_object* v___x_1697_; uint8_t v___x_1698_; 
v___x_1696_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures___closed__1));
v___x_1697_ = lean_array_get_size(v_children_1664_);
v___x_1698_ = lean_nat_dec_lt(v___x_1692_, v___x_1697_);
if (v___x_1698_ == 0)
{
lean_dec_ref(v_permFailures_1661_);
lean_dec_ref(v_permSuccesses_1660_);
v_val_1681_ = v___x_1696_;
goto v___jp_1680_;
}
else
{
uint8_t v___x_1699_; 
v___x_1699_ = lean_nat_dec_le(v___x_1697_, v___x_1697_);
if (v___x_1699_ == 0)
{
if (v___x_1698_ == 0)
{
lean_dec_ref(v_permFailures_1661_);
lean_dec_ref(v_permSuccesses_1660_);
v_val_1681_ = v___x_1696_;
goto v___jp_1680_;
}
else
{
size_t v___x_1700_; size_t v___x_1701_; lean_object* v___x_1702_; 
v___x_1700_ = ((size_t)0ULL);
v___x_1701_ = lean_usize_of_nat(v___x_1697_);
v___x_1702_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0(v_permSuccesses_1660_, v_permFailures_1661_, v_children_1664_, v___x_1700_, v___x_1701_, v___x_1696_);
v___y_1686_ = v___x_1702_;
goto v___jp_1685_;
}
}
else
{
size_t v___x_1703_; size_t v___x_1704_; lean_object* v___x_1705_; 
v___x_1703_ = ((size_t)0ULL);
v___x_1704_ = lean_usize_of_nat(v___x_1697_);
v___x_1705_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__0(v_permSuccesses_1660_, v_permFailures_1661_, v_children_1664_, v___x_1703_, v___x_1704_, v___x_1696_);
v___y_1686_ = v___x_1705_;
goto v___jp_1685_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1708_; 
lean_dec_ref(v_header_1663_);
lean_dec_ref(v_permFailures_1661_);
lean_dec_ref(v_permSuccesses_1660_);
v___x_1708_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3));
return v___x_1708_;
}
v___jp_1666_:
{
lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; 
v___x_1668_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1668_, 0, v_header_1663_);
lean_ctor_set(v___x_1668_, 1, v___y_1667_);
v___x_1669_ = lean_unsigned_to_nat(1u);
v___x_1670_ = lean_mk_empty_array_with_capacity(v___x_1669_);
v___x_1671_ = lean_array_push(v___x_1670_, v___x_1668_);
v___x_1672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1672_, 0, v___x_1671_);
v___x_1673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1673_, 0, v___x_1672_);
return v___x_1673_;
}
v___jp_1674_:
{
lean_object* v___x_1675_; 
v___x_1675_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__0));
return v___x_1675_;
}
v___jp_1680_:
{
lean_object* v___x_1682_; lean_object* v___x_1683_; uint8_t v___x_1684_; 
v___x_1682_ = lean_array_get_size(v_val_1681_);
v___x_1683_ = lean_unsigned_to_nat(0u);
v___x_1684_ = lean_nat_dec_eq(v___x_1682_, v___x_1683_);
if (v___x_1684_ == 0)
{
v___y_1667_ = v_val_1681_;
goto v___jp_1666_;
}
else
{
if (v___x_1679_ == 0)
{
lean_dec_ref(v_val_1681_);
lean_dec_ref(v_header_1663_);
goto v___jp_1674_;
}
else
{
v___y_1667_ = v_val_1681_;
goto v___jp_1666_;
}
}
}
v___jp_1685_:
{
v_val_1681_ = v___y_1686_;
goto v___jp_1680_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___boxed(lean_object* v_permSuccesses_1709_, lean_object* v_permFailures_1710_, lean_object* v_td_1711_, lean_object* v_header_1712_, lean_object* v_children_1713_, lean_object* v___y_1714_){
_start:
{
lean_object* v_res_1715_; 
v_res_1715_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0(v_permSuccesses_1709_, v_permFailures_1710_, v_td_1711_, v_header_1712_, v_children_1713_);
lean_dec_ref(v_children_1713_);
lean_dec_ref(v_td_1711_);
return v_res_1715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures(lean_object* v_permSuccesses_1719_, lean_object* v_permFailures_1720_, lean_object* v_msg_1721_){
_start:
{
lean_object* v___f_1723_; lean_object* v___f_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; 
v___f_1723_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__0));
v___f_1724_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___boxed), 6, 2);
lean_closure_set(v___f_1724_, 0, v_permSuccesses_1719_);
lean_closure_set(v___f_1724_, 1, v_permFailures_1720_);
v___x_1725_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__1));
v___x_1726_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v___f_1724_, v___x_1725_, v___f_1723_, v_msg_1721_);
return v___x_1726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___boxed(lean_object* v_permSuccesses_1727_, lean_object* v_permFailures_1728_, lean_object* v_msg_1729_, lean_object* v_a_1730_){
_start:
{
lean_object* v_res_1731_; 
v_res_1731_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures(v_permSuccesses_1727_, v_permFailures_1728_, v_msg_1729_);
return v_res_1731_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0(lean_object* v_s_1732_, lean_object* v_inst_1733_, lean_object* v_R_1734_, lean_object* v_a_1735_, uint8_t v_b_1736_, lean_object* v_c_1737_){
_start:
{
uint8_t v___x_1738_; 
v___x_1738_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg(v_s_1732_, v_a_1735_, v_b_1736_);
return v___x_1738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___boxed(lean_object* v_s_1739_, lean_object* v_inst_1740_, lean_object* v_R_1741_, lean_object* v_a_1742_, lean_object* v_b_1743_, lean_object* v_c_1744_){
_start:
{
uint8_t v_b_boxed_1745_; uint8_t v_res_1746_; lean_object* v_r_1747_; 
v_b_boxed_1745_ = lean_unbox(v_b_1743_);
v_res_1746_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0(v_s_1739_, v_inst_1740_, v_R_1741_, v_a_1742_, v_b_boxed_1745_, v_c_1744_);
lean_dec_ref(v_s_1739_);
v_r_1747_ = lean_box(v_res_1746_);
return v_r_1747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures_spec__0(lean_object* v_permSuccesses_1748_, lean_object* v_permFailures_1749_, lean_object* v_as_1750_, size_t v_i_1751_, size_t v_stop_1752_, lean_object* v_b_1753_){
_start:
{
uint8_t v___x_1755_; 
v___x_1755_ = lean_usize_dec_eq(v_i_1751_, v_stop_1752_);
if (v___x_1755_ == 0)
{
lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; size_t v___x_1759_; size_t v___x_1760_; 
v___x_1756_ = lean_array_uget_borrowed(v_as_1750_, v_i_1751_);
lean_inc(v___x_1756_);
lean_inc_ref(v_permFailures_1749_);
lean_inc_ref(v_permSuccesses_1748_);
v___x_1757_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures(v_permSuccesses_1748_, v_permFailures_1749_, v___x_1756_);
v___x_1758_ = l_Array_append___redArg(v_b_1753_, v___x_1757_);
lean_dec_ref(v___x_1757_);
v___x_1759_ = ((size_t)1ULL);
v___x_1760_ = lean_usize_add(v_i_1751_, v___x_1759_);
v_i_1751_ = v___x_1760_;
v_b_1753_ = v___x_1758_;
goto _start;
}
else
{
lean_dec_ref(v_permFailures_1749_);
lean_dec_ref(v_permSuccesses_1748_);
return v_b_1753_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures_spec__0___boxed(lean_object* v_permSuccesses_1762_, lean_object* v_permFailures_1763_, lean_object* v_as_1764_, lean_object* v_i_1765_, lean_object* v_stop_1766_, lean_object* v_b_1767_, lean_object* v___y_1768_){
_start:
{
size_t v_i_boxed_1769_; size_t v_stop_boxed_1770_; lean_object* v_res_1771_; 
v_i_boxed_1769_ = lean_unbox_usize(v_i_1765_);
lean_dec(v_i_1765_);
v_stop_boxed_1770_ = lean_unbox_usize(v_stop_1766_);
lean_dec(v_stop_1766_);
v_res_1771_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures_spec__0(v_permSuccesses_1762_, v_permFailures_1763_, v_as_1764_, v_i_boxed_1769_, v_stop_boxed_1770_, v_b_1767_);
lean_dec_ref(v_as_1764_);
return v_res_1771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___lam__0(lean_object* v_permSuccesses_1772_, lean_object* v_permFailures_1773_, lean_object* v_td_1774_, lean_object* v_header_1775_, lean_object* v_children_1776_){
_start:
{
lean_object* v_____do__lift_1779_; uint8_t v___y_1797_; lean_object* v_cls_1803_; lean_object* v_result_x3f_1804_; lean_object* v___x_1805_; uint8_t v___x_1806_; 
v_cls_1803_ = lean_ctor_get(v_td_1774_, 0);
v_result_x3f_1804_ = lean_ctor_get(v_td_1774_, 1);
v___x_1805_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__3));
v___x_1806_ = lean_name_eq(v_cls_1803_, v___x_1805_);
if (v___x_1806_ == 0)
{
lean_object* v___x_1807_; uint8_t v___x_1808_; 
v___x_1807_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2));
v___x_1808_ = lean_name_eq(v_cls_1803_, v___x_1807_);
if (v___x_1808_ == 0)
{
lean_object* v___x_1811_; uint8_t v___x_1812_; 
lean_dec_ref(v_permFailures_1773_);
lean_dec_ref(v_permSuccesses_1772_);
v___x_1811_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_onlyOnDefEqNodes___redArg___closed__4));
v___x_1812_ = l_Lean_Name_isPrefixOf(v___x_1811_, v_cls_1803_);
if (v___x_1812_ == 0)
{
goto v___jp_1809_;
}
else
{
if (v___x_1808_ == 0)
{
goto v___jp_1801_;
}
else
{
goto v___jp_1809_;
}
}
}
else
{
if (lean_obj_tag(v_result_x3f_1804_) == 1)
{
lean_object* v_val_1813_; uint8_t v___x_1814_; 
v_val_1813_ = lean_ctor_get(v_result_x3f_1804_, 0);
v___x_1814_ = lean_unbox(v_val_1813_);
if (v___x_1814_ == 1)
{
goto v___jp_1782_;
}
else
{
v___y_1797_ = v___x_1806_;
goto v___jp_1796_;
}
}
else
{
v___y_1797_ = v___x_1806_;
goto v___jp_1796_;
}
}
v___jp_1809_:
{
uint8_t v___x_1810_; 
v___x_1810_ = l_Lean_Name_isPrefixOf(v___x_1807_, v_cls_1803_);
if (v___x_1810_ == 0)
{
goto v___jp_1799_;
}
else
{
if (v___x_1808_ == 0)
{
goto v___jp_1801_;
}
else
{
goto v___jp_1799_;
}
}
}
}
else
{
lean_object* v___x_1815_; 
lean_dec_ref(v_permFailures_1773_);
lean_dec_ref(v_permSuccesses_1772_);
v___x_1815_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3));
return v___x_1815_;
}
v___jp_1778_:
{
lean_object* v___x_1780_; lean_object* v___x_1781_; 
v___x_1780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1780_, 0, v_____do__lift_1779_);
v___x_1781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1781_, 0, v___x_1780_);
return v___x_1781_;
}
v___jp_1782_:
{
lean_object* v___x_1783_; lean_object* v___x_1784_; uint8_t v___x_1785_; 
v___x_1783_ = lean_array_get_size(v_children_1776_);
v___x_1784_ = lean_unsigned_to_nat(0u);
v___x_1785_ = lean_nat_dec_eq(v___x_1783_, v___x_1784_);
if (v___x_1785_ == 0)
{
lean_object* v___x_1786_; uint8_t v___x_1787_; 
v___x_1786_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__1));
v___x_1787_ = lean_nat_dec_lt(v___x_1784_, v___x_1783_);
if (v___x_1787_ == 0)
{
lean_dec_ref(v_permFailures_1773_);
lean_dec_ref(v_permSuccesses_1772_);
v_____do__lift_1779_ = v___x_1786_;
goto v___jp_1778_;
}
else
{
uint8_t v___x_1788_; 
v___x_1788_ = lean_nat_dec_le(v___x_1783_, v___x_1783_);
if (v___x_1788_ == 0)
{
if (v___x_1787_ == 0)
{
lean_dec_ref(v_permFailures_1773_);
lean_dec_ref(v_permSuccesses_1772_);
v_____do__lift_1779_ = v___x_1786_;
goto v___jp_1778_;
}
else
{
size_t v___x_1789_; size_t v___x_1790_; lean_object* v___x_1791_; 
v___x_1789_ = ((size_t)0ULL);
v___x_1790_ = lean_usize_of_nat(v___x_1783_);
v___x_1791_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures_spec__0(v_permSuccesses_1772_, v_permFailures_1773_, v_children_1776_, v___x_1789_, v___x_1790_, v___x_1786_);
v_____do__lift_1779_ = v___x_1791_;
goto v___jp_1778_;
}
}
else
{
size_t v___x_1792_; size_t v___x_1793_; lean_object* v___x_1794_; 
v___x_1792_ = ((size_t)0ULL);
v___x_1793_ = lean_usize_of_nat(v___x_1783_);
v___x_1794_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures_spec__0(v_permSuccesses_1772_, v_permFailures_1773_, v_children_1776_, v___x_1792_, v___x_1793_, v___x_1786_);
v_____do__lift_1779_ = v___x_1794_;
goto v___jp_1778_;
}
}
}
else
{
lean_object* v___x_1795_; 
lean_dec_ref(v_permFailures_1773_);
lean_dec_ref(v_permSuccesses_1772_);
v___x_1795_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3));
return v___x_1795_;
}
}
v___jp_1796_:
{
if (v___y_1797_ == 0)
{
lean_object* v___x_1798_; 
lean_dec_ref(v_permFailures_1773_);
lean_dec_ref(v_permSuccesses_1772_);
v___x_1798_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3));
return v___x_1798_;
}
else
{
goto v___jp_1782_;
}
}
v___jp_1799_:
{
lean_object* v___x_1800_; 
v___x_1800_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__0));
return v___x_1800_;
}
v___jp_1801_:
{
lean_object* v___x_1802_; 
v___x_1802_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__3));
return v___x_1802_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___lam__0___boxed(lean_object* v_permSuccesses_1816_, lean_object* v_permFailures_1817_, lean_object* v_td_1818_, lean_object* v_header_1819_, lean_object* v_children_1820_, lean_object* v___y_1821_){
_start:
{
lean_object* v_res_1822_; 
v_res_1822_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___lam__0(v_permSuccesses_1816_, v_permFailures_1817_, v_td_1818_, v_header_1819_, v_children_1820_);
lean_dec_ref(v_children_1820_);
lean_dec_ref(v_header_1819_);
lean_dec_ref(v_td_1818_);
return v_res_1822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures(lean_object* v_permSuccesses_1823_, lean_object* v_permFailures_1824_, lean_object* v_msg_1825_){
_start:
{
lean_object* v___f_1827_; lean_object* v___f_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; 
v___f_1827_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__0));
v___f_1828_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___lam__0___boxed), 6, 2);
lean_closure_set(v___f_1828_, 0, v_permSuccesses_1823_);
lean_closure_set(v___f_1828_, 1, v_permFailures_1824_);
v___x_1829_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___closed__1));
v___x_1830_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v___f_1828_, v___x_1829_, v___f_1827_, v_msg_1825_);
return v___x_1830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures___boxed(lean_object* v_permSuccesses_1831_, lean_object* v_permFailures_1832_, lean_object* v_msg_1833_, lean_object* v_a_1834_){
_start:
{
lean_object* v_res_1835_; 
v_res_1835_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures(v_permSuccesses_1831_, v_permFailures_1832_, v_msg_1833_);
return v_res_1835_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__1(lean_object* v_x_1836_, lean_object* v_x_1837_){
_start:
{
if (lean_obj_tag(v_x_1836_) == 0)
{
if (lean_obj_tag(v_x_1837_) == 0)
{
uint8_t v___x_1838_; 
v___x_1838_ = 1;
return v___x_1838_;
}
else
{
uint8_t v___x_1839_; 
v___x_1839_ = 0;
return v___x_1839_;
}
}
else
{
if (lean_obj_tag(v_x_1837_) == 0)
{
uint8_t v___x_1840_; 
v___x_1840_ = 0;
return v___x_1840_;
}
else
{
lean_object* v_val_1841_; lean_object* v_val_1842_; uint8_t v___x_1843_; uint8_t v___x_1844_; uint8_t v___x_1845_; 
v_val_1841_ = lean_ctor_get(v_x_1836_, 0);
v_val_1842_ = lean_ctor_get(v_x_1837_, 0);
v___x_1843_ = lean_unbox(v_val_1841_);
v___x_1844_ = lean_unbox(v_val_1842_);
v___x_1845_ = l_Lean_instBEqTraceResult_beq(v___x_1843_, v___x_1844_);
return v___x_1845_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__1___boxed(lean_object* v_x_1846_, lean_object* v_x_1847_){
_start:
{
uint8_t v_res_1848_; lean_object* v_r_1849_; 
v_res_1848_ = lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__1(v_x_1846_, v_x_1847_);
lean_dec(v_x_1847_);
lean_dec(v_x_1846_);
v_r_1849_ = lean_box(v_res_1848_);
return v_r_1849_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1851_; lean_object* v___x_1852_; 
v___x_1851_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__0));
v___x_1852_ = lean_string_utf8_byte_size(v___x_1851_);
return v___x_1852_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__2(void){
_start:
{
lean_object* v___x_1853_; lean_object* v___x_1854_; uint8_t v___x_1855_; 
v___x_1853_ = lean_unsigned_to_nat(0u);
v___x_1854_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1);
v___x_1855_ = lean_nat_dec_eq(v___x_1854_, v___x_1853_);
return v___x_1855_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3(void){
_start:
{
lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; 
v___x_1856_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__1);
v___x_1857_ = lean_unsigned_to_nat(0u);
v___x_1858_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__0));
v___x_1859_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1859_, 0, v___x_1858_);
lean_ctor_set(v___x_1859_, 1, v___x_1857_);
lean_ctor_set(v___x_1859_, 2, v___x_1856_);
return v___x_1859_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__4(void){
_start:
{
lean_object* v___x_1860_; lean_object* v___x_1861_; 
v___x_1860_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3);
v___x_1861_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_1860_);
return v___x_1861_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__5(void){
_start:
{
lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; 
v___x_1862_ = lean_unsigned_to_nat(0u);
v___x_1863_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__4, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__4_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__4);
v___x_1864_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__3);
v___x_1865_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_1865_, 0, v___x_1864_);
lean_ctor_set(v___x_1865_, 1, v___x_1863_);
lean_ctor_set(v___x_1865_, 2, v___x_1862_);
lean_ctor_set(v___x_1865_, 3, v___x_1862_);
return v___x_1865_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0(lean_object* v_s_1866_){
_start:
{
lean_object* v___y_1868_; uint8_t v___x_1871_; 
v___x_1871_ = lean_uint8_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__2, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__2_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__2);
if (v___x_1871_ == 0)
{
lean_object* v___x_1872_; 
v___x_1872_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__5, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__5_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___closed__5);
v___y_1868_ = v___x_1872_;
goto v___jp_1867_;
}
else
{
lean_object* v___x_1873_; 
v___x_1873_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0___closed__6));
v___y_1868_ = v___x_1873_;
goto v___jp_1867_;
}
v___jp_1867_:
{
uint8_t v___x_1869_; uint8_t v___x_1870_; 
v___x_1869_ = 0;
lean_inc(v___y_1868_);
v___x_1870_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures_spec__0_spec__0___redArg(v_s_1866_, v___y_1868_, v___x_1869_);
return v___x_1870_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0___boxed(lean_object* v_s_1874_){
_start:
{
uint8_t v_res_1875_; lean_object* v_r_1876_; 
v_res_1875_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0(v_s_1874_);
lean_dec_ref(v_s_1874_);
v_r_1876_ = lean_box(v_res_1875_);
return v_r_1876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0(lean_object* v_td_1880_, lean_object* v_header_1881_, lean_object* v_children_1882_){
_start:
{
lean_object* v_cls_1886_; lean_object* v_result_x3f_1887_; lean_object* v___x_1888_; uint8_t v___x_1889_; 
v_cls_1886_ = lean_ctor_get(v_td_1880_, 0);
v_result_x3f_1887_ = lean_ctor_get(v_td_1880_, 1);
v___x_1888_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthAppFailures___lam__0___closed__2));
v___x_1889_ = lean_name_eq(v_cls_1886_, v___x_1888_);
if (v___x_1889_ == 0)
{
lean_dec_ref(v_header_1881_);
goto v___jp_1884_;
}
else
{
lean_object* v___x_1890_; uint8_t v___y_1892_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; uint8_t v___x_1902_; 
v___x_1890_ = l_Lean_MessageData_toString(v_header_1881_);
v___x_1899_ = lean_unsigned_to_nat(0u);
v___x_1900_ = lean_string_utf8_byte_size(v___x_1890_);
lean_inc_ref(v___x_1890_);
v___x_1901_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1901_, 0, v___x_1890_);
lean_ctor_set(v___x_1901_, 1, v___x_1899_);
lean_ctor_set(v___x_1901_, 2, v___x_1900_);
v___x_1902_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__0(v___x_1901_);
lean_dec_ref_known(v___x_1901_, 3);
if (v___x_1902_ == 0)
{
v___y_1892_ = v___x_1902_;
goto v___jp_1891_;
}
else
{
lean_object* v___x_1903_; uint8_t v___x_1904_; 
v___x_1903_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0___closed__0));
v___x_1904_ = lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps_spec__1(v_result_x3f_1887_, v___x_1903_);
v___y_1892_ = v___x_1904_;
goto v___jp_1891_;
}
v___jp_1891_:
{
if (v___y_1892_ == 0)
{
lean_dec_ref(v___x_1890_);
goto v___jp_1884_;
}
else
{
lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; 
v___x_1893_ = lp_mathlib_Lean_MessageData_extractInstName(v___x_1890_);
lean_dec_ref(v___x_1890_);
v___x_1894_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2);
v___x_1895_ = lean_box(0);
v___x_1896_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__0___redArg(v___x_1894_, v___x_1893_, v___x_1895_);
v___x_1897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1897_, 0, v___x_1896_);
v___x_1898_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1898_, 0, v___x_1897_);
return v___x_1898_;
}
}
}
v___jp_1884_:
{
lean_object* v___x_1885_; 
v___x_1885_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__0));
return v___x_1885_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0___boxed(lean_object* v_td_1905_, lean_object* v_header_1906_, lean_object* v_children_1907_, lean_object* v___y_1908_){
_start:
{
lean_object* v_res_1909_; 
v_res_1909_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___lam__0(v_td_1905_, v_header_1906_, v_children_1907_);
lean_dec_ref(v_children_1907_);
lean_dec_ref(v_td_1905_);
return v_res_1909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps(lean_object* v_msg_1911_){
_start:
{
lean_object* v___f_1913_; lean_object* v___f_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; 
v___f_1913_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___closed__0));
v___f_1914_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___closed__0));
v___x_1915_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2);
v___x_1916_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM_go___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findLeafFailures_spec__1___redArg(v___f_1913_, v___x_1915_, v___f_1914_, v_msg_1911_);
return v___x_1916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps___boxed(lean_object* v_msg_1917_, lean_object* v_a_1918_){
_start:
{
lean_object* v_res_1919_; 
v_res_1919_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps(v_msg_1917_);
return v_res_1919_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__0(uint8_t v_x_1920_){
_start:
{
uint8_t v___x_1921_; uint8_t v___x_1922_; 
v___x_1921_ = 0;
v___x_1922_ = l_Lean_instBEqTraceResult_beq(v_x_1920_, v___x_1921_);
return v___x_1922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__0___boxed(lean_object* v_x_1923_){
_start:
{
uint8_t v_x_3507__boxed_1924_; uint8_t v_res_1925_; lean_object* v_r_1926_; 
v_x_3507__boxed_1924_ = lean_unbox(v_x_1923_);
v_res_1925_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__0(v_x_3507__boxed_1924_);
v_r_1926_ = lean_box(v_res_1925_);
return v_r_1926_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__1(uint8_t v_x_1927_){
_start:
{
uint8_t v___x_1928_; uint8_t v___x_1929_; 
v___x_1928_ = 1;
v___x_1929_ = l_Lean_instBEqTraceResult_beq(v_x_1927_, v___x_1928_);
return v___x_1929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__1___boxed(lean_object* v_x_1930_){
_start:
{
uint8_t v_x_3514__boxed_1931_; uint8_t v_res_1932_; lean_object* v_r_1933_; 
v_x_3514__boxed_1931_ = lean_unbox(v_x_1930_);
v_res_1932_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___lam__1(v_x_3514__boxed_1931_);
v_r_1933_ = lean_box(v_res_1932_);
return v_r_1933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0(lean_object* v_as_1936_, size_t v_sz_1937_, size_t v_i_1938_, lean_object* v_b_1939_){
_start:
{
lean_object* v___y_1942_; lean_object* v___y_1943_; uint8_t v___x_1948_; 
v___x_1948_ = lean_usize_dec_lt(v_i_1938_, v_sz_1937_);
if (v___x_1948_ == 0)
{
return v_b_1939_;
}
else
{
lean_object* v___f_1949_; lean_object* v_a_1950_; lean_object* v___x_1951_; lean_object* v_fst_1952_; lean_object* v_snd_1953_; lean_object* v_size_1954_; lean_object* v_buckets_1955_; lean_object* v_size_1956_; lean_object* v___f_1957_; lean_object* v___y_1959_; uint8_t v___x_1969_; 
v___f_1949_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__0));
v_a_1950_ = lean_array_uget_borrowed(v_as_1936_, v_i_1938_);
lean_inc(v_a_1950_);
v___x_1951_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks(v___f_1949_, v_a_1950_);
v_fst_1952_ = lean_ctor_get(v_b_1939_, 0);
lean_inc(v_fst_1952_);
v_snd_1953_ = lean_ctor_get(v_b_1939_, 1);
lean_inc(v_snd_1953_);
lean_dec_ref(v_b_1939_);
v_size_1954_ = lean_ctor_get(v_fst_1952_, 0);
v_buckets_1955_ = lean_ctor_get(v_fst_1952_, 1);
v_size_1956_ = lean_ctor_get(v___x_1951_, 0);
lean_inc(v_size_1956_);
v___f_1957_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___closed__1));
v___x_1969_ = lean_nat_dec_le(v_size_1954_, v_size_1956_);
lean_dec(v_size_1956_);
if (v___x_1969_ == 0)
{
lean_object* v___x_1970_; 
v___x_1970_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1(v_fst_1952_, v___x_1951_);
lean_dec_ref(v___x_1951_);
v___y_1959_ = v___x_1970_;
goto v___jp_1958_;
}
else
{
size_t v_sz_1971_; size_t v___x_1972_; lean_object* v___x_1973_; 
lean_inc_ref(v_buckets_1955_);
lean_dec(v_fst_1952_);
v_sz_1971_ = lean_array_size(v_buckets_1955_);
v___x_1972_ = ((size_t)0ULL);
v___x_1973_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3(v_buckets_1955_, v_sz_1971_, v___x_1972_, v___x_1951_);
lean_dec_ref(v_buckets_1955_);
v___y_1959_ = v___x_1973_;
goto v___jp_1958_;
}
v___jp_1958_:
{
lean_object* v___x_1960_; lean_object* v_size_1961_; lean_object* v_buckets_1962_; lean_object* v_size_1963_; uint8_t v___x_1964_; 
lean_inc(v_a_1950_);
v___x_1960_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks(v___f_1957_, v_a_1950_);
v_size_1961_ = lean_ctor_get(v_snd_1953_, 0);
v_buckets_1962_ = lean_ctor_get(v_snd_1953_, 1);
v_size_1963_ = lean_ctor_get(v___x_1960_, 0);
lean_inc(v_size_1963_);
v___x_1964_ = lean_nat_dec_le(v_size_1961_, v_size_1963_);
lean_dec(v_size_1963_);
if (v___x_1964_ == 0)
{
lean_object* v___x_1965_; 
v___x_1965_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1(v_snd_1953_, v___x_1960_);
lean_dec_ref(v___x_1960_);
v___y_1942_ = v___y_1959_;
v___y_1943_ = v___x_1965_;
goto v___jp_1941_;
}
else
{
size_t v_sz_1966_; size_t v___x_1967_; lean_object* v___x_1968_; 
lean_inc_ref(v_buckets_1962_);
lean_dec(v_snd_1953_);
v_sz_1966_ = lean_array_size(v_buckets_1962_);
v___x_1967_ = ((size_t)0ULL);
v___x_1968_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3(v_buckets_1962_, v_sz_1966_, v___x_1967_, v___x_1960_);
lean_dec_ref(v_buckets_1962_);
v___y_1942_ = v___y_1959_;
v___y_1943_ = v___x_1968_;
goto v___jp_1941_;
}
}
}
v___jp_1941_:
{
lean_object* v___x_1944_; size_t v___x_1945_; size_t v___x_1946_; 
v___x_1944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1944_, 0, v___y_1942_);
lean_ctor_set(v___x_1944_, 1, v___y_1943_);
v___x_1945_ = ((size_t)1ULL);
v___x_1946_ = lean_usize_add(v_i_1938_, v___x_1945_);
v_i_1938_ = v___x_1946_;
v_b_1939_ = v___x_1944_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0___boxed(lean_object* v_as_1974_, lean_object* v_sz_1975_, lean_object* v_i_1976_, lean_object* v_b_1977_, lean_object* v___y_1978_){
_start:
{
size_t v_sz_boxed_1979_; size_t v_i_boxed_1980_; lean_object* v_res_1981_; 
v_sz_boxed_1979_ = lean_unbox_usize(v_sz_1975_);
lean_dec(v_sz_1975_);
v_i_boxed_1980_ = lean_unbox_usize(v_i_1976_);
lean_dec(v_i_1976_);
v_res_1981_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0(v_as_1974_, v_sz_boxed_1979_, v_i_boxed_1980_, v_b_1977_);
lean_dec_ref(v_as_1974_);
return v_res_1981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__3(lean_object* v_as_1982_, size_t v_sz_1983_, size_t v_i_1984_, lean_object* v_b_1985_){
_start:
{
uint8_t v___x_1987_; 
v___x_1987_ = lean_usize_dec_lt(v_i_1984_, v_sz_1983_);
if (v___x_1987_ == 0)
{
return v_b_1985_;
}
else
{
lean_object* v_a_1988_; lean_object* v___x_1989_; lean_object* v___y_1991_; lean_object* v_size_1995_; lean_object* v_buckets_1996_; lean_object* v_size_1997_; uint8_t v___x_1998_; 
v_a_1988_ = lean_array_uget_borrowed(v_as_1982_, v_i_1984_);
lean_inc(v_a_1988_);
v___x_1989_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthSuccessApps(v_a_1988_);
v_size_1995_ = lean_ctor_get(v_b_1985_, 0);
v_buckets_1996_ = lean_ctor_get(v_b_1985_, 1);
v_size_1997_ = lean_ctor_get(v___x_1989_, 0);
lean_inc(v_size_1997_);
v___x_1998_ = lean_nat_dec_le(v_size_1995_, v_size_1997_);
lean_dec(v_size_1997_);
if (v___x_1998_ == 0)
{
lean_object* v___x_1999_; 
v___x_1999_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertMany___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__1(v_b_1985_, v___x_1989_);
lean_dec_ref(v___x_1989_);
v___y_1991_ = v___x_1999_;
goto v___jp_1990_;
}
else
{
size_t v_sz_2000_; size_t v___x_2001_; lean_object* v___x_2002_; 
lean_inc_ref(v_buckets_1996_);
lean_dec_ref(v_b_1985_);
v_sz_2000_ = lean_array_size(v_buckets_1996_);
v___x_2001_ = ((size_t)0ULL);
v___x_2002_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks_spec__3(v_buckets_1996_, v_sz_2000_, v___x_2001_, v___x_1989_);
lean_dec_ref(v_buckets_1996_);
v___y_1991_ = v___x_2002_;
goto v___jp_1990_;
}
v___jp_1990_:
{
size_t v___x_1992_; size_t v___x_1993_; 
v___x_1992_ = ((size_t)1ULL);
v___x_1993_ = lean_usize_add(v_i_1984_, v___x_1992_);
v_i_1984_ = v___x_1993_;
v_b_1985_ = v___y_1991_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__3___boxed(lean_object* v_as_2003_, lean_object* v_sz_2004_, lean_object* v_i_2005_, lean_object* v_b_2006_, lean_object* v___y_2007_){
_start:
{
size_t v_sz_boxed_2008_; size_t v_i_boxed_2009_; lean_object* v_res_2010_; 
v_sz_boxed_2008_ = lean_unbox_usize(v_sz_2004_);
lean_dec(v_sz_2004_);
v_i_boxed_2009_ = lean_unbox_usize(v_i_2005_);
lean_dec(v_i_2005_);
v_res_2010_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__3(v_as_2003_, v_sz_boxed_2008_, v_i_boxed_2009_, v_b_2006_);
lean_dec_ref(v_as_2003_);
return v_res_2010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__5(lean_object* v_val_2011_, lean_object* v_as_2012_, size_t v_i_2013_, size_t v_stop_2014_, lean_object* v_b_2015_){
_start:
{
uint8_t v___x_2017_; 
v___x_2017_ = lean_usize_dec_eq(v_i_2013_, v_stop_2014_);
if (v___x_2017_ == 0)
{
lean_object* v___x_2018_; lean_object* v_fst_2019_; lean_object* v___x_2020_; lean_object* v_val_2022_; lean_object* v___x_2026_; uint8_t v___x_2027_; 
v___x_2018_ = lean_array_uget_borrowed(v_as_2012_, v_i_2013_);
v_fst_2019_ = lean_ctor_get(v___x_2018_, 0);
lean_inc(v_fst_2019_);
v___x_2020_ = l_Lean_MessageData_toString(v_fst_2019_);
v___x_2026_ = lp_mathlib_Lean_MessageData_extractInstName(v___x_2020_);
lean_dec_ref(v___x_2020_);
v___x_2027_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures_spec__1___redArg(v_val_2011_, v___x_2026_);
lean_dec_ref(v___x_2026_);
if (v___x_2027_ == 0)
{
v_val_2022_ = v_b_2015_;
goto v___jp_2021_;
}
else
{
lean_object* v___x_2028_; 
lean_inc(v___x_2018_);
v___x_2028_ = lean_array_push(v_b_2015_, v___x_2018_);
v_val_2022_ = v___x_2028_;
goto v___jp_2021_;
}
v___jp_2021_:
{
size_t v___x_2023_; size_t v___x_2024_; 
v___x_2023_ = ((size_t)1ULL);
v___x_2024_ = lean_usize_add(v_i_2013_, v___x_2023_);
v_i_2013_ = v___x_2024_;
v_b_2015_ = v_val_2022_;
goto _start;
}
}
else
{
return v_b_2015_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__5___boxed(lean_object* v_val_2029_, lean_object* v_as_2030_, lean_object* v_i_2031_, lean_object* v_stop_2032_, lean_object* v_b_2033_, lean_object* v___y_2034_){
_start:
{
size_t v_i_boxed_2035_; size_t v_stop_boxed_2036_; lean_object* v_res_2037_; 
v_i_boxed_2035_ = lean_unbox_usize(v_i_2031_);
lean_dec(v_i_2031_);
v_stop_boxed_2036_ = lean_unbox_usize(v_stop_2032_);
lean_dec(v_stop_2032_);
v_res_2037_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__5(v_val_2029_, v_as_2030_, v_i_boxed_2035_, v_stop_boxed_2036_, v_b_2033_);
lean_dec_ref(v_as_2030_);
lean_dec_ref(v_val_2029_);
return v_res_2037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__2(size_t v_sz_2038_, size_t v_i_2039_, lean_object* v_bs_2040_){
_start:
{
uint8_t v___x_2042_; 
v___x_2042_ = lean_usize_dec_lt(v_i_2039_, v_sz_2038_);
if (v___x_2042_ == 0)
{
return v_bs_2040_;
}
else
{
lean_object* v_v_2043_; lean_object* v_fst_2044_; lean_object* v_snd_2045_; lean_object* v___x_2047_; uint8_t v_isShared_2048_; uint8_t v_isSharedCheck_2059_; 
v_v_2043_ = lean_array_uget(v_bs_2040_, v_i_2039_);
v_fst_2044_ = lean_ctor_get(v_v_2043_, 0);
v_snd_2045_ = lean_ctor_get(v_v_2043_, 1);
v_isSharedCheck_2059_ = !lean_is_exclusive(v_v_2043_);
if (v_isSharedCheck_2059_ == 0)
{
v___x_2047_ = v_v_2043_;
v_isShared_2048_ = v_isSharedCheck_2059_;
goto v_resetjp_2046_;
}
else
{
lean_inc(v_snd_2045_);
lean_inc(v_fst_2044_);
lean_dec(v_v_2043_);
v___x_2047_ = lean_box(0);
v_isShared_2048_ = v_isSharedCheck_2059_;
goto v_resetjp_2046_;
}
v_resetjp_2046_:
{
lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v_bs_x27_2051_; lean_object* v___x_2053_; 
v___x_2049_ = lp_mathlib_Lean_MessageData_dedupByString(v_snd_2045_);
lean_dec(v_snd_2045_);
v___x_2050_ = lean_unsigned_to_nat(0u);
v_bs_x27_2051_ = lean_array_uset(v_bs_2040_, v_i_2039_, v___x_2050_);
if (v_isShared_2048_ == 0)
{
lean_ctor_set(v___x_2047_, 1, v___x_2049_);
v___x_2053_ = v___x_2047_;
goto v_reusejp_2052_;
}
else
{
lean_object* v_reuseFailAlloc_2058_; 
v_reuseFailAlloc_2058_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2058_, 0, v_fst_2044_);
lean_ctor_set(v_reuseFailAlloc_2058_, 1, v___x_2049_);
v___x_2053_ = v_reuseFailAlloc_2058_;
goto v_reusejp_2052_;
}
v_reusejp_2052_:
{
size_t v___x_2054_; size_t v___x_2055_; lean_object* v___x_2056_; 
v___x_2054_ = ((size_t)1ULL);
v___x_2055_ = lean_usize_add(v_i_2039_, v___x_2054_);
v___x_2056_ = lean_array_uset(v_bs_x27_2051_, v_i_2039_, v___x_2053_);
v_i_2039_ = v___x_2055_;
v_bs_2040_ = v___x_2056_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__2___boxed(lean_object* v_sz_2060_, lean_object* v_i_2061_, lean_object* v_bs_2062_, lean_object* v___y_2063_){
_start:
{
size_t v_sz_boxed_2064_; size_t v_i_boxed_2065_; lean_object* v_res_2066_; 
v_sz_boxed_2064_ = lean_unbox_usize(v_sz_2060_);
lean_dec(v_sz_2060_);
v_i_boxed_2065_ = lean_unbox_usize(v_i_2061_);
lean_dec(v_i_2061_);
v_res_2066_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__2(v_sz_boxed_2064_, v_i_boxed_2065_, v_bs_2062_);
return v_res_2066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__4(lean_object* v___x_2067_, lean_object* v___x_2068_, lean_object* v_as_2069_, size_t v_sz_2070_, size_t v_i_2071_, lean_object* v_b_2072_){
_start:
{
uint8_t v___x_2074_; 
v___x_2074_ = lean_usize_dec_lt(v_i_2071_, v_sz_2070_);
if (v___x_2074_ == 0)
{
lean_dec_ref(v___x_2068_);
lean_dec_ref(v___x_2067_);
return v_b_2072_;
}
else
{
lean_object* v_a_2075_; lean_object* v___x_2076_; lean_object* v___x_2077_; size_t v___x_2078_; size_t v___x_2079_; 
v_a_2075_ = lean_array_uget_borrowed(v_as_2069_, v_i_2071_);
lean_inc(v_a_2075_);
lean_inc_ref(v___x_2068_);
lean_inc_ref(v___x_2067_);
v___x_2076_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findSynthFailures(v___x_2067_, v___x_2068_, v_a_2075_);
v___x_2077_ = l_Array_append___redArg(v_b_2072_, v___x_2076_);
lean_dec_ref(v___x_2076_);
v___x_2078_ = ((size_t)1ULL);
v___x_2079_ = lean_usize_add(v_i_2071_, v___x_2078_);
v_i_2071_ = v___x_2079_;
v_b_2072_ = v___x_2077_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__4___boxed(lean_object* v___x_2081_, lean_object* v___x_2082_, lean_object* v_as_2083_, lean_object* v_sz_2084_, lean_object* v_i_2085_, lean_object* v_b_2086_, lean_object* v___y_2087_){
_start:
{
size_t v_sz_boxed_2088_; size_t v_i_boxed_2089_; lean_object* v_res_2090_; 
v_sz_boxed_2088_ = lean_unbox_usize(v_sz_2084_);
lean_dec(v_sz_2084_);
v_i_boxed_2089_ = lean_unbox_usize(v_i_2085_);
lean_dec(v_i_2085_);
v_res_2090_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__4(v___x_2081_, v___x_2082_, v_as_2083_, v_sz_boxed_2088_, v_i_boxed_2089_, v_b_2086_);
lean_dec_ref(v_as_2083_);
return v_res_2090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__1(lean_object* v___x_2091_, lean_object* v___x_2092_, lean_object* v_as_2093_, size_t v_sz_2094_, size_t v_i_2095_, lean_object* v_b_2096_){
_start:
{
uint8_t v___x_2098_; 
v___x_2098_ = lean_usize_dec_lt(v_i_2095_, v_sz_2094_);
if (v___x_2098_ == 0)
{
lean_dec_ref(v___x_2092_);
lean_dec_ref(v___x_2091_);
return v_b_2096_;
}
else
{
lean_object* v_a_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; size_t v___x_2102_; size_t v___x_2103_; 
v_a_2099_ = lean_array_uget_borrowed(v_as_2093_, v_i_2095_);
lean_inc(v_a_2099_);
lean_inc_ref(v___x_2092_);
lean_inc_ref(v___x_2091_);
v___x_2100_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_findTransitionFailures(v___x_2091_, v___x_2092_, v_a_2099_);
v___x_2101_ = l_Array_append___redArg(v_b_2096_, v___x_2100_);
lean_dec_ref(v___x_2100_);
v___x_2102_ = ((size_t)1ULL);
v___x_2103_ = lean_usize_add(v_i_2095_, v___x_2102_);
v_i_2095_ = v___x_2103_;
v_b_2096_ = v___x_2101_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__1___boxed(lean_object* v___x_2105_, lean_object* v___x_2106_, lean_object* v_as_2107_, lean_object* v_sz_2108_, lean_object* v_i_2109_, lean_object* v_b_2110_, lean_object* v___y_2111_){
_start:
{
size_t v_sz_boxed_2112_; size_t v_i_boxed_2113_; lean_object* v_res_2114_; 
v_sz_boxed_2112_ = lean_unbox_usize(v_sz_2108_);
lean_dec(v_sz_2108_);
v_i_boxed_2113_ = lean_unbox_usize(v_i_2109_);
lean_dec(v_i_2109_);
v_res_2114_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__1(v___x_2105_, v___x_2106_, v_as_2107_, v_sz_boxed_2112_, v_i_boxed_2113_, v_b_2110_);
lean_dec_ref(v_as_2107_);
return v_res_2114_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__0(void){
_start:
{
lean_object* v_permSuccesses_2115_; lean_object* v___x_2116_; 
v_permSuccesses_2115_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2);
v___x_2116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2116_, 0, v_permSuccesses_2115_);
lean_ctor_set(v___x_2116_, 1, v_permSuccesses_2115_);
return v___x_2116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces(lean_object* v_strictMsgs_2119_, lean_object* v_permMsgs_2120_, uint8_t v_includeSynth_2121_){
_start:
{
lean_object* v___x_2123_; lean_object* v_permSuccesses_2124_; lean_object* v___x_2125_; size_t v_sz_2126_; size_t v___x_2127_; lean_object* v___x_2128_; lean_object* v_fst_2129_; lean_object* v_snd_2130_; lean_object* v___x_2132_; uint8_t v_isShared_2133_; uint8_t v_isSharedCheck_2157_; 
v___x_2123_ = lean_unsigned_to_nat(0u);
v_permSuccesses_2124_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_collectIsDefEqChecks___lam__0___closed__2);
v___x_2125_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__0, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__0);
v_sz_2126_ = lean_array_size(v_permMsgs_2120_);
v___x_2127_ = ((size_t)0ULL);
v___x_2128_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__0(v_permMsgs_2120_, v_sz_2126_, v___x_2127_, v___x_2125_);
v_fst_2129_ = lean_ctor_get(v___x_2128_, 0);
v_snd_2130_ = lean_ctor_get(v___x_2128_, 1);
v_isSharedCheck_2157_ = !lean_is_exclusive(v___x_2128_);
if (v_isSharedCheck_2157_ == 0)
{
v___x_2132_ = v___x_2128_;
v_isShared_2133_ = v_isSharedCheck_2157_;
goto v_resetjp_2131_;
}
else
{
lean_inc(v_snd_2130_);
lean_inc(v_fst_2129_);
lean_dec(v___x_2128_);
v___x_2132_ = lean_box(0);
v_isShared_2133_ = v_isSharedCheck_2157_;
goto v_resetjp_2131_;
}
v_resetjp_2131_:
{
lean_object* v___x_2134_; size_t v_sz_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v_val_2139_; lean_object* v___y_2146_; 
v___x_2134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__1));
v_sz_2135_ = lean_array_size(v_strictMsgs_2119_);
lean_inc(v_snd_2130_);
lean_inc(v_fst_2129_);
v___x_2136_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__1(v_fst_2129_, v_snd_2130_, v_strictMsgs_2119_, v_sz_2135_, v___x_2127_, v___x_2134_);
v___x_2137_ = lp_mathlib_Lean_MessageData_dedupByString(v___x_2136_);
lean_dec_ref(v___x_2136_);
if (v_includeSynth_2121_ == 0)
{
lean_object* v___x_2147_; 
lean_del_object(v___x_2132_);
lean_dec(v_snd_2130_);
lean_dec(v_fst_2129_);
v___x_2147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2147_, 0, v___x_2137_);
lean_ctor_set(v___x_2147_, 1, v___x_2134_);
return v___x_2147_;
}
else
{
lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; uint8_t v___x_2151_; 
v___x_2148_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__3(v_permMsgs_2120_, v_sz_2126_, v___x_2127_, v_permSuccesses_2124_);
v___x_2149_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__4(v_fst_2129_, v_snd_2130_, v_strictMsgs_2119_, v_sz_2135_, v___x_2127_, v___x_2134_);
v___x_2150_ = lean_array_get_size(v___x_2149_);
v___x_2151_ = lean_nat_dec_lt(v___x_2123_, v___x_2150_);
if (v___x_2151_ == 0)
{
lean_dec_ref(v___x_2149_);
lean_dec_ref(v___x_2148_);
v_val_2139_ = v___x_2134_;
goto v___jp_2138_;
}
else
{
uint8_t v___x_2152_; 
v___x_2152_ = lean_nat_dec_le(v___x_2150_, v___x_2150_);
if (v___x_2152_ == 0)
{
if (v___x_2151_ == 0)
{
lean_dec_ref(v___x_2149_);
lean_dec_ref(v___x_2148_);
v_val_2139_ = v___x_2134_;
goto v___jp_2138_;
}
else
{
size_t v___x_2153_; lean_object* v___x_2154_; 
v___x_2153_ = lean_usize_of_nat(v___x_2150_);
v___x_2154_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__5(v___x_2148_, v___x_2149_, v___x_2127_, v___x_2153_, v___x_2134_);
lean_dec_ref(v___x_2149_);
lean_dec_ref(v___x_2148_);
v___y_2146_ = v___x_2154_;
goto v___jp_2145_;
}
}
else
{
size_t v___x_2155_; lean_object* v___x_2156_; 
v___x_2155_ = lean_usize_of_nat(v___x_2150_);
v___x_2156_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__5(v___x_2148_, v___x_2149_, v___x_2127_, v___x_2155_, v___x_2134_);
lean_dec_ref(v___x_2149_);
lean_dec_ref(v___x_2148_);
v___y_2146_ = v___x_2156_;
goto v___jp_2145_;
}
}
}
v___jp_2138_:
{
size_t v_sz_2140_; lean_object* v___x_2141_; lean_object* v___x_2143_; 
v_sz_2140_ = lean_array_size(v_val_2139_);
v___x_2141_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces_spec__2(v_sz_2140_, v___x_2127_, v_val_2139_);
if (v_isShared_2133_ == 0)
{
lean_ctor_set(v___x_2132_, 1, v___x_2141_);
lean_ctor_set(v___x_2132_, 0, v___x_2137_);
v___x_2143_ = v___x_2132_;
goto v_reusejp_2142_;
}
else
{
lean_object* v_reuseFailAlloc_2144_; 
v_reuseFailAlloc_2144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2144_, 0, v___x_2137_);
lean_ctor_set(v_reuseFailAlloc_2144_, 1, v___x_2141_);
v___x_2143_ = v_reuseFailAlloc_2144_;
goto v_reusejp_2142_;
}
v_reusejp_2142_:
{
return v___x_2143_;
}
}
v___jp_2145_:
{
v_val_2139_ = v___y_2146_;
goto v___jp_2138_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___boxed(lean_object* v_strictMsgs_2158_, lean_object* v_permMsgs_2159_, lean_object* v_includeSynth_2160_, lean_object* v_a_2161_){
_start:
{
uint8_t v_includeSynth_boxed_2162_; lean_object* v_res_2163_; 
v_includeSynth_boxed_2162_ = lean_unbox(v_includeSynth_2160_);
v_res_2163_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces(v_strictMsgs_2158_, v_permMsgs_2159_, v_includeSynth_boxed_2162_);
lean_dec_ref(v_permMsgs_2159_);
lean_dec_ref(v_strictMsgs_2158_);
return v_res_2163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0(lean_object* v_s_2166_){
_start:
{
lean_object* v___x_2167_; 
v___x_2167_ = ((lean_object*)(lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0___closed__0));
return v___x_2167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0___boxed(lean_object* v_s_2168_){
_start:
{
lean_object* v_res_2169_; 
v_res_2169_ = lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0(v_s_2168_);
lean_dec_ref(v_s_2168_);
return v_res_2169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__2(lean_object* v_a_2170_, lean_object* v_a_2171_){
_start:
{
if (lean_obj_tag(v_a_2170_) == 0)
{
lean_object* v___x_2172_; 
v___x_2172_ = l_List_reverse___redArg(v_a_2171_);
return v___x_2172_;
}
else
{
lean_object* v_head_2173_; lean_object* v_tail_2174_; lean_object* v___x_2176_; uint8_t v_isShared_2177_; uint8_t v_isSharedCheck_2183_; 
v_head_2173_ = lean_ctor_get(v_a_2170_, 0);
v_tail_2174_ = lean_ctor_get(v_a_2170_, 1);
v_isSharedCheck_2183_ = !lean_is_exclusive(v_a_2170_);
if (v_isSharedCheck_2183_ == 0)
{
v___x_2176_ = v_a_2170_;
v_isShared_2177_ = v_isSharedCheck_2183_;
goto v_resetjp_2175_;
}
else
{
lean_inc(v_tail_2174_);
lean_inc(v_head_2173_);
lean_dec(v_a_2170_);
v___x_2176_ = lean_box(0);
v_isShared_2177_ = v_isSharedCheck_2183_;
goto v_resetjp_2175_;
}
v_resetjp_2175_:
{
lean_object* v___x_2178_; lean_object* v___x_2180_; 
v___x_2178_ = l_String_Slice_toString(v_head_2173_);
lean_dec(v_head_2173_);
if (v_isShared_2177_ == 0)
{
lean_ctor_set(v___x_2176_, 1, v_a_2171_);
lean_ctor_set(v___x_2176_, 0, v___x_2178_);
v___x_2180_ = v___x_2176_;
goto v_reusejp_2179_;
}
else
{
lean_object* v_reuseFailAlloc_2182_; 
v_reuseFailAlloc_2182_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2182_, 0, v___x_2178_);
lean_ctor_set(v_reuseFailAlloc_2182_, 1, v_a_2171_);
v___x_2180_ = v_reuseFailAlloc_2182_;
goto v_reusejp_2179_;
}
v_reusejp_2179_:
{
v_a_2170_ = v_tail_2174_;
v_a_2171_ = v___x_2180_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___redArg(lean_object* v_s_2184_, lean_object* v___x_2185_, lean_object* v___x_2186_, lean_object* v_a_2187_, lean_object* v_b_2188_){
_start:
{
lean_object* v_it_2190_; lean_object* v_startInclusive_2191_; lean_object* v_endExclusive_2192_; 
if (lean_obj_tag(v_a_2187_) == 0)
{
lean_object* v_currPos_2196_; lean_object* v_searcher_2197_; lean_object* v___x_2199_; uint8_t v_isShared_2200_; uint8_t v_isSharedCheck_2232_; 
v_currPos_2196_ = lean_ctor_get(v_a_2187_, 0);
v_searcher_2197_ = lean_ctor_get(v_a_2187_, 1);
v_isSharedCheck_2232_ = !lean_is_exclusive(v_a_2187_);
if (v_isSharedCheck_2232_ == 0)
{
v___x_2199_ = v_a_2187_;
v_isShared_2200_ = v_isSharedCheck_2232_;
goto v_resetjp_2198_;
}
else
{
lean_inc(v_searcher_2197_);
lean_inc(v_currPos_2196_);
lean_dec(v_a_2187_);
v___x_2199_ = lean_box(0);
v_isShared_2200_ = v_isSharedCheck_2232_;
goto v_resetjp_2198_;
}
v_resetjp_2198_:
{
uint8_t v___y_2212_; lean_object* v_startInclusive_2216_; lean_object* v_endExclusive_2217_; lean_object* v___x_2218_; uint8_t v___x_2219_; 
v_startInclusive_2216_ = lean_ctor_get(v___x_2185_, 1);
v_endExclusive_2217_ = lean_ctor_get(v___x_2185_, 2);
v___x_2218_ = lean_nat_sub(v_endExclusive_2217_, v_startInclusive_2216_);
v___x_2219_ = lean_nat_dec_eq(v_searcher_2197_, v___x_2218_);
lean_dec(v___x_2218_);
if (v___x_2219_ == 0)
{
uint32_t v___x_2220_; uint8_t v___y_2222_; uint32_t v___x_2227_; uint8_t v___x_2228_; 
v___x_2220_ = lean_string_utf8_get_fast(v_s_2184_, v_searcher_2197_);
v___x_2227_ = 32;
v___x_2228_ = lean_uint32_dec_eq(v___x_2220_, v___x_2227_);
if (v___x_2228_ == 0)
{
uint32_t v___x_2229_; uint8_t v___x_2230_; 
v___x_2229_ = 9;
v___x_2230_ = lean_uint32_dec_eq(v___x_2220_, v___x_2229_);
v___y_2222_ = v___x_2230_;
goto v___jp_2221_;
}
else
{
v___y_2222_ = v___x_2228_;
goto v___jp_2221_;
}
v___jp_2221_:
{
if (v___y_2222_ == 0)
{
uint32_t v___x_2223_; uint8_t v___x_2224_; 
v___x_2223_ = 13;
v___x_2224_ = lean_uint32_dec_eq(v___x_2220_, v___x_2223_);
if (v___x_2224_ == 0)
{
uint32_t v___x_2225_; uint8_t v___x_2226_; 
v___x_2225_ = 10;
v___x_2226_ = lean_uint32_dec_eq(v___x_2220_, v___x_2225_);
v___y_2212_ = v___x_2226_;
goto v___jp_2211_;
}
else
{
v___y_2212_ = v___x_2224_;
goto v___jp_2211_;
}
}
else
{
goto v___jp_2201_;
}
}
}
else
{
lean_object* v___x_2231_; 
lean_del_object(v___x_2199_);
lean_dec(v_searcher_2197_);
v___x_2231_ = lean_box(1);
lean_inc(v___x_2186_);
v_it_2190_ = v___x_2231_;
v_startInclusive_2191_ = v_currPos_2196_;
v_endExclusive_2192_ = v___x_2186_;
goto v___jp_2189_;
}
v___jp_2201_:
{
lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v_slice_2205_; lean_object* v_nextIt_2207_; 
v___x_2202_ = lean_string_utf8_next_fast(v_s_2184_, v_searcher_2197_);
v___x_2203_ = lean_nat_sub(v___x_2202_, v_searcher_2197_);
v___x_2204_ = lean_nat_add(v_searcher_2197_, v___x_2203_);
lean_dec(v___x_2203_);
v_slice_2205_ = l_String_Slice_subslice_x21(v___x_2185_, v_currPos_2196_, v_searcher_2197_);
lean_inc(v___x_2204_);
if (v_isShared_2200_ == 0)
{
lean_ctor_set(v___x_2199_, 1, v___x_2204_);
lean_ctor_set(v___x_2199_, 0, v___x_2204_);
v_nextIt_2207_ = v___x_2199_;
goto v_reusejp_2206_;
}
else
{
lean_object* v_reuseFailAlloc_2210_; 
v_reuseFailAlloc_2210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2210_, 0, v___x_2204_);
lean_ctor_set(v_reuseFailAlloc_2210_, 1, v___x_2204_);
v_nextIt_2207_ = v_reuseFailAlloc_2210_;
goto v_reusejp_2206_;
}
v_reusejp_2206_:
{
lean_object* v_startInclusive_2208_; lean_object* v_endExclusive_2209_; 
v_startInclusive_2208_ = lean_ctor_get(v_slice_2205_, 0);
lean_inc(v_startInclusive_2208_);
v_endExclusive_2209_ = lean_ctor_get(v_slice_2205_, 1);
lean_inc(v_endExclusive_2209_);
lean_dec_ref(v_slice_2205_);
v_it_2190_ = v_nextIt_2207_;
v_startInclusive_2191_ = v_startInclusive_2208_;
v_endExclusive_2192_ = v_endExclusive_2209_;
goto v___jp_2189_;
}
}
v___jp_2211_:
{
if (v___y_2212_ == 0)
{
lean_object* v___x_2213_; lean_object* v___x_2214_; 
lean_del_object(v___x_2199_);
v___x_2213_ = lean_string_utf8_next_fast(v_s_2184_, v_searcher_2197_);
lean_dec(v_searcher_2197_);
v___x_2214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2214_, 0, v_currPos_2196_);
lean_ctor_set(v___x_2214_, 1, v___x_2213_);
v_a_2187_ = v___x_2214_;
goto _start;
}
else
{
goto v___jp_2201_;
}
}
}
}
else
{
lean_dec(v___x_2186_);
lean_dec_ref(v_s_2184_);
return v_b_2188_;
}
v___jp_2189_:
{
lean_object* v___x_2193_; lean_object* v___x_2194_; 
lean_inc_ref(v_s_2184_);
v___x_2193_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2193_, 0, v_s_2184_);
lean_ctor_set(v___x_2193_, 1, v_startInclusive_2191_);
lean_ctor_set(v___x_2193_, 2, v_endExclusive_2192_);
v___x_2194_ = lean_array_push(v_b_2188_, v___x_2193_);
v_a_2187_ = v_it_2190_;
v_b_2188_ = v___x_2194_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___redArg___boxed(lean_object* v_s_2233_, lean_object* v___x_2234_, lean_object* v___x_2235_, lean_object* v_a_2236_, lean_object* v_b_2237_){
_start:
{
lean_object* v_res_2238_; 
v_res_2238_ = lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___redArg(v_s_2233_, v___x_2234_, v___x_2235_, v_a_2236_, v_b_2237_);
lean_dec_ref(v___x_2234_);
return v_res_2238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3(lean_object* v_a_2240_, lean_object* v_a_2241_){
_start:
{
if (lean_obj_tag(v_a_2240_) == 0)
{
lean_object* v___x_2242_; 
v___x_2242_ = l_List_reverse___redArg(v_a_2241_);
return v___x_2242_;
}
else
{
lean_object* v_head_2243_; lean_object* v_tail_2244_; lean_object* v___x_2246_; uint8_t v_isShared_2247_; uint8_t v_isSharedCheck_2255_; 
v_head_2243_ = lean_ctor_get(v_a_2240_, 0);
v_tail_2244_ = lean_ctor_get(v_a_2240_, 1);
v_isSharedCheck_2255_ = !lean_is_exclusive(v_a_2240_);
if (v_isSharedCheck_2255_ == 0)
{
v___x_2246_ = v_a_2240_;
v_isShared_2247_ = v_isSharedCheck_2255_;
goto v_resetjp_2245_;
}
else
{
lean_inc(v_tail_2244_);
lean_inc(v_head_2243_);
lean_dec(v_a_2240_);
v___x_2246_ = lean_box(0);
v_isShared_2247_ = v_isSharedCheck_2255_;
goto v_resetjp_2245_;
}
v_resetjp_2245_:
{
lean_object* v___x_2248_; uint8_t v___x_2249_; 
v___x_2248_ = ((lean_object*)(lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3___closed__0));
v___x_2249_ = lean_string_dec_eq(v_head_2243_, v___x_2248_);
if (v___x_2249_ == 0)
{
lean_object* v___x_2251_; 
if (v_isShared_2247_ == 0)
{
lean_ctor_set(v___x_2246_, 1, v_a_2241_);
v___x_2251_ = v___x_2246_;
goto v_reusejp_2250_;
}
else
{
lean_object* v_reuseFailAlloc_2253_; 
v_reuseFailAlloc_2253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2253_, 0, v_head_2243_);
lean_ctor_set(v_reuseFailAlloc_2253_, 1, v_a_2241_);
v___x_2251_ = v_reuseFailAlloc_2253_;
goto v_reusejp_2250_;
}
v_reusejp_2250_:
{
v_a_2240_ = v_tail_2244_;
v_a_2241_ = v___x_2251_;
goto _start;
}
}
else
{
lean_del_object(v___x_2246_);
lean_dec(v_head_2243_);
v_a_2240_ = v_tail_2244_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___lam__0(lean_object* v___x_2256_, lean_object* v_tail_2257_, lean_object* v_s_2258_){
_start:
{
lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; 
v___x_2259_ = lean_string_utf8_byte_size(v_s_2258_);
lean_inc(v___x_2256_);
lean_inc_ref(v_s_2258_);
v___x_2260_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2260_, 0, v_s_2258_);
lean_ctor_set(v___x_2260_, 1, v___x_2256_);
lean_ctor_set(v___x_2260_, 2, v___x_2259_);
v___x_2261_ = lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__0(v___x_2260_);
v___x_2262_ = lean_mk_empty_array_with_capacity(v___x_2256_);
lean_dec(v___x_2256_);
v___x_2263_ = lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___redArg(v_s_2258_, v___x_2260_, v___x_2259_, v___x_2261_, v___x_2262_);
lean_dec_ref_known(v___x_2260_, 3);
v___x_2264_ = lean_array_to_list(v___x_2263_);
lean_inc(v_tail_2257_);
v___x_2265_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__2(v___x_2264_, v_tail_2257_);
v___x_2266_ = lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3(v___x_2265_, v_tail_2257_);
return v___x_2266_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__4(lean_object* v_x_2267_, lean_object* v_x_2268_){
_start:
{
if (lean_obj_tag(v_x_2267_) == 0)
{
if (lean_obj_tag(v_x_2268_) == 0)
{
uint8_t v___x_2269_; 
v___x_2269_ = 1;
return v___x_2269_;
}
else
{
uint8_t v___x_2270_; 
v___x_2270_ = 0;
return v___x_2270_;
}
}
else
{
if (lean_obj_tag(v_x_2268_) == 0)
{
uint8_t v___x_2271_; 
v___x_2271_ = 0;
return v___x_2271_;
}
else
{
lean_object* v_head_2272_; lean_object* v_tail_2273_; lean_object* v_head_2274_; lean_object* v_tail_2275_; uint8_t v___x_2276_; 
v_head_2272_ = lean_ctor_get(v_x_2267_, 0);
v_tail_2273_ = lean_ctor_get(v_x_2267_, 1);
v_head_2274_ = lean_ctor_get(v_x_2268_, 0);
v_tail_2275_ = lean_ctor_get(v_x_2268_, 1);
v___x_2276_ = lean_string_dec_eq(v_head_2272_, v_head_2274_);
if (v___x_2276_ == 0)
{
return v___x_2276_;
}
else
{
v_x_2267_ = v_tail_2273_;
v_x_2268_ = v_tail_2275_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__4___boxed(lean_object* v_x_2278_, lean_object* v_x_2279_){
_start:
{
uint8_t v_res_2280_; lean_object* v_r_2281_; 
v_res_2280_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__4(v_x_2278_, v_x_2279_);
lean_dec(v_x_2279_);
lean_dec(v_x_2278_);
v_r_2281_ = lean_box(v_res_2280_);
return v_r_2281_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr(lean_object* v_raw_2283_){
_start:
{
lean_object* v___x_2284_; uint8_t v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; 
v___x_2284_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___closed__0));
v___x_2285_ = 0;
v___x_2286_ = lean_unsigned_to_nat(0u);
v___x_2287_ = lean_box(0);
v___x_2288_ = l_String_splitOnAux(v_raw_2283_, v___x_2284_, v___x_2286_, v___x_2286_, v___x_2286_, v___x_2287_);
if (lean_obj_tag(v___x_2288_) == 1)
{
lean_object* v_tail_2289_; 
v_tail_2289_ = lean_ctor_get(v___x_2288_, 1);
lean_inc(v_tail_2289_);
if (lean_obj_tag(v_tail_2289_) == 1)
{
lean_object* v_tail_2290_; 
v_tail_2290_ = lean_ctor_get(v_tail_2289_, 1);
lean_inc(v_tail_2290_);
if (lean_obj_tag(v_tail_2290_) == 0)
{
lean_object* v_head_2291_; lean_object* v_head_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; uint8_t v___x_2295_; 
v_head_2291_ = lean_ctor_get(v___x_2288_, 0);
lean_inc(v_head_2291_);
lean_dec_ref_known(v___x_2288_, 2);
v_head_2292_ = lean_ctor_get(v_tail_2289_, 0);
lean_inc(v_head_2292_);
lean_dec_ref_known(v_tail_2289_, 2);
v___x_2293_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___lam__0(v___x_2286_, v_tail_2290_, v_head_2291_);
v___x_2294_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___lam__0(v___x_2286_, v_tail_2290_, v_head_2292_);
v___x_2295_ = lp_mathlib_List_beq___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__4(v___x_2293_, v___x_2294_);
lean_dec(v___x_2294_);
lean_dec(v___x_2293_);
return v___x_2295_;
}
else
{
lean_dec_ref_known(v_tail_2289_, 2);
lean_dec(v_tail_2290_);
lean_dec_ref_known(v___x_2288_, 2);
return v___x_2285_;
}
}
else
{
lean_dec(v_tail_2289_);
lean_dec_ref_known(v___x_2288_, 2);
return v___x_2285_;
}
}
else
{
lean_dec(v___x_2288_);
return v___x_2285_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr___boxed(lean_object* v_raw_2296_){
_start:
{
uint8_t v_res_2297_; lean_object* v_r_2298_; 
v_res_2297_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr(v_raw_2296_);
lean_dec_ref(v_raw_2296_);
v_r_2298_ = lean_box(v_res_2297_);
return v_r_2298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1(lean_object* v_s_2299_, lean_object* v___x_2300_, lean_object* v___x_2301_, lean_object* v_inst_2302_, lean_object* v_R_2303_, lean_object* v_a_2304_, lean_object* v_b_2305_){
_start:
{
lean_object* v___x_2306_; 
v___x_2306_ = lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___redArg(v_s_2299_, v___x_2300_, v___x_2301_, v_a_2304_, v_b_2305_);
return v___x_2306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1___boxed(lean_object* v_s_2307_, lean_object* v___x_2308_, lean_object* v___x_2309_, lean_object* v_inst_2310_, lean_object* v_R_2311_, lean_object* v_a_2312_, lean_object* v_b_2313_){
_start:
{
lean_object* v_res_2314_; 
v_res_2314_ = lp_mathlib___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__1(v_s_2307_, v___x_2308_, v___x_2309_, v_inst_2310_, v_R_2311_, v_a_2312_, v_b_2313_);
lean_dec_ref(v___x_2308_);
return v_res_2314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(lean_object* v_o_2318_, lean_object* v_k_2319_, uint8_t v_v_2320_){
_start:
{
lean_object* v_map_2321_; uint8_t v_hasTrace_2322_; lean_object* v___x_2324_; uint8_t v_isShared_2325_; uint8_t v_isSharedCheck_2336_; 
v_map_2321_ = lean_ctor_get(v_o_2318_, 0);
v_hasTrace_2322_ = lean_ctor_get_uint8(v_o_2318_, sizeof(void*)*1);
v_isSharedCheck_2336_ = !lean_is_exclusive(v_o_2318_);
if (v_isSharedCheck_2336_ == 0)
{
v___x_2324_ = v_o_2318_;
v_isShared_2325_ = v_isSharedCheck_2336_;
goto v_resetjp_2323_;
}
else
{
lean_inc(v_map_2321_);
lean_dec(v_o_2318_);
v___x_2324_ = lean_box(0);
v_isShared_2325_ = v_isSharedCheck_2336_;
goto v_resetjp_2323_;
}
v_resetjp_2323_:
{
lean_object* v___x_2326_; lean_object* v___x_2327_; 
v___x_2326_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_2326_, 0, v_v_2320_);
lean_inc(v_k_2319_);
v___x_2327_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_2319_, v___x_2326_, v_map_2321_);
if (v_hasTrace_2322_ == 0)
{
lean_object* v___x_2328_; uint8_t v___x_2329_; lean_object* v___x_2331_; 
v___x_2328_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__1));
v___x_2329_ = l_Lean_Name_isPrefixOf(v___x_2328_, v_k_2319_);
lean_dec(v_k_2319_);
if (v_isShared_2325_ == 0)
{
lean_ctor_set(v___x_2324_, 0, v___x_2327_);
v___x_2331_ = v___x_2324_;
goto v_reusejp_2330_;
}
else
{
lean_object* v_reuseFailAlloc_2332_; 
v_reuseFailAlloc_2332_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2332_, 0, v___x_2327_);
v___x_2331_ = v_reuseFailAlloc_2332_;
goto v_reusejp_2330_;
}
v_reusejp_2330_:
{
lean_ctor_set_uint8(v___x_2331_, sizeof(void*)*1, v___x_2329_);
return v___x_2331_;
}
}
else
{
lean_object* v___x_2334_; 
lean_dec(v_k_2319_);
if (v_isShared_2325_ == 0)
{
lean_ctor_set(v___x_2324_, 0, v___x_2327_);
v___x_2334_ = v___x_2324_;
goto v_reusejp_2333_;
}
else
{
lean_object* v_reuseFailAlloc_2335_; 
v_reuseFailAlloc_2335_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2335_, 0, v___x_2327_);
lean_ctor_set_uint8(v_reuseFailAlloc_2335_, sizeof(void*)*1, v_hasTrace_2322_);
v___x_2334_ = v_reuseFailAlloc_2335_;
goto v_reusejp_2333_;
}
v_reusejp_2333_:
{
return v___x_2334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___boxed(lean_object* v_o_2337_, lean_object* v_k_2338_, lean_object* v_v_2339_){
_start:
{
uint8_t v_v_boxed_2340_; lean_object* v_res_2341_; 
v_v_boxed_2340_ = lean_unbox(v_v_2339_);
v_res_2341_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_o_2337_, v_k_2338_, v_v_boxed_2340_);
return v_res_2341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0(lean_object* v_o_2347_){
_start:
{
lean_object* v___x_2348_; uint8_t v___x_2349_; lean_object* v___x_2350_; 
v___x_2348_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__0___closed__2));
v___x_2349_ = 1;
v___x_2350_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_o_2347_, v___x_2348_, v___x_2349_);
return v___x_2350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1(lean_object* v_o_2355_){
_start:
{
lean_object* v___x_2356_; uint8_t v___x_2357_; lean_object* v___x_2358_; 
v___x_2356_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations___lam__1___closed__1));
v___x_2357_ = 1;
v___x_2358_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_o_2355_, v___x_2356_, v___x_2357_);
return v___x_2358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg(lean_object* v_f_2371_, lean_object* v_as_x27_2372_, lean_object* v_b_2373_){
_start:
{
if (lean_obj_tag(v_as_x27_2372_) == 0)
{
lean_dec_ref(v_f_2371_);
lean_inc_ref(v_b_2373_);
return v_b_2373_;
}
else
{
lean_object* v_head_2375_; lean_object* v_tail_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; uint8_t v___x_2380_; 
v_head_2375_ = lean_ctor_get(v_as_x27_2372_, 0);
v_tail_2376_ = lean_ctor_get(v_as_x27_2372_, 1);
lean_inc(v_head_2375_);
lean_inc_ref(v_f_2371_);
v___x_2377_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_withPPOptions(v_f_2371_, v_head_2375_);
lean_inc_ref(v___x_2377_);
v___x_2378_ = l_Lean_MessageData_toString(v___x_2377_);
v___x_2379_ = lean_box(0);
v___x_2380_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr(v___x_2378_);
lean_dec_ref(v___x_2378_);
if (v___x_2380_ == 0)
{
lean_object* v___x_2381_; lean_object* v___x_2382_; 
lean_dec_ref(v_f_2371_);
v___x_2381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2381_, 0, v___x_2377_);
v___x_2382_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2382_, 0, v___x_2381_);
lean_ctor_set(v___x_2382_, 1, v___x_2379_);
return v___x_2382_;
}
else
{
lean_object* v___x_2383_; 
lean_dec_ref(v___x_2377_);
v___x_2383_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg___closed__0));
v_as_x27_2372_ = v_tail_2376_;
v_b_2373_ = v___x_2383_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg___boxed(lean_object* v_f_2385_, lean_object* v_as_x27_2386_, lean_object* v_b_2387_, lean_object* v___y_2388_){
_start:
{
lean_object* v_res_2389_; 
v_res_2389_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg(v_f_2385_, v_as_x27_2386_, v_b_2387_);
lean_dec_ref(v_b_2387_);
lean_dec(v_as_x27_2386_);
return v_res_2389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__1(size_t v_sz_2390_, size_t v_i_2391_, lean_object* v_bs_2392_){
_start:
{
uint8_t v___x_2394_; 
v___x_2394_ = lean_usize_dec_lt(v_i_2391_, v_sz_2390_);
if (v___x_2394_ == 0)
{
return v_bs_2392_;
}
else
{
lean_object* v_v_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; lean_object* v_bs_x27_2398_; lean_object* v_val_2400_; uint8_t v___x_2405_; 
v_v_2395_ = lean_array_uget(v_bs_2392_, v_i_2391_);
lean_inc(v_v_2395_);
v___x_2396_ = l_Lean_MessageData_toString(v_v_2395_);
v___x_2397_ = lean_unsigned_to_nat(0u);
v_bs_x27_2398_ = lean_array_uset(v_bs_2392_, v_i_2391_, v___x_2397_);
v___x_2405_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr(v___x_2396_);
lean_dec_ref(v___x_2396_);
if (v___x_2405_ == 0)
{
v_val_2400_ = v_v_2395_;
goto v___jp_2399_;
}
else
{
lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; lean_object* v_fst_2409_; 
v___x_2406_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations));
v___x_2407_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg___closed__0));
lean_inc(v_v_2395_);
v___x_2408_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg(v_v_2395_, v___x_2406_, v___x_2407_);
v_fst_2409_ = lean_ctor_get(v___x_2408_, 0);
lean_inc(v_fst_2409_);
lean_dec_ref(v___x_2408_);
if (lean_obj_tag(v_fst_2409_) == 0)
{
v_val_2400_ = v_v_2395_;
goto v___jp_2399_;
}
else
{
lean_object* v_val_2410_; 
lean_dec(v_v_2395_);
v_val_2410_ = lean_ctor_get(v_fst_2409_, 0);
lean_inc(v_val_2410_);
lean_dec_ref_known(v_fst_2409_, 1);
v_val_2400_ = v_val_2410_;
goto v___jp_2399_;
}
}
v___jp_2399_:
{
size_t v___x_2401_; size_t v___x_2402_; lean_object* v___x_2403_; 
v___x_2401_ = ((size_t)1ULL);
v___x_2402_ = lean_usize_add(v_i_2391_, v___x_2401_);
v___x_2403_ = lean_array_uset(v_bs_x27_2398_, v_i_2391_, v_val_2400_);
v_i_2391_ = v___x_2402_;
v_bs_2392_ = v___x_2403_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__1___boxed(lean_object* v_sz_2411_, lean_object* v_i_2412_, lean_object* v_bs_2413_, lean_object* v___y_2414_){
_start:
{
size_t v_sz_boxed_2415_; size_t v_i_boxed_2416_; lean_object* v_res_2417_; 
v_sz_boxed_2415_ = lean_unbox_usize(v_sz_2411_);
lean_dec(v_sz_2411_);
v_i_boxed_2416_ = lean_unbox_usize(v_i_2412_);
lean_dec(v_i_2412_);
v_res_2417_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__1(v_sz_boxed_2415_, v_i_boxed_2416_, v_bs_2413_);
return v_res_2417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures(lean_object* v_failures_2418_){
_start:
{
size_t v_sz_2420_; size_t v___x_2421_; lean_object* v___x_2422_; 
v_sz_2420_ = lean_array_size(v_failures_2418_);
v___x_2421_ = ((size_t)0ULL);
v___x_2422_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__1(v_sz_2420_, v___x_2421_, v_failures_2418_);
return v___x_2422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures___boxed(lean_object* v_failures_2423_, lean_object* v_a_2424_){
_start:
{
lean_object* v_res_2425_; 
v_res_2425_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures(v_failures_2423_);
return v_res_2425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0(lean_object* v_f_2426_, lean_object* v_as_2427_, lean_object* v_as_x27_2428_, lean_object* v_b_2429_, lean_object* v_a_2430_){
_start:
{
lean_object* v___x_2432_; 
v___x_2432_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___redArg(v_f_2426_, v_as_x27_2428_, v_b_2429_);
return v___x_2432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0___boxed(lean_object* v_f_2433_, lean_object* v_as_2434_, lean_object* v_as_x27_2435_, lean_object* v_b_2436_, lean_object* v_a_2437_, lean_object* v___y_2438_){
_start:
{
lean_object* v_res_2439_; 
v_res_2439_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures_spec__0(v_f_2433_, v_as_2434_, v_as_x27_2435_, v_b_2436_, v_a_2437_);
lean_dec_ref(v_b_2436_);
lean_dec(v_as_x27_2435_);
lean_dec(v_as_2434_);
return v_res_2439_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2(void){
_start:
{
lean_object* v___x_2443_; lean_object* v___x_2444_; 
v___x_2443_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__1));
v___x_2444_ = l_Lean_MessageData_ofFormat(v___x_2443_);
return v___x_2444_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4(void){
_start:
{
lean_object* v___x_2446_; lean_object* v___x_2447_; 
v___x_2446_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__3));
v___x_2447_ = l_Lean_stringToMessageData(v___x_2446_);
return v___x_2447_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6(void){
_start:
{
lean_object* v___x_2449_; lean_object* v___x_2450_; 
v___x_2449_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__5));
v___x_2450_ = l_Lean_stringToMessageData(v___x_2449_);
return v___x_2450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0(lean_object* v_kind_2451_, lean_object* v_inst_2452_, lean_object* v_inst_2453_, lean_object* v_inst_2454_, lean_object* v_inst_2455_, lean_object* v_____s_2456_){
_start:
{
lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v_report_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; lean_object* v___x_2463_; lean_object* v___x_2464_; lean_object* v___x_2465_; lean_object* v___x_2466_; 
v___x_2457_ = lean_array_to_list(v_____s_2456_);
v___x_2458_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v_report_2459_ = l_Lean_MessageData_joinSep(v___x_2457_, v___x_2458_);
v___x_2460_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_2461_ = l_Lean_stringToMessageData(v_kind_2451_);
v___x_2462_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2462_, 0, v___x_2460_);
lean_ctor_set(v___x_2462_, 1, v___x_2461_);
v___x_2463_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6);
v___x_2464_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2464_, 0, v___x_2462_);
lean_ctor_set(v___x_2464_, 1, v___x_2463_);
v___x_2465_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2465_, 0, v___x_2464_);
lean_ctor_set(v___x_2465_, 1, v_report_2459_);
v___x_2466_ = l_Lean_logWarning___redArg(v_inst_2452_, v_inst_2453_, v_inst_2454_, v_inst_2455_, v___x_2465_);
return v___x_2466_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1(void){
_start:
{
lean_object* v___x_2468_; lean_object* v___x_2469_; 
v___x_2468_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__0));
v___x_2469_ = l_Lean_stringToMessageData(v___x_2468_);
return v___x_2469_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3(void){
_start:
{
lean_object* v___x_2471_; lean_object* v___x_2472_; 
v___x_2471_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__2));
v___x_2472_ = l_Lean_stringToMessageData(v___x_2471_);
return v___x_2472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1(lean_object* v_failureEmoji_2473_, lean_object* v_f_2474_){
_start:
{
lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; 
v___x_2475_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1);
v___x_2476_ = l_Lean_stringToMessageData(v_failureEmoji_2473_);
v___x_2477_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2477_, 0, v___x_2475_);
lean_ctor_set(v___x_2477_, 1, v___x_2476_);
v___x_2478_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3);
v___x_2479_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2479_, 0, v___x_2477_);
lean_ctor_set(v___x_2479_, 1, v___x_2478_);
v___x_2480_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2480_, 0, v___x_2479_);
lean_ctor_set(v___x_2480_, 1, v_f_2474_);
return v___x_2480_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1(void){
_start:
{
lean_object* v___x_2482_; lean_object* v___x_2483_; 
v___x_2482_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__0));
v___x_2483_ = l_Lean_stringToMessageData(v___x_2482_);
return v___x_2483_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2(void){
_start:
{
lean_object* v___x_2484_; lean_object* v___x_2485_; 
v___x_2484_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__0));
v___x_2485_ = l_Lean_stringToMessageData(v___x_2484_);
return v___x_2485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2(lean_object* v_toApplicative_2486_, lean_object* v___f_2487_, lean_object* v_failureEmoji_2488_, lean_object* v_a_2489_, lean_object* v_x_2490_, lean_object* v___y_2491_){
_start:
{
lean_object* v_fst_2492_; lean_object* v_snd_2493_; lean_object* v___x_2495_; uint8_t v_isShared_2496_; uint8_t v_isSharedCheck_2517_; 
v_fst_2492_ = lean_ctor_get(v_a_2489_, 0);
v_snd_2493_ = lean_ctor_get(v_a_2489_, 1);
v_isSharedCheck_2517_ = !lean_is_exclusive(v_a_2489_);
if (v_isSharedCheck_2517_ == 0)
{
v___x_2495_ = v_a_2489_;
v_isShared_2496_ = v_isSharedCheck_2517_;
goto v_resetjp_2494_;
}
else
{
lean_inc(v_snd_2493_);
lean_inc(v_fst_2492_);
lean_dec(v_a_2489_);
v___x_2495_ = lean_box(0);
v_isShared_2496_ = v_isSharedCheck_2517_;
goto v_resetjp_2494_;
}
v_resetjp_2494_:
{
lean_object* v_toPure_2497_; lean_object* v___x_2498_; lean_object* v___x_2499_; lean_object* v___x_2500_; lean_object* v___x_2501_; lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2506_; 
v_toPure_2497_ = lean_ctor_get(v_toApplicative_2486_, 1);
lean_inc(v_toPure_2497_);
lean_dec_ref(v_toApplicative_2486_);
v___x_2498_ = lean_array_to_list(v_snd_2493_);
v___x_2499_ = lean_box(0);
v___x_2500_ = l_List_mapTR_loop___redArg(v___f_2487_, v___x_2498_, v___x_2499_);
v___x_2501_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v___x_2502_ = l_Lean_MessageData_joinSep(v___x_2500_, v___x_2501_);
v___x_2503_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1);
v___x_2504_ = l_Lean_stringToMessageData(v_failureEmoji_2488_);
if (v_isShared_2496_ == 0)
{
lean_ctor_set_tag(v___x_2495_, 7);
lean_ctor_set(v___x_2495_, 1, v___x_2504_);
lean_ctor_set(v___x_2495_, 0, v___x_2503_);
v___x_2506_ = v___x_2495_;
goto v_reusejp_2505_;
}
else
{
lean_object* v_reuseFailAlloc_2516_; 
v_reuseFailAlloc_2516_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2516_, 0, v___x_2503_);
lean_ctor_set(v_reuseFailAlloc_2516_, 1, v___x_2504_);
v___x_2506_ = v_reuseFailAlloc_2516_;
goto v_reusejp_2505_;
}
v_reusejp_2505_:
{
lean_object* v___x_2507_; lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; 
v___x_2507_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3);
v___x_2508_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2508_, 0, v___x_2506_);
lean_ctor_set(v___x_2508_, 1, v___x_2507_);
v___x_2509_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2509_, 0, v___x_2508_);
lean_ctor_set(v___x_2509_, 1, v_fst_2492_);
v___x_2510_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2);
v___x_2511_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2511_, 0, v___x_2509_);
lean_ctor_set(v___x_2511_, 1, v___x_2510_);
v___x_2512_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2512_, 0, v___x_2511_);
lean_ctor_set(v___x_2512_, 1, v___x_2502_);
v___x_2513_ = lean_array_push(v___y_2491_, v___x_2512_);
v___x_2514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2514_, 0, v___x_2513_);
v___x_2515_ = lean_apply_2(v_toPure_2497_, lean_box(0), v___x_2514_);
return v___x_2515_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__3(lean_object* v_failureEmoji_2518_, lean_object* v_f_2519_){
_start:
{
lean_object* v___x_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; 
v___x_2520_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1);
v___x_2521_ = l_Lean_stringToMessageData(v_failureEmoji_2518_);
v___x_2522_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2522_, 0, v___x_2520_);
lean_ctor_set(v___x_2522_, 1, v___x_2521_);
v___x_2523_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3);
v___x_2524_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2524_, 0, v___x_2522_);
lean_ctor_set(v___x_2524_, 1, v___x_2523_);
v___x_2525_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2525_, 0, v___x_2524_);
lean_ctor_set(v___x_2525_, 1, v_f_2519_);
return v___x_2525_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0(void){
_start:
{
uint8_t v___x_2526_; lean_object* v_failureEmoji_2527_; 
v___x_2526_ = 1;
v_failureEmoji_2527_ = l_Lean_TraceResult_toEmoji(v___x_2526_);
return v_failureEmoji_2527_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__1(void){
_start:
{
lean_object* v_failureEmoji_2528_; lean_object* v___f_2529_; 
v_failureEmoji_2528_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0);
v___f_2529_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1), 2, 1);
lean_closure_set(v___f_2529_, 0, v_failureEmoji_2528_);
return v___f_2529_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__2(void){
_start:
{
lean_object* v_failureEmoji_2530_; lean_object* v___f_2531_; 
v_failureEmoji_2530_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0);
v___f_2531_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__3), 2, 1);
lean_closure_set(v___f_2531_, 0, v_failureEmoji_2530_);
return v___f_2531_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4(void){
_start:
{
lean_object* v___x_2533_; lean_object* v___x_2534_; 
v___x_2533_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__3));
v___x_2534_ = l_Lean_stringToMessageData(v___x_2533_);
return v___x_2534_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6(void){
_start:
{
lean_object* v___x_2536_; lean_object* v___x_2537_; 
v___x_2536_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__5));
v___x_2537_ = l_Lean_stringToMessageData(v___x_2536_);
return v___x_2537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg(lean_object* v_inst_2538_, lean_object* v_inst_2539_, lean_object* v_inst_2540_, lean_object* v_inst_2541_, lean_object* v_kind_2542_, lean_object* v_uniqueFailures_2543_, lean_object* v_synthResults_2544_){
_start:
{
lean_object* v_failureEmoji_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; uint8_t v___x_2548_; 
v_failureEmoji_2545_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0);
v___x_2546_ = lean_array_get_size(v_synthResults_2544_);
v___x_2547_ = lean_unsigned_to_nat(0u);
v___x_2548_ = lean_nat_dec_eq(v___x_2546_, v___x_2547_);
if (v___x_2548_ == 0)
{
lean_object* v_toApplicative_2549_; lean_object* v_toBind_2550_; lean_object* v___f_2551_; lean_object* v___f_2552_; lean_object* v___f_2553_; lean_object* v_entries_2554_; size_t v_sz_2555_; size_t v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; 
lean_dec_ref(v_uniqueFailures_2543_);
v_toApplicative_2549_ = lean_ctor_get(v_inst_2538_, 0);
v_toBind_2550_ = lean_ctor_get(v_inst_2538_, 1);
lean_inc(v_toBind_2550_);
lean_inc_ref(v_inst_2538_);
v___f_2551_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0), 6, 5);
lean_closure_set(v___f_2551_, 0, v_kind_2542_);
lean_closure_set(v___f_2551_, 1, v_inst_2538_);
lean_closure_set(v___f_2551_, 2, v_inst_2539_);
lean_closure_set(v___f_2551_, 3, v_inst_2540_);
lean_closure_set(v___f_2551_, 4, v_inst_2541_);
v___f_2552_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__1);
lean_inc_ref(v_toApplicative_2549_);
v___f_2553_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2), 6, 3);
lean_closure_set(v___f_2553_, 0, v_toApplicative_2549_);
lean_closure_set(v___f_2553_, 1, v___f_2552_);
lean_closure_set(v___f_2553_, 2, v_failureEmoji_2545_);
v_entries_2554_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__1));
v_sz_2555_ = lean_array_size(v_synthResults_2544_);
v___x_2556_ = ((size_t)0ULL);
v___x_2557_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_2538_, v_synthResults_2544_, v___f_2553_, v_sz_2555_, v___x_2556_, v_entries_2554_);
v___x_2558_ = lean_apply_4(v_toBind_2550_, lean_box(0), lean_box(0), v___x_2557_, v___f_2551_);
return v___x_2558_;
}
else
{
lean_object* v___x_2559_; uint8_t v___x_2560_; 
lean_dec_ref(v_synthResults_2544_);
v___x_2559_ = lean_array_get_size(v_uniqueFailures_2543_);
v___x_2560_ = lean_nat_dec_eq(v___x_2559_, v___x_2547_);
if (v___x_2560_ == 0)
{
lean_object* v___f_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; lean_object* v_failureList_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; 
v___f_2561_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__2);
v___x_2562_ = lean_array_to_list(v_uniqueFailures_2543_);
v___x_2563_ = lean_box(0);
v___x_2564_ = l_List_mapTR_loop___redArg(v___f_2561_, v___x_2562_, v___x_2563_);
v___x_2565_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v_failureList_2566_ = l_Lean_MessageData_joinSep(v___x_2564_, v___x_2565_);
v___x_2567_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_2568_ = l_Lean_stringToMessageData(v_kind_2542_);
v___x_2569_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2569_, 0, v___x_2567_);
lean_ctor_set(v___x_2569_, 1, v___x_2568_);
v___x_2570_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4);
v___x_2571_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2571_, 0, v___x_2569_);
lean_ctor_set(v___x_2571_, 1, v___x_2570_);
v___x_2572_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2572_, 0, v___x_2571_);
lean_ctor_set(v___x_2572_, 1, v_failureList_2566_);
v___x_2573_ = l_Lean_logWarning___redArg(v_inst_2538_, v_inst_2539_, v_inst_2540_, v_inst_2541_, v___x_2572_);
return v___x_2573_;
}
else
{
lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; 
lean_dec_ref(v_uniqueFailures_2543_);
v___x_2574_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_2575_ = l_Lean_stringToMessageData(v_kind_2542_);
v___x_2576_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2576_, 0, v___x_2574_);
lean_ctor_set(v___x_2576_, 1, v___x_2575_);
v___x_2577_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6);
v___x_2578_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2578_, 0, v___x_2576_);
lean_ctor_set(v___x_2578_, 1, v___x_2577_);
v___x_2579_ = l_Lean_logWarning___redArg(v_inst_2538_, v_inst_2539_, v_inst_2540_, v_inst_2541_, v___x_2578_);
return v___x_2579_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse(lean_object* v_m_2580_, lean_object* v_inst_2581_, lean_object* v_inst_2582_, lean_object* v_inst_2583_, lean_object* v_inst_2584_, lean_object* v_kind_2585_, lean_object* v_uniqueFailures_2586_, lean_object* v_synthResults_2587_){
_start:
{
lean_object* v___x_2588_; 
v___x_2588_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg(v_inst_2581_, v_inst_2582_, v_inst_2583_, v_inst_2584_, v_kind_2585_, v_uniqueFailures_2586_, v_synthResults_2587_);
return v___x_2588_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; 
v___x_2626_ = lean_box(0);
v___x_2627_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2628_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2628_, 0, v___x_2627_);
lean_ctor_set(v___x_2628_, 1, v___x_2626_);
return v___x_2628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2630_; lean_object* v___x_2631_; 
v___x_2630_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0);
v___x_2631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2631_, 0, v___x_2630_);
return v___x_2631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___boxed(lean_object* v___y_2632_){
_start:
{
lean_object* v_res_2633_; 
v_res_2633_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg();
return v_res_2633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0(lean_object* v_00_u03b1_2634_, lean_object* v___y_2635_, lean_object* v___y_2636_, lean_object* v___y_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_){
_start:
{
lean_object* v___x_2644_; 
v___x_2644_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg();
return v___x_2644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___boxed(lean_object* v_00_u03b1_2645_, lean_object* v___y_2646_, lean_object* v___y_2647_, lean_object* v___y_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_){
_start:
{
lean_object* v_res_2655_; 
v_res_2655_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0(v_00_u03b1_2645_, v___y_2646_, v___y_2647_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_, v___y_2652_, v___y_2653_);
lean_dec(v___y_2653_);
lean_dec_ref(v___y_2652_);
lean_dec(v___y_2651_);
lean_dec_ref(v___y_2650_);
lean_dec(v___y_2649_);
lean_dec_ref(v___y_2648_);
lean_dec(v___y_2647_);
lean_dec_ref(v___y_2646_);
return v_res_2655_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(lean_object* v_opts_2656_, lean_object* v_opt_2657_){
_start:
{
lean_object* v_name_2658_; lean_object* v_defValue_2659_; lean_object* v_map_2660_; lean_object* v___x_2661_; 
v_name_2658_ = lean_ctor_get(v_opt_2657_, 0);
v_defValue_2659_ = lean_ctor_get(v_opt_2657_, 1);
v_map_2660_ = lean_ctor_get(v_opts_2656_, 0);
v___x_2661_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2660_, v_name_2658_);
if (lean_obj_tag(v___x_2661_) == 0)
{
uint8_t v___x_2662_; 
v___x_2662_ = lean_unbox(v_defValue_2659_);
return v___x_2662_;
}
else
{
lean_object* v_val_2663_; 
v_val_2663_ = lean_ctor_get(v___x_2661_, 0);
lean_inc(v_val_2663_);
lean_dec_ref_known(v___x_2661_, 1);
if (lean_obj_tag(v_val_2663_) == 1)
{
uint8_t v_v_2664_; 
v_v_2664_ = lean_ctor_get_uint8(v_val_2663_, 0);
lean_dec_ref_known(v_val_2663_, 0);
return v_v_2664_;
}
else
{
uint8_t v___x_2665_; 
lean_dec(v_val_2663_);
v___x_2665_ = lean_unbox(v_defValue_2659_);
return v___x_2665_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1___boxed(lean_object* v_opts_2666_, lean_object* v_opt_2667_){
_start:
{
uint8_t v_res_2668_; lean_object* v_r_2669_; 
v_res_2668_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(v_opts_2666_, v_opt_2667_);
lean_dec_ref(v_opt_2667_);
lean_dec_ref(v_opts_2666_);
v_r_2669_ = lean_box(v_res_2668_);
return v_r_2669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2(lean_object* v_opts_2670_, lean_object* v_opt_2671_){
_start:
{
lean_object* v_name_2672_; lean_object* v_defValue_2673_; lean_object* v_map_2674_; lean_object* v___x_2675_; 
v_name_2672_ = lean_ctor_get(v_opt_2671_, 0);
v_defValue_2673_ = lean_ctor_get(v_opt_2671_, 1);
v_map_2674_ = lean_ctor_get(v_opts_2670_, 0);
v___x_2675_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2674_, v_name_2672_);
if (lean_obj_tag(v___x_2675_) == 0)
{
lean_inc(v_defValue_2673_);
return v_defValue_2673_;
}
else
{
lean_object* v_val_2676_; 
v_val_2676_ = lean_ctor_get(v___x_2675_, 0);
lean_inc(v_val_2676_);
lean_dec_ref_known(v___x_2675_, 1);
if (lean_obj_tag(v_val_2676_) == 3)
{
lean_object* v_v_2677_; 
v_v_2677_ = lean_ctor_get(v_val_2676_, 0);
lean_inc(v_v_2677_);
lean_dec_ref_known(v_val_2676_, 1);
return v_v_2677_;
}
else
{
lean_dec(v_val_2676_);
lean_inc(v_defValue_2673_);
return v_defValue_2673_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2___boxed(lean_object* v_opts_2678_, lean_object* v_opt_2679_){
_start:
{
lean_object* v_res_2680_; 
v_res_2680_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2(v_opts_2678_, v_opt_2679_);
lean_dec_ref(v_opt_2679_);
lean_dec_ref(v_opts_2678_);
return v_res_2680_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; 
v___x_2681_ = lean_unsigned_to_nat(32u);
v___x_2682_ = lean_mk_empty_array_with_capacity(v___x_2681_);
v___x_2683_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2683_, 0, v___x_2682_);
return v___x_2683_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__6(void){
_start:
{
lean_object* v___x_2696_; 
v___x_2696_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2696_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__7(void){
_start:
{
lean_object* v___x_2697_; lean_object* v___x_2698_; 
v___x_2697_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__6, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__6);
v___x_2698_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2698_, 0, v___x_2697_);
return v___x_2698_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8(void){
_start:
{
lean_object* v___x_2699_; lean_object* v___x_2700_; 
v___x_2699_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__7, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__7);
v___x_2700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2700_, 0, v___x_2699_);
lean_ctor_set(v___x_2700_, 1, v___x_2699_);
return v___x_2700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0(lean_object* v___x_2701_, lean_object* v_traces_2702_, uint8_t v___x_2703_, lean_object* v___x_2704_, uint8_t v_strict_2705_, lean_object* v___y_2706_, lean_object* v___y_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_, lean_object* v___y_2713_){
_start:
{
lean_object* v_a_2716_; lean_object* v___x_2749_; lean_object* v_traceState_2750_; lean_object* v_env_2751_; lean_object* v_nextMacroScope_2752_; lean_object* v_ngen_2753_; lean_object* v_auxDeclNGen_2754_; lean_object* v_cache_2755_; lean_object* v_messages_2756_; lean_object* v_infoState_2757_; lean_object* v_snapshotTasks_2758_; lean_object* v___x_2760_; uint8_t v_isShared_2761_; uint8_t v_isSharedCheck_2914_; 
v___x_2749_ = lean_st_ref_take(v___y_2713_);
v_traceState_2750_ = lean_ctor_get(v___x_2749_, 4);
v_env_2751_ = lean_ctor_get(v___x_2749_, 0);
v_nextMacroScope_2752_ = lean_ctor_get(v___x_2749_, 1);
v_ngen_2753_ = lean_ctor_get(v___x_2749_, 2);
v_auxDeclNGen_2754_ = lean_ctor_get(v___x_2749_, 3);
v_cache_2755_ = lean_ctor_get(v___x_2749_, 5);
v_messages_2756_ = lean_ctor_get(v___x_2749_, 6);
v_infoState_2757_ = lean_ctor_get(v___x_2749_, 7);
v_snapshotTasks_2758_ = lean_ctor_get(v___x_2749_, 8);
v_isSharedCheck_2914_ = !lean_is_exclusive(v___x_2749_);
if (v_isSharedCheck_2914_ == 0)
{
v___x_2760_ = v___x_2749_;
v_isShared_2761_ = v_isSharedCheck_2914_;
goto v_resetjp_2759_;
}
else
{
lean_inc(v_snapshotTasks_2758_);
lean_inc(v_infoState_2757_);
lean_inc(v_messages_2756_);
lean_inc(v_cache_2755_);
lean_inc(v_traceState_2750_);
lean_inc(v_auxDeclNGen_2754_);
lean_inc(v_ngen_2753_);
lean_inc(v_nextMacroScope_2752_);
lean_inc(v_env_2751_);
lean_dec(v___x_2749_);
v___x_2760_ = lean_box(0);
v_isShared_2761_ = v_isSharedCheck_2914_;
goto v_resetjp_2759_;
}
v___jp_2715_:
{
lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v_traceState_2719_; lean_object* v_env_2720_; lean_object* v_nextMacroScope_2721_; lean_object* v_ngen_2722_; lean_object* v_auxDeclNGen_2723_; lean_object* v_cache_2724_; lean_object* v_messages_2725_; lean_object* v_infoState_2726_; lean_object* v_snapshotTasks_2727_; lean_object* v___x_2729_; uint8_t v_isShared_2730_; uint8_t v_isSharedCheck_2748_; 
v___x_2717_ = lean_st_ref_get(v___y_2713_);
v___x_2718_ = lean_st_ref_take(v___y_2713_);
v_traceState_2719_ = lean_ctor_get(v___x_2718_, 4);
v_env_2720_ = lean_ctor_get(v___x_2718_, 0);
v_nextMacroScope_2721_ = lean_ctor_get(v___x_2718_, 1);
v_ngen_2722_ = lean_ctor_get(v___x_2718_, 2);
v_auxDeclNGen_2723_ = lean_ctor_get(v___x_2718_, 3);
v_cache_2724_ = lean_ctor_get(v___x_2718_, 5);
v_messages_2725_ = lean_ctor_get(v___x_2718_, 6);
v_infoState_2726_ = lean_ctor_get(v___x_2718_, 7);
v_snapshotTasks_2727_ = lean_ctor_get(v___x_2718_, 8);
v_isSharedCheck_2748_ = !lean_is_exclusive(v___x_2718_);
if (v_isSharedCheck_2748_ == 0)
{
v___x_2729_ = v___x_2718_;
v_isShared_2730_ = v_isSharedCheck_2748_;
goto v_resetjp_2728_;
}
else
{
lean_inc(v_snapshotTasks_2727_);
lean_inc(v_infoState_2726_);
lean_inc(v_messages_2725_);
lean_inc(v_cache_2724_);
lean_inc(v_traceState_2719_);
lean_inc(v_auxDeclNGen_2723_);
lean_inc(v_ngen_2722_);
lean_inc(v_nextMacroScope_2721_);
lean_inc(v_env_2720_);
lean_dec(v___x_2718_);
v___x_2729_ = lean_box(0);
v_isShared_2730_ = v_isSharedCheck_2748_;
goto v_resetjp_2728_;
}
v_resetjp_2728_:
{
uint64_t v_tid_2731_; lean_object* v___x_2733_; uint8_t v_isShared_2734_; uint8_t v_isSharedCheck_2746_; 
v_tid_2731_ = lean_ctor_get_uint64(v_traceState_2719_, sizeof(void*)*1);
v_isSharedCheck_2746_ = !lean_is_exclusive(v_traceState_2719_);
if (v_isSharedCheck_2746_ == 0)
{
lean_object* v_unused_2747_; 
v_unused_2747_ = lean_ctor_get(v_traceState_2719_, 0);
lean_dec(v_unused_2747_);
v___x_2733_ = v_traceState_2719_;
v_isShared_2734_ = v_isSharedCheck_2746_;
goto v_resetjp_2732_;
}
else
{
lean_dec(v_traceState_2719_);
v___x_2733_ = lean_box(0);
v_isShared_2734_ = v_isSharedCheck_2746_;
goto v_resetjp_2732_;
}
v_resetjp_2732_:
{
lean_object* v___x_2736_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 0, v_traces_2702_);
v___x_2736_ = v___x_2733_;
goto v_reusejp_2735_;
}
else
{
lean_object* v_reuseFailAlloc_2745_; 
v_reuseFailAlloc_2745_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2745_, 0, v_traces_2702_);
lean_ctor_set_uint64(v_reuseFailAlloc_2745_, sizeof(void*)*1, v_tid_2731_);
v___x_2736_ = v_reuseFailAlloc_2745_;
goto v_reusejp_2735_;
}
v_reusejp_2735_:
{
lean_object* v___x_2738_; 
if (v_isShared_2730_ == 0)
{
lean_ctor_set(v___x_2729_, 4, v___x_2736_);
v___x_2738_ = v___x_2729_;
goto v_reusejp_2737_;
}
else
{
lean_object* v_reuseFailAlloc_2744_; 
v_reuseFailAlloc_2744_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2744_, 0, v_env_2720_);
lean_ctor_set(v_reuseFailAlloc_2744_, 1, v_nextMacroScope_2721_);
lean_ctor_set(v_reuseFailAlloc_2744_, 2, v_ngen_2722_);
lean_ctor_set(v_reuseFailAlloc_2744_, 3, v_auxDeclNGen_2723_);
lean_ctor_set(v_reuseFailAlloc_2744_, 4, v___x_2736_);
lean_ctor_set(v_reuseFailAlloc_2744_, 5, v_cache_2724_);
lean_ctor_set(v_reuseFailAlloc_2744_, 6, v_messages_2725_);
lean_ctor_set(v_reuseFailAlloc_2744_, 7, v_infoState_2726_);
lean_ctor_set(v_reuseFailAlloc_2744_, 8, v_snapshotTasks_2727_);
v___x_2738_ = v_reuseFailAlloc_2744_;
goto v_reusejp_2737_;
}
v_reusejp_2737_:
{
lean_object* v___x_2739_; lean_object* v_traceState_2740_; lean_object* v_traces_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; 
v___x_2739_ = lean_st_ref_set(v___y_2713_, v___x_2738_);
v_traceState_2740_ = lean_ctor_get(v___x_2717_, 4);
lean_inc_ref(v_traceState_2740_);
lean_dec(v___x_2717_);
v_traces_2741_ = lean_ctor_get(v_traceState_2740_, 0);
lean_inc_ref(v_traces_2741_);
lean_dec_ref(v_traceState_2740_);
v___x_2742_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2742_, 0, v_a_2716_);
lean_ctor_set(v___x_2742_, 1, v_traces_2741_);
v___x_2743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2743_, 0, v___x_2742_);
return v___x_2743_;
}
}
}
}
}
v_resetjp_2759_:
{
uint64_t v_tid_2762_; lean_object* v___x_2764_; uint8_t v_isShared_2765_; uint8_t v_isSharedCheck_2912_; 
v_tid_2762_ = lean_ctor_get_uint64(v_traceState_2750_, sizeof(void*)*1);
v_isSharedCheck_2912_ = !lean_is_exclusive(v_traceState_2750_);
if (v_isSharedCheck_2912_ == 0)
{
lean_object* v_unused_2913_; 
v_unused_2913_ = lean_ctor_get(v_traceState_2750_, 0);
lean_dec(v_unused_2913_);
v___x_2764_ = v_traceState_2750_;
v_isShared_2765_ = v_isSharedCheck_2912_;
goto v_resetjp_2763_;
}
else
{
lean_dec(v_traceState_2750_);
v___x_2764_ = lean_box(0);
v_isShared_2765_ = v_isSharedCheck_2912_;
goto v_resetjp_2763_;
}
v_resetjp_2763_:
{
lean_object* v___x_2766_; lean_object* v___x_2767_; lean_object* v___x_2768_; size_t v___x_2769_; lean_object* v___x_2770_; lean_object* v___x_2772_; 
v___x_2766_ = lean_unsigned_to_nat(32u);
v___x_2767_ = lean_mk_empty_array_with_capacity(v___x_2766_);
v___x_2768_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__0);
v___x_2769_ = ((size_t)5ULL);
lean_inc(v___x_2701_);
v___x_2770_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2770_, 0, v___x_2768_);
lean_ctor_set(v___x_2770_, 1, v___x_2767_);
lean_ctor_set(v___x_2770_, 2, v___x_2701_);
lean_ctor_set(v___x_2770_, 3, v___x_2701_);
lean_ctor_set_usize(v___x_2770_, 4, v___x_2769_);
if (v_isShared_2765_ == 0)
{
lean_ctor_set(v___x_2764_, 0, v___x_2770_);
v___x_2772_ = v___x_2764_;
goto v_reusejp_2771_;
}
else
{
lean_object* v_reuseFailAlloc_2911_; 
v_reuseFailAlloc_2911_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2911_, 0, v___x_2770_);
lean_ctor_set_uint64(v_reuseFailAlloc_2911_, sizeof(void*)*1, v_tid_2762_);
v___x_2772_ = v_reuseFailAlloc_2911_;
goto v_reusejp_2771_;
}
v_reusejp_2771_:
{
lean_object* v___x_2774_; 
if (v_isShared_2761_ == 0)
{
lean_ctor_set(v___x_2760_, 4, v___x_2772_);
v___x_2774_ = v___x_2760_;
goto v_reusejp_2773_;
}
else
{
lean_object* v_reuseFailAlloc_2910_; 
v_reuseFailAlloc_2910_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2910_, 0, v_env_2751_);
lean_ctor_set(v_reuseFailAlloc_2910_, 1, v_nextMacroScope_2752_);
lean_ctor_set(v_reuseFailAlloc_2910_, 2, v_ngen_2753_);
lean_ctor_set(v_reuseFailAlloc_2910_, 3, v_auxDeclNGen_2754_);
lean_ctor_set(v_reuseFailAlloc_2910_, 4, v___x_2772_);
lean_ctor_set(v_reuseFailAlloc_2910_, 5, v_cache_2755_);
lean_ctor_set(v_reuseFailAlloc_2910_, 6, v_messages_2756_);
lean_ctor_set(v_reuseFailAlloc_2910_, 7, v_infoState_2757_);
lean_ctor_set(v_reuseFailAlloc_2910_, 8, v_snapshotTasks_2758_);
v___x_2774_ = v_reuseFailAlloc_2910_;
goto v_reusejp_2773_;
}
v_reusejp_2773_:
{
lean_object* v___x_2775_; lean_object* v___x_2776_; 
v___x_2775_ = lean_st_ref_set(v___y_2713_, v___x_2774_);
v___x_2776_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_2707_, v___y_2709_, v___y_2711_, v___y_2713_);
if (lean_obj_tag(v___x_2776_) == 0)
{
lean_object* v_a_2777_; lean_object* v___x_2779_; uint8_t v_isShared_2780_; uint8_t v_isSharedCheck_2901_; 
v_a_2777_ = lean_ctor_get(v___x_2776_, 0);
v_isSharedCheck_2901_ = !lean_is_exclusive(v___x_2776_);
if (v_isSharedCheck_2901_ == 0)
{
v___x_2779_ = v___x_2776_;
v_isShared_2780_ = v_isSharedCheck_2901_;
goto v_resetjp_2778_;
}
else
{
lean_inc(v_a_2777_);
lean_dec(v___x_2776_);
v___x_2779_ = lean_box(0);
v_isShared_2780_ = v_isSharedCheck_2901_;
goto v_resetjp_2778_;
}
v_resetjp_2778_:
{
lean_object* v___y_2782_; uint8_t v___y_2783_; lean_object* v___x_2833_; lean_object* v_fileName_2834_; lean_object* v_fileMap_2835_; lean_object* v_options_2836_; lean_object* v_currRecDepth_2837_; lean_object* v_ref_2838_; lean_object* v_currNamespace_2839_; lean_object* v_openDecls_2840_; lean_object* v_initHeartbeats_2841_; lean_object* v_maxHeartbeats_2842_; lean_object* v_quotContext_2843_; lean_object* v_currMacroScope_2844_; lean_object* v_cancelTk_x3f_2845_; uint8_t v_suppressElabErrors_2846_; lean_object* v_inheritedTraceOptions_2847_; lean_object* v_env_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; uint8_t v___x_2854_; lean_object* v_fileName_2856_; lean_object* v_fileMap_2857_; lean_object* v_currRecDepth_2858_; lean_object* v_ref_2859_; lean_object* v_currNamespace_2860_; lean_object* v_openDecls_2861_; lean_object* v_initHeartbeats_2862_; lean_object* v_maxHeartbeats_2863_; lean_object* v_quotContext_2864_; lean_object* v_currMacroScope_2865_; lean_object* v_cancelTk_x3f_2866_; uint8_t v_suppressElabErrors_2867_; lean_object* v_inheritedTraceOptions_2868_; lean_object* v___y_2869_; uint8_t v___y_2879_; uint8_t v___x_2900_; 
v___x_2833_ = lean_st_ref_get(v___y_2713_);
v_fileName_2834_ = lean_ctor_get(v___y_2712_, 0);
v_fileMap_2835_ = lean_ctor_get(v___y_2712_, 1);
v_options_2836_ = lean_ctor_get(v___y_2712_, 2);
v_currRecDepth_2837_ = lean_ctor_get(v___y_2712_, 3);
v_ref_2838_ = lean_ctor_get(v___y_2712_, 5);
v_currNamespace_2839_ = lean_ctor_get(v___y_2712_, 6);
v_openDecls_2840_ = lean_ctor_get(v___y_2712_, 7);
v_initHeartbeats_2841_ = lean_ctor_get(v___y_2712_, 8);
v_maxHeartbeats_2842_ = lean_ctor_get(v___y_2712_, 9);
v_quotContext_2843_ = lean_ctor_get(v___y_2712_, 10);
v_currMacroScope_2844_ = lean_ctor_get(v___y_2712_, 11);
v_cancelTk_x3f_2845_ = lean_ctor_get(v___y_2712_, 12);
v_suppressElabErrors_2846_ = lean_ctor_get_uint8(v___y_2712_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2847_ = lean_ctor_get(v___y_2712_, 13);
v_env_2848_ = lean_ctor_get(v___x_2833_, 0);
lean_inc_ref(v_env_2848_);
lean_dec(v___x_2833_);
v___x_2849_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3));
lean_inc_ref(v_options_2836_);
v___x_2850_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_options_2836_, v___x_2849_, v_strict_2705_);
v___x_2851_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4));
v___x_2852_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v___x_2850_, v___x_2851_, v___x_2703_);
v___x_2853_ = l_Lean_diagnostics;
v___x_2854_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(v___x_2852_, v___x_2853_);
v___x_2900_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_2848_);
lean_dec_ref(v_env_2848_);
if (v___x_2900_ == 0)
{
if (v___x_2854_ == 0)
{
v___y_2879_ = v___x_2703_;
goto v___jp_2878_;
}
else
{
v___y_2879_ = v___x_2900_;
goto v___jp_2878_;
}
}
else
{
v___y_2879_ = v___x_2854_;
goto v___jp_2878_;
}
v___jp_2781_:
{
if (v___y_2783_ == 0)
{
lean_object* v___x_2784_; 
lean_del_object(v___x_2779_);
v___x_2784_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_2777_, v___y_2783_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_);
if (lean_obj_tag(v___x_2784_) == 0)
{
lean_object* v___x_2786_; uint8_t v_isShared_2787_; uint8_t v_isSharedCheck_2820_; 
v_isSharedCheck_2820_ = !lean_is_exclusive(v___x_2784_);
if (v_isSharedCheck_2820_ == 0)
{
lean_object* v_unused_2821_; 
v_unused_2821_ = lean_ctor_get(v___x_2784_, 0);
lean_dec(v_unused_2821_);
v___x_2786_ = v___x_2784_;
v_isShared_2787_ = v_isSharedCheck_2820_;
goto v_resetjp_2785_;
}
else
{
lean_dec(v___x_2784_);
v___x_2786_ = lean_box(0);
v_isShared_2787_ = v_isSharedCheck_2820_;
goto v_resetjp_2785_;
}
v_resetjp_2785_:
{
if (lean_obj_tag(v___y_2782_) == 1)
{
lean_object* v___x_2788_; lean_object* v_traceState_2789_; lean_object* v_env_2790_; lean_object* v_nextMacroScope_2791_; lean_object* v_ngen_2792_; lean_object* v_auxDeclNGen_2793_; lean_object* v_cache_2794_; lean_object* v_messages_2795_; lean_object* v_infoState_2796_; lean_object* v_snapshotTasks_2797_; lean_object* v___x_2799_; uint8_t v_isShared_2800_; uint8_t v_isSharedCheck_2817_; 
v___x_2788_ = lean_st_ref_take(v___y_2713_);
v_traceState_2789_ = lean_ctor_get(v___x_2788_, 4);
v_env_2790_ = lean_ctor_get(v___x_2788_, 0);
v_nextMacroScope_2791_ = lean_ctor_get(v___x_2788_, 1);
v_ngen_2792_ = lean_ctor_get(v___x_2788_, 2);
v_auxDeclNGen_2793_ = lean_ctor_get(v___x_2788_, 3);
v_cache_2794_ = lean_ctor_get(v___x_2788_, 5);
v_messages_2795_ = lean_ctor_get(v___x_2788_, 6);
v_infoState_2796_ = lean_ctor_get(v___x_2788_, 7);
v_snapshotTasks_2797_ = lean_ctor_get(v___x_2788_, 8);
v_isSharedCheck_2817_ = !lean_is_exclusive(v___x_2788_);
if (v_isSharedCheck_2817_ == 0)
{
v___x_2799_ = v___x_2788_;
v_isShared_2800_ = v_isSharedCheck_2817_;
goto v_resetjp_2798_;
}
else
{
lean_inc(v_snapshotTasks_2797_);
lean_inc(v_infoState_2796_);
lean_inc(v_messages_2795_);
lean_inc(v_cache_2794_);
lean_inc(v_traceState_2789_);
lean_inc(v_auxDeclNGen_2793_);
lean_inc(v_ngen_2792_);
lean_inc(v_nextMacroScope_2791_);
lean_inc(v_env_2790_);
lean_dec(v___x_2788_);
v___x_2799_ = lean_box(0);
v_isShared_2800_ = v_isSharedCheck_2817_;
goto v_resetjp_2798_;
}
v_resetjp_2798_:
{
uint64_t v_tid_2801_; lean_object* v___x_2803_; uint8_t v_isShared_2804_; uint8_t v_isSharedCheck_2815_; 
v_tid_2801_ = lean_ctor_get_uint64(v_traceState_2789_, sizeof(void*)*1);
v_isSharedCheck_2815_ = !lean_is_exclusive(v_traceState_2789_);
if (v_isSharedCheck_2815_ == 0)
{
lean_object* v_unused_2816_; 
v_unused_2816_ = lean_ctor_get(v_traceState_2789_, 0);
lean_dec(v_unused_2816_);
v___x_2803_ = v_traceState_2789_;
v_isShared_2804_ = v_isSharedCheck_2815_;
goto v_resetjp_2802_;
}
else
{
lean_dec(v_traceState_2789_);
v___x_2803_ = lean_box(0);
v_isShared_2804_ = v_isSharedCheck_2815_;
goto v_resetjp_2802_;
}
v_resetjp_2802_:
{
lean_object* v___x_2806_; 
if (v_isShared_2804_ == 0)
{
lean_ctor_set(v___x_2803_, 0, v_traces_2702_);
v___x_2806_ = v___x_2803_;
goto v_reusejp_2805_;
}
else
{
lean_object* v_reuseFailAlloc_2814_; 
v_reuseFailAlloc_2814_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2814_, 0, v_traces_2702_);
lean_ctor_set_uint64(v_reuseFailAlloc_2814_, sizeof(void*)*1, v_tid_2801_);
v___x_2806_ = v_reuseFailAlloc_2814_;
goto v_reusejp_2805_;
}
v_reusejp_2805_:
{
lean_object* v___x_2808_; 
if (v_isShared_2800_ == 0)
{
lean_ctor_set(v___x_2799_, 4, v___x_2806_);
v___x_2808_ = v___x_2799_;
goto v_reusejp_2807_;
}
else
{
lean_object* v_reuseFailAlloc_2813_; 
v_reuseFailAlloc_2813_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2813_, 0, v_env_2790_);
lean_ctor_set(v_reuseFailAlloc_2813_, 1, v_nextMacroScope_2791_);
lean_ctor_set(v_reuseFailAlloc_2813_, 2, v_ngen_2792_);
lean_ctor_set(v_reuseFailAlloc_2813_, 3, v_auxDeclNGen_2793_);
lean_ctor_set(v_reuseFailAlloc_2813_, 4, v___x_2806_);
lean_ctor_set(v_reuseFailAlloc_2813_, 5, v_cache_2794_);
lean_ctor_set(v_reuseFailAlloc_2813_, 6, v_messages_2795_);
lean_ctor_set(v_reuseFailAlloc_2813_, 7, v_infoState_2796_);
lean_ctor_set(v_reuseFailAlloc_2813_, 8, v_snapshotTasks_2797_);
v___x_2808_ = v_reuseFailAlloc_2813_;
goto v_reusejp_2807_;
}
v_reusejp_2807_:
{
lean_object* v___x_2809_; lean_object* v___x_2811_; 
v___x_2809_ = lean_st_ref_set(v___y_2713_, v___x_2808_);
if (v_isShared_2787_ == 0)
{
lean_ctor_set_tag(v___x_2786_, 1);
lean_ctor_set(v___x_2786_, 0, v___y_2782_);
v___x_2811_ = v___x_2786_;
goto v_reusejp_2810_;
}
else
{
lean_object* v_reuseFailAlloc_2812_; 
v_reuseFailAlloc_2812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2812_, 0, v___y_2782_);
v___x_2811_ = v_reuseFailAlloc_2812_;
goto v_reusejp_2810_;
}
v_reusejp_2810_:
{
return v___x_2811_;
}
}
}
}
}
}
else
{
lean_object* v___x_2818_; lean_object* v___x_2819_; 
lean_del_object(v___x_2786_);
v___x_2818_ = l_Lean_Exception_toMessageData(v___y_2782_);
v___x_2819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2819_, 0, v___x_2818_);
v_a_2716_ = v___x_2819_;
goto v___jp_2715_;
}
}
}
else
{
lean_object* v_a_2822_; lean_object* v___x_2824_; uint8_t v_isShared_2825_; uint8_t v_isSharedCheck_2829_; 
lean_dec_ref(v___y_2782_);
lean_dec_ref(v_traces_2702_);
v_a_2822_ = lean_ctor_get(v___x_2784_, 0);
v_isSharedCheck_2829_ = !lean_is_exclusive(v___x_2784_);
if (v_isSharedCheck_2829_ == 0)
{
v___x_2824_ = v___x_2784_;
v_isShared_2825_ = v_isSharedCheck_2829_;
goto v_resetjp_2823_;
}
else
{
lean_inc(v_a_2822_);
lean_dec(v___x_2784_);
v___x_2824_ = lean_box(0);
v_isShared_2825_ = v_isSharedCheck_2829_;
goto v_resetjp_2823_;
}
v_resetjp_2823_:
{
lean_object* v___x_2827_; 
if (v_isShared_2825_ == 0)
{
v___x_2827_ = v___x_2824_;
goto v_reusejp_2826_;
}
else
{
lean_object* v_reuseFailAlloc_2828_; 
v_reuseFailAlloc_2828_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2828_, 0, v_a_2822_);
v___x_2827_ = v_reuseFailAlloc_2828_;
goto v_reusejp_2826_;
}
v_reusejp_2826_:
{
return v___x_2827_;
}
}
}
}
else
{
lean_object* v___x_2831_; 
lean_dec(v_a_2777_);
lean_dec_ref(v_traces_2702_);
if (v_isShared_2780_ == 0)
{
lean_ctor_set_tag(v___x_2779_, 1);
lean_ctor_set(v___x_2779_, 0, v___y_2782_);
v___x_2831_ = v___x_2779_;
goto v_reusejp_2830_;
}
else
{
lean_object* v_reuseFailAlloc_2832_; 
v_reuseFailAlloc_2832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2832_, 0, v___y_2782_);
v___x_2831_ = v_reuseFailAlloc_2832_;
goto v_reusejp_2830_;
}
v_reusejp_2830_:
{
return v___x_2831_;
}
}
}
v___jp_2855_:
{
lean_object* v___x_2870_; lean_object* v___x_2871_; lean_object* v___x_2872_; lean_object* v___x_2873_; 
v___x_2870_ = l_Lean_maxRecDepth;
v___x_2871_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2(v___x_2852_, v___x_2870_);
lean_inc_ref(v_inheritedTraceOptions_2868_);
lean_inc(v_cancelTk_x3f_2866_);
lean_inc(v_currMacroScope_2865_);
lean_inc(v_quotContext_2864_);
lean_inc(v_maxHeartbeats_2863_);
lean_inc(v_initHeartbeats_2862_);
lean_inc(v_openDecls_2861_);
lean_inc(v_currNamespace_2860_);
lean_inc(v_ref_2859_);
lean_inc(v_currRecDepth_2858_);
lean_inc_ref(v_fileMap_2857_);
lean_inc_ref(v_fileName_2856_);
v___x_2872_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2872_, 0, v_fileName_2856_);
lean_ctor_set(v___x_2872_, 1, v_fileMap_2857_);
lean_ctor_set(v___x_2872_, 2, v___x_2852_);
lean_ctor_set(v___x_2872_, 3, v_currRecDepth_2858_);
lean_ctor_set(v___x_2872_, 4, v___x_2871_);
lean_ctor_set(v___x_2872_, 5, v_ref_2859_);
lean_ctor_set(v___x_2872_, 6, v_currNamespace_2860_);
lean_ctor_set(v___x_2872_, 7, v_openDecls_2861_);
lean_ctor_set(v___x_2872_, 8, v_initHeartbeats_2862_);
lean_ctor_set(v___x_2872_, 9, v_maxHeartbeats_2863_);
lean_ctor_set(v___x_2872_, 10, v_quotContext_2864_);
lean_ctor_set(v___x_2872_, 11, v_currMacroScope_2865_);
lean_ctor_set(v___x_2872_, 12, v_cancelTk_x3f_2866_);
lean_ctor_set(v___x_2872_, 13, v_inheritedTraceOptions_2868_);
lean_ctor_set_uint8(v___x_2872_, sizeof(void*)*14, v___x_2854_);
lean_ctor_set_uint8(v___x_2872_, sizeof(void*)*14 + 1, v_suppressElabErrors_2867_);
v___x_2873_ = l_Lean_Elab_Tactic_evalTactic(v___x_2704_, v___y_2706_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___x_2872_, v___y_2869_);
lean_dec_ref_known(v___x_2872_, 14);
if (lean_obj_tag(v___x_2873_) == 0)
{
lean_object* v___x_2874_; 
lean_dec_ref_known(v___x_2873_, 1);
lean_del_object(v___x_2779_);
lean_dec(v_a_2777_);
v___x_2874_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__5));
v_a_2716_ = v___x_2874_;
goto v___jp_2715_;
}
else
{
lean_object* v_a_2875_; uint8_t v___x_2876_; 
v_a_2875_ = lean_ctor_get(v___x_2873_, 0);
lean_inc(v_a_2875_);
lean_dec_ref_known(v___x_2873_, 1);
v___x_2876_ = l_Lean_Exception_isInterrupt(v_a_2875_);
if (v___x_2876_ == 0)
{
uint8_t v___x_2877_; 
lean_inc(v_a_2875_);
v___x_2877_ = l_Lean_Exception_isRuntime(v_a_2875_);
v___y_2782_ = v_a_2875_;
v___y_2783_ = v___x_2877_;
goto v___jp_2781_;
}
else
{
v___y_2782_ = v_a_2875_;
v___y_2783_ = v___x_2876_;
goto v___jp_2781_;
}
}
}
v___jp_2878_:
{
if (v___y_2879_ == 0)
{
lean_object* v___x_2880_; lean_object* v_env_2881_; lean_object* v_nextMacroScope_2882_; lean_object* v_ngen_2883_; lean_object* v_auxDeclNGen_2884_; lean_object* v_traceState_2885_; lean_object* v_messages_2886_; lean_object* v_infoState_2887_; lean_object* v_snapshotTasks_2888_; lean_object* v___x_2890_; uint8_t v_isShared_2891_; uint8_t v_isSharedCheck_2898_; 
v___x_2880_ = lean_st_ref_take(v___y_2713_);
v_env_2881_ = lean_ctor_get(v___x_2880_, 0);
v_nextMacroScope_2882_ = lean_ctor_get(v___x_2880_, 1);
v_ngen_2883_ = lean_ctor_get(v___x_2880_, 2);
v_auxDeclNGen_2884_ = lean_ctor_get(v___x_2880_, 3);
v_traceState_2885_ = lean_ctor_get(v___x_2880_, 4);
v_messages_2886_ = lean_ctor_get(v___x_2880_, 6);
v_infoState_2887_ = lean_ctor_get(v___x_2880_, 7);
v_snapshotTasks_2888_ = lean_ctor_get(v___x_2880_, 8);
v_isSharedCheck_2898_ = !lean_is_exclusive(v___x_2880_);
if (v_isSharedCheck_2898_ == 0)
{
lean_object* v_unused_2899_; 
v_unused_2899_ = lean_ctor_get(v___x_2880_, 5);
lean_dec(v_unused_2899_);
v___x_2890_ = v___x_2880_;
v_isShared_2891_ = v_isSharedCheck_2898_;
goto v_resetjp_2889_;
}
else
{
lean_inc(v_snapshotTasks_2888_);
lean_inc(v_infoState_2887_);
lean_inc(v_messages_2886_);
lean_inc(v_traceState_2885_);
lean_inc(v_auxDeclNGen_2884_);
lean_inc(v_ngen_2883_);
lean_inc(v_nextMacroScope_2882_);
lean_inc(v_env_2881_);
lean_dec(v___x_2880_);
v___x_2890_ = lean_box(0);
v_isShared_2891_ = v_isSharedCheck_2898_;
goto v_resetjp_2889_;
}
v_resetjp_2889_:
{
lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2895_; 
v___x_2892_ = l_Lean_Kernel_enableDiag(v_env_2881_, v___x_2854_);
v___x_2893_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8);
if (v_isShared_2891_ == 0)
{
lean_ctor_set(v___x_2890_, 5, v___x_2893_);
lean_ctor_set(v___x_2890_, 0, v___x_2892_);
v___x_2895_ = v___x_2890_;
goto v_reusejp_2894_;
}
else
{
lean_object* v_reuseFailAlloc_2897_; 
v_reuseFailAlloc_2897_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2897_, 0, v___x_2892_);
lean_ctor_set(v_reuseFailAlloc_2897_, 1, v_nextMacroScope_2882_);
lean_ctor_set(v_reuseFailAlloc_2897_, 2, v_ngen_2883_);
lean_ctor_set(v_reuseFailAlloc_2897_, 3, v_auxDeclNGen_2884_);
lean_ctor_set(v_reuseFailAlloc_2897_, 4, v_traceState_2885_);
lean_ctor_set(v_reuseFailAlloc_2897_, 5, v___x_2893_);
lean_ctor_set(v_reuseFailAlloc_2897_, 6, v_messages_2886_);
lean_ctor_set(v_reuseFailAlloc_2897_, 7, v_infoState_2887_);
lean_ctor_set(v_reuseFailAlloc_2897_, 8, v_snapshotTasks_2888_);
v___x_2895_ = v_reuseFailAlloc_2897_;
goto v_reusejp_2894_;
}
v_reusejp_2894_:
{
lean_object* v___x_2896_; 
v___x_2896_ = lean_st_ref_set(v___y_2713_, v___x_2895_);
v_fileName_2856_ = v_fileName_2834_;
v_fileMap_2857_ = v_fileMap_2835_;
v_currRecDepth_2858_ = v_currRecDepth_2837_;
v_ref_2859_ = v_ref_2838_;
v_currNamespace_2860_ = v_currNamespace_2839_;
v_openDecls_2861_ = v_openDecls_2840_;
v_initHeartbeats_2862_ = v_initHeartbeats_2841_;
v_maxHeartbeats_2863_ = v_maxHeartbeats_2842_;
v_quotContext_2864_ = v_quotContext_2843_;
v_currMacroScope_2865_ = v_currMacroScope_2844_;
v_cancelTk_x3f_2866_ = v_cancelTk_x3f_2845_;
v_suppressElabErrors_2867_ = v_suppressElabErrors_2846_;
v_inheritedTraceOptions_2868_ = v_inheritedTraceOptions_2847_;
v___y_2869_ = v___y_2713_;
goto v___jp_2855_;
}
}
}
else
{
v_fileName_2856_ = v_fileName_2834_;
v_fileMap_2857_ = v_fileMap_2835_;
v_currRecDepth_2858_ = v_currRecDepth_2837_;
v_ref_2859_ = v_ref_2838_;
v_currNamespace_2860_ = v_currNamespace_2839_;
v_openDecls_2861_ = v_openDecls_2840_;
v_initHeartbeats_2862_ = v_initHeartbeats_2841_;
v_maxHeartbeats_2863_ = v_maxHeartbeats_2842_;
v_quotContext_2864_ = v_quotContext_2843_;
v_currMacroScope_2865_ = v_currMacroScope_2844_;
v_cancelTk_x3f_2866_ = v_cancelTk_x3f_2845_;
v_suppressElabErrors_2867_ = v_suppressElabErrors_2846_;
v_inheritedTraceOptions_2868_ = v_inheritedTraceOptions_2847_;
v___y_2869_ = v___y_2713_;
goto v___jp_2855_;
}
}
}
}
else
{
lean_object* v_a_2902_; lean_object* v___x_2904_; uint8_t v_isShared_2905_; uint8_t v_isSharedCheck_2909_; 
lean_dec(v___x_2704_);
lean_dec_ref(v_traces_2702_);
v_a_2902_ = lean_ctor_get(v___x_2776_, 0);
v_isSharedCheck_2909_ = !lean_is_exclusive(v___x_2776_);
if (v_isSharedCheck_2909_ == 0)
{
v___x_2904_ = v___x_2776_;
v_isShared_2905_ = v_isSharedCheck_2909_;
goto v_resetjp_2903_;
}
else
{
lean_inc(v_a_2902_);
lean_dec(v___x_2776_);
v___x_2904_ = lean_box(0);
v_isShared_2905_ = v_isSharedCheck_2909_;
goto v_resetjp_2903_;
}
v_resetjp_2903_:
{
lean_object* v___x_2907_; 
if (v_isShared_2905_ == 0)
{
v___x_2907_ = v___x_2904_;
goto v_reusejp_2906_;
}
else
{
lean_object* v_reuseFailAlloc_2908_; 
v_reuseFailAlloc_2908_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2908_, 0, v_a_2902_);
v___x_2907_ = v_reuseFailAlloc_2908_;
goto v_reusejp_2906_;
}
v_reusejp_2906_:
{
return v___x_2907_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___boxed(lean_object* v___x_2915_, lean_object* v_traces_2916_, lean_object* v___x_2917_, lean_object* v___x_2918_, lean_object* v_strict_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_, lean_object* v___y_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_, lean_object* v___y_2927_, lean_object* v___y_2928_){
_start:
{
uint8_t v___x_24835__boxed_2929_; uint8_t v_strict_boxed_2930_; lean_object* v_res_2931_; 
v___x_24835__boxed_2929_ = lean_unbox(v___x_2917_);
v_strict_boxed_2930_ = lean_unbox(v_strict_2919_);
v_res_2931_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0(v___x_2915_, v_traces_2916_, v___x_24835__boxed_2929_, v___x_2918_, v_strict_boxed_2930_, v___y_2920_, v___y_2921_, v___y_2922_, v___y_2923_, v___y_2924_, v___y_2925_, v___y_2926_, v___y_2927_);
lean_dec(v___y_2927_);
lean_dec_ref(v___y_2926_);
lean_dec(v___y_2925_);
lean_dec_ref(v___y_2924_);
lean_dec(v___y_2923_);
lean_dec_ref(v___y_2922_);
lean_dec(v___y_2921_);
lean_dec_ref(v___y_2920_);
return v_res_2931_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0(uint8_t v___y_2938_, uint8_t v_suppressElabErrors_2939_, lean_object* v_x_2940_){
_start:
{
if (lean_obj_tag(v_x_2940_) == 1)
{
lean_object* v_pre_2941_; 
v_pre_2941_ = lean_ctor_get(v_x_2940_, 0);
switch(lean_obj_tag(v_pre_2941_))
{
case 1:
{
lean_object* v_pre_2942_; 
v_pre_2942_ = lean_ctor_get(v_pre_2941_, 0);
switch(lean_obj_tag(v_pre_2942_))
{
case 0:
{
lean_object* v_str_2943_; lean_object* v_str_2944_; lean_object* v___x_2945_; uint8_t v___x_2946_; 
v_str_2943_ = lean_ctor_get(v_x_2940_, 1);
v_str_2944_ = lean_ctor_get(v_pre_2941_, 1);
v___x_2945_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__0));
v___x_2946_ = lean_string_dec_eq(v_str_2944_, v___x_2945_);
if (v___x_2946_ == 0)
{
lean_object* v___x_2947_; uint8_t v___x_2948_; 
v___x_2947_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1___closed__2));
v___x_2948_ = lean_string_dec_eq(v_str_2944_, v___x_2947_);
if (v___x_2948_ == 0)
{
return v___y_2938_;
}
else
{
lean_object* v___x_2949_; uint8_t v___x_2950_; 
v___x_2949_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__1));
v___x_2950_ = lean_string_dec_eq(v_str_2943_, v___x_2949_);
if (v___x_2950_ == 0)
{
return v___y_2938_;
}
else
{
return v_suppressElabErrors_2939_;
}
}
}
else
{
lean_object* v___x_2951_; uint8_t v___x_2952_; 
v___x_2951_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__2));
v___x_2952_ = lean_string_dec_eq(v_str_2943_, v___x_2951_);
if (v___x_2952_ == 0)
{
return v___y_2938_;
}
else
{
return v_suppressElabErrors_2939_;
}
}
}
case 1:
{
lean_object* v_pre_2953_; 
v_pre_2953_ = lean_ctor_get(v_pre_2942_, 0);
if (lean_obj_tag(v_pre_2953_) == 0)
{
lean_object* v_str_2954_; lean_object* v_str_2955_; lean_object* v_str_2956_; lean_object* v___x_2957_; uint8_t v___x_2958_; 
v_str_2954_ = lean_ctor_get(v_x_2940_, 1);
v_str_2955_ = lean_ctor_get(v_pre_2941_, 1);
v_str_2956_ = lean_ctor_get(v_pre_2942_, 1);
v___x_2957_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__3));
v___x_2958_ = lean_string_dec_eq(v_str_2956_, v___x_2957_);
if (v___x_2958_ == 0)
{
return v___y_2938_;
}
else
{
lean_object* v___x_2959_; uint8_t v___x_2960_; 
v___x_2959_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__4));
v___x_2960_ = lean_string_dec_eq(v_str_2955_, v___x_2959_);
if (v___x_2960_ == 0)
{
return v___y_2938_;
}
else
{
lean_object* v___x_2961_; uint8_t v___x_2962_; 
v___x_2961_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___closed__5));
v___x_2962_ = lean_string_dec_eq(v_str_2954_, v___x_2961_);
if (v___x_2962_ == 0)
{
return v___y_2938_;
}
else
{
return v_suppressElabErrors_2939_;
}
}
}
}
else
{
return v___y_2938_;
}
}
default: 
{
return v___y_2938_;
}
}
}
case 0:
{
lean_object* v_str_2963_; lean_object* v___x_2964_; uint8_t v___x_2965_; 
v_str_2963_ = lean_ctor_get(v_x_2940_, 1);
v___x_2964_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0));
v___x_2965_ = lean_string_dec_eq(v_str_2963_, v___x_2964_);
if (v___x_2965_ == 0)
{
return v___y_2938_;
}
else
{
return v_suppressElabErrors_2939_;
}
}
default: 
{
return v___y_2938_;
}
}
}
else
{
return v___y_2938_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___boxed(lean_object* v___y_2966_, lean_object* v_suppressElabErrors_2967_, lean_object* v_x_2968_){
_start:
{
uint8_t v___y_25171__boxed_2969_; uint8_t v_suppressElabErrors_boxed_2970_; uint8_t v_res_2971_; lean_object* v_r_2972_; 
v___y_25171__boxed_2969_ = lean_unbox(v___y_2966_);
v_suppressElabErrors_boxed_2970_ = lean_unbox(v_suppressElabErrors_2967_);
v_res_2971_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0(v___y_25171__boxed_2969_, v_suppressElabErrors_boxed_2970_, v_x_2968_);
lean_dec(v_x_2968_);
v_r_2972_ = lean_box(v_res_2971_);
return v_r_2972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4_spec__8(lean_object* v_msgData_2973_, lean_object* v___y_2974_, lean_object* v___y_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_){
_start:
{
lean_object* v___x_2979_; lean_object* v_env_2980_; lean_object* v___x_2981_; lean_object* v_mctx_2982_; lean_object* v_lctx_2983_; lean_object* v_options_2984_; lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2987_; 
v___x_2979_ = lean_st_ref_get(v___y_2977_);
v_env_2980_ = lean_ctor_get(v___x_2979_, 0);
lean_inc_ref(v_env_2980_);
lean_dec(v___x_2979_);
v___x_2981_ = lean_st_ref_get(v___y_2975_);
v_mctx_2982_ = lean_ctor_get(v___x_2981_, 0);
lean_inc_ref(v_mctx_2982_);
lean_dec(v___x_2981_);
v_lctx_2983_ = lean_ctor_get(v___y_2974_, 2);
v_options_2984_ = lean_ctor_get(v___y_2976_, 2);
lean_inc_ref(v_options_2984_);
lean_inc_ref(v_lctx_2983_);
v___x_2985_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2985_, 0, v_env_2980_);
lean_ctor_set(v___x_2985_, 1, v_mctx_2982_);
lean_ctor_set(v___x_2985_, 2, v_lctx_2983_);
lean_ctor_set(v___x_2985_, 3, v_options_2984_);
v___x_2986_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2986_, 0, v___x_2985_);
lean_ctor_set(v___x_2986_, 1, v_msgData_2973_);
v___x_2987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2987_, 0, v___x_2986_);
return v___x_2987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4_spec__8___boxed(lean_object* v_msgData_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_){
_start:
{
lean_object* v_res_2994_; 
v_res_2994_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4_spec__8(v_msgData_2988_, v___y_2989_, v___y_2990_, v___y_2991_, v___y_2992_);
lean_dec(v___y_2992_);
lean_dec_ref(v___y_2991_);
lean_dec(v___y_2990_);
lean_dec_ref(v___y_2989_);
return v_res_2994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg(lean_object* v_ref_2995_, lean_object* v_msgData_2996_, uint8_t v_severity_2997_, uint8_t v_isSilent_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_, lean_object* v___y_3001_, lean_object* v___y_3002_){
_start:
{
lean_object* v___y_3005_; lean_object* v___y_3006_; uint8_t v___y_3007_; lean_object* v___y_3008_; lean_object* v___y_3009_; uint8_t v___y_3010_; lean_object* v___y_3011_; lean_object* v___y_3012_; lean_object* v___y_3013_; lean_object* v___y_3041_; uint8_t v___y_3042_; lean_object* v___y_3043_; lean_object* v___y_3044_; uint8_t v___y_3045_; lean_object* v___y_3046_; uint8_t v___y_3047_; lean_object* v___y_3048_; lean_object* v___y_3066_; uint8_t v___y_3067_; lean_object* v___y_3068_; lean_object* v___y_3069_; uint8_t v___y_3070_; lean_object* v___y_3071_; uint8_t v___y_3072_; lean_object* v___y_3073_; lean_object* v___y_3077_; uint8_t v___y_3078_; lean_object* v___y_3079_; uint8_t v___y_3080_; lean_object* v___y_3081_; lean_object* v___y_3082_; uint8_t v___y_3083_; uint8_t v___x_3088_; uint8_t v___y_3090_; lean_object* v___y_3091_; lean_object* v___y_3092_; lean_object* v___y_3093_; lean_object* v___y_3094_; uint8_t v___y_3095_; uint8_t v___y_3096_; uint8_t v___y_3098_; uint8_t v___x_3113_; 
v___x_3088_ = 2;
v___x_3113_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2997_, v___x_3088_);
if (v___x_3113_ == 0)
{
v___y_3098_ = v___x_3113_;
goto v___jp_3097_;
}
else
{
uint8_t v___x_3114_; 
lean_inc_ref(v_msgData_2996_);
v___x_3114_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2996_);
v___y_3098_ = v___x_3114_;
goto v___jp_3097_;
}
v___jp_3004_:
{
lean_object* v___x_3014_; lean_object* v_currNamespace_3015_; lean_object* v_openDecls_3016_; lean_object* v_env_3017_; lean_object* v_nextMacroScope_3018_; lean_object* v_ngen_3019_; lean_object* v_auxDeclNGen_3020_; lean_object* v_traceState_3021_; lean_object* v_cache_3022_; lean_object* v_messages_3023_; lean_object* v_infoState_3024_; lean_object* v_snapshotTasks_3025_; lean_object* v___x_3027_; uint8_t v_isShared_3028_; uint8_t v_isSharedCheck_3039_; 
v___x_3014_ = lean_st_ref_take(v___y_3013_);
v_currNamespace_3015_ = lean_ctor_get(v___y_3012_, 6);
v_openDecls_3016_ = lean_ctor_get(v___y_3012_, 7);
v_env_3017_ = lean_ctor_get(v___x_3014_, 0);
v_nextMacroScope_3018_ = lean_ctor_get(v___x_3014_, 1);
v_ngen_3019_ = lean_ctor_get(v___x_3014_, 2);
v_auxDeclNGen_3020_ = lean_ctor_get(v___x_3014_, 3);
v_traceState_3021_ = lean_ctor_get(v___x_3014_, 4);
v_cache_3022_ = lean_ctor_get(v___x_3014_, 5);
v_messages_3023_ = lean_ctor_get(v___x_3014_, 6);
v_infoState_3024_ = lean_ctor_get(v___x_3014_, 7);
v_snapshotTasks_3025_ = lean_ctor_get(v___x_3014_, 8);
v_isSharedCheck_3039_ = !lean_is_exclusive(v___x_3014_);
if (v_isSharedCheck_3039_ == 0)
{
v___x_3027_ = v___x_3014_;
v_isShared_3028_ = v_isSharedCheck_3039_;
goto v_resetjp_3026_;
}
else
{
lean_inc(v_snapshotTasks_3025_);
lean_inc(v_infoState_3024_);
lean_inc(v_messages_3023_);
lean_inc(v_cache_3022_);
lean_inc(v_traceState_3021_);
lean_inc(v_auxDeclNGen_3020_);
lean_inc(v_ngen_3019_);
lean_inc(v_nextMacroScope_3018_);
lean_inc(v_env_3017_);
lean_dec(v___x_3014_);
v___x_3027_ = lean_box(0);
v_isShared_3028_ = v_isSharedCheck_3039_;
goto v_resetjp_3026_;
}
v_resetjp_3026_:
{
lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; lean_object* v___x_3034_; 
lean_inc(v_openDecls_3016_);
lean_inc(v_currNamespace_3015_);
v___x_3029_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3029_, 0, v_currNamespace_3015_);
lean_ctor_set(v___x_3029_, 1, v_openDecls_3016_);
v___x_3030_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_3030_, 0, v___x_3029_);
lean_ctor_set(v___x_3030_, 1, v___y_3008_);
lean_inc_ref(v___y_3005_);
lean_inc_ref(v___y_3006_);
v___x_3031_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_3031_, 0, v___y_3006_);
lean_ctor_set(v___x_3031_, 1, v___y_3009_);
lean_ctor_set(v___x_3031_, 2, v___y_3011_);
lean_ctor_set(v___x_3031_, 3, v___y_3005_);
lean_ctor_set(v___x_3031_, 4, v___x_3030_);
lean_ctor_set_uint8(v___x_3031_, sizeof(void*)*5, v___y_3007_);
lean_ctor_set_uint8(v___x_3031_, sizeof(void*)*5 + 1, v___y_3010_);
lean_ctor_set_uint8(v___x_3031_, sizeof(void*)*5 + 2, v_isSilent_2998_);
v___x_3032_ = l_Lean_MessageLog_add(v___x_3031_, v_messages_3023_);
if (v_isShared_3028_ == 0)
{
lean_ctor_set(v___x_3027_, 6, v___x_3032_);
v___x_3034_ = v___x_3027_;
goto v_reusejp_3033_;
}
else
{
lean_object* v_reuseFailAlloc_3038_; 
v_reuseFailAlloc_3038_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3038_, 0, v_env_3017_);
lean_ctor_set(v_reuseFailAlloc_3038_, 1, v_nextMacroScope_3018_);
lean_ctor_set(v_reuseFailAlloc_3038_, 2, v_ngen_3019_);
lean_ctor_set(v_reuseFailAlloc_3038_, 3, v_auxDeclNGen_3020_);
lean_ctor_set(v_reuseFailAlloc_3038_, 4, v_traceState_3021_);
lean_ctor_set(v_reuseFailAlloc_3038_, 5, v_cache_3022_);
lean_ctor_set(v_reuseFailAlloc_3038_, 6, v___x_3032_);
lean_ctor_set(v_reuseFailAlloc_3038_, 7, v_infoState_3024_);
lean_ctor_set(v_reuseFailAlloc_3038_, 8, v_snapshotTasks_3025_);
v___x_3034_ = v_reuseFailAlloc_3038_;
goto v_reusejp_3033_;
}
v_reusejp_3033_:
{
lean_object* v___x_3035_; lean_object* v___x_3036_; lean_object* v___x_3037_; 
v___x_3035_ = lean_st_ref_set(v___y_3013_, v___x_3034_);
v___x_3036_ = lean_box(0);
v___x_3037_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3037_, 0, v___x_3036_);
return v___x_3037_;
}
}
}
v___jp_3040_:
{
lean_object* v___x_3049_; lean_object* v___x_3050_; lean_object* v_a_3051_; lean_object* v___x_3053_; uint8_t v_isShared_3054_; uint8_t v_isSharedCheck_3064_; 
v___x_3049_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2996_);
v___x_3050_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4_spec__8(v___x_3049_, v___y_2999_, v___y_3000_, v___y_3001_, v___y_3002_);
v_a_3051_ = lean_ctor_get(v___x_3050_, 0);
v_isSharedCheck_3064_ = !lean_is_exclusive(v___x_3050_);
if (v_isSharedCheck_3064_ == 0)
{
v___x_3053_ = v___x_3050_;
v_isShared_3054_ = v_isSharedCheck_3064_;
goto v_resetjp_3052_;
}
else
{
lean_inc(v_a_3051_);
lean_dec(v___x_3050_);
v___x_3053_ = lean_box(0);
v_isShared_3054_ = v_isSharedCheck_3064_;
goto v_resetjp_3052_;
}
v_resetjp_3052_:
{
lean_object* v___x_3055_; lean_object* v___x_3056_; lean_object* v___x_3057_; lean_object* v___x_3058_; 
lean_inc_ref_n(v___y_3046_, 2);
v___x_3055_ = l_Lean_FileMap_toPosition(v___y_3046_, v___y_3043_);
lean_dec(v___y_3043_);
v___x_3056_ = l_Lean_FileMap_toPosition(v___y_3046_, v___y_3048_);
lean_dec(v___y_3048_);
v___x_3057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3057_, 0, v___x_3056_);
v___x_3058_ = ((lean_object*)(lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3___closed__0));
if (v___y_3042_ == 0)
{
lean_del_object(v___x_3053_);
lean_dec_ref(v___y_3041_);
v___y_3005_ = v___x_3058_;
v___y_3006_ = v___y_3044_;
v___y_3007_ = v___y_3045_;
v___y_3008_ = v_a_3051_;
v___y_3009_ = v___x_3055_;
v___y_3010_ = v___y_3047_;
v___y_3011_ = v___x_3057_;
v___y_3012_ = v___y_3001_;
v___y_3013_ = v___y_3002_;
goto v___jp_3004_;
}
else
{
uint8_t v___x_3059_; 
lean_inc(v_a_3051_);
v___x_3059_ = l_Lean_MessageData_hasTag(v___y_3041_, v_a_3051_);
if (v___x_3059_ == 0)
{
lean_object* v___x_3060_; lean_object* v___x_3062_; 
lean_dec_ref_known(v___x_3057_, 1);
lean_dec_ref(v___x_3055_);
lean_dec(v_a_3051_);
v___x_3060_ = lean_box(0);
if (v_isShared_3054_ == 0)
{
lean_ctor_set(v___x_3053_, 0, v___x_3060_);
v___x_3062_ = v___x_3053_;
goto v_reusejp_3061_;
}
else
{
lean_object* v_reuseFailAlloc_3063_; 
v_reuseFailAlloc_3063_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3063_, 0, v___x_3060_);
v___x_3062_ = v_reuseFailAlloc_3063_;
goto v_reusejp_3061_;
}
v_reusejp_3061_:
{
return v___x_3062_;
}
}
else
{
lean_del_object(v___x_3053_);
v___y_3005_ = v___x_3058_;
v___y_3006_ = v___y_3044_;
v___y_3007_ = v___y_3045_;
v___y_3008_ = v_a_3051_;
v___y_3009_ = v___x_3055_;
v___y_3010_ = v___y_3047_;
v___y_3011_ = v___x_3057_;
v___y_3012_ = v___y_3001_;
v___y_3013_ = v___y_3002_;
goto v___jp_3004_;
}
}
}
}
v___jp_3065_:
{
lean_object* v___x_3074_; 
v___x_3074_ = l_Lean_Syntax_getTailPos_x3f(v___y_3068_, v___y_3070_);
lean_dec(v___y_3068_);
if (lean_obj_tag(v___x_3074_) == 0)
{
lean_inc(v___y_3073_);
v___y_3041_ = v___y_3066_;
v___y_3042_ = v___y_3067_;
v___y_3043_ = v___y_3073_;
v___y_3044_ = v___y_3069_;
v___y_3045_ = v___y_3070_;
v___y_3046_ = v___y_3071_;
v___y_3047_ = v___y_3072_;
v___y_3048_ = v___y_3073_;
goto v___jp_3040_;
}
else
{
lean_object* v_val_3075_; 
v_val_3075_ = lean_ctor_get(v___x_3074_, 0);
lean_inc(v_val_3075_);
lean_dec_ref_known(v___x_3074_, 1);
v___y_3041_ = v___y_3066_;
v___y_3042_ = v___y_3067_;
v___y_3043_ = v___y_3073_;
v___y_3044_ = v___y_3069_;
v___y_3045_ = v___y_3070_;
v___y_3046_ = v___y_3071_;
v___y_3047_ = v___y_3072_;
v___y_3048_ = v_val_3075_;
goto v___jp_3040_;
}
}
v___jp_3076_:
{
lean_object* v_ref_3084_; lean_object* v___x_3085_; 
v_ref_3084_ = l_Lean_replaceRef(v_ref_2995_, v___y_3082_);
v___x_3085_ = l_Lean_Syntax_getPos_x3f(v_ref_3084_, v___y_3080_);
if (lean_obj_tag(v___x_3085_) == 0)
{
lean_object* v___x_3086_; 
v___x_3086_ = lean_unsigned_to_nat(0u);
v___y_3066_ = v___y_3077_;
v___y_3067_ = v___y_3078_;
v___y_3068_ = v_ref_3084_;
v___y_3069_ = v___y_3079_;
v___y_3070_ = v___y_3080_;
v___y_3071_ = v___y_3081_;
v___y_3072_ = v___y_3083_;
v___y_3073_ = v___x_3086_;
goto v___jp_3065_;
}
else
{
lean_object* v_val_3087_; 
v_val_3087_ = lean_ctor_get(v___x_3085_, 0);
lean_inc(v_val_3087_);
lean_dec_ref_known(v___x_3085_, 1);
v___y_3066_ = v___y_3077_;
v___y_3067_ = v___y_3078_;
v___y_3068_ = v_ref_3084_;
v___y_3069_ = v___y_3079_;
v___y_3070_ = v___y_3080_;
v___y_3071_ = v___y_3081_;
v___y_3072_ = v___y_3083_;
v___y_3073_ = v_val_3087_;
goto v___jp_3065_;
}
}
v___jp_3089_:
{
if (v___y_3096_ == 0)
{
v___y_3077_ = v___y_3092_;
v___y_3078_ = v___y_3090_;
v___y_3079_ = v___y_3091_;
v___y_3080_ = v___y_3095_;
v___y_3081_ = v___y_3093_;
v___y_3082_ = v___y_3094_;
v___y_3083_ = v_severity_2997_;
goto v___jp_3076_;
}
else
{
v___y_3077_ = v___y_3092_;
v___y_3078_ = v___y_3090_;
v___y_3079_ = v___y_3091_;
v___y_3080_ = v___y_3095_;
v___y_3081_ = v___y_3093_;
v___y_3082_ = v___y_3094_;
v___y_3083_ = v___x_3088_;
goto v___jp_3076_;
}
}
v___jp_3097_:
{
if (v___y_3098_ == 0)
{
lean_object* v_fileName_3099_; lean_object* v_fileMap_3100_; lean_object* v_options_3101_; lean_object* v_ref_3102_; uint8_t v_suppressElabErrors_3103_; lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___f_3106_; uint8_t v___x_3107_; uint8_t v___x_3108_; 
v_fileName_3099_ = lean_ctor_get(v___y_3001_, 0);
v_fileMap_3100_ = lean_ctor_get(v___y_3001_, 1);
v_options_3101_ = lean_ctor_get(v___y_3001_, 2);
v_ref_3102_ = lean_ctor_get(v___y_3001_, 5);
v_suppressElabErrors_3103_ = lean_ctor_get_uint8(v___y_3001_, sizeof(void*)*14 + 1);
v___x_3104_ = lean_box(v___y_3098_);
v___x_3105_ = lean_box(v_suppressElabErrors_3103_);
v___f_3106_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3106_, 0, v___x_3104_);
lean_closure_set(v___f_3106_, 1, v___x_3105_);
v___x_3107_ = 1;
v___x_3108_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2997_, v___x_3107_);
if (v___x_3108_ == 0)
{
v___y_3090_ = v_suppressElabErrors_3103_;
v___y_3091_ = v_fileName_3099_;
v___y_3092_ = v___f_3106_;
v___y_3093_ = v_fileMap_3100_;
v___y_3094_ = v_ref_3102_;
v___y_3095_ = v___y_3098_;
v___y_3096_ = v___x_3108_;
goto v___jp_3089_;
}
else
{
lean_object* v___x_3109_; uint8_t v___x_3110_; 
v___x_3109_ = l_Lean_warningAsError;
v___x_3110_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(v_options_3101_, v___x_3109_);
v___y_3090_ = v_suppressElabErrors_3103_;
v___y_3091_ = v_fileName_3099_;
v___y_3092_ = v___f_3106_;
v___y_3093_ = v_fileMap_3100_;
v___y_3094_ = v_ref_3102_;
v___y_3095_ = v___y_3098_;
v___y_3096_ = v___x_3110_;
goto v___jp_3089_;
}
}
else
{
lean_object* v___x_3111_; lean_object* v___x_3112_; 
lean_dec_ref(v_msgData_2996_);
v___x_3111_ = lean_box(0);
v___x_3112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3112_, 0, v___x_3111_);
return v___x_3112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg___boxed(lean_object* v_ref_3115_, lean_object* v_msgData_3116_, lean_object* v_severity_3117_, lean_object* v_isSilent_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_, lean_object* v___y_3121_, lean_object* v___y_3122_, lean_object* v___y_3123_){
_start:
{
uint8_t v_severity_boxed_3124_; uint8_t v_isSilent_boxed_3125_; lean_object* v_res_3126_; 
v_severity_boxed_3124_ = lean_unbox(v_severity_3117_);
v_isSilent_boxed_3125_ = lean_unbox(v_isSilent_3118_);
v_res_3126_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg(v_ref_3115_, v_msgData_3116_, v_severity_boxed_3124_, v_isSilent_boxed_3125_, v___y_3119_, v___y_3120_, v___y_3121_, v___y_3122_);
lean_dec(v___y_3122_);
lean_dec_ref(v___y_3121_);
lean_dec(v___y_3120_);
lean_dec_ref(v___y_3119_);
lean_dec(v_ref_3115_);
return v_res_3126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3(lean_object* v_msgData_3127_, uint8_t v_severity_3128_, uint8_t v_isSilent_3129_, lean_object* v___y_3130_, lean_object* v___y_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_){
_start:
{
lean_object* v_ref_3139_; lean_object* v___x_3140_; 
v_ref_3139_ = lean_ctor_get(v___y_3136_, 5);
v___x_3140_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg(v_ref_3139_, v_msgData_3127_, v_severity_3128_, v_isSilent_3129_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
return v___x_3140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3___boxed(lean_object* v_msgData_3141_, lean_object* v_severity_3142_, lean_object* v_isSilent_3143_, lean_object* v___y_3144_, lean_object* v___y_3145_, lean_object* v___y_3146_, lean_object* v___y_3147_, lean_object* v___y_3148_, lean_object* v___y_3149_, lean_object* v___y_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_){
_start:
{
uint8_t v_severity_boxed_3153_; uint8_t v_isSilent_boxed_3154_; lean_object* v_res_3155_; 
v_severity_boxed_3153_ = lean_unbox(v_severity_3142_);
v_isSilent_boxed_3154_ = lean_unbox(v_isSilent_3143_);
v_res_3155_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3(v_msgData_3141_, v_severity_boxed_3153_, v_isSilent_boxed_3154_, v___y_3144_, v___y_3145_, v___y_3146_, v___y_3147_, v___y_3148_, v___y_3149_, v___y_3150_, v___y_3151_);
lean_dec(v___y_3151_);
lean_dec_ref(v___y_3150_);
lean_dec(v___y_3149_);
lean_dec_ref(v___y_3148_);
lean_dec(v___y_3147_);
lean_dec_ref(v___y_3146_);
lean_dec(v___y_3145_);
lean_dec_ref(v___y_3144_);
return v_res_3155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__6(lean_object* v_msgData_3156_, lean_object* v___y_3157_, lean_object* v___y_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_, lean_object* v___y_3163_, lean_object* v___y_3164_){
_start:
{
uint8_t v___x_3166_; uint8_t v___x_3167_; lean_object* v___x_3168_; 
v___x_3166_ = 0;
v___x_3167_ = 0;
v___x_3168_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3(v_msgData_3156_, v___x_3166_, v___x_3167_, v___y_3157_, v___y_3158_, v___y_3159_, v___y_3160_, v___y_3161_, v___y_3162_, v___y_3163_, v___y_3164_);
return v___x_3168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__6___boxed(lean_object* v_msgData_3169_, lean_object* v___y_3170_, lean_object* v___y_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_, lean_object* v___y_3175_, lean_object* v___y_3176_, lean_object* v___y_3177_, lean_object* v___y_3178_){
_start:
{
lean_object* v_res_3179_; 
v_res_3179_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__6(v_msgData_3169_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_, v___y_3174_, v___y_3175_, v___y_3176_, v___y_3177_);
lean_dec(v___y_3177_);
lean_dec_ref(v___y_3176_);
lean_dec(v___y_3175_);
lean_dec_ref(v___y_3174_);
lean_dec(v___y_3173_);
lean_dec_ref(v___y_3172_);
lean_dec(v___y_3171_);
lean_dec_ref(v___y_3170_);
return v_res_3179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__4(size_t v_sz_3180_, size_t v_i_3181_, lean_object* v_bs_3182_){
_start:
{
uint8_t v___x_3183_; 
v___x_3183_ = lean_usize_dec_lt(v_i_3181_, v_sz_3180_);
if (v___x_3183_ == 0)
{
return v_bs_3182_;
}
else
{
lean_object* v_v_3184_; lean_object* v_msg_3185_; lean_object* v___x_3186_; lean_object* v_bs_x27_3187_; size_t v___x_3188_; size_t v___x_3189_; lean_object* v___x_3190_; 
v_v_3184_ = lean_array_uget_borrowed(v_bs_3182_, v_i_3181_);
v_msg_3185_ = lean_ctor_get(v_v_3184_, 1);
lean_inc_ref(v_msg_3185_);
v___x_3186_ = lean_unsigned_to_nat(0u);
v_bs_x27_3187_ = lean_array_uset(v_bs_3182_, v_i_3181_, v___x_3186_);
v___x_3188_ = ((size_t)1ULL);
v___x_3189_ = lean_usize_add(v_i_3181_, v___x_3188_);
v___x_3190_ = lean_array_uset(v_bs_x27_3187_, v_i_3181_, v_msg_3185_);
v_i_3181_ = v___x_3189_;
v_bs_3182_ = v___x_3190_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__4___boxed(lean_object* v_sz_3192_, lean_object* v_i_3193_, lean_object* v_bs_3194_){
_start:
{
size_t v_sz_boxed_3195_; size_t v_i_boxed_3196_; lean_object* v_res_3197_; 
v_sz_boxed_3195_ = lean_unbox_usize(v_sz_3192_);
lean_dec(v_sz_3192_);
v_i_boxed_3196_ = lean_unbox_usize(v_i_3193_);
lean_dec(v_i_3193_);
v_res_3197_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__4(v_sz_boxed_3195_, v_i_boxed_3196_, v_bs_3194_);
return v_res_3197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3(lean_object* v_msgData_3198_, lean_object* v___y_3199_, lean_object* v___y_3200_, lean_object* v___y_3201_, lean_object* v___y_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_, lean_object* v___y_3206_){
_start:
{
uint8_t v___x_3208_; uint8_t v___x_3209_; lean_object* v___x_3210_; 
v___x_3208_ = 1;
v___x_3209_ = 0;
v___x_3210_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3(v_msgData_3198_, v___x_3208_, v___x_3209_, v___y_3199_, v___y_3200_, v___y_3201_, v___y_3202_, v___y_3203_, v___y_3204_, v___y_3205_, v___y_3206_);
return v___x_3210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3___boxed(lean_object* v_msgData_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_, lean_object* v___y_3219_, lean_object* v___y_3220_){
_start:
{
lean_object* v_res_3221_; 
v_res_3221_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3(v_msgData_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_, v___y_3217_, v___y_3218_, v___y_3219_);
lean_dec(v___y_3219_);
lean_dec_ref(v___y_3218_);
lean_dec(v___y_3217_);
lean_dec_ref(v___y_3216_);
lean_dec(v___y_3215_);
lean_dec_ref(v___y_3214_);
lean_dec(v___y_3213_);
lean_dec_ref(v___y_3212_);
return v_res_3221_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0(void){
_start:
{
lean_object* v_failureEmoji_3222_; lean_object* v___x_3223_; 
v_failureEmoji_3222_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__0);
v___x_3223_ = l_Lean_stringToMessageData(v_failureEmoji_3222_);
return v___x_3223_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__1(void){
_start:
{
lean_object* v___x_3224_; lean_object* v___x_3225_; lean_object* v___x_3226_; 
v___x_3224_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0);
v___x_3225_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__1);
v___x_3226_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3226_, 0, v___x_3225_);
lean_ctor_set(v___x_3226_, 1, v___x_3224_);
return v___x_3226_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2(void){
_start:
{
lean_object* v___x_3227_; lean_object* v___x_3228_; lean_object* v___x_3229_; 
v___x_3227_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3);
v___x_3228_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__1, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__1_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__1);
v___x_3229_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3229_, 0, v___x_3228_);
lean_ctor_set(v___x_3229_, 1, v___x_3227_);
return v___x_3229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8(lean_object* v_a_3230_, lean_object* v_a_3231_){
_start:
{
if (lean_obj_tag(v_a_3230_) == 0)
{
lean_object* v___x_3232_; 
v___x_3232_ = l_List_reverse___redArg(v_a_3231_);
return v___x_3232_;
}
else
{
lean_object* v_head_3233_; lean_object* v_tail_3234_; lean_object* v___x_3236_; uint8_t v_isShared_3237_; uint8_t v_isSharedCheck_3244_; 
v_head_3233_ = lean_ctor_get(v_a_3230_, 0);
v_tail_3234_ = lean_ctor_get(v_a_3230_, 1);
v_isSharedCheck_3244_ = !lean_is_exclusive(v_a_3230_);
if (v_isSharedCheck_3244_ == 0)
{
v___x_3236_ = v_a_3230_;
v_isShared_3237_ = v_isSharedCheck_3244_;
goto v_resetjp_3235_;
}
else
{
lean_inc(v_tail_3234_);
lean_inc(v_head_3233_);
lean_dec(v_a_3230_);
v___x_3236_ = lean_box(0);
v_isShared_3237_ = v_isSharedCheck_3244_;
goto v_resetjp_3235_;
}
v_resetjp_3235_:
{
lean_object* v___x_3238_; lean_object* v___x_3239_; lean_object* v___x_3241_; 
v___x_3238_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2);
v___x_3239_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3239_, 0, v___x_3238_);
lean_ctor_set(v___x_3239_, 1, v_head_3233_);
if (v_isShared_3237_ == 0)
{
lean_ctor_set(v___x_3236_, 1, v_a_3231_);
lean_ctor_set(v___x_3236_, 0, v___x_3239_);
v___x_3241_ = v___x_3236_;
goto v_reusejp_3240_;
}
else
{
lean_object* v_reuseFailAlloc_3243_; 
v_reuseFailAlloc_3243_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3243_, 0, v___x_3239_);
lean_ctor_set(v_reuseFailAlloc_3243_, 1, v_a_3231_);
v___x_3241_ = v_reuseFailAlloc_3243_;
goto v_reusejp_3240_;
}
v_reusejp_3240_:
{
v_a_3230_ = v_tail_3234_;
v_a_3231_ = v___x_3241_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__0(void){
_start:
{
lean_object* v___x_3245_; lean_object* v___x_3246_; lean_object* v___x_3247_; 
v___x_3245_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__0);
v___x_3246_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__1);
v___x_3247_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3247_, 0, v___x_3246_);
lean_ctor_set(v___x_3247_, 1, v___x_3245_);
return v___x_3247_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__1(void){
_start:
{
lean_object* v___x_3248_; lean_object* v___x_3249_; lean_object* v___x_3250_; 
v___x_3248_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__1___closed__3);
v___x_3249_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__0, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__0);
v___x_3250_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3250_, 0, v___x_3249_);
lean_ctor_set(v___x_3250_, 1, v___x_3248_);
return v___x_3250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6(lean_object* v_a_3251_, lean_object* v_a_3252_){
_start:
{
if (lean_obj_tag(v_a_3251_) == 0)
{
lean_object* v___x_3253_; 
v___x_3253_ = l_List_reverse___redArg(v_a_3252_);
return v___x_3253_;
}
else
{
lean_object* v_head_3254_; lean_object* v_tail_3255_; lean_object* v___x_3257_; uint8_t v_isShared_3258_; uint8_t v_isSharedCheck_3265_; 
v_head_3254_ = lean_ctor_get(v_a_3251_, 0);
v_tail_3255_ = lean_ctor_get(v_a_3251_, 1);
v_isSharedCheck_3265_ = !lean_is_exclusive(v_a_3251_);
if (v_isSharedCheck_3265_ == 0)
{
v___x_3257_ = v_a_3251_;
v_isShared_3258_ = v_isSharedCheck_3265_;
goto v_resetjp_3256_;
}
else
{
lean_inc(v_tail_3255_);
lean_inc(v_head_3254_);
lean_dec(v_a_3251_);
v___x_3257_ = lean_box(0);
v_isShared_3258_ = v_isSharedCheck_3265_;
goto v_resetjp_3256_;
}
v_resetjp_3256_:
{
lean_object* v___x_3259_; lean_object* v___x_3260_; lean_object* v___x_3262_; 
v___x_3259_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__1, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__1_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6___closed__1);
v___x_3260_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3260_, 0, v___x_3259_);
lean_ctor_set(v___x_3260_, 1, v_head_3254_);
if (v_isShared_3258_ == 0)
{
lean_ctor_set(v___x_3257_, 1, v_a_3252_);
lean_ctor_set(v___x_3257_, 0, v___x_3260_);
v___x_3262_ = v___x_3257_;
goto v_reusejp_3261_;
}
else
{
lean_object* v_reuseFailAlloc_3264_; 
v_reuseFailAlloc_3264_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3264_, 0, v___x_3260_);
lean_ctor_set(v_reuseFailAlloc_3264_, 1, v_a_3252_);
v___x_3262_ = v_reuseFailAlloc_3264_;
goto v_reusejp_3261_;
}
v_reusejp_3261_:
{
v_a_3251_ = v_tail_3255_;
v_a_3252_ = v___x_3262_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___redArg(lean_object* v_as_3266_, size_t v_sz_3267_, size_t v_i_3268_, lean_object* v_b_3269_){
_start:
{
uint8_t v___x_3271_; 
v___x_3271_ = lean_usize_dec_lt(v_i_3268_, v_sz_3267_);
if (v___x_3271_ == 0)
{
lean_object* v___x_3272_; 
v___x_3272_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3272_, 0, v_b_3269_);
return v___x_3272_;
}
else
{
lean_object* v_a_3273_; lean_object* v_fst_3274_; lean_object* v_snd_3275_; lean_object* v___x_3277_; uint8_t v_isShared_3278_; uint8_t v_isSharedCheck_3295_; 
v_a_3273_ = lean_array_uget(v_as_3266_, v_i_3268_);
v_fst_3274_ = lean_ctor_get(v_a_3273_, 0);
v_snd_3275_ = lean_ctor_get(v_a_3273_, 1);
v_isSharedCheck_3295_ = !lean_is_exclusive(v_a_3273_);
if (v_isSharedCheck_3295_ == 0)
{
v___x_3277_ = v_a_3273_;
v_isShared_3278_ = v_isSharedCheck_3295_;
goto v_resetjp_3276_;
}
else
{
lean_inc(v_snd_3275_);
lean_inc(v_fst_3274_);
lean_dec(v_a_3273_);
v___x_3277_ = lean_box(0);
v_isShared_3278_ = v_isSharedCheck_3295_;
goto v_resetjp_3276_;
}
v_resetjp_3276_:
{
lean_object* v___x_3279_; lean_object* v___x_3280_; lean_object* v___x_3281_; lean_object* v___x_3282_; lean_object* v___x_3283_; lean_object* v___x_3284_; lean_object* v___x_3286_; 
v___x_3279_ = lean_array_to_list(v_snd_3275_);
v___x_3280_ = lean_box(0);
v___x_3281_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6(v___x_3279_, v___x_3280_);
v___x_3282_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v___x_3283_ = l_Lean_MessageData_joinSep(v___x_3281_, v___x_3282_);
v___x_3284_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2);
if (v_isShared_3278_ == 0)
{
lean_ctor_set_tag(v___x_3277_, 7);
lean_ctor_set(v___x_3277_, 1, v_fst_3274_);
lean_ctor_set(v___x_3277_, 0, v___x_3284_);
v___x_3286_ = v___x_3277_;
goto v_reusejp_3285_;
}
else
{
lean_object* v_reuseFailAlloc_3294_; 
v_reuseFailAlloc_3294_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3294_, 0, v___x_3284_);
lean_ctor_set(v_reuseFailAlloc_3294_, 1, v_fst_3274_);
v___x_3286_ = v_reuseFailAlloc_3294_;
goto v_reusejp_3285_;
}
v_reusejp_3285_:
{
lean_object* v___x_3287_; lean_object* v___x_3288_; lean_object* v___x_3289_; lean_object* v___x_3290_; size_t v___x_3291_; size_t v___x_3292_; 
v___x_3287_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2);
v___x_3288_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3288_, 0, v___x_3286_);
lean_ctor_set(v___x_3288_, 1, v___x_3287_);
v___x_3289_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3289_, 0, v___x_3288_);
lean_ctor_set(v___x_3289_, 1, v___x_3283_);
v___x_3290_ = lean_array_push(v_b_3269_, v___x_3289_);
v___x_3291_ = ((size_t)1ULL);
v___x_3292_ = lean_usize_add(v_i_3268_, v___x_3291_);
v_i_3268_ = v___x_3292_;
v_b_3269_ = v___x_3290_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___redArg___boxed(lean_object* v_as_3296_, lean_object* v_sz_3297_, lean_object* v_i_3298_, lean_object* v_b_3299_, lean_object* v___y_3300_){
_start:
{
size_t v_sz_boxed_3301_; size_t v_i_boxed_3302_; lean_object* v_res_3303_; 
v_sz_boxed_3301_ = lean_unbox_usize(v_sz_3297_);
lean_dec(v_sz_3297_);
v_i_boxed_3302_ = lean_unbox_usize(v_i_3298_);
lean_dec(v_i_3298_);
v_res_3303_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___redArg(v_as_3296_, v_sz_boxed_3301_, v_i_boxed_3302_, v_b_3299_);
lean_dec_ref(v_as_3296_);
return v_res_3303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5(lean_object* v_kind_3304_, lean_object* v_uniqueFailures_3305_, lean_object* v_synthResults_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_, lean_object* v___y_3310_, lean_object* v___y_3311_, lean_object* v___y_3312_, lean_object* v___y_3313_, lean_object* v___y_3314_){
_start:
{
lean_object* v___x_3316_; lean_object* v___x_3317_; uint8_t v___x_3318_; 
v___x_3316_ = lean_array_get_size(v_synthResults_3306_);
v___x_3317_ = lean_unsigned_to_nat(0u);
v___x_3318_ = lean_nat_dec_eq(v___x_3316_, v___x_3317_);
if (v___x_3318_ == 0)
{
lean_object* v_entries_3319_; size_t v_sz_3320_; size_t v___x_3321_; lean_object* v___x_3322_; 
lean_dec_ref(v_uniqueFailures_3305_);
v_entries_3319_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__1));
v_sz_3320_ = lean_array_size(v_synthResults_3306_);
v___x_3321_ = ((size_t)0ULL);
v___x_3322_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___redArg(v_synthResults_3306_, v_sz_3320_, v___x_3321_, v_entries_3319_);
if (lean_obj_tag(v___x_3322_) == 0)
{
lean_object* v_a_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v_report_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; 
v_a_3323_ = lean_ctor_get(v___x_3322_, 0);
lean_inc(v_a_3323_);
lean_dec_ref_known(v___x_3322_, 1);
v___x_3324_ = lean_array_to_list(v_a_3323_);
v___x_3325_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v_report_3326_ = l_Lean_MessageData_joinSep(v___x_3324_, v___x_3325_);
v___x_3327_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_3328_ = l_Lean_stringToMessageData(v_kind_3304_);
v___x_3329_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3329_, 0, v___x_3327_);
lean_ctor_set(v___x_3329_, 1, v___x_3328_);
v___x_3330_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6);
v___x_3331_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3331_, 0, v___x_3329_);
lean_ctor_set(v___x_3331_, 1, v___x_3330_);
v___x_3332_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3332_, 0, v___x_3331_);
lean_ctor_set(v___x_3332_, 1, v_report_3326_);
v___x_3333_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3(v___x_3332_, v___y_3307_, v___y_3308_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_, v___y_3313_, v___y_3314_);
return v___x_3333_;
}
else
{
lean_object* v_a_3334_; lean_object* v___x_3336_; uint8_t v_isShared_3337_; uint8_t v_isSharedCheck_3341_; 
lean_dec_ref(v_kind_3304_);
v_a_3334_ = lean_ctor_get(v___x_3322_, 0);
v_isSharedCheck_3341_ = !lean_is_exclusive(v___x_3322_);
if (v_isSharedCheck_3341_ == 0)
{
v___x_3336_ = v___x_3322_;
v_isShared_3337_ = v_isSharedCheck_3341_;
goto v_resetjp_3335_;
}
else
{
lean_inc(v_a_3334_);
lean_dec(v___x_3322_);
v___x_3336_ = lean_box(0);
v_isShared_3337_ = v_isSharedCheck_3341_;
goto v_resetjp_3335_;
}
v_resetjp_3335_:
{
lean_object* v___x_3339_; 
if (v_isShared_3337_ == 0)
{
v___x_3339_ = v___x_3336_;
goto v_reusejp_3338_;
}
else
{
lean_object* v_reuseFailAlloc_3340_; 
v_reuseFailAlloc_3340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3340_, 0, v_a_3334_);
v___x_3339_ = v_reuseFailAlloc_3340_;
goto v_reusejp_3338_;
}
v_reusejp_3338_:
{
return v___x_3339_;
}
}
}
}
else
{
lean_object* v___x_3342_; uint8_t v___x_3343_; 
v___x_3342_ = lean_array_get_size(v_uniqueFailures_3305_);
v___x_3343_ = lean_nat_dec_eq(v___x_3342_, v___x_3317_);
if (v___x_3343_ == 0)
{
lean_object* v___x_3344_; lean_object* v___x_3345_; lean_object* v___x_3346_; lean_object* v___x_3347_; lean_object* v_failureList_3348_; lean_object* v___x_3349_; lean_object* v___x_3350_; lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; lean_object* v___x_3354_; lean_object* v___x_3355_; 
v___x_3344_ = lean_array_to_list(v_uniqueFailures_3305_);
v___x_3345_ = lean_box(0);
v___x_3346_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8(v___x_3344_, v___x_3345_);
v___x_3347_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v_failureList_3348_ = l_Lean_MessageData_joinSep(v___x_3346_, v___x_3347_);
v___x_3349_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_3350_ = l_Lean_stringToMessageData(v_kind_3304_);
v___x_3351_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3351_, 0, v___x_3349_);
lean_ctor_set(v___x_3351_, 1, v___x_3350_);
v___x_3352_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4);
v___x_3353_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3353_, 0, v___x_3351_);
lean_ctor_set(v___x_3353_, 1, v___x_3352_);
v___x_3354_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3354_, 0, v___x_3353_);
lean_ctor_set(v___x_3354_, 1, v_failureList_3348_);
v___x_3355_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3(v___x_3354_, v___y_3307_, v___y_3308_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_, v___y_3313_, v___y_3314_);
return v___x_3355_;
}
else
{
lean_object* v___x_3356_; lean_object* v___x_3357_; lean_object* v___x_3358_; lean_object* v___x_3359_; lean_object* v___x_3360_; lean_object* v___x_3361_; 
lean_dec_ref(v_uniqueFailures_3305_);
v___x_3356_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_3357_ = l_Lean_stringToMessageData(v_kind_3304_);
v___x_3358_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3358_, 0, v___x_3356_);
lean_ctor_set(v___x_3358_, 1, v___x_3357_);
v___x_3359_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6);
v___x_3360_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3360_, 0, v___x_3358_);
lean_ctor_set(v___x_3360_, 1, v___x_3359_);
v___x_3361_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3(v___x_3360_, v___y_3307_, v___y_3308_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_, v___y_3313_, v___y_3314_);
return v___x_3361_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5___boxed(lean_object* v_kind_3362_, lean_object* v_uniqueFailures_3363_, lean_object* v_synthResults_3364_, lean_object* v___y_3365_, lean_object* v___y_3366_, lean_object* v___y_3367_, lean_object* v___y_3368_, lean_object* v___y_3369_, lean_object* v___y_3370_, lean_object* v___y_3371_, lean_object* v___y_3372_, lean_object* v___y_3373_){
_start:
{
lean_object* v_res_3374_; 
v_res_3374_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5(v_kind_3362_, v_uniqueFailures_3363_, v_synthResults_3364_, v___y_3365_, v___y_3366_, v___y_3367_, v___y_3368_, v___y_3369_, v___y_3370_, v___y_3371_, v___y_3372_);
lean_dec(v___y_3372_);
lean_dec_ref(v___y_3371_);
lean_dec(v___y_3370_);
lean_dec_ref(v___y_3369_);
lean_dec(v___y_3368_);
lean_dec_ref(v___y_3367_);
lean_dec(v___y_3366_);
lean_dec_ref(v___y_3365_);
lean_dec_ref(v_synthResults_3364_);
return v_res_3374_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__2(void){
_start:
{
lean_object* v___x_3378_; lean_object* v___x_3379_; 
v___x_3378_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__1));
v___x_3379_ = l_Lean_MessageData_ofFormat(v___x_3378_);
return v___x_3379_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__5(void){
_start:
{
lean_object* v___x_3383_; lean_object* v___x_3384_; 
v___x_3383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__4));
v___x_3384_ = l_Lean_MessageData_ofFormat(v___x_3383_);
return v___x_3384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1(lean_object* v___x_3385_, uint8_t v___x_3386_, lean_object* v___x_3387_, lean_object* v___x_3388_, lean_object* v___y_3389_, lean_object* v___y_3390_, lean_object* v___y_3391_, lean_object* v___y_3392_, lean_object* v___y_3393_, lean_object* v___y_3394_, lean_object* v___y_3395_, lean_object* v___y_3396_){
_start:
{
lean_object* v___x_3398_; 
v___x_3398_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3390_, v___y_3392_, v___y_3394_, v___y_3396_);
if (lean_obj_tag(v___x_3398_) == 0)
{
lean_object* v_a_3399_; lean_object* v___x_3400_; lean_object* v_traceState_3401_; lean_object* v_traces_3402_; lean_object* v___x_3403_; 
v_a_3399_ = lean_ctor_get(v___x_3398_, 0);
lean_inc(v_a_3399_);
lean_dec_ref_known(v___x_3398_, 1);
v___x_3400_ = lean_st_ref_get(v___y_3396_);
v_traceState_3401_ = lean_ctor_get(v___x_3400_, 4);
lean_inc_ref(v_traceState_3401_);
lean_dec(v___x_3400_);
v_traces_3402_ = lean_ctor_get(v_traceState_3401_, 0);
lean_inc_ref_n(v_traces_3402_, 2);
lean_dec_ref(v_traceState_3401_);
lean_inc(v___x_3387_);
lean_inc(v___x_3385_);
v___x_3403_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0(v___x_3385_, v_traces_3402_, v___x_3386_, v___x_3387_, v___x_3386_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
if (lean_obj_tag(v___x_3403_) == 0)
{
lean_object* v_a_3404_; lean_object* v_fst_3405_; lean_object* v_snd_3406_; lean_object* v___x_3407_; 
v_a_3404_ = lean_ctor_get(v___x_3403_, 0);
lean_inc(v_a_3404_);
lean_dec_ref_known(v___x_3403_, 1);
v_fst_3405_ = lean_ctor_get(v_a_3404_, 0);
lean_inc(v_fst_3405_);
v_snd_3406_ = lean_ctor_get(v_a_3404_, 1);
lean_inc(v_snd_3406_);
lean_dec(v_a_3404_);
lean_inc(v_a_3399_);
v___x_3407_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3399_, v___x_3386_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
if (lean_obj_tag(v___x_3407_) == 0)
{
lean_dec_ref_known(v___x_3407_, 1);
if (lean_obj_tag(v_fst_3405_) == 0)
{
uint8_t v___x_3408_; lean_object* v___x_3409_; 
lean_dec_ref_known(v_fst_3405_, 1);
v___x_3408_ = 0;
lean_inc(v___x_3387_);
lean_inc(v___x_3385_);
v___x_3409_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0(v___x_3385_, v_traces_3402_, v___x_3386_, v___x_3387_, v___x_3408_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
if (lean_obj_tag(v___x_3409_) == 0)
{
lean_object* v_a_3410_; lean_object* v_fst_3411_; lean_object* v_snd_3412_; lean_object* v___x_3413_; 
v_a_3410_ = lean_ctor_get(v___x_3409_, 0);
lean_inc(v_a_3410_);
lean_dec_ref_known(v___x_3409_, 1);
v_fst_3411_ = lean_ctor_get(v_a_3410_, 0);
lean_inc(v_fst_3411_);
v_snd_3412_ = lean_ctor_get(v_a_3410_, 1);
lean_inc(v_snd_3412_);
lean_dec(v_a_3410_);
v___x_3413_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3399_, v___x_3386_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
if (lean_obj_tag(v___x_3413_) == 0)
{
lean_dec_ref_known(v___x_3413_, 1);
if (lean_obj_tag(v_fst_3411_) == 0)
{
lean_object* v___x_3414_; lean_object* v___x_3415_; 
lean_dec_ref_known(v_fst_3411_, 1);
lean_dec(v_snd_3412_);
lean_dec(v_snd_3406_);
lean_dec_ref(v___x_3388_);
lean_dec(v___x_3385_);
v___x_3414_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__2);
v___x_3415_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3(v___x_3414_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
if (lean_obj_tag(v___x_3415_) == 0)
{
lean_object* v___x_3416_; 
lean_dec_ref_known(v___x_3415_, 1);
v___x_3416_ = l_Lean_Elab_Tactic_evalTactic(v___x_3387_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
lean_dec_ref(v___y_3395_);
return v___x_3416_;
}
else
{
lean_dec_ref(v___y_3395_);
lean_dec(v___x_3387_);
return v___x_3415_;
}
}
else
{
lean_object* v___x_3417_; size_t v_sz_3418_; size_t v___x_3419_; lean_object* v___x_3420_; lean_object* v___x_3421_; size_t v_sz_3422_; lean_object* v___x_3423_; lean_object* v___x_3424_; lean_object* v_fst_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; lean_object* v___x_3428_; 
lean_dec_ref_known(v_fst_3411_, 1);
v___x_3417_ = l_Lean_PersistentArray_toArray___redArg(v_snd_3406_);
lean_dec(v_snd_3406_);
v_sz_3418_ = lean_array_size(v___x_3417_);
v___x_3419_ = ((size_t)0ULL);
v___x_3420_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__4(v_sz_3418_, v___x_3419_, v___x_3417_);
v___x_3421_ = l_Lean_PersistentArray_toArray___redArg(v_snd_3412_);
lean_dec(v_snd_3412_);
v_sz_3422_ = lean_array_size(v___x_3421_);
v___x_3423_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__4(v_sz_3422_, v___x_3419_, v___x_3421_);
v___x_3424_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces(v___x_3420_, v___x_3423_, v___x_3408_);
lean_dec_ref(v___x_3423_);
lean_dec_ref(v___x_3420_);
v_fst_3425_ = lean_ctor_get(v___x_3424_, 0);
lean_inc(v_fst_3425_);
lean_dec_ref(v___x_3424_);
v___x_3426_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures(v_fst_3425_);
v___x_3427_ = lean_mk_empty_array_with_capacity(v___x_3385_);
lean_dec(v___x_3385_);
v___x_3428_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5(v___x_3388_, v___x_3426_, v___x_3427_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
lean_dec_ref(v___x_3427_);
if (lean_obj_tag(v___x_3428_) == 0)
{
lean_object* v___x_3429_; lean_object* v_fileName_3430_; lean_object* v_fileMap_3431_; lean_object* v_options_3432_; lean_object* v_currRecDepth_3433_; lean_object* v_ref_3434_; lean_object* v_currNamespace_3435_; lean_object* v_openDecls_3436_; lean_object* v_initHeartbeats_3437_; lean_object* v_maxHeartbeats_3438_; lean_object* v_quotContext_3439_; lean_object* v_currMacroScope_3440_; lean_object* v_cancelTk_x3f_3441_; uint8_t v_suppressElabErrors_3442_; lean_object* v_inheritedTraceOptions_3443_; lean_object* v___x_3445_; uint8_t v_isShared_3446_; uint8_t v_isSharedCheck_3496_; 
lean_dec_ref_known(v___x_3428_, 1);
v___x_3429_ = lean_st_ref_get(v___y_3396_);
v_fileName_3430_ = lean_ctor_get(v___y_3395_, 0);
v_fileMap_3431_ = lean_ctor_get(v___y_3395_, 1);
v_options_3432_ = lean_ctor_get(v___y_3395_, 2);
v_currRecDepth_3433_ = lean_ctor_get(v___y_3395_, 3);
v_ref_3434_ = lean_ctor_get(v___y_3395_, 5);
v_currNamespace_3435_ = lean_ctor_get(v___y_3395_, 6);
v_openDecls_3436_ = lean_ctor_get(v___y_3395_, 7);
v_initHeartbeats_3437_ = lean_ctor_get(v___y_3395_, 8);
v_maxHeartbeats_3438_ = lean_ctor_get(v___y_3395_, 9);
v_quotContext_3439_ = lean_ctor_get(v___y_3395_, 10);
v_currMacroScope_3440_ = lean_ctor_get(v___y_3395_, 11);
v_cancelTk_x3f_3441_ = lean_ctor_get(v___y_3395_, 12);
v_suppressElabErrors_3442_ = lean_ctor_get_uint8(v___y_3395_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3443_ = lean_ctor_get(v___y_3395_, 13);
v_isSharedCheck_3496_ = !lean_is_exclusive(v___y_3395_);
if (v_isSharedCheck_3496_ == 0)
{
lean_object* v_unused_3497_; 
v_unused_3497_ = lean_ctor_get(v___y_3395_, 4);
lean_dec(v_unused_3497_);
v___x_3445_ = v___y_3395_;
v_isShared_3446_ = v_isSharedCheck_3496_;
goto v_resetjp_3444_;
}
else
{
lean_inc(v_inheritedTraceOptions_3443_);
lean_inc(v_cancelTk_x3f_3441_);
lean_inc(v_currMacroScope_3440_);
lean_inc(v_quotContext_3439_);
lean_inc(v_maxHeartbeats_3438_);
lean_inc(v_initHeartbeats_3437_);
lean_inc(v_openDecls_3436_);
lean_inc(v_currNamespace_3435_);
lean_inc(v_ref_3434_);
lean_inc(v_currRecDepth_3433_);
lean_inc(v_options_3432_);
lean_inc(v_fileMap_3431_);
lean_inc(v_fileName_3430_);
lean_dec(v___y_3395_);
v___x_3445_ = lean_box(0);
v_isShared_3446_ = v_isSharedCheck_3496_;
goto v_resetjp_3444_;
}
v_resetjp_3444_:
{
lean_object* v_env_3447_; lean_object* v___x_3448_; lean_object* v___x_3449_; lean_object* v___x_3450_; uint8_t v___x_3451_; lean_object* v_fileName_3453_; lean_object* v_fileMap_3454_; lean_object* v_currRecDepth_3455_; lean_object* v_ref_3456_; lean_object* v_currNamespace_3457_; lean_object* v_openDecls_3458_; lean_object* v_initHeartbeats_3459_; lean_object* v_maxHeartbeats_3460_; lean_object* v_quotContext_3461_; lean_object* v_currMacroScope_3462_; lean_object* v_cancelTk_x3f_3463_; uint8_t v_suppressElabErrors_3464_; lean_object* v_inheritedTraceOptions_3465_; lean_object* v___y_3466_; uint8_t v___y_3474_; uint8_t v___x_3495_; 
v_env_3447_ = lean_ctor_get(v___x_3429_, 0);
lean_inc_ref(v_env_3447_);
lean_dec(v___x_3429_);
v___x_3448_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3));
v___x_3449_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_options_3432_, v___x_3448_, v___x_3408_);
v___x_3450_ = l_Lean_diagnostics;
v___x_3451_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(v___x_3449_, v___x_3450_);
v___x_3495_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_3447_);
lean_dec_ref(v_env_3447_);
if (v___x_3495_ == 0)
{
if (v___x_3451_ == 0)
{
v___y_3474_ = v___x_3386_;
goto v___jp_3473_;
}
else
{
v___y_3474_ = v___x_3495_;
goto v___jp_3473_;
}
}
else
{
v___y_3474_ = v___x_3451_;
goto v___jp_3473_;
}
v___jp_3452_:
{
lean_object* v___x_3467_; lean_object* v___x_3468_; lean_object* v___x_3470_; 
v___x_3467_ = l_Lean_maxRecDepth;
v___x_3468_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2(v___x_3449_, v___x_3467_);
if (v_isShared_3446_ == 0)
{
lean_ctor_set(v___x_3445_, 13, v_inheritedTraceOptions_3465_);
lean_ctor_set(v___x_3445_, 12, v_cancelTk_x3f_3463_);
lean_ctor_set(v___x_3445_, 11, v_currMacroScope_3462_);
lean_ctor_set(v___x_3445_, 10, v_quotContext_3461_);
lean_ctor_set(v___x_3445_, 9, v_maxHeartbeats_3460_);
lean_ctor_set(v___x_3445_, 8, v_initHeartbeats_3459_);
lean_ctor_set(v___x_3445_, 7, v_openDecls_3458_);
lean_ctor_set(v___x_3445_, 6, v_currNamespace_3457_);
lean_ctor_set(v___x_3445_, 5, v_ref_3456_);
lean_ctor_set(v___x_3445_, 4, v___x_3468_);
lean_ctor_set(v___x_3445_, 3, v_currRecDepth_3455_);
lean_ctor_set(v___x_3445_, 2, v___x_3449_);
lean_ctor_set(v___x_3445_, 1, v_fileMap_3454_);
lean_ctor_set(v___x_3445_, 0, v_fileName_3453_);
v___x_3470_ = v___x_3445_;
goto v_reusejp_3469_;
}
else
{
lean_object* v_reuseFailAlloc_3472_; 
v_reuseFailAlloc_3472_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_3472_, 0, v_fileName_3453_);
lean_ctor_set(v_reuseFailAlloc_3472_, 1, v_fileMap_3454_);
lean_ctor_set(v_reuseFailAlloc_3472_, 2, v___x_3449_);
lean_ctor_set(v_reuseFailAlloc_3472_, 3, v_currRecDepth_3455_);
lean_ctor_set(v_reuseFailAlloc_3472_, 4, v___x_3468_);
lean_ctor_set(v_reuseFailAlloc_3472_, 5, v_ref_3456_);
lean_ctor_set(v_reuseFailAlloc_3472_, 6, v_currNamespace_3457_);
lean_ctor_set(v_reuseFailAlloc_3472_, 7, v_openDecls_3458_);
lean_ctor_set(v_reuseFailAlloc_3472_, 8, v_initHeartbeats_3459_);
lean_ctor_set(v_reuseFailAlloc_3472_, 9, v_maxHeartbeats_3460_);
lean_ctor_set(v_reuseFailAlloc_3472_, 10, v_quotContext_3461_);
lean_ctor_set(v_reuseFailAlloc_3472_, 11, v_currMacroScope_3462_);
lean_ctor_set(v_reuseFailAlloc_3472_, 12, v_cancelTk_x3f_3463_);
lean_ctor_set(v_reuseFailAlloc_3472_, 13, v_inheritedTraceOptions_3465_);
v___x_3470_ = v_reuseFailAlloc_3472_;
goto v_reusejp_3469_;
}
v_reusejp_3469_:
{
lean_object* v___x_3471_; 
lean_ctor_set_uint8(v___x_3470_, sizeof(void*)*14, v___x_3451_);
lean_ctor_set_uint8(v___x_3470_, sizeof(void*)*14 + 1, v_suppressElabErrors_3464_);
v___x_3471_ = l_Lean_Elab_Tactic_evalTactic(v___x_3387_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___x_3470_, v___y_3466_);
lean_dec_ref(v___x_3470_);
return v___x_3471_;
}
}
v___jp_3473_:
{
if (v___y_3474_ == 0)
{
lean_object* v___x_3475_; lean_object* v_env_3476_; lean_object* v_nextMacroScope_3477_; lean_object* v_ngen_3478_; lean_object* v_auxDeclNGen_3479_; lean_object* v_traceState_3480_; lean_object* v_messages_3481_; lean_object* v_infoState_3482_; lean_object* v_snapshotTasks_3483_; lean_object* v___x_3485_; uint8_t v_isShared_3486_; uint8_t v_isSharedCheck_3493_; 
v___x_3475_ = lean_st_ref_take(v___y_3396_);
v_env_3476_ = lean_ctor_get(v___x_3475_, 0);
v_nextMacroScope_3477_ = lean_ctor_get(v___x_3475_, 1);
v_ngen_3478_ = lean_ctor_get(v___x_3475_, 2);
v_auxDeclNGen_3479_ = lean_ctor_get(v___x_3475_, 3);
v_traceState_3480_ = lean_ctor_get(v___x_3475_, 4);
v_messages_3481_ = lean_ctor_get(v___x_3475_, 6);
v_infoState_3482_ = lean_ctor_get(v___x_3475_, 7);
v_snapshotTasks_3483_ = lean_ctor_get(v___x_3475_, 8);
v_isSharedCheck_3493_ = !lean_is_exclusive(v___x_3475_);
if (v_isSharedCheck_3493_ == 0)
{
lean_object* v_unused_3494_; 
v_unused_3494_ = lean_ctor_get(v___x_3475_, 5);
lean_dec(v_unused_3494_);
v___x_3485_ = v___x_3475_;
v_isShared_3486_ = v_isSharedCheck_3493_;
goto v_resetjp_3484_;
}
else
{
lean_inc(v_snapshotTasks_3483_);
lean_inc(v_infoState_3482_);
lean_inc(v_messages_3481_);
lean_inc(v_traceState_3480_);
lean_inc(v_auxDeclNGen_3479_);
lean_inc(v_ngen_3478_);
lean_inc(v_nextMacroScope_3477_);
lean_inc(v_env_3476_);
lean_dec(v___x_3475_);
v___x_3485_ = lean_box(0);
v_isShared_3486_ = v_isSharedCheck_3493_;
goto v_resetjp_3484_;
}
v_resetjp_3484_:
{
lean_object* v___x_3487_; lean_object* v___x_3488_; lean_object* v___x_3490_; 
v___x_3487_ = l_Lean_Kernel_enableDiag(v_env_3476_, v___x_3451_);
v___x_3488_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8);
if (v_isShared_3486_ == 0)
{
lean_ctor_set(v___x_3485_, 5, v___x_3488_);
lean_ctor_set(v___x_3485_, 0, v___x_3487_);
v___x_3490_ = v___x_3485_;
goto v_reusejp_3489_;
}
else
{
lean_object* v_reuseFailAlloc_3492_; 
v_reuseFailAlloc_3492_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3492_, 0, v___x_3487_);
lean_ctor_set(v_reuseFailAlloc_3492_, 1, v_nextMacroScope_3477_);
lean_ctor_set(v_reuseFailAlloc_3492_, 2, v_ngen_3478_);
lean_ctor_set(v_reuseFailAlloc_3492_, 3, v_auxDeclNGen_3479_);
lean_ctor_set(v_reuseFailAlloc_3492_, 4, v_traceState_3480_);
lean_ctor_set(v_reuseFailAlloc_3492_, 5, v___x_3488_);
lean_ctor_set(v_reuseFailAlloc_3492_, 6, v_messages_3481_);
lean_ctor_set(v_reuseFailAlloc_3492_, 7, v_infoState_3482_);
lean_ctor_set(v_reuseFailAlloc_3492_, 8, v_snapshotTasks_3483_);
v___x_3490_ = v_reuseFailAlloc_3492_;
goto v_reusejp_3489_;
}
v_reusejp_3489_:
{
lean_object* v___x_3491_; 
v___x_3491_ = lean_st_ref_set(v___y_3396_, v___x_3490_);
v_fileName_3453_ = v_fileName_3430_;
v_fileMap_3454_ = v_fileMap_3431_;
v_currRecDepth_3455_ = v_currRecDepth_3433_;
v_ref_3456_ = v_ref_3434_;
v_currNamespace_3457_ = v_currNamespace_3435_;
v_openDecls_3458_ = v_openDecls_3436_;
v_initHeartbeats_3459_ = v_initHeartbeats_3437_;
v_maxHeartbeats_3460_ = v_maxHeartbeats_3438_;
v_quotContext_3461_ = v_quotContext_3439_;
v_currMacroScope_3462_ = v_currMacroScope_3440_;
v_cancelTk_x3f_3463_ = v_cancelTk_x3f_3441_;
v_suppressElabErrors_3464_ = v_suppressElabErrors_3442_;
v_inheritedTraceOptions_3465_ = v_inheritedTraceOptions_3443_;
v___y_3466_ = v___y_3396_;
goto v___jp_3452_;
}
}
}
else
{
v_fileName_3453_ = v_fileName_3430_;
v_fileMap_3454_ = v_fileMap_3431_;
v_currRecDepth_3455_ = v_currRecDepth_3433_;
v_ref_3456_ = v_ref_3434_;
v_currNamespace_3457_ = v_currNamespace_3435_;
v_openDecls_3458_ = v_openDecls_3436_;
v_initHeartbeats_3459_ = v_initHeartbeats_3437_;
v_maxHeartbeats_3460_ = v_maxHeartbeats_3438_;
v_quotContext_3461_ = v_quotContext_3439_;
v_currMacroScope_3462_ = v_currMacroScope_3440_;
v_cancelTk_x3f_3463_ = v_cancelTk_x3f_3441_;
v_suppressElabErrors_3464_ = v_suppressElabErrors_3442_;
v_inheritedTraceOptions_3465_ = v_inheritedTraceOptions_3443_;
v___y_3466_ = v___y_3396_;
goto v___jp_3452_;
}
}
}
}
else
{
lean_dec_ref(v___y_3395_);
lean_dec(v___x_3387_);
return v___x_3428_;
}
}
}
else
{
lean_dec(v_snd_3412_);
lean_dec(v_fst_3411_);
lean_dec(v_snd_3406_);
lean_dec_ref(v___y_3395_);
lean_dec_ref(v___x_3388_);
lean_dec(v___x_3387_);
lean_dec(v___x_3385_);
return v___x_3413_;
}
}
else
{
lean_object* v_a_3498_; lean_object* v___x_3500_; uint8_t v_isShared_3501_; uint8_t v_isSharedCheck_3505_; 
lean_dec(v_snd_3406_);
lean_dec(v_a_3399_);
lean_dec_ref(v___y_3395_);
lean_dec_ref(v___x_3388_);
lean_dec(v___x_3387_);
lean_dec(v___x_3385_);
v_a_3498_ = lean_ctor_get(v___x_3409_, 0);
v_isSharedCheck_3505_ = !lean_is_exclusive(v___x_3409_);
if (v_isSharedCheck_3505_ == 0)
{
v___x_3500_ = v___x_3409_;
v_isShared_3501_ = v_isSharedCheck_3505_;
goto v_resetjp_3499_;
}
else
{
lean_inc(v_a_3498_);
lean_dec(v___x_3409_);
v___x_3500_ = lean_box(0);
v_isShared_3501_ = v_isSharedCheck_3505_;
goto v_resetjp_3499_;
}
v_resetjp_3499_:
{
lean_object* v___x_3503_; 
if (v_isShared_3501_ == 0)
{
v___x_3503_ = v___x_3500_;
goto v_reusejp_3502_;
}
else
{
lean_object* v_reuseFailAlloc_3504_; 
v_reuseFailAlloc_3504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3504_, 0, v_a_3498_);
v___x_3503_ = v_reuseFailAlloc_3504_;
goto v_reusejp_3502_;
}
v_reusejp_3502_:
{
return v___x_3503_;
}
}
}
}
else
{
lean_object* v___x_3506_; lean_object* v___x_3507_; 
lean_dec_ref_known(v_fst_3405_, 1);
lean_dec(v_snd_3406_);
lean_dec_ref(v_traces_3402_);
lean_dec(v_a_3399_);
lean_dec_ref(v___x_3388_);
lean_dec(v___x_3385_);
v___x_3506_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__5, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___closed__5);
v___x_3507_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__6(v___x_3506_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___y_3395_, v___y_3396_);
if (lean_obj_tag(v___x_3507_) == 0)
{
lean_object* v___x_3508_; lean_object* v_fileName_3509_; lean_object* v_fileMap_3510_; lean_object* v_options_3511_; lean_object* v_currRecDepth_3512_; lean_object* v_ref_3513_; lean_object* v_currNamespace_3514_; lean_object* v_openDecls_3515_; lean_object* v_initHeartbeats_3516_; lean_object* v_maxHeartbeats_3517_; lean_object* v_quotContext_3518_; lean_object* v_currMacroScope_3519_; lean_object* v_cancelTk_x3f_3520_; uint8_t v_suppressElabErrors_3521_; lean_object* v_inheritedTraceOptions_3522_; lean_object* v___x_3524_; uint8_t v_isShared_3525_; uint8_t v_isSharedCheck_3575_; 
lean_dec_ref_known(v___x_3507_, 1);
v___x_3508_ = lean_st_ref_get(v___y_3396_);
v_fileName_3509_ = lean_ctor_get(v___y_3395_, 0);
v_fileMap_3510_ = lean_ctor_get(v___y_3395_, 1);
v_options_3511_ = lean_ctor_get(v___y_3395_, 2);
v_currRecDepth_3512_ = lean_ctor_get(v___y_3395_, 3);
v_ref_3513_ = lean_ctor_get(v___y_3395_, 5);
v_currNamespace_3514_ = lean_ctor_get(v___y_3395_, 6);
v_openDecls_3515_ = lean_ctor_get(v___y_3395_, 7);
v_initHeartbeats_3516_ = lean_ctor_get(v___y_3395_, 8);
v_maxHeartbeats_3517_ = lean_ctor_get(v___y_3395_, 9);
v_quotContext_3518_ = lean_ctor_get(v___y_3395_, 10);
v_currMacroScope_3519_ = lean_ctor_get(v___y_3395_, 11);
v_cancelTk_x3f_3520_ = lean_ctor_get(v___y_3395_, 12);
v_suppressElabErrors_3521_ = lean_ctor_get_uint8(v___y_3395_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3522_ = lean_ctor_get(v___y_3395_, 13);
v_isSharedCheck_3575_ = !lean_is_exclusive(v___y_3395_);
if (v_isSharedCheck_3575_ == 0)
{
lean_object* v_unused_3576_; 
v_unused_3576_ = lean_ctor_get(v___y_3395_, 4);
lean_dec(v_unused_3576_);
v___x_3524_ = v___y_3395_;
v_isShared_3525_ = v_isSharedCheck_3575_;
goto v_resetjp_3523_;
}
else
{
lean_inc(v_inheritedTraceOptions_3522_);
lean_inc(v_cancelTk_x3f_3520_);
lean_inc(v_currMacroScope_3519_);
lean_inc(v_quotContext_3518_);
lean_inc(v_maxHeartbeats_3517_);
lean_inc(v_initHeartbeats_3516_);
lean_inc(v_openDecls_3515_);
lean_inc(v_currNamespace_3514_);
lean_inc(v_ref_3513_);
lean_inc(v_currRecDepth_3512_);
lean_inc(v_options_3511_);
lean_inc(v_fileMap_3510_);
lean_inc(v_fileName_3509_);
lean_dec(v___y_3395_);
v___x_3524_ = lean_box(0);
v_isShared_3525_ = v_isSharedCheck_3575_;
goto v_resetjp_3523_;
}
v_resetjp_3523_:
{
lean_object* v_env_3526_; lean_object* v___x_3527_; lean_object* v___x_3528_; lean_object* v___x_3529_; uint8_t v___x_3530_; lean_object* v_fileName_3532_; lean_object* v_fileMap_3533_; lean_object* v_currRecDepth_3534_; lean_object* v_ref_3535_; lean_object* v_currNamespace_3536_; lean_object* v_openDecls_3537_; lean_object* v_initHeartbeats_3538_; lean_object* v_maxHeartbeats_3539_; lean_object* v_quotContext_3540_; lean_object* v_currMacroScope_3541_; lean_object* v_cancelTk_x3f_3542_; uint8_t v_suppressElabErrors_3543_; lean_object* v_inheritedTraceOptions_3544_; lean_object* v___y_3545_; uint8_t v___y_3553_; uint8_t v___x_3574_; 
v_env_3526_ = lean_ctor_get(v___x_3508_, 0);
lean_inc_ref(v_env_3526_);
lean_dec(v___x_3508_);
v___x_3527_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3));
v___x_3528_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_options_3511_, v___x_3527_, v___x_3386_);
v___x_3529_ = l_Lean_diagnostics;
v___x_3530_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(v___x_3528_, v___x_3529_);
v___x_3574_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_3526_);
lean_dec_ref(v_env_3526_);
if (v___x_3574_ == 0)
{
if (v___x_3530_ == 0)
{
v___y_3553_ = v___x_3386_;
goto v___jp_3552_;
}
else
{
v___y_3553_ = v___x_3574_;
goto v___jp_3552_;
}
}
else
{
v___y_3553_ = v___x_3530_;
goto v___jp_3552_;
}
v___jp_3531_:
{
lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3549_; 
v___x_3546_ = l_Lean_maxRecDepth;
v___x_3547_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__2(v___x_3528_, v___x_3546_);
if (v_isShared_3525_ == 0)
{
lean_ctor_set(v___x_3524_, 13, v_inheritedTraceOptions_3544_);
lean_ctor_set(v___x_3524_, 12, v_cancelTk_x3f_3542_);
lean_ctor_set(v___x_3524_, 11, v_currMacroScope_3541_);
lean_ctor_set(v___x_3524_, 10, v_quotContext_3540_);
lean_ctor_set(v___x_3524_, 9, v_maxHeartbeats_3539_);
lean_ctor_set(v___x_3524_, 8, v_initHeartbeats_3538_);
lean_ctor_set(v___x_3524_, 7, v_openDecls_3537_);
lean_ctor_set(v___x_3524_, 6, v_currNamespace_3536_);
lean_ctor_set(v___x_3524_, 5, v_ref_3535_);
lean_ctor_set(v___x_3524_, 4, v___x_3547_);
lean_ctor_set(v___x_3524_, 3, v_currRecDepth_3534_);
lean_ctor_set(v___x_3524_, 2, v___x_3528_);
lean_ctor_set(v___x_3524_, 1, v_fileMap_3533_);
lean_ctor_set(v___x_3524_, 0, v_fileName_3532_);
v___x_3549_ = v___x_3524_;
goto v_reusejp_3548_;
}
else
{
lean_object* v_reuseFailAlloc_3551_; 
v_reuseFailAlloc_3551_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_3551_, 0, v_fileName_3532_);
lean_ctor_set(v_reuseFailAlloc_3551_, 1, v_fileMap_3533_);
lean_ctor_set(v_reuseFailAlloc_3551_, 2, v___x_3528_);
lean_ctor_set(v_reuseFailAlloc_3551_, 3, v_currRecDepth_3534_);
lean_ctor_set(v_reuseFailAlloc_3551_, 4, v___x_3547_);
lean_ctor_set(v_reuseFailAlloc_3551_, 5, v_ref_3535_);
lean_ctor_set(v_reuseFailAlloc_3551_, 6, v_currNamespace_3536_);
lean_ctor_set(v_reuseFailAlloc_3551_, 7, v_openDecls_3537_);
lean_ctor_set(v_reuseFailAlloc_3551_, 8, v_initHeartbeats_3538_);
lean_ctor_set(v_reuseFailAlloc_3551_, 9, v_maxHeartbeats_3539_);
lean_ctor_set(v_reuseFailAlloc_3551_, 10, v_quotContext_3540_);
lean_ctor_set(v_reuseFailAlloc_3551_, 11, v_currMacroScope_3541_);
lean_ctor_set(v_reuseFailAlloc_3551_, 12, v_cancelTk_x3f_3542_);
lean_ctor_set(v_reuseFailAlloc_3551_, 13, v_inheritedTraceOptions_3544_);
v___x_3549_ = v_reuseFailAlloc_3551_;
goto v_reusejp_3548_;
}
v_reusejp_3548_:
{
lean_object* v___x_3550_; 
lean_ctor_set_uint8(v___x_3549_, sizeof(void*)*14, v___x_3530_);
lean_ctor_set_uint8(v___x_3549_, sizeof(void*)*14 + 1, v_suppressElabErrors_3543_);
v___x_3550_ = l_Lean_Elab_Tactic_evalTactic(v___x_3387_, v___y_3389_, v___y_3390_, v___y_3391_, v___y_3392_, v___y_3393_, v___y_3394_, v___x_3549_, v___y_3545_);
lean_dec_ref(v___x_3549_);
return v___x_3550_;
}
}
v___jp_3552_:
{
if (v___y_3553_ == 0)
{
lean_object* v___x_3554_; lean_object* v_env_3555_; lean_object* v_nextMacroScope_3556_; lean_object* v_ngen_3557_; lean_object* v_auxDeclNGen_3558_; lean_object* v_traceState_3559_; lean_object* v_messages_3560_; lean_object* v_infoState_3561_; lean_object* v_snapshotTasks_3562_; lean_object* v___x_3564_; uint8_t v_isShared_3565_; uint8_t v_isSharedCheck_3572_; 
v___x_3554_ = lean_st_ref_take(v___y_3396_);
v_env_3555_ = lean_ctor_get(v___x_3554_, 0);
v_nextMacroScope_3556_ = lean_ctor_get(v___x_3554_, 1);
v_ngen_3557_ = lean_ctor_get(v___x_3554_, 2);
v_auxDeclNGen_3558_ = lean_ctor_get(v___x_3554_, 3);
v_traceState_3559_ = lean_ctor_get(v___x_3554_, 4);
v_messages_3560_ = lean_ctor_get(v___x_3554_, 6);
v_infoState_3561_ = lean_ctor_get(v___x_3554_, 7);
v_snapshotTasks_3562_ = lean_ctor_get(v___x_3554_, 8);
v_isSharedCheck_3572_ = !lean_is_exclusive(v___x_3554_);
if (v_isSharedCheck_3572_ == 0)
{
lean_object* v_unused_3573_; 
v_unused_3573_ = lean_ctor_get(v___x_3554_, 5);
lean_dec(v_unused_3573_);
v___x_3564_ = v___x_3554_;
v_isShared_3565_ = v_isSharedCheck_3572_;
goto v_resetjp_3563_;
}
else
{
lean_inc(v_snapshotTasks_3562_);
lean_inc(v_infoState_3561_);
lean_inc(v_messages_3560_);
lean_inc(v_traceState_3559_);
lean_inc(v_auxDeclNGen_3558_);
lean_inc(v_ngen_3557_);
lean_inc(v_nextMacroScope_3556_);
lean_inc(v_env_3555_);
lean_dec(v___x_3554_);
v___x_3564_ = lean_box(0);
v_isShared_3565_ = v_isSharedCheck_3572_;
goto v_resetjp_3563_;
}
v_resetjp_3563_:
{
lean_object* v___x_3566_; lean_object* v___x_3567_; lean_object* v___x_3569_; 
v___x_3566_ = l_Lean_Kernel_enableDiag(v_env_3555_, v___x_3530_);
v___x_3567_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__8);
if (v_isShared_3565_ == 0)
{
lean_ctor_set(v___x_3564_, 5, v___x_3567_);
lean_ctor_set(v___x_3564_, 0, v___x_3566_);
v___x_3569_ = v___x_3564_;
goto v_reusejp_3568_;
}
else
{
lean_object* v_reuseFailAlloc_3571_; 
v_reuseFailAlloc_3571_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3571_, 0, v___x_3566_);
lean_ctor_set(v_reuseFailAlloc_3571_, 1, v_nextMacroScope_3556_);
lean_ctor_set(v_reuseFailAlloc_3571_, 2, v_ngen_3557_);
lean_ctor_set(v_reuseFailAlloc_3571_, 3, v_auxDeclNGen_3558_);
lean_ctor_set(v_reuseFailAlloc_3571_, 4, v_traceState_3559_);
lean_ctor_set(v_reuseFailAlloc_3571_, 5, v___x_3567_);
lean_ctor_set(v_reuseFailAlloc_3571_, 6, v_messages_3560_);
lean_ctor_set(v_reuseFailAlloc_3571_, 7, v_infoState_3561_);
lean_ctor_set(v_reuseFailAlloc_3571_, 8, v_snapshotTasks_3562_);
v___x_3569_ = v_reuseFailAlloc_3571_;
goto v_reusejp_3568_;
}
v_reusejp_3568_:
{
lean_object* v___x_3570_; 
v___x_3570_ = lean_st_ref_set(v___y_3396_, v___x_3569_);
v_fileName_3532_ = v_fileName_3509_;
v_fileMap_3533_ = v_fileMap_3510_;
v_currRecDepth_3534_ = v_currRecDepth_3512_;
v_ref_3535_ = v_ref_3513_;
v_currNamespace_3536_ = v_currNamespace_3514_;
v_openDecls_3537_ = v_openDecls_3515_;
v_initHeartbeats_3538_ = v_initHeartbeats_3516_;
v_maxHeartbeats_3539_ = v_maxHeartbeats_3517_;
v_quotContext_3540_ = v_quotContext_3518_;
v_currMacroScope_3541_ = v_currMacroScope_3519_;
v_cancelTk_x3f_3542_ = v_cancelTk_x3f_3520_;
v_suppressElabErrors_3543_ = v_suppressElabErrors_3521_;
v_inheritedTraceOptions_3544_ = v_inheritedTraceOptions_3522_;
v___y_3545_ = v___y_3396_;
goto v___jp_3531_;
}
}
}
else
{
v_fileName_3532_ = v_fileName_3509_;
v_fileMap_3533_ = v_fileMap_3510_;
v_currRecDepth_3534_ = v_currRecDepth_3512_;
v_ref_3535_ = v_ref_3513_;
v_currNamespace_3536_ = v_currNamespace_3514_;
v_openDecls_3537_ = v_openDecls_3515_;
v_initHeartbeats_3538_ = v_initHeartbeats_3516_;
v_maxHeartbeats_3539_ = v_maxHeartbeats_3517_;
v_quotContext_3540_ = v_quotContext_3518_;
v_currMacroScope_3541_ = v_currMacroScope_3519_;
v_cancelTk_x3f_3542_ = v_cancelTk_x3f_3520_;
v_suppressElabErrors_3543_ = v_suppressElabErrors_3521_;
v_inheritedTraceOptions_3544_ = v_inheritedTraceOptions_3522_;
v___y_3545_ = v___y_3396_;
goto v___jp_3531_;
}
}
}
}
else
{
lean_dec_ref(v___y_3395_);
lean_dec(v___x_3387_);
return v___x_3507_;
}
}
}
else
{
lean_dec(v_snd_3406_);
lean_dec(v_fst_3405_);
lean_dec_ref(v_traces_3402_);
lean_dec(v_a_3399_);
lean_dec_ref(v___y_3395_);
lean_dec_ref(v___x_3388_);
lean_dec(v___x_3387_);
lean_dec(v___x_3385_);
return v___x_3407_;
}
}
else
{
lean_object* v_a_3577_; lean_object* v___x_3579_; uint8_t v_isShared_3580_; uint8_t v_isSharedCheck_3584_; 
lean_dec_ref(v_traces_3402_);
lean_dec(v_a_3399_);
lean_dec_ref(v___y_3395_);
lean_dec_ref(v___x_3388_);
lean_dec(v___x_3387_);
lean_dec(v___x_3385_);
v_a_3577_ = lean_ctor_get(v___x_3403_, 0);
v_isSharedCheck_3584_ = !lean_is_exclusive(v___x_3403_);
if (v_isSharedCheck_3584_ == 0)
{
v___x_3579_ = v___x_3403_;
v_isShared_3580_ = v_isSharedCheck_3584_;
goto v_resetjp_3578_;
}
else
{
lean_inc(v_a_3577_);
lean_dec(v___x_3403_);
v___x_3579_ = lean_box(0);
v_isShared_3580_ = v_isSharedCheck_3584_;
goto v_resetjp_3578_;
}
v_resetjp_3578_:
{
lean_object* v___x_3582_; 
if (v_isShared_3580_ == 0)
{
v___x_3582_ = v___x_3579_;
goto v_reusejp_3581_;
}
else
{
lean_object* v_reuseFailAlloc_3583_; 
v_reuseFailAlloc_3583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3583_, 0, v_a_3577_);
v___x_3582_ = v_reuseFailAlloc_3583_;
goto v_reusejp_3581_;
}
v_reusejp_3581_:
{
return v___x_3582_;
}
}
}
}
else
{
lean_object* v_a_3585_; lean_object* v___x_3587_; uint8_t v_isShared_3588_; uint8_t v_isSharedCheck_3592_; 
lean_dec_ref(v___y_3395_);
lean_dec_ref(v___x_3388_);
lean_dec(v___x_3387_);
lean_dec(v___x_3385_);
v_a_3585_ = lean_ctor_get(v___x_3398_, 0);
v_isSharedCheck_3592_ = !lean_is_exclusive(v___x_3398_);
if (v_isSharedCheck_3592_ == 0)
{
v___x_3587_ = v___x_3398_;
v_isShared_3588_ = v_isSharedCheck_3592_;
goto v_resetjp_3586_;
}
else
{
lean_inc(v_a_3585_);
lean_dec(v___x_3398_);
v___x_3587_ = lean_box(0);
v_isShared_3588_ = v_isSharedCheck_3592_;
goto v_resetjp_3586_;
}
v_resetjp_3586_:
{
lean_object* v___x_3590_; 
if (v_isShared_3588_ == 0)
{
v___x_3590_ = v___x_3587_;
goto v_reusejp_3589_;
}
else
{
lean_object* v_reuseFailAlloc_3591_; 
v_reuseFailAlloc_3591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3591_, 0, v_a_3585_);
v___x_3590_ = v_reuseFailAlloc_3591_;
goto v_reusejp_3589_;
}
v_reusejp_3589_:
{
return v___x_3590_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___boxed(lean_object* v___x_3593_, lean_object* v___x_3594_, lean_object* v___x_3595_, lean_object* v___x_3596_, lean_object* v___y_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_, lean_object* v___y_3600_, lean_object* v___y_3601_, lean_object* v___y_3602_, lean_object* v___y_3603_, lean_object* v___y_3604_, lean_object* v___y_3605_){
_start:
{
uint8_t v___x_25970__boxed_3606_; lean_object* v_res_3607_; 
v___x_25970__boxed_3606_ = lean_unbox(v___x_3594_);
v_res_3607_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1(v___x_3593_, v___x_25970__boxed_3606_, v___x_3595_, v___x_3596_, v___y_3597_, v___y_3598_, v___y_3599_, v___y_3600_, v___y_3601_, v___y_3602_, v___y_3603_, v___y_3604_);
lean_dec(v___y_3604_);
lean_dec(v___y_3602_);
lean_dec_ref(v___y_3601_);
lean_dec(v___y_3600_);
lean_dec_ref(v___y_3599_);
lean_dec(v___y_3598_);
lean_dec_ref(v___y_3597_);
return v_res_3607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1(lean_object* v_x_3608_, lean_object* v_a_3609_, lean_object* v_a_3610_, lean_object* v_a_3611_, lean_object* v_a_3612_, lean_object* v_a_3613_, lean_object* v_a_3614_, lean_object* v_a_3615_, lean_object* v_a_3616_){
_start:
{
lean_object* v___x_3618_; uint8_t v___x_3619_; 
v___x_3618_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__3));
lean_inc(v_x_3608_);
v___x_3619_ = l_Lean_Syntax_isOfKind(v_x_3608_, v___x_3618_);
if (v___x_3619_ == 0)
{
lean_object* v___x_3620_; 
lean_dec(v_x_3608_);
v___x_3620_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg();
return v___x_3620_;
}
else
{
lean_object* v___x_3621_; lean_object* v___x_3622_; lean_object* v___x_3623_; lean_object* v___x_3624_; lean_object* v___x_3625_; lean_object* v___f_3626_; lean_object* v___x_3627_; 
v___x_3621_ = lean_unsigned_to_nat(0u);
v___x_3622_ = lean_unsigned_to_nat(2u);
v___x_3623_ = l_Lean_Syntax_getArg(v_x_3608_, v___x_3622_);
lean_dec(v_x_3608_);
v___x_3624_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuse___closed__11));
v___x_3625_ = lean_box(v___x_3619_);
v___f_3626_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__1___boxed), 13, 4);
lean_closure_set(v___f_3626_, 0, v___x_3621_);
lean_closure_set(v___f_3626_, 1, v___x_3625_);
lean_closure_set(v___f_3626_, 2, v___x_3623_);
lean_closure_set(v___f_3626_, 3, v___x_3624_);
v___x_3627_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3626_, v_a_3609_, v_a_3610_, v_a_3611_, v_a_3612_, v_a_3613_, v_a_3614_, v_a_3615_, v_a_3616_);
return v___x_3627_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___boxed(lean_object* v_x_3628_, lean_object* v_a_3629_, lean_object* v_a_3630_, lean_object* v_a_3631_, lean_object* v_a_3632_, lean_object* v_a_3633_, lean_object* v_a_3634_, lean_object* v_a_3635_, lean_object* v_a_3636_, lean_object* v_a_3637_){
_start:
{
lean_object* v_res_3638_; 
v_res_3638_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1(v_x_3628_, v_a_3629_, v_a_3630_, v_a_3631_, v_a_3632_, v_a_3633_, v_a_3634_, v_a_3635_, v_a_3636_);
lean_dec(v_a_3636_);
lean_dec_ref(v_a_3635_);
lean_dec(v_a_3634_);
lean_dec_ref(v_a_3633_);
lean_dec(v_a_3632_);
lean_dec_ref(v_a_3631_);
lean_dec(v_a_3630_);
lean_dec_ref(v_a_3629_);
return v_res_3638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7(lean_object* v_as_3639_, size_t v_sz_3640_, size_t v_i_3641_, lean_object* v_b_3642_, lean_object* v___y_3643_, lean_object* v___y_3644_, lean_object* v___y_3645_, lean_object* v___y_3646_, lean_object* v___y_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_){
_start:
{
lean_object* v___x_3652_; 
v___x_3652_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___redArg(v_as_3639_, v_sz_3640_, v_i_3641_, v_b_3642_);
return v___x_3652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7___boxed(lean_object* v_as_3653_, lean_object* v_sz_3654_, lean_object* v_i_3655_, lean_object* v_b_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_, lean_object* v___y_3663_, lean_object* v___y_3664_, lean_object* v___y_3665_){
_start:
{
size_t v_sz_boxed_3666_; size_t v_i_boxed_3667_; lean_object* v_res_3668_; 
v_sz_boxed_3666_ = lean_unbox_usize(v_sz_3654_);
lean_dec(v_sz_3654_);
v_i_boxed_3667_ = lean_unbox_usize(v_i_3655_);
lean_dec(v_i_3655_);
v_res_3668_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__7(v_as_3653_, v_sz_boxed_3666_, v_i_boxed_3667_, v_b_3656_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_, v___y_3664_);
lean_dec(v___y_3664_);
lean_dec_ref(v___y_3663_);
lean_dec(v___y_3662_);
lean_dec_ref(v___y_3661_);
lean_dec(v___y_3660_);
lean_dec_ref(v___y_3659_);
lean_dec(v___y_3658_);
lean_dec_ref(v___y_3657_);
lean_dec_ref(v_as_3653_);
return v_res_3668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4(lean_object* v_ref_3669_, lean_object* v_msgData_3670_, uint8_t v_severity_3671_, uint8_t v_isSilent_3672_, lean_object* v___y_3673_, lean_object* v___y_3674_, lean_object* v___y_3675_, lean_object* v___y_3676_, lean_object* v___y_3677_, lean_object* v___y_3678_, lean_object* v___y_3679_, lean_object* v___y_3680_){
_start:
{
lean_object* v___x_3682_; 
v___x_3682_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___redArg(v_ref_3669_, v_msgData_3670_, v_severity_3671_, v_isSilent_3672_, v___y_3677_, v___y_3678_, v___y_3679_, v___y_3680_);
return v___x_3682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4___boxed(lean_object* v_ref_3683_, lean_object* v_msgData_3684_, lean_object* v_severity_3685_, lean_object* v_isSilent_3686_, lean_object* v___y_3687_, lean_object* v___y_3688_, lean_object* v___y_3689_, lean_object* v___y_3690_, lean_object* v___y_3691_, lean_object* v___y_3692_, lean_object* v___y_3693_, lean_object* v___y_3694_, lean_object* v___y_3695_){
_start:
{
uint8_t v_severity_boxed_3696_; uint8_t v_isSilent_boxed_3697_; lean_object* v_res_3698_; 
v_severity_boxed_3696_ = lean_unbox(v_severity_3685_);
v_isSilent_boxed_3697_ = lean_unbox(v_isSilent_3686_);
v_res_3698_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__3_spec__3_spec__4(v_ref_3683_, v_msgData_3684_, v_severity_boxed_3696_, v_isSilent_boxed_3697_, v___y_3687_, v___y_3688_, v___y_3689_, v___y_3690_, v___y_3691_, v___y_3692_, v___y_3693_, v___y_3694_);
lean_dec(v___y_3694_);
lean_dec_ref(v___y_3693_);
lean_dec(v___y_3692_);
lean_dec_ref(v___y_3691_);
lean_dec(v___y_3690_);
lean_dec_ref(v___y_3689_);
lean_dec(v___y_3688_);
lean_dec_ref(v___y_3687_);
lean_dec(v_ref_3683_);
return v_res_3698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3730_; lean_object* v___x_3731_; 
v___x_3730_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__0___redArg___closed__0);
v___x_3731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3731_, 0, v___x_3730_);
return v___x_3731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___redArg___boxed(lean_object* v___y_3732_){
_start:
{
lean_object* v_res_3733_; 
v_res_3733_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___redArg();
return v_res_3733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0(lean_object* v_00_u03b1_3734_, lean_object* v___y_3735_, lean_object* v___y_3736_){
_start:
{
lean_object* v___x_3738_; 
v___x_3738_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___redArg();
return v___x_3738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___boxed(lean_object* v_00_u03b1_3739_, lean_object* v___y_3740_, lean_object* v___y_3741_, lean_object* v___y_3742_){
_start:
{
lean_object* v_res_3743_; 
v_res_3743_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0(v_00_u03b1_3739_, v___y_3740_, v___y_3741_);
lean_dec(v___y_3741_);
lean_dec_ref(v___y_3740_);
return v_res_3743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0(uint8_t v___x_3752_, uint8_t v_strict_3753_, lean_object* v_scope_3754_){
_start:
{
lean_object* v_header_3755_; lean_object* v_opts_3756_; lean_object* v_currNamespace_3757_; lean_object* v_openDecls_3758_; lean_object* v_levelNames_3759_; lean_object* v_varDecls_3760_; lean_object* v_varUIds_3761_; lean_object* v_includedVars_3762_; lean_object* v_omittedVars_3763_; uint8_t v_isNoncomputable_3764_; uint8_t v_isPublic_3765_; uint8_t v_isMeta_3766_; lean_object* v_attrs_3767_; lean_object* v___x_3769_; uint8_t v_isShared_3770_; uint8_t v_isSharedCheck_3783_; 
v_header_3755_ = lean_ctor_get(v_scope_3754_, 0);
v_opts_3756_ = lean_ctor_get(v_scope_3754_, 1);
v_currNamespace_3757_ = lean_ctor_get(v_scope_3754_, 2);
v_openDecls_3758_ = lean_ctor_get(v_scope_3754_, 3);
v_levelNames_3759_ = lean_ctor_get(v_scope_3754_, 4);
v_varDecls_3760_ = lean_ctor_get(v_scope_3754_, 5);
v_varUIds_3761_ = lean_ctor_get(v_scope_3754_, 6);
v_includedVars_3762_ = lean_ctor_get(v_scope_3754_, 7);
v_omittedVars_3763_ = lean_ctor_get(v_scope_3754_, 8);
v_isNoncomputable_3764_ = lean_ctor_get_uint8(v_scope_3754_, sizeof(void*)*10);
v_isPublic_3765_ = lean_ctor_get_uint8(v_scope_3754_, sizeof(void*)*10 + 1);
v_isMeta_3766_ = lean_ctor_get_uint8(v_scope_3754_, sizeof(void*)*10 + 2);
v_attrs_3767_ = lean_ctor_get(v_scope_3754_, 9);
v_isSharedCheck_3783_ = !lean_is_exclusive(v_scope_3754_);
if (v_isSharedCheck_3783_ == 0)
{
v___x_3769_ = v_scope_3754_;
v_isShared_3770_ = v_isSharedCheck_3783_;
goto v_resetjp_3768_;
}
else
{
lean_inc(v_attrs_3767_);
lean_inc(v_omittedVars_3763_);
lean_inc(v_includedVars_3762_);
lean_inc(v_varUIds_3761_);
lean_inc(v_varDecls_3760_);
lean_inc(v_levelNames_3759_);
lean_inc(v_openDecls_3758_);
lean_inc(v_currNamespace_3757_);
lean_inc(v_opts_3756_);
lean_inc(v_header_3755_);
lean_dec(v_scope_3754_);
v___x_3769_ = lean_box(0);
v_isShared_3770_ = v_isSharedCheck_3783_;
goto v_resetjp_3768_;
}
v_resetjp_3768_:
{
lean_object* v___x_3771_; uint8_t v___x_3772_; lean_object* v___x_3773_; lean_object* v___x_3774_; lean_object* v___x_3775_; lean_object* v___x_3776_; lean_object* v___x_3777_; lean_object* v___x_3778_; lean_object* v___x_3779_; lean_object* v___x_3781_; 
v___x_3771_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__1));
v___x_3772_ = 0;
v___x_3773_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_opts_3756_, v___x_3771_, v___x_3772_);
v___x_3774_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3));
v___x_3775_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v___x_3773_, v___x_3774_, v_strict_3753_);
v___x_3776_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__4));
v___x_3777_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v___x_3775_, v___x_3776_, v___x_3752_);
v___x_3778_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___closed__2));
v___x_3779_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v___x_3777_, v___x_3778_, v___x_3752_);
if (v_isShared_3770_ == 0)
{
lean_ctor_set(v___x_3769_, 1, v___x_3779_);
v___x_3781_ = v___x_3769_;
goto v_reusejp_3780_;
}
else
{
lean_object* v_reuseFailAlloc_3782_; 
v_reuseFailAlloc_3782_ = lean_alloc_ctor(0, 10, 3);
lean_ctor_set(v_reuseFailAlloc_3782_, 0, v_header_3755_);
lean_ctor_set(v_reuseFailAlloc_3782_, 1, v___x_3779_);
lean_ctor_set(v_reuseFailAlloc_3782_, 2, v_currNamespace_3757_);
lean_ctor_set(v_reuseFailAlloc_3782_, 3, v_openDecls_3758_);
lean_ctor_set(v_reuseFailAlloc_3782_, 4, v_levelNames_3759_);
lean_ctor_set(v_reuseFailAlloc_3782_, 5, v_varDecls_3760_);
lean_ctor_set(v_reuseFailAlloc_3782_, 6, v_varUIds_3761_);
lean_ctor_set(v_reuseFailAlloc_3782_, 7, v_includedVars_3762_);
lean_ctor_set(v_reuseFailAlloc_3782_, 8, v_omittedVars_3763_);
lean_ctor_set(v_reuseFailAlloc_3782_, 9, v_attrs_3767_);
lean_ctor_set_uint8(v_reuseFailAlloc_3782_, sizeof(void*)*10, v_isNoncomputable_3764_);
lean_ctor_set_uint8(v_reuseFailAlloc_3782_, sizeof(void*)*10 + 1, v_isPublic_3765_);
lean_ctor_set_uint8(v_reuseFailAlloc_3782_, sizeof(void*)*10 + 2, v_isMeta_3766_);
v___x_3781_ = v_reuseFailAlloc_3782_;
goto v_reusejp_3780_;
}
v_reusejp_3780_:
{
return v___x_3781_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___boxed(lean_object* v___x_3784_, lean_object* v_strict_3785_, lean_object* v_scope_3786_){
_start:
{
uint8_t v___x_8632__boxed_3787_; uint8_t v_strict_boxed_3788_; lean_object* v_res_3789_; 
v___x_8632__boxed_3787_ = lean_unbox(v___x_3784_);
v_strict_boxed_3788_ = lean_unbox(v_strict_3785_);
v_res_3789_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0(v___x_8632__boxed_3787_, v_strict_boxed_3788_, v_scope_3786_);
return v_res_3789_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__1(void){
_start:
{
lean_object* v___x_3791_; lean_object* v___x_3792_; 
v___x_3791_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__0));
v___x_3792_ = l_Lean_stringToMessageData(v___x_3791_);
return v___x_3792_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__2(void){
_start:
{
lean_object* v___x_3793_; lean_object* v___x_3794_; 
v___x_3793_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__1);
v___x_3794_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3794_, 0, v___x_3793_);
return v___x_3794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1(lean_object* v___x_3795_, lean_object* v___y_3796_, lean_object* v___y_3797_){
_start:
{
lean_object* v___x_3799_; 
v___x_3799_ = l_Lean_Elab_Command_elabCommand(v___x_3795_, v___y_3796_, v___y_3797_);
if (lean_obj_tag(v___x_3799_) == 0)
{
lean_object* v___x_3801_; uint8_t v_isShared_3802_; uint8_t v_isSharedCheck_3814_; 
v_isSharedCheck_3814_ = !lean_is_exclusive(v___x_3799_);
if (v_isSharedCheck_3814_ == 0)
{
lean_object* v_unused_3815_; 
v_unused_3815_ = lean_ctor_get(v___x_3799_, 0);
lean_dec(v_unused_3815_);
v___x_3801_ = v___x_3799_;
v_isShared_3802_ = v_isSharedCheck_3814_;
goto v_resetjp_3800_;
}
else
{
lean_dec(v___x_3799_);
v___x_3801_ = lean_box(0);
v_isShared_3802_ = v_isSharedCheck_3814_;
goto v_resetjp_3800_;
}
v_resetjp_3800_:
{
lean_object* v___x_3803_; lean_object* v_messages_3804_; uint8_t v___x_3805_; 
v___x_3803_ = lean_st_ref_get(v___y_3797_);
v_messages_3804_ = lean_ctor_get(v___x_3803_, 1);
lean_inc_ref(v_messages_3804_);
lean_dec(v___x_3803_);
v___x_3805_ = l_Lean_MessageLog_hasErrors(v_messages_3804_);
lean_dec_ref(v_messages_3804_);
if (v___x_3805_ == 0)
{
lean_object* v___x_3806_; lean_object* v___x_3808_; 
v___x_3806_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__5));
if (v_isShared_3802_ == 0)
{
lean_ctor_set(v___x_3801_, 0, v___x_3806_);
v___x_3808_ = v___x_3801_;
goto v_reusejp_3807_;
}
else
{
lean_object* v_reuseFailAlloc_3809_; 
v_reuseFailAlloc_3809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3809_, 0, v___x_3806_);
v___x_3808_ = v_reuseFailAlloc_3809_;
goto v_reusejp_3807_;
}
v_reusejp_3807_:
{
return v___x_3808_;
}
}
else
{
lean_object* v___x_3810_; lean_object* v___x_3812_; 
v___x_3810_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___closed__2);
if (v_isShared_3802_ == 0)
{
lean_ctor_set(v___x_3801_, 0, v___x_3810_);
v___x_3812_ = v___x_3801_;
goto v_reusejp_3811_;
}
else
{
lean_object* v_reuseFailAlloc_3813_; 
v_reuseFailAlloc_3813_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3813_, 0, v___x_3810_);
v___x_3812_ = v_reuseFailAlloc_3813_;
goto v_reusejp_3811_;
}
v_reusejp_3811_:
{
return v___x_3812_;
}
}
}
}
else
{
lean_object* v_a_3816_; lean_object* v___x_3818_; uint8_t v_isShared_3819_; uint8_t v_isSharedCheck_3823_; 
v_a_3816_ = lean_ctor_get(v___x_3799_, 0);
v_isSharedCheck_3823_ = !lean_is_exclusive(v___x_3799_);
if (v_isSharedCheck_3823_ == 0)
{
v___x_3818_ = v___x_3799_;
v_isShared_3819_ = v_isSharedCheck_3823_;
goto v_resetjp_3817_;
}
else
{
lean_inc(v_a_3816_);
lean_dec(v___x_3799_);
v___x_3818_ = lean_box(0);
v_isShared_3819_ = v_isSharedCheck_3823_;
goto v_resetjp_3817_;
}
v_resetjp_3817_:
{
lean_object* v___x_3821_; 
if (v_isShared_3819_ == 0)
{
v___x_3821_ = v___x_3818_;
goto v_reusejp_3820_;
}
else
{
lean_object* v_reuseFailAlloc_3822_; 
v_reuseFailAlloc_3822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3822_, 0, v_a_3816_);
v___x_3821_ = v_reuseFailAlloc_3822_;
goto v_reusejp_3820_;
}
v_reusejp_3820_:
{
return v___x_3821_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___boxed(lean_object* v___x_3824_, lean_object* v___y_3825_, lean_object* v___y_3826_, lean_object* v___y_3827_){
_start:
{
lean_object* v_res_3828_; 
v_res_3828_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1(v___x_3824_, v___y_3825_, v___y_3826_);
lean_dec(v___y_3826_);
lean_dec_ref(v___y_3825_);
return v_res_3828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__2(lean_object* v___f_3829_, lean_object* v_opts_3830_, lean_object* v___y_3831_, lean_object* v___y_3832_){
_start:
{
lean_object* v___x_3834_; lean_object* v_messages_3835_; lean_object* v___x_3836_; lean_object* v___x_3837_; lean_object* v_a_3839_; lean_object* v___y_3847_; lean_object* v___x_3857_; 
v___x_3834_ = lean_st_ref_get(v___y_3832_);
v_messages_3835_ = lean_ctor_get(v___x_3834_, 1);
lean_inc_ref(v_messages_3835_);
lean_dec(v___x_3834_);
v___x_3836_ = l_Lean_MessageLog_toList(v_messages_3835_);
lean_dec_ref(v_messages_3835_);
v___x_3837_ = l_List_lengthTR___redArg(v___x_3836_);
lean_dec(v___x_3836_);
v___x_3857_ = l_Lean_Elab_Command_withScope___redArg(v_opts_3830_, v___f_3829_, v___y_3831_, v___y_3832_);
if (lean_obj_tag(v___x_3857_) == 0)
{
v___y_3847_ = v___x_3857_;
goto v___jp_3846_;
}
else
{
lean_object* v_a_3858_; uint8_t v___x_3859_; 
v_a_3858_ = lean_ctor_get(v___x_3857_, 0);
lean_inc(v_a_3858_);
v___x_3859_ = l_Lean_Exception_isInterrupt(v_a_3858_);
if (v___x_3859_ == 0)
{
if (lean_obj_tag(v_a_3858_) == 1)
{
lean_dec_ref_known(v_a_3858_, 2);
v___y_3847_ = v___x_3857_;
goto v___jp_3846_;
}
else
{
lean_object* v___x_3861_; uint8_t v_isShared_3862_; uint8_t v_isSharedCheck_3867_; 
v_isSharedCheck_3867_ = !lean_is_exclusive(v___x_3857_);
if (v_isSharedCheck_3867_ == 0)
{
lean_object* v_unused_3868_; 
v_unused_3868_ = lean_ctor_get(v___x_3857_, 0);
lean_dec(v_unused_3868_);
v___x_3861_ = v___x_3857_;
v_isShared_3862_ = v_isSharedCheck_3867_;
goto v_resetjp_3860_;
}
else
{
lean_dec(v___x_3857_);
v___x_3861_ = lean_box(0);
v_isShared_3862_ = v_isSharedCheck_3867_;
goto v_resetjp_3860_;
}
v_resetjp_3860_:
{
lean_object* v___x_3863_; lean_object* v___x_3865_; 
v___x_3863_ = l_Lean_Exception_toMessageData(v_a_3858_);
if (v_isShared_3862_ == 0)
{
lean_ctor_set_tag(v___x_3861_, 0);
lean_ctor_set(v___x_3861_, 0, v___x_3863_);
v___x_3865_ = v___x_3861_;
goto v_reusejp_3864_;
}
else
{
lean_object* v_reuseFailAlloc_3866_; 
v_reuseFailAlloc_3866_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3866_, 0, v___x_3863_);
v___x_3865_ = v_reuseFailAlloc_3866_;
goto v_reusejp_3864_;
}
v_reusejp_3864_:
{
v_a_3839_ = v___x_3865_;
goto v___jp_3838_;
}
}
}
}
else
{
lean_dec(v_a_3858_);
v___y_3847_ = v___x_3857_;
goto v___jp_3846_;
}
}
v___jp_3838_:
{
lean_object* v___x_3840_; lean_object* v_messages_3841_; lean_object* v___x_3842_; lean_object* v___x_3843_; lean_object* v___x_3844_; lean_object* v___x_3845_; 
v___x_3840_ = lean_st_ref_get(v___y_3832_);
v_messages_3841_ = lean_ctor_get(v___x_3840_, 1);
lean_inc_ref(v_messages_3841_);
lean_dec(v___x_3840_);
v___x_3842_ = l_Lean_MessageLog_toList(v_messages_3841_);
lean_dec_ref(v_messages_3841_);
v___x_3843_ = l_List_drop___redArg(v___x_3837_, v___x_3842_);
lean_dec(v___x_3842_);
v___x_3844_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3844_, 0, v_a_3839_);
lean_ctor_set(v___x_3844_, 1, v___x_3843_);
v___x_3845_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3845_, 0, v___x_3844_);
return v___x_3845_;
}
v___jp_3846_:
{
if (lean_obj_tag(v___y_3847_) == 0)
{
lean_object* v_a_3848_; 
v_a_3848_ = lean_ctor_get(v___y_3847_, 0);
lean_inc(v_a_3848_);
lean_dec_ref_known(v___y_3847_, 1);
v_a_3839_ = v_a_3848_;
goto v___jp_3838_;
}
else
{
lean_object* v_a_3849_; lean_object* v___x_3851_; uint8_t v_isShared_3852_; uint8_t v_isSharedCheck_3856_; 
lean_dec(v___x_3837_);
v_a_3849_ = lean_ctor_get(v___y_3847_, 0);
v_isSharedCheck_3856_ = !lean_is_exclusive(v___y_3847_);
if (v_isSharedCheck_3856_ == 0)
{
v___x_3851_ = v___y_3847_;
v_isShared_3852_ = v_isSharedCheck_3856_;
goto v_resetjp_3850_;
}
else
{
lean_inc(v_a_3849_);
lean_dec(v___y_3847_);
v___x_3851_ = lean_box(0);
v_isShared_3852_ = v_isSharedCheck_3856_;
goto v_resetjp_3850_;
}
v_resetjp_3850_:
{
lean_object* v___x_3854_; 
if (v_isShared_3852_ == 0)
{
v___x_3854_ = v___x_3851_;
goto v_reusejp_3853_;
}
else
{
lean_object* v_reuseFailAlloc_3855_; 
v_reuseFailAlloc_3855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3855_, 0, v_a_3849_);
v___x_3854_ = v_reuseFailAlloc_3855_;
goto v_reusejp_3853_;
}
v_reusejp_3853_:
{
return v___x_3854_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__2___boxed(lean_object* v___f_3869_, lean_object* v_opts_3870_, lean_object* v___y_3871_, lean_object* v___y_3872_, lean_object* v___y_3873_){
_start:
{
lean_object* v_res_3874_; 
v_res_3874_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__2(v___f_3869_, v_opts_3870_, v___y_3871_, v___y_3872_);
lean_dec(v___y_3872_);
lean_dec_ref(v___y_3871_);
return v_res_3874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__3(uint8_t v___x_3875_, lean_object* v_scope_3876_){
_start:
{
lean_object* v_header_3877_; lean_object* v_opts_3878_; lean_object* v_currNamespace_3879_; lean_object* v_openDecls_3880_; lean_object* v_levelNames_3881_; lean_object* v_varDecls_3882_; lean_object* v_varUIds_3883_; lean_object* v_includedVars_3884_; lean_object* v_omittedVars_3885_; uint8_t v_isNoncomputable_3886_; uint8_t v_isPublic_3887_; uint8_t v_isMeta_3888_; lean_object* v_attrs_3889_; lean_object* v___x_3891_; uint8_t v_isShared_3892_; uint8_t v_isSharedCheck_3898_; 
v_header_3877_ = lean_ctor_get(v_scope_3876_, 0);
v_opts_3878_ = lean_ctor_get(v_scope_3876_, 1);
v_currNamespace_3879_ = lean_ctor_get(v_scope_3876_, 2);
v_openDecls_3880_ = lean_ctor_get(v_scope_3876_, 3);
v_levelNames_3881_ = lean_ctor_get(v_scope_3876_, 4);
v_varDecls_3882_ = lean_ctor_get(v_scope_3876_, 5);
v_varUIds_3883_ = lean_ctor_get(v_scope_3876_, 6);
v_includedVars_3884_ = lean_ctor_get(v_scope_3876_, 7);
v_omittedVars_3885_ = lean_ctor_get(v_scope_3876_, 8);
v_isNoncomputable_3886_ = lean_ctor_get_uint8(v_scope_3876_, sizeof(void*)*10);
v_isPublic_3887_ = lean_ctor_get_uint8(v_scope_3876_, sizeof(void*)*10 + 1);
v_isMeta_3888_ = lean_ctor_get_uint8(v_scope_3876_, sizeof(void*)*10 + 2);
v_attrs_3889_ = lean_ctor_get(v_scope_3876_, 9);
v_isSharedCheck_3898_ = !lean_is_exclusive(v_scope_3876_);
if (v_isSharedCheck_3898_ == 0)
{
v___x_3891_ = v_scope_3876_;
v_isShared_3892_ = v_isSharedCheck_3898_;
goto v_resetjp_3890_;
}
else
{
lean_inc(v_attrs_3889_);
lean_inc(v_omittedVars_3885_);
lean_inc(v_includedVars_3884_);
lean_inc(v_varUIds_3883_);
lean_inc(v_varDecls_3882_);
lean_inc(v_levelNames_3881_);
lean_inc(v_openDecls_3880_);
lean_inc(v_currNamespace_3879_);
lean_inc(v_opts_3878_);
lean_inc(v_header_3877_);
lean_dec(v_scope_3876_);
v___x_3891_ = lean_box(0);
v_isShared_3892_ = v_isSharedCheck_3898_;
goto v_resetjp_3890_;
}
v_resetjp_3890_:
{
lean_object* v___x_3893_; lean_object* v___x_3894_; lean_object* v___x_3896_; 
v___x_3893_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1___lam__0___closed__3));
v___x_3894_ = lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0(v_opts_3878_, v___x_3893_, v___x_3875_);
if (v_isShared_3892_ == 0)
{
lean_ctor_set(v___x_3891_, 1, v___x_3894_);
v___x_3896_ = v___x_3891_;
goto v_reusejp_3895_;
}
else
{
lean_object* v_reuseFailAlloc_3897_; 
v_reuseFailAlloc_3897_ = lean_alloc_ctor(0, 10, 3);
lean_ctor_set(v_reuseFailAlloc_3897_, 0, v_header_3877_);
lean_ctor_set(v_reuseFailAlloc_3897_, 1, v___x_3894_);
lean_ctor_set(v_reuseFailAlloc_3897_, 2, v_currNamespace_3879_);
lean_ctor_set(v_reuseFailAlloc_3897_, 3, v_openDecls_3880_);
lean_ctor_set(v_reuseFailAlloc_3897_, 4, v_levelNames_3881_);
lean_ctor_set(v_reuseFailAlloc_3897_, 5, v_varDecls_3882_);
lean_ctor_set(v_reuseFailAlloc_3897_, 6, v_varUIds_3883_);
lean_ctor_set(v_reuseFailAlloc_3897_, 7, v_includedVars_3884_);
lean_ctor_set(v_reuseFailAlloc_3897_, 8, v_omittedVars_3885_);
lean_ctor_set(v_reuseFailAlloc_3897_, 9, v_attrs_3889_);
lean_ctor_set_uint8(v_reuseFailAlloc_3897_, sizeof(void*)*10, v_isNoncomputable_3886_);
lean_ctor_set_uint8(v_reuseFailAlloc_3897_, sizeof(void*)*10 + 1, v_isPublic_3887_);
lean_ctor_set_uint8(v_reuseFailAlloc_3897_, sizeof(void*)*10 + 2, v_isMeta_3888_);
v___x_3896_ = v_reuseFailAlloc_3897_;
goto v_reusejp_3895_;
}
v_reusejp_3895_:
{
return v___x_3896_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__3___boxed(lean_object* v___x_3899_, lean_object* v_scope_3900_){
_start:
{
uint8_t v___x_8841__boxed_3901_; lean_object* v_res_3902_; 
v___x_8841__boxed_3901_ = lean_unbox(v___x_3899_);
v_res_3902_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__3(v___x_8841__boxed_3901_, v_scope_3900_);
return v_res_3902_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_3903_; 
v___x_3903_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3903_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1(void){
_start:
{
lean_object* v___x_3904_; lean_object* v___x_3905_; 
v___x_3904_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__0);
v___x_3905_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3905_, 0, v___x_3904_);
return v___x_3905_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__2(void){
_start:
{
lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; 
v___x_3906_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1);
v___x_3907_ = lean_unsigned_to_nat(0u);
v___x_3908_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3908_, 0, v___x_3907_);
lean_ctor_set(v___x_3908_, 1, v___x_3907_);
lean_ctor_set(v___x_3908_, 2, v___x_3907_);
lean_ctor_set(v___x_3908_, 3, v___x_3907_);
lean_ctor_set(v___x_3908_, 4, v___x_3906_);
lean_ctor_set(v___x_3908_, 5, v___x_3906_);
lean_ctor_set(v___x_3908_, 6, v___x_3906_);
lean_ctor_set(v___x_3908_, 7, v___x_3906_);
lean_ctor_set(v___x_3908_, 8, v___x_3906_);
lean_ctor_set(v___x_3908_, 9, v___x_3906_);
return v___x_3908_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__3(void){
_start:
{
lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; 
v___x_3909_ = lean_unsigned_to_nat(32u);
v___x_3910_ = lean_mk_empty_array_with_capacity(v___x_3909_);
v___x_3911_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3911_, 0, v___x_3910_);
return v___x_3911_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__4(void){
_start:
{
size_t v___x_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; lean_object* v___x_3916_; lean_object* v___x_3917_; 
v___x_3912_ = ((size_t)5ULL);
v___x_3913_ = lean_unsigned_to_nat(0u);
v___x_3914_ = lean_unsigned_to_nat(32u);
v___x_3915_ = lean_mk_empty_array_with_capacity(v___x_3914_);
v___x_3916_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__3);
v___x_3917_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3917_, 0, v___x_3916_);
lean_ctor_set(v___x_3917_, 1, v___x_3915_);
lean_ctor_set(v___x_3917_, 2, v___x_3913_);
lean_ctor_set(v___x_3917_, 3, v___x_3913_);
lean_ctor_set_usize(v___x_3917_, 4, v___x_3912_);
return v___x_3917_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__5(void){
_start:
{
lean_object* v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; 
v___x_3918_ = lean_box(1);
v___x_3919_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__4);
v___x_3920_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__1);
v___x_3921_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3921_, 0, v___x_3920_);
lean_ctor_set(v___x_3921_, 1, v___x_3919_);
lean_ctor_set(v___x_3921_, 2, v___x_3918_);
return v___x_3921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg(lean_object* v_msgData_3922_, lean_object* v___y_3923_){
_start:
{
lean_object* v___x_3925_; lean_object* v_env_3926_; lean_object* v___x_3927_; lean_object* v_scopes_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; lean_object* v_opts_3931_; lean_object* v___x_3932_; lean_object* v___x_3933_; lean_object* v___x_3934_; lean_object* v___x_3935_; lean_object* v___x_3936_; 
v___x_3925_ = lean_st_ref_get(v___y_3923_);
v_env_3926_ = lean_ctor_get(v___x_3925_, 0);
lean_inc_ref(v_env_3926_);
lean_dec(v___x_3925_);
v___x_3927_ = lean_st_ref_get(v___y_3923_);
v_scopes_3928_ = lean_ctor_get(v___x_3927_, 2);
lean_inc(v_scopes_3928_);
lean_dec(v___x_3927_);
v___x_3929_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3930_ = l_List_head_x21___redArg(v___x_3929_, v_scopes_3928_);
lean_dec(v_scopes_3928_);
v_opts_3931_ = lean_ctor_get(v___x_3930_, 1);
lean_inc_ref(v_opts_3931_);
lean_dec(v___x_3930_);
v___x_3932_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__2);
v___x_3933_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___closed__5);
v___x_3934_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3934_, 0, v_env_3926_);
lean_ctor_set(v___x_3934_, 1, v___x_3932_);
lean_ctor_set(v___x_3934_, 2, v___x_3933_);
lean_ctor_set(v___x_3934_, 3, v_opts_3931_);
v___x_3935_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_3935_, 0, v___x_3934_);
lean_ctor_set(v___x_3935_, 1, v_msgData_3922_);
v___x_3936_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3936_, 0, v___x_3935_);
return v___x_3936_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg___boxed(lean_object* v_msgData_3937_, lean_object* v___y_3938_, lean_object* v___y_3939_){
_start:
{
lean_object* v_res_3940_; 
v_res_3940_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg(v_msgData_3937_, v___y_3938_);
lean_dec(v___y_3938_);
return v_res_3940_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___lam__0(uint8_t v___y_3941_, uint8_t v_suppressElabErrors_3942_, lean_object* v_x_3943_){
_start:
{
if (lean_obj_tag(v_x_3943_) == 1)
{
lean_object* v_pre_3944_; 
v_pre_3944_ = lean_ctor_get(v_x_3943_, 0);
if (lean_obj_tag(v_pre_3944_) == 0)
{
lean_object* v_str_3945_; lean_object* v___x_3946_; uint8_t v___x_3947_; 
v_str_3945_ = lean_ctor_get(v_x_3943_, 1);
v___x_3946_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_ppEscalations_spec__0___closed__0));
v___x_3947_ = lean_string_dec_eq(v_str_3945_, v___x_3946_);
if (v___x_3947_ == 0)
{
return v___y_3941_;
}
else
{
return v_suppressElabErrors_3942_;
}
}
else
{
return v___y_3941_;
}
}
else
{
return v___y_3941_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___lam__0___boxed(lean_object* v___y_3948_, lean_object* v_suppressElabErrors_3949_, lean_object* v_x_3950_){
_start:
{
uint8_t v___y_8965__boxed_3951_; uint8_t v_suppressElabErrors_boxed_3952_; uint8_t v_res_3953_; lean_object* v_r_3954_; 
v___y_8965__boxed_3951_ = lean_unbox(v___y_3948_);
v_suppressElabErrors_boxed_3952_ = lean_unbox(v_suppressElabErrors_3949_);
v_res_3953_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___lam__0(v___y_8965__boxed_3951_, v_suppressElabErrors_boxed_3952_, v_x_3950_);
lean_dec(v_x_3950_);
v_r_3954_ = lean_box(v_res_3953_);
return v_r_3954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2(lean_object* v_ref_3955_, lean_object* v_msgData_3956_, uint8_t v_severity_3957_, uint8_t v_isSilent_3958_, lean_object* v___y_3959_, lean_object* v___y_3960_){
_start:
{
lean_object* v___y_3963_; uint8_t v___y_3964_; lean_object* v___y_3965_; lean_object* v___y_3966_; uint8_t v___y_3967_; lean_object* v___y_3968_; lean_object* v___y_3969_; lean_object* v___y_3970_; uint8_t v___y_4027_; uint8_t v___y_4028_; uint8_t v___y_4029_; lean_object* v___y_4030_; lean_object* v___y_4031_; uint8_t v___y_4055_; lean_object* v___y_4056_; uint8_t v___y_4057_; uint8_t v___y_4058_; lean_object* v___y_4059_; uint8_t v___y_4063_; uint8_t v___y_4064_; uint8_t v___y_4065_; uint8_t v___x_4080_; uint8_t v___y_4082_; uint8_t v___y_4083_; uint8_t v___y_4084_; uint8_t v___y_4086_; uint8_t v___x_4098_; 
v___x_4080_ = 2;
v___x_4098_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3957_, v___x_4080_);
if (v___x_4098_ == 0)
{
v___y_4086_ = v___x_4098_;
goto v___jp_4085_;
}
else
{
uint8_t v___x_4099_; 
lean_inc_ref(v_msgData_3956_);
v___x_4099_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_3956_);
v___y_4086_ = v___x_4099_;
goto v___jp_4085_;
}
v___jp_3962_:
{
lean_object* v___x_3971_; 
v___x_3971_ = l_Lean_Elab_Command_getScope___redArg(v___y_3970_);
if (lean_obj_tag(v___x_3971_) == 0)
{
lean_object* v_a_3972_; lean_object* v___x_3973_; 
v_a_3972_ = lean_ctor_get(v___x_3971_, 0);
lean_inc(v_a_3972_);
lean_dec_ref_known(v___x_3971_, 1);
v___x_3973_ = l_Lean_Elab_Command_getScope___redArg(v___y_3970_);
if (lean_obj_tag(v___x_3973_) == 0)
{
lean_object* v_a_3974_; lean_object* v___x_3976_; uint8_t v_isShared_3977_; uint8_t v_isSharedCheck_4009_; 
v_a_3974_ = lean_ctor_get(v___x_3973_, 0);
v_isSharedCheck_4009_ = !lean_is_exclusive(v___x_3973_);
if (v_isSharedCheck_4009_ == 0)
{
v___x_3976_ = v___x_3973_;
v_isShared_3977_ = v_isSharedCheck_4009_;
goto v_resetjp_3975_;
}
else
{
lean_inc(v_a_3974_);
lean_dec(v___x_3973_);
v___x_3976_ = lean_box(0);
v_isShared_3977_ = v_isSharedCheck_4009_;
goto v_resetjp_3975_;
}
v_resetjp_3975_:
{
lean_object* v___x_3978_; lean_object* v_currNamespace_3979_; lean_object* v_openDecls_3980_; lean_object* v_env_3981_; lean_object* v_messages_3982_; lean_object* v_scopes_3983_; lean_object* v_usedQuotCtxts_3984_; lean_object* v_nextMacroScope_3985_; lean_object* v_maxRecDepth_3986_; lean_object* v_ngen_3987_; lean_object* v_auxDeclNGen_3988_; lean_object* v_infoState_3989_; lean_object* v_traceState_3990_; lean_object* v_snapshotTasks_3991_; lean_object* v_prevLinterStates_3992_; lean_object* v___x_3994_; uint8_t v_isShared_3995_; uint8_t v_isSharedCheck_4008_; 
v___x_3978_ = lean_st_ref_take(v___y_3970_);
v_currNamespace_3979_ = lean_ctor_get(v_a_3972_, 2);
lean_inc(v_currNamespace_3979_);
lean_dec(v_a_3972_);
v_openDecls_3980_ = lean_ctor_get(v_a_3974_, 3);
lean_inc(v_openDecls_3980_);
lean_dec(v_a_3974_);
v_env_3981_ = lean_ctor_get(v___x_3978_, 0);
v_messages_3982_ = lean_ctor_get(v___x_3978_, 1);
v_scopes_3983_ = lean_ctor_get(v___x_3978_, 2);
v_usedQuotCtxts_3984_ = lean_ctor_get(v___x_3978_, 3);
v_nextMacroScope_3985_ = lean_ctor_get(v___x_3978_, 4);
v_maxRecDepth_3986_ = lean_ctor_get(v___x_3978_, 5);
v_ngen_3987_ = lean_ctor_get(v___x_3978_, 6);
v_auxDeclNGen_3988_ = lean_ctor_get(v___x_3978_, 7);
v_infoState_3989_ = lean_ctor_get(v___x_3978_, 8);
v_traceState_3990_ = lean_ctor_get(v___x_3978_, 9);
v_snapshotTasks_3991_ = lean_ctor_get(v___x_3978_, 10);
v_prevLinterStates_3992_ = lean_ctor_get(v___x_3978_, 11);
v_isSharedCheck_4008_ = !lean_is_exclusive(v___x_3978_);
if (v_isSharedCheck_4008_ == 0)
{
v___x_3994_ = v___x_3978_;
v_isShared_3995_ = v_isSharedCheck_4008_;
goto v_resetjp_3993_;
}
else
{
lean_inc(v_prevLinterStates_3992_);
lean_inc(v_snapshotTasks_3991_);
lean_inc(v_traceState_3990_);
lean_inc(v_infoState_3989_);
lean_inc(v_auxDeclNGen_3988_);
lean_inc(v_ngen_3987_);
lean_inc(v_maxRecDepth_3986_);
lean_inc(v_nextMacroScope_3985_);
lean_inc(v_usedQuotCtxts_3984_);
lean_inc(v_scopes_3983_);
lean_inc(v_messages_3982_);
lean_inc(v_env_3981_);
lean_dec(v___x_3978_);
v___x_3994_ = lean_box(0);
v_isShared_3995_ = v_isSharedCheck_4008_;
goto v_resetjp_3993_;
}
v_resetjp_3993_:
{
lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; lean_object* v___x_4001_; 
v___x_3996_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3996_, 0, v_currNamespace_3979_);
lean_ctor_set(v___x_3996_, 1, v_openDecls_3980_);
v___x_3997_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_3997_, 0, v___x_3996_);
lean_ctor_set(v___x_3997_, 1, v___y_3966_);
lean_inc_ref(v___y_3969_);
lean_inc_ref(v___y_3963_);
v___x_3998_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_3998_, 0, v___y_3963_);
lean_ctor_set(v___x_3998_, 1, v___y_3968_);
lean_ctor_set(v___x_3998_, 2, v___y_3965_);
lean_ctor_set(v___x_3998_, 3, v___y_3969_);
lean_ctor_set(v___x_3998_, 4, v___x_3997_);
lean_ctor_set_uint8(v___x_3998_, sizeof(void*)*5, v___y_3967_);
lean_ctor_set_uint8(v___x_3998_, sizeof(void*)*5 + 1, v___y_3964_);
lean_ctor_set_uint8(v___x_3998_, sizeof(void*)*5 + 2, v_isSilent_3958_);
v___x_3999_ = l_Lean_MessageLog_add(v___x_3998_, v_messages_3982_);
if (v_isShared_3995_ == 0)
{
lean_ctor_set(v___x_3994_, 1, v___x_3999_);
v___x_4001_ = v___x_3994_;
goto v_reusejp_4000_;
}
else
{
lean_object* v_reuseFailAlloc_4007_; 
v_reuseFailAlloc_4007_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_4007_, 0, v_env_3981_);
lean_ctor_set(v_reuseFailAlloc_4007_, 1, v___x_3999_);
lean_ctor_set(v_reuseFailAlloc_4007_, 2, v_scopes_3983_);
lean_ctor_set(v_reuseFailAlloc_4007_, 3, v_usedQuotCtxts_3984_);
lean_ctor_set(v_reuseFailAlloc_4007_, 4, v_nextMacroScope_3985_);
lean_ctor_set(v_reuseFailAlloc_4007_, 5, v_maxRecDepth_3986_);
lean_ctor_set(v_reuseFailAlloc_4007_, 6, v_ngen_3987_);
lean_ctor_set(v_reuseFailAlloc_4007_, 7, v_auxDeclNGen_3988_);
lean_ctor_set(v_reuseFailAlloc_4007_, 8, v_infoState_3989_);
lean_ctor_set(v_reuseFailAlloc_4007_, 9, v_traceState_3990_);
lean_ctor_set(v_reuseFailAlloc_4007_, 10, v_snapshotTasks_3991_);
lean_ctor_set(v_reuseFailAlloc_4007_, 11, v_prevLinterStates_3992_);
v___x_4001_ = v_reuseFailAlloc_4007_;
goto v_reusejp_4000_;
}
v_reusejp_4000_:
{
lean_object* v___x_4002_; lean_object* v___x_4003_; lean_object* v___x_4005_; 
v___x_4002_ = lean_st_ref_set(v___y_3970_, v___x_4001_);
v___x_4003_ = lean_box(0);
if (v_isShared_3977_ == 0)
{
lean_ctor_set(v___x_3976_, 0, v___x_4003_);
v___x_4005_ = v___x_3976_;
goto v_reusejp_4004_;
}
else
{
lean_object* v_reuseFailAlloc_4006_; 
v_reuseFailAlloc_4006_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4006_, 0, v___x_4003_);
v___x_4005_ = v_reuseFailAlloc_4006_;
goto v_reusejp_4004_;
}
v_reusejp_4004_:
{
return v___x_4005_;
}
}
}
}
}
else
{
lean_object* v_a_4010_; lean_object* v___x_4012_; uint8_t v_isShared_4013_; uint8_t v_isSharedCheck_4017_; 
lean_dec(v_a_3972_);
lean_dec_ref(v___y_3968_);
lean_dec_ref(v___y_3966_);
lean_dec(v___y_3965_);
v_a_4010_ = lean_ctor_get(v___x_3973_, 0);
v_isSharedCheck_4017_ = !lean_is_exclusive(v___x_3973_);
if (v_isSharedCheck_4017_ == 0)
{
v___x_4012_ = v___x_3973_;
v_isShared_4013_ = v_isSharedCheck_4017_;
goto v_resetjp_4011_;
}
else
{
lean_inc(v_a_4010_);
lean_dec(v___x_3973_);
v___x_4012_ = lean_box(0);
v_isShared_4013_ = v_isSharedCheck_4017_;
goto v_resetjp_4011_;
}
v_resetjp_4011_:
{
lean_object* v___x_4015_; 
if (v_isShared_4013_ == 0)
{
v___x_4015_ = v___x_4012_;
goto v_reusejp_4014_;
}
else
{
lean_object* v_reuseFailAlloc_4016_; 
v_reuseFailAlloc_4016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4016_, 0, v_a_4010_);
v___x_4015_ = v_reuseFailAlloc_4016_;
goto v_reusejp_4014_;
}
v_reusejp_4014_:
{
return v___x_4015_;
}
}
}
}
else
{
lean_object* v_a_4018_; lean_object* v___x_4020_; uint8_t v_isShared_4021_; uint8_t v_isSharedCheck_4025_; 
lean_dec_ref(v___y_3968_);
lean_dec_ref(v___y_3966_);
lean_dec(v___y_3965_);
v_a_4018_ = lean_ctor_get(v___x_3971_, 0);
v_isSharedCheck_4025_ = !lean_is_exclusive(v___x_3971_);
if (v_isSharedCheck_4025_ == 0)
{
v___x_4020_ = v___x_3971_;
v_isShared_4021_ = v_isSharedCheck_4025_;
goto v_resetjp_4019_;
}
else
{
lean_inc(v_a_4018_);
lean_dec(v___x_3971_);
v___x_4020_ = lean_box(0);
v_isShared_4021_ = v_isSharedCheck_4025_;
goto v_resetjp_4019_;
}
v_resetjp_4019_:
{
lean_object* v___x_4023_; 
if (v_isShared_4021_ == 0)
{
v___x_4023_ = v___x_4020_;
goto v_reusejp_4022_;
}
else
{
lean_object* v_reuseFailAlloc_4024_; 
v_reuseFailAlloc_4024_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4024_, 0, v_a_4018_);
v___x_4023_ = v_reuseFailAlloc_4024_;
goto v_reusejp_4022_;
}
v_reusejp_4022_:
{
return v___x_4023_;
}
}
}
}
v___jp_4026_:
{
lean_object* v_fileName_4032_; lean_object* v_fileMap_4033_; uint8_t v_suppressElabErrors_4034_; lean_object* v___x_4035_; lean_object* v___x_4036_; lean_object* v_a_4037_; lean_object* v___x_4039_; uint8_t v_isShared_4040_; uint8_t v_isSharedCheck_4053_; 
v_fileName_4032_ = lean_ctor_get(v___y_3959_, 0);
v_fileMap_4033_ = lean_ctor_get(v___y_3959_, 1);
v_suppressElabErrors_4034_ = lean_ctor_get_uint8(v___y_3959_, sizeof(void*)*10);
v___x_4035_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_3956_);
v___x_4036_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg(v___x_4035_, v___y_3960_);
v_a_4037_ = lean_ctor_get(v___x_4036_, 0);
v_isSharedCheck_4053_ = !lean_is_exclusive(v___x_4036_);
if (v_isSharedCheck_4053_ == 0)
{
v___x_4039_ = v___x_4036_;
v_isShared_4040_ = v_isSharedCheck_4053_;
goto v_resetjp_4038_;
}
else
{
lean_inc(v_a_4037_);
lean_dec(v___x_4036_);
v___x_4039_ = lean_box(0);
v_isShared_4040_ = v_isSharedCheck_4053_;
goto v_resetjp_4038_;
}
v_resetjp_4038_:
{
lean_object* v___x_4041_; lean_object* v___x_4042_; lean_object* v___x_4043_; lean_object* v___x_4044_; 
lean_inc_ref_n(v_fileMap_4033_, 2);
v___x_4041_ = l_Lean_FileMap_toPosition(v_fileMap_4033_, v___y_4030_);
lean_dec(v___y_4030_);
v___x_4042_ = l_Lean_FileMap_toPosition(v_fileMap_4033_, v___y_4031_);
lean_dec(v___y_4031_);
v___x_4043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4043_, 0, v___x_4042_);
v___x_4044_ = ((lean_object*)(lp_mathlib_List_filterTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_isIdenticalSidesStr_spec__3___closed__0));
if (v_suppressElabErrors_4034_ == 0)
{
lean_del_object(v___x_4039_);
v___y_3963_ = v_fileName_4032_;
v___y_3964_ = v___y_4028_;
v___y_3965_ = v___x_4043_;
v___y_3966_ = v_a_4037_;
v___y_3967_ = v___y_4029_;
v___y_3968_ = v___x_4041_;
v___y_3969_ = v___x_4044_;
v___y_3970_ = v___y_3960_;
goto v___jp_3962_;
}
else
{
lean_object* v___x_4045_; lean_object* v___x_4046_; lean_object* v___f_4047_; uint8_t v___x_4048_; 
v___x_4045_ = lean_box(v___y_4027_);
v___x_4046_ = lean_box(v_suppressElabErrors_4034_);
v___f_4047_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___lam__0___boxed), 3, 2);
lean_closure_set(v___f_4047_, 0, v___x_4045_);
lean_closure_set(v___f_4047_, 1, v___x_4046_);
lean_inc(v_a_4037_);
v___x_4048_ = l_Lean_MessageData_hasTag(v___f_4047_, v_a_4037_);
if (v___x_4048_ == 0)
{
lean_object* v___x_4049_; lean_object* v___x_4051_; 
lean_dec_ref_known(v___x_4043_, 1);
lean_dec_ref(v___x_4041_);
lean_dec(v_a_4037_);
v___x_4049_ = lean_box(0);
if (v_isShared_4040_ == 0)
{
lean_ctor_set(v___x_4039_, 0, v___x_4049_);
v___x_4051_ = v___x_4039_;
goto v_reusejp_4050_;
}
else
{
lean_object* v_reuseFailAlloc_4052_; 
v_reuseFailAlloc_4052_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4052_, 0, v___x_4049_);
v___x_4051_ = v_reuseFailAlloc_4052_;
goto v_reusejp_4050_;
}
v_reusejp_4050_:
{
return v___x_4051_;
}
}
else
{
lean_del_object(v___x_4039_);
v___y_3963_ = v_fileName_4032_;
v___y_3964_ = v___y_4028_;
v___y_3965_ = v___x_4043_;
v___y_3966_ = v_a_4037_;
v___y_3967_ = v___y_4029_;
v___y_3968_ = v___x_4041_;
v___y_3969_ = v___x_4044_;
v___y_3970_ = v___y_3960_;
goto v___jp_3962_;
}
}
}
}
v___jp_4054_:
{
lean_object* v___x_4060_; 
v___x_4060_ = l_Lean_Syntax_getTailPos_x3f(v___y_4056_, v___y_4058_);
lean_dec(v___y_4056_);
if (lean_obj_tag(v___x_4060_) == 0)
{
lean_inc(v___y_4059_);
v___y_4027_ = v___y_4055_;
v___y_4028_ = v___y_4057_;
v___y_4029_ = v___y_4058_;
v___y_4030_ = v___y_4059_;
v___y_4031_ = v___y_4059_;
goto v___jp_4026_;
}
else
{
lean_object* v_val_4061_; 
v_val_4061_ = lean_ctor_get(v___x_4060_, 0);
lean_inc(v_val_4061_);
lean_dec_ref_known(v___x_4060_, 1);
v___y_4027_ = v___y_4055_;
v___y_4028_ = v___y_4057_;
v___y_4029_ = v___y_4058_;
v___y_4030_ = v___y_4059_;
v___y_4031_ = v_val_4061_;
goto v___jp_4026_;
}
}
v___jp_4062_:
{
lean_object* v___x_4066_; 
v___x_4066_ = l_Lean_Elab_Command_getRef___redArg(v___y_3959_);
if (lean_obj_tag(v___x_4066_) == 0)
{
lean_object* v_a_4067_; lean_object* v_ref_4068_; lean_object* v___x_4069_; 
v_a_4067_ = lean_ctor_get(v___x_4066_, 0);
lean_inc(v_a_4067_);
lean_dec_ref_known(v___x_4066_, 1);
v_ref_4068_ = l_Lean_replaceRef(v_ref_3955_, v_a_4067_);
lean_dec(v_a_4067_);
v___x_4069_ = l_Lean_Syntax_getPos_x3f(v_ref_4068_, v___y_4064_);
if (lean_obj_tag(v___x_4069_) == 0)
{
lean_object* v___x_4070_; 
v___x_4070_ = lean_unsigned_to_nat(0u);
v___y_4055_ = v___y_4063_;
v___y_4056_ = v_ref_4068_;
v___y_4057_ = v___y_4065_;
v___y_4058_ = v___y_4064_;
v___y_4059_ = v___x_4070_;
goto v___jp_4054_;
}
else
{
lean_object* v_val_4071_; 
v_val_4071_ = lean_ctor_get(v___x_4069_, 0);
lean_inc(v_val_4071_);
lean_dec_ref_known(v___x_4069_, 1);
v___y_4055_ = v___y_4063_;
v___y_4056_ = v_ref_4068_;
v___y_4057_ = v___y_4065_;
v___y_4058_ = v___y_4064_;
v___y_4059_ = v_val_4071_;
goto v___jp_4054_;
}
}
else
{
lean_object* v_a_4072_; lean_object* v___x_4074_; uint8_t v_isShared_4075_; uint8_t v_isSharedCheck_4079_; 
lean_dec_ref(v_msgData_3956_);
v_a_4072_ = lean_ctor_get(v___x_4066_, 0);
v_isSharedCheck_4079_ = !lean_is_exclusive(v___x_4066_);
if (v_isSharedCheck_4079_ == 0)
{
v___x_4074_ = v___x_4066_;
v_isShared_4075_ = v_isSharedCheck_4079_;
goto v_resetjp_4073_;
}
else
{
lean_inc(v_a_4072_);
lean_dec(v___x_4066_);
v___x_4074_ = lean_box(0);
v_isShared_4075_ = v_isSharedCheck_4079_;
goto v_resetjp_4073_;
}
v_resetjp_4073_:
{
lean_object* v___x_4077_; 
if (v_isShared_4075_ == 0)
{
v___x_4077_ = v___x_4074_;
goto v_reusejp_4076_;
}
else
{
lean_object* v_reuseFailAlloc_4078_; 
v_reuseFailAlloc_4078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4078_, 0, v_a_4072_);
v___x_4077_ = v_reuseFailAlloc_4078_;
goto v_reusejp_4076_;
}
v_reusejp_4076_:
{
return v___x_4077_;
}
}
}
}
v___jp_4081_:
{
if (v___y_4084_ == 0)
{
v___y_4063_ = v___y_4082_;
v___y_4064_ = v___y_4083_;
v___y_4065_ = v_severity_3957_;
goto v___jp_4062_;
}
else
{
v___y_4063_ = v___y_4082_;
v___y_4064_ = v___y_4083_;
v___y_4065_ = v___x_4080_;
goto v___jp_4062_;
}
}
v___jp_4085_:
{
if (v___y_4086_ == 0)
{
lean_object* v___x_4087_; lean_object* v_scopes_4088_; lean_object* v___x_4089_; lean_object* v___x_4090_; lean_object* v_opts_4091_; uint8_t v___x_4092_; uint8_t v___x_4093_; 
v___x_4087_ = lean_st_ref_get(v___y_3960_);
v_scopes_4088_ = lean_ctor_get(v___x_4087_, 2);
lean_inc(v_scopes_4088_);
lean_dec(v___x_4087_);
v___x_4089_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_4090_ = l_List_head_x21___redArg(v___x_4089_, v_scopes_4088_);
lean_dec(v_scopes_4088_);
v_opts_4091_ = lean_ctor_get(v___x_4090_, 1);
lean_inc_ref(v_opts_4091_);
lean_dec(v___x_4090_);
v___x_4092_ = 1;
v___x_4093_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3957_, v___x_4092_);
if (v___x_4093_ == 0)
{
lean_dec_ref(v_opts_4091_);
v___y_4082_ = v___y_4086_;
v___y_4083_ = v___y_4086_;
v___y_4084_ = v___x_4093_;
goto v___jp_4081_;
}
else
{
lean_object* v___x_4094_; uint8_t v___x_4095_; 
v___x_4094_ = l_Lean_warningAsError;
v___x_4095_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__1(v_opts_4091_, v___x_4094_);
lean_dec_ref(v_opts_4091_);
v___y_4082_ = v___y_4086_;
v___y_4083_ = v___y_4086_;
v___y_4084_ = v___x_4095_;
goto v___jp_4081_;
}
}
else
{
lean_object* v___x_4096_; lean_object* v___x_4097_; 
lean_dec_ref(v_msgData_3956_);
v___x_4096_ = lean_box(0);
v___x_4097_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4097_, 0, v___x_4096_);
return v___x_4097_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_4100_, lean_object* v_msgData_4101_, lean_object* v_severity_4102_, lean_object* v_isSilent_4103_, lean_object* v___y_4104_, lean_object* v___y_4105_, lean_object* v___y_4106_){
_start:
{
uint8_t v_severity_boxed_4107_; uint8_t v_isSilent_boxed_4108_; lean_object* v_res_4109_; 
v_severity_boxed_4107_ = lean_unbox(v_severity_4102_);
v_isSilent_boxed_4108_ = lean_unbox(v_isSilent_4103_);
v_res_4109_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2(v_ref_4100_, v_msgData_4101_, v_severity_boxed_4107_, v_isSilent_boxed_4108_, v___y_4104_, v___y_4105_);
lean_dec(v___y_4105_);
lean_dec_ref(v___y_4104_);
lean_dec(v_ref_4100_);
return v_res_4109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1(lean_object* v_msgData_4110_, uint8_t v_severity_4111_, uint8_t v_isSilent_4112_, lean_object* v___y_4113_, lean_object* v___y_4114_){
_start:
{
lean_object* v___x_4116_; 
v___x_4116_ = l_Lean_Elab_Command_getRef___redArg(v___y_4113_);
if (lean_obj_tag(v___x_4116_) == 0)
{
lean_object* v_a_4117_; lean_object* v___x_4118_; 
v_a_4117_ = lean_ctor_get(v___x_4116_, 0);
lean_inc(v_a_4117_);
lean_dec_ref_known(v___x_4116_, 1);
v___x_4118_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2(v_a_4117_, v_msgData_4110_, v_severity_4111_, v_isSilent_4112_, v___y_4113_, v___y_4114_);
lean_dec(v_a_4117_);
return v___x_4118_;
}
else
{
lean_object* v_a_4119_; lean_object* v___x_4121_; uint8_t v_isShared_4122_; uint8_t v_isSharedCheck_4126_; 
lean_dec_ref(v_msgData_4110_);
v_a_4119_ = lean_ctor_get(v___x_4116_, 0);
v_isSharedCheck_4126_ = !lean_is_exclusive(v___x_4116_);
if (v_isSharedCheck_4126_ == 0)
{
v___x_4121_ = v___x_4116_;
v_isShared_4122_ = v_isSharedCheck_4126_;
goto v_resetjp_4120_;
}
else
{
lean_inc(v_a_4119_);
lean_dec(v___x_4116_);
v___x_4121_ = lean_box(0);
v_isShared_4122_ = v_isSharedCheck_4126_;
goto v_resetjp_4120_;
}
v_resetjp_4120_:
{
lean_object* v___x_4124_; 
if (v_isShared_4122_ == 0)
{
v___x_4124_ = v___x_4121_;
goto v_reusejp_4123_;
}
else
{
lean_object* v_reuseFailAlloc_4125_; 
v_reuseFailAlloc_4125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4125_, 0, v_a_4119_);
v___x_4124_ = v_reuseFailAlloc_4125_;
goto v_reusejp_4123_;
}
v_reusejp_4123_:
{
return v___x_4124_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1___boxed(lean_object* v_msgData_4127_, lean_object* v_severity_4128_, lean_object* v_isSilent_4129_, lean_object* v___y_4130_, lean_object* v___y_4131_, lean_object* v___y_4132_){
_start:
{
uint8_t v_severity_boxed_4133_; uint8_t v_isSilent_boxed_4134_; lean_object* v_res_4135_; 
v_severity_boxed_4133_ = lean_unbox(v_severity_4128_);
v_isSilent_boxed_4134_ = lean_unbox(v_isSilent_4129_);
v_res_4135_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1(v_msgData_4127_, v_severity_boxed_4133_, v_isSilent_boxed_4134_, v___y_4130_, v___y_4131_);
lean_dec(v___y_4131_);
lean_dec_ref(v___y_4130_);
return v_res_4135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1(lean_object* v_msgData_4136_, lean_object* v___y_4137_, lean_object* v___y_4138_){
_start:
{
uint8_t v___x_4140_; uint8_t v___x_4141_; lean_object* v___x_4142_; 
v___x_4140_ = 1;
v___x_4141_ = 0;
v___x_4142_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1(v_msgData_4136_, v___x_4140_, v___x_4141_, v___y_4137_, v___y_4138_);
return v___x_4142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1___boxed(lean_object* v_msgData_4143_, lean_object* v___y_4144_, lean_object* v___y_4145_, lean_object* v___y_4146_){
_start:
{
lean_object* v_res_4147_; 
v_res_4147_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1(v_msgData_4143_, v___y_4144_, v___y_4145_);
lean_dec(v___y_4145_);
lean_dec_ref(v___y_4144_);
return v_res_4147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__2(lean_object* v_a_4148_, lean_object* v_a_4149_){
_start:
{
if (lean_obj_tag(v_a_4148_) == 0)
{
lean_object* v___x_4150_; 
v___x_4150_ = l_List_reverse___redArg(v_a_4149_);
return v___x_4150_;
}
else
{
lean_object* v_head_4151_; lean_object* v_tail_4152_; lean_object* v___x_4154_; uint8_t v_isShared_4155_; uint8_t v_isSharedCheck_4161_; 
v_head_4151_ = lean_ctor_get(v_a_4148_, 0);
v_tail_4152_ = lean_ctor_get(v_a_4148_, 1);
v_isSharedCheck_4161_ = !lean_is_exclusive(v_a_4148_);
if (v_isSharedCheck_4161_ == 0)
{
v___x_4154_ = v_a_4148_;
v_isShared_4155_ = v_isSharedCheck_4161_;
goto v_resetjp_4153_;
}
else
{
lean_inc(v_tail_4152_);
lean_inc(v_head_4151_);
lean_dec(v_a_4148_);
v___x_4154_ = lean_box(0);
v_isShared_4155_ = v_isSharedCheck_4161_;
goto v_resetjp_4153_;
}
v_resetjp_4153_:
{
lean_object* v_data_4156_; lean_object* v___x_4158_; 
v_data_4156_ = lean_ctor_get(v_head_4151_, 4);
lean_inc(v_data_4156_);
lean_dec(v_head_4151_);
if (v_isShared_4155_ == 0)
{
lean_ctor_set(v___x_4154_, 1, v_a_4149_);
lean_ctor_set(v___x_4154_, 0, v_data_4156_);
v___x_4158_ = v___x_4154_;
goto v_reusejp_4157_;
}
else
{
lean_object* v_reuseFailAlloc_4160_; 
v_reuseFailAlloc_4160_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4160_, 0, v_data_4156_);
lean_ctor_set(v_reuseFailAlloc_4160_, 1, v_a_4149_);
v___x_4158_ = v_reuseFailAlloc_4160_;
goto v_reusejp_4157_;
}
v_reusejp_4157_:
{
v_a_4148_ = v_tail_4152_;
v_a_4149_ = v___x_4158_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__5(lean_object* v_msgData_4162_, lean_object* v___y_4163_, lean_object* v___y_4164_){
_start:
{
uint8_t v___x_4166_; uint8_t v___x_4167_; lean_object* v___x_4168_; 
v___x_4166_ = 0;
v___x_4167_ = 0;
v___x_4168_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1(v_msgData_4162_, v___x_4166_, v___x_4167_, v___y_4163_, v___y_4164_);
return v___x_4168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__5___boxed(lean_object* v_msgData_4169_, lean_object* v___y_4170_, lean_object* v___y_4171_, lean_object* v___y_4172_){
_start:
{
lean_object* v_res_4173_; 
v_res_4173_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__5(v_msgData_4169_, v___y_4170_, v___y_4171_);
lean_dec(v___y_4171_);
lean_dec_ref(v___y_4170_);
return v_res_4173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___redArg(lean_object* v_as_4174_, size_t v_sz_4175_, size_t v_i_4176_, lean_object* v_b_4177_){
_start:
{
uint8_t v___x_4179_; 
v___x_4179_ = lean_usize_dec_lt(v_i_4176_, v_sz_4175_);
if (v___x_4179_ == 0)
{
lean_object* v___x_4180_; 
v___x_4180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4180_, 0, v_b_4177_);
return v___x_4180_;
}
else
{
lean_object* v_a_4181_; lean_object* v_fst_4182_; lean_object* v_snd_4183_; lean_object* v___x_4185_; uint8_t v_isShared_4186_; uint8_t v_isSharedCheck_4203_; 
v_a_4181_ = lean_array_uget(v_as_4174_, v_i_4176_);
v_fst_4182_ = lean_ctor_get(v_a_4181_, 0);
v_snd_4183_ = lean_ctor_get(v_a_4181_, 1);
v_isSharedCheck_4203_ = !lean_is_exclusive(v_a_4181_);
if (v_isSharedCheck_4203_ == 0)
{
v___x_4185_ = v_a_4181_;
v_isShared_4186_ = v_isSharedCheck_4203_;
goto v_resetjp_4184_;
}
else
{
lean_inc(v_snd_4183_);
lean_inc(v_fst_4182_);
lean_dec(v_a_4181_);
v___x_4185_ = lean_box(0);
v_isShared_4186_ = v_isSharedCheck_4203_;
goto v_resetjp_4184_;
}
v_resetjp_4184_:
{
lean_object* v___x_4187_; lean_object* v___x_4188_; lean_object* v___x_4189_; lean_object* v___x_4190_; lean_object* v___x_4191_; lean_object* v___x_4192_; lean_object* v___x_4194_; 
v___x_4187_ = lean_array_to_list(v_snd_4183_);
v___x_4188_ = lean_box(0);
v___x_4189_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__6(v___x_4187_, v___x_4188_);
v___x_4190_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v___x_4191_ = l_Lean_MessageData_joinSep(v___x_4189_, v___x_4190_);
v___x_4192_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2, &lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2_once, _init_lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8___closed__2);
if (v_isShared_4186_ == 0)
{
lean_ctor_set_tag(v___x_4185_, 7);
lean_ctor_set(v___x_4185_, 1, v_fst_4182_);
lean_ctor_set(v___x_4185_, 0, v___x_4192_);
v___x_4194_ = v___x_4185_;
goto v_reusejp_4193_;
}
else
{
lean_object* v_reuseFailAlloc_4202_; 
v_reuseFailAlloc_4202_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4202_, 0, v___x_4192_);
lean_ctor_set(v_reuseFailAlloc_4202_, 1, v_fst_4182_);
v___x_4194_ = v_reuseFailAlloc_4202_;
goto v_reusejp_4193_;
}
v_reusejp_4193_:
{
lean_object* v___x_4195_; lean_object* v___x_4196_; lean_object* v___x_4197_; lean_object* v___x_4198_; size_t v___x_4199_; size_t v___x_4200_; 
v___x_4195_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__2___closed__2);
v___x_4196_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4196_, 0, v___x_4194_);
lean_ctor_set(v___x_4196_, 1, v___x_4195_);
v___x_4197_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4197_, 0, v___x_4196_);
lean_ctor_set(v___x_4197_, 1, v___x_4191_);
v___x_4198_ = lean_array_push(v_b_4177_, v___x_4197_);
v___x_4199_ = ((size_t)1ULL);
v___x_4200_ = lean_usize_add(v_i_4176_, v___x_4199_);
v_i_4176_ = v___x_4200_;
v_b_4177_ = v___x_4198_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___redArg___boxed(lean_object* v_as_4204_, lean_object* v_sz_4205_, lean_object* v_i_4206_, lean_object* v_b_4207_, lean_object* v___y_4208_){
_start:
{
size_t v_sz_boxed_4209_; size_t v_i_boxed_4210_; lean_object* v_res_4211_; 
v_sz_boxed_4209_ = lean_unbox_usize(v_sz_4205_);
lean_dec(v_sz_4205_);
v_i_boxed_4210_ = lean_unbox_usize(v_i_4206_);
lean_dec(v_i_4206_);
v_res_4211_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___redArg(v_as_4204_, v_sz_boxed_4209_, v_i_boxed_4210_, v_b_4207_);
lean_dec_ref(v_as_4204_);
return v_res_4211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4(lean_object* v_kind_4212_, lean_object* v_uniqueFailures_4213_, lean_object* v_synthResults_4214_, lean_object* v___y_4215_, lean_object* v___y_4216_){
_start:
{
lean_object* v___x_4218_; lean_object* v___x_4219_; uint8_t v___x_4220_; 
v___x_4218_ = lean_array_get_size(v_synthResults_4214_);
v___x_4219_ = lean_unsigned_to_nat(0u);
v___x_4220_ = lean_nat_dec_eq(v___x_4218_, v___x_4219_);
if (v___x_4220_ == 0)
{
lean_object* v_entries_4221_; size_t v_sz_4222_; size_t v___x_4223_; lean_object* v___x_4224_; 
lean_dec_ref(v_uniqueFailures_4213_);
v_entries_4221_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces___closed__1));
v_sz_4222_ = lean_array_size(v_synthResults_4214_);
v___x_4223_ = ((size_t)0ULL);
v___x_4224_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___redArg(v_synthResults_4214_, v_sz_4222_, v___x_4223_, v_entries_4221_);
if (lean_obj_tag(v___x_4224_) == 0)
{
lean_object* v_a_4225_; lean_object* v___x_4226_; lean_object* v___x_4227_; lean_object* v_report_4228_; lean_object* v___x_4229_; lean_object* v___x_4230_; lean_object* v___x_4231_; lean_object* v___x_4232_; lean_object* v___x_4233_; lean_object* v___x_4234_; lean_object* v___x_4235_; 
v_a_4225_ = lean_ctor_get(v___x_4224_, 0);
lean_inc(v_a_4225_);
lean_dec_ref_known(v___x_4224_, 1);
v___x_4226_ = lean_array_to_list(v_a_4225_);
v___x_4227_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v_report_4228_ = l_Lean_MessageData_joinSep(v___x_4226_, v___x_4227_);
v___x_4229_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_4230_ = l_Lean_stringToMessageData(v_kind_4212_);
v___x_4231_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4231_, 0, v___x_4229_);
lean_ctor_set(v___x_4231_, 1, v___x_4230_);
v___x_4232_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__6);
v___x_4233_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4233_, 0, v___x_4231_);
lean_ctor_set(v___x_4233_, 1, v___x_4232_);
v___x_4234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4234_, 0, v___x_4233_);
lean_ctor_set(v___x_4234_, 1, v_report_4228_);
v___x_4235_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1(v___x_4234_, v___y_4215_, v___y_4216_);
return v___x_4235_;
}
else
{
lean_object* v_a_4236_; lean_object* v___x_4238_; uint8_t v_isShared_4239_; uint8_t v_isSharedCheck_4243_; 
lean_dec_ref(v_kind_4212_);
v_a_4236_ = lean_ctor_get(v___x_4224_, 0);
v_isSharedCheck_4243_ = !lean_is_exclusive(v___x_4224_);
if (v_isSharedCheck_4243_ == 0)
{
v___x_4238_ = v___x_4224_;
v_isShared_4239_ = v_isSharedCheck_4243_;
goto v_resetjp_4237_;
}
else
{
lean_inc(v_a_4236_);
lean_dec(v___x_4224_);
v___x_4238_ = lean_box(0);
v_isShared_4239_ = v_isSharedCheck_4243_;
goto v_resetjp_4237_;
}
v_resetjp_4237_:
{
lean_object* v___x_4241_; 
if (v_isShared_4239_ == 0)
{
v___x_4241_ = v___x_4238_;
goto v_reusejp_4240_;
}
else
{
lean_object* v_reuseFailAlloc_4242_; 
v_reuseFailAlloc_4242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4242_, 0, v_a_4236_);
v___x_4241_ = v_reuseFailAlloc_4242_;
goto v_reusejp_4240_;
}
v_reusejp_4240_:
{
return v___x_4241_;
}
}
}
}
else
{
lean_object* v___x_4244_; uint8_t v___x_4245_; 
v___x_4244_ = lean_array_get_size(v_uniqueFailures_4213_);
v___x_4245_ = lean_nat_dec_eq(v___x_4244_, v___x_4219_);
if (v___x_4245_ == 0)
{
lean_object* v___x_4246_; lean_object* v___x_4247_; lean_object* v___x_4248_; lean_object* v___x_4249_; lean_object* v_failureList_4250_; lean_object* v___x_4251_; lean_object* v___x_4252_; lean_object* v___x_4253_; lean_object* v___x_4254_; lean_object* v___x_4255_; lean_object* v___x_4256_; lean_object* v___x_4257_; 
v___x_4246_ = lean_array_to_list(v_uniqueFailures_4213_);
v___x_4247_ = lean_box(0);
v___x_4248_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuse__1_spec__5_spec__8(v___x_4246_, v___x_4247_);
v___x_4249_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__2);
v_failureList_4250_ = l_Lean_MessageData_joinSep(v___x_4248_, v___x_4249_);
v___x_4251_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_4252_ = l_Lean_stringToMessageData(v_kind_4212_);
v___x_4253_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4253_, 0, v___x_4251_);
lean_ctor_set(v___x_4253_, 1, v___x_4252_);
v___x_4254_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__4);
v___x_4255_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4255_, 0, v___x_4253_);
lean_ctor_set(v___x_4255_, 1, v___x_4254_);
v___x_4256_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4256_, 0, v___x_4255_);
lean_ctor_set(v___x_4256_, 1, v_failureList_4250_);
v___x_4257_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1(v___x_4256_, v___y_4215_, v___y_4216_);
return v___x_4257_;
}
else
{
lean_object* v___x_4258_; lean_object* v___x_4259_; lean_object* v___x_4260_; lean_object* v___x_4261_; lean_object* v___x_4262_; lean_object* v___x_4263_; 
lean_dec_ref(v_uniqueFailures_4213_);
v___x_4258_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___lam__0___closed__4);
v___x_4259_ = l_Lean_stringToMessageData(v_kind_4212_);
v___x_4260_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4260_, 0, v___x_4258_);
lean_ctor_set(v___x_4260_, 1, v___x_4259_);
v___x_4261_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6, &lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___redArg___closed__6);
v___x_4262_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4262_, 0, v___x_4260_);
lean_ctor_set(v___x_4262_, 1, v___x_4261_);
v___x_4263_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1(v___x_4262_, v___y_4215_, v___y_4216_);
return v___x_4263_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4___boxed(lean_object* v_kind_4264_, lean_object* v_uniqueFailures_4265_, lean_object* v_synthResults_4266_, lean_object* v___y_4267_, lean_object* v___y_4268_, lean_object* v___y_4269_){
_start:
{
lean_object* v_res_4270_; 
v_res_4270_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4(v_kind_4264_, v_uniqueFailures_4265_, v_synthResults_4266_, v___y_4267_, v___y_4268_);
lean_dec(v___y_4268_);
lean_dec_ref(v___y_4267_);
lean_dec_ref(v_synthResults_4266_);
return v_res_4270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___redArg(size_t v_sz_4271_, size_t v_i_4272_, lean_object* v_bs_4273_){
_start:
{
uint8_t v___x_4275_; 
v___x_4275_ = lean_usize_dec_lt(v_i_4272_, v_sz_4271_);
if (v___x_4275_ == 0)
{
lean_object* v___x_4276_; 
v___x_4276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4276_, 0, v_bs_4273_);
return v___x_4276_;
}
else
{
lean_object* v_v_4277_; lean_object* v_fst_4278_; lean_object* v_snd_4279_; lean_object* v___x_4281_; uint8_t v_isShared_4282_; uint8_t v_isSharedCheck_4293_; 
v_v_4277_ = lean_array_uget(v_bs_4273_, v_i_4272_);
v_fst_4278_ = lean_ctor_get(v_v_4277_, 0);
v_snd_4279_ = lean_ctor_get(v_v_4277_, 1);
v_isSharedCheck_4293_ = !lean_is_exclusive(v_v_4277_);
if (v_isSharedCheck_4293_ == 0)
{
v___x_4281_ = v_v_4277_;
v_isShared_4282_ = v_isSharedCheck_4293_;
goto v_resetjp_4280_;
}
else
{
lean_inc(v_snd_4279_);
lean_inc(v_fst_4278_);
lean_dec(v_v_4277_);
v___x_4281_ = lean_box(0);
v_isShared_4282_ = v_isSharedCheck_4293_;
goto v_resetjp_4280_;
}
v_resetjp_4280_:
{
lean_object* v___x_4283_; lean_object* v___x_4284_; lean_object* v_bs_x27_4285_; lean_object* v___x_4287_; 
v___x_4283_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures(v_snd_4279_);
v___x_4284_ = lean_unsigned_to_nat(0u);
v_bs_x27_4285_ = lean_array_uset(v_bs_4273_, v_i_4272_, v___x_4284_);
if (v_isShared_4282_ == 0)
{
lean_ctor_set(v___x_4281_, 1, v___x_4283_);
v___x_4287_ = v___x_4281_;
goto v_reusejp_4286_;
}
else
{
lean_object* v_reuseFailAlloc_4292_; 
v_reuseFailAlloc_4292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4292_, 0, v_fst_4278_);
lean_ctor_set(v_reuseFailAlloc_4292_, 1, v___x_4283_);
v___x_4287_ = v_reuseFailAlloc_4292_;
goto v_reusejp_4286_;
}
v_reusejp_4286_:
{
size_t v___x_4288_; size_t v___x_4289_; lean_object* v___x_4290_; 
v___x_4288_ = ((size_t)1ULL);
v___x_4289_ = lean_usize_add(v_i_4272_, v___x_4288_);
v___x_4290_ = lean_array_uset(v_bs_x27_4285_, v_i_4272_, v___x_4287_);
v_i_4272_ = v___x_4289_;
v_bs_4273_ = v___x_4290_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___redArg___boxed(lean_object* v_sz_4294_, lean_object* v_i_4295_, lean_object* v_bs_4296_, lean_object* v___y_4297_){
_start:
{
size_t v_sz_boxed_4298_; size_t v_i_boxed_4299_; lean_object* v_res_4300_; 
v_sz_boxed_4298_ = lean_unbox_usize(v_sz_4294_);
lean_dec(v_sz_4294_);
v_i_boxed_4299_ = lean_unbox_usize(v_i_4295_);
lean_dec(v_i_4295_);
v_res_4300_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___redArg(v_sz_boxed_4298_, v_i_boxed_4299_, v_bs_4296_);
return v_res_4300_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__2(void){
_start:
{
lean_object* v___x_4304_; lean_object* v___x_4305_; 
v___x_4304_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__1));
v___x_4305_ = l_Lean_MessageData_ofFormat(v___x_4304_);
return v___x_4305_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__6(void){
_start:
{
lean_object* v___x_4312_; lean_object* v___x_4313_; 
v___x_4312_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__5));
v___x_4313_ = l_Lean_MessageData_ofFormat(v___x_4312_);
return v___x_4313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1(lean_object* v_x_4314_, lean_object* v_a_4315_, lean_object* v_a_4316_){
_start:
{
lean_object* v___x_4318_; uint8_t v___x_4319_; 
v___x_4318_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__1));
lean_inc(v_x_4314_);
v___x_4319_ = l_Lean_Syntax_isOfKind(v_x_4314_, v___x_4318_);
if (v___x_4319_ == 0)
{
lean_object* v___x_4320_; 
lean_dec(v_x_4314_);
v___x_4320_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__0___redArg();
return v___x_4320_;
}
else
{
lean_object* v___x_4321_; lean_object* v___x_4322_; lean_object* v___x_4323_; lean_object* v___f_4324_; lean_object* v___x_4325_; lean_object* v___x_4326_; lean_object* v___x_4327_; lean_object* v___x_4328_; 
v___x_4321_ = lean_st_ref_get(v_a_4316_);
v___x_4322_ = lean_unsigned_to_nat(2u);
v___x_4323_ = l_Lean_Syntax_getArg(v_x_4314_, v___x_4322_);
lean_dec(v_x_4314_);
lean_inc(v___x_4323_);
v___f_4324_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__1___boxed), 4, 1);
lean_closure_set(v___f_4324_, 0, v___x_4323_);
v___x_4325_ = lean_box(v___x_4319_);
v___x_4326_ = lean_box(v___x_4319_);
v___x_4327_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___boxed), 3, 2);
lean_closure_set(v___x_4327_, 0, v___x_4325_);
lean_closure_set(v___x_4327_, 1, v___x_4326_);
lean_inc_ref(v___f_4324_);
v___x_4328_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__2(v___f_4324_, v___x_4327_, v_a_4315_, v_a_4316_);
if (lean_obj_tag(v___x_4328_) == 0)
{
lean_object* v_a_4329_; lean_object* v_fst_4330_; lean_object* v_snd_4331_; lean_object* v___x_4332_; 
v_a_4329_ = lean_ctor_get(v___x_4328_, 0);
lean_inc(v_a_4329_);
lean_dec_ref_known(v___x_4328_, 1);
v_fst_4330_ = lean_ctor_get(v_a_4329_, 0);
lean_inc(v_fst_4330_);
v_snd_4331_ = lean_ctor_get(v_a_4329_, 1);
lean_inc(v_snd_4331_);
lean_dec(v_a_4329_);
lean_inc(v___x_4321_);
v___x_4332_ = lean_st_ref_set(v_a_4316_, v___x_4321_);
if (lean_obj_tag(v_fst_4330_) == 0)
{
uint8_t v___x_4333_; lean_object* v___x_4334_; lean_object* v___x_4335_; lean_object* v___x_4336_; lean_object* v___x_4337_; 
lean_dec_ref_known(v_fst_4330_, 1);
v___x_4333_ = 0;
v___x_4334_ = lean_box(v___x_4319_);
v___x_4335_ = lean_box(v___x_4333_);
v___x_4336_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__0___boxed), 3, 2);
lean_closure_set(v___x_4336_, 0, v___x_4334_);
lean_closure_set(v___x_4336_, 1, v___x_4335_);
v___x_4337_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___lam__2(v___f_4324_, v___x_4336_, v_a_4315_, v_a_4316_);
if (lean_obj_tag(v___x_4337_) == 0)
{
lean_object* v_a_4338_; lean_object* v_fst_4339_; lean_object* v_snd_4340_; lean_object* v___x_4341_; 
v_a_4338_ = lean_ctor_get(v___x_4337_, 0);
lean_inc(v_a_4338_);
lean_dec_ref_known(v___x_4337_, 1);
v_fst_4339_ = lean_ctor_get(v_a_4338_, 0);
lean_inc(v_fst_4339_);
v_snd_4340_ = lean_ctor_get(v_a_4338_, 1);
lean_inc(v_snd_4340_);
lean_dec(v_a_4338_);
v___x_4341_ = lean_st_ref_set(v_a_4316_, v___x_4321_);
if (lean_obj_tag(v_fst_4339_) == 0)
{
lean_object* v___x_4342_; lean_object* v___x_4343_; 
lean_dec_ref_known(v_fst_4339_, 1);
lean_dec(v_snd_4340_);
lean_dec(v_snd_4331_);
v___x_4342_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__2, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__2);
v___x_4343_ = lp_mathlib_Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1(v___x_4342_, v_a_4315_, v_a_4316_);
if (lean_obj_tag(v___x_4343_) == 0)
{
lean_object* v___x_4344_; 
lean_dec_ref_known(v___x_4343_, 1);
v___x_4344_ = l_Lean_Elab_Command_elabCommand(v___x_4323_, v_a_4315_, v_a_4316_);
return v___x_4344_;
}
else
{
lean_dec(v___x_4323_);
return v___x_4343_;
}
}
else
{
lean_object* v___x_4345_; lean_object* v___x_4346_; lean_object* v___x_4347_; lean_object* v___x_4348_; lean_object* v___x_4349_; lean_object* v___x_4350_; lean_object* v_fst_4351_; lean_object* v_snd_4352_; lean_object* v___x_4353_; size_t v_sz_4354_; size_t v___x_4355_; lean_object* v___x_4356_; 
lean_dec_ref_known(v_fst_4339_, 1);
v___x_4345_ = lean_box(0);
v___x_4346_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__2(v_snd_4331_, v___x_4345_);
v___x_4347_ = lean_array_mk(v___x_4346_);
v___x_4348_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__2(v_snd_4340_, v___x_4345_);
v___x_4349_ = lean_array_mk(v___x_4348_);
v___x_4350_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_analyzeTraces(v___x_4347_, v___x_4349_, v___x_4319_);
lean_dec_ref(v___x_4349_);
lean_dec_ref(v___x_4347_);
v_fst_4351_ = lean_ctor_get(v___x_4350_, 0);
lean_inc(v_fst_4351_);
v_snd_4352_ = lean_ctor_get(v___x_4350_, 1);
lean_inc(v_snd_4352_);
lean_dec_ref(v___x_4350_);
v___x_4353_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_disambiguateFailures(v_fst_4351_);
v_sz_4354_ = lean_array_size(v_snd_4352_);
v___x_4355_ = ((size_t)0ULL);
v___x_4356_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___redArg(v_sz_4354_, v___x_4355_, v_snd_4352_);
if (lean_obj_tag(v___x_4356_) == 0)
{
lean_object* v_a_4357_; lean_object* v___x_4358_; lean_object* v___x_4359_; 
v_a_4357_ = lean_ctor_get(v___x_4356_, 0);
lean_inc(v_a_4357_);
lean_dec_ref_known(v___x_4356_, 1);
v___x_4358_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse_defeqAbuseCmd___closed__6));
v___x_4359_ = lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4(v___x_4358_, v___x_4353_, v_a_4357_, v_a_4315_, v_a_4316_);
lean_dec(v_a_4357_);
if (lean_obj_tag(v___x_4359_) == 0)
{
lean_object* v___f_4360_; lean_object* v___x_4361_; lean_object* v___x_4362_; 
lean_dec_ref_known(v___x_4359_, 1);
v___f_4360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__3));
v___x_4361_ = lean_alloc_closure((void*)(l_Lean_Elab_Command_elabCommand___boxed), 4, 1);
lean_closure_set(v___x_4361_, 0, v___x_4323_);
v___x_4362_ = l_Lean_Elab_Command_withScope___redArg(v___f_4360_, v___x_4361_, v_a_4315_, v_a_4316_);
return v___x_4362_;
}
else
{
lean_dec(v___x_4323_);
return v___x_4359_;
}
}
else
{
lean_object* v_a_4363_; lean_object* v___x_4365_; uint8_t v_isShared_4366_; uint8_t v_isSharedCheck_4370_; 
lean_dec_ref(v___x_4353_);
lean_dec(v___x_4323_);
v_a_4363_ = lean_ctor_get(v___x_4356_, 0);
v_isSharedCheck_4370_ = !lean_is_exclusive(v___x_4356_);
if (v_isSharedCheck_4370_ == 0)
{
v___x_4365_ = v___x_4356_;
v_isShared_4366_ = v_isSharedCheck_4370_;
goto v_resetjp_4364_;
}
else
{
lean_inc(v_a_4363_);
lean_dec(v___x_4356_);
v___x_4365_ = lean_box(0);
v_isShared_4366_ = v_isSharedCheck_4370_;
goto v_resetjp_4364_;
}
v_resetjp_4364_:
{
lean_object* v___x_4368_; 
if (v_isShared_4366_ == 0)
{
v___x_4368_ = v___x_4365_;
goto v_reusejp_4367_;
}
else
{
lean_object* v_reuseFailAlloc_4369_; 
v_reuseFailAlloc_4369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4369_, 0, v_a_4363_);
v___x_4368_ = v_reuseFailAlloc_4369_;
goto v_reusejp_4367_;
}
v_reusejp_4367_:
{
return v___x_4368_;
}
}
}
}
}
else
{
lean_object* v_a_4371_; lean_object* v___x_4373_; uint8_t v_isShared_4374_; uint8_t v_isSharedCheck_4378_; 
lean_dec(v_snd_4331_);
lean_dec(v___x_4323_);
lean_dec(v___x_4321_);
v_a_4371_ = lean_ctor_get(v___x_4337_, 0);
v_isSharedCheck_4378_ = !lean_is_exclusive(v___x_4337_);
if (v_isSharedCheck_4378_ == 0)
{
v___x_4373_ = v___x_4337_;
v_isShared_4374_ = v_isSharedCheck_4378_;
goto v_resetjp_4372_;
}
else
{
lean_inc(v_a_4371_);
lean_dec(v___x_4337_);
v___x_4373_ = lean_box(0);
v_isShared_4374_ = v_isSharedCheck_4378_;
goto v_resetjp_4372_;
}
v_resetjp_4372_:
{
lean_object* v___x_4376_; 
if (v_isShared_4374_ == 0)
{
v___x_4376_ = v___x_4373_;
goto v_reusejp_4375_;
}
else
{
lean_object* v_reuseFailAlloc_4377_; 
v_reuseFailAlloc_4377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4377_, 0, v_a_4371_);
v___x_4376_ = v_reuseFailAlloc_4377_;
goto v_reusejp_4375_;
}
v_reusejp_4375_:
{
return v___x_4376_;
}
}
}
}
else
{
lean_object* v___x_4379_; lean_object* v___x_4380_; 
lean_dec_ref_known(v_fst_4330_, 1);
lean_dec(v_snd_4331_);
lean_dec_ref(v___f_4324_);
lean_dec(v___x_4321_);
v___x_4379_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__6, &lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___closed__6);
v___x_4380_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__5(v___x_4379_, v_a_4315_, v_a_4316_);
if (lean_obj_tag(v___x_4380_) == 0)
{
lean_object* v___x_4381_; 
lean_dec_ref_known(v___x_4380_, 1);
v___x_4381_ = l_Lean_Elab_Command_elabCommand(v___x_4323_, v_a_4315_, v_a_4316_);
return v___x_4381_;
}
else
{
lean_dec(v___x_4323_);
return v___x_4380_;
}
}
}
else
{
lean_object* v_a_4382_; lean_object* v___x_4384_; uint8_t v_isShared_4385_; uint8_t v_isSharedCheck_4389_; 
lean_dec_ref(v___f_4324_);
lean_dec(v___x_4323_);
lean_dec(v___x_4321_);
v_a_4382_ = lean_ctor_get(v___x_4328_, 0);
v_isSharedCheck_4389_ = !lean_is_exclusive(v___x_4328_);
if (v_isSharedCheck_4389_ == 0)
{
v___x_4384_ = v___x_4328_;
v_isShared_4385_ = v_isSharedCheck_4389_;
goto v_resetjp_4383_;
}
else
{
lean_inc(v_a_4382_);
lean_dec(v___x_4328_);
v___x_4384_ = lean_box(0);
v_isShared_4385_ = v_isSharedCheck_4389_;
goto v_resetjp_4383_;
}
v_resetjp_4383_:
{
lean_object* v___x_4387_; 
if (v_isShared_4385_ == 0)
{
v___x_4387_ = v___x_4384_;
goto v_reusejp_4386_;
}
else
{
lean_object* v_reuseFailAlloc_4388_; 
v_reuseFailAlloc_4388_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4388_, 0, v_a_4382_);
v___x_4387_ = v_reuseFailAlloc_4388_;
goto v_reusejp_4386_;
}
v_reusejp_4386_:
{
return v___x_4387_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1___boxed(lean_object* v_x_4390_, lean_object* v_a_4391_, lean_object* v_a_4392_, lean_object* v_a_4393_){
_start:
{
lean_object* v_res_4394_; 
v_res_4394_ = lp_mathlib_Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1(v_x_4390_, v_a_4391_, v_a_4392_);
lean_dec(v_a_4392_);
lean_dec_ref(v_a_4391_);
return v_res_4394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3(size_t v_sz_4395_, size_t v_i_4396_, lean_object* v_bs_4397_, lean_object* v___y_4398_, lean_object* v___y_4399_){
_start:
{
lean_object* v___x_4401_; 
v___x_4401_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___redArg(v_sz_4395_, v_i_4396_, v_bs_4397_);
return v___x_4401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3___boxed(lean_object* v_sz_4402_, lean_object* v_i_4403_, lean_object* v_bs_4404_, lean_object* v___y_4405_, lean_object* v___y_4406_, lean_object* v___y_4407_){
_start:
{
size_t v_sz_boxed_4408_; size_t v_i_boxed_4409_; lean_object* v_res_4410_; 
v_sz_boxed_4408_ = lean_unbox_usize(v_sz_4402_);
lean_dec(v_sz_4402_);
v_i_boxed_4409_ = lean_unbox_usize(v_i_4403_);
lean_dec(v_i_4403_);
v_res_4410_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__3(v_sz_boxed_4408_, v_i_boxed_4409_, v_bs_4404_, v___y_4405_, v___y_4406_);
lean_dec(v___y_4406_);
lean_dec_ref(v___y_4405_);
return v_res_4410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5(lean_object* v_as_4411_, size_t v_sz_4412_, size_t v_i_4413_, lean_object* v_b_4414_, lean_object* v___y_4415_, lean_object* v___y_4416_){
_start:
{
lean_object* v___x_4418_; 
v___x_4418_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___redArg(v_as_4411_, v_sz_4412_, v_i_4413_, v_b_4414_);
return v___x_4418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5___boxed(lean_object* v_as_4419_, lean_object* v_sz_4420_, lean_object* v_i_4421_, lean_object* v_b_4422_, lean_object* v___y_4423_, lean_object* v___y_4424_, lean_object* v___y_4425_){
_start:
{
size_t v_sz_boxed_4426_; size_t v_i_boxed_4427_; lean_object* v_res_4428_; 
v_sz_boxed_4426_ = lean_unbox_usize(v_sz_4420_);
lean_dec(v_sz_4420_);
v_i_boxed_4427_ = lean_unbox_usize(v_i_4421_);
lean_dec(v_i_4421_);
v_res_4428_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_DefEqAbuse_0__Mathlib_Tactic_DefEqAbuse_reportDefEqAbuse___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__4_spec__5(v_as_4419_, v_sz_boxed_4426_, v_i_boxed_4427_, v_b_4422_, v___y_4423_, v___y_4424_);
lean_dec(v___y_4424_);
lean_dec_ref(v___y_4423_);
lean_dec_ref(v_as_4419_);
return v_res_4428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7(lean_object* v_msgData_4429_, lean_object* v___y_4430_, lean_object* v___y_4431_){
_start:
{
lean_object* v___x_4433_; 
v___x_4433_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___redArg(v_msgData_4429_, v___y_4431_);
return v___x_4433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7___boxed(lean_object* v_msgData_4434_, lean_object* v___y_4435_, lean_object* v___y_4436_, lean_object* v___y_4437_){
_start:
{
lean_object* v_res_4438_; 
v_res_4438_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_Tactic_DefEqAbuse___aux__Mathlib__Tactic__DefEqAbuse______elabRules__Mathlib__Tactic__DefEqAbuse__defeqAbuseCmd__1_spec__1_spec__1_spec__2_spec__7(v_msgData_4434_, v___y_4435_, v___y_4436_);
lean_dec(v___y_4436_);
lean_dec_ref(v___y_4435_);
return v_res_4438_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DefEqAbuse(uint8_t builtin) {
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
lean_object* runtime_initialize_mathlib_Mathlib_Lean_MessageData_Trace(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_DefEqAbuse(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_MessageData_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1 = _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__1);
lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3 = _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitTraceNodesM___auto__3);
lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__1 = _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__1);
lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__3 = _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithM___auto__3);
lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__1 = _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__1);
lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__3 = _init_lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__3();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_DefEqAbuse_0__Lean_MessageData_visitWithAndAscendM___auto__3);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_MessageData_Trace(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_DefEqAbuse(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Lean_MessageData_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DefEqAbuse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_DefEqAbuse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_DefEqAbuse(builtin);
}
#ifdef __cplusplus
}
#endif
