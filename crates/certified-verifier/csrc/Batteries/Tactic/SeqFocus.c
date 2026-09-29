// Lean compiler output
// Module: Batteries.Tactic.SeqFocus
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Basic
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_getUnsolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Elab_Tactic_setGoals___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_InternalExceptionId_getName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l_Lean_Elab_isAbortExceptionId(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticMap_tacs[_;]"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__2_value),LEAN_SCALAR_PTR_LITERAL(243, 18, 44, 109, 150, 48, 153, 18)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "map_tacs "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__10_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__11_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "; "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__14_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__15_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__10_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__17_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__17_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__20_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__20_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__21 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__21_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__21_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__0_value;
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__1 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__1_value;
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__2 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__2_value;
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__3 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__3_value;
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__4 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__4_value;
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__5 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__5_value;
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__6 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5_spec__10___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4_spec__7(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "internal exception: "};
static const lean_object* lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__0 = (const lean_object*)&lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "too many tactics"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__2;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "not enough tactics"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__4;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_seq__focus___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "seq_focus"};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 25, 59, 65, 212, 252, 96, 203)}};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_seq__focus___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " <;> "};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_seq__focus___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_seq__focus___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__7_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_seq__focus = (const lean_object*)&lp_batteries_Batteries_Tactic_seq__focus___closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "focus"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(198, 223, 207, 6, 131, 57, 182, 221)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__12_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "map_tacs"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__14_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__15;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__16_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_52_ = lean_box(0);
v___x_53_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_54_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___x_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg(){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___closed__0);
v___x_57_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg___boxed(lean_object* v___y_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg();
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0(lean_object* v_00_u03b1_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg();
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___boxed(lean_object* v_00_u03b1_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0(v_00_u03b1_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
return v_res_81_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___redArg(lean_object* v_keys_82_, lean_object* v_i_83_, lean_object* v_k_84_){
_start:
{
lean_object* v___x_85_; uint8_t v___x_86_; 
v___x_85_ = lean_array_get_size(v_keys_82_);
v___x_86_ = lean_nat_dec_lt(v_i_83_, v___x_85_);
if (v___x_86_ == 0)
{
lean_dec(v_i_83_);
return v___x_86_;
}
else
{
lean_object* v_k_x27_87_; uint8_t v___x_88_; 
v_k_x27_87_ = lean_array_fget_borrowed(v_keys_82_, v_i_83_);
v___x_88_ = l_Lean_instBEqMVarId_beq(v_k_84_, v_k_x27_87_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_89_ = lean_unsigned_to_nat(1u);
v___x_90_ = lean_nat_add(v_i_83_, v___x_89_);
lean_dec(v_i_83_);
v_i_83_ = v___x_90_;
goto _start;
}
else
{
lean_dec(v_i_83_);
return v___x_88_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___redArg___boxed(lean_object* v_keys_92_, lean_object* v_i_93_, lean_object* v_k_94_){
_start:
{
uint8_t v_res_95_; lean_object* v_r_96_; 
v_res_95_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___redArg(v_keys_92_, v_i_93_, v_k_94_);
lean_dec(v_k_94_);
lean_dec_ref(v_keys_92_);
v_r_96_ = lean_box(v_res_95_);
return v_r_96_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___redArg(lean_object* v_x_97_, size_t v_x_98_, lean_object* v_x_99_){
_start:
{
if (lean_obj_tag(v_x_97_) == 0)
{
lean_object* v_es_100_; lean_object* v___x_101_; size_t v___x_102_; size_t v___x_103_; lean_object* v_j_104_; lean_object* v___x_105_; 
v_es_100_ = lean_ctor_get(v_x_97_, 0);
v___x_101_ = lean_box(2);
v___x_102_ = ((size_t)31ULL);
v___x_103_ = lean_usize_land(v_x_98_, v___x_102_);
v_j_104_ = lean_usize_to_nat(v___x_103_);
v___x_105_ = lean_array_get_borrowed(v___x_101_, v_es_100_, v_j_104_);
lean_dec(v_j_104_);
switch(lean_obj_tag(v___x_105_))
{
case 0:
{
lean_object* v_key_106_; uint8_t v___x_107_; 
v_key_106_ = lean_ctor_get(v___x_105_, 0);
v___x_107_ = l_Lean_instBEqMVarId_beq(v_x_99_, v_key_106_);
return v___x_107_;
}
case 1:
{
lean_object* v_node_108_; size_t v___x_109_; size_t v___x_110_; 
v_node_108_ = lean_ctor_get(v___x_105_, 0);
v___x_109_ = ((size_t)5ULL);
v___x_110_ = lean_usize_shift_right(v_x_98_, v___x_109_);
v_x_97_ = v_node_108_;
v_x_98_ = v___x_110_;
goto _start;
}
default: 
{
uint8_t v___x_112_; 
v___x_112_ = 0;
return v___x_112_;
}
}
}
else
{
lean_object* v_ks_113_; lean_object* v___x_114_; uint8_t v___x_115_; 
v_ks_113_ = lean_ctor_get(v_x_97_, 0);
v___x_114_ = lean_unsigned_to_nat(0u);
v___x_115_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___redArg(v_ks_113_, v___x_114_, v_x_99_);
return v___x_115_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_116_, lean_object* v_x_117_, lean_object* v_x_118_){
_start:
{
size_t v_x_12452__boxed_119_; uint8_t v_res_120_; lean_object* v_r_121_; 
v_x_12452__boxed_119_ = lean_unbox_usize(v_x_117_);
lean_dec(v_x_117_);
v_res_120_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___redArg(v_x_116_, v_x_12452__boxed_119_, v_x_118_);
lean_dec(v_x_118_);
lean_dec_ref(v_x_116_);
v_r_121_ = lean_box(v_res_120_);
return v_r_121_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___redArg(lean_object* v_x_122_, lean_object* v_x_123_){
_start:
{
uint64_t v___x_124_; size_t v___x_125_; uint8_t v___x_126_; 
v___x_124_ = l_Lean_instHashableMVarId_hash(v_x_123_);
v___x_125_ = lean_uint64_to_usize(v___x_124_);
v___x_126_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___redArg(v_x_122_, v___x_125_, v_x_123_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___redArg___boxed(lean_object* v_x_127_, lean_object* v_x_128_){
_start:
{
uint8_t v_res_129_; lean_object* v_r_130_; 
v_res_129_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___redArg(v_x_127_, v_x_128_);
lean_dec(v_x_128_);
lean_dec_ref(v_x_127_);
v_r_130_ = lean_box(v_res_129_);
return v_r_130_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___redArg(lean_object* v_mvarId_131_, lean_object* v___y_132_){
_start:
{
lean_object* v___x_134_; lean_object* v_mctx_135_; lean_object* v_eAssignment_136_; uint8_t v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_134_ = lean_st_ref_get(v___y_132_);
v_mctx_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc_ref(v_mctx_135_);
lean_dec(v___x_134_);
v_eAssignment_136_ = lean_ctor_get(v_mctx_135_, 8);
lean_inc_ref(v_eAssignment_136_);
lean_dec_ref(v_mctx_135_);
v___x_137_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___redArg(v_eAssignment_136_, v_mvarId_131_);
lean_dec_ref(v_eAssignment_136_);
v___x_138_ = lean_box(v___x_137_);
v___x_139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___redArg___boxed(lean_object* v_mvarId_140_, lean_object* v___y_141_, lean_object* v___y_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___redArg(v_mvarId_140_, v___y_141_);
lean_dec(v___y_141_);
lean_dec(v_mvarId_140_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4_spec__7(lean_object* v_msgData_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
lean_object* v___x_150_; lean_object* v_env_151_; lean_object* v___x_152_; lean_object* v_mctx_153_; lean_object* v_lctx_154_; lean_object* v_options_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_150_ = lean_st_ref_get(v___y_148_);
v_env_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc_ref(v_env_151_);
lean_dec(v___x_150_);
v___x_152_ = lean_st_ref_get(v___y_146_);
v_mctx_153_ = lean_ctor_get(v___x_152_, 0);
lean_inc_ref(v_mctx_153_);
lean_dec(v___x_152_);
v_lctx_154_ = lean_ctor_get(v___y_145_, 2);
v_options_155_ = lean_ctor_get(v___y_147_, 2);
lean_inc_ref(v_options_155_);
lean_inc_ref(v_lctx_154_);
v___x_156_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_156_, 0, v_env_151_);
lean_ctor_set(v___x_156_, 1, v_mctx_153_);
lean_ctor_set(v___x_156_, 2, v_lctx_154_);
lean_ctor_set(v___x_156_, 3, v_options_155_);
v___x_157_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v_msgData_144_);
v___x_158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4_spec__7___boxed(lean_object* v_msgData_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4_spec__7(v_msgData_159_, v___y_160_, v___y_161_, v___y_162_, v___y_163_);
lean_dec(v___y_163_);
lean_dec_ref(v___y_162_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
return v_res_165_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0(uint8_t v___y_173_, uint8_t v_suppressElabErrors_174_, lean_object* v_x_175_){
_start:
{
if (lean_obj_tag(v_x_175_) == 1)
{
lean_object* v_pre_176_; 
v_pre_176_ = lean_ctor_get(v_x_175_, 0);
switch(lean_obj_tag(v_pre_176_))
{
case 1:
{
lean_object* v_pre_177_; 
v_pre_177_ = lean_ctor_get(v_pre_176_, 0);
switch(lean_obj_tag(v_pre_177_))
{
case 0:
{
lean_object* v_str_178_; lean_object* v_str_179_; lean_object* v___x_180_; uint8_t v___x_181_; 
v_str_178_ = lean_ctor_get(v_x_175_, 1);
v_str_179_ = lean_ctor_get(v_pre_176_, 1);
v___x_180_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__0));
v___x_181_ = lean_string_dec_eq(v_str_179_, v___x_180_);
if (v___x_181_ == 0)
{
lean_object* v___x_182_; uint8_t v___x_183_; 
v___x_182_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__1));
v___x_183_ = lean_string_dec_eq(v_str_179_, v___x_182_);
if (v___x_183_ == 0)
{
return v___y_173_;
}
else
{
lean_object* v___x_184_; uint8_t v___x_185_; 
v___x_184_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__1));
v___x_185_ = lean_string_dec_eq(v_str_178_, v___x_184_);
if (v___x_185_ == 0)
{
return v___y_173_;
}
else
{
return v_suppressElabErrors_174_;
}
}
}
else
{
lean_object* v___x_186_; uint8_t v___x_187_; 
v___x_186_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__2));
v___x_187_ = lean_string_dec_eq(v_str_178_, v___x_186_);
if (v___x_187_ == 0)
{
return v___y_173_;
}
else
{
return v_suppressElabErrors_174_;
}
}
}
case 1:
{
lean_object* v_pre_188_; 
v_pre_188_ = lean_ctor_get(v_pre_177_, 0);
if (lean_obj_tag(v_pre_188_) == 0)
{
lean_object* v_str_189_; lean_object* v_str_190_; lean_object* v_str_191_; lean_object* v___x_192_; uint8_t v___x_193_; 
v_str_189_ = lean_ctor_get(v_x_175_, 1);
v_str_190_ = lean_ctor_get(v_pre_176_, 1);
v_str_191_ = lean_ctor_get(v_pre_177_, 1);
v___x_192_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__3));
v___x_193_ = lean_string_dec_eq(v_str_191_, v___x_192_);
if (v___x_193_ == 0)
{
return v___y_173_;
}
else
{
lean_object* v___x_194_; uint8_t v___x_195_; 
v___x_194_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__4));
v___x_195_ = lean_string_dec_eq(v_str_190_, v___x_194_);
if (v___x_195_ == 0)
{
return v___y_173_;
}
else
{
lean_object* v___x_196_; uint8_t v___x_197_; 
v___x_196_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__5));
v___x_197_ = lean_string_dec_eq(v_str_189_, v___x_196_);
if (v___x_197_ == 0)
{
return v___y_173_;
}
else
{
return v_suppressElabErrors_174_;
}
}
}
}
else
{
return v___y_173_;
}
}
default: 
{
return v___y_173_;
}
}
}
case 0:
{
lean_object* v_str_198_; lean_object* v___x_199_; uint8_t v___x_200_; 
v_str_198_ = lean_ctor_get(v_x_175_, 1);
v___x_199_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___closed__6));
v___x_200_ = lean_string_dec_eq(v_str_198_, v___x_199_);
if (v___x_200_ == 0)
{
return v___y_173_;
}
else
{
return v_suppressElabErrors_174_;
}
}
default: 
{
return v___y_173_;
}
}
}
else
{
return v___y_173_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object* v___y_201_, lean_object* v_suppressElabErrors_202_, lean_object* v_x_203_){
_start:
{
uint8_t v___y_12558__boxed_204_; uint8_t v_suppressElabErrors_boxed_205_; uint8_t v_res_206_; lean_object* v_r_207_; 
v___y_12558__boxed_204_ = lean_unbox(v___y_201_);
v_suppressElabErrors_boxed_205_ = lean_unbox(v_suppressElabErrors_202_);
v_res_206_ = lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0(v___y_12558__boxed_204_, v_suppressElabErrors_boxed_205_, v_x_203_);
lean_dec(v_x_203_);
v_r_207_ = lean_box(v_res_206_);
return v_r_207_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5_spec__10(lean_object* v_opts_208_, lean_object* v_opt_209_){
_start:
{
lean_object* v_name_210_; lean_object* v_defValue_211_; lean_object* v_map_212_; lean_object* v___x_213_; 
v_name_210_ = lean_ctor_get(v_opt_209_, 0);
v_defValue_211_ = lean_ctor_get(v_opt_209_, 1);
v_map_212_ = lean_ctor_get(v_opts_208_, 0);
v___x_213_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_212_, v_name_210_);
if (lean_obj_tag(v___x_213_) == 0)
{
uint8_t v___x_214_; 
v___x_214_ = lean_unbox(v_defValue_211_);
return v___x_214_;
}
else
{
lean_object* v_val_215_; 
v_val_215_ = lean_ctor_get(v___x_213_, 0);
lean_inc(v_val_215_);
lean_dec_ref_known(v___x_213_, 1);
if (lean_obj_tag(v_val_215_) == 1)
{
uint8_t v_v_216_; 
v_v_216_ = lean_ctor_get_uint8(v_val_215_, 0);
lean_dec_ref_known(v_val_215_, 0);
return v_v_216_;
}
else
{
uint8_t v___x_217_; 
lean_dec(v_val_215_);
v___x_217_ = lean_unbox(v_defValue_211_);
return v___x_217_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5_spec__10___boxed(lean_object* v_opts_218_, lean_object* v_opt_219_){
_start:
{
uint8_t v_res_220_; lean_object* v_r_221_; 
v_res_220_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5_spec__10(v_opts_218_, v_opt_219_);
lean_dec_ref(v_opt_219_);
lean_dec_ref(v_opts_218_);
v_r_221_ = lean_box(v_res_220_);
return v_r_221_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg(lean_object* v_ref_223_, lean_object* v_msgData_224_, uint8_t v_severity_225_, uint8_t v_isSilent_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___y_233_; lean_object* v___y_234_; lean_object* v___y_235_; uint8_t v___y_236_; uint8_t v___y_237_; lean_object* v___y_238_; lean_object* v___y_239_; lean_object* v___y_240_; lean_object* v___y_241_; lean_object* v___y_269_; lean_object* v___y_270_; uint8_t v___y_271_; uint8_t v___y_272_; uint8_t v___y_273_; lean_object* v___y_274_; lean_object* v___y_275_; lean_object* v___y_276_; lean_object* v___y_294_; lean_object* v___y_295_; uint8_t v___y_296_; uint8_t v___y_297_; lean_object* v___y_298_; lean_object* v___y_299_; uint8_t v___y_300_; lean_object* v___y_301_; lean_object* v___y_305_; lean_object* v___y_306_; lean_object* v___y_307_; uint8_t v___y_308_; uint8_t v___y_309_; lean_object* v___y_310_; uint8_t v___y_311_; uint8_t v___x_316_; lean_object* v___y_318_; lean_object* v___y_319_; lean_object* v___y_320_; uint8_t v___y_321_; lean_object* v___y_322_; uint8_t v___y_323_; uint8_t v___y_324_; uint8_t v___y_326_; uint8_t v___x_341_; 
v___x_316_ = 2;
v___x_341_ = l_Lean_instBEqMessageSeverity_beq(v_severity_225_, v___x_316_);
if (v___x_341_ == 0)
{
v___y_326_ = v___x_341_;
goto v___jp_325_;
}
else
{
uint8_t v___x_342_; 
lean_inc_ref(v_msgData_224_);
v___x_342_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_224_);
v___y_326_ = v___x_342_;
goto v___jp_325_;
}
v___jp_232_:
{
lean_object* v___x_242_; lean_object* v_currNamespace_243_; lean_object* v_openDecls_244_; lean_object* v_env_245_; lean_object* v_nextMacroScope_246_; lean_object* v_ngen_247_; lean_object* v_auxDeclNGen_248_; lean_object* v_traceState_249_; lean_object* v_cache_250_; lean_object* v_messages_251_; lean_object* v_infoState_252_; lean_object* v_snapshotTasks_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_267_; 
v___x_242_ = lean_st_ref_take(v___y_241_);
v_currNamespace_243_ = lean_ctor_get(v___y_240_, 6);
v_openDecls_244_ = lean_ctor_get(v___y_240_, 7);
v_env_245_ = lean_ctor_get(v___x_242_, 0);
v_nextMacroScope_246_ = lean_ctor_get(v___x_242_, 1);
v_ngen_247_ = lean_ctor_get(v___x_242_, 2);
v_auxDeclNGen_248_ = lean_ctor_get(v___x_242_, 3);
v_traceState_249_ = lean_ctor_get(v___x_242_, 4);
v_cache_250_ = lean_ctor_get(v___x_242_, 5);
v_messages_251_ = lean_ctor_get(v___x_242_, 6);
v_infoState_252_ = lean_ctor_get(v___x_242_, 7);
v_snapshotTasks_253_ = lean_ctor_get(v___x_242_, 8);
v_isSharedCheck_267_ = !lean_is_exclusive(v___x_242_);
if (v_isSharedCheck_267_ == 0)
{
v___x_255_ = v___x_242_;
v_isShared_256_ = v_isSharedCheck_267_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_snapshotTasks_253_);
lean_inc(v_infoState_252_);
lean_inc(v_messages_251_);
lean_inc(v_cache_250_);
lean_inc(v_traceState_249_);
lean_inc(v_auxDeclNGen_248_);
lean_inc(v_ngen_247_);
lean_inc(v_nextMacroScope_246_);
lean_inc(v_env_245_);
lean_dec(v___x_242_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_267_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_262_; 
lean_inc(v_openDecls_244_);
lean_inc(v_currNamespace_243_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v_currNamespace_243_);
lean_ctor_set(v___x_257_, 1, v_openDecls_244_);
v___x_258_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v___y_233_);
lean_inc_ref(v___y_238_);
lean_inc_ref(v___y_234_);
v___x_259_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_259_, 0, v___y_234_);
lean_ctor_set(v___x_259_, 1, v___y_239_);
lean_ctor_set(v___x_259_, 2, v___y_235_);
lean_ctor_set(v___x_259_, 3, v___y_238_);
lean_ctor_set(v___x_259_, 4, v___x_258_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*5, v___y_237_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*5 + 1, v___y_236_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*5 + 2, v_isSilent_226_);
v___x_260_ = l_Lean_MessageLog_add(v___x_259_, v_messages_251_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 6, v___x_260_);
v___x_262_ = v___x_255_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_266_; 
v_reuseFailAlloc_266_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_266_, 0, v_env_245_);
lean_ctor_set(v_reuseFailAlloc_266_, 1, v_nextMacroScope_246_);
lean_ctor_set(v_reuseFailAlloc_266_, 2, v_ngen_247_);
lean_ctor_set(v_reuseFailAlloc_266_, 3, v_auxDeclNGen_248_);
lean_ctor_set(v_reuseFailAlloc_266_, 4, v_traceState_249_);
lean_ctor_set(v_reuseFailAlloc_266_, 5, v_cache_250_);
lean_ctor_set(v_reuseFailAlloc_266_, 6, v___x_260_);
lean_ctor_set(v_reuseFailAlloc_266_, 7, v_infoState_252_);
lean_ctor_set(v_reuseFailAlloc_266_, 8, v_snapshotTasks_253_);
v___x_262_ = v_reuseFailAlloc_266_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_263_ = lean_st_ref_set(v___y_241_, v___x_262_);
v___x_264_ = lean_box(0);
v___x_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
return v___x_265_;
}
}
}
v___jp_268_:
{
lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v_a_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_292_; 
v___x_277_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_224_);
v___x_278_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4_spec__7(v___x_277_, v___y_227_, v___y_228_, v___y_229_, v___y_230_);
v_a_279_ = lean_ctor_get(v___x_278_, 0);
v_isSharedCheck_292_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_292_ == 0)
{
v___x_281_ = v___x_278_;
v_isShared_282_ = v_isSharedCheck_292_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_a_279_);
lean_dec(v___x_278_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_292_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
lean_inc_ref_n(v___y_274_, 2);
v___x_283_ = l_Lean_FileMap_toPosition(v___y_274_, v___y_275_);
lean_dec(v___y_275_);
v___x_284_ = l_Lean_FileMap_toPosition(v___y_274_, v___y_276_);
lean_dec(v___y_276_);
v___x_285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_285_, 0, v___x_284_);
v___x_286_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___closed__0));
if (v___y_271_ == 0)
{
lean_del_object(v___x_281_);
lean_dec_ref(v___y_269_);
v___y_233_ = v_a_279_;
v___y_234_ = v___y_270_;
v___y_235_ = v___x_285_;
v___y_236_ = v___y_272_;
v___y_237_ = v___y_273_;
v___y_238_ = v___x_286_;
v___y_239_ = v___x_283_;
v___y_240_ = v___y_229_;
v___y_241_ = v___y_230_;
goto v___jp_232_;
}
else
{
uint8_t v___x_287_; 
lean_inc(v_a_279_);
v___x_287_ = l_Lean_MessageData_hasTag(v___y_269_, v_a_279_);
if (v___x_287_ == 0)
{
lean_object* v___x_288_; lean_object* v___x_290_; 
lean_dec_ref_known(v___x_285_, 1);
lean_dec_ref(v___x_283_);
lean_dec(v_a_279_);
v___x_288_ = lean_box(0);
if (v_isShared_282_ == 0)
{
lean_ctor_set(v___x_281_, 0, v___x_288_);
v___x_290_ = v___x_281_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v___x_288_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
else
{
lean_del_object(v___x_281_);
v___y_233_ = v_a_279_;
v___y_234_ = v___y_270_;
v___y_235_ = v___x_285_;
v___y_236_ = v___y_272_;
v___y_237_ = v___y_273_;
v___y_238_ = v___x_286_;
v___y_239_ = v___x_283_;
v___y_240_ = v___y_229_;
v___y_241_ = v___y_230_;
goto v___jp_232_;
}
}
}
}
v___jp_293_:
{
lean_object* v___x_302_; 
v___x_302_ = l_Lean_Syntax_getTailPos_x3f(v___y_298_, v___y_300_);
lean_dec(v___y_298_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_inc(v___y_301_);
v___y_269_ = v___y_294_;
v___y_270_ = v___y_295_;
v___y_271_ = v___y_297_;
v___y_272_ = v___y_296_;
v___y_273_ = v___y_300_;
v___y_274_ = v___y_299_;
v___y_275_ = v___y_301_;
v___y_276_ = v___y_301_;
goto v___jp_268_;
}
else
{
lean_object* v_val_303_; 
v_val_303_ = lean_ctor_get(v___x_302_, 0);
lean_inc(v_val_303_);
lean_dec_ref_known(v___x_302_, 1);
v___y_269_ = v___y_294_;
v___y_270_ = v___y_295_;
v___y_271_ = v___y_297_;
v___y_272_ = v___y_296_;
v___y_273_ = v___y_300_;
v___y_274_ = v___y_299_;
v___y_275_ = v___y_301_;
v___y_276_ = v_val_303_;
goto v___jp_268_;
}
}
v___jp_304_:
{
lean_object* v_ref_312_; lean_object* v___x_313_; 
v_ref_312_ = l_Lean_replaceRef(v_ref_223_, v___y_306_);
v___x_313_ = l_Lean_Syntax_getPos_x3f(v_ref_312_, v___y_309_);
if (lean_obj_tag(v___x_313_) == 0)
{
lean_object* v___x_314_; 
v___x_314_ = lean_unsigned_to_nat(0u);
v___y_294_ = v___y_305_;
v___y_295_ = v___y_307_;
v___y_296_ = v___y_311_;
v___y_297_ = v___y_308_;
v___y_298_ = v_ref_312_;
v___y_299_ = v___y_310_;
v___y_300_ = v___y_309_;
v___y_301_ = v___x_314_;
goto v___jp_293_;
}
else
{
lean_object* v_val_315_; 
v_val_315_ = lean_ctor_get(v___x_313_, 0);
lean_inc(v_val_315_);
lean_dec_ref_known(v___x_313_, 1);
v___y_294_ = v___y_305_;
v___y_295_ = v___y_307_;
v___y_296_ = v___y_311_;
v___y_297_ = v___y_308_;
v___y_298_ = v_ref_312_;
v___y_299_ = v___y_310_;
v___y_300_ = v___y_309_;
v___y_301_ = v_val_315_;
goto v___jp_293_;
}
}
v___jp_317_:
{
if (v___y_324_ == 0)
{
v___y_305_ = v___y_319_;
v___y_306_ = v___y_318_;
v___y_307_ = v___y_320_;
v___y_308_ = v___y_321_;
v___y_309_ = v___y_323_;
v___y_310_ = v___y_322_;
v___y_311_ = v_severity_225_;
goto v___jp_304_;
}
else
{
v___y_305_ = v___y_319_;
v___y_306_ = v___y_318_;
v___y_307_ = v___y_320_;
v___y_308_ = v___y_321_;
v___y_309_ = v___y_323_;
v___y_310_ = v___y_322_;
v___y_311_ = v___x_316_;
goto v___jp_304_;
}
}
v___jp_325_:
{
if (v___y_326_ == 0)
{
lean_object* v_fileName_327_; lean_object* v_fileMap_328_; lean_object* v_options_329_; lean_object* v_ref_330_; uint8_t v_suppressElabErrors_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___f_334_; uint8_t v___x_335_; uint8_t v___x_336_; 
v_fileName_327_ = lean_ctor_get(v___y_229_, 0);
v_fileMap_328_ = lean_ctor_get(v___y_229_, 1);
v_options_329_ = lean_ctor_get(v___y_229_, 2);
v_ref_330_ = lean_ctor_get(v___y_229_, 5);
v_suppressElabErrors_331_ = lean_ctor_get_uint8(v___y_229_, sizeof(void*)*14 + 1);
v___x_332_ = lean_box(v___y_326_);
v___x_333_ = lean_box(v_suppressElabErrors_331_);
v___f_334_ = lean_alloc_closure((void*)(lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_334_, 0, v___x_332_);
lean_closure_set(v___f_334_, 1, v___x_333_);
v___x_335_ = 1;
v___x_336_ = l_Lean_instBEqMessageSeverity_beq(v_severity_225_, v___x_335_);
if (v___x_336_ == 0)
{
v___y_318_ = v_ref_330_;
v___y_319_ = v___f_334_;
v___y_320_ = v_fileName_327_;
v___y_321_ = v_suppressElabErrors_331_;
v___y_322_ = v_fileMap_328_;
v___y_323_ = v___y_326_;
v___y_324_ = v___x_336_;
goto v___jp_317_;
}
else
{
lean_object* v___x_337_; uint8_t v___x_338_; 
v___x_337_ = l_Lean_warningAsError;
v___x_338_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5_spec__10(v_options_329_, v___x_337_);
v___y_318_ = v_ref_330_;
v___y_319_ = v___f_334_;
v___y_320_ = v_fileName_327_;
v___y_321_ = v_suppressElabErrors_331_;
v___y_322_ = v_fileMap_328_;
v___y_323_ = v___y_326_;
v___y_324_ = v___x_338_;
goto v___jp_317_;
}
}
else
{
lean_object* v___x_339_; lean_object* v___x_340_; 
lean_dec_ref(v_msgData_224_);
v___x_339_ = lean_box(0);
v___x_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
return v___x_340_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_ref_343_, lean_object* v_msgData_344_, lean_object* v_severity_345_, lean_object* v_isSilent_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_){
_start:
{
uint8_t v_severity_boxed_352_; uint8_t v_isSilent_boxed_353_; lean_object* v_res_354_; 
v_severity_boxed_352_ = lean_unbox(v_severity_345_);
v_isSilent_boxed_353_ = lean_unbox(v_isSilent_346_);
v_res_354_ = lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg(v_ref_343_, v_msgData_344_, v_severity_boxed_352_, v_isSilent_boxed_353_, v___y_347_, v___y_348_, v___y_349_, v___y_350_);
lean_dec(v___y_350_);
lean_dec_ref(v___y_349_);
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
lean_dec(v_ref_343_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3(lean_object* v_ref_355_, lean_object* v_msgData_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_){
_start:
{
uint8_t v___x_366_; uint8_t v___x_367_; lean_object* v___x_368_; 
v___x_366_ = 2;
v___x_367_ = 0;
v___x_368_ = lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg(v_ref_355_, v_msgData_356_, v___x_366_, v___x_367_, v___y_361_, v___y_362_, v___y_363_, v___y_364_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3___boxed(lean_object* v_ref_369_, lean_object* v_msgData_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_batteries_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3(v_ref_369_, v_msgData_370_, v___y_371_, v___y_372_, v___y_373_, v___y_374_, v___y_375_, v___y_376_, v___y_377_, v___y_378_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
lean_dec(v___y_376_);
lean_dec_ref(v___y_375_);
lean_dec(v___y_374_);
lean_dec_ref(v___y_373_);
lean_dec(v___y_372_);
lean_dec_ref(v___y_371_);
lean_dec(v_ref_369_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4_spec__7(lean_object* v_msgData_381_, uint8_t v_severity_382_, uint8_t v_isSilent_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_){
_start:
{
lean_object* v_ref_393_; lean_object* v___x_394_; 
v_ref_393_ = lean_ctor_get(v___y_390_, 5);
v___x_394_ = lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg(v_ref_393_, v_msgData_381_, v_severity_382_, v_isSilent_383_, v___y_388_, v___y_389_, v___y_390_, v___y_391_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4_spec__7___boxed(lean_object* v_msgData_395_, lean_object* v_severity_396_, lean_object* v_isSilent_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
uint8_t v_severity_boxed_407_; uint8_t v_isSilent_boxed_408_; lean_object* v_res_409_; 
v_severity_boxed_407_ = lean_unbox(v_severity_396_);
v_isSilent_boxed_408_ = lean_unbox(v_isSilent_397_);
v_res_409_ = lp_batteries_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4_spec__7(v_msgData_395_, v_severity_boxed_407_, v_isSilent_boxed_408_, v___y_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4(lean_object* v_msgData_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_){
_start:
{
uint8_t v___x_420_; uint8_t v___x_421_; lean_object* v___x_422_; 
v___x_420_ = 2;
v___x_421_ = 0;
v___x_422_ = lp_batteries_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4_spec__7(v_msgData_410_, v___x_420_, v___x_421_, v___y_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4___boxed(lean_object* v_msgData_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_){
_start:
{
lean_object* v_res_433_; 
v_res_433_ = lp_batteries_Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4(v_msgData_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_);
lean_dec(v___y_431_);
lean_dec_ref(v___y_430_);
lean_dec(v___y_429_);
lean_dec_ref(v___y_428_);
lean_dec(v___y_427_);
lean_dec_ref(v___y_426_);
lean_dec(v___y_425_);
lean_dec_ref(v___y_424_);
return v_res_433_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_435_; lean_object* v___x_436_; 
v___x_435_ = ((lean_object*)(lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__0));
v___x_436_ = l_Lean_stringToMessageData(v___x_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2(lean_object* v_ex_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
if (lean_obj_tag(v_ex_437_) == 0)
{
lean_object* v_ref_447_; lean_object* v_msg_448_; lean_object* v___x_449_; 
v_ref_447_ = lean_ctor_get(v_ex_437_, 0);
lean_inc(v_ref_447_);
v_msg_448_ = lean_ctor_get(v_ex_437_, 1);
lean_inc_ref(v_msg_448_);
lean_dec_ref_known(v_ex_437_, 2);
v___x_449_ = lp_batteries_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3(v_ref_447_, v_msg_448_, v___y_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_);
lean_dec(v_ref_447_);
return v___x_449_;
}
else
{
lean_object* v_id_450_; uint8_t v___y_452_; uint8_t v___x_474_; 
v_id_450_ = lean_ctor_get(v_ex_437_, 0);
lean_inc(v_id_450_);
v___x_474_ = l_Lean_Elab_isAbortExceptionId(v_id_450_);
if (v___x_474_ == 0)
{
uint8_t v___x_475_; 
v___x_475_ = l_Lean_Exception_isInterrupt(v_ex_437_);
lean_dec_ref_known(v_ex_437_, 2);
v___y_452_ = v___x_475_;
goto v___jp_451_;
}
else
{
lean_dec_ref_known(v_ex_437_, 2);
v___y_452_ = v___x_474_;
goto v___jp_451_;
}
v___jp_451_:
{
if (v___y_452_ == 0)
{
lean_object* v___x_453_; 
v___x_453_ = l_Lean_InternalExceptionId_getName(v_id_450_);
lean_dec(v_id_450_);
if (lean_obj_tag(v___x_453_) == 0)
{
lean_object* v_a_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v_a_454_ = lean_ctor_get(v___x_453_, 0);
lean_inc(v_a_454_);
lean_dec_ref_known(v___x_453_, 1);
v___x_455_ = lean_obj_once(&lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__1, &lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__1_once, _init_lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___closed__1);
v___x_456_ = l_Lean_MessageData_ofName(v_a_454_);
v___x_457_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_457_, 0, v___x_455_);
lean_ctor_set(v___x_457_, 1, v___x_456_);
v___x_458_ = lp_batteries_Lean_logError___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__4(v___x_457_, v___y_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_);
return v___x_458_;
}
else
{
lean_object* v_a_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_471_; 
v_a_459_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_471_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_471_ == 0)
{
v___x_461_ = v___x_453_;
v_isShared_462_ = v_isSharedCheck_471_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_a_459_);
lean_dec(v___x_453_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_471_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v_ref_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_469_; 
v_ref_463_ = lean_ctor_get(v___y_444_, 5);
v___x_464_ = lean_io_error_to_string(v_a_459_);
v___x_465_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
v___x_466_ = l_Lean_MessageData_ofFormat(v___x_465_);
lean_inc(v_ref_463_);
v___x_467_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_467_, 0, v_ref_463_);
lean_ctor_set(v___x_467_, 1, v___x_466_);
if (v_isShared_462_ == 0)
{
lean_ctor_set(v___x_461_, 0, v___x_467_);
v___x_469_ = v___x_461_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v___x_467_);
v___x_469_ = v_reuseFailAlloc_470_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
return v___x_469_;
}
}
}
}
else
{
lean_object* v___x_472_; lean_object* v___x_473_; 
lean_dec(v_id_450_);
v___x_472_ = lean_box(0);
v___x_473_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_473_, 0, v___x_472_);
return v___x_473_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2___boxed(lean_object* v_ex_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_){
_start:
{
lean_object* v_res_486_; 
v_res_486_ = lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2(v_ex_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_, v___y_482_, v___y_483_, v___y_484_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec(v___y_482_);
lean_dec_ref(v___y_481_);
lean_dec(v___y_480_);
lean_dec_ref(v___y_479_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__3(lean_object* v_as_487_, size_t v_sz_488_, size_t v_i_489_, lean_object* v_b_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_){
_start:
{
lean_object* v_a_501_; uint8_t v___x_505_; 
v___x_505_ = lean_usize_dec_lt(v_i_489_, v_sz_488_);
if (v___x_505_ == 0)
{
lean_object* v___x_506_; 
v___x_506_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_506_, 0, v_b_490_);
return v___x_506_;
}
else
{
lean_object* v_snd_507_; 
v_snd_507_ = lean_ctor_get(v_b_490_, 1);
lean_inc(v_snd_507_);
if (lean_obj_tag(v_snd_507_) == 0)
{
lean_object* v_fst_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_516_; 
v_fst_508_ = lean_ctor_get(v_b_490_, 0);
v_isSharedCheck_516_ = !lean_is_exclusive(v_b_490_);
if (v_isSharedCheck_516_ == 0)
{
lean_object* v_unused_517_; 
v_unused_517_ = lean_ctor_get(v_b_490_, 1);
lean_dec(v_unused_517_);
v___x_510_ = v_b_490_;
v_isShared_511_ = v_isSharedCheck_516_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_fst_508_);
lean_dec(v_b_490_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_516_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v___x_513_; 
if (v_isShared_511_ == 0)
{
v___x_513_ = v___x_510_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v_fst_508_);
lean_ctor_set(v_reuseFailAlloc_515_, 1, v_snd_507_);
v___x_513_ = v_reuseFailAlloc_515_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
lean_object* v___x_514_; 
v___x_514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_514_, 0, v___x_513_);
return v___x_514_;
}
}
}
else
{
lean_object* v_fst_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_617_; 
v_fst_518_ = lean_ctor_get(v_b_490_, 0);
v_isSharedCheck_617_ = !lean_is_exclusive(v_b_490_);
if (v_isSharedCheck_617_ == 0)
{
lean_object* v_unused_618_; 
v_unused_618_ = lean_ctor_get(v_b_490_, 1);
lean_dec(v_unused_618_);
v___x_520_ = v_b_490_;
v_isShared_521_ = v_isSharedCheck_617_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_fst_518_);
lean_dec(v_b_490_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_617_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v_head_522_; lean_object* v_tail_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_616_; 
v_head_522_ = lean_ctor_get(v_snd_507_, 0);
v_tail_523_ = lean_ctor_get(v_snd_507_, 1);
v_isSharedCheck_616_ = !lean_is_exclusive(v_snd_507_);
if (v_isSharedCheck_616_ == 0)
{
v___x_525_ = v_snd_507_;
v_isShared_526_ = v_isSharedCheck_616_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_tail_523_);
lean_inc(v_head_522_);
lean_dec(v_snd_507_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_616_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v_snd_528_; lean_object* v___x_532_; 
v___x_532_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___redArg(v_head_522_, v___y_496_);
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v_a_533_; uint8_t v___x_534_; 
v_a_533_ = lean_ctor_get(v___x_532_, 0);
lean_inc(v_a_533_);
lean_dec_ref_known(v___x_532_, 1);
v___x_534_ = lean_unbox(v_a_533_);
lean_dec(v_a_533_);
if (v___x_534_ == 0)
{
lean_object* v___x_535_; lean_object* v___x_537_; 
v___x_535_ = lean_box(0);
lean_inc(v_head_522_);
if (v_isShared_526_ == 0)
{
lean_ctor_set(v___x_525_, 1, v___x_535_);
v___x_537_ = v___x_525_;
goto v_reusejp_536_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v_head_522_);
lean_ctor_set(v_reuseFailAlloc_606_, 1, v___x_535_);
v___x_537_ = v_reuseFailAlloc_606_;
goto v_reusejp_536_;
}
v_reusejp_536_:
{
lean_object* v___x_538_; 
v___x_538_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_537_, v___y_492_);
if (lean_obj_tag(v___x_538_) == 0)
{
lean_object* v___x_539_; 
lean_dec_ref_known(v___x_538_, 1);
v___x_539_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_492_, v___y_494_, v___y_496_, v___y_498_);
if (lean_obj_tag(v___x_539_) == 0)
{
lean_object* v_a_540_; lean_object* v___x_542_; uint8_t v_isShared_543_; uint8_t v_isSharedCheck_589_; 
v_a_540_ = lean_ctor_get(v___x_539_, 0);
v_isSharedCheck_589_ = !lean_is_exclusive(v___x_539_);
if (v_isSharedCheck_589_ == 0)
{
v___x_542_ = v___x_539_;
v_isShared_543_ = v_isSharedCheck_589_;
goto v_resetjp_541_;
}
else
{
lean_inc(v_a_540_);
lean_dec(v___x_539_);
v___x_542_ = lean_box(0);
v_isShared_543_ = v_isSharedCheck_589_;
goto v_resetjp_541_;
}
v_resetjp_541_:
{
lean_object* v___y_545_; uint8_t v___y_546_; lean_object* v_a_579_; lean_object* v_a_582_; lean_object* v___x_583_; 
v_a_582_ = lean_array_uget_borrowed(v_as_487_, v_i_489_);
lean_inc(v_a_582_);
v___x_583_ = l_Lean_Elab_Tactic_evalTactic(v_a_582_, v___y_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
if (lean_obj_tag(v___x_583_) == 0)
{
lean_object* v___x_584_; 
lean_dec_ref_known(v___x_583_, 1);
v___x_584_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___y_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
if (lean_obj_tag(v___x_584_) == 0)
{
lean_object* v_a_585_; lean_object* v___x_586_; 
lean_del_object(v___x_542_);
lean_dec(v_a_540_);
lean_dec(v_head_522_);
v_a_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_a_585_);
lean_dec_ref_known(v___x_584_, 1);
v___x_586_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_fst_518_, v_a_585_);
v_snd_528_ = v___x_586_;
goto v___jp_527_;
}
else
{
lean_object* v_a_587_; 
v_a_587_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_a_587_);
lean_dec_ref_known(v___x_584_, 1);
v_a_579_ = v_a_587_;
goto v___jp_578_;
}
}
else
{
lean_object* v_a_588_; 
v_a_588_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_a_588_);
lean_dec_ref_known(v___x_583_, 1);
v_a_579_ = v_a_588_;
goto v___jp_578_;
}
v___jp_544_:
{
if (v___y_546_ == 0)
{
lean_object* v___x_547_; 
lean_del_object(v___x_542_);
v___x_547_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_540_, v___y_546_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
if (lean_obj_tag(v___x_547_) == 0)
{
lean_object* v___x_549_; uint8_t v_isShared_550_; uint8_t v_isSharedCheck_565_; 
v_isSharedCheck_565_ = !lean_is_exclusive(v___x_547_);
if (v_isSharedCheck_565_ == 0)
{
lean_object* v_unused_566_; 
v_unused_566_ = lean_ctor_get(v___x_547_, 0);
lean_dec(v_unused_566_);
v___x_549_ = v___x_547_;
v_isShared_550_ = v_isSharedCheck_565_;
goto v_resetjp_548_;
}
else
{
lean_dec(v___x_547_);
v___x_549_ = lean_box(0);
v_isShared_550_ = v_isSharedCheck_565_;
goto v_resetjp_548_;
}
v_resetjp_548_:
{
uint8_t v_recover_551_; 
v_recover_551_ = lean_ctor_get_uint8(v___y_491_, sizeof(void*)*1);
if (v_recover_551_ == 0)
{
lean_object* v___x_553_; 
lean_dec(v_tail_523_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
lean_dec(v_fst_518_);
if (v_isShared_550_ == 0)
{
lean_ctor_set_tag(v___x_549_, 1);
lean_ctor_set(v___x_549_, 0, v___y_545_);
v___x_553_ = v___x_549_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v___y_545_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
else
{
lean_object* v___x_555_; 
lean_del_object(v___x_549_);
v___x_555_ = lp_batteries_Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2(v___y_545_, v___y_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
if (lean_obj_tag(v___x_555_) == 0)
{
lean_object* v___x_556_; 
lean_dec_ref_known(v___x_555_, 1);
v___x_556_ = lean_array_push(v_fst_518_, v_head_522_);
v_snd_528_ = v___x_556_;
goto v___jp_527_;
}
else
{
lean_object* v_a_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_564_; 
lean_dec(v_tail_523_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
lean_dec(v_fst_518_);
v_a_557_ = lean_ctor_get(v___x_555_, 0);
v_isSharedCheck_564_ = !lean_is_exclusive(v___x_555_);
if (v_isSharedCheck_564_ == 0)
{
v___x_559_ = v___x_555_;
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_a_557_);
lean_dec(v___x_555_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
lean_object* v___x_562_; 
if (v_isShared_560_ == 0)
{
v___x_562_ = v___x_559_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v_a_557_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
}
}
}
else
{
lean_object* v_a_567_; lean_object* v___x_569_; uint8_t v_isShared_570_; uint8_t v_isSharedCheck_574_; 
lean_dec_ref(v___y_545_);
lean_dec(v_tail_523_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
lean_dec(v_fst_518_);
v_a_567_ = lean_ctor_get(v___x_547_, 0);
v_isSharedCheck_574_ = !lean_is_exclusive(v___x_547_);
if (v_isSharedCheck_574_ == 0)
{
v___x_569_ = v___x_547_;
v_isShared_570_ = v_isSharedCheck_574_;
goto v_resetjp_568_;
}
else
{
lean_inc(v_a_567_);
lean_dec(v___x_547_);
v___x_569_ = lean_box(0);
v_isShared_570_ = v_isSharedCheck_574_;
goto v_resetjp_568_;
}
v_resetjp_568_:
{
lean_object* v___x_572_; 
if (v_isShared_570_ == 0)
{
v___x_572_ = v___x_569_;
goto v_reusejp_571_;
}
else
{
lean_object* v_reuseFailAlloc_573_; 
v_reuseFailAlloc_573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_573_, 0, v_a_567_);
v___x_572_ = v_reuseFailAlloc_573_;
goto v_reusejp_571_;
}
v_reusejp_571_:
{
return v___x_572_;
}
}
}
}
else
{
lean_object* v___x_576_; 
lean_dec(v_a_540_);
lean_dec(v_tail_523_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
lean_dec(v_fst_518_);
if (v_isShared_543_ == 0)
{
lean_ctor_set_tag(v___x_542_, 1);
lean_ctor_set(v___x_542_, 0, v___y_545_);
v___x_576_ = v___x_542_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_577_; 
v_reuseFailAlloc_577_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_577_, 0, v___y_545_);
v___x_576_ = v_reuseFailAlloc_577_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
return v___x_576_;
}
}
}
v___jp_578_:
{
uint8_t v___x_580_; 
v___x_580_ = l_Lean_Exception_isInterrupt(v_a_579_);
if (v___x_580_ == 0)
{
uint8_t v___x_581_; 
lean_inc_ref(v_a_579_);
v___x_581_ = l_Lean_Exception_isRuntime(v_a_579_);
v___y_545_ = v_a_579_;
v___y_546_ = v___x_581_;
goto v___jp_544_;
}
else
{
v___y_545_ = v_a_579_;
v___y_546_ = v___x_580_;
goto v___jp_544_;
}
}
}
}
else
{
lean_object* v_a_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_597_; 
lean_dec(v_tail_523_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
lean_dec(v_fst_518_);
v_a_590_ = lean_ctor_get(v___x_539_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_539_);
if (v_isSharedCheck_597_ == 0)
{
v___x_592_ = v___x_539_;
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_a_590_);
lean_dec(v___x_539_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
lean_object* v___x_595_; 
if (v_isShared_593_ == 0)
{
v___x_595_ = v___x_592_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v_a_590_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
}
else
{
lean_object* v_a_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_605_; 
lean_dec(v_tail_523_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
lean_dec(v_fst_518_);
v_a_598_ = lean_ctor_get(v___x_538_, 0);
v_isSharedCheck_605_ = !lean_is_exclusive(v___x_538_);
if (v_isSharedCheck_605_ == 0)
{
v___x_600_ = v___x_538_;
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_a_598_);
lean_dec(v___x_538_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
lean_object* v___x_603_; 
if (v_isShared_601_ == 0)
{
v___x_603_ = v___x_600_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v_a_598_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
}
}
else
{
lean_object* v___x_607_; 
lean_del_object(v___x_525_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
v___x_607_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_607_, 0, v_fst_518_);
lean_ctor_set(v___x_607_, 1, v_tail_523_);
v_a_501_ = v___x_607_;
goto v___jp_500_;
}
}
else
{
lean_object* v_a_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_615_; 
lean_del_object(v___x_525_);
lean_dec(v_tail_523_);
lean_dec(v_head_522_);
lean_del_object(v___x_520_);
lean_dec(v_fst_518_);
v_a_608_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_615_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_615_ == 0)
{
v___x_610_ = v___x_532_;
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_a_608_);
lean_dec(v___x_532_);
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
v___jp_527_:
{
lean_object* v___x_530_; 
if (v_isShared_521_ == 0)
{
lean_ctor_set(v___x_520_, 1, v_tail_523_);
lean_ctor_set(v___x_520_, 0, v_snd_528_);
v___x_530_ = v___x_520_;
goto v_reusejp_529_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v_snd_528_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v_tail_523_);
v___x_530_ = v_reuseFailAlloc_531_;
goto v_reusejp_529_;
}
v_reusejp_529_:
{
v_a_501_ = v___x_530_;
goto v___jp_500_;
}
}
}
}
}
}
v___jp_500_:
{
size_t v___x_502_; size_t v___x_503_; 
v___x_502_ = ((size_t)1ULL);
v___x_503_ = lean_usize_add(v_i_489_, v___x_502_);
v_i_489_ = v___x_503_;
v_b_490_ = v_a_501_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__3___boxed(lean_object* v_as_619_, lean_object* v_sz_620_, lean_object* v_i_621_, lean_object* v_b_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_){
_start:
{
size_t v_sz_boxed_632_; size_t v_i_boxed_633_; lean_object* v_res_634_; 
v_sz_boxed_632_ = lean_unbox_usize(v_sz_620_);
lean_dec(v_sz_620_);
v_i_boxed_633_ = lean_unbox_usize(v_i_621_);
lean_dec(v_i_621_);
v_res_634_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__3(v_as_619_, v_sz_boxed_632_, v_i_boxed_633_, v_b_622_, v___y_623_, v___y_624_, v___y_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_, v___y_630_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
lean_dec(v___y_628_);
lean_dec_ref(v___y_627_);
lean_dec(v___y_626_);
lean_dec_ref(v___y_625_);
lean_dec(v___y_624_);
lean_dec_ref(v___y_623_);
lean_dec_ref(v_as_619_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg(lean_object* v_msg_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
lean_object* v_ref_641_; lean_object* v___x_642_; lean_object* v_a_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_651_; 
v_ref_641_ = lean_ctor_get(v___y_638_, 5);
v___x_642_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4_spec__7(v_msg_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_);
v_a_643_ = lean_ctor_get(v___x_642_, 0);
v_isSharedCheck_651_ = !lean_is_exclusive(v___x_642_);
if (v_isSharedCheck_651_ == 0)
{
v___x_645_ = v___x_642_;
v_isShared_646_ = v_isSharedCheck_651_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_a_643_);
lean_dec(v___x_642_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_651_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v___x_647_; lean_object* v___x_649_; 
lean_inc(v_ref_641_);
v___x_647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_647_, 0, v_ref_641_);
lean_ctor_set(v___x_647_, 1, v_a_643_);
if (v_isShared_646_ == 0)
{
lean_ctor_set_tag(v___x_645_, 1);
lean_ctor_set(v___x_645_, 0, v___x_647_);
v___x_649_ = v___x_645_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_647_);
v___x_649_ = v_reuseFailAlloc_650_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
return v___x_649_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg___boxed(lean_object* v_msg_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_){
_start:
{
lean_object* v_res_658_; 
v_res_658_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg(v_msg_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_);
lean_dec(v___y_656_);
lean_dec_ref(v___y_655_);
lean_dec(v___y_654_);
lean_dec_ref(v___y_653_);
return v_res_658_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__2(void){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_662_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__1));
v___x_663_ = l_Lean_stringToMessageData(v___x_662_);
return v___x_663_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__4(void){
_start:
{
lean_object* v___x_665_; lean_object* v___x_666_; 
v___x_665_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__3));
v___x_666_ = l_Lean_stringToMessageData(v___x_665_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1(lean_object* v_x_667_, lean_object* v_a_668_, lean_object* v_a_669_, lean_object* v_a_670_, lean_object* v_a_671_, lean_object* v_a_672_, lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_){
_start:
{
lean_object* v___x_677_; uint8_t v___x_678_; 
v___x_677_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3));
lean_inc(v_x_667_);
v___x_678_ = l_Lean_Syntax_isOfKind(v_x_667_, v___x_677_);
if (v___x_678_ == 0)
{
lean_object* v___x_679_; 
lean_dec(v_x_667_);
v___x_679_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__0___redArg();
return v___x_679_;
}
else
{
lean_object* v___x_680_; 
v___x_680_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v_a_668_, v_a_669_, v_a_670_, v_a_671_, v_a_672_, v_a_673_, v_a_674_, v_a_675_);
if (lean_obj_tag(v___x_680_) == 0)
{
lean_object* v_a_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v_ts_684_; lean_object* v___x_685_; lean_object* v___y_687_; lean_object* v___y_688_; lean_object* v___y_689_; lean_object* v___y_690_; lean_object* v___y_691_; lean_object* v___y_692_; lean_object* v___y_693_; lean_object* v___y_694_; lean_object* v___x_712_; lean_object* v___x_713_; uint8_t v___x_714_; 
v_a_681_ = lean_ctor_get(v___x_680_, 0);
lean_inc(v_a_681_);
lean_dec_ref_known(v___x_680_, 1);
v___x_682_ = lean_unsigned_to_nat(2u);
v___x_683_ = l_Lean_Syntax_getArg(v_x_667_, v___x_682_);
lean_dec(v_x_667_);
v_ts_684_ = l_Lean_Syntax_getArgs(v___x_683_);
lean_dec(v___x_683_);
v___x_685_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_ts_684_);
lean_dec_ref(v_ts_684_);
v___x_712_ = lean_array_get_size(v___x_685_);
v___x_713_ = l_List_lengthTR___redArg(v_a_681_);
v___x_714_ = lean_nat_dec_lt(v___x_712_, v___x_713_);
if (v___x_714_ == 0)
{
uint8_t v___x_715_; 
v___x_715_ = lean_nat_dec_lt(v___x_713_, v___x_712_);
lean_dec(v___x_713_);
if (v___x_715_ == 0)
{
v___y_687_ = v_a_668_;
v___y_688_ = v_a_669_;
v___y_689_ = v_a_670_;
v___y_690_ = v_a_671_;
v___y_691_ = v_a_672_;
v___y_692_ = v_a_673_;
v___y_693_ = v_a_674_;
v___y_694_ = v_a_675_;
goto v___jp_686_;
}
else
{
lean_object* v___x_716_; lean_object* v___x_717_; 
lean_dec_ref(v___x_685_);
lean_dec(v_a_681_);
v___x_716_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__2, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__2_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__2);
v___x_717_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg(v___x_716_, v_a_672_, v_a_673_, v_a_674_, v_a_675_);
return v___x_717_;
}
}
else
{
lean_object* v___x_718_; lean_object* v___x_719_; 
lean_dec(v___x_713_);
lean_dec_ref(v___x_685_);
lean_dec(v_a_681_);
v___x_718_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__4, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__4_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__4);
v___x_719_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg(v___x_718_, v_a_672_, v_a_673_, v_a_674_, v_a_675_);
return v___x_719_;
}
v___jp_686_:
{
lean_object* v___x_695_; lean_object* v___x_696_; size_t v_sz_697_; size_t v___x_698_; lean_object* v___x_699_; 
v___x_695_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___closed__0));
v___x_696_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_696_, 0, v___x_695_);
lean_ctor_set(v___x_696_, 1, v_a_681_);
v_sz_697_ = lean_array_size(v___x_685_);
v___x_698_ = ((size_t)0ULL);
v___x_699_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__3(v___x_685_, v_sz_697_, v___x_698_, v___x_696_, v___y_687_, v___y_688_, v___y_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_);
lean_dec_ref(v___x_685_);
if (lean_obj_tag(v___x_699_) == 0)
{
lean_object* v_a_700_; lean_object* v_fst_701_; lean_object* v___x_702_; lean_object* v___x_703_; 
v_a_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_a_700_);
lean_dec_ref_known(v___x_699_, 1);
v_fst_701_ = lean_ctor_get(v_a_700_, 0);
lean_inc(v_fst_701_);
lean_dec(v_a_700_);
v___x_702_ = lean_array_to_list(v_fst_701_);
v___x_703_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_702_, v___y_688_);
return v___x_703_;
}
else
{
lean_object* v_a_704_; lean_object* v___x_706_; uint8_t v_isShared_707_; uint8_t v_isSharedCheck_711_; 
v_a_704_ = lean_ctor_get(v___x_699_, 0);
v_isSharedCheck_711_ = !lean_is_exclusive(v___x_699_);
if (v_isSharedCheck_711_ == 0)
{
v___x_706_ = v___x_699_;
v_isShared_707_ = v_isSharedCheck_711_;
goto v_resetjp_705_;
}
else
{
lean_inc(v_a_704_);
lean_dec(v___x_699_);
v___x_706_ = lean_box(0);
v_isShared_707_ = v_isSharedCheck_711_;
goto v_resetjp_705_;
}
v_resetjp_705_:
{
lean_object* v___x_709_; 
if (v_isShared_707_ == 0)
{
v___x_709_ = v___x_706_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_710_; 
v_reuseFailAlloc_710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_710_, 0, v_a_704_);
v___x_709_ = v_reuseFailAlloc_710_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
return v___x_709_;
}
}
}
}
}
else
{
lean_object* v_a_720_; lean_object* v___x_722_; uint8_t v_isShared_723_; uint8_t v_isSharedCheck_727_; 
lean_dec(v_x_667_);
v_a_720_ = lean_ctor_get(v___x_680_, 0);
v_isSharedCheck_727_ = !lean_is_exclusive(v___x_680_);
if (v_isSharedCheck_727_ == 0)
{
v___x_722_ = v___x_680_;
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
else
{
lean_inc(v_a_720_);
lean_dec(v___x_680_);
v___x_722_ = lean_box(0);
v_isShared_723_ = v_isSharedCheck_727_;
goto v_resetjp_721_;
}
v_resetjp_721_:
{
lean_object* v___x_725_; 
if (v_isShared_723_ == 0)
{
v___x_725_ = v___x_722_;
goto v_reusejp_724_;
}
else
{
lean_object* v_reuseFailAlloc_726_; 
v_reuseFailAlloc_726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_726_, 0, v_a_720_);
v___x_725_ = v_reuseFailAlloc_726_;
goto v_reusejp_724_;
}
v_reusejp_724_:
{
return v___x_725_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1___boxed(lean_object* v_x_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_, lean_object* v_a_732_, lean_object* v_a_733_, lean_object* v_a_734_, lean_object* v_a_735_, lean_object* v_a_736_, lean_object* v_a_737_){
_start:
{
lean_object* v_res_738_; 
v_res_738_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1(v_x_728_, v_a_729_, v_a_730_, v_a_731_, v_a_732_, v_a_733_, v_a_734_, v_a_735_, v_a_736_);
lean_dec(v_a_736_);
lean_dec_ref(v_a_735_);
lean_dec(v_a_734_);
lean_dec_ref(v_a_733_);
lean_dec(v_a_732_);
lean_dec_ref(v_a_731_);
lean_dec(v_a_730_);
lean_dec_ref(v_a_729_);
return v_res_738_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1(lean_object* v_mvarId_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_){
_start:
{
lean_object* v___x_749_; 
v___x_749_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___redArg(v_mvarId_739_, v___y_745_);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1___boxed(lean_object* v_mvarId_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_){
_start:
{
lean_object* v_res_760_; 
v_res_760_ = lp_batteries_Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1(v_mvarId_750_, v___y_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_, v___y_758_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_757_);
lean_dec(v___y_756_);
lean_dec_ref(v___y_755_);
lean_dec(v___y_754_);
lean_dec_ref(v___y_753_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v_mvarId_750_);
return v_res_760_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4(lean_object* v_00_u03b1_761_, lean_object* v_msg_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___redArg(v_msg_762_, v___y_767_, v___y_768_, v___y_769_, v___y_770_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4___boxed(lean_object* v_00_u03b1_773_, lean_object* v_msg_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__4(v_00_u03b1_773_, v_msg_774_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
lean_dec(v___y_782_);
lean_dec_ref(v___y_781_);
lean_dec(v___y_780_);
lean_dec_ref(v___y_779_);
lean_dec(v___y_778_);
lean_dec_ref(v___y_777_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
return v_res_784_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1(lean_object* v_00_u03b2_785_, lean_object* v_x_786_, lean_object* v_x_787_){
_start:
{
uint8_t v___x_788_; 
v___x_788_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___redArg(v_x_786_, v_x_787_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1___boxed(lean_object* v_00_u03b2_789_, lean_object* v_x_790_, lean_object* v_x_791_){
_start:
{
uint8_t v_res_792_; lean_object* v_r_793_; 
v_res_792_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1(v_00_u03b2_789_, v_x_790_, v_x_791_);
lean_dec(v_x_791_);
lean_dec_ref(v_x_790_);
v_r_793_ = lean_box(v_res_792_);
return v_r_793_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_794_, lean_object* v_x_795_, size_t v_x_796_, lean_object* v_x_797_){
_start:
{
uint8_t v___x_798_; 
v___x_798_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___redArg(v_x_795_, v_x_796_, v_x_797_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_799_, lean_object* v_x_800_, lean_object* v_x_801_, lean_object* v_x_802_){
_start:
{
size_t v_x_13530__boxed_803_; uint8_t v_res_804_; lean_object* v_r_805_; 
v_x_13530__boxed_803_ = lean_unbox_usize(v_x_801_);
lean_dec(v_x_801_);
v_res_804_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2(v_00_u03b2_799_, v_x_800_, v_x_13530__boxed_803_, v_x_802_);
lean_dec(v_x_802_);
lean_dec_ref(v_x_800_);
v_r_805_ = lean_box(v_res_804_);
return v_r_805_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5(lean_object* v_ref_806_, lean_object* v_msgData_807_, uint8_t v_severity_808_, uint8_t v_isSilent_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_){
_start:
{
lean_object* v___x_819_; 
v___x_819_ = lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___redArg(v_ref_806_, v_msgData_807_, v_severity_808_, v_isSilent_809_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5___boxed(lean_object* v_ref_820_, lean_object* v_msgData_821_, lean_object* v_severity_822_, lean_object* v_isSilent_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_){
_start:
{
uint8_t v_severity_boxed_833_; uint8_t v_isSilent_boxed_834_; lean_object* v_res_835_; 
v_severity_boxed_833_ = lean_unbox(v_severity_822_);
v_isSilent_boxed_834_ = lean_unbox(v_isSilent_823_);
v_res_835_ = lp_batteries_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__2_spec__3_spec__5(v_ref_820_, v_msgData_821_, v_severity_boxed_833_, v_isSilent_boxed_834_, v___y_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_);
lean_dec(v___y_831_);
lean_dec_ref(v___y_830_);
lean_dec(v___y_829_);
lean_dec_ref(v___y_828_);
lean_dec(v___y_827_);
lean_dec_ref(v___y_826_);
lean_dec(v___y_825_);
lean_dec_ref(v___y_824_);
lean_dec(v_ref_820_);
return v_res_835_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7(lean_object* v_00_u03b2_836_, lean_object* v_keys_837_, lean_object* v_vals_838_, lean_object* v_heq_839_, lean_object* v_i_840_, lean_object* v_k_841_){
_start:
{
uint8_t v___x_842_; 
v___x_842_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___redArg(v_keys_837_, v_i_840_, v_k_841_);
return v___x_842_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7___boxed(lean_object* v_00_u03b2_843_, lean_object* v_keys_844_, lean_object* v_vals_845_, lean_object* v_heq_846_, lean_object* v_i_847_, lean_object* v_k_848_){
_start:
{
uint8_t v_res_849_; lean_object* v_r_850_; 
v_res_849_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______elabRules__Batteries__Tactic__tacticMap__tacs_x5b___x3b_x5d__1_spec__1_spec__1_spec__2_spec__7(v_00_u03b2_843_, v_keys_844_, v_vals_845_, v_heq_846_, v_i_847_, v_k_848_);
lean_dec(v_k_848_);
lean_dec_ref(v_vals_845_);
lean_dec_ref(v_keys_844_);
v_r_850_ = lean_box(v_res_849_);
return v_r_850_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__15(void){
_start:
{
lean_object* v___x_909_; 
v___x_909_ = l_Array_mkArray0(lean_box(0));
return v___x_909_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1(lean_object* v_x_911_, lean_object* v_a_912_, lean_object* v_a_913_){
_start:
{
lean_object* v___x_914_; uint8_t v___x_915_; 
v___x_914_ = ((lean_object*)(lp_batteries_Batteries_Tactic_seq__focus___closed__1));
lean_inc(v_x_911_);
v___x_915_ = l_Lean_Syntax_isOfKind(v_x_911_, v___x_914_);
if (v___x_915_ == 0)
{
lean_object* v___x_916_; lean_object* v___x_917_; 
lean_dec(v_x_911_);
v___x_916_ = lean_box(1);
v___x_917_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_917_, 0, v___x_916_);
lean_ctor_set(v___x_917_, 1, v_a_913_);
return v___x_917_;
}
else
{
lean_object* v_ref_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v_ts_923_; uint8_t v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; 
v_ref_918_ = lean_ctor_get(v_a_912_, 5);
v___x_919_ = lean_unsigned_to_nat(0u);
v___x_920_ = l_Lean_Syntax_getArg(v_x_911_, v___x_919_);
v___x_921_ = lean_unsigned_to_nat(3u);
v___x_922_ = l_Lean_Syntax_getArg(v_x_911_, v___x_921_);
lean_dec(v_x_911_);
v_ts_923_ = l_Lean_Syntax_getArgs(v___x_922_);
lean_dec(v___x_922_);
v___x_924_ = 0;
v___x_925_ = l_Lean_SourceInfo_fromRef(v_ref_918_, v___x_924_);
v___x_926_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__2));
v___x_927_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__3));
lean_inc_n(v___x_925_, 16);
v___x_928_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_928_, 0, v___x_925_);
lean_ctor_set(v___x_928_, 1, v___x_926_);
v___x_929_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__5));
v___x_930_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__7));
v___x_931_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__9));
v___x_932_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__11));
v___x_933_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__12));
v___x_934_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_934_, 0, v___x_925_);
lean_ctor_set(v___x_934_, 1, v___x_933_);
v___x_935_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__13));
v___x_936_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_936_, 0, v___x_925_);
lean_ctor_set(v___x_936_, 1, v___x_935_);
v___x_937_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__3));
v___x_938_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__14));
v___x_939_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_939_, 0, v___x_925_);
lean_ctor_set(v___x_939_, 1, v___x_938_);
v___x_940_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__8));
v___x_941_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_941_, 0, v___x_925_);
lean_ctor_set(v___x_941_, 1, v___x_940_);
v___x_942_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__15, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__15_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__15);
v___x_943_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_ts_923_);
lean_dec_ref(v_ts_923_);
v___x_944_ = l_Lean_Syntax_SepArray_ofElems(v___x_935_, v___x_943_);
lean_dec_ref(v___x_943_);
v___x_945_ = l_Array_append___redArg(v___x_942_, v___x_944_);
lean_dec_ref(v___x_944_);
v___x_946_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_946_, 0, v___x_925_);
lean_ctor_set(v___x_946_, 1, v___x_931_);
lean_ctor_set(v___x_946_, 2, v___x_945_);
v___x_947_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticMap__tacs_x5b___x3b_x5d___closed__18));
v___x_948_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_948_, 0, v___x_925_);
lean_ctor_set(v___x_948_, 1, v___x_947_);
v___x_949_ = l_Lean_Syntax_node4(v___x_925_, v___x_937_, v___x_939_, v___x_941_, v___x_946_, v___x_948_);
v___x_950_ = l_Lean_Syntax_node3(v___x_925_, v___x_931_, v___x_920_, v___x_936_, v___x_949_);
v___x_951_ = l_Lean_Syntax_node1(v___x_925_, v___x_930_, v___x_950_);
v___x_952_ = l_Lean_Syntax_node1(v___x_925_, v___x_929_, v___x_951_);
v___x_953_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___closed__16));
v___x_954_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_954_, 0, v___x_925_);
lean_ctor_set(v___x_954_, 1, v___x_953_);
v___x_955_ = l_Lean_Syntax_node3(v___x_925_, v___x_932_, v___x_934_, v___x_952_, v___x_954_);
v___x_956_ = l_Lean_Syntax_node1(v___x_925_, v___x_931_, v___x_955_);
v___x_957_ = l_Lean_Syntax_node1(v___x_925_, v___x_930_, v___x_956_);
v___x_958_ = l_Lean_Syntax_node1(v___x_925_, v___x_929_, v___x_957_);
v___x_959_ = l_Lean_Syntax_node2(v___x_925_, v___x_927_, v___x_928_, v___x_958_);
v___x_960_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_960_, 0, v___x_959_);
lean_ctor_set(v___x_960_, 1, v_a_913_);
return v___x_960_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1___boxed(lean_object* v_x_961_, lean_object* v_a_962_, lean_object* v_a_963_){
_start:
{
lean_object* v_res_964_; 
v_res_964_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__SeqFocus______macroRules__Batteries__Tactic__seq__focus__1(v_x_961_, v_a_962_, v_a_963_);
lean_dec_ref(v_a_962_);
return v_res_964_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
}
#ifdef __cplusplus
}
#endif
