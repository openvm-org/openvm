// Lean compiler output
// Module: Aesop.Script.UScriptToSScript
// Imports: public import Init public meta import Init public import Aesop.Script.UScript public import Aesop.Script.SScript
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
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_array_size(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_script;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_MessageData_paren(lean_object*);
lean_object* l_runST___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* lp_batteries_Array_sortDedup___redArg(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_io_get_num_heartbeats();
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_empty_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_empty_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_node_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_node_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "- "};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__1;
static const lean_string_object lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__3;
static const lean_string_object lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " → "};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5;
static const lean_string_object lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7;
static const lean_array_object lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2___closed__0 = (const lean_object*)&lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__9 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__9_value;
static lean_once_cell_t lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10;
static const lean_string_object lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__11 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__11_value;
static lean_once_cell_t lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__12;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_StepTree_toMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "empty"};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_StepTree_toMessageData___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_StepTree_toMessageData___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_StepTree_toMessageData___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_instToMessageDataStepTree___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Script_StepTree_toMessageData, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_instToMessageDataStepTree___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_instToMessageDataStepTree___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Script_instToMessageDataStepTree = (const lean_object*)&lp_aesop_Aesop_Script_instToMessageDataStepTree___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Script_UScript_toStepTree___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_UScript_toStepTree___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Script_UScript_toStepTree___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_UScript_toStepTree___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_toStepTree(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_toStepTree___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Array_append___redArg___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__1_value),((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__3_value),((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__4_value),((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__5_value),((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__9_value),((lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__10_value;
static const lean_closure_object lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Script_sortDedupArrays___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__11 = (const lean_object*)&lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__11_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_isConsecutiveSequence_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Script_isConsecutiveSequence(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_isConsecutiveSequence___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_isConsecutiveSequence_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___closed__0 = (const lean_object*)&lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1___boxed(lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___redArg(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_focusableGoals___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_focusableGoals___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_focusableGoals(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_numSiblings___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_numSiblings___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_numSiblings(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_panic___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__8(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__0;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__2;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__3;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__4;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "internal error: "};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = ": unknown goal '\?"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "focus"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13_spec__17(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "applyTactic"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "goal position: "};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__1_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__2;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "visible goals: "};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__3_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__4;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "focusable: "};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__5_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__6;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__8_value;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "siblings: "};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__9_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__10;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "aesop: internal error while structuring script: unknown sibling count for goal "};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__11_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__12;
static const lean_string_object lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "applying step:"};
static const lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__13_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__14;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Converting ordered unstructured script to structured script"};
static const lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__0 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__0_value;
static const lean_ctor_object lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__0_value)}};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__1 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__1_value;
static lean_once_cell_t lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__2;
static lean_once_cell_t lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Script_orderedUScriptToSScript_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Script_orderedUScriptToSScript_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "focusable goals: "};
static const lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1;
static const lean_string_object lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "step tree:"};
static const lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__8(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__8___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__0 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__0_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__1;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "unstructured script:"};
static const lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1;
static const lean_closure_object lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorIdx(lean_object* v_x_1_){
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_aesop_Aesop_Script_StepTree_ctorIdx(v_x_4_);
lean_dec(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorElim___redArg(lean_object* v_t_6_, lean_object* v_k_7_){
_start:
{
if (lean_obj_tag(v_t_6_) == 0)
{
return v_k_7_;
}
else
{
lean_object* v_step_8_; lean_object* v_index_9_; lean_object* v_children_10_; lean_object* v___x_11_; 
v_step_8_ = lean_ctor_get(v_t_6_, 0);
lean_inc_ref(v_step_8_);
v_index_9_ = lean_ctor_get(v_t_6_, 1);
lean_inc(v_index_9_);
v_children_10_ = lean_ctor_get(v_t_6_, 2);
lean_inc_ref(v_children_10_);
lean_dec_ref_known(v_t_6_, 3);
v___x_11_ = lean_apply_3(v_k_7_, v_step_8_, v_index_9_, v_children_10_);
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorElim(lean_object* v_motive__1_12_, lean_object* v_ctorIdx_13_, lean_object* v_t_14_, lean_object* v_h_15_, lean_object* v_k_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lp_aesop_Aesop_Script_StepTree_ctorElim___redArg(v_t_14_, v_k_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_ctorElim___boxed(lean_object* v_motive__1_18_, lean_object* v_ctorIdx_19_, lean_object* v_t_20_, lean_object* v_h_21_, lean_object* v_k_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_aesop_Aesop_Script_StepTree_ctorElim(v_motive__1_18_, v_ctorIdx_19_, v_t_20_, v_h_21_, v_k_22_);
lean_dec(v_ctorIdx_19_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_empty_elim___redArg(lean_object* v_t_24_, lean_object* v_empty_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_aesop_Aesop_Script_StepTree_ctorElim___redArg(v_t_24_, v_empty_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_empty_elim(lean_object* v_motive__1_27_, lean_object* v_t_28_, lean_object* v_h_29_, lean_object* v_empty_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_aesop_Aesop_Script_StepTree_ctorElim___redArg(v_t_28_, v_empty_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_node_elim___redArg(lean_object* v_t_32_, lean_object* v_node_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_aesop_Aesop_Script_StepTree_ctorElim___redArg(v_t_32_, v_node_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_node_elim(lean_object* v_motive__1_35_, lean_object* v_t_36_, lean_object* v_h_37_, lean_object* v_node_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_aesop_Aesop_Script_StepTree_ctorElim___redArg(v_t_36_, v_node_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0(size_t v_sz_40_, size_t v_i_41_, lean_object* v_bs_42_){
_start:
{
uint8_t v___x_43_; 
v___x_43_ = lean_usize_dec_lt(v_i_41_, v_sz_40_);
if (v___x_43_ == 0)
{
return v_bs_42_;
}
else
{
lean_object* v_v_44_; lean_object* v_goal_45_; lean_object* v___x_46_; lean_object* v_bs_x27_47_; size_t v___x_48_; size_t v___x_49_; lean_object* v___x_50_; 
v_v_44_ = lean_array_uget_borrowed(v_bs_42_, v_i_41_);
v_goal_45_ = lean_ctor_get(v_v_44_, 0);
lean_inc(v_goal_45_);
v___x_46_ = lean_unsigned_to_nat(0u);
v_bs_x27_47_ = lean_array_uset(v_bs_42_, v_i_41_, v___x_46_);
v___x_48_ = ((size_t)1ULL);
v___x_49_ = lean_usize_add(v_i_41_, v___x_48_);
v___x_50_ = lean_array_uset(v_bs_x27_47_, v_i_41_, v_goal_45_);
v_i_41_ = v___x_49_;
v_bs_42_ = v___x_50_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0___boxed(lean_object* v_sz_52_, lean_object* v_i_53_, lean_object* v_bs_54_){
_start:
{
size_t v_sz_boxed_55_; size_t v_i_boxed_56_; lean_object* v_res_57_; 
v_sz_boxed_55_ = lean_unbox_usize(v_sz_52_);
lean_dec(v_sz_52_);
v_i_boxed_56_ = lean_unbox_usize(v_i_53_);
lean_dec(v_i_53_);
v_res_57_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0(v_sz_boxed_55_, v_i_boxed_56_, v_bs_54_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__1(lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
if (lean_obj_tag(v_a_58_) == 0)
{
lean_object* v___x_60_; 
v___x_60_ = l_List_reverse___redArg(v_a_59_);
return v___x_60_;
}
else
{
lean_object* v_head_61_; lean_object* v_tail_62_; lean_object* v___x_64_; uint8_t v_isShared_65_; uint8_t v_isSharedCheck_71_; 
v_head_61_ = lean_ctor_get(v_a_58_, 0);
v_tail_62_ = lean_ctor_get(v_a_58_, 1);
v_isSharedCheck_71_ = !lean_is_exclusive(v_a_58_);
if (v_isSharedCheck_71_ == 0)
{
v___x_64_ = v_a_58_;
v_isShared_65_ = v_isSharedCheck_71_;
goto v_resetjp_63_;
}
else
{
lean_inc(v_tail_62_);
lean_inc(v_head_61_);
lean_dec(v_a_58_);
v___x_64_ = lean_box(0);
v_isShared_65_ = v_isSharedCheck_71_;
goto v_resetjp_63_;
}
v_resetjp_63_:
{
lean_object* v___x_66_; lean_object* v___x_68_; 
v___x_66_ = l_Lean_MessageData_ofName(v_head_61_);
if (v_isShared_65_ == 0)
{
lean_ctor_set(v___x_64_, 1, v_a_59_);
lean_ctor_set(v___x_64_, 0, v___x_66_);
v___x_68_ = v___x_64_;
goto v_reusejp_67_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v___x_66_);
lean_ctor_set(v_reuseFailAlloc_70_, 1, v_a_59_);
v___x_68_ = v_reuseFailAlloc_70_;
goto v_reusejp_67_;
}
v_reusejp_67_:
{
v_a_58_ = v_tail_62_;
v_a_59_ = v___x_68_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__1(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__0));
v___x_74_ = l_Lean_stringToMessageData(v___x_73_);
return v___x_74_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__3(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_76_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__2));
v___x_77_ = l_Lean_stringToMessageData(v___x_76_);
return v___x_77_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_79_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__4));
v___x_80_ = l_Lean_stringToMessageData(v___x_79_);
return v___x_80_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_82_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__6));
v___x_83_ = l_Lean_stringToMessageData(v___x_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2(lean_object* v_as_86_, lean_object* v_start_87_, lean_object* v_stop_88_){
_start:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = ((lean_object*)(lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2___closed__0));
v___x_90_ = lean_nat_dec_lt(v_start_87_, v_stop_88_);
if (v___x_90_ == 0)
{
return v___x_89_;
}
else
{
lean_object* v___x_91_; uint8_t v___x_92_; 
v___x_91_ = lean_array_get_size(v_as_86_);
v___x_92_ = lean_nat_dec_le(v_stop_88_, v___x_91_);
if (v___x_92_ == 0)
{
uint8_t v___x_93_; 
v___x_93_ = lean_nat_dec_lt(v_start_87_, v___x_91_);
if (v___x_93_ == 0)
{
return v___x_89_;
}
else
{
size_t v___x_94_; size_t v___x_95_; lean_object* v___x_96_; 
v___x_94_ = lean_usize_of_nat(v_start_87_);
v___x_95_ = lean_usize_of_nat(v___x_91_);
v___x_96_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2_spec__2(v_as_86_, v___x_94_, v___x_95_, v___x_89_);
return v___x_96_;
}
}
else
{
size_t v___x_97_; size_t v___x_98_; lean_object* v___x_99_; 
v___x_97_ = lean_usize_of_nat(v_start_87_);
v___x_98_ = lean_usize_of_nat(v_stop_88_);
v___x_99_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2_spec__2(v_as_86_, v___x_97_, v___x_98_, v___x_89_);
return v___x_99_;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_103_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__9));
v___x_104_ = l_Lean_MessageData_ofFormat(v___x_103_);
return v___x_104_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__12(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__11));
v___x_107_ = l_Lean_stringToMessageData(v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData_x3f(lean_object* v_x_108_){
_start:
{
if (lean_obj_tag(v_x_108_) == 0)
{
lean_object* v___x_109_; 
v___x_109_ = lean_box(0);
return v___x_109_;
}
else
{
lean_object* v_step_110_; lean_object* v_tactic_111_; lean_object* v_index_112_; lean_object* v_children_113_; lean_object* v_preGoal_114_; lean_object* v_postGoals_115_; lean_object* v_uTactic_116_; lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_159_; 
v_step_110_ = lean_ctor_get(v_x_108_, 0);
lean_inc_ref(v_step_110_);
v_tactic_111_ = lean_ctor_get(v_step_110_, 2);
lean_inc_ref(v_tactic_111_);
v_index_112_ = lean_ctor_get(v_x_108_, 1);
lean_inc(v_index_112_);
v_children_113_ = lean_ctor_get(v_x_108_, 2);
lean_inc_ref(v_children_113_);
lean_dec_ref_known(v_x_108_, 3);
v_preGoal_114_ = lean_ctor_get(v_step_110_, 1);
lean_inc(v_preGoal_114_);
v_postGoals_115_ = lean_ctor_get(v_step_110_, 4);
lean_inc_ref(v_postGoals_115_);
lean_dec_ref(v_step_110_);
v_uTactic_116_ = lean_ctor_get(v_tactic_111_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v_tactic_111_);
if (v_isSharedCheck_159_ == 0)
{
lean_object* v_unused_160_; 
v_unused_160_ = lean_ctor_get(v_tactic_111_, 1);
lean_dec(v_unused_160_);
v___x_118_ = v_tactic_111_;
v_isShared_119_ = v_isSharedCheck_159_;
goto v_resetjp_117_;
}
else
{
lean_inc(v_uTactic_116_);
lean_dec(v_tactic_111_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_159_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
size_t v_sz_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_126_; 
v_sz_120_ = lean_array_size(v_postGoals_115_);
v___x_121_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__1, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__1_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__1);
v___x_122_ = l_Nat_reprFast(v_index_112_);
v___x_123_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
v___x_124_ = l_Lean_MessageData_ofFormat(v___x_123_);
if (v_isShared_119_ == 0)
{
lean_ctor_set_tag(v___x_118_, 7);
lean_ctor_set(v___x_118_, 1, v___x_124_);
lean_ctor_set(v___x_118_, 0, v___x_121_);
v___x_126_ = v___x_118_;
goto v_reusejp_125_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v___x_121_);
lean_ctor_set(v_reuseFailAlloc_158_, 1, v___x_124_);
v___x_126_ = v_reuseFailAlloc_158_;
goto v_reusejp_125_;
}
v_reusejp_125_:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; size_t v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___y_146_; lean_object* v___x_149_; lean_object* v___x_150_; uint8_t v___x_151_; 
v___x_127_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__3, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__3_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__3);
v___x_128_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_126_);
lean_ctor_set(v___x_128_, 1, v___x_127_);
v___x_129_ = l_Lean_MessageData_ofName(v_preGoal_114_);
v___x_130_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5);
v___x_131_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_129_);
lean_ctor_set(v___x_131_, 1, v___x_130_);
v___x_132_ = ((size_t)0ULL);
v___x_133_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0(v_sz_120_, v___x_132_, v_postGoals_115_);
v___x_134_ = lean_array_to_list(v___x_133_);
v___x_135_ = lean_box(0);
v___x_136_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__1(v___x_134_, v___x_135_);
v___x_137_ = l_Lean_MessageData_ofList(v___x_136_);
v___x_138_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_131_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
v___x_139_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7);
v___x_140_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_138_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
v___x_141_ = l_Lean_MessageData_ofSyntax(v_uTactic_116_);
v___x_142_ = l_Lean_indentD(v___x_141_);
v___x_143_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_140_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_128_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
v___x_149_ = lean_array_get_size(v_children_113_);
v___x_150_ = lean_unsigned_to_nat(0u);
v___x_151_ = lean_nat_dec_eq(v___x_149_, v___x_150_);
if (v___x_151_ == 0)
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_152_ = lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2(v_children_113_, v___x_150_, v___x_149_);
lean_dec_ref(v_children_113_);
v___x_153_ = lean_array_to_list(v___x_152_);
v___x_154_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10);
v___x_155_ = l_Lean_MessageData_joinSep(v___x_153_, v___x_154_);
v___x_156_ = l_Lean_indentD(v___x_155_);
v___y_146_ = v___x_156_;
goto v___jp_145_;
}
else
{
lean_object* v___x_157_; 
lean_dec_ref(v_children_113_);
v___x_157_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__12, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__12_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__12);
v___y_146_ = v___x_157_;
goto v___jp_145_;
}
v___jp_145_:
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_144_);
lean_ctor_set(v___x_147_, 1, v___y_146_);
v___x_148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
return v___x_148_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2_spec__2(lean_object* v_as_161_, size_t v_i_162_, size_t v_stop_163_, lean_object* v_b_164_){
_start:
{
lean_object* v___y_166_; uint8_t v___x_170_; 
v___x_170_ = lean_usize_dec_eq(v_i_162_, v_stop_163_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_171_ = lean_array_uget_borrowed(v_as_161_, v_i_162_);
lean_inc(v___x_171_);
v___x_172_ = lp_aesop_Aesop_Script_StepTree_toMessageData_x3f(v___x_171_);
if (lean_obj_tag(v___x_172_) == 0)
{
v___y_166_ = v_b_164_;
goto v___jp_165_;
}
else
{
lean_object* v_val_173_; lean_object* v___x_174_; 
v_val_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc(v_val_173_);
lean_dec_ref_known(v___x_172_, 1);
v___x_174_ = lean_array_push(v_b_164_, v_val_173_);
v___y_166_ = v___x_174_;
goto v___jp_165_;
}
}
else
{
return v_b_164_;
}
v___jp_165_:
{
size_t v___x_167_; size_t v___x_168_; 
v___x_167_ = ((size_t)1ULL);
v___x_168_ = lean_usize_add(v_i_162_, v___x_167_);
v_i_162_ = v___x_168_;
v_b_164_ = v___y_166_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2_spec__2___boxed(lean_object* v_as_175_, lean_object* v_i_176_, lean_object* v_stop_177_, lean_object* v_b_178_){
_start:
{
size_t v_i_boxed_179_; size_t v_stop_boxed_180_; lean_object* v_res_181_; 
v_i_boxed_179_ = lean_unbox_usize(v_i_176_);
lean_dec(v_i_176_);
v_stop_boxed_180_ = lean_unbox_usize(v_stop_177_);
lean_dec(v_stop_177_);
v_res_181_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2_spec__2(v_as_175_, v_i_boxed_179_, v_stop_boxed_180_, v_b_178_);
lean_dec_ref(v_as_175_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2___boxed(lean_object* v_as_182_, lean_object* v_start_183_, lean_object* v_stop_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2(v_as_182_, v_start_183_, v_stop_184_);
lean_dec(v_stop_184_);
lean_dec(v_start_183_);
lean_dec_ref(v_as_182_);
return v_res_185_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_StepTree_toMessageData___closed__2(void){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_189_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData___closed__1));
v___x_190_ = l_Lean_MessageData_ofFormat(v___x_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_toMessageData(lean_object* v_t_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_aesop_Aesop_Script_StepTree_toMessageData_x3f(v_t_191_);
if (lean_obj_tag(v___x_192_) == 0)
{
lean_object* v___x_193_; 
v___x_193_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData___closed__2, &lp_aesop_Aesop_Script_StepTree_toMessageData___closed__2_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData___closed__2);
return v___x_193_;
}
else
{
lean_object* v_val_194_; 
v_val_194_ = lean_ctor_get(v___x_192_, 0);
lean_inc(v_val_194_);
lean_dec_ref_known(v___x_192_, 1);
return v_val_194_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___redArg(lean_object* v_a_197_, lean_object* v_x_198_){
_start:
{
if (lean_obj_tag(v_x_198_) == 0)
{
lean_object* v___x_199_; 
v___x_199_ = lean_box(0);
return v___x_199_;
}
else
{
lean_object* v_key_200_; lean_object* v_value_201_; lean_object* v_tail_202_; uint8_t v___x_203_; 
v_key_200_ = lean_ctor_get(v_x_198_, 0);
v_value_201_ = lean_ctor_get(v_x_198_, 1);
v_tail_202_ = lean_ctor_get(v_x_198_, 2);
v___x_203_ = l_Lean_instBEqMVarId_beq(v_key_200_, v_a_197_);
if (v___x_203_ == 0)
{
v_x_198_ = v_tail_202_;
goto _start;
}
else
{
lean_object* v___x_205_; 
lean_inc(v_value_201_);
v___x_205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_205_, 0, v_value_201_);
return v___x_205_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___redArg___boxed(lean_object* v_a_206_, lean_object* v_x_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___redArg(v_a_206_, v_x_207_);
lean_dec(v_x_207_);
lean_dec(v_a_206_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg(lean_object* v_m_209_, lean_object* v_a_210_){
_start:
{
lean_object* v_buckets_211_; lean_object* v___x_212_; uint64_t v___x_213_; uint64_t v___x_214_; uint64_t v___x_215_; uint64_t v_fold_216_; uint64_t v___x_217_; uint64_t v___x_218_; uint64_t v___x_219_; size_t v___x_220_; size_t v___x_221_; size_t v___x_222_; size_t v___x_223_; size_t v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v_buckets_211_ = lean_ctor_get(v_m_209_, 1);
v___x_212_ = lean_array_get_size(v_buckets_211_);
v___x_213_ = l_Lean_instHashableMVarId_hash(v_a_210_);
v___x_214_ = 32ULL;
v___x_215_ = lean_uint64_shift_right(v___x_213_, v___x_214_);
v_fold_216_ = lean_uint64_xor(v___x_213_, v___x_215_);
v___x_217_ = 16ULL;
v___x_218_ = lean_uint64_shift_right(v_fold_216_, v___x_217_);
v___x_219_ = lean_uint64_xor(v_fold_216_, v___x_218_);
v___x_220_ = lean_uint64_to_usize(v___x_219_);
v___x_221_ = lean_usize_of_nat(v___x_212_);
v___x_222_ = ((size_t)1ULL);
v___x_223_ = lean_usize_sub(v___x_221_, v___x_222_);
v___x_224_ = lean_usize_land(v___x_220_, v___x_223_);
v___x_225_ = lean_array_uget_borrowed(v_buckets_211_, v___x_224_);
v___x_226_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___redArg(v_a_210_, v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg___boxed(lean_object* v_m_227_, lean_object* v_a_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg(v_m_227_, v_a_228_);
lean_dec(v_a_228_);
lean_dec_ref(v_m_227_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__1(lean_object* v_m_230_, size_t v_sz_231_, size_t v_i_232_, lean_object* v_bs_233_){
_start:
{
uint8_t v___x_234_; 
v___x_234_ = lean_usize_dec_lt(v_i_232_, v_sz_231_);
if (v___x_234_ == 0)
{
return v_bs_233_;
}
else
{
lean_object* v_v_235_; lean_object* v_goal_236_; lean_object* v___x_237_; lean_object* v_bs_x27_238_; lean_object* v___x_239_; size_t v___x_240_; size_t v___x_241_; lean_object* v___x_242_; 
v_v_235_ = lean_array_uget_borrowed(v_bs_233_, v_i_232_);
v_goal_236_ = lean_ctor_get(v_v_235_, 0);
lean_inc(v_goal_236_);
v___x_237_ = lean_unsigned_to_nat(0u);
v_bs_x27_238_ = lean_array_uset(v_bs_233_, v_i_232_, v___x_237_);
v___x_239_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go(v_m_230_, v_goal_236_);
lean_dec(v_goal_236_);
v___x_240_ = ((size_t)1ULL);
v___x_241_ = lean_usize_add(v_i_232_, v___x_240_);
v___x_242_ = lean_array_uset(v_bs_x27_238_, v_i_232_, v___x_239_);
v_i_232_ = v___x_241_;
v_bs_233_ = v___x_242_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go(lean_object* v_m_244_, lean_object* v_goal_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg(v_m_244_, v_goal_245_);
if (lean_obj_tag(v___x_246_) == 1)
{
lean_object* v_val_247_; lean_object* v_snd_248_; lean_object* v_fst_249_; lean_object* v_postGoals_250_; size_t v_sz_251_; size_t v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v_val_247_ = lean_ctor_get(v___x_246_, 0);
lean_inc(v_val_247_);
lean_dec_ref_known(v___x_246_, 1);
v_snd_248_ = lean_ctor_get(v_val_247_, 1);
lean_inc(v_snd_248_);
v_fst_249_ = lean_ctor_get(v_val_247_, 0);
lean_inc(v_fst_249_);
lean_dec(v_val_247_);
v_postGoals_250_ = lean_ctor_get(v_snd_248_, 4);
v_sz_251_ = lean_array_size(v_postGoals_250_);
v___x_252_ = ((size_t)0ULL);
lean_inc_ref(v_postGoals_250_);
v___x_253_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__1(v_m_244_, v_sz_251_, v___x_252_, v_postGoals_250_);
v___x_254_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_254_, 0, v_snd_248_);
lean_ctor_set(v___x_254_, 1, v_fst_249_);
lean_ctor_set(v___x_254_, 2, v___x_253_);
return v___x_254_;
}
else
{
lean_object* v___x_255_; 
lean_dec(v___x_246_);
v___x_255_ = lean_box(0);
return v___x_255_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go___boxed(lean_object* v_m_256_, lean_object* v_goal_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go(v_m_256_, v_goal_257_);
lean_dec(v_goal_257_);
lean_dec_ref(v_m_256_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__1___boxed(lean_object* v_m_259_, lean_object* v_sz_260_, lean_object* v_i_261_, lean_object* v_bs_262_){
_start:
{
size_t v_sz_boxed_263_; size_t v_i_boxed_264_; lean_object* v_res_265_; 
v_sz_boxed_263_ = lean_unbox_usize(v_sz_260_);
lean_dec(v_sz_260_);
v_i_boxed_264_ = lean_unbox_usize(v_i_261_);
lean_dec(v_i_261_);
v_res_265_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__1(v_m_259_, v_sz_boxed_263_, v_i_boxed_264_, v_bs_262_);
lean_dec_ref(v_m_259_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0(lean_object* v_00_u03b2_266_, lean_object* v_m_267_, lean_object* v_a_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg(v_m_267_, v_a_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___boxed(lean_object* v_00_u03b2_270_, lean_object* v_m_271_, lean_object* v_a_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0(v_00_u03b2_270_, v_m_271_, v_a_272_);
lean_dec(v_a_272_);
lean_dec_ref(v_m_271_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0(lean_object* v_00_u03b2_274_, lean_object* v_a_275_, lean_object* v_x_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___redArg(v_a_275_, v_x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0___boxed(lean_object* v_00_u03b2_278_, lean_object* v_a_279_, lean_object* v_x_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0_spec__0(v_00_u03b2_278_, v_a_279_, v_x_280_);
lean_dec(v_x_280_);
lean_dec(v_a_279_);
return v_res_281_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg(lean_object* v_a_282_, lean_object* v_x_283_){
_start:
{
if (lean_obj_tag(v_x_283_) == 0)
{
uint8_t v___x_284_; 
v___x_284_ = 0;
return v___x_284_;
}
else
{
lean_object* v_key_285_; lean_object* v_tail_286_; uint8_t v___x_287_; 
v_key_285_ = lean_ctor_get(v_x_283_, 0);
v_tail_286_ = lean_ctor_get(v_x_283_, 2);
v___x_287_ = l_Lean_instBEqMVarId_beq(v_key_285_, v_a_282_);
if (v___x_287_ == 0)
{
v_x_283_ = v_tail_286_;
goto _start;
}
else
{
return v___x_287_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg___boxed(lean_object* v_a_289_, lean_object* v_x_290_){
_start:
{
uint8_t v_res_291_; lean_object* v_r_292_; 
v_res_291_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg(v_a_289_, v_x_290_);
lean_dec(v_x_290_);
lean_dec(v_a_289_);
v_r_292_ = lean_box(v_res_291_);
return v_r_292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_293_, lean_object* v_x_294_){
_start:
{
if (lean_obj_tag(v_x_294_) == 0)
{
return v_x_293_;
}
else
{
lean_object* v_key_295_; lean_object* v_value_296_; lean_object* v_tail_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_320_; 
v_key_295_ = lean_ctor_get(v_x_294_, 0);
v_value_296_ = lean_ctor_get(v_x_294_, 1);
v_tail_297_ = lean_ctor_get(v_x_294_, 2);
v_isSharedCheck_320_ = !lean_is_exclusive(v_x_294_);
if (v_isSharedCheck_320_ == 0)
{
v___x_299_ = v_x_294_;
v_isShared_300_ = v_isSharedCheck_320_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_tail_297_);
lean_inc(v_value_296_);
lean_inc(v_key_295_);
lean_dec(v_x_294_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_320_;
goto v_resetjp_298_;
}
v_resetjp_298_:
{
lean_object* v___x_301_; uint64_t v___x_302_; uint64_t v___x_303_; uint64_t v___x_304_; uint64_t v_fold_305_; uint64_t v___x_306_; uint64_t v___x_307_; uint64_t v___x_308_; size_t v___x_309_; size_t v___x_310_; size_t v___x_311_; size_t v___x_312_; size_t v___x_313_; lean_object* v___x_314_; lean_object* v___x_316_; 
v___x_301_ = lean_array_get_size(v_x_293_);
v___x_302_ = l_Lean_instHashableMVarId_hash(v_key_295_);
v___x_303_ = 32ULL;
v___x_304_ = lean_uint64_shift_right(v___x_302_, v___x_303_);
v_fold_305_ = lean_uint64_xor(v___x_302_, v___x_304_);
v___x_306_ = 16ULL;
v___x_307_ = lean_uint64_shift_right(v_fold_305_, v___x_306_);
v___x_308_ = lean_uint64_xor(v_fold_305_, v___x_307_);
v___x_309_ = lean_uint64_to_usize(v___x_308_);
v___x_310_ = lean_usize_of_nat(v___x_301_);
v___x_311_ = ((size_t)1ULL);
v___x_312_ = lean_usize_sub(v___x_310_, v___x_311_);
v___x_313_ = lean_usize_land(v___x_309_, v___x_312_);
v___x_314_ = lean_array_uget_borrowed(v_x_293_, v___x_313_);
lean_inc(v___x_314_);
if (v_isShared_300_ == 0)
{
lean_ctor_set(v___x_299_, 2, v___x_314_);
v___x_316_ = v___x_299_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v_key_295_);
lean_ctor_set(v_reuseFailAlloc_319_, 1, v_value_296_);
lean_ctor_set(v_reuseFailAlloc_319_, 2, v___x_314_);
v___x_316_ = v_reuseFailAlloc_319_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
lean_object* v___x_317_; 
v___x_317_ = lean_array_uset(v_x_293_, v___x_313_, v___x_316_);
v_x_293_ = v___x_317_;
v_x_294_ = v_tail_297_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2___redArg(lean_object* v_i_321_, lean_object* v_source_322_, lean_object* v_target_323_){
_start:
{
lean_object* v___x_324_; uint8_t v___x_325_; 
v___x_324_ = lean_array_get_size(v_source_322_);
v___x_325_ = lean_nat_dec_lt(v_i_321_, v___x_324_);
if (v___x_325_ == 0)
{
lean_dec_ref(v_source_322_);
lean_dec(v_i_321_);
return v_target_323_;
}
else
{
lean_object* v_es_326_; lean_object* v___x_327_; lean_object* v_source_328_; lean_object* v_target_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v_es_326_ = lean_array_fget(v_source_322_, v_i_321_);
v___x_327_ = lean_box(0);
v_source_328_ = lean_array_fset(v_source_322_, v_i_321_, v___x_327_);
v_target_329_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2_spec__4___redArg(v_target_323_, v_es_326_);
v___x_330_ = lean_unsigned_to_nat(1u);
v___x_331_ = lean_nat_add(v_i_321_, v___x_330_);
lean_dec(v_i_321_);
v_i_321_ = v___x_331_;
v_source_322_ = v_source_328_;
v_target_323_ = v_target_329_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1___redArg(lean_object* v_data_333_){
_start:
{
lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v_nbuckets_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_334_ = lean_array_get_size(v_data_333_);
v___x_335_ = lean_unsigned_to_nat(2u);
v_nbuckets_336_ = lean_nat_mul(v___x_334_, v___x_335_);
v___x_337_ = lean_unsigned_to_nat(0u);
v___x_338_ = lean_box(0);
v___x_339_ = lean_mk_array(v_nbuckets_336_, v___x_338_);
v___x_340_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2___redArg(v___x_337_, v_data_333_, v___x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__2___redArg(lean_object* v_a_341_, lean_object* v_b_342_, lean_object* v_x_343_){
_start:
{
if (lean_obj_tag(v_x_343_) == 0)
{
lean_dec(v_b_342_);
lean_dec(v_a_341_);
return v_x_343_;
}
else
{
lean_object* v_key_344_; lean_object* v_value_345_; lean_object* v_tail_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_358_; 
v_key_344_ = lean_ctor_get(v_x_343_, 0);
v_value_345_ = lean_ctor_get(v_x_343_, 1);
v_tail_346_ = lean_ctor_get(v_x_343_, 2);
v_isSharedCheck_358_ = !lean_is_exclusive(v_x_343_);
if (v_isSharedCheck_358_ == 0)
{
v___x_348_ = v_x_343_;
v_isShared_349_ = v_isSharedCheck_358_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_tail_346_);
lean_inc(v_value_345_);
lean_inc(v_key_344_);
lean_dec(v_x_343_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_358_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
uint8_t v___x_350_; 
v___x_350_ = l_Lean_instBEqMVarId_beq(v_key_344_, v_a_341_);
if (v___x_350_ == 0)
{
lean_object* v___x_351_; lean_object* v___x_353_; 
v___x_351_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__2___redArg(v_a_341_, v_b_342_, v_tail_346_);
if (v_isShared_349_ == 0)
{
lean_ctor_set(v___x_348_, 2, v___x_351_);
v___x_353_ = v___x_348_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v_key_344_);
lean_ctor_set(v_reuseFailAlloc_354_, 1, v_value_345_);
lean_ctor_set(v_reuseFailAlloc_354_, 2, v___x_351_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
else
{
lean_object* v___x_356_; 
lean_dec(v_value_345_);
lean_dec(v_key_344_);
if (v_isShared_349_ == 0)
{
lean_ctor_set(v___x_348_, 1, v_b_342_);
lean_ctor_set(v___x_348_, 0, v_a_341_);
v___x_356_ = v___x_348_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v_a_341_);
lean_ctor_set(v_reuseFailAlloc_357_, 1, v_b_342_);
lean_ctor_set(v_reuseFailAlloc_357_, 2, v_tail_346_);
v___x_356_ = v_reuseFailAlloc_357_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
return v___x_356_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0___redArg(lean_object* v_m_359_, lean_object* v_a_360_, lean_object* v_b_361_){
_start:
{
lean_object* v_size_362_; lean_object* v_buckets_363_; lean_object* v___x_365_; uint8_t v_isShared_366_; uint8_t v_isSharedCheck_406_; 
v_size_362_ = lean_ctor_get(v_m_359_, 0);
v_buckets_363_ = lean_ctor_get(v_m_359_, 1);
v_isSharedCheck_406_ = !lean_is_exclusive(v_m_359_);
if (v_isSharedCheck_406_ == 0)
{
v___x_365_ = v_m_359_;
v_isShared_366_ = v_isSharedCheck_406_;
goto v_resetjp_364_;
}
else
{
lean_inc(v_buckets_363_);
lean_inc(v_size_362_);
lean_dec(v_m_359_);
v___x_365_ = lean_box(0);
v_isShared_366_ = v_isSharedCheck_406_;
goto v_resetjp_364_;
}
v_resetjp_364_:
{
lean_object* v___x_367_; uint64_t v___x_368_; uint64_t v___x_369_; uint64_t v___x_370_; uint64_t v_fold_371_; uint64_t v___x_372_; uint64_t v___x_373_; uint64_t v___x_374_; size_t v___x_375_; size_t v___x_376_; size_t v___x_377_; size_t v___x_378_; size_t v___x_379_; lean_object* v_bkt_380_; uint8_t v___x_381_; 
v___x_367_ = lean_array_get_size(v_buckets_363_);
v___x_368_ = l_Lean_instHashableMVarId_hash(v_a_360_);
v___x_369_ = 32ULL;
v___x_370_ = lean_uint64_shift_right(v___x_368_, v___x_369_);
v_fold_371_ = lean_uint64_xor(v___x_368_, v___x_370_);
v___x_372_ = 16ULL;
v___x_373_ = lean_uint64_shift_right(v_fold_371_, v___x_372_);
v___x_374_ = lean_uint64_xor(v_fold_371_, v___x_373_);
v___x_375_ = lean_uint64_to_usize(v___x_374_);
v___x_376_ = lean_usize_of_nat(v___x_367_);
v___x_377_ = ((size_t)1ULL);
v___x_378_ = lean_usize_sub(v___x_376_, v___x_377_);
v___x_379_ = lean_usize_land(v___x_375_, v___x_378_);
v_bkt_380_ = lean_array_uget_borrowed(v_buckets_363_, v___x_379_);
v___x_381_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg(v_a_360_, v_bkt_380_);
if (v___x_381_ == 0)
{
lean_object* v___x_382_; lean_object* v_size_x27_383_; lean_object* v___x_384_; lean_object* v_buckets_x27_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; uint8_t v___x_391_; 
v___x_382_ = lean_unsigned_to_nat(1u);
v_size_x27_383_ = lean_nat_add(v_size_362_, v___x_382_);
lean_dec(v_size_362_);
lean_inc(v_bkt_380_);
v___x_384_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_384_, 0, v_a_360_);
lean_ctor_set(v___x_384_, 1, v_b_361_);
lean_ctor_set(v___x_384_, 2, v_bkt_380_);
v_buckets_x27_385_ = lean_array_uset(v_buckets_363_, v___x_379_, v___x_384_);
v___x_386_ = lean_unsigned_to_nat(4u);
v___x_387_ = lean_nat_mul(v_size_x27_383_, v___x_386_);
v___x_388_ = lean_unsigned_to_nat(3u);
v___x_389_ = lean_nat_div(v___x_387_, v___x_388_);
lean_dec(v___x_387_);
v___x_390_ = lean_array_get_size(v_buckets_x27_385_);
v___x_391_ = lean_nat_dec_le(v___x_389_, v___x_390_);
lean_dec(v___x_389_);
if (v___x_391_ == 0)
{
lean_object* v_val_392_; lean_object* v___x_394_; 
v_val_392_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1___redArg(v_buckets_x27_385_);
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 1, v_val_392_);
lean_ctor_set(v___x_365_, 0, v_size_x27_383_);
v___x_394_ = v___x_365_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v_size_x27_383_);
lean_ctor_set(v_reuseFailAlloc_395_, 1, v_val_392_);
v___x_394_ = v_reuseFailAlloc_395_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
return v___x_394_;
}
}
else
{
lean_object* v___x_397_; 
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 1, v_buckets_x27_385_);
lean_ctor_set(v___x_365_, 0, v_size_x27_383_);
v___x_397_ = v___x_365_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v_size_x27_383_);
lean_ctor_set(v_reuseFailAlloc_398_, 1, v_buckets_x27_385_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
else
{
lean_object* v___x_399_; lean_object* v_buckets_x27_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_404_; 
lean_inc(v_bkt_380_);
v___x_399_ = lean_box(0);
v_buckets_x27_400_ = lean_array_uset(v_buckets_363_, v___x_379_, v___x_399_);
v___x_401_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__2___redArg(v_a_360_, v_b_361_, v_bkt_380_);
v___x_402_ = lean_array_uset(v_buckets_x27_400_, v___x_379_, v___x_401_);
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 1, v___x_402_);
v___x_404_ = v___x_365_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_size_362_);
lean_ctor_set(v_reuseFailAlloc_405_, 1, v___x_402_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
return v___x_404_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___redArg(lean_object* v_s_407_, lean_object* v_range_408_, lean_object* v_b_409_, lean_object* v_i_410_){
_start:
{
lean_object* v_stop_411_; lean_object* v_step_412_; uint8_t v___x_413_; 
v_stop_411_ = lean_ctor_get(v_range_408_, 1);
v_step_412_ = lean_ctor_get(v_range_408_, 2);
v___x_413_ = lean_nat_dec_lt(v_i_410_, v_stop_411_);
if (v___x_413_ == 0)
{
lean_dec(v_i_410_);
return v_b_409_;
}
else
{
lean_object* v___x_414_; lean_object* v_preGoal_415_; lean_object* v___x_416_; lean_object* v_preGoalMap_417_; lean_object* v___x_418_; 
v___x_414_ = lean_array_fget_borrowed(v_s_407_, v_i_410_);
v_preGoal_415_ = lean_ctor_get(v___x_414_, 1);
lean_inc(v___x_414_);
lean_inc(v_i_410_);
v___x_416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_416_, 0, v_i_410_);
lean_ctor_set(v___x_416_, 1, v___x_414_);
lean_inc(v_preGoal_415_);
v_preGoalMap_417_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0___redArg(v_b_409_, v_preGoal_415_, v___x_416_);
v___x_418_ = lean_nat_add(v_i_410_, v_step_412_);
lean_dec(v_i_410_);
v_b_409_ = v_preGoalMap_417_;
v_i_410_ = v___x_418_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___redArg___boxed(lean_object* v_s_420_, lean_object* v_range_421_, lean_object* v_b_422_, lean_object* v_i_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___redArg(v_s_420_, v_range_421_, v_b_422_, v_i_423_);
lean_dec_ref(v_range_421_);
lean_dec_ref(v_s_420_);
return v_res_424_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__0(void){
_start:
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; 
v___x_425_ = lean_box(0);
v___x_426_ = lean_unsigned_to_nat(16u);
v___x_427_ = lean_mk_array(v___x_426_, v___x_425_);
return v___x_427_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__1(void){
_start:
{
lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v_preGoalMap_430_; 
v___x_428_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_toStepTree___closed__0, &lp_aesop_Aesop_Script_UScript_toStepTree___closed__0_once, _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__0);
v___x_429_ = lean_unsigned_to_nat(0u);
v_preGoalMap_430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_preGoalMap_430_, 0, v___x_429_);
lean_ctor_set(v_preGoalMap_430_, 1, v___x_428_);
return v_preGoalMap_430_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_toStepTree(lean_object* v_s_431_){
_start:
{
lean_object* v___x_432_; lean_object* v___x_433_; uint8_t v___x_434_; 
v___x_432_ = lean_unsigned_to_nat(0u);
v___x_433_ = lean_array_get_size(v_s_431_);
v___x_434_ = lean_nat_dec_lt(v___x_432_, v___x_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; 
v___x_435_ = lean_box(0);
return v___x_435_;
}
else
{
lean_object* v_preGoalMap_436_; lean_object* v___x_437_; lean_object* v_preGoal_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v_preGoalMap_436_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_toStepTree___closed__1, &lp_aesop_Aesop_Script_UScript_toStepTree___closed__1_once, _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__1);
v___x_437_ = lean_array_fget_borrowed(v_s_431_, v___x_432_);
v_preGoal_438_ = lean_ctor_get(v___x_437_, 1);
v___x_439_ = lean_unsigned_to_nat(1u);
v___x_440_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_440_, 0, v___x_432_);
lean_ctor_set(v___x_440_, 1, v___x_433_);
lean_ctor_set(v___x_440_, 2, v___x_439_);
v___x_441_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___redArg(v_s_431_, v___x_440_, v_preGoalMap_436_, v___x_432_);
lean_dec_ref_known(v___x_440_, 3);
v___x_442_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go(v___x_441_, v_preGoal_438_);
lean_dec_ref(v___x_441_);
return v___x_442_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_UScript_toStepTree___boxed(lean_object* v_s_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_aesop_Aesop_Script_UScript_toStepTree(v_s_443_);
lean_dec_ref(v_s_443_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0(lean_object* v_00_u03b2_445_, lean_object* v_m_446_, lean_object* v_a_447_, lean_object* v_b_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0___redArg(v_m_446_, v_a_447_, v_b_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1(lean_object* v_s_450_, lean_object* v_range_451_, lean_object* v_b_452_, lean_object* v_i_453_, lean_object* v_hs_454_, lean_object* v_hl_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___redArg(v_s_450_, v_range_451_, v_b_452_, v_i_453_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1___boxed(lean_object* v_s_457_, lean_object* v_range_458_, lean_object* v_b_459_, lean_object* v_i_460_, lean_object* v_hs_461_, lean_object* v_hl_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_Script_UScript_toStepTree_spec__1(v_s_457_, v_range_458_, v_b_459_, v_i_460_, v_hs_461_, v_hl_462_);
lean_dec_ref(v_range_458_);
lean_dec_ref(v_s_457_);
return v_res_463_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0(lean_object* v_00_u03b2_464_, lean_object* v_a_465_, lean_object* v_x_466_){
_start:
{
uint8_t v___x_467_; 
v___x_467_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg(v_a_465_, v_x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___boxed(lean_object* v_00_u03b2_468_, lean_object* v_a_469_, lean_object* v_x_470_){
_start:
{
uint8_t v_res_471_; lean_object* v_r_472_; 
v_res_471_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0(v_00_u03b2_468_, v_a_469_, v_x_470_);
lean_dec(v_x_470_);
lean_dec(v_a_469_);
v_r_472_ = lean_box(v_res_471_);
return v_r_472_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1(lean_object* v_00_u03b2_473_, lean_object* v_data_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1___redArg(v_data_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__2(lean_object* v_00_u03b2_476_, lean_object* v_a_477_, lean_object* v_b_478_, lean_object* v_x_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__2___redArg(v_a_477_, v_b_478_, v_x_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_481_, lean_object* v_i_482_, lean_object* v_source_483_, lean_object* v_target_484_){
_start:
{
lean_object* v___x_485_; 
v___x_485_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2___redArg(v_i_482_, v_source_483_, v_target_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_486_, lean_object* v_x_487_, lean_object* v_x_488_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1_spec__2_spec__4___redArg(v_x_487_, v_x_488_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___lam__0(lean_object* v_x1_490_, lean_object* v_x2_491_){
_start:
{
lean_object* v___x_492_; lean_object* v___x_493_; 
v___x_492_ = lean_array_get_size(v_x2_491_);
v___x_493_ = lean_nat_add(v_x1_490_, v___x_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg___lam__0___boxed(lean_object* v_x1_494_, lean_object* v_x2_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_aesop_Aesop_Script_sortDedupArrays___redArg___lam__0(v_x1_494_, v_x2_495_);
lean_dec_ref(v_x2_495_);
lean_dec(v_x1_494_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___redArg(lean_object* v_inst_518_, lean_object* v_as_519_){
_start:
{
lean_object* v___f_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___y_524_; lean_object* v___x_539_; uint8_t v___x_540_; 
v___f_520_ = ((lean_object*)(lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__0));
v___x_521_ = lean_unsigned_to_nat(0u);
v___x_522_ = lean_array_get_size(v_as_519_);
v___x_539_ = ((lean_object*)(lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__10));
v___x_540_ = lean_nat_dec_lt(v___x_521_, v___x_522_);
if (v___x_540_ == 0)
{
v___y_524_ = v___x_521_;
goto v___jp_523_;
}
else
{
lean_object* v___f_541_; uint8_t v___x_542_; 
v___f_541_ = ((lean_object*)(lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__11));
v___x_542_ = lean_nat_dec_le(v___x_522_, v___x_522_);
if (v___x_542_ == 0)
{
if (v___x_540_ == 0)
{
v___y_524_ = v___x_521_;
goto v___jp_523_;
}
else
{
size_t v___x_543_; size_t v___x_544_; lean_object* v___x_545_; 
v___x_543_ = ((size_t)0ULL);
v___x_544_ = lean_usize_of_nat(v___x_522_);
lean_inc_ref(v_as_519_);
v___x_545_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_539_, v___f_541_, v_as_519_, v___x_543_, v___x_544_, v___x_521_);
v___y_524_ = v___x_545_;
goto v___jp_523_;
}
}
else
{
size_t v___x_546_; size_t v___x_547_; lean_object* v___x_548_; 
v___x_546_ = ((size_t)0ULL);
v___x_547_ = lean_usize_of_nat(v___x_522_);
lean_inc_ref(v_as_519_);
v___x_548_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_539_, v___f_541_, v_as_519_, v___x_546_, v___x_547_, v___x_521_);
v___y_524_ = v___x_548_;
goto v___jp_523_;
}
}
v___jp_523_:
{
lean_object* v___x_525_; lean_object* v___x_526_; uint8_t v___x_527_; 
v___x_525_ = lean_mk_empty_array_with_capacity(v___y_524_);
lean_dec(v___y_524_);
v___x_526_ = ((lean_object*)(lp_aesop_Aesop_Script_sortDedupArrays___redArg___closed__10));
v___x_527_ = lean_nat_dec_lt(v___x_521_, v___x_522_);
if (v___x_527_ == 0)
{
lean_object* v___x_528_; 
lean_dec_ref(v_as_519_);
v___x_528_ = lp_batteries_Array_sortDedup___redArg(v_inst_518_, v___x_525_);
return v___x_528_;
}
else
{
uint8_t v___x_529_; 
v___x_529_ = lean_nat_dec_le(v___x_522_, v___x_522_);
if (v___x_529_ == 0)
{
if (v___x_527_ == 0)
{
lean_object* v___x_530_; 
lean_dec_ref(v_as_519_);
v___x_530_ = lp_batteries_Array_sortDedup___redArg(v_inst_518_, v___x_525_);
return v___x_530_;
}
else
{
size_t v___x_531_; size_t v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_531_ = ((size_t)0ULL);
v___x_532_ = lean_usize_of_nat(v___x_522_);
v___x_533_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_526_, v___f_520_, v_as_519_, v___x_531_, v___x_532_, v___x_525_);
v___x_534_ = lp_batteries_Array_sortDedup___redArg(v_inst_518_, v___x_533_);
return v___x_534_;
}
}
else
{
size_t v___x_535_; size_t v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; 
v___x_535_ = ((size_t)0ULL);
v___x_536_ = lean_usize_of_nat(v___x_522_);
v___x_537_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_526_, v___f_520_, v_as_519_, v___x_535_, v___x_536_, v___x_525_);
v___x_538_ = lp_batteries_Array_sortDedup___redArg(v_inst_518_, v___x_537_);
return v___x_538_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays(lean_object* v_00_u03b1_549_, lean_object* v_inst_550_, lean_object* v_as_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lp_aesop_Aesop_Script_sortDedupArrays___redArg(v_inst_550_, v_as_551_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_isConsecutiveSequence_spec__0___redArg(lean_object* v_a_553_, lean_object* v_b_554_){
_start:
{
lean_object* v_array_555_; lean_object* v_start_556_; lean_object* v_stop_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_586_; 
v_array_555_ = lean_ctor_get(v_a_553_, 0);
v_start_556_ = lean_ctor_get(v_a_553_, 1);
v_stop_557_ = lean_ctor_get(v_a_553_, 2);
v_isSharedCheck_586_ = !lean_is_exclusive(v_a_553_);
if (v_isSharedCheck_586_ == 0)
{
v___x_559_ = v_a_553_;
v_isShared_560_ = v_isSharedCheck_586_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_stop_557_);
lean_inc(v_start_556_);
lean_inc(v_array_555_);
lean_dec(v_a_553_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_586_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
uint8_t v___x_561_; 
v___x_561_ = lean_nat_dec_lt(v_start_556_, v_stop_557_);
if (v___x_561_ == 0)
{
lean_del_object(v___x_559_);
lean_dec(v_stop_557_);
lean_dec(v_start_556_);
lean_dec_ref(v_array_555_);
return v_b_554_;
}
else
{
lean_object* v_snd_562_; lean_object* v___x_564_; uint8_t v_isShared_565_; uint8_t v_isSharedCheck_584_; 
v_snd_562_ = lean_ctor_get(v_b_554_, 1);
v_isSharedCheck_584_ = !lean_is_exclusive(v_b_554_);
if (v_isSharedCheck_584_ == 0)
{
lean_object* v_unused_585_; 
v_unused_585_ = lean_ctor_get(v_b_554_, 0);
lean_dec(v_unused_585_);
v___x_564_ = v_b_554_;
v_isShared_565_ = v_isSharedCheck_584_;
goto v_resetjp_563_;
}
else
{
lean_inc(v_snd_562_);
lean_dec(v_b_554_);
v___x_564_ = lean_box(0);
v_isShared_565_ = v_isSharedCheck_584_;
goto v_resetjp_563_;
}
v_resetjp_563_:
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; uint8_t v___x_569_; 
v___x_566_ = lean_unsigned_to_nat(1u);
v___x_567_ = lean_array_fget(v_array_555_, v_start_556_);
v___x_568_ = lean_nat_add(v_snd_562_, v___x_566_);
v___x_569_ = lean_nat_dec_eq(v___x_567_, v___x_568_);
lean_dec(v___x_568_);
if (v___x_569_ == 0)
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_573_; 
lean_dec(v___x_567_);
lean_del_object(v___x_559_);
lean_dec(v_stop_557_);
lean_dec(v_start_556_);
lean_dec_ref(v_array_555_);
v___x_570_ = lean_box(v___x_569_);
v___x_571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_571_, 0, v___x_570_);
if (v_isShared_565_ == 0)
{
lean_ctor_set(v___x_564_, 0, v___x_571_);
v___x_573_ = v___x_564_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v___x_571_);
lean_ctor_set(v_reuseFailAlloc_574_, 1, v_snd_562_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
return v___x_573_;
}
}
else
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_578_; 
lean_dec(v_snd_562_);
v___x_575_ = lean_box(0);
v___x_576_ = lean_nat_add(v_start_556_, v___x_566_);
lean_dec(v_start_556_);
if (v_isShared_560_ == 0)
{
lean_ctor_set(v___x_559_, 1, v___x_576_);
v___x_578_ = v___x_559_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v_array_555_);
lean_ctor_set(v_reuseFailAlloc_583_, 1, v___x_576_);
lean_ctor_set(v_reuseFailAlloc_583_, 2, v_stop_557_);
v___x_578_ = v_reuseFailAlloc_583_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
lean_object* v___x_580_; 
if (v_isShared_565_ == 0)
{
lean_ctor_set(v___x_564_, 1, v___x_567_);
lean_ctor_set(v___x_564_, 0, v___x_575_);
v___x_580_ = v___x_564_;
goto v_reusejp_579_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v___x_575_);
lean_ctor_set(v_reuseFailAlloc_582_, 1, v___x_567_);
v___x_580_ = v_reuseFailAlloc_582_;
goto v_reusejp_579_;
}
v_reusejp_579_:
{
v_a_553_ = v___x_578_;
v_b_554_ = v___x_580_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Script_isConsecutiveSequence(lean_object* v_ns_587_){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; uint8_t v___x_590_; 
v___x_588_ = lean_unsigned_to_nat(0u);
v___x_589_ = lean_array_get_size(v_ns_587_);
v___x_590_ = lean_nat_dec_lt(v___x_588_, v___x_589_);
if (v___x_590_ == 0)
{
uint8_t v___x_591_; 
lean_dec_ref(v_ns_587_);
v___x_591_ = 1;
return v___x_591_;
}
else
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v_fst_598_; 
v___x_592_ = lean_array_fget(v_ns_587_, v___x_588_);
v___x_593_ = lean_unsigned_to_nat(1u);
v___x_594_ = l_Array_toSubarray___redArg(v_ns_587_, v___x_593_, v___x_589_);
v___x_595_ = lean_box(0);
v___x_596_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_595_);
lean_ctor_set(v___x_596_, 1, v___x_592_);
v___x_597_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_isConsecutiveSequence_spec__0___redArg(v___x_594_, v___x_596_);
v_fst_598_ = lean_ctor_get(v___x_597_, 0);
lean_inc(v_fst_598_);
lean_dec_ref(v___x_597_);
if (lean_obj_tag(v_fst_598_) == 0)
{
return v___x_590_;
}
else
{
lean_object* v_val_599_; uint8_t v___x_600_; 
v_val_599_ = lean_ctor_get(v_fst_598_, 0);
lean_inc(v_val_599_);
lean_dec_ref_known(v_fst_598_, 1);
v___x_600_ = lean_unbox(v_val_599_);
lean_dec(v_val_599_);
return v___x_600_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_isConsecutiveSequence___boxed(lean_object* v_ns_601_){
_start:
{
uint8_t v_res_602_; lean_object* v_r_603_; 
v_res_602_ = lp_aesop_Aesop_Script_isConsecutiveSequence(v_ns_601_);
v_r_603_ = lean_box(v_res_602_);
return v_r_603_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_isConsecutiveSequence_spec__0(lean_object* v_inst_604_, lean_object* v_R_605_, lean_object* v_a_606_, lean_object* v_b_607_, lean_object* v_c_608_){
_start:
{
lean_object* v___x_609_; 
v___x_609_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_Script_isConsecutiveSequence_spec__0___redArg(v_a_606_, v_b_607_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___lam__0(lean_object* v_x_610_, lean_object* v_x_611_){
_start:
{
lean_inc(v_x_610_);
return v_x_610_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___lam__0___boxed(lean_object* v_x_612_, lean_object* v_x_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___lam__0(v_x_612_, v_x_613_);
lean_dec(v_x_613_);
lean_dec(v_x_612_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5_spec__8(lean_object* v_f_615_, lean_object* v_xs_616_, lean_object* v_acc_617_, lean_object* v_i_618_, lean_object* v_hd_619_){
_start:
{
lean_object* v___x_620_; uint8_t v___x_621_; 
v___x_620_ = lean_array_get_size(v_xs_616_);
v___x_621_ = lean_nat_dec_lt(v_i_618_, v___x_620_);
if (v___x_621_ == 0)
{
lean_object* v___x_622_; 
lean_dec(v_i_618_);
lean_dec_ref(v_f_615_);
v___x_622_ = lean_array_push(v_acc_617_, v_hd_619_);
return v___x_622_;
}
else
{
lean_object* v_x_623_; uint8_t v___x_629_; 
v_x_623_ = lean_array_fget_borrowed(v_xs_616_, v_i_618_);
v___x_629_ = lean_nat_dec_lt(v_x_623_, v_hd_619_);
if (v___x_629_ == 0)
{
uint8_t v___x_630_; 
v___x_630_ = lean_nat_dec_eq(v_x_623_, v_hd_619_);
if (v___x_630_ == 0)
{
goto v___jp_624_;
}
else
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v___x_631_ = lean_unsigned_to_nat(1u);
v___x_632_ = lean_nat_add(v_i_618_, v___x_631_);
lean_dec(v_i_618_);
lean_inc_ref(v_f_615_);
lean_inc(v_x_623_);
v___x_633_ = lean_apply_2(v_f_615_, v_hd_619_, v_x_623_);
v_i_618_ = v___x_632_;
v_hd_619_ = v___x_633_;
goto _start;
}
}
else
{
goto v___jp_624_;
}
v___jp_624_:
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_625_ = lean_array_push(v_acc_617_, v_hd_619_);
v___x_626_ = lean_unsigned_to_nat(1u);
v___x_627_ = lean_nat_add(v_i_618_, v___x_626_);
lean_dec(v_i_618_);
lean_inc(v_x_623_);
v_acc_617_ = v___x_625_;
v_i_618_ = v___x_627_;
v_hd_619_ = v_x_623_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5_spec__8___boxed(lean_object* v_f_635_, lean_object* v_xs_636_, lean_object* v_acc_637_, lean_object* v_i_638_, lean_object* v_hd_639_){
_start:
{
lean_object* v_res_640_; 
v_res_640_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5_spec__8(v_f_635_, v_xs_636_, v_acc_637_, v_i_638_, v_hd_639_);
lean_dec_ref(v_xs_636_);
return v_res_640_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5(lean_object* v_f_641_, lean_object* v_xs_642_){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; uint8_t v___x_645_; 
v___x_643_ = lean_unsigned_to_nat(0u);
v___x_644_ = lean_array_get_size(v_xs_642_);
v___x_645_ = lean_nat_dec_lt(v___x_643_, v___x_644_);
if (v___x_645_ == 0)
{
lean_dec_ref(v_f_641_);
lean_inc_ref(v_xs_642_);
return v_xs_642_;
}
else
{
lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; 
v___x_646_ = lean_mk_empty_array_with_capacity(v___x_644_);
v___x_647_ = lean_unsigned_to_nat(1u);
v___x_648_ = lean_array_fget_borrowed(v_xs_642_, v___x_643_);
lean_inc(v___x_648_);
v___x_649_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5_spec__8(v_f_641_, v_xs_642_, v___x_646_, v___x_647_, v___x_648_);
return v___x_649_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5___boxed(lean_object* v_f_650_, lean_object* v_xs_651_){
_start:
{
lean_object* v_res_652_; 
v_res_652_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5(v_f_650_, v_xs_651_);
lean_dec_ref(v_xs_651_);
return v_res_652_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3(lean_object* v_xs_654_){
_start:
{
lean_object* v___f_655_; lean_object* v___x_656_; 
v___f_655_ = ((lean_object*)(lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___closed__0));
v___x_656_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3_spec__5(v___f_655_, v_xs_654_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3___boxed(lean_object* v_xs_657_){
_start:
{
lean_object* v_res_658_; 
v_res_658_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3(v_xs_657_);
lean_dec_ref(v_xs_657_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___redArg(lean_object* v_hi_659_, lean_object* v_pivot_660_, lean_object* v_as_661_, lean_object* v_i_662_, lean_object* v_k_663_){
_start:
{
uint8_t v___x_664_; 
v___x_664_ = lean_nat_dec_lt(v_k_663_, v_hi_659_);
if (v___x_664_ == 0)
{
lean_object* v___x_665_; lean_object* v___x_666_; 
lean_dec(v_k_663_);
v___x_665_ = lean_array_fswap(v_as_661_, v_i_662_, v_hi_659_);
v___x_666_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_666_, 0, v_i_662_);
lean_ctor_set(v___x_666_, 1, v___x_665_);
return v___x_666_;
}
else
{
lean_object* v___x_667_; uint8_t v___x_668_; 
v___x_667_ = lean_array_fget_borrowed(v_as_661_, v_k_663_);
v___x_668_ = lean_nat_dec_lt(v___x_667_, v_pivot_660_);
if (v___x_668_ == 0)
{
lean_object* v___x_669_; lean_object* v___x_670_; 
v___x_669_ = lean_unsigned_to_nat(1u);
v___x_670_ = lean_nat_add(v_k_663_, v___x_669_);
lean_dec(v_k_663_);
v_k_663_ = v___x_670_;
goto _start;
}
else
{
lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; 
v___x_672_ = lean_array_fswap(v_as_661_, v_i_662_, v_k_663_);
v___x_673_ = lean_unsigned_to_nat(1u);
v___x_674_ = lean_nat_add(v_i_662_, v___x_673_);
lean_dec(v_i_662_);
v___x_675_ = lean_nat_add(v_k_663_, v___x_673_);
lean_dec(v_k_663_);
v_as_661_ = v___x_672_;
v_i_662_ = v___x_674_;
v_k_663_ = v___x_675_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___redArg___boxed(lean_object* v_hi_677_, lean_object* v_pivot_678_, lean_object* v_as_679_, lean_object* v_i_680_, lean_object* v_k_681_){
_start:
{
lean_object* v_res_682_; 
v_res_682_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___redArg(v_hi_677_, v_pivot_678_, v_as_679_, v_i_680_, v_k_681_);
lean_dec(v_pivot_678_);
lean_dec(v_hi_677_);
return v_res_682_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg(lean_object* v_n_683_, lean_object* v_as_684_, lean_object* v_lo_685_, lean_object* v_hi_686_){
_start:
{
lean_object* v___y_688_; uint8_t v___x_698_; 
v___x_698_ = lean_nat_dec_lt(v_lo_685_, v_hi_686_);
if (v___x_698_ == 0)
{
lean_dec(v_lo_685_);
return v_as_684_;
}
else
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v_mid_701_; lean_object* v___y_703_; lean_object* v___y_709_; lean_object* v___x_714_; lean_object* v___x_715_; uint8_t v___x_716_; 
v___x_699_ = lean_nat_add(v_lo_685_, v_hi_686_);
v___x_700_ = lean_unsigned_to_nat(1u);
v_mid_701_ = lean_nat_shiftr(v___x_699_, v___x_700_);
lean_dec(v___x_699_);
v___x_714_ = lean_array_fget_borrowed(v_as_684_, v_mid_701_);
v___x_715_ = lean_array_fget_borrowed(v_as_684_, v_lo_685_);
v___x_716_ = lean_nat_dec_lt(v___x_714_, v___x_715_);
if (v___x_716_ == 0)
{
v___y_709_ = v_as_684_;
goto v___jp_708_;
}
else
{
lean_object* v___x_717_; 
v___x_717_ = lean_array_fswap(v_as_684_, v_lo_685_, v_mid_701_);
v___y_709_ = v___x_717_;
goto v___jp_708_;
}
v___jp_702_:
{
lean_object* v___x_704_; lean_object* v___x_705_; uint8_t v___x_706_; 
v___x_704_ = lean_array_fget_borrowed(v___y_703_, v_mid_701_);
v___x_705_ = lean_array_fget_borrowed(v___y_703_, v_hi_686_);
v___x_706_ = lean_nat_dec_lt(v___x_704_, v___x_705_);
if (v___x_706_ == 0)
{
lean_dec(v_mid_701_);
v___y_688_ = v___y_703_;
goto v___jp_687_;
}
else
{
lean_object* v___x_707_; 
v___x_707_ = lean_array_fswap(v___y_703_, v_mid_701_, v_hi_686_);
lean_dec(v_mid_701_);
v___y_688_ = v___x_707_;
goto v___jp_687_;
}
}
v___jp_708_:
{
lean_object* v___x_710_; lean_object* v___x_711_; uint8_t v___x_712_; 
v___x_710_ = lean_array_fget_borrowed(v___y_709_, v_hi_686_);
v___x_711_ = lean_array_fget_borrowed(v___y_709_, v_lo_685_);
v___x_712_ = lean_nat_dec_lt(v___x_710_, v___x_711_);
if (v___x_712_ == 0)
{
v___y_703_ = v___y_709_;
goto v___jp_702_;
}
else
{
lean_object* v___x_713_; 
v___x_713_ = lean_array_fswap(v___y_709_, v_lo_685_, v_hi_686_);
v___y_703_ = v___x_713_;
goto v___jp_702_;
}
}
}
v___jp_687_:
{
lean_object* v_pivot_689_; lean_object* v___x_690_; lean_object* v_fst_691_; lean_object* v_snd_692_; uint8_t v___x_693_; 
v_pivot_689_ = lean_array_fget(v___y_688_, v_hi_686_);
lean_inc_n(v_lo_685_, 2);
v___x_690_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___redArg(v_hi_686_, v_pivot_689_, v___y_688_, v_lo_685_, v_lo_685_);
lean_dec(v_pivot_689_);
v_fst_691_ = lean_ctor_get(v___x_690_, 0);
lean_inc(v_fst_691_);
v_snd_692_ = lean_ctor_get(v___x_690_, 1);
lean_inc(v_snd_692_);
lean_dec_ref(v___x_690_);
v___x_693_ = lean_nat_dec_le(v_hi_686_, v_fst_691_);
if (v___x_693_ == 0)
{
lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; 
v___x_694_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg(v_n_683_, v_snd_692_, v_lo_685_, v_fst_691_);
v___x_695_ = lean_unsigned_to_nat(1u);
v___x_696_ = lean_nat_add(v_fst_691_, v___x_695_);
lean_dec(v_fst_691_);
v_as_684_ = v___x_694_;
v_lo_685_ = v___x_696_;
goto _start;
}
else
{
lean_dec(v_fst_691_);
lean_dec(v_lo_685_);
return v_snd_692_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_n_718_, lean_object* v_as_719_, lean_object* v_lo_720_, lean_object* v_hi_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg(v_n_718_, v_as_719_, v_lo_720_, v_hi_721_);
lean_dec(v_hi_721_);
lean_dec(v_n_718_);
return v_res_722_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1(lean_object* v_xs_723_){
_start:
{
lean_object* v___x_724_; lean_object* v___y_726_; lean_object* v___y_727_; lean_object* v___x_730_; uint8_t v___x_731_; 
v___x_724_ = lean_array_get_size(v_xs_723_);
v___x_730_ = lean_unsigned_to_nat(0u);
v___x_731_ = lean_nat_dec_eq(v___x_724_, v___x_730_);
if (v___x_731_ == 0)
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___y_735_; uint8_t v___x_737_; 
v___x_732_ = lean_unsigned_to_nat(1u);
v___x_733_ = lean_nat_sub(v___x_724_, v___x_732_);
v___x_737_ = lean_nat_dec_le(v___x_730_, v___x_733_);
if (v___x_737_ == 0)
{
lean_inc(v___x_733_);
v___y_735_ = v___x_733_;
goto v___jp_734_;
}
else
{
v___y_735_ = v___x_730_;
goto v___jp_734_;
}
v___jp_734_:
{
uint8_t v___x_736_; 
v___x_736_ = lean_nat_dec_le(v___y_735_, v___x_733_);
if (v___x_736_ == 0)
{
lean_dec(v___x_733_);
lean_inc(v___y_735_);
v___y_726_ = v___y_735_;
v___y_727_ = v___y_735_;
goto v___jp_725_;
}
else
{
v___y_726_ = v___y_735_;
v___y_727_ = v___x_733_;
goto v___jp_725_;
}
}
}
else
{
lean_object* v___x_738_; 
v___x_738_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3(v_xs_723_);
lean_dec_ref(v_xs_723_);
return v___x_738_;
}
v___jp_725_:
{
lean_object* v___x_728_; lean_object* v___x_729_; 
v___x_728_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg(v___x_724_, v_xs_723_, v___y_726_, v___y_727_);
lean_dec(v___y_727_);
v___x_729_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__3(v___x_728_);
lean_dec_ref(v___x_728_);
return v___x_729_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__3(lean_object* v_as_739_, size_t v_i_740_, size_t v_stop_741_, lean_object* v_b_742_){
_start:
{
uint8_t v___x_743_; 
v___x_743_ = lean_usize_dec_eq(v_i_740_, v_stop_741_);
if (v___x_743_ == 0)
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; size_t v___x_747_; size_t v___x_748_; 
v___x_744_ = lean_array_uget_borrowed(v_as_739_, v_i_740_);
v___x_745_ = lean_array_get_size(v___x_744_);
v___x_746_ = lean_nat_add(v_b_742_, v___x_745_);
lean_dec(v_b_742_);
v___x_747_ = ((size_t)1ULL);
v___x_748_ = lean_usize_add(v_i_740_, v___x_747_);
v_i_740_ = v___x_748_;
v_b_742_ = v___x_746_;
goto _start;
}
else
{
return v_b_742_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__3___boxed(lean_object* v_as_750_, lean_object* v_i_751_, lean_object* v_stop_752_, lean_object* v_b_753_){
_start:
{
size_t v_i_boxed_754_; size_t v_stop_boxed_755_; lean_object* v_res_756_; 
v_i_boxed_754_ = lean_unbox_usize(v_i_751_);
lean_dec(v_i_751_);
v_stop_boxed_755_ = lean_unbox_usize(v_stop_752_);
lean_dec(v_stop_752_);
v_res_756_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__3(v_as_750_, v_i_boxed_754_, v_stop_boxed_755_, v_b_753_);
lean_dec_ref(v_as_750_);
return v_res_756_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__2(lean_object* v_as_757_, size_t v_i_758_, size_t v_stop_759_, lean_object* v_b_760_){
_start:
{
uint8_t v___x_761_; 
v___x_761_ = lean_usize_dec_eq(v_i_758_, v_stop_759_);
if (v___x_761_ == 0)
{
lean_object* v___x_762_; lean_object* v___x_763_; size_t v___x_764_; size_t v___x_765_; 
v___x_762_ = lean_array_uget_borrowed(v_as_757_, v_i_758_);
v___x_763_ = l_Array_append___redArg(v_b_760_, v___x_762_);
v___x_764_ = ((size_t)1ULL);
v___x_765_ = lean_usize_add(v_i_758_, v___x_764_);
v_i_758_ = v___x_765_;
v_b_760_ = v___x_763_;
goto _start;
}
else
{
return v_b_760_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__2___boxed(lean_object* v_as_767_, lean_object* v_i_768_, lean_object* v_stop_769_, lean_object* v_b_770_){
_start:
{
size_t v_i_boxed_771_; size_t v_stop_boxed_772_; lean_object* v_res_773_; 
v_i_boxed_771_ = lean_unbox_usize(v_i_768_);
lean_dec(v_i_768_);
v_stop_boxed_772_ = lean_unbox_usize(v_stop_769_);
lean_dec(v_stop_769_);
v_res_773_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__2(v_as_767_, v_i_boxed_771_, v_stop_boxed_772_, v_b_770_);
lean_dec_ref(v_as_767_);
return v_res_773_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1(lean_object* v_as_774_){
_start:
{
lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___y_778_; uint8_t v___x_792_; 
v___x_775_ = lean_unsigned_to_nat(0u);
v___x_776_ = lean_array_get_size(v_as_774_);
v___x_792_ = lean_nat_dec_lt(v___x_775_, v___x_776_);
if (v___x_792_ == 0)
{
v___y_778_ = v___x_775_;
goto v___jp_777_;
}
else
{
uint8_t v___x_793_; 
v___x_793_ = lean_nat_dec_le(v___x_776_, v___x_776_);
if (v___x_793_ == 0)
{
if (v___x_792_ == 0)
{
v___y_778_ = v___x_775_;
goto v___jp_777_;
}
else
{
size_t v___x_794_; size_t v___x_795_; lean_object* v___x_796_; 
v___x_794_ = ((size_t)0ULL);
v___x_795_ = lean_usize_of_nat(v___x_776_);
v___x_796_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__3(v_as_774_, v___x_794_, v___x_795_, v___x_775_);
v___y_778_ = v___x_796_;
goto v___jp_777_;
}
}
else
{
size_t v___x_797_; size_t v___x_798_; lean_object* v___x_799_; 
v___x_797_ = ((size_t)0ULL);
v___x_798_ = lean_usize_of_nat(v___x_776_);
v___x_799_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__3(v_as_774_, v___x_797_, v___x_798_, v___x_775_);
v___y_778_ = v___x_799_;
goto v___jp_777_;
}
}
v___jp_777_:
{
lean_object* v___x_779_; uint8_t v___x_780_; 
v___x_779_ = lean_mk_empty_array_with_capacity(v___y_778_);
lean_dec(v___y_778_);
v___x_780_ = lean_nat_dec_lt(v___x_775_, v___x_776_);
if (v___x_780_ == 0)
{
lean_object* v___x_781_; 
v___x_781_ = lp_aesop_Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1(v___x_779_);
return v___x_781_;
}
else
{
uint8_t v___x_782_; 
v___x_782_ = lean_nat_dec_le(v___x_776_, v___x_776_);
if (v___x_782_ == 0)
{
if (v___x_780_ == 0)
{
lean_object* v___x_783_; 
v___x_783_ = lp_aesop_Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1(v___x_779_);
return v___x_783_;
}
else
{
size_t v___x_784_; size_t v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; 
v___x_784_ = ((size_t)0ULL);
v___x_785_ = lean_usize_of_nat(v___x_776_);
v___x_786_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__2(v_as_774_, v___x_784_, v___x_785_, v___x_779_);
v___x_787_ = lp_aesop_Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1(v___x_786_);
return v___x_787_;
}
}
else
{
size_t v___x_788_; size_t v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; 
v___x_788_ = ((size_t)0ULL);
v___x_789_ = lean_usize_of_nat(v___x_776_);
v___x_790_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__2(v_as_774_, v___x_788_, v___x_789_, v___x_779_);
v___x_791_ = lp_aesop_Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1(v___x_790_);
return v___x_791_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1___boxed(lean_object* v_as_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_aesop_Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1(v_as_800_);
lean_dec_ref(v_as_800_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg(lean_object* v_a_804_, lean_object* v_a_805_){
_start:
{
if (lean_obj_tag(v_a_804_) == 0)
{
lean_object* v___x_807_; 
v___x_807_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg___closed__0));
return v___x_807_;
}
else
{
lean_object* v_step_808_; lean_object* v_index_809_; lean_object* v_children_810_; size_t v_sz_811_; size_t v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___y_822_; uint8_t v___x_827_; 
v_step_808_ = lean_ctor_get(v_a_804_, 0);
lean_inc_ref(v_step_808_);
v_index_809_ = lean_ctor_get(v_a_804_, 1);
lean_inc_n(v_index_809_, 2);
v_children_810_ = lean_ctor_get(v_a_804_, 2);
lean_inc_ref(v_children_810_);
lean_dec_ref_known(v_a_804_, 3);
v_sz_811_ = lean_array_size(v_children_810_);
v___x_812_ = ((size_t)0ULL);
v___x_813_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___redArg(v_sz_811_, v___x_812_, v_children_810_, v_a_805_);
v___x_814_ = lp_aesop_Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1(v___x_813_);
lean_dec_ref(v___x_813_);
v___x_815_ = lean_array_get_size(v___x_814_);
v___x_816_ = lean_unsigned_to_nat(1u);
v___x_817_ = lean_nat_add(v___x_815_, v___x_816_);
v___x_818_ = lean_mk_empty_array_with_capacity(v___x_817_);
lean_dec(v___x_817_);
v___x_819_ = lean_array_push(v___x_818_, v_index_809_);
v___x_820_ = l_Array_append___redArg(v___x_819_, v___x_814_);
lean_inc_ref(v___x_820_);
v___x_827_ = lp_aesop_Aesop_Script_isConsecutiveSequence(v___x_820_);
if (v___x_827_ == 0)
{
lean_dec_ref(v___x_814_);
lean_dec(v_index_809_);
lean_dec_ref(v_step_808_);
return v___x_820_;
}
else
{
lean_object* v___x_828_; uint8_t v___x_829_; 
v___x_828_ = lean_nat_sub(v___x_815_, v___x_816_);
v___x_829_ = lean_nat_dec_lt(v___x_828_, v___x_815_);
if (v___x_829_ == 0)
{
lean_dec(v___x_828_);
lean_dec_ref(v___x_814_);
v___y_822_ = v_index_809_;
goto v___jp_821_;
}
else
{
lean_object* v___x_830_; 
lean_dec(v_index_809_);
v___x_830_ = lean_array_fget(v___x_814_, v___x_828_);
lean_dec(v___x_828_);
lean_dec_ref(v___x_814_);
v___y_822_ = v___x_830_;
goto v___jp_821_;
}
}
v___jp_821_:
{
lean_object* v___x_823_; lean_object* v_preGoal_824_; lean_object* v___x_825_; lean_object* v___x_826_; 
v___x_823_ = lean_st_ref_take(v_a_805_);
v_preGoal_824_ = lean_ctor_get(v_step_808_, 1);
lean_inc(v_preGoal_824_);
lean_dec_ref(v_step_808_);
v___x_825_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0___redArg(v___x_823_, v_preGoal_824_, v___y_822_);
v___x_826_ = lean_st_ref_set(v_a_805_, v___x_825_);
return v___x_820_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___redArg(size_t v_sz_831_, size_t v_i_832_, lean_object* v_bs_833_, lean_object* v___y_834_){
_start:
{
uint8_t v___x_836_; 
v___x_836_ = lean_usize_dec_lt(v_i_832_, v_sz_831_);
if (v___x_836_ == 0)
{
return v_bs_833_;
}
else
{
lean_object* v_v_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v_bs_x27_840_; size_t v___x_841_; size_t v___x_842_; lean_object* v___x_843_; 
v_v_837_ = lean_array_uget_borrowed(v_bs_833_, v_i_832_);
lean_inc(v_v_837_);
v___x_838_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg(v_v_837_, v___y_834_);
v___x_839_ = lean_unsigned_to_nat(0u);
v_bs_x27_840_ = lean_array_uset(v_bs_833_, v_i_832_, v___x_839_);
v___x_841_ = ((size_t)1ULL);
v___x_842_ = lean_usize_add(v_i_832_, v___x_841_);
v___x_843_ = lean_array_uset(v_bs_x27_840_, v_i_832_, v___x_838_);
v_i_832_ = v___x_842_;
v_bs_833_ = v___x_843_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___redArg___boxed(lean_object* v_sz_845_, lean_object* v_i_846_, lean_object* v_bs_847_, lean_object* v___y_848_, lean_object* v___y_849_){
_start:
{
size_t v_sz_boxed_850_; size_t v_i_boxed_851_; lean_object* v_res_852_; 
v_sz_boxed_850_ = lean_unbox_usize(v_sz_845_);
lean_dec(v_sz_845_);
v_i_boxed_851_ = lean_unbox_usize(v_i_846_);
lean_dec(v_i_846_);
v_res_852_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___redArg(v_sz_boxed_850_, v_i_boxed_851_, v_bs_847_, v___y_848_);
lean_dec(v___y_848_);
return v_res_852_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg___boxed(lean_object* v_a_853_, lean_object* v_a_854_, lean_object* v_a_855_){
_start:
{
lean_object* v_res_856_; 
v_res_856_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg(v_a_853_, v_a_854_);
lean_dec(v_a_854_);
return v_res_856_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go(lean_object* v_00_u03c3_857_, lean_object* v_a_858_, lean_object* v_a_859_){
_start:
{
lean_object* v___x_861_; 
v___x_861_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg(v_a_858_, v_a_859_);
return v___x_861_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___boxed(lean_object* v_00_u03c3_862_, lean_object* v_a_863_, lean_object* v_a_864_, lean_object* v_a_865_){
_start:
{
lean_object* v_res_866_; 
v_res_866_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go(v_00_u03c3_862_, v_a_863_, v_a_864_);
lean_dec(v_a_864_);
return v_res_866_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0(lean_object* v_00_u03c3_867_, size_t v_sz_868_, size_t v_i_869_, lean_object* v_bs_870_, lean_object* v___y_871_){
_start:
{
lean_object* v___x_873_; 
v___x_873_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___redArg(v_sz_868_, v_i_869_, v_bs_870_, v___y_871_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0___boxed(lean_object* v_00_u03c3_874_, lean_object* v_sz_875_, lean_object* v_i_876_, lean_object* v_bs_877_, lean_object* v___y_878_, lean_object* v___y_879_){
_start:
{
size_t v_sz_boxed_880_; size_t v_i_boxed_881_; lean_object* v_res_882_; 
v_sz_boxed_880_ = lean_unbox_usize(v_sz_875_);
lean_dec(v_sz_875_);
v_i_boxed_881_ = lean_unbox_usize(v_i_876_);
lean_dec(v_i_876_);
v_res_882_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__0(v_00_u03c3_874_, v_sz_boxed_880_, v_i_boxed_881_, v_bs_877_, v___y_878_);
lean_dec(v___y_878_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2(lean_object* v_n_883_, lean_object* v_as_884_, lean_object* v_lo_885_, lean_object* v_hi_886_, lean_object* v_w_887_, lean_object* v_hlo_888_, lean_object* v_hhi_889_){
_start:
{
lean_object* v___x_890_; 
v___x_890_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___redArg(v_n_883_, v_as_884_, v_lo_885_, v_hi_886_);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2___boxed(lean_object* v_n_891_, lean_object* v_as_892_, lean_object* v_lo_893_, lean_object* v_hi_894_, lean_object* v_w_895_, lean_object* v_hlo_896_, lean_object* v_hhi_897_){
_start:
{
lean_object* v_res_898_; 
v_res_898_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2(v_n_891_, v_as_892_, v_lo_893_, v_hi_894_, v_w_895_, v_hlo_896_, v_hhi_897_);
lean_dec(v_hi_894_);
lean_dec(v_n_891_);
return v_res_898_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3(lean_object* v_n_899_, lean_object* v_lo_900_, lean_object* v_hi_901_, lean_object* v_hhi_902_, lean_object* v_pivot_903_, lean_object* v_as_904_, lean_object* v_i_905_, lean_object* v_k_906_, lean_object* v_ilo_907_, lean_object* v_ik_908_, lean_object* v_w_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___redArg(v_hi_901_, v_pivot_903_, v_as_904_, v_i_905_, v_k_906_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3___boxed(lean_object* v_n_911_, lean_object* v_lo_912_, lean_object* v_hi_913_, lean_object* v_hhi_914_, lean_object* v_pivot_915_, lean_object* v_as_916_, lean_object* v_i_917_, lean_object* v_k_918_, lean_object* v_ilo_919_, lean_object* v_ik_920_, lean_object* v_w_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_Script_sortDedupArrays___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go_spec__1_spec__1_spec__2_spec__3(v_n_911_, v_lo_912_, v_hi_913_, v_hhi_914_, v_pivot_915_, v_as_916_, v_i_917_, v_k_918_, v_ilo_919_, v_ik_920_, v_w_921_);
lean_dec(v_pivot_915_);
lean_dec(v_hi_913_);
lean_dec(v_lo_912_);
lean_dec(v_n_911_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_focusableGoals___lam__0(lean_object* v_t_923_, lean_object* v_x_924_){
_start:
{
lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; 
v___x_926_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_toStepTree___closed__1, &lp_aesop_Aesop_Script_UScript_toStepTree___closed__1_once, _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__1);
v___x_927_ = lean_st_mk_ref(v___x_926_);
v___x_928_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_focusableGoals_go___redArg(v_t_923_, v___x_927_);
v___x_929_ = lean_st_ref_get(v___x_927_);
lean_dec(v___x_927_);
v___x_930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_930_, 0, v___x_928_);
lean_ctor_set(v___x_930_, 1, v___x_929_);
return v___x_930_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_focusableGoals___lam__0___boxed(lean_object* v_t_931_, lean_object* v_x_932_, lean_object* v___y_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_aesop_Aesop_Script_StepTree_focusableGoals___lam__0(v_t_931_, v_x_932_);
return v_res_934_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_focusableGoals(lean_object* v_t_935_){
_start:
{
lean_object* v___f_936_; lean_object* v___x_937_; lean_object* v_snd_938_; 
v___f_936_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_StepTree_focusableGoals___lam__0___boxed), 3, 1);
lean_closure_set(v___f_936_, 0, v_t_935_);
v___x_937_ = l_runST___redArg(v___f_936_);
v_snd_938_ = lean_ctor_get(v___x_937_, 1);
lean_inc(v_snd_938_);
lean_dec(v___x_937_);
return v_snd_938_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg(lean_object* v_parentNumGoals_939_, lean_object* v_a_940_, lean_object* v_a_941_){
_start:
{
if (lean_obj_tag(v_a_940_) == 0)
{
lean_object* v___x_943_; 
v___x_943_ = lean_box(0);
return v___x_943_;
}
else
{
lean_object* v_step_944_; lean_object* v_children_945_; lean_object* v___x_946_; lean_object* v_preGoal_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; uint8_t v___x_955_; 
v_step_944_ = lean_ctor_get(v_a_940_, 0);
lean_inc_ref(v_step_944_);
v_children_945_ = lean_ctor_get(v_a_940_, 2);
lean_inc_ref(v_children_945_);
lean_dec_ref_known(v_a_940_, 3);
v___x_946_ = lean_st_ref_take(v_a_941_);
v_preGoal_947_ = lean_ctor_get(v_step_944_, 1);
lean_inc(v_preGoal_947_);
lean_dec_ref(v_step_944_);
v___x_948_ = lean_unsigned_to_nat(1u);
v___x_949_ = lean_nat_sub(v_parentNumGoals_939_, v___x_948_);
v___x_950_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0___redArg(v___x_946_, v_preGoal_947_, v___x_949_);
v___x_951_ = lean_st_ref_set(v_a_941_, v___x_950_);
v___x_952_ = lean_array_get_size(v_children_945_);
v___x_953_ = lean_unsigned_to_nat(0u);
v___x_954_ = lean_box(0);
v___x_955_ = lean_nat_dec_lt(v___x_953_, v___x_952_);
if (v___x_955_ == 0)
{
lean_dec_ref(v_children_945_);
return v___x_954_;
}
else
{
uint8_t v___x_956_; 
v___x_956_ = lean_nat_dec_le(v___x_952_, v___x_952_);
if (v___x_956_ == 0)
{
if (v___x_955_ == 0)
{
lean_dec_ref(v_children_945_);
return v___x_954_;
}
else
{
size_t v___x_957_; size_t v___x_958_; lean_object* v___x_959_; 
v___x_957_ = ((size_t)0ULL);
v___x_958_ = lean_usize_of_nat(v___x_952_);
v___x_959_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg(v___x_952_, v_children_945_, v___x_957_, v___x_958_, v___x_954_, v_a_941_);
lean_dec_ref(v_children_945_);
return v___x_959_;
}
}
else
{
size_t v___x_960_; size_t v___x_961_; lean_object* v___x_962_; 
v___x_960_ = ((size_t)0ULL);
v___x_961_ = lean_usize_of_nat(v___x_952_);
v___x_962_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg(v___x_952_, v_children_945_, v___x_960_, v___x_961_, v___x_954_, v_a_941_);
lean_dec_ref(v_children_945_);
return v___x_962_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg(lean_object* v___x_963_, lean_object* v_as_964_, size_t v_i_965_, size_t v_stop_966_, lean_object* v_b_967_, lean_object* v___y_968_){
_start:
{
uint8_t v___x_970_; 
v___x_970_ = lean_usize_dec_eq(v_i_965_, v_stop_966_);
if (v___x_970_ == 0)
{
lean_object* v___x_971_; lean_object* v___x_972_; size_t v___x_973_; size_t v___x_974_; 
v___x_971_ = lean_array_uget_borrowed(v_as_964_, v_i_965_);
lean_inc(v___x_971_);
v___x_972_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg(v___x_963_, v___x_971_, v___y_968_);
v___x_973_ = ((size_t)1ULL);
v___x_974_ = lean_usize_add(v_i_965_, v___x_973_);
v_i_965_ = v___x_974_;
v_b_967_ = v___x_972_;
goto _start;
}
else
{
return v_b_967_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg___boxed(lean_object* v___x_976_, lean_object* v_as_977_, lean_object* v_i_978_, lean_object* v_stop_979_, lean_object* v_b_980_, lean_object* v___y_981_, lean_object* v___y_982_){
_start:
{
size_t v_i_boxed_983_; size_t v_stop_boxed_984_; lean_object* v_res_985_; 
v_i_boxed_983_ = lean_unbox_usize(v_i_978_);
lean_dec(v_i_978_);
v_stop_boxed_984_ = lean_unbox_usize(v_stop_979_);
lean_dec(v_stop_979_);
v_res_985_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg(v___x_976_, v_as_977_, v_i_boxed_983_, v_stop_boxed_984_, v_b_980_, v___y_981_);
lean_dec(v___y_981_);
lean_dec_ref(v_as_977_);
lean_dec(v___x_976_);
return v_res_985_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg___boxed(lean_object* v_parentNumGoals_986_, lean_object* v_a_987_, lean_object* v_a_988_, lean_object* v_a_989_){
_start:
{
lean_object* v_res_990_; 
v_res_990_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg(v_parentNumGoals_986_, v_a_987_, v_a_988_);
lean_dec(v_a_988_);
lean_dec(v_parentNumGoals_986_);
return v_res_990_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go(lean_object* v_00_u03c3_991_, lean_object* v_parentNumGoals_992_, lean_object* v_a_993_, lean_object* v_a_994_){
_start:
{
lean_object* v___x_996_; 
v___x_996_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg(v_parentNumGoals_992_, v_a_993_, v_a_994_);
return v___x_996_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___boxed(lean_object* v_00_u03c3_997_, lean_object* v_parentNumGoals_998_, lean_object* v_a_999_, lean_object* v_a_1000_, lean_object* v_a_1001_){
_start:
{
lean_object* v_res_1002_; 
v_res_1002_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go(v_00_u03c3_997_, v_parentNumGoals_998_, v_a_999_, v_a_1000_);
lean_dec(v_a_1000_);
lean_dec(v_parentNumGoals_998_);
return v_res_1002_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0(lean_object* v_00_u03c3_1003_, lean_object* v___x_1004_, lean_object* v_as_1005_, size_t v_i_1006_, size_t v_stop_1007_, lean_object* v_b_1008_, lean_object* v___y_1009_){
_start:
{
lean_object* v___x_1011_; 
v___x_1011_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___redArg(v___x_1004_, v_as_1005_, v_i_1006_, v_stop_1007_, v_b_1008_, v___y_1009_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0___boxed(lean_object* v_00_u03c3_1012_, lean_object* v___x_1013_, lean_object* v_as_1014_, lean_object* v_i_1015_, lean_object* v_stop_1016_, lean_object* v_b_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_){
_start:
{
size_t v_i_boxed_1020_; size_t v_stop_boxed_1021_; lean_object* v_res_1022_; 
v_i_boxed_1020_ = lean_unbox_usize(v_i_1015_);
lean_dec(v_i_1015_);
v_stop_boxed_1021_ = lean_unbox_usize(v_stop_1016_);
lean_dec(v_stop_1016_);
v_res_1022_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go_spec__0(v_00_u03c3_1012_, v___x_1013_, v_as_1014_, v_i_boxed_1020_, v_stop_boxed_1021_, v_b_1017_, v___y_1018_);
lean_dec(v___y_1018_);
lean_dec_ref(v_as_1014_);
lean_dec(v___x_1013_);
return v_res_1022_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_numSiblings___lam__0(lean_object* v_t_1023_, lean_object* v_x_1024_){
_start:
{
lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v___x_1026_ = lean_unsigned_to_nat(0u);
v___x_1027_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_toStepTree___closed__1, &lp_aesop_Aesop_Script_UScript_toStepTree___closed__1_once, _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__1);
v___x_1028_ = lean_st_mk_ref(v___x_1027_);
v___x_1029_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_StepTree_numSiblings_go___redArg(v___x_1026_, v_t_1023_, v___x_1028_);
v___x_1030_ = lean_st_ref_get(v___x_1028_);
lean_dec(v___x_1028_);
v___x_1031_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1031_, 0, v___x_1029_);
lean_ctor_set(v___x_1031_, 1, v___x_1030_);
return v___x_1031_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_numSiblings___lam__0___boxed(lean_object* v_t_1032_, lean_object* v_x_1033_, lean_object* v___y_1034_){
_start:
{
lean_object* v_res_1035_; 
v_res_1035_ = lp_aesop_Aesop_Script_StepTree_numSiblings___lam__0(v_t_1032_, v_x_1033_);
return v_res_1035_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_StepTree_numSiblings(lean_object* v_t_1036_){
_start:
{
lean_object* v___f_1037_; lean_object* v___x_1038_; lean_object* v_snd_1039_; 
v___f_1037_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_StepTree_numSiblings___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1037_, 0, v_t_1036_);
v___x_1038_ = l_runST___redArg(v___f_1037_);
v_snd_1039_ = lean_ctor_get(v___x_1038_, 1);
lean_inc(v_snd_1039_);
lean_dec(v___x_1038_);
return v_snd_1039_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__8(lean_object* v_msg_1040_){
_start:
{
lean_object* v___x_1041_; lean_object* v___x_1042_; 
v___x_1041_ = lean_unsigned_to_nat(0u);
v___x_1042_ = lean_panic_fn_borrowed(v___x_1041_, v_msg_1040_);
return v___x_1042_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg(lean_object* v_m_1043_, lean_object* v_a_1044_){
_start:
{
lean_object* v_buckets_1045_; lean_object* v___x_1046_; uint64_t v___x_1047_; uint64_t v___x_1048_; uint64_t v___x_1049_; uint64_t v_fold_1050_; uint64_t v___x_1051_; uint64_t v___x_1052_; uint64_t v___x_1053_; size_t v___x_1054_; size_t v___x_1055_; size_t v___x_1056_; size_t v___x_1057_; size_t v___x_1058_; lean_object* v___x_1059_; uint8_t v___x_1060_; 
v_buckets_1045_ = lean_ctor_get(v_m_1043_, 1);
v___x_1046_ = lean_array_get_size(v_buckets_1045_);
v___x_1047_ = l_Lean_instHashableMVarId_hash(v_a_1044_);
v___x_1048_ = 32ULL;
v___x_1049_ = lean_uint64_shift_right(v___x_1047_, v___x_1048_);
v_fold_1050_ = lean_uint64_xor(v___x_1047_, v___x_1049_);
v___x_1051_ = 16ULL;
v___x_1052_ = lean_uint64_shift_right(v_fold_1050_, v___x_1051_);
v___x_1053_ = lean_uint64_xor(v_fold_1050_, v___x_1052_);
v___x_1054_ = lean_uint64_to_usize(v___x_1053_);
v___x_1055_ = lean_usize_of_nat(v___x_1046_);
v___x_1056_ = ((size_t)1ULL);
v___x_1057_ = lean_usize_sub(v___x_1055_, v___x_1056_);
v___x_1058_ = lean_usize_land(v___x_1054_, v___x_1057_);
v___x_1059_ = lean_array_uget_borrowed(v_buckets_1045_, v___x_1058_);
v___x_1060_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg(v_a_1044_, v___x_1059_);
return v___x_1060_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg___boxed(lean_object* v_m_1061_, lean_object* v_a_1062_){
_start:
{
uint8_t v_res_1063_; lean_object* v_r_1064_; 
v_res_1063_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg(v_m_1061_, v_a_1062_);
lean_dec(v_a_1062_);
lean_dec_ref(v_m_1061_);
v_r_1064_ = lean_box(v_res_1063_);
return v_r_1064_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___redArg(lean_object* v___x_1065_, lean_object* v_snd_1066_, lean_object* v_as_1067_, size_t v_sz_1068_, size_t v_i_1069_, lean_object* v_b_1070_){
_start:
{
lean_object* v_a_1073_; uint8_t v___x_1077_; 
v___x_1077_ = lean_usize_dec_lt(v_i_1069_, v_sz_1068_);
if (v___x_1077_ == 0)
{
lean_object* v___x_1078_; 
v___x_1078_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1078_, 0, v_b_1070_);
return v___x_1078_;
}
else
{
lean_object* v_a_1079_; lean_object* v_goal_1080_; uint8_t v___x_1081_; 
v_a_1079_ = lean_array_uget_borrowed(v_as_1067_, v_i_1069_);
v_goal_1080_ = lean_ctor_get(v_a_1079_, 0);
v___x_1081_ = l_Lean_instBEqMVarId_beq(v_goal_1080_, v___x_1065_);
if (v___x_1081_ == 0)
{
lean_object* v_invisibleGoals_1082_; uint8_t v___x_1083_; 
v_invisibleGoals_1082_ = lean_ctor_get(v_snd_1066_, 1);
v___x_1083_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg(v_invisibleGoals_1082_, v_goal_1080_);
if (v___x_1083_ == 0)
{
v_a_1073_ = v_b_1070_;
goto v___jp_1072_;
}
else
{
lean_object* v___x_1084_; 
lean_inc(v_a_1079_);
v___x_1084_ = lean_array_push(v_b_1070_, v_a_1079_);
v_a_1073_ = v___x_1084_;
goto v___jp_1072_;
}
}
else
{
lean_object* v_visibleGoals_1085_; lean_object* v___x_1086_; 
v_visibleGoals_1085_ = lean_ctor_get(v_snd_1066_, 0);
v___x_1086_ = l_Array_append___redArg(v_b_1070_, v_visibleGoals_1085_);
v_a_1073_ = v___x_1086_;
goto v___jp_1072_;
}
}
v___jp_1072_:
{
size_t v___x_1074_; size_t v___x_1075_; 
v___x_1074_ = ((size_t)1ULL);
v___x_1075_ = lean_usize_add(v_i_1069_, v___x_1074_);
v_i_1069_ = v___x_1075_;
v_b_1070_ = v_a_1073_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___redArg___boxed(lean_object* v___x_1087_, lean_object* v_snd_1088_, lean_object* v_as_1089_, lean_object* v_sz_1090_, lean_object* v_i_1091_, lean_object* v_b_1092_, lean_object* v___y_1093_){
_start:
{
size_t v_sz_boxed_1094_; size_t v_i_boxed_1095_; lean_object* v_res_1096_; 
v_sz_boxed_1094_ = lean_unbox_usize(v_sz_1090_);
lean_dec(v_sz_1090_);
v_i_boxed_1095_ = lean_unbox_usize(v_i_1091_);
lean_dec(v_i_1091_);
v_res_1096_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___redArg(v___x_1087_, v_snd_1088_, v_as_1089_, v_sz_boxed_1094_, v_i_boxed_1095_, v_b_1092_);
lean_dec_ref(v_as_1089_);
lean_dec_ref(v_snd_1088_);
lean_dec(v___x_1087_);
return v_res_1096_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__2___redArg(lean_object* v_m_1097_, lean_object* v_a_1098_, lean_object* v_b_1099_){
_start:
{
lean_object* v_size_1100_; lean_object* v_buckets_1101_; lean_object* v___x_1102_; uint64_t v___x_1103_; uint64_t v___x_1104_; uint64_t v___x_1105_; uint64_t v_fold_1106_; uint64_t v___x_1107_; uint64_t v___x_1108_; uint64_t v___x_1109_; size_t v___x_1110_; size_t v___x_1111_; size_t v___x_1112_; size_t v___x_1113_; size_t v___x_1114_; lean_object* v_bkt_1115_; uint8_t v___x_1116_; 
v_size_1100_ = lean_ctor_get(v_m_1097_, 0);
v_buckets_1101_ = lean_ctor_get(v_m_1097_, 1);
v___x_1102_ = lean_array_get_size(v_buckets_1101_);
v___x_1103_ = l_Lean_instHashableMVarId_hash(v_a_1098_);
v___x_1104_ = 32ULL;
v___x_1105_ = lean_uint64_shift_right(v___x_1103_, v___x_1104_);
v_fold_1106_ = lean_uint64_xor(v___x_1103_, v___x_1105_);
v___x_1107_ = 16ULL;
v___x_1108_ = lean_uint64_shift_right(v_fold_1106_, v___x_1107_);
v___x_1109_ = lean_uint64_xor(v_fold_1106_, v___x_1108_);
v___x_1110_ = lean_uint64_to_usize(v___x_1109_);
v___x_1111_ = lean_usize_of_nat(v___x_1102_);
v___x_1112_ = ((size_t)1ULL);
v___x_1113_ = lean_usize_sub(v___x_1111_, v___x_1112_);
v___x_1114_ = lean_usize_land(v___x_1110_, v___x_1113_);
v_bkt_1115_ = lean_array_uget_borrowed(v_buckets_1101_, v___x_1114_);
v___x_1116_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__0___redArg(v_a_1098_, v_bkt_1115_);
if (v___x_1116_ == 0)
{
lean_object* v___x_1118_; uint8_t v_isShared_1119_; uint8_t v_isSharedCheck_1137_; 
lean_inc_ref(v_buckets_1101_);
lean_inc(v_size_1100_);
v_isSharedCheck_1137_ = !lean_is_exclusive(v_m_1097_);
if (v_isSharedCheck_1137_ == 0)
{
lean_object* v_unused_1138_; lean_object* v_unused_1139_; 
v_unused_1138_ = lean_ctor_get(v_m_1097_, 1);
lean_dec(v_unused_1138_);
v_unused_1139_ = lean_ctor_get(v_m_1097_, 0);
lean_dec(v_unused_1139_);
v___x_1118_ = v_m_1097_;
v_isShared_1119_ = v_isSharedCheck_1137_;
goto v_resetjp_1117_;
}
else
{
lean_dec(v_m_1097_);
v___x_1118_ = lean_box(0);
v_isShared_1119_ = v_isSharedCheck_1137_;
goto v_resetjp_1117_;
}
v_resetjp_1117_:
{
lean_object* v___x_1120_; lean_object* v_size_x27_1121_; lean_object* v___x_1122_; lean_object* v_buckets_x27_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; uint8_t v___x_1129_; 
v___x_1120_ = lean_unsigned_to_nat(1u);
v_size_x27_1121_ = lean_nat_add(v_size_1100_, v___x_1120_);
lean_dec(v_size_1100_);
lean_inc(v_bkt_1115_);
v___x_1122_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1122_, 0, v_a_1098_);
lean_ctor_set(v___x_1122_, 1, v_b_1099_);
lean_ctor_set(v___x_1122_, 2, v_bkt_1115_);
v_buckets_x27_1123_ = lean_array_uset(v_buckets_1101_, v___x_1114_, v___x_1122_);
v___x_1124_ = lean_unsigned_to_nat(4u);
v___x_1125_ = lean_nat_mul(v_size_x27_1121_, v___x_1124_);
v___x_1126_ = lean_unsigned_to_nat(3u);
v___x_1127_ = lean_nat_div(v___x_1125_, v___x_1126_);
lean_dec(v___x_1125_);
v___x_1128_ = lean_array_get_size(v_buckets_x27_1123_);
v___x_1129_ = lean_nat_dec_le(v___x_1127_, v___x_1128_);
lean_dec(v___x_1127_);
if (v___x_1129_ == 0)
{
lean_object* v_val_1130_; lean_object* v___x_1132_; 
v_val_1130_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Script_UScript_toStepTree_spec__0_spec__1___redArg(v_buckets_x27_1123_);
if (v_isShared_1119_ == 0)
{
lean_ctor_set(v___x_1118_, 1, v_val_1130_);
lean_ctor_set(v___x_1118_, 0, v_size_x27_1121_);
v___x_1132_ = v___x_1118_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1133_; 
v_reuseFailAlloc_1133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1133_, 0, v_size_x27_1121_);
lean_ctor_set(v_reuseFailAlloc_1133_, 1, v_val_1130_);
v___x_1132_ = v_reuseFailAlloc_1133_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
return v___x_1132_;
}
}
else
{
lean_object* v___x_1135_; 
if (v_isShared_1119_ == 0)
{
lean_ctor_set(v___x_1118_, 1, v_buckets_x27_1123_);
lean_ctor_set(v___x_1118_, 0, v_size_x27_1121_);
v___x_1135_ = v___x_1118_;
goto v_reusejp_1134_;
}
else
{
lean_object* v_reuseFailAlloc_1136_; 
v_reuseFailAlloc_1136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1136_, 0, v_size_x27_1121_);
lean_ctor_set(v_reuseFailAlloc_1136_, 1, v_buckets_x27_1123_);
v___x_1135_ = v_reuseFailAlloc_1136_;
goto v_reusejp_1134_;
}
v_reusejp_1134_:
{
return v___x_1135_;
}
}
}
}
else
{
lean_dec(v_b_1099_);
lean_dec(v_a_1098_);
return v_m_1097_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___redArg(lean_object* v_snd_1140_, lean_object* v_a_1141_, lean_object* v_a_1142_){
_start:
{
if (lean_obj_tag(v_a_1141_) == 0)
{
lean_object* v___x_1144_; lean_object* v___x_1145_; 
v___x_1144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1144_, 0, v_a_1142_);
v___x_1145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1145_, 0, v___x_1144_);
return v___x_1145_;
}
else
{
lean_object* v_key_1146_; lean_object* v_tail_1147_; lean_object* v_invisibleGoals_1148_; uint8_t v___x_1149_; 
v_key_1146_ = lean_ctor_get(v_a_1141_, 0);
lean_inc(v_key_1146_);
v_tail_1147_ = lean_ctor_get(v_a_1141_, 2);
lean_inc(v_tail_1147_);
lean_dec_ref_known(v_a_1141_, 3);
v_invisibleGoals_1148_ = lean_ctor_get(v_snd_1140_, 1);
v___x_1149_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg(v_invisibleGoals_1148_, v_key_1146_);
if (v___x_1149_ == 0)
{
lean_dec(v_key_1146_);
v_a_1141_ = v_tail_1147_;
goto _start;
}
else
{
lean_object* v___x_1151_; lean_object* v___x_1152_; 
v___x_1151_ = lean_box(0);
v___x_1152_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__2___redArg(v_a_1142_, v_key_1146_, v___x_1151_);
v_a_1141_ = v_tail_1147_;
v_a_1142_ = v___x_1152_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___redArg___boxed(lean_object* v_snd_1154_, lean_object* v_a_1155_, lean_object* v_a_1156_, lean_object* v___y_1157_){
_start:
{
lean_object* v_res_1158_; 
v_res_1158_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___redArg(v_snd_1154_, v_a_1155_, v_a_1156_);
lean_dec_ref(v_snd_1154_);
return v_res_1158_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__5(lean_object* v_snd_1159_, lean_object* v_as_1160_, size_t v_sz_1161_, size_t v_i_1162_, lean_object* v_b_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_){
_start:
{
uint8_t v___x_1167_; 
v___x_1167_ = lean_usize_dec_lt(v_i_1162_, v_sz_1161_);
if (v___x_1167_ == 0)
{
lean_object* v___x_1168_; 
v___x_1168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1168_, 0, v_b_1163_);
return v___x_1168_;
}
else
{
lean_object* v_a_1169_; lean_object* v___x_1170_; 
v_a_1169_ = lean_array_uget_borrowed(v_as_1160_, v_i_1162_);
lean_inc(v_a_1169_);
v___x_1170_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___redArg(v_snd_1159_, v_a_1169_, v_b_1163_);
if (lean_obj_tag(v___x_1170_) == 0)
{
lean_object* v_a_1171_; lean_object* v___x_1173_; uint8_t v_isShared_1174_; uint8_t v_isSharedCheck_1183_; 
v_a_1171_ = lean_ctor_get(v___x_1170_, 0);
v_isSharedCheck_1183_ = !lean_is_exclusive(v___x_1170_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1173_ = v___x_1170_;
v_isShared_1174_ = v_isSharedCheck_1183_;
goto v_resetjp_1172_;
}
else
{
lean_inc(v_a_1171_);
lean_dec(v___x_1170_);
v___x_1173_ = lean_box(0);
v_isShared_1174_ = v_isSharedCheck_1183_;
goto v_resetjp_1172_;
}
v_resetjp_1172_:
{
if (lean_obj_tag(v_a_1171_) == 0)
{
lean_object* v_a_1175_; lean_object* v___x_1177_; 
v_a_1175_ = lean_ctor_get(v_a_1171_, 0);
lean_inc(v_a_1175_);
lean_dec_ref_known(v_a_1171_, 1);
if (v_isShared_1174_ == 0)
{
lean_ctor_set(v___x_1173_, 0, v_a_1175_);
v___x_1177_ = v___x_1173_;
goto v_reusejp_1176_;
}
else
{
lean_object* v_reuseFailAlloc_1178_; 
v_reuseFailAlloc_1178_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1178_, 0, v_a_1175_);
v___x_1177_ = v_reuseFailAlloc_1178_;
goto v_reusejp_1176_;
}
v_reusejp_1176_:
{
return v___x_1177_;
}
}
else
{
lean_object* v_a_1179_; size_t v___x_1180_; size_t v___x_1181_; 
lean_del_object(v___x_1173_);
v_a_1179_ = lean_ctor_get(v_a_1171_, 0);
lean_inc(v_a_1179_);
lean_dec_ref_known(v_a_1171_, 1);
v___x_1180_ = ((size_t)1ULL);
v___x_1181_ = lean_usize_add(v_i_1162_, v___x_1180_);
v_i_1162_ = v___x_1181_;
v_b_1163_ = v_a_1179_;
goto _start;
}
}
}
else
{
lean_object* v_a_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1191_; 
v_a_1184_ = lean_ctor_get(v___x_1170_, 0);
v_isSharedCheck_1191_ = !lean_is_exclusive(v___x_1170_);
if (v_isSharedCheck_1191_ == 0)
{
v___x_1186_ = v___x_1170_;
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_a_1184_);
lean_dec(v___x_1170_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1189_; 
if (v_isShared_1187_ == 0)
{
v___x_1189_ = v___x_1186_;
goto v_reusejp_1188_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_a_1184_);
v___x_1189_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1188_;
}
v_reusejp_1188_:
{
return v___x_1189_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__5___boxed(lean_object* v_snd_1192_, lean_object* v_as_1193_, lean_object* v_sz_1194_, lean_object* v_i_1195_, lean_object* v_b_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_){
_start:
{
size_t v_sz_boxed_1200_; size_t v_i_boxed_1201_; lean_object* v_res_1202_; 
v_sz_boxed_1200_ = lean_unbox_usize(v_sz_1194_);
lean_dec(v_sz_1194_);
v_i_boxed_1201_ = lean_unbox_usize(v_i_1195_);
lean_dec(v_i_1195_);
v_res_1202_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__5(v_snd_1192_, v_as_1193_, v_sz_boxed_1200_, v_i_boxed_1201_, v_b_1196_, v___y_1197_, v___y_1198_);
lean_dec(v___y_1198_);
lean_dec_ref(v___y_1197_);
lean_dec_ref(v_as_1193_);
lean_dec_ref(v_snd_1192_);
return v_res_1202_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___redArg(lean_object* v_goal_1203_, lean_object* v_as_1204_, size_t v_sz_1205_, size_t v_i_1206_, lean_object* v_b_1207_){
_start:
{
lean_object* v_a_1210_; uint8_t v___x_1214_; 
v___x_1214_ = lean_usize_dec_lt(v_i_1206_, v_sz_1205_);
if (v___x_1214_ == 0)
{
lean_object* v___x_1215_; 
v___x_1215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1215_, 0, v_b_1207_);
return v___x_1215_;
}
else
{
lean_object* v_a_1216_; lean_object* v_goal_1217_; uint8_t v___x_1218_; 
v_a_1216_ = lean_array_uget_borrowed(v_as_1204_, v_i_1206_);
v_goal_1217_ = lean_ctor_get(v_a_1216_, 0);
v___x_1218_ = l_Lean_instBEqMVarId_beq(v_goal_1217_, v_goal_1203_);
if (v___x_1218_ == 0)
{
lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1219_ = lean_box(0);
lean_inc(v_goal_1217_);
v___x_1220_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__2___redArg(v_b_1207_, v_goal_1217_, v___x_1219_);
v_a_1210_ = v___x_1220_;
goto v___jp_1209_;
}
else
{
v_a_1210_ = v_b_1207_;
goto v___jp_1209_;
}
}
v___jp_1209_:
{
size_t v___x_1211_; size_t v___x_1212_; 
v___x_1211_ = ((size_t)1ULL);
v___x_1212_ = lean_usize_add(v_i_1206_, v___x_1211_);
v_i_1206_ = v___x_1212_;
v_b_1207_ = v_a_1210_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___redArg___boxed(lean_object* v_goal_1221_, lean_object* v_as_1222_, lean_object* v_sz_1223_, lean_object* v_i_1224_, lean_object* v_b_1225_, lean_object* v___y_1226_){
_start:
{
size_t v_sz_boxed_1227_; size_t v_i_boxed_1228_; lean_object* v_res_1229_; 
v_sz_boxed_1227_ = lean_unbox_usize(v_sz_1223_);
lean_dec(v_sz_1223_);
v_i_boxed_1228_ = lean_unbox_usize(v_i_1224_);
lean_dec(v_i_1224_);
v_res_1229_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___redArg(v_goal_1221_, v_as_1222_, v_sz_boxed_1227_, v_i_boxed_1228_, v_b_1225_);
lean_dec_ref(v_as_1222_);
lean_dec(v_goal_1221_);
return v_res_1229_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8(lean_object* v_goal_1233_, lean_object* v_as_1234_, size_t v_sz_1235_, size_t v_i_1236_, lean_object* v_b_1237_){
_start:
{
uint8_t v___x_1238_; 
v___x_1238_ = lean_usize_dec_lt(v_i_1236_, v_sz_1235_);
if (v___x_1238_ == 0)
{
lean_inc_ref(v_b_1237_);
return v_b_1237_;
}
else
{
lean_object* v_a_1239_; lean_object* v_goal_1240_; lean_object* v___x_1241_; uint8_t v___x_1242_; 
v_a_1239_ = lean_array_uget_borrowed(v_as_1234_, v_i_1236_);
v_goal_1240_ = lean_ctor_get(v_a_1239_, 0);
v___x_1241_ = lean_box(0);
v___x_1242_ = l_Lean_instBEqMVarId_beq(v_goal_1240_, v_goal_1233_);
if (v___x_1242_ == 0)
{
lean_object* v___x_1243_; size_t v___x_1244_; size_t v___x_1245_; 
v___x_1243_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8___closed__0));
v___x_1244_ = ((size_t)1ULL);
v___x_1245_ = lean_usize_add(v_i_1236_, v___x_1244_);
v_i_1236_ = v___x_1245_;
v_b_1237_ = v___x_1243_;
goto _start;
}
else
{
lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; 
lean_inc(v_a_1239_);
v___x_1247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1247_, 0, v_a_1239_);
v___x_1248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1248_, 0, v___x_1247_);
v___x_1249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1248_);
lean_ctor_set(v___x_1249_, 1, v___x_1241_);
return v___x_1249_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8___boxed(lean_object* v_goal_1250_, lean_object* v_as_1251_, lean_object* v_sz_1252_, lean_object* v_i_1253_, lean_object* v_b_1254_){
_start:
{
size_t v_sz_boxed_1255_; size_t v_i_boxed_1256_; lean_object* v_res_1257_; 
v_sz_boxed_1255_ = lean_unbox_usize(v_sz_1252_);
lean_dec(v_sz_1252_);
v_i_boxed_1256_ = lean_unbox_usize(v_i_1253_);
lean_dec(v_i_1253_);
v_res_1257_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8(v_goal_1250_, v_as_1251_, v_sz_boxed_1255_, v_i_boxed_1256_, v_b_1254_);
lean_dec_ref(v_b_1254_);
lean_dec_ref(v_as_1251_);
lean_dec(v_goal_1250_);
return v_res_1257_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__0(void){
_start:
{
lean_object* v___x_1258_; 
v___x_1258_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1258_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1(void){
_start:
{
lean_object* v___x_1259_; lean_object* v___x_1260_; 
v___x_1259_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__0, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__0_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__0);
v___x_1260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1260_, 0, v___x_1259_);
return v___x_1260_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__2(void){
_start:
{
lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; 
v___x_1261_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1);
v___x_1262_ = lean_unsigned_to_nat(0u);
v___x_1263_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1262_);
lean_ctor_set(v___x_1263_, 1, v___x_1262_);
lean_ctor_set(v___x_1263_, 2, v___x_1262_);
lean_ctor_set(v___x_1263_, 3, v___x_1262_);
lean_ctor_set(v___x_1263_, 4, v___x_1261_);
lean_ctor_set(v___x_1263_, 5, v___x_1261_);
lean_ctor_set(v___x_1263_, 6, v___x_1261_);
lean_ctor_set(v___x_1263_, 7, v___x_1261_);
lean_ctor_set(v___x_1263_, 8, v___x_1261_);
lean_ctor_set(v___x_1263_, 9, v___x_1261_);
return v___x_1263_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__3(void){
_start:
{
lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; 
v___x_1264_ = lean_unsigned_to_nat(32u);
v___x_1265_ = lean_mk_empty_array_with_capacity(v___x_1264_);
v___x_1266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1266_, 0, v___x_1265_);
return v___x_1266_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__4(void){
_start:
{
size_t v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; 
v___x_1267_ = ((size_t)5ULL);
v___x_1268_ = lean_unsigned_to_nat(0u);
v___x_1269_ = lean_unsigned_to_nat(32u);
v___x_1270_ = lean_mk_empty_array_with_capacity(v___x_1269_);
v___x_1271_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__3, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__3_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__3);
v___x_1272_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1272_, 0, v___x_1271_);
lean_ctor_set(v___x_1272_, 1, v___x_1270_);
lean_ctor_set(v___x_1272_, 2, v___x_1268_);
lean_ctor_set(v___x_1272_, 3, v___x_1268_);
lean_ctor_set_usize(v___x_1272_, 4, v___x_1267_);
return v___x_1272_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__5(void){
_start:
{
lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; 
v___x_1273_ = lean_box(1);
v___x_1274_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__4);
v___x_1275_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__1);
v___x_1276_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1276_, 0, v___x_1275_);
lean_ctor_set(v___x_1276_, 1, v___x_1274_);
lean_ctor_set(v___x_1276_, 2, v___x_1273_);
return v___x_1276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14(lean_object* v_msgData_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_){
_start:
{
lean_object* v___x_1281_; lean_object* v_env_1282_; lean_object* v_options_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___x_1281_ = lean_st_ref_get(v___y_1279_);
v_env_1282_ = lean_ctor_get(v___x_1281_, 0);
lean_inc_ref(v_env_1282_);
lean_dec(v___x_1281_);
v_options_1283_ = lean_ctor_get(v___y_1278_, 2);
v___x_1284_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__2, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__2_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__2);
v___x_1285_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__5, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__5_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___closed__5);
lean_inc_ref(v_options_1283_);
v___x_1286_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1286_, 0, v_env_1282_);
lean_ctor_set(v___x_1286_, 1, v___x_1284_);
lean_ctor_set(v___x_1286_, 2, v___x_1285_);
lean_ctor_set(v___x_1286_, 3, v_options_1283_);
v___x_1287_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1286_);
lean_ctor_set(v___x_1287_, 1, v_msgData_1277_);
v___x_1288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1287_);
return v___x_1288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14___boxed(lean_object* v_msgData_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_){
_start:
{
lean_object* v_res_1293_; 
v_res_1293_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14(v_msgData_1289_, v___y_1290_, v___y_1291_);
lean_dec(v___y_1291_);
lean_dec_ref(v___y_1290_);
return v_res_1293_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg(lean_object* v_msg_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_){
_start:
{
lean_object* v_ref_1298_; lean_object* v___x_1299_; lean_object* v_a_1300_; lean_object* v___x_1302_; uint8_t v_isShared_1303_; uint8_t v_isSharedCheck_1308_; 
v_ref_1298_ = lean_ctor_get(v___y_1295_, 5);
v___x_1299_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14(v_msg_1294_, v___y_1295_, v___y_1296_);
v_a_1300_ = lean_ctor_get(v___x_1299_, 0);
v_isSharedCheck_1308_ = !lean_is_exclusive(v___x_1299_);
if (v_isSharedCheck_1308_ == 0)
{
v___x_1302_ = v___x_1299_;
v_isShared_1303_ = v_isSharedCheck_1308_;
goto v_resetjp_1301_;
}
else
{
lean_inc(v_a_1300_);
lean_dec(v___x_1299_);
v___x_1302_ = lean_box(0);
v_isShared_1303_ = v_isSharedCheck_1308_;
goto v_resetjp_1301_;
}
v_resetjp_1301_:
{
lean_object* v___x_1304_; lean_object* v___x_1306_; 
lean_inc(v_ref_1298_);
v___x_1304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1304_, 0, v_ref_1298_);
lean_ctor_set(v___x_1304_, 1, v_a_1300_);
if (v_isShared_1303_ == 0)
{
lean_ctor_set_tag(v___x_1302_, 1);
lean_ctor_set(v___x_1302_, 0, v___x_1304_);
v___x_1306_ = v___x_1302_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1307_; 
v_reuseFailAlloc_1307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1307_, 0, v___x_1304_);
v___x_1306_ = v_reuseFailAlloc_1307_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
return v___x_1306_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg___boxed(lean_object* v_msg_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_){
_start:
{
lean_object* v_res_1313_; 
v_res_1313_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg(v_msg_1309_, v___y_1310_, v___y_1311_);
lean_dec(v___y_1311_);
lean_dec_ref(v___y_1310_);
return v_res_1313_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__1(void){
_start:
{
lean_object* v___x_1315_; lean_object* v___x_1316_; 
v___x_1315_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__0));
v___x_1316_ = l_Lean_stringToMessageData(v___x_1315_);
return v___x_1316_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__3(void){
_start:
{
lean_object* v___x_1318_; lean_object* v___x_1319_; 
v___x_1318_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__2));
v___x_1319_ = l_Lean_stringToMessageData(v___x_1318_);
return v___x_1319_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__5(void){
_start:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; 
v___x_1321_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__4));
v___x_1322_ = l_Lean_stringToMessageData(v___x_1321_);
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg(lean_object* v_goal_1323_, lean_object* v_pre_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_){
_start:
{
lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; 
v___x_1328_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__1, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__1);
v___x_1329_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1329_, 0, v___x_1328_);
lean_ctor_set(v___x_1329_, 1, v_pre_1324_);
v___x_1330_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__3, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__3_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__3);
v___x_1331_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1331_, 0, v___x_1329_);
lean_ctor_set(v___x_1331_, 1, v___x_1330_);
v___x_1332_ = l_Lean_MessageData_ofName(v_goal_1323_);
v___x_1333_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1333_, 0, v___x_1331_);
lean_ctor_set(v___x_1333_, 1, v___x_1332_);
v___x_1334_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__5, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___closed__5);
v___x_1335_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1335_, 0, v___x_1333_);
lean_ctor_set(v___x_1335_, 1, v___x_1334_);
v___x_1336_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg(v___x_1335_, v___y_1325_, v___y_1326_);
return v___x_1336_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg___boxed(lean_object* v_goal_1337_, lean_object* v_pre_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
lean_object* v_res_1342_; 
v_res_1342_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg(v_goal_1337_, v_pre_1338_, v___y_1339_, v___y_1340_);
lean_dec(v___y_1340_);
lean_dec_ref(v___y_1339_);
return v_res_1342_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__2(void){
_start:
{
lean_object* v___x_1346_; lean_object* v___x_1347_; 
v___x_1346_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__1));
v___x_1347_ = l_Lean_MessageData_ofFormat(v___x_1346_);
return v___x_1347_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6(lean_object* v_ts_1348_, lean_object* v_goal_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_){
_start:
{
lean_object* v_visibleGoals_1356_; lean_object* v_invisibleGoals_1357_; lean_object* v___x_1359_; uint8_t v_isShared_1360_; uint8_t v_isSharedCheck_1391_; 
v_visibleGoals_1356_ = lean_ctor_get(v_ts_1348_, 0);
v_invisibleGoals_1357_ = lean_ctor_get(v_ts_1348_, 1);
v_isSharedCheck_1391_ = !lean_is_exclusive(v_ts_1348_);
if (v_isSharedCheck_1391_ == 0)
{
v___x_1359_ = v_ts_1348_;
v_isShared_1360_ = v_isSharedCheck_1391_;
goto v_resetjp_1358_;
}
else
{
lean_inc(v_invisibleGoals_1357_);
lean_inc(v_visibleGoals_1356_);
lean_dec(v_ts_1348_);
v___x_1359_ = lean_box(0);
v_isShared_1360_ = v_isSharedCheck_1391_;
goto v_resetjp_1358_;
}
v___jp_1353_:
{
lean_object* v___x_1354_; lean_object* v___x_1355_; 
v___x_1354_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__2, &lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__2_once, _init_lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___closed__2);
v___x_1355_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg(v_goal_1349_, v___x_1354_, v___y_1350_, v___y_1351_);
return v___x_1355_;
}
v_resetjp_1358_:
{
lean_object* v___x_1361_; size_t v_sz_1362_; size_t v___x_1363_; lean_object* v___x_1364_; lean_object* v_fst_1365_; 
v___x_1361_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8___closed__0));
v_sz_1362_ = lean_array_size(v_visibleGoals_1356_);
v___x_1363_ = ((size_t)0ULL);
v___x_1364_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__8(v_goal_1349_, v_visibleGoals_1356_, v_sz_1362_, v___x_1363_, v___x_1361_);
v_fst_1365_ = lean_ctor_get(v___x_1364_, 0);
lean_inc(v_fst_1365_);
lean_dec_ref(v___x_1364_);
if (lean_obj_tag(v_fst_1365_) == 0)
{
lean_del_object(v___x_1359_);
lean_dec_ref(v_invisibleGoals_1357_);
lean_dec_ref(v_visibleGoals_1356_);
goto v___jp_1353_;
}
else
{
lean_object* v_val_1366_; 
v_val_1366_ = lean_ctor_get(v_fst_1365_, 0);
lean_inc(v_val_1366_);
lean_dec_ref_known(v_fst_1365_, 1);
if (lean_obj_tag(v_val_1366_) == 1)
{
lean_object* v_val_1367_; lean_object* v___x_1368_; 
v_val_1367_ = lean_ctor_get(v_val_1366_, 0);
lean_inc(v_val_1367_);
lean_dec_ref_known(v_val_1366_, 1);
v___x_1368_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___redArg(v_goal_1349_, v_visibleGoals_1356_, v_sz_1362_, v___x_1363_, v_invisibleGoals_1357_);
lean_dec_ref(v_visibleGoals_1356_);
lean_dec(v_goal_1349_);
if (lean_obj_tag(v___x_1368_) == 0)
{
lean_object* v_a_1369_; lean_object* v___x_1371_; uint8_t v_isShared_1372_; uint8_t v_isSharedCheck_1382_; 
v_a_1369_ = lean_ctor_get(v___x_1368_, 0);
v_isSharedCheck_1382_ = !lean_is_exclusive(v___x_1368_);
if (v_isSharedCheck_1382_ == 0)
{
v___x_1371_ = v___x_1368_;
v_isShared_1372_ = v_isSharedCheck_1382_;
goto v_resetjp_1370_;
}
else
{
lean_inc(v_a_1369_);
lean_dec(v___x_1368_);
v___x_1371_ = lean_box(0);
v_isShared_1372_ = v_isSharedCheck_1382_;
goto v_resetjp_1370_;
}
v_resetjp_1370_:
{
lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1377_; 
v___x_1373_ = lean_unsigned_to_nat(1u);
v___x_1374_ = lean_mk_empty_array_with_capacity(v___x_1373_);
v___x_1375_ = lean_array_push(v___x_1374_, v_val_1367_);
if (v_isShared_1360_ == 0)
{
lean_ctor_set(v___x_1359_, 1, v_a_1369_);
lean_ctor_set(v___x_1359_, 0, v___x_1375_);
v___x_1377_ = v___x_1359_;
goto v_reusejp_1376_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v___x_1375_);
lean_ctor_set(v_reuseFailAlloc_1381_, 1, v_a_1369_);
v___x_1377_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1376_;
}
v_reusejp_1376_:
{
lean_object* v___x_1379_; 
if (v_isShared_1372_ == 0)
{
lean_ctor_set(v___x_1371_, 0, v___x_1377_);
v___x_1379_ = v___x_1371_;
goto v_reusejp_1378_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v___x_1377_);
v___x_1379_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1378_;
}
v_reusejp_1378_:
{
return v___x_1379_;
}
}
}
}
else
{
lean_object* v_a_1383_; lean_object* v___x_1385_; uint8_t v_isShared_1386_; uint8_t v_isSharedCheck_1390_; 
lean_dec(v_val_1367_);
lean_del_object(v___x_1359_);
v_a_1383_ = lean_ctor_get(v___x_1368_, 0);
v_isSharedCheck_1390_ = !lean_is_exclusive(v___x_1368_);
if (v_isSharedCheck_1390_ == 0)
{
v___x_1385_ = v___x_1368_;
v_isShared_1386_ = v_isSharedCheck_1390_;
goto v_resetjp_1384_;
}
else
{
lean_inc(v_a_1383_);
lean_dec(v___x_1368_);
v___x_1385_ = lean_box(0);
v_isShared_1386_ = v_isSharedCheck_1390_;
goto v_resetjp_1384_;
}
v_resetjp_1384_:
{
lean_object* v___x_1388_; 
if (v_isShared_1386_ == 0)
{
v___x_1388_ = v___x_1385_;
goto v_reusejp_1387_;
}
else
{
lean_object* v_reuseFailAlloc_1389_; 
v_reuseFailAlloc_1389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1389_, 0, v_a_1383_);
v___x_1388_ = v_reuseFailAlloc_1389_;
goto v_reusejp_1387_;
}
v_reusejp_1387_:
{
return v___x_1388_;
}
}
}
}
else
{
lean_dec(v_val_1366_);
lean_del_object(v___x_1359_);
lean_dec_ref(v_invisibleGoals_1357_);
lean_dec_ref(v_visibleGoals_1356_);
goto v___jp_1353_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6___boxed(lean_object* v_ts_1392_, lean_object* v_goal_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_){
_start:
{
lean_object* v_res_1397_; 
v_res_1397_ = lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6(v_ts_1392_, v_goal_1393_, v___y_1394_, v___y_1395_);
lean_dec(v___y_1395_);
lean_dec_ref(v___y_1394_);
return v_res_1397_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13_spec__17(lean_object* v_x_1398_, lean_object* v_r_1399_, lean_object* v_as_1400_, size_t v_sz_1401_, size_t v_i_1402_, lean_object* v_b_1403_){
_start:
{
lean_object* v_a_1405_; uint8_t v___x_1409_; 
v___x_1409_ = lean_usize_dec_lt(v_i_1402_, v_sz_1401_);
if (v___x_1409_ == 0)
{
return v_b_1403_;
}
else
{
lean_object* v_fst_1410_; lean_object* v_snd_1411_; lean_object* v___x_1413_; uint8_t v_isShared_1414_; uint8_t v_isSharedCheck_1428_; 
v_fst_1410_ = lean_ctor_get(v_b_1403_, 0);
v_snd_1411_ = lean_ctor_get(v_b_1403_, 1);
v_isSharedCheck_1428_ = !lean_is_exclusive(v_b_1403_);
if (v_isSharedCheck_1428_ == 0)
{
v___x_1413_ = v_b_1403_;
v_isShared_1414_ = v_isSharedCheck_1428_;
goto v_resetjp_1412_;
}
else
{
lean_inc(v_snd_1411_);
lean_inc(v_fst_1410_);
lean_dec(v_b_1403_);
v___x_1413_ = lean_box(0);
v_isShared_1414_ = v_isSharedCheck_1428_;
goto v_resetjp_1412_;
}
v_resetjp_1412_:
{
lean_object* v_a_1415_; lean_object* v_goal_1416_; lean_object* v_goal_1417_; uint8_t v___x_1418_; 
v_a_1415_ = lean_array_uget_borrowed(v_as_1400_, v_i_1402_);
v_goal_1416_ = lean_ctor_get(v_a_1415_, 0);
v_goal_1417_ = lean_ctor_get(v_x_1398_, 0);
v___x_1418_ = l_Lean_instBEqMVarId_beq(v_goal_1416_, v_goal_1417_);
if (v___x_1418_ == 0)
{
lean_object* v___x_1419_; lean_object* v___x_1421_; 
lean_inc(v_a_1415_);
v___x_1419_ = lean_array_push(v_snd_1411_, v_a_1415_);
if (v_isShared_1414_ == 0)
{
lean_ctor_set(v___x_1413_, 1, v___x_1419_);
v___x_1421_ = v___x_1413_;
goto v_reusejp_1420_;
}
else
{
lean_object* v_reuseFailAlloc_1422_; 
v_reuseFailAlloc_1422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1422_, 0, v_fst_1410_);
lean_ctor_set(v_reuseFailAlloc_1422_, 1, v___x_1419_);
v___x_1421_ = v_reuseFailAlloc_1422_;
goto v_reusejp_1420_;
}
v_reusejp_1420_:
{
v_a_1405_ = v___x_1421_;
goto v___jp_1404_;
}
}
else
{
lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1426_; 
lean_dec(v_fst_1410_);
v___x_1423_ = l_Array_append___redArg(v_snd_1411_, v_r_1399_);
v___x_1424_ = lean_box(v___x_1418_);
if (v_isShared_1414_ == 0)
{
lean_ctor_set(v___x_1413_, 1, v___x_1423_);
lean_ctor_set(v___x_1413_, 0, v___x_1424_);
v___x_1426_ = v___x_1413_;
goto v_reusejp_1425_;
}
else
{
lean_object* v_reuseFailAlloc_1427_; 
v_reuseFailAlloc_1427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1427_, 0, v___x_1424_);
lean_ctor_set(v_reuseFailAlloc_1427_, 1, v___x_1423_);
v___x_1426_ = v_reuseFailAlloc_1427_;
goto v_reusejp_1425_;
}
v_reusejp_1425_:
{
v_a_1405_ = v___x_1426_;
goto v___jp_1404_;
}
}
}
}
v___jp_1404_:
{
size_t v___x_1406_; size_t v___x_1407_; 
v___x_1406_ = ((size_t)1ULL);
v___x_1407_ = lean_usize_add(v_i_1402_, v___x_1406_);
v_i_1402_ = v___x_1407_;
v_b_1403_ = v_a_1405_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13_spec__17___boxed(lean_object* v_x_1429_, lean_object* v_r_1430_, lean_object* v_as_1431_, lean_object* v_sz_1432_, lean_object* v_i_1433_, lean_object* v_b_1434_){
_start:
{
size_t v_sz_boxed_1435_; size_t v_i_boxed_1436_; lean_object* v_res_1437_; 
v_sz_boxed_1435_ = lean_unbox_usize(v_sz_1432_);
lean_dec(v_sz_1432_);
v_i_boxed_1436_ = lean_unbox_usize(v_i_1433_);
lean_dec(v_i_1433_);
v_res_1437_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13_spec__17(v_x_1429_, v_r_1430_, v_as_1431_, v_sz_boxed_1435_, v_i_boxed_1436_, v_b_1434_);
lean_dec_ref(v_as_1431_);
lean_dec_ref(v_r_1430_);
lean_dec_ref(v_x_1429_);
return v_res_1437_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13(lean_object* v_xs_1438_, lean_object* v_x_1439_, lean_object* v_r_1440_){
_start:
{
uint8_t v_found_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v_ys_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; size_t v_sz_1450_; size_t v___x_1451_; lean_object* v___x_1452_; lean_object* v_fst_1453_; uint8_t v___x_1454_; 
v_found_1441_ = 0;
v___x_1442_ = lean_array_get_size(v_xs_1438_);
v___x_1443_ = lean_unsigned_to_nat(1u);
v___x_1444_ = lean_nat_sub(v___x_1442_, v___x_1443_);
v___x_1445_ = lean_array_get_size(v_r_1440_);
v___x_1446_ = lean_nat_add(v___x_1444_, v___x_1445_);
lean_dec(v___x_1444_);
v_ys_1447_ = lean_mk_empty_array_with_capacity(v___x_1446_);
lean_dec(v___x_1446_);
v___x_1448_ = lean_box(v_found_1441_);
v___x_1449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1449_, 0, v___x_1448_);
lean_ctor_set(v___x_1449_, 1, v_ys_1447_);
v_sz_1450_ = lean_array_size(v_xs_1438_);
v___x_1451_ = ((size_t)0ULL);
v___x_1452_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13_spec__17(v_x_1439_, v_r_1440_, v_xs_1438_, v_sz_1450_, v___x_1451_, v___x_1449_);
v_fst_1453_ = lean_ctor_get(v___x_1452_, 0);
lean_inc(v_fst_1453_);
v___x_1454_ = lean_unbox(v_fst_1453_);
lean_dec(v_fst_1453_);
if (v___x_1454_ == 0)
{
lean_object* v___x_1455_; 
lean_dec_ref(v___x_1452_);
v___x_1455_ = lean_box(0);
return v___x_1455_;
}
else
{
lean_object* v_snd_1456_; lean_object* v___x_1457_; 
v_snd_1456_ = lean_ctor_get(v___x_1452_, 1);
lean_inc(v_snd_1456_);
lean_dec_ref(v___x_1452_);
v___x_1457_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1457_, 0, v_snd_1456_);
return v___x_1457_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13___boxed(lean_object* v_xs_1458_, lean_object* v_x_1459_, lean_object* v_r_1460_){
_start:
{
lean_object* v_res_1461_; 
v_res_1461_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13(v_xs_1458_, v_x_1459_, v_r_1460_);
lean_dec_ref(v_r_1460_);
lean_dec_ref(v_x_1459_);
lean_dec_ref(v_xs_1458_);
return v_res_1461_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__2(void){
_start:
{
lean_object* v___x_1465_; lean_object* v___x_1466_; 
v___x_1465_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__1));
v___x_1466_ = l_Lean_MessageData_ofFormat(v___x_1465_);
return v___x_1466_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11(lean_object* v_ts_1467_, lean_object* v_inGoal_1468_, lean_object* v_outGoals_1469_, lean_object* v_preMCtx_1470_, lean_object* v_postMCtx_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_){
_start:
{
lean_object* v_visibleGoals_1475_; lean_object* v_invisibleGoals_1476_; lean_object* v___x_1478_; uint8_t v_isShared_1479_; uint8_t v_isSharedCheck_1497_; 
v_visibleGoals_1475_ = lean_ctor_get(v_ts_1467_, 0);
v_invisibleGoals_1476_ = lean_ctor_get(v_ts_1467_, 1);
v_isSharedCheck_1497_ = !lean_is_exclusive(v_ts_1467_);
if (v_isSharedCheck_1497_ == 0)
{
v___x_1478_ = v_ts_1467_;
v_isShared_1479_ = v_isSharedCheck_1497_;
goto v_resetjp_1477_;
}
else
{
lean_inc(v_invisibleGoals_1476_);
lean_inc(v_visibleGoals_1475_);
lean_dec(v_ts_1467_);
v___x_1478_ = lean_box(0);
v_isShared_1479_ = v_isSharedCheck_1497_;
goto v_resetjp_1477_;
}
v_resetjp_1477_:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; 
v___x_1480_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_toStepTree___closed__1, &lp_aesop_Aesop_Script_UScript_toStepTree___closed__1_once, _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__1);
lean_inc(v_inGoal_1468_);
v___x_1481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1481_, 0, v_inGoal_1468_);
lean_ctor_set(v___x_1481_, 1, v___x_1480_);
v___x_1482_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___at___00Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11_spec__13(v_visibleGoals_1475_, v___x_1481_, v_outGoals_1469_);
lean_dec_ref_known(v___x_1481_, 2);
lean_dec_ref(v_visibleGoals_1475_);
if (lean_obj_tag(v___x_1482_) == 1)
{
lean_object* v_val_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1494_; 
lean_dec(v_inGoal_1468_);
v_val_1483_ = lean_ctor_get(v___x_1482_, 0);
v_isSharedCheck_1494_ = !lean_is_exclusive(v___x_1482_);
if (v_isSharedCheck_1494_ == 0)
{
v___x_1485_ = v___x_1482_;
v_isShared_1486_ = v_isSharedCheck_1494_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_val_1483_);
lean_dec(v___x_1482_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1494_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v_ts_1488_; 
if (v_isShared_1479_ == 0)
{
lean_ctor_set(v___x_1478_, 0, v_val_1483_);
v_ts_1488_ = v___x_1478_;
goto v_reusejp_1487_;
}
else
{
lean_object* v_reuseFailAlloc_1493_; 
v_reuseFailAlloc_1493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1493_, 0, v_val_1483_);
lean_ctor_set(v_reuseFailAlloc_1493_, 1, v_invisibleGoals_1476_);
v_ts_1488_ = v_reuseFailAlloc_1493_;
goto v_reusejp_1487_;
}
v_reusejp_1487_:
{
lean_object* v___x_1489_; lean_object* v___x_1491_; 
v___x_1489_ = lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(v_ts_1488_, v_preMCtx_1470_, v_postMCtx_1471_);
if (v_isShared_1486_ == 0)
{
lean_ctor_set_tag(v___x_1485_, 0);
lean_ctor_set(v___x_1485_, 0, v___x_1489_);
v___x_1491_ = v___x_1485_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v___x_1489_);
v___x_1491_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
return v___x_1491_;
}
}
}
}
else
{
lean_object* v___x_1495_; lean_object* v___x_1496_; 
lean_dec(v___x_1482_);
lean_del_object(v___x_1478_);
lean_dec_ref(v_invisibleGoals_1476_);
v___x_1495_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__2, &lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__2_once, _init_lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___closed__2);
v___x_1496_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg(v_inGoal_1468_, v___x_1495_, v___y_1472_, v___y_1473_);
return v___x_1496_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11___boxed(lean_object* v_ts_1498_, lean_object* v_inGoal_1499_, lean_object* v_outGoals_1500_, lean_object* v_preMCtx_1501_, lean_object* v_postMCtx_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_){
_start:
{
lean_object* v_res_1506_; 
v_res_1506_ = lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11(v_ts_1498_, v_inGoal_1499_, v_outGoals_1500_, v_preMCtx_1501_, v_postMCtx_1502_, v___y_1503_, v___y_1504_);
lean_dec(v___y_1504_);
lean_dec_ref(v___y_1503_);
lean_dec_ref(v_postMCtx_1502_);
lean_dec_ref(v_preMCtx_1501_);
lean_dec_ref(v_outGoals_1500_);
return v_res_1506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7(lean_object* v_tacticState_1507_, lean_object* v_step_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_){
_start:
{
lean_object* v_preState_1512_; lean_object* v_meta_1513_; lean_object* v_postState_1514_; lean_object* v_meta_1515_; lean_object* v_preGoal_1516_; lean_object* v_postGoals_1517_; lean_object* v_mctx_1518_; lean_object* v_mctx_1519_; lean_object* v___x_1520_; 
v_preState_1512_ = lean_ctor_get(v_step_1508_, 0);
v_meta_1513_ = lean_ctor_get(v_preState_1512_, 1);
lean_inc_ref(v_meta_1513_);
v_postState_1514_ = lean_ctor_get(v_step_1508_, 3);
v_meta_1515_ = lean_ctor_get(v_postState_1514_, 1);
lean_inc_ref(v_meta_1515_);
v_preGoal_1516_ = lean_ctor_get(v_step_1508_, 1);
lean_inc(v_preGoal_1516_);
v_postGoals_1517_ = lean_ctor_get(v_step_1508_, 4);
lean_inc_ref(v_postGoals_1517_);
lean_dec_ref(v_step_1508_);
v_mctx_1518_ = lean_ctor_get(v_meta_1513_, 0);
lean_inc_ref(v_mctx_1518_);
lean_dec_ref(v_meta_1513_);
v_mctx_1519_ = lean_ctor_get(v_meta_1515_, 0);
lean_inc_ref(v_mctx_1519_);
lean_dec_ref(v_meta_1515_);
v___x_1520_ = lp_aesop_Aesop_Script_TacticState_applyTactic___at___00Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7_spec__11(v_tacticState_1507_, v_preGoal_1516_, v_postGoals_1517_, v_mctx_1518_, v_mctx_1519_, v___y_1509_, v___y_1510_);
lean_dec_ref(v_mctx_1519_);
lean_dec_ref(v_mctx_1518_);
lean_dec_ref(v_postGoals_1517_);
return v___x_1520_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7___boxed(lean_object* v_tacticState_1521_, lean_object* v_step_1522_, lean_object* v___y_1523_, lean_object* v___y_1524_, lean_object* v___y_1525_){
_start:
{
lean_object* v_res_1526_; 
v_res_1526_ = lp_aesop_Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7(v_tacticState_1521_, v_step_1522_, v___y_1523_, v___y_1524_);
lean_dec(v___y_1524_);
lean_dec_ref(v___y_1523_);
return v_res_1526_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(lean_object* v_opts_1527_, lean_object* v_opt_1528_){
_start:
{
lean_object* v_name_1529_; lean_object* v_defValue_1530_; lean_object* v_map_1531_; lean_object* v___x_1532_; 
v_name_1529_ = lean_ctor_get(v_opt_1528_, 0);
v_defValue_1530_ = lean_ctor_get(v_opt_1528_, 1);
v_map_1531_ = lean_ctor_get(v_opts_1527_, 0);
v___x_1532_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1531_, v_name_1529_);
if (lean_obj_tag(v___x_1532_) == 0)
{
uint8_t v___x_1533_; 
v___x_1533_ = lean_unbox(v_defValue_1530_);
return v___x_1533_;
}
else
{
lean_object* v_val_1534_; 
v_val_1534_ = lean_ctor_get(v___x_1532_, 0);
lean_inc(v_val_1534_);
lean_dec_ref_known(v___x_1532_, 1);
if (lean_obj_tag(v_val_1534_) == 1)
{
uint8_t v_v_1535_; 
v_v_1535_ = lean_ctor_get_uint8(v_val_1534_, 0);
lean_dec_ref_known(v_val_1534_, 0);
return v_v_1535_;
}
else
{
uint8_t v___x_1536_; 
lean_dec(v_val_1534_);
v___x_1536_ = lean_unbox(v_defValue_1530_);
return v___x_1536_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0___boxed(lean_object* v_opts_1537_, lean_object* v_opt_1538_){
_start:
{
uint8_t v_res_1539_; lean_object* v_r_1540_; 
v_res_1539_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(v_opts_1537_, v_opt_1538_);
lean_dec_ref(v_opt_1538_);
lean_dec_ref(v_opts_1537_);
v_r_1540_ = lean_box(v_res_1539_);
return v_r_1540_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(lean_object* v_opt_1541_, lean_object* v___y_1542_){
_start:
{
lean_object* v_options_1544_; lean_object* v_option_1545_; uint8_t v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; 
v_options_1544_ = lean_ctor_get(v___y_1542_, 2);
v_option_1545_ = lean_ctor_get(v_opt_1541_, 1);
v___x_1546_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(v_options_1544_, v_option_1545_);
v___x_1547_ = lean_box(v___x_1546_);
v___x_1548_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1548_, 0, v___x_1547_);
return v___x_1548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg___boxed(lean_object* v_opt_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_){
_start:
{
lean_object* v_res_1552_; 
v_res_1552_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v_opt_1549_, v___y_1550_);
lean_dec_ref(v___y_1550_);
lean_dec_ref(v_opt_1549_);
return v_res_1552_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0(void){
_start:
{
lean_object* v___x_1553_; double v___x_1554_; 
v___x_1553_ = lean_unsigned_to_nat(0u);
v___x_1554_ = lean_float_of_nat(v___x_1553_);
return v___x_1554_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(lean_object* v_cls_1555_, lean_object* v_msg_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_){
_start:
{
lean_object* v_ref_1560_; lean_object* v___x_1561_; lean_object* v_a_1562_; lean_object* v___x_1564_; uint8_t v_isShared_1565_; uint8_t v_isSharedCheck_1606_; 
v_ref_1560_ = lean_ctor_get(v___y_1557_, 5);
v___x_1561_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14(v_msg_1556_, v___y_1557_, v___y_1558_);
v_a_1562_ = lean_ctor_get(v___x_1561_, 0);
v_isSharedCheck_1606_ = !lean_is_exclusive(v___x_1561_);
if (v_isSharedCheck_1606_ == 0)
{
v___x_1564_ = v___x_1561_;
v_isShared_1565_ = v_isSharedCheck_1606_;
goto v_resetjp_1563_;
}
else
{
lean_inc(v_a_1562_);
lean_dec(v___x_1561_);
v___x_1564_ = lean_box(0);
v_isShared_1565_ = v_isSharedCheck_1606_;
goto v_resetjp_1563_;
}
v_resetjp_1563_:
{
lean_object* v___x_1566_; lean_object* v_traceState_1567_; lean_object* v_env_1568_; lean_object* v_nextMacroScope_1569_; lean_object* v_ngen_1570_; lean_object* v_auxDeclNGen_1571_; lean_object* v_cache_1572_; lean_object* v_messages_1573_; lean_object* v_infoState_1574_; lean_object* v_snapshotTasks_1575_; lean_object* v___x_1577_; uint8_t v_isShared_1578_; uint8_t v_isSharedCheck_1605_; 
v___x_1566_ = lean_st_ref_take(v___y_1558_);
v_traceState_1567_ = lean_ctor_get(v___x_1566_, 4);
v_env_1568_ = lean_ctor_get(v___x_1566_, 0);
v_nextMacroScope_1569_ = lean_ctor_get(v___x_1566_, 1);
v_ngen_1570_ = lean_ctor_get(v___x_1566_, 2);
v_auxDeclNGen_1571_ = lean_ctor_get(v___x_1566_, 3);
v_cache_1572_ = lean_ctor_get(v___x_1566_, 5);
v_messages_1573_ = lean_ctor_get(v___x_1566_, 6);
v_infoState_1574_ = lean_ctor_get(v___x_1566_, 7);
v_snapshotTasks_1575_ = lean_ctor_get(v___x_1566_, 8);
v_isSharedCheck_1605_ = !lean_is_exclusive(v___x_1566_);
if (v_isSharedCheck_1605_ == 0)
{
v___x_1577_ = v___x_1566_;
v_isShared_1578_ = v_isSharedCheck_1605_;
goto v_resetjp_1576_;
}
else
{
lean_inc(v_snapshotTasks_1575_);
lean_inc(v_infoState_1574_);
lean_inc(v_messages_1573_);
lean_inc(v_cache_1572_);
lean_inc(v_traceState_1567_);
lean_inc(v_auxDeclNGen_1571_);
lean_inc(v_ngen_1570_);
lean_inc(v_nextMacroScope_1569_);
lean_inc(v_env_1568_);
lean_dec(v___x_1566_);
v___x_1577_ = lean_box(0);
v_isShared_1578_ = v_isSharedCheck_1605_;
goto v_resetjp_1576_;
}
v_resetjp_1576_:
{
uint64_t v_tid_1579_; lean_object* v_traces_1580_; lean_object* v___x_1582_; uint8_t v_isShared_1583_; uint8_t v_isSharedCheck_1604_; 
v_tid_1579_ = lean_ctor_get_uint64(v_traceState_1567_, sizeof(void*)*1);
v_traces_1580_ = lean_ctor_get(v_traceState_1567_, 0);
v_isSharedCheck_1604_ = !lean_is_exclusive(v_traceState_1567_);
if (v_isSharedCheck_1604_ == 0)
{
v___x_1582_ = v_traceState_1567_;
v_isShared_1583_ = v_isSharedCheck_1604_;
goto v_resetjp_1581_;
}
else
{
lean_inc(v_traces_1580_);
lean_dec(v_traceState_1567_);
v___x_1582_ = lean_box(0);
v_isShared_1583_ = v_isSharedCheck_1604_;
goto v_resetjp_1581_;
}
v_resetjp_1581_:
{
lean_object* v___x_1584_; double v___x_1585_; uint8_t v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1594_; 
v___x_1584_ = lean_box(0);
v___x_1585_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0);
v___x_1586_ = 0;
v___x_1587_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__11));
v___x_1588_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1588_, 0, v_cls_1555_);
lean_ctor_set(v___x_1588_, 1, v___x_1584_);
lean_ctor_set(v___x_1588_, 2, v___x_1587_);
lean_ctor_set_float(v___x_1588_, sizeof(void*)*3, v___x_1585_);
lean_ctor_set_float(v___x_1588_, sizeof(void*)*3 + 8, v___x_1585_);
lean_ctor_set_uint8(v___x_1588_, sizeof(void*)*3 + 16, v___x_1586_);
v___x_1589_ = ((lean_object*)(lp_aesop_Array_filterMapM___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__2___closed__0));
v___x_1590_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1590_, 0, v___x_1588_);
lean_ctor_set(v___x_1590_, 1, v_a_1562_);
lean_ctor_set(v___x_1590_, 2, v___x_1589_);
lean_inc(v_ref_1560_);
v___x_1591_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1591_, 0, v_ref_1560_);
lean_ctor_set(v___x_1591_, 1, v___x_1590_);
v___x_1592_ = l_Lean_PersistentArray_push___redArg(v_traces_1580_, v___x_1591_);
if (v_isShared_1583_ == 0)
{
lean_ctor_set(v___x_1582_, 0, v___x_1592_);
v___x_1594_ = v___x_1582_;
goto v_reusejp_1593_;
}
else
{
lean_object* v_reuseFailAlloc_1603_; 
v_reuseFailAlloc_1603_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1603_, 0, v___x_1592_);
lean_ctor_set_uint64(v_reuseFailAlloc_1603_, sizeof(void*)*1, v_tid_1579_);
v___x_1594_ = v_reuseFailAlloc_1603_;
goto v_reusejp_1593_;
}
v_reusejp_1593_:
{
lean_object* v___x_1596_; 
if (v_isShared_1578_ == 0)
{
lean_ctor_set(v___x_1577_, 4, v___x_1594_);
v___x_1596_ = v___x_1577_;
goto v_reusejp_1595_;
}
else
{
lean_object* v_reuseFailAlloc_1602_; 
v_reuseFailAlloc_1602_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1602_, 0, v_env_1568_);
lean_ctor_set(v_reuseFailAlloc_1602_, 1, v_nextMacroScope_1569_);
lean_ctor_set(v_reuseFailAlloc_1602_, 2, v_ngen_1570_);
lean_ctor_set(v_reuseFailAlloc_1602_, 3, v_auxDeclNGen_1571_);
lean_ctor_set(v_reuseFailAlloc_1602_, 4, v___x_1594_);
lean_ctor_set(v_reuseFailAlloc_1602_, 5, v_cache_1572_);
lean_ctor_set(v_reuseFailAlloc_1602_, 6, v_messages_1573_);
lean_ctor_set(v_reuseFailAlloc_1602_, 7, v_infoState_1574_);
lean_ctor_set(v_reuseFailAlloc_1602_, 8, v_snapshotTasks_1575_);
v___x_1596_ = v_reuseFailAlloc_1602_;
goto v_reusejp_1595_;
}
v_reusejp_1595_:
{
lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1600_; 
v___x_1597_ = lean_st_ref_set(v___y_1558_, v___x_1596_);
v___x_1598_ = lean_box(0);
if (v_isShared_1565_ == 0)
{
lean_ctor_set(v___x_1564_, 0, v___x_1598_);
v___x_1600_ = v___x_1564_;
goto v_reusejp_1599_;
}
else
{
lean_object* v_reuseFailAlloc_1601_; 
v_reuseFailAlloc_1601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1601_, 0, v___x_1598_);
v___x_1600_ = v_reuseFailAlloc_1601_;
goto v_reusejp_1599_;
}
v_reusejp_1599_:
{
return v___x_1600_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___boxed(lean_object* v_cls_1607_, lean_object* v_msg_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_){
_start:
{
lean_object* v_res_1612_; 
v_res_1612_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_cls_1607_, v_msg_1608_, v___y_1609_, v___y_1610_);
lean_dec(v___y_1610_);
lean_dec_ref(v___y_1609_);
return v_res_1612_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__2(void){
_start:
{
lean_object* v___x_1616_; lean_object* v___x_1617_; 
v___x_1616_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__1));
v___x_1617_ = l_Lean_stringToMessageData(v___x_1616_);
return v___x_1617_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__4(void){
_start:
{
lean_object* v___x_1619_; lean_object* v___x_1620_; 
v___x_1619_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__3));
v___x_1620_ = l_Lean_stringToMessageData(v___x_1619_);
return v___x_1620_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__6(void){
_start:
{
lean_object* v___x_1622_; lean_object* v___x_1623_; 
v___x_1622_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__5));
v___x_1623_ = l_Lean_stringToMessageData(v___x_1622_);
return v___x_1623_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__10(void){
_start:
{
lean_object* v___x_1627_; lean_object* v___x_1628_; 
v___x_1627_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__9));
v___x_1628_ = l_Lean_stringToMessageData(v___x_1627_);
return v___x_1628_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__12(void){
_start:
{
lean_object* v___x_1630_; lean_object* v___x_1631_; 
v___x_1630_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__11));
v___x_1631_ = l_Lean_stringToMessageData(v___x_1630_);
return v___x_1631_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__14(void){
_start:
{
lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1633_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__13));
v___x_1634_ = l_Lean_stringToMessageData(v___x_1633_);
return v___x_1634_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(lean_object* v_uscript_1635_, lean_object* v_focusable_1636_, lean_object* v_numSiblings_1637_, lean_object* v_start_1638_, lean_object* v_stop_1639_, lean_object* v_tacticState_1640_, lean_object* v_a_1641_, lean_object* v_a_1642_){
_start:
{
lean_object* v___y_1645_; lean_object* v___y_1646_; lean_object* v___y_1647_; lean_object* v___y_1648_; lean_object* v___y_1649_; lean_object* v_fst_1650_; lean_object* v_snd_1651_; uint8_t v___x_1708_; 
v___x_1708_ = lean_nat_dec_lt(v_stop_1639_, v_start_1638_);
if (v___x_1708_ == 0)
{
lean_object* v___x_1709_; uint8_t v___x_1710_; 
v___x_1709_ = lean_array_get_size(v_uscript_1635_);
v___x_1710_ = lean_nat_dec_lt(v_start_1638_, v___x_1709_);
if (v___x_1710_ == 0)
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; 
v___x_1711_ = lean_box(0);
v___x_1712_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1712_, 0, v___x_1711_);
lean_ctor_set(v___x_1712_, 1, v_tacticState_1640_);
v___x_1713_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1713_, 0, v___x_1712_);
return v___x_1713_;
}
else
{
lean_object* v___x_1714_; lean_object* v___x_1715_; 
v___x_1714_ = lp_aesop_Aesop_TraceOption_script;
v___x_1715_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_1714_, v_a_1641_);
if (lean_obj_tag(v___x_1715_) == 0)
{
lean_object* v_a_1716_; lean_object* v___x_1718_; uint8_t v_isShared_1719_; uint8_t v_isSharedCheck_2003_; 
v_a_1716_ = lean_ctor_get(v___x_1715_, 0);
v_isSharedCheck_2003_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1718_ = v___x_1715_;
v_isShared_1719_ = v_isSharedCheck_2003_;
goto v_resetjp_1717_;
}
else
{
lean_inc(v_a_1716_);
lean_dec(v___x_1715_);
v___x_1718_ = lean_box(0);
v_isShared_1719_ = v_isSharedCheck_2003_;
goto v_resetjp_1717_;
}
v_resetjp_1717_:
{
lean_object* v___x_1720_; lean_object* v___y_1722_; lean_object* v___y_1723_; lean_object* v___y_1724_; lean_object* v___y_1757_; lean_object* v___y_1758_; lean_object* v___y_1759_; lean_object* v___y_1760_; lean_object* v___y_1761_; lean_object* v___y_1762_; lean_object* v___y_1797_; lean_object* v___y_1798_; lean_object* v___y_1799_; lean_object* v___y_1800_; lean_object* v___y_1801_; lean_object* v___y_1842_; lean_object* v___y_1843_; lean_object* v___y_1844_; lean_object* v___y_1845_; lean_object* v___y_1846_; lean_object* v___y_1879_; lean_object* v___y_1880_; lean_object* v___y_1881_; lean_object* v___y_1882_; lean_object* v___y_1883_; lean_object* v___y_1884_; lean_object* v___y_1885_; lean_object* v___y_1886_; lean_object* v___y_1900_; lean_object* v___y_1901_; lean_object* v___y_1902_; lean_object* v___y_1903_; lean_object* v___y_1921_; lean_object* v___y_1922_; uint8_t v___x_1962_; 
v___x_1720_ = lean_array_fget_borrowed(v_uscript_1635_, v_start_1638_);
v___x_1962_ = lean_unbox(v_a_1716_);
lean_dec(v_a_1716_);
if (v___x_1962_ == 0)
{
v___y_1921_ = v_a_1641_;
v___y_1922_ = v_a_1642_;
goto v___jp_1920_;
}
else
{
lean_object* v_tactic_1963_; lean_object* v_traceClass_1964_; lean_object* v_preGoal_1965_; lean_object* v_postGoals_1966_; lean_object* v_uTactic_1967_; lean_object* v___x_1969_; uint8_t v_isShared_1970_; uint8_t v_isSharedCheck_2001_; 
v_tactic_1963_ = lean_ctor_get(v___x_1720_, 2);
lean_inc_ref(v_tactic_1963_);
v_traceClass_1964_ = lean_ctor_get(v___x_1714_, 0);
v_preGoal_1965_ = lean_ctor_get(v___x_1720_, 1);
v_postGoals_1966_ = lean_ctor_get(v___x_1720_, 4);
v_uTactic_1967_ = lean_ctor_get(v_tactic_1963_, 0);
v_isSharedCheck_2001_ = !lean_is_exclusive(v_tactic_1963_);
if (v_isSharedCheck_2001_ == 0)
{
lean_object* v_unused_2002_; 
v_unused_2002_ = lean_ctor_get(v_tactic_1963_, 1);
lean_dec(v_unused_2002_);
v___x_1969_ = v_tactic_1963_;
v_isShared_1970_ = v_isSharedCheck_2001_;
goto v_resetjp_1968_;
}
else
{
lean_inc(v_uTactic_1967_);
lean_dec(v_tactic_1963_);
v___x_1969_ = lean_box(0);
v_isShared_1970_ = v_isSharedCheck_2001_;
goto v_resetjp_1968_;
}
v_resetjp_1968_:
{
size_t v_sz_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; lean_object* v___x_1976_; 
v_sz_1971_ = lean_array_size(v_postGoals_1966_);
v___x_1972_ = lean_obj_once(&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__14, &lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__14_once, _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__14);
lean_inc(v_preGoal_1965_);
v___x_1973_ = l_Lean_MessageData_ofName(v_preGoal_1965_);
v___x_1974_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5);
if (v_isShared_1970_ == 0)
{
lean_ctor_set_tag(v___x_1969_, 7);
lean_ctor_set(v___x_1969_, 1, v___x_1974_);
lean_ctor_set(v___x_1969_, 0, v___x_1973_);
v___x_1976_ = v___x_1969_;
goto v_reusejp_1975_;
}
else
{
lean_object* v_reuseFailAlloc_2000_; 
v_reuseFailAlloc_2000_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2000_, 0, v___x_1973_);
lean_ctor_set(v_reuseFailAlloc_2000_, 1, v___x_1974_);
v___x_1976_ = v_reuseFailAlloc_2000_;
goto v_reusejp_1975_;
}
v_reusejp_1975_:
{
size_t v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; 
v___x_1977_ = ((size_t)0ULL);
lean_inc_ref(v_postGoals_1966_);
v___x_1978_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0(v_sz_1971_, v___x_1977_, v_postGoals_1966_);
v___x_1979_ = lean_array_to_list(v___x_1978_);
v___x_1980_ = lean_box(0);
v___x_1981_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__1(v___x_1979_, v___x_1980_);
v___x_1982_ = l_Lean_MessageData_ofList(v___x_1981_);
v___x_1983_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1983_, 0, v___x_1976_);
lean_ctor_set(v___x_1983_, 1, v___x_1982_);
v___x_1984_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7);
v___x_1985_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1985_, 0, v___x_1983_);
lean_ctor_set(v___x_1985_, 1, v___x_1984_);
v___x_1986_ = l_Lean_MessageData_ofSyntax(v_uTactic_1967_);
v___x_1987_ = l_Lean_indentD(v___x_1986_);
v___x_1988_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1988_, 0, v___x_1985_);
lean_ctor_set(v___x_1988_, 1, v___x_1987_);
v___x_1989_ = l_Lean_indentD(v___x_1988_);
v___x_1990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1990_, 0, v___x_1972_);
lean_ctor_set(v___x_1990_, 1, v___x_1989_);
lean_inc(v_traceClass_1964_);
v___x_1991_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_1964_, v___x_1990_, v_a_1641_, v_a_1642_);
if (lean_obj_tag(v___x_1991_) == 0)
{
lean_dec_ref_known(v___x_1991_, 1);
v___y_1921_ = v_a_1641_;
v___y_1922_ = v_a_1642_;
goto v___jp_1920_;
}
else
{
lean_object* v_a_1992_; lean_object* v___x_1994_; uint8_t v_isShared_1995_; uint8_t v_isSharedCheck_1999_; 
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v_a_1992_ = lean_ctor_get(v___x_1991_, 0);
v_isSharedCheck_1999_ = !lean_is_exclusive(v___x_1991_);
if (v_isSharedCheck_1999_ == 0)
{
v___x_1994_ = v___x_1991_;
v_isShared_1995_ = v_isSharedCheck_1999_;
goto v_resetjp_1993_;
}
else
{
lean_inc(v_a_1992_);
lean_dec(v___x_1991_);
v___x_1994_ = lean_box(0);
v_isShared_1995_ = v_isSharedCheck_1999_;
goto v_resetjp_1993_;
}
v_resetjp_1993_:
{
lean_object* v___x_1997_; 
if (v_isShared_1995_ == 0)
{
v___x_1997_ = v___x_1994_;
goto v_reusejp_1996_;
}
else
{
lean_object* v_reuseFailAlloc_1998_; 
v_reuseFailAlloc_1998_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1998_, 0, v_a_1992_);
v___x_1997_ = v_reuseFailAlloc_1998_;
goto v_reusejp_1996_;
}
v_reusejp_1996_:
{
return v___x_1997_;
}
}
}
}
}
}
v___jp_1721_:
{
lean_object* v___x_1725_; 
lean_inc(v___x_1720_);
v___x_1725_ = lp_aesop_Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7(v_tacticState_1640_, v___x_1720_, v___y_1724_, v___y_1722_);
if (lean_obj_tag(v___x_1725_) == 0)
{
lean_object* v_a_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; 
v_a_1726_ = lean_ctor_get(v___x_1725_, 0);
lean_inc(v_a_1726_);
lean_dec_ref_known(v___x_1725_, 1);
v___x_1727_ = lean_unsigned_to_nat(1u);
v___x_1728_ = lean_nat_add(v_start_1638_, v___x_1727_);
v___x_1729_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_1635_, v_focusable_1636_, v_numSiblings_1637_, v___x_1728_, v_stop_1639_, v_a_1726_, v___y_1724_, v___y_1722_);
lean_dec(v___x_1728_);
if (lean_obj_tag(v___x_1729_) == 0)
{
lean_object* v_a_1730_; lean_object* v___x_1732_; uint8_t v_isShared_1733_; uint8_t v_isSharedCheck_1747_; 
v_a_1730_ = lean_ctor_get(v___x_1729_, 0);
v_isSharedCheck_1747_ = !lean_is_exclusive(v___x_1729_);
if (v_isSharedCheck_1747_ == 0)
{
v___x_1732_ = v___x_1729_;
v_isShared_1733_ = v_isSharedCheck_1747_;
goto v_resetjp_1731_;
}
else
{
lean_inc(v_a_1730_);
lean_dec(v___x_1729_);
v___x_1732_ = lean_box(0);
v_isShared_1733_ = v_isSharedCheck_1747_;
goto v_resetjp_1731_;
}
v_resetjp_1731_:
{
lean_object* v_fst_1734_; lean_object* v_snd_1735_; lean_object* v___x_1737_; uint8_t v_isShared_1738_; uint8_t v_isSharedCheck_1746_; 
v_fst_1734_ = lean_ctor_get(v_a_1730_, 0);
v_snd_1735_ = lean_ctor_get(v_a_1730_, 1);
v_isSharedCheck_1746_ = !lean_is_exclusive(v_a_1730_);
if (v_isSharedCheck_1746_ == 0)
{
v___x_1737_ = v_a_1730_;
v_isShared_1738_ = v_isSharedCheck_1746_;
goto v_resetjp_1736_;
}
else
{
lean_inc(v_snd_1735_);
lean_inc(v_fst_1734_);
lean_dec(v_a_1730_);
v___x_1737_ = lean_box(0);
v_isShared_1738_ = v_isSharedCheck_1746_;
goto v_resetjp_1736_;
}
v_resetjp_1736_:
{
lean_object* v___x_1739_; lean_object* v___x_1741_; 
lean_inc(v___x_1720_);
v___x_1739_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1739_, 0, v___y_1723_);
lean_ctor_set(v___x_1739_, 1, v___x_1720_);
lean_ctor_set(v___x_1739_, 2, v_fst_1734_);
if (v_isShared_1738_ == 0)
{
lean_ctor_set(v___x_1737_, 0, v___x_1739_);
v___x_1741_ = v___x_1737_;
goto v_reusejp_1740_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v___x_1739_);
lean_ctor_set(v_reuseFailAlloc_1745_, 1, v_snd_1735_);
v___x_1741_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1740_;
}
v_reusejp_1740_:
{
lean_object* v___x_1743_; 
if (v_isShared_1733_ == 0)
{
lean_ctor_set(v___x_1732_, 0, v___x_1741_);
v___x_1743_ = v___x_1732_;
goto v_reusejp_1742_;
}
else
{
lean_object* v_reuseFailAlloc_1744_; 
v_reuseFailAlloc_1744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1744_, 0, v___x_1741_);
v___x_1743_ = v_reuseFailAlloc_1744_;
goto v_reusejp_1742_;
}
v_reusejp_1742_:
{
return v___x_1743_;
}
}
}
}
}
else
{
lean_dec(v___y_1723_);
return v___x_1729_;
}
}
else
{
lean_object* v_a_1748_; lean_object* v___x_1750_; uint8_t v_isShared_1751_; uint8_t v_isSharedCheck_1755_; 
lean_dec(v___y_1723_);
v_a_1748_ = lean_ctor_get(v___x_1725_, 0);
v_isSharedCheck_1755_ = !lean_is_exclusive(v___x_1725_);
if (v_isSharedCheck_1755_ == 0)
{
v___x_1750_ = v___x_1725_;
v_isShared_1751_ = v_isSharedCheck_1755_;
goto v_resetjp_1749_;
}
else
{
lean_inc(v_a_1748_);
lean_dec(v___x_1725_);
v___x_1750_ = lean_box(0);
v_isShared_1751_ = v_isSharedCheck_1755_;
goto v_resetjp_1749_;
}
v_resetjp_1749_:
{
lean_object* v___x_1753_; 
if (v_isShared_1751_ == 0)
{
v___x_1753_ = v___x_1750_;
goto v_reusejp_1752_;
}
else
{
lean_object* v_reuseFailAlloc_1754_; 
v_reuseFailAlloc_1754_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1754_, 0, v_a_1748_);
v___x_1753_ = v_reuseFailAlloc_1754_;
goto v_reusejp_1752_;
}
v_reusejp_1752_:
{
return v___x_1753_;
}
}
}
}
v___jp_1756_:
{
if (lean_obj_tag(v___y_1757_) == 0)
{
lean_dec(v___y_1760_);
lean_dec(v___y_1759_);
v___y_1722_ = v___y_1762_;
v___y_1723_ = v___y_1758_;
v___y_1724_ = v___y_1761_;
goto v___jp_1721_;
}
else
{
lean_object* v_val_1763_; lean_object* v___x_1764_; uint8_t v___x_1765_; 
v_val_1763_ = lean_ctor_get(v___y_1757_, 0);
lean_inc(v_val_1763_);
lean_dec_ref_known(v___y_1757_, 1);
v___x_1764_ = lean_unsigned_to_nat(0u);
v___x_1765_ = lean_nat_dec_eq(v___y_1759_, v___x_1764_);
lean_dec(v___y_1759_);
if (v___x_1765_ == 0)
{
lean_object* v___x_1766_; 
lean_inc(v___y_1760_);
lean_inc_ref(v_tacticState_1640_);
v___x_1766_ = lp_aesop_Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6(v_tacticState_1640_, v___y_1760_, v___y_1761_, v___y_1762_);
if (lean_obj_tag(v___x_1766_) == 0)
{
lean_object* v_a_1767_; lean_object* v___x_1768_; 
v_a_1767_ = lean_ctor_get(v___x_1766_, 0);
lean_inc(v_a_1767_);
lean_dec_ref_known(v___x_1766_, 1);
lean_inc(v___x_1720_);
v___x_1768_ = lp_aesop_Aesop_Script_TacticState_applyStep___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__7(v_a_1767_, v___x_1720_, v___y_1761_, v___y_1762_);
if (lean_obj_tag(v___x_1768_) == 0)
{
lean_object* v_a_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; 
v_a_1769_ = lean_ctor_get(v___x_1768_, 0);
lean_inc(v_a_1769_);
lean_dec_ref_known(v___x_1768_, 1);
v___x_1770_ = lean_unsigned_to_nat(1u);
v___x_1771_ = lean_nat_add(v_start_1638_, v___x_1770_);
v___x_1772_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_1635_, v_focusable_1636_, v_numSiblings_1637_, v___x_1771_, v_val_1763_, v_a_1769_, v___y_1761_, v___y_1762_);
lean_dec(v___x_1771_);
if (lean_obj_tag(v___x_1772_) == 0)
{
lean_object* v_a_1773_; lean_object* v_fst_1774_; lean_object* v_snd_1775_; lean_object* v___x_1776_; 
v_a_1773_ = lean_ctor_get(v___x_1772_, 0);
lean_inc(v_a_1773_);
lean_dec_ref_known(v___x_1772_, 1);
v_fst_1774_ = lean_ctor_get(v_a_1773_, 0);
lean_inc(v_fst_1774_);
v_snd_1775_ = lean_ctor_get(v_a_1773_, 1);
lean_inc(v_snd_1775_);
lean_dec(v_a_1773_);
lean_inc(v___x_1720_);
v___x_1776_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1776_, 0, v___x_1764_);
lean_ctor_set(v___x_1776_, 1, v___x_1720_);
lean_ctor_set(v___x_1776_, 2, v_fst_1774_);
v___y_1645_ = v___y_1762_;
v___y_1646_ = v___y_1758_;
v___y_1647_ = v___y_1761_;
v___y_1648_ = v___y_1760_;
v___y_1649_ = v_val_1763_;
v_fst_1650_ = v___x_1776_;
v_snd_1651_ = v_snd_1775_;
goto v___jp_1644_;
}
else
{
if (lean_obj_tag(v___x_1772_) == 0)
{
lean_object* v_a_1777_; lean_object* v_fst_1778_; lean_object* v_snd_1779_; 
v_a_1777_ = lean_ctor_get(v___x_1772_, 0);
lean_inc(v_a_1777_);
lean_dec_ref_known(v___x_1772_, 1);
v_fst_1778_ = lean_ctor_get(v_a_1777_, 0);
lean_inc(v_fst_1778_);
v_snd_1779_ = lean_ctor_get(v_a_1777_, 1);
lean_inc(v_snd_1779_);
lean_dec(v_a_1777_);
v___y_1645_ = v___y_1762_;
v___y_1646_ = v___y_1758_;
v___y_1647_ = v___y_1761_;
v___y_1648_ = v___y_1760_;
v___y_1649_ = v_val_1763_;
v_fst_1650_ = v_fst_1778_;
v_snd_1651_ = v_snd_1779_;
goto v___jp_1644_;
}
else
{
lean_dec(v_val_1763_);
lean_dec(v___y_1760_);
lean_dec(v___y_1758_);
lean_dec_ref(v_tacticState_1640_);
return v___x_1772_;
}
}
}
else
{
lean_object* v_a_1780_; lean_object* v___x_1782_; uint8_t v_isShared_1783_; uint8_t v_isSharedCheck_1787_; 
lean_dec(v_val_1763_);
lean_dec(v___y_1760_);
lean_dec(v___y_1758_);
lean_dec_ref(v_tacticState_1640_);
v_a_1780_ = lean_ctor_get(v___x_1768_, 0);
v_isSharedCheck_1787_ = !lean_is_exclusive(v___x_1768_);
if (v_isSharedCheck_1787_ == 0)
{
v___x_1782_ = v___x_1768_;
v_isShared_1783_ = v_isSharedCheck_1787_;
goto v_resetjp_1781_;
}
else
{
lean_inc(v_a_1780_);
lean_dec(v___x_1768_);
v___x_1782_ = lean_box(0);
v_isShared_1783_ = v_isSharedCheck_1787_;
goto v_resetjp_1781_;
}
v_resetjp_1781_:
{
lean_object* v___x_1785_; 
if (v_isShared_1783_ == 0)
{
v___x_1785_ = v___x_1782_;
goto v_reusejp_1784_;
}
else
{
lean_object* v_reuseFailAlloc_1786_; 
v_reuseFailAlloc_1786_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1786_, 0, v_a_1780_);
v___x_1785_ = v_reuseFailAlloc_1786_;
goto v_reusejp_1784_;
}
v_reusejp_1784_:
{
return v___x_1785_;
}
}
}
}
else
{
lean_object* v_a_1788_; lean_object* v___x_1790_; uint8_t v_isShared_1791_; uint8_t v_isSharedCheck_1795_; 
lean_dec(v_val_1763_);
lean_dec(v___y_1760_);
lean_dec(v___y_1758_);
lean_dec_ref(v_tacticState_1640_);
v_a_1788_ = lean_ctor_get(v___x_1766_, 0);
v_isSharedCheck_1795_ = !lean_is_exclusive(v___x_1766_);
if (v_isSharedCheck_1795_ == 0)
{
v___x_1790_ = v___x_1766_;
v_isShared_1791_ = v_isSharedCheck_1795_;
goto v_resetjp_1789_;
}
else
{
lean_inc(v_a_1788_);
lean_dec(v___x_1766_);
v___x_1790_ = lean_box(0);
v_isShared_1791_ = v_isSharedCheck_1795_;
goto v_resetjp_1789_;
}
v_resetjp_1789_:
{
lean_object* v___x_1793_; 
if (v_isShared_1791_ == 0)
{
v___x_1793_ = v___x_1790_;
goto v_reusejp_1792_;
}
else
{
lean_object* v_reuseFailAlloc_1794_; 
v_reuseFailAlloc_1794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1794_, 0, v_a_1788_);
v___x_1793_ = v_reuseFailAlloc_1794_;
goto v_reusejp_1792_;
}
v_reusejp_1792_:
{
return v___x_1793_;
}
}
}
}
else
{
lean_dec(v_val_1763_);
lean_dec(v___y_1760_);
v___y_1722_ = v___y_1762_;
v___y_1723_ = v___y_1758_;
v___y_1724_ = v___y_1761_;
goto v___jp_1721_;
}
}
}
v___jp_1796_:
{
lean_object* v___x_1802_; 
v___x_1802_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(v_tacticState_1640_, v___y_1799_);
if (lean_obj_tag(v___x_1802_) == 1)
{
lean_object* v_val_1803_; lean_object* v___x_1805_; uint8_t v_isShared_1806_; uint8_t v_isSharedCheck_1835_; 
lean_del_object(v___x_1718_);
v_val_1803_ = lean_ctor_get(v___x_1802_, 0);
v_isSharedCheck_1835_ = !lean_is_exclusive(v___x_1802_);
if (v_isSharedCheck_1835_ == 0)
{
v___x_1805_ = v___x_1802_;
v_isShared_1806_ = v_isSharedCheck_1835_;
goto v_resetjp_1804_;
}
else
{
lean_inc(v_val_1803_);
lean_dec(v___x_1802_);
v___x_1805_ = lean_box(0);
v_isShared_1806_ = v_isSharedCheck_1835_;
goto v_resetjp_1804_;
}
v_resetjp_1804_:
{
lean_object* v___x_1807_; 
v___x_1807_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_1714_, v___y_1800_);
if (lean_obj_tag(v___x_1807_) == 0)
{
lean_object* v_a_1808_; uint8_t v___x_1809_; 
v_a_1808_ = lean_ctor_get(v___x_1807_, 0);
lean_inc(v_a_1808_);
lean_dec_ref_known(v___x_1807_, 1);
v___x_1809_ = lean_unbox(v_a_1808_);
lean_dec(v_a_1808_);
if (v___x_1809_ == 0)
{
lean_del_object(v___x_1805_);
v___y_1757_ = v___y_1797_;
v___y_1758_ = v_val_1803_;
v___y_1759_ = v___y_1798_;
v___y_1760_ = v___y_1799_;
v___y_1761_ = v___y_1800_;
v___y_1762_ = v___y_1801_;
goto v___jp_1756_;
}
else
{
lean_object* v_traceClass_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1814_; 
v_traceClass_1810_ = lean_ctor_get(v___x_1714_, 0);
v___x_1811_ = lean_obj_once(&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__2, &lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__2_once, _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__2);
lean_inc(v_val_1803_);
v___x_1812_ = l_Nat_reprFast(v_val_1803_);
if (v_isShared_1806_ == 0)
{
lean_ctor_set_tag(v___x_1805_, 3);
lean_ctor_set(v___x_1805_, 0, v___x_1812_);
v___x_1814_ = v___x_1805_;
goto v_reusejp_1813_;
}
else
{
lean_object* v_reuseFailAlloc_1826_; 
v_reuseFailAlloc_1826_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1826_, 0, v___x_1812_);
v___x_1814_ = v_reuseFailAlloc_1826_;
goto v_reusejp_1813_;
}
v_reusejp_1813_:
{
lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; 
v___x_1815_ = l_Lean_MessageData_ofFormat(v___x_1814_);
v___x_1816_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1816_, 0, v___x_1811_);
lean_ctor_set(v___x_1816_, 1, v___x_1815_);
lean_inc(v_traceClass_1810_);
v___x_1817_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_1810_, v___x_1816_, v___y_1800_, v___y_1801_);
if (lean_obj_tag(v___x_1817_) == 0)
{
lean_dec_ref_known(v___x_1817_, 1);
v___y_1757_ = v___y_1797_;
v___y_1758_ = v_val_1803_;
v___y_1759_ = v___y_1798_;
v___y_1760_ = v___y_1799_;
v___y_1761_ = v___y_1800_;
v___y_1762_ = v___y_1801_;
goto v___jp_1756_;
}
else
{
lean_object* v_a_1818_; lean_object* v___x_1820_; uint8_t v_isShared_1821_; uint8_t v_isSharedCheck_1825_; 
lean_dec(v_val_1803_);
lean_dec(v___y_1799_);
lean_dec(v___y_1798_);
lean_dec(v___y_1797_);
lean_dec_ref(v_tacticState_1640_);
v_a_1818_ = lean_ctor_get(v___x_1817_, 0);
v_isSharedCheck_1825_ = !lean_is_exclusive(v___x_1817_);
if (v_isSharedCheck_1825_ == 0)
{
v___x_1820_ = v___x_1817_;
v_isShared_1821_ = v_isSharedCheck_1825_;
goto v_resetjp_1819_;
}
else
{
lean_inc(v_a_1818_);
lean_dec(v___x_1817_);
v___x_1820_ = lean_box(0);
v_isShared_1821_ = v_isSharedCheck_1825_;
goto v_resetjp_1819_;
}
v_resetjp_1819_:
{
lean_object* v___x_1823_; 
if (v_isShared_1821_ == 0)
{
v___x_1823_ = v___x_1820_;
goto v_reusejp_1822_;
}
else
{
lean_object* v_reuseFailAlloc_1824_; 
v_reuseFailAlloc_1824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1824_, 0, v_a_1818_);
v___x_1823_ = v_reuseFailAlloc_1824_;
goto v_reusejp_1822_;
}
v_reusejp_1822_:
{
return v___x_1823_;
}
}
}
}
}
}
else
{
lean_object* v_a_1827_; lean_object* v___x_1829_; uint8_t v_isShared_1830_; uint8_t v_isSharedCheck_1834_; 
lean_del_object(v___x_1805_);
lean_dec(v_val_1803_);
lean_dec(v___y_1799_);
lean_dec(v___y_1798_);
lean_dec(v___y_1797_);
lean_dec_ref(v_tacticState_1640_);
v_a_1827_ = lean_ctor_get(v___x_1807_, 0);
v_isSharedCheck_1834_ = !lean_is_exclusive(v___x_1807_);
if (v_isSharedCheck_1834_ == 0)
{
v___x_1829_ = v___x_1807_;
v_isShared_1830_ = v_isSharedCheck_1834_;
goto v_resetjp_1828_;
}
else
{
lean_inc(v_a_1827_);
lean_dec(v___x_1807_);
v___x_1829_ = lean_box(0);
v_isShared_1830_ = v_isSharedCheck_1834_;
goto v_resetjp_1828_;
}
v_resetjp_1828_:
{
lean_object* v___x_1832_; 
if (v_isShared_1830_ == 0)
{
v___x_1832_ = v___x_1829_;
goto v_reusejp_1831_;
}
else
{
lean_object* v_reuseFailAlloc_1833_; 
v_reuseFailAlloc_1833_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1833_, 0, v_a_1827_);
v___x_1832_ = v_reuseFailAlloc_1833_;
goto v_reusejp_1831_;
}
v_reusejp_1831_:
{
return v___x_1832_;
}
}
}
}
}
else
{
lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1839_; 
lean_dec(v___x_1802_);
lean_dec(v___y_1799_);
lean_dec(v___y_1798_);
lean_dec(v___y_1797_);
v___x_1836_ = lean_box(0);
v___x_1837_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1837_, 0, v___x_1836_);
lean_ctor_set(v___x_1837_, 1, v_tacticState_1640_);
if (v_isShared_1719_ == 0)
{
lean_ctor_set(v___x_1718_, 0, v___x_1837_);
v___x_1839_ = v___x_1718_;
goto v_reusejp_1838_;
}
else
{
lean_object* v_reuseFailAlloc_1840_; 
v_reuseFailAlloc_1840_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1840_, 0, v___x_1837_);
v___x_1839_ = v_reuseFailAlloc_1840_;
goto v_reusejp_1838_;
}
v_reusejp_1838_:
{
return v___x_1839_;
}
}
}
v___jp_1841_:
{
lean_object* v___x_1847_; 
v___x_1847_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_1714_, v___y_1845_);
if (lean_obj_tag(v___x_1847_) == 0)
{
lean_object* v_a_1848_; uint8_t v___x_1849_; 
v_a_1848_ = lean_ctor_get(v___x_1847_, 0);
lean_inc(v_a_1848_);
lean_dec_ref_known(v___x_1847_, 1);
v___x_1849_ = lean_unbox(v_a_1848_);
lean_dec(v_a_1848_);
if (v___x_1849_ == 0)
{
v___y_1797_ = v___y_1842_;
v___y_1798_ = v___y_1843_;
v___y_1799_ = v___y_1844_;
v___y_1800_ = v___y_1845_;
v___y_1801_ = v___y_1846_;
goto v___jp_1796_;
}
else
{
lean_object* v_traceClass_1850_; lean_object* v_visibleGoals_1851_; lean_object* v___x_1852_; size_t v_sz_1853_; size_t v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; 
v_traceClass_1850_ = lean_ctor_get(v___x_1714_, 0);
v_visibleGoals_1851_ = lean_ctor_get(v_tacticState_1640_, 0);
v___x_1852_ = lean_obj_once(&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__4, &lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__4_once, _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__4);
v_sz_1853_ = lean_array_size(v_visibleGoals_1851_);
v___x_1854_ = ((size_t)0ULL);
lean_inc_ref(v_visibleGoals_1851_);
v___x_1855_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0(v_sz_1853_, v___x_1854_, v_visibleGoals_1851_);
v___x_1856_ = lean_array_to_list(v___x_1855_);
v___x_1857_ = lean_box(0);
v___x_1858_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__1(v___x_1856_, v___x_1857_);
v___x_1859_ = l_Lean_MessageData_ofList(v___x_1858_);
v___x_1860_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1860_, 0, v___x_1852_);
lean_ctor_set(v___x_1860_, 1, v___x_1859_);
lean_inc(v_traceClass_1850_);
v___x_1861_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_1850_, v___x_1860_, v___y_1845_, v___y_1846_);
if (lean_obj_tag(v___x_1861_) == 0)
{
lean_dec_ref_known(v___x_1861_, 1);
v___y_1797_ = v___y_1842_;
v___y_1798_ = v___y_1843_;
v___y_1799_ = v___y_1844_;
v___y_1800_ = v___y_1845_;
v___y_1801_ = v___y_1846_;
goto v___jp_1796_;
}
else
{
lean_object* v_a_1862_; lean_object* v___x_1864_; uint8_t v_isShared_1865_; uint8_t v_isSharedCheck_1869_; 
lean_dec(v___y_1844_);
lean_dec(v___y_1843_);
lean_dec(v___y_1842_);
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v_a_1862_ = lean_ctor_get(v___x_1861_, 0);
v_isSharedCheck_1869_ = !lean_is_exclusive(v___x_1861_);
if (v_isSharedCheck_1869_ == 0)
{
v___x_1864_ = v___x_1861_;
v_isShared_1865_ = v_isSharedCheck_1869_;
goto v_resetjp_1863_;
}
else
{
lean_inc(v_a_1862_);
lean_dec(v___x_1861_);
v___x_1864_ = lean_box(0);
v_isShared_1865_ = v_isSharedCheck_1869_;
goto v_resetjp_1863_;
}
v_resetjp_1863_:
{
lean_object* v___x_1867_; 
if (v_isShared_1865_ == 0)
{
v___x_1867_ = v___x_1864_;
goto v_reusejp_1866_;
}
else
{
lean_object* v_reuseFailAlloc_1868_; 
v_reuseFailAlloc_1868_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1868_, 0, v_a_1862_);
v___x_1867_ = v_reuseFailAlloc_1868_;
goto v_reusejp_1866_;
}
v_reusejp_1866_:
{
return v___x_1867_;
}
}
}
}
}
else
{
lean_object* v_a_1870_; lean_object* v___x_1872_; uint8_t v_isShared_1873_; uint8_t v_isSharedCheck_1877_; 
lean_dec(v___y_1844_);
lean_dec(v___y_1843_);
lean_dec(v___y_1842_);
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v_a_1870_ = lean_ctor_get(v___x_1847_, 0);
v_isSharedCheck_1877_ = !lean_is_exclusive(v___x_1847_);
if (v_isSharedCheck_1877_ == 0)
{
v___x_1872_ = v___x_1847_;
v_isShared_1873_ = v_isSharedCheck_1877_;
goto v_resetjp_1871_;
}
else
{
lean_inc(v_a_1870_);
lean_dec(v___x_1847_);
v___x_1872_ = lean_box(0);
v_isShared_1873_ = v_isSharedCheck_1877_;
goto v_resetjp_1871_;
}
v_resetjp_1871_:
{
lean_object* v___x_1875_; 
if (v_isShared_1873_ == 0)
{
v___x_1875_ = v___x_1872_;
goto v_reusejp_1874_;
}
else
{
lean_object* v_reuseFailAlloc_1876_; 
v_reuseFailAlloc_1876_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1876_, 0, v_a_1870_);
v___x_1875_ = v_reuseFailAlloc_1876_;
goto v_reusejp_1874_;
}
v_reusejp_1874_:
{
return v___x_1875_;
}
}
}
}
v___jp_1878_:
{
lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; 
lean_inc_ref(v___y_1886_);
v___x_1887_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1887_, 0, v___y_1886_);
v___x_1888_ = l_Lean_MessageData_ofFormat(v___x_1887_);
lean_inc_ref(v___y_1882_);
v___x_1889_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1889_, 0, v___y_1882_);
lean_ctor_set(v___x_1889_, 1, v___x_1888_);
v___x_1890_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v___y_1885_, v___x_1889_, v___y_1884_, v___y_1880_);
if (lean_obj_tag(v___x_1890_) == 0)
{
lean_dec_ref_known(v___x_1890_, 1);
v___y_1842_ = v___y_1879_;
v___y_1843_ = v___y_1881_;
v___y_1844_ = v___y_1883_;
v___y_1845_ = v___y_1884_;
v___y_1846_ = v___y_1880_;
goto v___jp_1841_;
}
else
{
lean_object* v_a_1891_; lean_object* v___x_1893_; uint8_t v_isShared_1894_; uint8_t v_isSharedCheck_1898_; 
lean_dec(v___y_1883_);
lean_dec(v___y_1881_);
lean_dec(v___y_1879_);
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v_a_1891_ = lean_ctor_get(v___x_1890_, 0);
v_isSharedCheck_1898_ = !lean_is_exclusive(v___x_1890_);
if (v_isSharedCheck_1898_ == 0)
{
v___x_1893_ = v___x_1890_;
v_isShared_1894_ = v_isSharedCheck_1898_;
goto v_resetjp_1892_;
}
else
{
lean_inc(v_a_1891_);
lean_dec(v___x_1890_);
v___x_1893_ = lean_box(0);
v_isShared_1894_ = v_isSharedCheck_1898_;
goto v_resetjp_1892_;
}
v_resetjp_1892_:
{
lean_object* v___x_1896_; 
if (v_isShared_1894_ == 0)
{
v___x_1896_ = v___x_1893_;
goto v_reusejp_1895_;
}
else
{
lean_object* v_reuseFailAlloc_1897_; 
v_reuseFailAlloc_1897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1897_, 0, v_a_1891_);
v___x_1896_ = v_reuseFailAlloc_1897_;
goto v_reusejp_1895_;
}
v_reusejp_1895_:
{
return v___x_1896_;
}
}
}
}
v___jp_1899_:
{
lean_object* v___x_1904_; 
v___x_1904_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_1714_, v___y_1902_);
if (lean_obj_tag(v___x_1904_) == 0)
{
lean_object* v_a_1905_; lean_object* v___x_1906_; uint8_t v___x_1907_; 
v_a_1905_ = lean_ctor_get(v___x_1904_, 0);
lean_inc(v_a_1905_);
lean_dec_ref_known(v___x_1904_, 1);
v___x_1906_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg(v_focusable_1636_, v___y_1901_);
v___x_1907_ = lean_unbox(v_a_1905_);
lean_dec(v_a_1905_);
if (v___x_1907_ == 0)
{
v___y_1842_ = v___x_1906_;
v___y_1843_ = v___y_1900_;
v___y_1844_ = v___y_1901_;
v___y_1845_ = v___y_1902_;
v___y_1846_ = v___y_1903_;
goto v___jp_1841_;
}
else
{
lean_object* v_traceClass_1908_; lean_object* v___x_1909_; 
v_traceClass_1908_ = lean_ctor_get(v___x_1714_, 0);
v___x_1909_ = lean_obj_once(&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__6, &lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__6_once, _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__6);
if (lean_obj_tag(v___x_1906_) == 0)
{
lean_object* v___x_1910_; 
v___x_1910_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__7));
lean_inc(v_traceClass_1908_);
v___y_1879_ = v___x_1906_;
v___y_1880_ = v___y_1903_;
v___y_1881_ = v___y_1900_;
v___y_1882_ = v___x_1909_;
v___y_1883_ = v___y_1901_;
v___y_1884_ = v___y_1902_;
v___y_1885_ = v_traceClass_1908_;
v___y_1886_ = v___x_1910_;
goto v___jp_1878_;
}
else
{
lean_object* v___x_1911_; 
v___x_1911_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__8));
lean_inc(v_traceClass_1908_);
v___y_1879_ = v___x_1906_;
v___y_1880_ = v___y_1903_;
v___y_1881_ = v___y_1900_;
v___y_1882_ = v___x_1909_;
v___y_1883_ = v___y_1901_;
v___y_1884_ = v___y_1902_;
v___y_1885_ = v_traceClass_1908_;
v___y_1886_ = v___x_1911_;
goto v___jp_1878_;
}
}
}
else
{
lean_object* v_a_1912_; lean_object* v___x_1914_; uint8_t v_isShared_1915_; uint8_t v_isSharedCheck_1919_; 
lean_dec(v___y_1901_);
lean_dec(v___y_1900_);
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v_a_1912_ = lean_ctor_get(v___x_1904_, 0);
v_isSharedCheck_1919_ = !lean_is_exclusive(v___x_1904_);
if (v_isSharedCheck_1919_ == 0)
{
v___x_1914_ = v___x_1904_;
v_isShared_1915_ = v_isSharedCheck_1919_;
goto v_resetjp_1913_;
}
else
{
lean_inc(v_a_1912_);
lean_dec(v___x_1904_);
v___x_1914_ = lean_box(0);
v_isShared_1915_ = v_isSharedCheck_1919_;
goto v_resetjp_1913_;
}
v_resetjp_1913_:
{
lean_object* v___x_1917_; 
if (v_isShared_1915_ == 0)
{
v___x_1917_ = v___x_1914_;
goto v_reusejp_1916_;
}
else
{
lean_object* v_reuseFailAlloc_1918_; 
v_reuseFailAlloc_1918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1918_, 0, v_a_1912_);
v___x_1917_ = v_reuseFailAlloc_1918_;
goto v_reusejp_1916_;
}
v_reusejp_1916_:
{
return v___x_1917_;
}
}
}
}
v___jp_1920_:
{
lean_object* v_preGoal_1923_; lean_object* v___x_1924_; 
v_preGoal_1923_ = lean_ctor_get(v___x_1720_, 1);
v___x_1924_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_UScript_toStepTree_go_spec__0___redArg(v_numSiblings_1637_, v_preGoal_1923_);
if (lean_obj_tag(v___x_1924_) == 1)
{
lean_object* v_val_1925_; lean_object* v___x_1927_; uint8_t v_isShared_1928_; uint8_t v_isSharedCheck_1957_; 
v_val_1925_ = lean_ctor_get(v___x_1924_, 0);
v_isSharedCheck_1957_ = !lean_is_exclusive(v___x_1924_);
if (v_isSharedCheck_1957_ == 0)
{
v___x_1927_ = v___x_1924_;
v_isShared_1928_ = v_isSharedCheck_1957_;
goto v_resetjp_1926_;
}
else
{
lean_inc(v_val_1925_);
lean_dec(v___x_1924_);
v___x_1927_ = lean_box(0);
v_isShared_1928_ = v_isSharedCheck_1957_;
goto v_resetjp_1926_;
}
v_resetjp_1926_:
{
lean_object* v___x_1929_; 
v___x_1929_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_1714_, v___y_1921_);
if (lean_obj_tag(v___x_1929_) == 0)
{
lean_object* v_a_1930_; uint8_t v___x_1931_; 
v_a_1930_ = lean_ctor_get(v___x_1929_, 0);
lean_inc(v_a_1930_);
lean_dec_ref_known(v___x_1929_, 1);
v___x_1931_ = lean_unbox(v_a_1930_);
lean_dec(v_a_1930_);
if (v___x_1931_ == 0)
{
lean_del_object(v___x_1927_);
lean_inc(v_preGoal_1923_);
v___y_1900_ = v_val_1925_;
v___y_1901_ = v_preGoal_1923_;
v___y_1902_ = v___y_1921_;
v___y_1903_ = v___y_1922_;
goto v___jp_1899_;
}
else
{
lean_object* v_traceClass_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1936_; 
v_traceClass_1932_ = lean_ctor_get(v___x_1714_, 0);
v___x_1933_ = lean_obj_once(&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__10, &lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__10_once, _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__10);
lean_inc(v_val_1925_);
v___x_1934_ = l_Nat_reprFast(v_val_1925_);
if (v_isShared_1928_ == 0)
{
lean_ctor_set_tag(v___x_1927_, 3);
lean_ctor_set(v___x_1927_, 0, v___x_1934_);
v___x_1936_ = v___x_1927_;
goto v_reusejp_1935_;
}
else
{
lean_object* v_reuseFailAlloc_1948_; 
v_reuseFailAlloc_1948_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1948_, 0, v___x_1934_);
v___x_1936_ = v_reuseFailAlloc_1948_;
goto v_reusejp_1935_;
}
v_reusejp_1935_:
{
lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; 
v___x_1937_ = l_Lean_MessageData_ofFormat(v___x_1936_);
v___x_1938_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1938_, 0, v___x_1933_);
lean_ctor_set(v___x_1938_, 1, v___x_1937_);
lean_inc(v_traceClass_1932_);
v___x_1939_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_1932_, v___x_1938_, v___y_1921_, v___y_1922_);
if (lean_obj_tag(v___x_1939_) == 0)
{
lean_dec_ref_known(v___x_1939_, 1);
lean_inc(v_preGoal_1923_);
v___y_1900_ = v_val_1925_;
v___y_1901_ = v_preGoal_1923_;
v___y_1902_ = v___y_1921_;
v___y_1903_ = v___y_1922_;
goto v___jp_1899_;
}
else
{
lean_object* v_a_1940_; lean_object* v___x_1942_; uint8_t v_isShared_1943_; uint8_t v_isSharedCheck_1947_; 
lean_dec(v_val_1925_);
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v_a_1940_ = lean_ctor_get(v___x_1939_, 0);
v_isSharedCheck_1947_ = !lean_is_exclusive(v___x_1939_);
if (v_isSharedCheck_1947_ == 0)
{
v___x_1942_ = v___x_1939_;
v_isShared_1943_ = v_isSharedCheck_1947_;
goto v_resetjp_1941_;
}
else
{
lean_inc(v_a_1940_);
lean_dec(v___x_1939_);
v___x_1942_ = lean_box(0);
v_isShared_1943_ = v_isSharedCheck_1947_;
goto v_resetjp_1941_;
}
v_resetjp_1941_:
{
lean_object* v___x_1945_; 
if (v_isShared_1943_ == 0)
{
v___x_1945_ = v___x_1942_;
goto v_reusejp_1944_;
}
else
{
lean_object* v_reuseFailAlloc_1946_; 
v_reuseFailAlloc_1946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1946_, 0, v_a_1940_);
v___x_1945_ = v_reuseFailAlloc_1946_;
goto v_reusejp_1944_;
}
v_reusejp_1944_:
{
return v___x_1945_;
}
}
}
}
}
}
else
{
lean_object* v_a_1949_; lean_object* v___x_1951_; uint8_t v_isShared_1952_; uint8_t v_isSharedCheck_1956_; 
lean_del_object(v___x_1927_);
lean_dec(v_val_1925_);
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v_a_1949_ = lean_ctor_get(v___x_1929_, 0);
v_isSharedCheck_1956_ = !lean_is_exclusive(v___x_1929_);
if (v_isSharedCheck_1956_ == 0)
{
v___x_1951_ = v___x_1929_;
v_isShared_1952_ = v_isSharedCheck_1956_;
goto v_resetjp_1950_;
}
else
{
lean_inc(v_a_1949_);
lean_dec(v___x_1929_);
v___x_1951_ = lean_box(0);
v_isShared_1952_ = v_isSharedCheck_1956_;
goto v_resetjp_1950_;
}
v_resetjp_1950_:
{
lean_object* v___x_1954_; 
if (v_isShared_1952_ == 0)
{
v___x_1954_ = v___x_1951_;
goto v_reusejp_1953_;
}
else
{
lean_object* v_reuseFailAlloc_1955_; 
v_reuseFailAlloc_1955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1955_, 0, v_a_1949_);
v___x_1954_ = v_reuseFailAlloc_1955_;
goto v_reusejp_1953_;
}
v_reusejp_1953_:
{
return v___x_1954_;
}
}
}
}
}
else
{
lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; 
lean_dec(v___x_1924_);
lean_del_object(v___x_1718_);
lean_dec_ref(v_tacticState_1640_);
v___x_1958_ = lean_obj_once(&lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__12, &lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__12_once, _init_lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__12);
lean_inc(v_preGoal_1923_);
v___x_1959_ = l_Lean_MessageData_ofName(v_preGoal_1923_);
v___x_1960_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1960_, 0, v___x_1958_);
lean_ctor_set(v___x_1960_, 1, v___x_1959_);
v___x_1961_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg(v___x_1960_, v___y_1921_, v___y_1922_);
return v___x_1961_;
}
}
}
}
else
{
lean_object* v_a_2004_; lean_object* v___x_2006_; uint8_t v_isShared_2007_; uint8_t v_isSharedCheck_2011_; 
lean_dec_ref(v_tacticState_1640_);
v_a_2004_ = lean_ctor_get(v___x_1715_, 0);
v_isSharedCheck_2011_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_2011_ == 0)
{
v___x_2006_ = v___x_1715_;
v_isShared_2007_ = v_isSharedCheck_2011_;
goto v_resetjp_2005_;
}
else
{
lean_inc(v_a_2004_);
lean_dec(v___x_1715_);
v___x_2006_ = lean_box(0);
v_isShared_2007_ = v_isSharedCheck_2011_;
goto v_resetjp_2005_;
}
v_resetjp_2005_:
{
lean_object* v___x_2009_; 
if (v_isShared_2007_ == 0)
{
v___x_2009_ = v___x_2006_;
goto v_reusejp_2008_;
}
else
{
lean_object* v_reuseFailAlloc_2010_; 
v_reuseFailAlloc_2010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2010_, 0, v_a_2004_);
v___x_2009_ = v_reuseFailAlloc_2010_;
goto v_reusejp_2008_;
}
v_reusejp_2008_:
{
return v___x_2009_;
}
}
}
}
}
else
{
lean_object* v___x_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; 
v___x_2012_ = lean_box(0);
v___x_2013_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2013_, 0, v___x_2012_);
lean_ctor_set(v___x_2013_, 1, v_tacticState_1640_);
v___x_2014_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2014_, 0, v___x_2013_);
return v___x_2014_;
}
v___jp_1644_:
{
lean_object* v_visibleGoals_1652_; lean_object* v_invisibleGoals_1653_; lean_object* v___x_1655_; uint8_t v_isShared_1656_; uint8_t v_isSharedCheck_1707_; 
v_visibleGoals_1652_ = lean_ctor_get(v_tacticState_1640_, 0);
v_invisibleGoals_1653_ = lean_ctor_get(v_tacticState_1640_, 1);
v_isSharedCheck_1707_ = !lean_is_exclusive(v_tacticState_1640_);
if (v_isSharedCheck_1707_ == 0)
{
v___x_1655_ = v_tacticState_1640_;
v_isShared_1656_ = v_isSharedCheck_1707_;
goto v_resetjp_1654_;
}
else
{
lean_inc(v_invisibleGoals_1653_);
lean_inc(v_visibleGoals_1652_);
lean_dec(v_tacticState_1640_);
v___x_1655_ = lean_box(0);
v_isShared_1656_ = v_isSharedCheck_1707_;
goto v_resetjp_1654_;
}
v_resetjp_1654_:
{
lean_object* v_visibleGoals_1657_; size_t v_sz_1658_; size_t v___x_1659_; lean_object* v___x_1660_; 
v_visibleGoals_1657_ = ((lean_object*)(lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___closed__0));
v_sz_1658_ = lean_array_size(v_visibleGoals_1652_);
v___x_1659_ = ((size_t)0ULL);
v___x_1660_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___redArg(v___y_1648_, v_snd_1651_, v_visibleGoals_1652_, v_sz_1658_, v___x_1659_, v_visibleGoals_1657_);
lean_dec_ref(v_visibleGoals_1652_);
lean_dec(v___y_1648_);
if (lean_obj_tag(v___x_1660_) == 0)
{
lean_object* v_a_1661_; lean_object* v_buckets_1662_; lean_object* v_invisibleGoals_1663_; size_t v_sz_1664_; lean_object* v___x_1665_; 
v_a_1661_ = lean_ctor_get(v___x_1660_, 0);
lean_inc(v_a_1661_);
lean_dec_ref_known(v___x_1660_, 1);
v_buckets_1662_ = lean_ctor_get(v_invisibleGoals_1653_, 1);
lean_inc_ref(v_buckets_1662_);
lean_dec_ref(v_invisibleGoals_1653_);
v_invisibleGoals_1663_ = lean_obj_once(&lp_aesop_Aesop_Script_UScript_toStepTree___closed__1, &lp_aesop_Aesop_Script_UScript_toStepTree___closed__1_once, _init_lp_aesop_Aesop_Script_UScript_toStepTree___closed__1);
v_sz_1664_ = lean_array_size(v_buckets_1662_);
v___x_1665_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__5(v_snd_1651_, v_buckets_1662_, v_sz_1664_, v___x_1659_, v_invisibleGoals_1663_, v___y_1647_, v___y_1645_);
lean_dec_ref(v_buckets_1662_);
lean_dec_ref(v_snd_1651_);
if (lean_obj_tag(v___x_1665_) == 0)
{
lean_object* v_a_1666_; lean_object* v___x_1668_; 
v_a_1666_ = lean_ctor_get(v___x_1665_, 0);
lean_inc(v_a_1666_);
lean_dec_ref_known(v___x_1665_, 1);
if (v_isShared_1656_ == 0)
{
lean_ctor_set(v___x_1655_, 1, v_a_1666_);
lean_ctor_set(v___x_1655_, 0, v_a_1661_);
v___x_1668_ = v___x_1655_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1690_; 
v_reuseFailAlloc_1690_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1690_, 0, v_a_1661_);
lean_ctor_set(v_reuseFailAlloc_1690_, 1, v_a_1666_);
v___x_1668_ = v_reuseFailAlloc_1690_;
goto v_reusejp_1667_;
}
v_reusejp_1667_:
{
lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; 
v___x_1669_ = lean_unsigned_to_nat(1u);
v___x_1670_ = lean_nat_add(v___y_1649_, v___x_1669_);
lean_dec(v___y_1649_);
v___x_1671_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_1635_, v_focusable_1636_, v_numSiblings_1637_, v___x_1670_, v_stop_1639_, v___x_1668_, v___y_1647_, v___y_1645_);
lean_dec(v___x_1670_);
if (lean_obj_tag(v___x_1671_) == 0)
{
lean_object* v_a_1672_; lean_object* v___x_1674_; uint8_t v_isShared_1675_; uint8_t v_isSharedCheck_1689_; 
v_a_1672_ = lean_ctor_get(v___x_1671_, 0);
v_isSharedCheck_1689_ = !lean_is_exclusive(v___x_1671_);
if (v_isSharedCheck_1689_ == 0)
{
v___x_1674_ = v___x_1671_;
v_isShared_1675_ = v_isSharedCheck_1689_;
goto v_resetjp_1673_;
}
else
{
lean_inc(v_a_1672_);
lean_dec(v___x_1671_);
v___x_1674_ = lean_box(0);
v_isShared_1675_ = v_isSharedCheck_1689_;
goto v_resetjp_1673_;
}
v_resetjp_1673_:
{
lean_object* v_fst_1676_; lean_object* v_snd_1677_; lean_object* v___x_1679_; uint8_t v_isShared_1680_; uint8_t v_isSharedCheck_1688_; 
v_fst_1676_ = lean_ctor_get(v_a_1672_, 0);
v_snd_1677_ = lean_ctor_get(v_a_1672_, 1);
v_isSharedCheck_1688_ = !lean_is_exclusive(v_a_1672_);
if (v_isSharedCheck_1688_ == 0)
{
v___x_1679_ = v_a_1672_;
v_isShared_1680_ = v_isSharedCheck_1688_;
goto v_resetjp_1678_;
}
else
{
lean_inc(v_snd_1677_);
lean_inc(v_fst_1676_);
lean_dec(v_a_1672_);
v___x_1679_ = lean_box(0);
v_isShared_1680_ = v_isSharedCheck_1688_;
goto v_resetjp_1678_;
}
v_resetjp_1678_:
{
lean_object* v___x_1681_; lean_object* v___x_1683_; 
v___x_1681_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1681_, 0, v___y_1646_);
lean_ctor_set(v___x_1681_, 1, v_fst_1650_);
lean_ctor_set(v___x_1681_, 2, v_fst_1676_);
if (v_isShared_1680_ == 0)
{
lean_ctor_set(v___x_1679_, 0, v___x_1681_);
v___x_1683_ = v___x_1679_;
goto v_reusejp_1682_;
}
else
{
lean_object* v_reuseFailAlloc_1687_; 
v_reuseFailAlloc_1687_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1687_, 0, v___x_1681_);
lean_ctor_set(v_reuseFailAlloc_1687_, 1, v_snd_1677_);
v___x_1683_ = v_reuseFailAlloc_1687_;
goto v_reusejp_1682_;
}
v_reusejp_1682_:
{
lean_object* v___x_1685_; 
if (v_isShared_1675_ == 0)
{
lean_ctor_set(v___x_1674_, 0, v___x_1683_);
v___x_1685_ = v___x_1674_;
goto v_reusejp_1684_;
}
else
{
lean_object* v_reuseFailAlloc_1686_; 
v_reuseFailAlloc_1686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1686_, 0, v___x_1683_);
v___x_1685_ = v_reuseFailAlloc_1686_;
goto v_reusejp_1684_;
}
v_reusejp_1684_:
{
return v___x_1685_;
}
}
}
}
}
else
{
lean_dec(v_fst_1650_);
lean_dec(v___y_1646_);
return v___x_1671_;
}
}
}
else
{
lean_object* v_a_1691_; lean_object* v___x_1693_; uint8_t v_isShared_1694_; uint8_t v_isSharedCheck_1698_; 
lean_dec(v_a_1661_);
lean_del_object(v___x_1655_);
lean_dec(v_fst_1650_);
lean_dec(v___y_1649_);
lean_dec(v___y_1646_);
v_a_1691_ = lean_ctor_get(v___x_1665_, 0);
v_isSharedCheck_1698_ = !lean_is_exclusive(v___x_1665_);
if (v_isSharedCheck_1698_ == 0)
{
v___x_1693_ = v___x_1665_;
v_isShared_1694_ = v_isSharedCheck_1698_;
goto v_resetjp_1692_;
}
else
{
lean_inc(v_a_1691_);
lean_dec(v___x_1665_);
v___x_1693_ = lean_box(0);
v_isShared_1694_ = v_isSharedCheck_1698_;
goto v_resetjp_1692_;
}
v_resetjp_1692_:
{
lean_object* v___x_1696_; 
if (v_isShared_1694_ == 0)
{
v___x_1696_ = v___x_1693_;
goto v_reusejp_1695_;
}
else
{
lean_object* v_reuseFailAlloc_1697_; 
v_reuseFailAlloc_1697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1697_, 0, v_a_1691_);
v___x_1696_ = v_reuseFailAlloc_1697_;
goto v_reusejp_1695_;
}
v_reusejp_1695_:
{
return v___x_1696_;
}
}
}
}
else
{
lean_object* v_a_1699_; lean_object* v___x_1701_; uint8_t v_isShared_1702_; uint8_t v_isSharedCheck_1706_; 
lean_del_object(v___x_1655_);
lean_dec_ref(v_invisibleGoals_1653_);
lean_dec_ref(v_snd_1651_);
lean_dec(v_fst_1650_);
lean_dec(v___y_1649_);
lean_dec(v___y_1646_);
v_a_1699_ = lean_ctor_get(v___x_1660_, 0);
v_isSharedCheck_1706_ = !lean_is_exclusive(v___x_1660_);
if (v_isSharedCheck_1706_ == 0)
{
v___x_1701_ = v___x_1660_;
v_isShared_1702_ = v_isSharedCheck_1706_;
goto v_resetjp_1700_;
}
else
{
lean_inc(v_a_1699_);
lean_dec(v___x_1660_);
v___x_1701_ = lean_box(0);
v_isShared_1702_ = v_isSharedCheck_1706_;
goto v_resetjp_1700_;
}
v_resetjp_1700_:
{
lean_object* v___x_1704_; 
if (v_isShared_1702_ == 0)
{
v___x_1704_ = v___x_1701_;
goto v_reusejp_1703_;
}
else
{
lean_object* v_reuseFailAlloc_1705_; 
v_reuseFailAlloc_1705_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1705_, 0, v_a_1699_);
v___x_1704_ = v_reuseFailAlloc_1705_;
goto v_reusejp_1703_;
}
v_reusejp_1703_:
{
return v___x_1704_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go___boxed(lean_object* v_uscript_2015_, lean_object* v_focusable_2016_, lean_object* v_numSiblings_2017_, lean_object* v_start_2018_, lean_object* v_stop_2019_, lean_object* v_tacticState_2020_, lean_object* v_a_2021_, lean_object* v_a_2022_, lean_object* v_a_2023_){
_start:
{
lean_object* v_res_2024_; 
v_res_2024_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_2015_, v_focusable_2016_, v_numSiblings_2017_, v_start_2018_, v_stop_2019_, v_tacticState_2020_, v_a_2021_, v_a_2022_);
lean_dec(v_a_2022_);
lean_dec_ref(v_a_2021_);
lean_dec(v_stop_2019_);
lean_dec(v_start_2018_);
lean_dec_ref(v_numSiblings_2017_);
lean_dec_ref(v_focusable_2016_);
lean_dec_ref(v_uscript_2015_);
return v_res_2024_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0(lean_object* v_opt_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_){
_start:
{
lean_object* v___x_2029_; 
v___x_2029_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v_opt_2025_, v___y_2026_);
return v___x_2029_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___boxed(lean_object* v_opt_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_){
_start:
{
lean_object* v_res_2034_; 
v_res_2034_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0(v_opt_2030_, v___y_2031_, v___y_2032_);
lean_dec(v___y_2032_);
lean_dec_ref(v___y_2031_);
lean_dec_ref(v_opt_2030_);
return v_res_2034_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1(lean_object* v_00_u03b2_2035_, lean_object* v_m_2036_, lean_object* v_a_2037_){
_start:
{
uint8_t v___x_2038_; 
v___x_2038_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___redArg(v_m_2036_, v_a_2037_);
return v___x_2038_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1___boxed(lean_object* v_00_u03b2_2039_, lean_object* v_m_2040_, lean_object* v_a_2041_){
_start:
{
uint8_t v_res_2042_; lean_object* v_r_2043_; 
v_res_2042_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__1(v_00_u03b2_2039_, v_m_2040_, v_a_2041_);
lean_dec(v_a_2041_);
lean_dec_ref(v_m_2040_);
v_r_2043_ = lean_box(v_res_2042_);
return v_r_2043_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__2(lean_object* v_00_u03b2_2044_, lean_object* v_m_2045_, lean_object* v_a_2046_, lean_object* v_b_2047_){
_start:
{
lean_object* v___x_2048_; 
v___x_2048_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__2___redArg(v_m_2045_, v_a_2046_, v_b_2047_);
return v___x_2048_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3(lean_object* v_snd_2049_, lean_object* v_a_2050_, lean_object* v_a_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_){
_start:
{
lean_object* v___x_2055_; 
v___x_2055_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___redArg(v_snd_2049_, v_a_2050_, v_a_2051_);
return v___x_2055_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3___boxed(lean_object* v_snd_2056_, lean_object* v_a_2057_, lean_object* v_a_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_){
_start:
{
lean_object* v_res_2062_; 
v_res_2062_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__3(v_snd_2056_, v_a_2057_, v_a_2058_, v___y_2059_, v___y_2060_);
lean_dec(v___y_2060_);
lean_dec_ref(v___y_2059_);
lean_dec_ref(v_snd_2056_);
return v_res_2062_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4(lean_object* v___x_2063_, lean_object* v_snd_2064_, lean_object* v_as_2065_, size_t v_sz_2066_, size_t v_i_2067_, lean_object* v_b_2068_, lean_object* v___y_2069_, lean_object* v___y_2070_){
_start:
{
lean_object* v___x_2072_; 
v___x_2072_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___redArg(v___x_2063_, v_snd_2064_, v_as_2065_, v_sz_2066_, v_i_2067_, v_b_2068_);
return v___x_2072_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4___boxed(lean_object* v___x_2073_, lean_object* v_snd_2074_, lean_object* v_as_2075_, lean_object* v_sz_2076_, lean_object* v_i_2077_, lean_object* v_b_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_){
_start:
{
size_t v_sz_boxed_2082_; size_t v_i_boxed_2083_; lean_object* v_res_2084_; 
v_sz_boxed_2082_ = lean_unbox_usize(v_sz_2076_);
lean_dec(v_sz_2076_);
v_i_boxed_2083_ = lean_unbox_usize(v_i_2077_);
lean_dec(v_i_2077_);
v_res_2084_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__4(v___x_2073_, v_snd_2074_, v_as_2075_, v_sz_boxed_2082_, v_i_boxed_2083_, v_b_2078_, v___y_2079_, v___y_2080_);
lean_dec(v___y_2080_);
lean_dec_ref(v___y_2079_);
lean_dec_ref(v_as_2075_);
lean_dec_ref(v_snd_2074_);
lean_dec(v___x_2073_);
return v_res_2084_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10(lean_object* v_00_u03b1_2085_, lean_object* v_msg_2086_, lean_object* v___y_2087_, lean_object* v___y_2088_){
_start:
{
lean_object* v___x_2090_; 
v___x_2090_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___redArg(v_msg_2086_, v___y_2087_, v___y_2088_);
return v___x_2090_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10___boxed(lean_object* v_00_u03b1_2091_, lean_object* v_msg_2092_, lean_object* v___y_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_){
_start:
{
lean_object* v_res_2096_; 
v_res_2096_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__10(v_00_u03b1_2091_, v_msg_2092_, v___y_2093_, v___y_2094_);
lean_dec(v___y_2094_);
lean_dec_ref(v___y_2093_);
return v_res_2096_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7(lean_object* v_00_u03b1_2097_, lean_object* v_goal_2098_, lean_object* v_pre_2099_, lean_object* v___y_2100_, lean_object* v___y_2101_){
_start:
{
lean_object* v___x_2103_; 
v___x_2103_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___redArg(v_goal_2098_, v_pre_2099_, v___y_2100_, v___y_2101_);
return v___x_2103_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7___boxed(lean_object* v_00_u03b1_2104_, lean_object* v_goal_2105_, lean_object* v_pre_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_){
_start:
{
lean_object* v_res_2110_; 
v_res_2110_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__7(v_00_u03b1_2104_, v_goal_2105_, v_pre_2106_, v___y_2107_, v___y_2108_);
lean_dec(v___y_2108_);
lean_dec_ref(v___y_2107_);
return v_res_2110_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9(lean_object* v_goal_2111_, lean_object* v_as_2112_, size_t v_sz_2113_, size_t v_i_2114_, lean_object* v_b_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_){
_start:
{
lean_object* v___x_2119_; 
v___x_2119_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___redArg(v_goal_2111_, v_as_2112_, v_sz_2113_, v_i_2114_, v_b_2115_);
return v___x_2119_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9___boxed(lean_object* v_goal_2120_, lean_object* v_as_2121_, lean_object* v_sz_2122_, lean_object* v_i_2123_, lean_object* v_b_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_){
_start:
{
size_t v_sz_boxed_2128_; size_t v_i_boxed_2129_; lean_object* v_res_2130_; 
v_sz_boxed_2128_ = lean_unbox_usize(v_sz_2122_);
lean_dec(v_sz_2122_);
v_i_boxed_2129_ = lean_unbox_usize(v_i_2123_);
lean_dec(v_i_2123_);
v_res_2130_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticState_focus___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__6_spec__9(v_goal_2120_, v_as_2121_, v_sz_boxed_2128_, v_i_boxed_2129_, v_b_2124_, v___y_2125_, v___y_2126_);
lean_dec(v___y_2126_);
lean_dec_ref(v___y_2125_);
lean_dec_ref(v_as_2121_);
lean_dec(v_goal_2120_);
return v_res_2130_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; 
v___x_2131_ = lean_unsigned_to_nat(32u);
v___x_2132_ = lean_mk_empty_array_with_capacity(v___x_2131_);
v___x_2133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2133_, 0, v___x_2132_);
return v___x_2133_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__1(void){
_start:
{
size_t v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; 
v___x_2134_ = ((size_t)5ULL);
v___x_2135_ = lean_unsigned_to_nat(0u);
v___x_2136_ = lean_unsigned_to_nat(32u);
v___x_2137_ = lean_mk_empty_array_with_capacity(v___x_2136_);
v___x_2138_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__0);
v___x_2139_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2139_, 0, v___x_2138_);
lean_ctor_set(v___x_2139_, 1, v___x_2137_);
lean_ctor_set(v___x_2139_, 2, v___x_2135_);
lean_ctor_set(v___x_2139_, 3, v___x_2135_);
lean_ctor_set_usize(v___x_2139_, 4, v___x_2134_);
return v___x_2139_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg(lean_object* v___y_2140_){
_start:
{
lean_object* v___x_2142_; lean_object* v_traceState_2143_; lean_object* v_traces_2144_; lean_object* v___x_2145_; lean_object* v_traceState_2146_; lean_object* v_env_2147_; lean_object* v_nextMacroScope_2148_; lean_object* v_ngen_2149_; lean_object* v_auxDeclNGen_2150_; lean_object* v_cache_2151_; lean_object* v_messages_2152_; lean_object* v_infoState_2153_; lean_object* v_snapshotTasks_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2173_; 
v___x_2142_ = lean_st_ref_get(v___y_2140_);
v_traceState_2143_ = lean_ctor_get(v___x_2142_, 4);
lean_inc_ref(v_traceState_2143_);
lean_dec(v___x_2142_);
v_traces_2144_ = lean_ctor_get(v_traceState_2143_, 0);
lean_inc_ref(v_traces_2144_);
lean_dec_ref(v_traceState_2143_);
v___x_2145_ = lean_st_ref_take(v___y_2140_);
v_traceState_2146_ = lean_ctor_get(v___x_2145_, 4);
v_env_2147_ = lean_ctor_get(v___x_2145_, 0);
v_nextMacroScope_2148_ = lean_ctor_get(v___x_2145_, 1);
v_ngen_2149_ = lean_ctor_get(v___x_2145_, 2);
v_auxDeclNGen_2150_ = lean_ctor_get(v___x_2145_, 3);
v_cache_2151_ = lean_ctor_get(v___x_2145_, 5);
v_messages_2152_ = lean_ctor_get(v___x_2145_, 6);
v_infoState_2153_ = lean_ctor_get(v___x_2145_, 7);
v_snapshotTasks_2154_ = lean_ctor_get(v___x_2145_, 8);
v_isSharedCheck_2173_ = !lean_is_exclusive(v___x_2145_);
if (v_isSharedCheck_2173_ == 0)
{
v___x_2156_ = v___x_2145_;
v_isShared_2157_ = v_isSharedCheck_2173_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_snapshotTasks_2154_);
lean_inc(v_infoState_2153_);
lean_inc(v_messages_2152_);
lean_inc(v_cache_2151_);
lean_inc(v_traceState_2146_);
lean_inc(v_auxDeclNGen_2150_);
lean_inc(v_ngen_2149_);
lean_inc(v_nextMacroScope_2148_);
lean_inc(v_env_2147_);
lean_dec(v___x_2145_);
v___x_2156_ = lean_box(0);
v_isShared_2157_ = v_isSharedCheck_2173_;
goto v_resetjp_2155_;
}
v_resetjp_2155_:
{
uint64_t v_tid_2158_; lean_object* v___x_2160_; uint8_t v_isShared_2161_; uint8_t v_isSharedCheck_2171_; 
v_tid_2158_ = lean_ctor_get_uint64(v_traceState_2146_, sizeof(void*)*1);
v_isSharedCheck_2171_ = !lean_is_exclusive(v_traceState_2146_);
if (v_isSharedCheck_2171_ == 0)
{
lean_object* v_unused_2172_; 
v_unused_2172_ = lean_ctor_get(v_traceState_2146_, 0);
lean_dec(v_unused_2172_);
v___x_2160_ = v_traceState_2146_;
v_isShared_2161_ = v_isSharedCheck_2171_;
goto v_resetjp_2159_;
}
else
{
lean_dec(v_traceState_2146_);
v___x_2160_ = lean_box(0);
v_isShared_2161_ = v_isSharedCheck_2171_;
goto v_resetjp_2159_;
}
v_resetjp_2159_:
{
lean_object* v___x_2162_; lean_object* v___x_2164_; 
v___x_2162_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___closed__1);
if (v_isShared_2161_ == 0)
{
lean_ctor_set(v___x_2160_, 0, v___x_2162_);
v___x_2164_ = v___x_2160_;
goto v_reusejp_2163_;
}
else
{
lean_object* v_reuseFailAlloc_2170_; 
v_reuseFailAlloc_2170_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2170_, 0, v___x_2162_);
lean_ctor_set_uint64(v_reuseFailAlloc_2170_, sizeof(void*)*1, v_tid_2158_);
v___x_2164_ = v_reuseFailAlloc_2170_;
goto v_reusejp_2163_;
}
v_reusejp_2163_:
{
lean_object* v___x_2166_; 
if (v_isShared_2157_ == 0)
{
lean_ctor_set(v___x_2156_, 4, v___x_2164_);
v___x_2166_ = v___x_2156_;
goto v_reusejp_2165_;
}
else
{
lean_object* v_reuseFailAlloc_2169_; 
v_reuseFailAlloc_2169_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2169_, 0, v_env_2147_);
lean_ctor_set(v_reuseFailAlloc_2169_, 1, v_nextMacroScope_2148_);
lean_ctor_set(v_reuseFailAlloc_2169_, 2, v_ngen_2149_);
lean_ctor_set(v_reuseFailAlloc_2169_, 3, v_auxDeclNGen_2150_);
lean_ctor_set(v_reuseFailAlloc_2169_, 4, v___x_2164_);
lean_ctor_set(v_reuseFailAlloc_2169_, 5, v_cache_2151_);
lean_ctor_set(v_reuseFailAlloc_2169_, 6, v_messages_2152_);
lean_ctor_set(v_reuseFailAlloc_2169_, 7, v_infoState_2153_);
lean_ctor_set(v_reuseFailAlloc_2169_, 8, v_snapshotTasks_2154_);
v___x_2166_ = v_reuseFailAlloc_2169_;
goto v_reusejp_2165_;
}
v_reusejp_2165_:
{
lean_object* v___x_2167_; lean_object* v___x_2168_; 
v___x_2167_ = lean_st_ref_set(v___y_2140_, v___x_2166_);
v___x_2168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2168_, 0, v_traces_2144_);
return v___x_2168_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg___boxed(lean_object* v___y_2174_, lean_object* v___y_2175_){
_start:
{
lean_object* v_res_2176_; 
v_res_2176_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg(v___y_2174_);
lean_dec(v___y_2174_);
return v_res_2176_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5(lean_object* v___y_2177_, lean_object* v___y_2178_){
_start:
{
lean_object* v___x_2180_; 
v___x_2180_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg(v___y_2178_);
return v___x_2180_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___boxed(lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_){
_start:
{
lean_object* v_res_2184_; 
v_res_2184_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5(v___y_2181_, v___y_2182_);
lean_dec(v___y_2182_);
lean_dec_ref(v___y_2181_);
return v_res_2184_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2186_; lean_object* v___x_2187_; 
v___x_2186_ = ((lean_object*)(lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__0));
v___x_2187_ = l_Lean_stringToMessageData(v___x_2186_);
return v___x_2187_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0(lean_object* v_x_2188_, lean_object* v___y_2189_, lean_object* v___y_2190_){
_start:
{
lean_object* v___x_2192_; lean_object* v___x_2193_; 
v___x_2192_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___closed__1);
v___x_2193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2193_, 0, v___x_2192_);
return v___x_2193_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0___boxed(lean_object* v_x_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_){
_start:
{
lean_object* v_res_2198_; 
v_res_2198_ = lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__0(v_x_2194_, v___y_2195_, v___y_2196_);
lean_dec(v___y_2196_);
lean_dec_ref(v___y_2195_);
lean_dec_ref(v_x_2194_);
return v_res_2198_;
}
}
static lean_object* _init_lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__2(void){
_start:
{
lean_object* v___x_2202_; lean_object* v___x_2203_; 
v___x_2202_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__1));
v___x_2203_ = l_Lean_MessageData_ofFormat(v___x_2202_);
return v___x_2203_;
}
}
static lean_object* _init_lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__3(void){
_start:
{
lean_object* v___x_2204_; lean_object* v___x_2205_; 
v___x_2204_ = lean_box(1);
v___x_2205_ = l_Lean_MessageData_ofFormat(v___x_2204_);
return v___x_2205_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1(lean_object* v_a_2206_, lean_object* v_a_2207_){
_start:
{
if (lean_obj_tag(v_a_2206_) == 0)
{
lean_object* v___x_2208_; 
v___x_2208_ = l_List_reverse___redArg(v_a_2207_);
return v___x_2208_;
}
else
{
lean_object* v_head_2209_; lean_object* v_tail_2210_; lean_object* v___x_2212_; uint8_t v_isShared_2213_; uint8_t v_isSharedCheck_2236_; 
v_head_2209_ = lean_ctor_get(v_a_2206_, 0);
v_tail_2210_ = lean_ctor_get(v_a_2206_, 1);
v_isSharedCheck_2236_ = !lean_is_exclusive(v_a_2206_);
if (v_isSharedCheck_2236_ == 0)
{
v___x_2212_ = v_a_2206_;
v_isShared_2213_ = v_isSharedCheck_2236_;
goto v_resetjp_2211_;
}
else
{
lean_inc(v_tail_2210_);
lean_inc(v_head_2209_);
lean_dec(v_a_2206_);
v___x_2212_ = lean_box(0);
v_isShared_2213_ = v_isSharedCheck_2236_;
goto v_resetjp_2211_;
}
v_resetjp_2211_:
{
lean_object* v_fst_2214_; lean_object* v_snd_2215_; lean_object* v___x_2217_; uint8_t v_isShared_2218_; uint8_t v_isSharedCheck_2235_; 
v_fst_2214_ = lean_ctor_get(v_head_2209_, 0);
v_snd_2215_ = lean_ctor_get(v_head_2209_, 1);
v_isSharedCheck_2235_ = !lean_is_exclusive(v_head_2209_);
if (v_isSharedCheck_2235_ == 0)
{
v___x_2217_ = v_head_2209_;
v_isShared_2218_ = v_isSharedCheck_2235_;
goto v_resetjp_2216_;
}
else
{
lean_inc(v_snd_2215_);
lean_inc(v_fst_2214_);
lean_dec(v_head_2209_);
v___x_2217_ = lean_box(0);
v_isShared_2218_ = v_isSharedCheck_2235_;
goto v_resetjp_2216_;
}
v_resetjp_2216_:
{
lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2222_; 
v___x_2219_ = l_Lean_MessageData_ofName(v_fst_2214_);
v___x_2220_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__2, &lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__2_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__2);
if (v_isShared_2218_ == 0)
{
lean_ctor_set_tag(v___x_2217_, 7);
lean_ctor_set(v___x_2217_, 1, v___x_2220_);
lean_ctor_set(v___x_2217_, 0, v___x_2219_);
v___x_2222_ = v___x_2217_;
goto v_reusejp_2221_;
}
else
{
lean_object* v_reuseFailAlloc_2234_; 
v_reuseFailAlloc_2234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2234_, 0, v___x_2219_);
lean_ctor_set(v_reuseFailAlloc_2234_, 1, v___x_2220_);
v___x_2222_ = v_reuseFailAlloc_2234_;
goto v_reusejp_2221_;
}
v_reusejp_2221_:
{
lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2231_; 
v___x_2223_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__3, &lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__3_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1___closed__3);
v___x_2224_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2224_, 0, v___x_2222_);
lean_ctor_set(v___x_2224_, 1, v___x_2223_);
v___x_2225_ = l_Nat_reprFast(v_snd_2215_);
v___x_2226_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2226_, 0, v___x_2225_);
v___x_2227_ = l_Lean_MessageData_ofFormat(v___x_2226_);
v___x_2228_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2228_, 0, v___x_2224_);
lean_ctor_set(v___x_2228_, 1, v___x_2227_);
v___x_2229_ = l_Lean_MessageData_paren(v___x_2228_);
if (v_isShared_2213_ == 0)
{
lean_ctor_set(v___x_2212_, 1, v_a_2207_);
lean_ctor_set(v___x_2212_, 0, v___x_2229_);
v___x_2231_ = v___x_2212_;
goto v_reusejp_2230_;
}
else
{
lean_object* v_reuseFailAlloc_2233_; 
v_reuseFailAlloc_2233_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2233_, 0, v___x_2229_);
lean_ctor_set(v_reuseFailAlloc_2233_, 1, v_a_2207_);
v___x_2231_ = v_reuseFailAlloc_2233_;
goto v_reusejp_2230_;
}
v_reusejp_2230_:
{
v_a_2206_ = v_tail_2210_;
v_a_2207_ = v___x_2231_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Script_orderedUScriptToSScript_spec__2(lean_object* v_x_2237_, lean_object* v_x_2238_){
_start:
{
if (lean_obj_tag(v_x_2238_) == 0)
{
return v_x_2237_;
}
else
{
lean_object* v_key_2239_; lean_object* v_value_2240_; lean_object* v_tail_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; 
v_key_2239_ = lean_ctor_get(v_x_2238_, 0);
v_value_2240_ = lean_ctor_get(v_x_2238_, 1);
v_tail_2241_ = lean_ctor_get(v_x_2238_, 2);
lean_inc(v_value_2240_);
lean_inc(v_key_2239_);
v___x_2242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2242_, 0, v_key_2239_);
lean_ctor_set(v___x_2242_, 1, v_value_2240_);
v___x_2243_ = lean_array_push(v_x_2237_, v___x_2242_);
v_x_2237_ = v___x_2243_;
v_x_2238_ = v_tail_2241_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Script_orderedUScriptToSScript_spec__2___boxed(lean_object* v_x_2245_, lean_object* v_x_2246_){
_start:
{
lean_object* v_res_2247_; 
v_res_2247_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Script_orderedUScriptToSScript_spec__2(v_x_2245_, v_x_2246_);
lean_dec(v_x_2246_);
return v_res_2247_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(lean_object* v_as_2248_, size_t v_i_2249_, size_t v_stop_2250_, lean_object* v_b_2251_){
_start:
{
uint8_t v___x_2252_; 
v___x_2252_ = lean_usize_dec_eq(v_i_2249_, v_stop_2250_);
if (v___x_2252_ == 0)
{
lean_object* v___x_2253_; lean_object* v___x_2254_; size_t v___x_2255_; size_t v___x_2256_; 
v___x_2253_ = lean_array_uget_borrowed(v_as_2248_, v_i_2249_);
v___x_2254_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Script_orderedUScriptToSScript_spec__2(v_b_2251_, v___x_2253_);
v___x_2255_ = ((size_t)1ULL);
v___x_2256_ = lean_usize_add(v_i_2249_, v___x_2255_);
v_i_2249_ = v___x_2256_;
v_b_2251_ = v___x_2254_;
goto _start;
}
else
{
return v_b_2251_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3___boxed(lean_object* v_as_2258_, lean_object* v_i_2259_, lean_object* v_stop_2260_, lean_object* v_b_2261_){
_start:
{
size_t v_i_boxed_2262_; size_t v_stop_boxed_2263_; lean_object* v_res_2264_; 
v_i_boxed_2262_ = lean_unbox_usize(v_i_2259_);
lean_dec(v_i_2259_);
v_stop_boxed_2263_ = lean_unbox_usize(v_stop_2260_);
lean_dec(v_stop_2260_);
v_res_2264_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_as_2258_, v_i_boxed_2262_, v_stop_boxed_2263_, v_b_2261_);
lean_dec_ref(v_as_2258_);
return v_res_2264_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0(size_t v_sz_2265_, size_t v_i_2266_, lean_object* v_bs_2267_){
_start:
{
uint8_t v___x_2268_; 
v___x_2268_ = lean_usize_dec_lt(v_i_2266_, v_sz_2265_);
if (v___x_2268_ == 0)
{
return v_bs_2267_;
}
else
{
lean_object* v_v_2269_; lean_object* v_fst_2270_; lean_object* v_snd_2271_; lean_object* v___x_2273_; uint8_t v_isShared_2274_; uint8_t v_isSharedCheck_2284_; 
v_v_2269_ = lean_array_uget(v_bs_2267_, v_i_2266_);
v_fst_2270_ = lean_ctor_get(v_v_2269_, 0);
v_snd_2271_ = lean_ctor_get(v_v_2269_, 1);
v_isSharedCheck_2284_ = !lean_is_exclusive(v_v_2269_);
if (v_isSharedCheck_2284_ == 0)
{
v___x_2273_ = v_v_2269_;
v_isShared_2274_ = v_isSharedCheck_2284_;
goto v_resetjp_2272_;
}
else
{
lean_inc(v_snd_2271_);
lean_inc(v_fst_2270_);
lean_dec(v_v_2269_);
v___x_2273_ = lean_box(0);
v_isShared_2274_ = v_isSharedCheck_2284_;
goto v_resetjp_2272_;
}
v_resetjp_2272_:
{
lean_object* v___x_2275_; lean_object* v_bs_x27_2276_; lean_object* v___x_2278_; 
v___x_2275_ = lean_unsigned_to_nat(0u);
v_bs_x27_2276_ = lean_array_uset(v_bs_2267_, v_i_2266_, v___x_2275_);
if (v_isShared_2274_ == 0)
{
v___x_2278_ = v___x_2273_;
goto v_reusejp_2277_;
}
else
{
lean_object* v_reuseFailAlloc_2283_; 
v_reuseFailAlloc_2283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2283_, 0, v_fst_2270_);
lean_ctor_set(v_reuseFailAlloc_2283_, 1, v_snd_2271_);
v___x_2278_ = v_reuseFailAlloc_2283_;
goto v_reusejp_2277_;
}
v_reusejp_2277_:
{
size_t v___x_2279_; size_t v___x_2280_; lean_object* v___x_2281_; 
v___x_2279_ = ((size_t)1ULL);
v___x_2280_ = lean_usize_add(v_i_2266_, v___x_2279_);
v___x_2281_ = lean_array_uset(v_bs_x27_2276_, v_i_2266_, v___x_2278_);
v_i_2266_ = v___x_2280_;
v_bs_2267_ = v___x_2281_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0___boxed(lean_object* v_sz_2285_, lean_object* v_i_2286_, lean_object* v_bs_2287_){
_start:
{
size_t v_sz_boxed_2288_; size_t v_i_boxed_2289_; lean_object* v_res_2290_; 
v_sz_boxed_2288_ = lean_unbox_usize(v_sz_2285_);
lean_dec(v_sz_2285_);
v_i_boxed_2289_ = lean_unbox_usize(v_i_2286_);
lean_dec(v_i_2286_);
v_res_2290_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0(v_sz_boxed_2288_, v_i_boxed_2289_, v_bs_2287_);
return v_res_2290_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1(void){
_start:
{
lean_object* v___x_2292_; lean_object* v___x_2293_; 
v___x_2292_ = ((lean_object*)(lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__0));
v___x_2293_ = l_Lean_stringToMessageData(v___x_2292_);
return v___x_2293_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3(void){
_start:
{
lean_object* v___x_2295_; lean_object* v___x_2296_; 
v___x_2295_ = ((lean_object*)(lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__2));
v___x_2296_ = l_Lean_stringToMessageData(v___x_2295_);
return v___x_2296_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1(lean_object* v___x_2297_, lean_object* v_uscript_2298_, lean_object* v_tacticState_2299_, lean_object* v_____r_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_){
_start:
{
lean_object* v___y_2305_; lean_object* v___y_2306_; lean_object* v___y_2307_; lean_object* v___y_2308_; lean_object* v___y_2332_; lean_object* v___y_2333_; lean_object* v___y_2334_; lean_object* v___y_2335_; lean_object* v___y_2336_; lean_object* v___y_2337_; lean_object* v___y_2338_; lean_object* v___x_2356_; lean_object* v_a_2357_; lean_object* v___x_2358_; lean_object* v___y_2360_; lean_object* v___y_2361_; uint8_t v___x_2382_; 
v___x_2356_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2297_, v___y_2301_);
v_a_2357_ = lean_ctor_get(v___x_2356_, 0);
lean_inc(v_a_2357_);
lean_dec_ref(v___x_2356_);
v___x_2358_ = lp_aesop_Aesop_Script_UScript_toStepTree(v_uscript_2298_);
v___x_2382_ = lean_unbox(v_a_2357_);
lean_dec(v_a_2357_);
if (v___x_2382_ == 0)
{
v___y_2360_ = v___y_2301_;
v___y_2361_ = v___y_2302_;
goto v___jp_2359_;
}
else
{
lean_object* v_traceClass_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; 
v_traceClass_2383_ = lean_ctor_get(v___x_2297_, 0);
v___x_2384_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3);
lean_inc(v___x_2358_);
v___x_2385_ = lp_aesop_Aesop_Script_StepTree_toMessageData(v___x_2358_);
v___x_2386_ = l_Lean_indentD(v___x_2385_);
v___x_2387_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2387_, 0, v___x_2384_);
lean_ctor_set(v___x_2387_, 1, v___x_2386_);
lean_inc(v_traceClass_2383_);
v___x_2388_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_2383_, v___x_2387_, v___y_2301_, v___y_2302_);
if (lean_obj_tag(v___x_2388_) == 0)
{
lean_dec_ref_known(v___x_2388_, 1);
v___y_2360_ = v___y_2301_;
v___y_2361_ = v___y_2302_;
goto v___jp_2359_;
}
else
{
lean_object* v_a_2389_; lean_object* v___x_2391_; uint8_t v_isShared_2392_; uint8_t v_isSharedCheck_2396_; 
lean_dec(v___x_2358_);
lean_dec_ref(v_tacticState_2299_);
lean_dec_ref(v___x_2297_);
v_a_2389_ = lean_ctor_get(v___x_2388_, 0);
v_isSharedCheck_2396_ = !lean_is_exclusive(v___x_2388_);
if (v_isSharedCheck_2396_ == 0)
{
v___x_2391_ = v___x_2388_;
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
else
{
lean_inc(v_a_2389_);
lean_dec(v___x_2388_);
v___x_2391_ = lean_box(0);
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
v_resetjp_2390_:
{
lean_object* v___x_2394_; 
if (v_isShared_2392_ == 0)
{
v___x_2394_ = v___x_2391_;
goto v_reusejp_2393_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v_a_2389_);
v___x_2394_ = v_reuseFailAlloc_2395_;
goto v_reusejp_2393_;
}
v_reusejp_2393_:
{
return v___x_2394_;
}
}
}
}
v___jp_2304_:
{
lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; 
v___x_2309_ = lean_unsigned_to_nat(0u);
v___x_2310_ = lean_array_get_size(v_uscript_2298_);
v___x_2311_ = lean_unsigned_to_nat(1u);
v___x_2312_ = lean_nat_sub(v___x_2310_, v___x_2311_);
v___x_2313_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_2298_, v___y_2306_, v___y_2305_, v___x_2309_, v___x_2312_, v_tacticState_2299_, v___y_2307_, v___y_2308_);
lean_dec(v___x_2312_);
lean_dec_ref(v___y_2305_);
lean_dec_ref(v___y_2306_);
if (lean_obj_tag(v___x_2313_) == 0)
{
lean_object* v_a_2314_; lean_object* v___x_2316_; uint8_t v_isShared_2317_; uint8_t v_isSharedCheck_2322_; 
v_a_2314_ = lean_ctor_get(v___x_2313_, 0);
v_isSharedCheck_2322_ = !lean_is_exclusive(v___x_2313_);
if (v_isSharedCheck_2322_ == 0)
{
v___x_2316_ = v___x_2313_;
v_isShared_2317_ = v_isSharedCheck_2322_;
goto v_resetjp_2315_;
}
else
{
lean_inc(v_a_2314_);
lean_dec(v___x_2313_);
v___x_2316_ = lean_box(0);
v_isShared_2317_ = v_isSharedCheck_2322_;
goto v_resetjp_2315_;
}
v_resetjp_2315_:
{
lean_object* v_fst_2318_; lean_object* v___x_2320_; 
v_fst_2318_ = lean_ctor_get(v_a_2314_, 0);
lean_inc(v_fst_2318_);
lean_dec(v_a_2314_);
if (v_isShared_2317_ == 0)
{
lean_ctor_set(v___x_2316_, 0, v_fst_2318_);
v___x_2320_ = v___x_2316_;
goto v_reusejp_2319_;
}
else
{
lean_object* v_reuseFailAlloc_2321_; 
v_reuseFailAlloc_2321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2321_, 0, v_fst_2318_);
v___x_2320_ = v_reuseFailAlloc_2321_;
goto v_reusejp_2319_;
}
v_reusejp_2319_:
{
return v___x_2320_;
}
}
}
else
{
lean_object* v_a_2323_; lean_object* v___x_2325_; uint8_t v_isShared_2326_; uint8_t v_isSharedCheck_2330_; 
v_a_2323_ = lean_ctor_get(v___x_2313_, 0);
v_isSharedCheck_2330_ = !lean_is_exclusive(v___x_2313_);
if (v_isSharedCheck_2330_ == 0)
{
v___x_2325_ = v___x_2313_;
v_isShared_2326_ = v_isSharedCheck_2330_;
goto v_resetjp_2324_;
}
else
{
lean_inc(v_a_2323_);
lean_dec(v___x_2313_);
v___x_2325_ = lean_box(0);
v_isShared_2326_ = v_isSharedCheck_2330_;
goto v_resetjp_2324_;
}
v_resetjp_2324_:
{
lean_object* v___x_2328_; 
if (v_isShared_2326_ == 0)
{
v___x_2328_ = v___x_2325_;
goto v_reusejp_2327_;
}
else
{
lean_object* v_reuseFailAlloc_2329_; 
v_reuseFailAlloc_2329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2329_, 0, v_a_2323_);
v___x_2328_ = v_reuseFailAlloc_2329_;
goto v_reusejp_2327_;
}
v_reusejp_2327_:
{
return v___x_2328_;
}
}
}
}
v___jp_2331_:
{
size_t v_sz_2339_; size_t v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; 
v_sz_2339_ = lean_array_size(v___y_2338_);
v___x_2340_ = ((size_t)0ULL);
v___x_2341_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0(v_sz_2339_, v___x_2340_, v___y_2338_);
v___x_2342_ = lean_array_to_list(v___x_2341_);
v___x_2343_ = lean_box(0);
v___x_2344_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1(v___x_2342_, v___x_2343_);
v___x_2345_ = l_Lean_MessageData_ofList(v___x_2344_);
lean_inc_ref(v___y_2336_);
v___x_2346_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2346_, 0, v___y_2336_);
lean_ctor_set(v___x_2346_, 1, v___x_2345_);
v___x_2347_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v___y_2335_, v___x_2346_, v___y_2337_, v___y_2334_);
if (lean_obj_tag(v___x_2347_) == 0)
{
lean_dec_ref_known(v___x_2347_, 1);
v___y_2305_ = v___y_2332_;
v___y_2306_ = v___y_2333_;
v___y_2307_ = v___y_2337_;
v___y_2308_ = v___y_2334_;
goto v___jp_2304_;
}
else
{
lean_object* v_a_2348_; lean_object* v___x_2350_; uint8_t v_isShared_2351_; uint8_t v_isSharedCheck_2355_; 
lean_dec_ref(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec_ref(v_tacticState_2299_);
v_a_2348_ = lean_ctor_get(v___x_2347_, 0);
v_isSharedCheck_2355_ = !lean_is_exclusive(v___x_2347_);
if (v_isSharedCheck_2355_ == 0)
{
v___x_2350_ = v___x_2347_;
v_isShared_2351_ = v_isSharedCheck_2355_;
goto v_resetjp_2349_;
}
else
{
lean_inc(v_a_2348_);
lean_dec(v___x_2347_);
v___x_2350_ = lean_box(0);
v_isShared_2351_ = v_isSharedCheck_2355_;
goto v_resetjp_2349_;
}
v_resetjp_2349_:
{
lean_object* v___x_2353_; 
if (v_isShared_2351_ == 0)
{
v___x_2353_ = v___x_2350_;
goto v_reusejp_2352_;
}
else
{
lean_object* v_reuseFailAlloc_2354_; 
v_reuseFailAlloc_2354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2354_, 0, v_a_2348_);
v___x_2353_ = v_reuseFailAlloc_2354_;
goto v_reusejp_2352_;
}
v_reusejp_2352_:
{
return v___x_2353_;
}
}
}
}
v___jp_2359_:
{
lean_object* v___x_2362_; lean_object* v_a_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; uint8_t v___x_2366_; 
v___x_2362_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2297_, v___y_2360_);
v_a_2363_ = lean_ctor_get(v___x_2362_, 0);
lean_inc(v_a_2363_);
lean_dec_ref(v___x_2362_);
lean_inc(v___x_2358_);
v___x_2364_ = lp_aesop_Aesop_Script_StepTree_focusableGoals(v___x_2358_);
v___x_2365_ = lp_aesop_Aesop_Script_StepTree_numSiblings(v___x_2358_);
v___x_2366_ = lean_unbox(v_a_2363_);
lean_dec(v_a_2363_);
if (v___x_2366_ == 0)
{
lean_dec_ref(v___x_2297_);
v___y_2305_ = v___x_2365_;
v___y_2306_ = v___x_2364_;
v___y_2307_ = v___y_2360_;
v___y_2308_ = v___y_2361_;
goto v___jp_2304_;
}
else
{
lean_object* v_traceClass_2367_; lean_object* v_size_2368_; lean_object* v_buckets_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; uint8_t v___x_2374_; 
v_traceClass_2367_ = lean_ctor_get(v___x_2297_, 0);
lean_inc(v_traceClass_2367_);
lean_dec_ref(v___x_2297_);
v_size_2368_ = lean_ctor_get(v___x_2364_, 0);
lean_inc(v_size_2368_);
v_buckets_2369_ = lean_ctor_get(v___x_2364_, 1);
lean_inc_ref(v_buckets_2369_);
v___x_2370_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1);
v___x_2371_ = lean_mk_empty_array_with_capacity(v_size_2368_);
lean_dec(v_size_2368_);
v___x_2372_ = lean_unsigned_to_nat(0u);
v___x_2373_ = lean_array_get_size(v_buckets_2369_);
v___x_2374_ = lean_nat_dec_lt(v___x_2372_, v___x_2373_);
if (v___x_2374_ == 0)
{
lean_dec_ref(v_buckets_2369_);
v___y_2332_ = v___x_2365_;
v___y_2333_ = v___x_2364_;
v___y_2334_ = v___y_2361_;
v___y_2335_ = v_traceClass_2367_;
v___y_2336_ = v___x_2370_;
v___y_2337_ = v___y_2360_;
v___y_2338_ = v___x_2371_;
goto v___jp_2331_;
}
else
{
uint8_t v___x_2375_; 
v___x_2375_ = lean_nat_dec_le(v___x_2373_, v___x_2373_);
if (v___x_2375_ == 0)
{
if (v___x_2374_ == 0)
{
lean_dec_ref(v_buckets_2369_);
v___y_2332_ = v___x_2365_;
v___y_2333_ = v___x_2364_;
v___y_2334_ = v___y_2361_;
v___y_2335_ = v_traceClass_2367_;
v___y_2336_ = v___x_2370_;
v___y_2337_ = v___y_2360_;
v___y_2338_ = v___x_2371_;
goto v___jp_2331_;
}
else
{
size_t v___x_2376_; size_t v___x_2377_; lean_object* v___x_2378_; 
v___x_2376_ = ((size_t)0ULL);
v___x_2377_ = lean_usize_of_nat(v___x_2373_);
v___x_2378_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2369_, v___x_2376_, v___x_2377_, v___x_2371_);
lean_dec_ref(v_buckets_2369_);
v___y_2332_ = v___x_2365_;
v___y_2333_ = v___x_2364_;
v___y_2334_ = v___y_2361_;
v___y_2335_ = v_traceClass_2367_;
v___y_2336_ = v___x_2370_;
v___y_2337_ = v___y_2360_;
v___y_2338_ = v___x_2378_;
goto v___jp_2331_;
}
}
else
{
size_t v___x_2379_; size_t v___x_2380_; lean_object* v___x_2381_; 
v___x_2379_ = ((size_t)0ULL);
v___x_2380_ = lean_usize_of_nat(v___x_2373_);
v___x_2381_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2369_, v___x_2379_, v___x_2380_, v___x_2371_);
lean_dec_ref(v_buckets_2369_);
v___y_2332_ = v___x_2365_;
v___y_2333_ = v___x_2364_;
v___y_2334_ = v___y_2361_;
v___y_2335_ = v_traceClass_2367_;
v___y_2336_ = v___x_2370_;
v___y_2337_ = v___y_2360_;
v___y_2338_ = v___x_2381_;
goto v___jp_2331_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___boxed(lean_object* v___x_2397_, lean_object* v_uscript_2398_, lean_object* v_tacticState_2399_, lean_object* v_____r_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_){
_start:
{
lean_object* v_res_2404_; 
v_res_2404_ = lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1(v___x_2397_, v_uscript_2398_, v_tacticState_2399_, v_____r_2400_, v___y_2401_, v___y_2402_);
lean_dec(v___y_2402_);
lean_dec_ref(v___y_2401_);
lean_dec_ref(v_uscript_2398_);
return v_res_2404_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__2(lean_object* v___x_2405_, lean_object* v_uscript_2406_, lean_object* v_tacticState_2407_, lean_object* v_____r_2408_, lean_object* v___y_2409_, lean_object* v___y_2410_){
_start:
{
lean_object* v___y_2413_; lean_object* v___y_2414_; lean_object* v___y_2415_; lean_object* v___y_2416_; lean_object* v___y_2440_; lean_object* v___y_2441_; lean_object* v___y_2442_; lean_object* v___y_2443_; lean_object* v___y_2444_; lean_object* v___y_2445_; lean_object* v___y_2446_; lean_object* v___x_2464_; lean_object* v_a_2465_; lean_object* v___x_2466_; lean_object* v___y_2468_; lean_object* v___y_2469_; uint8_t v___x_2490_; 
v___x_2464_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2405_, v___y_2409_);
v_a_2465_ = lean_ctor_get(v___x_2464_, 0);
lean_inc(v_a_2465_);
lean_dec_ref(v___x_2464_);
v___x_2466_ = lp_aesop_Aesop_Script_UScript_toStepTree(v_uscript_2406_);
v___x_2490_ = lean_unbox(v_a_2465_);
lean_dec(v_a_2465_);
if (v___x_2490_ == 0)
{
v___y_2468_ = v___y_2409_;
v___y_2469_ = v___y_2410_;
goto v___jp_2467_;
}
else
{
lean_object* v_traceClass_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; 
v_traceClass_2491_ = lean_ctor_get(v___x_2405_, 0);
v___x_2492_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3);
lean_inc(v___x_2466_);
v___x_2493_ = lp_aesop_Aesop_Script_StepTree_toMessageData(v___x_2466_);
v___x_2494_ = l_Lean_indentD(v___x_2493_);
v___x_2495_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2495_, 0, v___x_2492_);
lean_ctor_set(v___x_2495_, 1, v___x_2494_);
lean_inc(v_traceClass_2491_);
v___x_2496_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_2491_, v___x_2495_, v___y_2409_, v___y_2410_);
if (lean_obj_tag(v___x_2496_) == 0)
{
lean_dec_ref_known(v___x_2496_, 1);
v___y_2468_ = v___y_2409_;
v___y_2469_ = v___y_2410_;
goto v___jp_2467_;
}
else
{
lean_object* v_a_2497_; lean_object* v___x_2499_; uint8_t v_isShared_2500_; uint8_t v_isSharedCheck_2504_; 
lean_dec(v___x_2466_);
lean_dec_ref(v_tacticState_2407_);
lean_dec_ref(v___x_2405_);
v_a_2497_ = lean_ctor_get(v___x_2496_, 0);
v_isSharedCheck_2504_ = !lean_is_exclusive(v___x_2496_);
if (v_isSharedCheck_2504_ == 0)
{
v___x_2499_ = v___x_2496_;
v_isShared_2500_ = v_isSharedCheck_2504_;
goto v_resetjp_2498_;
}
else
{
lean_inc(v_a_2497_);
lean_dec(v___x_2496_);
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
v___jp_2412_:
{
lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; 
v___x_2417_ = lean_unsigned_to_nat(0u);
v___x_2418_ = lean_array_get_size(v_uscript_2406_);
v___x_2419_ = lean_unsigned_to_nat(1u);
v___x_2420_ = lean_nat_sub(v___x_2418_, v___x_2419_);
v___x_2421_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_2406_, v___y_2413_, v___y_2414_, v___x_2417_, v___x_2420_, v_tacticState_2407_, v___y_2415_, v___y_2416_);
lean_dec(v___x_2420_);
lean_dec_ref(v___y_2414_);
lean_dec_ref(v___y_2413_);
if (lean_obj_tag(v___x_2421_) == 0)
{
lean_object* v_a_2422_; lean_object* v___x_2424_; uint8_t v_isShared_2425_; uint8_t v_isSharedCheck_2430_; 
v_a_2422_ = lean_ctor_get(v___x_2421_, 0);
v_isSharedCheck_2430_ = !lean_is_exclusive(v___x_2421_);
if (v_isSharedCheck_2430_ == 0)
{
v___x_2424_ = v___x_2421_;
v_isShared_2425_ = v_isSharedCheck_2430_;
goto v_resetjp_2423_;
}
else
{
lean_inc(v_a_2422_);
lean_dec(v___x_2421_);
v___x_2424_ = lean_box(0);
v_isShared_2425_ = v_isSharedCheck_2430_;
goto v_resetjp_2423_;
}
v_resetjp_2423_:
{
lean_object* v_fst_2426_; lean_object* v___x_2428_; 
v_fst_2426_ = lean_ctor_get(v_a_2422_, 0);
lean_inc(v_fst_2426_);
lean_dec(v_a_2422_);
if (v_isShared_2425_ == 0)
{
lean_ctor_set(v___x_2424_, 0, v_fst_2426_);
v___x_2428_ = v___x_2424_;
goto v_reusejp_2427_;
}
else
{
lean_object* v_reuseFailAlloc_2429_; 
v_reuseFailAlloc_2429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2429_, 0, v_fst_2426_);
v___x_2428_ = v_reuseFailAlloc_2429_;
goto v_reusejp_2427_;
}
v_reusejp_2427_:
{
return v___x_2428_;
}
}
}
else
{
lean_object* v_a_2431_; lean_object* v___x_2433_; uint8_t v_isShared_2434_; uint8_t v_isSharedCheck_2438_; 
v_a_2431_ = lean_ctor_get(v___x_2421_, 0);
v_isSharedCheck_2438_ = !lean_is_exclusive(v___x_2421_);
if (v_isSharedCheck_2438_ == 0)
{
v___x_2433_ = v___x_2421_;
v_isShared_2434_ = v_isSharedCheck_2438_;
goto v_resetjp_2432_;
}
else
{
lean_inc(v_a_2431_);
lean_dec(v___x_2421_);
v___x_2433_ = lean_box(0);
v_isShared_2434_ = v_isSharedCheck_2438_;
goto v_resetjp_2432_;
}
v_resetjp_2432_:
{
lean_object* v___x_2436_; 
if (v_isShared_2434_ == 0)
{
v___x_2436_ = v___x_2433_;
goto v_reusejp_2435_;
}
else
{
lean_object* v_reuseFailAlloc_2437_; 
v_reuseFailAlloc_2437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2437_, 0, v_a_2431_);
v___x_2436_ = v_reuseFailAlloc_2437_;
goto v_reusejp_2435_;
}
v_reusejp_2435_:
{
return v___x_2436_;
}
}
}
}
v___jp_2439_:
{
size_t v_sz_2447_; size_t v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; 
v_sz_2447_ = lean_array_size(v___y_2446_);
v___x_2448_ = ((size_t)0ULL);
v___x_2449_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0(v_sz_2447_, v___x_2448_, v___y_2446_);
v___x_2450_ = lean_array_to_list(v___x_2449_);
v___x_2451_ = lean_box(0);
v___x_2452_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1(v___x_2450_, v___x_2451_);
v___x_2453_ = l_Lean_MessageData_ofList(v___x_2452_);
lean_inc_ref(v___y_2445_);
v___x_2454_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2454_, 0, v___y_2445_);
lean_ctor_set(v___x_2454_, 1, v___x_2453_);
v___x_2455_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v___y_2441_, v___x_2454_, v___y_2442_, v___y_2444_);
if (lean_obj_tag(v___x_2455_) == 0)
{
lean_dec_ref_known(v___x_2455_, 1);
v___y_2413_ = v___y_2440_;
v___y_2414_ = v___y_2443_;
v___y_2415_ = v___y_2442_;
v___y_2416_ = v___y_2444_;
goto v___jp_2412_;
}
else
{
lean_object* v_a_2456_; lean_object* v___x_2458_; uint8_t v_isShared_2459_; uint8_t v_isSharedCheck_2463_; 
lean_dec_ref(v___y_2443_);
lean_dec_ref(v___y_2440_);
lean_dec_ref(v_tacticState_2407_);
v_a_2456_ = lean_ctor_get(v___x_2455_, 0);
v_isSharedCheck_2463_ = !lean_is_exclusive(v___x_2455_);
if (v_isSharedCheck_2463_ == 0)
{
v___x_2458_ = v___x_2455_;
v_isShared_2459_ = v_isSharedCheck_2463_;
goto v_resetjp_2457_;
}
else
{
lean_inc(v_a_2456_);
lean_dec(v___x_2455_);
v___x_2458_ = lean_box(0);
v_isShared_2459_ = v_isSharedCheck_2463_;
goto v_resetjp_2457_;
}
v_resetjp_2457_:
{
lean_object* v___x_2461_; 
if (v_isShared_2459_ == 0)
{
v___x_2461_ = v___x_2458_;
goto v_reusejp_2460_;
}
else
{
lean_object* v_reuseFailAlloc_2462_; 
v_reuseFailAlloc_2462_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2462_, 0, v_a_2456_);
v___x_2461_ = v_reuseFailAlloc_2462_;
goto v_reusejp_2460_;
}
v_reusejp_2460_:
{
return v___x_2461_;
}
}
}
}
v___jp_2467_:
{
lean_object* v___x_2470_; lean_object* v_a_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; uint8_t v___x_2474_; 
v___x_2470_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2405_, v___y_2468_);
v_a_2471_ = lean_ctor_get(v___x_2470_, 0);
lean_inc(v_a_2471_);
lean_dec_ref(v___x_2470_);
lean_inc(v___x_2466_);
v___x_2472_ = lp_aesop_Aesop_Script_StepTree_focusableGoals(v___x_2466_);
v___x_2473_ = lp_aesop_Aesop_Script_StepTree_numSiblings(v___x_2466_);
v___x_2474_ = lean_unbox(v_a_2471_);
lean_dec(v_a_2471_);
if (v___x_2474_ == 0)
{
lean_dec_ref(v___x_2405_);
v___y_2413_ = v___x_2472_;
v___y_2414_ = v___x_2473_;
v___y_2415_ = v___y_2468_;
v___y_2416_ = v___y_2469_;
goto v___jp_2412_;
}
else
{
lean_object* v_traceClass_2475_; lean_object* v_size_2476_; lean_object* v_buckets_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; uint8_t v___x_2482_; 
v_traceClass_2475_ = lean_ctor_get(v___x_2405_, 0);
lean_inc(v_traceClass_2475_);
lean_dec_ref(v___x_2405_);
v_size_2476_ = lean_ctor_get(v___x_2472_, 0);
lean_inc(v_size_2476_);
v_buckets_2477_ = lean_ctor_get(v___x_2472_, 1);
lean_inc_ref(v_buckets_2477_);
v___x_2478_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1);
v___x_2479_ = lean_mk_empty_array_with_capacity(v_size_2476_);
lean_dec(v_size_2476_);
v___x_2480_ = lean_unsigned_to_nat(0u);
v___x_2481_ = lean_array_get_size(v_buckets_2477_);
v___x_2482_ = lean_nat_dec_lt(v___x_2480_, v___x_2481_);
if (v___x_2482_ == 0)
{
lean_dec_ref(v_buckets_2477_);
v___y_2440_ = v___x_2472_;
v___y_2441_ = v_traceClass_2475_;
v___y_2442_ = v___y_2468_;
v___y_2443_ = v___x_2473_;
v___y_2444_ = v___y_2469_;
v___y_2445_ = v___x_2478_;
v___y_2446_ = v___x_2479_;
goto v___jp_2439_;
}
else
{
uint8_t v___x_2483_; 
v___x_2483_ = lean_nat_dec_le(v___x_2481_, v___x_2481_);
if (v___x_2483_ == 0)
{
if (v___x_2482_ == 0)
{
lean_dec_ref(v_buckets_2477_);
v___y_2440_ = v___x_2472_;
v___y_2441_ = v_traceClass_2475_;
v___y_2442_ = v___y_2468_;
v___y_2443_ = v___x_2473_;
v___y_2444_ = v___y_2469_;
v___y_2445_ = v___x_2478_;
v___y_2446_ = v___x_2479_;
goto v___jp_2439_;
}
else
{
size_t v___x_2484_; size_t v___x_2485_; lean_object* v___x_2486_; 
v___x_2484_ = ((size_t)0ULL);
v___x_2485_ = lean_usize_of_nat(v___x_2481_);
v___x_2486_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2477_, v___x_2484_, v___x_2485_, v___x_2479_);
lean_dec_ref(v_buckets_2477_);
v___y_2440_ = v___x_2472_;
v___y_2441_ = v_traceClass_2475_;
v___y_2442_ = v___y_2468_;
v___y_2443_ = v___x_2473_;
v___y_2444_ = v___y_2469_;
v___y_2445_ = v___x_2478_;
v___y_2446_ = v___x_2486_;
goto v___jp_2439_;
}
}
else
{
size_t v___x_2487_; size_t v___x_2488_; lean_object* v___x_2489_; 
v___x_2487_ = ((size_t)0ULL);
v___x_2488_ = lean_usize_of_nat(v___x_2481_);
v___x_2489_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2477_, v___x_2487_, v___x_2488_, v___x_2479_);
lean_dec_ref(v_buckets_2477_);
v___y_2440_ = v___x_2472_;
v___y_2441_ = v_traceClass_2475_;
v___y_2442_ = v___y_2468_;
v___y_2443_ = v___x_2473_;
v___y_2444_ = v___y_2469_;
v___y_2445_ = v___x_2478_;
v___y_2446_ = v___x_2489_;
goto v___jp_2439_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__2___boxed(lean_object* v___x_2505_, lean_object* v_uscript_2506_, lean_object* v_tacticState_2507_, lean_object* v_____r_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_, lean_object* v___y_2511_){
_start:
{
lean_object* v_res_2512_; 
v_res_2512_ = lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__2(v___x_2505_, v_uscript_2506_, v_tacticState_2507_, v_____r_2508_, v___y_2509_, v___y_2510_);
lean_dec(v___y_2510_);
lean_dec_ref(v___y_2509_);
lean_dec_ref(v_uscript_2506_);
return v_res_2512_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6_spec__7(size_t v_sz_2513_, size_t v_i_2514_, lean_object* v_bs_2515_){
_start:
{
uint8_t v___x_2516_; 
v___x_2516_ = lean_usize_dec_lt(v_i_2514_, v_sz_2513_);
if (v___x_2516_ == 0)
{
return v_bs_2515_;
}
else
{
lean_object* v_v_2517_; lean_object* v_msg_2518_; lean_object* v___x_2519_; lean_object* v_bs_x27_2520_; size_t v___x_2521_; size_t v___x_2522_; lean_object* v___x_2523_; 
v_v_2517_ = lean_array_uget_borrowed(v_bs_2515_, v_i_2514_);
v_msg_2518_ = lean_ctor_get(v_v_2517_, 1);
lean_inc_ref(v_msg_2518_);
v___x_2519_ = lean_unsigned_to_nat(0u);
v_bs_x27_2520_ = lean_array_uset(v_bs_2515_, v_i_2514_, v___x_2519_);
v___x_2521_ = ((size_t)1ULL);
v___x_2522_ = lean_usize_add(v_i_2514_, v___x_2521_);
v___x_2523_ = lean_array_uset(v_bs_x27_2520_, v_i_2514_, v_msg_2518_);
v_i_2514_ = v___x_2522_;
v_bs_2515_ = v___x_2523_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6_spec__7___boxed(lean_object* v_sz_2525_, lean_object* v_i_2526_, lean_object* v_bs_2527_){
_start:
{
size_t v_sz_boxed_2528_; size_t v_i_boxed_2529_; lean_object* v_res_2530_; 
v_sz_boxed_2528_ = lean_unbox_usize(v_sz_2525_);
lean_dec(v_sz_2525_);
v_i_boxed_2529_ = lean_unbox_usize(v_i_2526_);
lean_dec(v_i_2526_);
v_res_2530_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6_spec__7(v_sz_boxed_2528_, v_i_boxed_2529_, v_bs_2527_);
return v_res_2530_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6(lean_object* v_oldTraces_2531_, lean_object* v_data_2532_, lean_object* v_ref_2533_, lean_object* v_msg_2534_, lean_object* v___y_2535_, lean_object* v___y_2536_){
_start:
{
lean_object* v_fileName_2538_; lean_object* v_fileMap_2539_; lean_object* v_options_2540_; lean_object* v_currRecDepth_2541_; lean_object* v_maxRecDepth_2542_; lean_object* v_ref_2543_; lean_object* v_currNamespace_2544_; lean_object* v_openDecls_2545_; lean_object* v_initHeartbeats_2546_; lean_object* v_maxHeartbeats_2547_; lean_object* v_quotContext_2548_; lean_object* v_currMacroScope_2549_; uint8_t v_diag_2550_; lean_object* v_cancelTk_x3f_2551_; uint8_t v_suppressElabErrors_2552_; lean_object* v_inheritedTraceOptions_2553_; lean_object* v___x_2554_; lean_object* v_traceState_2555_; lean_object* v_traces_2556_; lean_object* v_ref_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; size_t v_sz_2560_; size_t v___x_2561_; lean_object* v___x_2562_; lean_object* v_msg_2563_; lean_object* v___x_2564_; lean_object* v_a_2565_; lean_object* v___x_2567_; uint8_t v_isShared_2568_; uint8_t v_isSharedCheck_2602_; 
v_fileName_2538_ = lean_ctor_get(v___y_2535_, 0);
v_fileMap_2539_ = lean_ctor_get(v___y_2535_, 1);
v_options_2540_ = lean_ctor_get(v___y_2535_, 2);
v_currRecDepth_2541_ = lean_ctor_get(v___y_2535_, 3);
v_maxRecDepth_2542_ = lean_ctor_get(v___y_2535_, 4);
v_ref_2543_ = lean_ctor_get(v___y_2535_, 5);
v_currNamespace_2544_ = lean_ctor_get(v___y_2535_, 6);
v_openDecls_2545_ = lean_ctor_get(v___y_2535_, 7);
v_initHeartbeats_2546_ = lean_ctor_get(v___y_2535_, 8);
v_maxHeartbeats_2547_ = lean_ctor_get(v___y_2535_, 9);
v_quotContext_2548_ = lean_ctor_get(v___y_2535_, 10);
v_currMacroScope_2549_ = lean_ctor_get(v___y_2535_, 11);
v_diag_2550_ = lean_ctor_get_uint8(v___y_2535_, sizeof(void*)*14);
v_cancelTk_x3f_2551_ = lean_ctor_get(v___y_2535_, 12);
v_suppressElabErrors_2552_ = lean_ctor_get_uint8(v___y_2535_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2553_ = lean_ctor_get(v___y_2535_, 13);
v___x_2554_ = lean_st_ref_get(v___y_2536_);
v_traceState_2555_ = lean_ctor_get(v___x_2554_, 4);
lean_inc_ref(v_traceState_2555_);
lean_dec(v___x_2554_);
v_traces_2556_ = lean_ctor_get(v_traceState_2555_, 0);
lean_inc_ref(v_traces_2556_);
lean_dec_ref(v_traceState_2555_);
v_ref_2557_ = l_Lean_replaceRef(v_ref_2533_, v_ref_2543_);
lean_inc_ref(v_inheritedTraceOptions_2553_);
lean_inc(v_cancelTk_x3f_2551_);
lean_inc(v_currMacroScope_2549_);
lean_inc(v_quotContext_2548_);
lean_inc(v_maxHeartbeats_2547_);
lean_inc(v_initHeartbeats_2546_);
lean_inc(v_openDecls_2545_);
lean_inc(v_currNamespace_2544_);
lean_inc(v_maxRecDepth_2542_);
lean_inc(v_currRecDepth_2541_);
lean_inc_ref(v_options_2540_);
lean_inc_ref(v_fileMap_2539_);
lean_inc_ref(v_fileName_2538_);
v___x_2558_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2558_, 0, v_fileName_2538_);
lean_ctor_set(v___x_2558_, 1, v_fileMap_2539_);
lean_ctor_set(v___x_2558_, 2, v_options_2540_);
lean_ctor_set(v___x_2558_, 3, v_currRecDepth_2541_);
lean_ctor_set(v___x_2558_, 4, v_maxRecDepth_2542_);
lean_ctor_set(v___x_2558_, 5, v_ref_2557_);
lean_ctor_set(v___x_2558_, 6, v_currNamespace_2544_);
lean_ctor_set(v___x_2558_, 7, v_openDecls_2545_);
lean_ctor_set(v___x_2558_, 8, v_initHeartbeats_2546_);
lean_ctor_set(v___x_2558_, 9, v_maxHeartbeats_2547_);
lean_ctor_set(v___x_2558_, 10, v_quotContext_2548_);
lean_ctor_set(v___x_2558_, 11, v_currMacroScope_2549_);
lean_ctor_set(v___x_2558_, 12, v_cancelTk_x3f_2551_);
lean_ctor_set(v___x_2558_, 13, v_inheritedTraceOptions_2553_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*14, v_diag_2550_);
lean_ctor_set_uint8(v___x_2558_, sizeof(void*)*14 + 1, v_suppressElabErrors_2552_);
v___x_2559_ = l_Lean_PersistentArray_toArray___redArg(v_traces_2556_);
lean_dec_ref(v_traces_2556_);
v_sz_2560_ = lean_array_size(v___x_2559_);
v___x_2561_ = ((size_t)0ULL);
v___x_2562_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6_spec__7(v_sz_2560_, v___x_2561_, v___x_2559_);
v_msg_2563_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_2563_, 0, v_data_2532_);
lean_ctor_set(v_msg_2563_, 1, v_msg_2534_);
lean_ctor_set(v_msg_2563_, 2, v___x_2562_);
v___x_2564_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9_spec__14(v_msg_2563_, v___x_2558_, v___y_2536_);
lean_dec_ref_known(v___x_2558_, 14);
v_a_2565_ = lean_ctor_get(v___x_2564_, 0);
v_isSharedCheck_2602_ = !lean_is_exclusive(v___x_2564_);
if (v_isSharedCheck_2602_ == 0)
{
v___x_2567_ = v___x_2564_;
v_isShared_2568_ = v_isSharedCheck_2602_;
goto v_resetjp_2566_;
}
else
{
lean_inc(v_a_2565_);
lean_dec(v___x_2564_);
v___x_2567_ = lean_box(0);
v_isShared_2568_ = v_isSharedCheck_2602_;
goto v_resetjp_2566_;
}
v_resetjp_2566_:
{
lean_object* v___x_2569_; lean_object* v_traceState_2570_; lean_object* v_env_2571_; lean_object* v_nextMacroScope_2572_; lean_object* v_ngen_2573_; lean_object* v_auxDeclNGen_2574_; lean_object* v_cache_2575_; lean_object* v_messages_2576_; lean_object* v_infoState_2577_; lean_object* v_snapshotTasks_2578_; lean_object* v___x_2580_; uint8_t v_isShared_2581_; uint8_t v_isSharedCheck_2601_; 
v___x_2569_ = lean_st_ref_take(v___y_2536_);
v_traceState_2570_ = lean_ctor_get(v___x_2569_, 4);
v_env_2571_ = lean_ctor_get(v___x_2569_, 0);
v_nextMacroScope_2572_ = lean_ctor_get(v___x_2569_, 1);
v_ngen_2573_ = lean_ctor_get(v___x_2569_, 2);
v_auxDeclNGen_2574_ = lean_ctor_get(v___x_2569_, 3);
v_cache_2575_ = lean_ctor_get(v___x_2569_, 5);
v_messages_2576_ = lean_ctor_get(v___x_2569_, 6);
v_infoState_2577_ = lean_ctor_get(v___x_2569_, 7);
v_snapshotTasks_2578_ = lean_ctor_get(v___x_2569_, 8);
v_isSharedCheck_2601_ = !lean_is_exclusive(v___x_2569_);
if (v_isSharedCheck_2601_ == 0)
{
v___x_2580_ = v___x_2569_;
v_isShared_2581_ = v_isSharedCheck_2601_;
goto v_resetjp_2579_;
}
else
{
lean_inc(v_snapshotTasks_2578_);
lean_inc(v_infoState_2577_);
lean_inc(v_messages_2576_);
lean_inc(v_cache_2575_);
lean_inc(v_traceState_2570_);
lean_inc(v_auxDeclNGen_2574_);
lean_inc(v_ngen_2573_);
lean_inc(v_nextMacroScope_2572_);
lean_inc(v_env_2571_);
lean_dec(v___x_2569_);
v___x_2580_ = lean_box(0);
v_isShared_2581_ = v_isSharedCheck_2601_;
goto v_resetjp_2579_;
}
v_resetjp_2579_:
{
uint64_t v_tid_2582_; lean_object* v___x_2584_; uint8_t v_isShared_2585_; uint8_t v_isSharedCheck_2599_; 
v_tid_2582_ = lean_ctor_get_uint64(v_traceState_2570_, sizeof(void*)*1);
v_isSharedCheck_2599_ = !lean_is_exclusive(v_traceState_2570_);
if (v_isSharedCheck_2599_ == 0)
{
lean_object* v_unused_2600_; 
v_unused_2600_ = lean_ctor_get(v_traceState_2570_, 0);
lean_dec(v_unused_2600_);
v___x_2584_ = v_traceState_2570_;
v_isShared_2585_ = v_isSharedCheck_2599_;
goto v_resetjp_2583_;
}
else
{
lean_dec(v_traceState_2570_);
v___x_2584_ = lean_box(0);
v_isShared_2585_ = v_isSharedCheck_2599_;
goto v_resetjp_2583_;
}
v_resetjp_2583_:
{
lean_object* v___x_2586_; lean_object* v___x_2587_; lean_object* v___x_2589_; 
v___x_2586_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2586_, 0, v_ref_2533_);
lean_ctor_set(v___x_2586_, 1, v_a_2565_);
v___x_2587_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_2531_, v___x_2586_);
if (v_isShared_2585_ == 0)
{
lean_ctor_set(v___x_2584_, 0, v___x_2587_);
v___x_2589_ = v___x_2584_;
goto v_reusejp_2588_;
}
else
{
lean_object* v_reuseFailAlloc_2598_; 
v_reuseFailAlloc_2598_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2598_, 0, v___x_2587_);
lean_ctor_set_uint64(v_reuseFailAlloc_2598_, sizeof(void*)*1, v_tid_2582_);
v___x_2589_ = v_reuseFailAlloc_2598_;
goto v_reusejp_2588_;
}
v_reusejp_2588_:
{
lean_object* v___x_2591_; 
if (v_isShared_2581_ == 0)
{
lean_ctor_set(v___x_2580_, 4, v___x_2589_);
v___x_2591_ = v___x_2580_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2597_; 
v_reuseFailAlloc_2597_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2597_, 0, v_env_2571_);
lean_ctor_set(v_reuseFailAlloc_2597_, 1, v_nextMacroScope_2572_);
lean_ctor_set(v_reuseFailAlloc_2597_, 2, v_ngen_2573_);
lean_ctor_set(v_reuseFailAlloc_2597_, 3, v_auxDeclNGen_2574_);
lean_ctor_set(v_reuseFailAlloc_2597_, 4, v___x_2589_);
lean_ctor_set(v_reuseFailAlloc_2597_, 5, v_cache_2575_);
lean_ctor_set(v_reuseFailAlloc_2597_, 6, v_messages_2576_);
lean_ctor_set(v_reuseFailAlloc_2597_, 7, v_infoState_2577_);
lean_ctor_set(v_reuseFailAlloc_2597_, 8, v_snapshotTasks_2578_);
v___x_2591_ = v_reuseFailAlloc_2597_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v___x_2595_; 
v___x_2592_ = lean_st_ref_set(v___y_2536_, v___x_2591_);
v___x_2593_ = lean_box(0);
if (v_isShared_2568_ == 0)
{
lean_ctor_set(v___x_2567_, 0, v___x_2593_);
v___x_2595_ = v___x_2567_;
goto v_reusejp_2594_;
}
else
{
lean_object* v_reuseFailAlloc_2596_; 
v_reuseFailAlloc_2596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2596_, 0, v___x_2593_);
v___x_2595_ = v_reuseFailAlloc_2596_;
goto v_reusejp_2594_;
}
v_reusejp_2594_:
{
return v___x_2595_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6___boxed(lean_object* v_oldTraces_2603_, lean_object* v_data_2604_, lean_object* v_ref_2605_, lean_object* v_msg_2606_, lean_object* v___y_2607_, lean_object* v___y_2608_, lean_object* v___y_2609_){
_start:
{
lean_object* v_res_2610_; 
v_res_2610_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6(v_oldTraces_2603_, v_data_2604_, v_ref_2605_, v_msg_2606_, v___y_2607_, v___y_2608_);
lean_dec(v___y_2608_);
lean_dec_ref(v___y_2607_);
return v_res_2610_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__9(lean_object* v_opts_2611_, lean_object* v_opt_2612_){
_start:
{
lean_object* v_name_2613_; lean_object* v_defValue_2614_; lean_object* v_map_2615_; lean_object* v___x_2616_; 
v_name_2613_ = lean_ctor_get(v_opt_2612_, 0);
v_defValue_2614_ = lean_ctor_get(v_opt_2612_, 1);
v_map_2615_ = lean_ctor_get(v_opts_2611_, 0);
v___x_2616_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2615_, v_name_2613_);
if (lean_obj_tag(v___x_2616_) == 0)
{
lean_inc(v_defValue_2614_);
return v_defValue_2614_;
}
else
{
lean_object* v_val_2617_; 
v_val_2617_ = lean_ctor_get(v___x_2616_, 0);
lean_inc(v_val_2617_);
lean_dec_ref_known(v___x_2616_, 1);
if (lean_obj_tag(v_val_2617_) == 3)
{
lean_object* v_v_2618_; 
v_v_2618_ = lean_ctor_get(v_val_2617_, 0);
lean_inc(v_v_2618_);
lean_dec_ref_known(v_val_2617_, 1);
return v_v_2618_;
}
else
{
lean_dec(v_val_2617_);
lean_inc(v_defValue_2614_);
return v_defValue_2614_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__9___boxed(lean_object* v_opts_2619_, lean_object* v_opt_2620_){
_start:
{
lean_object* v_res_2621_; 
v_res_2621_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__9(v_opts_2619_, v_opt_2620_);
lean_dec_ref(v_opt_2620_);
lean_dec_ref(v_opts_2619_);
return v_res_2621_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__8(lean_object* v_e_2622_){
_start:
{
if (lean_obj_tag(v_e_2622_) == 0)
{
uint8_t v___x_2623_; 
v___x_2623_ = 2;
return v___x_2623_;
}
else
{
uint8_t v___x_2624_; 
v___x_2624_ = 0;
return v___x_2624_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__8___boxed(lean_object* v_e_2625_){
_start:
{
uint8_t v_res_2626_; lean_object* v_r_2627_; 
v_res_2626_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__8(v_e_2625_);
lean_dec_ref(v_e_2625_);
v_r_2627_ = lean_box(v_res_2626_);
return v_r_2627_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg(lean_object* v_x_2628_){
_start:
{
if (lean_obj_tag(v_x_2628_) == 0)
{
lean_object* v_a_2630_; lean_object* v___x_2632_; uint8_t v_isShared_2633_; uint8_t v_isSharedCheck_2637_; 
v_a_2630_ = lean_ctor_get(v_x_2628_, 0);
v_isSharedCheck_2637_ = !lean_is_exclusive(v_x_2628_);
if (v_isSharedCheck_2637_ == 0)
{
v___x_2632_ = v_x_2628_;
v_isShared_2633_ = v_isSharedCheck_2637_;
goto v_resetjp_2631_;
}
else
{
lean_inc(v_a_2630_);
lean_dec(v_x_2628_);
v___x_2632_ = lean_box(0);
v_isShared_2633_ = v_isSharedCheck_2637_;
goto v_resetjp_2631_;
}
v_resetjp_2631_:
{
lean_object* v___x_2635_; 
if (v_isShared_2633_ == 0)
{
lean_ctor_set_tag(v___x_2632_, 1);
v___x_2635_ = v___x_2632_;
goto v_reusejp_2634_;
}
else
{
lean_object* v_reuseFailAlloc_2636_; 
v_reuseFailAlloc_2636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2636_, 0, v_a_2630_);
v___x_2635_ = v_reuseFailAlloc_2636_;
goto v_reusejp_2634_;
}
v_reusejp_2634_:
{
return v___x_2635_;
}
}
}
else
{
lean_object* v_a_2638_; lean_object* v___x_2640_; uint8_t v_isShared_2641_; uint8_t v_isSharedCheck_2645_; 
v_a_2638_ = lean_ctor_get(v_x_2628_, 0);
v_isSharedCheck_2645_ = !lean_is_exclusive(v_x_2628_);
if (v_isSharedCheck_2645_ == 0)
{
v___x_2640_ = v_x_2628_;
v_isShared_2641_ = v_isSharedCheck_2645_;
goto v_resetjp_2639_;
}
else
{
lean_inc(v_a_2638_);
lean_dec(v_x_2628_);
v___x_2640_ = lean_box(0);
v_isShared_2641_ = v_isSharedCheck_2645_;
goto v_resetjp_2639_;
}
v_resetjp_2639_:
{
lean_object* v___x_2643_; 
if (v_isShared_2641_ == 0)
{
lean_ctor_set_tag(v___x_2640_, 0);
v___x_2643_ = v___x_2640_;
goto v_reusejp_2642_;
}
else
{
lean_object* v_reuseFailAlloc_2644_; 
v_reuseFailAlloc_2644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2644_, 0, v_a_2638_);
v___x_2643_ = v_reuseFailAlloc_2644_;
goto v_reusejp_2642_;
}
v_reusejp_2642_:
{
return v___x_2643_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg___boxed(lean_object* v_x_2646_, lean_object* v___y_2647_){
_start:
{
lean_object* v_res_2648_; 
v_res_2648_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg(v_x_2646_);
return v_res_2648_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__1(void){
_start:
{
lean_object* v___x_2650_; lean_object* v___x_2651_; 
v___x_2650_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__0));
v___x_2651_ = l_Lean_stringToMessageData(v___x_2650_);
return v___x_2651_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__2(void){
_start:
{
lean_object* v___x_2652_; double v___x_2653_; 
v___x_2652_ = lean_unsigned_to_nat(1000u);
v___x_2653_ = lean_float_of_nat(v___x_2652_);
return v___x_2653_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6(lean_object* v_cls_2654_, uint8_t v_collapsed_2655_, lean_object* v_tag_2656_, lean_object* v_opts_2657_, uint8_t v_clsEnabled_2658_, lean_object* v_oldTraces_2659_, lean_object* v_msg_2660_, lean_object* v_resStartStop_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_){
_start:
{
lean_object* v_fst_2665_; lean_object* v_snd_2666_; lean_object* v___y_2668_; lean_object* v___y_2669_; lean_object* v_data_2670_; lean_object* v_fst_2681_; lean_object* v_snd_2682_; lean_object* v___x_2683_; uint8_t v___x_2684_; lean_object* v___y_2686_; lean_object* v_a_2687_; uint8_t v___y_2702_; double v___y_2733_; 
v_fst_2665_ = lean_ctor_get(v_resStartStop_2661_, 0);
lean_inc(v_fst_2665_);
v_snd_2666_ = lean_ctor_get(v_resStartStop_2661_, 1);
lean_inc(v_snd_2666_);
lean_dec_ref(v_resStartStop_2661_);
v_fst_2681_ = lean_ctor_get(v_snd_2666_, 0);
lean_inc(v_fst_2681_);
v_snd_2682_ = lean_ctor_get(v_snd_2666_, 1);
lean_inc(v_snd_2682_);
lean_dec(v_snd_2666_);
v___x_2683_ = l_Lean_trace_profiler;
v___x_2684_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(v_opts_2657_, v___x_2683_);
if (v___x_2684_ == 0)
{
v___y_2702_ = v___x_2684_;
goto v___jp_2701_;
}
else
{
lean_object* v___x_2738_; uint8_t v___x_2739_; 
v___x_2738_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2739_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(v_opts_2657_, v___x_2738_);
if (v___x_2739_ == 0)
{
lean_object* v___x_2740_; lean_object* v___x_2741_; double v___x_2742_; double v___x_2743_; double v___x_2744_; 
v___x_2740_ = l_Lean_trace_profiler_threshold;
v___x_2741_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__9(v_opts_2657_, v___x_2740_);
v___x_2742_ = lean_float_of_nat(v___x_2741_);
v___x_2743_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__2);
v___x_2744_ = lean_float_div(v___x_2742_, v___x_2743_);
v___y_2733_ = v___x_2744_;
goto v___jp_2732_;
}
else
{
lean_object* v___x_2745_; lean_object* v___x_2746_; double v___x_2747_; 
v___x_2745_ = l_Lean_trace_profiler_threshold;
v___x_2746_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__9(v_opts_2657_, v___x_2745_);
v___x_2747_ = lean_float_of_nat(v___x_2746_);
v___y_2733_ = v___x_2747_;
goto v___jp_2732_;
}
}
v___jp_2667_:
{
lean_object* v___x_2671_; 
lean_inc(v___y_2668_);
v___x_2671_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__6(v_oldTraces_2659_, v_data_2670_, v___y_2668_, v___y_2669_, v___y_2662_, v___y_2663_);
if (lean_obj_tag(v___x_2671_) == 0)
{
lean_object* v___x_2672_; 
lean_dec_ref_known(v___x_2671_, 1);
v___x_2672_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg(v_fst_2665_);
return v___x_2672_;
}
else
{
lean_object* v_a_2673_; lean_object* v___x_2675_; uint8_t v_isShared_2676_; uint8_t v_isSharedCheck_2680_; 
lean_dec(v_fst_2665_);
v_a_2673_ = lean_ctor_get(v___x_2671_, 0);
v_isSharedCheck_2680_ = !lean_is_exclusive(v___x_2671_);
if (v_isSharedCheck_2680_ == 0)
{
v___x_2675_ = v___x_2671_;
v_isShared_2676_ = v_isSharedCheck_2680_;
goto v_resetjp_2674_;
}
else
{
lean_inc(v_a_2673_);
lean_dec(v___x_2671_);
v___x_2675_ = lean_box(0);
v_isShared_2676_ = v_isSharedCheck_2680_;
goto v_resetjp_2674_;
}
v_resetjp_2674_:
{
lean_object* v___x_2678_; 
if (v_isShared_2676_ == 0)
{
v___x_2678_ = v___x_2675_;
goto v_reusejp_2677_;
}
else
{
lean_object* v_reuseFailAlloc_2679_; 
v_reuseFailAlloc_2679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2679_, 0, v_a_2673_);
v___x_2678_ = v_reuseFailAlloc_2679_;
goto v_reusejp_2677_;
}
v_reusejp_2677_:
{
return v___x_2678_;
}
}
}
}
v___jp_2685_:
{
uint8_t v_result_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; double v___x_2691_; lean_object* v_data_2692_; 
v_result_2688_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__8(v_fst_2665_);
v___x_2689_ = lean_box(v_result_2688_);
v___x_2690_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2690_, 0, v___x_2689_);
v___x_2691_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9___closed__0);
lean_inc_ref(v_tag_2656_);
lean_inc_ref(v___x_2690_);
lean_inc(v_cls_2654_);
v_data_2692_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2692_, 0, v_cls_2654_);
lean_ctor_set(v_data_2692_, 1, v___x_2690_);
lean_ctor_set(v_data_2692_, 2, v_tag_2656_);
lean_ctor_set_float(v_data_2692_, sizeof(void*)*3, v___x_2691_);
lean_ctor_set_float(v_data_2692_, sizeof(void*)*3 + 8, v___x_2691_);
lean_ctor_set_uint8(v_data_2692_, sizeof(void*)*3 + 16, v_collapsed_2655_);
if (v___x_2684_ == 0)
{
lean_dec_ref_known(v___x_2690_, 1);
lean_dec(v_snd_2682_);
lean_dec(v_fst_2681_);
lean_dec_ref(v_tag_2656_);
lean_dec(v_cls_2654_);
v___y_2668_ = v___y_2686_;
v___y_2669_ = v_a_2687_;
v_data_2670_ = v_data_2692_;
goto v___jp_2667_;
}
else
{
lean_object* v_data_2693_; double v___x_2694_; double v___x_2695_; 
lean_dec_ref_known(v_data_2692_, 3);
v_data_2693_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2693_, 0, v_cls_2654_);
lean_ctor_set(v_data_2693_, 1, v___x_2690_);
lean_ctor_set(v_data_2693_, 2, v_tag_2656_);
v___x_2694_ = lean_unbox_float(v_fst_2681_);
lean_dec(v_fst_2681_);
lean_ctor_set_float(v_data_2693_, sizeof(void*)*3, v___x_2694_);
v___x_2695_ = lean_unbox_float(v_snd_2682_);
lean_dec(v_snd_2682_);
lean_ctor_set_float(v_data_2693_, sizeof(void*)*3 + 8, v___x_2695_);
lean_ctor_set_uint8(v_data_2693_, sizeof(void*)*3 + 16, v_collapsed_2655_);
v___y_2668_ = v___y_2686_;
v___y_2669_ = v_a_2687_;
v_data_2670_ = v_data_2693_;
goto v___jp_2667_;
}
}
v___jp_2696_:
{
lean_object* v_ref_2697_; lean_object* v___x_2698_; 
v_ref_2697_ = lean_ctor_get(v___y_2662_, 5);
lean_inc(v___y_2663_);
lean_inc_ref(v___y_2662_);
lean_inc(v_fst_2665_);
v___x_2698_ = lean_apply_4(v_msg_2660_, v_fst_2665_, v___y_2662_, v___y_2663_, lean_box(0));
if (lean_obj_tag(v___x_2698_) == 0)
{
lean_object* v_a_2699_; 
v_a_2699_ = lean_ctor_get(v___x_2698_, 0);
lean_inc(v_a_2699_);
lean_dec_ref_known(v___x_2698_, 1);
v___y_2686_ = v_ref_2697_;
v_a_2687_ = v_a_2699_;
goto v___jp_2685_;
}
else
{
lean_object* v___x_2700_; 
lean_dec_ref_known(v___x_2698_, 1);
v___x_2700_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___closed__1);
v___y_2686_ = v_ref_2697_;
v_a_2687_ = v___x_2700_;
goto v___jp_2685_;
}
}
v___jp_2701_:
{
if (v_clsEnabled_2658_ == 0)
{
if (v___y_2702_ == 0)
{
lean_object* v___x_2703_; lean_object* v_traceState_2704_; lean_object* v_env_2705_; lean_object* v_nextMacroScope_2706_; lean_object* v_ngen_2707_; lean_object* v_auxDeclNGen_2708_; lean_object* v_cache_2709_; lean_object* v_messages_2710_; lean_object* v_infoState_2711_; lean_object* v_snapshotTasks_2712_; lean_object* v___x_2714_; uint8_t v_isShared_2715_; uint8_t v_isSharedCheck_2731_; 
lean_dec(v_snd_2682_);
lean_dec(v_fst_2681_);
lean_dec_ref(v_msg_2660_);
lean_dec_ref(v_tag_2656_);
lean_dec(v_cls_2654_);
v___x_2703_ = lean_st_ref_take(v___y_2663_);
v_traceState_2704_ = lean_ctor_get(v___x_2703_, 4);
v_env_2705_ = lean_ctor_get(v___x_2703_, 0);
v_nextMacroScope_2706_ = lean_ctor_get(v___x_2703_, 1);
v_ngen_2707_ = lean_ctor_get(v___x_2703_, 2);
v_auxDeclNGen_2708_ = lean_ctor_get(v___x_2703_, 3);
v_cache_2709_ = lean_ctor_get(v___x_2703_, 5);
v_messages_2710_ = lean_ctor_get(v___x_2703_, 6);
v_infoState_2711_ = lean_ctor_get(v___x_2703_, 7);
v_snapshotTasks_2712_ = lean_ctor_get(v___x_2703_, 8);
v_isSharedCheck_2731_ = !lean_is_exclusive(v___x_2703_);
if (v_isSharedCheck_2731_ == 0)
{
v___x_2714_ = v___x_2703_;
v_isShared_2715_ = v_isSharedCheck_2731_;
goto v_resetjp_2713_;
}
else
{
lean_inc(v_snapshotTasks_2712_);
lean_inc(v_infoState_2711_);
lean_inc(v_messages_2710_);
lean_inc(v_cache_2709_);
lean_inc(v_traceState_2704_);
lean_inc(v_auxDeclNGen_2708_);
lean_inc(v_ngen_2707_);
lean_inc(v_nextMacroScope_2706_);
lean_inc(v_env_2705_);
lean_dec(v___x_2703_);
v___x_2714_ = lean_box(0);
v_isShared_2715_ = v_isSharedCheck_2731_;
goto v_resetjp_2713_;
}
v_resetjp_2713_:
{
uint64_t v_tid_2716_; lean_object* v_traces_2717_; lean_object* v___x_2719_; uint8_t v_isShared_2720_; uint8_t v_isSharedCheck_2730_; 
v_tid_2716_ = lean_ctor_get_uint64(v_traceState_2704_, sizeof(void*)*1);
v_traces_2717_ = lean_ctor_get(v_traceState_2704_, 0);
v_isSharedCheck_2730_ = !lean_is_exclusive(v_traceState_2704_);
if (v_isSharedCheck_2730_ == 0)
{
v___x_2719_ = v_traceState_2704_;
v_isShared_2720_ = v_isSharedCheck_2730_;
goto v_resetjp_2718_;
}
else
{
lean_inc(v_traces_2717_);
lean_dec(v_traceState_2704_);
v___x_2719_ = lean_box(0);
v_isShared_2720_ = v_isSharedCheck_2730_;
goto v_resetjp_2718_;
}
v_resetjp_2718_:
{
lean_object* v___x_2721_; lean_object* v___x_2723_; 
v___x_2721_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2659_, v_traces_2717_);
lean_dec_ref(v_traces_2717_);
if (v_isShared_2720_ == 0)
{
lean_ctor_set(v___x_2719_, 0, v___x_2721_);
v___x_2723_ = v___x_2719_;
goto v_reusejp_2722_;
}
else
{
lean_object* v_reuseFailAlloc_2729_; 
v_reuseFailAlloc_2729_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2729_, 0, v___x_2721_);
lean_ctor_set_uint64(v_reuseFailAlloc_2729_, sizeof(void*)*1, v_tid_2716_);
v___x_2723_ = v_reuseFailAlloc_2729_;
goto v_reusejp_2722_;
}
v_reusejp_2722_:
{
lean_object* v___x_2725_; 
if (v_isShared_2715_ == 0)
{
lean_ctor_set(v___x_2714_, 4, v___x_2723_);
v___x_2725_ = v___x_2714_;
goto v_reusejp_2724_;
}
else
{
lean_object* v_reuseFailAlloc_2728_; 
v_reuseFailAlloc_2728_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2728_, 0, v_env_2705_);
lean_ctor_set(v_reuseFailAlloc_2728_, 1, v_nextMacroScope_2706_);
lean_ctor_set(v_reuseFailAlloc_2728_, 2, v_ngen_2707_);
lean_ctor_set(v_reuseFailAlloc_2728_, 3, v_auxDeclNGen_2708_);
lean_ctor_set(v_reuseFailAlloc_2728_, 4, v___x_2723_);
lean_ctor_set(v_reuseFailAlloc_2728_, 5, v_cache_2709_);
lean_ctor_set(v_reuseFailAlloc_2728_, 6, v_messages_2710_);
lean_ctor_set(v_reuseFailAlloc_2728_, 7, v_infoState_2711_);
lean_ctor_set(v_reuseFailAlloc_2728_, 8, v_snapshotTasks_2712_);
v___x_2725_ = v_reuseFailAlloc_2728_;
goto v_reusejp_2724_;
}
v_reusejp_2724_:
{
lean_object* v___x_2726_; lean_object* v___x_2727_; 
v___x_2726_ = lean_st_ref_set(v___y_2663_, v___x_2725_);
v___x_2727_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg(v_fst_2665_);
return v___x_2727_;
}
}
}
}
}
else
{
goto v___jp_2696_;
}
}
else
{
goto v___jp_2696_;
}
}
v___jp_2732_:
{
double v___x_2734_; double v___x_2735_; double v___x_2736_; uint8_t v___x_2737_; 
v___x_2734_ = lean_unbox_float(v_snd_2682_);
v___x_2735_ = lean_unbox_float(v_fst_2681_);
v___x_2736_ = lean_float_sub(v___x_2734_, v___x_2735_);
v___x_2737_ = lean_float_decLt(v___y_2733_, v___x_2736_);
v___y_2702_ = v___x_2737_;
goto v___jp_2701_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6___boxed(lean_object* v_cls_2748_, lean_object* v_collapsed_2749_, lean_object* v_tag_2750_, lean_object* v_opts_2751_, lean_object* v_clsEnabled_2752_, lean_object* v_oldTraces_2753_, lean_object* v_msg_2754_, lean_object* v_resStartStop_2755_, lean_object* v___y_2756_, lean_object* v___y_2757_, lean_object* v___y_2758_){
_start:
{
uint8_t v_collapsed_boxed_2759_; uint8_t v_clsEnabled_boxed_2760_; lean_object* v_res_2761_; 
v_collapsed_boxed_2759_ = lean_unbox(v_collapsed_2749_);
v_clsEnabled_boxed_2760_ = lean_unbox(v_clsEnabled_2752_);
v_res_2761_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6(v_cls_2748_, v_collapsed_boxed_2759_, v_tag_2750_, v_opts_2751_, v_clsEnabled_boxed_2760_, v_oldTraces_2753_, v_msg_2754_, v_resStartStop_2755_, v___y_2756_, v___y_2757_);
lean_dec(v___y_2757_);
lean_dec_ref(v___y_2756_);
lean_dec_ref(v_opts_2751_);
return v_res_2761_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4(size_t v_sz_2762_, size_t v_i_2763_, lean_object* v_bs_2764_){
_start:
{
uint8_t v___x_2765_; 
v___x_2765_ = lean_usize_dec_lt(v_i_2763_, v_sz_2762_);
if (v___x_2765_ == 0)
{
return v_bs_2764_;
}
else
{
lean_object* v_v_2766_; lean_object* v_tactic_2767_; lean_object* v_preGoal_2768_; lean_object* v_postGoals_2769_; lean_object* v_uTactic_2770_; lean_object* v___x_2772_; uint8_t v_isShared_2773_; uint8_t v_isSharedCheck_2798_; 
v_v_2766_ = lean_array_uget_borrowed(v_bs_2764_, v_i_2763_);
v_tactic_2767_ = lean_ctor_get(v_v_2766_, 2);
lean_inc_ref(v_tactic_2767_);
v_preGoal_2768_ = lean_ctor_get(v_v_2766_, 1);
lean_inc(v_preGoal_2768_);
v_postGoals_2769_ = lean_ctor_get(v_v_2766_, 4);
lean_inc_ref(v_postGoals_2769_);
v_uTactic_2770_ = lean_ctor_get(v_tactic_2767_, 0);
v_isSharedCheck_2798_ = !lean_is_exclusive(v_tactic_2767_);
if (v_isSharedCheck_2798_ == 0)
{
lean_object* v_unused_2799_; 
v_unused_2799_ = lean_ctor_get(v_tactic_2767_, 1);
lean_dec(v_unused_2799_);
v___x_2772_ = v_tactic_2767_;
v_isShared_2773_ = v_isSharedCheck_2798_;
goto v_resetjp_2771_;
}
else
{
lean_inc(v_uTactic_2770_);
lean_dec(v_tactic_2767_);
v___x_2772_ = lean_box(0);
v_isShared_2773_ = v_isSharedCheck_2798_;
goto v_resetjp_2771_;
}
v_resetjp_2771_:
{
size_t v_sz_2774_; lean_object* v___x_2775_; lean_object* v_bs_x27_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2780_; 
v_sz_2774_ = lean_array_size(v_postGoals_2769_);
v___x_2775_ = lean_unsigned_to_nat(0u);
v_bs_x27_2776_ = lean_array_uset(v_bs_2764_, v_i_2763_, v___x_2775_);
v___x_2777_ = l_Lean_MessageData_ofName(v_preGoal_2768_);
v___x_2778_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__5);
if (v_isShared_2773_ == 0)
{
lean_ctor_set_tag(v___x_2772_, 7);
lean_ctor_set(v___x_2772_, 1, v___x_2778_);
lean_ctor_set(v___x_2772_, 0, v___x_2777_);
v___x_2780_ = v___x_2772_;
goto v_reusejp_2779_;
}
else
{
lean_object* v_reuseFailAlloc_2797_; 
v_reuseFailAlloc_2797_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2797_, 0, v___x_2777_);
lean_ctor_set(v_reuseFailAlloc_2797_, 1, v___x_2778_);
v___x_2780_ = v_reuseFailAlloc_2797_;
goto v_reusejp_2779_;
}
v_reusejp_2779_:
{
size_t v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v___x_2792_; size_t v___x_2793_; size_t v___x_2794_; lean_object* v___x_2795_; 
v___x_2781_ = ((size_t)0ULL);
v___x_2782_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__0(v_sz_2774_, v___x_2781_, v_postGoals_2769_);
v___x_2783_ = lean_array_to_list(v___x_2782_);
v___x_2784_ = lean_box(0);
v___x_2785_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_StepTree_toMessageData_x3f_spec__1(v___x_2783_, v___x_2784_);
v___x_2786_ = l_Lean_MessageData_ofList(v___x_2785_);
v___x_2787_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2787_, 0, v___x_2780_);
lean_ctor_set(v___x_2787_, 1, v___x_2786_);
v___x_2788_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__7);
v___x_2789_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2789_, 0, v___x_2787_);
lean_ctor_set(v___x_2789_, 1, v___x_2788_);
v___x_2790_ = l_Lean_MessageData_ofSyntax(v_uTactic_2770_);
v___x_2791_ = l_Lean_indentD(v___x_2790_);
v___x_2792_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2792_, 0, v___x_2789_);
lean_ctor_set(v___x_2792_, 1, v___x_2791_);
v___x_2793_ = ((size_t)1ULL);
v___x_2794_ = lean_usize_add(v_i_2763_, v___x_2793_);
v___x_2795_ = lean_array_uset(v_bs_x27_2776_, v_i_2763_, v___x_2792_);
v_i_2763_ = v___x_2794_;
v_bs_2764_ = v___x_2795_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4___boxed(lean_object* v_sz_2800_, lean_object* v_i_2801_, lean_object* v_bs_2802_){
_start:
{
size_t v_sz_boxed_2803_; size_t v_i_boxed_2804_; lean_object* v_res_2805_; 
v_sz_boxed_2803_ = lean_unbox_usize(v_sz_2800_);
lean_dec(v_sz_2800_);
v_i_boxed_2804_ = lean_unbox_usize(v_i_2801_);
lean_dec(v_i_2801_);
v_res_2805_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4(v_sz_boxed_2803_, v_i_boxed_2804_, v_bs_2802_);
return v_res_2805_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1(void){
_start:
{
lean_object* v___x_2807_; lean_object* v___x_2808_; 
v___x_2807_ = ((lean_object*)(lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__0));
v___x_2808_ = l_Lean_stringToMessageData(v___x_2807_);
return v___x_2808_;
}
}
static double _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__5(void){
_start:
{
lean_object* v___x_2813_; double v___x_2814_; 
v___x_2813_ = lean_unsigned_to_nat(1000000000u);
v___x_2814_ = lean_float_of_nat(v___x_2813_);
return v___x_2814_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript(lean_object* v_uscript_2815_, lean_object* v_tacticState_2816_, lean_object* v_a_2817_, lean_object* v_a_2818_){
_start:
{
lean_object* v___y_2821_; lean_object* v___y_2822_; lean_object* v___y_2823_; lean_object* v___y_2824_; lean_object* v___y_2848_; lean_object* v___y_2849_; lean_object* v___y_2850_; lean_object* v___y_2851_; lean_object* v___y_2852_; lean_object* v___y_2853_; lean_object* v___y_2854_; lean_object* v___y_2873_; lean_object* v___y_2874_; lean_object* v___y_2875_; lean_object* v___y_2876_; lean_object* v___y_2900_; lean_object* v___y_2901_; lean_object* v___y_2902_; lean_object* v___y_2903_; lean_object* v___y_2904_; lean_object* v___y_2905_; lean_object* v___y_2906_; lean_object* v_options_2924_; lean_object* v_inheritedTraceOptions_2925_; uint8_t v_hasTrace_2926_; lean_object* v___x_2927_; lean_object* v___y_2929_; lean_object* v___y_2930_; lean_object* v___y_2931_; lean_object* v___y_2953_; lean_object* v___y_2954_; lean_object* v___y_2974_; lean_object* v___y_2975_; lean_object* v___y_2976_; lean_object* v___y_2998_; lean_object* v___y_2999_; 
v_options_2924_ = lean_ctor_get(v_a_2817_, 2);
v_inheritedTraceOptions_2925_ = lean_ctor_get(v_a_2817_, 13);
v_hasTrace_2926_ = lean_ctor_get_uint8(v_options_2924_, sizeof(void*)*1);
v___x_2927_ = lp_aesop_Aesop_TraceOption_script;
if (v_hasTrace_2926_ == 0)
{
lean_object* v___x_3018_; lean_object* v_a_3019_; uint8_t v___x_3020_; 
v___x_3018_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v_a_2817_);
v_a_3019_ = lean_ctor_get(v___x_3018_, 0);
lean_inc(v_a_3019_);
lean_dec_ref(v___x_3018_);
v___x_3020_ = lean_unbox(v_a_3019_);
lean_dec(v_a_3019_);
if (v___x_3020_ == 0)
{
v___y_2998_ = v_a_2817_;
v___y_2999_ = v_a_2818_;
goto v___jp_2997_;
}
else
{
lean_object* v_traceClass_3021_; lean_object* v___x_3022_; size_t v_sz_3023_; size_t v___x_3024_; lean_object* v___x_3025_; lean_object* v___x_3026_; lean_object* v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; 
v_traceClass_3021_ = lean_ctor_get(v___x_2927_, 0);
v___x_3022_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1);
v_sz_3023_ = lean_array_size(v_uscript_2815_);
v___x_3024_ = ((size_t)0ULL);
lean_inc_ref(v_uscript_2815_);
v___x_3025_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4(v_sz_3023_, v___x_3024_, v_uscript_2815_);
v___x_3026_ = lean_array_to_list(v___x_3025_);
v___x_3027_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10);
v___x_3028_ = l_Lean_MessageData_joinSep(v___x_3026_, v___x_3027_);
v___x_3029_ = l_Lean_indentD(v___x_3028_);
v___x_3030_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3030_, 0, v___x_3022_);
lean_ctor_set(v___x_3030_, 1, v___x_3029_);
lean_inc(v_traceClass_3021_);
v___x_3031_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_3021_, v___x_3030_, v_a_2817_, v_a_2818_);
if (lean_obj_tag(v___x_3031_) == 0)
{
lean_dec_ref_known(v___x_3031_, 1);
v___y_2998_ = v_a_2817_;
v___y_2999_ = v_a_2818_;
goto v___jp_2997_;
}
else
{
lean_object* v_a_3032_; lean_object* v___x_3034_; uint8_t v_isShared_3035_; uint8_t v_isSharedCheck_3039_; 
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_3032_ = lean_ctor_get(v___x_3031_, 0);
v_isSharedCheck_3039_ = !lean_is_exclusive(v___x_3031_);
if (v_isSharedCheck_3039_ == 0)
{
v___x_3034_ = v___x_3031_;
v_isShared_3035_ = v_isSharedCheck_3039_;
goto v_resetjp_3033_;
}
else
{
lean_inc(v_a_3032_);
lean_dec(v___x_3031_);
v___x_3034_ = lean_box(0);
v_isShared_3035_ = v_isSharedCheck_3039_;
goto v_resetjp_3033_;
}
v_resetjp_3033_:
{
lean_object* v___x_3037_; 
if (v_isShared_3035_ == 0)
{
v___x_3037_ = v___x_3034_;
goto v_reusejp_3036_;
}
else
{
lean_object* v_reuseFailAlloc_3038_; 
v_reuseFailAlloc_3038_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3038_, 0, v_a_3032_);
v___x_3037_ = v_reuseFailAlloc_3038_;
goto v_reusejp_3036_;
}
v_reusejp_3036_:
{
return v___x_3037_;
}
}
}
}
}
else
{
lean_object* v_traceClass_3040_; lean_object* v___f_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; uint8_t v___x_3045_; lean_object* v___y_3047_; lean_object* v___y_3048_; lean_object* v_a_3049_; lean_object* v___y_3062_; lean_object* v___y_3063_; lean_object* v_a_3064_; lean_object* v___y_3067_; lean_object* v___y_3068_; lean_object* v___y_3069_; lean_object* v___y_3080_; lean_object* v___y_3081_; lean_object* v_a_3082_; lean_object* v___y_3092_; lean_object* v___y_3093_; lean_object* v_a_3094_; lean_object* v___y_3097_; lean_object* v___y_3098_; lean_object* v___y_3099_; 
v_traceClass_3040_ = lean_ctor_get(v___x_2927_, 0);
v___f_3041_ = ((lean_object*)(lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__2));
v___x_3042_ = ((lean_object*)(lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__11));
v___x_3043_ = ((lean_object*)(lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__4));
lean_inc(v_traceClass_3040_);
v___x_3044_ = l_Lean_Name_append(v___x_3043_, v_traceClass_3040_);
v___x_3045_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2925_, v_options_2924_, v___x_3044_);
lean_dec(v___x_3044_);
if (v___x_3045_ == 0)
{
lean_object* v___x_3152_; uint8_t v___x_3153_; 
v___x_3152_ = l_Lean_trace_profiler;
v___x_3153_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(v_options_2924_, v___x_3152_);
if (v___x_3153_ == 0)
{
lean_object* v___x_3154_; lean_object* v_a_3155_; uint8_t v___x_3156_; 
v___x_3154_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v_a_2817_);
v_a_3155_ = lean_ctor_get(v___x_3154_, 0);
lean_inc(v_a_3155_);
lean_dec_ref(v___x_3154_);
v___x_3156_ = lean_unbox(v_a_3155_);
lean_dec(v_a_3155_);
if (v___x_3156_ == 0)
{
v___y_2953_ = v_a_2817_;
v___y_2954_ = v_a_2818_;
goto v___jp_2952_;
}
else
{
lean_object* v___x_3157_; size_t v_sz_3158_; size_t v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; 
v___x_3157_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1);
v_sz_3158_ = lean_array_size(v_uscript_2815_);
v___x_3159_ = ((size_t)0ULL);
lean_inc_ref(v_uscript_2815_);
v___x_3160_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4(v_sz_3158_, v___x_3159_, v_uscript_2815_);
v___x_3161_ = lean_array_to_list(v___x_3160_);
v___x_3162_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10);
v___x_3163_ = l_Lean_MessageData_joinSep(v___x_3161_, v___x_3162_);
v___x_3164_ = l_Lean_indentD(v___x_3163_);
v___x_3165_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3165_, 0, v___x_3157_);
lean_ctor_set(v___x_3165_, 1, v___x_3164_);
lean_inc(v_traceClass_3040_);
v___x_3166_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_3040_, v___x_3165_, v_a_2817_, v_a_2818_);
if (lean_obj_tag(v___x_3166_) == 0)
{
lean_dec_ref_known(v___x_3166_, 1);
v___y_2953_ = v_a_2817_;
v___y_2954_ = v_a_2818_;
goto v___jp_2952_;
}
else
{
lean_object* v_a_3167_; lean_object* v___x_3169_; uint8_t v_isShared_3170_; uint8_t v_isSharedCheck_3174_; 
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_3167_ = lean_ctor_get(v___x_3166_, 0);
v_isSharedCheck_3174_ = !lean_is_exclusive(v___x_3166_);
if (v_isSharedCheck_3174_ == 0)
{
v___x_3169_ = v___x_3166_;
v_isShared_3170_ = v_isSharedCheck_3174_;
goto v_resetjp_3168_;
}
else
{
lean_inc(v_a_3167_);
lean_dec(v___x_3166_);
v___x_3169_ = lean_box(0);
v_isShared_3170_ = v_isSharedCheck_3174_;
goto v_resetjp_3168_;
}
v_resetjp_3168_:
{
lean_object* v___x_3172_; 
if (v_isShared_3170_ == 0)
{
v___x_3172_ = v___x_3169_;
goto v_reusejp_3171_;
}
else
{
lean_object* v_reuseFailAlloc_3173_; 
v_reuseFailAlloc_3173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3173_, 0, v_a_3167_);
v___x_3172_ = v_reuseFailAlloc_3173_;
goto v_reusejp_3171_;
}
v_reusejp_3171_:
{
return v___x_3172_;
}
}
}
}
}
else
{
goto v___jp_3109_;
}
}
else
{
goto v___jp_3109_;
}
v___jp_3046_:
{
lean_object* v___x_3050_; double v___x_3051_; double v___x_3052_; double v___x_3053_; double v___x_3054_; double v___x_3055_; lean_object* v___x_3056_; lean_object* v___x_3057_; lean_object* v___x_3058_; lean_object* v___x_3059_; lean_object* v___x_3060_; 
v___x_3050_ = lean_io_mono_nanos_now();
v___x_3051_ = lean_float_of_nat(v___y_3048_);
v___x_3052_ = lean_float_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__5, &lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__5_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__5);
v___x_3053_ = lean_float_div(v___x_3051_, v___x_3052_);
v___x_3054_ = lean_float_of_nat(v___x_3050_);
v___x_3055_ = lean_float_div(v___x_3054_, v___x_3052_);
v___x_3056_ = lean_box_float(v___x_3053_);
v___x_3057_ = lean_box_float(v___x_3055_);
v___x_3058_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3058_, 0, v___x_3056_);
lean_ctor_set(v___x_3058_, 1, v___x_3057_);
v___x_3059_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3059_, 0, v_a_3049_);
lean_ctor_set(v___x_3059_, 1, v___x_3058_);
lean_inc(v_traceClass_3040_);
v___x_3060_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6(v_traceClass_3040_, v_hasTrace_2926_, v___x_3042_, v_options_2924_, v___x_3045_, v___y_3047_, v___f_3041_, v___x_3059_, v_a_2817_, v_a_2818_);
return v___x_3060_;
}
v___jp_3061_:
{
lean_object* v___x_3065_; 
v___x_3065_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3065_, 0, v_a_3064_);
v___y_3047_ = v___y_3062_;
v___y_3048_ = v___y_3063_;
v_a_3049_ = v___x_3065_;
goto v___jp_3046_;
}
v___jp_3066_:
{
if (lean_obj_tag(v___y_3069_) == 0)
{
lean_object* v_a_3070_; lean_object* v___x_3072_; uint8_t v_isShared_3073_; uint8_t v_isSharedCheck_3077_; 
v_a_3070_ = lean_ctor_get(v___y_3069_, 0);
v_isSharedCheck_3077_ = !lean_is_exclusive(v___y_3069_);
if (v_isSharedCheck_3077_ == 0)
{
v___x_3072_ = v___y_3069_;
v_isShared_3073_ = v_isSharedCheck_3077_;
goto v_resetjp_3071_;
}
else
{
lean_inc(v_a_3070_);
lean_dec(v___y_3069_);
v___x_3072_ = lean_box(0);
v_isShared_3073_ = v_isSharedCheck_3077_;
goto v_resetjp_3071_;
}
v_resetjp_3071_:
{
lean_object* v___x_3075_; 
if (v_isShared_3073_ == 0)
{
lean_ctor_set_tag(v___x_3072_, 1);
v___x_3075_ = v___x_3072_;
goto v_reusejp_3074_;
}
else
{
lean_object* v_reuseFailAlloc_3076_; 
v_reuseFailAlloc_3076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3076_, 0, v_a_3070_);
v___x_3075_ = v_reuseFailAlloc_3076_;
goto v_reusejp_3074_;
}
v_reusejp_3074_:
{
v___y_3047_ = v___y_3067_;
v___y_3048_ = v___y_3068_;
v_a_3049_ = v___x_3075_;
goto v___jp_3046_;
}
}
}
else
{
lean_object* v_a_3078_; 
v_a_3078_ = lean_ctor_get(v___y_3069_, 0);
lean_inc(v_a_3078_);
lean_dec_ref_known(v___y_3069_, 1);
v___y_3062_ = v___y_3067_;
v___y_3063_ = v___y_3068_;
v_a_3064_ = v_a_3078_;
goto v___jp_3061_;
}
}
v___jp_3079_:
{
lean_object* v___x_3083_; double v___x_3084_; double v___x_3085_; lean_object* v___x_3086_; lean_object* v___x_3087_; lean_object* v___x_3088_; lean_object* v___x_3089_; lean_object* v___x_3090_; 
v___x_3083_ = lean_io_get_num_heartbeats();
v___x_3084_ = lean_float_of_nat(v___y_3081_);
v___x_3085_ = lean_float_of_nat(v___x_3083_);
v___x_3086_ = lean_box_float(v___x_3084_);
v___x_3087_ = lean_box_float(v___x_3085_);
v___x_3088_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3088_, 0, v___x_3086_);
lean_ctor_set(v___x_3088_, 1, v___x_3087_);
v___x_3089_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3089_, 0, v_a_3082_);
lean_ctor_set(v___x_3089_, 1, v___x_3088_);
lean_inc(v_traceClass_3040_);
v___x_3090_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6(v_traceClass_3040_, v_hasTrace_2926_, v___x_3042_, v_options_2924_, v___x_3045_, v___y_3080_, v___f_3041_, v___x_3089_, v_a_2817_, v_a_2818_);
return v___x_3090_;
}
v___jp_3091_:
{
lean_object* v___x_3095_; 
v___x_3095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3095_, 0, v_a_3094_);
v___y_3080_ = v___y_3092_;
v___y_3081_ = v___y_3093_;
v_a_3082_ = v___x_3095_;
goto v___jp_3079_;
}
v___jp_3096_:
{
if (lean_obj_tag(v___y_3099_) == 0)
{
lean_object* v_a_3100_; lean_object* v___x_3102_; uint8_t v_isShared_3103_; uint8_t v_isSharedCheck_3107_; 
v_a_3100_ = lean_ctor_get(v___y_3099_, 0);
v_isSharedCheck_3107_ = !lean_is_exclusive(v___y_3099_);
if (v_isSharedCheck_3107_ == 0)
{
v___x_3102_ = v___y_3099_;
v_isShared_3103_ = v_isSharedCheck_3107_;
goto v_resetjp_3101_;
}
else
{
lean_inc(v_a_3100_);
lean_dec(v___y_3099_);
v___x_3102_ = lean_box(0);
v_isShared_3103_ = v_isSharedCheck_3107_;
goto v_resetjp_3101_;
}
v_resetjp_3101_:
{
lean_object* v___x_3105_; 
if (v_isShared_3103_ == 0)
{
lean_ctor_set_tag(v___x_3102_, 1);
v___x_3105_ = v___x_3102_;
goto v_reusejp_3104_;
}
else
{
lean_object* v_reuseFailAlloc_3106_; 
v_reuseFailAlloc_3106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3106_, 0, v_a_3100_);
v___x_3105_ = v_reuseFailAlloc_3106_;
goto v_reusejp_3104_;
}
v_reusejp_3104_:
{
v___y_3080_ = v___y_3097_;
v___y_3081_ = v___y_3098_;
v_a_3082_ = v___x_3105_;
goto v___jp_3079_;
}
}
}
else
{
lean_object* v_a_3108_; 
v_a_3108_ = lean_ctor_get(v___y_3099_, 0);
lean_inc(v_a_3108_);
lean_dec_ref_known(v___y_3099_, 1);
v___y_3092_ = v___y_3097_;
v___y_3093_ = v___y_3098_;
v_a_3094_ = v_a_3108_;
goto v___jp_3091_;
}
}
v___jp_3109_:
{
lean_object* v___x_3110_; lean_object* v_a_3111_; lean_object* v___x_3112_; uint8_t v___x_3113_; 
v___x_3110_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Script_orderedUScriptToSScript_spec__5___redArg(v_a_2818_);
v_a_3111_ = lean_ctor_get(v___x_3110_, 0);
lean_inc(v_a_3111_);
lean_dec_ref(v___x_3110_);
v___x_3112_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3113_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0_spec__0(v_options_2924_, v___x_3112_);
if (v___x_3113_ == 0)
{
lean_object* v___x_3114_; lean_object* v___x_3115_; lean_object* v_a_3116_; uint8_t v___x_3117_; 
v___x_3114_ = lean_io_mono_nanos_now();
v___x_3115_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v_a_2817_);
v_a_3116_ = lean_ctor_get(v___x_3115_, 0);
lean_inc(v_a_3116_);
lean_dec_ref(v___x_3115_);
v___x_3117_ = lean_unbox(v_a_3116_);
lean_dec(v_a_3116_);
if (v___x_3117_ == 0)
{
lean_object* v___x_3118_; lean_object* v___x_3119_; 
v___x_3118_ = lean_box(0);
v___x_3119_ = lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1(v___x_2927_, v_uscript_2815_, v_tacticState_2816_, v___x_3118_, v_a_2817_, v_a_2818_);
lean_dec_ref(v_uscript_2815_);
v___y_3067_ = v_a_3111_;
v___y_3068_ = v___x_3114_;
v___y_3069_ = v___x_3119_;
goto v___jp_3066_;
}
else
{
lean_object* v___x_3120_; size_t v_sz_3121_; size_t v___x_3122_; lean_object* v___x_3123_; lean_object* v___x_3124_; lean_object* v___x_3125_; lean_object* v___x_3126_; lean_object* v___x_3127_; lean_object* v___x_3128_; lean_object* v___x_3129_; 
v___x_3120_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1);
v_sz_3121_ = lean_array_size(v_uscript_2815_);
v___x_3122_ = ((size_t)0ULL);
lean_inc_ref(v_uscript_2815_);
v___x_3123_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4(v_sz_3121_, v___x_3122_, v_uscript_2815_);
v___x_3124_ = lean_array_to_list(v___x_3123_);
v___x_3125_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10);
v___x_3126_ = l_Lean_MessageData_joinSep(v___x_3124_, v___x_3125_);
v___x_3127_ = l_Lean_indentD(v___x_3126_);
v___x_3128_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3128_, 0, v___x_3120_);
lean_ctor_set(v___x_3128_, 1, v___x_3127_);
lean_inc(v_traceClass_3040_);
v___x_3129_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_3040_, v___x_3128_, v_a_2817_, v_a_2818_);
if (lean_obj_tag(v___x_3129_) == 0)
{
lean_object* v_a_3130_; lean_object* v___x_3131_; 
v_a_3130_ = lean_ctor_get(v___x_3129_, 0);
lean_inc(v_a_3130_);
lean_dec_ref_known(v___x_3129_, 1);
v___x_3131_ = lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1(v___x_2927_, v_uscript_2815_, v_tacticState_2816_, v_a_3130_, v_a_2817_, v_a_2818_);
lean_dec_ref(v_uscript_2815_);
v___y_3067_ = v_a_3111_;
v___y_3068_ = v___x_3114_;
v___y_3069_ = v___x_3131_;
goto v___jp_3066_;
}
else
{
lean_object* v_a_3132_; 
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_3132_ = lean_ctor_get(v___x_3129_, 0);
lean_inc(v_a_3132_);
lean_dec_ref_known(v___x_3129_, 1);
v___y_3062_ = v_a_3111_;
v___y_3063_ = v___x_3114_;
v_a_3064_ = v_a_3132_;
goto v___jp_3061_;
}
}
}
else
{
lean_object* v___x_3133_; lean_object* v___x_3134_; lean_object* v_a_3135_; uint8_t v___x_3136_; 
v___x_3133_ = lean_io_get_num_heartbeats();
v___x_3134_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v_a_2817_);
v_a_3135_ = lean_ctor_get(v___x_3134_, 0);
lean_inc(v_a_3135_);
lean_dec_ref(v___x_3134_);
v___x_3136_ = lean_unbox(v_a_3135_);
lean_dec(v_a_3135_);
if (v___x_3136_ == 0)
{
lean_object* v___x_3137_; lean_object* v___x_3138_; 
v___x_3137_ = lean_box(0);
v___x_3138_ = lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__2(v___x_2927_, v_uscript_2815_, v_tacticState_2816_, v___x_3137_, v_a_2817_, v_a_2818_);
lean_dec_ref(v_uscript_2815_);
v___y_3097_ = v_a_3111_;
v___y_3098_ = v___x_3133_;
v___y_3099_ = v___x_3138_;
goto v___jp_3096_;
}
else
{
lean_object* v___x_3139_; size_t v_sz_3140_; size_t v___x_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; lean_object* v___x_3144_; lean_object* v___x_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; 
v___x_3139_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___closed__1);
v_sz_3140_ = lean_array_size(v_uscript_2815_);
v___x_3141_ = ((size_t)0ULL);
lean_inc_ref(v_uscript_2815_);
v___x_3142_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__4(v_sz_3140_, v___x_3141_, v_uscript_2815_);
v___x_3143_ = lean_array_to_list(v___x_3142_);
v___x_3144_ = lean_obj_once(&lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10, &lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10_once, _init_lp_aesop_Aesop_Script_StepTree_toMessageData_x3f___closed__10);
v___x_3145_ = l_Lean_MessageData_joinSep(v___x_3143_, v___x_3144_);
v___x_3146_ = l_Lean_indentD(v___x_3145_);
v___x_3147_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3147_, 0, v___x_3139_);
lean_ctor_set(v___x_3147_, 1, v___x_3146_);
lean_inc(v_traceClass_3040_);
v___x_3148_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_3040_, v___x_3147_, v_a_2817_, v_a_2818_);
if (lean_obj_tag(v___x_3148_) == 0)
{
lean_object* v_a_3149_; lean_object* v___x_3150_; 
v_a_3149_ = lean_ctor_get(v___x_3148_, 0);
lean_inc(v_a_3149_);
lean_dec_ref_known(v___x_3148_, 1);
v___x_3150_ = lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__2(v___x_2927_, v_uscript_2815_, v_tacticState_2816_, v_a_3149_, v_a_2817_, v_a_2818_);
lean_dec_ref(v_uscript_2815_);
v___y_3097_ = v_a_3111_;
v___y_3098_ = v___x_3133_;
v___y_3099_ = v___x_3150_;
goto v___jp_3096_;
}
else
{
lean_object* v_a_3151_; 
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_3151_ = lean_ctor_get(v___x_3148_, 0);
lean_inc(v_a_3151_);
lean_dec_ref_known(v___x_3148_, 1);
v___y_3092_ = v_a_3111_;
v___y_3093_ = v___x_3133_;
v_a_3094_ = v_a_3151_;
goto v___jp_3091_;
}
}
}
}
}
v___jp_2820_:
{
lean_object* v___x_2825_; lean_object* v___x_2826_; lean_object* v___x_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; 
v___x_2825_ = lean_unsigned_to_nat(0u);
v___x_2826_ = lean_array_get_size(v_uscript_2815_);
v___x_2827_ = lean_unsigned_to_nat(1u);
v___x_2828_ = lean_nat_sub(v___x_2826_, v___x_2827_);
v___x_2829_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_2815_, v___y_2821_, v___y_2822_, v___x_2825_, v___x_2828_, v_tacticState_2816_, v___y_2823_, v___y_2824_);
lean_dec(v___x_2828_);
lean_dec_ref(v___y_2822_);
lean_dec_ref(v___y_2821_);
lean_dec_ref(v_uscript_2815_);
if (lean_obj_tag(v___x_2829_) == 0)
{
lean_object* v_a_2830_; lean_object* v___x_2832_; uint8_t v_isShared_2833_; uint8_t v_isSharedCheck_2838_; 
v_a_2830_ = lean_ctor_get(v___x_2829_, 0);
v_isSharedCheck_2838_ = !lean_is_exclusive(v___x_2829_);
if (v_isSharedCheck_2838_ == 0)
{
v___x_2832_ = v___x_2829_;
v_isShared_2833_ = v_isSharedCheck_2838_;
goto v_resetjp_2831_;
}
else
{
lean_inc(v_a_2830_);
lean_dec(v___x_2829_);
v___x_2832_ = lean_box(0);
v_isShared_2833_ = v_isSharedCheck_2838_;
goto v_resetjp_2831_;
}
v_resetjp_2831_:
{
lean_object* v_fst_2834_; lean_object* v___x_2836_; 
v_fst_2834_ = lean_ctor_get(v_a_2830_, 0);
lean_inc(v_fst_2834_);
lean_dec(v_a_2830_);
if (v_isShared_2833_ == 0)
{
lean_ctor_set(v___x_2832_, 0, v_fst_2834_);
v___x_2836_ = v___x_2832_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2837_; 
v_reuseFailAlloc_2837_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2837_, 0, v_fst_2834_);
v___x_2836_ = v_reuseFailAlloc_2837_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
return v___x_2836_;
}
}
}
else
{
lean_object* v_a_2839_; lean_object* v___x_2841_; uint8_t v_isShared_2842_; uint8_t v_isSharedCheck_2846_; 
v_a_2839_ = lean_ctor_get(v___x_2829_, 0);
v_isSharedCheck_2846_ = !lean_is_exclusive(v___x_2829_);
if (v_isSharedCheck_2846_ == 0)
{
v___x_2841_ = v___x_2829_;
v_isShared_2842_ = v_isSharedCheck_2846_;
goto v_resetjp_2840_;
}
else
{
lean_inc(v_a_2839_);
lean_dec(v___x_2829_);
v___x_2841_ = lean_box(0);
v_isShared_2842_ = v_isSharedCheck_2846_;
goto v_resetjp_2840_;
}
v_resetjp_2840_:
{
lean_object* v___x_2844_; 
if (v_isShared_2842_ == 0)
{
v___x_2844_ = v___x_2841_;
goto v_reusejp_2843_;
}
else
{
lean_object* v_reuseFailAlloc_2845_; 
v_reuseFailAlloc_2845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2845_, 0, v_a_2839_);
v___x_2844_ = v_reuseFailAlloc_2845_;
goto v_reusejp_2843_;
}
v_reusejp_2843_:
{
return v___x_2844_;
}
}
}
}
v___jp_2847_:
{
size_t v_sz_2855_; size_t v___x_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; 
v_sz_2855_ = lean_array_size(v___y_2854_);
v___x_2856_ = ((size_t)0ULL);
v___x_2857_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0(v_sz_2855_, v___x_2856_, v___y_2854_);
v___x_2858_ = lean_array_to_list(v___x_2857_);
v___x_2859_ = lean_box(0);
v___x_2860_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1(v___x_2858_, v___x_2859_);
v___x_2861_ = l_Lean_MessageData_ofList(v___x_2860_);
lean_inc_ref(v___y_2850_);
v___x_2862_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2862_, 0, v___y_2850_);
lean_ctor_set(v___x_2862_, 1, v___x_2861_);
v___x_2863_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v___y_2848_, v___x_2862_, v___y_2853_, v___y_2851_);
if (lean_obj_tag(v___x_2863_) == 0)
{
lean_dec_ref_known(v___x_2863_, 1);
v___y_2821_ = v___y_2849_;
v___y_2822_ = v___y_2852_;
v___y_2823_ = v___y_2853_;
v___y_2824_ = v___y_2851_;
goto v___jp_2820_;
}
else
{
lean_object* v_a_2864_; lean_object* v___x_2866_; uint8_t v_isShared_2867_; uint8_t v_isSharedCheck_2871_; 
lean_dec_ref(v___y_2852_);
lean_dec_ref(v___y_2849_);
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_2864_ = lean_ctor_get(v___x_2863_, 0);
v_isSharedCheck_2871_ = !lean_is_exclusive(v___x_2863_);
if (v_isSharedCheck_2871_ == 0)
{
v___x_2866_ = v___x_2863_;
v_isShared_2867_ = v_isSharedCheck_2871_;
goto v_resetjp_2865_;
}
else
{
lean_inc(v_a_2864_);
lean_dec(v___x_2863_);
v___x_2866_ = lean_box(0);
v_isShared_2867_ = v_isSharedCheck_2871_;
goto v_resetjp_2865_;
}
v_resetjp_2865_:
{
lean_object* v___x_2869_; 
if (v_isShared_2867_ == 0)
{
v___x_2869_ = v___x_2866_;
goto v_reusejp_2868_;
}
else
{
lean_object* v_reuseFailAlloc_2870_; 
v_reuseFailAlloc_2870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2870_, 0, v_a_2864_);
v___x_2869_ = v_reuseFailAlloc_2870_;
goto v_reusejp_2868_;
}
v_reusejp_2868_:
{
return v___x_2869_;
}
}
}
}
v___jp_2872_:
{
lean_object* v___x_2877_; lean_object* v___x_2878_; lean_object* v___x_2879_; lean_object* v___x_2880_; lean_object* v___x_2881_; 
v___x_2877_ = lean_unsigned_to_nat(0u);
v___x_2878_ = lean_array_get_size(v_uscript_2815_);
v___x_2879_ = lean_unsigned_to_nat(1u);
v___x_2880_ = lean_nat_sub(v___x_2878_, v___x_2879_);
v___x_2881_ = lp_aesop___private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go(v_uscript_2815_, v___y_2873_, v___y_2874_, v___x_2877_, v___x_2880_, v_tacticState_2816_, v___y_2875_, v___y_2876_);
lean_dec(v___x_2880_);
lean_dec_ref(v___y_2874_);
lean_dec_ref(v___y_2873_);
lean_dec_ref(v_uscript_2815_);
if (lean_obj_tag(v___x_2881_) == 0)
{
lean_object* v_a_2882_; lean_object* v___x_2884_; uint8_t v_isShared_2885_; uint8_t v_isSharedCheck_2890_; 
v_a_2882_ = lean_ctor_get(v___x_2881_, 0);
v_isSharedCheck_2890_ = !lean_is_exclusive(v___x_2881_);
if (v_isSharedCheck_2890_ == 0)
{
v___x_2884_ = v___x_2881_;
v_isShared_2885_ = v_isSharedCheck_2890_;
goto v_resetjp_2883_;
}
else
{
lean_inc(v_a_2882_);
lean_dec(v___x_2881_);
v___x_2884_ = lean_box(0);
v_isShared_2885_ = v_isSharedCheck_2890_;
goto v_resetjp_2883_;
}
v_resetjp_2883_:
{
lean_object* v_fst_2886_; lean_object* v___x_2888_; 
v_fst_2886_ = lean_ctor_get(v_a_2882_, 0);
lean_inc(v_fst_2886_);
lean_dec(v_a_2882_);
if (v_isShared_2885_ == 0)
{
lean_ctor_set(v___x_2884_, 0, v_fst_2886_);
v___x_2888_ = v___x_2884_;
goto v_reusejp_2887_;
}
else
{
lean_object* v_reuseFailAlloc_2889_; 
v_reuseFailAlloc_2889_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2889_, 0, v_fst_2886_);
v___x_2888_ = v_reuseFailAlloc_2889_;
goto v_reusejp_2887_;
}
v_reusejp_2887_:
{
return v___x_2888_;
}
}
}
else
{
lean_object* v_a_2891_; lean_object* v___x_2893_; uint8_t v_isShared_2894_; uint8_t v_isSharedCheck_2898_; 
v_a_2891_ = lean_ctor_get(v___x_2881_, 0);
v_isSharedCheck_2898_ = !lean_is_exclusive(v___x_2881_);
if (v_isSharedCheck_2898_ == 0)
{
v___x_2893_ = v___x_2881_;
v_isShared_2894_ = v_isSharedCheck_2898_;
goto v_resetjp_2892_;
}
else
{
lean_inc(v_a_2891_);
lean_dec(v___x_2881_);
v___x_2893_ = lean_box(0);
v_isShared_2894_ = v_isSharedCheck_2898_;
goto v_resetjp_2892_;
}
v_resetjp_2892_:
{
lean_object* v___x_2896_; 
if (v_isShared_2894_ == 0)
{
v___x_2896_ = v___x_2893_;
goto v_reusejp_2895_;
}
else
{
lean_object* v_reuseFailAlloc_2897_; 
v_reuseFailAlloc_2897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2897_, 0, v_a_2891_);
v___x_2896_ = v_reuseFailAlloc_2897_;
goto v_reusejp_2895_;
}
v_reusejp_2895_:
{
return v___x_2896_;
}
}
}
}
v___jp_2899_:
{
size_t v_sz_2907_; size_t v___x_2908_; lean_object* v___x_2909_; lean_object* v___x_2910_; lean_object* v___x_2911_; lean_object* v___x_2912_; lean_object* v___x_2913_; lean_object* v___x_2914_; lean_object* v___x_2915_; 
v_sz_2907_ = lean_array_size(v___y_2906_);
v___x_2908_ = ((size_t)0ULL);
v___x_2909_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_orderedUScriptToSScript_spec__0(v_sz_2907_, v___x_2908_, v___y_2906_);
v___x_2910_ = lean_array_to_list(v___x_2909_);
v___x_2911_ = lean_box(0);
v___x_2912_ = lp_aesop_List_mapTR_loop___at___00Aesop_Script_orderedUScriptToSScript_spec__1(v___x_2910_, v___x_2911_);
v___x_2913_ = l_Lean_MessageData_ofList(v___x_2912_);
lean_inc_ref(v___y_2900_);
v___x_2914_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2914_, 0, v___y_2900_);
lean_ctor_set(v___x_2914_, 1, v___x_2913_);
v___x_2915_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v___y_2902_, v___x_2914_, v___y_2905_, v___y_2901_);
if (lean_obj_tag(v___x_2915_) == 0)
{
lean_dec_ref_known(v___x_2915_, 1);
v___y_2873_ = v___y_2903_;
v___y_2874_ = v___y_2904_;
v___y_2875_ = v___y_2905_;
v___y_2876_ = v___y_2901_;
goto v___jp_2872_;
}
else
{
lean_object* v_a_2916_; lean_object* v___x_2918_; uint8_t v_isShared_2919_; uint8_t v_isSharedCheck_2923_; 
lean_dec_ref(v___y_2904_);
lean_dec_ref(v___y_2903_);
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_2916_ = lean_ctor_get(v___x_2915_, 0);
v_isSharedCheck_2923_ = !lean_is_exclusive(v___x_2915_);
if (v_isSharedCheck_2923_ == 0)
{
v___x_2918_ = v___x_2915_;
v_isShared_2919_ = v_isSharedCheck_2923_;
goto v_resetjp_2917_;
}
else
{
lean_inc(v_a_2916_);
lean_dec(v___x_2915_);
v___x_2918_ = lean_box(0);
v_isShared_2919_ = v_isSharedCheck_2923_;
goto v_resetjp_2917_;
}
v_resetjp_2917_:
{
lean_object* v___x_2921_; 
if (v_isShared_2919_ == 0)
{
v___x_2921_ = v___x_2918_;
goto v_reusejp_2920_;
}
else
{
lean_object* v_reuseFailAlloc_2922_; 
v_reuseFailAlloc_2922_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2922_, 0, v_a_2916_);
v___x_2921_ = v_reuseFailAlloc_2922_;
goto v_reusejp_2920_;
}
v_reusejp_2920_:
{
return v___x_2921_;
}
}
}
}
v___jp_2928_:
{
lean_object* v___x_2932_; lean_object* v_a_2933_; lean_object* v___x_2934_; lean_object* v___x_2935_; uint8_t v___x_2936_; 
v___x_2932_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v___y_2930_);
v_a_2933_ = lean_ctor_get(v___x_2932_, 0);
lean_inc(v_a_2933_);
lean_dec_ref(v___x_2932_);
lean_inc(v___y_2929_);
v___x_2934_ = lp_aesop_Aesop_Script_StepTree_focusableGoals(v___y_2929_);
v___x_2935_ = lp_aesop_Aesop_Script_StepTree_numSiblings(v___y_2929_);
v___x_2936_ = lean_unbox(v_a_2933_);
lean_dec(v_a_2933_);
if (v___x_2936_ == 0)
{
v___y_2821_ = v___x_2934_;
v___y_2822_ = v___x_2935_;
v___y_2823_ = v___y_2930_;
v___y_2824_ = v___y_2931_;
goto v___jp_2820_;
}
else
{
lean_object* v_traceClass_2937_; lean_object* v_size_2938_; lean_object* v_buckets_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; uint8_t v___x_2944_; 
v_traceClass_2937_ = lean_ctor_get(v___x_2927_, 0);
v_size_2938_ = lean_ctor_get(v___x_2934_, 0);
lean_inc(v_size_2938_);
v_buckets_2939_ = lean_ctor_get(v___x_2934_, 1);
lean_inc_ref(v_buckets_2939_);
v___x_2940_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1);
v___x_2941_ = lean_mk_empty_array_with_capacity(v_size_2938_);
lean_dec(v_size_2938_);
v___x_2942_ = lean_unsigned_to_nat(0u);
v___x_2943_ = lean_array_get_size(v_buckets_2939_);
v___x_2944_ = lean_nat_dec_lt(v___x_2942_, v___x_2943_);
if (v___x_2944_ == 0)
{
lean_dec_ref(v_buckets_2939_);
lean_inc(v_traceClass_2937_);
v___y_2848_ = v_traceClass_2937_;
v___y_2849_ = v___x_2934_;
v___y_2850_ = v___x_2940_;
v___y_2851_ = v___y_2931_;
v___y_2852_ = v___x_2935_;
v___y_2853_ = v___y_2930_;
v___y_2854_ = v___x_2941_;
goto v___jp_2847_;
}
else
{
uint8_t v___x_2945_; 
v___x_2945_ = lean_nat_dec_le(v___x_2943_, v___x_2943_);
if (v___x_2945_ == 0)
{
if (v___x_2944_ == 0)
{
lean_dec_ref(v_buckets_2939_);
lean_inc(v_traceClass_2937_);
v___y_2848_ = v_traceClass_2937_;
v___y_2849_ = v___x_2934_;
v___y_2850_ = v___x_2940_;
v___y_2851_ = v___y_2931_;
v___y_2852_ = v___x_2935_;
v___y_2853_ = v___y_2930_;
v___y_2854_ = v___x_2941_;
goto v___jp_2847_;
}
else
{
size_t v___x_2946_; size_t v___x_2947_; lean_object* v___x_2948_; 
v___x_2946_ = ((size_t)0ULL);
v___x_2947_ = lean_usize_of_nat(v___x_2943_);
v___x_2948_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2939_, v___x_2946_, v___x_2947_, v___x_2941_);
lean_dec_ref(v_buckets_2939_);
lean_inc(v_traceClass_2937_);
v___y_2848_ = v_traceClass_2937_;
v___y_2849_ = v___x_2934_;
v___y_2850_ = v___x_2940_;
v___y_2851_ = v___y_2931_;
v___y_2852_ = v___x_2935_;
v___y_2853_ = v___y_2930_;
v___y_2854_ = v___x_2948_;
goto v___jp_2847_;
}
}
else
{
size_t v___x_2949_; size_t v___x_2950_; lean_object* v___x_2951_; 
v___x_2949_ = ((size_t)0ULL);
v___x_2950_ = lean_usize_of_nat(v___x_2943_);
v___x_2951_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2939_, v___x_2949_, v___x_2950_, v___x_2941_);
lean_dec_ref(v_buckets_2939_);
lean_inc(v_traceClass_2937_);
v___y_2848_ = v_traceClass_2937_;
v___y_2849_ = v___x_2934_;
v___y_2850_ = v___x_2940_;
v___y_2851_ = v___y_2931_;
v___y_2852_ = v___x_2935_;
v___y_2853_ = v___y_2930_;
v___y_2854_ = v___x_2951_;
goto v___jp_2847_;
}
}
}
}
v___jp_2952_:
{
lean_object* v___x_2955_; lean_object* v_a_2956_; lean_object* v___x_2957_; uint8_t v___x_2958_; 
v___x_2955_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v___y_2953_);
v_a_2956_ = lean_ctor_get(v___x_2955_, 0);
lean_inc(v_a_2956_);
lean_dec_ref(v___x_2955_);
v___x_2957_ = lp_aesop_Aesop_Script_UScript_toStepTree(v_uscript_2815_);
v___x_2958_ = lean_unbox(v_a_2956_);
lean_dec(v_a_2956_);
if (v___x_2958_ == 0)
{
v___y_2929_ = v___x_2957_;
v___y_2930_ = v___y_2953_;
v___y_2931_ = v___y_2954_;
goto v___jp_2928_;
}
else
{
lean_object* v_traceClass_2959_; lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v___x_2962_; lean_object* v___x_2963_; lean_object* v___x_2964_; 
v_traceClass_2959_ = lean_ctor_get(v___x_2927_, 0);
v___x_2960_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3);
lean_inc(v___x_2957_);
v___x_2961_ = lp_aesop_Aesop_Script_StepTree_toMessageData(v___x_2957_);
v___x_2962_ = l_Lean_indentD(v___x_2961_);
v___x_2963_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2963_, 0, v___x_2960_);
lean_ctor_set(v___x_2963_, 1, v___x_2962_);
lean_inc(v_traceClass_2959_);
v___x_2964_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_2959_, v___x_2963_, v___y_2953_, v___y_2954_);
if (lean_obj_tag(v___x_2964_) == 0)
{
lean_dec_ref_known(v___x_2964_, 1);
v___y_2929_ = v___x_2957_;
v___y_2930_ = v___y_2953_;
v___y_2931_ = v___y_2954_;
goto v___jp_2928_;
}
else
{
lean_object* v_a_2965_; lean_object* v___x_2967_; uint8_t v_isShared_2968_; uint8_t v_isSharedCheck_2972_; 
lean_dec(v___x_2957_);
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_2965_ = lean_ctor_get(v___x_2964_, 0);
v_isSharedCheck_2972_ = !lean_is_exclusive(v___x_2964_);
if (v_isSharedCheck_2972_ == 0)
{
v___x_2967_ = v___x_2964_;
v_isShared_2968_ = v_isSharedCheck_2972_;
goto v_resetjp_2966_;
}
else
{
lean_inc(v_a_2965_);
lean_dec(v___x_2964_);
v___x_2967_ = lean_box(0);
v_isShared_2968_ = v_isSharedCheck_2972_;
goto v_resetjp_2966_;
}
v_resetjp_2966_:
{
lean_object* v___x_2970_; 
if (v_isShared_2968_ == 0)
{
v___x_2970_ = v___x_2967_;
goto v_reusejp_2969_;
}
else
{
lean_object* v_reuseFailAlloc_2971_; 
v_reuseFailAlloc_2971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2971_, 0, v_a_2965_);
v___x_2970_ = v_reuseFailAlloc_2971_;
goto v_reusejp_2969_;
}
v_reusejp_2969_:
{
return v___x_2970_;
}
}
}
}
}
v___jp_2973_:
{
lean_object* v___x_2977_; lean_object* v_a_2978_; lean_object* v___x_2979_; lean_object* v___x_2980_; uint8_t v___x_2981_; 
v___x_2977_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v___y_2975_);
v_a_2978_ = lean_ctor_get(v___x_2977_, 0);
lean_inc(v_a_2978_);
lean_dec_ref(v___x_2977_);
lean_inc(v___y_2974_);
v___x_2979_ = lp_aesop_Aesop_Script_StepTree_focusableGoals(v___y_2974_);
v___x_2980_ = lp_aesop_Aesop_Script_StepTree_numSiblings(v___y_2974_);
v___x_2981_ = lean_unbox(v_a_2978_);
lean_dec(v_a_2978_);
if (v___x_2981_ == 0)
{
v___y_2873_ = v___x_2979_;
v___y_2874_ = v___x_2980_;
v___y_2875_ = v___y_2975_;
v___y_2876_ = v___y_2976_;
goto v___jp_2872_;
}
else
{
lean_object* v_traceClass_2982_; lean_object* v_size_2983_; lean_object* v_buckets_2984_; lean_object* v___x_2985_; lean_object* v___x_2986_; lean_object* v___x_2987_; lean_object* v___x_2988_; uint8_t v___x_2989_; 
v_traceClass_2982_ = lean_ctor_get(v___x_2927_, 0);
v_size_2983_ = lean_ctor_get(v___x_2979_, 0);
lean_inc(v_size_2983_);
v_buckets_2984_ = lean_ctor_get(v___x_2979_, 1);
lean_inc_ref(v_buckets_2984_);
v___x_2985_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__1);
v___x_2986_ = lean_mk_empty_array_with_capacity(v_size_2983_);
lean_dec(v_size_2983_);
v___x_2987_ = lean_unsigned_to_nat(0u);
v___x_2988_ = lean_array_get_size(v_buckets_2984_);
v___x_2989_ = lean_nat_dec_lt(v___x_2987_, v___x_2988_);
if (v___x_2989_ == 0)
{
lean_dec_ref(v_buckets_2984_);
lean_inc(v_traceClass_2982_);
v___y_2900_ = v___x_2985_;
v___y_2901_ = v___y_2976_;
v___y_2902_ = v_traceClass_2982_;
v___y_2903_ = v___x_2979_;
v___y_2904_ = v___x_2980_;
v___y_2905_ = v___y_2975_;
v___y_2906_ = v___x_2986_;
goto v___jp_2899_;
}
else
{
uint8_t v___x_2990_; 
v___x_2990_ = lean_nat_dec_le(v___x_2988_, v___x_2988_);
if (v___x_2990_ == 0)
{
if (v___x_2989_ == 0)
{
lean_dec_ref(v_buckets_2984_);
lean_inc(v_traceClass_2982_);
v___y_2900_ = v___x_2985_;
v___y_2901_ = v___y_2976_;
v___y_2902_ = v_traceClass_2982_;
v___y_2903_ = v___x_2979_;
v___y_2904_ = v___x_2980_;
v___y_2905_ = v___y_2975_;
v___y_2906_ = v___x_2986_;
goto v___jp_2899_;
}
else
{
size_t v___x_2991_; size_t v___x_2992_; lean_object* v___x_2993_; 
v___x_2991_ = ((size_t)0ULL);
v___x_2992_ = lean_usize_of_nat(v___x_2988_);
v___x_2993_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2984_, v___x_2991_, v___x_2992_, v___x_2986_);
lean_dec_ref(v_buckets_2984_);
lean_inc(v_traceClass_2982_);
v___y_2900_ = v___x_2985_;
v___y_2901_ = v___y_2976_;
v___y_2902_ = v_traceClass_2982_;
v___y_2903_ = v___x_2979_;
v___y_2904_ = v___x_2980_;
v___y_2905_ = v___y_2975_;
v___y_2906_ = v___x_2993_;
goto v___jp_2899_;
}
}
else
{
size_t v___x_2994_; size_t v___x_2995_; lean_object* v___x_2996_; 
v___x_2994_ = ((size_t)0ULL);
v___x_2995_ = lean_usize_of_nat(v___x_2988_);
v___x_2996_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_orderedUScriptToSScript_spec__3(v_buckets_2984_, v___x_2994_, v___x_2995_, v___x_2986_);
lean_dec_ref(v_buckets_2984_);
lean_inc(v_traceClass_2982_);
v___y_2900_ = v___x_2985_;
v___y_2901_ = v___y_2976_;
v___y_2902_ = v_traceClass_2982_;
v___y_2903_ = v___x_2979_;
v___y_2904_ = v___x_2980_;
v___y_2905_ = v___y_2975_;
v___y_2906_ = v___x_2996_;
goto v___jp_2899_;
}
}
}
}
v___jp_2997_:
{
lean_object* v___x_3000_; lean_object* v_a_3001_; lean_object* v___x_3002_; uint8_t v___x_3003_; 
v___x_3000_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__0___redArg(v___x_2927_, v___y_2998_);
v_a_3001_ = lean_ctor_get(v___x_3000_, 0);
lean_inc(v_a_3001_);
lean_dec_ref(v___x_3000_);
v___x_3002_ = lp_aesop_Aesop_Script_UScript_toStepTree(v_uscript_2815_);
v___x_3003_ = lean_unbox(v_a_3001_);
lean_dec(v_a_3001_);
if (v___x_3003_ == 0)
{
v___y_2974_ = v___x_3002_;
v___y_2975_ = v___y_2998_;
v___y_2976_ = v___y_2999_;
goto v___jp_2973_;
}
else
{
lean_object* v_traceClass_3004_; lean_object* v___x_3005_; lean_object* v___x_3006_; lean_object* v___x_3007_; lean_object* v___x_3008_; lean_object* v___x_3009_; 
v_traceClass_3004_ = lean_ctor_get(v___x_2927_, 0);
v___x_3005_ = lean_obj_once(&lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3, &lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3_once, _init_lp_aesop_Aesop_Script_orderedUScriptToSScript___lam__1___closed__3);
lean_inc(v___x_3002_);
v___x_3006_ = lp_aesop_Aesop_Script_StepTree_toMessageData(v___x_3002_);
v___x_3007_ = l_Lean_indentD(v___x_3006_);
v___x_3008_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3008_, 0, v___x_3005_);
lean_ctor_set(v___x_3008_, 1, v___x_3007_);
lean_inc(v_traceClass_3004_);
v___x_3009_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Script_UScriptToSScript_0__Aesop_Script_orderedUScriptToSScript_go_spec__9(v_traceClass_3004_, v___x_3008_, v___y_2998_, v___y_2999_);
if (lean_obj_tag(v___x_3009_) == 0)
{
lean_dec_ref_known(v___x_3009_, 1);
v___y_2974_ = v___x_3002_;
v___y_2975_ = v___y_2998_;
v___y_2976_ = v___y_2999_;
goto v___jp_2973_;
}
else
{
lean_object* v_a_3010_; lean_object* v___x_3012_; uint8_t v_isShared_3013_; uint8_t v_isSharedCheck_3017_; 
lean_dec(v___x_3002_);
lean_dec_ref(v_tacticState_2816_);
lean_dec_ref(v_uscript_2815_);
v_a_3010_ = lean_ctor_get(v___x_3009_, 0);
v_isSharedCheck_3017_ = !lean_is_exclusive(v___x_3009_);
if (v_isSharedCheck_3017_ == 0)
{
v___x_3012_ = v___x_3009_;
v_isShared_3013_ = v_isSharedCheck_3017_;
goto v_resetjp_3011_;
}
else
{
lean_inc(v_a_3010_);
lean_dec(v___x_3009_);
v___x_3012_ = lean_box(0);
v_isShared_3013_ = v_isSharedCheck_3017_;
goto v_resetjp_3011_;
}
v_resetjp_3011_:
{
lean_object* v___x_3015_; 
if (v_isShared_3013_ == 0)
{
v___x_3015_ = v___x_3012_;
goto v_reusejp_3014_;
}
else
{
lean_object* v_reuseFailAlloc_3016_; 
v_reuseFailAlloc_3016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3016_, 0, v_a_3010_);
v___x_3015_ = v_reuseFailAlloc_3016_;
goto v_reusejp_3014_;
}
v_reusejp_3014_:
{
return v___x_3015_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_orderedUScriptToSScript___boxed(lean_object* v_uscript_3175_, lean_object* v_tacticState_3176_, lean_object* v_a_3177_, lean_object* v_a_3178_, lean_object* v_a_3179_){
_start:
{
lean_object* v_res_3180_; 
v_res_3180_ = lp_aesop_Aesop_Script_orderedUScriptToSScript(v_uscript_3175_, v_tacticState_3176_, v_a_3177_, v_a_3178_);
lean_dec(v_a_3178_);
lean_dec_ref(v_a_3177_);
return v_res_3180_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7(lean_object* v_00_u03b1_3181_, lean_object* v_x_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_){
_start:
{
lean_object* v___x_3186_; 
v___x_3186_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___redArg(v_x_3182_);
return v___x_3186_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7___boxed(lean_object* v_00_u03b1_3187_, lean_object* v_x_3188_, lean_object* v___y_3189_, lean_object* v___y_3190_, lean_object* v___y_3191_){
_start:
{
lean_object* v_res_3192_; 
v_res_3192_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Script_orderedUScriptToSScript_spec__6_spec__7(v_00_u03b1_3187_, v_x_3188_, v___y_3189_, v___y_3190_);
lean_dec(v___y_3190_);
lean_dec_ref(v___y_3189_);
return v_res_3192_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_UScript(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_SScript(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_UScriptToSScript(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_UScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_SScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_UScriptToSScript(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Script_UScript(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_SScript(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_UScriptToSScript(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_UScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_SScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_UScriptToSScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_UScriptToSScript(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_UScriptToSScript(builtin);
}
#ifdef __cplusplus
}
#endif
