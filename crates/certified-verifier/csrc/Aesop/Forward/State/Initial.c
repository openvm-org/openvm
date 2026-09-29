// Lean compiler output
// Module: Aesop.Forward.State.Initial
// Imports: public import Init public meta import Init public import Aesop.Forward.State public import Aesop.RuleSet
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
uint8_t lean_usize_dec_lt(size_t, size_t);
extern lean_object* lp_aesop_Aesop_TraceOption_forward;
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_statefulForward;
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_enqueueTargetPatSubsts(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_constForwardRuleMatches(lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1(lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2(lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5_spec__10(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8_spec__13(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__7(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__0;
static const lean_string_object lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__1_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__2 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "match for constant rule "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__2_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__3_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__4_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__5_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__6_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__8_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__10_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__11_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__12 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__12_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__13 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__13_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__14 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__14_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__15 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__0;
static lean_once_cell_t lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__1;
static lean_once_cell_t lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__2;
static lean_once_cell_t lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3;
static const lean_array_object lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0(lean_object* v_opts_1_, lean_object* v_opt_2_){
_start:
{
lean_object* v_name_3_; lean_object* v_defValue_4_; lean_object* v_map_5_; lean_object* v___x_6_; 
v_name_3_ = lean_ctor_get(v_opt_2_, 0);
v_defValue_4_ = lean_ctor_get(v_opt_2_, 1);
v_map_5_ = lean_ctor_get(v_opts_1_, 0);
v___x_6_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_5_, v_name_3_);
if (lean_obj_tag(v___x_6_) == 0)
{
uint8_t v___x_7_; 
v___x_7_ = lean_unbox(v_defValue_4_);
return v___x_7_;
}
else
{
lean_object* v_val_8_; 
v_val_8_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_val_8_);
lean_dec_ref_known(v___x_6_, 1);
if (lean_obj_tag(v_val_8_) == 1)
{
uint8_t v_v_9_; 
v_v_9_ = lean_ctor_get_uint8(v_val_8_, 0);
lean_dec_ref_known(v_val_8_, 0);
return v_v_9_;
}
else
{
uint8_t v___x_10_; 
lean_dec(v_val_8_);
v___x_10_ = lean_unbox(v_defValue_4_);
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0___boxed(lean_object* v_opts_11_, lean_object* v_opt_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0(v_opts_11_, v_opt_12_);
lean_dec_ref(v_opt_12_);
lean_dec_ref(v_opts_11_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__0(void){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_15_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__1(void){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_16_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__0);
v___x_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1(lean_object* v_00_u03b2_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1___closed__1);
return v___x_19_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__0(void){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_20_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__1(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__0);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2(lean_object* v_00_u03b2_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2___closed__1);
return v___x_24_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__0(void){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_25_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__1(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_26_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__0);
v___x_27_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3(lean_object* v_00_u03b2_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3___closed__1);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___lam__0(lean_object* v_x_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_){
_start:
{
lean_object* v___x_37_; 
lean_inc(v___y_31_);
v___x_37_ = lean_apply_6(v_x_30_, v___y_31_, v___y_32_, v___y_33_, v___y_34_, v___y_35_, lean_box(0));
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___lam__0___boxed(lean_object* v_x_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___lam__0(v_x_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_);
lean_dec(v___y_39_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg(lean_object* v_mvarId_46_, lean_object* v_x_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_){
_start:
{
lean_object* v___f_54_; lean_object* v___x_55_; 
lean_inc(v___y_48_);
v___f_54_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_54_, 0, v_x_47_);
lean_closure_set(v___f_54_, 1, v___y_48_);
v___x_55_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_46_, v___f_54_, v___y_49_, v___y_50_, v___y_51_, v___y_52_);
if (lean_obj_tag(v___x_55_) == 0)
{
return v___x_55_;
}
else
{
lean_object* v_a_56_; lean_object* v___x_58_; uint8_t v_isShared_59_; uint8_t v_isSharedCheck_63_; 
v_a_56_ = lean_ctor_get(v___x_55_, 0);
v_isSharedCheck_63_ = !lean_is_exclusive(v___x_55_);
if (v_isSharedCheck_63_ == 0)
{
v___x_58_ = v___x_55_;
v_isShared_59_ = v_isSharedCheck_63_;
goto v_resetjp_57_;
}
else
{
lean_inc(v_a_56_);
lean_dec(v___x_55_);
v___x_58_ = lean_box(0);
v_isShared_59_ = v_isSharedCheck_63_;
goto v_resetjp_57_;
}
v_resetjp_57_:
{
lean_object* v___x_61_; 
if (v_isShared_59_ == 0)
{
v___x_61_ = v___x_58_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_62_; 
v_reuseFailAlloc_62_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_62_, 0, v_a_56_);
v___x_61_ = v_reuseFailAlloc_62_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
return v___x_61_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg___boxed(lean_object* v_mvarId_64_, lean_object* v_x_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg(v_mvarId_64_, v_x_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_, v___y_70_);
lean_dec(v___y_70_);
lean_dec_ref(v___y_69_);
lean_dec(v___y_68_);
lean_dec_ref(v___y_67_);
lean_dec(v___y_66_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8(lean_object* v_00_u03b1_73_, lean_object* v_mvarId_74_, lean_object* v_x_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg(v_mvarId_74_, v_x_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___boxed(lean_object* v_00_u03b1_83_, lean_object* v_mvarId_84_, lean_object* v_x_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8(v_00_u03b1_83_, v_mvarId_84_, v_x_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
lean_dec(v___y_86_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__9(lean_object* v_opts_93_, lean_object* v_opt_94_){
_start:
{
lean_object* v_name_95_; lean_object* v_defValue_96_; lean_object* v_map_97_; lean_object* v___x_98_; 
v_name_95_ = lean_ctor_get(v_opt_94_, 0);
v_defValue_96_ = lean_ctor_get(v_opt_94_, 1);
v_map_97_ = lean_ctor_get(v_opts_93_, 0);
v___x_98_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_97_, v_name_95_);
if (lean_obj_tag(v___x_98_) == 0)
{
lean_inc(v_defValue_96_);
return v_defValue_96_;
}
else
{
lean_object* v_val_99_; 
v_val_99_ = lean_ctor_get(v___x_98_, 0);
lean_inc(v_val_99_);
lean_dec_ref_known(v___x_98_, 1);
if (lean_obj_tag(v_val_99_) == 0)
{
lean_object* v_v_100_; 
v_v_100_ = lean_ctor_get(v_val_99_, 0);
lean_inc_ref(v_v_100_);
lean_dec_ref_known(v_val_99_, 1);
return v_v_100_;
}
else
{
lean_dec(v_val_99_);
lean_inc(v_defValue_96_);
return v_defValue_96_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__9___boxed(lean_object* v_opts_101_, lean_object* v_opt_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__9(v_opts_101_, v_opt_102_);
lean_dec_ref(v_opt_102_);
lean_dec_ref(v_opts_101_);
return v_res_103_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0(uint8_t v___x_104_, lean_object* v_x_105_){
_start:
{
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0___boxed(lean_object* v___x_106_, lean_object* v_x_107_){
_start:
{
uint8_t v___x_16753__boxed_108_; uint8_t v_res_109_; lean_object* v_r_110_; 
v___x_16753__boxed_108_ = lean_unbox(v___x_106_);
v_res_109_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0(v___x_16753__boxed_108_, v_x_107_);
lean_dec_ref(v_x_107_);
v_r_110_ = lean_box(v_res_109_);
return v_r_110_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5_spec__10(lean_object* v_rs_111_, uint8_t v___x_112_, lean_object* v_as_113_, size_t v_sz_114_, size_t v_i_115_, lean_object* v_b_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
uint8_t v___x_123_; 
v___x_123_ = lean_usize_dec_lt(v_i_115_, v_sz_114_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; 
lean_dec_ref(v_rs_111_);
v___x_124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_124_, 0, v_b_116_);
return v___x_124_;
}
else
{
lean_object* v_snd_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_166_; 
v_snd_125_ = lean_ctor_get(v_b_116_, 1);
v_isSharedCheck_166_ = !lean_is_exclusive(v_b_116_);
if (v_isSharedCheck_166_ == 0)
{
lean_object* v_unused_167_; 
v_unused_167_ = lean_ctor_get(v_b_116_, 0);
lean_dec(v_unused_167_);
v___x_127_ = v_b_116_;
v_isShared_128_ = v_isSharedCheck_166_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_snd_125_);
lean_dec(v_b_116_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_166_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_129_; lean_object* v_a_131_; lean_object* v_a_138_; 
v___x_129_ = lean_box(0);
v_a_138_ = lean_array_uget_borrowed(v_as_113_, v_i_115_);
if (lean_obj_tag(v_a_138_) == 0)
{
v_a_131_ = v_snd_125_;
goto v___jp_130_;
}
else
{
lean_object* v_val_139_; uint8_t v___x_140_; 
v_val_139_ = lean_ctor_get(v_a_138_, 0);
v___x_140_ = l_Lean_LocalDecl_isImplementationDetail(v_val_139_);
if (v___x_140_ == 0)
{
lean_object* v___x_141_; lean_object* v___f_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_141_ = lean_box(v___x_112_);
v___f_142_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0___boxed), 2, 1);
lean_closure_set(v___f_142_, 0, v___x_141_);
v___x_143_ = l_Lean_LocalDecl_type(v_val_139_);
lean_inc_ref(v_rs_111_);
v___x_144_ = lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(v_rs_111_, v___x_143_, v___f_142_, v___y_118_, v___y_119_, v___y_120_, v___y_121_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_object* v_a_145_; lean_object* v___x_146_; 
v_a_145_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_a_145_);
lean_dec_ref_known(v___x_144_, 1);
lean_inc(v_val_139_);
lean_inc_ref(v_rs_111_);
v___x_146_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(v_rs_111_, v_val_139_, v___y_117_, v___y_118_, v___y_119_, v___y_120_, v___y_121_);
if (lean_obj_tag(v___x_146_) == 0)
{
lean_object* v_a_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v_a_147_ = lean_ctor_get(v___x_146_, 0);
lean_inc(v_a_147_);
lean_dec_ref_known(v___x_146_, 1);
v___x_148_ = l_Lean_LocalDecl_fvarId(v_val_139_);
v___x_149_ = lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(v___x_148_, v_a_145_, v_a_147_, v_snd_125_);
lean_dec(v_a_147_);
v_a_131_ = v___x_149_;
goto v___jp_130_;
}
else
{
lean_object* v_a_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_157_; 
lean_dec(v_a_145_);
lean_del_object(v___x_127_);
lean_dec(v_snd_125_);
lean_dec_ref(v_rs_111_);
v_a_150_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_157_ == 0)
{
v___x_152_ = v___x_146_;
v_isShared_153_ = v_isSharedCheck_157_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_a_150_);
lean_dec(v___x_146_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_157_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v___x_155_; 
if (v_isShared_153_ == 0)
{
v___x_155_ = v___x_152_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v_a_150_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
}
}
else
{
lean_object* v_a_158_; lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_165_; 
lean_del_object(v___x_127_);
lean_dec(v_snd_125_);
lean_dec_ref(v_rs_111_);
v_a_158_ = lean_ctor_get(v___x_144_, 0);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_165_ == 0)
{
v___x_160_ = v___x_144_;
v_isShared_161_ = v_isSharedCheck_165_;
goto v_resetjp_159_;
}
else
{
lean_inc(v_a_158_);
lean_dec(v___x_144_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_165_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
lean_object* v___x_163_; 
if (v_isShared_161_ == 0)
{
v___x_163_ = v___x_160_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v_a_158_);
v___x_163_ = v_reuseFailAlloc_164_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
return v___x_163_;
}
}
}
}
else
{
v_a_131_ = v_snd_125_;
goto v___jp_130_;
}
}
v___jp_130_:
{
lean_object* v___x_133_; 
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 1, v_a_131_);
lean_ctor_set(v___x_127_, 0, v___x_129_);
v___x_133_ = v___x_127_;
goto v_reusejp_132_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v___x_129_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v_a_131_);
v___x_133_ = v_reuseFailAlloc_137_;
goto v_reusejp_132_;
}
v_reusejp_132_:
{
size_t v___x_134_; size_t v___x_135_; 
v___x_134_ = ((size_t)1ULL);
v___x_135_ = lean_usize_add(v_i_115_, v___x_134_);
v_i_115_ = v___x_135_;
v_b_116_ = v___x_133_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5_spec__10___boxed(lean_object* v_rs_168_, lean_object* v___x_169_, lean_object* v_as_170_, lean_object* v_sz_171_, lean_object* v_i_172_, lean_object* v_b_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_){
_start:
{
uint8_t v___x_16759__boxed_180_; size_t v_sz_boxed_181_; size_t v_i_boxed_182_; lean_object* v_res_183_; 
v___x_16759__boxed_180_ = lean_unbox(v___x_169_);
v_sz_boxed_181_ = lean_unbox_usize(v_sz_171_);
lean_dec(v_sz_171_);
v_i_boxed_182_ = lean_unbox_usize(v_i_172_);
lean_dec(v_i_172_);
v_res_183_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5_spec__10(v_rs_168_, v___x_16759__boxed_180_, v_as_170_, v_sz_boxed_181_, v_i_boxed_182_, v_b_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v___y_176_);
lean_dec_ref(v___y_175_);
lean_dec(v___y_174_);
lean_dec_ref(v_as_170_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5(lean_object* v_rs_184_, uint8_t v___x_185_, lean_object* v_as_186_, size_t v_sz_187_, size_t v_i_188_, lean_object* v_b_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
uint8_t v___x_196_; 
v___x_196_ = lean_usize_dec_lt(v_i_188_, v_sz_187_);
if (v___x_196_ == 0)
{
lean_object* v___x_197_; 
lean_dec_ref(v_rs_184_);
v___x_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_197_, 0, v_b_189_);
return v___x_197_;
}
else
{
lean_object* v_snd_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_239_; 
v_snd_198_ = lean_ctor_get(v_b_189_, 1);
v_isSharedCheck_239_ = !lean_is_exclusive(v_b_189_);
if (v_isSharedCheck_239_ == 0)
{
lean_object* v_unused_240_; 
v_unused_240_ = lean_ctor_get(v_b_189_, 0);
lean_dec(v_unused_240_);
v___x_200_ = v_b_189_;
v_isShared_201_ = v_isSharedCheck_239_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_snd_198_);
lean_dec(v_b_189_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_239_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_202_; lean_object* v_a_204_; lean_object* v_a_211_; 
v___x_202_ = lean_box(0);
v_a_211_ = lean_array_uget_borrowed(v_as_186_, v_i_188_);
if (lean_obj_tag(v_a_211_) == 0)
{
v_a_204_ = v_snd_198_;
goto v___jp_203_;
}
else
{
lean_object* v_val_212_; uint8_t v___x_213_; 
v_val_212_ = lean_ctor_get(v_a_211_, 0);
v___x_213_ = l_Lean_LocalDecl_isImplementationDetail(v_val_212_);
if (v___x_213_ == 0)
{
lean_object* v___x_214_; lean_object* v___f_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_214_ = lean_box(v___x_185_);
v___f_215_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0___boxed), 2, 1);
lean_closure_set(v___f_215_, 0, v___x_214_);
v___x_216_ = l_Lean_LocalDecl_type(v_val_212_);
lean_inc_ref(v_rs_184_);
v___x_217_ = lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(v_rs_184_, v___x_216_, v___f_215_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
if (lean_obj_tag(v___x_217_) == 0)
{
lean_object* v_a_218_; lean_object* v___x_219_; 
v_a_218_ = lean_ctor_get(v___x_217_, 0);
lean_inc(v_a_218_);
lean_dec_ref_known(v___x_217_, 1);
lean_inc(v_val_212_);
lean_inc_ref(v_rs_184_);
v___x_219_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(v_rs_184_, v_val_212_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
if (lean_obj_tag(v___x_219_) == 0)
{
lean_object* v_a_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v_a_220_ = lean_ctor_get(v___x_219_, 0);
lean_inc(v_a_220_);
lean_dec_ref_known(v___x_219_, 1);
v___x_221_ = l_Lean_LocalDecl_fvarId(v_val_212_);
v___x_222_ = lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(v___x_221_, v_a_218_, v_a_220_, v_snd_198_);
lean_dec(v_a_220_);
v_a_204_ = v___x_222_;
goto v___jp_203_;
}
else
{
lean_object* v_a_223_; lean_object* v___x_225_; uint8_t v_isShared_226_; uint8_t v_isSharedCheck_230_; 
lean_dec(v_a_218_);
lean_del_object(v___x_200_);
lean_dec(v_snd_198_);
lean_dec_ref(v_rs_184_);
v_a_223_ = lean_ctor_get(v___x_219_, 0);
v_isSharedCheck_230_ = !lean_is_exclusive(v___x_219_);
if (v_isSharedCheck_230_ == 0)
{
v___x_225_ = v___x_219_;
v_isShared_226_ = v_isSharedCheck_230_;
goto v_resetjp_224_;
}
else
{
lean_inc(v_a_223_);
lean_dec(v___x_219_);
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
else
{
lean_object* v_a_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_238_; 
lean_del_object(v___x_200_);
lean_dec(v_snd_198_);
lean_dec_ref(v_rs_184_);
v_a_231_ = lean_ctor_get(v___x_217_, 0);
v_isSharedCheck_238_ = !lean_is_exclusive(v___x_217_);
if (v_isSharedCheck_238_ == 0)
{
v___x_233_ = v___x_217_;
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_a_231_);
lean_dec(v___x_217_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v___x_236_; 
if (v_isShared_234_ == 0)
{
v___x_236_ = v___x_233_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v_a_231_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
else
{
v_a_204_ = v_snd_198_;
goto v___jp_203_;
}
}
v___jp_203_:
{
lean_object* v___x_206_; 
if (v_isShared_201_ == 0)
{
lean_ctor_set(v___x_200_, 1, v_a_204_);
lean_ctor_set(v___x_200_, 0, v___x_202_);
v___x_206_ = v___x_200_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_202_);
lean_ctor_set(v_reuseFailAlloc_210_, 1, v_a_204_);
v___x_206_ = v_reuseFailAlloc_210_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
size_t v___x_207_; size_t v___x_208_; lean_object* v___x_209_; 
v___x_207_ = ((size_t)1ULL);
v___x_208_ = lean_usize_add(v_i_188_, v___x_207_);
v___x_209_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5_spec__10(v_rs_184_, v___x_185_, v_as_186_, v_sz_187_, v___x_208_, v___x_206_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
return v___x_209_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___boxed(lean_object* v_rs_241_, lean_object* v___x_242_, lean_object* v_as_243_, lean_object* v_sz_244_, lean_object* v_i_245_, lean_object* v_b_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_){
_start:
{
uint8_t v___x_16866__boxed_253_; size_t v_sz_boxed_254_; size_t v_i_boxed_255_; lean_object* v_res_256_; 
v___x_16866__boxed_253_ = lean_unbox(v___x_242_);
v_sz_boxed_254_ = lean_unbox_usize(v_sz_244_);
lean_dec(v_sz_244_);
v_i_boxed_255_ = lean_unbox_usize(v_i_245_);
lean_dec(v_i_245_);
v_res_256_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5(v_rs_241_, v___x_16866__boxed_253_, v_as_243_, v_sz_boxed_254_, v_i_boxed_255_, v_b_246_, v___y_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_);
lean_dec(v___y_251_);
lean_dec_ref(v___y_250_);
lean_dec(v___y_249_);
lean_dec_ref(v___y_248_);
lean_dec(v___y_247_);
lean_dec_ref(v_as_243_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8_spec__13(lean_object* v_rs_257_, uint8_t v___x_258_, lean_object* v_as_259_, size_t v_sz_260_, size_t v_i_261_, lean_object* v_b_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
uint8_t v___x_269_; 
v___x_269_ = lean_usize_dec_lt(v_i_261_, v_sz_260_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; 
lean_dec_ref(v_rs_257_);
v___x_270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_270_, 0, v_b_262_);
return v___x_270_;
}
else
{
lean_object* v_snd_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_312_; 
v_snd_271_ = lean_ctor_get(v_b_262_, 1);
v_isSharedCheck_312_ = !lean_is_exclusive(v_b_262_);
if (v_isSharedCheck_312_ == 0)
{
lean_object* v_unused_313_; 
v_unused_313_ = lean_ctor_get(v_b_262_, 0);
lean_dec(v_unused_313_);
v___x_273_ = v_b_262_;
v_isShared_274_ = v_isSharedCheck_312_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_snd_271_);
lean_dec(v_b_262_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_312_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
lean_object* v___x_275_; lean_object* v_a_277_; lean_object* v_a_284_; 
v___x_275_ = lean_box(0);
v_a_284_ = lean_array_uget_borrowed(v_as_259_, v_i_261_);
if (lean_obj_tag(v_a_284_) == 0)
{
v_a_277_ = v_snd_271_;
goto v___jp_276_;
}
else
{
lean_object* v_val_285_; uint8_t v___x_286_; 
v_val_285_ = lean_ctor_get(v_a_284_, 0);
v___x_286_ = l_Lean_LocalDecl_isImplementationDetail(v_val_285_);
if (v___x_286_ == 0)
{
lean_object* v___x_287_; lean_object* v___f_288_; lean_object* v___x_289_; lean_object* v___x_290_; 
v___x_287_ = lean_box(v___x_258_);
v___f_288_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0___boxed), 2, 1);
lean_closure_set(v___f_288_, 0, v___x_287_);
v___x_289_ = l_Lean_LocalDecl_type(v_val_285_);
lean_inc_ref(v_rs_257_);
v___x_290_ = lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(v_rs_257_, v___x_289_, v___f_288_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
if (lean_obj_tag(v___x_290_) == 0)
{
lean_object* v_a_291_; lean_object* v___x_292_; 
v_a_291_ = lean_ctor_get(v___x_290_, 0);
lean_inc(v_a_291_);
lean_dec_ref_known(v___x_290_, 1);
lean_inc(v_val_285_);
lean_inc_ref(v_rs_257_);
v___x_292_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(v_rs_257_, v_val_285_, v___y_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
if (lean_obj_tag(v___x_292_) == 0)
{
lean_object* v_a_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v_a_293_ = lean_ctor_get(v___x_292_, 0);
lean_inc(v_a_293_);
lean_dec_ref_known(v___x_292_, 1);
v___x_294_ = l_Lean_LocalDecl_fvarId(v_val_285_);
v___x_295_ = lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(v___x_294_, v_a_291_, v_a_293_, v_snd_271_);
lean_dec(v_a_293_);
v_a_277_ = v___x_295_;
goto v___jp_276_;
}
else
{
lean_object* v_a_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_303_; 
lean_dec(v_a_291_);
lean_del_object(v___x_273_);
lean_dec(v_snd_271_);
lean_dec_ref(v_rs_257_);
v_a_296_ = lean_ctor_get(v___x_292_, 0);
v_isSharedCheck_303_ = !lean_is_exclusive(v___x_292_);
if (v_isSharedCheck_303_ == 0)
{
v___x_298_ = v___x_292_;
v_isShared_299_ = v_isSharedCheck_303_;
goto v_resetjp_297_;
}
else
{
lean_inc(v_a_296_);
lean_dec(v___x_292_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_303_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
lean_object* v___x_301_; 
if (v_isShared_299_ == 0)
{
v___x_301_ = v___x_298_;
goto v_reusejp_300_;
}
else
{
lean_object* v_reuseFailAlloc_302_; 
v_reuseFailAlloc_302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_302_, 0, v_a_296_);
v___x_301_ = v_reuseFailAlloc_302_;
goto v_reusejp_300_;
}
v_reusejp_300_:
{
return v___x_301_;
}
}
}
}
else
{
lean_object* v_a_304_; lean_object* v___x_306_; uint8_t v_isShared_307_; uint8_t v_isSharedCheck_311_; 
lean_del_object(v___x_273_);
lean_dec(v_snd_271_);
lean_dec_ref(v_rs_257_);
v_a_304_ = lean_ctor_get(v___x_290_, 0);
v_isSharedCheck_311_ = !lean_is_exclusive(v___x_290_);
if (v_isSharedCheck_311_ == 0)
{
v___x_306_ = v___x_290_;
v_isShared_307_ = v_isSharedCheck_311_;
goto v_resetjp_305_;
}
else
{
lean_inc(v_a_304_);
lean_dec(v___x_290_);
v___x_306_ = lean_box(0);
v_isShared_307_ = v_isSharedCheck_311_;
goto v_resetjp_305_;
}
v_resetjp_305_:
{
lean_object* v___x_309_; 
if (v_isShared_307_ == 0)
{
v___x_309_ = v___x_306_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_310_; 
v_reuseFailAlloc_310_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_310_, 0, v_a_304_);
v___x_309_ = v_reuseFailAlloc_310_;
goto v_reusejp_308_;
}
v_reusejp_308_:
{
return v___x_309_;
}
}
}
}
else
{
v_a_277_ = v_snd_271_;
goto v___jp_276_;
}
}
v___jp_276_:
{
lean_object* v___x_279_; 
if (v_isShared_274_ == 0)
{
lean_ctor_set(v___x_273_, 1, v_a_277_);
lean_ctor_set(v___x_273_, 0, v___x_275_);
v___x_279_ = v___x_273_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v___x_275_);
lean_ctor_set(v_reuseFailAlloc_283_, 1, v_a_277_);
v___x_279_ = v_reuseFailAlloc_283_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
size_t v___x_280_; size_t v___x_281_; 
v___x_280_ = ((size_t)1ULL);
v___x_281_ = lean_usize_add(v_i_261_, v___x_280_);
v_i_261_ = v___x_281_;
v_b_262_ = v___x_279_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8_spec__13___boxed(lean_object* v_rs_314_, lean_object* v___x_315_, lean_object* v_as_316_, lean_object* v_sz_317_, lean_object* v_i_318_, lean_object* v_b_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
uint8_t v___x_16973__boxed_326_; size_t v_sz_boxed_327_; size_t v_i_boxed_328_; lean_object* v_res_329_; 
v___x_16973__boxed_326_ = lean_unbox(v___x_315_);
v_sz_boxed_327_ = lean_unbox_usize(v_sz_317_);
lean_dec(v_sz_317_);
v_i_boxed_328_ = lean_unbox_usize(v_i_318_);
lean_dec(v_i_318_);
v_res_329_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8_spec__13(v_rs_314_, v___x_16973__boxed_326_, v_as_316_, v_sz_boxed_327_, v_i_boxed_328_, v_b_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
lean_dec(v___y_320_);
lean_dec_ref(v_as_316_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8(lean_object* v_rs_330_, uint8_t v___x_331_, lean_object* v_as_332_, size_t v_sz_333_, size_t v_i_334_, lean_object* v_b_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
uint8_t v___x_342_; 
v___x_342_ = lean_usize_dec_lt(v_i_334_, v_sz_333_);
if (v___x_342_ == 0)
{
lean_object* v___x_343_; 
lean_dec_ref(v_rs_330_);
v___x_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_343_, 0, v_b_335_);
return v___x_343_;
}
else
{
lean_object* v_snd_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_385_; 
v_snd_344_ = lean_ctor_get(v_b_335_, 1);
v_isSharedCheck_385_ = !lean_is_exclusive(v_b_335_);
if (v_isSharedCheck_385_ == 0)
{
lean_object* v_unused_386_; 
v_unused_386_ = lean_ctor_get(v_b_335_, 0);
lean_dec(v_unused_386_);
v___x_346_ = v_b_335_;
v_isShared_347_ = v_isSharedCheck_385_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_snd_344_);
lean_dec(v_b_335_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_385_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_348_; lean_object* v_a_350_; lean_object* v_a_357_; 
v___x_348_ = lean_box(0);
v_a_357_ = lean_array_uget_borrowed(v_as_332_, v_i_334_);
if (lean_obj_tag(v_a_357_) == 0)
{
v_a_350_ = v_snd_344_;
goto v___jp_349_;
}
else
{
lean_object* v_val_358_; uint8_t v___x_359_; 
v_val_358_ = lean_ctor_get(v_a_357_, 0);
v___x_359_ = l_Lean_LocalDecl_isImplementationDetail(v_val_358_);
if (v___x_359_ == 0)
{
lean_object* v___x_360_; lean_object* v___f_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_360_ = lean_box(v___x_331_);
v___f_361_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5___lam__0___boxed), 2, 1);
lean_closure_set(v___f_361_, 0, v___x_360_);
v___x_362_ = l_Lean_LocalDecl_type(v_val_358_);
lean_inc_ref(v_rs_330_);
v___x_363_ = lp_aesop_Aesop_LocalRuleSet_applicableForwardRulesWith(v_rs_330_, v___x_362_, v___f_361_, v___y_337_, v___y_338_, v___y_339_, v___y_340_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v_a_364_; lean_object* v___x_365_; 
v_a_364_ = lean_ctor_get(v___x_363_, 0);
lean_inc(v_a_364_);
lean_dec_ref_known(v___x_363_, 1);
lean_inc(v_val_358_);
lean_inc_ref(v_rs_330_);
v___x_365_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInLocalDecl(v_rs_330_, v_val_358_, v___y_336_, v___y_337_, v___y_338_, v___y_339_, v___y_340_);
if (lean_obj_tag(v___x_365_) == 0)
{
lean_object* v_a_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v_a_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_a_366_);
lean_dec_ref_known(v___x_365_, 1);
v___x_367_ = l_Lean_LocalDecl_fvarId(v_val_358_);
v___x_368_ = lp_aesop_Aesop_ForwardState_enqueueHypWithPatSubsts(v___x_367_, v_a_364_, v_a_366_, v_snd_344_);
lean_dec(v_a_366_);
v_a_350_ = v___x_368_;
goto v___jp_349_;
}
else
{
lean_object* v_a_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_376_; 
lean_dec(v_a_364_);
lean_del_object(v___x_346_);
lean_dec(v_snd_344_);
lean_dec_ref(v_rs_330_);
v_a_369_ = lean_ctor_get(v___x_365_, 0);
v_isSharedCheck_376_ = !lean_is_exclusive(v___x_365_);
if (v_isSharedCheck_376_ == 0)
{
v___x_371_ = v___x_365_;
v_isShared_372_ = v_isSharedCheck_376_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_a_369_);
lean_dec(v___x_365_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_376_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_374_; 
if (v_isShared_372_ == 0)
{
v___x_374_ = v___x_371_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v_a_369_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
}
}
else
{
lean_object* v_a_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_384_; 
lean_del_object(v___x_346_);
lean_dec(v_snd_344_);
lean_dec_ref(v_rs_330_);
v_a_377_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_384_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_384_ == 0)
{
v___x_379_ = v___x_363_;
v_isShared_380_ = v_isSharedCheck_384_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_a_377_);
lean_dec(v___x_363_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_384_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
lean_object* v___x_382_; 
if (v_isShared_380_ == 0)
{
v___x_382_ = v___x_379_;
goto v_reusejp_381_;
}
else
{
lean_object* v_reuseFailAlloc_383_; 
v_reuseFailAlloc_383_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_383_, 0, v_a_377_);
v___x_382_ = v_reuseFailAlloc_383_;
goto v_reusejp_381_;
}
v_reusejp_381_:
{
return v___x_382_;
}
}
}
}
else
{
v_a_350_ = v_snd_344_;
goto v___jp_349_;
}
}
v___jp_349_:
{
lean_object* v___x_352_; 
if (v_isShared_347_ == 0)
{
lean_ctor_set(v___x_346_, 1, v_a_350_);
lean_ctor_set(v___x_346_, 0, v___x_348_);
v___x_352_ = v___x_346_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v___x_348_);
lean_ctor_set(v_reuseFailAlloc_356_, 1, v_a_350_);
v___x_352_ = v_reuseFailAlloc_356_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
size_t v___x_353_; size_t v___x_354_; lean_object* v___x_355_; 
v___x_353_ = ((size_t)1ULL);
v___x_354_ = lean_usize_add(v_i_334_, v___x_353_);
v___x_355_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8_spec__13(v_rs_330_, v___x_331_, v_as_332_, v_sz_333_, v___x_354_, v___x_352_, v___y_336_, v___y_337_, v___y_338_, v___y_339_, v___y_340_);
return v___x_355_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8___boxed(lean_object* v_rs_387_, lean_object* v___x_388_, lean_object* v_as_389_, lean_object* v_sz_390_, lean_object* v_i_391_, lean_object* v_b_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_){
_start:
{
uint8_t v___x_17080__boxed_399_; size_t v_sz_boxed_400_; size_t v_i_boxed_401_; lean_object* v_res_402_; 
v___x_17080__boxed_399_ = lean_unbox(v___x_388_);
v_sz_boxed_400_ = lean_unbox_usize(v_sz_390_);
lean_dec(v_sz_390_);
v_i_boxed_401_ = lean_unbox_usize(v_i_391_);
lean_dec(v_i_391_);
v_res_402_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8(v_rs_387_, v___x_17080__boxed_399_, v_as_389_, v_sz_boxed_400_, v_i_boxed_401_, v_b_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_, v___y_397_);
lean_dec(v___y_397_);
lean_dec_ref(v___y_396_);
lean_dec(v___y_395_);
lean_dec_ref(v___y_394_);
lean_dec(v___y_393_);
lean_dec_ref(v_as_389_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4(lean_object* v_init_403_, lean_object* v_rs_404_, uint8_t v___x_405_, lean_object* v_n_406_, lean_object* v_b_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_){
_start:
{
if (lean_obj_tag(v_n_406_) == 0)
{
lean_object* v_cs_414_; lean_object* v___x_415_; lean_object* v___x_416_; size_t v_sz_417_; size_t v___x_418_; lean_object* v___x_419_; 
v_cs_414_ = lean_ctor_get(v_n_406_, 0);
v___x_415_ = lean_box(0);
v___x_416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_416_, 0, v___x_415_);
lean_ctor_set(v___x_416_, 1, v_b_407_);
v_sz_417_ = lean_array_size(v_cs_414_);
v___x_418_ = ((size_t)0ULL);
v___x_419_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__7(v_init_403_, v_rs_404_, v___x_405_, v_cs_414_, v_sz_417_, v___x_418_, v___x_416_, v___y_408_, v___y_409_, v___y_410_, v___y_411_, v___y_412_);
if (lean_obj_tag(v___x_419_) == 0)
{
lean_object* v_a_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_434_; 
v_a_420_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_434_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_434_ == 0)
{
v___x_422_ = v___x_419_;
v_isShared_423_ = v_isSharedCheck_434_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_a_420_);
lean_dec(v___x_419_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_434_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v_fst_424_; 
v_fst_424_ = lean_ctor_get(v_a_420_, 0);
if (lean_obj_tag(v_fst_424_) == 0)
{
lean_object* v_snd_425_; lean_object* v___x_426_; lean_object* v___x_428_; 
v_snd_425_ = lean_ctor_get(v_a_420_, 1);
lean_inc(v_snd_425_);
lean_dec(v_a_420_);
v___x_426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_426_, 0, v_snd_425_);
if (v_isShared_423_ == 0)
{
lean_ctor_set(v___x_422_, 0, v___x_426_);
v___x_428_ = v___x_422_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v___x_426_);
v___x_428_ = v_reuseFailAlloc_429_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
return v___x_428_;
}
}
else
{
lean_object* v_val_430_; lean_object* v___x_432_; 
lean_inc_ref(v_fst_424_);
lean_dec(v_a_420_);
v_val_430_ = lean_ctor_get(v_fst_424_, 0);
lean_inc(v_val_430_);
lean_dec_ref_known(v_fst_424_, 1);
if (v_isShared_423_ == 0)
{
lean_ctor_set(v___x_422_, 0, v_val_430_);
v___x_432_ = v___x_422_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v_val_430_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
}
else
{
lean_object* v_a_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_442_; 
v_a_435_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_442_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_442_ == 0)
{
v___x_437_ = v___x_419_;
v_isShared_438_ = v_isSharedCheck_442_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_a_435_);
lean_dec(v___x_419_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_442_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v___x_440_; 
if (v_isShared_438_ == 0)
{
v___x_440_ = v___x_437_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v_a_435_);
v___x_440_ = v_reuseFailAlloc_441_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
return v___x_440_;
}
}
}
}
else
{
lean_object* v_vs_443_; lean_object* v___x_444_; lean_object* v___x_445_; size_t v_sz_446_; size_t v___x_447_; lean_object* v___x_448_; 
v_vs_443_ = lean_ctor_get(v_n_406_, 0);
v___x_444_ = lean_box(0);
v___x_445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_445_, 0, v___x_444_);
lean_ctor_set(v___x_445_, 1, v_b_407_);
v_sz_446_ = lean_array_size(v_vs_443_);
v___x_447_ = ((size_t)0ULL);
v___x_448_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__8(v_rs_404_, v___x_405_, v_vs_443_, v_sz_446_, v___x_447_, v___x_445_, v___y_408_, v___y_409_, v___y_410_, v___y_411_, v___y_412_);
if (lean_obj_tag(v___x_448_) == 0)
{
lean_object* v_a_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_463_; 
v_a_449_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_463_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_463_ == 0)
{
v___x_451_ = v___x_448_;
v_isShared_452_ = v_isSharedCheck_463_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_a_449_);
lean_dec(v___x_448_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_463_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v_fst_453_; 
v_fst_453_ = lean_ctor_get(v_a_449_, 0);
if (lean_obj_tag(v_fst_453_) == 0)
{
lean_object* v_snd_454_; lean_object* v___x_455_; lean_object* v___x_457_; 
v_snd_454_ = lean_ctor_get(v_a_449_, 1);
lean_inc(v_snd_454_);
lean_dec(v_a_449_);
v___x_455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_455_, 0, v_snd_454_);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 0, v___x_455_);
v___x_457_ = v___x_451_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_455_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
else
{
lean_object* v_val_459_; lean_object* v___x_461_; 
lean_inc_ref(v_fst_453_);
lean_dec(v_a_449_);
v_val_459_ = lean_ctor_get(v_fst_453_, 0);
lean_inc(v_val_459_);
lean_dec_ref_known(v_fst_453_, 1);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 0, v_val_459_);
v___x_461_ = v___x_451_;
goto v_reusejp_460_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v_val_459_);
v___x_461_ = v_reuseFailAlloc_462_;
goto v_reusejp_460_;
}
v_reusejp_460_:
{
return v___x_461_;
}
}
}
}
else
{
lean_object* v_a_464_; lean_object* v___x_466_; uint8_t v_isShared_467_; uint8_t v_isSharedCheck_471_; 
v_a_464_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_471_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_471_ == 0)
{
v___x_466_ = v___x_448_;
v_isShared_467_ = v_isSharedCheck_471_;
goto v_resetjp_465_;
}
else
{
lean_inc(v_a_464_);
lean_dec(v___x_448_);
v___x_466_ = lean_box(0);
v_isShared_467_ = v_isSharedCheck_471_;
goto v_resetjp_465_;
}
v_resetjp_465_:
{
lean_object* v___x_469_; 
if (v_isShared_467_ == 0)
{
v___x_469_ = v___x_466_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v_a_464_);
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
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__7(lean_object* v_init_472_, lean_object* v_rs_473_, uint8_t v___x_474_, lean_object* v_as_475_, size_t v_sz_476_, size_t v_i_477_, lean_object* v_b_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_){
_start:
{
uint8_t v___x_485_; 
v___x_485_ = lean_usize_dec_lt(v_i_477_, v_sz_476_);
if (v___x_485_ == 0)
{
lean_object* v___x_486_; 
lean_dec_ref(v_rs_473_);
v___x_486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_486_, 0, v_b_478_);
return v___x_486_;
}
else
{
lean_object* v_snd_487_; lean_object* v___x_489_; uint8_t v_isShared_490_; uint8_t v_isSharedCheck_521_; 
v_snd_487_ = lean_ctor_get(v_b_478_, 1);
v_isSharedCheck_521_ = !lean_is_exclusive(v_b_478_);
if (v_isSharedCheck_521_ == 0)
{
lean_object* v_unused_522_; 
v_unused_522_ = lean_ctor_get(v_b_478_, 0);
lean_dec(v_unused_522_);
v___x_489_ = v_b_478_;
v_isShared_490_ = v_isSharedCheck_521_;
goto v_resetjp_488_;
}
else
{
lean_inc(v_snd_487_);
lean_dec(v_b_478_);
v___x_489_ = lean_box(0);
v_isShared_490_ = v_isSharedCheck_521_;
goto v_resetjp_488_;
}
v_resetjp_488_:
{
lean_object* v_a_491_; lean_object* v___x_492_; 
v_a_491_ = lean_array_uget_borrowed(v_as_475_, v_i_477_);
lean_inc(v_snd_487_);
lean_inc_ref(v_rs_473_);
v___x_492_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4(v_init_472_, v_rs_473_, v___x_474_, v_a_491_, v_snd_487_, v___y_479_, v___y_480_, v___y_481_, v___y_482_, v___y_483_);
if (lean_obj_tag(v___x_492_) == 0)
{
lean_object* v_a_493_; lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_512_; 
v_a_493_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_512_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_512_ == 0)
{
v___x_495_ = v___x_492_;
v_isShared_496_ = v_isSharedCheck_512_;
goto v_resetjp_494_;
}
else
{
lean_inc(v_a_493_);
lean_dec(v___x_492_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_512_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
if (lean_obj_tag(v_a_493_) == 0)
{
lean_object* v___x_497_; lean_object* v___x_499_; 
lean_dec_ref(v_rs_473_);
v___x_497_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_497_, 0, v_a_493_);
if (v_isShared_490_ == 0)
{
lean_ctor_set(v___x_489_, 0, v___x_497_);
v___x_499_ = v___x_489_;
goto v_reusejp_498_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v___x_497_);
lean_ctor_set(v_reuseFailAlloc_503_, 1, v_snd_487_);
v___x_499_ = v_reuseFailAlloc_503_;
goto v_reusejp_498_;
}
v_reusejp_498_:
{
lean_object* v___x_501_; 
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 0, v___x_499_);
v___x_501_ = v___x_495_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v___x_499_);
v___x_501_ = v_reuseFailAlloc_502_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
return v___x_501_;
}
}
}
else
{
lean_object* v_a_504_; lean_object* v___x_505_; lean_object* v___x_507_; 
lean_del_object(v___x_495_);
lean_dec(v_snd_487_);
v_a_504_ = lean_ctor_get(v_a_493_, 0);
lean_inc(v_a_504_);
lean_dec_ref_known(v_a_493_, 1);
v___x_505_ = lean_box(0);
if (v_isShared_490_ == 0)
{
lean_ctor_set(v___x_489_, 1, v_a_504_);
lean_ctor_set(v___x_489_, 0, v___x_505_);
v___x_507_ = v___x_489_;
goto v_reusejp_506_;
}
else
{
lean_object* v_reuseFailAlloc_511_; 
v_reuseFailAlloc_511_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_511_, 0, v___x_505_);
lean_ctor_set(v_reuseFailAlloc_511_, 1, v_a_504_);
v___x_507_ = v_reuseFailAlloc_511_;
goto v_reusejp_506_;
}
v_reusejp_506_:
{
size_t v___x_508_; size_t v___x_509_; 
v___x_508_ = ((size_t)1ULL);
v___x_509_ = lean_usize_add(v_i_477_, v___x_508_);
v_i_477_ = v___x_509_;
v_b_478_ = v___x_507_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_513_; lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_520_; 
lean_del_object(v___x_489_);
lean_dec(v_snd_487_);
lean_dec_ref(v_rs_473_);
v_a_513_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_520_ == 0)
{
v___x_515_ = v___x_492_;
v_isShared_516_ = v_isSharedCheck_520_;
goto v_resetjp_514_;
}
else
{
lean_inc(v_a_513_);
lean_dec(v___x_492_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_520_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___x_518_; 
if (v_isShared_516_ == 0)
{
v___x_518_ = v___x_515_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_a_513_);
v___x_518_ = v_reuseFailAlloc_519_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
return v___x_518_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__7___boxed(lean_object* v_init_523_, lean_object* v_rs_524_, lean_object* v___x_525_, lean_object* v_as_526_, lean_object* v_sz_527_, lean_object* v_i_528_, lean_object* v_b_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_){
_start:
{
uint8_t v___x_17187__boxed_536_; size_t v_sz_boxed_537_; size_t v_i_boxed_538_; lean_object* v_res_539_; 
v___x_17187__boxed_536_ = lean_unbox(v___x_525_);
v_sz_boxed_537_ = lean_unbox_usize(v_sz_527_);
lean_dec(v_sz_527_);
v_i_boxed_538_ = lean_unbox_usize(v_i_528_);
lean_dec(v_i_528_);
v_res_539_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4_spec__7(v_init_523_, v_rs_524_, v___x_17187__boxed_536_, v_as_526_, v_sz_boxed_537_, v_i_boxed_538_, v_b_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_, v___y_534_);
lean_dec(v___y_534_);
lean_dec_ref(v___y_533_);
lean_dec(v___y_532_);
lean_dec_ref(v___y_531_);
lean_dec(v___y_530_);
lean_dec_ref(v_as_526_);
lean_dec_ref(v_init_523_);
return v_res_539_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4___boxed(lean_object* v_init_540_, lean_object* v_rs_541_, lean_object* v___x_542_, lean_object* v_n_543_, lean_object* v_b_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_){
_start:
{
uint8_t v___x_17210__boxed_551_; lean_object* v_res_552_; 
v___x_17210__boxed_551_ = lean_unbox(v___x_542_);
v_res_552_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4(v_init_540_, v_rs_541_, v___x_17210__boxed_551_, v_n_543_, v_b_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_, v___y_549_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
lean_dec(v___y_547_);
lean_dec_ref(v___y_546_);
lean_dec(v___y_545_);
lean_dec_ref(v_n_543_);
lean_dec_ref(v_init_540_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4(lean_object* v_rs_553_, uint8_t v___x_554_, lean_object* v_t_555_, lean_object* v_init_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_){
_start:
{
lean_object* v_root_563_; lean_object* v_tail_564_; lean_object* v___x_565_; 
v_root_563_ = lean_ctor_get(v_t_555_, 0);
v_tail_564_ = lean_ctor_get(v_t_555_, 1);
lean_inc_ref(v_rs_553_);
lean_inc_ref(v_init_556_);
v___x_565_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__4(v_init_556_, v_rs_553_, v___x_554_, v_root_563_, v_init_556_, v___y_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_);
lean_dec_ref(v_init_556_);
if (lean_obj_tag(v___x_565_) == 0)
{
lean_object* v_a_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_602_; 
v_a_566_ = lean_ctor_get(v___x_565_, 0);
v_isSharedCheck_602_ = !lean_is_exclusive(v___x_565_);
if (v_isSharedCheck_602_ == 0)
{
v___x_568_ = v___x_565_;
v_isShared_569_ = v_isSharedCheck_602_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_a_566_);
lean_dec(v___x_565_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_602_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
if (lean_obj_tag(v_a_566_) == 0)
{
lean_object* v_a_570_; lean_object* v___x_572_; 
lean_dec_ref(v_rs_553_);
v_a_570_ = lean_ctor_get(v_a_566_, 0);
lean_inc(v_a_570_);
lean_dec_ref_known(v_a_566_, 1);
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 0, v_a_570_);
v___x_572_ = v___x_568_;
goto v_reusejp_571_;
}
else
{
lean_object* v_reuseFailAlloc_573_; 
v_reuseFailAlloc_573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_573_, 0, v_a_570_);
v___x_572_ = v_reuseFailAlloc_573_;
goto v_reusejp_571_;
}
v_reusejp_571_:
{
return v___x_572_;
}
}
else
{
lean_object* v_a_574_; lean_object* v___x_575_; lean_object* v___x_576_; size_t v_sz_577_; size_t v___x_578_; lean_object* v___x_579_; 
lean_del_object(v___x_568_);
v_a_574_ = lean_ctor_get(v_a_566_, 0);
lean_inc(v_a_574_);
lean_dec_ref_known(v_a_566_, 1);
v___x_575_ = lean_box(0);
v___x_576_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_576_, 0, v___x_575_);
lean_ctor_set(v___x_576_, 1, v_a_574_);
v_sz_577_ = lean_array_size(v_tail_564_);
v___x_578_ = ((size_t)0ULL);
v___x_579_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4_spec__5(v_rs_553_, v___x_554_, v_tail_564_, v_sz_577_, v___x_578_, v___x_576_, v___y_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_);
if (lean_obj_tag(v___x_579_) == 0)
{
lean_object* v_a_580_; lean_object* v___x_582_; uint8_t v_isShared_583_; uint8_t v_isSharedCheck_593_; 
v_a_580_ = lean_ctor_get(v___x_579_, 0);
v_isSharedCheck_593_ = !lean_is_exclusive(v___x_579_);
if (v_isSharedCheck_593_ == 0)
{
v___x_582_ = v___x_579_;
v_isShared_583_ = v_isSharedCheck_593_;
goto v_resetjp_581_;
}
else
{
lean_inc(v_a_580_);
lean_dec(v___x_579_);
v___x_582_ = lean_box(0);
v_isShared_583_ = v_isSharedCheck_593_;
goto v_resetjp_581_;
}
v_resetjp_581_:
{
lean_object* v_fst_584_; 
v_fst_584_ = lean_ctor_get(v_a_580_, 0);
if (lean_obj_tag(v_fst_584_) == 0)
{
lean_object* v_snd_585_; lean_object* v___x_587_; 
v_snd_585_ = lean_ctor_get(v_a_580_, 1);
lean_inc(v_snd_585_);
lean_dec(v_a_580_);
if (v_isShared_583_ == 0)
{
lean_ctor_set(v___x_582_, 0, v_snd_585_);
v___x_587_ = v___x_582_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_588_; 
v_reuseFailAlloc_588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_588_, 0, v_snd_585_);
v___x_587_ = v_reuseFailAlloc_588_;
goto v_reusejp_586_;
}
v_reusejp_586_:
{
return v___x_587_;
}
}
else
{
lean_object* v_val_589_; lean_object* v___x_591_; 
lean_inc_ref(v_fst_584_);
lean_dec(v_a_580_);
v_val_589_ = lean_ctor_get(v_fst_584_, 0);
lean_inc(v_val_589_);
lean_dec_ref_known(v_fst_584_, 1);
if (v_isShared_583_ == 0)
{
lean_ctor_set(v___x_582_, 0, v_val_589_);
v___x_591_ = v___x_582_;
goto v_reusejp_590_;
}
else
{
lean_object* v_reuseFailAlloc_592_; 
v_reuseFailAlloc_592_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_592_, 0, v_val_589_);
v___x_591_ = v_reuseFailAlloc_592_;
goto v_reusejp_590_;
}
v_reusejp_590_:
{
return v___x_591_;
}
}
}
}
else
{
lean_object* v_a_594_; lean_object* v___x_596_; uint8_t v_isShared_597_; uint8_t v_isSharedCheck_601_; 
v_a_594_ = lean_ctor_get(v___x_579_, 0);
v_isSharedCheck_601_ = !lean_is_exclusive(v___x_579_);
if (v_isSharedCheck_601_ == 0)
{
v___x_596_ = v___x_579_;
v_isShared_597_ = v_isSharedCheck_601_;
goto v_resetjp_595_;
}
else
{
lean_inc(v_a_594_);
lean_dec(v___x_579_);
v___x_596_ = lean_box(0);
v_isShared_597_ = v_isSharedCheck_601_;
goto v_resetjp_595_;
}
v_resetjp_595_:
{
lean_object* v___x_599_; 
if (v_isShared_597_ == 0)
{
v___x_599_ = v___x_596_;
goto v_reusejp_598_;
}
else
{
lean_object* v_reuseFailAlloc_600_; 
v_reuseFailAlloc_600_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_600_, 0, v_a_594_);
v___x_599_ = v_reuseFailAlloc_600_;
goto v_reusejp_598_;
}
v_reusejp_598_:
{
return v___x_599_;
}
}
}
}
}
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
lean_dec_ref(v_rs_553_);
v_a_603_ = lean_ctor_get(v___x_565_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_565_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_565_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_565_);
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
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4___boxed(lean_object* v_rs_611_, lean_object* v___x_612_, lean_object* v_t_613_, lean_object* v_init_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_){
_start:
{
uint8_t v___x_17406__boxed_621_; lean_object* v_res_622_; 
v___x_17406__boxed_621_ = lean_unbox(v___x_612_);
v_res_622_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4(v_rs_611_, v___x_17406__boxed_621_, v_t_613_, v_init_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_, v___y_619_);
lean_dec(v___y_619_);
lean_dec_ref(v___y_618_);
lean_dec(v___y_617_);
lean_dec_ref(v___y_616_);
lean_dec(v___y_615_);
lean_dec_ref(v_t_613_);
return v_res_622_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6_spec__8(lean_object* v_msgData_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_){
_start:
{
lean_object* v___x_629_; lean_object* v_env_630_; lean_object* v___x_631_; lean_object* v_mctx_632_; lean_object* v_lctx_633_; lean_object* v_options_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_629_ = lean_st_ref_get(v___y_627_);
v_env_630_ = lean_ctor_get(v___x_629_, 0);
lean_inc_ref(v_env_630_);
lean_dec(v___x_629_);
v___x_631_ = lean_st_ref_get(v___y_625_);
v_mctx_632_ = lean_ctor_get(v___x_631_, 0);
lean_inc_ref(v_mctx_632_);
lean_dec(v___x_631_);
v_lctx_633_ = lean_ctor_get(v___y_624_, 2);
v_options_634_ = lean_ctor_get(v___y_626_, 2);
lean_inc_ref(v_options_634_);
lean_inc_ref(v_lctx_633_);
v___x_635_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_635_, 0, v_env_630_);
lean_ctor_set(v___x_635_, 1, v_mctx_632_);
lean_ctor_set(v___x_635_, 2, v_lctx_633_);
lean_ctor_set(v___x_635_, 3, v_options_634_);
v___x_636_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_636_, 0, v___x_635_);
lean_ctor_set(v___x_636_, 1, v_msgData_623_);
v___x_637_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_637_, 0, v___x_636_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6_spec__8___boxed(lean_object* v_msgData_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6_spec__8(v_msgData_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_);
lean_dec(v___y_642_);
lean_dec_ref(v___y_641_);
lean_dec(v___y_640_);
lean_dec_ref(v___y_639_);
return v_res_644_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_645_; double v___x_646_; 
v___x_645_ = lean_unsigned_to_nat(0u);
v___x_646_ = lean_float_of_nat(v___x_645_);
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg(lean_object* v_cls_650_, lean_object* v_msg_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_){
_start:
{
lean_object* v_ref_657_; lean_object* v___x_658_; lean_object* v_a_659_; lean_object* v___x_661_; uint8_t v_isShared_662_; uint8_t v_isSharedCheck_703_; 
v_ref_657_ = lean_ctor_get(v___y_654_, 5);
v___x_658_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6_spec__8(v_msg_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_);
v_a_659_ = lean_ctor_get(v___x_658_, 0);
v_isSharedCheck_703_ = !lean_is_exclusive(v___x_658_);
if (v_isSharedCheck_703_ == 0)
{
v___x_661_ = v___x_658_;
v_isShared_662_ = v_isSharedCheck_703_;
goto v_resetjp_660_;
}
else
{
lean_inc(v_a_659_);
lean_dec(v___x_658_);
v___x_661_ = lean_box(0);
v_isShared_662_ = v_isSharedCheck_703_;
goto v_resetjp_660_;
}
v_resetjp_660_:
{
lean_object* v___x_663_; lean_object* v_traceState_664_; lean_object* v_env_665_; lean_object* v_nextMacroScope_666_; lean_object* v_ngen_667_; lean_object* v_auxDeclNGen_668_; lean_object* v_cache_669_; lean_object* v_messages_670_; lean_object* v_infoState_671_; lean_object* v_snapshotTasks_672_; lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_702_; 
v___x_663_ = lean_st_ref_take(v___y_655_);
v_traceState_664_ = lean_ctor_get(v___x_663_, 4);
v_env_665_ = lean_ctor_get(v___x_663_, 0);
v_nextMacroScope_666_ = lean_ctor_get(v___x_663_, 1);
v_ngen_667_ = lean_ctor_get(v___x_663_, 2);
v_auxDeclNGen_668_ = lean_ctor_get(v___x_663_, 3);
v_cache_669_ = lean_ctor_get(v___x_663_, 5);
v_messages_670_ = lean_ctor_get(v___x_663_, 6);
v_infoState_671_ = lean_ctor_get(v___x_663_, 7);
v_snapshotTasks_672_ = lean_ctor_get(v___x_663_, 8);
v_isSharedCheck_702_ = !lean_is_exclusive(v___x_663_);
if (v_isSharedCheck_702_ == 0)
{
v___x_674_ = v___x_663_;
v_isShared_675_ = v_isSharedCheck_702_;
goto v_resetjp_673_;
}
else
{
lean_inc(v_snapshotTasks_672_);
lean_inc(v_infoState_671_);
lean_inc(v_messages_670_);
lean_inc(v_cache_669_);
lean_inc(v_traceState_664_);
lean_inc(v_auxDeclNGen_668_);
lean_inc(v_ngen_667_);
lean_inc(v_nextMacroScope_666_);
lean_inc(v_env_665_);
lean_dec(v___x_663_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_702_;
goto v_resetjp_673_;
}
v_resetjp_673_:
{
uint64_t v_tid_676_; lean_object* v_traces_677_; lean_object* v___x_679_; uint8_t v_isShared_680_; uint8_t v_isSharedCheck_701_; 
v_tid_676_ = lean_ctor_get_uint64(v_traceState_664_, sizeof(void*)*1);
v_traces_677_ = lean_ctor_get(v_traceState_664_, 0);
v_isSharedCheck_701_ = !lean_is_exclusive(v_traceState_664_);
if (v_isSharedCheck_701_ == 0)
{
v___x_679_ = v_traceState_664_;
v_isShared_680_ = v_isSharedCheck_701_;
goto v_resetjp_678_;
}
else
{
lean_inc(v_traces_677_);
lean_dec(v_traceState_664_);
v___x_679_ = lean_box(0);
v_isShared_680_ = v_isSharedCheck_701_;
goto v_resetjp_678_;
}
v_resetjp_678_:
{
lean_object* v___x_681_; double v___x_682_; uint8_t v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_691_; 
v___x_681_ = lean_box(0);
v___x_682_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__0);
v___x_683_ = 0;
v___x_684_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__1));
v___x_685_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_685_, 0, v_cls_650_);
lean_ctor_set(v___x_685_, 1, v___x_681_);
lean_ctor_set(v___x_685_, 2, v___x_684_);
lean_ctor_set_float(v___x_685_, sizeof(void*)*3, v___x_682_);
lean_ctor_set_float(v___x_685_, sizeof(void*)*3 + 8, v___x_682_);
lean_ctor_set_uint8(v___x_685_, sizeof(void*)*3 + 16, v___x_683_);
v___x_686_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__2));
v___x_687_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_687_, 0, v___x_685_);
lean_ctor_set(v___x_687_, 1, v_a_659_);
lean_ctor_set(v___x_687_, 2, v___x_686_);
lean_inc(v_ref_657_);
v___x_688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_688_, 0, v_ref_657_);
lean_ctor_set(v___x_688_, 1, v___x_687_);
v___x_689_ = l_Lean_PersistentArray_push___redArg(v_traces_677_, v___x_688_);
if (v_isShared_680_ == 0)
{
lean_ctor_set(v___x_679_, 0, v___x_689_);
v___x_691_ = v___x_679_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v___x_689_);
lean_ctor_set_uint64(v_reuseFailAlloc_700_, sizeof(void*)*1, v_tid_676_);
v___x_691_ = v_reuseFailAlloc_700_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
lean_object* v___x_693_; 
if (v_isShared_675_ == 0)
{
lean_ctor_set(v___x_674_, 4, v___x_691_);
v___x_693_ = v___x_674_;
goto v_reusejp_692_;
}
else
{
lean_object* v_reuseFailAlloc_699_; 
v_reuseFailAlloc_699_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_699_, 0, v_env_665_);
lean_ctor_set(v_reuseFailAlloc_699_, 1, v_nextMacroScope_666_);
lean_ctor_set(v_reuseFailAlloc_699_, 2, v_ngen_667_);
lean_ctor_set(v_reuseFailAlloc_699_, 3, v_auxDeclNGen_668_);
lean_ctor_set(v_reuseFailAlloc_699_, 4, v___x_691_);
lean_ctor_set(v_reuseFailAlloc_699_, 5, v_cache_669_);
lean_ctor_set(v_reuseFailAlloc_699_, 6, v_messages_670_);
lean_ctor_set(v_reuseFailAlloc_699_, 7, v_infoState_671_);
lean_ctor_set(v_reuseFailAlloc_699_, 8, v_snapshotTasks_672_);
v___x_693_ = v_reuseFailAlloc_699_;
goto v_reusejp_692_;
}
v_reusejp_692_:
{
lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_697_; 
v___x_694_ = lean_st_ref_set(v___y_655_, v___x_693_);
v___x_695_ = lean_box(0);
if (v_isShared_662_ == 0)
{
lean_ctor_set(v___x_661_, 0, v___x_695_);
v___x_697_ = v___x_661_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_698_; 
v_reuseFailAlloc_698_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_698_, 0, v___x_695_);
v___x_697_ = v_reuseFailAlloc_698_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
return v___x_697_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___boxed(lean_object* v_cls_704_, lean_object* v_msg_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg(v_cls_704_, v_msg_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
lean_dec(v___y_709_);
lean_dec_ref(v___y_708_);
lean_dec(v___y_707_);
lean_dec_ref(v___y_706_);
return v_res_711_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__1(void){
_start:
{
lean_object* v___x_713_; lean_object* v___x_714_; 
v___x_713_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__0));
v___x_714_ = l_Lean_stringToMessageData(v___x_713_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7(uint8_t v_a_729_, lean_object* v_as_730_, size_t v_sz_731_, size_t v_i_732_, lean_object* v_b_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
uint8_t v___x_740_; 
v___x_740_ = lean_usize_dec_lt(v_i_732_, v_sz_731_);
if (v___x_740_ == 0)
{
lean_object* v___x_741_; 
v___x_741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_741_, 0, v_b_733_);
return v___x_741_;
}
else
{
lean_object* v___x_742_; lean_object* v_traceClass_743_; lean_object* v_a_744_; lean_object* v_rule_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_796_; 
v___x_742_ = lp_aesop_Aesop_TraceOption_forward;
v_traceClass_743_ = lean_ctor_get(v___x_742_, 0);
v_a_744_ = lean_array_uget(v_as_730_, v_i_732_);
v_rule_745_ = lean_ctor_get(v_a_744_, 0);
v_isSharedCheck_796_ = !lean_is_exclusive(v_a_744_);
if (v_isSharedCheck_796_ == 0)
{
lean_object* v_unused_797_; 
v_unused_797_ = lean_ctor_get(v_a_744_, 1);
lean_dec(v_unused_797_);
v___x_747_ = v_a_744_;
v_isShared_748_ = v_isSharedCheck_796_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_rule_745_);
lean_dec(v_a_744_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_796_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
lean_object* v_name_749_; lean_object* v_name_750_; uint8_t v_builder_751_; uint8_t v_phase_752_; uint8_t v_scope_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___y_757_; lean_object* v___y_758_; lean_object* v___y_759_; lean_object* v___y_774_; lean_object* v___y_775_; lean_object* v___y_776_; lean_object* v___y_782_; 
v_name_749_ = lean_ctor_get(v_rule_745_, 1);
lean_inc_ref(v_name_749_);
lean_dec_ref(v_rule_745_);
v_name_750_ = lean_ctor_get(v_name_749_, 0);
lean_inc(v_name_750_);
v_builder_751_ = lean_ctor_get_uint8(v_name_749_, sizeof(void*)*1 + 8);
v_phase_752_ = lean_ctor_get_uint8(v_name_749_, sizeof(void*)*1 + 9);
v_scope_753_ = lean_ctor_get_uint8(v_name_749_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_749_);
v___x_754_ = lean_box(0);
v___x_755_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__1);
switch(v_phase_752_)
{
case 0:
{
lean_object* v___x_793_; 
v___x_793_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__13));
v___y_782_ = v___x_793_;
goto v___jp_781_;
}
case 1:
{
lean_object* v___x_794_; 
v___x_794_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__14));
v___y_782_ = v___x_794_;
goto v___jp_781_;
}
default: 
{
lean_object* v___x_795_; 
v___x_795_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__15));
v___y_782_ = v___x_795_;
goto v___jp_781_;
}
}
v___jp_756_:
{
lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_767_; 
v___x_760_ = lean_string_append(v___y_758_, v___y_759_);
v___x_761_ = lean_string_append(v___x_760_, v___y_757_);
v___x_762_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_750_, v_a_729_);
v___x_763_ = lean_string_append(v___x_761_, v___x_762_);
lean_dec_ref(v___x_762_);
v___x_764_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_764_, 0, v___x_763_);
v___x_765_ = l_Lean_MessageData_ofFormat(v___x_764_);
if (v_isShared_748_ == 0)
{
lean_ctor_set_tag(v___x_747_, 7);
lean_ctor_set(v___x_747_, 1, v___x_765_);
lean_ctor_set(v___x_747_, 0, v___x_755_);
v___x_767_ = v___x_747_;
goto v_reusejp_766_;
}
else
{
lean_object* v_reuseFailAlloc_772_; 
v_reuseFailAlloc_772_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_772_, 0, v___x_755_);
lean_ctor_set(v_reuseFailAlloc_772_, 1, v___x_765_);
v___x_767_ = v_reuseFailAlloc_772_;
goto v_reusejp_766_;
}
v_reusejp_766_:
{
lean_object* v___x_768_; 
lean_inc(v_traceClass_743_);
v___x_768_ = lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg(v_traceClass_743_, v___x_767_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_768_) == 0)
{
size_t v___x_769_; size_t v___x_770_; 
lean_dec_ref_known(v___x_768_, 1);
v___x_769_ = ((size_t)1ULL);
v___x_770_ = lean_usize_add(v_i_732_, v___x_769_);
v_i_732_ = v___x_770_;
v_b_733_ = v___x_754_;
goto _start;
}
else
{
return v___x_768_;
}
}
}
v___jp_773_:
{
lean_object* v___x_777_; lean_object* v___x_778_; 
v___x_777_ = lean_string_append(v___y_774_, v___y_776_);
v___x_778_ = lean_string_append(v___x_777_, v___y_775_);
if (v_scope_753_ == 0)
{
lean_object* v___x_779_; 
v___x_779_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__2));
v___y_757_ = v___y_775_;
v___y_758_ = v___x_778_;
v___y_759_ = v___x_779_;
goto v___jp_756_;
}
else
{
lean_object* v___x_780_; 
v___x_780_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__3));
v___y_757_ = v___y_775_;
v___y_758_ = v___x_778_;
v___y_759_ = v___x_780_;
goto v___jp_756_;
}
}
v___jp_781_:
{
lean_object* v___x_783_; lean_object* v___x_784_; 
v___x_783_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__4));
lean_inc_ref(v___y_782_);
v___x_784_ = lean_string_append(v___y_782_, v___x_783_);
switch(v_builder_751_)
{
case 0:
{
lean_object* v___x_785_; 
v___x_785_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__5));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_785_;
goto v___jp_773_;
}
case 1:
{
lean_object* v___x_786_; 
v___x_786_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__6));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_786_;
goto v___jp_773_;
}
case 2:
{
lean_object* v___x_787_; 
v___x_787_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__7));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_787_;
goto v___jp_773_;
}
case 3:
{
lean_object* v___x_788_; 
v___x_788_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__8));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_788_;
goto v___jp_773_;
}
case 4:
{
lean_object* v___x_789_; 
v___x_789_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__9));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_789_;
goto v___jp_773_;
}
case 5:
{
lean_object* v___x_790_; 
v___x_790_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__10));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_790_;
goto v___jp_773_;
}
case 6:
{
lean_object* v___x_791_; 
v___x_791_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__11));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_791_;
goto v___jp_773_;
}
default: 
{
lean_object* v___x_792_; 
v___x_792_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___closed__12));
v___y_774_ = v___x_784_;
v___y_775_ = v___x_783_;
v___y_776_ = v___x_792_;
goto v___jp_773_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7___boxed(lean_object* v_a_798_, lean_object* v_as_799_, lean_object* v_sz_800_, lean_object* v_i_801_, lean_object* v_b_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_){
_start:
{
uint8_t v_a_17673__boxed_809_; size_t v_sz_boxed_810_; size_t v_i_boxed_811_; lean_object* v_res_812_; 
v_a_17673__boxed_809_ = lean_unbox(v_a_798_);
v_sz_boxed_810_ = lean_unbox_usize(v_sz_800_);
lean_dec(v_sz_800_);
v_i_boxed_811_ = lean_unbox_usize(v_i_801_);
lean_dec(v_i_801_);
v_res_812_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7(v_a_17673__boxed_809_, v_as_799_, v_sz_boxed_810_, v_i_boxed_811_, v_b_802_, v___y_803_, v___y_804_, v___y_805_, v___y_806_, v___y_807_);
lean_dec(v___y_807_);
lean_dec_ref(v___y_806_);
lean_dec(v___y_805_);
lean_dec_ref(v___y_804_);
lean_dec(v___y_803_);
lean_dec_ref(v_as_799_);
return v_res_812_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg(lean_object* v_opt_813_, lean_object* v___y_814_){
_start:
{
lean_object* v_options_816_; lean_object* v_option_817_; uint8_t v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; 
v_options_816_ = lean_ctor_get(v___y_814_, 2);
v_option_817_ = lean_ctor_get(v_opt_813_, 1);
v___x_818_ = lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0(v_options_816_, v_option_817_);
v___x_819_ = lean_box(v___x_818_);
v___x_820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_820_, 0, v___x_819_);
return v___x_820_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg___boxed(lean_object* v_opt_821_, lean_object* v___y_822_, lean_object* v___y_823_){
_start:
{
lean_object* v_res_824_; 
v_res_824_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg(v_opt_821_, v___y_822_);
lean_dec_ref(v___y_822_);
lean_dec_ref(v_opt_821_);
return v_res_824_;
}
}
static lean_object* _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__0(void){
_start:
{
lean_object* v___x_825_; 
v___x_825_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__1(lean_box(0));
return v___x_825_;
}
}
static lean_object* _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__1(void){
_start:
{
lean_object* v___x_826_; 
v___x_826_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__2(lean_box(0));
return v___x_826_;
}
}
static lean_object* _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__2(void){
_start:
{
lean_object* v___x_827_; 
v___x_827_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__3(lean_box(0));
return v___x_827_;
}
}
static lean_object* _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3(void){
_start:
{
lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_828_ = lean_obj_once(&lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__2, &lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__2_once, _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__2);
v___x_829_ = lean_obj_once(&lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__1, &lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__1_once, _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__1);
v___x_830_ = lean_obj_once(&lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__0, &lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__0_once, _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__0);
v___x_831_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_831_, 0, v___x_830_);
lean_ctor_set(v___x_831_, 1, v___x_829_);
lean_ctor_set(v___x_831_, 2, v___x_828_);
return v___x_831_;
}
}
static lean_object* _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__5(void){
_start:
{
lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; 
v___x_834_ = ((lean_object*)(lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__4));
v___x_835_ = lean_obj_once(&lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3, &lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3_once, _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3);
v___x_836_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_836_, 0, v___x_835_);
lean_ctor_set(v___x_836_, 1, v___x_834_);
return v___x_836_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0(lean_object* v_rs_837_, lean_object* v_goal_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_){
_start:
{
lean_object* v_options_845_; lean_object* v___x_846_; uint8_t v___x_847_; 
v_options_845_ = lean_ctor_get(v___y_842_, 2);
v___x_846_ = lp_aesop_Aesop_aesop_dev_statefulForward;
v___x_847_ = lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0(v_options_845_, v___x_846_);
if (v___x_847_ == 0)
{
lean_object* v___x_848_; lean_object* v___x_849_; 
lean_dec(v_goal_838_);
lean_dec_ref(v_rs_837_);
v___x_848_ = lean_obj_once(&lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__5, &lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__5_once, _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__5);
v___x_849_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
return v___x_849_;
}
else
{
lean_object* v_lctx_850_; lean_object* v_decls_851_; lean_object* v_fs_852_; lean_object* v___x_853_; 
v_lctx_850_ = lean_ctor_get(v___y_840_, 2);
v_decls_851_ = lean_ctor_get(v_lctx_850_, 1);
v_fs_852_ = lean_obj_once(&lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3, &lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3_once, _init_lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___closed__3);
lean_inc_ref(v_rs_837_);
v___x_853_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__4(v_rs_837_, v___x_847_, v_decls_851_, v_fs_852_, v___y_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_);
if (lean_obj_tag(v___x_853_) == 0)
{
lean_object* v_a_854_; lean_object* v___x_855_; 
v_a_854_ = lean_ctor_get(v___x_853_, 0);
lean_inc(v_a_854_);
lean_dec_ref_known(v___x_853_, 1);
v___x_855_ = l_Lean_MVarId_getType(v_goal_838_, v___y_840_, v___y_841_, v___y_842_, v___y_843_);
if (lean_obj_tag(v___x_855_) == 0)
{
lean_object* v_a_856_; lean_object* v___x_857_; 
v_a_856_ = lean_ctor_get(v___x_855_, 0);
lean_inc(v_a_856_);
lean_dec_ref_known(v___x_855_, 1);
lean_inc_ref(v_rs_837_);
v___x_857_ = lp_aesop_Aesop_LocalRuleSet_forwardRulePatternSubstsInExpr(v_rs_837_, v_a_856_, v___y_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_);
if (lean_obj_tag(v___x_857_) == 0)
{
lean_object* v_a_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v_a_861_; lean_object* v___x_863_; uint8_t v_isShared_864_; uint8_t v_isSharedCheck_886_; 
v_a_858_ = lean_ctor_get(v___x_857_, 0);
lean_inc(v_a_858_);
lean_dec_ref_known(v___x_857_, 1);
v___x_859_ = lp_aesop_Aesop_TraceOption_forward;
v___x_860_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg(v___x_859_, v___y_842_);
v_a_861_ = lean_ctor_get(v___x_860_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v___x_860_);
if (v_isSharedCheck_886_ == 0)
{
v___x_863_ = v___x_860_;
v_isShared_864_ = v_isSharedCheck_886_;
goto v_resetjp_862_;
}
else
{
lean_inc(v_a_861_);
lean_dec(v___x_860_);
v___x_863_ = lean_box(0);
v_isShared_864_ = v_isSharedCheck_886_;
goto v_resetjp_862_;
}
v_resetjp_862_:
{
lean_object* v___x_865_; lean_object* v___x_866_; uint8_t v___x_872_; 
v___x_865_ = lp_aesop_Aesop_ForwardState_enqueueTargetPatSubsts(v_a_858_, v_a_854_);
lean_dec(v_a_858_);
v___x_866_ = lp_aesop_Aesop_LocalRuleSet_constForwardRuleMatches(v_rs_837_);
lean_dec_ref(v_rs_837_);
v___x_872_ = lean_unbox(v_a_861_);
if (v___x_872_ == 0)
{
lean_dec(v_a_861_);
goto v___jp_867_;
}
else
{
lean_object* v___x_873_; size_t v_sz_874_; size_t v___x_875_; uint8_t v___x_876_; lean_object* v___x_877_; 
v___x_873_ = lean_box(0);
v_sz_874_ = lean_array_size(v___x_866_);
v___x_875_ = ((size_t)0ULL);
v___x_876_ = lean_unbox(v_a_861_);
lean_dec(v_a_861_);
v___x_877_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__7(v___x_876_, v___x_866_, v_sz_874_, v___x_875_, v___x_873_, v___y_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_);
if (lean_obj_tag(v___x_877_) == 0)
{
lean_dec_ref_known(v___x_877_, 1);
goto v___jp_867_;
}
else
{
lean_object* v_a_878_; lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_885_; 
lean_dec_ref(v___x_866_);
lean_dec_ref(v___x_865_);
lean_del_object(v___x_863_);
v_a_878_ = lean_ctor_get(v___x_877_, 0);
v_isSharedCheck_885_ = !lean_is_exclusive(v___x_877_);
if (v_isSharedCheck_885_ == 0)
{
v___x_880_ = v___x_877_;
v_isShared_881_ = v_isSharedCheck_885_;
goto v_resetjp_879_;
}
else
{
lean_inc(v_a_878_);
lean_dec(v___x_877_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_885_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v___x_883_; 
if (v_isShared_881_ == 0)
{
v___x_883_ = v___x_880_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v_a_878_);
v___x_883_ = v_reuseFailAlloc_884_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
return v___x_883_;
}
}
}
}
v___jp_867_:
{
lean_object* v___x_868_; lean_object* v___x_870_; 
v___x_868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_868_, 0, v___x_865_);
lean_ctor_set(v___x_868_, 1, v___x_866_);
if (v_isShared_864_ == 0)
{
lean_ctor_set(v___x_863_, 0, v___x_868_);
v___x_870_ = v___x_863_;
goto v_reusejp_869_;
}
else
{
lean_object* v_reuseFailAlloc_871_; 
v_reuseFailAlloc_871_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_871_, 0, v___x_868_);
v___x_870_ = v_reuseFailAlloc_871_;
goto v_reusejp_869_;
}
v_reusejp_869_:
{
return v___x_870_;
}
}
}
}
else
{
lean_object* v_a_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_894_; 
lean_dec(v_a_854_);
lean_dec_ref(v_rs_837_);
v_a_887_ = lean_ctor_get(v___x_857_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_894_ == 0)
{
v___x_889_ = v___x_857_;
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_a_887_);
lean_dec(v___x_857_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___x_892_; 
if (v_isShared_890_ == 0)
{
v___x_892_ = v___x_889_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v_a_887_);
v___x_892_ = v_reuseFailAlloc_893_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
return v___x_892_;
}
}
}
}
else
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_902_; 
lean_dec(v_a_854_);
lean_dec_ref(v_rs_837_);
v_a_895_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_902_ == 0)
{
v___x_897_ = v___x_855_;
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___x_855_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_900_; 
if (v_isShared_898_ == 0)
{
v___x_900_ = v___x_897_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_a_895_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
return v___x_900_;
}
}
}
}
else
{
lean_object* v_a_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_910_; 
lean_dec(v_goal_838_);
lean_dec_ref(v_rs_837_);
v_a_903_ = lean_ctor_get(v___x_853_, 0);
v_isSharedCheck_910_ = !lean_is_exclusive(v___x_853_);
if (v_isSharedCheck_910_ == 0)
{
v___x_905_ = v___x_853_;
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_a_903_);
lean_dec(v___x_853_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
lean_object* v___x_908_; 
if (v_isShared_906_ == 0)
{
v___x_908_ = v___x_905_;
goto v_reusejp_907_;
}
else
{
lean_object* v_reuseFailAlloc_909_; 
v_reuseFailAlloc_909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_909_, 0, v_a_903_);
v___x_908_ = v_reuseFailAlloc_909_;
goto v_reusejp_907_;
}
v_reusejp_907_:
{
return v___x_908_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___boxed(lean_object* v_rs_911_, lean_object* v_goal_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_){
_start:
{
lean_object* v_res_919_; 
v_res_919_ = lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0(v_rs_911_, v_goal_912_, v___y_913_, v___y_914_, v___y_915_, v___y_916_, v___y_917_);
lean_dec(v___y_917_);
lean_dec_ref(v___y_916_);
lean_dec(v___y_915_);
lean_dec_ref(v___y_914_);
lean_dec(v___y_913_);
return v_res_919_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(lean_object* v_goal_920_, lean_object* v_rs_921_, lean_object* v_a_922_, lean_object* v_a_923_, lean_object* v_a_924_, lean_object* v_a_925_, lean_object* v_a_926_){
_start:
{
lean_object* v_options_928_; lean_object* v___f_929_; lean_object* v___y_973_; lean_object* v___x_977_; uint8_t v___x_978_; 
v_options_928_ = lean_ctor_get(v_a_925_, 2);
lean_inc(v_goal_920_);
v___f_929_ = lean_alloc_closure((void*)(lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___lam__0___boxed), 8, 2);
lean_closure_set(v___f_929_, 0, v_rs_921_);
lean_closure_set(v___f_929_, 1, v_goal_920_);
v___x_977_ = lp_aesop_Aesop_aesop_collectStats;
v___x_978_ = lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__0(v_options_928_, v___x_977_);
if (v___x_978_ == 0)
{
lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v_a_981_; uint8_t v___x_982_; 
v___x_979_ = lp_aesop_Aesop_TraceOption_stats;
v___x_980_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg(v___x_979_, v_a_925_);
v_a_981_ = lean_ctor_get(v___x_980_, 0);
lean_inc(v_a_981_);
v___x_982_ = lean_unbox(v_a_981_);
lean_dec(v_a_981_);
if (v___x_982_ == 0)
{
lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; uint8_t v___x_986_; 
lean_dec_ref(v___x_980_);
v___x_983_ = lp_aesop_Aesop_aesop_stats_file;
v___x_984_ = lp_aesop_Lean_Option_get___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__9(v_options_928_, v___x_983_);
v___x_985_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg___closed__1));
v___x_986_ = lean_string_dec_eq(v___x_984_, v___x_985_);
lean_dec_ref(v___x_984_);
if (v___x_986_ == 0)
{
goto v___jp_930_;
}
else
{
lean_object* v___x_987_; 
v___x_987_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg(v_goal_920_, v___f_929_, v_a_922_, v_a_923_, v_a_924_, v_a_925_, v_a_926_);
return v___x_987_;
}
}
else
{
v___y_973_ = v___x_980_;
goto v___jp_972_;
}
}
else
{
goto v___jp_930_;
}
v___jp_930_:
{
lean_object* v___x_931_; lean_object* v___x_932_; 
v___x_931_ = lean_io_mono_nanos_now();
v___x_932_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg(v_goal_920_, v___f_929_, v_a_922_, v_a_923_, v_a_924_, v_a_925_, v_a_926_);
if (lean_obj_tag(v___x_932_) == 0)
{
lean_object* v_a_933_; lean_object* v___x_935_; uint8_t v_isShared_936_; uint8_t v_isSharedCheck_971_; 
v_a_933_ = lean_ctor_get(v___x_932_, 0);
v_isSharedCheck_971_ = !lean_is_exclusive(v___x_932_);
if (v_isSharedCheck_971_ == 0)
{
v___x_935_ = v___x_932_;
v_isShared_936_ = v_isSharedCheck_971_;
goto v_resetjp_934_;
}
else
{
lean_inc(v_a_933_);
lean_dec(v___x_932_);
v___x_935_ = lean_box(0);
v_isShared_936_ = v_isSharedCheck_971_;
goto v_resetjp_934_;
}
v_resetjp_934_:
{
lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v_stats_939_; lean_object* v_rulePatternCache_940_; lean_object* v___x_942_; uint8_t v_isShared_943_; uint8_t v_isSharedCheck_970_; 
v___x_937_ = lean_io_mono_nanos_now();
v___x_938_ = lean_st_ref_take(v_a_922_);
v_stats_939_ = lean_ctor_get(v___x_938_, 1);
v_rulePatternCache_940_ = lean_ctor_get(v___x_938_, 0);
v_isSharedCheck_970_ = !lean_is_exclusive(v___x_938_);
if (v_isSharedCheck_970_ == 0)
{
v___x_942_ = v___x_938_;
v_isShared_943_ = v_isSharedCheck_970_;
goto v_resetjp_941_;
}
else
{
lean_inc(v_stats_939_);
lean_inc(v_rulePatternCache_940_);
lean_dec(v___x_938_);
v___x_942_ = lean_box(0);
v_isShared_943_ = v_isSharedCheck_970_;
goto v_resetjp_941_;
}
v_resetjp_941_:
{
lean_object* v_total_944_; lean_object* v_configParsing_945_; lean_object* v_ruleSetConstruction_946_; lean_object* v_search_947_; lean_object* v_ruleSelection_948_; lean_object* v_script_949_; lean_object* v_forwardState_950_; lean_object* v_scriptGenerated_951_; lean_object* v_ruleStats_952_; lean_object* v_goalStats_953_; lean_object* v___x_955_; uint8_t v_isShared_956_; uint8_t v_isSharedCheck_969_; 
v_total_944_ = lean_ctor_get(v_stats_939_, 0);
v_configParsing_945_ = lean_ctor_get(v_stats_939_, 1);
v_ruleSetConstruction_946_ = lean_ctor_get(v_stats_939_, 2);
v_search_947_ = lean_ctor_get(v_stats_939_, 3);
v_ruleSelection_948_ = lean_ctor_get(v_stats_939_, 4);
v_script_949_ = lean_ctor_get(v_stats_939_, 5);
v_forwardState_950_ = lean_ctor_get(v_stats_939_, 6);
v_scriptGenerated_951_ = lean_ctor_get(v_stats_939_, 7);
v_ruleStats_952_ = lean_ctor_get(v_stats_939_, 8);
v_goalStats_953_ = lean_ctor_get(v_stats_939_, 9);
v_isSharedCheck_969_ = !lean_is_exclusive(v_stats_939_);
if (v_isSharedCheck_969_ == 0)
{
v___x_955_ = v_stats_939_;
v_isShared_956_ = v_isSharedCheck_969_;
goto v_resetjp_954_;
}
else
{
lean_inc(v_goalStats_953_);
lean_inc(v_ruleStats_952_);
lean_inc(v_scriptGenerated_951_);
lean_inc(v_forwardState_950_);
lean_inc(v_script_949_);
lean_inc(v_ruleSelection_948_);
lean_inc(v_search_947_);
lean_inc(v_ruleSetConstruction_946_);
lean_inc(v_configParsing_945_);
lean_inc(v_total_944_);
lean_dec(v_stats_939_);
v___x_955_ = lean_box(0);
v_isShared_956_ = v_isSharedCheck_969_;
goto v_resetjp_954_;
}
v_resetjp_954_:
{
lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_960_; 
v___x_957_ = lean_nat_sub(v___x_937_, v___x_931_);
lean_dec(v___x_931_);
lean_dec(v___x_937_);
v___x_958_ = lean_nat_add(v_forwardState_950_, v___x_957_);
lean_dec(v___x_957_);
lean_dec(v_forwardState_950_);
if (v_isShared_956_ == 0)
{
lean_ctor_set(v___x_955_, 6, v___x_958_);
v___x_960_ = v___x_955_;
goto v_reusejp_959_;
}
else
{
lean_object* v_reuseFailAlloc_968_; 
v_reuseFailAlloc_968_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_968_, 0, v_total_944_);
lean_ctor_set(v_reuseFailAlloc_968_, 1, v_configParsing_945_);
lean_ctor_set(v_reuseFailAlloc_968_, 2, v_ruleSetConstruction_946_);
lean_ctor_set(v_reuseFailAlloc_968_, 3, v_search_947_);
lean_ctor_set(v_reuseFailAlloc_968_, 4, v_ruleSelection_948_);
lean_ctor_set(v_reuseFailAlloc_968_, 5, v_script_949_);
lean_ctor_set(v_reuseFailAlloc_968_, 6, v___x_958_);
lean_ctor_set(v_reuseFailAlloc_968_, 7, v_scriptGenerated_951_);
lean_ctor_set(v_reuseFailAlloc_968_, 8, v_ruleStats_952_);
lean_ctor_set(v_reuseFailAlloc_968_, 9, v_goalStats_953_);
v___x_960_ = v_reuseFailAlloc_968_;
goto v_reusejp_959_;
}
v_reusejp_959_:
{
lean_object* v___x_962_; 
if (v_isShared_943_ == 0)
{
lean_ctor_set(v___x_942_, 1, v___x_960_);
v___x_962_ = v___x_942_;
goto v_reusejp_961_;
}
else
{
lean_object* v_reuseFailAlloc_967_; 
v_reuseFailAlloc_967_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_967_, 0, v_rulePatternCache_940_);
lean_ctor_set(v_reuseFailAlloc_967_, 1, v___x_960_);
v___x_962_ = v_reuseFailAlloc_967_;
goto v_reusejp_961_;
}
v_reusejp_961_:
{
lean_object* v___x_963_; lean_object* v___x_965_; 
v___x_963_ = lean_st_ref_set(v_a_922_, v___x_962_);
if (v_isShared_936_ == 0)
{
v___x_965_ = v___x_935_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v_a_933_);
v___x_965_ = v_reuseFailAlloc_966_;
goto v_reusejp_964_;
}
v_reusejp_964_:
{
return v___x_965_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_931_);
return v___x_932_;
}
}
v___jp_972_:
{
lean_object* v_a_974_; uint8_t v___x_975_; 
v_a_974_ = lean_ctor_get(v___y_973_, 0);
lean_inc(v_a_974_);
lean_dec_ref(v___y_973_);
v___x_975_ = lean_unbox(v_a_974_);
lean_dec(v_a_974_);
if (v___x_975_ == 0)
{
lean_object* v___x_976_; 
v___x_976_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__8___redArg(v_goal_920_, v___f_929_, v_a_922_, v_a_923_, v_a_924_, v_a_925_, v_a_926_);
return v___x_976_;
}
else
{
goto v___jp_930_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState___boxed(lean_object* v_goal_988_, lean_object* v_rs_989_, lean_object* v_a_990_, lean_object* v_a_991_, lean_object* v_a_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_){
_start:
{
lean_object* v_res_996_; 
v_res_996_ = lp_aesop_Aesop_LocalRuleSet_mkInitialForwardState(v_goal_988_, v_rs_989_, v_a_990_, v_a_991_, v_a_992_, v_a_993_, v_a_994_);
lean_dec(v_a_994_);
lean_dec_ref(v_a_993_);
lean_dec(v_a_992_);
lean_dec_ref(v_a_991_);
lean_dec(v_a_990_);
return v_res_996_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5(lean_object* v_opt_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_){
_start:
{
lean_object* v___x_1004_; 
v___x_1004_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___redArg(v_opt_997_, v___y_1001_);
return v___x_1004_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5___boxed(lean_object* v_opt_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_){
_start:
{
lean_object* v_res_1012_; 
v_res_1012_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__5(v_opt_1005_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_, v___y_1010_);
lean_dec(v___y_1010_);
lean_dec_ref(v___y_1009_);
lean_dec(v___y_1008_);
lean_dec_ref(v___y_1007_);
lean_dec(v___y_1006_);
lean_dec_ref(v_opt_1005_);
return v_res_1012_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6(lean_object* v_cls_1013_, lean_object* v_msg_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_){
_start:
{
lean_object* v___x_1021_; 
v___x_1021_ = lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___redArg(v_cls_1013_, v_msg_1014_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
return v___x_1021_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6___boxed(lean_object* v_cls_1022_, lean_object* v_msg_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_){
_start:
{
lean_object* v_res_1030_; 
v_res_1030_ = lp_aesop_Lean_addTrace___at___00Aesop_LocalRuleSet_mkInitialForwardState_spec__6(v_cls_1022_, v_msg_1023_, v___y_1024_, v___y_1025_, v___y_1026_, v___y_1027_, v___y_1028_);
lean_dec(v___y_1028_);
lean_dec_ref(v___y_1027_);
lean_dec(v___y_1026_);
lean_dec_ref(v___y_1025_);
lean_dec(v___y_1024_);
return v_res_1030_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_State(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleSet(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Forward_State_Initial(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Forward_State_Initial(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_State(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleSet(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_State_Initial(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State_Initial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Forward_State_Initial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Forward_State_Initial(builtin);
}
#ifdef __cplusplus
}
#endif
