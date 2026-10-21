// Lean compiler output
// Module: Mathlib.Lean.ContextInfo
// Imports: public import Init public meta import Init public meta import Mathlib.Lean.Elab.Tactic.Meta public import Mathlib.Tactic.Linter.Header
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
lean_object* lp_mathlib_Lean_Elab_runTactic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_PersistentArray_toList___redArg(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Meta_withLocalInstancesImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_mk_io_user_error(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_append(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_toString(lean_object*);
lean_object* l_Lean_InternalExceptionId_getName(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
extern lean_object* l_Lean_maxRecDepth;
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* l_Lean_Core_getMaxHeartbeats(lean_object*);
extern lean_object* l_Lean_firstFrontendMacroScope;
lean_object* lean_nat_add(lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* lean_io_get_num_heartbeats();
extern lean_object* l_Lean_inheritedTraceOptions;
extern lean_object* l_Lean_diagnostics;
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
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedMetavarDecl_default;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_instInhabitedCommandElabM(lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "internal exception "};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "internal exception #"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " (unknown)"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__8;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13;
static const lean_array_object lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__0(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__7;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__1(lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Lean_Elab_ContextInfo_runTactic_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Lean_Elab_ContextInfo_runTactic_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Lean.Data.PersistentHashMap"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Lean.PersistentHashMap.find!"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "key is not in the map"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Mathlib.Lean.ContextInfo"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Lean.Elab.ContextInfo.runTactic"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "ContextInfo.runTactic: `goal` must be an element of `i.goalsBefore`"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_ContextInfo_runTacticCode_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_ContextInfo_runTacticCode_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "Lean.Elab.ContextInfo.runTacticCapturingInfoTree"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 85, .m_capacity = 85, .m_length = 84, .m_data = "ContextInfo.runTacticCapturingInfoTree: `goal` must be an element of `i.goalsBefore`"};
static const lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0(lean_object* v_opts_1_, lean_object* v_opt_2_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0___boxed(lean_object* v_opts_11_, lean_object* v_opt_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0(v_opts_11_, v_opt_12_);
lean_dec_ref(v_opt_12_);
lean_dec_ref(v_opts_11_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1(lean_object* v_opts_15_, lean_object* v_opt_16_){
_start:
{
lean_object* v_name_17_; lean_object* v_defValue_18_; lean_object* v_map_19_; lean_object* v___x_20_; 
v_name_17_ = lean_ctor_get(v_opt_16_, 0);
v_defValue_18_ = lean_ctor_get(v_opt_16_, 1);
v_map_19_ = lean_ctor_get(v_opts_15_, 0);
v___x_20_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_19_, v_name_17_);
if (lean_obj_tag(v___x_20_) == 0)
{
lean_inc(v_defValue_18_);
return v_defValue_18_;
}
else
{
lean_object* v_val_21_; 
v_val_21_ = lean_ctor_get(v___x_20_, 0);
lean_inc(v_val_21_);
lean_dec_ref_known(v___x_20_, 1);
if (lean_obj_tag(v_val_21_) == 3)
{
lean_object* v_v_22_; 
v_v_22_ = lean_ctor_get(v_val_21_, 0);
lean_inc(v_v_22_);
lean_dec_ref_known(v_val_21_, 1);
return v_v_22_;
}
else
{
lean_dec(v_val_21_);
lean_inc(v_defValue_18_);
return v_defValue_18_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1___boxed(lean_object* v_opts_23_, lean_object* v_opt_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1(v_opts_23_, v_opt_24_);
lean_dec_ref(v_opt_24_);
lean_dec_ref(v_opts_23_);
return v_res_25_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = l_Lean_maxRecDepth;
v___x_30_ = l_Lean_Options_empty;
v___x_31_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = l_Lean_Options_empty;
v___x_33_ = l_Lean_Core_getMaxHeartbeats(v___x_32_);
return v___x_33_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5(void){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_34_ = lean_unsigned_to_nat(1u);
v___x_35_ = l_Lean_firstFrontendMacroScope;
v___x_36_ = lean_nat_add(v___x_35_, v___x_34_);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__6(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_37_ = lean_unsigned_to_nat(32u);
v___x_38_ = lean_mk_empty_array_with_capacity(v___x_37_);
v___x_39_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_39_, 0, v___x_38_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7(void){
_start:
{
size_t v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_40_ = ((size_t)5ULL);
v___x_41_ = lean_unsigned_to_nat(0u);
v___x_42_ = lean_unsigned_to_nat(32u);
v___x_43_ = lean_mk_empty_array_with_capacity(v___x_42_);
v___x_44_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__6, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__6_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__6);
v___x_45_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
lean_ctor_set(v___x_45_, 2, v___x_41_);
lean_ctor_set(v___x_45_, 3, v___x_41_);
lean_ctor_set_usize(v___x_45_, 4, v___x_40_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__8(void){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_46_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__8, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__8_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__8);
v___x_48_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9);
v___x_50_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
lean_ctor_set(v___x_50_, 1, v___x_49_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_51_ = l_Lean_NameSet_empty;
v___x_52_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7);
v___x_53_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_53_, 0, v___x_52_);
lean_ctor_set(v___x_53_, 1, v___x_52_);
lean_ctor_set(v___x_53_, 2, v___x_51_);
return v___x_53_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12(void){
_start:
{
lean_object* v___x_54_; uint64_t v___x_55_; lean_object* v___x_56_; 
v___x_54_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7);
v___x_55_ = 0ULL;
v___x_56_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_56_, 0, v___x_54_);
lean_ctor_set_uint64(v___x_56_, sizeof(void*)*1, v___x_55_);
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; uint8_t v___x_59_; lean_object* v___x_60_; 
v___x_57_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__7);
v___x_58_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__9);
v___x_59_ = 1;
v___x_60_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_60_, 0, v___x_58_);
lean_ctor_set(v___x_60_, 1, v___x_58_);
lean_ctor_set(v___x_60_, 2, v___x_57_);
lean_ctor_set_uint8(v___x_60_, sizeof(void*)*3, v___x_59_);
return v___x_60_;
}
}
static uint8_t _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; uint8_t v___x_65_; 
v___x_63_ = l_Lean_diagnostics;
v___x_64_ = l_Lean_Options_empty;
v___x_65_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0(v___x_64_, v___x_63_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg(lean_object* v_info_66_, lean_object* v_x_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v_toCommandContextInfo_71_; lean_object* v_parentDecl_x3f_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_323_; 
v_toCommandContextInfo_71_ = lean_ctor_get(v_info_66_, 0);
v_parentDecl_x3f_72_ = lean_ctor_get(v_info_66_, 1);
v_isSharedCheck_323_ = !lean_is_exclusive(v_info_66_);
if (v_isSharedCheck_323_ == 0)
{
lean_object* v_unused_324_; 
v_unused_324_ = lean_ctor_get(v_info_66_, 2);
lean_dec(v_unused_324_);
v___x_74_ = v_info_66_;
v_isShared_75_ = v_isSharedCheck_323_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_parentDecl_x3f_72_);
lean_inc(v_toCommandContextInfo_71_);
lean_dec(v_info_66_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_323_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v_env_76_; lean_object* v_options_77_; lean_object* v_currNamespace_78_; lean_object* v_openDecls_79_; lean_object* v_ngen_80_; lean_object* v_fileName_81_; lean_object* v_fileMap_82_; lean_object* v_ref_83_; lean_object* v_a_85_; lean_object* v_a_92_; lean_object* v___y_95_; uint8_t v___y_96_; lean_object* v___y_97_; uint8_t v___y_98_; lean_object* v_fileName_99_; lean_object* v_fileMap_100_; lean_object* v_currRecDepth_101_; lean_object* v_ref_102_; lean_object* v_currNamespace_103_; lean_object* v_openDecls_104_; lean_object* v_initHeartbeats_105_; lean_object* v_maxHeartbeats_106_; lean_object* v_quotContext_107_; lean_object* v_currMacroScope_108_; lean_object* v_cancelTk_x3f_109_; uint8_t v_suppressElabErrors_110_; lean_object* v_inheritedTraceOptions_111_; lean_object* v___y_112_; lean_object* v___y_177_; uint8_t v___y_178_; lean_object* v___y_179_; uint8_t v___y_180_; lean_object* v___y_181_; lean_object* v___y_182_; lean_object* v___y_197_; lean_object* v___y_198_; uint8_t v___y_199_; lean_object* v___y_200_; lean_object* v___y_201_; lean_object* v___y_202_; uint8_t v___y_203_; uint8_t v___y_204_; uint8_t v___x_224_; lean_object* v_env_225_; lean_object* v___x_226_; uint8_t v___y_228_; lean_object* v___y_229_; lean_object* v___y_230_; lean_object* v___y_231_; uint8_t v___y_232_; lean_object* v___y_233_; lean_object* v___y_234_; uint8_t v___y_264_; lean_object* v___y_265_; lean_object* v___y_266_; lean_object* v___y_267_; lean_object* v___y_268_; uint8_t v___y_269_; uint8_t v___y_270_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___y_300_; 
v_env_76_ = lean_ctor_get(v_toCommandContextInfo_71_, 0);
lean_inc_ref(v_env_76_);
v_options_77_ = lean_ctor_get(v_toCommandContextInfo_71_, 4);
lean_inc_ref(v_options_77_);
v_currNamespace_78_ = lean_ctor_get(v_toCommandContextInfo_71_, 5);
lean_inc(v_currNamespace_78_);
v_openDecls_79_ = lean_ctor_get(v_toCommandContextInfo_71_, 6);
lean_inc(v_openDecls_79_);
v_ngen_80_ = lean_ctor_get(v_toCommandContextInfo_71_, 7);
lean_inc_ref(v_ngen_80_);
lean_dec_ref(v_toCommandContextInfo_71_);
v_fileName_81_ = lean_ctor_get(v_a_68_, 0);
v_fileMap_82_ = lean_ctor_get(v_a_68_, 1);
v_ref_83_ = lean_ctor_get(v_a_68_, 7);
v___x_224_ = 0;
v_env_225_ = l_Lean_Environment_setExporting(v_env_76_, v___x_224_);
v___x_226_ = l_Lean_Options_empty;
v___x_290_ = lean_unsigned_to_nat(0u);
v___x_291_ = lean_unsigned_to_nat(1000u);
v___x_292_ = lean_box(0);
v___x_293_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4);
v___x_294_ = lean_box(0);
v___x_295_ = l_Lean_firstFrontendMacroScope;
v___x_296_ = lean_box(0);
v___x_297_ = lean_unsigned_to_nat(1u);
v___x_298_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5);
if (lean_obj_tag(v_parentDecl_x3f_72_) == 0)
{
v___y_300_ = v___x_294_;
goto v___jp_299_;
}
else
{
lean_object* v_val_322_; 
v_val_322_ = lean_ctor_get(v_parentDecl_x3f_72_, 0);
lean_inc(v_val_322_);
lean_dec_ref_known(v_parentDecl_x3f_72_, 1);
v___y_300_ = v_val_322_;
goto v___jp_299_;
}
v___jp_84_:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_86_ = lean_io_error_to_string(v_a_85_);
v___x_87_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
v___x_88_ = l_Lean_MessageData_ofFormat(v___x_87_);
lean_inc(v_ref_83_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v_ref_83_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
v___x_90_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
return v___x_90_;
}
v___jp_91_:
{
lean_object* v___x_93_; 
v___x_93_ = lean_mk_io_user_error(v_a_92_);
v_a_85_ = v___x_93_;
goto v___jp_84_;
}
v___jp_94_:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_113_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1(v_options_77_, v___y_95_);
v___x_114_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_114_, 0, v_fileName_99_);
lean_ctor_set(v___x_114_, 1, v_fileMap_100_);
lean_ctor_set(v___x_114_, 2, v_options_77_);
lean_ctor_set(v___x_114_, 3, v_currRecDepth_101_);
lean_ctor_set(v___x_114_, 4, v___x_113_);
lean_ctor_set(v___x_114_, 5, v_ref_102_);
lean_ctor_set(v___x_114_, 6, v_currNamespace_103_);
lean_ctor_set(v___x_114_, 7, v_openDecls_104_);
lean_ctor_set(v___x_114_, 8, v_initHeartbeats_105_);
lean_ctor_set(v___x_114_, 9, v_maxHeartbeats_106_);
lean_ctor_set(v___x_114_, 10, v_quotContext_107_);
lean_ctor_set(v___x_114_, 11, v_currMacroScope_108_);
lean_ctor_set(v___x_114_, 12, v_cancelTk_x3f_109_);
lean_ctor_set(v___x_114_, 13, v_inheritedTraceOptions_111_);
lean_ctor_set_uint8(v___x_114_, sizeof(void*)*14, v___y_98_);
lean_ctor_set_uint8(v___x_114_, sizeof(void*)*14 + 1, v_suppressElabErrors_110_);
v___x_115_ = lean_apply_3(v_x_67_, v___x_114_, v___y_112_, lean_box(0));
if (lean_obj_tag(v___x_115_) == 0)
{
lean_object* v_a_116_; lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_160_; 
v_a_116_ = lean_ctor_get(v___x_115_, 0);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_115_);
if (v_isSharedCheck_160_ == 0)
{
v___x_118_ = v___x_115_;
v_isShared_119_ = v_isSharedCheck_160_;
goto v_resetjp_117_;
}
else
{
lean_inc(v_a_116_);
lean_dec(v___x_115_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_160_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v_traceState_122_; lean_object* v_traceState_123_; lean_object* v_env_124_; lean_object* v_messages_125_; lean_object* v_scopes_126_; lean_object* v_usedQuotCtxts_127_; lean_object* v_nextMacroScope_128_; lean_object* v_maxRecDepth_129_; lean_object* v_ngen_130_; lean_object* v_auxDeclNGen_131_; lean_object* v_infoState_132_; lean_object* v_snapshotTasks_133_; lean_object* v_prevLinterStates_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_158_; 
v___x_120_ = lean_st_ref_get(v___y_97_);
lean_dec(v___y_97_);
v___x_121_ = lean_st_ref_take(v_a_69_);
v_traceState_122_ = lean_ctor_get(v___x_121_, 9);
lean_inc_ref(v_traceState_122_);
v_traceState_123_ = lean_ctor_get(v___x_120_, 4);
lean_inc_ref(v_traceState_123_);
v_env_124_ = lean_ctor_get(v___x_121_, 0);
v_messages_125_ = lean_ctor_get(v___x_121_, 1);
v_scopes_126_ = lean_ctor_get(v___x_121_, 2);
v_usedQuotCtxts_127_ = lean_ctor_get(v___x_121_, 3);
v_nextMacroScope_128_ = lean_ctor_get(v___x_121_, 4);
v_maxRecDepth_129_ = lean_ctor_get(v___x_121_, 5);
v_ngen_130_ = lean_ctor_get(v___x_121_, 6);
v_auxDeclNGen_131_ = lean_ctor_get(v___x_121_, 7);
v_infoState_132_ = lean_ctor_get(v___x_121_, 8);
v_snapshotTasks_133_ = lean_ctor_get(v___x_121_, 10);
v_prevLinterStates_134_ = lean_ctor_get(v___x_121_, 11);
v_isSharedCheck_158_ = !lean_is_exclusive(v___x_121_);
if (v_isSharedCheck_158_ == 0)
{
lean_object* v_unused_159_; 
v_unused_159_ = lean_ctor_get(v___x_121_, 9);
lean_dec(v_unused_159_);
v___x_136_ = v___x_121_;
v_isShared_137_ = v_isSharedCheck_158_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_prevLinterStates_134_);
lean_inc(v_snapshotTasks_133_);
lean_inc(v_infoState_132_);
lean_inc(v_auxDeclNGen_131_);
lean_inc(v_ngen_130_);
lean_inc(v_maxRecDepth_129_);
lean_inc(v_nextMacroScope_128_);
lean_inc(v_usedQuotCtxts_127_);
lean_inc(v_scopes_126_);
lean_inc(v_messages_125_);
lean_inc(v_env_124_);
lean_dec(v___x_121_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_158_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v_messages_138_; uint64_t v_tid_139_; lean_object* v_traces_140_; lean_object* v_traces_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_157_; 
v_messages_138_ = lean_ctor_get(v___x_120_, 6);
lean_inc_ref(v_messages_138_);
lean_dec(v___x_120_);
v_tid_139_ = lean_ctor_get_uint64(v_traceState_122_, sizeof(void*)*1);
v_traces_140_ = lean_ctor_get(v_traceState_122_, 0);
lean_inc_ref(v_traces_140_);
lean_dec_ref(v_traceState_122_);
v_traces_141_ = lean_ctor_get(v_traceState_123_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v_traceState_123_);
if (v_isSharedCheck_157_ == 0)
{
v___x_143_ = v_traceState_123_;
v_isShared_144_ = v_isSharedCheck_157_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_traces_141_);
lean_dec(v_traceState_123_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_157_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_148_; 
v___x_145_ = l_Lean_MessageLog_append(v_messages_125_, v_messages_138_);
v___x_146_ = l_Lean_PersistentArray_append___redArg(v_traces_140_, v_traces_141_);
lean_dec_ref(v_traces_141_);
if (v_isShared_144_ == 0)
{
lean_ctor_set(v___x_143_, 0, v___x_146_);
v___x_148_ = v___x_143_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v___x_146_);
v___x_148_ = v_reuseFailAlloc_156_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
lean_object* v___x_150_; 
lean_ctor_set_uint64(v___x_148_, sizeof(void*)*1, v_tid_139_);
if (v_isShared_137_ == 0)
{
lean_ctor_set(v___x_136_, 9, v___x_148_);
lean_ctor_set(v___x_136_, 1, v___x_145_);
v___x_150_ = v___x_136_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_155_; 
v_reuseFailAlloc_155_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_155_, 0, v_env_124_);
lean_ctor_set(v_reuseFailAlloc_155_, 1, v___x_145_);
lean_ctor_set(v_reuseFailAlloc_155_, 2, v_scopes_126_);
lean_ctor_set(v_reuseFailAlloc_155_, 3, v_usedQuotCtxts_127_);
lean_ctor_set(v_reuseFailAlloc_155_, 4, v_nextMacroScope_128_);
lean_ctor_set(v_reuseFailAlloc_155_, 5, v_maxRecDepth_129_);
lean_ctor_set(v_reuseFailAlloc_155_, 6, v_ngen_130_);
lean_ctor_set(v_reuseFailAlloc_155_, 7, v_auxDeclNGen_131_);
lean_ctor_set(v_reuseFailAlloc_155_, 8, v_infoState_132_);
lean_ctor_set(v_reuseFailAlloc_155_, 9, v___x_148_);
lean_ctor_set(v_reuseFailAlloc_155_, 10, v_snapshotTasks_133_);
lean_ctor_set(v_reuseFailAlloc_155_, 11, v_prevLinterStates_134_);
v___x_150_ = v_reuseFailAlloc_155_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
lean_object* v___x_151_; lean_object* v___x_153_; 
v___x_151_ = lean_st_ref_set(v_a_69_, v___x_150_);
if (v_isShared_119_ == 0)
{
v___x_153_ = v___x_118_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_a_116_);
v___x_153_ = v_reuseFailAlloc_154_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
return v___x_153_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_161_; 
lean_dec(v___y_97_);
v_a_161_ = lean_ctor_get(v___x_115_, 0);
lean_inc(v_a_161_);
lean_dec_ref_known(v___x_115_, 1);
if (lean_obj_tag(v_a_161_) == 0)
{
lean_object* v_msg_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v_msg_162_ = lean_ctor_get(v_a_161_, 1);
lean_inc_ref(v_msg_162_);
lean_dec_ref_known(v_a_161_, 2);
v___x_163_ = l_Lean_MessageData_toString(v_msg_162_);
v___x_164_ = lean_mk_io_user_error(v___x_163_);
v_a_85_ = v___x_164_;
goto v___jp_84_;
}
else
{
lean_object* v_id_165_; lean_object* v___x_166_; 
v_id_165_ = lean_ctor_get(v_a_161_, 0);
lean_inc(v_id_165_);
lean_dec_ref_known(v_a_161_, 2);
v___x_166_ = l_Lean_InternalExceptionId_getName(v_id_165_);
if (lean_obj_tag(v___x_166_) == 0)
{
lean_object* v_a_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
lean_dec(v_id_165_);
v_a_167_ = lean_ctor_get(v___x_166_, 0);
lean_inc(v_a_167_);
lean_dec_ref_known(v___x_166_, 1);
v___x_168_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__0));
v___x_169_ = l_Lean_Name_toString(v_a_167_, v___y_96_);
v___x_170_ = lean_string_append(v___x_168_, v___x_169_);
lean_dec_ref(v___x_169_);
v_a_92_ = v___x_170_;
goto v___jp_91_;
}
else
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
lean_dec_ref_known(v___x_166_, 1);
v___x_171_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__1));
v___x_172_ = l_Nat_reprFast(v_id_165_);
v___x_173_ = lean_string_append(v___x_171_, v___x_172_);
lean_dec_ref(v___x_172_);
v___x_174_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__2));
v___x_175_ = lean_string_append(v___x_173_, v___x_174_);
v_a_92_ = v___x_175_;
goto v___jp_91_;
}
}
}
}
v___jp_176_:
{
lean_object* v_fileName_183_; lean_object* v_fileMap_184_; lean_object* v_currRecDepth_185_; lean_object* v_ref_186_; lean_object* v_currNamespace_187_; lean_object* v_openDecls_188_; lean_object* v_initHeartbeats_189_; lean_object* v_maxHeartbeats_190_; lean_object* v_quotContext_191_; lean_object* v_currMacroScope_192_; lean_object* v_cancelTk_x3f_193_; uint8_t v_suppressElabErrors_194_; lean_object* v_inheritedTraceOptions_195_; 
v_fileName_183_ = lean_ctor_get(v___y_181_, 0);
lean_inc_ref(v_fileName_183_);
v_fileMap_184_ = lean_ctor_get(v___y_181_, 1);
lean_inc_ref(v_fileMap_184_);
v_currRecDepth_185_ = lean_ctor_get(v___y_181_, 3);
lean_inc(v_currRecDepth_185_);
v_ref_186_ = lean_ctor_get(v___y_181_, 5);
lean_inc(v_ref_186_);
v_currNamespace_187_ = lean_ctor_get(v___y_181_, 6);
lean_inc(v_currNamespace_187_);
v_openDecls_188_ = lean_ctor_get(v___y_181_, 7);
lean_inc(v_openDecls_188_);
v_initHeartbeats_189_ = lean_ctor_get(v___y_181_, 8);
lean_inc(v_initHeartbeats_189_);
v_maxHeartbeats_190_ = lean_ctor_get(v___y_181_, 9);
lean_inc(v_maxHeartbeats_190_);
v_quotContext_191_ = lean_ctor_get(v___y_181_, 10);
lean_inc(v_quotContext_191_);
v_currMacroScope_192_ = lean_ctor_get(v___y_181_, 11);
lean_inc(v_currMacroScope_192_);
v_cancelTk_x3f_193_ = lean_ctor_get(v___y_181_, 12);
lean_inc(v_cancelTk_x3f_193_);
v_suppressElabErrors_194_ = lean_ctor_get_uint8(v___y_181_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_195_ = lean_ctor_get(v___y_181_, 13);
lean_inc_ref(v_inheritedTraceOptions_195_);
lean_dec_ref(v___y_181_);
v___y_95_ = v___y_177_;
v___y_96_ = v___y_178_;
v___y_97_ = v___y_179_;
v___y_98_ = v___y_180_;
v_fileName_99_ = v_fileName_183_;
v_fileMap_100_ = v_fileMap_184_;
v_currRecDepth_101_ = v_currRecDepth_185_;
v_ref_102_ = v_ref_186_;
v_currNamespace_103_ = v_currNamespace_187_;
v_openDecls_104_ = v_openDecls_188_;
v_initHeartbeats_105_ = v_initHeartbeats_189_;
v_maxHeartbeats_106_ = v_maxHeartbeats_190_;
v_quotContext_107_ = v_quotContext_191_;
v_currMacroScope_108_ = v_currMacroScope_192_;
v_cancelTk_x3f_109_ = v_cancelTk_x3f_193_;
v_suppressElabErrors_110_ = v_suppressElabErrors_194_;
v_inheritedTraceOptions_111_ = v_inheritedTraceOptions_195_;
v___y_112_ = v___y_182_;
goto v___jp_94_;
}
v___jp_196_:
{
if (v___y_204_ == 0)
{
lean_object* v___x_205_; lean_object* v_env_206_; lean_object* v_nextMacroScope_207_; lean_object* v_ngen_208_; lean_object* v_auxDeclNGen_209_; lean_object* v_traceState_210_; lean_object* v_messages_211_; lean_object* v_infoState_212_; lean_object* v_snapshotTasks_213_; lean_object* v___x_215_; uint8_t v_isShared_216_; uint8_t v_isSharedCheck_222_; 
v___x_205_ = lean_st_ref_take(v___y_200_);
v_env_206_ = lean_ctor_get(v___x_205_, 0);
v_nextMacroScope_207_ = lean_ctor_get(v___x_205_, 1);
v_ngen_208_ = lean_ctor_get(v___x_205_, 2);
v_auxDeclNGen_209_ = lean_ctor_get(v___x_205_, 3);
v_traceState_210_ = lean_ctor_get(v___x_205_, 4);
v_messages_211_ = lean_ctor_get(v___x_205_, 6);
v_infoState_212_ = lean_ctor_get(v___x_205_, 7);
v_snapshotTasks_213_ = lean_ctor_get(v___x_205_, 8);
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_205_);
if (v_isSharedCheck_222_ == 0)
{
lean_object* v_unused_223_; 
v_unused_223_ = lean_ctor_get(v___x_205_, 5);
lean_dec(v_unused_223_);
v___x_215_ = v___x_205_;
v_isShared_216_ = v_isSharedCheck_222_;
goto v_resetjp_214_;
}
else
{
lean_inc(v_snapshotTasks_213_);
lean_inc(v_infoState_212_);
lean_inc(v_messages_211_);
lean_inc(v_traceState_210_);
lean_inc(v_auxDeclNGen_209_);
lean_inc(v_ngen_208_);
lean_inc(v_nextMacroScope_207_);
lean_inc(v_env_206_);
lean_dec(v___x_205_);
v___x_215_ = lean_box(0);
v_isShared_216_ = v_isSharedCheck_222_;
goto v_resetjp_214_;
}
v_resetjp_214_:
{
lean_object* v___x_217_; lean_object* v___x_219_; 
v___x_217_ = l_Lean_Kernel_enableDiag(v_env_206_, v___y_203_);
lean_inc_ref(v___y_201_);
if (v_isShared_216_ == 0)
{
lean_ctor_set(v___x_215_, 5, v___y_201_);
lean_ctor_set(v___x_215_, 0, v___x_217_);
v___x_219_ = v___x_215_;
goto v_reusejp_218_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v___x_217_);
lean_ctor_set(v_reuseFailAlloc_221_, 1, v_nextMacroScope_207_);
lean_ctor_set(v_reuseFailAlloc_221_, 2, v_ngen_208_);
lean_ctor_set(v_reuseFailAlloc_221_, 3, v_auxDeclNGen_209_);
lean_ctor_set(v_reuseFailAlloc_221_, 4, v_traceState_210_);
lean_ctor_set(v_reuseFailAlloc_221_, 5, v___y_201_);
lean_ctor_set(v_reuseFailAlloc_221_, 6, v_messages_211_);
lean_ctor_set(v_reuseFailAlloc_221_, 7, v_infoState_212_);
lean_ctor_set(v_reuseFailAlloc_221_, 8, v_snapshotTasks_213_);
v___x_219_ = v_reuseFailAlloc_221_;
goto v_reusejp_218_;
}
v_reusejp_218_:
{
lean_object* v___x_220_; 
v___x_220_ = lean_st_ref_set(v___y_200_, v___x_219_);
v___y_177_ = v___y_197_;
v___y_178_ = v___y_199_;
v___y_179_ = v___y_202_;
v___y_180_ = v___y_203_;
v___y_181_ = v___y_198_;
v___y_182_ = v___y_200_;
goto v___jp_176_;
}
}
}
else
{
v___y_177_ = v___y_197_;
v___y_178_ = v___y_199_;
v___y_179_ = v___y_202_;
v___y_180_ = v___y_203_;
v___y_181_ = v___y_198_;
v___y_182_ = v___y_200_;
goto v___jp_176_;
}
}
v___jp_227_:
{
lean_object* v___x_235_; lean_object* v_fileName_236_; lean_object* v_fileMap_237_; lean_object* v_currRecDepth_238_; lean_object* v_ref_239_; lean_object* v_currNamespace_240_; lean_object* v_openDecls_241_; lean_object* v_initHeartbeats_242_; lean_object* v_maxHeartbeats_243_; lean_object* v_quotContext_244_; lean_object* v_currMacroScope_245_; lean_object* v_cancelTk_x3f_246_; uint8_t v_suppressElabErrors_247_; lean_object* v_inheritedTraceOptions_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_260_; 
v___x_235_ = lean_st_ref_get(v___y_234_);
v_fileName_236_ = lean_ctor_get(v___y_233_, 0);
v_fileMap_237_ = lean_ctor_get(v___y_233_, 1);
v_currRecDepth_238_ = lean_ctor_get(v___y_233_, 3);
v_ref_239_ = lean_ctor_get(v___y_233_, 5);
v_currNamespace_240_ = lean_ctor_get(v___y_233_, 6);
v_openDecls_241_ = lean_ctor_get(v___y_233_, 7);
v_initHeartbeats_242_ = lean_ctor_get(v___y_233_, 8);
v_maxHeartbeats_243_ = lean_ctor_get(v___y_233_, 9);
v_quotContext_244_ = lean_ctor_get(v___y_233_, 10);
v_currMacroScope_245_ = lean_ctor_get(v___y_233_, 11);
v_cancelTk_x3f_246_ = lean_ctor_get(v___y_233_, 12);
v_suppressElabErrors_247_ = lean_ctor_get_uint8(v___y_233_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_248_ = lean_ctor_get(v___y_233_, 13);
v_isSharedCheck_260_ = !lean_is_exclusive(v___y_233_);
if (v_isSharedCheck_260_ == 0)
{
lean_object* v_unused_261_; lean_object* v_unused_262_; 
v_unused_261_ = lean_ctor_get(v___y_233_, 4);
lean_dec(v_unused_261_);
v_unused_262_ = lean_ctor_get(v___y_233_, 2);
lean_dec(v_unused_262_);
v___x_250_ = v___y_233_;
v_isShared_251_ = v_isSharedCheck_260_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_inheritedTraceOptions_248_);
lean_inc(v_cancelTk_x3f_246_);
lean_inc(v_currMacroScope_245_);
lean_inc(v_quotContext_244_);
lean_inc(v_maxHeartbeats_243_);
lean_inc(v_initHeartbeats_242_);
lean_inc(v_openDecls_241_);
lean_inc(v_currNamespace_240_);
lean_inc(v_ref_239_);
lean_inc(v_currRecDepth_238_);
lean_inc(v_fileMap_237_);
lean_inc(v_fileName_236_);
lean_dec(v___y_233_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_260_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v_env_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_256_; 
v_env_252_ = lean_ctor_get(v___x_235_, 0);
lean_inc_ref(v_env_252_);
lean_dec(v___x_235_);
v___x_253_ = l_Lean_maxRecDepth;
v___x_254_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3);
lean_inc_ref(v_inheritedTraceOptions_248_);
lean_inc(v_cancelTk_x3f_246_);
lean_inc(v_currMacroScope_245_);
lean_inc(v_quotContext_244_);
lean_inc(v_maxHeartbeats_243_);
lean_inc(v_initHeartbeats_242_);
lean_inc(v_openDecls_241_);
lean_inc(v_currNamespace_240_);
lean_inc(v_ref_239_);
lean_inc(v_currRecDepth_238_);
lean_inc_ref(v_fileMap_237_);
lean_inc_ref(v_fileName_236_);
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 4, v___x_254_);
lean_ctor_set(v___x_250_, 2, v___x_226_);
v___x_256_ = v___x_250_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v_fileName_236_);
lean_ctor_set(v_reuseFailAlloc_259_, 1, v_fileMap_237_);
lean_ctor_set(v_reuseFailAlloc_259_, 2, v___x_226_);
lean_ctor_set(v_reuseFailAlloc_259_, 3, v_currRecDepth_238_);
lean_ctor_set(v_reuseFailAlloc_259_, 4, v___x_254_);
lean_ctor_set(v_reuseFailAlloc_259_, 5, v_ref_239_);
lean_ctor_set(v_reuseFailAlloc_259_, 6, v_currNamespace_240_);
lean_ctor_set(v_reuseFailAlloc_259_, 7, v_openDecls_241_);
lean_ctor_set(v_reuseFailAlloc_259_, 8, v_initHeartbeats_242_);
lean_ctor_set(v_reuseFailAlloc_259_, 9, v_maxHeartbeats_243_);
lean_ctor_set(v_reuseFailAlloc_259_, 10, v_quotContext_244_);
lean_ctor_set(v_reuseFailAlloc_259_, 11, v_currMacroScope_245_);
lean_ctor_set(v_reuseFailAlloc_259_, 12, v_cancelTk_x3f_246_);
lean_ctor_set(v_reuseFailAlloc_259_, 13, v_inheritedTraceOptions_248_);
lean_ctor_set_uint8(v_reuseFailAlloc_259_, sizeof(void*)*14 + 1, v_suppressElabErrors_247_);
v___x_256_ = v_reuseFailAlloc_259_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
uint8_t v___x_257_; uint8_t v___x_258_; 
lean_ctor_set_uint8(v___x_256_, sizeof(void*)*14, v___y_232_);
v___x_257_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0(v_options_77_, v___y_229_);
v___x_258_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_252_);
lean_dec_ref(v_env_252_);
if (v___x_258_ == 0)
{
if (v___x_257_ == 0)
{
lean_dec_ref(v___x_256_);
v___y_95_ = v___x_253_;
v___y_96_ = v___y_228_;
v___y_97_ = v___y_231_;
v___y_98_ = v___x_257_;
v_fileName_99_ = v_fileName_236_;
v_fileMap_100_ = v_fileMap_237_;
v_currRecDepth_101_ = v_currRecDepth_238_;
v_ref_102_ = v_ref_239_;
v_currNamespace_103_ = v_currNamespace_240_;
v_openDecls_104_ = v_openDecls_241_;
v_initHeartbeats_105_ = v_initHeartbeats_242_;
v_maxHeartbeats_106_ = v_maxHeartbeats_243_;
v_quotContext_107_ = v_quotContext_244_;
v_currMacroScope_108_ = v_currMacroScope_245_;
v_cancelTk_x3f_109_ = v_cancelTk_x3f_246_;
v_suppressElabErrors_110_ = v_suppressElabErrors_247_;
v_inheritedTraceOptions_111_ = v_inheritedTraceOptions_248_;
v___y_112_ = v___y_234_;
goto v___jp_94_;
}
else
{
lean_dec_ref(v_inheritedTraceOptions_248_);
lean_dec(v_cancelTk_x3f_246_);
lean_dec(v_currMacroScope_245_);
lean_dec(v_quotContext_244_);
lean_dec(v_maxHeartbeats_243_);
lean_dec(v_initHeartbeats_242_);
lean_dec(v_openDecls_241_);
lean_dec(v_currNamespace_240_);
lean_dec(v_ref_239_);
lean_dec(v_currRecDepth_238_);
lean_dec_ref(v_fileMap_237_);
lean_dec_ref(v_fileName_236_);
v___y_197_ = v___x_253_;
v___y_198_ = v___x_256_;
v___y_199_ = v___y_228_;
v___y_200_ = v___y_234_;
v___y_201_ = v___y_230_;
v___y_202_ = v___y_231_;
v___y_203_ = v___x_257_;
v___y_204_ = v___x_258_;
goto v___jp_196_;
}
}
else
{
lean_dec_ref(v_inheritedTraceOptions_248_);
lean_dec(v_cancelTk_x3f_246_);
lean_dec(v_currMacroScope_245_);
lean_dec(v_quotContext_244_);
lean_dec(v_maxHeartbeats_243_);
lean_dec(v_initHeartbeats_242_);
lean_dec(v_openDecls_241_);
lean_dec(v_currNamespace_240_);
lean_dec(v_ref_239_);
lean_dec(v_currRecDepth_238_);
lean_dec_ref(v_fileMap_237_);
lean_dec_ref(v_fileName_236_);
v___y_197_ = v___x_253_;
v___y_198_ = v___x_256_;
v___y_199_ = v___y_228_;
v___y_200_ = v___y_234_;
v___y_201_ = v___y_230_;
v___y_202_ = v___y_231_;
v___y_203_ = v___x_257_;
v___y_204_ = v___x_257_;
goto v___jp_196_;
}
}
}
}
v___jp_263_:
{
if (v___y_270_ == 0)
{
lean_object* v___x_271_; lean_object* v_env_272_; lean_object* v_nextMacroScope_273_; lean_object* v_ngen_274_; lean_object* v_auxDeclNGen_275_; lean_object* v_traceState_276_; lean_object* v_messages_277_; lean_object* v_infoState_278_; lean_object* v_snapshotTasks_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_288_; 
v___x_271_ = lean_st_ref_take(v___y_267_);
v_env_272_ = lean_ctor_get(v___x_271_, 0);
v_nextMacroScope_273_ = lean_ctor_get(v___x_271_, 1);
v_ngen_274_ = lean_ctor_get(v___x_271_, 2);
v_auxDeclNGen_275_ = lean_ctor_get(v___x_271_, 3);
v_traceState_276_ = lean_ctor_get(v___x_271_, 4);
v_messages_277_ = lean_ctor_get(v___x_271_, 6);
v_infoState_278_ = lean_ctor_get(v___x_271_, 7);
v_snapshotTasks_279_ = lean_ctor_get(v___x_271_, 8);
v_isSharedCheck_288_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_288_ == 0)
{
lean_object* v_unused_289_; 
v_unused_289_ = lean_ctor_get(v___x_271_, 5);
lean_dec(v_unused_289_);
v___x_281_ = v___x_271_;
v_isShared_282_ = v_isSharedCheck_288_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_snapshotTasks_279_);
lean_inc(v_infoState_278_);
lean_inc(v_messages_277_);
lean_inc(v_traceState_276_);
lean_inc(v_auxDeclNGen_275_);
lean_inc(v_ngen_274_);
lean_inc(v_nextMacroScope_273_);
lean_inc(v_env_272_);
lean_dec(v___x_271_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_288_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_283_; lean_object* v___x_285_; 
v___x_283_ = l_Lean_Kernel_enableDiag(v_env_272_, v___y_269_);
lean_inc_ref(v___y_266_);
if (v_isShared_282_ == 0)
{
lean_ctor_set(v___x_281_, 5, v___y_266_);
lean_ctor_set(v___x_281_, 0, v___x_283_);
v___x_285_ = v___x_281_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_287_; 
v_reuseFailAlloc_287_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_287_, 0, v___x_283_);
lean_ctor_set(v_reuseFailAlloc_287_, 1, v_nextMacroScope_273_);
lean_ctor_set(v_reuseFailAlloc_287_, 2, v_ngen_274_);
lean_ctor_set(v_reuseFailAlloc_287_, 3, v_auxDeclNGen_275_);
lean_ctor_set(v_reuseFailAlloc_287_, 4, v_traceState_276_);
lean_ctor_set(v_reuseFailAlloc_287_, 5, v___y_266_);
lean_ctor_set(v_reuseFailAlloc_287_, 6, v_messages_277_);
lean_ctor_set(v_reuseFailAlloc_287_, 7, v_infoState_278_);
lean_ctor_set(v_reuseFailAlloc_287_, 8, v_snapshotTasks_279_);
v___x_285_ = v_reuseFailAlloc_287_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
lean_object* v___x_286_; 
v___x_286_ = lean_st_ref_set(v___y_267_, v___x_285_);
lean_inc(v___y_267_);
v___y_228_ = v___y_264_;
v___y_229_ = v___y_265_;
v___y_230_ = v___y_266_;
v___y_231_ = v___y_267_;
v___y_232_ = v___y_269_;
v___y_233_ = v___y_268_;
v___y_234_ = v___y_267_;
goto v___jp_227_;
}
}
}
else
{
lean_inc(v___y_267_);
v___y_228_ = v___y_264_;
v___y_229_ = v___y_265_;
v___y_230_ = v___y_266_;
v___y_231_ = v___y_267_;
v___y_232_ = v___y_269_;
v___y_233_ = v___y_268_;
v___y_234_ = v___y_267_;
goto v___jp_227_;
}
}
v___jp_299_:
{
lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_306_; 
v___x_301_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10);
v___x_302_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11);
v___x_303_ = lean_io_get_num_heartbeats();
v___x_304_ = lean_box(0);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 2, v___x_304_);
lean_ctor_set(v___x_74_, 1, v___x_297_);
lean_ctor_set(v___x_74_, 0, v___y_300_);
v___x_306_ = v___x_74_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v___y_300_);
lean_ctor_set(v_reuseFailAlloc_321_, 1, v___x_297_);
lean_ctor_set(v_reuseFailAlloc_321_, 2, v___x_304_);
v___x_306_ = v_reuseFailAlloc_321_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
lean_object* v___x_307_; uint8_t v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v_env_317_; lean_object* v___x_318_; uint8_t v___x_319_; uint8_t v___x_320_; 
v___x_307_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12);
v___x_308_ = 1;
v___x_309_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13);
v___x_310_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__14));
v___x_311_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v___x_311_, 0, v_env_225_);
lean_ctor_set(v___x_311_, 1, v___x_298_);
lean_ctor_set(v___x_311_, 2, v_ngen_80_);
lean_ctor_set(v___x_311_, 3, v___x_306_);
lean_ctor_set(v___x_311_, 4, v___x_307_);
lean_ctor_set(v___x_311_, 5, v___x_301_);
lean_ctor_set(v___x_311_, 6, v___x_302_);
lean_ctor_set(v___x_311_, 7, v___x_309_);
lean_ctor_set(v___x_311_, 8, v___x_310_);
v___x_312_ = lean_st_mk_ref(v___x_311_);
v___x_313_ = l_Lean_inheritedTraceOptions;
v___x_314_ = lean_st_ref_get(v___x_313_);
v___x_315_ = lean_st_ref_get(v___x_312_);
lean_inc_ref(v_fileMap_82_);
lean_inc_ref(v_fileName_81_);
v___x_316_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_316_, 0, v_fileName_81_);
lean_ctor_set(v___x_316_, 1, v_fileMap_82_);
lean_ctor_set(v___x_316_, 2, v___x_226_);
lean_ctor_set(v___x_316_, 3, v___x_290_);
lean_ctor_set(v___x_316_, 4, v___x_291_);
lean_ctor_set(v___x_316_, 5, v___x_292_);
lean_ctor_set(v___x_316_, 6, v_currNamespace_78_);
lean_ctor_set(v___x_316_, 7, v_openDecls_79_);
lean_ctor_set(v___x_316_, 8, v___x_303_);
lean_ctor_set(v___x_316_, 9, v___x_293_);
lean_ctor_set(v___x_316_, 10, v___x_294_);
lean_ctor_set(v___x_316_, 11, v___x_295_);
lean_ctor_set(v___x_316_, 12, v___x_296_);
lean_ctor_set(v___x_316_, 13, v___x_314_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*14, v___x_224_);
lean_ctor_set_uint8(v___x_316_, sizeof(void*)*14 + 1, v___x_224_);
v_env_317_ = lean_ctor_get(v___x_315_, 0);
lean_inc_ref(v_env_317_);
lean_dec(v___x_315_);
v___x_318_ = l_Lean_diagnostics;
v___x_319_ = lean_uint8_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15);
v___x_320_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_317_);
lean_dec_ref(v_env_317_);
if (v___x_320_ == 0)
{
if (v___x_319_ == 0)
{
lean_inc(v___x_312_);
v___y_228_ = v___x_308_;
v___y_229_ = v___x_318_;
v___y_230_ = v___x_301_;
v___y_231_ = v___x_312_;
v___y_232_ = v___x_319_;
v___y_233_ = v___x_316_;
v___y_234_ = v___x_312_;
goto v___jp_227_;
}
else
{
v___y_264_ = v___x_308_;
v___y_265_ = v___x_318_;
v___y_266_ = v___x_301_;
v___y_267_ = v___x_312_;
v___y_268_ = v___x_316_;
v___y_269_ = v___x_319_;
v___y_270_ = v___x_320_;
goto v___jp_263_;
}
}
else
{
v___y_264_ = v___x_308_;
v___y_265_ = v___x_318_;
v___y_266_ = v___x_301_;
v___y_267_ = v___x_312_;
v___y_268_ = v___x_316_;
v___y_269_ = v___x_319_;
v___y_270_ = v___x_319_;
goto v___jp_263_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___boxed(lean_object* v_info_325_, lean_object* v_x_326_, lean_object* v_a_327_, lean_object* v_a_328_, lean_object* v_a_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg(v_info_325_, v_x_326_, v_a_327_, v_a_328_);
lean_dec(v_a_328_);
lean_dec_ref(v_a_327_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages(lean_object* v_00_u03b1_331_, lean_object* v_info_332_, lean_object* v_x_333_, lean_object* v_a_334_, lean_object* v_a_335_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg(v_info_332_, v_x_333_, v_a_334_, v_a_335_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___boxed(lean_object* v_00_u03b1_338_, lean_object* v_info_339_, lean_object* v_x_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages(v_00_u03b1_338_, v_info_339_, v_x_340_, v_a_341_, v_a_342_);
lean_dec(v_a_342_);
lean_dec_ref(v_a_341_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___redArg(lean_object* v_decls_345_, lean_object* v_x_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = l_Lean_Meta_withLocalInstancesImp___redArg(v_decls_345_, v_x_346_, v___y_347_, v___y_348_, v___y_349_, v___y_350_);
if (lean_obj_tag(v___x_352_) == 0)
{
lean_object* v_a_353_; lean_object* v___x_355_; uint8_t v_isShared_356_; uint8_t v_isSharedCheck_360_; 
v_a_353_ = lean_ctor_get(v___x_352_, 0);
v_isSharedCheck_360_ = !lean_is_exclusive(v___x_352_);
if (v_isSharedCheck_360_ == 0)
{
v___x_355_ = v___x_352_;
v_isShared_356_ = v_isSharedCheck_360_;
goto v_resetjp_354_;
}
else
{
lean_inc(v_a_353_);
lean_dec(v___x_352_);
v___x_355_ = lean_box(0);
v_isShared_356_ = v_isSharedCheck_360_;
goto v_resetjp_354_;
}
v_resetjp_354_:
{
lean_object* v___x_358_; 
if (v_isShared_356_ == 0)
{
v___x_358_ = v___x_355_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v_a_353_);
v___x_358_ = v_reuseFailAlloc_359_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
return v___x_358_;
}
}
}
else
{
lean_object* v_a_361_; lean_object* v___x_363_; uint8_t v_isShared_364_; uint8_t v_isSharedCheck_368_; 
v_a_361_ = lean_ctor_get(v___x_352_, 0);
v_isSharedCheck_368_ = !lean_is_exclusive(v___x_352_);
if (v_isSharedCheck_368_ == 0)
{
v___x_363_ = v___x_352_;
v_isShared_364_ = v_isSharedCheck_368_;
goto v_resetjp_362_;
}
else
{
lean_inc(v_a_361_);
lean_dec(v___x_352_);
v___x_363_ = lean_box(0);
v_isShared_364_ = v_isSharedCheck_368_;
goto v_resetjp_362_;
}
v_resetjp_362_:
{
lean_object* v___x_366_; 
if (v_isShared_364_ == 0)
{
v___x_366_ = v___x_363_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v_a_361_);
v___x_366_ = v_reuseFailAlloc_367_;
goto v_reusejp_365_;
}
v_reusejp_365_:
{
return v___x_366_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___redArg___boxed(lean_object* v_decls_369_, lean_object* v_x_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___redArg(v_decls_369_, v_x_370_, v___y_371_, v___y_372_, v___y_373_, v___y_374_);
lean_dec(v___y_374_);
lean_dec_ref(v___y_373_);
lean_dec(v___y_372_);
lean_dec_ref(v___y_371_);
lean_dec(v_decls_369_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1(lean_object* v_00_u03b1_377_, lean_object* v_decls_378_, lean_object* v_x_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___redArg(v_decls_378_, v_x_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___boxed(lean_object* v_00_u03b1_386_, lean_object* v_decls_387_, lean_object* v_x_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1(v_00_u03b1_386_, v_decls_387_, v_x_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
lean_dec(v_decls_387_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___lam__0(lean_object* v___x_395_, lean_object* v___x_396_, lean_object* v_x_397_, lean_object* v___x_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_402_ = lean_st_mk_ref(v___x_395_);
v___x_403_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__1___redArg(v___x_396_, v_x_397_, v___x_398_, v___x_402_, v___y_399_, v___y_400_);
if (lean_obj_tag(v___x_403_) == 0)
{
lean_object* v_a_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_413_; 
v_a_404_ = lean_ctor_get(v___x_403_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_403_);
if (v_isSharedCheck_413_ == 0)
{
v___x_406_ = v___x_403_;
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_a_404_);
lean_dec(v___x_403_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_411_; 
v___x_408_ = lean_st_ref_get(v___x_402_);
lean_dec(v___x_402_);
v___x_409_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_409_, 0, v_a_404_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
if (v_isShared_407_ == 0)
{
lean_ctor_set(v___x_406_, 0, v___x_409_);
v___x_411_ = v___x_406_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_409_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
else
{
lean_object* v_a_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_421_; 
lean_dec(v___x_402_);
v_a_414_ = lean_ctor_get(v___x_403_, 0);
v_isSharedCheck_421_ = !lean_is_exclusive(v___x_403_);
if (v_isSharedCheck_421_ == 0)
{
v___x_416_ = v___x_403_;
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_a_414_);
lean_dec(v___x_403_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_419_; 
if (v_isShared_417_ == 0)
{
v___x_419_ = v___x_416_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_420_; 
v_reuseFailAlloc_420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_420_, 0, v_a_414_);
v___x_419_ = v_reuseFailAlloc_420_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
return v___x_419_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___lam__0___boxed(lean_object* v___x_422_, lean_object* v___x_423_, lean_object* v_x_424_, lean_object* v___x_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
lean_object* v_res_429_; 
v_res_429_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___lam__0(v___x_422_, v___x_423_, v_x_424_, v___x_425_, v___y_426_, v___y_427_);
lean_dec(v___y_427_);
lean_dec_ref(v___y_426_);
lean_dec_ref(v___x_425_);
lean_dec(v___x_423_);
return v_res_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__0(lean_object* v_a_430_, lean_object* v_a_431_){
_start:
{
if (lean_obj_tag(v_a_430_) == 0)
{
lean_object* v___x_432_; 
v___x_432_ = lean_array_to_list(v_a_431_);
return v___x_432_;
}
else
{
lean_object* v_head_433_; 
v_head_433_ = lean_ctor_get(v_a_430_, 0);
if (lean_obj_tag(v_head_433_) == 0)
{
lean_object* v_tail_434_; 
v_tail_434_ = lean_ctor_get(v_a_430_, 1);
lean_inc(v_tail_434_);
lean_dec_ref_known(v_a_430_, 2);
v_a_430_ = v_tail_434_;
goto _start;
}
else
{
lean_object* v_tail_436_; lean_object* v_val_437_; lean_object* v___x_438_; 
lean_inc_ref(v_head_433_);
v_tail_436_ = lean_ctor_get(v_a_430_, 1);
lean_inc(v_tail_436_);
lean_dec_ref_known(v_a_430_, 2);
v_val_437_ = lean_ctor_get(v_head_433_, 0);
lean_inc(v_val_437_);
lean_dec_ref_known(v_head_433_, 1);
v___x_438_ = lean_array_push(v_a_431_, v_val_437_);
v_a_430_ = v_tail_436_;
v_a_431_ = v___x_438_;
goto _start;
}
}
}
}
static uint64_t _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__2(void){
_start:
{
lean_object* v___x_448_; uint64_t v___x_449_; 
v___x_448_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__1));
v___x_449_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_448_);
return v___x_449_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3(void){
_start:
{
uint64_t v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_450_ = lean_uint64_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__2, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__2);
v___x_451_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__1));
v___x_452_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_452_, 0, v___x_451_);
lean_ctor_set_uint64(v___x_452_, sizeof(void*)*1, v___x_450_);
return v___x_452_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__4(void){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_453_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5(void){
_start:
{
lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_454_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__4, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__4_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__4);
v___x_455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_455_, 0, v___x_454_);
return v___x_455_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6(void){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_456_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5);
v___x_457_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_457_, 0, v___x_456_);
lean_ctor_set(v___x_457_, 1, v___x_456_);
lean_ctor_set(v___x_457_, 2, v___x_456_);
lean_ctor_set(v___x_457_, 3, v___x_456_);
lean_ctor_set(v___x_457_, 4, v___x_456_);
lean_ctor_set(v___x_457_, 5, v___x_456_);
return v___x_457_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__7(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; 
v___x_458_ = lean_unsigned_to_nat(32u);
v___x_459_ = lean_mk_empty_array_with_capacity(v___x_458_);
v___x_460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
return v___x_460_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8(void){
_start:
{
size_t v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_461_ = ((size_t)5ULL);
v___x_462_ = lean_unsigned_to_nat(0u);
v___x_463_ = lean_unsigned_to_nat(32u);
v___x_464_ = lean_mk_empty_array_with_capacity(v___x_463_);
v___x_465_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__7, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__7_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__7);
v___x_466_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_466_, 0, v___x_465_);
lean_ctor_set(v___x_466_, 1, v___x_464_);
lean_ctor_set(v___x_466_, 2, v___x_462_);
lean_ctor_set(v___x_466_, 3, v___x_462_);
lean_ctor_set_usize(v___x_466_, 4, v___x_461_);
return v___x_466_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9(void){
_start:
{
lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_467_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__5);
v___x_468_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_468_, 0, v___x_467_);
lean_ctor_set(v___x_468_, 1, v___x_467_);
lean_ctor_set(v___x_468_, 2, v___x_467_);
lean_ctor_set(v___x_468_, 3, v___x_467_);
lean_ctor_set(v___x_468_, 4, v___x_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg(lean_object* v_info_469_, lean_object* v_lctx_470_, lean_object* v_x_471_, lean_object* v_a_472_, lean_object* v_a_473_){
_start:
{
lean_object* v_decls_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; uint8_t v___x_479_; uint8_t v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v_toCommandContextInfo_484_; lean_object* v_mctx_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___f_492_; lean_object* v___x_493_; 
v_decls_475_ = lean_ctor_get(v_lctx_470_, 1);
lean_inc_ref(v_decls_475_);
v___x_476_ = lean_box(1);
v___x_477_ = lean_unsigned_to_nat(0u);
v___x_478_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__0));
v___x_479_ = 0;
v___x_480_ = 1;
v___x_481_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3);
v___x_482_ = lean_box(0);
v___x_483_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_483_, 0, v___x_481_);
lean_ctor_set(v___x_483_, 1, v___x_476_);
lean_ctor_set(v___x_483_, 2, v_lctx_470_);
lean_ctor_set(v___x_483_, 3, v___x_478_);
lean_ctor_set(v___x_483_, 4, v___x_482_);
lean_ctor_set(v___x_483_, 5, v___x_477_);
lean_ctor_set(v___x_483_, 6, v___x_482_);
lean_ctor_set_uint8(v___x_483_, sizeof(void*)*7, v___x_479_);
lean_ctor_set_uint8(v___x_483_, sizeof(void*)*7 + 1, v___x_479_);
lean_ctor_set_uint8(v___x_483_, sizeof(void*)*7 + 2, v___x_479_);
lean_ctor_set_uint8(v___x_483_, sizeof(void*)*7 + 3, v___x_480_);
v_toCommandContextInfo_484_ = lean_ctor_get(v_info_469_, 0);
v_mctx_485_ = lean_ctor_get(v_toCommandContextInfo_484_, 3);
v___x_486_ = l_Lean_PersistentArray_toList___redArg(v_decls_475_);
lean_dec_ref(v_decls_475_);
v___x_487_ = lp_mathlib_List_filterMapTR_go___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__0(v___x_486_, v___x_478_);
v___x_488_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6);
v___x_489_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8);
v___x_490_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9);
lean_inc_ref(v_mctx_485_);
v___x_491_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_491_, 0, v_mctx_485_);
lean_ctor_set(v___x_491_, 1, v___x_488_);
lean_ctor_set(v___x_491_, 2, v___x_476_);
lean_ctor_set(v___x_491_, 3, v___x_489_);
lean_ctor_set(v___x_491_, 4, v___x_490_);
v___f_492_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___lam__0___boxed), 7, 4);
lean_closure_set(v___f_492_, 0, v___x_491_);
lean_closure_set(v___f_492_, 1, v___x_487_);
lean_closure_set(v___f_492_, 2, v_x_471_);
lean_closure_set(v___f_492_, 3, v___x_483_);
v___x_493_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg(v_info_469_, v___f_492_, v_a_472_, v_a_473_);
if (lean_obj_tag(v___x_493_) == 0)
{
lean_object* v_a_494_; lean_object* v___x_496_; uint8_t v_isShared_497_; uint8_t v_isSharedCheck_502_; 
v_a_494_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_502_ == 0)
{
v___x_496_ = v___x_493_;
v_isShared_497_ = v_isSharedCheck_502_;
goto v_resetjp_495_;
}
else
{
lean_inc(v_a_494_);
lean_dec(v___x_493_);
v___x_496_ = lean_box(0);
v_isShared_497_ = v_isSharedCheck_502_;
goto v_resetjp_495_;
}
v_resetjp_495_:
{
lean_object* v_fst_498_; lean_object* v___x_500_; 
v_fst_498_ = lean_ctor_get(v_a_494_, 0);
lean_inc(v_fst_498_);
lean_dec(v_a_494_);
if (v_isShared_497_ == 0)
{
lean_ctor_set(v___x_496_, 0, v_fst_498_);
v___x_500_ = v___x_496_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_fst_498_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
else
{
lean_object* v_a_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_510_; 
v_a_503_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_510_ == 0)
{
v___x_505_ = v___x_493_;
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_a_503_);
lean_dec(v___x_493_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_508_; 
if (v_isShared_506_ == 0)
{
v___x_508_ = v___x_505_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v_a_503_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___boxed(lean_object* v_info_511_, lean_object* v_lctx_512_, lean_object* v_x_513_, lean_object* v_a_514_, lean_object* v_a_515_, lean_object* v_a_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg(v_info_511_, v_lctx_512_, v_x_513_, v_a_514_, v_a_515_);
lean_dec(v_a_515_);
lean_dec_ref(v_a_514_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages(lean_object* v_00_u03b1_518_, lean_object* v_info_519_, lean_object* v_lctx_520_, lean_object* v_x_521_, lean_object* v_a_522_, lean_object* v_a_523_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg(v_info_519_, v_lctx_520_, v_x_521_, v_a_522_, v_a_523_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___boxed(lean_object* v_00_u03b1_526_, lean_object* v_info_527_, lean_object* v_lctx_528_, lean_object* v_x_529_, lean_object* v_a_530_, lean_object* v_a_531_, lean_object* v_a_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages(v_00_u03b1_526_, v_info_527_, v_lctx_528_, v_x_529_, v_a_530_, v_a_531_);
lean_dec(v_a_531_);
lean_dec_ref(v_a_530_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__1(lean_object* v_msg_534_){
_start:
{
lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_535_ = l_Lean_instInhabitedMetavarDecl_default;
v___x_536_ = lean_panic_fn_borrowed(v___x_535_, v_msg_534_);
return v___x_536_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___closed__0(void){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = l_Lean_Elab_Command_instInhabitedCommandElabM(lean_box(0));
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3(lean_object* v_msg_538_, lean_object* v___y_539_, lean_object* v___y_540_){
_start:
{
lean_object* v___x_542_; lean_object* v___x_399__overap_543_; lean_object* v___x_544_; 
v___x_542_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___closed__0, &lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___closed__0_once, _init_lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___closed__0);
v___x_399__overap_543_ = lean_panic_fn_borrowed(v___x_542_, v_msg_538_);
lean_inc(v___y_540_);
lean_inc_ref(v___y_539_);
v___x_544_ = lean_apply_3(v___x_399__overap_543_, v___y_539_, v___y_540_, lean_box(0));
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3___boxed(lean_object* v_msg_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_){
_start:
{
lean_object* v_res_549_; 
v_res_549_ = lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3(v_msg_545_, v___y_546_, v___y_547_);
lean_dec(v___y_547_);
lean_dec_ref(v___y_546_);
return v_res_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___lam__0(lean_object* v_goal_550_, lean_object* v_x_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = l_Lean_MVarId_getType(v_goal_550_, v___y_552_, v___y_553_, v___y_554_, v___y_555_);
if (lean_obj_tag(v___x_557_) == 0)
{
lean_object* v_a_558_; lean_object* v___x_559_; lean_object* v___x_560_; 
v_a_558_ = lean_ctor_get(v___x_557_, 0);
lean_inc(v_a_558_);
lean_dec_ref_known(v___x_557_, 1);
v___x_559_ = lean_box(0);
v___x_560_ = l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(v_a_558_, v___x_559_, v___y_552_, v___y_553_, v___y_554_, v___y_555_);
if (lean_obj_tag(v___x_560_) == 0)
{
lean_object* v_a_561_; lean_object* v___x_562_; lean_object* v___x_563_; 
v_a_561_ = lean_ctor_get(v___x_560_, 0);
lean_inc(v_a_561_);
lean_dec_ref_known(v___x_560_, 1);
v___x_562_ = l_Lean_Expr_mvarId_x21(v_a_561_);
lean_dec(v_a_561_);
v___x_563_ = lean_apply_6(v_x_551_, v___x_562_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, lean_box(0));
return v___x_563_;
}
else
{
lean_object* v_a_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_571_; 
lean_dec(v___y_555_);
lean_dec_ref(v___y_554_);
lean_dec(v___y_553_);
lean_dec_ref(v___y_552_);
lean_dec_ref(v_x_551_);
v_a_564_ = lean_ctor_get(v___x_560_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v___x_560_);
if (v_isSharedCheck_571_ == 0)
{
v___x_566_ = v___x_560_;
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_a_564_);
lean_dec(v___x_560_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
lean_object* v___x_569_; 
if (v_isShared_567_ == 0)
{
v___x_569_ = v___x_566_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_a_564_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
}
else
{
lean_object* v_a_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_579_; 
lean_dec(v___y_555_);
lean_dec_ref(v___y_554_);
lean_dec(v___y_553_);
lean_dec_ref(v___y_552_);
lean_dec_ref(v_x_551_);
v_a_572_ = lean_ctor_get(v___x_557_, 0);
v_isSharedCheck_579_ = !lean_is_exclusive(v___x_557_);
if (v_isSharedCheck_579_ == 0)
{
v___x_574_ = v___x_557_;
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_a_572_);
lean_dec(v___x_557_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_579_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v___x_577_; 
if (v_isShared_575_ == 0)
{
v___x_577_ = v___x_574_;
goto v_reusejp_576_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v_a_572_);
v___x_577_ = v_reuseFailAlloc_578_;
goto v_reusejp_576_;
}
v_reusejp_576_:
{
return v___x_577_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___lam__0___boxed(lean_object* v_goal_580_, lean_object* v_x_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_){
_start:
{
lean_object* v_res_587_; 
v_res_587_ = lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___lam__0(v_goal_580_, v_x_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_);
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___redArg(lean_object* v_keys_588_, lean_object* v_vals_589_, lean_object* v_i_590_, lean_object* v_k_591_){
_start:
{
lean_object* v___x_592_; uint8_t v___x_593_; 
v___x_592_ = lean_array_get_size(v_keys_588_);
v___x_593_ = lean_nat_dec_lt(v_i_590_, v___x_592_);
if (v___x_593_ == 0)
{
lean_object* v___x_594_; 
lean_dec(v_i_590_);
v___x_594_ = lean_box(0);
return v___x_594_;
}
else
{
lean_object* v_k_x27_595_; uint8_t v___x_596_; 
v_k_x27_595_ = lean_array_fget_borrowed(v_keys_588_, v_i_590_);
v___x_596_ = l_Lean_instBEqMVarId_beq(v_k_591_, v_k_x27_595_);
if (v___x_596_ == 0)
{
lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_597_ = lean_unsigned_to_nat(1u);
v___x_598_ = lean_nat_add(v_i_590_, v___x_597_);
lean_dec(v_i_590_);
v_i_590_ = v___x_598_;
goto _start;
}
else
{
lean_object* v___x_600_; lean_object* v___x_601_; 
v___x_600_ = lean_array_fget_borrowed(v_vals_589_, v_i_590_);
lean_dec(v_i_590_);
lean_inc(v___x_600_);
v___x_601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_601_, 0, v___x_600_);
return v___x_601_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_keys_602_, lean_object* v_vals_603_, lean_object* v_i_604_, lean_object* v_k_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___redArg(v_keys_602_, v_vals_603_, v_i_604_, v_k_605_);
lean_dec(v_k_605_);
lean_dec_ref(v_vals_603_);
lean_dec_ref(v_keys_602_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___redArg(lean_object* v_x_607_, size_t v_x_608_, lean_object* v_x_609_){
_start:
{
if (lean_obj_tag(v_x_607_) == 0)
{
lean_object* v_es_610_; lean_object* v___x_611_; size_t v___x_612_; size_t v___x_613_; lean_object* v_j_614_; lean_object* v___x_615_; 
v_es_610_ = lean_ctor_get(v_x_607_, 0);
v___x_611_ = lean_box(2);
v___x_612_ = ((size_t)31ULL);
v___x_613_ = lean_usize_land(v_x_608_, v___x_612_);
v_j_614_ = lean_usize_to_nat(v___x_613_);
v___x_615_ = lean_array_get_borrowed(v___x_611_, v_es_610_, v_j_614_);
lean_dec(v_j_614_);
switch(lean_obj_tag(v___x_615_))
{
case 0:
{
lean_object* v_key_616_; lean_object* v_val_617_; uint8_t v___x_618_; 
v_key_616_ = lean_ctor_get(v___x_615_, 0);
v_val_617_ = lean_ctor_get(v___x_615_, 1);
v___x_618_ = l_Lean_instBEqMVarId_beq(v_x_609_, v_key_616_);
if (v___x_618_ == 0)
{
lean_object* v___x_619_; 
v___x_619_ = lean_box(0);
return v___x_619_;
}
else
{
lean_object* v___x_620_; 
lean_inc(v_val_617_);
v___x_620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_620_, 0, v_val_617_);
return v___x_620_;
}
}
case 1:
{
lean_object* v_node_621_; size_t v___x_622_; size_t v___x_623_; 
v_node_621_ = lean_ctor_get(v___x_615_, 0);
v___x_622_ = ((size_t)5ULL);
v___x_623_ = lean_usize_shift_right(v_x_608_, v___x_622_);
v_x_607_ = v_node_621_;
v_x_608_ = v___x_623_;
goto _start;
}
default: 
{
lean_object* v___x_625_; 
v___x_625_ = lean_box(0);
return v___x_625_;
}
}
}
else
{
lean_object* v_ks_626_; lean_object* v_vs_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v_ks_626_ = lean_ctor_get(v_x_607_, 0);
v_vs_627_ = lean_ctor_get(v_x_607_, 1);
v___x_628_ = lean_unsigned_to_nat(0u);
v___x_629_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___redArg(v_ks_626_, v_vs_627_, v___x_628_, v_x_609_);
return v___x_629_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___redArg___boxed(lean_object* v_x_630_, lean_object* v_x_631_, lean_object* v_x_632_){
_start:
{
size_t v_x_615__boxed_633_; lean_object* v_res_634_; 
v_x_615__boxed_633_ = lean_unbox_usize(v_x_631_);
lean_dec(v_x_631_);
v_res_634_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___redArg(v_x_630_, v_x_615__boxed_633_, v_x_632_);
lean_dec(v_x_632_);
lean_dec_ref(v_x_630_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg(lean_object* v_x_635_, lean_object* v_x_636_){
_start:
{
uint64_t v___x_637_; size_t v___x_638_; lean_object* v___x_639_; 
v___x_637_ = l_Lean_instHashableMVarId_hash(v_x_636_);
v___x_638_ = lean_uint64_to_usize(v___x_637_);
v___x_639_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___redArg(v_x_635_, v___x_638_, v_x_636_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg___boxed(lean_object* v_x_640_, lean_object* v_x_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg(v_x_640_, v_x_641_);
lean_dec(v_x_641_);
lean_dec_ref(v_x_640_);
return v_res_642_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Lean_Elab_ContextInfo_runTactic_spec__2(lean_object* v_a_643_, lean_object* v_x_644_){
_start:
{
if (lean_obj_tag(v_x_644_) == 0)
{
uint8_t v___x_645_; 
v___x_645_ = 0;
return v___x_645_;
}
else
{
lean_object* v_head_646_; lean_object* v_tail_647_; uint8_t v___x_648_; 
v_head_646_ = lean_ctor_get(v_x_644_, 0);
v_tail_647_ = lean_ctor_get(v_x_644_, 1);
v___x_648_ = l_Lean_instBEqMVarId_beq(v_a_643_, v_head_646_);
if (v___x_648_ == 0)
{
v_x_644_ = v_tail_647_;
goto _start;
}
else
{
return v___x_648_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Lean_Elab_ContextInfo_runTactic_spec__2___boxed(lean_object* v_a_650_, lean_object* v_x_651_){
_start:
{
uint8_t v_res_652_; lean_object* v_r_653_; 
v_res_652_ = lp_mathlib_List_elem___at___00Lean_Elab_ContextInfo_runTactic_spec__2(v_a_650_, v_x_651_);
lean_dec(v_x_651_);
lean_dec(v_a_650_);
v_r_653_ = lean_box(v_res_652_);
return v_r_653_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3(void){
_start:
{
lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
v___x_657_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__2));
v___x_658_ = lean_unsigned_to_nat(14u);
v___x_659_ = lean_unsigned_to_nat(178u);
v___x_660_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__1));
v___x_661_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__0));
v___x_662_ = l_mkPanicMessageWithDecl(v___x_661_, v___x_660_, v___x_659_, v___x_658_, v___x_657_);
return v___x_662_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__7(void){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_666_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__6));
v___x_667_ = lean_unsigned_to_nat(4u);
v___x_668_ = lean_unsigned_to_nat(75u);
v___x_669_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__5));
v___x_670_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__4));
v___x_671_ = l_mkPanicMessageWithDecl(v___x_670_, v___x_669_, v___x_668_, v___x_667_, v___x_666_);
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg(lean_object* v_ctx_672_, lean_object* v_i_673_, lean_object* v_goal_674_, lean_object* v_x_675_, lean_object* v_a_676_, lean_object* v_a_677_){
_start:
{
lean_object* v_mctxBefore_679_; lean_object* v_goalsBefore_680_; lean_object* v___f_681_; lean_object* v___y_683_; lean_object* v___y_684_; lean_object* v___y_685_; lean_object* v___y_689_; lean_object* v___y_690_; uint8_t v___x_696_; 
v_mctxBefore_679_ = lean_ctor_get(v_i_673_, 1);
v_goalsBefore_680_ = lean_ctor_get(v_i_673_, 2);
lean_inc(v_goal_674_);
v___f_681_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_681_, 0, v_goal_674_);
lean_closure_set(v___f_681_, 1, v_x_675_);
v___x_696_ = lp_mathlib_List_elem___at___00Lean_Elab_ContextInfo_runTactic_spec__2(v_goal_674_, v_goalsBefore_680_);
if (v___x_696_ == 0)
{
lean_object* v___x_697_; lean_object* v___x_698_; 
v___x_697_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__7, &lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__7_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__7);
v___x_698_ = lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3(v___x_697_, v_a_676_, v_a_677_);
if (lean_obj_tag(v___x_698_) == 0)
{
lean_dec_ref_known(v___x_698_, 1);
v___y_689_ = v_a_676_;
v___y_690_ = v_a_677_;
goto v___jp_688_;
}
else
{
lean_object* v_a_699_; lean_object* v___x_701_; uint8_t v_isShared_702_; uint8_t v_isSharedCheck_706_; 
lean_dec_ref(v___f_681_);
lean_dec(v_goal_674_);
lean_dec_ref(v_ctx_672_);
v_a_699_ = lean_ctor_get(v___x_698_, 0);
v_isSharedCheck_706_ = !lean_is_exclusive(v___x_698_);
if (v_isSharedCheck_706_ == 0)
{
v___x_701_ = v___x_698_;
v_isShared_702_ = v_isSharedCheck_706_;
goto v_resetjp_700_;
}
else
{
lean_inc(v_a_699_);
lean_dec(v___x_698_);
v___x_701_ = lean_box(0);
v_isShared_702_ = v_isSharedCheck_706_;
goto v_resetjp_700_;
}
v_resetjp_700_:
{
lean_object* v___x_704_; 
if (v_isShared_702_ == 0)
{
v___x_704_ = v___x_701_;
goto v_reusejp_703_;
}
else
{
lean_object* v_reuseFailAlloc_705_; 
v_reuseFailAlloc_705_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_705_, 0, v_a_699_);
v___x_704_ = v_reuseFailAlloc_705_;
goto v_reusejp_703_;
}
v_reusejp_703_:
{
return v___x_704_;
}
}
}
}
else
{
v___y_689_ = v_a_676_;
v___y_690_ = v_a_677_;
goto v___jp_688_;
}
v___jp_682_:
{
lean_object* v_lctx_686_; lean_object* v___x_687_; 
v_lctx_686_ = lean_ctor_get(v___y_685_, 1);
lean_inc_ref(v_lctx_686_);
lean_dec_ref(v___y_685_);
v___x_687_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg(v_ctx_672_, v_lctx_686_, v___f_681_, v___y_683_, v___y_684_);
return v___x_687_;
}
v___jp_688_:
{
lean_object* v_decls_691_; lean_object* v___x_692_; 
v_decls_691_ = lean_ctor_get(v_mctxBefore_679_, 5);
v___x_692_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg(v_decls_691_, v_goal_674_);
lean_dec(v_goal_674_);
if (lean_obj_tag(v___x_692_) == 0)
{
lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_693_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3, &lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3);
v___x_694_ = lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__1(v___x_693_);
v___y_683_ = v___y_689_;
v___y_684_ = v___y_690_;
v___y_685_ = v___x_694_;
goto v___jp_682_;
}
else
{
lean_object* v_val_695_; 
v_val_695_ = lean_ctor_get(v___x_692_, 0);
lean_inc(v_val_695_);
lean_dec_ref_known(v___x_692_, 1);
v___y_683_ = v___y_689_;
v___y_684_ = v___y_690_;
v___y_685_ = v_val_695_;
goto v___jp_682_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___boxed(lean_object* v_ctx_707_, lean_object* v_i_708_, lean_object* v_goal_709_, lean_object* v_x_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_){
_start:
{
lean_object* v_res_714_; 
v_res_714_ = lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg(v_ctx_707_, v_i_708_, v_goal_709_, v_x_710_, v_a_711_, v_a_712_);
lean_dec(v_a_712_);
lean_dec_ref(v_a_711_);
lean_dec_ref(v_i_708_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic(lean_object* v_00_u03b1_715_, lean_object* v_ctx_716_, lean_object* v_i_717_, lean_object* v_goal_718_, lean_object* v_x_719_, lean_object* v_a_720_, lean_object* v_a_721_){
_start:
{
lean_object* v___x_723_; 
v___x_723_ = lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg(v_ctx_716_, v_i_717_, v_goal_718_, v_x_719_, v_a_720_, v_a_721_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTactic___boxed(lean_object* v_00_u03b1_724_, lean_object* v_ctx_725_, lean_object* v_i_726_, lean_object* v_goal_727_, lean_object* v_x_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_){
_start:
{
lean_object* v_res_732_; 
v_res_732_ = lp_mathlib_Lean_Elab_ContextInfo_runTactic(v_00_u03b1_724_, v_ctx_725_, v_i_726_, v_goal_727_, v_x_728_, v_a_729_, v_a_730_);
lean_dec(v_a_730_);
lean_dec_ref(v_a_729_);
lean_dec_ref(v_i_726_);
return v_res_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0(lean_object* v_00_u03b2_733_, lean_object* v_x_734_, lean_object* v_x_735_){
_start:
{
lean_object* v___x_736_; 
v___x_736_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg(v_x_734_, v_x_735_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___boxed(lean_object* v_00_u03b2_737_, lean_object* v_x_738_, lean_object* v_x_739_){
_start:
{
lean_object* v_res_740_; 
v_res_740_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0(v_00_u03b2_737_, v_x_738_, v_x_739_);
lean_dec(v_x_739_);
lean_dec_ref(v_x_738_);
return v_res_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0(lean_object* v_00_u03b2_741_, lean_object* v_x_742_, size_t v_x_743_, lean_object* v_x_744_){
_start:
{
lean_object* v___x_745_; 
v___x_745_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___redArg(v_x_742_, v_x_743_, v_x_744_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0___boxed(lean_object* v_00_u03b2_746_, lean_object* v_x_747_, lean_object* v_x_748_, lean_object* v_x_749_){
_start:
{
size_t v_x_806__boxed_750_; lean_object* v_res_751_; 
v_x_806__boxed_750_ = lean_unbox_usize(v_x_748_);
lean_dec(v_x_748_);
v_res_751_ = lp_mathlib_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0(v_00_u03b2_746_, v_x_747_, v_x_806__boxed_750_, v_x_749_);
lean_dec(v_x_749_);
lean_dec_ref(v_x_747_);
return v_res_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_752_, lean_object* v_keys_753_, lean_object* v_vals_754_, lean_object* v_heq_755_, lean_object* v_i_756_, lean_object* v_k_757_){
_start:
{
lean_object* v___x_758_; 
v___x_758_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___redArg(v_keys_753_, v_vals_754_, v_i_756_, v_k_757_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b2_759_, lean_object* v_keys_760_, lean_object* v_vals_761_, lean_object* v_heq_762_, lean_object* v_i_763_, lean_object* v_k_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_mathlib_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0_spec__0_spec__3(v_00_u03b2_759_, v_keys_760_, v_vals_761_, v_heq_762_, v_i_763_, v_k_764_);
lean_dec(v_k_764_);
lean_dec_ref(v_vals_761_);
lean_dec_ref(v_keys_760_);
return v_res_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__0(lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_){
_start:
{
lean_object* v___x_773_; 
lean_inc_ref(v___y_766_);
v___x_773_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_773_, 0, v___y_766_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__0___boxed(lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
lean_object* v_res_781_; 
v_res_781_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__0(v___y_774_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_);
lean_dec(v___y_779_);
lean_dec_ref(v___y_778_);
lean_dec(v___y_777_);
lean_dec_ref(v___y_776_);
lean_dec(v___y_775_);
lean_dec_ref(v___y_774_);
return v_res_781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__1(lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_){
_start:
{
lean_object* v___x_789_; lean_object* v___x_790_; 
v___x_789_ = lean_st_ref_get(v___y_783_);
v___x_790_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_790_, 0, v___x_789_);
return v___x_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__1___boxed(lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_){
_start:
{
lean_object* v_res_798_; 
v_res_798_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__1(v___y_791_, v___y_792_, v___y_793_, v___y_794_, v___y_795_, v___y_796_);
lean_dec(v___y_796_);
lean_dec_ref(v___y_795_);
lean_dec(v___y_794_);
lean_dec_ref(v___y_793_);
lean_dec(v___y_792_);
lean_dec_ref(v___y_791_);
return v_res_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_ContextInfo_runTacticCode_spec__0(lean_object* v___x_799_, lean_object* v_x_800_, lean_object* v_x_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_){
_start:
{
if (lean_obj_tag(v_x_800_) == 0)
{
lean_object* v___x_807_; lean_object* v___x_808_; 
lean_dec_ref(v___x_799_);
v___x_807_ = l_List_reverse___redArg(v_x_801_);
v___x_808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_808_, 0, v___x_807_);
return v___x_808_;
}
else
{
lean_object* v_head_809_; lean_object* v_tail_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_828_; 
v_head_809_ = lean_ctor_get(v_x_800_, 0);
v_tail_810_ = lean_ctor_get(v_x_800_, 1);
v_isSharedCheck_828_ = !lean_is_exclusive(v_x_800_);
if (v_isSharedCheck_828_ == 0)
{
v___x_812_ = v_x_800_;
v_isShared_813_ = v_isSharedCheck_828_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_tail_810_);
lean_inc(v_head_809_);
lean_dec(v_x_800_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_828_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___x_814_; 
lean_inc_ref(v___x_799_);
lean_inc(v___y_805_);
lean_inc_ref(v___y_804_);
lean_inc(v___y_803_);
lean_inc_ref(v___y_802_);
v___x_814_ = lean_apply_6(v___x_799_, v_head_809_, v___y_802_, v___y_803_, v___y_804_, v___y_805_, lean_box(0));
if (lean_obj_tag(v___x_814_) == 0)
{
lean_object* v_a_815_; lean_object* v___x_817_; 
v_a_815_ = lean_ctor_get(v___x_814_, 0);
lean_inc(v_a_815_);
lean_dec_ref_known(v___x_814_, 1);
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 1, v_x_801_);
lean_ctor_set(v___x_812_, 0, v_a_815_);
v___x_817_ = v___x_812_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_819_; 
v_reuseFailAlloc_819_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_819_, 0, v_a_815_);
lean_ctor_set(v_reuseFailAlloc_819_, 1, v_x_801_);
v___x_817_ = v_reuseFailAlloc_819_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
v_x_800_ = v_tail_810_;
v_x_801_ = v___x_817_;
goto _start;
}
}
else
{
lean_object* v_a_820_; lean_object* v___x_822_; uint8_t v_isShared_823_; uint8_t v_isSharedCheck_827_; 
lean_del_object(v___x_812_);
lean_dec(v_tail_810_);
lean_dec(v_x_801_);
lean_dec_ref(v___x_799_);
v_a_820_ = lean_ctor_get(v___x_814_, 0);
v_isSharedCheck_827_ = !lean_is_exclusive(v___x_814_);
if (v_isSharedCheck_827_ == 0)
{
v___x_822_ = v___x_814_;
v_isShared_823_ = v_isSharedCheck_827_;
goto v_resetjp_821_;
}
else
{
lean_inc(v_a_820_);
lean_dec(v___x_814_);
v___x_822_ = lean_box(0);
v_isShared_823_ = v_isSharedCheck_827_;
goto v_resetjp_821_;
}
v_resetjp_821_:
{
lean_object* v___x_825_; 
if (v_isShared_823_ == 0)
{
v___x_825_ = v___x_822_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_826_; 
v_reuseFailAlloc_826_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_826_, 0, v_a_820_);
v___x_825_ = v_reuseFailAlloc_826_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
return v___x_825_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_ContextInfo_runTacticCode_spec__0___boxed(lean_object* v___x_829_, lean_object* v_x_830_, lean_object* v_x_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_){
_start:
{
lean_object* v_res_837_; 
v_res_837_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_ContextInfo_runTacticCode_spec__0(v___x_829_, v_x_830_, v_x_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_);
lean_dec(v___y_835_);
lean_dec_ref(v___y_834_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
return v_res_837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__2(lean_object* v_code_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_m_841_, lean_object* v_goal_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = lp_mathlib_Lean_Elab_runTactic_x27(v_goal_842_, v_code_838_, v_a_839_, v_a_840_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
if (lean_obj_tag(v___x_848_) == 0)
{
lean_object* v_a_849_; lean_object* v_snd_850_; lean_object* v___x_851_; lean_object* v___x_852_; 
v_a_849_ = lean_ctor_get(v___x_848_, 0);
lean_inc(v_a_849_);
lean_dec_ref_known(v___x_848_, 1);
v_snd_850_ = lean_ctor_get(v_m_841_, 1);
lean_inc(v_snd_850_);
lean_dec_ref(v_m_841_);
v___x_851_ = lean_box(0);
v___x_852_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_ContextInfo_runTacticCode_spec__0(v_snd_850_, v_a_849_, v___x_851_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
return v___x_852_;
}
else
{
lean_object* v_a_853_; lean_object* v___x_855_; uint8_t v_isShared_856_; uint8_t v_isSharedCheck_860_; 
lean_dec_ref(v_m_841_);
v_a_853_ = lean_ctor_get(v___x_848_, 0);
v_isSharedCheck_860_ = !lean_is_exclusive(v___x_848_);
if (v_isSharedCheck_860_ == 0)
{
v___x_855_ = v___x_848_;
v_isShared_856_ = v_isSharedCheck_860_;
goto v_resetjp_854_;
}
else
{
lean_inc(v_a_853_);
lean_dec(v___x_848_);
v___x_855_ = lean_box(0);
v_isShared_856_ = v_isSharedCheck_860_;
goto v_resetjp_854_;
}
v_resetjp_854_:
{
lean_object* v___x_858_; 
if (v_isShared_856_ == 0)
{
v___x_858_ = v___x_855_;
goto v_reusejp_857_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v_a_853_);
v___x_858_ = v_reuseFailAlloc_859_;
goto v_reusejp_857_;
}
v_reusejp_857_:
{
return v___x_858_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__2___boxed(lean_object* v_code_861_, lean_object* v_a_862_, lean_object* v_a_863_, lean_object* v_m_864_, lean_object* v_goal_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_){
_start:
{
lean_object* v_res_871_; 
v_res_871_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__2(v_code_861_, v_a_862_, v_a_863_, v_m_864_, v_goal_865_, v___y_866_, v___y_867_, v___y_868_, v___y_869_);
lean_dec(v___y_869_);
lean_dec_ref(v___y_868_);
lean_dec(v___y_867_);
lean_dec_ref(v___y_866_);
return v_res_871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__3(lean_object* v_a_872_, lean_object* v_messages_873_, lean_object* v_a_x3f_874_){
_start:
{
lean_object* v___x_876_; lean_object* v_env_877_; lean_object* v_scopes_878_; lean_object* v_usedQuotCtxts_879_; lean_object* v_nextMacroScope_880_; lean_object* v_maxRecDepth_881_; lean_object* v_ngen_882_; lean_object* v_auxDeclNGen_883_; lean_object* v_infoState_884_; lean_object* v_traceState_885_; lean_object* v_snapshotTasks_886_; lean_object* v_prevLinterStates_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_897_; 
v___x_876_ = lean_st_ref_take(v_a_872_);
v_env_877_ = lean_ctor_get(v___x_876_, 0);
v_scopes_878_ = lean_ctor_get(v___x_876_, 2);
v_usedQuotCtxts_879_ = lean_ctor_get(v___x_876_, 3);
v_nextMacroScope_880_ = lean_ctor_get(v___x_876_, 4);
v_maxRecDepth_881_ = lean_ctor_get(v___x_876_, 5);
v_ngen_882_ = lean_ctor_get(v___x_876_, 6);
v_auxDeclNGen_883_ = lean_ctor_get(v___x_876_, 7);
v_infoState_884_ = lean_ctor_get(v___x_876_, 8);
v_traceState_885_ = lean_ctor_get(v___x_876_, 9);
v_snapshotTasks_886_ = lean_ctor_get(v___x_876_, 10);
v_prevLinterStates_887_ = lean_ctor_get(v___x_876_, 11);
v_isSharedCheck_897_ = !lean_is_exclusive(v___x_876_);
if (v_isSharedCheck_897_ == 0)
{
lean_object* v_unused_898_; 
v_unused_898_ = lean_ctor_get(v___x_876_, 1);
lean_dec(v_unused_898_);
v___x_889_ = v___x_876_;
v_isShared_890_ = v_isSharedCheck_897_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_prevLinterStates_887_);
lean_inc(v_snapshotTasks_886_);
lean_inc(v_traceState_885_);
lean_inc(v_infoState_884_);
lean_inc(v_auxDeclNGen_883_);
lean_inc(v_ngen_882_);
lean_inc(v_maxRecDepth_881_);
lean_inc(v_nextMacroScope_880_);
lean_inc(v_usedQuotCtxts_879_);
lean_inc(v_scopes_878_);
lean_inc(v_env_877_);
lean_dec(v___x_876_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_897_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___x_892_; 
if (v_isShared_890_ == 0)
{
lean_ctor_set(v___x_889_, 1, v_messages_873_);
v___x_892_ = v___x_889_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_896_; 
v_reuseFailAlloc_896_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_896_, 0, v_env_877_);
lean_ctor_set(v_reuseFailAlloc_896_, 1, v_messages_873_);
lean_ctor_set(v_reuseFailAlloc_896_, 2, v_scopes_878_);
lean_ctor_set(v_reuseFailAlloc_896_, 3, v_usedQuotCtxts_879_);
lean_ctor_set(v_reuseFailAlloc_896_, 4, v_nextMacroScope_880_);
lean_ctor_set(v_reuseFailAlloc_896_, 5, v_maxRecDepth_881_);
lean_ctor_set(v_reuseFailAlloc_896_, 6, v_ngen_882_);
lean_ctor_set(v_reuseFailAlloc_896_, 7, v_auxDeclNGen_883_);
lean_ctor_set(v_reuseFailAlloc_896_, 8, v_infoState_884_);
lean_ctor_set(v_reuseFailAlloc_896_, 9, v_traceState_885_);
lean_ctor_set(v_reuseFailAlloc_896_, 10, v_snapshotTasks_886_);
lean_ctor_set(v_reuseFailAlloc_896_, 11, v_prevLinterStates_887_);
v___x_892_ = v_reuseFailAlloc_896_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_893_ = lean_st_ref_set(v_a_872_, v___x_892_);
v___x_894_ = lean_box(0);
v___x_895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_895_, 0, v___x_894_);
return v___x_895_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__3___boxed(lean_object* v_a_899_, lean_object* v_messages_900_, lean_object* v_a_x3f_901_, lean_object* v___y_902_){
_start:
{
lean_object* v_res_903_; 
v_res_903_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__3(v_a_899_, v_messages_900_, v_a_x3f_901_);
lean_dec(v_a_x3f_901_);
lean_dec(v_a_899_);
return v_res_903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode(lean_object* v_ctx_906_, lean_object* v_i_907_, lean_object* v_goal_908_, lean_object* v_code_909_, lean_object* v_m_910_, lean_object* v_a_911_, lean_object* v_a_912_){
_start:
{
lean_object* v___f_914_; lean_object* v___x_915_; 
v___f_914_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__0));
v___x_915_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_914_, v_a_911_, v_a_912_);
if (lean_obj_tag(v___x_915_) == 0)
{
lean_object* v_a_916_; lean_object* v___f_917_; lean_object* v___x_918_; 
v_a_916_ = lean_ctor_get(v___x_915_, 0);
lean_inc(v_a_916_);
lean_dec_ref_known(v___x_915_, 1);
v___f_917_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__1));
v___x_918_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_917_, v_a_911_, v_a_912_);
if (lean_obj_tag(v___x_918_) == 0)
{
lean_object* v_a_919_; lean_object* v___x_920_; lean_object* v_messages_921_; lean_object* v___f_922_; lean_object* v_r_923_; 
v_a_919_ = lean_ctor_get(v___x_918_, 0);
lean_inc(v_a_919_);
lean_dec_ref_known(v___x_918_, 1);
v___x_920_ = lean_st_ref_get(v_a_912_);
v_messages_921_ = lean_ctor_get(v___x_920_, 1);
lean_inc_ref(v_messages_921_);
lean_dec(v___x_920_);
v___f_922_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__2___boxed), 10, 4);
lean_closure_set(v___f_922_, 0, v_code_909_);
lean_closure_set(v___f_922_, 1, v_a_916_);
lean_closure_set(v___f_922_, 2, v_a_919_);
lean_closure_set(v___f_922_, 3, v_m_910_);
v_r_923_ = lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg(v_ctx_906_, v_i_907_, v_goal_908_, v___f_922_, v_a_911_, v_a_912_);
if (lean_obj_tag(v_r_923_) == 0)
{
lean_object* v_a_924_; lean_object* v___x_926_; uint8_t v_isShared_927_; uint8_t v_isSharedCheck_940_; 
v_a_924_ = lean_ctor_get(v_r_923_, 0);
v_isSharedCheck_940_ = !lean_is_exclusive(v_r_923_);
if (v_isSharedCheck_940_ == 0)
{
v___x_926_ = v_r_923_;
v_isShared_927_ = v_isSharedCheck_940_;
goto v_resetjp_925_;
}
else
{
lean_inc(v_a_924_);
lean_dec(v_r_923_);
v___x_926_ = lean_box(0);
v_isShared_927_ = v_isSharedCheck_940_;
goto v_resetjp_925_;
}
v_resetjp_925_:
{
lean_object* v___x_929_; 
lean_inc(v_a_924_);
if (v_isShared_927_ == 0)
{
lean_ctor_set_tag(v___x_926_, 1);
v___x_929_ = v___x_926_;
goto v_reusejp_928_;
}
else
{
lean_object* v_reuseFailAlloc_939_; 
v_reuseFailAlloc_939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_939_, 0, v_a_924_);
v___x_929_ = v_reuseFailAlloc_939_;
goto v_reusejp_928_;
}
v_reusejp_928_:
{
lean_object* v___x_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_937_; 
v___x_930_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__3(v_a_912_, v_messages_921_, v___x_929_);
lean_dec_ref(v___x_929_);
v_isSharedCheck_937_ = !lean_is_exclusive(v___x_930_);
if (v_isSharedCheck_937_ == 0)
{
lean_object* v_unused_938_; 
v_unused_938_ = lean_ctor_get(v___x_930_, 0);
lean_dec(v_unused_938_);
v___x_932_ = v___x_930_;
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
else
{
lean_dec(v___x_930_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_935_; 
if (v_isShared_933_ == 0)
{
lean_ctor_set(v___x_932_, 0, v_a_924_);
v___x_935_ = v___x_932_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v_a_924_);
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
else
{
lean_object* v_a_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_950_; 
v_a_941_ = lean_ctor_get(v_r_923_, 0);
lean_inc(v_a_941_);
lean_dec_ref_known(v_r_923_, 1);
v___x_942_ = lean_box(0);
v___x_943_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___lam__3(v_a_912_, v_messages_921_, v___x_942_);
v_isSharedCheck_950_ = !lean_is_exclusive(v___x_943_);
if (v_isSharedCheck_950_ == 0)
{
lean_object* v_unused_951_; 
v_unused_951_ = lean_ctor_get(v___x_943_, 0);
lean_dec(v_unused_951_);
v___x_945_ = v___x_943_;
v_isShared_946_ = v_isSharedCheck_950_;
goto v_resetjp_944_;
}
else
{
lean_dec(v___x_943_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_950_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
lean_object* v___x_948_; 
if (v_isShared_946_ == 0)
{
lean_ctor_set_tag(v___x_945_, 1);
lean_ctor_set(v___x_945_, 0, v_a_941_);
v___x_948_ = v___x_945_;
goto v_reusejp_947_;
}
else
{
lean_object* v_reuseFailAlloc_949_; 
v_reuseFailAlloc_949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_949_, 0, v_a_941_);
v___x_948_ = v_reuseFailAlloc_949_;
goto v_reusejp_947_;
}
v_reusejp_947_:
{
return v___x_948_;
}
}
}
}
else
{
lean_object* v_a_952_; lean_object* v___x_954_; uint8_t v_isShared_955_; uint8_t v_isSharedCheck_959_; 
lean_dec(v_a_916_);
lean_dec_ref(v_m_910_);
lean_dec(v_code_909_);
lean_dec(v_goal_908_);
lean_dec_ref(v_ctx_906_);
v_a_952_ = lean_ctor_get(v___x_918_, 0);
v_isSharedCheck_959_ = !lean_is_exclusive(v___x_918_);
if (v_isSharedCheck_959_ == 0)
{
v___x_954_ = v___x_918_;
v_isShared_955_ = v_isSharedCheck_959_;
goto v_resetjp_953_;
}
else
{
lean_inc(v_a_952_);
lean_dec(v___x_918_);
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
else
{
lean_object* v_a_960_; lean_object* v___x_962_; uint8_t v_isShared_963_; uint8_t v_isSharedCheck_967_; 
lean_dec_ref(v_m_910_);
lean_dec(v_code_909_);
lean_dec(v_goal_908_);
lean_dec_ref(v_ctx_906_);
v_a_960_ = lean_ctor_get(v___x_915_, 0);
v_isSharedCheck_967_ = !lean_is_exclusive(v___x_915_);
if (v_isSharedCheck_967_ == 0)
{
v___x_962_ = v___x_915_;
v_isShared_963_ = v_isSharedCheck_967_;
goto v_resetjp_961_;
}
else
{
lean_inc(v_a_960_);
lean_dec(v___x_915_);
v___x_962_ = lean_box(0);
v_isShared_963_ = v_isSharedCheck_967_;
goto v_resetjp_961_;
}
v_resetjp_961_:
{
lean_object* v___x_965_; 
if (v_isShared_963_ == 0)
{
v___x_965_ = v___x_962_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v_a_960_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___boxed(lean_object* v_ctx_968_, lean_object* v_i_969_, lean_object* v_goal_970_, lean_object* v_code_971_, lean_object* v_m_972_, lean_object* v_a_973_, lean_object* v_a_974_, lean_object* v_a_975_){
_start:
{
lean_object* v_res_976_; 
v_res_976_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode(v_ctx_968_, v_i_969_, v_goal_970_, v_code_971_, v_m_972_, v_a_973_, v_a_974_);
lean_dec(v_a_974_);
lean_dec_ref(v_a_973_);
lean_dec_ref(v_i_969_);
return v_res_976_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___redArg(lean_object* v_info_977_, lean_object* v_x_978_, lean_object* v_a_979_, lean_object* v_a_980_){
_start:
{
lean_object* v_toCommandContextInfo_982_; lean_object* v_parentDecl_x3f_983_; lean_object* v___x_985_; uint8_t v_isShared_986_; uint8_t v_isSharedCheck_1239_; 
v_toCommandContextInfo_982_ = lean_ctor_get(v_info_977_, 0);
v_parentDecl_x3f_983_ = lean_ctor_get(v_info_977_, 1);
v_isSharedCheck_1239_ = !lean_is_exclusive(v_info_977_);
if (v_isSharedCheck_1239_ == 0)
{
lean_object* v_unused_1240_; 
v_unused_1240_ = lean_ctor_get(v_info_977_, 2);
lean_dec(v_unused_1240_);
v___x_985_ = v_info_977_;
v_isShared_986_ = v_isSharedCheck_1239_;
goto v_resetjp_984_;
}
else
{
lean_inc(v_parentDecl_x3f_983_);
lean_inc(v_toCommandContextInfo_982_);
lean_dec(v_info_977_);
v___x_985_ = lean_box(0);
v_isShared_986_ = v_isSharedCheck_1239_;
goto v_resetjp_984_;
}
v_resetjp_984_:
{
lean_object* v_env_987_; lean_object* v_options_988_; lean_object* v_currNamespace_989_; lean_object* v_openDecls_990_; lean_object* v_ngen_991_; lean_object* v_fileName_992_; lean_object* v_fileMap_993_; lean_object* v_ref_994_; lean_object* v_a_996_; lean_object* v_a_1003_; uint8_t v___y_1006_; uint8_t v___y_1007_; lean_object* v___y_1008_; lean_object* v___y_1009_; lean_object* v_fileName_1010_; lean_object* v_fileMap_1011_; lean_object* v_currRecDepth_1012_; lean_object* v_ref_1013_; lean_object* v_currNamespace_1014_; lean_object* v_openDecls_1015_; lean_object* v_initHeartbeats_1016_; lean_object* v_maxHeartbeats_1017_; lean_object* v_quotContext_1018_; lean_object* v_currMacroScope_1019_; lean_object* v_cancelTk_x3f_1020_; uint8_t v_suppressElabErrors_1021_; lean_object* v_inheritedTraceOptions_1022_; lean_object* v___y_1023_; uint8_t v___y_1091_; uint8_t v___y_1092_; lean_object* v___y_1093_; lean_object* v___y_1094_; lean_object* v___y_1095_; lean_object* v___y_1096_; lean_object* v___y_1111_; lean_object* v___y_1112_; uint8_t v___y_1113_; lean_object* v___y_1114_; lean_object* v___y_1115_; uint8_t v___y_1116_; lean_object* v___y_1117_; uint8_t v___y_1118_; uint8_t v___x_1138_; lean_object* v_env_1139_; lean_object* v___x_1140_; lean_object* v___y_1142_; lean_object* v___y_1143_; lean_object* v___y_1144_; uint8_t v___y_1145_; uint8_t v___y_1146_; lean_object* v___y_1147_; lean_object* v___y_1148_; lean_object* v___y_1178_; lean_object* v___y_1179_; lean_object* v___y_1180_; uint8_t v___y_1181_; lean_object* v___y_1182_; uint8_t v___y_1183_; uint8_t v___y_1184_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___y_1214_; 
v_env_987_ = lean_ctor_get(v_toCommandContextInfo_982_, 0);
lean_inc_ref(v_env_987_);
v_options_988_ = lean_ctor_get(v_toCommandContextInfo_982_, 4);
lean_inc_ref(v_options_988_);
v_currNamespace_989_ = lean_ctor_get(v_toCommandContextInfo_982_, 5);
lean_inc(v_currNamespace_989_);
v_openDecls_990_ = lean_ctor_get(v_toCommandContextInfo_982_, 6);
lean_inc(v_openDecls_990_);
v_ngen_991_ = lean_ctor_get(v_toCommandContextInfo_982_, 7);
lean_inc_ref(v_ngen_991_);
lean_dec_ref(v_toCommandContextInfo_982_);
v_fileName_992_ = lean_ctor_get(v_a_979_, 0);
v_fileMap_993_ = lean_ctor_get(v_a_979_, 1);
v_ref_994_ = lean_ctor_get(v_a_979_, 7);
v___x_1138_ = 0;
v_env_1139_ = l_Lean_Environment_setExporting(v_env_987_, v___x_1138_);
v___x_1140_ = l_Lean_Options_empty;
v___x_1204_ = lean_unsigned_to_nat(0u);
v___x_1205_ = lean_unsigned_to_nat(1000u);
v___x_1206_ = lean_box(0);
v___x_1207_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__4);
v___x_1208_ = lean_box(0);
v___x_1209_ = l_Lean_firstFrontendMacroScope;
v___x_1210_ = lean_box(0);
v___x_1211_ = lean_unsigned_to_nat(1u);
v___x_1212_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__5);
if (lean_obj_tag(v_parentDecl_x3f_983_) == 0)
{
v___y_1214_ = v___x_1208_;
goto v___jp_1213_;
}
else
{
lean_object* v_val_1238_; 
v_val_1238_ = lean_ctor_get(v_parentDecl_x3f_983_, 0);
lean_inc(v_val_1238_);
lean_dec_ref_known(v_parentDecl_x3f_983_, 1);
v___y_1214_ = v_val_1238_;
goto v___jp_1213_;
}
v___jp_995_:
{
lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; 
v___x_997_ = lean_io_error_to_string(v_a_996_);
v___x_998_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_998_, 0, v___x_997_);
v___x_999_ = l_Lean_MessageData_ofFormat(v___x_998_);
lean_inc(v_ref_994_);
v___x_1000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1000_, 0, v_ref_994_);
lean_ctor_set(v___x_1000_, 1, v___x_999_);
v___x_1001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1001_, 0, v___x_1000_);
return v___x_1001_;
}
v___jp_1002_:
{
lean_object* v___x_1004_; 
v___x_1004_ = lean_mk_io_user_error(v_a_1003_);
v_a_996_ = v___x_1004_;
goto v___jp_995_;
}
v___jp_1005_:
{
lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; 
v___x_1024_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__1(v_options_988_, v___y_1009_);
v___x_1025_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1025_, 0, v_fileName_1010_);
lean_ctor_set(v___x_1025_, 1, v_fileMap_1011_);
lean_ctor_set(v___x_1025_, 2, v_options_988_);
lean_ctor_set(v___x_1025_, 3, v_currRecDepth_1012_);
lean_ctor_set(v___x_1025_, 4, v___x_1024_);
lean_ctor_set(v___x_1025_, 5, v_ref_1013_);
lean_ctor_set(v___x_1025_, 6, v_currNamespace_1014_);
lean_ctor_set(v___x_1025_, 7, v_openDecls_1015_);
lean_ctor_set(v___x_1025_, 8, v_initHeartbeats_1016_);
lean_ctor_set(v___x_1025_, 9, v_maxHeartbeats_1017_);
lean_ctor_set(v___x_1025_, 10, v_quotContext_1018_);
lean_ctor_set(v___x_1025_, 11, v_currMacroScope_1019_);
lean_ctor_set(v___x_1025_, 12, v_cancelTk_x3f_1020_);
lean_ctor_set(v___x_1025_, 13, v_inheritedTraceOptions_1022_);
lean_ctor_set_uint8(v___x_1025_, sizeof(void*)*14, v___y_1006_);
lean_ctor_set_uint8(v___x_1025_, sizeof(void*)*14 + 1, v_suppressElabErrors_1021_);
v___x_1026_ = lean_apply_3(v_x_978_, v___x_1025_, v___y_1023_, lean_box(0));
if (lean_obj_tag(v___x_1026_) == 0)
{
lean_object* v_a_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1074_; 
v_a_1027_ = lean_ctor_get(v___x_1026_, 0);
v_isSharedCheck_1074_ = !lean_is_exclusive(v___x_1026_);
if (v_isSharedCheck_1074_ == 0)
{
v___x_1029_ = v___x_1026_;
v_isShared_1030_ = v_isSharedCheck_1074_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_a_1027_);
lean_dec(v___x_1026_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1074_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v_traceState_1033_; lean_object* v_traceState_1034_; lean_object* v_env_1035_; lean_object* v_messages_1036_; lean_object* v_scopes_1037_; lean_object* v_usedQuotCtxts_1038_; lean_object* v_nextMacroScope_1039_; lean_object* v_maxRecDepth_1040_; lean_object* v_ngen_1041_; lean_object* v_auxDeclNGen_1042_; lean_object* v_infoState_1043_; lean_object* v_snapshotTasks_1044_; lean_object* v_prevLinterStates_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1072_; 
v___x_1031_ = lean_st_ref_get(v___y_1008_);
lean_dec(v___y_1008_);
v___x_1032_ = lean_st_ref_take(v_a_980_);
v_traceState_1033_ = lean_ctor_get(v___x_1032_, 9);
lean_inc_ref(v_traceState_1033_);
v_traceState_1034_ = lean_ctor_get(v___x_1031_, 4);
lean_inc_ref(v_traceState_1034_);
v_env_1035_ = lean_ctor_get(v___x_1032_, 0);
v_messages_1036_ = lean_ctor_get(v___x_1032_, 1);
v_scopes_1037_ = lean_ctor_get(v___x_1032_, 2);
v_usedQuotCtxts_1038_ = lean_ctor_get(v___x_1032_, 3);
v_nextMacroScope_1039_ = lean_ctor_get(v___x_1032_, 4);
v_maxRecDepth_1040_ = lean_ctor_get(v___x_1032_, 5);
v_ngen_1041_ = lean_ctor_get(v___x_1032_, 6);
v_auxDeclNGen_1042_ = lean_ctor_get(v___x_1032_, 7);
v_infoState_1043_ = lean_ctor_get(v___x_1032_, 8);
v_snapshotTasks_1044_ = lean_ctor_get(v___x_1032_, 10);
v_prevLinterStates_1045_ = lean_ctor_get(v___x_1032_, 11);
v_isSharedCheck_1072_ = !lean_is_exclusive(v___x_1032_);
if (v_isSharedCheck_1072_ == 0)
{
lean_object* v_unused_1073_; 
v_unused_1073_ = lean_ctor_get(v___x_1032_, 9);
lean_dec(v_unused_1073_);
v___x_1047_ = v___x_1032_;
v_isShared_1048_ = v_isSharedCheck_1072_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_prevLinterStates_1045_);
lean_inc(v_snapshotTasks_1044_);
lean_inc(v_infoState_1043_);
lean_inc(v_auxDeclNGen_1042_);
lean_inc(v_ngen_1041_);
lean_inc(v_maxRecDepth_1040_);
lean_inc(v_nextMacroScope_1039_);
lean_inc(v_usedQuotCtxts_1038_);
lean_inc(v_scopes_1037_);
lean_inc(v_messages_1036_);
lean_inc(v_env_1035_);
lean_dec(v___x_1032_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1072_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v_messages_1049_; lean_object* v_infoState_1050_; uint64_t v_tid_1051_; lean_object* v_traces_1052_; lean_object* v_traces_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1071_; 
v_messages_1049_ = lean_ctor_get(v___x_1031_, 6);
lean_inc_ref(v_messages_1049_);
v_infoState_1050_ = lean_ctor_get(v___x_1031_, 7);
lean_inc_ref(v_infoState_1050_);
lean_dec(v___x_1031_);
v_tid_1051_ = lean_ctor_get_uint64(v_traceState_1033_, sizeof(void*)*1);
v_traces_1052_ = lean_ctor_get(v_traceState_1033_, 0);
lean_inc_ref(v_traces_1052_);
lean_dec_ref(v_traceState_1033_);
v_traces_1053_ = lean_ctor_get(v_traceState_1034_, 0);
v_isSharedCheck_1071_ = !lean_is_exclusive(v_traceState_1034_);
if (v_isSharedCheck_1071_ == 0)
{
v___x_1055_ = v_traceState_1034_;
v_isShared_1056_ = v_isSharedCheck_1071_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_traces_1053_);
lean_dec(v_traceState_1034_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1071_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1060_; 
v___x_1057_ = l_Lean_MessageLog_append(v_messages_1036_, v_messages_1049_);
v___x_1058_ = l_Lean_PersistentArray_append___redArg(v_traces_1052_, v_traces_1053_);
lean_dec_ref(v_traces_1053_);
if (v_isShared_1056_ == 0)
{
lean_ctor_set(v___x_1055_, 0, v___x_1058_);
v___x_1060_ = v___x_1055_;
goto v_reusejp_1059_;
}
else
{
lean_object* v_reuseFailAlloc_1070_; 
v_reuseFailAlloc_1070_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1070_, 0, v___x_1058_);
v___x_1060_ = v_reuseFailAlloc_1070_;
goto v_reusejp_1059_;
}
v_reusejp_1059_:
{
lean_object* v___x_1062_; 
lean_ctor_set_uint64(v___x_1060_, sizeof(void*)*1, v_tid_1051_);
if (v_isShared_1048_ == 0)
{
lean_ctor_set(v___x_1047_, 9, v___x_1060_);
lean_ctor_set(v___x_1047_, 1, v___x_1057_);
v___x_1062_ = v___x_1047_;
goto v_reusejp_1061_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v_env_1035_);
lean_ctor_set(v_reuseFailAlloc_1069_, 1, v___x_1057_);
lean_ctor_set(v_reuseFailAlloc_1069_, 2, v_scopes_1037_);
lean_ctor_set(v_reuseFailAlloc_1069_, 3, v_usedQuotCtxts_1038_);
lean_ctor_set(v_reuseFailAlloc_1069_, 4, v_nextMacroScope_1039_);
lean_ctor_set(v_reuseFailAlloc_1069_, 5, v_maxRecDepth_1040_);
lean_ctor_set(v_reuseFailAlloc_1069_, 6, v_ngen_1041_);
lean_ctor_set(v_reuseFailAlloc_1069_, 7, v_auxDeclNGen_1042_);
lean_ctor_set(v_reuseFailAlloc_1069_, 8, v_infoState_1043_);
lean_ctor_set(v_reuseFailAlloc_1069_, 9, v___x_1060_);
lean_ctor_set(v_reuseFailAlloc_1069_, 10, v_snapshotTasks_1044_);
lean_ctor_set(v_reuseFailAlloc_1069_, 11, v_prevLinterStates_1045_);
v___x_1062_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1061_;
}
v_reusejp_1061_:
{
lean_object* v___x_1063_; lean_object* v_trees_1064_; lean_object* v___x_1065_; lean_object* v___x_1067_; 
v___x_1063_ = lean_st_ref_set(v_a_980_, v___x_1062_);
v_trees_1064_ = lean_ctor_get(v_infoState_1050_, 2);
lean_inc_ref(v_trees_1064_);
lean_dec_ref(v_infoState_1050_);
v___x_1065_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1065_, 0, v_a_1027_);
lean_ctor_set(v___x_1065_, 1, v_trees_1064_);
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 0, v___x_1065_);
v___x_1067_ = v___x_1029_;
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
}
}
}
}
}
else
{
lean_object* v_a_1075_; 
lean_dec(v___y_1008_);
v_a_1075_ = lean_ctor_get(v___x_1026_, 0);
lean_inc(v_a_1075_);
lean_dec_ref_known(v___x_1026_, 1);
if (lean_obj_tag(v_a_1075_) == 0)
{
lean_object* v_msg_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; 
v_msg_1076_ = lean_ctor_get(v_a_1075_, 1);
lean_inc_ref(v_msg_1076_);
lean_dec_ref_known(v_a_1075_, 2);
v___x_1077_ = l_Lean_MessageData_toString(v_msg_1076_);
v___x_1078_ = lean_mk_io_user_error(v___x_1077_);
v_a_996_ = v___x_1078_;
goto v___jp_995_;
}
else
{
lean_object* v_id_1079_; lean_object* v___x_1080_; 
v_id_1079_ = lean_ctor_get(v_a_1075_, 0);
lean_inc(v_id_1079_);
lean_dec_ref_known(v_a_1075_, 2);
v___x_1080_ = l_Lean_InternalExceptionId_getName(v_id_1079_);
if (lean_obj_tag(v___x_1080_) == 0)
{
lean_object* v_a_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; 
lean_dec(v_id_1079_);
v_a_1081_ = lean_ctor_get(v___x_1080_, 0);
lean_inc(v_a_1081_);
lean_dec_ref_known(v___x_1080_, 1);
v___x_1082_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__0));
v___x_1083_ = l_Lean_Name_toString(v_a_1081_, v___y_1007_);
v___x_1084_ = lean_string_append(v___x_1082_, v___x_1083_);
lean_dec_ref(v___x_1083_);
v_a_1003_ = v___x_1084_;
goto v___jp_1002_;
}
else
{
lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; 
lean_dec_ref_known(v___x_1080_, 1);
v___x_1085_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__1));
v___x_1086_ = l_Nat_reprFast(v_id_1079_);
v___x_1087_ = lean_string_append(v___x_1085_, v___x_1086_);
lean_dec_ref(v___x_1086_);
v___x_1088_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__2));
v___x_1089_ = lean_string_append(v___x_1087_, v___x_1088_);
v_a_1003_ = v___x_1089_;
goto v___jp_1002_;
}
}
}
}
v___jp_1090_:
{
lean_object* v_fileName_1097_; lean_object* v_fileMap_1098_; lean_object* v_currRecDepth_1099_; lean_object* v_ref_1100_; lean_object* v_currNamespace_1101_; lean_object* v_openDecls_1102_; lean_object* v_initHeartbeats_1103_; lean_object* v_maxHeartbeats_1104_; lean_object* v_quotContext_1105_; lean_object* v_currMacroScope_1106_; lean_object* v_cancelTk_x3f_1107_; uint8_t v_suppressElabErrors_1108_; lean_object* v_inheritedTraceOptions_1109_; 
v_fileName_1097_ = lean_ctor_get(v___y_1095_, 0);
lean_inc_ref(v_fileName_1097_);
v_fileMap_1098_ = lean_ctor_get(v___y_1095_, 1);
lean_inc_ref(v_fileMap_1098_);
v_currRecDepth_1099_ = lean_ctor_get(v___y_1095_, 3);
lean_inc(v_currRecDepth_1099_);
v_ref_1100_ = lean_ctor_get(v___y_1095_, 5);
lean_inc(v_ref_1100_);
v_currNamespace_1101_ = lean_ctor_get(v___y_1095_, 6);
lean_inc(v_currNamespace_1101_);
v_openDecls_1102_ = lean_ctor_get(v___y_1095_, 7);
lean_inc(v_openDecls_1102_);
v_initHeartbeats_1103_ = lean_ctor_get(v___y_1095_, 8);
lean_inc(v_initHeartbeats_1103_);
v_maxHeartbeats_1104_ = lean_ctor_get(v___y_1095_, 9);
lean_inc(v_maxHeartbeats_1104_);
v_quotContext_1105_ = lean_ctor_get(v___y_1095_, 10);
lean_inc(v_quotContext_1105_);
v_currMacroScope_1106_ = lean_ctor_get(v___y_1095_, 11);
lean_inc(v_currMacroScope_1106_);
v_cancelTk_x3f_1107_ = lean_ctor_get(v___y_1095_, 12);
lean_inc(v_cancelTk_x3f_1107_);
v_suppressElabErrors_1108_ = lean_ctor_get_uint8(v___y_1095_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1109_ = lean_ctor_get(v___y_1095_, 13);
lean_inc_ref(v_inheritedTraceOptions_1109_);
lean_dec_ref(v___y_1095_);
v___y_1006_ = v___y_1091_;
v___y_1007_ = v___y_1092_;
v___y_1008_ = v___y_1093_;
v___y_1009_ = v___y_1094_;
v_fileName_1010_ = v_fileName_1097_;
v_fileMap_1011_ = v_fileMap_1098_;
v_currRecDepth_1012_ = v_currRecDepth_1099_;
v_ref_1013_ = v_ref_1100_;
v_currNamespace_1014_ = v_currNamespace_1101_;
v_openDecls_1015_ = v_openDecls_1102_;
v_initHeartbeats_1016_ = v_initHeartbeats_1103_;
v_maxHeartbeats_1017_ = v_maxHeartbeats_1104_;
v_quotContext_1018_ = v_quotContext_1105_;
v_currMacroScope_1019_ = v_currMacroScope_1106_;
v_cancelTk_x3f_1020_ = v_cancelTk_x3f_1107_;
v_suppressElabErrors_1021_ = v_suppressElabErrors_1108_;
v_inheritedTraceOptions_1022_ = v_inheritedTraceOptions_1109_;
v___y_1023_ = v___y_1096_;
goto v___jp_1005_;
}
v___jp_1110_:
{
if (v___y_1118_ == 0)
{
lean_object* v___x_1119_; lean_object* v_env_1120_; lean_object* v_nextMacroScope_1121_; lean_object* v_ngen_1122_; lean_object* v_auxDeclNGen_1123_; lean_object* v_traceState_1124_; lean_object* v_messages_1125_; lean_object* v_infoState_1126_; lean_object* v_snapshotTasks_1127_; lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1136_; 
v___x_1119_ = lean_st_ref_take(v___y_1112_);
v_env_1120_ = lean_ctor_get(v___x_1119_, 0);
v_nextMacroScope_1121_ = lean_ctor_get(v___x_1119_, 1);
v_ngen_1122_ = lean_ctor_get(v___x_1119_, 2);
v_auxDeclNGen_1123_ = lean_ctor_get(v___x_1119_, 3);
v_traceState_1124_ = lean_ctor_get(v___x_1119_, 4);
v_messages_1125_ = lean_ctor_get(v___x_1119_, 6);
v_infoState_1126_ = lean_ctor_get(v___x_1119_, 7);
v_snapshotTasks_1127_ = lean_ctor_get(v___x_1119_, 8);
v_isSharedCheck_1136_ = !lean_is_exclusive(v___x_1119_);
if (v_isSharedCheck_1136_ == 0)
{
lean_object* v_unused_1137_; 
v_unused_1137_ = lean_ctor_get(v___x_1119_, 5);
lean_dec(v_unused_1137_);
v___x_1129_ = v___x_1119_;
v_isShared_1130_ = v_isSharedCheck_1136_;
goto v_resetjp_1128_;
}
else
{
lean_inc(v_snapshotTasks_1127_);
lean_inc(v_infoState_1126_);
lean_inc(v_messages_1125_);
lean_inc(v_traceState_1124_);
lean_inc(v_auxDeclNGen_1123_);
lean_inc(v_ngen_1122_);
lean_inc(v_nextMacroScope_1121_);
lean_inc(v_env_1120_);
lean_dec(v___x_1119_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1136_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v___x_1131_; lean_object* v___x_1133_; 
v___x_1131_ = l_Lean_Kernel_enableDiag(v_env_1120_, v___y_1113_);
lean_inc_ref(v___y_1114_);
if (v_isShared_1130_ == 0)
{
lean_ctor_set(v___x_1129_, 5, v___y_1114_);
lean_ctor_set(v___x_1129_, 0, v___x_1131_);
v___x_1133_ = v___x_1129_;
goto v_reusejp_1132_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v___x_1131_);
lean_ctor_set(v_reuseFailAlloc_1135_, 1, v_nextMacroScope_1121_);
lean_ctor_set(v_reuseFailAlloc_1135_, 2, v_ngen_1122_);
lean_ctor_set(v_reuseFailAlloc_1135_, 3, v_auxDeclNGen_1123_);
lean_ctor_set(v_reuseFailAlloc_1135_, 4, v_traceState_1124_);
lean_ctor_set(v_reuseFailAlloc_1135_, 5, v___y_1114_);
lean_ctor_set(v_reuseFailAlloc_1135_, 6, v_messages_1125_);
lean_ctor_set(v_reuseFailAlloc_1135_, 7, v_infoState_1126_);
lean_ctor_set(v_reuseFailAlloc_1135_, 8, v_snapshotTasks_1127_);
v___x_1133_ = v_reuseFailAlloc_1135_;
goto v_reusejp_1132_;
}
v_reusejp_1132_:
{
lean_object* v___x_1134_; 
v___x_1134_ = lean_st_ref_set(v___y_1112_, v___x_1133_);
v___y_1091_ = v___y_1113_;
v___y_1092_ = v___y_1116_;
v___y_1093_ = v___y_1115_;
v___y_1094_ = v___y_1117_;
v___y_1095_ = v___y_1111_;
v___y_1096_ = v___y_1112_;
goto v___jp_1090_;
}
}
}
else
{
v___y_1091_ = v___y_1113_;
v___y_1092_ = v___y_1116_;
v___y_1093_ = v___y_1115_;
v___y_1094_ = v___y_1117_;
v___y_1095_ = v___y_1111_;
v___y_1096_ = v___y_1112_;
goto v___jp_1090_;
}
}
v___jp_1141_:
{
lean_object* v___x_1149_; lean_object* v_fileName_1150_; lean_object* v_fileMap_1151_; lean_object* v_currRecDepth_1152_; lean_object* v_ref_1153_; lean_object* v_currNamespace_1154_; lean_object* v_openDecls_1155_; lean_object* v_initHeartbeats_1156_; lean_object* v_maxHeartbeats_1157_; lean_object* v_quotContext_1158_; lean_object* v_currMacroScope_1159_; lean_object* v_cancelTk_x3f_1160_; uint8_t v_suppressElabErrors_1161_; lean_object* v_inheritedTraceOptions_1162_; lean_object* v___x_1164_; uint8_t v_isShared_1165_; uint8_t v_isSharedCheck_1174_; 
v___x_1149_ = lean_st_ref_get(v___y_1148_);
v_fileName_1150_ = lean_ctor_get(v___y_1147_, 0);
v_fileMap_1151_ = lean_ctor_get(v___y_1147_, 1);
v_currRecDepth_1152_ = lean_ctor_get(v___y_1147_, 3);
v_ref_1153_ = lean_ctor_get(v___y_1147_, 5);
v_currNamespace_1154_ = lean_ctor_get(v___y_1147_, 6);
v_openDecls_1155_ = lean_ctor_get(v___y_1147_, 7);
v_initHeartbeats_1156_ = lean_ctor_get(v___y_1147_, 8);
v_maxHeartbeats_1157_ = lean_ctor_get(v___y_1147_, 9);
v_quotContext_1158_ = lean_ctor_get(v___y_1147_, 10);
v_currMacroScope_1159_ = lean_ctor_get(v___y_1147_, 11);
v_cancelTk_x3f_1160_ = lean_ctor_get(v___y_1147_, 12);
v_suppressElabErrors_1161_ = lean_ctor_get_uint8(v___y_1147_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1162_ = lean_ctor_get(v___y_1147_, 13);
v_isSharedCheck_1174_ = !lean_is_exclusive(v___y_1147_);
if (v_isSharedCheck_1174_ == 0)
{
lean_object* v_unused_1175_; lean_object* v_unused_1176_; 
v_unused_1175_ = lean_ctor_get(v___y_1147_, 4);
lean_dec(v_unused_1175_);
v_unused_1176_ = lean_ctor_get(v___y_1147_, 2);
lean_dec(v_unused_1176_);
v___x_1164_ = v___y_1147_;
v_isShared_1165_ = v_isSharedCheck_1174_;
goto v_resetjp_1163_;
}
else
{
lean_inc(v_inheritedTraceOptions_1162_);
lean_inc(v_cancelTk_x3f_1160_);
lean_inc(v_currMacroScope_1159_);
lean_inc(v_quotContext_1158_);
lean_inc(v_maxHeartbeats_1157_);
lean_inc(v_initHeartbeats_1156_);
lean_inc(v_openDecls_1155_);
lean_inc(v_currNamespace_1154_);
lean_inc(v_ref_1153_);
lean_inc(v_currRecDepth_1152_);
lean_inc(v_fileMap_1151_);
lean_inc(v_fileName_1150_);
lean_dec(v___y_1147_);
v___x_1164_ = lean_box(0);
v_isShared_1165_ = v_isSharedCheck_1174_;
goto v_resetjp_1163_;
}
v_resetjp_1163_:
{
lean_object* v_env_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1170_; 
v_env_1166_ = lean_ctor_get(v___x_1149_, 0);
lean_inc_ref(v_env_1166_);
lean_dec(v___x_1149_);
v___x_1167_ = l_Lean_maxRecDepth;
v___x_1168_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__3);
lean_inc_ref(v_inheritedTraceOptions_1162_);
lean_inc(v_cancelTk_x3f_1160_);
lean_inc(v_currMacroScope_1159_);
lean_inc(v_quotContext_1158_);
lean_inc(v_maxHeartbeats_1157_);
lean_inc(v_initHeartbeats_1156_);
lean_inc(v_openDecls_1155_);
lean_inc(v_currNamespace_1154_);
lean_inc(v_ref_1153_);
lean_inc(v_currRecDepth_1152_);
lean_inc_ref(v_fileMap_1151_);
lean_inc_ref(v_fileName_1150_);
if (v_isShared_1165_ == 0)
{
lean_ctor_set(v___x_1164_, 4, v___x_1168_);
lean_ctor_set(v___x_1164_, 2, v___x_1140_);
v___x_1170_ = v___x_1164_;
goto v_reusejp_1169_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v_fileName_1150_);
lean_ctor_set(v_reuseFailAlloc_1173_, 1, v_fileMap_1151_);
lean_ctor_set(v_reuseFailAlloc_1173_, 2, v___x_1140_);
lean_ctor_set(v_reuseFailAlloc_1173_, 3, v_currRecDepth_1152_);
lean_ctor_set(v_reuseFailAlloc_1173_, 4, v___x_1168_);
lean_ctor_set(v_reuseFailAlloc_1173_, 5, v_ref_1153_);
lean_ctor_set(v_reuseFailAlloc_1173_, 6, v_currNamespace_1154_);
lean_ctor_set(v_reuseFailAlloc_1173_, 7, v_openDecls_1155_);
lean_ctor_set(v_reuseFailAlloc_1173_, 8, v_initHeartbeats_1156_);
lean_ctor_set(v_reuseFailAlloc_1173_, 9, v_maxHeartbeats_1157_);
lean_ctor_set(v_reuseFailAlloc_1173_, 10, v_quotContext_1158_);
lean_ctor_set(v_reuseFailAlloc_1173_, 11, v_currMacroScope_1159_);
lean_ctor_set(v_reuseFailAlloc_1173_, 12, v_cancelTk_x3f_1160_);
lean_ctor_set(v_reuseFailAlloc_1173_, 13, v_inheritedTraceOptions_1162_);
lean_ctor_set_uint8(v_reuseFailAlloc_1173_, sizeof(void*)*14 + 1, v_suppressElabErrors_1161_);
v___x_1170_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1169_;
}
v_reusejp_1169_:
{
uint8_t v___x_1171_; uint8_t v___x_1172_; 
lean_ctor_set_uint8(v___x_1170_, sizeof(void*)*14, v___y_1146_);
v___x_1171_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_ContextInfo_runCoreMWithMessages_spec__0(v_options_988_, v___y_1142_);
v___x_1172_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_1166_);
lean_dec_ref(v_env_1166_);
if (v___x_1172_ == 0)
{
if (v___x_1171_ == 0)
{
lean_dec_ref(v___x_1170_);
v___y_1006_ = v___x_1171_;
v___y_1007_ = v___y_1145_;
v___y_1008_ = v___y_1144_;
v___y_1009_ = v___x_1167_;
v_fileName_1010_ = v_fileName_1150_;
v_fileMap_1011_ = v_fileMap_1151_;
v_currRecDepth_1012_ = v_currRecDepth_1152_;
v_ref_1013_ = v_ref_1153_;
v_currNamespace_1014_ = v_currNamespace_1154_;
v_openDecls_1015_ = v_openDecls_1155_;
v_initHeartbeats_1016_ = v_initHeartbeats_1156_;
v_maxHeartbeats_1017_ = v_maxHeartbeats_1157_;
v_quotContext_1018_ = v_quotContext_1158_;
v_currMacroScope_1019_ = v_currMacroScope_1159_;
v_cancelTk_x3f_1020_ = v_cancelTk_x3f_1160_;
v_suppressElabErrors_1021_ = v_suppressElabErrors_1161_;
v_inheritedTraceOptions_1022_ = v_inheritedTraceOptions_1162_;
v___y_1023_ = v___y_1148_;
goto v___jp_1005_;
}
else
{
lean_dec_ref(v_inheritedTraceOptions_1162_);
lean_dec(v_cancelTk_x3f_1160_);
lean_dec(v_currMacroScope_1159_);
lean_dec(v_quotContext_1158_);
lean_dec(v_maxHeartbeats_1157_);
lean_dec(v_initHeartbeats_1156_);
lean_dec(v_openDecls_1155_);
lean_dec(v_currNamespace_1154_);
lean_dec(v_ref_1153_);
lean_dec(v_currRecDepth_1152_);
lean_dec_ref(v_fileMap_1151_);
lean_dec_ref(v_fileName_1150_);
v___y_1111_ = v___x_1170_;
v___y_1112_ = v___y_1148_;
v___y_1113_ = v___x_1171_;
v___y_1114_ = v___y_1143_;
v___y_1115_ = v___y_1144_;
v___y_1116_ = v___y_1145_;
v___y_1117_ = v___x_1167_;
v___y_1118_ = v___x_1172_;
goto v___jp_1110_;
}
}
else
{
lean_dec_ref(v_inheritedTraceOptions_1162_);
lean_dec(v_cancelTk_x3f_1160_);
lean_dec(v_currMacroScope_1159_);
lean_dec(v_quotContext_1158_);
lean_dec(v_maxHeartbeats_1157_);
lean_dec(v_initHeartbeats_1156_);
lean_dec(v_openDecls_1155_);
lean_dec(v_currNamespace_1154_);
lean_dec(v_ref_1153_);
lean_dec(v_currRecDepth_1152_);
lean_dec_ref(v_fileMap_1151_);
lean_dec_ref(v_fileName_1150_);
v___y_1111_ = v___x_1170_;
v___y_1112_ = v___y_1148_;
v___y_1113_ = v___x_1171_;
v___y_1114_ = v___y_1143_;
v___y_1115_ = v___y_1144_;
v___y_1116_ = v___y_1145_;
v___y_1117_ = v___x_1167_;
v___y_1118_ = v___x_1171_;
goto v___jp_1110_;
}
}
}
}
v___jp_1177_:
{
if (v___y_1184_ == 0)
{
lean_object* v___x_1185_; lean_object* v_env_1186_; lean_object* v_nextMacroScope_1187_; lean_object* v_ngen_1188_; lean_object* v_auxDeclNGen_1189_; lean_object* v_traceState_1190_; lean_object* v_messages_1191_; lean_object* v_infoState_1192_; lean_object* v_snapshotTasks_1193_; lean_object* v___x_1195_; uint8_t v_isShared_1196_; uint8_t v_isSharedCheck_1202_; 
v___x_1185_ = lean_st_ref_take(v___y_1182_);
v_env_1186_ = lean_ctor_get(v___x_1185_, 0);
v_nextMacroScope_1187_ = lean_ctor_get(v___x_1185_, 1);
v_ngen_1188_ = lean_ctor_get(v___x_1185_, 2);
v_auxDeclNGen_1189_ = lean_ctor_get(v___x_1185_, 3);
v_traceState_1190_ = lean_ctor_get(v___x_1185_, 4);
v_messages_1191_ = lean_ctor_get(v___x_1185_, 6);
v_infoState_1192_ = lean_ctor_get(v___x_1185_, 7);
v_snapshotTasks_1193_ = lean_ctor_get(v___x_1185_, 8);
v_isSharedCheck_1202_ = !lean_is_exclusive(v___x_1185_);
if (v_isSharedCheck_1202_ == 0)
{
lean_object* v_unused_1203_; 
v_unused_1203_ = lean_ctor_get(v___x_1185_, 5);
lean_dec(v_unused_1203_);
v___x_1195_ = v___x_1185_;
v_isShared_1196_ = v_isSharedCheck_1202_;
goto v_resetjp_1194_;
}
else
{
lean_inc(v_snapshotTasks_1193_);
lean_inc(v_infoState_1192_);
lean_inc(v_messages_1191_);
lean_inc(v_traceState_1190_);
lean_inc(v_auxDeclNGen_1189_);
lean_inc(v_ngen_1188_);
lean_inc(v_nextMacroScope_1187_);
lean_inc(v_env_1186_);
lean_dec(v___x_1185_);
v___x_1195_ = lean_box(0);
v_isShared_1196_ = v_isSharedCheck_1202_;
goto v_resetjp_1194_;
}
v_resetjp_1194_:
{
lean_object* v___x_1197_; lean_object* v___x_1199_; 
v___x_1197_ = l_Lean_Kernel_enableDiag(v_env_1186_, v___y_1183_);
lean_inc_ref(v___y_1180_);
if (v_isShared_1196_ == 0)
{
lean_ctor_set(v___x_1195_, 5, v___y_1180_);
lean_ctor_set(v___x_1195_, 0, v___x_1197_);
v___x_1199_ = v___x_1195_;
goto v_reusejp_1198_;
}
else
{
lean_object* v_reuseFailAlloc_1201_; 
v_reuseFailAlloc_1201_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1201_, 0, v___x_1197_);
lean_ctor_set(v_reuseFailAlloc_1201_, 1, v_nextMacroScope_1187_);
lean_ctor_set(v_reuseFailAlloc_1201_, 2, v_ngen_1188_);
lean_ctor_set(v_reuseFailAlloc_1201_, 3, v_auxDeclNGen_1189_);
lean_ctor_set(v_reuseFailAlloc_1201_, 4, v_traceState_1190_);
lean_ctor_set(v_reuseFailAlloc_1201_, 5, v___y_1180_);
lean_ctor_set(v_reuseFailAlloc_1201_, 6, v_messages_1191_);
lean_ctor_set(v_reuseFailAlloc_1201_, 7, v_infoState_1192_);
lean_ctor_set(v_reuseFailAlloc_1201_, 8, v_snapshotTasks_1193_);
v___x_1199_ = v_reuseFailAlloc_1201_;
goto v_reusejp_1198_;
}
v_reusejp_1198_:
{
lean_object* v___x_1200_; 
v___x_1200_ = lean_st_ref_set(v___y_1182_, v___x_1199_);
lean_inc(v___y_1182_);
v___y_1142_ = v___y_1178_;
v___y_1143_ = v___y_1180_;
v___y_1144_ = v___y_1182_;
v___y_1145_ = v___y_1181_;
v___y_1146_ = v___y_1183_;
v___y_1147_ = v___y_1179_;
v___y_1148_ = v___y_1182_;
goto v___jp_1141_;
}
}
}
else
{
lean_inc(v___y_1182_);
v___y_1142_ = v___y_1178_;
v___y_1143_ = v___y_1180_;
v___y_1144_ = v___y_1182_;
v___y_1145_ = v___y_1181_;
v___y_1146_ = v___y_1183_;
v___y_1147_ = v___y_1179_;
v___y_1148_ = v___y_1182_;
goto v___jp_1141_;
}
}
v___jp_1213_:
{
lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1222_; 
v___x_1215_ = lean_unsigned_to_nat(32u);
v___x_1216_ = lean_mk_empty_array_with_capacity(v___x_1215_);
lean_dec_ref(v___x_1216_);
v___x_1217_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__10);
v___x_1218_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__11);
v___x_1219_ = lean_io_get_num_heartbeats();
v___x_1220_ = lean_box(0);
if (v_isShared_986_ == 0)
{
lean_ctor_set(v___x_985_, 2, v___x_1220_);
lean_ctor_set(v___x_985_, 1, v___x_1211_);
lean_ctor_set(v___x_985_, 0, v___y_1214_);
v___x_1222_ = v___x_985_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1237_; 
v_reuseFailAlloc_1237_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1237_, 0, v___y_1214_);
lean_ctor_set(v_reuseFailAlloc_1237_, 1, v___x_1211_);
lean_ctor_set(v_reuseFailAlloc_1237_, 2, v___x_1220_);
v___x_1222_ = v_reuseFailAlloc_1237_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
lean_object* v___x_1223_; uint8_t v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v_env_1233_; lean_object* v___x_1234_; uint8_t v___x_1235_; uint8_t v___x_1236_; 
v___x_1223_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__12);
v___x_1224_ = 1;
v___x_1225_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__13);
v___x_1226_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__14));
v___x_1227_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v___x_1227_, 0, v_env_1139_);
lean_ctor_set(v___x_1227_, 1, v___x_1212_);
lean_ctor_set(v___x_1227_, 2, v_ngen_991_);
lean_ctor_set(v___x_1227_, 3, v___x_1222_);
lean_ctor_set(v___x_1227_, 4, v___x_1223_);
lean_ctor_set(v___x_1227_, 5, v___x_1217_);
lean_ctor_set(v___x_1227_, 6, v___x_1218_);
lean_ctor_set(v___x_1227_, 7, v___x_1225_);
lean_ctor_set(v___x_1227_, 8, v___x_1226_);
v___x_1228_ = lean_st_mk_ref(v___x_1227_);
v___x_1229_ = l_Lean_inheritedTraceOptions;
v___x_1230_ = lean_st_ref_get(v___x_1229_);
v___x_1231_ = lean_st_ref_get(v___x_1228_);
lean_inc_ref(v_fileMap_993_);
lean_inc_ref(v_fileName_992_);
v___x_1232_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1232_, 0, v_fileName_992_);
lean_ctor_set(v___x_1232_, 1, v_fileMap_993_);
lean_ctor_set(v___x_1232_, 2, v___x_1140_);
lean_ctor_set(v___x_1232_, 3, v___x_1204_);
lean_ctor_set(v___x_1232_, 4, v___x_1205_);
lean_ctor_set(v___x_1232_, 5, v___x_1206_);
lean_ctor_set(v___x_1232_, 6, v_currNamespace_989_);
lean_ctor_set(v___x_1232_, 7, v_openDecls_990_);
lean_ctor_set(v___x_1232_, 8, v___x_1219_);
lean_ctor_set(v___x_1232_, 9, v___x_1207_);
lean_ctor_set(v___x_1232_, 10, v___x_1208_);
lean_ctor_set(v___x_1232_, 11, v___x_1209_);
lean_ctor_set(v___x_1232_, 12, v___x_1210_);
lean_ctor_set(v___x_1232_, 13, v___x_1230_);
lean_ctor_set_uint8(v___x_1232_, sizeof(void*)*14, v___x_1138_);
lean_ctor_set_uint8(v___x_1232_, sizeof(void*)*14 + 1, v___x_1138_);
v_env_1233_ = lean_ctor_get(v___x_1231_, 0);
lean_inc_ref(v_env_1233_);
lean_dec(v___x_1231_);
v___x_1234_ = l_Lean_diagnostics;
v___x_1235_ = lean_uint8_once(&lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15, &lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runCoreMWithMessages___redArg___closed__15);
v___x_1236_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_1233_);
lean_dec_ref(v_env_1233_);
if (v___x_1236_ == 0)
{
if (v___x_1235_ == 0)
{
lean_inc(v___x_1228_);
v___y_1142_ = v___x_1234_;
v___y_1143_ = v___x_1217_;
v___y_1144_ = v___x_1228_;
v___y_1145_ = v___x_1224_;
v___y_1146_ = v___x_1235_;
v___y_1147_ = v___x_1232_;
v___y_1148_ = v___x_1228_;
goto v___jp_1141_;
}
else
{
v___y_1178_ = v___x_1234_;
v___y_1179_ = v___x_1232_;
v___y_1180_ = v___x_1217_;
v___y_1181_ = v___x_1224_;
v___y_1182_ = v___x_1228_;
v___y_1183_ = v___x_1235_;
v___y_1184_ = v___x_1236_;
goto v___jp_1177_;
}
}
else
{
v___y_1178_ = v___x_1234_;
v___y_1179_ = v___x_1232_;
v___y_1180_ = v___x_1217_;
v___y_1181_ = v___x_1224_;
v___y_1182_ = v___x_1228_;
v___y_1183_ = v___x_1235_;
v___y_1184_ = v___x_1235_;
goto v___jp_1177_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___redArg___boxed(lean_object* v_info_1241_, lean_object* v_x_1242_, lean_object* v_a_1243_, lean_object* v_a_1244_, lean_object* v_a_1245_){
_start:
{
lean_object* v_res_1246_; 
v_res_1246_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___redArg(v_info_1241_, v_x_1242_, v_a_1243_, v_a_1244_);
lean_dec(v_a_1244_);
lean_dec_ref(v_a_1243_);
return v_res_1246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree(lean_object* v_00_u03b1_1247_, lean_object* v_info_1248_, lean_object* v_x_1249_, lean_object* v_a_1250_, lean_object* v_a_1251_){
_start:
{
lean_object* v___x_1253_; 
v___x_1253_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___redArg(v_info_1248_, v_x_1249_, v_a_1250_, v_a_1251_);
return v___x_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___boxed(lean_object* v_00_u03b1_1254_, lean_object* v_info_1255_, lean_object* v_x_1256_, lean_object* v_a_1257_, lean_object* v_a_1258_, lean_object* v_a_1259_){
_start:
{
lean_object* v_res_1260_; 
v_res_1260_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree(v_00_u03b1_1254_, v_info_1255_, v_x_1256_, v_a_1257_, v_a_1258_);
lean_dec(v_a_1258_);
lean_dec_ref(v_a_1257_);
return v_res_1260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___redArg(lean_object* v_info_1261_, lean_object* v_lctx_1262_, lean_object* v_x_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_){
_start:
{
lean_object* v_decls_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; uint8_t v___x_1271_; uint8_t v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v_toCommandContextInfo_1276_; lean_object* v_mctx_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___f_1286_; lean_object* v___x_1287_; 
v_decls_1267_ = lean_ctor_get(v_lctx_1262_, 1);
lean_inc_ref(v_decls_1267_);
v___x_1268_ = lean_box(1);
v___x_1269_ = lean_unsigned_to_nat(0u);
v___x_1270_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__0));
v___x_1271_ = 0;
v___x_1272_ = 1;
v___x_1273_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__3);
v___x_1274_ = lean_box(0);
v___x_1275_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1275_, 0, v___x_1273_);
lean_ctor_set(v___x_1275_, 1, v___x_1268_);
lean_ctor_set(v___x_1275_, 2, v_lctx_1262_);
lean_ctor_set(v___x_1275_, 3, v___x_1270_);
lean_ctor_set(v___x_1275_, 4, v___x_1274_);
lean_ctor_set(v___x_1275_, 5, v___x_1269_);
lean_ctor_set(v___x_1275_, 6, v___x_1274_);
lean_ctor_set_uint8(v___x_1275_, sizeof(void*)*7, v___x_1271_);
lean_ctor_set_uint8(v___x_1275_, sizeof(void*)*7 + 1, v___x_1271_);
lean_ctor_set_uint8(v___x_1275_, sizeof(void*)*7 + 2, v___x_1271_);
lean_ctor_set_uint8(v___x_1275_, sizeof(void*)*7 + 3, v___x_1272_);
v_toCommandContextInfo_1276_ = lean_ctor_get(v_info_1261_, 0);
v_mctx_1277_ = lean_ctor_get(v_toCommandContextInfo_1276_, 3);
v___x_1278_ = l_Lean_PersistentArray_toList___redArg(v_decls_1267_);
lean_dec_ref(v_decls_1267_);
v___x_1279_ = lp_mathlib_List_filterMapTR_go___at___00Lean_Elab_ContextInfo_runMetaMWithMessages_spec__0(v___x_1278_, v___x_1270_);
v___x_1280_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__6);
v___x_1281_ = lean_unsigned_to_nat(32u);
v___x_1282_ = lean_mk_empty_array_with_capacity(v___x_1281_);
lean_dec_ref(v___x_1282_);
v___x_1283_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__8);
v___x_1284_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9, &lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___closed__9);
lean_inc_ref(v_mctx_1277_);
v___x_1285_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1285_, 0, v_mctx_1277_);
lean_ctor_set(v___x_1285_, 1, v___x_1280_);
lean_ctor_set(v___x_1285_, 2, v___x_1268_);
lean_ctor_set(v___x_1285_, 3, v___x_1283_);
lean_ctor_set(v___x_1285_, 4, v___x_1284_);
v___f_1286_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_ContextInfo_runMetaMWithMessages___redArg___lam__0___boxed), 7, 4);
lean_closure_set(v___f_1286_, 0, v___x_1285_);
lean_closure_set(v___f_1286_, 1, v___x_1279_);
lean_closure_set(v___f_1286_, 2, v_x_1263_);
lean_closure_set(v___f_1286_, 3, v___x_1275_);
v___x_1287_ = lp_mathlib_Lean_Elab_ContextInfo_runCoreMCapturingInfoTree___redArg(v_info_1261_, v___f_1286_, v_a_1264_, v_a_1265_);
if (lean_obj_tag(v___x_1287_) == 0)
{
lean_object* v_a_1288_; lean_object* v___x_1290_; uint8_t v_isShared_1291_; uint8_t v_isSharedCheck_1306_; 
v_a_1288_ = lean_ctor_get(v___x_1287_, 0);
v_isSharedCheck_1306_ = !lean_is_exclusive(v___x_1287_);
if (v_isSharedCheck_1306_ == 0)
{
v___x_1290_ = v___x_1287_;
v_isShared_1291_ = v_isSharedCheck_1306_;
goto v_resetjp_1289_;
}
else
{
lean_inc(v_a_1288_);
lean_dec(v___x_1287_);
v___x_1290_ = lean_box(0);
v_isShared_1291_ = v_isSharedCheck_1306_;
goto v_resetjp_1289_;
}
v_resetjp_1289_:
{
lean_object* v_fst_1292_; lean_object* v_snd_1293_; lean_object* v_fst_1294_; lean_object* v___x_1296_; uint8_t v_isShared_1297_; uint8_t v_isSharedCheck_1304_; 
v_fst_1292_ = lean_ctor_get(v_a_1288_, 0);
lean_inc(v_fst_1292_);
v_snd_1293_ = lean_ctor_get(v_a_1288_, 1);
lean_inc(v_snd_1293_);
lean_dec(v_a_1288_);
v_fst_1294_ = lean_ctor_get(v_fst_1292_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v_fst_1292_);
if (v_isSharedCheck_1304_ == 0)
{
lean_object* v_unused_1305_; 
v_unused_1305_ = lean_ctor_get(v_fst_1292_, 1);
lean_dec(v_unused_1305_);
v___x_1296_ = v_fst_1292_;
v_isShared_1297_ = v_isSharedCheck_1304_;
goto v_resetjp_1295_;
}
else
{
lean_inc(v_fst_1294_);
lean_dec(v_fst_1292_);
v___x_1296_ = lean_box(0);
v_isShared_1297_ = v_isSharedCheck_1304_;
goto v_resetjp_1295_;
}
v_resetjp_1295_:
{
lean_object* v___x_1299_; 
if (v_isShared_1297_ == 0)
{
lean_ctor_set(v___x_1296_, 1, v_snd_1293_);
v___x_1299_ = v___x_1296_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_fst_1294_);
lean_ctor_set(v_reuseFailAlloc_1303_, 1, v_snd_1293_);
v___x_1299_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
lean_object* v___x_1301_; 
if (v_isShared_1291_ == 0)
{
lean_ctor_set(v___x_1290_, 0, v___x_1299_);
v___x_1301_ = v___x_1290_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v___x_1299_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
}
}
}
else
{
lean_object* v_a_1307_; lean_object* v___x_1309_; uint8_t v_isShared_1310_; uint8_t v_isSharedCheck_1314_; 
v_a_1307_ = lean_ctor_get(v___x_1287_, 0);
v_isSharedCheck_1314_ = !lean_is_exclusive(v___x_1287_);
if (v_isSharedCheck_1314_ == 0)
{
v___x_1309_ = v___x_1287_;
v_isShared_1310_ = v_isSharedCheck_1314_;
goto v_resetjp_1308_;
}
else
{
lean_inc(v_a_1307_);
lean_dec(v___x_1287_);
v___x_1309_ = lean_box(0);
v_isShared_1310_ = v_isSharedCheck_1314_;
goto v_resetjp_1308_;
}
v_resetjp_1308_:
{
lean_object* v___x_1312_; 
if (v_isShared_1310_ == 0)
{
v___x_1312_ = v___x_1309_;
goto v_reusejp_1311_;
}
else
{
lean_object* v_reuseFailAlloc_1313_; 
v_reuseFailAlloc_1313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1313_, 0, v_a_1307_);
v___x_1312_ = v_reuseFailAlloc_1313_;
goto v_reusejp_1311_;
}
v_reusejp_1311_:
{
return v___x_1312_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___redArg___boxed(lean_object* v_info_1315_, lean_object* v_lctx_1316_, lean_object* v_x_1317_, lean_object* v_a_1318_, lean_object* v_a_1319_, lean_object* v_a_1320_){
_start:
{
lean_object* v_res_1321_; 
v_res_1321_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___redArg(v_info_1315_, v_lctx_1316_, v_x_1317_, v_a_1318_, v_a_1319_);
lean_dec(v_a_1319_);
lean_dec_ref(v_a_1318_);
return v_res_1321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree(lean_object* v_00_u03b1_1322_, lean_object* v_info_1323_, lean_object* v_lctx_1324_, lean_object* v_x_1325_, lean_object* v_a_1326_, lean_object* v_a_1327_){
_start:
{
lean_object* v___x_1329_; 
v___x_1329_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___redArg(v_info_1323_, v_lctx_1324_, v_x_1325_, v_a_1326_, v_a_1327_);
return v___x_1329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___boxed(lean_object* v_00_u03b1_1330_, lean_object* v_info_1331_, lean_object* v_lctx_1332_, lean_object* v_x_1333_, lean_object* v_a_1334_, lean_object* v_a_1335_, lean_object* v_a_1336_){
_start:
{
lean_object* v_res_1337_; 
v_res_1337_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree(v_00_u03b1_1330_, v_info_1331_, v_lctx_1332_, v_x_1333_, v_a_1334_, v_a_1335_);
lean_dec(v_a_1335_);
lean_dec_ref(v_a_1334_);
return v_res_1337_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__2(void){
_start:
{
lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; 
v___x_1340_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__1));
v___x_1341_ = lean_unsigned_to_nat(4u);
v___x_1342_ = lean_unsigned_to_nat(132u);
v___x_1343_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__0));
v___x_1344_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__4));
v___x_1345_ = l_mkPanicMessageWithDecl(v___x_1344_, v___x_1343_, v___x_1342_, v___x_1341_, v___x_1340_);
return v___x_1345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg(lean_object* v_ctx_1346_, lean_object* v_i_1347_, lean_object* v_goal_1348_, lean_object* v_x_1349_, lean_object* v_a_1350_, lean_object* v_a_1351_){
_start:
{
lean_object* v_mctxBefore_1353_; lean_object* v_goalsBefore_1354_; lean_object* v___f_1355_; lean_object* v___y_1357_; lean_object* v___y_1358_; lean_object* v___y_1359_; lean_object* v___y_1363_; lean_object* v___y_1364_; uint8_t v___x_1370_; 
v_mctxBefore_1353_ = lean_ctor_get(v_i_1347_, 1);
v_goalsBefore_1354_ = lean_ctor_get(v_i_1347_, 2);
lean_inc(v_goal_1348_);
v___f_1355_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1355_, 0, v_goal_1348_);
lean_closure_set(v___f_1355_, 1, v_x_1349_);
v___x_1370_ = lp_mathlib_List_elem___at___00Lean_Elab_ContextInfo_runTactic_spec__2(v_goal_1348_, v_goalsBefore_1354_);
if (v___x_1370_ == 0)
{
lean_object* v___x_1371_; lean_object* v___x_1372_; 
v___x_1371_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__2, &lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___closed__2);
v___x_1372_ = lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__3(v___x_1371_, v_a_1350_, v_a_1351_);
if (lean_obj_tag(v___x_1372_) == 0)
{
lean_dec_ref_known(v___x_1372_, 1);
v___y_1363_ = v_a_1350_;
v___y_1364_ = v_a_1351_;
goto v___jp_1362_;
}
else
{
lean_object* v_a_1373_; lean_object* v___x_1375_; uint8_t v_isShared_1376_; uint8_t v_isSharedCheck_1380_; 
lean_dec_ref(v___f_1355_);
lean_dec(v_goal_1348_);
lean_dec_ref(v_ctx_1346_);
v_a_1373_ = lean_ctor_get(v___x_1372_, 0);
v_isSharedCheck_1380_ = !lean_is_exclusive(v___x_1372_);
if (v_isSharedCheck_1380_ == 0)
{
v___x_1375_ = v___x_1372_;
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
else
{
lean_inc(v_a_1373_);
lean_dec(v___x_1372_);
v___x_1375_ = lean_box(0);
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
v_resetjp_1374_:
{
lean_object* v___x_1378_; 
if (v_isShared_1376_ == 0)
{
v___x_1378_ = v___x_1375_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v_a_1373_);
v___x_1378_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
return v___x_1378_;
}
}
}
}
else
{
v___y_1363_ = v_a_1350_;
v___y_1364_ = v_a_1351_;
goto v___jp_1362_;
}
v___jp_1356_:
{
lean_object* v_lctx_1360_; lean_object* v___x_1361_; 
v_lctx_1360_ = lean_ctor_get(v___y_1359_, 1);
lean_inc_ref(v_lctx_1360_);
lean_dec_ref(v___y_1359_);
v___x_1361_ = lp_mathlib_Lean_Elab_ContextInfo_runMetaMCapturingInfoTree___redArg(v_ctx_1346_, v_lctx_1360_, v___f_1355_, v___y_1358_, v___y_1357_);
return v___x_1361_;
}
v___jp_1362_:
{
lean_object* v_decls_1365_; lean_object* v___x_1366_; 
v_decls_1365_ = lean_ctor_get(v_mctxBefore_1353_, 5);
v___x_1366_ = lp_mathlib_Lean_PersistentHashMap_find_x3f___at___00Lean_Elab_ContextInfo_runTactic_spec__0___redArg(v_decls_1365_, v_goal_1348_);
lean_dec(v_goal_1348_);
if (lean_obj_tag(v___x_1366_) == 0)
{
lean_object* v___x_1367_; lean_object* v___x_1368_; 
v___x_1367_ = lean_obj_once(&lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3, &lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_ContextInfo_runTactic___redArg___closed__3);
v___x_1368_ = lp_mathlib_panic___at___00Lean_Elab_ContextInfo_runTactic_spec__1(v___x_1367_);
v___y_1357_ = v___y_1364_;
v___y_1358_ = v___y_1363_;
v___y_1359_ = v___x_1368_;
goto v___jp_1356_;
}
else
{
lean_object* v_val_1369_; 
v_val_1369_ = lean_ctor_get(v___x_1366_, 0);
lean_inc(v_val_1369_);
lean_dec_ref_known(v___x_1366_, 1);
v___y_1357_ = v___y_1364_;
v___y_1358_ = v___y_1363_;
v___y_1359_ = v_val_1369_;
goto v___jp_1356_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg___boxed(lean_object* v_ctx_1381_, lean_object* v_i_1382_, lean_object* v_goal_1383_, lean_object* v_x_1384_, lean_object* v_a_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_){
_start:
{
lean_object* v_res_1388_; 
v_res_1388_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg(v_ctx_1381_, v_i_1382_, v_goal_1383_, v_x_1384_, v_a_1385_, v_a_1386_);
lean_dec(v_a_1386_);
lean_dec_ref(v_a_1385_);
lean_dec_ref(v_i_1382_);
return v_res_1388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree(lean_object* v_00_u03b1_1389_, lean_object* v_ctx_1390_, lean_object* v_i_1391_, lean_object* v_goal_1392_, lean_object* v_x_1393_, lean_object* v_a_1394_, lean_object* v_a_1395_){
_start:
{
lean_object* v___x_1397_; 
v___x_1397_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg(v_ctx_1390_, v_i_1391_, v_goal_1392_, v_x_1393_, v_a_1394_, v_a_1395_);
return v___x_1397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___boxed(lean_object* v_00_u03b1_1398_, lean_object* v_ctx_1399_, lean_object* v_i_1400_, lean_object* v_goal_1401_, lean_object* v_x_1402_, lean_object* v_a_1403_, lean_object* v_a_1404_, lean_object* v_a_1405_){
_start:
{
lean_object* v_res_1406_; 
v_res_1406_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree(v_00_u03b1_1398_, v_ctx_1399_, v_i_1400_, v_goal_1401_, v_x_1402_, v_a_1403_, v_a_1404_);
lean_dec(v_a_1404_);
lean_dec_ref(v_a_1403_);
lean_dec_ref(v_i_1400_);
return v_res_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___lam__2(lean_object* v_code_1407_, lean_object* v_a_1408_, lean_object* v_a_1409_, lean_object* v_goal_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_){
_start:
{
lean_object* v___x_1416_; 
v___x_1416_ = lp_mathlib_Lean_Elab_runTactic_x27(v_goal_1410_, v_code_1407_, v_a_1408_, v_a_1409_, v___y_1411_, v___y_1412_, v___y_1413_, v___y_1414_);
return v___x_1416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___lam__2___boxed(lean_object* v_code_1417_, lean_object* v_a_1418_, lean_object* v_a_1419_, lean_object* v_goal_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_){
_start:
{
lean_object* v_res_1426_; 
v_res_1426_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___lam__2(v_code_1417_, v_a_1418_, v_a_1419_, v_goal_1420_, v___y_1421_, v___y_1422_, v___y_1423_, v___y_1424_);
lean_dec(v___y_1424_);
lean_dec_ref(v___y_1423_);
lean_dec(v___y_1422_);
lean_dec_ref(v___y_1421_);
return v_res_1426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree(lean_object* v_ctx_1427_, lean_object* v_i_1428_, lean_object* v_goal_1429_, lean_object* v_code_1430_, lean_object* v_a_1431_, lean_object* v_a_1432_){
_start:
{
lean_object* v___f_1434_; lean_object* v___x_1435_; 
v___f_1434_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__0));
v___x_1435_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_1434_, v_a_1431_, v_a_1432_);
if (lean_obj_tag(v___x_1435_) == 0)
{
lean_object* v_a_1436_; lean_object* v___f_1437_; lean_object* v___x_1438_; 
v_a_1436_ = lean_ctor_get(v___x_1435_, 0);
lean_inc(v_a_1436_);
lean_dec_ref_known(v___x_1435_, 1);
v___f_1437_ = ((lean_object*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCode___closed__1));
v___x_1438_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_1437_, v_a_1431_, v_a_1432_);
if (lean_obj_tag(v___x_1438_) == 0)
{
lean_object* v_a_1439_; lean_object* v___f_1440_; lean_object* v___x_1441_; 
v_a_1439_ = lean_ctor_get(v___x_1438_, 0);
lean_inc(v_a_1439_);
lean_dec_ref_known(v___x_1438_, 1);
v___f_1440_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___lam__2___boxed), 9, 3);
lean_closure_set(v___f_1440_, 0, v_code_1430_);
lean_closure_set(v___f_1440_, 1, v_a_1436_);
lean_closure_set(v___f_1440_, 2, v_a_1439_);
v___x_1441_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCapturingInfoTree___redArg(v_ctx_1427_, v_i_1428_, v_goal_1429_, v___f_1440_, v_a_1431_, v_a_1432_);
return v___x_1441_;
}
else
{
lean_object* v_a_1442_; lean_object* v___x_1444_; uint8_t v_isShared_1445_; uint8_t v_isSharedCheck_1449_; 
lean_dec(v_a_1436_);
lean_dec(v_code_1430_);
lean_dec(v_goal_1429_);
lean_dec_ref(v_ctx_1427_);
v_a_1442_ = lean_ctor_get(v___x_1438_, 0);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___x_1438_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1444_ = v___x_1438_;
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
else
{
lean_inc(v_a_1442_);
lean_dec(v___x_1438_);
v___x_1444_ = lean_box(0);
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
v_resetjp_1443_:
{
lean_object* v___x_1447_; 
if (v_isShared_1445_ == 0)
{
v___x_1447_ = v___x_1444_;
goto v_reusejp_1446_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v_a_1442_);
v___x_1447_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1446_;
}
v_reusejp_1446_:
{
return v___x_1447_;
}
}
}
}
else
{
lean_object* v_a_1450_; lean_object* v___x_1452_; uint8_t v_isShared_1453_; uint8_t v_isSharedCheck_1457_; 
lean_dec(v_code_1430_);
lean_dec(v_goal_1429_);
lean_dec_ref(v_ctx_1427_);
v_a_1450_ = lean_ctor_get(v___x_1435_, 0);
v_isSharedCheck_1457_ = !lean_is_exclusive(v___x_1435_);
if (v_isSharedCheck_1457_ == 0)
{
v___x_1452_ = v___x_1435_;
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
else
{
lean_inc(v_a_1450_);
lean_dec(v___x_1435_);
v___x_1452_ = lean_box(0);
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
v_resetjp_1451_:
{
lean_object* v___x_1455_; 
if (v_isShared_1453_ == 0)
{
v___x_1455_ = v___x_1452_;
goto v_reusejp_1454_;
}
else
{
lean_object* v_reuseFailAlloc_1456_; 
v_reuseFailAlloc_1456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1456_, 0, v_a_1450_);
v___x_1455_ = v_reuseFailAlloc_1456_;
goto v_reusejp_1454_;
}
v_reusejp_1454_:
{
return v___x_1455_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree___boxed(lean_object* v_ctx_1458_, lean_object* v_i_1459_, lean_object* v_goal_1460_, lean_object* v_code_1461_, lean_object* v_a_1462_, lean_object* v_a_1463_, lean_object* v_a_1464_){
_start:
{
lean_object* v_res_1465_; 
v_res_1465_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree(v_ctx_1458_, v_i_1459_, v_goal_1460_, v_code_1461_, v_a_1462_, v_a_1463_);
lean_dec(v_a_1463_);
lean_dec_ref(v_a_1462_);
lean_dec_ref(v_i_1459_);
return v_res_1465_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_ContextInfo(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_ContextInfo(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_ContextInfo(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_ContextInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_ContextInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_ContextInfo(builtin);
}
#ifdef __cplusplus
}
#endif
