// Lean compiler output
// Module: Aesop.Tree.AddRapp
// Imports: public import Init public meta import Init public import Aesop.Tree.TreeM import Batteries.Lean.Meta.SavedState import Aesop.Forward.State.ApplyGoalDiff import Aesop.Tree.Traversal import Aesop.Util.UnionFind
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
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint64_t lean_usize_to_uint64(size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lp_aesop_Aesop_Goal_currentGoal(lean_object*);
lean_object* lp_aesop_Aesop_diffGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_st_ref_take(lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lp_aesop_Aesop_Goal_originalGoalId(lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getRootGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Subgoal_mvarId(lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardState_applyGoalDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatches_update(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getAndIncrementNextGoalId___redArg(lean_object*);
extern lean_object* lp_aesop_Aesop_Iteration_none;
lean_object* l_Subarray_empty(lean_object*);
lean_object* lp_aesop_Aesop_Subgoal_mvarId___boxed(lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedGoalDiff_default;
lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lp_aesop_Aesop_RegularRule_name(lean_object*);
lean_object* lp_aesop_Aesop_RegularRule_tac(lean_object*);
lean_object* lp_aesop_Aesop_RuleTacDescr_forwardRuleMatches_x3f(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_aesop_Aesop_incrementNumGoals___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_incrementNumRapps___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getAndIncrementNextRappId___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_DeclNameGenerator_mkChild(lean_object*);
lean_object* lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(lean_object*);
static const lean_array_object lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___closed__0 = (const lean_object*)&lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_findPathForAssignedMVars_spec__9(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_findPathForAssignedMVars_spec__9___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12_spec__16(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_findPathForAssignedMVars___lam__0(lean_object*, uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_findPathForAssignedMVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_findPathForAssignedMVars_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_findPathForAssignedMVars_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_findPathForAssignedMVars_spec__6(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10_spec__15(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11_spec__17(lean_object*, uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11(lean_object*, uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7_spec__18(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7_spec__18___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4_spec__15___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_findPathForAssignedMVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_findPathForAssignedMVars___closed__0 = (const lean_object*)&lp_aesop_Aesop_findPathForAssignedMVars___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_findPathForAssignedMVars___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_findPathForAssignedMVars___closed__1;
static lean_once_cell_t lp_aesop_Aesop_findPathForAssignedMVars___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_findPathForAssignedMVars___closed__2;
static const lean_string_object lp_aesop_Aesop_findPathForAssignedMVars___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "aesop: internal error: introducing rapps not found for these mvars: "};
static const lean_object* lp_aesop_Aesop_findPathForAssignedMVars___closed__3 = (const lean_object*)&lp_aesop_Aesop_findPathForAssignedMVars___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_findPathForAssignedMVars___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_findPathForAssignedMVars___closed__4;
static lean_once_cell_t lp_aesop_Aesop_findPathForAssignedMVars___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_findPathForAssignedMVars___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_findPathForAssignedMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_findPathForAssignedMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4_spec__15(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoalsToCopy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoalsToCopy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2(lean_object*, lean_object*, lean_object*, double, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_copyGoals(lean_object*, lean_object*, lean_object*, double, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_copyGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, double, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal___boxed(lean_object**);
static const lean_array_object lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_addRappUnsafe_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_addRappUnsafe_spec__3___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_addRappUnsafe_spec__3___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_addRappUnsafe_spec__3 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_addRappUnsafe_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20_spec__31___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__10(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_insert___at___00Aesop_addRappUnsafe_spec__8(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__9(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__1(lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__15(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__22(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__11(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__11___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41(size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___redArg(size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg(size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58_spec__64___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___redArg(size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___redArg(lean_object*, size_t);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__44(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__45(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__45___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__39(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__40(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__40___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0;
static lean_once_cell_t lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27_spec__32___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34_spec__42___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__28(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__28___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__29(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__29___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__31(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__31___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63_spec__65___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__58___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48___redArg(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(1ULL)}};
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43___boxed__const__1 = (const lean_object*)&lp_aesop_Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__44(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__44___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36___boxed(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__1;
static lean_once_cell_t lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__2;
static lean_once_cell_t lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___boxed(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__0;
static lean_once_cell_t lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__2_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__3_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__4_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__5_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__6_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__8_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__10_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__11_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__12 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__12_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "in rapp for rule "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__13 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__13_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__14;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__15 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__15_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__16 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__16_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__17 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__17_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, double, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__21(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__20(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_addRappUnsafe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_addRappUnsafe___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_addRappUnsafe___closed__0 = (const lean_object*)&lp_aesop_Aesop_addRappUnsafe___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_addRappUnsafe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Subgoal_mvarId___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_addRappUnsafe___closed__1 = (const lean_object*)&lp_aesop_Aesop_addRappUnsafe___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_addRappUnsafe___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_aesop_Aesop_addRappUnsafe___boxed__const__1 = (const lean_object*)&lp_aesop_Aesop_addRappUnsafe___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27_spec__32(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42(lean_object*, lean_object*, size_t);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20_spec__31(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55(lean_object*, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34_spec__42(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__58(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58_spec__64(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63_spec__65(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches(lean_object* v_r_3_){
_start:
{
lean_object* v_appliedRule_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v_appliedRule_4_ = lean_ctor_get(v_r_3_, 2);
v___x_5_ = lp_aesop_Aesop_RegularRule_tac(v_appliedRule_4_);
v___x_6_ = lp_aesop_Aesop_RuleTacDescr_forwardRuleMatches_x3f(v___x_5_);
if (lean_obj_tag(v___x_6_) == 0)
{
lean_object* v___x_7_; 
v___x_7_ = ((lean_object*)(lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___closed__0));
return v___x_7_;
}
else
{
lean_object* v_val_8_; 
v_val_8_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_val_8_);
lean_dec_ref_known(v___x_6_, 1);
return v_val_8_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___boxed(lean_object* v_r_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches(v_r_9_);
lean_dec_ref(v_r_9_);
return v_res_10_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0(lean_object* v_s_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; uint8_t v___x_14_; 
v___x_12_ = lean_array_get_size(v_s_11_);
v___x_13_ = lean_unsigned_to_nat(0u);
v___x_14_ = lean_nat_dec_eq(v___x_12_, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0___boxed(lean_object* v_s_15_){
_start:
{
uint8_t v_res_16_; lean_object* v_r_17_; 
v_res_16_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0(v_s_15_);
lean_dec_ref(v_s_15_);
v_r_17_ = lean_box(v_res_16_);
return v_r_17_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_findPathForAssignedMVars_spec__9(lean_object* v_s_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_array_get_size(v_s_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_findPathForAssignedMVars_spec__9___boxed(lean_object* v_s_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_aesop_Aesop_UnorderedArraySet_size___at___00Aesop_findPathForAssignedMVars_spec__9(v_s_20_);
lean_dec_ref(v_s_20_);
return v_res_21_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12_spec__16(lean_object* v_a_22_, lean_object* v_as_23_, size_t v_i_24_, size_t v_stop_25_){
_start:
{
uint8_t v___x_26_; 
v___x_26_ = lean_usize_dec_eq(v_i_24_, v_stop_25_);
if (v___x_26_ == 0)
{
lean_object* v___x_27_; uint8_t v___x_28_; 
v___x_27_ = lean_array_uget_borrowed(v_as_23_, v_i_24_);
v___x_28_ = l_Lean_instBEqMVarId_beq(v_a_22_, v___x_27_);
if (v___x_28_ == 0)
{
size_t v___x_29_; size_t v___x_30_; 
v___x_29_ = ((size_t)1ULL);
v___x_30_ = lean_usize_add(v_i_24_, v___x_29_);
v_i_24_ = v___x_30_;
goto _start;
}
else
{
return v___x_28_;
}
}
else
{
uint8_t v___x_32_; 
v___x_32_ = 0;
return v___x_32_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12_spec__16___boxed(lean_object* v_a_33_, lean_object* v_as_34_, lean_object* v_i_35_, lean_object* v_stop_36_){
_start:
{
size_t v_i_boxed_37_; size_t v_stop_boxed_38_; uint8_t v_res_39_; lean_object* v_r_40_; 
v_i_boxed_37_ = lean_unbox_usize(v_i_35_);
lean_dec(v_i_35_);
v_stop_boxed_38_ = lean_unbox_usize(v_stop_36_);
lean_dec(v_stop_36_);
v_res_39_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12_spec__16(v_a_33_, v_as_34_, v_i_boxed_37_, v_stop_boxed_38_);
lean_dec_ref(v_as_34_);
lean_dec(v_a_33_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(lean_object* v_as_41_, lean_object* v_a_42_){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; uint8_t v___x_45_; 
v___x_43_ = lean_unsigned_to_nat(0u);
v___x_44_ = lean_array_get_size(v_as_41_);
v___x_45_ = lean_nat_dec_lt(v___x_43_, v___x_44_);
if (v___x_45_ == 0)
{
return v___x_45_;
}
else
{
if (v___x_45_ == 0)
{
return v___x_45_;
}
else
{
size_t v___x_46_; size_t v___x_47_; uint8_t v___x_48_; 
v___x_46_ = ((size_t)0ULL);
v___x_47_ = lean_usize_of_nat(v___x_44_);
v___x_48_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12_spec__16(v_a_42_, v_as_41_, v___x_46_, v___x_47_);
return v___x_48_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12___boxed(lean_object* v_as_49_, lean_object* v_a_50_){
_start:
{
uint8_t v_res_51_; lean_object* v_r_52_; 
v_res_51_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v_as_49_, v_a_50_);
lean_dec(v_a_50_);
lean_dec_ref(v_as_49_);
v_r_52_ = lean_box(v_res_51_);
return v_r_52_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8(lean_object* v_x_53_, lean_object* v_s_54_){
_start:
{
uint8_t v___x_55_; 
v___x_55_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v_s_54_, v_x_53_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8___boxed(lean_object* v_x_56_, lean_object* v_s_57_){
_start:
{
uint8_t v_res_58_; lean_object* v_r_59_; 
v_res_58_ = lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8(v_x_56_, v_s_57_);
lean_dec_ref(v_s_57_);
lean_dec(v_x_56_);
v_r_59_ = lean_box(v_res_58_);
return v_r_59_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_findPathForAssignedMVars___lam__0(lean_object* v_mvars_60_, uint8_t v___y_61_, uint8_t v___x_62_, lean_object* v_x_63_){
_start:
{
uint8_t v___x_64_; 
v___x_64_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v_mvars_60_, v_x_63_);
if (v___x_64_ == 0)
{
return v___y_61_;
}
else
{
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_findPathForAssignedMVars___lam__0___boxed(lean_object* v_mvars_65_, lean_object* v___y_66_, lean_object* v___x_67_, lean_object* v_x_68_){
_start:
{
uint8_t v___y_38780__boxed_69_; uint8_t v___x_38781__boxed_70_; uint8_t v_res_71_; lean_object* v_r_72_; 
v___y_38780__boxed_69_ = lean_unbox(v___y_66_);
v___x_38781__boxed_70_ = lean_unbox(v___x_67_);
v_res_71_ = lp_aesop_Aesop_findPathForAssignedMVars___lam__0(v_mvars_65_, v___y_38780__boxed_69_, v___x_38781__boxed_70_, v_x_68_);
lean_dec(v_x_68_);
lean_dec_ref(v_mvars_65_);
v_r_72_ = lean_box(v_res_71_);
return v_r_72_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7_spec__10(lean_object* v_msgData_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_){
_start:
{
lean_object* v___x_79_; lean_object* v_env_80_; lean_object* v___x_81_; lean_object* v_mctx_82_; lean_object* v_lctx_83_; lean_object* v_options_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_79_ = lean_st_ref_get(v___y_77_);
v_env_80_ = lean_ctor_get(v___x_79_, 0);
lean_inc_ref(v_env_80_);
lean_dec(v___x_79_);
v___x_81_ = lean_st_ref_get(v___y_75_);
v_mctx_82_ = lean_ctor_get(v___x_81_, 0);
lean_inc_ref(v_mctx_82_);
lean_dec(v___x_81_);
v_lctx_83_ = lean_ctor_get(v___y_74_, 2);
v_options_84_ = lean_ctor_get(v___y_76_, 2);
lean_inc_ref(v_options_84_);
lean_inc_ref(v_lctx_83_);
v___x_85_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_85_, 0, v_env_80_);
lean_ctor_set(v___x_85_, 1, v_mctx_82_);
lean_ctor_set(v___x_85_, 2, v_lctx_83_);
lean_ctor_set(v___x_85_, 3, v_options_84_);
v___x_86_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
lean_ctor_set(v___x_86_, 1, v_msgData_73_);
v___x_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7_spec__10___boxed(lean_object* v_msgData_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7_spec__10(v_msgData_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg(lean_object* v_msg_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v_ref_101_; lean_object* v___x_102_; lean_object* v_a_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_111_; 
v_ref_101_ = lean_ctor_get(v___y_98_, 5);
v___x_102_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7_spec__10(v_msg_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
v_a_103_ = lean_ctor_get(v___x_102_, 0);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_102_);
if (v_isSharedCheck_111_ == 0)
{
v___x_105_ = v___x_102_;
v_isShared_106_ = v_isSharedCheck_111_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_a_103_);
lean_dec(v___x_102_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_111_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v___x_107_; lean_object* v___x_109_; 
lean_inc(v_ref_101_);
v___x_107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_107_, 0, v_ref_101_);
lean_ctor_set(v___x_107_, 1, v_a_103_);
if (v_isShared_106_ == 0)
{
lean_ctor_set_tag(v___x_105_, 1);
lean_ctor_set(v___x_105_, 0, v___x_107_);
v___x_109_ = v___x_105_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v___x_107_);
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
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg___boxed(lean_object* v_msg_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg(v_msg_112_, v___y_113_, v___y_114_, v___y_115_, v___y_116_);
lean_dec(v___y_116_);
lean_dec_ref(v___y_115_);
lean_dec(v___y_114_);
lean_dec_ref(v___y_113_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_findPathForAssignedMVars_spec__5(size_t v_sz_119_, size_t v_i_120_, lean_object* v_bs_121_){
_start:
{
uint8_t v___x_122_; 
v___x_122_ = lean_usize_dec_lt(v_i_120_, v_sz_119_);
if (v___x_122_ == 0)
{
return v_bs_121_;
}
else
{
lean_object* v_v_123_; lean_object* v___x_124_; lean_object* v_bs_x27_125_; size_t v___x_126_; size_t v___x_127_; lean_object* v___x_128_; 
v_v_123_ = lean_array_uget(v_bs_121_, v_i_120_);
v___x_124_ = lean_unsigned_to_nat(0u);
v_bs_x27_125_ = lean_array_uset(v_bs_121_, v_i_120_, v___x_124_);
v___x_126_ = ((size_t)1ULL);
v___x_127_ = lean_usize_add(v_i_120_, v___x_126_);
v___x_128_ = lean_array_uset(v_bs_x27_125_, v_i_120_, v_v_123_);
v_i_120_ = v___x_127_;
v_bs_121_ = v___x_128_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_findPathForAssignedMVars_spec__5___boxed(lean_object* v_sz_130_, lean_object* v_i_131_, lean_object* v_bs_132_){
_start:
{
size_t v_sz_boxed_133_; size_t v_i_boxed_134_; lean_object* v_res_135_; 
v_sz_boxed_133_ = lean_unbox_usize(v_sz_130_);
lean_dec(v_sz_130_);
v_i_boxed_134_ = lean_unbox_usize(v_i_131_);
lean_dec(v_i_131_);
v_res_135_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_findPathForAssignedMVars_spec__5(v_sz_boxed_133_, v_i_boxed_134_, v_bs_132_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_findPathForAssignedMVars_spec__6(lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
if (lean_obj_tag(v_a_136_) == 0)
{
lean_object* v___x_138_; 
v___x_138_ = l_List_reverse___redArg(v_a_137_);
return v___x_138_;
}
else
{
lean_object* v_head_139_; lean_object* v_tail_140_; lean_object* v___x_142_; uint8_t v_isShared_143_; uint8_t v_isSharedCheck_149_; 
v_head_139_ = lean_ctor_get(v_a_136_, 0);
v_tail_140_ = lean_ctor_get(v_a_136_, 1);
v_isSharedCheck_149_ = !lean_is_exclusive(v_a_136_);
if (v_isSharedCheck_149_ == 0)
{
v___x_142_ = v_a_136_;
v_isShared_143_ = v_isSharedCheck_149_;
goto v_resetjp_141_;
}
else
{
lean_inc(v_tail_140_);
lean_inc(v_head_139_);
lean_dec(v_a_136_);
v___x_142_ = lean_box(0);
v_isShared_143_ = v_isSharedCheck_149_;
goto v_resetjp_141_;
}
v_resetjp_141_:
{
lean_object* v___x_144_; lean_object* v___x_146_; 
v___x_144_ = l_Lean_MessageData_ofName(v_head_139_);
if (v_isShared_143_ == 0)
{
lean_ctor_set(v___x_142_, 1, v_a_137_);
lean_ctor_set(v___x_142_, 0, v___x_144_);
v___x_146_ = v___x_142_;
goto v_reusejp_145_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_144_);
lean_ctor_set(v_reuseFailAlloc_148_, 1, v_a_137_);
v___x_146_ = v_reuseFailAlloc_148_;
goto v_reusejp_145_;
}
v_reusejp_145_:
{
v_a_136_ = v_tail_140_;
v_a_137_ = v___x_146_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10_spec__15(lean_object* v_p_150_, lean_object* v_as_151_, size_t v_i_152_, size_t v_stop_153_){
_start:
{
uint8_t v___x_154_; 
v___x_154_ = lean_usize_dec_eq(v_i_152_, v_stop_153_);
if (v___x_154_ == 0)
{
lean_object* v___x_155_; lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_155_ = lean_array_uget_borrowed(v_as_151_, v_i_152_);
lean_inc_ref(v_p_150_);
lean_inc(v___x_155_);
v___x_156_ = lean_apply_1(v_p_150_, v___x_155_);
v___x_157_ = lean_unbox(v___x_156_);
if (v___x_157_ == 0)
{
size_t v___x_158_; size_t v___x_159_; 
v___x_158_ = ((size_t)1ULL);
v___x_159_ = lean_usize_add(v_i_152_, v___x_158_);
v_i_152_ = v___x_159_;
goto _start;
}
else
{
uint8_t v___x_161_; 
lean_dec_ref(v_p_150_);
v___x_161_ = lean_unbox(v___x_156_);
return v___x_161_;
}
}
else
{
uint8_t v___x_162_; 
lean_dec_ref(v_p_150_);
v___x_162_ = 0;
return v___x_162_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10_spec__15___boxed(lean_object* v_p_163_, lean_object* v_as_164_, lean_object* v_i_165_, lean_object* v_stop_166_){
_start:
{
size_t v_i_boxed_167_; size_t v_stop_boxed_168_; uint8_t v_res_169_; lean_object* v_r_170_; 
v_i_boxed_167_ = lean_unbox_usize(v_i_165_);
lean_dec(v_i_165_);
v_stop_boxed_168_ = lean_unbox_usize(v_stop_166_);
lean_dec(v_stop_166_);
v_res_169_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10_spec__15(v_p_163_, v_as_164_, v_i_boxed_167_, v_stop_boxed_168_);
lean_dec_ref(v_as_164_);
v_r_170_ = lean_box(v_res_169_);
return v_r_170_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10(lean_object* v_p_171_, lean_object* v_s_172_, lean_object* v_start_173_, lean_object* v_stop_174_){
_start:
{
lean_object* v___y_176_; uint8_t v___x_181_; 
v___x_181_ = lean_nat_dec_lt(v_start_173_, v_stop_174_);
if (v___x_181_ == 0)
{
lean_dec(v_stop_174_);
lean_dec_ref(v_p_171_);
return v___x_181_;
}
else
{
lean_object* v___x_182_; uint8_t v___x_183_; 
v___x_182_ = lean_array_get_size(v_s_172_);
v___x_183_ = lean_nat_dec_le(v_stop_174_, v___x_182_);
if (v___x_183_ == 0)
{
lean_dec(v_stop_174_);
v___y_176_ = v___x_182_;
goto v___jp_175_;
}
else
{
v___y_176_ = v_stop_174_;
goto v___jp_175_;
}
}
v___jp_175_:
{
uint8_t v___x_177_; 
v___x_177_ = lean_nat_dec_lt(v_start_173_, v___y_176_);
if (v___x_177_ == 0)
{
lean_dec(v___y_176_);
lean_dec_ref(v_p_171_);
return v___x_177_;
}
else
{
size_t v___x_178_; size_t v___x_179_; uint8_t v___x_180_; 
v___x_178_ = lean_usize_of_nat(v_start_173_);
v___x_179_ = lean_usize_of_nat(v___y_176_);
lean_dec(v___y_176_);
v___x_180_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10_spec__15(v_p_171_, v_s_172_, v___x_178_, v___x_179_);
return v___x_180_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10___boxed(lean_object* v_p_184_, lean_object* v_s_185_, lean_object* v_start_186_, lean_object* v_stop_187_){
_start:
{
uint8_t v_res_188_; lean_object* v_r_189_; 
v_res_188_ = lp_aesop_Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10(v_p_184_, v_s_185_, v_start_186_, v_stop_187_);
lean_dec(v_start_186_);
lean_dec_ref(v_s_185_);
v_r_189_ = lean_box(v_res_188_);
return v_r_189_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11_spec__17(lean_object* v___x_190_, uint8_t v___y_191_, uint8_t v___x_192_, lean_object* v_as_193_, size_t v_i_194_, size_t v_stop_195_, lean_object* v_b_196_){
_start:
{
lean_object* v___y_198_; uint8_t v___x_202_; 
v___x_202_ = lean_usize_dec_eq(v_i_194_, v_stop_195_);
if (v___x_202_ == 0)
{
lean_object* v___x_203_; uint8_t v___y_205_; uint8_t v___x_207_; 
v___x_203_ = lean_array_uget_borrowed(v_as_193_, v_i_194_);
v___x_207_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v___x_190_, v___x_203_);
if (v___x_207_ == 0)
{
v___y_205_ = v___y_191_;
goto v___jp_204_;
}
else
{
v___y_205_ = v___x_192_;
goto v___jp_204_;
}
v___jp_204_:
{
if (v___y_205_ == 0)
{
v___y_198_ = v_b_196_;
goto v___jp_197_;
}
else
{
lean_object* v___x_206_; 
lean_inc(v___x_203_);
v___x_206_ = lean_array_push(v_b_196_, v___x_203_);
v___y_198_ = v___x_206_;
goto v___jp_197_;
}
}
}
else
{
return v_b_196_;
}
v___jp_197_:
{
size_t v___x_199_; size_t v___x_200_; 
v___x_199_ = ((size_t)1ULL);
v___x_200_ = lean_usize_add(v_i_194_, v___x_199_);
v_i_194_ = v___x_200_;
v_b_196_ = v___y_198_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11_spec__17___boxed(lean_object* v___x_208_, lean_object* v___y_209_, lean_object* v___x_210_, lean_object* v_as_211_, lean_object* v_i_212_, lean_object* v_stop_213_, lean_object* v_b_214_){
_start:
{
uint8_t v___y_38930__boxed_215_; uint8_t v___x_38931__boxed_216_; size_t v_i_boxed_217_; size_t v_stop_boxed_218_; lean_object* v_res_219_; 
v___y_38930__boxed_215_ = lean_unbox(v___y_209_);
v___x_38931__boxed_216_ = lean_unbox(v___x_210_);
v_i_boxed_217_ = lean_unbox_usize(v_i_212_);
lean_dec(v_i_212_);
v_stop_boxed_218_ = lean_unbox_usize(v_stop_213_);
lean_dec(v_stop_213_);
v_res_219_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11_spec__17(v___x_208_, v___y_38930__boxed_215_, v___x_38931__boxed_216_, v_as_211_, v_i_boxed_217_, v_stop_boxed_218_, v_b_214_);
lean_dec_ref(v_as_211_);
lean_dec_ref(v___x_208_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11(lean_object* v___x_220_, uint8_t v___y_221_, uint8_t v___x_222_, lean_object* v_as_223_, size_t v_i_224_, size_t v_stop_225_, lean_object* v_b_226_){
_start:
{
lean_object* v___y_228_; uint8_t v___x_232_; 
v___x_232_ = lean_usize_dec_eq(v_i_224_, v_stop_225_);
if (v___x_232_ == 0)
{
lean_object* v___x_233_; uint8_t v___y_235_; uint8_t v___x_237_; 
v___x_233_ = lean_array_uget_borrowed(v_as_223_, v_i_224_);
v___x_237_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v___x_220_, v___x_233_);
if (v___x_237_ == 0)
{
v___y_235_ = v___y_221_;
goto v___jp_234_;
}
else
{
v___y_235_ = v___x_222_;
goto v___jp_234_;
}
v___jp_234_:
{
if (v___y_235_ == 0)
{
v___y_228_ = v_b_226_;
goto v___jp_227_;
}
else
{
lean_object* v___x_236_; 
lean_inc(v___x_233_);
v___x_236_ = lean_array_push(v_b_226_, v___x_233_);
v___y_228_ = v___x_236_;
goto v___jp_227_;
}
}
}
else
{
return v_b_226_;
}
v___jp_227_:
{
size_t v___x_229_; size_t v___x_230_; lean_object* v___x_231_; 
v___x_229_ = ((size_t)1ULL);
v___x_230_ = lean_usize_add(v_i_224_, v___x_229_);
v___x_231_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11_spec__17(v___x_220_, v___y_221_, v___x_222_, v_as_223_, v___x_230_, v_stop_225_, v___y_228_);
return v___x_231_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11___boxed(lean_object* v___x_238_, lean_object* v___y_239_, lean_object* v___x_240_, lean_object* v_as_241_, lean_object* v_i_242_, lean_object* v_stop_243_, lean_object* v_b_244_){
_start:
{
uint8_t v___y_38962__boxed_245_; uint8_t v___x_38963__boxed_246_; size_t v_i_boxed_247_; size_t v_stop_boxed_248_; lean_object* v_res_249_; 
v___y_38962__boxed_245_ = lean_unbox(v___y_239_);
v___x_38963__boxed_246_ = lean_unbox(v___x_240_);
v_i_boxed_247_ = lean_unbox_usize(v_i_242_);
lean_dec(v_i_242_);
v_stop_boxed_248_ = lean_unbox_usize(v_stop_243_);
lean_dec(v_stop_243_);
v_res_249_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11(v___x_238_, v___y_38962__boxed_245_, v___x_38963__boxed_246_, v_as_241_, v_i_boxed_247_, v_stop_boxed_248_, v_b_244_);
lean_dec_ref(v_as_241_);
lean_dec_ref(v___x_238_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7_spec__18(lean_object* v_xs_250_, lean_object* v_v_251_, lean_object* v_i_252_){
_start:
{
lean_object* v___x_253_; uint8_t v___x_254_; 
v___x_253_ = lean_array_get_size(v_xs_250_);
v___x_254_ = lean_nat_dec_lt(v_i_252_, v___x_253_);
if (v___x_254_ == 0)
{
lean_object* v___x_255_; 
lean_dec(v_i_252_);
v___x_255_ = lean_box(0);
return v___x_255_;
}
else
{
lean_object* v___x_256_; uint8_t v___x_257_; 
v___x_256_ = lean_array_fget_borrowed(v_xs_250_, v_i_252_);
v___x_257_ = l_Lean_instBEqMVarId_beq(v___x_256_, v_v_251_);
if (v___x_257_ == 0)
{
lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_258_ = lean_unsigned_to_nat(1u);
v___x_259_ = lean_nat_add(v_i_252_, v___x_258_);
lean_dec(v_i_252_);
v_i_252_ = v___x_259_;
goto _start;
}
else
{
lean_object* v___x_261_; 
v___x_261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_261_, 0, v_i_252_);
return v___x_261_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7_spec__18___boxed(lean_object* v_xs_262_, lean_object* v_v_263_, lean_object* v_i_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7_spec__18(v_xs_262_, v_v_263_, v_i_264_);
lean_dec(v_v_263_);
lean_dec_ref(v_xs_262_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7(lean_object* v_xs_266_, lean_object* v_v_267_){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = lean_unsigned_to_nat(0u);
v___x_269_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7_spec__18(v_xs_266_, v_v_267_, v___x_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7___boxed(lean_object* v_xs_270_, lean_object* v_v_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7(v_xs_270_, v_v_271_);
lean_dec(v_v_271_);
lean_dec_ref(v_xs_270_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4(lean_object* v_as_273_, lean_object* v_a_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4_spec__7(v_as_273_, v_a_274_);
if (lean_obj_tag(v___x_275_) == 0)
{
return v_as_273_;
}
else
{
lean_object* v_val_276_; lean_object* v___x_277_; 
v_val_276_ = lean_ctor_get(v___x_275_, 0);
lean_inc(v_val_276_);
lean_dec_ref_known(v___x_275_, 1);
v___x_277_ = l_Array_eraseIdx___redArg(v_as_273_, v_val_276_);
return v___x_277_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4___boxed(lean_object* v_as_278_, lean_object* v_a_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_aesop_Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4(v_as_278_, v_a_279_);
lean_dec(v_a_279_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2(lean_object* v_x_281_, lean_object* v_s_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_aesop_Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4(v_s_282_, v_x_281_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2___boxed(lean_object* v_x_284_, lean_object* v_s_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_aesop_Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2(v_x_284_, v_s_285_);
lean_dec(v_x_284_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___redArg(lean_object* v_val_287_, lean_object* v_as_288_, size_t v_sz_289_, size_t v_i_290_, lean_object* v_b_291_){
_start:
{
uint8_t v___x_293_; 
v___x_293_ = lean_usize_dec_lt(v_i_290_, v_sz_289_);
if (v___x_293_ == 0)
{
lean_object* v___x_294_; 
v___x_294_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_294_, 0, v_b_291_);
return v___x_294_;
}
else
{
lean_object* v___x_295_; lean_object* v_a_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; size_t v___x_300_; size_t v___x_301_; 
v___x_295_ = lean_st_ref_take(v_val_287_);
v_a_296_ = lean_array_uget_borrowed(v_as_288_, v_i_290_);
v___x_297_ = lp_aesop_Array_erase___at___00Aesop_UnorderedArraySet_erase___at___00Aesop_findPathForAssignedMVars_spec__2_spec__4(v___x_295_, v_a_296_);
v___x_298_ = lean_st_ref_set(v_val_287_, v___x_297_);
v___x_299_ = lean_box(0);
v___x_300_ = ((size_t)1ULL);
v___x_301_ = lean_usize_add(v_i_290_, v___x_300_);
v_i_290_ = v___x_301_;
v_b_291_ = v___x_299_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___redArg___boxed(lean_object* v_val_303_, lean_object* v_as_304_, lean_object* v_sz_305_, lean_object* v_i_306_, lean_object* v_b_307_, lean_object* v___y_308_){
_start:
{
size_t v_sz_boxed_309_; size_t v_i_boxed_310_; lean_object* v_res_311_; 
v_sz_boxed_309_ = lean_unbox_usize(v_sz_305_);
lean_dec(v_sz_305_);
v_i_boxed_310_ = lean_unbox_usize(v_i_306_);
lean_dec(v_i_306_);
v_res_311_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___redArg(v_val_303_, v_as_304_, v_sz_boxed_309_, v_i_boxed_310_, v_b_307_);
lean_dec_ref(v_as_304_);
lean_dec(v_val_303_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4_spec__15___redArg(lean_object* v_x_312_, lean_object* v_x_313_){
_start:
{
if (lean_obj_tag(v_x_313_) == 0)
{
return v_x_312_;
}
else
{
lean_object* v_key_314_; lean_object* v_value_315_; lean_object* v_tail_316_; lean_object* v___x_318_; uint8_t v_isShared_319_; uint8_t v_isSharedCheck_339_; 
v_key_314_ = lean_ctor_get(v_x_313_, 0);
v_value_315_ = lean_ctor_get(v_x_313_, 1);
v_tail_316_ = lean_ctor_get(v_x_313_, 2);
v_isSharedCheck_339_ = !lean_is_exclusive(v_x_313_);
if (v_isSharedCheck_339_ == 0)
{
v___x_318_ = v_x_313_;
v_isShared_319_ = v_isSharedCheck_339_;
goto v_resetjp_317_;
}
else
{
lean_inc(v_tail_316_);
lean_inc(v_value_315_);
lean_inc(v_key_314_);
lean_dec(v_x_313_);
v___x_318_ = lean_box(0);
v_isShared_319_ = v_isSharedCheck_339_;
goto v_resetjp_317_;
}
v_resetjp_317_:
{
lean_object* v___x_320_; uint64_t v___x_321_; uint64_t v___x_322_; uint64_t v___x_323_; uint64_t v_fold_324_; uint64_t v___x_325_; uint64_t v___x_326_; uint64_t v___x_327_; size_t v___x_328_; size_t v___x_329_; size_t v___x_330_; size_t v___x_331_; size_t v___x_332_; lean_object* v___x_333_; lean_object* v___x_335_; 
v___x_320_ = lean_array_get_size(v_x_312_);
v___x_321_ = lean_uint64_of_nat(v_key_314_);
v___x_322_ = 32ULL;
v___x_323_ = lean_uint64_shift_right(v___x_321_, v___x_322_);
v_fold_324_ = lean_uint64_xor(v___x_321_, v___x_323_);
v___x_325_ = 16ULL;
v___x_326_ = lean_uint64_shift_right(v_fold_324_, v___x_325_);
v___x_327_ = lean_uint64_xor(v_fold_324_, v___x_326_);
v___x_328_ = lean_uint64_to_usize(v___x_327_);
v___x_329_ = lean_usize_of_nat(v___x_320_);
v___x_330_ = ((size_t)1ULL);
v___x_331_ = lean_usize_sub(v___x_329_, v___x_330_);
v___x_332_ = lean_usize_land(v___x_328_, v___x_331_);
v___x_333_ = lean_array_uget_borrowed(v_x_312_, v___x_332_);
lean_inc(v___x_333_);
if (v_isShared_319_ == 0)
{
lean_ctor_set(v___x_318_, 2, v___x_333_);
v___x_335_ = v___x_318_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v_key_314_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v_value_315_);
lean_ctor_set(v_reuseFailAlloc_338_, 2, v___x_333_);
v___x_335_ = v_reuseFailAlloc_338_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_336_; 
v___x_336_ = lean_array_uset(v_x_312_, v___x_332_, v___x_335_);
v_x_312_ = v___x_336_;
v_x_313_ = v_tail_316_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4___redArg(lean_object* v_i_340_, lean_object* v_source_341_, lean_object* v_target_342_){
_start:
{
lean_object* v___x_343_; uint8_t v___x_344_; 
v___x_343_ = lean_array_get_size(v_source_341_);
v___x_344_ = lean_nat_dec_lt(v_i_340_, v___x_343_);
if (v___x_344_ == 0)
{
lean_dec_ref(v_source_341_);
lean_dec(v_i_340_);
return v_target_342_;
}
else
{
lean_object* v_es_345_; lean_object* v___x_346_; lean_object* v_source_347_; lean_object* v_target_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v_es_345_ = lean_array_fget(v_source_341_, v_i_340_);
v___x_346_ = lean_box(0);
v_source_347_ = lean_array_fset(v_source_341_, v_i_340_, v___x_346_);
v_target_348_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4_spec__15___redArg(v_target_342_, v_es_345_);
v___x_349_ = lean_unsigned_to_nat(1u);
v___x_350_ = lean_nat_add(v_i_340_, v___x_349_);
lean_dec(v_i_340_);
v_i_340_ = v___x_350_;
v_source_341_ = v_source_347_;
v_target_342_ = v_target_348_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2___redArg(lean_object* v_data_352_){
_start:
{
lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v_nbuckets_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_353_ = lean_array_get_size(v_data_352_);
v___x_354_ = lean_unsigned_to_nat(2u);
v_nbuckets_355_ = lean_nat_mul(v___x_353_, v___x_354_);
v___x_356_ = lean_unsigned_to_nat(0u);
v___x_357_ = lean_box(0);
v___x_358_ = lean_mk_array(v_nbuckets_355_, v___x_357_);
v___x_359_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4___redArg(v___x_356_, v_data_352_, v___x_358_);
return v___x_359_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg(lean_object* v_a_360_, lean_object* v_x_361_){
_start:
{
if (lean_obj_tag(v_x_361_) == 0)
{
uint8_t v___x_362_; 
v___x_362_ = 0;
return v___x_362_;
}
else
{
lean_object* v_key_363_; lean_object* v_tail_364_; uint8_t v___x_365_; 
v_key_363_ = lean_ctor_get(v_x_361_, 0);
v_tail_364_ = lean_ctor_get(v_x_361_, 2);
v___x_365_ = lean_nat_dec_eq(v_key_363_, v_a_360_);
if (v___x_365_ == 0)
{
v_x_361_ = v_tail_364_;
goto _start;
}
else
{
return v___x_365_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg___boxed(lean_object* v_a_367_, lean_object* v_x_368_){
_start:
{
uint8_t v_res_369_; lean_object* v_r_370_; 
v_res_369_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg(v_a_367_, v_x_368_);
lean_dec(v_x_368_);
lean_dec(v_a_367_);
v_r_370_ = lean_box(v_res_369_);
return v_r_370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1___redArg(lean_object* v_m_371_, lean_object* v_a_372_, lean_object* v_b_373_){
_start:
{
lean_object* v_size_374_; lean_object* v_buckets_375_; lean_object* v___x_376_; uint64_t v___x_377_; uint64_t v___x_378_; uint64_t v___x_379_; uint64_t v_fold_380_; uint64_t v___x_381_; uint64_t v___x_382_; uint64_t v___x_383_; size_t v___x_384_; size_t v___x_385_; size_t v___x_386_; size_t v___x_387_; size_t v___x_388_; lean_object* v_bkt_389_; uint8_t v___x_390_; 
v_size_374_ = lean_ctor_get(v_m_371_, 0);
v_buckets_375_ = lean_ctor_get(v_m_371_, 1);
v___x_376_ = lean_array_get_size(v_buckets_375_);
v___x_377_ = lean_uint64_of_nat(v_a_372_);
v___x_378_ = 32ULL;
v___x_379_ = lean_uint64_shift_right(v___x_377_, v___x_378_);
v_fold_380_ = lean_uint64_xor(v___x_377_, v___x_379_);
v___x_381_ = 16ULL;
v___x_382_ = lean_uint64_shift_right(v_fold_380_, v___x_381_);
v___x_383_ = lean_uint64_xor(v_fold_380_, v___x_382_);
v___x_384_ = lean_uint64_to_usize(v___x_383_);
v___x_385_ = lean_usize_of_nat(v___x_376_);
v___x_386_ = ((size_t)1ULL);
v___x_387_ = lean_usize_sub(v___x_385_, v___x_386_);
v___x_388_ = lean_usize_land(v___x_384_, v___x_387_);
v_bkt_389_ = lean_array_uget_borrowed(v_buckets_375_, v___x_388_);
v___x_390_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg(v_a_372_, v_bkt_389_);
if (v___x_390_ == 0)
{
lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_411_; 
lean_inc_ref(v_buckets_375_);
lean_inc(v_size_374_);
v_isSharedCheck_411_ = !lean_is_exclusive(v_m_371_);
if (v_isSharedCheck_411_ == 0)
{
lean_object* v_unused_412_; lean_object* v_unused_413_; 
v_unused_412_ = lean_ctor_get(v_m_371_, 1);
lean_dec(v_unused_412_);
v_unused_413_ = lean_ctor_get(v_m_371_, 0);
lean_dec(v_unused_413_);
v___x_392_ = v_m_371_;
v_isShared_393_ = v_isSharedCheck_411_;
goto v_resetjp_391_;
}
else
{
lean_dec(v_m_371_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_411_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
lean_object* v___x_394_; lean_object* v_size_x27_395_; lean_object* v___x_396_; lean_object* v_buckets_x27_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; 
v___x_394_ = lean_unsigned_to_nat(1u);
v_size_x27_395_ = lean_nat_add(v_size_374_, v___x_394_);
lean_dec(v_size_374_);
lean_inc(v_bkt_389_);
v___x_396_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_396_, 0, v_a_372_);
lean_ctor_set(v___x_396_, 1, v_b_373_);
lean_ctor_set(v___x_396_, 2, v_bkt_389_);
v_buckets_x27_397_ = lean_array_uset(v_buckets_375_, v___x_388_, v___x_396_);
v___x_398_ = lean_unsigned_to_nat(4u);
v___x_399_ = lean_nat_mul(v_size_x27_395_, v___x_398_);
v___x_400_ = lean_unsigned_to_nat(3u);
v___x_401_ = lean_nat_div(v___x_399_, v___x_400_);
lean_dec(v___x_399_);
v___x_402_ = lean_array_get_size(v_buckets_x27_397_);
v___x_403_ = lean_nat_dec_le(v___x_401_, v___x_402_);
lean_dec(v___x_401_);
if (v___x_403_ == 0)
{
lean_object* v_val_404_; lean_object* v___x_406_; 
v_val_404_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2___redArg(v_buckets_x27_397_);
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 1, v_val_404_);
lean_ctor_set(v___x_392_, 0, v_size_x27_395_);
v___x_406_ = v___x_392_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_407_; 
v_reuseFailAlloc_407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_407_, 0, v_size_x27_395_);
lean_ctor_set(v_reuseFailAlloc_407_, 1, v_val_404_);
v___x_406_ = v_reuseFailAlloc_407_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
return v___x_406_;
}
}
else
{
lean_object* v___x_409_; 
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 1, v_buckets_x27_397_);
lean_ctor_set(v___x_392_, 0, v_size_x27_395_);
v___x_409_ = v___x_392_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v_size_x27_395_);
lean_ctor_set(v_reuseFailAlloc_410_, 1, v_buckets_x27_397_);
v___x_409_ = v_reuseFailAlloc_410_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
return v___x_409_;
}
}
}
}
else
{
lean_dec(v_b_373_);
lean_dec(v_a_372_);
return v_m_371_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4(lean_object* v_val_414_, lean_object* v_val_415_, lean_object* v_val_416_, uint8_t v___x_417_, lean_object* v_x_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
switch(lean_obj_tag(v_x_418_))
{
case 0:
{
lean_object* v_gref_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_457_; 
v_gref_430_ = lean_ctor_get(v_x_418_, 0);
v_isSharedCheck_457_ = !lean_is_exclusive(v_x_418_);
if (v_isSharedCheck_457_ == 0)
{
v___x_432_ = v_x_418_;
v_isShared_433_ = v_isSharedCheck_457_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_gref_430_);
lean_dec(v_x_418_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_457_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v_elimGoal_442_; lean_object* v___x_443_; lean_object* v_parent_444_; lean_object* v___x_446_; 
v___x_434_ = lean_st_ref_get(v_gref_430_);
v___x_435_ = lean_st_ref_take(v_val_414_);
v___x_436_ = lp_aesop_Aesop_Goal_originalGoalId(v___x_434_);
v___x_437_ = lean_box(0);
v___x_438_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1___redArg(v___x_435_, v___x_436_, v___x_437_);
v___x_439_ = lean_st_ref_set(v_val_414_, v___x_438_);
v___x_440_ = lean_st_ref_get(v_gref_430_);
lean_dec(v_gref_430_);
v___x_441_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_442_ = lean_ctor_get(v___x_441_, 1);
lean_inc_ref(v_elimGoal_442_);
v___x_443_ = lean_apply_1(v_elimGoal_442_, v___x_440_);
v_parent_444_ = lean_ctor_get(v___x_443_, 1);
lean_inc(v_parent_444_);
lean_dec_ref(v___x_443_);
if (v_isShared_433_ == 0)
{
lean_ctor_set_tag(v___x_432_, 2);
lean_ctor_set(v___x_432_, 0, v_parent_444_);
v___x_446_ = v___x_432_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_456_; 
v_reuseFailAlloc_456_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_456_, 0, v_parent_444_);
v___x_446_ = v_reuseFailAlloc_456_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
lean_object* v___x_447_; 
v___x_447_ = lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4(v_val_414_, v_val_415_, v_val_416_, v___x_417_, v___x_446_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_454_; 
v_isSharedCheck_454_ = !lean_is_exclusive(v___x_447_);
if (v_isSharedCheck_454_ == 0)
{
lean_object* v_unused_455_; 
v_unused_455_ = lean_ctor_get(v___x_447_, 0);
lean_dec(v_unused_455_);
v___x_449_ = v___x_447_;
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
else
{
lean_dec(v___x_447_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_454_;
goto v_resetjp_448_;
}
v_resetjp_448_:
{
lean_object* v___x_452_; 
if (v_isShared_450_ == 0)
{
lean_ctor_set(v___x_449_, 0, v___x_437_);
v___x_452_ = v___x_449_;
goto v_reusejp_451_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v___x_437_);
v___x_452_ = v_reuseFailAlloc_453_;
goto v_reusejp_451_;
}
v_reusejp_451_:
{
return v___x_452_;
}
}
}
else
{
return v___x_447_;
}
}
}
}
case 1:
{
lean_object* v_rref_458_; lean_object* v___x_460_; uint8_t v_isShared_461_; uint8_t v_isSharedCheck_501_; 
v_rref_458_ = lean_ctor_get(v_x_418_, 0);
v_isSharedCheck_501_ = !lean_is_exclusive(v_x_418_);
if (v_isSharedCheck_501_ == 0)
{
v___x_460_ = v_x_418_;
v_isShared_461_ = v_isSharedCheck_501_;
goto v_resetjp_459_;
}
else
{
lean_inc(v_rref_458_);
lean_dec(v_x_418_);
v___x_460_ = lean_box(0);
v_isShared_461_ = v_isSharedCheck_501_;
goto v_resetjp_459_;
}
v_resetjp_459_:
{
lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v_elimRapp_467_; lean_object* v___x_468_; lean_object* v_introducedMVars_469_; lean_object* v___x_470_; size_t v_sz_471_; size_t v___x_472_; lean_object* v___x_473_; 
v___x_462_ = lean_st_ref_take(v_val_415_);
lean_inc(v_rref_458_);
v___x_463_ = lean_array_push(v___x_462_, v_rref_458_);
v___x_464_ = lean_st_ref_set(v_val_415_, v___x_463_);
v___x_465_ = lean_st_ref_get(v_rref_458_);
v___x_466_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_467_ = lean_ctor_get(v___x_466_, 3);
lean_inc_ref(v_elimRapp_467_);
v___x_468_ = lean_apply_1(v_elimRapp_467_, v___x_465_);
v_introducedMVars_469_ = lean_ctor_get(v___x_468_, 7);
lean_inc_ref(v_introducedMVars_469_);
lean_dec_ref(v___x_468_);
v___x_470_ = lean_box(0);
v_sz_471_ = lean_array_size(v_introducedMVars_469_);
v___x_472_ = ((size_t)0ULL);
v___x_473_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___redArg(v_val_416_, v_introducedMVars_469_, v_sz_471_, v___x_472_, v___x_470_);
lean_dec_ref(v_introducedMVars_469_);
if (lean_obj_tag(v___x_473_) == 0)
{
lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_499_; 
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_473_);
if (v_isSharedCheck_499_ == 0)
{
lean_object* v_unused_500_; 
v_unused_500_ = lean_ctor_get(v___x_473_, 0);
lean_dec(v_unused_500_);
v___x_475_ = v___x_473_;
v_isShared_476_ = v_isSharedCheck_499_;
goto v_resetjp_474_;
}
else
{
lean_dec(v___x_473_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_499_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___x_477_; uint8_t v___x_495_; 
v___x_477_ = lean_st_ref_get(v_val_416_);
v___x_495_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0(v___x_477_);
lean_dec(v___x_477_);
if (v___x_495_ == 0)
{
lean_del_object(v___x_475_);
goto v___jp_478_;
}
else
{
if (v___x_417_ == 0)
{
lean_object* v___x_497_; 
lean_del_object(v___x_460_);
lean_dec(v_rref_458_);
if (v_isShared_476_ == 0)
{
lean_ctor_set(v___x_475_, 0, v___x_470_);
v___x_497_ = v___x_475_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_470_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
else
{
lean_del_object(v___x_475_);
goto v___jp_478_;
}
}
v___jp_478_:
{
lean_object* v___x_479_; lean_object* v_elimRapp_480_; lean_object* v___x_481_; lean_object* v_parent_482_; lean_object* v___x_484_; 
v___x_479_ = lean_st_ref_get(v_rref_458_);
lean_dec(v_rref_458_);
v_elimRapp_480_ = lean_ctor_get(v___x_466_, 3);
lean_inc_ref(v_elimRapp_480_);
v___x_481_ = lean_apply_1(v_elimRapp_480_, v___x_479_);
v_parent_482_ = lean_ctor_get(v___x_481_, 1);
lean_inc(v_parent_482_);
lean_dec_ref(v___x_481_);
if (v_isShared_461_ == 0)
{
lean_ctor_set_tag(v___x_460_, 0);
lean_ctor_set(v___x_460_, 0, v_parent_482_);
v___x_484_ = v___x_460_;
goto v_reusejp_483_;
}
else
{
lean_object* v_reuseFailAlloc_494_; 
v_reuseFailAlloc_494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_494_, 0, v_parent_482_);
v___x_484_ = v_reuseFailAlloc_494_;
goto v_reusejp_483_;
}
v_reusejp_483_:
{
lean_object* v___x_485_; 
v___x_485_ = lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4(v_val_414_, v_val_415_, v_val_416_, v___x_417_, v___x_484_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
if (lean_obj_tag(v___x_485_) == 0)
{
lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_492_; 
v_isSharedCheck_492_ = !lean_is_exclusive(v___x_485_);
if (v_isSharedCheck_492_ == 0)
{
lean_object* v_unused_493_; 
v_unused_493_ = lean_ctor_get(v___x_485_, 0);
lean_dec(v_unused_493_);
v___x_487_ = v___x_485_;
v_isShared_488_ = v_isSharedCheck_492_;
goto v_resetjp_486_;
}
else
{
lean_dec(v___x_485_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_492_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_490_; 
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 0, v___x_470_);
v___x_490_ = v___x_487_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_491_; 
v_reuseFailAlloc_491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_491_, 0, v___x_470_);
v___x_490_ = v_reuseFailAlloc_491_;
goto v_reusejp_489_;
}
v_reusejp_489_:
{
return v___x_490_;
}
}
}
else
{
return v___x_485_;
}
}
}
}
}
else
{
lean_del_object(v___x_460_);
lean_dec(v_rref_458_);
return v___x_473_;
}
}
}
default: 
{
lean_object* v_cref_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_516_; 
v_cref_502_ = lean_ctor_get(v_x_418_, 0);
v_isSharedCheck_516_ = !lean_is_exclusive(v_x_418_);
if (v_isSharedCheck_516_ == 0)
{
v___x_504_ = v_x_418_;
v_isShared_505_ = v_isSharedCheck_516_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_cref_502_);
lean_dec(v_x_418_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_516_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v_elimMVarCluster_508_; lean_object* v___x_509_; lean_object* v_parent_x3f_510_; 
v___x_506_ = lean_st_ref_get(v_cref_502_);
lean_dec(v_cref_502_);
v___x_507_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_508_ = lean_ctor_get(v___x_507_, 5);
lean_inc_ref(v_elimMVarCluster_508_);
v___x_509_ = lean_apply_1(v_elimMVarCluster_508_, v___x_506_);
v_parent_x3f_510_ = lean_ctor_get(v___x_509_, 0);
lean_inc(v_parent_x3f_510_);
lean_dec_ref(v___x_509_);
if (lean_obj_tag(v_parent_x3f_510_) == 1)
{
lean_object* v_val_511_; lean_object* v___x_513_; 
v_val_511_ = lean_ctor_get(v_parent_x3f_510_, 0);
lean_inc(v_val_511_);
lean_dec_ref_known(v_parent_x3f_510_, 1);
if (v_isShared_505_ == 0)
{
lean_ctor_set_tag(v___x_504_, 1);
lean_ctor_set(v___x_504_, 0, v_val_511_);
v___x_513_ = v___x_504_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v_val_511_);
v___x_513_ = v_reuseFailAlloc_515_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
lean_object* v___x_514_; 
v___x_514_ = lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4(v_val_414_, v_val_415_, v_val_416_, v___x_417_, v___x_513_, v___y_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
if (lean_obj_tag(v___x_514_) == 0)
{
lean_dec_ref_known(v___x_514_, 1);
goto v___jp_427_;
}
else
{
return v___x_514_;
}
}
}
else
{
lean_dec(v_parent_x3f_510_);
lean_del_object(v___x_504_);
goto v___jp_427_;
}
}
}
}
v___jp_427_:
{
lean_object* v___x_428_; lean_object* v___x_429_; 
v___x_428_ = lean_box(0);
v___x_429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_429_, 0, v___x_428_);
return v___x_429_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4___boxed(lean_object* v_val_517_, lean_object* v_val_518_, lean_object* v_val_519_, lean_object* v___x_520_, lean_object* v_x_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
uint8_t v___x_39214__boxed_530_; lean_object* v_res_531_; 
v___x_39214__boxed_530_ = lean_unbox(v___x_520_);
v_res_531_ = lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4(v_val_517_, v_val_518_, v_val_519_, v___x_39214__boxed_530_, v_x_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
lean_dec(v___y_524_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec(v_val_519_);
lean_dec(v_val_518_);
lean_dec(v_val_517_);
return v_res_531_;
}
}
static lean_object* _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__1(void){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_534_ = lean_box(0);
v___x_535_ = lean_unsigned_to_nat(16u);
v___x_536_ = lean_mk_array(v___x_535_, v___x_534_);
return v___x_536_;
}
}
static lean_object* _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__2(void){
_start:
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v___x_537_ = lean_obj_once(&lp_aesop_Aesop_findPathForAssignedMVars___closed__1, &lp_aesop_Aesop_findPathForAssignedMVars___closed__1_once, _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__1);
v___x_538_ = lean_unsigned_to_nat(0u);
v___x_539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_538_);
lean_ctor_set(v___x_539_, 1, v___x_537_);
return v___x_539_;
}
}
static lean_object* _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__4(void){
_start:
{
lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_541_ = ((lean_object*)(lp_aesop_Aesop_findPathForAssignedMVars___closed__3));
v___x_542_ = l_Lean_stringToMessageData(v___x_541_);
return v___x_542_;
}
}
static lean_object* _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__5(void){
_start:
{
lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v___x_543_ = lean_obj_once(&lp_aesop_Aesop_findPathForAssignedMVars___closed__2, &lp_aesop_Aesop_findPathForAssignedMVars___closed__2_once, _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__2);
v___x_544_ = ((lean_object*)(lp_aesop_Aesop_findPathForAssignedMVars___closed__0));
v___x_545_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_545_, 0, v___x_544_);
lean_ctor_set(v___x_545_, 1, v___x_543_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_findPathForAssignedMVars(lean_object* v_assignedMVars_546_, lean_object* v_start_547_, lean_object* v_a_548_, lean_object* v_a_549_, lean_object* v_a_550_, lean_object* v_a_551_, lean_object* v_a_552_, lean_object* v_a_553_, lean_object* v_a_554_){
_start:
{
uint8_t v___x_556_; 
v___x_556_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0(v_assignedMVars_546_);
if (v___x_556_ == 0)
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___y_569_; lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_557_ = lean_st_mk_ref(v_assignedMVars_546_);
v___x_558_ = lean_unsigned_to_nat(0u);
v___x_559_ = ((lean_object*)(lp_aesop_Aesop_findPathForAssignedMVars___closed__0));
v___x_560_ = lean_st_mk_ref(v___x_559_);
v___x_561_ = lean_obj_once(&lp_aesop_Aesop_findPathForAssignedMVars___closed__2, &lp_aesop_Aesop_findPathForAssignedMVars___closed__2_once, _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__2);
v___x_562_ = lean_st_mk_ref(v___x_561_);
v___x_588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_588_, 0, v_start_547_);
v___x_589_ = lp_aesop_Aesop_traverseUp___at___00Aesop_findPathForAssignedMVars_spec__4(v___x_562_, v___x_560_, v___x_557_, v___x_556_, v___x_588_, v_a_548_, v_a_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_, v_a_554_);
if (lean_obj_tag(v___x_589_) == 0)
{
lean_object* v___x_590_; uint8_t v___y_592_; uint8_t v___x_621_; 
lean_dec_ref_known(v___x_589_, 1);
v___x_590_ = lean_st_ref_get(v___x_557_);
lean_dec(v___x_557_);
v___x_621_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___at___00Aesop_findPathForAssignedMVars_spec__0(v___x_590_);
if (v___x_621_ == 0)
{
uint8_t v___x_622_; 
v___x_622_ = 1;
v___y_592_ = v___x_622_;
goto v___jp_591_;
}
else
{
if (v___x_556_ == 0)
{
lean_dec(v___x_590_);
goto v___jp_563_;
}
else
{
v___y_592_ = v___x_556_;
goto v___jp_591_;
}
}
v___jp_591_:
{
lean_object* v___x_593_; 
v___x_593_ = lp_aesop_Aesop_getRootGoal(v_a_548_, v_a_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_, v_a_554_);
if (lean_obj_tag(v___x_593_) == 0)
{
lean_object* v_a_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v_elimGoal_597_; lean_object* v___x_598_; lean_object* v_mvars_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___f_602_; lean_object* v___x_603_; uint8_t v___x_604_; 
v_a_594_ = lean_ctor_get(v___x_593_, 0);
lean_inc(v_a_594_);
lean_dec_ref_known(v___x_593_, 1);
v___x_595_ = lean_st_ref_get(v_a_594_);
lean_dec(v_a_594_);
v___x_596_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_597_ = lean_ctor_get(v___x_596_, 1);
lean_inc_ref(v_elimGoal_597_);
v___x_598_ = lean_apply_1(v_elimGoal_597_, v___x_595_);
v_mvars_599_ = lean_ctor_get(v___x_598_, 7);
lean_inc_ref_n(v_mvars_599_, 2);
lean_dec_ref(v___x_598_);
v___x_600_ = lean_box(v___y_592_);
v___x_601_ = lean_box(v___x_556_);
v___f_602_ = lean_alloc_closure((void*)(lp_aesop_Aesop_findPathForAssignedMVars___lam__0___boxed), 4, 3);
lean_closure_set(v___f_602_, 0, v_mvars_599_);
lean_closure_set(v___f_602_, 1, v___x_600_);
lean_closure_set(v___f_602_, 2, v___x_601_);
v___x_603_ = lean_array_get_size(v___x_590_);
v___x_604_ = lp_aesop_Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10(v___f_602_, v___x_590_, v___x_558_, v___x_603_);
if (v___x_604_ == 0)
{
lean_dec_ref(v_mvars_599_);
lean_dec(v___x_590_);
goto v___jp_563_;
}
else
{
uint8_t v___x_605_; 
lean_dec(v___x_562_);
lean_dec(v___x_560_);
v___x_605_ = lean_nat_dec_lt(v___x_558_, v___x_603_);
if (v___x_605_ == 0)
{
lean_dec_ref(v_mvars_599_);
lean_dec(v___x_590_);
v___y_569_ = v___x_559_;
goto v___jp_568_;
}
else
{
uint8_t v___x_606_; 
v___x_606_ = lean_nat_dec_le(v___x_603_, v___x_603_);
if (v___x_606_ == 0)
{
if (v___x_605_ == 0)
{
lean_dec_ref(v_mvars_599_);
lean_dec(v___x_590_);
v___y_569_ = v___x_559_;
goto v___jp_568_;
}
else
{
size_t v___x_607_; size_t v___x_608_; lean_object* v___x_609_; 
v___x_607_ = ((size_t)0ULL);
v___x_608_ = lean_usize_of_nat(v___x_603_);
v___x_609_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11(v_mvars_599_, v___y_592_, v___x_556_, v___x_590_, v___x_607_, v___x_608_, v___x_559_);
lean_dec(v___x_590_);
lean_dec_ref(v_mvars_599_);
v___y_569_ = v___x_609_;
goto v___jp_568_;
}
}
else
{
size_t v___x_610_; size_t v___x_611_; lean_object* v___x_612_; 
v___x_610_ = ((size_t)0ULL);
v___x_611_ = lean_usize_of_nat(v___x_603_);
v___x_612_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_findPathForAssignedMVars_spec__11(v_mvars_599_, v___y_592_, v___x_556_, v___x_590_, v___x_610_, v___x_611_, v___x_559_);
lean_dec(v___x_590_);
lean_dec_ref(v_mvars_599_);
v___y_569_ = v___x_612_;
goto v___jp_568_;
}
}
}
}
else
{
lean_object* v_a_613_; lean_object* v___x_615_; uint8_t v_isShared_616_; uint8_t v_isSharedCheck_620_; 
lean_dec(v___x_590_);
lean_dec(v___x_562_);
lean_dec(v___x_560_);
v_a_613_ = lean_ctor_get(v___x_593_, 0);
v_isSharedCheck_620_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_620_ == 0)
{
v___x_615_ = v___x_593_;
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
else
{
lean_inc(v_a_613_);
lean_dec(v___x_593_);
v___x_615_ = lean_box(0);
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
v_resetjp_614_:
{
lean_object* v___x_618_; 
if (v_isShared_616_ == 0)
{
v___x_618_ = v___x_615_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_a_613_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
}
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
lean_dec(v___x_562_);
lean_dec(v___x_560_);
lean_dec(v___x_557_);
v_a_623_ = lean_ctor_get(v___x_589_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_589_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_589_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_589_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
v___jp_563_:
{
lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; 
v___x_564_ = lean_st_ref_get(v___x_560_);
lean_dec(v___x_560_);
v___x_565_ = lean_st_ref_get(v___x_562_);
lean_dec(v___x_562_);
v___x_566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_564_);
lean_ctor_set(v___x_566_, 1, v___x_565_);
v___x_567_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_567_, 0, v___x_566_);
return v___x_567_;
}
v___jp_568_:
{
size_t v_sz_570_; size_t v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v_a_580_; lean_object* v___x_582_; uint8_t v_isShared_583_; uint8_t v_isSharedCheck_587_; 
v_sz_570_ = lean_array_size(v___y_569_);
v___x_571_ = ((size_t)0ULL);
v___x_572_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_findPathForAssignedMVars_spec__5(v_sz_570_, v___x_571_, v___y_569_);
v___x_573_ = lean_obj_once(&lp_aesop_Aesop_findPathForAssignedMVars___closed__4, &lp_aesop_Aesop_findPathForAssignedMVars___closed__4_once, _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__4);
v___x_574_ = lean_array_to_list(v___x_572_);
v___x_575_ = lean_box(0);
v___x_576_ = lp_aesop_List_mapTR_loop___at___00Aesop_findPathForAssignedMVars_spec__6(v___x_574_, v___x_575_);
v___x_577_ = l_Lean_MessageData_ofList(v___x_576_);
v___x_578_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_578_, 0, v___x_573_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v___x_579_ = lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg(v___x_578_, v_a_551_, v_a_552_, v_a_553_, v_a_554_);
v_a_580_ = lean_ctor_get(v___x_579_, 0);
v_isSharedCheck_587_ = !lean_is_exclusive(v___x_579_);
if (v_isSharedCheck_587_ == 0)
{
v___x_582_ = v___x_579_;
v_isShared_583_ = v_isSharedCheck_587_;
goto v_resetjp_581_;
}
else
{
lean_inc(v_a_580_);
lean_dec(v___x_579_);
v___x_582_ = lean_box(0);
v_isShared_583_ = v_isSharedCheck_587_;
goto v_resetjp_581_;
}
v_resetjp_581_:
{
lean_object* v___x_585_; 
if (v_isShared_583_ == 0)
{
v___x_585_ = v___x_582_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_586_; 
v_reuseFailAlloc_586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_586_, 0, v_a_580_);
v___x_585_ = v_reuseFailAlloc_586_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
return v___x_585_;
}
}
}
}
else
{
lean_object* v___x_631_; lean_object* v___x_632_; 
lean_dec(v_start_547_);
lean_dec_ref(v_assignedMVars_546_);
v___x_631_ = lean_obj_once(&lp_aesop_Aesop_findPathForAssignedMVars___closed__5, &lp_aesop_Aesop_findPathForAssignedMVars___closed__5_once, _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__5);
v___x_632_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_632_, 0, v___x_631_);
return v___x_632_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_findPathForAssignedMVars___boxed(lean_object* v_assignedMVars_633_, lean_object* v_start_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_aesop_Aesop_findPathForAssignedMVars(v_assignedMVars_633_, v_start_634_, v_a_635_, v_a_636_, v_a_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_);
lean_dec(v_a_641_);
lean_dec_ref(v_a_640_);
lean_dec(v_a_639_);
lean_dec_ref(v_a_638_);
lean_dec(v_a_637_);
lean_dec(v_a_636_);
lean_dec_ref(v_a_635_);
return v_res_643_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1(lean_object* v_00_u03b2_644_, lean_object* v_m_645_, lean_object* v_a_646_, lean_object* v_b_647_){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1___redArg(v_m_645_, v_a_646_, v_b_647_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3(lean_object* v_val_649_, lean_object* v_as_650_, size_t v_sz_651_, size_t v_i_652_, lean_object* v_b_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_){
_start:
{
lean_object* v___x_662_; 
v___x_662_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___redArg(v_val_649_, v_as_650_, v_sz_651_, v_i_652_, v_b_653_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3___boxed(lean_object* v_val_663_, lean_object* v_as_664_, lean_object* v_sz_665_, lean_object* v_i_666_, lean_object* v_b_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_){
_start:
{
size_t v_sz_boxed_676_; size_t v_i_boxed_677_; lean_object* v_res_678_; 
v_sz_boxed_676_ = lean_unbox_usize(v_sz_665_);
lean_dec(v_sz_665_);
v_i_boxed_677_ = lean_unbox_usize(v_i_666_);
lean_dec(v_i_666_);
v_res_678_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_findPathForAssignedMVars_spec__3(v_val_663_, v_as_664_, v_sz_boxed_676_, v_i_boxed_677_, v_b_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_, v___y_674_);
lean_dec(v___y_674_);
lean_dec_ref(v___y_673_);
lean_dec(v___y_672_);
lean_dec_ref(v___y_671_);
lean_dec(v___y_670_);
lean_dec(v___y_669_);
lean_dec_ref(v___y_668_);
lean_dec_ref(v_as_664_);
lean_dec(v_val_663_);
return v_res_678_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7(lean_object* v_00_u03b1_679_, lean_object* v_msg_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_){
_start:
{
lean_object* v___x_689_; 
v___x_689_ = lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg(v_msg_680_, v___y_684_, v___y_685_, v___y_686_, v___y_687_);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___boxed(lean_object* v_00_u03b1_690_, lean_object* v_msg_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_){
_start:
{
lean_object* v_res_700_; 
v_res_700_ = lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7(v_00_u03b1_690_, v_msg_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_, v___y_696_, v___y_697_, v___y_698_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
lean_dec(v___y_696_);
lean_dec_ref(v___y_695_);
lean_dec(v___y_694_);
lean_dec(v___y_693_);
lean_dec_ref(v___y_692_);
return v_res_700_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1(lean_object* v_00_u03b2_701_, lean_object* v_a_702_, lean_object* v_x_703_){
_start:
{
uint8_t v___x_704_; 
v___x_704_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg(v_a_702_, v_x_703_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___boxed(lean_object* v_00_u03b2_705_, lean_object* v_a_706_, lean_object* v_x_707_){
_start:
{
uint8_t v_res_708_; lean_object* v_r_709_; 
v_res_708_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1(v_00_u03b2_705_, v_a_706_, v_x_707_);
lean_dec(v_x_707_);
lean_dec(v_a_706_);
v_r_709_ = lean_box(v_res_708_);
return v_r_709_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2(lean_object* v_00_u03b2_710_, lean_object* v_data_711_){
_start:
{
lean_object* v___x_712_; 
v___x_712_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2___redArg(v_data_711_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_713_, lean_object* v_i_714_, lean_object* v_source_715_, lean_object* v_target_716_){
_start:
{
lean_object* v___x_717_; 
v___x_717_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4___redArg(v_i_714_, v_source_715_, v_target_716_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4_spec__15(lean_object* v_00_u03b2_718_, lean_object* v_x_719_, lean_object* v_x_720_){
_start:
{
lean_object* v___x_721_; 
v___x_721_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__2_spec__4_spec__15___redArg(v_x_719_, v_x_720_);
return v___x_721_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___lam__0(lean_object* v_assignedMVars_722_, lean_object* v_x_723_){
_start:
{
uint8_t v___x_724_; 
v___x_724_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v_assignedMVars_722_, v_x_723_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___lam__0___boxed(lean_object* v_assignedMVars_725_, lean_object* v_x_726_){
_start:
{
uint8_t v_res_727_; lean_object* v_r_728_; 
v_res_727_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___lam__0(v_assignedMVars_725_, v_x_726_);
lean_dec(v_x_726_);
lean_dec_ref(v_assignedMVars_725_);
v_r_728_ = lean_box(v_res_727_);
return v_r_728_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg(lean_object* v_m_729_, lean_object* v_a_730_){
_start:
{
lean_object* v_buckets_731_; lean_object* v___x_732_; uint64_t v___x_733_; uint64_t v___x_734_; uint64_t v___x_735_; uint64_t v_fold_736_; uint64_t v___x_737_; uint64_t v___x_738_; uint64_t v___x_739_; size_t v___x_740_; size_t v___x_741_; size_t v___x_742_; size_t v___x_743_; size_t v___x_744_; lean_object* v___x_745_; uint8_t v___x_746_; 
v_buckets_731_ = lean_ctor_get(v_m_729_, 1);
v___x_732_ = lean_array_get_size(v_buckets_731_);
v___x_733_ = lean_uint64_of_nat(v_a_730_);
v___x_734_ = 32ULL;
v___x_735_ = lean_uint64_shift_right(v___x_733_, v___x_734_);
v_fold_736_ = lean_uint64_xor(v___x_733_, v___x_735_);
v___x_737_ = 16ULL;
v___x_738_ = lean_uint64_shift_right(v_fold_736_, v___x_737_);
v___x_739_ = lean_uint64_xor(v_fold_736_, v___x_738_);
v___x_740_ = lean_uint64_to_usize(v___x_739_);
v___x_741_ = lean_usize_of_nat(v___x_732_);
v___x_742_ = ((size_t)1ULL);
v___x_743_ = lean_usize_sub(v___x_741_, v___x_742_);
v___x_744_ = lean_usize_land(v___x_740_, v___x_743_);
v___x_745_ = lean_array_uget_borrowed(v_buckets_731_, v___x_744_);
v___x_746_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1_spec__1___redArg(v_a_730_, v___x_745_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg___boxed(lean_object* v_m_747_, lean_object* v_a_748_){
_start:
{
uint8_t v_res_749_; lean_object* v_r_750_; 
v_res_749_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg(v_m_747_, v_a_748_);
lean_dec(v_a_748_);
lean_dec_ref(v_m_747_);
v_r_750_ = lean_box(v_res_749_);
return v_r_750_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg(lean_object* v_snd_751_, lean_object* v_assignedMVars_752_, lean_object* v_as_753_, size_t v_sz_754_, size_t v_i_755_, lean_object* v_b_756_){
_start:
{
lean_object* v_a_759_; uint8_t v___x_763_; 
v___x_763_ = lean_usize_dec_lt(v_i_755_, v_sz_754_);
if (v___x_763_ == 0)
{
lean_object* v___x_764_; 
lean_dec_ref(v_assignedMVars_752_);
v___x_764_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_764_, 0, v_b_756_);
return v___x_764_;
}
else
{
lean_object* v_a_765_; lean_object* v___x_766_; lean_object* v_fst_767_; lean_object* v_snd_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_791_; 
v_a_765_ = lean_array_uget_borrowed(v_as_753_, v_i_755_);
v___x_766_ = lean_st_ref_get(v_a_765_);
v_fst_767_ = lean_ctor_get(v_b_756_, 0);
v_snd_768_ = lean_ctor_get(v_b_756_, 1);
v_isSharedCheck_791_ = !lean_is_exclusive(v_b_756_);
if (v_isSharedCheck_791_ == 0)
{
v___x_770_ = v_b_756_;
v_isShared_771_ = v_isSharedCheck_791_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_snd_768_);
lean_inc(v_fst_767_);
lean_dec(v_b_756_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_791_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_776_; uint8_t v___x_777_; 
lean_inc(v___x_766_);
v___x_776_ = lp_aesop_Aesop_Goal_originalGoalId(v___x_766_);
v___x_777_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg(v_snd_751_, v___x_776_);
if (v___x_777_ == 0)
{
uint8_t v___x_778_; 
v___x_778_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg(v_snd_768_, v___x_776_);
if (v___x_778_ == 0)
{
lean_object* v___x_779_; lean_object* v_elimGoal_780_; lean_object* v___x_781_; lean_object* v_mvars_782_; lean_object* v___f_783_; lean_object* v___x_784_; lean_object* v___x_785_; uint8_t v___x_786_; 
v___x_779_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_780_ = lean_ctor_get(v___x_779_, 1);
lean_inc_ref(v_elimGoal_780_);
v___x_781_ = lean_apply_1(v_elimGoal_780_, v___x_766_);
v_mvars_782_ = lean_ctor_get(v___x_781_, 7);
lean_inc_ref(v_mvars_782_);
lean_dec_ref(v___x_781_);
lean_inc_ref(v_assignedMVars_752_);
v___f_783_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_783_, 0, v_assignedMVars_752_);
v___x_784_ = lean_unsigned_to_nat(0u);
v___x_785_ = lean_array_get_size(v_mvars_782_);
v___x_786_ = lp_aesop_Aesop_UnorderedArraySet_any___at___00Aesop_findPathForAssignedMVars_spec__10(v___f_783_, v_mvars_782_, v___x_784_, v___x_785_);
lean_dec_ref(v_mvars_782_);
if (v___x_786_ == 0)
{
lean_dec(v___x_776_);
goto v___jp_772_;
}
else
{
lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; 
lean_del_object(v___x_770_);
lean_inc(v_a_765_);
v___x_787_ = lean_array_push(v_fst_767_, v_a_765_);
v___x_788_ = lean_box(0);
v___x_789_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_findPathForAssignedMVars_spec__1___redArg(v_snd_768_, v___x_776_, v___x_788_);
v___x_790_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_790_, 0, v___x_787_);
lean_ctor_set(v___x_790_, 1, v___x_789_);
v_a_759_ = v___x_790_;
goto v___jp_758_;
}
}
else
{
lean_dec(v___x_776_);
lean_dec(v___x_766_);
goto v___jp_772_;
}
}
else
{
lean_dec(v___x_776_);
lean_dec(v___x_766_);
goto v___jp_772_;
}
v___jp_772_:
{
lean_object* v___x_774_; 
if (v_isShared_771_ == 0)
{
v___x_774_ = v___x_770_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v_fst_767_);
lean_ctor_set(v_reuseFailAlloc_775_, 1, v_snd_768_);
v___x_774_ = v_reuseFailAlloc_775_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
v_a_759_ = v___x_774_;
goto v___jp_758_;
}
}
}
}
v___jp_758_:
{
size_t v___x_760_; size_t v___x_761_; 
v___x_760_ = ((size_t)1ULL);
v___x_761_ = lean_usize_add(v_i_755_, v___x_760_);
v_i_755_ = v___x_761_;
v_b_756_ = v_a_759_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg___boxed(lean_object* v_snd_792_, lean_object* v_assignedMVars_793_, lean_object* v_as_794_, lean_object* v_sz_795_, lean_object* v_i_796_, lean_object* v_b_797_, lean_object* v___y_798_){
_start:
{
size_t v_sz_boxed_799_; size_t v_i_boxed_800_; lean_object* v_res_801_; 
v_sz_boxed_799_ = lean_unbox_usize(v_sz_795_);
lean_dec(v_sz_795_);
v_i_boxed_800_ = lean_unbox_usize(v_i_796_);
lean_dec(v_i_796_);
v_res_801_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg(v_snd_792_, v_assignedMVars_793_, v_as_794_, v_sz_boxed_799_, v_i_boxed_800_, v_b_797_);
lean_dec_ref(v_as_794_);
lean_dec_ref(v_snd_792_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__2(lean_object* v_snd_802_, lean_object* v_assignedMVars_803_, lean_object* v_as_804_, size_t v_sz_805_, size_t v_i_806_, lean_object* v_b_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_){
_start:
{
uint8_t v___x_816_; 
v___x_816_ = lean_usize_dec_lt(v_i_806_, v_sz_805_);
if (v___x_816_ == 0)
{
lean_object* v___x_817_; 
lean_dec_ref(v_assignedMVars_803_);
v___x_817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_817_, 0, v_b_807_);
return v___x_817_;
}
else
{
lean_object* v_a_818_; lean_object* v___x_819_; lean_object* v_fst_820_; lean_object* v_snd_821_; lean_object* v___x_823_; uint8_t v_isShared_824_; uint8_t v_isSharedCheck_848_; 
v_a_818_ = lean_array_uget_borrowed(v_as_804_, v_i_806_);
v___x_819_ = lean_st_ref_get(v_a_818_);
v_fst_820_ = lean_ctor_get(v_b_807_, 0);
v_snd_821_ = lean_ctor_get(v_b_807_, 1);
v_isSharedCheck_848_ = !lean_is_exclusive(v_b_807_);
if (v_isSharedCheck_848_ == 0)
{
v___x_823_ = v_b_807_;
v_isShared_824_ = v_isSharedCheck_848_;
goto v_resetjp_822_;
}
else
{
lean_inc(v_snd_821_);
lean_inc(v_fst_820_);
lean_dec(v_b_807_);
v___x_823_ = lean_box(0);
v_isShared_824_ = v_isSharedCheck_848_;
goto v_resetjp_822_;
}
v_resetjp_822_:
{
lean_object* v___x_825_; lean_object* v_elimMVarCluster_826_; lean_object* v___x_827_; lean_object* v_goals_828_; lean_object* v___x_830_; 
v___x_825_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_826_ = lean_ctor_get(v___x_825_, 5);
lean_inc_ref(v_elimMVarCluster_826_);
v___x_827_ = lean_apply_1(v_elimMVarCluster_826_, v___x_819_);
v_goals_828_ = lean_ctor_get(v___x_827_, 1);
lean_inc_ref(v_goals_828_);
lean_dec_ref(v___x_827_);
if (v_isShared_824_ == 0)
{
v___x_830_ = v___x_823_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v_fst_820_);
lean_ctor_set(v_reuseFailAlloc_847_, 1, v_snd_821_);
v___x_830_ = v_reuseFailAlloc_847_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
size_t v_sz_831_; size_t v___x_832_; lean_object* v___x_833_; 
v_sz_831_ = lean_array_size(v_goals_828_);
v___x_832_ = ((size_t)0ULL);
lean_inc_ref(v_assignedMVars_803_);
v___x_833_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg(v_snd_802_, v_assignedMVars_803_, v_goals_828_, v_sz_831_, v___x_832_, v___x_830_);
lean_dec_ref(v_goals_828_);
if (lean_obj_tag(v___x_833_) == 0)
{
lean_object* v_a_834_; lean_object* v_fst_835_; lean_object* v_snd_836_; lean_object* v___x_838_; uint8_t v_isShared_839_; uint8_t v_isSharedCheck_846_; 
v_a_834_ = lean_ctor_get(v___x_833_, 0);
lean_inc(v_a_834_);
lean_dec_ref_known(v___x_833_, 1);
v_fst_835_ = lean_ctor_get(v_a_834_, 0);
v_snd_836_ = lean_ctor_get(v_a_834_, 1);
v_isSharedCheck_846_ = !lean_is_exclusive(v_a_834_);
if (v_isSharedCheck_846_ == 0)
{
v___x_838_ = v_a_834_;
v_isShared_839_ = v_isSharedCheck_846_;
goto v_resetjp_837_;
}
else
{
lean_inc(v_snd_836_);
lean_inc(v_fst_835_);
lean_dec(v_a_834_);
v___x_838_ = lean_box(0);
v_isShared_839_ = v_isSharedCheck_846_;
goto v_resetjp_837_;
}
v_resetjp_837_:
{
lean_object* v___x_841_; 
if (v_isShared_839_ == 0)
{
v___x_841_ = v___x_838_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_fst_835_);
lean_ctor_set(v_reuseFailAlloc_845_, 1, v_snd_836_);
v___x_841_ = v_reuseFailAlloc_845_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
size_t v___x_842_; size_t v___x_843_; 
v___x_842_ = ((size_t)1ULL);
v___x_843_ = lean_usize_add(v_i_806_, v___x_842_);
v_i_806_ = v___x_843_;
v_b_807_ = v___x_841_;
goto _start;
}
}
}
else
{
lean_dec_ref(v_assignedMVars_803_);
return v___x_833_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__2___boxed(lean_object* v_snd_849_, lean_object* v_assignedMVars_850_, lean_object* v_as_851_, lean_object* v_sz_852_, lean_object* v_i_853_, lean_object* v_b_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_){
_start:
{
size_t v_sz_boxed_863_; size_t v_i_boxed_864_; lean_object* v_res_865_; 
v_sz_boxed_863_ = lean_unbox_usize(v_sz_852_);
lean_dec(v_sz_852_);
v_i_boxed_864_ = lean_unbox_usize(v_i_853_);
lean_dec(v_i_853_);
v_res_865_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__2(v_snd_849_, v_assignedMVars_850_, v_as_851_, v_sz_boxed_863_, v_i_boxed_864_, v_b_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_, v___y_860_, v___y_861_);
lean_dec(v___y_861_);
lean_dec_ref(v___y_860_);
lean_dec(v___y_859_);
lean_dec_ref(v___y_858_);
lean_dec(v___y_857_);
lean_dec(v___y_856_);
lean_dec_ref(v___y_855_);
lean_dec_ref(v_as_851_);
lean_dec_ref(v_snd_849_);
return v_res_865_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__3(lean_object* v_snd_866_, lean_object* v_assignedMVars_867_, lean_object* v_as_868_, size_t v_sz_869_, size_t v_i_870_, lean_object* v_b_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_){
_start:
{
uint8_t v___x_880_; 
v___x_880_ = lean_usize_dec_lt(v_i_870_, v_sz_869_);
if (v___x_880_ == 0)
{
lean_object* v___x_881_; 
lean_dec_ref(v_assignedMVars_867_);
v___x_881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_881_, 0, v_b_871_);
return v___x_881_;
}
else
{
lean_object* v_a_882_; lean_object* v___x_883_; lean_object* v_fst_884_; lean_object* v_snd_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_912_; 
v_a_882_ = lean_array_uget_borrowed(v_as_868_, v_i_870_);
v___x_883_ = lean_st_ref_get(v_a_882_);
v_fst_884_ = lean_ctor_get(v_b_871_, 0);
v_snd_885_ = lean_ctor_get(v_b_871_, 1);
v_isSharedCheck_912_ = !lean_is_exclusive(v_b_871_);
if (v_isSharedCheck_912_ == 0)
{
v___x_887_ = v_b_871_;
v_isShared_888_ = v_isSharedCheck_912_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_snd_885_);
lean_inc(v_fst_884_);
lean_dec(v_b_871_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_912_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_889_; lean_object* v_elimRapp_890_; lean_object* v___x_891_; lean_object* v_children_892_; lean_object* v___x_894_; 
v___x_889_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_890_ = lean_ctor_get(v___x_889_, 3);
lean_inc_ref(v_elimRapp_890_);
v___x_891_ = lean_apply_1(v_elimRapp_890_, v___x_883_);
v_children_892_ = lean_ctor_get(v___x_891_, 2);
lean_inc_ref(v_children_892_);
lean_dec_ref(v___x_891_);
if (v_isShared_888_ == 0)
{
v___x_894_ = v___x_887_;
goto v_reusejp_893_;
}
else
{
lean_object* v_reuseFailAlloc_911_; 
v_reuseFailAlloc_911_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_911_, 0, v_fst_884_);
lean_ctor_set(v_reuseFailAlloc_911_, 1, v_snd_885_);
v___x_894_ = v_reuseFailAlloc_911_;
goto v_reusejp_893_;
}
v_reusejp_893_:
{
size_t v_sz_895_; size_t v___x_896_; lean_object* v___x_897_; 
v_sz_895_ = lean_array_size(v_children_892_);
v___x_896_ = ((size_t)0ULL);
lean_inc_ref(v_assignedMVars_867_);
v___x_897_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__2(v_snd_866_, v_assignedMVars_867_, v_children_892_, v_sz_895_, v___x_896_, v___x_894_, v___y_872_, v___y_873_, v___y_874_, v___y_875_, v___y_876_, v___y_877_, v___y_878_);
lean_dec_ref(v_children_892_);
if (lean_obj_tag(v___x_897_) == 0)
{
lean_object* v_a_898_; lean_object* v_fst_899_; lean_object* v_snd_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_910_; 
v_a_898_ = lean_ctor_get(v___x_897_, 0);
lean_inc(v_a_898_);
lean_dec_ref_known(v___x_897_, 1);
v_fst_899_ = lean_ctor_get(v_a_898_, 0);
v_snd_900_ = lean_ctor_get(v_a_898_, 1);
v_isSharedCheck_910_ = !lean_is_exclusive(v_a_898_);
if (v_isSharedCheck_910_ == 0)
{
v___x_902_ = v_a_898_;
v_isShared_903_ = v_isSharedCheck_910_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_snd_900_);
lean_inc(v_fst_899_);
lean_dec(v_a_898_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_910_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_905_; 
if (v_isShared_903_ == 0)
{
v___x_905_ = v___x_902_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_909_; 
v_reuseFailAlloc_909_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_909_, 0, v_fst_899_);
lean_ctor_set(v_reuseFailAlloc_909_, 1, v_snd_900_);
v___x_905_ = v_reuseFailAlloc_909_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
size_t v___x_906_; size_t v___x_907_; 
v___x_906_ = ((size_t)1ULL);
v___x_907_ = lean_usize_add(v_i_870_, v___x_906_);
v_i_870_ = v___x_907_;
v_b_871_ = v___x_905_;
goto _start;
}
}
}
else
{
lean_dec_ref(v_assignedMVars_867_);
return v___x_897_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__3___boxed(lean_object* v_snd_913_, lean_object* v_assignedMVars_914_, lean_object* v_as_915_, lean_object* v_sz_916_, lean_object* v_i_917_, lean_object* v_b_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_){
_start:
{
size_t v_sz_boxed_927_; size_t v_i_boxed_928_; lean_object* v_res_929_; 
v_sz_boxed_927_ = lean_unbox_usize(v_sz_916_);
lean_dec(v_sz_916_);
v_i_boxed_928_ = lean_unbox_usize(v_i_917_);
lean_dec(v_i_917_);
v_res_929_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__3(v_snd_913_, v_assignedMVars_914_, v_as_915_, v_sz_boxed_927_, v_i_boxed_928_, v_b_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_, v___y_923_, v___y_924_, v___y_925_);
lean_dec(v___y_925_);
lean_dec_ref(v___y_924_);
lean_dec(v___y_923_);
lean_dec_ref(v___y_922_);
lean_dec(v___y_921_);
lean_dec(v___y_920_);
lean_dec_ref(v___y_919_);
lean_dec_ref(v_as_915_);
lean_dec_ref(v_snd_913_);
return v_res_929_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoalsToCopy(lean_object* v_assignedMVars_930_, lean_object* v_start_931_, lean_object* v_a_932_, lean_object* v_a_933_, lean_object* v_a_934_, lean_object* v_a_935_, lean_object* v_a_936_, lean_object* v_a_937_, lean_object* v_a_938_){
_start:
{
lean_object* v___x_940_; 
lean_inc_ref(v_assignedMVars_930_);
v___x_940_ = lp_aesop_Aesop_findPathForAssignedMVars(v_assignedMVars_930_, v_start_931_, v_a_932_, v_a_933_, v_a_934_, v_a_935_, v_a_936_, v_a_937_, v_a_938_);
if (lean_obj_tag(v___x_940_) == 0)
{
lean_object* v_a_941_; lean_object* v_fst_942_; lean_object* v_snd_943_; lean_object* v___x_944_; size_t v_sz_945_; size_t v___x_946_; lean_object* v___x_947_; 
v_a_941_ = lean_ctor_get(v___x_940_, 0);
lean_inc(v_a_941_);
lean_dec_ref_known(v___x_940_, 1);
v_fst_942_ = lean_ctor_get(v_a_941_, 0);
lean_inc(v_fst_942_);
v_snd_943_ = lean_ctor_get(v_a_941_, 1);
lean_inc(v_snd_943_);
lean_dec(v_a_941_);
v___x_944_ = lean_obj_once(&lp_aesop_Aesop_findPathForAssignedMVars___closed__5, &lp_aesop_Aesop_findPathForAssignedMVars___closed__5_once, _init_lp_aesop_Aesop_findPathForAssignedMVars___closed__5);
v_sz_945_ = lean_array_size(v_fst_942_);
v___x_946_ = ((size_t)0ULL);
v___x_947_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__3(v_snd_943_, v_assignedMVars_930_, v_fst_942_, v_sz_945_, v___x_946_, v___x_944_, v_a_932_, v_a_933_, v_a_934_, v_a_935_, v_a_936_, v_a_937_, v_a_938_);
lean_dec(v_fst_942_);
lean_dec(v_snd_943_);
if (lean_obj_tag(v___x_947_) == 0)
{
lean_object* v_a_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_956_; 
v_a_948_ = lean_ctor_get(v___x_947_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_947_);
if (v_isSharedCheck_956_ == 0)
{
v___x_950_ = v___x_947_;
v_isShared_951_ = v_isSharedCheck_956_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_a_948_);
lean_dec(v___x_947_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_956_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v_fst_952_; lean_object* v___x_954_; 
v_fst_952_ = lean_ctor_get(v_a_948_, 0);
lean_inc(v_fst_952_);
lean_dec(v_a_948_);
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 0, v_fst_952_);
v___x_954_ = v___x_950_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v_fst_952_);
v___x_954_ = v_reuseFailAlloc_955_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
return v___x_954_;
}
}
}
else
{
lean_object* v_a_957_; lean_object* v___x_959_; uint8_t v_isShared_960_; uint8_t v_isSharedCheck_964_; 
v_a_957_ = lean_ctor_get(v___x_947_, 0);
v_isSharedCheck_964_ = !lean_is_exclusive(v___x_947_);
if (v_isSharedCheck_964_ == 0)
{
v___x_959_ = v___x_947_;
v_isShared_960_ = v_isSharedCheck_964_;
goto v_resetjp_958_;
}
else
{
lean_inc(v_a_957_);
lean_dec(v___x_947_);
v___x_959_ = lean_box(0);
v_isShared_960_ = v_isSharedCheck_964_;
goto v_resetjp_958_;
}
v_resetjp_958_:
{
lean_object* v___x_962_; 
if (v_isShared_960_ == 0)
{
v___x_962_ = v___x_959_;
goto v_reusejp_961_;
}
else
{
lean_object* v_reuseFailAlloc_963_; 
v_reuseFailAlloc_963_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_963_, 0, v_a_957_);
v___x_962_ = v_reuseFailAlloc_963_;
goto v_reusejp_961_;
}
v_reusejp_961_:
{
return v___x_962_;
}
}
}
}
else
{
lean_object* v_a_965_; lean_object* v___x_967_; uint8_t v_isShared_968_; uint8_t v_isSharedCheck_972_; 
lean_dec_ref(v_assignedMVars_930_);
v_a_965_ = lean_ctor_get(v___x_940_, 0);
v_isSharedCheck_972_ = !lean_is_exclusive(v___x_940_);
if (v_isSharedCheck_972_ == 0)
{
v___x_967_ = v___x_940_;
v_isShared_968_ = v_isSharedCheck_972_;
goto v_resetjp_966_;
}
else
{
lean_inc(v_a_965_);
lean_dec(v___x_940_);
v___x_967_ = lean_box(0);
v_isShared_968_ = v_isSharedCheck_972_;
goto v_resetjp_966_;
}
v_resetjp_966_:
{
lean_object* v___x_970_; 
if (v_isShared_968_ == 0)
{
v___x_970_ = v___x_967_;
goto v_reusejp_969_;
}
else
{
lean_object* v_reuseFailAlloc_971_; 
v_reuseFailAlloc_971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_971_, 0, v_a_965_);
v___x_970_ = v_reuseFailAlloc_971_;
goto v_reusejp_969_;
}
v_reusejp_969_:
{
return v___x_970_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getGoalsToCopy___boxed(lean_object* v_assignedMVars_973_, lean_object* v_start_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_, lean_object* v_a_978_, lean_object* v_a_979_, lean_object* v_a_980_, lean_object* v_a_981_, lean_object* v_a_982_){
_start:
{
lean_object* v_res_983_; 
v_res_983_ = lp_aesop_Aesop_getGoalsToCopy(v_assignedMVars_973_, v_start_974_, v_a_975_, v_a_976_, v_a_977_, v_a_978_, v_a_979_, v_a_980_, v_a_981_);
lean_dec(v_a_981_);
lean_dec_ref(v_a_980_);
lean_dec(v_a_979_);
lean_dec_ref(v_a_978_);
lean_dec(v_a_977_);
lean_dec(v_a_976_);
lean_dec_ref(v_a_975_);
return v_res_983_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0(lean_object* v_00_u03b2_984_, lean_object* v_m_985_, lean_object* v_a_986_){
_start:
{
uint8_t v___x_987_; 
v___x_987_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___redArg(v_m_985_, v_a_986_);
return v___x_987_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0___boxed(lean_object* v_00_u03b2_988_, lean_object* v_m_989_, lean_object* v_a_990_){
_start:
{
uint8_t v_res_991_; lean_object* v_r_992_; 
v_res_991_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_getGoalsToCopy_spec__0(v_00_u03b2_988_, v_m_989_, v_a_990_);
lean_dec(v_a_990_);
lean_dec_ref(v_m_989_);
v_r_992_ = lean_box(v_res_991_);
return v_r_992_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1(lean_object* v_snd_993_, lean_object* v_assignedMVars_994_, lean_object* v_as_995_, size_t v_sz_996_, size_t v_i_997_, lean_object* v_b_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___redArg(v_snd_993_, v_assignedMVars_994_, v_as_995_, v_sz_996_, v_i_997_, v_b_998_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1___boxed(lean_object* v_snd_1008_, lean_object* v_assignedMVars_1009_, lean_object* v_as_1010_, lean_object* v_sz_1011_, lean_object* v_i_1012_, lean_object* v_b_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_){
_start:
{
size_t v_sz_boxed_1022_; size_t v_i_boxed_1023_; lean_object* v_res_1024_; 
v_sz_boxed_1022_ = lean_unbox_usize(v_sz_1011_);
lean_dec(v_sz_1011_);
v_i_boxed_1023_ = lean_unbox_usize(v_i_1012_);
lean_dec(v_i_1012_);
v_res_1024_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_getGoalsToCopy_spec__1(v_snd_1008_, v_assignedMVars_1009_, v_as_1010_, v_sz_boxed_1022_, v_i_boxed_1023_, v_b_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_);
lean_dec(v___y_1020_);
lean_dec_ref(v___y_1019_);
lean_dec(v___y_1018_);
lean_dec_ref(v___y_1017_);
lean_dec(v___y_1016_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
lean_dec_ref(v_as_1010_);
lean_dec_ref(v_snd_1008_);
return v_res_1024_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg(lean_object* v_s_1025_, lean_object* v_x_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_){
_start:
{
lean_object* v___x_1035_; 
v___x_1035_ = l_Lean_Meta_saveState___redArg(v___y_1031_, v___y_1033_);
if (lean_obj_tag(v___x_1035_) == 0)
{
lean_object* v_a_1036_; lean_object* v_a_1038_; lean_object* v___x_1056_; 
v_a_1036_ = lean_ctor_get(v___x_1035_, 0);
lean_inc(v_a_1036_);
lean_dec_ref_known(v___x_1035_, 1);
v___x_1056_ = l_Lean_Meta_SavedState_restore___redArg(v_s_1025_, v___y_1031_, v___y_1033_);
if (lean_obj_tag(v___x_1056_) == 0)
{
lean_object* v___x_1057_; 
lean_dec_ref_known(v___x_1056_, 1);
lean_inc(v___y_1033_);
lean_inc_ref(v___y_1032_);
lean_inc(v___y_1031_);
lean_inc_ref(v___y_1030_);
lean_inc(v___y_1029_);
lean_inc(v___y_1028_);
lean_inc_ref(v___y_1027_);
v___x_1057_ = lean_apply_8(v_x_1026_, v___y_1027_, v___y_1028_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, lean_box(0));
if (lean_obj_tag(v___x_1057_) == 0)
{
lean_object* v_a_1058_; lean_object* v___x_1059_; 
v_a_1058_ = lean_ctor_get(v___x_1057_, 0);
lean_inc(v_a_1058_);
lean_dec_ref_known(v___x_1057_, 1);
v___x_1059_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1036_, v___y_1031_, v___y_1033_);
lean_dec(v_a_1036_);
if (lean_obj_tag(v___x_1059_) == 0)
{
lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1066_; 
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1066_ == 0)
{
lean_object* v_unused_1067_; 
v_unused_1067_ = lean_ctor_get(v___x_1059_, 0);
lean_dec(v_unused_1067_);
v___x_1061_ = v___x_1059_;
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
else
{
lean_dec(v___x_1059_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v___x_1064_; 
if (v_isShared_1062_ == 0)
{
lean_ctor_set(v___x_1061_, 0, v_a_1058_);
v___x_1064_ = v___x_1061_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v_a_1058_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
}
else
{
lean_object* v_a_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1075_; 
lean_dec(v_a_1058_);
v_a_1068_ = lean_ctor_get(v___x_1059_, 0);
v_isSharedCheck_1075_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1075_ == 0)
{
v___x_1070_ = v___x_1059_;
v_isShared_1071_ = v_isSharedCheck_1075_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_a_1068_);
lean_dec(v___x_1059_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1075_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v___x_1073_; 
if (v_isShared_1071_ == 0)
{
v___x_1073_ = v___x_1070_;
goto v_reusejp_1072_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v_a_1068_);
v___x_1073_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1072_;
}
v_reusejp_1072_:
{
return v___x_1073_;
}
}
}
}
else
{
lean_object* v_a_1076_; 
v_a_1076_ = lean_ctor_get(v___x_1057_, 0);
lean_inc(v_a_1076_);
lean_dec_ref_known(v___x_1057_, 1);
v_a_1038_ = v_a_1076_;
goto v___jp_1037_;
}
}
else
{
lean_object* v_a_1077_; 
lean_dec_ref(v_x_1026_);
v_a_1077_ = lean_ctor_get(v___x_1056_, 0);
lean_inc(v_a_1077_);
lean_dec_ref_known(v___x_1056_, 1);
v_a_1038_ = v_a_1077_;
goto v___jp_1037_;
}
v___jp_1037_:
{
lean_object* v___x_1039_; 
v___x_1039_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1036_, v___y_1031_, v___y_1033_);
lean_dec(v_a_1036_);
if (lean_obj_tag(v___x_1039_) == 0)
{
lean_object* v___x_1041_; uint8_t v_isShared_1042_; uint8_t v_isSharedCheck_1046_; 
v_isSharedCheck_1046_ = !lean_is_exclusive(v___x_1039_);
if (v_isSharedCheck_1046_ == 0)
{
lean_object* v_unused_1047_; 
v_unused_1047_ = lean_ctor_get(v___x_1039_, 0);
lean_dec(v_unused_1047_);
v___x_1041_ = v___x_1039_;
v_isShared_1042_ = v_isSharedCheck_1046_;
goto v_resetjp_1040_;
}
else
{
lean_dec(v___x_1039_);
v___x_1041_ = lean_box(0);
v_isShared_1042_ = v_isSharedCheck_1046_;
goto v_resetjp_1040_;
}
v_resetjp_1040_:
{
lean_object* v___x_1044_; 
if (v_isShared_1042_ == 0)
{
lean_ctor_set_tag(v___x_1041_, 1);
lean_ctor_set(v___x_1041_, 0, v_a_1038_);
v___x_1044_ = v___x_1041_;
goto v_reusejp_1043_;
}
else
{
lean_object* v_reuseFailAlloc_1045_; 
v_reuseFailAlloc_1045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1045_, 0, v_a_1038_);
v___x_1044_ = v_reuseFailAlloc_1045_;
goto v_reusejp_1043_;
}
v_reusejp_1043_:
{
return v___x_1044_;
}
}
}
else
{
lean_object* v_a_1048_; lean_object* v___x_1050_; uint8_t v_isShared_1051_; uint8_t v_isSharedCheck_1055_; 
lean_dec_ref(v_a_1038_);
v_a_1048_ = lean_ctor_get(v___x_1039_, 0);
v_isSharedCheck_1055_ = !lean_is_exclusive(v___x_1039_);
if (v_isSharedCheck_1055_ == 0)
{
v___x_1050_ = v___x_1039_;
v_isShared_1051_ = v_isSharedCheck_1055_;
goto v_resetjp_1049_;
}
else
{
lean_inc(v_a_1048_);
lean_dec(v___x_1039_);
v___x_1050_ = lean_box(0);
v_isShared_1051_ = v_isSharedCheck_1055_;
goto v_resetjp_1049_;
}
v_resetjp_1049_:
{
lean_object* v___x_1053_; 
if (v_isShared_1051_ == 0)
{
v___x_1053_ = v___x_1050_;
goto v_reusejp_1052_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v_a_1048_);
v___x_1053_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1052_;
}
v_reusejp_1052_:
{
return v___x_1053_;
}
}
}
}
}
else
{
lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1085_; 
lean_dec_ref(v_x_1026_);
v_a_1078_ = lean_ctor_get(v___x_1035_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_1035_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1080_ = v___x_1035_;
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1035_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1083_; 
if (v_isShared_1081_ == 0)
{
v___x_1083_ = v___x_1080_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_a_1078_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
return v___x_1083_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg___boxed(lean_object* v_s_1086_, lean_object* v_x_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_){
_start:
{
lean_object* v_res_1096_; 
v_res_1096_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg(v_s_1086_, v_x_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
lean_dec(v___y_1094_);
lean_dec_ref(v___y_1093_);
lean_dec(v___y_1092_);
lean_dec_ref(v___y_1091_);
lean_dec(v___y_1090_);
lean_dec(v___y_1089_);
lean_dec_ref(v___y_1088_);
lean_dec_ref(v_s_1086_);
return v_res_1096_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1(lean_object* v_00_u03b1_1097_, lean_object* v_s_1098_, lean_object* v_x_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_){
_start:
{
lean_object* v___x_1108_; 
v___x_1108_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg(v_s_1098_, v_x_1099_, v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_);
return v___x_1108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___boxed(lean_object* v_00_u03b1_1109_, lean_object* v_s_1110_, lean_object* v_x_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_){
_start:
{
lean_object* v_res_1120_; 
v_res_1120_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1(v_00_u03b1_1109_, v_s_1110_, v_x_1111_, v___y_1112_, v___y_1113_, v___y_1114_, v___y_1115_, v___y_1116_, v___y_1117_, v___y_1118_);
lean_dec(v___y_1118_);
lean_dec_ref(v___y_1117_);
lean_dec(v___y_1116_);
lean_dec_ref(v___y_1115_);
lean_dec(v___y_1114_);
lean_dec(v___y_1113_);
lean_dec_ref(v___y_1112_);
lean_dec_ref(v_s_1110_);
return v_res_1120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__0(lean_object* v_x_1121_, lean_object* v_x_1122_){
_start:
{
if (lean_obj_tag(v_x_1122_) == 0)
{
return v_x_1121_;
}
else
{
lean_object* v_key_1123_; lean_object* v_tail_1124_; lean_object* v___x_1125_; 
v_key_1123_ = lean_ctor_get(v_x_1122_, 0);
lean_inc(v_key_1123_);
v_tail_1124_ = lean_ctor_get(v_x_1122_, 2);
lean_inc(v_tail_1124_);
lean_dec_ref_known(v_x_1122_, 3);
v___x_1125_ = lean_array_push(v_x_1121_, v_key_1123_);
v_x_1121_ = v___x_1125_;
v_x_1122_ = v_tail_1124_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__1(lean_object* v_as_1127_, size_t v_i_1128_, size_t v_stop_1129_, lean_object* v_b_1130_){
_start:
{
uint8_t v___x_1131_; 
v___x_1131_ = lean_usize_dec_eq(v_i_1128_, v_stop_1129_);
if (v___x_1131_ == 0)
{
lean_object* v___x_1132_; lean_object* v___x_1133_; size_t v___x_1134_; size_t v___x_1135_; 
v___x_1132_ = lean_array_uget_borrowed(v_as_1127_, v_i_1128_);
lean_inc(v___x_1132_);
v___x_1133_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__0(v_b_1130_, v___x_1132_);
v___x_1134_ = ((size_t)1ULL);
v___x_1135_ = lean_usize_add(v_i_1128_, v___x_1134_);
v_i_1128_ = v___x_1135_;
v_b_1130_ = v___x_1133_;
goto _start;
}
else
{
return v_b_1130_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__1___boxed(lean_object* v_as_1137_, lean_object* v_i_1138_, lean_object* v_stop_1139_, lean_object* v_b_1140_){
_start:
{
size_t v_i_boxed_1141_; size_t v_stop_boxed_1142_; lean_object* v_res_1143_; 
v_i_boxed_1141_ = lean_unbox_usize(v_i_1138_);
lean_dec(v_i_1138_);
v_stop_boxed_1142_ = lean_unbox_usize(v_stop_1139_);
lean_dec(v_stop_1139_);
v_res_1143_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__1(v_as_1137_, v_i_boxed_1141_, v_stop_boxed_1142_, v_b_1140_);
lean_dec_ref(v_as_1137_);
return v_res_1143_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0(lean_object* v_xs_1144_){
_start:
{
lean_object* v_size_1145_; lean_object* v_buckets_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; uint8_t v___x_1150_; 
v_size_1145_ = lean_ctor_get(v_xs_1144_, 0);
v_buckets_1146_ = lean_ctor_get(v_xs_1144_, 1);
v___x_1147_ = lean_mk_empty_array_with_capacity(v_size_1145_);
v___x_1148_ = lean_unsigned_to_nat(0u);
v___x_1149_ = lean_array_get_size(v_buckets_1146_);
v___x_1150_ = lean_nat_dec_lt(v___x_1148_, v___x_1149_);
if (v___x_1150_ == 0)
{
return v___x_1147_;
}
else
{
uint8_t v___x_1151_; 
v___x_1151_ = lean_nat_dec_le(v___x_1149_, v___x_1149_);
if (v___x_1151_ == 0)
{
if (v___x_1150_ == 0)
{
return v___x_1147_;
}
else
{
size_t v___x_1152_; size_t v___x_1153_; lean_object* v___x_1154_; 
v___x_1152_ = ((size_t)0ULL);
v___x_1153_ = lean_usize_of_nat(v___x_1149_);
v___x_1154_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__1(v_buckets_1146_, v___x_1152_, v___x_1153_, v___x_1147_);
return v___x_1154_;
}
}
else
{
size_t v___x_1155_; size_t v___x_1156_; lean_object* v___x_1157_; 
v___x_1155_ = ((size_t)0ULL);
v___x_1156_ = lean_usize_of_nat(v___x_1149_);
v___x_1157_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0_spec__1(v_buckets_1146_, v___x_1155_, v___x_1156_, v___x_1147_);
return v___x_1157_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0___boxed(lean_object* v_xs_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0(v_xs_1158_);
lean_dec_ref(v_xs_1158_);
return v_res_1159_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___lam__0(lean_object* v_start_1160_, lean_object* v_val_1161_, lean_object* v_ruleSet_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_){
_start:
{
lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v_elimGoal_1173_; lean_object* v___x_1174_; lean_object* v_preNormGoal_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; 
v___x_1171_ = lean_st_ref_get(v_start_1160_);
v___x_1172_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1173_ = lean_ctor_get(v___x_1172_, 1);
lean_inc_ref(v_elimGoal_1173_);
v___x_1174_ = lean_apply_1(v_elimGoal_1173_, v_val_1161_);
v_preNormGoal_1175_ = lean_ctor_get(v___x_1174_, 5);
lean_inc_n(v_preNormGoal_1175_, 2);
lean_dec_ref(v___x_1174_);
lean_inc(v___x_1171_);
v___x_1176_ = lp_aesop_Aesop_Goal_currentGoal(v___x_1171_);
v___x_1177_ = lp_aesop_Aesop_diffGoals(v___x_1176_, v_preNormGoal_1175_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_);
if (lean_obj_tag(v___x_1177_) == 0)
{
lean_object* v_a_1178_; lean_object* v___x_1179_; lean_object* v_forwardState_1180_; lean_object* v_forwardRuleMatches_1181_; lean_object* v___x_1182_; 
v_a_1178_ = lean_ctor_get(v___x_1177_, 0);
lean_inc_n(v_a_1178_, 2);
lean_dec_ref_known(v___x_1177_, 1);
lean_inc_ref(v_elimGoal_1173_);
v___x_1179_ = lean_apply_1(v_elimGoal_1173_, v___x_1171_);
v_forwardState_1180_ = lean_ctor_get(v___x_1179_, 8);
lean_inc_ref(v_forwardState_1180_);
v_forwardRuleMatches_1181_ = lean_ctor_get(v___x_1179_, 9);
lean_inc_ref(v_forwardRuleMatches_1181_);
lean_dec_ref(v___x_1179_);
v___x_1182_ = lp_aesop_Aesop_ForwardState_applyGoalDiff(v_ruleSet_1162_, v_a_1178_, v_forwardState_1180_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_);
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_object* v_a_1183_; uint8_t v___x_1184_; lean_object* v___x_1185_; 
v_a_1183_ = lean_ctor_get(v___x_1182_, 0);
lean_inc(v_a_1183_);
lean_dec_ref_known(v___x_1182_, 1);
v___x_1184_ = 0;
v___x_1185_ = l_Lean_MVarId_getMVarDependencies(v_preNormGoal_1175_, v___x_1184_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_);
if (lean_obj_tag(v___x_1185_) == 0)
{
lean_object* v_a_1186_; lean_object* v___x_1188_; uint8_t v_isShared_1189_; uint8_t v_isSharedCheck_1199_; 
v_a_1186_ = lean_ctor_get(v___x_1185_, 0);
v_isSharedCheck_1199_ = !lean_is_exclusive(v___x_1185_);
if (v_isSharedCheck_1199_ == 0)
{
v___x_1188_ = v___x_1185_;
v_isShared_1189_ = v_isSharedCheck_1199_;
goto v_resetjp_1187_;
}
else
{
lean_inc(v_a_1186_);
lean_dec(v___x_1185_);
v___x_1188_ = lean_box(0);
v_isShared_1189_ = v_isSharedCheck_1199_;
goto v_resetjp_1187_;
}
v_resetjp_1187_:
{
lean_object* v_removedFVars_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1197_; 
v_removedFVars_1190_ = lean_ctor_get(v_a_1178_, 3);
lean_inc_ref(v_removedFVars_1190_);
lean_dec(v_a_1178_);
v___x_1191_ = ((lean_object*)(lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___closed__0));
v___x_1192_ = lp_aesop_Aesop_ForwardRuleMatches_update(v___x_1191_, v_removedFVars_1190_, v___x_1191_, v_forwardRuleMatches_1181_);
v___x_1193_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___at___00Aesop_copyGoals_spec__0(v_a_1186_);
lean_dec(v_a_1186_);
v___x_1194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1194_, 0, v___x_1192_);
lean_ctor_set(v___x_1194_, 1, v___x_1193_);
v___x_1195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1195_, 0, v_a_1183_);
lean_ctor_set(v___x_1195_, 1, v___x_1194_);
if (v_isShared_1189_ == 0)
{
lean_ctor_set(v___x_1188_, 0, v___x_1195_);
v___x_1197_ = v___x_1188_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v___x_1195_);
v___x_1197_ = v_reuseFailAlloc_1198_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
return v___x_1197_;
}
}
}
else
{
lean_object* v_a_1200_; lean_object* v___x_1202_; uint8_t v_isShared_1203_; uint8_t v_isSharedCheck_1207_; 
lean_dec(v_a_1183_);
lean_dec_ref(v_forwardRuleMatches_1181_);
lean_dec(v_a_1178_);
v_a_1200_ = lean_ctor_get(v___x_1185_, 0);
v_isSharedCheck_1207_ = !lean_is_exclusive(v___x_1185_);
if (v_isSharedCheck_1207_ == 0)
{
v___x_1202_ = v___x_1185_;
v_isShared_1203_ = v_isSharedCheck_1207_;
goto v_resetjp_1201_;
}
else
{
lean_inc(v_a_1200_);
lean_dec(v___x_1185_);
v___x_1202_ = lean_box(0);
v_isShared_1203_ = v_isSharedCheck_1207_;
goto v_resetjp_1201_;
}
v_resetjp_1201_:
{
lean_object* v___x_1205_; 
if (v_isShared_1203_ == 0)
{
v___x_1205_ = v___x_1202_;
goto v_reusejp_1204_;
}
else
{
lean_object* v_reuseFailAlloc_1206_; 
v_reuseFailAlloc_1206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1206_, 0, v_a_1200_);
v___x_1205_ = v_reuseFailAlloc_1206_;
goto v_reusejp_1204_;
}
v_reusejp_1204_:
{
return v___x_1205_;
}
}
}
}
else
{
lean_object* v_a_1208_; lean_object* v___x_1210_; uint8_t v_isShared_1211_; uint8_t v_isSharedCheck_1215_; 
lean_dec_ref(v_forwardRuleMatches_1181_);
lean_dec(v_a_1178_);
lean_dec(v_preNormGoal_1175_);
v_a_1208_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1215_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1215_ == 0)
{
v___x_1210_ = v___x_1182_;
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
else
{
lean_inc(v_a_1208_);
lean_dec(v___x_1182_);
v___x_1210_ = lean_box(0);
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
v_resetjp_1209_:
{
lean_object* v___x_1213_; 
if (v_isShared_1211_ == 0)
{
v___x_1213_ = v___x_1210_;
goto v_reusejp_1212_;
}
else
{
lean_object* v_reuseFailAlloc_1214_; 
v_reuseFailAlloc_1214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1214_, 0, v_a_1208_);
v___x_1213_ = v_reuseFailAlloc_1214_;
goto v_reusejp_1212_;
}
v_reusejp_1212_:
{
return v___x_1213_;
}
}
}
}
else
{
lean_object* v_a_1216_; lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1223_; 
lean_dec(v_preNormGoal_1175_);
lean_dec(v___x_1171_);
lean_dec_ref(v_ruleSet_1162_);
v_a_1216_ = lean_ctor_get(v___x_1177_, 0);
v_isSharedCheck_1223_ = !lean_is_exclusive(v___x_1177_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1218_ = v___x_1177_;
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
else
{
lean_inc(v_a_1216_);
lean_dec(v___x_1177_);
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
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___lam__0___boxed(lean_object* v_start_1224_, lean_object* v_val_1225_, lean_object* v_ruleSet_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_){
_start:
{
lean_object* v_res_1235_; 
v_res_1235_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___lam__0(v_start_1224_, v_val_1225_, v_ruleSet_1226_, v___y_1227_, v___y_1228_, v___y_1229_, v___y_1230_, v___y_1231_, v___y_1232_, v___y_1233_);
lean_dec(v___y_1233_);
lean_dec_ref(v___y_1232_);
lean_dec(v___y_1231_);
lean_dec_ref(v___y_1230_);
lean_dec(v___y_1229_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
lean_dec(v_start_1224_);
return v_res_1235_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1236_; 
v___x_1236_ = l_Subarray_empty(lean_box(0));
return v___x_1236_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2(lean_object* v_start_1237_, lean_object* v_parentMetaState_1238_, lean_object* v_depth_1239_, double v_parentSuccessProbability_1240_, size_t v_sz_1241_, size_t v_i_1242_, lean_object* v_bs_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_){
_start:
{
uint8_t v___x_1252_; 
v___x_1252_ = lean_usize_dec_lt(v_i_1242_, v_sz_1241_);
if (v___x_1252_ == 0)
{
lean_object* v___x_1253_; 
lean_dec(v_depth_1239_);
lean_dec(v_start_1237_);
v___x_1253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1253_, 0, v_bs_1243_);
return v___x_1253_;
}
else
{
lean_object* v_v_1254_; lean_object* v___x_1255_; lean_object* v_currentIteration_1256_; lean_object* v_ruleSet_1257_; lean_object* v___f_1258_; lean_object* v___x_1259_; 
v_v_1254_ = lean_array_uget_borrowed(v_bs_1243_, v_i_1242_);
v___x_1255_ = lean_st_ref_get(v_v_1254_);
v_currentIteration_1256_ = lean_ctor_get(v___y_1244_, 0);
v_ruleSet_1257_ = lean_ctor_get(v___y_1244_, 1);
lean_inc_ref(v_ruleSet_1257_);
lean_inc(v___x_1255_);
lean_inc(v_start_1237_);
v___f_1258_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___lam__0___boxed), 11, 3);
lean_closure_set(v___f_1258_, 0, v_start_1237_);
lean_closure_set(v___f_1258_, 1, v___x_1255_);
lean_closure_set(v___f_1258_, 2, v_ruleSet_1257_);
v___x_1259_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg(v_parentMetaState_1238_, v___f_1258_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_);
if (lean_obj_tag(v___x_1259_) == 0)
{
lean_object* v_a_1260_; lean_object* v_snd_1261_; lean_object* v_fst_1262_; lean_object* v_fst_1263_; lean_object* v_snd_1264_; lean_object* v___x_1266_; uint8_t v_isShared_1267_; uint8_t v_isSharedCheck_1321_; 
v_a_1260_ = lean_ctor_get(v___x_1259_, 0);
lean_inc(v_a_1260_);
lean_dec_ref_known(v___x_1259_, 1);
v_snd_1261_ = lean_ctor_get(v_a_1260_, 1);
lean_inc(v_snd_1261_);
v_fst_1262_ = lean_ctor_get(v_a_1260_, 0);
lean_inc(v_fst_1262_);
lean_dec(v_a_1260_);
v_fst_1263_ = lean_ctor_get(v_snd_1261_, 0);
v_snd_1264_ = lean_ctor_get(v_snd_1261_, 1);
v_isSharedCheck_1321_ = !lean_is_exclusive(v_snd_1261_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1266_ = v_snd_1261_;
v_isShared_1267_ = v_isSharedCheck_1321_;
goto v_resetjp_1265_;
}
else
{
lean_inc(v_snd_1264_);
lean_inc(v_fst_1263_);
lean_dec(v_snd_1261_);
v___x_1266_ = lean_box(0);
v_isShared_1267_ = v_isSharedCheck_1321_;
goto v_resetjp_1265_;
}
v_resetjp_1265_:
{
lean_object* v___x_1268_; 
v___x_1268_ = lp_aesop_Aesop_getAndIncrementNextGoalId___redArg(v___y_1245_);
if (lean_obj_tag(v___x_1268_) == 0)
{
lean_object* v_a_1269_; lean_object* v___x_1270_; lean_object* v_introGoal_1271_; lean_object* v_elimGoal_1272_; lean_object* v___x_1273_; lean_object* v_id_1274_; lean_object* v_preNormGoal_1275_; lean_object* v___x_1277_; uint8_t v_isShared_1278_; uint8_t v_isSharedCheck_1300_; 
v_a_1269_ = lean_ctor_get(v___x_1268_, 0);
lean_inc(v_a_1269_);
lean_dec_ref_known(v___x_1268_, 1);
v___x_1270_ = lp_aesop_Aesop_treeImpl;
v_introGoal_1271_ = lean_ctor_get(v___x_1270_, 0);
v_elimGoal_1272_ = lean_ctor_get(v___x_1270_, 1);
lean_inc_ref(v_elimGoal_1272_);
lean_inc(v___x_1255_);
v___x_1273_ = lean_apply_1(v_elimGoal_1272_, v___x_1255_);
v_id_1274_ = lean_ctor_get(v___x_1273_, 0);
v_preNormGoal_1275_ = lean_ctor_get(v___x_1273_, 5);
v_isSharedCheck_1300_ = !lean_is_exclusive(v___x_1273_);
if (v_isSharedCheck_1300_ == 0)
{
lean_object* v_unused_1301_; lean_object* v_unused_1302_; lean_object* v_unused_1303_; lean_object* v_unused_1304_; lean_object* v_unused_1305_; lean_object* v_unused_1306_; lean_object* v_unused_1307_; lean_object* v_unused_1308_; lean_object* v_unused_1309_; lean_object* v_unused_1310_; lean_object* v_unused_1311_; lean_object* v_unused_1312_; 
v_unused_1301_ = lean_ctor_get(v___x_1273_, 13);
lean_dec(v_unused_1301_);
v_unused_1302_ = lean_ctor_get(v___x_1273_, 12);
lean_dec(v_unused_1302_);
v_unused_1303_ = lean_ctor_get(v___x_1273_, 11);
lean_dec(v_unused_1303_);
v_unused_1304_ = lean_ctor_get(v___x_1273_, 10);
lean_dec(v_unused_1304_);
v_unused_1305_ = lean_ctor_get(v___x_1273_, 9);
lean_dec(v_unused_1305_);
v_unused_1306_ = lean_ctor_get(v___x_1273_, 8);
lean_dec(v_unused_1306_);
v_unused_1307_ = lean_ctor_get(v___x_1273_, 7);
lean_dec(v_unused_1307_);
v_unused_1308_ = lean_ctor_get(v___x_1273_, 6);
lean_dec(v_unused_1308_);
v_unused_1309_ = lean_ctor_get(v___x_1273_, 4);
lean_dec(v_unused_1309_);
v_unused_1310_ = lean_ctor_get(v___x_1273_, 3);
lean_dec(v_unused_1310_);
v_unused_1311_ = lean_ctor_get(v___x_1273_, 2);
lean_dec(v_unused_1311_);
v_unused_1312_ = lean_ctor_get(v___x_1273_, 1);
lean_dec(v_unused_1312_);
v___x_1277_ = v___x_1273_;
v_isShared_1278_ = v_isSharedCheck_1300_;
goto v_resetjp_1276_;
}
else
{
lean_inc(v_preNormGoal_1275_);
lean_inc(v_id_1274_);
lean_dec(v___x_1273_);
v___x_1277_ = lean_box(0);
v_isShared_1278_ = v_isSharedCheck_1300_;
goto v_resetjp_1276_;
}
v_resetjp_1276_:
{
lean_object* v___x_1279_; lean_object* v_bs_x27_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1285_; 
v___x_1279_ = lean_unsigned_to_nat(0u);
v_bs_x27_1280_ = lean_array_uset(v_bs_1243_, v_i_1242_, v___x_1279_);
v___x_1281_ = lean_box(0);
v___x_1282_ = ((lean_object*)(lp_aesop_Aesop_findPathForAssignedMVars___closed__0));
v___x_1283_ = lp_aesop_Aesop_Goal_originalGoalId(v___x_1255_);
if (v_isShared_1267_ == 0)
{
lean_ctor_set_tag(v___x_1266_, 1);
lean_ctor_set(v___x_1266_, 1, v___x_1283_);
lean_ctor_set(v___x_1266_, 0, v_id_1274_);
v___x_1285_ = v___x_1266_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1299_; 
v_reuseFailAlloc_1299_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1299_, 0, v_id_1274_);
lean_ctor_set(v_reuseFailAlloc_1299_, 1, v___x_1283_);
v___x_1285_ = v_reuseFailAlloc_1299_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
uint8_t v___x_1286_; uint8_t v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1292_; 
v___x_1286_ = 0;
v___x_1287_ = 0;
v___x_1288_ = lean_box(0);
v___x_1289_ = lp_aesop_Aesop_Iteration_none;
v___x_1290_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0);
lean_inc(v_currentIteration_1256_);
lean_inc(v_depth_1239_);
if (v_isShared_1278_ == 0)
{
lean_ctor_set(v___x_1277_, 13, v___x_1282_);
lean_ctor_set(v___x_1277_, 12, v___x_1290_);
lean_ctor_set(v___x_1277_, 11, v___x_1289_);
lean_ctor_set(v___x_1277_, 10, v_currentIteration_1256_);
lean_ctor_set(v___x_1277_, 9, v_fst_1263_);
lean_ctor_set(v___x_1277_, 8, v_fst_1262_);
lean_ctor_set(v___x_1277_, 7, v_snd_1264_);
lean_ctor_set(v___x_1277_, 6, v___x_1288_);
lean_ctor_set(v___x_1277_, 4, v_depth_1239_);
lean_ctor_set(v___x_1277_, 3, v___x_1285_);
lean_ctor_set(v___x_1277_, 2, v___x_1282_);
lean_ctor_set(v___x_1277_, 1, v___x_1281_);
lean_ctor_set(v___x_1277_, 0, v_a_1269_);
v___x_1292_ = v___x_1277_;
goto v_reusejp_1291_;
}
else
{
lean_object* v_reuseFailAlloc_1298_; 
v_reuseFailAlloc_1298_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1298_, 0, v_a_1269_);
lean_ctor_set(v_reuseFailAlloc_1298_, 1, v___x_1281_);
lean_ctor_set(v_reuseFailAlloc_1298_, 2, v___x_1282_);
lean_ctor_set(v_reuseFailAlloc_1298_, 3, v___x_1285_);
lean_ctor_set(v_reuseFailAlloc_1298_, 4, v_depth_1239_);
lean_ctor_set(v_reuseFailAlloc_1298_, 5, v_preNormGoal_1275_);
lean_ctor_set(v_reuseFailAlloc_1298_, 6, v___x_1288_);
lean_ctor_set(v_reuseFailAlloc_1298_, 7, v_snd_1264_);
lean_ctor_set(v_reuseFailAlloc_1298_, 8, v_fst_1262_);
lean_ctor_set(v_reuseFailAlloc_1298_, 9, v_fst_1263_);
lean_ctor_set(v_reuseFailAlloc_1298_, 10, v_currentIteration_1256_);
lean_ctor_set(v_reuseFailAlloc_1298_, 11, v___x_1289_);
lean_ctor_set(v_reuseFailAlloc_1298_, 12, v___x_1290_);
lean_ctor_set(v_reuseFailAlloc_1298_, 13, v___x_1282_);
v___x_1292_ = v_reuseFailAlloc_1298_;
goto v_reusejp_1291_;
}
v_reusejp_1291_:
{
lean_object* v___x_1293_; size_t v___x_1294_; size_t v___x_1295_; lean_object* v___x_1296_; 
lean_ctor_set_uint8(v___x_1292_, sizeof(void*)*14 + 8, v___x_1286_);
lean_ctor_set_uint8(v___x_1292_, sizeof(void*)*14 + 9, v___x_1287_);
lean_ctor_set_uint8(v___x_1292_, sizeof(void*)*14 + 10, v___x_1287_);
lean_ctor_set_float(v___x_1292_, sizeof(void*)*14, v_parentSuccessProbability_1240_);
lean_ctor_set_uint8(v___x_1292_, sizeof(void*)*14 + 11, v___x_1287_);
lean_inc(v_introGoal_1271_);
v___x_1293_ = lean_apply_1(v_introGoal_1271_, v___x_1292_);
v___x_1294_ = ((size_t)1ULL);
v___x_1295_ = lean_usize_add(v_i_1242_, v___x_1294_);
v___x_1296_ = lean_array_uset(v_bs_x27_1280_, v_i_1242_, v___x_1293_);
v_i_1242_ = v___x_1295_;
v_bs_1243_ = v___x_1296_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1313_; lean_object* v___x_1315_; uint8_t v_isShared_1316_; uint8_t v_isSharedCheck_1320_; 
lean_del_object(v___x_1266_);
lean_dec(v_snd_1264_);
lean_dec(v_fst_1263_);
lean_dec(v_fst_1262_);
lean_dec(v___x_1255_);
lean_dec_ref(v_bs_1243_);
lean_dec(v_depth_1239_);
lean_dec(v_start_1237_);
v_a_1313_ = lean_ctor_get(v___x_1268_, 0);
v_isSharedCheck_1320_ = !lean_is_exclusive(v___x_1268_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1315_ = v___x_1268_;
v_isShared_1316_ = v_isSharedCheck_1320_;
goto v_resetjp_1314_;
}
else
{
lean_inc(v_a_1313_);
lean_dec(v___x_1268_);
v___x_1315_ = lean_box(0);
v_isShared_1316_ = v_isSharedCheck_1320_;
goto v_resetjp_1314_;
}
v_resetjp_1314_:
{
lean_object* v___x_1318_; 
if (v_isShared_1316_ == 0)
{
v___x_1318_ = v___x_1315_;
goto v_reusejp_1317_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v_a_1313_);
v___x_1318_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1317_;
}
v_reusejp_1317_:
{
return v___x_1318_;
}
}
}
}
}
else
{
lean_object* v_a_1322_; lean_object* v___x_1324_; uint8_t v_isShared_1325_; uint8_t v_isSharedCheck_1329_; 
lean_dec(v___x_1255_);
lean_dec_ref(v_bs_1243_);
lean_dec(v_depth_1239_);
lean_dec(v_start_1237_);
v_a_1322_ = lean_ctor_get(v___x_1259_, 0);
v_isSharedCheck_1329_ = !lean_is_exclusive(v___x_1259_);
if (v_isSharedCheck_1329_ == 0)
{
v___x_1324_ = v___x_1259_;
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
else
{
lean_inc(v_a_1322_);
lean_dec(v___x_1259_);
v___x_1324_ = lean_box(0);
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
v_resetjp_1323_:
{
lean_object* v___x_1327_; 
if (v_isShared_1325_ == 0)
{
v___x_1327_ = v___x_1324_;
goto v_reusejp_1326_;
}
else
{
lean_object* v_reuseFailAlloc_1328_; 
v_reuseFailAlloc_1328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1328_, 0, v_a_1322_);
v___x_1327_ = v_reuseFailAlloc_1328_;
goto v_reusejp_1326_;
}
v_reusejp_1326_:
{
return v___x_1327_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___boxed(lean_object* v_start_1330_, lean_object* v_parentMetaState_1331_, lean_object* v_depth_1332_, lean_object* v_parentSuccessProbability_1333_, lean_object* v_sz_1334_, lean_object* v_i_1335_, lean_object* v_bs_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_){
_start:
{
double v_parentSuccessProbability_boxed_1345_; size_t v_sz_boxed_1346_; size_t v_i_boxed_1347_; lean_object* v_res_1348_; 
v_parentSuccessProbability_boxed_1345_ = lean_unbox_float(v_parentSuccessProbability_1333_);
lean_dec_ref(v_parentSuccessProbability_1333_);
v_sz_boxed_1346_ = lean_unbox_usize(v_sz_1334_);
lean_dec(v_sz_1334_);
v_i_boxed_1347_ = lean_unbox_usize(v_i_1335_);
lean_dec(v_i_1335_);
v_res_1348_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2(v_start_1330_, v_parentMetaState_1331_, v_depth_1332_, v_parentSuccessProbability_boxed_1345_, v_sz_boxed_1346_, v_i_boxed_1347_, v_bs_1336_, v___y_1337_, v___y_1338_, v___y_1339_, v___y_1340_, v___y_1341_, v___y_1342_, v___y_1343_);
lean_dec(v___y_1343_);
lean_dec_ref(v___y_1342_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
lean_dec(v___y_1339_);
lean_dec(v___y_1338_);
lean_dec_ref(v___y_1337_);
lean_dec_ref(v_parentMetaState_1331_);
return v_res_1348_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_copyGoals(lean_object* v_assignedMVars_1349_, lean_object* v_start_1350_, lean_object* v_parentMetaState_1351_, double v_parentSuccessProbability_1352_, lean_object* v_depth_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_, lean_object* v_a_1357_, lean_object* v_a_1358_, lean_object* v_a_1359_, lean_object* v_a_1360_){
_start:
{
lean_object* v___x_1362_; 
lean_inc(v_start_1350_);
v___x_1362_ = lp_aesop_Aesop_getGoalsToCopy(v_assignedMVars_1349_, v_start_1350_, v_a_1354_, v_a_1355_, v_a_1356_, v_a_1357_, v_a_1358_, v_a_1359_, v_a_1360_);
if (lean_obj_tag(v___x_1362_) == 0)
{
lean_object* v_a_1363_; size_t v_sz_1364_; size_t v___x_1365_; lean_object* v___x_1366_; 
v_a_1363_ = lean_ctor_get(v___x_1362_, 0);
lean_inc(v_a_1363_);
lean_dec_ref_known(v___x_1362_, 1);
v_sz_1364_ = lean_array_size(v_a_1363_);
v___x_1365_ = ((size_t)0ULL);
v___x_1366_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2(v_start_1350_, v_parentMetaState_1351_, v_depth_1353_, v_parentSuccessProbability_1352_, v_sz_1364_, v___x_1365_, v_a_1363_, v_a_1354_, v_a_1355_, v_a_1356_, v_a_1357_, v_a_1358_, v_a_1359_, v_a_1360_);
return v___x_1366_;
}
else
{
lean_object* v_a_1367_; lean_object* v___x_1369_; uint8_t v_isShared_1370_; uint8_t v_isSharedCheck_1374_; 
lean_dec(v_depth_1353_);
lean_dec(v_start_1350_);
v_a_1367_ = lean_ctor_get(v___x_1362_, 0);
v_isSharedCheck_1374_ = !lean_is_exclusive(v___x_1362_);
if (v_isSharedCheck_1374_ == 0)
{
v___x_1369_ = v___x_1362_;
v_isShared_1370_ = v_isSharedCheck_1374_;
goto v_resetjp_1368_;
}
else
{
lean_inc(v_a_1367_);
lean_dec(v___x_1362_);
v___x_1369_ = lean_box(0);
v_isShared_1370_ = v_isSharedCheck_1374_;
goto v_resetjp_1368_;
}
v_resetjp_1368_:
{
lean_object* v___x_1372_; 
if (v_isShared_1370_ == 0)
{
v___x_1372_ = v___x_1369_;
goto v_reusejp_1371_;
}
else
{
lean_object* v_reuseFailAlloc_1373_; 
v_reuseFailAlloc_1373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1373_, 0, v_a_1367_);
v___x_1372_ = v_reuseFailAlloc_1373_;
goto v_reusejp_1371_;
}
v_reusejp_1371_:
{
return v___x_1372_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_copyGoals___boxed(lean_object* v_assignedMVars_1375_, lean_object* v_start_1376_, lean_object* v_parentMetaState_1377_, lean_object* v_parentSuccessProbability_1378_, lean_object* v_depth_1379_, lean_object* v_a_1380_, lean_object* v_a_1381_, lean_object* v_a_1382_, lean_object* v_a_1383_, lean_object* v_a_1384_, lean_object* v_a_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_){
_start:
{
double v_parentSuccessProbability_boxed_1388_; lean_object* v_res_1389_; 
v_parentSuccessProbability_boxed_1388_ = lean_unbox_float(v_parentSuccessProbability_1378_);
lean_dec_ref(v_parentSuccessProbability_1378_);
v_res_1389_ = lp_aesop_Aesop_copyGoals(v_assignedMVars_1375_, v_start_1376_, v_parentMetaState_1377_, v_parentSuccessProbability_boxed_1388_, v_depth_1379_, v_a_1380_, v_a_1381_, v_a_1382_, v_a_1383_, v_a_1384_, v_a_1385_, v_a_1386_);
lean_dec(v_a_1386_);
lean_dec_ref(v_a_1385_);
lean_dec(v_a_1384_);
lean_dec_ref(v_a_1383_);
lean_dec(v_a_1382_);
lean_dec(v_a_1381_);
lean_dec_ref(v_a_1380_);
lean_dec_ref(v_parentMetaState_1377_);
return v_res_1389_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal___lam__0(lean_object* v_ruleSet_1390_, lean_object* v_goal_1391_, lean_object* v_parentForwardState_1392_, lean_object* v_consumedForwardRuleMatches_1393_, lean_object* v_parentForwardMatches_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_, lean_object* v___y_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_){
_start:
{
lean_object* v___x_1403_; 
lean_inc_ref(v_goal_1391_);
v___x_1403_ = lp_aesop_Aesop_ForwardState_applyGoalDiff(v_ruleSet_1390_, v_goal_1391_, v_parentForwardState_1392_, v___y_1397_, v___y_1398_, v___y_1399_, v___y_1400_, v___y_1401_);
if (lean_obj_tag(v___x_1403_) == 0)
{
lean_object* v_a_1404_; lean_object* v___x_1406_; uint8_t v_isShared_1407_; uint8_t v_isSharedCheck_1415_; 
v_a_1404_ = lean_ctor_get(v___x_1403_, 0);
v_isSharedCheck_1415_ = !lean_is_exclusive(v___x_1403_);
if (v_isSharedCheck_1415_ == 0)
{
v___x_1406_ = v___x_1403_;
v_isShared_1407_ = v_isSharedCheck_1415_;
goto v_resetjp_1405_;
}
else
{
lean_inc(v_a_1404_);
lean_dec(v___x_1403_);
v___x_1406_ = lean_box(0);
v_isShared_1407_ = v_isSharedCheck_1415_;
goto v_resetjp_1405_;
}
v_resetjp_1405_:
{
lean_object* v_removedFVars_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1413_; 
v_removedFVars_1408_ = lean_ctor_get(v_goal_1391_, 3);
lean_inc_ref(v_removedFVars_1408_);
lean_dec_ref(v_goal_1391_);
v___x_1409_ = ((lean_object*)(lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches___closed__0));
v___x_1410_ = lp_aesop_Aesop_ForwardRuleMatches_update(v___x_1409_, v_removedFVars_1408_, v_consumedForwardRuleMatches_1393_, v_parentForwardMatches_1394_);
v___x_1411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1411_, 0, v_a_1404_);
lean_ctor_set(v___x_1411_, 1, v___x_1410_);
if (v_isShared_1407_ == 0)
{
lean_ctor_set(v___x_1406_, 0, v___x_1411_);
v___x_1413_ = v___x_1406_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1414_; 
v_reuseFailAlloc_1414_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1414_, 0, v___x_1411_);
v___x_1413_ = v_reuseFailAlloc_1414_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
return v___x_1413_;
}
}
}
else
{
lean_object* v_a_1416_; lean_object* v___x_1418_; uint8_t v_isShared_1419_; uint8_t v_isSharedCheck_1423_; 
lean_dec_ref(v_parentForwardMatches_1394_);
lean_dec_ref(v_goal_1391_);
v_a_1416_ = lean_ctor_get(v___x_1403_, 0);
v_isSharedCheck_1423_ = !lean_is_exclusive(v___x_1403_);
if (v_isSharedCheck_1423_ == 0)
{
v___x_1418_ = v___x_1403_;
v_isShared_1419_ = v_isSharedCheck_1423_;
goto v_resetjp_1417_;
}
else
{
lean_inc(v_a_1416_);
lean_dec(v___x_1403_);
v___x_1418_ = lean_box(0);
v_isShared_1419_ = v_isSharedCheck_1423_;
goto v_resetjp_1417_;
}
v_resetjp_1417_:
{
lean_object* v___x_1421_; 
if (v_isShared_1419_ == 0)
{
v___x_1421_ = v___x_1418_;
goto v_reusejp_1420_;
}
else
{
lean_object* v_reuseFailAlloc_1422_; 
v_reuseFailAlloc_1422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1422_, 0, v_a_1416_);
v___x_1421_ = v_reuseFailAlloc_1422_;
goto v_reusejp_1420_;
}
v_reusejp_1420_:
{
return v___x_1421_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal___lam__0___boxed(lean_object* v_ruleSet_1424_, lean_object* v_goal_1425_, lean_object* v_parentForwardState_1426_, lean_object* v_consumedForwardRuleMatches_1427_, lean_object* v_parentForwardMatches_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_){
_start:
{
lean_object* v_res_1437_; 
v_res_1437_ = lp_aesop_Aesop_makeInitialGoal___lam__0(v_ruleSet_1424_, v_goal_1425_, v_parentForwardState_1426_, v_consumedForwardRuleMatches_1427_, v_parentForwardMatches_1428_, v___y_1429_, v___y_1430_, v___y_1431_, v___y_1432_, v___y_1433_, v___y_1434_, v___y_1435_);
lean_dec(v___y_1435_);
lean_dec_ref(v___y_1434_);
lean_dec(v___y_1433_);
lean_dec_ref(v___y_1432_);
lean_dec(v___y_1431_);
lean_dec(v___y_1430_);
lean_dec_ref(v___y_1429_);
lean_dec_ref(v_consumedForwardRuleMatches_1427_);
return v_res_1437_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal(lean_object* v_goal_1438_, lean_object* v_mvars_1439_, lean_object* v_parent_1440_, lean_object* v_parentMetaState_1441_, lean_object* v_parentForwardState_1442_, lean_object* v_parentForwardMatches_1443_, lean_object* v_consumedForwardRuleMatches_1444_, lean_object* v_depth_1445_, double v_successProbability_1446_, lean_object* v_origin_1447_, lean_object* v_a_1448_, lean_object* v_a_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_){
_start:
{
lean_object* v_currentIteration_1456_; lean_object* v_ruleSet_1457_; lean_object* v___f_1458_; lean_object* v___x_1459_; 
v_currentIteration_1456_ = lean_ctor_get(v_a_1448_, 0);
v_ruleSet_1457_ = lean_ctor_get(v_a_1448_, 1);
lean_inc_ref(v_goal_1438_);
lean_inc_ref(v_ruleSet_1457_);
v___f_1458_ = lean_alloc_closure((void*)(lp_aesop_Aesop_makeInitialGoal___lam__0___boxed), 13, 5);
lean_closure_set(v___f_1458_, 0, v_ruleSet_1457_);
lean_closure_set(v___f_1458_, 1, v_goal_1438_);
lean_closure_set(v___f_1458_, 2, v_parentForwardState_1442_);
lean_closure_set(v___f_1458_, 3, v_consumedForwardRuleMatches_1444_);
lean_closure_set(v___f_1458_, 4, v_parentForwardMatches_1443_);
v___x_1459_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg(v_parentMetaState_1441_, v___f_1458_, v_a_1448_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_, v_a_1454_);
if (lean_obj_tag(v___x_1459_) == 0)
{
lean_object* v_a_1460_; lean_object* v_fst_1461_; lean_object* v_snd_1462_; lean_object* v___x_1463_; 
v_a_1460_ = lean_ctor_get(v___x_1459_, 0);
lean_inc(v_a_1460_);
lean_dec_ref_known(v___x_1459_, 1);
v_fst_1461_ = lean_ctor_get(v_a_1460_, 0);
lean_inc(v_fst_1461_);
v_snd_1462_ = lean_ctor_get(v_a_1460_, 1);
lean_inc(v_snd_1462_);
lean_dec(v_a_1460_);
v___x_1463_ = lp_aesop_Aesop_getAndIncrementNextGoalId___redArg(v_a_1449_);
if (lean_obj_tag(v___x_1463_) == 0)
{
lean_object* v_a_1464_; lean_object* v___x_1466_; uint8_t v_isShared_1467_; uint8_t v_isSharedCheck_1482_; 
v_a_1464_ = lean_ctor_get(v___x_1463_, 0);
v_isSharedCheck_1482_ = !lean_is_exclusive(v___x_1463_);
if (v_isSharedCheck_1482_ == 0)
{
v___x_1466_ = v___x_1463_;
v_isShared_1467_ = v_isSharedCheck_1482_;
goto v_resetjp_1465_;
}
else
{
lean_inc(v_a_1464_);
lean_dec(v___x_1463_);
v___x_1466_ = lean_box(0);
v_isShared_1467_ = v_isSharedCheck_1482_;
goto v_resetjp_1465_;
}
v_resetjp_1465_:
{
lean_object* v___x_1468_; uint8_t v___x_1469_; uint8_t v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v_introGoal_1477_; lean_object* v___x_1478_; lean_object* v___x_1480_; 
v___x_1468_ = ((lean_object*)(lp_aesop_Aesop_findPathForAssignedMVars___closed__0));
v___x_1469_ = 0;
v___x_1470_ = 0;
v___x_1471_ = lp_aesop_Aesop_Subgoal_mvarId(v_goal_1438_);
lean_dec_ref(v_goal_1438_);
v___x_1472_ = lean_box(0);
v___x_1473_ = lp_aesop_Aesop_Iteration_none;
v___x_1474_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_copyGoals_spec__2___closed__0);
lean_inc(v_currentIteration_1456_);
v___x_1475_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v___x_1475_, 0, v_a_1464_);
lean_ctor_set(v___x_1475_, 1, v_parent_1440_);
lean_ctor_set(v___x_1475_, 2, v___x_1468_);
lean_ctor_set(v___x_1475_, 3, v_origin_1447_);
lean_ctor_set(v___x_1475_, 4, v_depth_1445_);
lean_ctor_set(v___x_1475_, 5, v___x_1471_);
lean_ctor_set(v___x_1475_, 6, v___x_1472_);
lean_ctor_set(v___x_1475_, 7, v_mvars_1439_);
lean_ctor_set(v___x_1475_, 8, v_fst_1461_);
lean_ctor_set(v___x_1475_, 9, v_snd_1462_);
lean_ctor_set(v___x_1475_, 10, v_currentIteration_1456_);
lean_ctor_set(v___x_1475_, 11, v___x_1473_);
lean_ctor_set(v___x_1475_, 12, v___x_1474_);
lean_ctor_set(v___x_1475_, 13, v___x_1468_);
lean_ctor_set_uint8(v___x_1475_, sizeof(void*)*14 + 8, v___x_1469_);
lean_ctor_set_uint8(v___x_1475_, sizeof(void*)*14 + 9, v___x_1470_);
lean_ctor_set_uint8(v___x_1475_, sizeof(void*)*14 + 10, v___x_1470_);
lean_ctor_set_float(v___x_1475_, sizeof(void*)*14, v_successProbability_1446_);
lean_ctor_set_uint8(v___x_1475_, sizeof(void*)*14 + 11, v___x_1470_);
v___x_1476_ = lp_aesop_Aesop_treeImpl;
v_introGoal_1477_ = lean_ctor_get(v___x_1476_, 0);
lean_inc(v_introGoal_1477_);
v___x_1478_ = lean_apply_1(v_introGoal_1477_, v___x_1475_);
if (v_isShared_1467_ == 0)
{
lean_ctor_set(v___x_1466_, 0, v___x_1478_);
v___x_1480_ = v___x_1466_;
goto v_reusejp_1479_;
}
else
{
lean_object* v_reuseFailAlloc_1481_; 
v_reuseFailAlloc_1481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1481_, 0, v___x_1478_);
v___x_1480_ = v_reuseFailAlloc_1481_;
goto v_reusejp_1479_;
}
v_reusejp_1479_:
{
return v___x_1480_;
}
}
}
else
{
lean_object* v_a_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1490_; 
lean_dec(v_snd_1462_);
lean_dec(v_fst_1461_);
lean_dec(v_origin_1447_);
lean_dec(v_depth_1445_);
lean_dec(v_parent_1440_);
lean_dec_ref(v_mvars_1439_);
lean_dec_ref(v_goal_1438_);
v_a_1483_ = lean_ctor_get(v___x_1463_, 0);
v_isSharedCheck_1490_ = !lean_is_exclusive(v___x_1463_);
if (v_isSharedCheck_1490_ == 0)
{
v___x_1485_ = v___x_1463_;
v_isShared_1486_ = v_isSharedCheck_1490_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_a_1483_);
lean_dec(v___x_1463_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1490_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v___x_1488_; 
if (v_isShared_1486_ == 0)
{
v___x_1488_ = v___x_1485_;
goto v_reusejp_1487_;
}
else
{
lean_object* v_reuseFailAlloc_1489_; 
v_reuseFailAlloc_1489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1489_, 0, v_a_1483_);
v___x_1488_ = v_reuseFailAlloc_1489_;
goto v_reusejp_1487_;
}
v_reusejp_1487_:
{
return v___x_1488_;
}
}
}
}
else
{
lean_object* v_a_1491_; lean_object* v___x_1493_; uint8_t v_isShared_1494_; uint8_t v_isSharedCheck_1498_; 
lean_dec(v_origin_1447_);
lean_dec(v_depth_1445_);
lean_dec(v_parent_1440_);
lean_dec_ref(v_mvars_1439_);
lean_dec_ref(v_goal_1438_);
v_a_1491_ = lean_ctor_get(v___x_1459_, 0);
v_isSharedCheck_1498_ = !lean_is_exclusive(v___x_1459_);
if (v_isSharedCheck_1498_ == 0)
{
v___x_1493_ = v___x_1459_;
v_isShared_1494_ = v_isSharedCheck_1498_;
goto v_resetjp_1492_;
}
else
{
lean_inc(v_a_1491_);
lean_dec(v___x_1459_);
v___x_1493_ = lean_box(0);
v_isShared_1494_ = v_isSharedCheck_1498_;
goto v_resetjp_1492_;
}
v_resetjp_1492_:
{
lean_object* v___x_1496_; 
if (v_isShared_1494_ == 0)
{
v___x_1496_ = v___x_1493_;
goto v_reusejp_1495_;
}
else
{
lean_object* v_reuseFailAlloc_1497_; 
v_reuseFailAlloc_1497_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1497_, 0, v_a_1491_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_makeInitialGoal___boxed(lean_object** _args){
lean_object* v_goal_1499_ = _args[0];
lean_object* v_mvars_1500_ = _args[1];
lean_object* v_parent_1501_ = _args[2];
lean_object* v_parentMetaState_1502_ = _args[3];
lean_object* v_parentForwardState_1503_ = _args[4];
lean_object* v_parentForwardMatches_1504_ = _args[5];
lean_object* v_consumedForwardRuleMatches_1505_ = _args[6];
lean_object* v_depth_1506_ = _args[7];
lean_object* v_successProbability_1507_ = _args[8];
lean_object* v_origin_1508_ = _args[9];
lean_object* v_a_1509_ = _args[10];
lean_object* v_a_1510_ = _args[11];
lean_object* v_a_1511_ = _args[12];
lean_object* v_a_1512_ = _args[13];
lean_object* v_a_1513_ = _args[14];
lean_object* v_a_1514_ = _args[15];
lean_object* v_a_1515_ = _args[16];
lean_object* v_a_1516_ = _args[17];
_start:
{
double v_successProbability_boxed_1517_; lean_object* v_res_1518_; 
v_successProbability_boxed_1517_ = lean_unbox_float(v_successProbability_1507_);
lean_dec_ref(v_successProbability_1507_);
v_res_1518_ = lp_aesop_Aesop_makeInitialGoal(v_goal_1499_, v_mvars_1500_, v_parent_1501_, v_parentMetaState_1502_, v_parentForwardState_1503_, v_parentForwardMatches_1504_, v_consumedForwardRuleMatches_1505_, v_depth_1506_, v_successProbability_boxed_1517_, v_origin_1508_, v_a_1509_, v_a_1510_, v_a_1511_, v_a_1512_, v_a_1513_, v_a_1514_, v_a_1515_);
lean_dec(v_a_1515_);
lean_dec_ref(v_a_1514_);
lean_dec(v_a_1513_);
lean_dec_ref(v_a_1512_);
lean_dec(v_a_1511_);
lean_dec(v_a_1510_);
lean_dec_ref(v_a_1509_);
lean_dec_ref(v_parentMetaState_1502_);
return v_res_1518_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__0(lean_object* v_x_1522_){
_start:
{
lean_object* v___x_1523_; lean_object* v_elimGoal_1524_; lean_object* v___x_1525_; lean_object* v_mvars_1526_; 
v___x_1523_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1524_ = lean_ctor_get(v___x_1523_, 1);
lean_inc_ref(v_elimGoal_1524_);
v___x_1525_ = lean_apply_1(v_elimGoal_1524_, v_x_1522_);
v_mvars_1526_ = lean_ctor_get(v___x_1525_, 7);
lean_inc_ref(v_mvars_1526_);
lean_dec_ref(v___x_1525_);
return v_mvars_1526_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20_spec__31___redArg(lean_object* v_x_1527_, lean_object* v_x_1528_){
_start:
{
if (lean_obj_tag(v_x_1528_) == 0)
{
return v_x_1527_;
}
else
{
lean_object* v_key_1529_; lean_object* v_value_1530_; lean_object* v_tail_1531_; lean_object* v___x_1533_; uint8_t v_isShared_1534_; uint8_t v_isSharedCheck_1554_; 
v_key_1529_ = lean_ctor_get(v_x_1528_, 0);
v_value_1530_ = lean_ctor_get(v_x_1528_, 1);
v_tail_1531_ = lean_ctor_get(v_x_1528_, 2);
v_isSharedCheck_1554_ = !lean_is_exclusive(v_x_1528_);
if (v_isSharedCheck_1554_ == 0)
{
v___x_1533_ = v_x_1528_;
v_isShared_1534_ = v_isSharedCheck_1554_;
goto v_resetjp_1532_;
}
else
{
lean_inc(v_tail_1531_);
lean_inc(v_value_1530_);
lean_inc(v_key_1529_);
lean_dec(v_x_1528_);
v___x_1533_ = lean_box(0);
v_isShared_1534_ = v_isSharedCheck_1554_;
goto v_resetjp_1532_;
}
v_resetjp_1532_:
{
lean_object* v___x_1535_; uint64_t v___x_1536_; uint64_t v___x_1537_; uint64_t v___x_1538_; uint64_t v_fold_1539_; uint64_t v___x_1540_; uint64_t v___x_1541_; uint64_t v___x_1542_; size_t v___x_1543_; size_t v___x_1544_; size_t v___x_1545_; size_t v___x_1546_; size_t v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1550_; 
v___x_1535_ = lean_array_get_size(v_x_1527_);
v___x_1536_ = l_Lean_instHashableMVarId_hash(v_key_1529_);
v___x_1537_ = 32ULL;
v___x_1538_ = lean_uint64_shift_right(v___x_1536_, v___x_1537_);
v_fold_1539_ = lean_uint64_xor(v___x_1536_, v___x_1538_);
v___x_1540_ = 16ULL;
v___x_1541_ = lean_uint64_shift_right(v_fold_1539_, v___x_1540_);
v___x_1542_ = lean_uint64_xor(v_fold_1539_, v___x_1541_);
v___x_1543_ = lean_uint64_to_usize(v___x_1542_);
v___x_1544_ = lean_usize_of_nat(v___x_1535_);
v___x_1545_ = ((size_t)1ULL);
v___x_1546_ = lean_usize_sub(v___x_1544_, v___x_1545_);
v___x_1547_ = lean_usize_land(v___x_1543_, v___x_1546_);
v___x_1548_ = lean_array_uget_borrowed(v_x_1527_, v___x_1547_);
lean_inc(v___x_1548_);
if (v_isShared_1534_ == 0)
{
lean_ctor_set(v___x_1533_, 2, v___x_1548_);
v___x_1550_ = v___x_1533_;
goto v_reusejp_1549_;
}
else
{
lean_object* v_reuseFailAlloc_1553_; 
v_reuseFailAlloc_1553_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1553_, 0, v_key_1529_);
lean_ctor_set(v_reuseFailAlloc_1553_, 1, v_value_1530_);
lean_ctor_set(v_reuseFailAlloc_1553_, 2, v___x_1548_);
v___x_1550_ = v_reuseFailAlloc_1553_;
goto v_reusejp_1549_;
}
v_reusejp_1549_:
{
lean_object* v___x_1551_; 
v___x_1551_ = lean_array_uset(v_x_1527_, v___x_1547_, v___x_1550_);
v_x_1527_ = v___x_1551_;
v_x_1528_ = v_tail_1531_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20___redArg(lean_object* v_i_1555_, lean_object* v_source_1556_, lean_object* v_target_1557_){
_start:
{
lean_object* v___x_1558_; uint8_t v___x_1559_; 
v___x_1558_ = lean_array_get_size(v_source_1556_);
v___x_1559_ = lean_nat_dec_lt(v_i_1555_, v___x_1558_);
if (v___x_1559_ == 0)
{
lean_dec_ref(v_source_1556_);
lean_dec(v_i_1555_);
return v_target_1557_;
}
else
{
lean_object* v_es_1560_; lean_object* v___x_1561_; lean_object* v_source_1562_; lean_object* v_target_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; 
v_es_1560_ = lean_array_fget(v_source_1556_, v_i_1555_);
v___x_1561_ = lean_box(0);
v_source_1562_ = lean_array_fset(v_source_1556_, v_i_1555_, v___x_1561_);
v_target_1563_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20_spec__31___redArg(v_target_1557_, v_es_1560_);
v___x_1564_ = lean_unsigned_to_nat(1u);
v___x_1565_ = lean_nat_add(v_i_1555_, v___x_1564_);
lean_dec(v_i_1555_);
v_i_1555_ = v___x_1565_;
v_source_1556_ = v_source_1562_;
v_target_1557_ = v_target_1563_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18___redArg(lean_object* v_data_1567_){
_start:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v_nbuckets_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; 
v___x_1568_ = lean_array_get_size(v_data_1567_);
v___x_1569_ = lean_unsigned_to_nat(2u);
v_nbuckets_1570_ = lean_nat_mul(v___x_1568_, v___x_1569_);
v___x_1571_ = lean_unsigned_to_nat(0u);
v___x_1572_ = lean_box(0);
v___x_1573_ = lean_mk_array(v_nbuckets_1570_, v___x_1572_);
v___x_1574_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20___redArg(v___x_1571_, v_data_1567_, v___x_1573_);
return v___x_1574_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg(lean_object* v_a_1575_, lean_object* v_x_1576_){
_start:
{
if (lean_obj_tag(v_x_1576_) == 0)
{
uint8_t v___x_1577_; 
v___x_1577_ = 0;
return v___x_1577_;
}
else
{
lean_object* v_key_1578_; lean_object* v_tail_1579_; uint8_t v___x_1580_; 
v_key_1578_ = lean_ctor_get(v_x_1576_, 0);
v_tail_1579_ = lean_ctor_get(v_x_1576_, 2);
v___x_1580_ = l_Lean_instBEqMVarId_beq(v_key_1578_, v_a_1575_);
if (v___x_1580_ == 0)
{
v_x_1576_ = v_tail_1579_;
goto _start;
}
else
{
return v___x_1580_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg___boxed(lean_object* v_a_1582_, lean_object* v_x_1583_){
_start:
{
uint8_t v_res_1584_; lean_object* v_r_1585_; 
v_res_1584_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg(v_a_1582_, v_x_1583_);
lean_dec(v_x_1583_);
lean_dec(v_a_1582_);
v_r_1585_ = lean_box(v_res_1584_);
return v_r_1585_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13___redArg(lean_object* v_m_1586_, lean_object* v_a_1587_, lean_object* v_b_1588_){
_start:
{
lean_object* v_size_1589_; lean_object* v_buckets_1590_; lean_object* v___x_1591_; uint64_t v___x_1592_; uint64_t v___x_1593_; uint64_t v___x_1594_; uint64_t v_fold_1595_; uint64_t v___x_1596_; uint64_t v___x_1597_; uint64_t v___x_1598_; size_t v___x_1599_; size_t v___x_1600_; size_t v___x_1601_; size_t v___x_1602_; size_t v___x_1603_; lean_object* v_bkt_1604_; uint8_t v___x_1605_; 
v_size_1589_ = lean_ctor_get(v_m_1586_, 0);
v_buckets_1590_ = lean_ctor_get(v_m_1586_, 1);
v___x_1591_ = lean_array_get_size(v_buckets_1590_);
v___x_1592_ = l_Lean_instHashableMVarId_hash(v_a_1587_);
v___x_1593_ = 32ULL;
v___x_1594_ = lean_uint64_shift_right(v___x_1592_, v___x_1593_);
v_fold_1595_ = lean_uint64_xor(v___x_1592_, v___x_1594_);
v___x_1596_ = 16ULL;
v___x_1597_ = lean_uint64_shift_right(v_fold_1595_, v___x_1596_);
v___x_1598_ = lean_uint64_xor(v_fold_1595_, v___x_1597_);
v___x_1599_ = lean_uint64_to_usize(v___x_1598_);
v___x_1600_ = lean_usize_of_nat(v___x_1591_);
v___x_1601_ = ((size_t)1ULL);
v___x_1602_ = lean_usize_sub(v___x_1600_, v___x_1601_);
v___x_1603_ = lean_usize_land(v___x_1599_, v___x_1602_);
v_bkt_1604_ = lean_array_uget_borrowed(v_buckets_1590_, v___x_1603_);
v___x_1605_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg(v_a_1587_, v_bkt_1604_);
if (v___x_1605_ == 0)
{
lean_object* v___x_1607_; uint8_t v_isShared_1608_; uint8_t v_isSharedCheck_1626_; 
lean_inc_ref(v_buckets_1590_);
lean_inc(v_size_1589_);
v_isSharedCheck_1626_ = !lean_is_exclusive(v_m_1586_);
if (v_isSharedCheck_1626_ == 0)
{
lean_object* v_unused_1627_; lean_object* v_unused_1628_; 
v_unused_1627_ = lean_ctor_get(v_m_1586_, 1);
lean_dec(v_unused_1627_);
v_unused_1628_ = lean_ctor_get(v_m_1586_, 0);
lean_dec(v_unused_1628_);
v___x_1607_ = v_m_1586_;
v_isShared_1608_ = v_isSharedCheck_1626_;
goto v_resetjp_1606_;
}
else
{
lean_dec(v_m_1586_);
v___x_1607_ = lean_box(0);
v_isShared_1608_ = v_isSharedCheck_1626_;
goto v_resetjp_1606_;
}
v_resetjp_1606_:
{
lean_object* v___x_1609_; lean_object* v_size_x27_1610_; lean_object* v___x_1611_; lean_object* v_buckets_x27_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; uint8_t v___x_1618_; 
v___x_1609_ = lean_unsigned_to_nat(1u);
v_size_x27_1610_ = lean_nat_add(v_size_1589_, v___x_1609_);
lean_dec(v_size_1589_);
lean_inc(v_bkt_1604_);
v___x_1611_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1611_, 0, v_a_1587_);
lean_ctor_set(v___x_1611_, 1, v_b_1588_);
lean_ctor_set(v___x_1611_, 2, v_bkt_1604_);
v_buckets_x27_1612_ = lean_array_uset(v_buckets_1590_, v___x_1603_, v___x_1611_);
v___x_1613_ = lean_unsigned_to_nat(4u);
v___x_1614_ = lean_nat_mul(v_size_x27_1610_, v___x_1613_);
v___x_1615_ = lean_unsigned_to_nat(3u);
v___x_1616_ = lean_nat_div(v___x_1614_, v___x_1615_);
lean_dec(v___x_1614_);
v___x_1617_ = lean_array_get_size(v_buckets_x27_1612_);
v___x_1618_ = lean_nat_dec_le(v___x_1616_, v___x_1617_);
lean_dec(v___x_1616_);
if (v___x_1618_ == 0)
{
lean_object* v_val_1619_; lean_object* v___x_1621_; 
v_val_1619_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18___redArg(v_buckets_x27_1612_);
if (v_isShared_1608_ == 0)
{
lean_ctor_set(v___x_1607_, 1, v_val_1619_);
lean_ctor_set(v___x_1607_, 0, v_size_x27_1610_);
v___x_1621_ = v___x_1607_;
goto v_reusejp_1620_;
}
else
{
lean_object* v_reuseFailAlloc_1622_; 
v_reuseFailAlloc_1622_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1622_, 0, v_size_x27_1610_);
lean_ctor_set(v_reuseFailAlloc_1622_, 1, v_val_1619_);
v___x_1621_ = v_reuseFailAlloc_1622_;
goto v_reusejp_1620_;
}
v_reusejp_1620_:
{
return v___x_1621_;
}
}
else
{
lean_object* v___x_1624_; 
if (v_isShared_1608_ == 0)
{
lean_ctor_set(v___x_1607_, 1, v_buckets_x27_1612_);
lean_ctor_set(v___x_1607_, 0, v_size_x27_1610_);
v___x_1624_ = v___x_1607_;
goto v_reusejp_1623_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_size_x27_1610_);
lean_ctor_set(v_reuseFailAlloc_1625_, 1, v_buckets_x27_1612_);
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
else
{
lean_dec(v_b_1588_);
lean_dec(v_a_1587_);
return v_m_1586_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__2(lean_object* v_a_1629_, lean_object* v_a_1630_){
_start:
{
if (lean_obj_tag(v_a_1629_) == 0)
{
lean_object* v___x_1631_; 
v___x_1631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1631_, 0, v_a_1630_);
return v___x_1631_;
}
else
{
lean_object* v_key_1632_; lean_object* v_tail_1633_; lean_object* v___x_1634_; lean_object* v_r_1635_; 
v_key_1632_ = lean_ctor_get(v_a_1629_, 0);
lean_inc(v_key_1632_);
v_tail_1633_ = lean_ctor_get(v_a_1629_, 2);
lean_inc(v_tail_1633_);
lean_dec_ref_known(v_a_1629_, 3);
v___x_1634_ = lean_box(0);
v_r_1635_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13___redArg(v_a_1630_, v_key_1632_, v___x_1634_);
v_a_1629_ = v_tail_1633_;
v_a_1630_ = v_r_1635_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__3(lean_object* v_as_1637_, size_t v_sz_1638_, size_t v_i_1639_, lean_object* v_b_1640_){
_start:
{
uint8_t v___x_1641_; 
v___x_1641_ = lean_usize_dec_lt(v_i_1639_, v_sz_1638_);
if (v___x_1641_ == 0)
{
return v_b_1640_;
}
else
{
lean_object* v_a_1642_; lean_object* v___x_1643_; 
v_a_1642_ = lean_array_uget_borrowed(v_as_1637_, v_i_1639_);
lean_inc(v_a_1642_);
v___x_1643_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__2(v_a_1642_, v_b_1640_);
if (lean_obj_tag(v___x_1643_) == 0)
{
lean_object* v_a_1644_; 
v_a_1644_ = lean_ctor_get(v___x_1643_, 0);
lean_inc(v_a_1644_);
lean_dec_ref_known(v___x_1643_, 1);
return v_a_1644_;
}
else
{
lean_object* v_a_1645_; size_t v___x_1646_; size_t v___x_1647_; 
v_a_1645_ = lean_ctor_get(v___x_1643_, 0);
lean_inc(v_a_1645_);
lean_dec_ref_known(v___x_1643_, 1);
v___x_1646_ = ((size_t)1ULL);
v___x_1647_ = lean_usize_add(v_i_1639_, v___x_1646_);
v_i_1639_ = v___x_1647_;
v_b_1640_ = v_a_1645_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__3___boxed(lean_object* v_as_1649_, lean_object* v_sz_1650_, lean_object* v_i_1651_, lean_object* v_b_1652_){
_start:
{
size_t v_sz_boxed_1653_; size_t v_i_boxed_1654_; lean_object* v_res_1655_; 
v_sz_boxed_1653_ = lean_unbox_usize(v_sz_1650_);
lean_dec(v_sz_1650_);
v_i_boxed_1654_ = lean_unbox_usize(v_i_1651_);
lean_dec(v_i_1651_);
v_res_1655_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__3(v_as_1649_, v_sz_boxed_1653_, v_i_boxed_1654_, v_b_1652_);
lean_dec_ref(v_as_1649_);
return v_res_1655_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1(lean_object* v_m_1656_, lean_object* v_l_1657_){
_start:
{
lean_object* v_buckets_1658_; size_t v_sz_1659_; size_t v___x_1660_; lean_object* v___x_1661_; 
v_buckets_1658_ = lean_ctor_get(v_l_1657_, 1);
v_sz_1659_ = lean_array_size(v_buckets_1658_);
v___x_1660_ = ((size_t)0ULL);
v___x_1661_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1_spec__3(v_buckets_1658_, v_sz_1659_, v___x_1660_, v_m_1656_);
return v___x_1661_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1___boxed(lean_object* v_m_1662_, lean_object* v_l_1663_){
_start:
{
lean_object* v_res_1664_; 
v_res_1664_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1(v_m_1662_, v_l_1663_);
lean_dec_ref(v_l_1663_);
return v_res_1664_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__10(lean_object* v_as_1665_, size_t v_i_1666_, size_t v_stop_1667_, lean_object* v_b_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_){
_start:
{
lean_object* v_a_1675_; uint8_t v___x_1679_; 
v___x_1679_ = lean_usize_dec_eq(v_i_1666_, v_stop_1667_);
if (v___x_1679_ == 0)
{
lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; 
v___x_1680_ = lean_array_uget_borrowed(v_as_1665_, v_i_1666_);
v___x_1681_ = lp_aesop_Aesop_Subgoal_mvarId(v___x_1680_);
v___x_1682_ = l_Lean_MVarId_getMVarDependencies(v___x_1681_, v___x_1679_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_);
if (lean_obj_tag(v___x_1682_) == 0)
{
lean_object* v_a_1683_; lean_object* v___x_1684_; 
v_a_1683_ = lean_ctor_get(v___x_1682_, 0);
lean_inc(v_a_1683_);
lean_dec_ref_known(v___x_1682_, 1);
v___x_1684_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__1(v_b_1668_, v_a_1683_);
lean_dec(v_a_1683_);
v_a_1675_ = v___x_1684_;
goto v___jp_1674_;
}
else
{
lean_dec_ref(v_b_1668_);
if (lean_obj_tag(v___x_1682_) == 0)
{
lean_object* v_a_1685_; 
v_a_1685_ = lean_ctor_get(v___x_1682_, 0);
lean_inc(v_a_1685_);
lean_dec_ref_known(v___x_1682_, 1);
v_a_1675_ = v_a_1685_;
goto v___jp_1674_;
}
else
{
return v___x_1682_;
}
}
}
else
{
lean_object* v___x_1686_; 
v___x_1686_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1686_, 0, v_b_1668_);
return v___x_1686_;
}
v___jp_1674_:
{
size_t v___x_1676_; size_t v___x_1677_; 
v___x_1676_ = ((size_t)1ULL);
v___x_1677_ = lean_usize_add(v_i_1666_, v___x_1676_);
v_i_1666_ = v___x_1677_;
v_b_1668_ = v_a_1675_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__10___boxed(lean_object* v_as_1687_, lean_object* v_i_1688_, lean_object* v_stop_1689_, lean_object* v_b_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_){
_start:
{
size_t v_i_boxed_1696_; size_t v_stop_boxed_1697_; lean_object* v_res_1698_; 
v_i_boxed_1696_ = lean_unbox_usize(v_i_1688_);
lean_dec(v_i_1688_);
v_stop_boxed_1697_ = lean_unbox_usize(v_stop_1689_);
lean_dec(v_stop_1689_);
v_res_1698_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__10(v_as_1687_, v_i_boxed_1696_, v_stop_boxed_1697_, v_b_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_);
lean_dec(v___y_1694_);
lean_dec_ref(v___y_1693_);
lean_dec(v___y_1692_);
lean_dec_ref(v___y_1691_);
lean_dec_ref(v_as_1687_);
return v_res_1698_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_insert___at___00Aesop_addRappUnsafe_spec__8(lean_object* v_x_1699_, lean_object* v_x_1700_){
_start:
{
uint8_t v___x_1701_; 
v___x_1701_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v_x_1700_, v_x_1699_);
if (v___x_1701_ == 0)
{
lean_object* v___x_1702_; 
v___x_1702_ = lean_array_push(v_x_1700_, v_x_1699_);
return v___x_1702_;
}
else
{
lean_dec(v_x_1699_);
return v_x_1700_;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg(lean_object* v_m_1703_, lean_object* v_a_1704_){
_start:
{
lean_object* v_buckets_1705_; lean_object* v___x_1706_; uint64_t v___x_1707_; uint64_t v___x_1708_; uint64_t v___x_1709_; uint64_t v_fold_1710_; uint64_t v___x_1711_; uint64_t v___x_1712_; uint64_t v___x_1713_; size_t v___x_1714_; size_t v___x_1715_; size_t v___x_1716_; size_t v___x_1717_; size_t v___x_1718_; lean_object* v___x_1719_; uint8_t v___x_1720_; 
v_buckets_1705_ = lean_ctor_get(v_m_1703_, 1);
v___x_1706_ = lean_array_get_size(v_buckets_1705_);
v___x_1707_ = l_Lean_instHashableMVarId_hash(v_a_1704_);
v___x_1708_ = 32ULL;
v___x_1709_ = lean_uint64_shift_right(v___x_1707_, v___x_1708_);
v_fold_1710_ = lean_uint64_xor(v___x_1707_, v___x_1709_);
v___x_1711_ = 16ULL;
v___x_1712_ = lean_uint64_shift_right(v_fold_1710_, v___x_1711_);
v___x_1713_ = lean_uint64_xor(v_fold_1710_, v___x_1712_);
v___x_1714_ = lean_uint64_to_usize(v___x_1713_);
v___x_1715_ = lean_usize_of_nat(v___x_1706_);
v___x_1716_ = ((size_t)1ULL);
v___x_1717_ = lean_usize_sub(v___x_1715_, v___x_1716_);
v___x_1718_ = lean_usize_land(v___x_1714_, v___x_1717_);
v___x_1719_ = lean_array_uget_borrowed(v_buckets_1705_, v___x_1718_);
v___x_1720_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg(v_a_1704_, v___x_1719_);
return v___x_1720_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg___boxed(lean_object* v_m_1721_, lean_object* v_a_1722_){
_start:
{
uint8_t v_res_1723_; lean_object* v_r_1724_; 
v_res_1723_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg(v_m_1721_, v_a_1722_);
lean_dec(v_a_1722_);
lean_dec_ref(v_m_1721_);
v_r_1724_ = lean_box(v_res_1723_);
return v_r_1724_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___redArg(lean_object* v_keys_1725_, lean_object* v_i_1726_, lean_object* v_k_1727_){
_start:
{
lean_object* v___x_1728_; uint8_t v___x_1729_; 
v___x_1728_ = lean_array_get_size(v_keys_1725_);
v___x_1729_ = lean_nat_dec_lt(v_i_1726_, v___x_1728_);
if (v___x_1729_ == 0)
{
lean_dec(v_i_1726_);
return v___x_1729_;
}
else
{
lean_object* v_k_x27_1730_; uint8_t v___x_1731_; 
v_k_x27_1730_ = lean_array_fget_borrowed(v_keys_1725_, v_i_1726_);
v___x_1731_ = l_Lean_instBEqMVarId_beq(v_k_1727_, v_k_x27_1730_);
if (v___x_1731_ == 0)
{
lean_object* v___x_1732_; lean_object* v___x_1733_; 
v___x_1732_ = lean_unsigned_to_nat(1u);
v___x_1733_ = lean_nat_add(v_i_1726_, v___x_1732_);
lean_dec(v_i_1726_);
v_i_1726_ = v___x_1733_;
goto _start;
}
else
{
lean_dec(v_i_1726_);
return v___x_1731_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___redArg___boxed(lean_object* v_keys_1735_, lean_object* v_i_1736_, lean_object* v_k_1737_){
_start:
{
uint8_t v_res_1738_; lean_object* v_r_1739_; 
v_res_1738_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___redArg(v_keys_1735_, v_i_1736_, v_k_1737_);
lean_dec(v_k_1737_);
lean_dec_ref(v_keys_1735_);
v_r_1739_ = lean_box(v_res_1738_);
return v_r_1739_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___redArg(lean_object* v_x_1740_, size_t v_x_1741_, lean_object* v_x_1742_){
_start:
{
if (lean_obj_tag(v_x_1740_) == 0)
{
lean_object* v_es_1743_; lean_object* v___x_1744_; size_t v___x_1745_; size_t v___x_1746_; lean_object* v_j_1747_; lean_object* v___x_1748_; 
v_es_1743_ = lean_ctor_get(v_x_1740_, 0);
v___x_1744_ = lean_box(2);
v___x_1745_ = ((size_t)31ULL);
v___x_1746_ = lean_usize_land(v_x_1741_, v___x_1745_);
v_j_1747_ = lean_usize_to_nat(v___x_1746_);
v___x_1748_ = lean_array_get_borrowed(v___x_1744_, v_es_1743_, v_j_1747_);
lean_dec(v_j_1747_);
switch(lean_obj_tag(v___x_1748_))
{
case 0:
{
lean_object* v_key_1749_; uint8_t v___x_1750_; 
v_key_1749_ = lean_ctor_get(v___x_1748_, 0);
v___x_1750_ = l_Lean_instBEqMVarId_beq(v_x_1742_, v_key_1749_);
return v___x_1750_;
}
case 1:
{
lean_object* v_node_1751_; size_t v___x_1752_; size_t v___x_1753_; 
v_node_1751_ = lean_ctor_get(v___x_1748_, 0);
v___x_1752_ = ((size_t)5ULL);
v___x_1753_ = lean_usize_shift_right(v_x_1741_, v___x_1752_);
v_x_1740_ = v_node_1751_;
v_x_1741_ = v___x_1753_;
goto _start;
}
default: 
{
uint8_t v___x_1755_; 
v___x_1755_ = 0;
return v___x_1755_;
}
}
}
else
{
lean_object* v_ks_1756_; lean_object* v___x_1757_; uint8_t v___x_1758_; 
v_ks_1756_ = lean_ctor_get(v_x_1740_, 0);
v___x_1757_ = lean_unsigned_to_nat(0u);
v___x_1758_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___redArg(v_ks_1756_, v___x_1757_, v_x_1742_);
return v___x_1758_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___redArg___boxed(lean_object* v_x_1759_, lean_object* v_x_1760_, lean_object* v_x_1761_){
_start:
{
size_t v_x_122742__boxed_1762_; uint8_t v_res_1763_; lean_object* v_r_1764_; 
v_x_122742__boxed_1762_ = lean_unbox_usize(v_x_1760_);
lean_dec(v_x_1760_);
v_res_1763_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___redArg(v_x_1759_, v_x_122742__boxed_1762_, v_x_1761_);
lean_dec(v_x_1761_);
lean_dec_ref(v_x_1759_);
v_r_1764_ = lean_box(v_res_1763_);
return v_r_1764_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(lean_object* v_x_1765_, lean_object* v_x_1766_){
_start:
{
uint64_t v___x_1767_; size_t v___x_1768_; uint8_t v___x_1769_; 
v___x_1767_ = l_Lean_instHashableMVarId_hash(v_x_1766_);
v___x_1768_ = lean_uint64_to_usize(v___x_1767_);
v___x_1769_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___redArg(v_x_1765_, v___x_1768_, v_x_1766_);
return v___x_1769_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg___boxed(lean_object* v_x_1770_, lean_object* v_x_1771_){
_start:
{
uint8_t v_res_1772_; lean_object* v_r_1773_; 
v_res_1772_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(v_x_1770_, v_x_1771_);
lean_dec(v_x_1771_);
lean_dec_ref(v_x_1770_);
v_r_1773_ = lean_box(v_res_1772_);
return v_r_1773_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___redArg(lean_object* v_mvarId_1774_, lean_object* v___y_1775_){
_start:
{
lean_object* v___x_1777_; lean_object* v_mctx_1778_; lean_object* v_eAssignment_1779_; lean_object* v_dAssignment_1780_; uint8_t v___x_1781_; 
v___x_1777_ = lean_st_ref_get(v___y_1775_);
v_mctx_1778_ = lean_ctor_get(v___x_1777_, 0);
lean_inc_ref(v_mctx_1778_);
lean_dec(v___x_1777_);
v_eAssignment_1779_ = lean_ctor_get(v_mctx_1778_, 8);
lean_inc_ref(v_eAssignment_1779_);
v_dAssignment_1780_ = lean_ctor_get(v_mctx_1778_, 9);
lean_inc_ref(v_dAssignment_1780_);
lean_dec_ref(v_mctx_1778_);
v___x_1781_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(v_eAssignment_1779_, v_mvarId_1774_);
lean_dec_ref(v_eAssignment_1779_);
if (v___x_1781_ == 0)
{
uint8_t v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; 
v___x_1782_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(v_dAssignment_1780_, v_mvarId_1774_);
lean_dec_ref(v_dAssignment_1780_);
v___x_1783_ = lean_box(v___x_1782_);
v___x_1784_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1784_, 0, v___x_1783_);
return v___x_1784_;
}
else
{
lean_object* v___x_1785_; lean_object* v___x_1786_; 
lean_dec_ref(v_dAssignment_1780_);
v___x_1785_ = lean_box(v___x_1781_);
v___x_1786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1786_, 0, v___x_1785_);
return v___x_1786_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___redArg___boxed(lean_object* v_mvarId_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_){
_start:
{
lean_object* v_res_1790_; 
v_res_1790_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___redArg(v_mvarId_1787_, v___y_1788_);
lean_dec(v___y_1788_);
lean_dec(v_mvarId_1787_);
return v_res_1790_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__9(lean_object* v_a_1791_, lean_object* v_as_1792_, size_t v_sz_1793_, size_t v_i_1794_, lean_object* v_b_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_){
_start:
{
lean_object* v_a_1802_; uint8_t v___x_1806_; 
v___x_1806_ = lean_usize_dec_lt(v_i_1794_, v_sz_1793_);
if (v___x_1806_ == 0)
{
lean_object* v___x_1807_; 
v___x_1807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1807_, 0, v_b_1795_);
return v___x_1807_;
}
else
{
lean_object* v_a_1808_; lean_object* v___x_1809_; 
v_a_1808_ = lean_array_uget_borrowed(v_as_1792_, v_i_1794_);
v___x_1809_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___redArg(v_a_1808_, v___y_1797_);
if (lean_obj_tag(v___x_1809_) == 0)
{
lean_object* v_a_1810_; uint8_t v___x_1811_; 
v_a_1810_ = lean_ctor_get(v___x_1809_, 0);
lean_inc(v_a_1810_);
lean_dec_ref_known(v___x_1809_, 1);
v___x_1811_ = lean_unbox(v_a_1810_);
lean_dec(v_a_1810_);
if (v___x_1811_ == 0)
{
lean_object* v_fst_1812_; lean_object* v_snd_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1825_; 
v_fst_1812_ = lean_ctor_get(v_b_1795_, 0);
v_snd_1813_ = lean_ctor_get(v_b_1795_, 1);
v_isSharedCheck_1825_ = !lean_is_exclusive(v_b_1795_);
if (v_isSharedCheck_1825_ == 0)
{
v___x_1815_ = v_b_1795_;
v_isShared_1816_ = v_isSharedCheck_1825_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_snd_1813_);
lean_inc(v_fst_1812_);
lean_dec(v_b_1795_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1825_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
uint8_t v___x_1817_; 
v___x_1817_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg(v_a_1791_, v_a_1808_);
if (v___x_1817_ == 0)
{
lean_object* v___x_1818_; lean_object* v___x_1820_; 
lean_inc(v_a_1808_);
v___x_1818_ = lp_aesop_Aesop_UnorderedArraySet_insert___at___00Aesop_addRappUnsafe_spec__8(v_a_1808_, v_snd_1813_);
if (v_isShared_1816_ == 0)
{
lean_ctor_set(v___x_1815_, 1, v___x_1818_);
v___x_1820_ = v___x_1815_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1821_; 
v_reuseFailAlloc_1821_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1821_, 0, v_fst_1812_);
lean_ctor_set(v_reuseFailAlloc_1821_, 1, v___x_1818_);
v___x_1820_ = v_reuseFailAlloc_1821_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
v_a_1802_ = v___x_1820_;
goto v___jp_1801_;
}
}
else
{
lean_object* v___x_1823_; 
if (v_isShared_1816_ == 0)
{
v___x_1823_ = v___x_1815_;
goto v_reusejp_1822_;
}
else
{
lean_object* v_reuseFailAlloc_1824_; 
v_reuseFailAlloc_1824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1824_, 0, v_fst_1812_);
lean_ctor_set(v_reuseFailAlloc_1824_, 1, v_snd_1813_);
v___x_1823_ = v_reuseFailAlloc_1824_;
goto v_reusejp_1822_;
}
v_reusejp_1822_:
{
v_a_1802_ = v___x_1823_;
goto v___jp_1801_;
}
}
}
}
else
{
lean_object* v_fst_1826_; lean_object* v_snd_1827_; lean_object* v___x_1829_; uint8_t v_isShared_1830_; uint8_t v_isSharedCheck_1836_; 
v_fst_1826_ = lean_ctor_get(v_b_1795_, 0);
v_snd_1827_ = lean_ctor_get(v_b_1795_, 1);
v_isSharedCheck_1836_ = !lean_is_exclusive(v_b_1795_);
if (v_isSharedCheck_1836_ == 0)
{
v___x_1829_ = v_b_1795_;
v_isShared_1830_ = v_isSharedCheck_1836_;
goto v_resetjp_1828_;
}
else
{
lean_inc(v_snd_1827_);
lean_inc(v_fst_1826_);
lean_dec(v_b_1795_);
v___x_1829_ = lean_box(0);
v_isShared_1830_ = v_isSharedCheck_1836_;
goto v_resetjp_1828_;
}
v_resetjp_1828_:
{
lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1834_; 
lean_inc_n(v_a_1808_, 2);
v___x_1831_ = lp_aesop_Aesop_UnorderedArraySet_insert___at___00Aesop_addRappUnsafe_spec__8(v_a_1808_, v_fst_1826_);
v___x_1832_ = lp_aesop_Aesop_UnorderedArraySet_insert___at___00Aesop_addRappUnsafe_spec__8(v_a_1808_, v_snd_1827_);
if (v_isShared_1830_ == 0)
{
lean_ctor_set(v___x_1829_, 1, v___x_1832_);
lean_ctor_set(v___x_1829_, 0, v___x_1831_);
v___x_1834_ = v___x_1829_;
goto v_reusejp_1833_;
}
else
{
lean_object* v_reuseFailAlloc_1835_; 
v_reuseFailAlloc_1835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1835_, 0, v___x_1831_);
lean_ctor_set(v_reuseFailAlloc_1835_, 1, v___x_1832_);
v___x_1834_ = v_reuseFailAlloc_1835_;
goto v_reusejp_1833_;
}
v_reusejp_1833_:
{
v_a_1802_ = v___x_1834_;
goto v___jp_1801_;
}
}
}
}
else
{
lean_object* v_a_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1844_; 
lean_dec_ref(v_b_1795_);
v_a_1837_ = lean_ctor_get(v___x_1809_, 0);
v_isSharedCheck_1844_ = !lean_is_exclusive(v___x_1809_);
if (v_isSharedCheck_1844_ == 0)
{
v___x_1839_ = v___x_1809_;
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_a_1837_);
lean_dec(v___x_1809_);
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
v___jp_1801_:
{
size_t v___x_1803_; size_t v___x_1804_; 
v___x_1803_ = ((size_t)1ULL);
v___x_1804_ = lean_usize_add(v_i_1794_, v___x_1803_);
v_i_1794_ = v___x_1804_;
v_b_1795_ = v_a_1802_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__9___boxed(lean_object* v_a_1845_, lean_object* v_as_1846_, lean_object* v_sz_1847_, lean_object* v_i_1848_, lean_object* v_b_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_){
_start:
{
size_t v_sz_boxed_1855_; size_t v_i_boxed_1856_; lean_object* v_res_1857_; 
v_sz_boxed_1855_ = lean_unbox_usize(v_sz_1847_);
lean_dec(v_sz_1847_);
v_i_boxed_1856_ = lean_unbox_usize(v_i_1848_);
lean_dec(v_i_1848_);
v_res_1857_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__9(v_a_1845_, v_as_1846_, v_sz_boxed_1855_, v_i_boxed_1856_, v_b_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_);
lean_dec(v___y_1853_);
lean_dec_ref(v___y_1852_);
lean_dec(v___y_1851_);
lean_dec_ref(v___y_1850_);
lean_dec_ref(v_as_1846_);
lean_dec_ref(v_a_1845_);
return v_res_1857_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__1(lean_object* v_elimGoal_1858_, lean_object* v_val_1859_, lean_object* v___x_1860_, size_t v___x_1861_, lean_object* v___x_1862_, lean_object* v___x_1863_, lean_object* v___x_1864_, lean_object* v_goals_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_, lean_object* v___y_1869_){
_start:
{
lean_object* v_a_1872_; lean_object* v___y_1905_; uint8_t v___x_1915_; 
v___x_1915_ = lean_nat_dec_lt(v___x_1862_, v___x_1863_);
if (v___x_1915_ == 0)
{
v_a_1872_ = v___x_1864_;
goto v___jp_1871_;
}
else
{
uint8_t v___x_1916_; 
v___x_1916_ = lean_nat_dec_le(v___x_1863_, v___x_1863_);
if (v___x_1916_ == 0)
{
if (v___x_1915_ == 0)
{
v_a_1872_ = v___x_1864_;
goto v___jp_1871_;
}
else
{
size_t v___x_1917_; lean_object* v___x_1918_; 
v___x_1917_ = lean_usize_of_nat(v___x_1863_);
v___x_1918_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__10(v_goals_1865_, v___x_1861_, v___x_1917_, v___x_1864_, v___y_1866_, v___y_1867_, v___y_1868_, v___y_1869_);
v___y_1905_ = v___x_1918_;
goto v___jp_1904_;
}
}
else
{
size_t v___x_1919_; lean_object* v___x_1920_; 
v___x_1919_ = lean_usize_of_nat(v___x_1863_);
v___x_1920_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__10(v_goals_1865_, v___x_1861_, v___x_1919_, v___x_1864_, v___y_1866_, v___y_1867_, v___y_1868_, v___y_1869_);
v___y_1905_ = v___x_1920_;
goto v___jp_1904_;
}
}
v___jp_1871_:
{
lean_object* v___x_1873_; lean_object* v_mvars_1874_; lean_object* v___x_1875_; size_t v_sz_1876_; lean_object* v___x_1877_; 
v___x_1873_ = lean_apply_1(v_elimGoal_1858_, v_val_1859_);
v_mvars_1874_ = lean_ctor_get(v___x_1873_, 7);
lean_inc_ref(v_mvars_1874_);
lean_dec_ref(v___x_1873_);
lean_inc_ref(v___x_1860_);
v___x_1875_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1875_, 0, v___x_1860_);
lean_ctor_set(v___x_1875_, 1, v___x_1860_);
v_sz_1876_ = lean_array_size(v_mvars_1874_);
v___x_1877_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__9(v_a_1872_, v_mvars_1874_, v_sz_1876_, v___x_1861_, v___x_1875_, v___y_1866_, v___y_1867_, v___y_1868_, v___y_1869_);
lean_dec_ref(v_mvars_1874_);
if (lean_obj_tag(v___x_1877_) == 0)
{
lean_object* v_a_1878_; lean_object* v___x_1880_; uint8_t v_isShared_1881_; uint8_t v_isSharedCheck_1895_; 
v_a_1878_ = lean_ctor_get(v___x_1877_, 0);
v_isSharedCheck_1895_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1895_ == 0)
{
v___x_1880_ = v___x_1877_;
v_isShared_1881_ = v_isSharedCheck_1895_;
goto v_resetjp_1879_;
}
else
{
lean_inc(v_a_1878_);
lean_dec(v___x_1877_);
v___x_1880_ = lean_box(0);
v_isShared_1881_ = v_isSharedCheck_1895_;
goto v_resetjp_1879_;
}
v_resetjp_1879_:
{
lean_object* v_fst_1882_; lean_object* v_snd_1883_; lean_object* v___x_1885_; uint8_t v_isShared_1886_; uint8_t v_isSharedCheck_1894_; 
v_fst_1882_ = lean_ctor_get(v_a_1878_, 0);
v_snd_1883_ = lean_ctor_get(v_a_1878_, 1);
v_isSharedCheck_1894_ = !lean_is_exclusive(v_a_1878_);
if (v_isSharedCheck_1894_ == 0)
{
v___x_1885_ = v_a_1878_;
v_isShared_1886_ = v_isSharedCheck_1894_;
goto v_resetjp_1884_;
}
else
{
lean_inc(v_snd_1883_);
lean_inc(v_fst_1882_);
lean_dec(v_a_1878_);
v___x_1885_ = lean_box(0);
v_isShared_1886_ = v_isSharedCheck_1894_;
goto v_resetjp_1884_;
}
v_resetjp_1884_:
{
lean_object* v___x_1888_; 
if (v_isShared_1886_ == 0)
{
v___x_1888_ = v___x_1885_;
goto v_reusejp_1887_;
}
else
{
lean_object* v_reuseFailAlloc_1893_; 
v_reuseFailAlloc_1893_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1893_, 0, v_fst_1882_);
lean_ctor_set(v_reuseFailAlloc_1893_, 1, v_snd_1883_);
v___x_1888_ = v_reuseFailAlloc_1893_;
goto v_reusejp_1887_;
}
v_reusejp_1887_:
{
lean_object* v___x_1889_; lean_object* v___x_1891_; 
v___x_1889_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1889_, 0, v_a_1872_);
lean_ctor_set(v___x_1889_, 1, v___x_1888_);
if (v_isShared_1881_ == 0)
{
lean_ctor_set(v___x_1880_, 0, v___x_1889_);
v___x_1891_ = v___x_1880_;
goto v_reusejp_1890_;
}
else
{
lean_object* v_reuseFailAlloc_1892_; 
v_reuseFailAlloc_1892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1892_, 0, v___x_1889_);
v___x_1891_ = v_reuseFailAlloc_1892_;
goto v_reusejp_1890_;
}
v_reusejp_1890_:
{
return v___x_1891_;
}
}
}
}
}
else
{
lean_object* v_a_1896_; lean_object* v___x_1898_; uint8_t v_isShared_1899_; uint8_t v_isSharedCheck_1903_; 
lean_dec_ref(v_a_1872_);
v_a_1896_ = lean_ctor_get(v___x_1877_, 0);
v_isSharedCheck_1903_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1903_ == 0)
{
v___x_1898_ = v___x_1877_;
v_isShared_1899_ = v_isSharedCheck_1903_;
goto v_resetjp_1897_;
}
else
{
lean_inc(v_a_1896_);
lean_dec(v___x_1877_);
v___x_1898_ = lean_box(0);
v_isShared_1899_ = v_isSharedCheck_1903_;
goto v_resetjp_1897_;
}
v_resetjp_1897_:
{
lean_object* v___x_1901_; 
if (v_isShared_1899_ == 0)
{
v___x_1901_ = v___x_1898_;
goto v_reusejp_1900_;
}
else
{
lean_object* v_reuseFailAlloc_1902_; 
v_reuseFailAlloc_1902_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1902_, 0, v_a_1896_);
v___x_1901_ = v_reuseFailAlloc_1902_;
goto v_reusejp_1900_;
}
v_reusejp_1900_:
{
return v___x_1901_;
}
}
}
}
v___jp_1904_:
{
if (lean_obj_tag(v___y_1905_) == 0)
{
lean_object* v_a_1906_; 
v_a_1906_ = lean_ctor_get(v___y_1905_, 0);
lean_inc(v_a_1906_);
lean_dec_ref_known(v___y_1905_, 1);
v_a_1872_ = v_a_1906_;
goto v___jp_1871_;
}
else
{
lean_object* v_a_1907_; lean_object* v___x_1909_; uint8_t v_isShared_1910_; uint8_t v_isSharedCheck_1914_; 
lean_dec_ref(v___x_1860_);
lean_dec(v_val_1859_);
lean_dec_ref(v_elimGoal_1858_);
v_a_1907_ = lean_ctor_get(v___y_1905_, 0);
v_isSharedCheck_1914_ = !lean_is_exclusive(v___y_1905_);
if (v_isSharedCheck_1914_ == 0)
{
v___x_1909_ = v___y_1905_;
v_isShared_1910_ = v_isSharedCheck_1914_;
goto v_resetjp_1908_;
}
else
{
lean_inc(v_a_1907_);
lean_dec(v___y_1905_);
v___x_1909_ = lean_box(0);
v_isShared_1910_ = v_isSharedCheck_1914_;
goto v_resetjp_1908_;
}
v_resetjp_1908_:
{
lean_object* v___x_1912_; 
if (v_isShared_1910_ == 0)
{
v___x_1912_ = v___x_1909_;
goto v_reusejp_1911_;
}
else
{
lean_object* v_reuseFailAlloc_1913_; 
v_reuseFailAlloc_1913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1913_, 0, v_a_1907_);
v___x_1912_ = v_reuseFailAlloc_1913_;
goto v_reusejp_1911_;
}
v_reusejp_1911_:
{
return v___x_1912_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__1___boxed(lean_object* v_elimGoal_1921_, lean_object* v_val_1922_, lean_object* v___x_1923_, lean_object* v___x_1924_, lean_object* v___x_1925_, lean_object* v___x_1926_, lean_object* v___x_1927_, lean_object* v_goals_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_){
_start:
{
size_t v___x_122912__boxed_1934_; lean_object* v_res_1935_; 
v___x_122912__boxed_1934_ = lean_unbox_usize(v___x_1924_);
lean_dec(v___x_1924_);
v_res_1935_ = lp_aesop_Aesop_addRappUnsafe___lam__1(v_elimGoal_1921_, v_val_1922_, v___x_1923_, v___x_122912__boxed_1934_, v___x_1925_, v___x_1926_, v___x_1927_, v_goals_1928_, v___y_1929_, v___y_1930_, v___y_1931_, v___y_1932_);
lean_dec(v___y_1932_);
lean_dec_ref(v___y_1931_);
lean_dec(v___y_1930_);
lean_dec_ref(v___y_1929_);
lean_dec_ref(v_goals_1928_);
lean_dec(v___x_1926_);
lean_dec(v___x_1925_);
return v_res_1935_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___redArg(lean_object* v_mvarId_1936_, lean_object* v___y_1937_){
_start:
{
lean_object* v___x_1939_; lean_object* v_mctx_1940_; lean_object* v_eAssignment_1941_; lean_object* v_dAssignment_1942_; uint8_t v___x_1943_; 
v___x_1939_ = lean_st_ref_get(v___y_1937_);
v_mctx_1940_ = lean_ctor_get(v___x_1939_, 0);
lean_inc_ref(v_mctx_1940_);
lean_dec(v___x_1939_);
v_eAssignment_1941_ = lean_ctor_get(v_mctx_1940_, 8);
lean_inc_ref(v_eAssignment_1941_);
v_dAssignment_1942_ = lean_ctor_get(v_mctx_1940_, 9);
lean_inc_ref(v_dAssignment_1942_);
lean_dec_ref(v_mctx_1940_);
v___x_1943_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(v_eAssignment_1941_, v_mvarId_1936_);
lean_dec_ref(v_eAssignment_1941_);
if (v___x_1943_ == 0)
{
uint8_t v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; 
v___x_1944_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(v_dAssignment_1942_, v_mvarId_1936_);
lean_dec_ref(v_dAssignment_1942_);
v___x_1945_ = lean_box(v___x_1944_);
v___x_1946_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1946_, 0, v___x_1945_);
return v___x_1946_;
}
else
{
lean_object* v___x_1947_; lean_object* v___x_1948_; 
lean_dec_ref(v_dAssignment_1942_);
v___x_1947_ = lean_box(v___x_1943_);
v___x_1948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1948_, 0, v___x_1947_);
return v___x_1948_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___redArg___boxed(lean_object* v_mvarId_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_){
_start:
{
lean_object* v_res_1952_; 
v_res_1952_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___redArg(v_mvarId_1949_, v___y_1950_);
lean_dec(v___y_1950_);
lean_dec(v_mvarId_1949_);
return v_res_1952_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__15(lean_object* v_val_1953_, lean_object* v___y_1954_, lean_object* v_as_1955_, size_t v_sz_1956_, size_t v_i_1957_, lean_object* v_b_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_){
_start:
{
lean_object* v_a_1968_; uint8_t v___x_1972_; 
v___x_1972_ = lean_usize_dec_lt(v_i_1957_, v_sz_1956_);
if (v___x_1972_ == 0)
{
lean_object* v___x_1973_; 
lean_dec(v_val_1953_);
v___x_1973_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1973_, 0, v_b_1958_);
return v___x_1973_;
}
else
{
lean_object* v_a_1974_; uint8_t v_a_1976_; uint8_t v___x_1989_; 
v_a_1974_ = lean_array_uget_borrowed(v_as_1955_, v_i_1957_);
v___x_1989_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg(v___y_1954_, v_a_1974_);
if (v___x_1989_ == 0)
{
lean_object* v___x_1990_; 
v___x_1990_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___redArg(v_a_1974_, v___y_1963_);
if (lean_obj_tag(v___x_1990_) == 0)
{
lean_object* v_a_1991_; uint8_t v___x_1992_; 
v_a_1991_ = lean_ctor_get(v___x_1990_, 0);
lean_inc(v_a_1991_);
lean_dec_ref_known(v___x_1990_, 1);
v___x_1992_ = lean_unbox(v_a_1991_);
lean_dec(v_a_1991_);
v_a_1976_ = v___x_1992_;
goto v___jp_1975_;
}
else
{
lean_object* v_a_1993_; lean_object* v___x_1995_; uint8_t v_isShared_1996_; uint8_t v_isSharedCheck_2000_; 
lean_dec_ref(v_b_1958_);
lean_dec(v_val_1953_);
v_a_1993_ = lean_ctor_get(v___x_1990_, 0);
v_isSharedCheck_2000_ = !lean_is_exclusive(v___x_1990_);
if (v_isSharedCheck_2000_ == 0)
{
v___x_1995_ = v___x_1990_;
v_isShared_1996_ = v_isSharedCheck_2000_;
goto v_resetjp_1994_;
}
else
{
lean_inc(v_a_1993_);
lean_dec(v___x_1990_);
v___x_1995_ = lean_box(0);
v_isShared_1996_ = v_isSharedCheck_2000_;
goto v_resetjp_1994_;
}
v_resetjp_1994_:
{
lean_object* v___x_1998_; 
if (v_isShared_1996_ == 0)
{
v___x_1998_ = v___x_1995_;
goto v_reusejp_1997_;
}
else
{
lean_object* v_reuseFailAlloc_1999_; 
v_reuseFailAlloc_1999_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1999_, 0, v_a_1993_);
v___x_1998_ = v_reuseFailAlloc_1999_;
goto v_reusejp_1997_;
}
v_reusejp_1997_:
{
return v___x_1998_;
}
}
}
}
else
{
v_a_1976_ = v___x_1989_;
goto v___jp_1975_;
}
v___jp_1975_:
{
if (v_a_1976_ == 0)
{
lean_object* v___x_1977_; lean_object* v___x_1978_; 
lean_inc(v_val_1953_);
v___x_1977_ = lp_aesop_Aesop_Goal_currentGoal(v_val_1953_);
lean_inc(v_a_1974_);
v___x_1978_ = lp_aesop_Aesop_diffGoals(v___x_1977_, v_a_1974_, v___y_1961_, v___y_1962_, v___y_1963_, v___y_1964_, v___y_1965_);
if (lean_obj_tag(v___x_1978_) == 0)
{
lean_object* v_a_1979_; lean_object* v___x_1980_; 
v_a_1979_ = lean_ctor_get(v___x_1978_, 0);
lean_inc(v_a_1979_);
lean_dec_ref_known(v___x_1978_, 1);
v___x_1980_ = lean_array_push(v_b_1958_, v_a_1979_);
v_a_1968_ = v___x_1980_;
goto v___jp_1967_;
}
else
{
lean_object* v_a_1981_; lean_object* v___x_1983_; uint8_t v_isShared_1984_; uint8_t v_isSharedCheck_1988_; 
lean_dec_ref(v_b_1958_);
lean_dec(v_val_1953_);
v_a_1981_ = lean_ctor_get(v___x_1978_, 0);
v_isSharedCheck_1988_ = !lean_is_exclusive(v___x_1978_);
if (v_isSharedCheck_1988_ == 0)
{
v___x_1983_ = v___x_1978_;
v_isShared_1984_ = v_isSharedCheck_1988_;
goto v_resetjp_1982_;
}
else
{
lean_inc(v_a_1981_);
lean_dec(v___x_1978_);
v___x_1983_ = lean_box(0);
v_isShared_1984_ = v_isSharedCheck_1988_;
goto v_resetjp_1982_;
}
v_resetjp_1982_:
{
lean_object* v___x_1986_; 
if (v_isShared_1984_ == 0)
{
v___x_1986_ = v___x_1983_;
goto v_reusejp_1985_;
}
else
{
lean_object* v_reuseFailAlloc_1987_; 
v_reuseFailAlloc_1987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1987_, 0, v_a_1981_);
v___x_1986_ = v_reuseFailAlloc_1987_;
goto v_reusejp_1985_;
}
v_reusejp_1985_:
{
return v___x_1986_;
}
}
}
}
else
{
v_a_1968_ = v_b_1958_;
goto v___jp_1967_;
}
}
}
v___jp_1967_:
{
size_t v___x_1969_; size_t v___x_1970_; 
v___x_1969_ = ((size_t)1ULL);
v___x_1970_ = lean_usize_add(v_i_1957_, v___x_1969_);
v_i_1957_ = v___x_1970_;
v_b_1958_ = v_a_1968_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__15___boxed(lean_object* v_val_2001_, lean_object* v___y_2002_, lean_object* v_as_2003_, lean_object* v_sz_2004_, lean_object* v_i_2005_, lean_object* v_b_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_, lean_object* v___y_2009_, lean_object* v___y_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_){
_start:
{
size_t v_sz_boxed_2015_; size_t v_i_boxed_2016_; lean_object* v_res_2017_; 
v_sz_boxed_2015_ = lean_unbox_usize(v_sz_2004_);
lean_dec(v_sz_2004_);
v_i_boxed_2016_ = lean_unbox_usize(v_i_2005_);
lean_dec(v_i_2005_);
v_res_2017_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__15(v_val_2001_, v___y_2002_, v_as_2003_, v_sz_boxed_2015_, v_i_boxed_2016_, v_b_2006_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_, v___y_2011_, v___y_2012_, v___y_2013_);
lean_dec(v___y_2013_);
lean_dec_ref(v___y_2012_);
lean_dec(v___y_2011_);
lean_dec_ref(v___y_2010_);
lean_dec(v___y_2009_);
lean_dec(v___y_2008_);
lean_dec_ref(v___y_2007_);
lean_dec_ref(v_as_2003_);
lean_dec_ref(v___y_2002_);
return v_res_2017_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__2(lean_object* v_val_2018_, lean_object* v___y_2019_, lean_object* v_mvars_2020_, size_t v_sz_2021_, size_t v___x_2022_, lean_object* v___x_2023_, lean_object* v___y_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_){
_start:
{
lean_object* v___x_2032_; 
v___x_2032_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__15(v_val_2018_, v___y_2019_, v_mvars_2020_, v_sz_2021_, v___x_2022_, v___x_2023_, v___y_2024_, v___y_2025_, v___y_2026_, v___y_2027_, v___y_2028_, v___y_2029_, v___y_2030_);
return v___x_2032_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___lam__2___boxed(lean_object* v_val_2033_, lean_object* v___y_2034_, lean_object* v_mvars_2035_, lean_object* v_sz_2036_, lean_object* v___x_2037_, lean_object* v___x_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_){
_start:
{
size_t v_sz_boxed_2047_; size_t v___x_123153__boxed_2048_; lean_object* v_res_2049_; 
v_sz_boxed_2047_ = lean_unbox_usize(v_sz_2036_);
lean_dec(v_sz_2036_);
v___x_123153__boxed_2048_ = lean_unbox_usize(v___x_2037_);
lean_dec(v___x_2037_);
v_res_2049_ = lp_aesop_Aesop_addRappUnsafe___lam__2(v_val_2033_, v___y_2034_, v_mvars_2035_, v_sz_boxed_2047_, v___x_123153__boxed_2048_, v___x_2038_, v___y_2039_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_, v___y_2044_, v___y_2045_);
lean_dec(v___y_2045_);
lean_dec_ref(v___y_2044_);
lean_dec(v___y_2043_);
lean_dec_ref(v___y_2042_);
lean_dec(v___y_2041_);
lean_dec(v___y_2040_);
lean_dec_ref(v___y_2039_);
lean_dec_ref(v_mvars_2035_);
lean_dec_ref(v___y_2034_);
return v_res_2049_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0_spec__0(lean_object* v_as_2050_, size_t v_sz_2051_, size_t v_i_2052_, lean_object* v_b_2053_){
_start:
{
uint8_t v___x_2054_; 
v___x_2054_ = lean_usize_dec_lt(v_i_2052_, v_sz_2051_);
if (v___x_2054_ == 0)
{
return v_b_2053_;
}
else
{
lean_object* v_a_2055_; lean_object* v___x_2056_; lean_object* v_r_2057_; size_t v___x_2058_; size_t v___x_2059_; 
v_a_2055_ = lean_array_uget_borrowed(v_as_2050_, v_i_2052_);
v___x_2056_ = lean_box(0);
lean_inc(v_a_2055_);
v_r_2057_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13___redArg(v_b_2053_, v_a_2055_, v___x_2056_);
v___x_2058_ = ((size_t)1ULL);
v___x_2059_ = lean_usize_add(v_i_2052_, v___x_2058_);
v_i_2052_ = v___x_2059_;
v_b_2053_ = v_r_2057_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0_spec__0___boxed(lean_object* v_as_2061_, lean_object* v_sz_2062_, lean_object* v_i_2063_, lean_object* v_b_2064_){
_start:
{
size_t v_sz_boxed_2065_; size_t v_i_boxed_2066_; lean_object* v_res_2067_; 
v_sz_boxed_2065_ = lean_unbox_usize(v_sz_2062_);
lean_dec(v_sz_2062_);
v_i_boxed_2066_ = lean_unbox_usize(v_i_2063_);
lean_dec(v_i_2063_);
v_res_2067_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0_spec__0(v_as_2061_, v_sz_boxed_2065_, v_i_boxed_2066_, v_b_2064_);
lean_dec_ref(v_as_2061_);
return v_res_2067_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0(lean_object* v_m_2068_, lean_object* v_l_2069_){
_start:
{
size_t v_sz_2070_; size_t v___x_2071_; lean_object* v___x_2072_; 
v_sz_2070_ = lean_array_size(v_l_2069_);
v___x_2071_ = ((size_t)0ULL);
v___x_2072_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0_spec__0(v_l_2069_, v_sz_2070_, v___x_2071_, v_m_2068_);
return v___x_2072_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0___boxed(lean_object* v_m_2073_, lean_object* v_l_2074_){
_start:
{
lean_object* v_res_2075_; 
v_res_2075_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0(v_m_2073_, v_l_2074_);
lean_dec_ref(v_l_2074_);
return v_res_2075_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__22(lean_object* v_as_2076_, size_t v_i_2077_, size_t v_stop_2078_, lean_object* v_b_2079_){
_start:
{
uint8_t v___x_2080_; 
v___x_2080_ = lean_usize_dec_eq(v_i_2077_, v_stop_2078_);
if (v___x_2080_ == 0)
{
lean_object* v___x_2081_; lean_object* v_elimGoal_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v_mvars_2085_; lean_object* v___x_2086_; size_t v___x_2087_; size_t v___x_2088_; 
v___x_2081_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2082_ = lean_ctor_get(v___x_2081_, 1);
v___x_2083_ = lean_array_uget_borrowed(v_as_2076_, v_i_2077_);
lean_inc_ref(v_elimGoal_2082_);
lean_inc(v___x_2083_);
v___x_2084_ = lean_apply_1(v_elimGoal_2082_, v___x_2083_);
v_mvars_2085_ = lean_ctor_get(v___x_2084_, 7);
lean_inc_ref(v_mvars_2085_);
lean_dec_ref(v___x_2084_);
v___x_2086_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Aesop_addRappUnsafe_spec__0(v_b_2079_, v_mvars_2085_);
lean_dec_ref(v_mvars_2085_);
v___x_2087_ = ((size_t)1ULL);
v___x_2088_ = lean_usize_add(v_i_2077_, v___x_2087_);
v_i_2077_ = v___x_2088_;
v_b_2079_ = v___x_2086_;
goto _start;
}
else
{
return v_b_2079_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__22___boxed(lean_object* v_as_2090_, lean_object* v_i_2091_, lean_object* v_stop_2092_, lean_object* v_b_2093_){
_start:
{
size_t v_i_boxed_2094_; size_t v_stop_boxed_2095_; lean_object* v_res_2096_; 
v_i_boxed_2094_ = lean_unbox_usize(v_i_2091_);
lean_dec(v_i_2091_);
v_stop_boxed_2095_ = lean_unbox_usize(v_stop_2092_);
lean_dec(v_stop_2092_);
v_res_2096_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__22(v_as_2090_, v_i_boxed_2094_, v_stop_boxed_2095_, v_b_2093_);
lean_dec_ref(v_as_2090_);
return v_res_2096_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__11(size_t v_sz_2097_, size_t v_i_2098_, lean_object* v_bs_2099_){
_start:
{
uint8_t v___x_2100_; 
v___x_2100_ = lean_usize_dec_lt(v_i_2098_, v_sz_2097_);
if (v___x_2100_ == 0)
{
return v_bs_2099_;
}
else
{
lean_object* v___x_2101_; lean_object* v_oldGoal_2102_; lean_object* v_addedFVars_2103_; lean_object* v_removedFVars_2104_; uint8_t v_targetChanged_2105_; lean_object* v___x_2106_; lean_object* v_elimGoal_2107_; lean_object* v_v_2108_; lean_object* v___x_2109_; lean_object* v_preNormGoal_2110_; lean_object* v___x_2111_; lean_object* v_bs_x27_2112_; lean_object* v___x_2113_; size_t v___x_2114_; size_t v___x_2115_; lean_object* v___x_2116_; 
v___x_2101_ = lp_aesop_Aesop_instInhabitedGoalDiff_default;
v_oldGoal_2102_ = lean_ctor_get(v___x_2101_, 0);
v_addedFVars_2103_ = lean_ctor_get(v___x_2101_, 2);
v_removedFVars_2104_ = lean_ctor_get(v___x_2101_, 3);
v_targetChanged_2105_ = lean_ctor_get_uint8(v___x_2101_, sizeof(void*)*4);
v___x_2106_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2107_ = lean_ctor_get(v___x_2106_, 1);
v_v_2108_ = lean_array_uget_borrowed(v_bs_2099_, v_i_2098_);
lean_inc_ref(v_elimGoal_2107_);
lean_inc(v_v_2108_);
v___x_2109_ = lean_apply_1(v_elimGoal_2107_, v_v_2108_);
v_preNormGoal_2110_ = lean_ctor_get(v___x_2109_, 5);
lean_inc(v_preNormGoal_2110_);
lean_dec_ref(v___x_2109_);
v___x_2111_ = lean_unsigned_to_nat(0u);
v_bs_x27_2112_ = lean_array_uset(v_bs_2099_, v_i_2098_, v___x_2111_);
lean_inc_ref(v_removedFVars_2104_);
lean_inc_ref(v_addedFVars_2103_);
lean_inc(v_oldGoal_2102_);
v___x_2113_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_2113_, 0, v_oldGoal_2102_);
lean_ctor_set(v___x_2113_, 1, v_preNormGoal_2110_);
lean_ctor_set(v___x_2113_, 2, v_addedFVars_2103_);
lean_ctor_set(v___x_2113_, 3, v_removedFVars_2104_);
lean_ctor_set_uint8(v___x_2113_, sizeof(void*)*4, v_targetChanged_2105_);
v___x_2114_ = ((size_t)1ULL);
v___x_2115_ = lean_usize_add(v_i_2098_, v___x_2114_);
v___x_2116_ = lean_array_uset(v_bs_x27_2112_, v_i_2098_, v___x_2113_);
v_i_2098_ = v___x_2115_;
v_bs_2099_ = v___x_2116_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__11___boxed(lean_object* v_sz_2118_, lean_object* v_i_2119_, lean_object* v_bs_2120_){
_start:
{
size_t v_sz_boxed_2121_; size_t v_i_boxed_2122_; lean_object* v_res_2123_; 
v_sz_boxed_2121_ = lean_unbox_usize(v_sz_2118_);
lean_dec(v_sz_2118_);
v_i_boxed_2122_ = lean_unbox_usize(v_i_2119_);
lean_dec(v_i_2119_);
v_res_2123_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__11(v_sz_boxed_2121_, v_i_boxed_2122_, v_bs_2120_);
return v_res_2123_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41(size_t v_i_2124_, lean_object* v_u_2125_){
_start:
{
lean_object* v_parents_2126_; lean_object* v_parent_2127_; size_t v___x_2128_; uint8_t v___x_2129_; 
v_parents_2126_ = lean_ctor_get(v_u_2125_, 0);
v_parent_2127_ = lean_array_uget_borrowed(v_parents_2126_, v_i_2124_);
v___x_2128_ = lean_unbox_usize(v_parent_2127_);
v___x_2129_ = lean_usize_dec_eq(v___x_2128_, v_i_2124_);
if (v___x_2129_ == 0)
{
size_t v___x_2130_; lean_object* v___x_2131_; lean_object* v_snd_2132_; lean_object* v_fst_2133_; lean_object* v___x_2135_; uint8_t v_isShared_2136_; uint8_t v_isSharedCheck_2151_; 
v___x_2130_ = lean_unbox_usize(v_parent_2127_);
v___x_2131_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41(v___x_2130_, v_u_2125_);
v_snd_2132_ = lean_ctor_get(v___x_2131_, 1);
v_fst_2133_ = lean_ctor_get(v___x_2131_, 0);
v_isSharedCheck_2151_ = !lean_is_exclusive(v___x_2131_);
if (v_isSharedCheck_2151_ == 0)
{
v___x_2135_ = v___x_2131_;
v_isShared_2136_ = v_isSharedCheck_2151_;
goto v_resetjp_2134_;
}
else
{
lean_inc(v_snd_2132_);
lean_inc(v_fst_2133_);
lean_dec(v___x_2131_);
v___x_2135_ = lean_box(0);
v_isShared_2136_ = v_isSharedCheck_2151_;
goto v_resetjp_2134_;
}
v_resetjp_2134_:
{
lean_object* v_parents_2137_; lean_object* v_sizes_2138_; lean_object* v_toRep_2139_; lean_object* v___x_2141_; uint8_t v_isShared_2142_; uint8_t v_isSharedCheck_2150_; 
v_parents_2137_ = lean_ctor_get(v_snd_2132_, 0);
v_sizes_2138_ = lean_ctor_get(v_snd_2132_, 1);
v_toRep_2139_ = lean_ctor_get(v_snd_2132_, 2);
v_isSharedCheck_2150_ = !lean_is_exclusive(v_snd_2132_);
if (v_isSharedCheck_2150_ == 0)
{
v___x_2141_ = v_snd_2132_;
v_isShared_2142_ = v_isSharedCheck_2150_;
goto v_resetjp_2140_;
}
else
{
lean_inc(v_toRep_2139_);
lean_inc(v_sizes_2138_);
lean_inc(v_parents_2137_);
lean_dec(v_snd_2132_);
v___x_2141_ = lean_box(0);
v_isShared_2142_ = v_isSharedCheck_2150_;
goto v_resetjp_2140_;
}
v_resetjp_2140_:
{
lean_object* v___x_2143_; lean_object* v___x_2145_; 
lean_inc(v_fst_2133_);
v___x_2143_ = lean_array_uset(v_parents_2137_, v_i_2124_, v_fst_2133_);
if (v_isShared_2142_ == 0)
{
lean_ctor_set(v___x_2141_, 0, v___x_2143_);
v___x_2145_ = v___x_2141_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2149_; 
v_reuseFailAlloc_2149_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2149_, 0, v___x_2143_);
lean_ctor_set(v_reuseFailAlloc_2149_, 1, v_sizes_2138_);
lean_ctor_set(v_reuseFailAlloc_2149_, 2, v_toRep_2139_);
v___x_2145_ = v_reuseFailAlloc_2149_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
lean_object* v___x_2147_; 
if (v_isShared_2136_ == 0)
{
lean_ctor_set(v___x_2135_, 1, v___x_2145_);
v___x_2147_ = v___x_2135_;
goto v_reusejp_2146_;
}
else
{
lean_object* v_reuseFailAlloc_2148_; 
v_reuseFailAlloc_2148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2148_, 0, v_fst_2133_);
lean_ctor_set(v_reuseFailAlloc_2148_, 1, v___x_2145_);
v___x_2147_ = v_reuseFailAlloc_2148_;
goto v_reusejp_2146_;
}
v_reusejp_2146_:
{
return v___x_2147_;
}
}
}
}
}
else
{
lean_object* v___x_2152_; 
lean_inc(v_parent_2127_);
v___x_2152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2152_, 0, v_parent_2127_);
lean_ctor_set(v___x_2152_, 1, v_u_2125_);
return v___x_2152_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41___boxed(lean_object* v_i_2153_, lean_object* v_u_2154_){
_start:
{
size_t v_i_boxed_2155_; lean_object* v_res_2156_; 
v_i_boxed_2155_ = lean_unbox_usize(v_i_2153_);
lean_dec(v_i_2153_);
v_res_2156_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41(v_i_boxed_2155_, v_u_2154_);
return v_res_2156_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___redArg(size_t v_a_2157_, lean_object* v_x_2158_){
_start:
{
if (lean_obj_tag(v_x_2158_) == 0)
{
uint8_t v___x_2159_; 
v___x_2159_ = 0;
return v___x_2159_;
}
else
{
lean_object* v_key_2160_; lean_object* v_tail_2161_; size_t v___x_2162_; uint8_t v___x_2163_; 
v_key_2160_ = lean_ctor_get(v_x_2158_, 0);
v_tail_2161_ = lean_ctor_get(v_x_2158_, 2);
v___x_2162_ = lean_unbox_usize(v_key_2160_);
v___x_2163_ = lean_usize_dec_eq(v___x_2162_, v_a_2157_);
if (v___x_2163_ == 0)
{
v_x_2158_ = v_tail_2161_;
goto _start;
}
else
{
return v___x_2163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___redArg___boxed(lean_object* v_a_2165_, lean_object* v_x_2166_){
_start:
{
size_t v_a_boxed_2167_; uint8_t v_res_2168_; lean_object* v_r_2169_; 
v_a_boxed_2167_ = lean_unbox_usize(v_a_2165_);
lean_dec(v_a_2165_);
v_res_2168_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___redArg(v_a_boxed_2167_, v_x_2166_);
lean_dec(v_x_2166_);
v_r_2169_ = lean_box(v_res_2168_);
return v_r_2169_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg(size_t v_a_2170_, lean_object* v_b_2171_, lean_object* v_x_2172_){
_start:
{
if (lean_obj_tag(v_x_2172_) == 0)
{
lean_dec(v_b_2171_);
return v_x_2172_;
}
else
{
lean_object* v_key_2173_; lean_object* v_value_2174_; lean_object* v_tail_2175_; lean_object* v___x_2177_; uint8_t v_isShared_2178_; uint8_t v_isSharedCheck_2189_; 
v_key_2173_ = lean_ctor_get(v_x_2172_, 0);
v_value_2174_ = lean_ctor_get(v_x_2172_, 1);
v_tail_2175_ = lean_ctor_get(v_x_2172_, 2);
v_isSharedCheck_2189_ = !lean_is_exclusive(v_x_2172_);
if (v_isSharedCheck_2189_ == 0)
{
v___x_2177_ = v_x_2172_;
v_isShared_2178_ = v_isSharedCheck_2189_;
goto v_resetjp_2176_;
}
else
{
lean_inc(v_tail_2175_);
lean_inc(v_value_2174_);
lean_inc(v_key_2173_);
lean_dec(v_x_2172_);
v___x_2177_ = lean_box(0);
v_isShared_2178_ = v_isSharedCheck_2189_;
goto v_resetjp_2176_;
}
v_resetjp_2176_:
{
size_t v___x_2179_; uint8_t v___x_2180_; 
v___x_2179_ = lean_unbox_usize(v_key_2173_);
v___x_2180_ = lean_usize_dec_eq(v___x_2179_, v_a_2170_);
if (v___x_2180_ == 0)
{
lean_object* v___x_2181_; lean_object* v___x_2183_; 
v___x_2181_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg(v_a_2170_, v_b_2171_, v_tail_2175_);
if (v_isShared_2178_ == 0)
{
lean_ctor_set(v___x_2177_, 2, v___x_2181_);
v___x_2183_ = v___x_2177_;
goto v_reusejp_2182_;
}
else
{
lean_object* v_reuseFailAlloc_2184_; 
v_reuseFailAlloc_2184_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2184_, 0, v_key_2173_);
lean_ctor_set(v_reuseFailAlloc_2184_, 1, v_value_2174_);
lean_ctor_set(v_reuseFailAlloc_2184_, 2, v___x_2181_);
v___x_2183_ = v_reuseFailAlloc_2184_;
goto v_reusejp_2182_;
}
v_reusejp_2182_:
{
return v___x_2183_;
}
}
else
{
lean_object* v___x_2185_; lean_object* v___x_2187_; 
lean_dec(v_value_2174_);
lean_dec(v_key_2173_);
v___x_2185_ = lean_box_usize(v_a_2170_);
if (v_isShared_2178_ == 0)
{
lean_ctor_set(v___x_2177_, 1, v_b_2171_);
lean_ctor_set(v___x_2177_, 0, v___x_2185_);
v___x_2187_ = v___x_2177_;
goto v_reusejp_2186_;
}
else
{
lean_object* v_reuseFailAlloc_2188_; 
v_reuseFailAlloc_2188_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2188_, 0, v___x_2185_);
lean_ctor_set(v_reuseFailAlloc_2188_, 1, v_b_2171_);
lean_ctor_set(v_reuseFailAlloc_2188_, 2, v_tail_2175_);
v___x_2187_ = v_reuseFailAlloc_2188_;
goto v_reusejp_2186_;
}
v_reusejp_2186_:
{
return v___x_2187_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg___boxed(lean_object* v_a_2190_, lean_object* v_b_2191_, lean_object* v_x_2192_){
_start:
{
size_t v_a_boxed_2193_; lean_object* v_res_2194_; 
v_a_boxed_2193_ = lean_unbox_usize(v_a_2190_);
lean_dec(v_a_2190_);
v_res_2194_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg(v_a_boxed_2193_, v_b_2191_, v_x_2192_);
return v_res_2194_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58_spec__64___redArg(lean_object* v_x_2195_, lean_object* v_x_2196_){
_start:
{
if (lean_obj_tag(v_x_2196_) == 0)
{
return v_x_2195_;
}
else
{
lean_object* v_key_2197_; lean_object* v_value_2198_; lean_object* v_tail_2199_; lean_object* v___x_2201_; uint8_t v_isShared_2202_; uint8_t v_isSharedCheck_2223_; 
v_key_2197_ = lean_ctor_get(v_x_2196_, 0);
v_value_2198_ = lean_ctor_get(v_x_2196_, 1);
v_tail_2199_ = lean_ctor_get(v_x_2196_, 2);
v_isSharedCheck_2223_ = !lean_is_exclusive(v_x_2196_);
if (v_isSharedCheck_2223_ == 0)
{
v___x_2201_ = v_x_2196_;
v_isShared_2202_ = v_isSharedCheck_2223_;
goto v_resetjp_2200_;
}
else
{
lean_inc(v_tail_2199_);
lean_inc(v_value_2198_);
lean_inc(v_key_2197_);
lean_dec(v_x_2196_);
v___x_2201_ = lean_box(0);
v_isShared_2202_ = v_isSharedCheck_2223_;
goto v_resetjp_2200_;
}
v_resetjp_2200_:
{
lean_object* v___x_2203_; size_t v___x_2204_; uint64_t v___x_2205_; uint64_t v___x_2206_; uint64_t v___x_2207_; uint64_t v_fold_2208_; uint64_t v___x_2209_; uint64_t v___x_2210_; uint64_t v___x_2211_; size_t v___x_2212_; size_t v___x_2213_; size_t v___x_2214_; size_t v___x_2215_; size_t v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2219_; 
v___x_2203_ = lean_array_get_size(v_x_2195_);
v___x_2204_ = lean_unbox_usize(v_key_2197_);
v___x_2205_ = lean_usize_to_uint64(v___x_2204_);
v___x_2206_ = 32ULL;
v___x_2207_ = lean_uint64_shift_right(v___x_2205_, v___x_2206_);
v_fold_2208_ = lean_uint64_xor(v___x_2205_, v___x_2207_);
v___x_2209_ = 16ULL;
v___x_2210_ = lean_uint64_shift_right(v_fold_2208_, v___x_2209_);
v___x_2211_ = lean_uint64_xor(v_fold_2208_, v___x_2210_);
v___x_2212_ = lean_uint64_to_usize(v___x_2211_);
v___x_2213_ = lean_usize_of_nat(v___x_2203_);
v___x_2214_ = ((size_t)1ULL);
v___x_2215_ = lean_usize_sub(v___x_2213_, v___x_2214_);
v___x_2216_ = lean_usize_land(v___x_2212_, v___x_2215_);
v___x_2217_ = lean_array_uget_borrowed(v_x_2195_, v___x_2216_);
lean_inc(v___x_2217_);
if (v_isShared_2202_ == 0)
{
lean_ctor_set(v___x_2201_, 2, v___x_2217_);
v___x_2219_ = v___x_2201_;
goto v_reusejp_2218_;
}
else
{
lean_object* v_reuseFailAlloc_2222_; 
v_reuseFailAlloc_2222_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2222_, 0, v_key_2197_);
lean_ctor_set(v_reuseFailAlloc_2222_, 1, v_value_2198_);
lean_ctor_set(v_reuseFailAlloc_2222_, 2, v___x_2217_);
v___x_2219_ = v_reuseFailAlloc_2222_;
goto v_reusejp_2218_;
}
v_reusejp_2218_:
{
lean_object* v___x_2220_; 
v___x_2220_ = lean_array_uset(v_x_2195_, v___x_2216_, v___x_2219_);
v_x_2195_ = v___x_2220_;
v_x_2196_ = v_tail_2199_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58___redArg(lean_object* v_i_2224_, lean_object* v_source_2225_, lean_object* v_target_2226_){
_start:
{
lean_object* v___x_2227_; uint8_t v___x_2228_; 
v___x_2227_ = lean_array_get_size(v_source_2225_);
v___x_2228_ = lean_nat_dec_lt(v_i_2224_, v___x_2227_);
if (v___x_2228_ == 0)
{
lean_dec_ref(v_source_2225_);
lean_dec(v_i_2224_);
return v_target_2226_;
}
else
{
lean_object* v_es_2229_; lean_object* v___x_2230_; lean_object* v_source_2231_; lean_object* v_target_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; 
v_es_2229_ = lean_array_fget(v_source_2225_, v_i_2224_);
v___x_2230_ = lean_box(0);
v_source_2231_ = lean_array_fset(v_source_2225_, v_i_2224_, v___x_2230_);
v_target_2232_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58_spec__64___redArg(v_target_2226_, v_es_2229_);
v___x_2233_ = lean_unsigned_to_nat(1u);
v___x_2234_ = lean_nat_add(v_i_2224_, v___x_2233_);
lean_dec(v_i_2224_);
v_i_2224_ = v___x_2234_;
v_source_2225_ = v_source_2231_;
v_target_2226_ = v_target_2232_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54___redArg(lean_object* v_data_2236_){
_start:
{
lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v_nbuckets_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; 
v___x_2237_ = lean_array_get_size(v_data_2236_);
v___x_2238_ = lean_unsigned_to_nat(2u);
v_nbuckets_2239_ = lean_nat_mul(v___x_2237_, v___x_2238_);
v___x_2240_ = lean_unsigned_to_nat(0u);
v___x_2241_ = lean_box(0);
v___x_2242_ = lean_mk_array(v_nbuckets_2239_, v___x_2241_);
v___x_2243_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58___redArg(v___x_2240_, v_data_2236_, v___x_2242_);
return v___x_2243_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg(lean_object* v_m_2244_, size_t v_a_2245_, lean_object* v_b_2246_){
_start:
{
lean_object* v_size_2247_; lean_object* v_buckets_2248_; lean_object* v___x_2250_; uint8_t v_isShared_2251_; uint8_t v_isSharedCheck_2292_; 
v_size_2247_ = lean_ctor_get(v_m_2244_, 0);
v_buckets_2248_ = lean_ctor_get(v_m_2244_, 1);
v_isSharedCheck_2292_ = !lean_is_exclusive(v_m_2244_);
if (v_isSharedCheck_2292_ == 0)
{
v___x_2250_ = v_m_2244_;
v_isShared_2251_ = v_isSharedCheck_2292_;
goto v_resetjp_2249_;
}
else
{
lean_inc(v_buckets_2248_);
lean_inc(v_size_2247_);
lean_dec(v_m_2244_);
v___x_2250_ = lean_box(0);
v_isShared_2251_ = v_isSharedCheck_2292_;
goto v_resetjp_2249_;
}
v_resetjp_2249_:
{
lean_object* v___x_2252_; uint64_t v___x_2253_; uint64_t v___x_2254_; uint64_t v___x_2255_; uint64_t v_fold_2256_; uint64_t v___x_2257_; uint64_t v___x_2258_; uint64_t v___x_2259_; size_t v___x_2260_; size_t v___x_2261_; size_t v___x_2262_; size_t v___x_2263_; size_t v___x_2264_; lean_object* v_bkt_2265_; uint8_t v___x_2266_; 
v___x_2252_ = lean_array_get_size(v_buckets_2248_);
v___x_2253_ = lean_usize_to_uint64(v_a_2245_);
v___x_2254_ = 32ULL;
v___x_2255_ = lean_uint64_shift_right(v___x_2253_, v___x_2254_);
v_fold_2256_ = lean_uint64_xor(v___x_2253_, v___x_2255_);
v___x_2257_ = 16ULL;
v___x_2258_ = lean_uint64_shift_right(v_fold_2256_, v___x_2257_);
v___x_2259_ = lean_uint64_xor(v_fold_2256_, v___x_2258_);
v___x_2260_ = lean_uint64_to_usize(v___x_2259_);
v___x_2261_ = lean_usize_of_nat(v___x_2252_);
v___x_2262_ = ((size_t)1ULL);
v___x_2263_ = lean_usize_sub(v___x_2261_, v___x_2262_);
v___x_2264_ = lean_usize_land(v___x_2260_, v___x_2263_);
v_bkt_2265_ = lean_array_uget_borrowed(v_buckets_2248_, v___x_2264_);
v___x_2266_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___redArg(v_a_2245_, v_bkt_2265_);
if (v___x_2266_ == 0)
{
lean_object* v___x_2267_; lean_object* v_size_x27_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v_buckets_x27_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; uint8_t v___x_2277_; 
v___x_2267_ = lean_unsigned_to_nat(1u);
v_size_x27_2268_ = lean_nat_add(v_size_2247_, v___x_2267_);
lean_dec(v_size_2247_);
v___x_2269_ = lean_box_usize(v_a_2245_);
lean_inc(v_bkt_2265_);
v___x_2270_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2270_, 0, v___x_2269_);
lean_ctor_set(v___x_2270_, 1, v_b_2246_);
lean_ctor_set(v___x_2270_, 2, v_bkt_2265_);
v_buckets_x27_2271_ = lean_array_uset(v_buckets_2248_, v___x_2264_, v___x_2270_);
v___x_2272_ = lean_unsigned_to_nat(4u);
v___x_2273_ = lean_nat_mul(v_size_x27_2268_, v___x_2272_);
v___x_2274_ = lean_unsigned_to_nat(3u);
v___x_2275_ = lean_nat_div(v___x_2273_, v___x_2274_);
lean_dec(v___x_2273_);
v___x_2276_ = lean_array_get_size(v_buckets_x27_2271_);
v___x_2277_ = lean_nat_dec_le(v___x_2275_, v___x_2276_);
lean_dec(v___x_2275_);
if (v___x_2277_ == 0)
{
lean_object* v_val_2278_; lean_object* v___x_2280_; 
v_val_2278_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54___redArg(v_buckets_x27_2271_);
if (v_isShared_2251_ == 0)
{
lean_ctor_set(v___x_2250_, 1, v_val_2278_);
lean_ctor_set(v___x_2250_, 0, v_size_x27_2268_);
v___x_2280_ = v___x_2250_;
goto v_reusejp_2279_;
}
else
{
lean_object* v_reuseFailAlloc_2281_; 
v_reuseFailAlloc_2281_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2281_, 0, v_size_x27_2268_);
lean_ctor_set(v_reuseFailAlloc_2281_, 1, v_val_2278_);
v___x_2280_ = v_reuseFailAlloc_2281_;
goto v_reusejp_2279_;
}
v_reusejp_2279_:
{
return v___x_2280_;
}
}
else
{
lean_object* v___x_2283_; 
if (v_isShared_2251_ == 0)
{
lean_ctor_set(v___x_2250_, 1, v_buckets_x27_2271_);
lean_ctor_set(v___x_2250_, 0, v_size_x27_2268_);
v___x_2283_ = v___x_2250_;
goto v_reusejp_2282_;
}
else
{
lean_object* v_reuseFailAlloc_2284_; 
v_reuseFailAlloc_2284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2284_, 0, v_size_x27_2268_);
lean_ctor_set(v_reuseFailAlloc_2284_, 1, v_buckets_x27_2271_);
v___x_2283_ = v_reuseFailAlloc_2284_;
goto v_reusejp_2282_;
}
v_reusejp_2282_:
{
return v___x_2283_;
}
}
}
else
{
lean_object* v___x_2285_; lean_object* v_buckets_x27_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2290_; 
lean_inc(v_bkt_2265_);
v___x_2285_ = lean_box(0);
v_buckets_x27_2286_ = lean_array_uset(v_buckets_2248_, v___x_2264_, v___x_2285_);
v___x_2287_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg(v_a_2245_, v_b_2246_, v_bkt_2265_);
v___x_2288_ = lean_array_uset(v_buckets_x27_2286_, v___x_2264_, v___x_2287_);
if (v_isShared_2251_ == 0)
{
lean_ctor_set(v___x_2250_, 1, v___x_2288_);
v___x_2290_ = v___x_2250_;
goto v_reusejp_2289_;
}
else
{
lean_object* v_reuseFailAlloc_2291_; 
v_reuseFailAlloc_2291_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2291_, 0, v_size_2247_);
lean_ctor_set(v_reuseFailAlloc_2291_, 1, v___x_2288_);
v___x_2290_ = v_reuseFailAlloc_2291_;
goto v_reusejp_2289_;
}
v_reusejp_2289_:
{
return v___x_2290_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg___boxed(lean_object* v_m_2293_, lean_object* v_a_2294_, lean_object* v_b_2295_){
_start:
{
size_t v_a_boxed_2296_; lean_object* v_res_2297_; 
v_a_boxed_2296_ = lean_unbox_usize(v_a_2294_);
lean_dec(v_a_2294_);
v_res_2297_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg(v_m_2293_, v_a_boxed_2296_, v_b_2295_);
return v_res_2297_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___redArg(size_t v_a_2298_, lean_object* v_x_2299_){
_start:
{
if (lean_obj_tag(v_x_2299_) == 0)
{
lean_object* v___x_2300_; 
v___x_2300_ = lean_box(0);
return v___x_2300_;
}
else
{
lean_object* v_key_2301_; lean_object* v_value_2302_; lean_object* v_tail_2303_; size_t v___x_2304_; uint8_t v___x_2305_; 
v_key_2301_ = lean_ctor_get(v_x_2299_, 0);
v_value_2302_ = lean_ctor_get(v_x_2299_, 1);
v_tail_2303_ = lean_ctor_get(v_x_2299_, 2);
v___x_2304_ = lean_unbox_usize(v_key_2301_);
v___x_2305_ = lean_usize_dec_eq(v___x_2304_, v_a_2298_);
if (v___x_2305_ == 0)
{
v_x_2299_ = v_tail_2303_;
goto _start;
}
else
{
lean_object* v___x_2307_; 
lean_inc(v_value_2302_);
v___x_2307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2307_, 0, v_value_2302_);
return v___x_2307_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___redArg___boxed(lean_object* v_a_2308_, lean_object* v_x_2309_){
_start:
{
size_t v_a_boxed_2310_; lean_object* v_res_2311_; 
v_a_boxed_2310_ = lean_unbox_usize(v_a_2308_);
lean_dec(v_a_2308_);
v_res_2311_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___redArg(v_a_boxed_2310_, v_x_2309_);
lean_dec(v_x_2309_);
return v_res_2311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___redArg(lean_object* v_m_2312_, size_t v_a_2313_){
_start:
{
lean_object* v_buckets_2314_; lean_object* v___x_2315_; uint64_t v___x_2316_; uint64_t v___x_2317_; uint64_t v___x_2318_; uint64_t v_fold_2319_; uint64_t v___x_2320_; uint64_t v___x_2321_; uint64_t v___x_2322_; size_t v___x_2323_; size_t v___x_2324_; size_t v___x_2325_; size_t v___x_2326_; size_t v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; 
v_buckets_2314_ = lean_ctor_get(v_m_2312_, 1);
v___x_2315_ = lean_array_get_size(v_buckets_2314_);
v___x_2316_ = lean_usize_to_uint64(v_a_2313_);
v___x_2317_ = 32ULL;
v___x_2318_ = lean_uint64_shift_right(v___x_2316_, v___x_2317_);
v_fold_2319_ = lean_uint64_xor(v___x_2316_, v___x_2318_);
v___x_2320_ = 16ULL;
v___x_2321_ = lean_uint64_shift_right(v_fold_2319_, v___x_2320_);
v___x_2322_ = lean_uint64_xor(v_fold_2319_, v___x_2321_);
v___x_2323_ = lean_uint64_to_usize(v___x_2322_);
v___x_2324_ = lean_usize_of_nat(v___x_2315_);
v___x_2325_ = ((size_t)1ULL);
v___x_2326_ = lean_usize_sub(v___x_2324_, v___x_2325_);
v___x_2327_ = lean_usize_land(v___x_2323_, v___x_2326_);
v___x_2328_ = lean_array_uget_borrowed(v_buckets_2314_, v___x_2327_);
v___x_2329_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___redArg(v_a_2313_, v___x_2328_);
return v___x_2329_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___redArg___boxed(lean_object* v_m_2330_, lean_object* v_a_2331_){
_start:
{
size_t v_a_boxed_2332_; lean_object* v_res_2333_; 
v_a_boxed_2332_ = lean_unbox_usize(v_a_2331_);
lean_dec(v_a_2331_);
v_res_2333_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___redArg(v_m_2330_, v_a_boxed_2332_);
lean_dec_ref(v_m_2330_);
return v_res_2333_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__44(lean_object* v_x_2334_, lean_object* v_x_2335_){
_start:
{
if (lean_obj_tag(v_x_2335_) == 0)
{
return v_x_2334_;
}
else
{
lean_object* v_key_2336_; lean_object* v_value_2337_; lean_object* v_tail_2338_; lean_object* v_fst_2339_; lean_object* v_snd_2340_; size_t v___x_2341_; lean_object* v___x_2342_; lean_object* v_fst_2343_; lean_object* v_snd_2344_; lean_object* v___x_2346_; uint8_t v_isShared_2347_; uint8_t v_isSharedCheck_2367_; 
v_key_2336_ = lean_ctor_get(v_x_2335_, 0);
lean_inc(v_key_2336_);
v_value_2337_ = lean_ctor_get(v_x_2335_, 1);
lean_inc(v_value_2337_);
v_tail_2338_ = lean_ctor_get(v_x_2335_, 2);
lean_inc(v_tail_2338_);
lean_dec_ref_known(v_x_2335_, 3);
v_fst_2339_ = lean_ctor_get(v_x_2334_, 0);
lean_inc(v_fst_2339_);
v_snd_2340_ = lean_ctor_get(v_x_2334_, 1);
lean_inc(v_snd_2340_);
lean_dec_ref(v_x_2334_);
v___x_2341_ = lean_unbox_usize(v_value_2337_);
lean_dec(v_value_2337_);
v___x_2342_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41(v___x_2341_, v_snd_2340_);
v_fst_2343_ = lean_ctor_get(v___x_2342_, 0);
v_snd_2344_ = lean_ctor_get(v___x_2342_, 1);
v_isSharedCheck_2367_ = !lean_is_exclusive(v___x_2342_);
if (v_isSharedCheck_2367_ == 0)
{
v___x_2346_ = v___x_2342_;
v_isShared_2347_ = v_isSharedCheck_2367_;
goto v_resetjp_2345_;
}
else
{
lean_inc(v_snd_2344_);
lean_inc(v_fst_2343_);
lean_dec(v___x_2342_);
v___x_2346_ = lean_box(0);
v_isShared_2347_ = v_isSharedCheck_2367_;
goto v_resetjp_2345_;
}
v_resetjp_2345_:
{
size_t v___x_2348_; lean_object* v___x_2349_; 
v___x_2348_ = lean_unbox_usize(v_fst_2343_);
v___x_2349_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___redArg(v_fst_2339_, v___x_2348_);
if (lean_obj_tag(v___x_2349_) == 0)
{
lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; size_t v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2356_; 
v___x_2350_ = lean_unsigned_to_nat(1u);
v___x_2351_ = lean_mk_empty_array_with_capacity(v___x_2350_);
v___x_2352_ = lean_array_push(v___x_2351_, v_key_2336_);
v___x_2353_ = lean_unbox_usize(v_fst_2343_);
lean_dec(v_fst_2343_);
v___x_2354_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg(v_fst_2339_, v___x_2353_, v___x_2352_);
if (v_isShared_2347_ == 0)
{
lean_ctor_set(v___x_2346_, 0, v___x_2354_);
v___x_2356_ = v___x_2346_;
goto v_reusejp_2355_;
}
else
{
lean_object* v_reuseFailAlloc_2358_; 
v_reuseFailAlloc_2358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2358_, 0, v___x_2354_);
lean_ctor_set(v_reuseFailAlloc_2358_, 1, v_snd_2344_);
v___x_2356_ = v_reuseFailAlloc_2358_;
goto v_reusejp_2355_;
}
v_reusejp_2355_:
{
v_x_2334_ = v___x_2356_;
v_x_2335_ = v_tail_2338_;
goto _start;
}
}
else
{
lean_object* v_val_2359_; lean_object* v___x_2360_; size_t v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2364_; 
v_val_2359_ = lean_ctor_get(v___x_2349_, 0);
lean_inc(v_val_2359_);
lean_dec_ref_known(v___x_2349_, 1);
v___x_2360_ = lean_array_push(v_val_2359_, v_key_2336_);
v___x_2361_ = lean_unbox_usize(v_fst_2343_);
lean_dec(v_fst_2343_);
v___x_2362_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg(v_fst_2339_, v___x_2361_, v___x_2360_);
if (v_isShared_2347_ == 0)
{
lean_ctor_set(v___x_2346_, 0, v___x_2362_);
v___x_2364_ = v___x_2346_;
goto v_reusejp_2363_;
}
else
{
lean_object* v_reuseFailAlloc_2366_; 
v_reuseFailAlloc_2366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2366_, 0, v___x_2362_);
lean_ctor_set(v_reuseFailAlloc_2366_, 1, v_snd_2344_);
v___x_2364_ = v_reuseFailAlloc_2366_;
goto v_reusejp_2363_;
}
v_reusejp_2363_:
{
v_x_2334_ = v___x_2364_;
v_x_2335_ = v_tail_2338_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__45(lean_object* v_as_2368_, size_t v_i_2369_, size_t v_stop_2370_, lean_object* v_b_2371_){
_start:
{
uint8_t v___x_2372_; 
v___x_2372_ = lean_usize_dec_eq(v_i_2369_, v_stop_2370_);
if (v___x_2372_ == 0)
{
lean_object* v___x_2373_; lean_object* v___x_2374_; size_t v___x_2375_; size_t v___x_2376_; 
v___x_2373_ = lean_array_uget_borrowed(v_as_2368_, v_i_2369_);
lean_inc(v___x_2373_);
v___x_2374_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__44(v_b_2371_, v___x_2373_);
v___x_2375_ = ((size_t)1ULL);
v___x_2376_ = lean_usize_add(v_i_2369_, v___x_2375_);
v_i_2369_ = v___x_2376_;
v_b_2371_ = v___x_2374_;
goto _start;
}
else
{
return v_b_2371_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__45___boxed(lean_object* v_as_2378_, lean_object* v_i_2379_, lean_object* v_stop_2380_, lean_object* v_b_2381_){
_start:
{
size_t v_i_boxed_2382_; size_t v_stop_boxed_2383_; lean_object* v_res_2384_; 
v_i_boxed_2382_ = lean_unbox_usize(v_i_2379_);
lean_dec(v_i_2379_);
v_stop_boxed_2383_ = lean_unbox_usize(v_stop_2380_);
lean_dec(v_stop_2380_);
v_res_2384_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__45(v_as_2378_, v_i_boxed_2382_, v_stop_boxed_2383_, v_b_2381_);
lean_dec_ref(v_as_2378_);
return v_res_2384_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__39(lean_object* v_x_2385_, lean_object* v_x_2386_){
_start:
{
if (lean_obj_tag(v_x_2386_) == 0)
{
return v_x_2385_;
}
else
{
lean_object* v_value_2387_; lean_object* v_tail_2388_; lean_object* v___x_2389_; 
v_value_2387_ = lean_ctor_get(v_x_2386_, 1);
lean_inc(v_value_2387_);
v_tail_2388_ = lean_ctor_get(v_x_2386_, 2);
lean_inc(v_tail_2388_);
lean_dec_ref_known(v_x_2386_, 3);
v___x_2389_ = lean_array_push(v_x_2385_, v_value_2387_);
v_x_2385_ = v___x_2389_;
v_x_2386_ = v_tail_2388_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__40(lean_object* v_as_2391_, size_t v_i_2392_, size_t v_stop_2393_, lean_object* v_b_2394_){
_start:
{
uint8_t v___x_2395_; 
v___x_2395_ = lean_usize_dec_eq(v_i_2392_, v_stop_2393_);
if (v___x_2395_ == 0)
{
lean_object* v___x_2396_; lean_object* v___x_2397_; size_t v___x_2398_; size_t v___x_2399_; 
v___x_2396_ = lean_array_uget_borrowed(v_as_2391_, v_i_2392_);
lean_inc(v___x_2396_);
v___x_2397_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__39(v_b_2394_, v___x_2396_);
v___x_2398_ = ((size_t)1ULL);
v___x_2399_ = lean_usize_add(v_i_2392_, v___x_2398_);
v_i_2392_ = v___x_2399_;
v_b_2394_ = v___x_2397_;
goto _start;
}
else
{
return v_b_2394_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__40___boxed(lean_object* v_as_2401_, lean_object* v_i_2402_, lean_object* v_stop_2403_, lean_object* v_b_2404_){
_start:
{
size_t v_i_boxed_2405_; size_t v_stop_boxed_2406_; lean_object* v_res_2407_; 
v_i_boxed_2405_ = lean_unbox_usize(v_i_2402_);
lean_dec(v_i_2402_);
v_stop_boxed_2406_ = lean_unbox_usize(v_stop_2403_);
lean_dec(v_stop_2403_);
v_res_2407_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__40(v_as_2401_, v_i_boxed_2405_, v_stop_boxed_2406_, v_b_2404_);
lean_dec_ref(v_as_2401_);
return v_res_2407_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0(void){
_start:
{
lean_object* v___x_2408_; lean_object* v___x_2409_; lean_object* v___x_2410_; 
v___x_2408_ = lean_box(0);
v___x_2409_ = lean_unsigned_to_nat(16u);
v___x_2410_ = lean_mk_array(v___x_2409_, v___x_2408_);
return v___x_2410_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__1(void){
_start:
{
lean_object* v___x_2411_; lean_object* v___x_2412_; lean_object* v___x_2413_; 
v___x_2411_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0, &lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0_once, _init_lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0);
v___x_2412_ = lean_unsigned_to_nat(0u);
v___x_2413_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2413_, 0, v___x_2412_);
lean_ctor_set(v___x_2413_, 1, v___x_2411_);
return v___x_2413_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32(lean_object* v_u_2414_){
_start:
{
lean_object* v_size_2416_; lean_object* v_buckets_2417_; lean_object* v_snd_2418_; lean_object* v___y_2435_; lean_object* v_toRep_2440_; lean_object* v_buckets_2441_; lean_object* v___x_2443_; uint8_t v_isShared_2444_; uint8_t v_isSharedCheck_2460_; 
v_toRep_2440_ = lean_ctor_get(v_u_2414_, 2);
lean_inc_ref(v_toRep_2440_);
v_buckets_2441_ = lean_ctor_get(v_toRep_2440_, 1);
v_isSharedCheck_2460_ = !lean_is_exclusive(v_toRep_2440_);
if (v_isSharedCheck_2460_ == 0)
{
lean_object* v_unused_2461_; 
v_unused_2461_ = lean_ctor_get(v_toRep_2440_, 0);
lean_dec(v_unused_2461_);
v___x_2443_ = v_toRep_2440_;
v_isShared_2444_ = v_isSharedCheck_2460_;
goto v_resetjp_2442_;
}
else
{
lean_inc(v_buckets_2441_);
lean_dec(v_toRep_2440_);
v___x_2443_ = lean_box(0);
v_isShared_2444_ = v_isSharedCheck_2460_;
goto v_resetjp_2442_;
}
v___jp_2415_:
{
lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; uint8_t v___x_2422_; 
v___x_2419_ = lean_mk_empty_array_with_capacity(v_size_2416_);
lean_dec(v_size_2416_);
v___x_2420_ = lean_unsigned_to_nat(0u);
v___x_2421_ = lean_array_get_size(v_buckets_2417_);
v___x_2422_ = lean_nat_dec_lt(v___x_2420_, v___x_2421_);
if (v___x_2422_ == 0)
{
lean_object* v___x_2423_; 
lean_dec_ref(v_buckets_2417_);
v___x_2423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2423_, 0, v___x_2419_);
lean_ctor_set(v___x_2423_, 1, v_snd_2418_);
return v___x_2423_;
}
else
{
uint8_t v___x_2424_; 
v___x_2424_ = lean_nat_dec_le(v___x_2421_, v___x_2421_);
if (v___x_2424_ == 0)
{
if (v___x_2422_ == 0)
{
lean_object* v___x_2425_; 
lean_dec_ref(v_buckets_2417_);
v___x_2425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2425_, 0, v___x_2419_);
lean_ctor_set(v___x_2425_, 1, v_snd_2418_);
return v___x_2425_;
}
else
{
size_t v___x_2426_; size_t v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; 
v___x_2426_ = ((size_t)0ULL);
v___x_2427_ = lean_usize_of_nat(v___x_2421_);
v___x_2428_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__40(v_buckets_2417_, v___x_2426_, v___x_2427_, v___x_2419_);
lean_dec_ref(v_buckets_2417_);
v___x_2429_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2429_, 0, v___x_2428_);
lean_ctor_set(v___x_2429_, 1, v_snd_2418_);
return v___x_2429_;
}
}
else
{
size_t v___x_2430_; size_t v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; 
v___x_2430_ = ((size_t)0ULL);
v___x_2431_ = lean_usize_of_nat(v___x_2421_);
v___x_2432_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__40(v_buckets_2417_, v___x_2430_, v___x_2431_, v___x_2419_);
lean_dec_ref(v_buckets_2417_);
v___x_2433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2433_, 0, v___x_2432_);
lean_ctor_set(v___x_2433_, 1, v_snd_2418_);
return v___x_2433_;
}
}
}
v___jp_2434_:
{
lean_object* v_fst_2436_; lean_object* v_snd_2437_; lean_object* v_size_2438_; lean_object* v_buckets_2439_; 
v_fst_2436_ = lean_ctor_get(v___y_2435_, 0);
lean_inc(v_fst_2436_);
v_snd_2437_ = lean_ctor_get(v___y_2435_, 1);
lean_inc(v_snd_2437_);
lean_dec_ref(v___y_2435_);
v_size_2438_ = lean_ctor_get(v_fst_2436_, 0);
lean_inc(v_size_2438_);
v_buckets_2439_ = lean_ctor_get(v_fst_2436_, 1);
lean_inc_ref(v_buckets_2439_);
lean_dec(v_fst_2436_);
v_size_2416_ = v_size_2438_;
v_buckets_2417_ = v_buckets_2439_;
v_snd_2418_ = v_snd_2437_;
goto v___jp_2415_;
}
v_resetjp_2442_:
{
lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; uint8_t v___x_2448_; 
v___x_2445_ = lean_unsigned_to_nat(0u);
v___x_2446_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0, &lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0_once, _init_lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__0);
v___x_2447_ = lean_array_get_size(v_buckets_2441_);
v___x_2448_ = lean_nat_dec_lt(v___x_2445_, v___x_2447_);
if (v___x_2448_ == 0)
{
lean_del_object(v___x_2443_);
lean_dec_ref(v_buckets_2441_);
v_size_2416_ = v___x_2445_;
v_buckets_2417_ = v___x_2446_;
v_snd_2418_ = v_u_2414_;
goto v___jp_2415_;
}
else
{
lean_object* v___x_2449_; lean_object* v___x_2451_; 
v___x_2449_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__1, &lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__1_once, _init_lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32___closed__1);
lean_inc_ref(v_u_2414_);
if (v_isShared_2444_ == 0)
{
lean_ctor_set(v___x_2443_, 1, v_u_2414_);
lean_ctor_set(v___x_2443_, 0, v___x_2449_);
v___x_2451_ = v___x_2443_;
goto v_reusejp_2450_;
}
else
{
lean_object* v_reuseFailAlloc_2459_; 
v_reuseFailAlloc_2459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2459_, 0, v___x_2449_);
lean_ctor_set(v_reuseFailAlloc_2459_, 1, v_u_2414_);
v___x_2451_ = v_reuseFailAlloc_2459_;
goto v_reusejp_2450_;
}
v_reusejp_2450_:
{
uint8_t v___x_2452_; 
v___x_2452_ = lean_nat_dec_le(v___x_2447_, v___x_2447_);
if (v___x_2452_ == 0)
{
if (v___x_2448_ == 0)
{
lean_dec_ref(v___x_2451_);
lean_dec_ref(v_buckets_2441_);
v_size_2416_ = v___x_2445_;
v_buckets_2417_ = v___x_2446_;
v_snd_2418_ = v_u_2414_;
goto v___jp_2415_;
}
else
{
size_t v___x_2453_; size_t v___x_2454_; lean_object* v___x_2455_; 
lean_dec_ref(v_u_2414_);
v___x_2453_ = ((size_t)0ULL);
v___x_2454_ = lean_usize_of_nat(v___x_2447_);
v___x_2455_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__45(v_buckets_2441_, v___x_2453_, v___x_2454_, v___x_2451_);
lean_dec_ref(v_buckets_2441_);
v___y_2435_ = v___x_2455_;
goto v___jp_2434_;
}
}
else
{
size_t v___x_2456_; size_t v___x_2457_; lean_object* v___x_2458_; 
lean_dec_ref(v_u_2414_);
v___x_2456_ = ((size_t)0ULL);
v___x_2457_ = lean_usize_of_nat(v___x_2447_);
v___x_2458_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__45(v_buckets_2441_, v___x_2456_, v___x_2457_, v___x_2451_);
lean_dec_ref(v_buckets_2441_);
v___y_2435_ = v___x_2458_;
goto v___jp_2434_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27_spec__32___redArg(lean_object* v_a_2462_, lean_object* v_b_2463_, lean_object* v_x_2464_){
_start:
{
if (lean_obj_tag(v_x_2464_) == 0)
{
lean_dec(v_b_2463_);
lean_dec(v_a_2462_);
return v_x_2464_;
}
else
{
lean_object* v_key_2465_; lean_object* v_value_2466_; lean_object* v_tail_2467_; lean_object* v___x_2469_; uint8_t v_isShared_2470_; uint8_t v_isSharedCheck_2479_; 
v_key_2465_ = lean_ctor_get(v_x_2464_, 0);
v_value_2466_ = lean_ctor_get(v_x_2464_, 1);
v_tail_2467_ = lean_ctor_get(v_x_2464_, 2);
v_isSharedCheck_2479_ = !lean_is_exclusive(v_x_2464_);
if (v_isSharedCheck_2479_ == 0)
{
v___x_2469_ = v_x_2464_;
v_isShared_2470_ = v_isSharedCheck_2479_;
goto v_resetjp_2468_;
}
else
{
lean_inc(v_tail_2467_);
lean_inc(v_value_2466_);
lean_inc(v_key_2465_);
lean_dec(v_x_2464_);
v___x_2469_ = lean_box(0);
v_isShared_2470_ = v_isSharedCheck_2479_;
goto v_resetjp_2468_;
}
v_resetjp_2468_:
{
uint8_t v___x_2471_; 
v___x_2471_ = l_Lean_instBEqMVarId_beq(v_key_2465_, v_a_2462_);
if (v___x_2471_ == 0)
{
lean_object* v___x_2472_; lean_object* v___x_2474_; 
v___x_2472_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27_spec__32___redArg(v_a_2462_, v_b_2463_, v_tail_2467_);
if (v_isShared_2470_ == 0)
{
lean_ctor_set(v___x_2469_, 2, v___x_2472_);
v___x_2474_ = v___x_2469_;
goto v_reusejp_2473_;
}
else
{
lean_object* v_reuseFailAlloc_2475_; 
v_reuseFailAlloc_2475_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2475_, 0, v_key_2465_);
lean_ctor_set(v_reuseFailAlloc_2475_, 1, v_value_2466_);
lean_ctor_set(v_reuseFailAlloc_2475_, 2, v___x_2472_);
v___x_2474_ = v_reuseFailAlloc_2475_;
goto v_reusejp_2473_;
}
v_reusejp_2473_:
{
return v___x_2474_;
}
}
else
{
lean_object* v___x_2477_; 
lean_dec(v_value_2466_);
lean_dec(v_key_2465_);
if (v_isShared_2470_ == 0)
{
lean_ctor_set(v___x_2469_, 1, v_b_2463_);
lean_ctor_set(v___x_2469_, 0, v_a_2462_);
v___x_2477_ = v___x_2469_;
goto v_reusejp_2476_;
}
else
{
lean_object* v_reuseFailAlloc_2478_; 
v_reuseFailAlloc_2478_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2478_, 0, v_a_2462_);
lean_ctor_set(v_reuseFailAlloc_2478_, 1, v_b_2463_);
lean_ctor_set(v_reuseFailAlloc_2478_, 2, v_tail_2467_);
v___x_2477_ = v_reuseFailAlloc_2478_;
goto v_reusejp_2476_;
}
v_reusejp_2476_:
{
return v___x_2477_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27___redArg(lean_object* v_m_2480_, lean_object* v_a_2481_, lean_object* v_b_2482_){
_start:
{
lean_object* v_size_2483_; lean_object* v_buckets_2484_; lean_object* v___x_2486_; uint8_t v_isShared_2487_; uint8_t v_isSharedCheck_2527_; 
v_size_2483_ = lean_ctor_get(v_m_2480_, 0);
v_buckets_2484_ = lean_ctor_get(v_m_2480_, 1);
v_isSharedCheck_2527_ = !lean_is_exclusive(v_m_2480_);
if (v_isSharedCheck_2527_ == 0)
{
v___x_2486_ = v_m_2480_;
v_isShared_2487_ = v_isSharedCheck_2527_;
goto v_resetjp_2485_;
}
else
{
lean_inc(v_buckets_2484_);
lean_inc(v_size_2483_);
lean_dec(v_m_2480_);
v___x_2486_ = lean_box(0);
v_isShared_2487_ = v_isSharedCheck_2527_;
goto v_resetjp_2485_;
}
v_resetjp_2485_:
{
lean_object* v___x_2488_; uint64_t v___x_2489_; uint64_t v___x_2490_; uint64_t v___x_2491_; uint64_t v_fold_2492_; uint64_t v___x_2493_; uint64_t v___x_2494_; uint64_t v___x_2495_; size_t v___x_2496_; size_t v___x_2497_; size_t v___x_2498_; size_t v___x_2499_; size_t v___x_2500_; lean_object* v_bkt_2501_; uint8_t v___x_2502_; 
v___x_2488_ = lean_array_get_size(v_buckets_2484_);
v___x_2489_ = l_Lean_instHashableMVarId_hash(v_a_2481_);
v___x_2490_ = 32ULL;
v___x_2491_ = lean_uint64_shift_right(v___x_2489_, v___x_2490_);
v_fold_2492_ = lean_uint64_xor(v___x_2489_, v___x_2491_);
v___x_2493_ = 16ULL;
v___x_2494_ = lean_uint64_shift_right(v_fold_2492_, v___x_2493_);
v___x_2495_ = lean_uint64_xor(v_fold_2492_, v___x_2494_);
v___x_2496_ = lean_uint64_to_usize(v___x_2495_);
v___x_2497_ = lean_usize_of_nat(v___x_2488_);
v___x_2498_ = ((size_t)1ULL);
v___x_2499_ = lean_usize_sub(v___x_2497_, v___x_2498_);
v___x_2500_ = lean_usize_land(v___x_2496_, v___x_2499_);
v_bkt_2501_ = lean_array_uget_borrowed(v_buckets_2484_, v___x_2500_);
v___x_2502_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg(v_a_2481_, v_bkt_2501_);
if (v___x_2502_ == 0)
{
lean_object* v___x_2503_; lean_object* v_size_x27_2504_; lean_object* v___x_2505_; lean_object* v_buckets_x27_2506_; lean_object* v___x_2507_; lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; uint8_t v___x_2512_; 
v___x_2503_ = lean_unsigned_to_nat(1u);
v_size_x27_2504_ = lean_nat_add(v_size_2483_, v___x_2503_);
lean_dec(v_size_2483_);
lean_inc(v_bkt_2501_);
v___x_2505_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2505_, 0, v_a_2481_);
lean_ctor_set(v___x_2505_, 1, v_b_2482_);
lean_ctor_set(v___x_2505_, 2, v_bkt_2501_);
v_buckets_x27_2506_ = lean_array_uset(v_buckets_2484_, v___x_2500_, v___x_2505_);
v___x_2507_ = lean_unsigned_to_nat(4u);
v___x_2508_ = lean_nat_mul(v_size_x27_2504_, v___x_2507_);
v___x_2509_ = lean_unsigned_to_nat(3u);
v___x_2510_ = lean_nat_div(v___x_2508_, v___x_2509_);
lean_dec(v___x_2508_);
v___x_2511_ = lean_array_get_size(v_buckets_x27_2506_);
v___x_2512_ = lean_nat_dec_le(v___x_2510_, v___x_2511_);
lean_dec(v___x_2510_);
if (v___x_2512_ == 0)
{
lean_object* v_val_2513_; lean_object* v___x_2515_; 
v_val_2513_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18___redArg(v_buckets_x27_2506_);
if (v_isShared_2487_ == 0)
{
lean_ctor_set(v___x_2486_, 1, v_val_2513_);
lean_ctor_set(v___x_2486_, 0, v_size_x27_2504_);
v___x_2515_ = v___x_2486_;
goto v_reusejp_2514_;
}
else
{
lean_object* v_reuseFailAlloc_2516_; 
v_reuseFailAlloc_2516_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2516_, 0, v_size_x27_2504_);
lean_ctor_set(v_reuseFailAlloc_2516_, 1, v_val_2513_);
v___x_2515_ = v_reuseFailAlloc_2516_;
goto v_reusejp_2514_;
}
v_reusejp_2514_:
{
return v___x_2515_;
}
}
else
{
lean_object* v___x_2518_; 
if (v_isShared_2487_ == 0)
{
lean_ctor_set(v___x_2486_, 1, v_buckets_x27_2506_);
lean_ctor_set(v___x_2486_, 0, v_size_x27_2504_);
v___x_2518_ = v___x_2486_;
goto v_reusejp_2517_;
}
else
{
lean_object* v_reuseFailAlloc_2519_; 
v_reuseFailAlloc_2519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2519_, 0, v_size_x27_2504_);
lean_ctor_set(v_reuseFailAlloc_2519_, 1, v_buckets_x27_2506_);
v___x_2518_ = v_reuseFailAlloc_2519_;
goto v_reusejp_2517_;
}
v_reusejp_2517_:
{
return v___x_2518_;
}
}
}
else
{
lean_object* v___x_2520_; lean_object* v_buckets_x27_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2525_; 
lean_inc(v_bkt_2501_);
v___x_2520_ = lean_box(0);
v_buckets_x27_2521_ = lean_array_uset(v_buckets_2484_, v___x_2500_, v___x_2520_);
v___x_2522_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27_spec__32___redArg(v_a_2481_, v_b_2482_, v_bkt_2501_);
v___x_2523_ = lean_array_uset(v_buckets_x27_2521_, v___x_2500_, v___x_2522_);
if (v_isShared_2487_ == 0)
{
lean_ctor_set(v___x_2486_, 1, v___x_2523_);
v___x_2525_ = v___x_2486_;
goto v_reusejp_2524_;
}
else
{
lean_object* v_reuseFailAlloc_2526_; 
v_reuseFailAlloc_2526_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2526_, 0, v_size_2483_);
lean_ctor_set(v_reuseFailAlloc_2526_, 1, v___x_2523_);
v___x_2525_ = v_reuseFailAlloc_2526_;
goto v_reusejp_2524_;
}
v_reusejp_2524_:
{
return v___x_2525_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34_spec__42___redArg(lean_object* v_a_2528_, lean_object* v_x_2529_){
_start:
{
if (lean_obj_tag(v_x_2529_) == 0)
{
lean_object* v___x_2530_; 
lean_dec(v_a_2528_);
v___x_2530_ = lean_box(0);
return v___x_2530_;
}
else
{
lean_object* v_key_2531_; lean_object* v_value_2532_; lean_object* v_tail_2533_; lean_object* v___x_2534_; lean_object* v_elimGoal_2535_; lean_object* v___x_2536_; lean_object* v_id_2537_; lean_object* v___x_2538_; lean_object* v_id_2539_; uint8_t v___x_2540_; 
v_key_2531_ = lean_ctor_get(v_x_2529_, 0);
lean_inc(v_key_2531_);
v_value_2532_ = lean_ctor_get(v_x_2529_, 1);
lean_inc(v_value_2532_);
v_tail_2533_ = lean_ctor_get(v_x_2529_, 2);
lean_inc(v_tail_2533_);
lean_dec_ref_known(v_x_2529_, 3);
v___x_2534_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2535_ = lean_ctor_get(v___x_2534_, 1);
lean_inc_ref_n(v_elimGoal_2535_, 2);
v___x_2536_ = lean_apply_1(v_elimGoal_2535_, v_key_2531_);
v_id_2537_ = lean_ctor_get(v___x_2536_, 0);
lean_inc(v_id_2537_);
lean_dec_ref(v___x_2536_);
lean_inc(v_a_2528_);
v___x_2538_ = lean_apply_1(v_elimGoal_2535_, v_a_2528_);
v_id_2539_ = lean_ctor_get(v___x_2538_, 0);
lean_inc(v_id_2539_);
lean_dec_ref(v___x_2538_);
v___x_2540_ = lean_nat_dec_eq(v_id_2537_, v_id_2539_);
lean_dec(v_id_2539_);
lean_dec(v_id_2537_);
if (v___x_2540_ == 0)
{
lean_dec(v_value_2532_);
v_x_2529_ = v_tail_2533_;
goto _start;
}
else
{
lean_object* v___x_2542_; 
lean_dec(v_tail_2533_);
lean_dec(v_a_2528_);
v___x_2542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2542_, 0, v_value_2532_);
return v___x_2542_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___redArg(lean_object* v_m_2543_, lean_object* v_a_2544_){
_start:
{
lean_object* v_buckets_2545_; lean_object* v___x_2546_; lean_object* v_elimGoal_2547_; lean_object* v___x_2548_; lean_object* v_id_2549_; lean_object* v___x_2550_; uint64_t v___x_2551_; uint64_t v___x_2552_; uint64_t v___x_2553_; uint64_t v_fold_2554_; uint64_t v___x_2555_; uint64_t v___x_2556_; uint64_t v___x_2557_; size_t v___x_2558_; size_t v___x_2559_; size_t v___x_2560_; size_t v___x_2561_; size_t v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; 
v_buckets_2545_ = lean_ctor_get(v_m_2543_, 1);
v___x_2546_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2547_ = lean_ctor_get(v___x_2546_, 1);
lean_inc_ref(v_elimGoal_2547_);
lean_inc(v_a_2544_);
v___x_2548_ = lean_apply_1(v_elimGoal_2547_, v_a_2544_);
v_id_2549_ = lean_ctor_get(v___x_2548_, 0);
lean_inc(v_id_2549_);
lean_dec_ref(v___x_2548_);
v___x_2550_ = lean_array_get_size(v_buckets_2545_);
v___x_2551_ = lean_uint64_of_nat(v_id_2549_);
lean_dec(v_id_2549_);
v___x_2552_ = 32ULL;
v___x_2553_ = lean_uint64_shift_right(v___x_2551_, v___x_2552_);
v_fold_2554_ = lean_uint64_xor(v___x_2551_, v___x_2553_);
v___x_2555_ = 16ULL;
v___x_2556_ = lean_uint64_shift_right(v_fold_2554_, v___x_2555_);
v___x_2557_ = lean_uint64_xor(v_fold_2554_, v___x_2556_);
v___x_2558_ = lean_uint64_to_usize(v___x_2557_);
v___x_2559_ = lean_usize_of_nat(v___x_2550_);
v___x_2560_ = ((size_t)1ULL);
v___x_2561_ = lean_usize_sub(v___x_2559_, v___x_2560_);
v___x_2562_ = lean_usize_land(v___x_2558_, v___x_2561_);
v___x_2563_ = lean_array_uget_borrowed(v_buckets_2545_, v___x_2562_);
lean_inc(v___x_2563_);
v___x_2564_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34_spec__42___redArg(v_a_2544_, v___x_2563_);
return v___x_2564_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___redArg___boxed(lean_object* v_m_2565_, lean_object* v_a_2566_){
_start:
{
lean_object* v_res_2567_; 
v_res_2567_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___redArg(v_m_2565_, v_a_2566_);
lean_dec_ref(v_m_2565_);
return v_res_2567_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28(lean_object* v_x_2568_, lean_object* v_u_2569_){
_start:
{
lean_object* v_toRep_2570_; lean_object* v___x_2571_; 
v_toRep_2570_ = lean_ctor_get(v_u_2569_, 2);
v___x_2571_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___redArg(v_toRep_2570_, v_x_2568_);
if (lean_obj_tag(v___x_2571_) == 0)
{
lean_object* v___x_2572_; 
v___x_2572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2572_, 0, v___x_2571_);
lean_ctor_set(v___x_2572_, 1, v_u_2569_);
return v___x_2572_;
}
else
{
lean_object* v_val_2573_; lean_object* v___x_2575_; uint8_t v_isShared_2576_; uint8_t v_isSharedCheck_2591_; 
v_val_2573_ = lean_ctor_get(v___x_2571_, 0);
v_isSharedCheck_2591_ = !lean_is_exclusive(v___x_2571_);
if (v_isSharedCheck_2591_ == 0)
{
v___x_2575_ = v___x_2571_;
v_isShared_2576_ = v_isSharedCheck_2591_;
goto v_resetjp_2574_;
}
else
{
lean_inc(v_val_2573_);
lean_dec(v___x_2571_);
v___x_2575_ = lean_box(0);
v_isShared_2576_ = v_isSharedCheck_2591_;
goto v_resetjp_2574_;
}
v_resetjp_2574_:
{
size_t v___x_2577_; lean_object* v___x_2578_; lean_object* v_fst_2579_; lean_object* v_snd_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2590_; 
v___x_2577_ = lean_unbox_usize(v_val_2573_);
lean_dec(v_val_2573_);
v___x_2578_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__41(v___x_2577_, v_u_2569_);
v_fst_2579_ = lean_ctor_get(v___x_2578_, 0);
v_snd_2580_ = lean_ctor_get(v___x_2578_, 1);
v_isSharedCheck_2590_ = !lean_is_exclusive(v___x_2578_);
if (v_isSharedCheck_2590_ == 0)
{
v___x_2582_ = v___x_2578_;
v_isShared_2583_ = v_isSharedCheck_2590_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_snd_2580_);
lean_inc(v_fst_2579_);
lean_dec(v___x_2578_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2590_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
lean_object* v___x_2585_; 
if (v_isShared_2576_ == 0)
{
lean_ctor_set(v___x_2575_, 0, v_fst_2579_);
v___x_2585_ = v___x_2575_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2589_; 
v_reuseFailAlloc_2589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2589_, 0, v_fst_2579_);
v___x_2585_ = v_reuseFailAlloc_2589_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
lean_object* v___x_2587_; 
if (v_isShared_2583_ == 0)
{
lean_ctor_set(v___x_2582_, 0, v___x_2585_);
v___x_2587_ = v___x_2582_;
goto v_reusejp_2586_;
}
else
{
lean_object* v_reuseFailAlloc_2588_; 
v_reuseFailAlloc_2588_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2588_, 0, v___x_2585_);
lean_ctor_set(v_reuseFailAlloc_2588_, 1, v_snd_2580_);
v___x_2587_ = v_reuseFailAlloc_2588_;
goto v_reusejp_2586_;
}
v_reusejp_2586_:
{
return v___x_2587_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25(lean_object* v_x_2592_, lean_object* v_y_2593_, lean_object* v_u_2594_){
_start:
{
lean_object* v___x_2595_; lean_object* v_fst_2596_; 
lean_inc_ref(v_u_2594_);
v___x_2595_ = lp_aesop_Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28(v_x_2592_, v_u_2594_);
v_fst_2596_ = lean_ctor_get(v___x_2595_, 0);
lean_inc(v_fst_2596_);
if (lean_obj_tag(v_fst_2596_) == 1)
{
lean_object* v_snd_2597_; lean_object* v_val_2598_; lean_object* v___x_2599_; lean_object* v_fst_2600_; 
lean_dec_ref(v_u_2594_);
v_snd_2597_ = lean_ctor_get(v___x_2595_, 1);
lean_inc_n(v_snd_2597_, 2);
lean_dec_ref(v___x_2595_);
v_val_2598_ = lean_ctor_get(v_fst_2596_, 0);
lean_inc(v_val_2598_);
lean_dec_ref_known(v_fst_2596_, 1);
v___x_2599_ = lp_aesop_Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28(v_y_2593_, v_snd_2597_);
v_fst_2600_ = lean_ctor_get(v___x_2599_, 0);
lean_inc(v_fst_2600_);
if (lean_obj_tag(v_fst_2600_) == 1)
{
lean_object* v_snd_2601_; lean_object* v_val_2602_; size_t v___x_2603_; size_t v___x_2604_; uint8_t v___x_2605_; 
lean_dec(v_snd_2597_);
v_snd_2601_ = lean_ctor_get(v___x_2599_, 1);
lean_inc(v_snd_2601_);
lean_dec_ref(v___x_2599_);
v_val_2602_ = lean_ctor_get(v_fst_2600_, 0);
lean_inc(v_val_2602_);
lean_dec_ref_known(v_fst_2600_, 1);
v___x_2603_ = lean_unbox_usize(v_val_2598_);
v___x_2604_ = lean_unbox_usize(v_val_2602_);
v___x_2605_ = lean_usize_dec_eq(v___x_2603_, v___x_2604_);
if (v___x_2605_ == 0)
{
lean_object* v_parents_2606_; lean_object* v_sizes_2607_; lean_object* v_toRep_2608_; lean_object* v___x_2610_; uint8_t v_isShared_2611_; uint8_t v_isSharedCheck_2641_; 
v_parents_2606_ = lean_ctor_get(v_snd_2601_, 0);
v_sizes_2607_ = lean_ctor_get(v_snd_2601_, 1);
v_toRep_2608_ = lean_ctor_get(v_snd_2601_, 2);
v_isSharedCheck_2641_ = !lean_is_exclusive(v_snd_2601_);
if (v_isSharedCheck_2641_ == 0)
{
v___x_2610_ = v_snd_2601_;
v_isShared_2611_ = v_isSharedCheck_2641_;
goto v_resetjp_2609_;
}
else
{
lean_inc(v_toRep_2608_);
lean_inc(v_sizes_2607_);
lean_inc(v_parents_2606_);
lean_dec(v_snd_2601_);
v___x_2610_ = lean_box(0);
v_isShared_2611_ = v_isSharedCheck_2641_;
goto v_resetjp_2609_;
}
v_resetjp_2609_:
{
size_t v___x_2612_; lean_object* v_xSize_2613_; size_t v___x_2614_; lean_object* v_ySize_2615_; size_t v___x_2616_; size_t v___x_2617_; uint8_t v___x_2618_; 
v___x_2612_ = lean_unbox_usize(v_val_2598_);
v_xSize_2613_ = lean_array_uget_borrowed(v_sizes_2607_, v___x_2612_);
v___x_2614_ = lean_unbox_usize(v_val_2602_);
v_ySize_2615_ = lean_array_uget_borrowed(v_sizes_2607_, v___x_2614_);
v___x_2616_ = lean_unbox_usize(v_xSize_2613_);
v___x_2617_ = lean_unbox_usize(v_ySize_2615_);
v___x_2618_ = lean_usize_dec_lt(v___x_2616_, v___x_2617_);
if (v___x_2618_ == 0)
{
size_t v___x_2619_; lean_object* v___x_2620_; size_t v___x_2621_; size_t v___x_2622_; size_t v___x_2623_; size_t v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2628_; 
v___x_2619_ = lean_unbox_usize(v_val_2602_);
lean_dec(v_val_2602_);
lean_inc(v_val_2598_);
v___x_2620_ = lean_array_uset(v_parents_2606_, v___x_2619_, v_val_2598_);
v___x_2621_ = lean_unbox_usize(v_xSize_2613_);
v___x_2622_ = lean_unbox_usize(v_ySize_2615_);
v___x_2623_ = lean_usize_add(v___x_2621_, v___x_2622_);
v___x_2624_ = lean_unbox_usize(v_val_2598_);
lean_dec(v_val_2598_);
v___x_2625_ = lean_box_usize(v___x_2623_);
v___x_2626_ = lean_array_uset(v_sizes_2607_, v___x_2624_, v___x_2625_);
if (v_isShared_2611_ == 0)
{
lean_ctor_set(v___x_2610_, 1, v___x_2626_);
lean_ctor_set(v___x_2610_, 0, v___x_2620_);
v___x_2628_ = v___x_2610_;
goto v_reusejp_2627_;
}
else
{
lean_object* v_reuseFailAlloc_2629_; 
v_reuseFailAlloc_2629_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2629_, 0, v___x_2620_);
lean_ctor_set(v_reuseFailAlloc_2629_, 1, v___x_2626_);
lean_ctor_set(v_reuseFailAlloc_2629_, 2, v_toRep_2608_);
v___x_2628_ = v_reuseFailAlloc_2629_;
goto v_reusejp_2627_;
}
v_reusejp_2627_:
{
return v___x_2628_;
}
}
else
{
size_t v___x_2630_; lean_object* v___x_2631_; size_t v___x_2632_; size_t v___x_2633_; size_t v___x_2634_; size_t v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2639_; 
v___x_2630_ = lean_unbox_usize(v_val_2598_);
lean_dec(v_val_2598_);
lean_inc(v_val_2602_);
v___x_2631_ = lean_array_uset(v_parents_2606_, v___x_2630_, v_val_2602_);
v___x_2632_ = lean_unbox_usize(v_xSize_2613_);
v___x_2633_ = lean_unbox_usize(v_ySize_2615_);
v___x_2634_ = lean_usize_add(v___x_2632_, v___x_2633_);
v___x_2635_ = lean_unbox_usize(v_val_2602_);
lean_dec(v_val_2602_);
v___x_2636_ = lean_box_usize(v___x_2634_);
v___x_2637_ = lean_array_uset(v_sizes_2607_, v___x_2635_, v___x_2636_);
if (v_isShared_2611_ == 0)
{
lean_ctor_set(v___x_2610_, 1, v___x_2637_);
lean_ctor_set(v___x_2610_, 0, v___x_2631_);
v___x_2639_ = v___x_2610_;
goto v_reusejp_2638_;
}
else
{
lean_object* v_reuseFailAlloc_2640_; 
v_reuseFailAlloc_2640_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2640_, 0, v___x_2631_);
lean_ctor_set(v_reuseFailAlloc_2640_, 1, v___x_2637_);
lean_ctor_set(v_reuseFailAlloc_2640_, 2, v_toRep_2608_);
v___x_2639_ = v_reuseFailAlloc_2640_;
goto v_reusejp_2638_;
}
v_reusejp_2638_:
{
return v___x_2639_;
}
}
}
}
else
{
lean_dec(v_val_2602_);
lean_dec(v_val_2598_);
return v_snd_2601_;
}
}
else
{
lean_dec(v_fst_2600_);
lean_dec_ref(v___x_2599_);
lean_dec(v_val_2598_);
return v_snd_2597_;
}
}
else
{
lean_dec(v_fst_2596_);
lean_dec_ref(v___x_2595_);
lean_dec(v_y_2593_);
return v_u_2594_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__28(lean_object* v_a_2642_, lean_object* v_as_2643_, size_t v_sz_2644_, size_t v_i_2645_, lean_object* v_b_2646_){
_start:
{
uint8_t v___x_2647_; 
v___x_2647_ = lean_usize_dec_lt(v_i_2645_, v_sz_2644_);
if (v___x_2647_ == 0)
{
lean_dec(v_a_2642_);
return v_b_2646_;
}
else
{
lean_object* v_a_2648_; lean_object* v___x_2649_; size_t v___x_2650_; size_t v___x_2651_; 
v_a_2648_ = lean_array_uget_borrowed(v_as_2643_, v_i_2645_);
lean_inc(v_a_2648_);
lean_inc(v_a_2642_);
v___x_2649_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25(v_a_2642_, v_a_2648_, v_b_2646_);
v___x_2650_ = ((size_t)1ULL);
v___x_2651_ = lean_usize_add(v_i_2645_, v___x_2650_);
v_i_2645_ = v___x_2651_;
v_b_2646_ = v___x_2649_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__28___boxed(lean_object* v_a_2653_, lean_object* v_as_2654_, lean_object* v_sz_2655_, lean_object* v_i_2656_, lean_object* v_b_2657_){
_start:
{
size_t v_sz_boxed_2658_; size_t v_i_boxed_2659_; lean_object* v_res_2660_; 
v_sz_boxed_2658_ = lean_unbox_usize(v_sz_2655_);
lean_dec(v_sz_2655_);
v_i_boxed_2659_ = lean_unbox_usize(v_i_2656_);
lean_dec(v_i_2656_);
v_res_2660_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__28(v_a_2653_, v_as_2654_, v_sz_boxed_2658_, v_i_boxed_2659_, v_b_2657_);
lean_dec_ref(v_as_2654_);
return v_res_2660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___redArg(lean_object* v_a_2661_, lean_object* v_x_2662_){
_start:
{
if (lean_obj_tag(v_x_2662_) == 0)
{
lean_object* v___x_2663_; 
v___x_2663_ = lean_box(0);
return v___x_2663_;
}
else
{
lean_object* v_key_2664_; lean_object* v_value_2665_; lean_object* v_tail_2666_; uint8_t v___x_2667_; 
v_key_2664_ = lean_ctor_get(v_x_2662_, 0);
v_value_2665_ = lean_ctor_get(v_x_2662_, 1);
v_tail_2666_ = lean_ctor_get(v_x_2662_, 2);
v___x_2667_ = l_Lean_instBEqMVarId_beq(v_key_2664_, v_a_2661_);
if (v___x_2667_ == 0)
{
v_x_2662_ = v_tail_2666_;
goto _start;
}
else
{
lean_object* v___x_2669_; 
lean_inc(v_value_2665_);
v___x_2669_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2669_, 0, v_value_2665_);
return v___x_2669_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___redArg___boxed(lean_object* v_a_2670_, lean_object* v_x_2671_){
_start:
{
lean_object* v_res_2672_; 
v_res_2672_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___redArg(v_a_2670_, v_x_2671_);
lean_dec(v_x_2671_);
lean_dec(v_a_2670_);
return v_res_2672_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___redArg(lean_object* v_m_2673_, lean_object* v_a_2674_){
_start:
{
lean_object* v_buckets_2675_; lean_object* v___x_2676_; uint64_t v___x_2677_; uint64_t v___x_2678_; uint64_t v___x_2679_; uint64_t v_fold_2680_; uint64_t v___x_2681_; uint64_t v___x_2682_; uint64_t v___x_2683_; size_t v___x_2684_; size_t v___x_2685_; size_t v___x_2686_; size_t v___x_2687_; size_t v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; 
v_buckets_2675_ = lean_ctor_get(v_m_2673_, 1);
v___x_2676_ = lean_array_get_size(v_buckets_2675_);
v___x_2677_ = l_Lean_instHashableMVarId_hash(v_a_2674_);
v___x_2678_ = 32ULL;
v___x_2679_ = lean_uint64_shift_right(v___x_2677_, v___x_2678_);
v_fold_2680_ = lean_uint64_xor(v___x_2677_, v___x_2679_);
v___x_2681_ = 16ULL;
v___x_2682_ = lean_uint64_shift_right(v_fold_2680_, v___x_2681_);
v___x_2683_ = lean_uint64_xor(v_fold_2680_, v___x_2682_);
v___x_2684_ = lean_uint64_to_usize(v___x_2683_);
v___x_2685_ = lean_usize_of_nat(v___x_2676_);
v___x_2686_ = ((size_t)1ULL);
v___x_2687_ = lean_usize_sub(v___x_2685_, v___x_2686_);
v___x_2688_ = lean_usize_land(v___x_2684_, v___x_2687_);
v___x_2689_ = lean_array_uget_borrowed(v_buckets_2675_, v___x_2688_);
v___x_2690_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___redArg(v_a_2674_, v___x_2689_);
return v___x_2690_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___redArg___boxed(lean_object* v_m_2691_, lean_object* v_a_2692_){
_start:
{
lean_object* v_res_2693_; 
v_res_2693_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___redArg(v_m_2691_, v_a_2692_);
lean_dec(v_a_2692_);
lean_dec_ref(v_m_2691_);
return v_res_2693_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__29(lean_object* v_a_2694_, lean_object* v_as_2695_, size_t v_sz_2696_, size_t v_i_2697_, lean_object* v_b_2698_){
_start:
{
lean_object* v_a_2700_; uint8_t v___x_2704_; 
v___x_2704_ = lean_usize_dec_lt(v_i_2697_, v_sz_2696_);
if (v___x_2704_ == 0)
{
lean_dec(v_a_2694_);
return v_b_2698_;
}
else
{
lean_object* v_fst_2705_; lean_object* v_snd_2706_; lean_object* v___x_2708_; uint8_t v_isShared_2709_; uint8_t v_isSharedCheck_2728_; 
v_fst_2705_ = lean_ctor_get(v_b_2698_, 0);
v_snd_2706_ = lean_ctor_get(v_b_2698_, 1);
v_isSharedCheck_2728_ = !lean_is_exclusive(v_b_2698_);
if (v_isSharedCheck_2728_ == 0)
{
v___x_2708_ = v_b_2698_;
v_isShared_2709_ = v_isSharedCheck_2728_;
goto v_resetjp_2707_;
}
else
{
lean_inc(v_snd_2706_);
lean_inc(v_fst_2705_);
lean_dec(v_b_2698_);
v___x_2708_ = lean_box(0);
v_isShared_2709_ = v_isSharedCheck_2728_;
goto v_resetjp_2707_;
}
v_resetjp_2707_:
{
lean_object* v_a_2710_; lean_object* v___x_2711_; 
v_a_2710_ = lean_array_uget_borrowed(v_as_2695_, v_i_2697_);
v___x_2711_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___redArg(v_snd_2706_, v_a_2710_);
if (lean_obj_tag(v___x_2711_) == 0)
{
lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2717_; 
v___x_2712_ = lean_unsigned_to_nat(1u);
v___x_2713_ = lean_mk_empty_array_with_capacity(v___x_2712_);
lean_inc(v_a_2694_);
v___x_2714_ = lean_array_push(v___x_2713_, v_a_2694_);
lean_inc(v_a_2710_);
v___x_2715_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27___redArg(v_snd_2706_, v_a_2710_, v___x_2714_);
if (v_isShared_2709_ == 0)
{
lean_ctor_set(v___x_2708_, 1, v___x_2715_);
v___x_2717_ = v___x_2708_;
goto v_reusejp_2716_;
}
else
{
lean_object* v_reuseFailAlloc_2718_; 
v_reuseFailAlloc_2718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2718_, 0, v_fst_2705_);
lean_ctor_set(v_reuseFailAlloc_2718_, 1, v___x_2715_);
v___x_2717_ = v_reuseFailAlloc_2718_;
goto v_reusejp_2716_;
}
v_reusejp_2716_:
{
v_a_2700_ = v___x_2717_;
goto v___jp_2699_;
}
}
else
{
lean_object* v_val_2719_; size_t v_sz_2720_; size_t v___x_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___x_2726_; 
v_val_2719_ = lean_ctor_get(v___x_2711_, 0);
lean_inc(v_val_2719_);
lean_dec_ref_known(v___x_2711_, 1);
v_sz_2720_ = lean_array_size(v_val_2719_);
v___x_2721_ = ((size_t)0ULL);
lean_inc_n(v_a_2694_, 2);
v___x_2722_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__28(v_a_2694_, v_val_2719_, v_sz_2720_, v___x_2721_, v_fst_2705_);
v___x_2723_ = lean_array_push(v_val_2719_, v_a_2694_);
lean_inc(v_a_2710_);
v___x_2724_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27___redArg(v_snd_2706_, v_a_2710_, v___x_2723_);
if (v_isShared_2709_ == 0)
{
lean_ctor_set(v___x_2708_, 1, v___x_2724_);
lean_ctor_set(v___x_2708_, 0, v___x_2722_);
v___x_2726_ = v___x_2708_;
goto v_reusejp_2725_;
}
else
{
lean_object* v_reuseFailAlloc_2727_; 
v_reuseFailAlloc_2727_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2727_, 0, v___x_2722_);
lean_ctor_set(v_reuseFailAlloc_2727_, 1, v___x_2724_);
v___x_2726_ = v_reuseFailAlloc_2727_;
goto v_reusejp_2725_;
}
v_reusejp_2725_:
{
v_a_2700_ = v___x_2726_;
goto v___jp_2699_;
}
}
}
}
v___jp_2699_:
{
size_t v___x_2701_; size_t v___x_2702_; 
v___x_2701_ = ((size_t)1ULL);
v___x_2702_ = lean_usize_add(v_i_2697_, v___x_2701_);
v_i_2697_ = v___x_2702_;
v_b_2698_ = v_a_2700_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__29___boxed(lean_object* v_a_2729_, lean_object* v_as_2730_, lean_object* v_sz_2731_, lean_object* v_i_2732_, lean_object* v_b_2733_){
_start:
{
size_t v_sz_boxed_2734_; size_t v_i_boxed_2735_; lean_object* v_res_2736_; 
v_sz_boxed_2734_ = lean_unbox_usize(v_sz_2731_);
lean_dec(v_sz_2731_);
v_i_boxed_2735_ = lean_unbox_usize(v_i_2732_);
lean_dec(v_i_2732_);
v_res_2736_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__29(v_a_2729_, v_as_2730_, v_sz_boxed_2734_, v_i_boxed_2735_, v_b_2733_);
lean_dec_ref(v_as_2730_);
return v_res_2736_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__31(lean_object* v_f_2737_, lean_object* v_as_2738_, size_t v_sz_2739_, size_t v_i_2740_, lean_object* v_b_2741_){
_start:
{
uint8_t v___x_2742_; 
v___x_2742_ = lean_usize_dec_lt(v_i_2740_, v_sz_2739_);
if (v___x_2742_ == 0)
{
lean_dec_ref(v_f_2737_);
return v_b_2741_;
}
else
{
lean_object* v_fst_2743_; lean_object* v_snd_2744_; lean_object* v___x_2746_; uint8_t v_isShared_2747_; uint8_t v_isSharedCheck_2768_; 
v_fst_2743_ = lean_ctor_get(v_b_2741_, 0);
v_snd_2744_ = lean_ctor_get(v_b_2741_, 1);
v_isSharedCheck_2768_ = !lean_is_exclusive(v_b_2741_);
if (v_isSharedCheck_2768_ == 0)
{
v___x_2746_ = v_b_2741_;
v_isShared_2747_ = v_isSharedCheck_2768_;
goto v_resetjp_2745_;
}
else
{
lean_inc(v_snd_2744_);
lean_inc(v_fst_2743_);
lean_dec(v_b_2741_);
v___x_2746_ = lean_box(0);
v_isShared_2747_ = v_isSharedCheck_2768_;
goto v_resetjp_2745_;
}
v_resetjp_2745_:
{
lean_object* v_a_2748_; lean_object* v___x_2749_; lean_object* v___x_2751_; 
v_a_2748_ = lean_array_uget_borrowed(v_as_2738_, v_i_2740_);
lean_inc_ref(v_f_2737_);
lean_inc(v_a_2748_);
v___x_2749_ = lean_apply_1(v_f_2737_, v_a_2748_);
if (v_isShared_2747_ == 0)
{
v___x_2751_ = v___x_2746_;
goto v_reusejp_2750_;
}
else
{
lean_object* v_reuseFailAlloc_2767_; 
v_reuseFailAlloc_2767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2767_, 0, v_fst_2743_);
lean_ctor_set(v_reuseFailAlloc_2767_, 1, v_snd_2744_);
v___x_2751_ = v_reuseFailAlloc_2767_;
goto v_reusejp_2750_;
}
v_reusejp_2750_:
{
size_t v_sz_2752_; size_t v___x_2753_; lean_object* v___x_2754_; lean_object* v_fst_2755_; lean_object* v_snd_2756_; lean_object* v___x_2758_; uint8_t v_isShared_2759_; uint8_t v_isSharedCheck_2766_; 
v_sz_2752_ = lean_array_size(v___x_2749_);
v___x_2753_ = ((size_t)0ULL);
lean_inc(v_a_2748_);
v___x_2754_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__29(v_a_2748_, v___x_2749_, v_sz_2752_, v___x_2753_, v___x_2751_);
lean_dec_ref(v___x_2749_);
v_fst_2755_ = lean_ctor_get(v___x_2754_, 0);
v_snd_2756_ = lean_ctor_get(v___x_2754_, 1);
v_isSharedCheck_2766_ = !lean_is_exclusive(v___x_2754_);
if (v_isSharedCheck_2766_ == 0)
{
v___x_2758_ = v___x_2754_;
v_isShared_2759_ = v_isSharedCheck_2766_;
goto v_resetjp_2757_;
}
else
{
lean_inc(v_snd_2756_);
lean_inc(v_fst_2755_);
lean_dec(v___x_2754_);
v___x_2758_ = lean_box(0);
v_isShared_2759_ = v_isSharedCheck_2766_;
goto v_resetjp_2757_;
}
v_resetjp_2757_:
{
lean_object* v___x_2761_; 
if (v_isShared_2759_ == 0)
{
v___x_2761_ = v___x_2758_;
goto v_reusejp_2760_;
}
else
{
lean_object* v_reuseFailAlloc_2765_; 
v_reuseFailAlloc_2765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2765_, 0, v_fst_2755_);
lean_ctor_set(v_reuseFailAlloc_2765_, 1, v_snd_2756_);
v___x_2761_ = v_reuseFailAlloc_2765_;
goto v_reusejp_2760_;
}
v_reusejp_2760_:
{
size_t v___x_2762_; size_t v___x_2763_; 
v___x_2762_ = ((size_t)1ULL);
v___x_2763_ = lean_usize_add(v_i_2740_, v___x_2762_);
v_i_2740_ = v___x_2763_;
v_b_2741_ = v___x_2761_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__31___boxed(lean_object* v_f_2769_, lean_object* v_as_2770_, lean_object* v_sz_2771_, lean_object* v_i_2772_, lean_object* v_b_2773_){
_start:
{
size_t v_sz_boxed_2774_; size_t v_i_boxed_2775_; lean_object* v_res_2776_; 
v_sz_boxed_2774_ = lean_unbox_usize(v_sz_2771_);
lean_dec(v_sz_2771_);
v_i_boxed_2775_ = lean_unbox_usize(v_i_2772_);
lean_dec(v_i_2772_);
v_res_2776_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__31(v_f_2769_, v_as_2770_, v_sz_boxed_2774_, v_i_boxed_2775_, v_b_2773_);
lean_dec_ref(v_as_2770_);
return v_res_2776_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg(lean_object* v_a_2777_, lean_object* v_x_2778_){
_start:
{
if (lean_obj_tag(v_x_2778_) == 0)
{
uint8_t v___x_2779_; 
lean_dec(v_a_2777_);
v___x_2779_ = 0;
return v___x_2779_;
}
else
{
lean_object* v_key_2780_; lean_object* v_tail_2781_; lean_object* v___x_2782_; lean_object* v_elimGoal_2783_; lean_object* v___x_2784_; lean_object* v_id_2785_; lean_object* v___x_2786_; lean_object* v_id_2787_; uint8_t v___x_2788_; 
v_key_2780_ = lean_ctor_get(v_x_2778_, 0);
lean_inc(v_key_2780_);
v_tail_2781_ = lean_ctor_get(v_x_2778_, 2);
lean_inc(v_tail_2781_);
lean_dec_ref_known(v_x_2778_, 3);
v___x_2782_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2783_ = lean_ctor_get(v___x_2782_, 1);
lean_inc_ref_n(v_elimGoal_2783_, 2);
v___x_2784_ = lean_apply_1(v_elimGoal_2783_, v_key_2780_);
v_id_2785_ = lean_ctor_get(v___x_2784_, 0);
lean_inc(v_id_2785_);
lean_dec_ref(v___x_2784_);
lean_inc(v_a_2777_);
v___x_2786_ = lean_apply_1(v_elimGoal_2783_, v_a_2777_);
v_id_2787_ = lean_ctor_get(v___x_2786_, 0);
lean_inc(v_id_2787_);
lean_dec_ref(v___x_2786_);
v___x_2788_ = lean_nat_dec_eq(v_id_2785_, v_id_2787_);
lean_dec(v_id_2787_);
lean_dec(v_id_2785_);
if (v___x_2788_ == 0)
{
v_x_2778_ = v_tail_2781_;
goto _start;
}
else
{
lean_dec(v_tail_2781_);
lean_dec(v_a_2777_);
return v___x_2788_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg___boxed(lean_object* v_a_2790_, lean_object* v_x_2791_){
_start:
{
uint8_t v_res_2792_; lean_object* v_r_2793_; 
v_res_2792_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg(v_a_2790_, v_x_2791_);
v_r_2793_ = lean_box(v_res_2792_);
return v_r_2793_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___redArg(lean_object* v_m_2794_, lean_object* v_a_2795_){
_start:
{
lean_object* v_buckets_2796_; lean_object* v___x_2797_; lean_object* v_elimGoal_2798_; lean_object* v___x_2799_; lean_object* v_id_2800_; lean_object* v___x_2801_; uint64_t v___x_2802_; uint64_t v___x_2803_; uint64_t v___x_2804_; uint64_t v_fold_2805_; uint64_t v___x_2806_; uint64_t v___x_2807_; uint64_t v___x_2808_; size_t v___x_2809_; size_t v___x_2810_; size_t v___x_2811_; size_t v___x_2812_; size_t v___x_2813_; lean_object* v___x_2814_; uint8_t v___x_2815_; 
v_buckets_2796_ = lean_ctor_get(v_m_2794_, 1);
v___x_2797_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2798_ = lean_ctor_get(v___x_2797_, 1);
lean_inc_ref(v_elimGoal_2798_);
lean_inc(v_a_2795_);
v___x_2799_ = lean_apply_1(v_elimGoal_2798_, v_a_2795_);
v_id_2800_ = lean_ctor_get(v___x_2799_, 0);
lean_inc(v_id_2800_);
lean_dec_ref(v___x_2799_);
v___x_2801_ = lean_array_get_size(v_buckets_2796_);
v___x_2802_ = lean_uint64_of_nat(v_id_2800_);
lean_dec(v_id_2800_);
v___x_2803_ = 32ULL;
v___x_2804_ = lean_uint64_shift_right(v___x_2802_, v___x_2803_);
v_fold_2805_ = lean_uint64_xor(v___x_2802_, v___x_2804_);
v___x_2806_ = 16ULL;
v___x_2807_ = lean_uint64_shift_right(v_fold_2805_, v___x_2806_);
v___x_2808_ = lean_uint64_xor(v_fold_2805_, v___x_2807_);
v___x_2809_ = lean_uint64_to_usize(v___x_2808_);
v___x_2810_ = lean_usize_of_nat(v___x_2801_);
v___x_2811_ = ((size_t)1ULL);
v___x_2812_ = lean_usize_sub(v___x_2810_, v___x_2811_);
v___x_2813_ = lean_usize_land(v___x_2809_, v___x_2812_);
v___x_2814_ = lean_array_uget_borrowed(v_buckets_2796_, v___x_2813_);
lean_inc(v___x_2814_);
v___x_2815_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg(v_a_2795_, v___x_2814_);
return v___x_2815_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___redArg___boxed(lean_object* v_m_2816_, lean_object* v_a_2817_){
_start:
{
uint8_t v_res_2818_; lean_object* v_r_2819_; 
v_res_2818_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___redArg(v_m_2816_, v_a_2817_);
lean_dec_ref(v_m_2816_);
v_r_2819_ = lean_box(v_res_2818_);
return v_r_2819_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63_spec__65___redArg(lean_object* v_x_2820_, lean_object* v_x_2821_){
_start:
{
if (lean_obj_tag(v_x_2821_) == 0)
{
return v_x_2820_;
}
else
{
lean_object* v_key_2822_; lean_object* v_value_2823_; lean_object* v_tail_2824_; lean_object* v___x_2826_; uint8_t v_isShared_2827_; uint8_t v_isSharedCheck_2851_; 
v_key_2822_ = lean_ctor_get(v_x_2821_, 0);
v_value_2823_ = lean_ctor_get(v_x_2821_, 1);
v_tail_2824_ = lean_ctor_get(v_x_2821_, 2);
v_isSharedCheck_2851_ = !lean_is_exclusive(v_x_2821_);
if (v_isSharedCheck_2851_ == 0)
{
v___x_2826_ = v_x_2821_;
v_isShared_2827_ = v_isSharedCheck_2851_;
goto v_resetjp_2825_;
}
else
{
lean_inc(v_tail_2824_);
lean_inc(v_value_2823_);
lean_inc(v_key_2822_);
lean_dec(v_x_2821_);
v___x_2826_ = lean_box(0);
v_isShared_2827_ = v_isSharedCheck_2851_;
goto v_resetjp_2825_;
}
v_resetjp_2825_:
{
lean_object* v___x_2828_; lean_object* v_elimGoal_2829_; lean_object* v___x_2830_; lean_object* v_id_2831_; lean_object* v___x_2832_; uint64_t v___x_2833_; uint64_t v___x_2834_; uint64_t v___x_2835_; uint64_t v_fold_2836_; uint64_t v___x_2837_; uint64_t v___x_2838_; uint64_t v___x_2839_; size_t v___x_2840_; size_t v___x_2841_; size_t v___x_2842_; size_t v___x_2843_; size_t v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2847_; 
v___x_2828_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2829_ = lean_ctor_get(v___x_2828_, 1);
lean_inc_ref(v_elimGoal_2829_);
lean_inc(v_key_2822_);
v___x_2830_ = lean_apply_1(v_elimGoal_2829_, v_key_2822_);
v_id_2831_ = lean_ctor_get(v___x_2830_, 0);
lean_inc(v_id_2831_);
lean_dec_ref(v___x_2830_);
v___x_2832_ = lean_array_get_size(v_x_2820_);
v___x_2833_ = lean_uint64_of_nat(v_id_2831_);
lean_dec(v_id_2831_);
v___x_2834_ = 32ULL;
v___x_2835_ = lean_uint64_shift_right(v___x_2833_, v___x_2834_);
v_fold_2836_ = lean_uint64_xor(v___x_2833_, v___x_2835_);
v___x_2837_ = 16ULL;
v___x_2838_ = lean_uint64_shift_right(v_fold_2836_, v___x_2837_);
v___x_2839_ = lean_uint64_xor(v_fold_2836_, v___x_2838_);
v___x_2840_ = lean_uint64_to_usize(v___x_2839_);
v___x_2841_ = lean_usize_of_nat(v___x_2832_);
v___x_2842_ = ((size_t)1ULL);
v___x_2843_ = lean_usize_sub(v___x_2841_, v___x_2842_);
v___x_2844_ = lean_usize_land(v___x_2840_, v___x_2843_);
v___x_2845_ = lean_array_uget_borrowed(v_x_2820_, v___x_2844_);
lean_inc(v___x_2845_);
if (v_isShared_2827_ == 0)
{
lean_ctor_set(v___x_2826_, 2, v___x_2845_);
v___x_2847_ = v___x_2826_;
goto v_reusejp_2846_;
}
else
{
lean_object* v_reuseFailAlloc_2850_; 
v_reuseFailAlloc_2850_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2850_, 0, v_key_2822_);
lean_ctor_set(v_reuseFailAlloc_2850_, 1, v_value_2823_);
lean_ctor_set(v_reuseFailAlloc_2850_, 2, v___x_2845_);
v___x_2847_ = v_reuseFailAlloc_2850_;
goto v_reusejp_2846_;
}
v_reusejp_2846_:
{
lean_object* v___x_2848_; 
v___x_2848_ = lean_array_uset(v_x_2820_, v___x_2844_, v___x_2847_);
v_x_2820_ = v___x_2848_;
v_x_2821_ = v_tail_2824_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63___redArg(lean_object* v_i_2852_, lean_object* v_source_2853_, lean_object* v_target_2854_){
_start:
{
lean_object* v___x_2855_; uint8_t v___x_2856_; 
v___x_2855_ = lean_array_get_size(v_source_2853_);
v___x_2856_ = lean_nat_dec_lt(v_i_2852_, v___x_2855_);
if (v___x_2856_ == 0)
{
lean_dec_ref(v_source_2853_);
lean_dec(v_i_2852_);
return v_target_2854_;
}
else
{
lean_object* v_es_2857_; lean_object* v___x_2858_; lean_object* v_source_2859_; lean_object* v_target_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; 
v_es_2857_ = lean_array_fget(v_source_2853_, v_i_2852_);
v___x_2858_ = lean_box(0);
v_source_2859_ = lean_array_fset(v_source_2853_, v_i_2852_, v___x_2858_);
v_target_2860_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63_spec__65___redArg(v_target_2854_, v_es_2857_);
v___x_2861_ = lean_unsigned_to_nat(1u);
v___x_2862_ = lean_nat_add(v_i_2852_, v___x_2861_);
lean_dec(v_i_2852_);
v_i_2852_ = v___x_2862_;
v_source_2853_ = v_source_2859_;
v_target_2854_ = v_target_2860_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57___redArg(lean_object* v_data_2864_){
_start:
{
lean_object* v___x_2865_; lean_object* v___x_2866_; lean_object* v_nbuckets_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; lean_object* v___x_2870_; lean_object* v___x_2871_; 
v___x_2865_ = lean_array_get_size(v_data_2864_);
v___x_2866_ = lean_unsigned_to_nat(2u);
v_nbuckets_2867_ = lean_nat_mul(v___x_2865_, v___x_2866_);
v___x_2868_ = lean_unsigned_to_nat(0u);
v___x_2869_ = lean_box(0);
v___x_2870_ = lean_mk_array(v_nbuckets_2867_, v___x_2869_);
v___x_2871_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63___redArg(v___x_2868_, v_data_2864_, v___x_2870_);
return v___x_2871_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__58___redArg(lean_object* v_a_2872_, lean_object* v_b_2873_, lean_object* v_x_2874_){
_start:
{
if (lean_obj_tag(v_x_2874_) == 0)
{
lean_dec(v_b_2873_);
lean_dec(v_a_2872_);
return v_x_2874_;
}
else
{
lean_object* v_key_2875_; lean_object* v_value_2876_; lean_object* v_tail_2877_; lean_object* v___x_2879_; uint8_t v_isShared_2880_; uint8_t v_isSharedCheck_2895_; 
v_key_2875_ = lean_ctor_get(v_x_2874_, 0);
v_value_2876_ = lean_ctor_get(v_x_2874_, 1);
v_tail_2877_ = lean_ctor_get(v_x_2874_, 2);
v_isSharedCheck_2895_ = !lean_is_exclusive(v_x_2874_);
if (v_isSharedCheck_2895_ == 0)
{
v___x_2879_ = v_x_2874_;
v_isShared_2880_ = v_isSharedCheck_2895_;
goto v_resetjp_2878_;
}
else
{
lean_inc(v_tail_2877_);
lean_inc(v_value_2876_);
lean_inc(v_key_2875_);
lean_dec(v_x_2874_);
v___x_2879_ = lean_box(0);
v_isShared_2880_ = v_isSharedCheck_2895_;
goto v_resetjp_2878_;
}
v_resetjp_2878_:
{
lean_object* v___x_2881_; lean_object* v_elimGoal_2882_; lean_object* v___x_2883_; lean_object* v_id_2884_; lean_object* v___x_2885_; lean_object* v_id_2886_; uint8_t v___x_2887_; 
v___x_2881_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2882_ = lean_ctor_get(v___x_2881_, 1);
lean_inc_ref_n(v_elimGoal_2882_, 2);
lean_inc(v_key_2875_);
v___x_2883_ = lean_apply_1(v_elimGoal_2882_, v_key_2875_);
v_id_2884_ = lean_ctor_get(v___x_2883_, 0);
lean_inc(v_id_2884_);
lean_dec_ref(v___x_2883_);
lean_inc(v_a_2872_);
v___x_2885_ = lean_apply_1(v_elimGoal_2882_, v_a_2872_);
v_id_2886_ = lean_ctor_get(v___x_2885_, 0);
lean_inc(v_id_2886_);
lean_dec_ref(v___x_2885_);
v___x_2887_ = lean_nat_dec_eq(v_id_2884_, v_id_2886_);
lean_dec(v_id_2886_);
lean_dec(v_id_2884_);
if (v___x_2887_ == 0)
{
lean_object* v___x_2888_; lean_object* v___x_2890_; 
v___x_2888_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__58___redArg(v_a_2872_, v_b_2873_, v_tail_2877_);
if (v_isShared_2880_ == 0)
{
lean_ctor_set(v___x_2879_, 2, v___x_2888_);
v___x_2890_ = v___x_2879_;
goto v_reusejp_2889_;
}
else
{
lean_object* v_reuseFailAlloc_2891_; 
v_reuseFailAlloc_2891_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2891_, 0, v_key_2875_);
lean_ctor_set(v_reuseFailAlloc_2891_, 1, v_value_2876_);
lean_ctor_set(v_reuseFailAlloc_2891_, 2, v___x_2888_);
v___x_2890_ = v_reuseFailAlloc_2891_;
goto v_reusejp_2889_;
}
v_reusejp_2889_:
{
return v___x_2890_;
}
}
else
{
lean_object* v___x_2893_; 
lean_dec(v_value_2876_);
lean_dec(v_key_2875_);
if (v_isShared_2880_ == 0)
{
lean_ctor_set(v___x_2879_, 1, v_b_2873_);
lean_ctor_set(v___x_2879_, 0, v_a_2872_);
v___x_2893_ = v___x_2879_;
goto v_reusejp_2892_;
}
else
{
lean_object* v_reuseFailAlloc_2894_; 
v_reuseFailAlloc_2894_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2894_, 0, v_a_2872_);
lean_ctor_set(v_reuseFailAlloc_2894_, 1, v_b_2873_);
lean_ctor_set(v_reuseFailAlloc_2894_, 2, v_tail_2877_);
v___x_2893_ = v_reuseFailAlloc_2894_;
goto v_reusejp_2892_;
}
v_reusejp_2892_:
{
return v___x_2893_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48___redArg(lean_object* v_m_2896_, lean_object* v_a_2897_, lean_object* v_b_2898_){
_start:
{
lean_object* v_size_2899_; lean_object* v_buckets_2900_; lean_object* v___x_2902_; uint8_t v_isShared_2903_; uint8_t v_isSharedCheck_2947_; 
v_size_2899_ = lean_ctor_get(v_m_2896_, 0);
v_buckets_2900_ = lean_ctor_get(v_m_2896_, 1);
v_isSharedCheck_2947_ = !lean_is_exclusive(v_m_2896_);
if (v_isSharedCheck_2947_ == 0)
{
v___x_2902_ = v_m_2896_;
v_isShared_2903_ = v_isSharedCheck_2947_;
goto v_resetjp_2901_;
}
else
{
lean_inc(v_buckets_2900_);
lean_inc(v_size_2899_);
lean_dec(v_m_2896_);
v___x_2902_ = lean_box(0);
v_isShared_2903_ = v_isSharedCheck_2947_;
goto v_resetjp_2901_;
}
v_resetjp_2901_:
{
lean_object* v___x_2904_; lean_object* v_elimGoal_2905_; lean_object* v___x_2906_; lean_object* v_id_2907_; lean_object* v___x_2908_; uint64_t v___x_2909_; uint64_t v___x_2910_; uint64_t v___x_2911_; uint64_t v_fold_2912_; uint64_t v___x_2913_; uint64_t v___x_2914_; uint64_t v___x_2915_; size_t v___x_2916_; size_t v___x_2917_; size_t v___x_2918_; size_t v___x_2919_; size_t v___x_2920_; lean_object* v_bkt_2921_; uint8_t v___x_2922_; 
v___x_2904_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2905_ = lean_ctor_get(v___x_2904_, 1);
lean_inc_ref(v_elimGoal_2905_);
lean_inc_n(v_a_2897_, 2);
v___x_2906_ = lean_apply_1(v_elimGoal_2905_, v_a_2897_);
v_id_2907_ = lean_ctor_get(v___x_2906_, 0);
lean_inc(v_id_2907_);
lean_dec_ref(v___x_2906_);
v___x_2908_ = lean_array_get_size(v_buckets_2900_);
v___x_2909_ = lean_uint64_of_nat(v_id_2907_);
lean_dec(v_id_2907_);
v___x_2910_ = 32ULL;
v___x_2911_ = lean_uint64_shift_right(v___x_2909_, v___x_2910_);
v_fold_2912_ = lean_uint64_xor(v___x_2909_, v___x_2911_);
v___x_2913_ = 16ULL;
v___x_2914_ = lean_uint64_shift_right(v_fold_2912_, v___x_2913_);
v___x_2915_ = lean_uint64_xor(v_fold_2912_, v___x_2914_);
v___x_2916_ = lean_uint64_to_usize(v___x_2915_);
v___x_2917_ = lean_usize_of_nat(v___x_2908_);
v___x_2918_ = ((size_t)1ULL);
v___x_2919_ = lean_usize_sub(v___x_2917_, v___x_2918_);
v___x_2920_ = lean_usize_land(v___x_2916_, v___x_2919_);
v_bkt_2921_ = lean_array_uget_borrowed(v_buckets_2900_, v___x_2920_);
lean_inc(v_bkt_2921_);
v___x_2922_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg(v_a_2897_, v_bkt_2921_);
if (v___x_2922_ == 0)
{
lean_object* v___x_2923_; lean_object* v_size_x27_2924_; lean_object* v___x_2925_; lean_object* v_buckets_x27_2926_; lean_object* v___x_2927_; lean_object* v___x_2928_; lean_object* v___x_2929_; lean_object* v___x_2930_; lean_object* v___x_2931_; uint8_t v___x_2932_; 
v___x_2923_ = lean_unsigned_to_nat(1u);
v_size_x27_2924_ = lean_nat_add(v_size_2899_, v___x_2923_);
lean_dec(v_size_2899_);
lean_inc(v_bkt_2921_);
v___x_2925_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2925_, 0, v_a_2897_);
lean_ctor_set(v___x_2925_, 1, v_b_2898_);
lean_ctor_set(v___x_2925_, 2, v_bkt_2921_);
v_buckets_x27_2926_ = lean_array_uset(v_buckets_2900_, v___x_2920_, v___x_2925_);
v___x_2927_ = lean_unsigned_to_nat(4u);
v___x_2928_ = lean_nat_mul(v_size_x27_2924_, v___x_2927_);
v___x_2929_ = lean_unsigned_to_nat(3u);
v___x_2930_ = lean_nat_div(v___x_2928_, v___x_2929_);
lean_dec(v___x_2928_);
v___x_2931_ = lean_array_get_size(v_buckets_x27_2926_);
v___x_2932_ = lean_nat_dec_le(v___x_2930_, v___x_2931_);
lean_dec(v___x_2930_);
if (v___x_2932_ == 0)
{
lean_object* v_val_2933_; lean_object* v___x_2935_; 
v_val_2933_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57___redArg(v_buckets_x27_2926_);
if (v_isShared_2903_ == 0)
{
lean_ctor_set(v___x_2902_, 1, v_val_2933_);
lean_ctor_set(v___x_2902_, 0, v_size_x27_2924_);
v___x_2935_ = v___x_2902_;
goto v_reusejp_2934_;
}
else
{
lean_object* v_reuseFailAlloc_2936_; 
v_reuseFailAlloc_2936_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2936_, 0, v_size_x27_2924_);
lean_ctor_set(v_reuseFailAlloc_2936_, 1, v_val_2933_);
v___x_2935_ = v_reuseFailAlloc_2936_;
goto v_reusejp_2934_;
}
v_reusejp_2934_:
{
return v___x_2935_;
}
}
else
{
lean_object* v___x_2938_; 
if (v_isShared_2903_ == 0)
{
lean_ctor_set(v___x_2902_, 1, v_buckets_x27_2926_);
lean_ctor_set(v___x_2902_, 0, v_size_x27_2924_);
v___x_2938_ = v___x_2902_;
goto v_reusejp_2937_;
}
else
{
lean_object* v_reuseFailAlloc_2939_; 
v_reuseFailAlloc_2939_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2939_, 0, v_size_x27_2924_);
lean_ctor_set(v_reuseFailAlloc_2939_, 1, v_buckets_x27_2926_);
v___x_2938_ = v_reuseFailAlloc_2939_;
goto v_reusejp_2937_;
}
v_reusejp_2937_:
{
return v___x_2938_;
}
}
}
else
{
lean_object* v___x_2940_; lean_object* v_buckets_x27_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; lean_object* v___x_2945_; 
lean_inc(v_bkt_2921_);
v___x_2940_ = lean_box(0);
v_buckets_x27_2941_ = lean_array_uset(v_buckets_2900_, v___x_2920_, v___x_2940_);
v___x_2942_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__58___redArg(v_a_2897_, v_b_2898_, v_bkt_2921_);
v___x_2943_ = lean_array_uset(v_buckets_x27_2941_, v___x_2920_, v___x_2942_);
if (v_isShared_2903_ == 0)
{
lean_ctor_set(v___x_2902_, 1, v___x_2943_);
v___x_2945_ = v___x_2902_;
goto v_reusejp_2944_;
}
else
{
lean_object* v_reuseFailAlloc_2946_; 
v_reuseFailAlloc_2946_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2946_, 0, v_size_2899_);
lean_ctor_set(v_reuseFailAlloc_2946_, 1, v___x_2943_);
v___x_2945_ = v_reuseFailAlloc_2946_;
goto v_reusejp_2944_;
}
v_reusejp_2944_:
{
return v___x_2945_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43(lean_object* v_x_2950_, lean_object* v_u_2951_){
_start:
{
lean_object* v_parents_2952_; lean_object* v_toRep_2953_; uint8_t v___x_2954_; 
v_parents_2952_ = lean_ctor_get(v_u_2951_, 0);
v_toRep_2953_ = lean_ctor_get(v_u_2951_, 2);
lean_inc(v_x_2950_);
v___x_2954_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___redArg(v_toRep_2953_, v_x_2950_);
if (v___x_2954_ == 0)
{
lean_object* v___x_2956_; uint8_t v_isShared_2957_; uint8_t v_isSharedCheck_2969_; 
lean_inc_ref(v_toRep_2953_);
lean_inc_ref(v_parents_2952_);
v_isSharedCheck_2969_ = !lean_is_exclusive(v_u_2951_);
if (v_isSharedCheck_2969_ == 0)
{
lean_object* v_unused_2970_; lean_object* v_unused_2971_; lean_object* v_unused_2972_; 
v_unused_2970_ = lean_ctor_get(v_u_2951_, 2);
lean_dec(v_unused_2970_);
v_unused_2971_ = lean_ctor_get(v_u_2951_, 1);
lean_dec(v_unused_2971_);
v_unused_2972_ = lean_ctor_get(v_u_2951_, 0);
lean_dec(v_unused_2972_);
v___x_2956_ = v_u_2951_;
v_isShared_2957_ = v_isSharedCheck_2969_;
goto v_resetjp_2955_;
}
else
{
lean_dec(v_u_2951_);
v___x_2956_ = lean_box(0);
v_isShared_2957_ = v_isSharedCheck_2969_;
goto v_resetjp_2955_;
}
v_resetjp_2955_:
{
lean_object* v___x_2958_; size_t v_rep_2959_; lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v___x_2962_; lean_object* v___x_2963_; lean_object* v___x_2964_; lean_object* v___x_2965_; lean_object* v___x_2967_; 
v___x_2958_ = lean_array_get_size(v_parents_2952_);
v_rep_2959_ = lean_usize_of_nat(v___x_2958_);
v___x_2960_ = lean_box_usize(v_rep_2959_);
lean_inc_ref(v_parents_2952_);
v___x_2961_ = lean_array_push(v_parents_2952_, v___x_2960_);
v___x_2962_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43___boxed__const__1));
v___x_2963_ = lean_array_push(v_parents_2952_, v___x_2962_);
v___x_2964_ = lean_box_usize(v_rep_2959_);
v___x_2965_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48___redArg(v_toRep_2953_, v_x_2950_, v___x_2964_);
if (v_isShared_2957_ == 0)
{
lean_ctor_set(v___x_2956_, 2, v___x_2965_);
lean_ctor_set(v___x_2956_, 1, v___x_2963_);
lean_ctor_set(v___x_2956_, 0, v___x_2961_);
v___x_2967_ = v___x_2956_;
goto v_reusejp_2966_;
}
else
{
lean_object* v_reuseFailAlloc_2968_; 
v_reuseFailAlloc_2968_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2968_, 0, v___x_2961_);
lean_ctor_set(v_reuseFailAlloc_2968_, 1, v___x_2963_);
lean_ctor_set(v_reuseFailAlloc_2968_, 2, v___x_2965_);
v___x_2967_ = v_reuseFailAlloc_2968_;
goto v_reusejp_2966_;
}
v_reusejp_2966_:
{
return v___x_2967_;
}
}
}
else
{
lean_dec(v_x_2950_);
return v_u_2951_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__44(lean_object* v_as_2973_, size_t v_i_2974_, size_t v_stop_2975_, lean_object* v_b_2976_){
_start:
{
uint8_t v___x_2977_; 
v___x_2977_ = lean_usize_dec_eq(v_i_2974_, v_stop_2975_);
if (v___x_2977_ == 0)
{
lean_object* v___x_2978_; lean_object* v___x_2979_; size_t v___x_2980_; size_t v___x_2981_; 
v___x_2978_ = lean_array_uget_borrowed(v_as_2973_, v_i_2974_);
lean_inc(v___x_2978_);
v___x_2979_ = lp_aesop_Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43(v___x_2978_, v_b_2976_);
v___x_2980_ = ((size_t)1ULL);
v___x_2981_ = lean_usize_add(v_i_2974_, v___x_2980_);
v_i_2974_ = v___x_2981_;
v_b_2976_ = v___x_2979_;
goto _start;
}
else
{
return v_b_2976_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__44___boxed(lean_object* v_as_2983_, lean_object* v_i_2984_, lean_object* v_stop_2985_, lean_object* v_b_2986_){
_start:
{
size_t v_i_boxed_2987_; size_t v_stop_boxed_2988_; lean_object* v_res_2989_; 
v_i_boxed_2987_ = lean_unbox_usize(v_i_2984_);
lean_dec(v_i_2984_);
v_stop_boxed_2988_ = lean_unbox_usize(v_stop_2985_);
lean_dec(v_stop_2985_);
v_res_2989_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__44(v_as_2983_, v_i_boxed_2987_, v_stop_boxed_2988_, v_b_2986_);
lean_dec_ref(v_as_2983_);
return v_res_2989_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36(lean_object* v_xs_2990_, lean_object* v_u_2991_){
_start:
{
lean_object* v___x_2992_; lean_object* v___x_2993_; uint8_t v___x_2994_; 
v___x_2992_ = lean_unsigned_to_nat(0u);
v___x_2993_ = lean_array_get_size(v_xs_2990_);
v___x_2994_ = lean_nat_dec_lt(v___x_2992_, v___x_2993_);
if (v___x_2994_ == 0)
{
return v_u_2991_;
}
else
{
uint8_t v___x_2995_; 
v___x_2995_ = lean_nat_dec_le(v___x_2993_, v___x_2993_);
if (v___x_2995_ == 0)
{
if (v___x_2994_ == 0)
{
return v_u_2991_;
}
else
{
size_t v___x_2996_; size_t v___x_2997_; lean_object* v___x_2998_; 
v___x_2996_ = ((size_t)0ULL);
v___x_2997_ = lean_usize_of_nat(v___x_2993_);
v___x_2998_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__44(v_xs_2990_, v___x_2996_, v___x_2997_, v_u_2991_);
return v___x_2998_;
}
}
else
{
size_t v___x_2999_; size_t v___x_3000_; lean_object* v___x_3001_; 
v___x_2999_ = ((size_t)0ULL);
v___x_3000_ = lean_usize_of_nat(v___x_2993_);
v___x_3001_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__44(v_xs_2990_, v___x_2999_, v___x_3000_, v_u_2991_);
return v___x_3001_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36___boxed(lean_object* v_xs_3002_, lean_object* v_u_3003_){
_start:
{
lean_object* v_res_3004_; 
v_res_3004_ = lp_aesop_Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36(v_xs_3002_, v_u_3003_);
lean_dec_ref(v_xs_3002_);
return v_res_3004_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__1(void){
_start:
{
lean_object* v___x_3007_; lean_object* v___x_3008_; lean_object* v___x_3009_; 
v___x_3007_ = lean_box(0);
v___x_3008_ = lean_unsigned_to_nat(16u);
v___x_3009_ = lean_mk_array(v___x_3008_, v___x_3007_);
return v___x_3009_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__2(void){
_start:
{
lean_object* v___x_3010_; lean_object* v___x_3011_; lean_object* v___x_3012_; 
v___x_3010_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__1, &lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__1_once, _init_lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__1);
v___x_3011_ = lean_unsigned_to_nat(0u);
v___x_3012_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3012_, 0, v___x_3011_);
lean_ctor_set(v___x_3012_, 1, v___x_3010_);
return v___x_3012_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__3(void){
_start:
{
lean_object* v___x_3013_; lean_object* v___x_3014_; lean_object* v___x_3015_; 
v___x_3013_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__2, &lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__2_once, _init_lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__2);
v___x_3014_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__0));
v___x_3015_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3015_, 0, v___x_3014_);
lean_ctor_set(v___x_3015_, 1, v___x_3014_);
lean_ctor_set(v___x_3015_, 2, v___x_3013_);
return v___x_3015_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30(lean_object* v_xs_3016_){
_start:
{
lean_object* v___x_3017_; lean_object* v___x_3018_; 
v___x_3017_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__3, &lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__3_once, _init_lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___closed__3);
v___x_3018_ = lp_aesop_Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36(v_xs_3016_, v___x_3017_);
return v___x_3018_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30___boxed(lean_object* v_xs_3019_){
_start:
{
lean_object* v_res_3020_; 
v_res_3020_ = lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30(v_xs_3019_);
lean_dec_ref(v_xs_3019_);
return v_res_3020_;
}
}
static lean_object* _init_lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__0(void){
_start:
{
lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; 
v___x_3021_ = lean_box(0);
v___x_3022_ = lean_unsigned_to_nat(16u);
v___x_3023_ = lean_mk_array(v___x_3022_, v___x_3021_);
return v___x_3023_;
}
}
static lean_object* _init_lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1(void){
_start:
{
lean_object* v___x_3024_; lean_object* v___x_3025_; lean_object* v_aOccs_3026_; 
v___x_3024_ = lean_obj_once(&lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__0, &lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__0_once, _init_lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__0);
v___x_3025_ = lean_unsigned_to_nat(0u);
v_aOccs_3026_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_aOccs_3026_, 0, v___x_3025_);
lean_ctor_set(v_aOccs_3026_, 1, v___x_3024_);
return v_aOccs_3026_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19(lean_object* v_f_3027_, lean_object* v_as_3028_){
_start:
{
lean_object* v_clusters_3029_; lean_object* v_aOccs_3030_; lean_object* v___x_3031_; size_t v_sz_3032_; size_t v___x_3033_; lean_object* v___x_3034_; lean_object* v_fst_3035_; lean_object* v___x_3036_; lean_object* v_fst_3037_; 
v_clusters_3029_ = lp_aesop_Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30(v_as_3028_);
v_aOccs_3030_ = lean_obj_once(&lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1, &lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1_once, _init_lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1);
v___x_3031_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3031_, 0, v_clusters_3029_);
lean_ctor_set(v___x_3031_, 1, v_aOccs_3030_);
v_sz_3032_ = lean_array_size(v_as_3028_);
v___x_3033_ = ((size_t)0ULL);
v___x_3034_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__31(v_f_3027_, v_as_3028_, v_sz_3032_, v___x_3033_, v___x_3031_);
v_fst_3035_ = lean_ctor_get(v___x_3034_, 0);
lean_inc(v_fst_3035_);
lean_dec_ref(v___x_3034_);
v___x_3036_ = lp_aesop_Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32(v_fst_3035_);
v_fst_3037_ = lean_ctor_get(v___x_3036_, 0);
lean_inc(v_fst_3037_);
lean_dec_ref(v___x_3036_);
return v_fst_3037_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___boxed(lean_object* v_f_3038_, lean_object* v_as_3039_){
_start:
{
lean_object* v_res_3040_; 
v_res_3040_ = lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19(v_f_3038_, v_as_3039_);
lean_dec_ref(v_as_3039_);
return v_res_3040_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__2(size_t v_sz_3041_, size_t v_i_3042_, lean_object* v_bs_3043_){
_start:
{
uint8_t v___x_3044_; 
v___x_3044_ = lean_usize_dec_lt(v_i_3042_, v_sz_3041_);
if (v___x_3044_ == 0)
{
return v_bs_3043_;
}
else
{
lean_object* v_v_3045_; lean_object* v___x_3046_; lean_object* v_bs_x27_3047_; lean_object* v___x_3048_; size_t v___x_3049_; size_t v___x_3050_; lean_object* v___x_3051_; 
v_v_3045_ = lean_array_uget(v_bs_3043_, v_i_3042_);
v___x_3046_ = lean_unsigned_to_nat(0u);
v_bs_x27_3047_ = lean_array_uset(v_bs_3043_, v_i_3042_, v___x_3046_);
v___x_3048_ = lp_aesop_Aesop_Subgoal_mvarId(v_v_3045_);
lean_dec(v_v_3045_);
v___x_3049_ = ((size_t)1ULL);
v___x_3050_ = lean_usize_add(v_i_3042_, v___x_3049_);
v___x_3051_ = lean_array_uset(v_bs_x27_3047_, v_i_3042_, v___x_3048_);
v_i_3042_ = v___x_3050_;
v_bs_3043_ = v___x_3051_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__2___boxed(lean_object* v_sz_3053_, lean_object* v_i_3054_, lean_object* v_bs_3055_){
_start:
{
size_t v_sz_boxed_3056_; size_t v_i_boxed_3057_; lean_object* v_res_3058_; 
v_sz_boxed_3056_ = lean_unbox_usize(v_sz_3053_);
lean_dec(v_sz_3053_);
v_i_boxed_3057_ = lean_unbox_usize(v_i_3054_);
lean_dec(v_i_3054_);
v_res_3058_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__2(v_sz_boxed_3056_, v_i_boxed_3057_, v_bs_3055_);
return v_res_3058_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17(lean_object* v_fst_3062_, lean_object* v_as_3063_, size_t v_sz_3064_, size_t v_i_3065_, lean_object* v_b_3066_){
_start:
{
uint8_t v___x_3067_; 
v___x_3067_ = lean_usize_dec_lt(v_i_3065_, v_sz_3064_);
if (v___x_3067_ == 0)
{
lean_inc_ref(v_b_3066_);
return v_b_3066_;
}
else
{
lean_object* v___x_3068_; lean_object* v_elimGoal_3069_; lean_object* v_a_3070_; lean_object* v___x_3071_; lean_object* v_preNormGoal_3072_; lean_object* v___x_3073_; lean_object* v___x_3074_; uint8_t v___x_3075_; 
v___x_3068_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3069_ = lean_ctor_get(v___x_3068_, 1);
v_a_3070_ = lean_array_uget_borrowed(v_as_3063_, v_i_3065_);
lean_inc_ref(v_elimGoal_3069_);
lean_inc(v_a_3070_);
v___x_3071_ = lean_apply_1(v_elimGoal_3069_, v_a_3070_);
v_preNormGoal_3072_ = lean_ctor_get(v___x_3071_, 5);
lean_inc(v_preNormGoal_3072_);
lean_dec_ref(v___x_3071_);
v___x_3073_ = lean_box(0);
v___x_3074_ = lp_aesop_Aesop_Subgoal_mvarId(v_fst_3062_);
v___x_3075_ = l_Lean_instBEqMVarId_beq(v_preNormGoal_3072_, v___x_3074_);
lean_dec(v___x_3074_);
lean_dec(v_preNormGoal_3072_);
if (v___x_3075_ == 0)
{
lean_object* v___x_3076_; size_t v___x_3077_; size_t v___x_3078_; 
v___x_3076_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17___closed__0));
v___x_3077_ = ((size_t)1ULL);
v___x_3078_ = lean_usize_add(v_i_3065_, v___x_3077_);
v_i_3065_ = v___x_3078_;
v_b_3066_ = v___x_3076_;
goto _start;
}
else
{
lean_object* v___x_3080_; lean_object* v___x_3081_; lean_object* v___x_3082_; 
lean_inc(v_a_3070_);
v___x_3080_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3080_, 0, v_a_3070_);
v___x_3081_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3081_, 0, v___x_3080_);
v___x_3082_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3082_, 0, v___x_3081_);
lean_ctor_set(v___x_3082_, 1, v___x_3073_);
return v___x_3082_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17___boxed(lean_object* v_fst_3083_, lean_object* v_as_3084_, lean_object* v_sz_3085_, lean_object* v_i_3086_, lean_object* v_b_3087_){
_start:
{
size_t v_sz_boxed_3088_; size_t v_i_boxed_3089_; lean_object* v_res_3090_; 
v_sz_boxed_3088_ = lean_unbox_usize(v_sz_3085_);
lean_dec(v_sz_3085_);
v_i_boxed_3089_ = lean_unbox_usize(v_i_3086_);
lean_dec(v_i_3086_);
v_res_3090_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17(v_fst_3083_, v_as_3084_, v_sz_boxed_3088_, v_i_boxed_3089_, v_b_3087_);
lean_dec_ref(v_b_3087_);
lean_dec_ref(v_as_3084_);
lean_dec_ref(v_fst_3083_);
return v_res_3090_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16(lean_object* v_fst_3094_, lean_object* v_as_3095_, size_t v_sz_3096_, size_t v_i_3097_, lean_object* v_b_3098_){
_start:
{
uint8_t v___x_3099_; 
v___x_3099_ = lean_usize_dec_lt(v_i_3097_, v_sz_3096_);
if (v___x_3099_ == 0)
{
lean_inc_ref(v_b_3098_);
return v_b_3098_;
}
else
{
lean_object* v___x_3100_; lean_object* v_a_3101_; lean_object* v___x_3102_; lean_object* v___x_3103_; uint8_t v___x_3104_; 
v___x_3100_ = lean_box(0);
v_a_3101_ = lean_array_uget_borrowed(v_as_3095_, v_i_3097_);
v___x_3102_ = lp_aesop_Aesop_Subgoal_mvarId(v_a_3101_);
v___x_3103_ = lp_aesop_Aesop_Subgoal_mvarId(v_fst_3094_);
v___x_3104_ = l_Lean_instBEqMVarId_beq(v___x_3102_, v___x_3103_);
lean_dec(v___x_3103_);
lean_dec(v___x_3102_);
if (v___x_3104_ == 0)
{
lean_object* v___x_3105_; size_t v___x_3106_; size_t v___x_3107_; 
v___x_3105_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16___closed__0));
v___x_3106_ = ((size_t)1ULL);
v___x_3107_ = lean_usize_add(v_i_3097_, v___x_3106_);
v_i_3097_ = v___x_3107_;
v_b_3098_ = v___x_3105_;
goto _start;
}
else
{
lean_object* v___x_3109_; lean_object* v___x_3110_; lean_object* v___x_3111_; 
lean_inc(v_a_3101_);
v___x_3109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3109_, 0, v_a_3101_);
v___x_3110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3110_, 0, v___x_3109_);
v___x_3111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3111_, 0, v___x_3110_);
lean_ctor_set(v___x_3111_, 1, v___x_3100_);
return v___x_3111_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16___boxed(lean_object* v_fst_3112_, lean_object* v_as_3113_, lean_object* v_sz_3114_, lean_object* v_i_3115_, lean_object* v_b_3116_){
_start:
{
size_t v_sz_boxed_3117_; size_t v_i_boxed_3118_; lean_object* v_res_3119_; 
v_sz_boxed_3117_ = lean_unbox_usize(v_sz_3114_);
lean_dec(v_sz_3114_);
v_i_boxed_3118_ = lean_unbox_usize(v_i_3115_);
lean_dec(v_i_3115_);
v_res_3119_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16(v_fst_3112_, v_as_3113_, v_sz_boxed_3117_, v_i_boxed_3118_, v_b_3116_);
lean_dec_ref(v_b_3116_);
lean_dec_ref(v_as_3113_);
lean_dec_ref(v_fst_3112_);
return v_res_3119_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__1(void){
_start:
{
lean_object* v___x_3121_; lean_object* v___x_3122_; 
v___x_3121_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__0));
v___x_3122_ = l_Lean_stringToMessageData(v___x_3121_);
return v___x_3122_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__14(void){
_start:
{
lean_object* v___x_3135_; lean_object* v___x_3136_; 
v___x_3135_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__13));
v___x_3136_ = l_Lean_stringToMessageData(v___x_3135_);
return v___x_3136_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18(lean_object* v_val_3140_, lean_object* v_r_3141_, lean_object* v___x_3142_, lean_object* v___x_3143_, double v___x_3144_, lean_object* v___x_3145_, lean_object* v_a_3146_, lean_object* v_a_3147_, size_t v_sz_3148_, size_t v_i_3149_, lean_object* v_bs_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_, lean_object* v___y_3153_, lean_object* v___y_3154_, lean_object* v___y_3155_, lean_object* v___y_3156_, lean_object* v___y_3157_){
_start:
{
uint8_t v___x_3159_; 
v___x_3159_ = lean_usize_dec_lt(v_i_3149_, v_sz_3148_);
if (v___x_3159_ == 0)
{
lean_object* v___x_3160_; 
lean_dec(v___x_3143_);
lean_dec(v_val_3140_);
v___x_3160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3160_, 0, v_bs_3150_);
return v___x_3160_;
}
else
{
lean_object* v_v_3161_; lean_object* v_fst_3162_; lean_object* v_snd_3163_; lean_object* v___x_3165_; uint8_t v_isShared_3166_; uint8_t v_isSharedCheck_3283_; 
v_v_3161_ = lean_array_uget(v_bs_3150_, v_i_3149_);
v_fst_3162_ = lean_ctor_get(v_v_3161_, 0);
v_snd_3163_ = lean_ctor_get(v_v_3161_, 1);
v_isSharedCheck_3283_ = !lean_is_exclusive(v_v_3161_);
if (v_isSharedCheck_3283_ == 0)
{
v___x_3165_ = v_v_3161_;
v_isShared_3166_ = v_isSharedCheck_3283_;
goto v_resetjp_3164_;
}
else
{
lean_inc(v_snd_3163_);
lean_inc(v_fst_3162_);
lean_dec(v_v_3161_);
v___x_3165_ = lean_box(0);
v_isShared_3166_ = v_isSharedCheck_3283_;
goto v_resetjp_3164_;
}
v_resetjp_3164_:
{
lean_object* v___x_3167_; lean_object* v___x_3168_; size_t v_sz_3169_; size_t v___x_3170_; lean_object* v___x_3171_; lean_object* v_fst_3172_; lean_object* v___x_3174_; uint8_t v_isShared_3175_; uint8_t v_isSharedCheck_3281_; 
v___x_3167_ = lean_box(0);
v___x_3168_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17___closed__0));
v_sz_3169_ = lean_array_size(v_a_3147_);
v___x_3170_ = ((size_t)0ULL);
v___x_3171_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__17(v_fst_3162_, v_a_3147_, v_sz_3169_, v___x_3170_, v___x_3168_);
v_fst_3172_ = lean_ctor_get(v___x_3171_, 0);
v_isSharedCheck_3281_ = !lean_is_exclusive(v___x_3171_);
if (v_isSharedCheck_3281_ == 0)
{
lean_object* v_unused_3282_; 
v_unused_3282_ = lean_ctor_get(v___x_3171_, 1);
lean_dec(v_unused_3282_);
v___x_3174_ = v___x_3171_;
v_isShared_3175_ = v_isSharedCheck_3281_;
goto v_resetjp_3173_;
}
else
{
lean_inc(v_fst_3172_);
lean_dec(v___x_3171_);
v___x_3174_ = lean_box(0);
v_isShared_3175_ = v_isSharedCheck_3281_;
goto v_resetjp_3173_;
}
v_resetjp_3173_:
{
lean_object* v___x_3176_; lean_object* v_bs_x27_3177_; lean_object* v_a_3179_; lean_object* v___y_3185_; lean_object* v___y_3196_; lean_object* v___y_3197_; lean_object* v___y_3198_; lean_object* v___y_3199_; lean_object* v_name_3200_; lean_object* v___y_3201_; lean_object* v___y_3220_; lean_object* v___y_3221_; lean_object* v___y_3222_; lean_object* v_name_3223_; uint8_t v_scope_3224_; lean_object* v___y_3225_; lean_object* v___y_3226_; lean_object* v___y_3232_; lean_object* v___y_3233_; lean_object* v___y_3234_; lean_object* v___y_3235_; lean_object* v___y_3250_; lean_object* v___y_3251_; uint8_t v___y_3252_; lean_object* v___y_3260_; 
v___x_3176_ = lean_unsigned_to_nat(0u);
v_bs_x27_3177_ = lean_array_uset(v_bs_3150_, v_i_3149_, v___x_3176_);
if (lean_obj_tag(v_fst_3172_) == 0)
{
goto v___jp_3273_;
}
else
{
lean_object* v_val_3279_; 
v_val_3279_ = lean_ctor_get(v_fst_3172_, 0);
lean_inc(v_val_3279_);
lean_dec_ref_known(v_fst_3172_, 1);
if (lean_obj_tag(v_val_3279_) == 1)
{
lean_object* v_val_3280_; 
lean_del_object(v___x_3174_);
lean_del_object(v___x_3165_);
lean_dec(v_snd_3163_);
lean_dec(v_fst_3162_);
v_val_3280_ = lean_ctor_get(v_val_3279_, 0);
lean_inc(v_val_3280_);
lean_dec_ref_known(v_val_3279_, 1);
v_a_3179_ = v_val_3280_;
goto v___jp_3178_;
}
else
{
lean_dec(v_val_3279_);
goto v___jp_3273_;
}
}
v___jp_3178_:
{
size_t v___x_3180_; size_t v___x_3181_; lean_object* v___x_3182_; 
v___x_3180_ = ((size_t)1ULL);
v___x_3181_ = lean_usize_add(v_i_3149_, v___x_3180_);
v___x_3182_ = lean_array_uset(v_bs_x27_3177_, v_i_3149_, v_a_3179_);
v_i_3149_ = v___x_3181_;
v_bs_3150_ = v___x_3182_;
goto _start;
}
v___jp_3184_:
{
if (lean_obj_tag(v___y_3185_) == 0)
{
lean_object* v_a_3186_; 
v_a_3186_ = lean_ctor_get(v___y_3185_, 0);
lean_inc(v_a_3186_);
lean_dec_ref_known(v___y_3185_, 1);
v_a_3179_ = v_a_3186_;
goto v___jp_3178_;
}
else
{
lean_object* v_a_3187_; lean_object* v___x_3189_; uint8_t v_isShared_3190_; uint8_t v_isSharedCheck_3194_; 
lean_dec_ref(v_bs_x27_3177_);
lean_dec(v___x_3143_);
lean_dec(v_val_3140_);
v_a_3187_ = lean_ctor_get(v___y_3185_, 0);
v_isSharedCheck_3194_ = !lean_is_exclusive(v___y_3185_);
if (v_isSharedCheck_3194_ == 0)
{
v___x_3189_ = v___y_3185_;
v_isShared_3190_ = v_isSharedCheck_3194_;
goto v_resetjp_3188_;
}
else
{
lean_inc(v_a_3187_);
lean_dec(v___y_3185_);
v___x_3189_ = lean_box(0);
v_isShared_3190_ = v_isSharedCheck_3194_;
goto v_resetjp_3188_;
}
v_resetjp_3188_:
{
lean_object* v___x_3192_; 
if (v_isShared_3190_ == 0)
{
v___x_3192_ = v___x_3189_;
goto v_reusejp_3191_;
}
else
{
lean_object* v_reuseFailAlloc_3193_; 
v_reuseFailAlloc_3193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3193_, 0, v_a_3187_);
v___x_3192_ = v_reuseFailAlloc_3193_;
goto v_reusejp_3191_;
}
v_reusejp_3191_:
{
return v___x_3192_;
}
}
}
}
v___jp_3195_:
{
lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; lean_object* v___x_3206_; lean_object* v___x_3207_; lean_object* v___x_3209_; 
v___x_3202_ = lean_string_append(v___y_3197_, v___y_3201_);
v___x_3203_ = lean_string_append(v___x_3202_, v___y_3196_);
v___x_3204_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_3200_, v___x_3159_);
v___x_3205_ = lean_string_append(v___x_3203_, v___x_3204_);
lean_dec_ref(v___x_3204_);
v___x_3206_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3206_, 0, v___x_3205_);
v___x_3207_ = l_Lean_MessageData_ofFormat(v___x_3206_);
lean_inc_ref(v___y_3199_);
if (v_isShared_3175_ == 0)
{
lean_ctor_set_tag(v___x_3174_, 7);
lean_ctor_set(v___x_3174_, 1, v___x_3207_);
lean_ctor_set(v___x_3174_, 0, v___y_3199_);
v___x_3209_ = v___x_3174_;
goto v_reusejp_3208_;
}
else
{
lean_object* v_reuseFailAlloc_3218_; 
v_reuseFailAlloc_3218_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3218_, 0, v___y_3199_);
lean_ctor_set(v_reuseFailAlloc_3218_, 1, v___x_3207_);
v___x_3209_ = v_reuseFailAlloc_3218_;
goto v_reusejp_3208_;
}
v_reusejp_3208_:
{
lean_object* v___x_3210_; lean_object* v___x_3212_; 
v___x_3210_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__1);
if (v_isShared_3166_ == 0)
{
lean_ctor_set_tag(v___x_3165_, 7);
lean_ctor_set(v___x_3165_, 1, v___x_3210_);
lean_ctor_set(v___x_3165_, 0, v___x_3209_);
v___x_3212_ = v___x_3165_;
goto v_reusejp_3211_;
}
else
{
lean_object* v_reuseFailAlloc_3217_; 
v_reuseFailAlloc_3217_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3217_, 0, v___x_3209_);
lean_ctor_set(v_reuseFailAlloc_3217_, 1, v___x_3210_);
v___x_3212_ = v_reuseFailAlloc_3217_;
goto v_reusejp_3211_;
}
v_reusejp_3211_:
{
lean_object* v___x_3213_; lean_object* v___x_3214_; lean_object* v___x_3215_; lean_object* v___x_3216_; 
v___x_3213_ = l_Lean_Exception_toMessageData(v___y_3198_);
v___x_3214_ = l_Lean_indentD(v___x_3213_);
v___x_3215_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3215_, 0, v___x_3212_);
lean_ctor_set(v___x_3215_, 1, v___x_3214_);
v___x_3216_ = lp_aesop_Lean_throwError___at___00Aesop_findPathForAssignedMVars_spec__7___redArg(v___x_3215_, v___y_3154_, v___y_3155_, v___y_3156_, v___y_3157_);
v___y_3185_ = v___x_3216_;
goto v___jp_3184_;
}
}
}
v___jp_3219_:
{
lean_object* v___x_3227_; lean_object* v___x_3228_; 
v___x_3227_ = lean_string_append(v___y_3225_, v___y_3226_);
v___x_3228_ = lean_string_append(v___x_3227_, v___y_3220_);
if (v_scope_3224_ == 0)
{
lean_object* v___x_3229_; 
v___x_3229_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__2));
v___y_3196_ = v___y_3220_;
v___y_3197_ = v___x_3228_;
v___y_3198_ = v___y_3221_;
v___y_3199_ = v___y_3222_;
v_name_3200_ = v_name_3223_;
v___y_3201_ = v___x_3229_;
goto v___jp_3195_;
}
else
{
lean_object* v___x_3230_; 
v___x_3230_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__3));
v___y_3196_ = v___y_3220_;
v___y_3197_ = v___x_3228_;
v___y_3198_ = v___y_3221_;
v___y_3199_ = v___y_3222_;
v_name_3200_ = v_name_3223_;
v___y_3201_ = v___x_3230_;
goto v___jp_3195_;
}
}
v___jp_3231_:
{
lean_object* v_name_3236_; uint8_t v_builder_3237_; uint8_t v_scope_3238_; lean_object* v___x_3239_; lean_object* v___x_3240_; 
v_name_3236_ = lean_ctor_get(v___y_3234_, 0);
lean_inc(v_name_3236_);
v_builder_3237_ = lean_ctor_get_uint8(v___y_3234_, sizeof(void*)*1 + 8);
v_scope_3238_ = lean_ctor_get_uint8(v___y_3234_, sizeof(void*)*1 + 10);
lean_dec_ref(v___y_3234_);
v___x_3239_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__4));
lean_inc_ref(v___y_3235_);
v___x_3240_ = lean_string_append(v___y_3235_, v___x_3239_);
switch(v_builder_3237_)
{
case 0:
{
lean_object* v___x_3241_; 
v___x_3241_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__5));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3241_;
goto v___jp_3219_;
}
case 1:
{
lean_object* v___x_3242_; 
v___x_3242_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__6));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3242_;
goto v___jp_3219_;
}
case 2:
{
lean_object* v___x_3243_; 
v___x_3243_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__7));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3243_;
goto v___jp_3219_;
}
case 3:
{
lean_object* v___x_3244_; 
v___x_3244_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__8));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3244_;
goto v___jp_3219_;
}
case 4:
{
lean_object* v___x_3245_; 
v___x_3245_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__9));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3245_;
goto v___jp_3219_;
}
case 5:
{
lean_object* v___x_3246_; 
v___x_3246_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__10));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3246_;
goto v___jp_3219_;
}
case 6:
{
lean_object* v___x_3247_; 
v___x_3247_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__11));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3247_;
goto v___jp_3219_;
}
default: 
{
lean_object* v___x_3248_; 
v___x_3248_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__12));
v___y_3220_ = v___x_3239_;
v___y_3221_ = v___y_3232_;
v___y_3222_ = v___y_3233_;
v_name_3223_ = v_name_3236_;
v_scope_3224_ = v_scope_3238_;
v___y_3225_ = v___x_3240_;
v___y_3226_ = v___x_3248_;
goto v___jp_3219_;
}
}
}
v___jp_3249_:
{
if (v___y_3252_ == 0)
{
lean_object* v___x_3253_; uint8_t v_phase_3254_; lean_object* v___x_3255_; 
lean_dec_ref(v___y_3250_);
v___x_3253_ = lp_aesop_Aesop_RegularRule_name(v___x_3145_);
v_phase_3254_ = lean_ctor_get_uint8(v___x_3253_, sizeof(void*)*1 + 9);
v___x_3255_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__14, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__14_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__14);
switch(v_phase_3254_)
{
case 0:
{
lean_object* v___x_3256_; 
v___x_3256_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__15));
v___y_3232_ = v___y_3251_;
v___y_3233_ = v___x_3255_;
v___y_3234_ = v___x_3253_;
v___y_3235_ = v___x_3256_;
goto v___jp_3231_;
}
case 1:
{
lean_object* v___x_3257_; 
v___x_3257_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__16));
v___y_3232_ = v___y_3251_;
v___y_3233_ = v___x_3255_;
v___y_3234_ = v___x_3253_;
v___y_3235_ = v___x_3257_;
goto v___jp_3231_;
}
default: 
{
lean_object* v___x_3258_; 
v___x_3258_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___closed__17));
v___y_3232_ = v___y_3251_;
v___y_3233_ = v___x_3255_;
v___y_3234_ = v___x_3253_;
v___y_3235_ = v___x_3258_;
goto v___jp_3231_;
}
}
}
else
{
lean_dec_ref(v___y_3251_);
lean_del_object(v___x_3174_);
lean_del_object(v___x_3165_);
v___y_3185_ = v___y_3250_;
goto v___jp_3184_;
}
}
v___jp_3259_:
{
lean_object* v___x_3261_; lean_object* v_elimGoal_3262_; lean_object* v___x_3263_; lean_object* v_forwardState_3264_; lean_object* v_forwardRuleMatches_3265_; lean_object* v___x_3266_; lean_object* v___x_3267_; 
v___x_3261_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3262_ = lean_ctor_get(v___x_3261_, 1);
lean_inc_ref(v_elimGoal_3262_);
lean_inc(v_val_3140_);
v___x_3263_ = lean_apply_1(v_elimGoal_3262_, v_val_3140_);
v_forwardState_3264_ = lean_ctor_get(v___x_3263_, 8);
lean_inc_ref(v_forwardState_3264_);
v_forwardRuleMatches_3265_ = lean_ctor_get(v___x_3263_, 9);
lean_inc_ref(v_forwardRuleMatches_3265_);
lean_dec_ref(v___x_3263_);
v___x_3266_ = lp_aesop_Aesop_AddRapp_consumedForwardRuleMatches(v_r_3141_);
lean_inc(v___y_3260_);
lean_inc(v___x_3143_);
v___x_3267_ = lp_aesop_Aesop_makeInitialGoal(v_fst_3162_, v_snd_3163_, v___x_3167_, v___x_3142_, v_forwardState_3264_, v_forwardRuleMatches_3265_, v___x_3266_, v___x_3143_, v___x_3144_, v___y_3260_, v___y_3151_, v___y_3152_, v___y_3153_, v___y_3154_, v___y_3155_, v___y_3156_, v___y_3157_);
if (lean_obj_tag(v___x_3267_) == 0)
{
lean_del_object(v___x_3174_);
lean_del_object(v___x_3165_);
v___y_3185_ = v___x_3267_;
goto v___jp_3184_;
}
else
{
lean_object* v_a_3268_; uint8_t v___x_3269_; 
v_a_3268_ = lean_ctor_get(v___x_3267_, 0);
lean_inc(v_a_3268_);
v___x_3269_ = l_Lean_Exception_isInterrupt(v_a_3268_);
if (v___x_3269_ == 0)
{
uint8_t v___x_3270_; 
lean_inc(v_a_3268_);
v___x_3270_ = l_Lean_Exception_isRuntime(v_a_3268_);
v___y_3250_ = v___x_3267_;
v___y_3251_ = v_a_3268_;
v___y_3252_ = v___x_3270_;
goto v___jp_3249_;
}
else
{
v___y_3250_ = v___x_3267_;
v___y_3251_ = v_a_3268_;
v___y_3252_ = v___x_3269_;
goto v___jp_3249_;
}
}
}
v___jp_3271_:
{
lean_object* v___x_3272_; 
v___x_3272_ = lean_box(0);
v___y_3260_ = v___x_3272_;
goto v___jp_3259_;
}
v___jp_3273_:
{
size_t v_sz_3274_; lean_object* v___x_3275_; lean_object* v_fst_3276_; 
v_sz_3274_ = lean_array_size(v_a_3146_);
v___x_3275_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__16(v_fst_3162_, v_a_3146_, v_sz_3274_, v___x_3170_, v___x_3168_);
v_fst_3276_ = lean_ctor_get(v___x_3275_, 0);
lean_inc(v_fst_3276_);
lean_dec_ref(v___x_3275_);
if (lean_obj_tag(v_fst_3276_) == 0)
{
goto v___jp_3271_;
}
else
{
lean_object* v_val_3277_; 
v_val_3277_ = lean_ctor_get(v_fst_3276_, 0);
lean_inc(v_val_3277_);
lean_dec_ref_known(v_fst_3276_, 1);
if (lean_obj_tag(v_val_3277_) == 0)
{
goto v___jp_3271_;
}
else
{
lean_object* v___x_3278_; 
lean_dec_ref_known(v_val_3277_, 1);
v___x_3278_ = lean_box(2);
v___y_3260_ = v___x_3278_;
goto v___jp_3259_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18___boxed(lean_object** _args){
lean_object* v_val_3284_ = _args[0];
lean_object* v_r_3285_ = _args[1];
lean_object* v___x_3286_ = _args[2];
lean_object* v___x_3287_ = _args[3];
lean_object* v___x_3288_ = _args[4];
lean_object* v___x_3289_ = _args[5];
lean_object* v_a_3290_ = _args[6];
lean_object* v_a_3291_ = _args[7];
lean_object* v_sz_3292_ = _args[8];
lean_object* v_i_3293_ = _args[9];
lean_object* v_bs_3294_ = _args[10];
lean_object* v___y_3295_ = _args[11];
lean_object* v___y_3296_ = _args[12];
lean_object* v___y_3297_ = _args[13];
lean_object* v___y_3298_ = _args[14];
lean_object* v___y_3299_ = _args[15];
lean_object* v___y_3300_ = _args[16];
lean_object* v___y_3301_ = _args[17];
lean_object* v___y_3302_ = _args[18];
_start:
{
double v___x_124728__boxed_3303_; size_t v_sz_boxed_3304_; size_t v_i_boxed_3305_; lean_object* v_res_3306_; 
v___x_124728__boxed_3303_ = lean_unbox_float(v___x_3288_);
lean_dec_ref(v___x_3288_);
v_sz_boxed_3304_ = lean_unbox_usize(v_sz_3292_);
lean_dec(v_sz_3292_);
v_i_boxed_3305_ = lean_unbox_usize(v_i_3293_);
lean_dec(v_i_3293_);
v_res_3306_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18(v_val_3284_, v_r_3285_, v___x_3286_, v___x_3287_, v___x_124728__boxed_3303_, v___x_3289_, v_a_3290_, v_a_3291_, v_sz_boxed_3304_, v_i_boxed_3305_, v_bs_3294_, v___y_3295_, v___y_3296_, v___y_3297_, v___y_3298_, v___y_3299_, v___y_3300_, v___y_3301_);
lean_dec(v___y_3301_);
lean_dec_ref(v___y_3300_);
lean_dec(v___y_3299_);
lean_dec_ref(v___y_3298_);
lean_dec(v___y_3297_);
lean_dec(v___y_3296_);
lean_dec_ref(v___y_3295_);
lean_dec_ref(v_a_3291_);
lean_dec_ref(v_a_3290_);
lean_dec_ref(v___x_3289_);
lean_dec_ref(v___x_3286_);
lean_dec_ref(v_r_3285_);
return v_res_3306_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___redArg(lean_object* v___x_3307_, lean_object* v_as_3308_, size_t v_sz_3309_, size_t v_i_3310_, lean_object* v_b_3311_){
_start:
{
lean_object* v_a_3314_; uint8_t v___x_3318_; 
v___x_3318_ = lean_usize_dec_lt(v_i_3310_, v_sz_3309_);
if (v___x_3318_ == 0)
{
lean_object* v___x_3319_; 
v___x_3319_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3319_, 0, v_b_3311_);
return v___x_3319_;
}
else
{
lean_object* v_fst_3320_; lean_object* v_snd_3321_; lean_object* v___x_3323_; uint8_t v_isShared_3324_; uint8_t v_isSharedCheck_3336_; 
v_fst_3320_ = lean_ctor_get(v_b_3311_, 0);
v_snd_3321_ = lean_ctor_get(v_b_3311_, 1);
v_isSharedCheck_3336_ = !lean_is_exclusive(v_b_3311_);
if (v_isSharedCheck_3336_ == 0)
{
v___x_3323_ = v_b_3311_;
v_isShared_3324_ = v_isSharedCheck_3336_;
goto v_resetjp_3322_;
}
else
{
lean_inc(v_snd_3321_);
lean_inc(v_fst_3320_);
lean_dec(v_b_3311_);
v___x_3323_ = lean_box(0);
v_isShared_3324_ = v_isSharedCheck_3336_;
goto v_resetjp_3322_;
}
v_resetjp_3322_:
{
lean_object* v_a_3329_; uint8_t v___x_3330_; 
v_a_3329_ = lean_array_uget_borrowed(v_as_3308_, v_i_3310_);
v___x_3330_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_findPathForAssignedMVars_spec__8_spec__12(v___x_3307_, v_a_3329_);
if (v___x_3330_ == 0)
{
uint8_t v___x_3331_; 
v___x_3331_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg(v_snd_3321_, v_a_3329_);
if (v___x_3331_ == 0)
{
lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; 
lean_del_object(v___x_3323_);
lean_inc_n(v_a_3329_, 2);
v___x_3332_ = lp_aesop_Aesop_UnorderedArraySet_insert___at___00Aesop_addRappUnsafe_spec__8(v_a_3329_, v_fst_3320_);
v___x_3333_ = lean_box(0);
v___x_3334_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13___redArg(v_snd_3321_, v_a_3329_, v___x_3333_);
v___x_3335_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3335_, 0, v___x_3332_);
lean_ctor_set(v___x_3335_, 1, v___x_3334_);
v_a_3314_ = v___x_3335_;
goto v___jp_3313_;
}
else
{
goto v___jp_3325_;
}
}
else
{
goto v___jp_3325_;
}
v___jp_3325_:
{
lean_object* v___x_3327_; 
if (v_isShared_3324_ == 0)
{
v___x_3327_ = v___x_3323_;
goto v_reusejp_3326_;
}
else
{
lean_object* v_reuseFailAlloc_3328_; 
v_reuseFailAlloc_3328_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3328_, 0, v_fst_3320_);
lean_ctor_set(v_reuseFailAlloc_3328_, 1, v_snd_3321_);
v___x_3327_ = v_reuseFailAlloc_3328_;
goto v_reusejp_3326_;
}
v_reusejp_3326_:
{
v_a_3314_ = v___x_3327_;
goto v___jp_3313_;
}
}
}
}
v___jp_3313_:
{
size_t v___x_3315_; size_t v___x_3316_; 
v___x_3315_ = ((size_t)1ULL);
v___x_3316_ = lean_usize_add(v_i_3310_, v___x_3315_);
v_i_3310_ = v___x_3316_;
v_b_3311_ = v_a_3314_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___redArg___boxed(lean_object* v___x_3337_, lean_object* v_as_3338_, lean_object* v_sz_3339_, lean_object* v_i_3340_, lean_object* v_b_3341_, lean_object* v___y_3342_){
_start:
{
size_t v_sz_boxed_3343_; size_t v_i_boxed_3344_; lean_object* v_res_3345_; 
v_sz_boxed_3343_ = lean_unbox_usize(v_sz_3339_);
lean_dec(v_sz_3339_);
v_i_boxed_3344_ = lean_unbox_usize(v_i_3340_);
lean_dec(v_i_3340_);
v_res_3345_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___redArg(v___x_3337_, v_as_3338_, v_sz_boxed_3343_, v_i_boxed_3344_, v_b_3341_);
lean_dec_ref(v_as_3338_);
lean_dec_ref(v___x_3337_);
return v_res_3345_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__21(lean_object* v___x_3346_, lean_object* v_as_3347_, size_t v_sz_3348_, size_t v_i_3349_, lean_object* v_b_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_, lean_object* v___y_3355_, lean_object* v___y_3356_, lean_object* v___y_3357_){
_start:
{
uint8_t v___x_3359_; 
v___x_3359_ = lean_usize_dec_lt(v_i_3349_, v_sz_3348_);
if (v___x_3359_ == 0)
{
lean_object* v___x_3360_; 
v___x_3360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3360_, 0, v_b_3350_);
return v___x_3360_;
}
else
{
lean_object* v_fst_3361_; lean_object* v_snd_3362_; lean_object* v___x_3364_; uint8_t v_isShared_3365_; uint8_t v_isSharedCheck_3390_; 
v_fst_3361_ = lean_ctor_get(v_b_3350_, 0);
v_snd_3362_ = lean_ctor_get(v_b_3350_, 1);
v_isSharedCheck_3390_ = !lean_is_exclusive(v_b_3350_);
if (v_isSharedCheck_3390_ == 0)
{
v___x_3364_ = v_b_3350_;
v_isShared_3365_ = v_isSharedCheck_3390_;
goto v_resetjp_3363_;
}
else
{
lean_inc(v_snd_3362_);
lean_inc(v_fst_3361_);
lean_dec(v_b_3350_);
v___x_3364_ = lean_box(0);
v_isShared_3365_ = v_isSharedCheck_3390_;
goto v_resetjp_3363_;
}
v_resetjp_3363_:
{
lean_object* v___x_3366_; lean_object* v_elimGoal_3367_; lean_object* v_a_3368_; lean_object* v___x_3369_; lean_object* v_mvars_3370_; lean_object* v___x_3372_; 
v___x_3366_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_3367_ = lean_ctor_get(v___x_3366_, 1);
v_a_3368_ = lean_array_uget_borrowed(v_as_3347_, v_i_3349_);
lean_inc_ref(v_elimGoal_3367_);
lean_inc(v_a_3368_);
v___x_3369_ = lean_apply_1(v_elimGoal_3367_, v_a_3368_);
v_mvars_3370_ = lean_ctor_get(v___x_3369_, 7);
lean_inc_ref(v_mvars_3370_);
lean_dec_ref(v___x_3369_);
if (v_isShared_3365_ == 0)
{
v___x_3372_ = v___x_3364_;
goto v_reusejp_3371_;
}
else
{
lean_object* v_reuseFailAlloc_3389_; 
v_reuseFailAlloc_3389_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3389_, 0, v_fst_3361_);
lean_ctor_set(v_reuseFailAlloc_3389_, 1, v_snd_3362_);
v___x_3372_ = v_reuseFailAlloc_3389_;
goto v_reusejp_3371_;
}
v_reusejp_3371_:
{
size_t v_sz_3373_; size_t v___x_3374_; lean_object* v___x_3375_; 
v_sz_3373_ = lean_array_size(v_mvars_3370_);
v___x_3374_ = ((size_t)0ULL);
v___x_3375_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___redArg(v___x_3346_, v_mvars_3370_, v_sz_3373_, v___x_3374_, v___x_3372_);
lean_dec_ref(v_mvars_3370_);
if (lean_obj_tag(v___x_3375_) == 0)
{
lean_object* v_a_3376_; lean_object* v_fst_3377_; lean_object* v_snd_3378_; lean_object* v___x_3380_; uint8_t v_isShared_3381_; uint8_t v_isSharedCheck_3388_; 
v_a_3376_ = lean_ctor_get(v___x_3375_, 0);
lean_inc(v_a_3376_);
lean_dec_ref_known(v___x_3375_, 1);
v_fst_3377_ = lean_ctor_get(v_a_3376_, 0);
v_snd_3378_ = lean_ctor_get(v_a_3376_, 1);
v_isSharedCheck_3388_ = !lean_is_exclusive(v_a_3376_);
if (v_isSharedCheck_3388_ == 0)
{
v___x_3380_ = v_a_3376_;
v_isShared_3381_ = v_isSharedCheck_3388_;
goto v_resetjp_3379_;
}
else
{
lean_inc(v_snd_3378_);
lean_inc(v_fst_3377_);
lean_dec(v_a_3376_);
v___x_3380_ = lean_box(0);
v_isShared_3381_ = v_isSharedCheck_3388_;
goto v_resetjp_3379_;
}
v_resetjp_3379_:
{
lean_object* v___x_3383_; 
if (v_isShared_3381_ == 0)
{
v___x_3383_ = v___x_3380_;
goto v_reusejp_3382_;
}
else
{
lean_object* v_reuseFailAlloc_3387_; 
v_reuseFailAlloc_3387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3387_, 0, v_fst_3377_);
lean_ctor_set(v_reuseFailAlloc_3387_, 1, v_snd_3378_);
v___x_3383_ = v_reuseFailAlloc_3387_;
goto v_reusejp_3382_;
}
v_reusejp_3382_:
{
size_t v___x_3384_; size_t v___x_3385_; 
v___x_3384_ = ((size_t)1ULL);
v___x_3385_ = lean_usize_add(v_i_3349_, v___x_3384_);
v_i_3349_ = v___x_3385_;
v_b_3350_ = v___x_3383_;
goto _start;
}
}
}
else
{
return v___x_3375_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__21___boxed(lean_object* v___x_3391_, lean_object* v_as_3392_, lean_object* v_sz_3393_, lean_object* v_i_3394_, lean_object* v_b_3395_, lean_object* v___y_3396_, lean_object* v___y_3397_, lean_object* v___y_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_, lean_object* v___y_3401_, lean_object* v___y_3402_, lean_object* v___y_3403_){
_start:
{
size_t v_sz_boxed_3404_; size_t v_i_boxed_3405_; lean_object* v_res_3406_; 
v_sz_boxed_3404_ = lean_unbox_usize(v_sz_3393_);
lean_dec(v_sz_3393_);
v_i_boxed_3405_ = lean_unbox_usize(v_i_3394_);
lean_dec(v_i_3394_);
v_res_3406_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__21(v___x_3391_, v_as_3392_, v_sz_boxed_3404_, v_i_boxed_3405_, v_b_3395_, v___y_3396_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_, v___y_3402_);
lean_dec(v___y_3402_);
lean_dec_ref(v___y_3401_);
lean_dec(v___y_3400_);
lean_dec_ref(v___y_3399_);
lean_dec(v___y_3398_);
lean_dec(v___y_3397_);
lean_dec_ref(v___y_3396_);
lean_dec_ref(v_as_3392_);
lean_dec_ref(v___x_3391_);
return v_res_3406_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg(lean_object* v_val_3407_, lean_object* v_as_3408_, size_t v_i_3409_, size_t v_stop_3410_, lean_object* v_b_3411_){
_start:
{
uint8_t v___x_3413_; 
v___x_3413_ = lean_usize_dec_eq(v_i_3409_, v_stop_3410_);
if (v___x_3413_ == 0)
{
lean_object* v___x_3414_; lean_object* v___x_3415_; lean_object* v___x_3416_; lean_object* v_introGoal_3417_; lean_object* v_elimGoal_3418_; lean_object* v___x_3419_; lean_object* v_id_3420_; lean_object* v_children_3421_; lean_object* v_origin_3422_; lean_object* v_depth_3423_; uint8_t v_state_3424_; uint8_t v_isIrrelevant_3425_; uint8_t v_isForcedUnprovable_3426_; lean_object* v_preNormGoal_3427_; lean_object* v_normalizationState_3428_; lean_object* v_mvars_3429_; lean_object* v_forwardState_3430_; lean_object* v_forwardRuleMatches_3431_; double v_successProbability_3432_; lean_object* v_addedInIteration_3433_; lean_object* v_lastExpandedInIteration_3434_; uint8_t v_unsafeRulesSelected_3435_; lean_object* v_unsafeQueue_3436_; lean_object* v_failedRapps_3437_; lean_object* v___x_3439_; uint8_t v_isShared_3440_; uint8_t v_isSharedCheck_3449_; 
v___x_3414_ = lean_array_uget_borrowed(v_as_3408_, v_i_3409_);
v___x_3415_ = lean_st_ref_take(v___x_3414_);
v___x_3416_ = lp_aesop_Aesop_treeImpl;
v_introGoal_3417_ = lean_ctor_get(v___x_3416_, 0);
v_elimGoal_3418_ = lean_ctor_get(v___x_3416_, 1);
lean_inc_ref(v_elimGoal_3418_);
v___x_3419_ = lean_apply_1(v_elimGoal_3418_, v___x_3415_);
v_id_3420_ = lean_ctor_get(v___x_3419_, 0);
v_children_3421_ = lean_ctor_get(v___x_3419_, 2);
v_origin_3422_ = lean_ctor_get(v___x_3419_, 3);
v_depth_3423_ = lean_ctor_get(v___x_3419_, 4);
v_state_3424_ = lean_ctor_get_uint8(v___x_3419_, sizeof(void*)*14 + 8);
v_isIrrelevant_3425_ = lean_ctor_get_uint8(v___x_3419_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_3426_ = lean_ctor_get_uint8(v___x_3419_, sizeof(void*)*14 + 10);
v_preNormGoal_3427_ = lean_ctor_get(v___x_3419_, 5);
v_normalizationState_3428_ = lean_ctor_get(v___x_3419_, 6);
v_mvars_3429_ = lean_ctor_get(v___x_3419_, 7);
v_forwardState_3430_ = lean_ctor_get(v___x_3419_, 8);
v_forwardRuleMatches_3431_ = lean_ctor_get(v___x_3419_, 9);
v_successProbability_3432_ = lean_ctor_get_float(v___x_3419_, sizeof(void*)*14);
v_addedInIteration_3433_ = lean_ctor_get(v___x_3419_, 10);
v_lastExpandedInIteration_3434_ = lean_ctor_get(v___x_3419_, 11);
v_unsafeRulesSelected_3435_ = lean_ctor_get_uint8(v___x_3419_, sizeof(void*)*14 + 11);
v_unsafeQueue_3436_ = lean_ctor_get(v___x_3419_, 12);
v_failedRapps_3437_ = lean_ctor_get(v___x_3419_, 13);
v_isSharedCheck_3449_ = !lean_is_exclusive(v___x_3419_);
if (v_isSharedCheck_3449_ == 0)
{
lean_object* v_unused_3450_; 
v_unused_3450_ = lean_ctor_get(v___x_3419_, 1);
lean_dec(v_unused_3450_);
v___x_3439_ = v___x_3419_;
v_isShared_3440_ = v_isSharedCheck_3449_;
goto v_resetjp_3438_;
}
else
{
lean_inc(v_failedRapps_3437_);
lean_inc(v_unsafeQueue_3436_);
lean_inc(v_lastExpandedInIteration_3434_);
lean_inc(v_addedInIteration_3433_);
lean_inc(v_forwardRuleMatches_3431_);
lean_inc(v_forwardState_3430_);
lean_inc(v_mvars_3429_);
lean_inc(v_normalizationState_3428_);
lean_inc(v_preNormGoal_3427_);
lean_inc(v_depth_3423_);
lean_inc(v_origin_3422_);
lean_inc(v_children_3421_);
lean_inc(v_id_3420_);
lean_dec(v___x_3419_);
v___x_3439_ = lean_box(0);
v_isShared_3440_ = v_isSharedCheck_3449_;
goto v_resetjp_3438_;
}
v_resetjp_3438_:
{
lean_object* v___x_3442_; 
lean_inc(v_val_3407_);
if (v_isShared_3440_ == 0)
{
lean_ctor_set(v___x_3439_, 1, v_val_3407_);
v___x_3442_ = v___x_3439_;
goto v_reusejp_3441_;
}
else
{
lean_object* v_reuseFailAlloc_3448_; 
v_reuseFailAlloc_3448_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_3448_, 0, v_id_3420_);
lean_ctor_set(v_reuseFailAlloc_3448_, 1, v_val_3407_);
lean_ctor_set(v_reuseFailAlloc_3448_, 2, v_children_3421_);
lean_ctor_set(v_reuseFailAlloc_3448_, 3, v_origin_3422_);
lean_ctor_set(v_reuseFailAlloc_3448_, 4, v_depth_3423_);
lean_ctor_set(v_reuseFailAlloc_3448_, 5, v_preNormGoal_3427_);
lean_ctor_set(v_reuseFailAlloc_3448_, 6, v_normalizationState_3428_);
lean_ctor_set(v_reuseFailAlloc_3448_, 7, v_mvars_3429_);
lean_ctor_set(v_reuseFailAlloc_3448_, 8, v_forwardState_3430_);
lean_ctor_set(v_reuseFailAlloc_3448_, 9, v_forwardRuleMatches_3431_);
lean_ctor_set(v_reuseFailAlloc_3448_, 10, v_addedInIteration_3433_);
lean_ctor_set(v_reuseFailAlloc_3448_, 11, v_lastExpandedInIteration_3434_);
lean_ctor_set(v_reuseFailAlloc_3448_, 12, v_unsafeQueue_3436_);
lean_ctor_set(v_reuseFailAlloc_3448_, 13, v_failedRapps_3437_);
lean_ctor_set_uint8(v_reuseFailAlloc_3448_, sizeof(void*)*14 + 8, v_state_3424_);
lean_ctor_set_uint8(v_reuseFailAlloc_3448_, sizeof(void*)*14 + 9, v_isIrrelevant_3425_);
lean_ctor_set_uint8(v_reuseFailAlloc_3448_, sizeof(void*)*14 + 10, v_isForcedUnprovable_3426_);
lean_ctor_set_float(v_reuseFailAlloc_3448_, sizeof(void*)*14, v_successProbability_3432_);
lean_ctor_set_uint8(v_reuseFailAlloc_3448_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_3435_);
v___x_3442_ = v_reuseFailAlloc_3448_;
goto v_reusejp_3441_;
}
v_reusejp_3441_:
{
lean_object* v___x_3443_; lean_object* v___x_3444_; size_t v___x_3445_; size_t v___x_3446_; 
lean_inc(v_introGoal_3417_);
v___x_3443_ = lean_apply_1(v_introGoal_3417_, v___x_3442_);
v___x_3444_ = lean_st_ref_set(v___x_3414_, v___x_3443_);
v___x_3445_ = ((size_t)1ULL);
v___x_3446_ = lean_usize_add(v_i_3409_, v___x_3445_);
v_i_3409_ = v___x_3446_;
v_b_3411_ = v___x_3444_;
goto _start;
}
}
}
else
{
lean_object* v___x_3451_; 
lean_dec(v_val_3407_);
v___x_3451_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3451_, 0, v_b_3411_);
return v___x_3451_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg___boxed(lean_object* v_val_3452_, lean_object* v_as_3453_, lean_object* v_i_3454_, lean_object* v_stop_3455_, lean_object* v_b_3456_, lean_object* v___y_3457_){
_start:
{
size_t v_i_boxed_3458_; size_t v_stop_boxed_3459_; lean_object* v_res_3460_; 
v_i_boxed_3458_ = lean_unbox_usize(v_i_3454_);
lean_dec(v_i_3454_);
v_stop_boxed_3459_ = lean_unbox_usize(v_stop_3455_);
lean_dec(v_stop_3455_);
v_res_3460_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg(v_val_3452_, v_as_3453_, v_i_boxed_3458_, v_stop_boxed_3459_, v_b_3456_);
lean_dec_ref(v_as_3453_);
return v_res_3460_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___redArg(size_t v_sz_3461_, size_t v_i_3462_, lean_object* v_bs_3463_){
_start:
{
uint8_t v___x_3465_; 
v___x_3465_ = lean_usize_dec_lt(v_i_3462_, v_sz_3461_);
if (v___x_3465_ == 0)
{
lean_object* v___x_3466_; 
v___x_3466_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3466_, 0, v_bs_3463_);
return v___x_3466_;
}
else
{
lean_object* v_v_3467_; lean_object* v___x_3468_; lean_object* v___x_3469_; lean_object* v_bs_x27_3470_; size_t v___x_3471_; size_t v___x_3472_; lean_object* v___x_3473_; 
v_v_3467_ = lean_array_uget_borrowed(v_bs_3463_, v_i_3462_);
lean_inc(v_v_3467_);
v___x_3468_ = lean_st_mk_ref(v_v_3467_);
v___x_3469_ = lean_unsigned_to_nat(0u);
v_bs_x27_3470_ = lean_array_uset(v_bs_3463_, v_i_3462_, v___x_3469_);
v___x_3471_ = ((size_t)1ULL);
v___x_3472_ = lean_usize_add(v_i_3462_, v___x_3471_);
v___x_3473_ = lean_array_uset(v_bs_x27_3470_, v_i_3462_, v___x_3468_);
v_i_3462_ = v___x_3472_;
v_bs_3463_ = v___x_3473_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___redArg___boxed(lean_object* v_sz_3475_, lean_object* v_i_3476_, lean_object* v_bs_3477_, lean_object* v___y_3478_){
_start:
{
size_t v_sz_boxed_3479_; size_t v_i_boxed_3480_; lean_object* v_res_3481_; 
v_sz_boxed_3479_ = lean_unbox_usize(v_sz_3475_);
lean_dec(v_sz_3475_);
v_i_boxed_3480_ = lean_unbox_usize(v_i_3476_);
lean_dec(v_i_3476_);
v_res_3481_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___redArg(v_sz_boxed_3479_, v_i_boxed_3480_, v_bs_3477_);
return v_res_3481_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__20(lean_object* v_val_3482_, size_t v_sz_3483_, size_t v_i_3484_, lean_object* v_bs_3485_, lean_object* v___y_3486_, lean_object* v___y_3487_, lean_object* v___y_3488_, lean_object* v___y_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_){
_start:
{
uint8_t v___x_3494_; 
v___x_3494_ = lean_usize_dec_lt(v_i_3484_, v_sz_3483_);
if (v___x_3494_ == 0)
{
lean_object* v___x_3495_; 
lean_dec(v_val_3482_);
v___x_3495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3495_, 0, v_bs_3485_);
return v___x_3495_;
}
else
{
lean_object* v_v_3496_; size_t v_sz_3497_; size_t v___x_3498_; lean_object* v___x_3499_; 
v_v_3496_ = lean_array_uget_borrowed(v_bs_3485_, v_i_3484_);
v_sz_3497_ = lean_array_size(v_v_3496_);
v___x_3498_ = ((size_t)0ULL);
lean_inc(v_v_3496_);
v___x_3499_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___redArg(v_sz_3497_, v___x_3498_, v_v_3496_);
if (lean_obj_tag(v___x_3499_) == 0)
{
lean_object* v_a_3500_; lean_object* v___x_3502_; uint8_t v_isShared_3503_; uint8_t v_isSharedCheck_3539_; 
v_a_3500_ = lean_ctor_get(v___x_3499_, 0);
v_isSharedCheck_3539_ = !lean_is_exclusive(v___x_3499_);
if (v_isSharedCheck_3539_ == 0)
{
v___x_3502_ = v___x_3499_;
v_isShared_3503_ = v_isSharedCheck_3539_;
goto v_resetjp_3501_;
}
else
{
lean_inc(v_a_3500_);
lean_dec(v___x_3499_);
v___x_3502_ = lean_box(0);
v_isShared_3503_ = v_isSharedCheck_3539_;
goto v_resetjp_3501_;
}
v_resetjp_3501_:
{
lean_object* v___x_3504_; lean_object* v_introMVarCluster_3505_; uint8_t v___x_3506_; uint8_t v___x_3507_; lean_object* v___x_3509_; 
v___x_3504_ = lp_aesop_Aesop_treeImpl;
v_introMVarCluster_3505_ = lean_ctor_get(v___x_3504_, 4);
v___x_3506_ = 0;
v___x_3507_ = 0;
lean_inc(v_val_3482_);
if (v_isShared_3503_ == 0)
{
lean_ctor_set_tag(v___x_3502_, 1);
lean_ctor_set(v___x_3502_, 0, v_val_3482_);
v___x_3509_ = v___x_3502_;
goto v_reusejp_3508_;
}
else
{
lean_object* v_reuseFailAlloc_3538_; 
v_reuseFailAlloc_3538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3538_, 0, v_val_3482_);
v___x_3509_ = v_reuseFailAlloc_3538_;
goto v_reusejp_3508_;
}
v_reusejp_3508_:
{
lean_object* v___x_3510_; lean_object* v___x_3511_; lean_object* v___x_3512_; lean_object* v___x_3513_; lean_object* v_bs_x27_3514_; lean_object* v___y_3521_; lean_object* v___x_3530_; uint8_t v___x_3531_; 
lean_inc(v_a_3500_);
v___x_3510_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v___x_3510_, 0, v___x_3509_);
lean_ctor_set(v___x_3510_, 1, v_a_3500_);
lean_ctor_set_uint8(v___x_3510_, sizeof(void*)*2, v___x_3506_);
lean_ctor_set_uint8(v___x_3510_, sizeof(void*)*2 + 1, v___x_3507_);
lean_inc(v_introMVarCluster_3505_);
v___x_3511_ = lean_apply_1(v_introMVarCluster_3505_, v___x_3510_);
v___x_3512_ = lean_st_mk_ref(v___x_3511_);
v___x_3513_ = lean_unsigned_to_nat(0u);
v_bs_x27_3514_ = lean_array_uset(v_bs_3485_, v_i_3484_, v___x_3513_);
v___x_3530_ = lean_array_get_size(v_a_3500_);
v___x_3531_ = lean_nat_dec_lt(v___x_3513_, v___x_3530_);
if (v___x_3531_ == 0)
{
lean_dec(v_a_3500_);
goto v___jp_3515_;
}
else
{
lean_object* v___x_3532_; uint8_t v___x_3533_; 
v___x_3532_ = lean_box(0);
v___x_3533_ = lean_nat_dec_le(v___x_3530_, v___x_3530_);
if (v___x_3533_ == 0)
{
if (v___x_3531_ == 0)
{
lean_dec(v_a_3500_);
goto v___jp_3515_;
}
else
{
size_t v___x_3534_; lean_object* v___x_3535_; 
v___x_3534_ = lean_usize_of_nat(v___x_3530_);
lean_inc(v___x_3512_);
v___x_3535_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg(v___x_3512_, v_a_3500_, v___x_3498_, v___x_3534_, v___x_3532_);
lean_dec(v_a_3500_);
v___y_3521_ = v___x_3535_;
goto v___jp_3520_;
}
}
else
{
size_t v___x_3536_; lean_object* v___x_3537_; 
v___x_3536_ = lean_usize_of_nat(v___x_3530_);
lean_inc(v___x_3512_);
v___x_3537_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg(v___x_3512_, v_a_3500_, v___x_3498_, v___x_3536_, v___x_3532_);
lean_dec(v_a_3500_);
v___y_3521_ = v___x_3537_;
goto v___jp_3520_;
}
}
v___jp_3515_:
{
size_t v___x_3516_; size_t v___x_3517_; lean_object* v___x_3518_; 
v___x_3516_ = ((size_t)1ULL);
v___x_3517_ = lean_usize_add(v_i_3484_, v___x_3516_);
v___x_3518_ = lean_array_uset(v_bs_x27_3514_, v_i_3484_, v___x_3512_);
v_i_3484_ = v___x_3517_;
v_bs_3485_ = v___x_3518_;
goto _start;
}
v___jp_3520_:
{
if (lean_obj_tag(v___y_3521_) == 0)
{
lean_dec_ref_known(v___y_3521_, 1);
goto v___jp_3515_;
}
else
{
lean_object* v_a_3522_; lean_object* v___x_3524_; uint8_t v_isShared_3525_; uint8_t v_isSharedCheck_3529_; 
lean_dec_ref(v_bs_x27_3514_);
lean_dec(v___x_3512_);
lean_dec(v_val_3482_);
v_a_3522_ = lean_ctor_get(v___y_3521_, 0);
v_isSharedCheck_3529_ = !lean_is_exclusive(v___y_3521_);
if (v_isSharedCheck_3529_ == 0)
{
v___x_3524_ = v___y_3521_;
v_isShared_3525_ = v_isSharedCheck_3529_;
goto v_resetjp_3523_;
}
else
{
lean_inc(v_a_3522_);
lean_dec(v___y_3521_);
v___x_3524_ = lean_box(0);
v_isShared_3525_ = v_isSharedCheck_3529_;
goto v_resetjp_3523_;
}
v_resetjp_3523_:
{
lean_object* v___x_3527_; 
if (v_isShared_3525_ == 0)
{
v___x_3527_ = v___x_3524_;
goto v_reusejp_3526_;
}
else
{
lean_object* v_reuseFailAlloc_3528_; 
v_reuseFailAlloc_3528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3528_, 0, v_a_3522_);
v___x_3527_ = v_reuseFailAlloc_3528_;
goto v_reusejp_3526_;
}
v_reusejp_3526_:
{
return v___x_3527_;
}
}
}
}
}
}
}
else
{
lean_dec_ref(v_bs_3485_);
lean_dec(v_val_3482_);
return v___x_3499_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__20___boxed(lean_object* v_val_3540_, lean_object* v_sz_3541_, lean_object* v_i_3542_, lean_object* v_bs_3543_, lean_object* v___y_3544_, lean_object* v___y_3545_, lean_object* v___y_3546_, lean_object* v___y_3547_, lean_object* v___y_3548_, lean_object* v___y_3549_, lean_object* v___y_3550_, lean_object* v___y_3551_){
_start:
{
size_t v_sz_boxed_3552_; size_t v_i_boxed_3553_; lean_object* v_res_3554_; 
v_sz_boxed_3552_ = lean_unbox_usize(v_sz_3541_);
lean_dec(v_sz_3541_);
v_i_boxed_3553_ = lean_unbox_usize(v_i_3542_);
lean_dec(v_i_3542_);
v_res_3554_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__20(v_val_3540_, v_sz_boxed_3552_, v_i_boxed_3553_, v_bs_3543_, v___y_3544_, v___y_3545_, v___y_3546_, v___y_3547_, v___y_3548_, v___y_3549_, v___y_3550_);
lean_dec(v___y_3550_);
lean_dec_ref(v___y_3549_);
lean_dec(v___y_3548_);
lean_dec_ref(v___y_3547_);
lean_dec(v___y_3546_);
lean_dec(v___y_3545_);
lean_dec_ref(v___y_3544_);
return v_res_3554_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe(lean_object* v_r_3559_, lean_object* v_a_3560_, lean_object* v_a_3561_, lean_object* v_a_3562_, lean_object* v_a_3563_, lean_object* v_a_3564_, lean_object* v_a_3565_, lean_object* v_a_3566_){
_start:
{
lean_object* v_toRuleApplication_3568_; lean_object* v_parent_3569_; lean_object* v_appliedRule_3570_; double v_successProbability_3571_; lean_object* v___x_3572_; lean_object* v___x_3573_; lean_object* v_goals_3574_; lean_object* v_postState_3575_; lean_object* v_scriptSteps_x3f_3576_; lean_object* v___f_3577_; lean_object* v___f_3578_; lean_object* v___y_3580_; lean_object* v___y_3581_; lean_object* v___y_3582_; size_t v___y_3583_; lean_object* v___y_3584_; lean_object* v___y_3585_; lean_object* v___y_3586_; lean_object* v___y_3587_; lean_object* v___y_3588_; lean_object* v___y_3589_; lean_object* v___y_3590_; lean_object* v___y_3591_; size_t v___y_3592_; lean_object* v___y_3593_; lean_object* v___y_3594_; lean_object* v___y_3595_; lean_object* v___y_3596_; lean_object* v___y_3597_; lean_object* v___y_3598_; lean_object* v___y_3599_; lean_object* v___y_3600_; lean_object* v___y_3601_; lean_object* v___y_3602_; lean_object* v___y_3603_; lean_object* v___y_3604_; lean_object* v___y_3605_; lean_object* v___y_3606_; lean_object* v_auxDeclNGen_3836_; lean_object* v___y_3837_; lean_object* v___y_3838_; lean_object* v___y_3839_; lean_object* v___y_3840_; lean_object* v___y_3841_; lean_object* v___y_3842_; lean_object* v___y_3843_; 
v_toRuleApplication_3568_ = lean_ctor_get(v_r_3559_, 0);
v_parent_3569_ = lean_ctor_get(v_r_3559_, 1);
lean_inc(v_parent_3569_);
v_appliedRule_3570_ = lean_ctor_get(v_r_3559_, 2);
lean_inc_ref(v_appliedRule_3570_);
v_successProbability_3571_ = lean_ctor_get_float(v_r_3559_, sizeof(void*)*3);
v___x_3572_ = lean_st_ref_get(v_parent_3569_);
v___x_3573_ = lp_aesop_Aesop_Goal_parentRapp_x3f(v___x_3572_);
v_goals_3574_ = lean_ctor_get(v_toRuleApplication_3568_, 0);
v_postState_3575_ = lean_ctor_get(v_toRuleApplication_3568_, 1);
lean_inc_ref(v_postState_3575_);
v_scriptSteps_x3f_3576_ = lean_ctor_get(v_toRuleApplication_3568_, 2);
v___f_3577_ = ((lean_object*)(lp_aesop_Aesop_addRappUnsafe___closed__0));
v___f_3578_ = ((lean_object*)(lp_aesop_Aesop_addRappUnsafe___closed__1));
if (lean_obj_tag(v___x_3573_) == 0)
{
lean_object* v___x_3942_; lean_object* v_auxDeclNGen_3943_; lean_object* v___x_3944_; lean_object* v_fst_3945_; lean_object* v_snd_3946_; lean_object* v___x_3947_; lean_object* v_env_3948_; lean_object* v_nextMacroScope_3949_; lean_object* v_ngen_3950_; lean_object* v_traceState_3951_; lean_object* v_cache_3952_; lean_object* v_messages_3953_; lean_object* v_infoState_3954_; lean_object* v_snapshotTasks_3955_; lean_object* v___x_3957_; uint8_t v_isShared_3958_; uint8_t v_isSharedCheck_3963_; 
v___x_3942_ = lean_st_ref_get(v_a_3566_);
v_auxDeclNGen_3943_ = lean_ctor_get(v___x_3942_, 3);
lean_inc_ref(v_auxDeclNGen_3943_);
lean_dec(v___x_3942_);
v___x_3944_ = l_Lean_DeclNameGenerator_mkChild(v_auxDeclNGen_3943_);
v_fst_3945_ = lean_ctor_get(v___x_3944_, 0);
lean_inc(v_fst_3945_);
v_snd_3946_ = lean_ctor_get(v___x_3944_, 1);
lean_inc(v_snd_3946_);
lean_dec_ref(v___x_3944_);
v___x_3947_ = lean_st_ref_take(v_a_3566_);
v_env_3948_ = lean_ctor_get(v___x_3947_, 0);
v_nextMacroScope_3949_ = lean_ctor_get(v___x_3947_, 1);
v_ngen_3950_ = lean_ctor_get(v___x_3947_, 2);
v_traceState_3951_ = lean_ctor_get(v___x_3947_, 4);
v_cache_3952_ = lean_ctor_get(v___x_3947_, 5);
v_messages_3953_ = lean_ctor_get(v___x_3947_, 6);
v_infoState_3954_ = lean_ctor_get(v___x_3947_, 7);
v_snapshotTasks_3955_ = lean_ctor_get(v___x_3947_, 8);
v_isSharedCheck_3963_ = !lean_is_exclusive(v___x_3947_);
if (v_isSharedCheck_3963_ == 0)
{
lean_object* v_unused_3964_; 
v_unused_3964_ = lean_ctor_get(v___x_3947_, 3);
lean_dec(v_unused_3964_);
v___x_3957_ = v___x_3947_;
v_isShared_3958_ = v_isSharedCheck_3963_;
goto v_resetjp_3956_;
}
else
{
lean_inc(v_snapshotTasks_3955_);
lean_inc(v_infoState_3954_);
lean_inc(v_messages_3953_);
lean_inc(v_cache_3952_);
lean_inc(v_traceState_3951_);
lean_inc(v_ngen_3950_);
lean_inc(v_nextMacroScope_3949_);
lean_inc(v_env_3948_);
lean_dec(v___x_3947_);
v___x_3957_ = lean_box(0);
v_isShared_3958_ = v_isSharedCheck_3963_;
goto v_resetjp_3956_;
}
v_resetjp_3956_:
{
lean_object* v___x_3960_; 
if (v_isShared_3958_ == 0)
{
lean_ctor_set(v___x_3957_, 3, v_snd_3946_);
v___x_3960_ = v___x_3957_;
goto v_reusejp_3959_;
}
else
{
lean_object* v_reuseFailAlloc_3962_; 
v_reuseFailAlloc_3962_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3962_, 0, v_env_3948_);
lean_ctor_set(v_reuseFailAlloc_3962_, 1, v_nextMacroScope_3949_);
lean_ctor_set(v_reuseFailAlloc_3962_, 2, v_ngen_3950_);
lean_ctor_set(v_reuseFailAlloc_3962_, 3, v_snd_3946_);
lean_ctor_set(v_reuseFailAlloc_3962_, 4, v_traceState_3951_);
lean_ctor_set(v_reuseFailAlloc_3962_, 5, v_cache_3952_);
lean_ctor_set(v_reuseFailAlloc_3962_, 6, v_messages_3953_);
lean_ctor_set(v_reuseFailAlloc_3962_, 7, v_infoState_3954_);
lean_ctor_set(v_reuseFailAlloc_3962_, 8, v_snapshotTasks_3955_);
v___x_3960_ = v_reuseFailAlloc_3962_;
goto v_reusejp_3959_;
}
v_reusejp_3959_:
{
lean_object* v___x_3961_; 
v___x_3961_ = lean_st_ref_set(v_a_3566_, v___x_3960_);
v_auxDeclNGen_3836_ = v_fst_3945_;
v___y_3837_ = v_a_3560_;
v___y_3838_ = v_a_3561_;
v___y_3839_ = v_a_3562_;
v___y_3840_ = v_a_3563_;
v___y_3841_ = v_a_3564_;
v___y_3842_ = v_a_3565_;
v___y_3843_ = v_a_3566_;
goto v___jp_3835_;
}
}
}
else
{
lean_object* v_val_3965_; lean_object* v___x_3966_; 
v_val_3965_ = lean_ctor_get(v___x_3573_, 0);
lean_inc(v_val_3965_);
lean_dec_ref_known(v___x_3573_, 1);
v___x_3966_ = lp_aesop_Aesop_RappRef_getChildAuxDeclNameGenerator(v_val_3965_);
lean_dec(v_val_3965_);
v_auxDeclNGen_3836_ = v___x_3966_;
v___y_3837_ = v_a_3560_;
v___y_3838_ = v_a_3561_;
v___y_3839_ = v_a_3562_;
v___y_3840_ = v_a_3563_;
v___y_3841_ = v_a_3564_;
v___y_3842_ = v_a_3565_;
v___y_3843_ = v_a_3566_;
goto v___jp_3835_;
}
v___jp_3579_:
{
lean_object* v___x_3607_; size_t v_sz_3608_; lean_object* v___x_3609_; lean_object* v___x_3610_; lean_object* v___f_3611_; lean_object* v___x_3612_; 
v___x_3607_ = lean_mk_empty_array_with_capacity(v___y_3591_);
lean_dec(v___y_3591_);
v_sz_3608_ = lean_array_size(v___y_3594_);
v___x_3609_ = lean_box_usize(v_sz_3608_);
v___x_3610_ = lean_box_usize(v___y_3583_);
v___f_3611_ = lean_alloc_closure((void*)(lp_aesop_Aesop_addRappUnsafe___lam__2___boxed), 14, 6);
lean_closure_set(v___f_3611_, 0, v___y_3582_);
lean_closure_set(v___f_3611_, 1, v___y_3606_);
lean_closure_set(v___f_3611_, 2, v___y_3580_);
lean_closure_set(v___f_3611_, 3, v___x_3609_);
lean_closure_set(v___f_3611_, 4, v___x_3610_);
lean_closure_set(v___f_3611_, 5, v___x_3607_);
v___x_3612_ = lp_aesop_Aesop_runInMetaState___at___00Aesop_copyGoals_spec__1___redArg(v_postState_3575_, v___f_3611_, v___y_3598_, v___y_3602_, v___y_3586_, v___y_3584_, v___y_3588_, v___y_3597_, v___y_3599_);
if (lean_obj_tag(v___x_3612_) == 0)
{
lean_object* v_a_3613_; lean_object* v___x_3614_; lean_object* v___x_3615_; lean_object* v___x_3616_; lean_object* v___x_3617_; 
v_a_3613_ = lean_ctor_get(v___x_3612_, 0);
lean_inc(v_a_3613_);
lean_dec_ref_known(v___x_3612_, 1);
lean_inc_ref(v_goals_3574_);
v___x_3614_ = l_Array_append___redArg(v_goals_3574_, v___y_3603_);
lean_dec_ref(v___y_3603_);
v___x_3615_ = l_Array_append___redArg(v___x_3614_, v_a_3613_);
v___x_3616_ = lean_alloc_closure((void*)(lp_aesop_Aesop_partitionGoalsAndMVars___boxed), 8, 3);
lean_closure_set(v___x_3616_, 0, lean_box(0));
lean_closure_set(v___x_3616_, 1, v___f_3578_);
lean_closure_set(v___x_3616_, 2, v___x_3615_);
lean_inc_ref(v_postState_3575_);
v___x_3617_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_postState_3575_, v___x_3616_, v___y_3584_, v___y_3588_, v___y_3597_, v___y_3599_);
if (lean_obj_tag(v___x_3617_) == 0)
{
lean_object* v_a_3618_; lean_object* v_fst_3619_; lean_object* v___x_3621_; uint8_t v_isShared_3622_; uint8_t v_isSharedCheck_3817_; 
v_a_3618_ = lean_ctor_get(v___x_3617_, 0);
lean_inc(v_a_3618_);
lean_dec_ref_known(v___x_3617_, 1);
v_fst_3619_ = lean_ctor_get(v_a_3618_, 0);
v_isSharedCheck_3817_ = !lean_is_exclusive(v_a_3618_);
if (v_isSharedCheck_3817_ == 0)
{
lean_object* v_unused_3818_; 
v_unused_3818_ = lean_ctor_get(v_a_3618_, 1);
lean_dec(v_unused_3818_);
v___x_3621_ = v_a_3618_;
v_isShared_3622_ = v_isSharedCheck_3817_;
goto v_resetjp_3620_;
}
else
{
lean_inc(v_fst_3619_);
lean_dec(v_a_3618_);
v___x_3621_ = lean_box(0);
v_isShared_3622_ = v_isSharedCheck_3817_;
goto v_resetjp_3620_;
}
v_resetjp_3620_:
{
size_t v_sz_3623_; lean_object* v___x_3624_; 
v_sz_3623_ = lean_array_size(v_fst_3619_);
v___x_3624_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__18(v___y_3589_, v_r_3559_, v_postState_3575_, v___y_3605_, v_successProbability_3571_, v_appliedRule_3570_, v_a_3613_, v___y_3581_, v_sz_3623_, v___y_3592_, v_fst_3619_, v___y_3598_, v___y_3602_, v___y_3586_, v___y_3584_, v___y_3588_, v___y_3597_, v___y_3599_);
lean_dec_ref(v___y_3581_);
lean_dec(v_a_3613_);
lean_dec_ref(v_appliedRule_3570_);
lean_dec_ref(v_postState_3575_);
lean_dec_ref(v_r_3559_);
if (lean_obj_tag(v___x_3624_) == 0)
{
lean_object* v_a_3625_; lean_object* v___x_3626_; size_t v_sz_3627_; lean_object* v___x_3628_; 
v_a_3625_ = lean_ctor_get(v___x_3624_, 0);
lean_inc(v_a_3625_);
lean_dec_ref_known(v___x_3624_, 1);
v___x_3626_ = lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19(v___f_3577_, v_a_3625_);
v_sz_3627_ = lean_array_size(v___x_3626_);
lean_inc(v___y_3604_);
v___x_3628_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__20(v___y_3604_, v_sz_3627_, v___y_3592_, v___x_3626_, v___y_3598_, v___y_3602_, v___y_3586_, v___y_3584_, v___y_3588_, v___y_3597_, v___y_3599_);
if (lean_obj_tag(v___x_3628_) == 0)
{
lean_object* v_a_3629_; lean_object* v___x_3630_; lean_object* v_root_3631_; lean_object* v_rootMetaState_3632_; lean_object* v_numGoals_3633_; lean_object* v_numRapps_3634_; lean_object* v_nextGoalId_3635_; lean_object* v_nextRappId_3636_; lean_object* v_allIntroducedMVars_3637_; lean_object* v___x_3639_; uint8_t v_isShared_3640_; uint8_t v_isSharedCheck_3800_; 
v_a_3629_ = lean_ctor_get(v___x_3628_, 0);
lean_inc(v_a_3629_);
lean_dec_ref_known(v___x_3628_, 1);
v___x_3630_ = lean_st_ref_take(v___y_3602_);
v_root_3631_ = lean_ctor_get(v___x_3630_, 0);
v_rootMetaState_3632_ = lean_ctor_get(v___x_3630_, 1);
v_numGoals_3633_ = lean_ctor_get(v___x_3630_, 2);
v_numRapps_3634_ = lean_ctor_get(v___x_3630_, 3);
v_nextGoalId_3635_ = lean_ctor_get(v___x_3630_, 4);
v_nextRappId_3636_ = lean_ctor_get(v___x_3630_, 5);
v_allIntroducedMVars_3637_ = lean_ctor_get(v___x_3630_, 6);
v_isSharedCheck_3800_ = !lean_is_exclusive(v___x_3630_);
if (v_isSharedCheck_3800_ == 0)
{
v___x_3639_ = v___x_3630_;
v_isShared_3640_ = v_isSharedCheck_3800_;
goto v_resetjp_3638_;
}
else
{
lean_inc(v_allIntroducedMVars_3637_);
lean_inc(v_nextRappId_3636_);
lean_inc(v_nextGoalId_3635_);
lean_inc(v_numRapps_3634_);
lean_inc(v_numGoals_3633_);
lean_inc(v_rootMetaState_3632_);
lean_inc(v_root_3631_);
lean_dec(v___x_3630_);
v___x_3639_ = lean_box(0);
v_isShared_3640_ = v_isSharedCheck_3800_;
goto v_resetjp_3638_;
}
v_resetjp_3638_:
{
lean_object* v___x_3642_; 
if (v_isShared_3640_ == 0)
{
lean_ctor_set(v___x_3639_, 6, v___y_3601_);
v___x_3642_ = v___x_3639_;
goto v_reusejp_3641_;
}
else
{
lean_object* v_reuseFailAlloc_3799_; 
v_reuseFailAlloc_3799_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_3799_, 0, v_root_3631_);
lean_ctor_set(v_reuseFailAlloc_3799_, 1, v_rootMetaState_3632_);
lean_ctor_set(v_reuseFailAlloc_3799_, 2, v_numGoals_3633_);
lean_ctor_set(v_reuseFailAlloc_3799_, 3, v_numRapps_3634_);
lean_ctor_set(v_reuseFailAlloc_3799_, 4, v_nextGoalId_3635_);
lean_ctor_set(v_reuseFailAlloc_3799_, 5, v_nextRappId_3636_);
lean_ctor_set(v_reuseFailAlloc_3799_, 6, v___y_3601_);
v___x_3642_ = v_reuseFailAlloc_3799_;
goto v_reusejp_3641_;
}
v_reusejp_3641_:
{
lean_object* v___x_3643_; lean_object* v___x_3645_; 
v___x_3643_ = lean_st_ref_set(v___y_3602_, v___x_3642_);
if (v_isShared_3622_ == 0)
{
lean_ctor_set(v___x_3621_, 1, v_allIntroducedMVars_3637_);
lean_ctor_set(v___x_3621_, 0, v___y_3593_);
v___x_3645_ = v___x_3621_;
goto v_reusejp_3644_;
}
else
{
lean_object* v_reuseFailAlloc_3798_; 
v_reuseFailAlloc_3798_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3798_, 0, v___y_3593_);
lean_ctor_set(v_reuseFailAlloc_3798_, 1, v_allIntroducedMVars_3637_);
v___x_3645_ = v_reuseFailAlloc_3798_;
goto v_reusejp_3644_;
}
v_reusejp_3644_:
{
size_t v_sz_3646_; lean_object* v___x_3647_; 
v_sz_3646_ = lean_array_size(v_a_3625_);
v___x_3647_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__21(v___y_3594_, v_a_3625_, v_sz_3646_, v___y_3592_, v___x_3645_, v___y_3598_, v___y_3602_, v___y_3586_, v___y_3584_, v___y_3588_, v___y_3597_, v___y_3599_);
lean_dec_ref(v___y_3594_);
if (lean_obj_tag(v___x_3647_) == 0)
{
lean_object* v_a_3648_; lean_object* v___x_3649_; lean_object* v_fst_3650_; lean_object* v_snd_3651_; lean_object* v_root_3652_; lean_object* v_rootMetaState_3653_; lean_object* v_numGoals_3654_; lean_object* v_numRapps_3655_; lean_object* v_nextGoalId_3656_; lean_object* v_nextRappId_3657_; lean_object* v___x_3659_; uint8_t v_isShared_3660_; uint8_t v_isSharedCheck_3788_; 
v_a_3648_ = lean_ctor_get(v___x_3647_, 0);
lean_inc(v_a_3648_);
lean_dec_ref_known(v___x_3647_, 1);
v___x_3649_ = lean_st_ref_take(v___y_3602_);
v_fst_3650_ = lean_ctor_get(v_a_3648_, 0);
lean_inc(v_fst_3650_);
v_snd_3651_ = lean_ctor_get(v_a_3648_, 1);
lean_inc(v_snd_3651_);
lean_dec(v_a_3648_);
v_root_3652_ = lean_ctor_get(v___x_3649_, 0);
v_rootMetaState_3653_ = lean_ctor_get(v___x_3649_, 1);
v_numGoals_3654_ = lean_ctor_get(v___x_3649_, 2);
v_numRapps_3655_ = lean_ctor_get(v___x_3649_, 3);
v_nextGoalId_3656_ = lean_ctor_get(v___x_3649_, 4);
v_nextRappId_3657_ = lean_ctor_get(v___x_3649_, 5);
v_isSharedCheck_3788_ = !lean_is_exclusive(v___x_3649_);
if (v_isSharedCheck_3788_ == 0)
{
lean_object* v_unused_3789_; 
v_unused_3789_ = lean_ctor_get(v___x_3649_, 6);
lean_dec(v_unused_3789_);
v___x_3659_ = v___x_3649_;
v_isShared_3660_ = v_isSharedCheck_3788_;
goto v_resetjp_3658_;
}
else
{
lean_inc(v_nextRappId_3657_);
lean_inc(v_nextGoalId_3656_);
lean_inc(v_numRapps_3655_);
lean_inc(v_numGoals_3654_);
lean_inc(v_rootMetaState_3653_);
lean_inc(v_root_3652_);
lean_dec(v___x_3649_);
v___x_3659_ = lean_box(0);
v_isShared_3660_ = v_isSharedCheck_3788_;
goto v_resetjp_3658_;
}
v_resetjp_3658_:
{
lean_object* v___x_3662_; 
if (v_isShared_3660_ == 0)
{
lean_ctor_set(v___x_3659_, 6, v_snd_3651_);
v___x_3662_ = v___x_3659_;
goto v_reusejp_3661_;
}
else
{
lean_object* v_reuseFailAlloc_3787_; 
v_reuseFailAlloc_3787_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_3787_, 0, v_root_3652_);
lean_ctor_set(v_reuseFailAlloc_3787_, 1, v_rootMetaState_3653_);
lean_ctor_set(v_reuseFailAlloc_3787_, 2, v_numGoals_3654_);
lean_ctor_set(v_reuseFailAlloc_3787_, 3, v_numRapps_3655_);
lean_ctor_set(v_reuseFailAlloc_3787_, 4, v_nextGoalId_3656_);
lean_ctor_set(v_reuseFailAlloc_3787_, 5, v_nextRappId_3657_);
lean_ctor_set(v_reuseFailAlloc_3787_, 6, v_snd_3651_);
v___x_3662_ = v_reuseFailAlloc_3787_;
goto v_reusejp_3661_;
}
v_reusejp_3661_:
{
lean_object* v___x_3663_; lean_object* v___x_3664_; lean_object* v___x_3665_; lean_object* v_id_3666_; lean_object* v_parent_3667_; uint8_t v_state_3668_; uint8_t v_isIrrelevant_3669_; lean_object* v_appliedRule_3670_; lean_object* v_scriptSteps_x3f_3671_; lean_object* v_originalSubgoals_3672_; double v_successProbability_3673_; lean_object* v_metaState_3674_; lean_object* v_introducedMVars_3675_; lean_object* v_assignedMVars_3676_; lean_object* v___x_3678_; uint8_t v_isShared_3679_; uint8_t v_isSharedCheck_3785_; 
v___x_3663_ = lean_st_ref_set(v___y_3602_, v___x_3662_);
v___x_3664_ = lean_st_ref_take(v___y_3604_);
lean_inc_ref(v___y_3596_);
v___x_3665_ = lean_apply_1(v___y_3596_, v___x_3664_);
v_id_3666_ = lean_ctor_get(v___x_3665_, 0);
v_parent_3667_ = lean_ctor_get(v___x_3665_, 1);
v_state_3668_ = lean_ctor_get_uint8(v___x_3665_, sizeof(void*)*9 + 8);
v_isIrrelevant_3669_ = lean_ctor_get_uint8(v___x_3665_, sizeof(void*)*9 + 9);
v_appliedRule_3670_ = lean_ctor_get(v___x_3665_, 3);
v_scriptSteps_x3f_3671_ = lean_ctor_get(v___x_3665_, 4);
v_originalSubgoals_3672_ = lean_ctor_get(v___x_3665_, 5);
v_successProbability_3673_ = lean_ctor_get_float(v___x_3665_, sizeof(void*)*9);
v_metaState_3674_ = lean_ctor_get(v___x_3665_, 6);
v_introducedMVars_3675_ = lean_ctor_get(v___x_3665_, 7);
v_assignedMVars_3676_ = lean_ctor_get(v___x_3665_, 8);
v_isSharedCheck_3785_ = !lean_is_exclusive(v___x_3665_);
if (v_isSharedCheck_3785_ == 0)
{
lean_object* v_unused_3786_; 
v_unused_3786_ = lean_ctor_get(v___x_3665_, 2);
lean_dec(v_unused_3786_);
v___x_3678_ = v___x_3665_;
v_isShared_3679_ = v_isSharedCheck_3785_;
goto v_resetjp_3677_;
}
else
{
lean_inc(v_assignedMVars_3676_);
lean_inc(v_introducedMVars_3675_);
lean_inc(v_metaState_3674_);
lean_inc(v_originalSubgoals_3672_);
lean_inc(v_scriptSteps_x3f_3671_);
lean_inc(v_appliedRule_3670_);
lean_inc(v_parent_3667_);
lean_inc(v_id_3666_);
lean_dec(v___x_3665_);
v___x_3678_ = lean_box(0);
v_isShared_3679_ = v_isSharedCheck_3785_;
goto v_resetjp_3677_;
}
v_resetjp_3677_:
{
lean_object* v___x_3681_; 
if (v_isShared_3679_ == 0)
{
lean_ctor_set(v___x_3678_, 2, v_a_3629_);
v___x_3681_ = v___x_3678_;
goto v_reusejp_3680_;
}
else
{
lean_object* v_reuseFailAlloc_3784_; 
v_reuseFailAlloc_3784_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_3784_, 0, v_id_3666_);
lean_ctor_set(v_reuseFailAlloc_3784_, 1, v_parent_3667_);
lean_ctor_set(v_reuseFailAlloc_3784_, 2, v_a_3629_);
lean_ctor_set(v_reuseFailAlloc_3784_, 3, v_appliedRule_3670_);
lean_ctor_set(v_reuseFailAlloc_3784_, 4, v_scriptSteps_x3f_3671_);
lean_ctor_set(v_reuseFailAlloc_3784_, 5, v_originalSubgoals_3672_);
lean_ctor_set(v_reuseFailAlloc_3784_, 6, v_metaState_3674_);
lean_ctor_set(v_reuseFailAlloc_3784_, 7, v_introducedMVars_3675_);
lean_ctor_set(v_reuseFailAlloc_3784_, 8, v_assignedMVars_3676_);
lean_ctor_set_uint8(v_reuseFailAlloc_3784_, sizeof(void*)*9 + 8, v_state_3668_);
lean_ctor_set_uint8(v_reuseFailAlloc_3784_, sizeof(void*)*9 + 9, v_isIrrelevant_3669_);
lean_ctor_set_float(v_reuseFailAlloc_3784_, sizeof(void*)*9, v_successProbability_3673_);
v___x_3681_ = v_reuseFailAlloc_3784_;
goto v_reusejp_3680_;
}
v_reusejp_3680_:
{
lean_object* v___x_3682_; lean_object* v___x_3683_; lean_object* v_id_3684_; lean_object* v_parent_3685_; lean_object* v_children_3686_; uint8_t v_state_3687_; uint8_t v_isIrrelevant_3688_; lean_object* v_appliedRule_3689_; lean_object* v_scriptSteps_x3f_3690_; lean_object* v_originalSubgoals_3691_; double v_successProbability_3692_; lean_object* v_metaState_3693_; lean_object* v_assignedMVars_3694_; lean_object* v___x_3696_; uint8_t v_isShared_3697_; uint8_t v_isSharedCheck_3782_; 
lean_inc(v___y_3590_);
v___x_3682_ = lean_apply_1(v___y_3590_, v___x_3681_);
lean_inc_ref(v___y_3596_);
v___x_3683_ = lean_apply_1(v___y_3596_, v___x_3682_);
v_id_3684_ = lean_ctor_get(v___x_3683_, 0);
v_parent_3685_ = lean_ctor_get(v___x_3683_, 1);
v_children_3686_ = lean_ctor_get(v___x_3683_, 2);
v_state_3687_ = lean_ctor_get_uint8(v___x_3683_, sizeof(void*)*9 + 8);
v_isIrrelevant_3688_ = lean_ctor_get_uint8(v___x_3683_, sizeof(void*)*9 + 9);
v_appliedRule_3689_ = lean_ctor_get(v___x_3683_, 3);
v_scriptSteps_x3f_3690_ = lean_ctor_get(v___x_3683_, 4);
v_originalSubgoals_3691_ = lean_ctor_get(v___x_3683_, 5);
v_successProbability_3692_ = lean_ctor_get_float(v___x_3683_, sizeof(void*)*9);
v_metaState_3693_ = lean_ctor_get(v___x_3683_, 6);
v_assignedMVars_3694_ = lean_ctor_get(v___x_3683_, 8);
v_isSharedCheck_3782_ = !lean_is_exclusive(v___x_3683_);
if (v_isSharedCheck_3782_ == 0)
{
lean_object* v_unused_3783_; 
v_unused_3783_ = lean_ctor_get(v___x_3683_, 7);
lean_dec(v_unused_3783_);
v___x_3696_ = v___x_3683_;
v_isShared_3697_ = v_isSharedCheck_3782_;
goto v_resetjp_3695_;
}
else
{
lean_inc(v_assignedMVars_3694_);
lean_inc(v_metaState_3693_);
lean_inc(v_originalSubgoals_3691_);
lean_inc(v_scriptSteps_x3f_3690_);
lean_inc(v_appliedRule_3689_);
lean_inc(v_children_3686_);
lean_inc(v_parent_3685_);
lean_inc(v_id_3684_);
lean_dec(v___x_3683_);
v___x_3696_ = lean_box(0);
v_isShared_3697_ = v_isSharedCheck_3782_;
goto v_resetjp_3695_;
}
v_resetjp_3695_:
{
lean_object* v___x_3699_; 
if (v_isShared_3697_ == 0)
{
lean_ctor_set(v___x_3696_, 7, v_fst_3650_);
v___x_3699_ = v___x_3696_;
goto v_reusejp_3698_;
}
else
{
lean_object* v_reuseFailAlloc_3781_; 
v_reuseFailAlloc_3781_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_3781_, 0, v_id_3684_);
lean_ctor_set(v_reuseFailAlloc_3781_, 1, v_parent_3685_);
lean_ctor_set(v_reuseFailAlloc_3781_, 2, v_children_3686_);
lean_ctor_set(v_reuseFailAlloc_3781_, 3, v_appliedRule_3689_);
lean_ctor_set(v_reuseFailAlloc_3781_, 4, v_scriptSteps_x3f_3690_);
lean_ctor_set(v_reuseFailAlloc_3781_, 5, v_originalSubgoals_3691_);
lean_ctor_set(v_reuseFailAlloc_3781_, 6, v_metaState_3693_);
lean_ctor_set(v_reuseFailAlloc_3781_, 7, v_fst_3650_);
lean_ctor_set(v_reuseFailAlloc_3781_, 8, v_assignedMVars_3694_);
lean_ctor_set_uint8(v_reuseFailAlloc_3781_, sizeof(void*)*9 + 8, v_state_3687_);
lean_ctor_set_uint8(v_reuseFailAlloc_3781_, sizeof(void*)*9 + 9, v_isIrrelevant_3688_);
lean_ctor_set_float(v_reuseFailAlloc_3781_, sizeof(void*)*9, v_successProbability_3692_);
v___x_3699_ = v_reuseFailAlloc_3781_;
goto v_reusejp_3698_;
}
v_reusejp_3698_:
{
lean_object* v___x_3700_; lean_object* v___x_3701_; lean_object* v_id_3702_; lean_object* v_parent_3703_; lean_object* v_children_3704_; uint8_t v_state_3705_; uint8_t v_isIrrelevant_3706_; lean_object* v_appliedRule_3707_; lean_object* v_scriptSteps_x3f_3708_; lean_object* v_originalSubgoals_3709_; double v_successProbability_3710_; lean_object* v_metaState_3711_; lean_object* v_introducedMVars_3712_; lean_object* v___x_3714_; uint8_t v_isShared_3715_; uint8_t v_isSharedCheck_3779_; 
lean_inc(v___y_3590_);
v___x_3700_ = lean_apply_1(v___y_3590_, v___x_3699_);
v___x_3701_ = lean_apply_1(v___y_3596_, v___x_3700_);
v_id_3702_ = lean_ctor_get(v___x_3701_, 0);
v_parent_3703_ = lean_ctor_get(v___x_3701_, 1);
v_children_3704_ = lean_ctor_get(v___x_3701_, 2);
v_state_3705_ = lean_ctor_get_uint8(v___x_3701_, sizeof(void*)*9 + 8);
v_isIrrelevant_3706_ = lean_ctor_get_uint8(v___x_3701_, sizeof(void*)*9 + 9);
v_appliedRule_3707_ = lean_ctor_get(v___x_3701_, 3);
v_scriptSteps_x3f_3708_ = lean_ctor_get(v___x_3701_, 4);
v_originalSubgoals_3709_ = lean_ctor_get(v___x_3701_, 5);
v_successProbability_3710_ = lean_ctor_get_float(v___x_3701_, sizeof(void*)*9);
v_metaState_3711_ = lean_ctor_get(v___x_3701_, 6);
v_introducedMVars_3712_ = lean_ctor_get(v___x_3701_, 7);
v_isSharedCheck_3779_ = !lean_is_exclusive(v___x_3701_);
if (v_isSharedCheck_3779_ == 0)
{
lean_object* v_unused_3780_; 
v_unused_3780_ = lean_ctor_get(v___x_3701_, 8);
lean_dec(v_unused_3780_);
v___x_3714_ = v___x_3701_;
v_isShared_3715_ = v_isSharedCheck_3779_;
goto v_resetjp_3713_;
}
else
{
lean_inc(v_introducedMVars_3712_);
lean_inc(v_metaState_3711_);
lean_inc(v_originalSubgoals_3709_);
lean_inc(v_scriptSteps_x3f_3708_);
lean_inc(v_appliedRule_3707_);
lean_inc(v_children_3704_);
lean_inc(v_parent_3703_);
lean_inc(v_id_3702_);
lean_dec(v___x_3701_);
v___x_3714_ = lean_box(0);
v_isShared_3715_ = v_isSharedCheck_3779_;
goto v_resetjp_3713_;
}
v_resetjp_3713_:
{
lean_object* v___x_3717_; 
if (v_isShared_3715_ == 0)
{
lean_ctor_set(v___x_3714_, 8, v___y_3600_);
v___x_3717_ = v___x_3714_;
goto v_reusejp_3716_;
}
else
{
lean_object* v_reuseFailAlloc_3778_; 
v_reuseFailAlloc_3778_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_3778_, 0, v_id_3702_);
lean_ctor_set(v_reuseFailAlloc_3778_, 1, v_parent_3703_);
lean_ctor_set(v_reuseFailAlloc_3778_, 2, v_children_3704_);
lean_ctor_set(v_reuseFailAlloc_3778_, 3, v_appliedRule_3707_);
lean_ctor_set(v_reuseFailAlloc_3778_, 4, v_scriptSteps_x3f_3708_);
lean_ctor_set(v_reuseFailAlloc_3778_, 5, v_originalSubgoals_3709_);
lean_ctor_set(v_reuseFailAlloc_3778_, 6, v_metaState_3711_);
lean_ctor_set(v_reuseFailAlloc_3778_, 7, v_introducedMVars_3712_);
lean_ctor_set(v_reuseFailAlloc_3778_, 8, v___y_3600_);
lean_ctor_set_uint8(v_reuseFailAlloc_3778_, sizeof(void*)*9 + 8, v_state_3705_);
lean_ctor_set_uint8(v_reuseFailAlloc_3778_, sizeof(void*)*9 + 9, v_isIrrelevant_3706_);
lean_ctor_set_float(v_reuseFailAlloc_3778_, sizeof(void*)*9, v_successProbability_3710_);
v___x_3717_ = v_reuseFailAlloc_3778_;
goto v_reusejp_3716_;
}
v_reusejp_3716_:
{
lean_object* v___x_3718_; lean_object* v___x_3719_; lean_object* v___x_3720_; lean_object* v___x_3721_; lean_object* v_id_3722_; lean_object* v_parent_3723_; lean_object* v_children_3724_; lean_object* v_origin_3725_; lean_object* v_depth_3726_; uint8_t v_state_3727_; uint8_t v_isIrrelevant_3728_; uint8_t v_isForcedUnprovable_3729_; lean_object* v_preNormGoal_3730_; lean_object* v_normalizationState_3731_; lean_object* v_mvars_3732_; lean_object* v_forwardState_3733_; lean_object* v_forwardRuleMatches_3734_; double v_successProbability_3735_; lean_object* v_addedInIteration_3736_; lean_object* v_lastExpandedInIteration_3737_; uint8_t v_unsafeRulesSelected_3738_; lean_object* v_unsafeQueue_3739_; lean_object* v_failedRapps_3740_; lean_object* v___x_3742_; uint8_t v_isShared_3743_; uint8_t v_isSharedCheck_3777_; 
v___x_3718_ = lean_apply_1(v___y_3590_, v___x_3717_);
v___x_3719_ = lean_st_ref_set(v___y_3604_, v___x_3718_);
v___x_3720_ = lean_st_ref_take(v_parent_3569_);
v___x_3721_ = lean_apply_1(v___y_3585_, v___x_3720_);
v_id_3722_ = lean_ctor_get(v___x_3721_, 0);
v_parent_3723_ = lean_ctor_get(v___x_3721_, 1);
v_children_3724_ = lean_ctor_get(v___x_3721_, 2);
v_origin_3725_ = lean_ctor_get(v___x_3721_, 3);
v_depth_3726_ = lean_ctor_get(v___x_3721_, 4);
v_state_3727_ = lean_ctor_get_uint8(v___x_3721_, sizeof(void*)*14 + 8);
v_isIrrelevant_3728_ = lean_ctor_get_uint8(v___x_3721_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_3729_ = lean_ctor_get_uint8(v___x_3721_, sizeof(void*)*14 + 10);
v_preNormGoal_3730_ = lean_ctor_get(v___x_3721_, 5);
v_normalizationState_3731_ = lean_ctor_get(v___x_3721_, 6);
v_mvars_3732_ = lean_ctor_get(v___x_3721_, 7);
v_forwardState_3733_ = lean_ctor_get(v___x_3721_, 8);
v_forwardRuleMatches_3734_ = lean_ctor_get(v___x_3721_, 9);
v_successProbability_3735_ = lean_ctor_get_float(v___x_3721_, sizeof(void*)*14);
v_addedInIteration_3736_ = lean_ctor_get(v___x_3721_, 10);
v_lastExpandedInIteration_3737_ = lean_ctor_get(v___x_3721_, 11);
v_unsafeRulesSelected_3738_ = lean_ctor_get_uint8(v___x_3721_, sizeof(void*)*14 + 11);
v_unsafeQueue_3739_ = lean_ctor_get(v___x_3721_, 12);
v_failedRapps_3740_ = lean_ctor_get(v___x_3721_, 13);
v_isSharedCheck_3777_ = !lean_is_exclusive(v___x_3721_);
if (v_isSharedCheck_3777_ == 0)
{
v___x_3742_ = v___x_3721_;
v_isShared_3743_ = v_isSharedCheck_3777_;
goto v_resetjp_3741_;
}
else
{
lean_inc(v_failedRapps_3740_);
lean_inc(v_unsafeQueue_3739_);
lean_inc(v_lastExpandedInIteration_3737_);
lean_inc(v_addedInIteration_3736_);
lean_inc(v_forwardRuleMatches_3734_);
lean_inc(v_forwardState_3733_);
lean_inc(v_mvars_3732_);
lean_inc(v_normalizationState_3731_);
lean_inc(v_preNormGoal_3730_);
lean_inc(v_depth_3726_);
lean_inc(v_origin_3725_);
lean_inc(v_children_3724_);
lean_inc(v_parent_3723_);
lean_inc(v_id_3722_);
lean_dec(v___x_3721_);
v___x_3742_ = lean_box(0);
v_isShared_3743_ = v_isSharedCheck_3777_;
goto v_resetjp_3741_;
}
v_resetjp_3741_:
{
lean_object* v___x_3744_; lean_object* v___x_3746_; 
lean_inc(v___y_3604_);
v___x_3744_ = lean_array_push(v_children_3724_, v___y_3604_);
if (v_isShared_3743_ == 0)
{
lean_ctor_set(v___x_3742_, 2, v___x_3744_);
v___x_3746_ = v___x_3742_;
goto v_reusejp_3745_;
}
else
{
lean_object* v_reuseFailAlloc_3776_; 
v_reuseFailAlloc_3776_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_3776_, 0, v_id_3722_);
lean_ctor_set(v_reuseFailAlloc_3776_, 1, v_parent_3723_);
lean_ctor_set(v_reuseFailAlloc_3776_, 2, v___x_3744_);
lean_ctor_set(v_reuseFailAlloc_3776_, 3, v_origin_3725_);
lean_ctor_set(v_reuseFailAlloc_3776_, 4, v_depth_3726_);
lean_ctor_set(v_reuseFailAlloc_3776_, 5, v_preNormGoal_3730_);
lean_ctor_set(v_reuseFailAlloc_3776_, 6, v_normalizationState_3731_);
lean_ctor_set(v_reuseFailAlloc_3776_, 7, v_mvars_3732_);
lean_ctor_set(v_reuseFailAlloc_3776_, 8, v_forwardState_3733_);
lean_ctor_set(v_reuseFailAlloc_3776_, 9, v_forwardRuleMatches_3734_);
lean_ctor_set(v_reuseFailAlloc_3776_, 10, v_addedInIteration_3736_);
lean_ctor_set(v_reuseFailAlloc_3776_, 11, v_lastExpandedInIteration_3737_);
lean_ctor_set(v_reuseFailAlloc_3776_, 12, v_unsafeQueue_3739_);
lean_ctor_set(v_reuseFailAlloc_3776_, 13, v_failedRapps_3740_);
lean_ctor_set_uint8(v_reuseFailAlloc_3776_, sizeof(void*)*14 + 8, v_state_3727_);
lean_ctor_set_uint8(v_reuseFailAlloc_3776_, sizeof(void*)*14 + 9, v_isIrrelevant_3728_);
lean_ctor_set_uint8(v_reuseFailAlloc_3776_, sizeof(void*)*14 + 10, v_isForcedUnprovable_3729_);
lean_ctor_set_float(v_reuseFailAlloc_3776_, sizeof(void*)*14, v_successProbability_3735_);
lean_ctor_set_uint8(v_reuseFailAlloc_3776_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_3738_);
v___x_3746_ = v_reuseFailAlloc_3776_;
goto v_reusejp_3745_;
}
v_reusejp_3745_:
{
lean_object* v___x_3747_; lean_object* v___x_3748_; lean_object* v___x_3749_; lean_object* v___x_3750_; 
v___x_3747_ = lean_apply_1(v___y_3595_, v___x_3746_);
v___x_3748_ = lean_st_ref_set(v_parent_3569_, v___x_3747_);
lean_dec(v_parent_3569_);
v___x_3749_ = lean_array_get_size(v_a_3625_);
lean_dec(v_a_3625_);
v___x_3750_ = lp_aesop_Aesop_incrementNumGoals___redArg(v___x_3749_, v___y_3602_);
if (lean_obj_tag(v___x_3750_) == 0)
{
lean_object* v___x_3751_; 
lean_dec_ref_known(v___x_3750_, 1);
v___x_3751_ = lp_aesop_Aesop_incrementNumRapps___redArg(v___y_3587_, v___y_3602_);
if (lean_obj_tag(v___x_3751_) == 0)
{
lean_object* v___x_3753_; uint8_t v_isShared_3754_; uint8_t v_isSharedCheck_3758_; 
v_isSharedCheck_3758_ = !lean_is_exclusive(v___x_3751_);
if (v_isSharedCheck_3758_ == 0)
{
lean_object* v_unused_3759_; 
v_unused_3759_ = lean_ctor_get(v___x_3751_, 0);
lean_dec(v_unused_3759_);
v___x_3753_ = v___x_3751_;
v_isShared_3754_ = v_isSharedCheck_3758_;
goto v_resetjp_3752_;
}
else
{
lean_dec(v___x_3751_);
v___x_3753_ = lean_box(0);
v_isShared_3754_ = v_isSharedCheck_3758_;
goto v_resetjp_3752_;
}
v_resetjp_3752_:
{
lean_object* v___x_3756_; 
if (v_isShared_3754_ == 0)
{
lean_ctor_set(v___x_3753_, 0, v___y_3604_);
v___x_3756_ = v___x_3753_;
goto v_reusejp_3755_;
}
else
{
lean_object* v_reuseFailAlloc_3757_; 
v_reuseFailAlloc_3757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3757_, 0, v___y_3604_);
v___x_3756_ = v_reuseFailAlloc_3757_;
goto v_reusejp_3755_;
}
v_reusejp_3755_:
{
return v___x_3756_;
}
}
}
else
{
lean_object* v_a_3760_; lean_object* v___x_3762_; uint8_t v_isShared_3763_; uint8_t v_isSharedCheck_3767_; 
lean_dec(v___y_3604_);
v_a_3760_ = lean_ctor_get(v___x_3751_, 0);
v_isSharedCheck_3767_ = !lean_is_exclusive(v___x_3751_);
if (v_isSharedCheck_3767_ == 0)
{
v___x_3762_ = v___x_3751_;
v_isShared_3763_ = v_isSharedCheck_3767_;
goto v_resetjp_3761_;
}
else
{
lean_inc(v_a_3760_);
lean_dec(v___x_3751_);
v___x_3762_ = lean_box(0);
v_isShared_3763_ = v_isSharedCheck_3767_;
goto v_resetjp_3761_;
}
v_resetjp_3761_:
{
lean_object* v___x_3765_; 
if (v_isShared_3763_ == 0)
{
v___x_3765_ = v___x_3762_;
goto v_reusejp_3764_;
}
else
{
lean_object* v_reuseFailAlloc_3766_; 
v_reuseFailAlloc_3766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3766_, 0, v_a_3760_);
v___x_3765_ = v_reuseFailAlloc_3766_;
goto v_reusejp_3764_;
}
v_reusejp_3764_:
{
return v___x_3765_;
}
}
}
}
else
{
lean_object* v_a_3768_; lean_object* v___x_3770_; uint8_t v_isShared_3771_; uint8_t v_isSharedCheck_3775_; 
lean_dec(v___y_3604_);
v_a_3768_ = lean_ctor_get(v___x_3750_, 0);
v_isSharedCheck_3775_ = !lean_is_exclusive(v___x_3750_);
if (v_isSharedCheck_3775_ == 0)
{
v___x_3770_ = v___x_3750_;
v_isShared_3771_ = v_isSharedCheck_3775_;
goto v_resetjp_3769_;
}
else
{
lean_inc(v_a_3768_);
lean_dec(v___x_3750_);
v___x_3770_ = lean_box(0);
v_isShared_3771_ = v_isSharedCheck_3775_;
goto v_resetjp_3769_;
}
v_resetjp_3769_:
{
lean_object* v___x_3773_; 
if (v_isShared_3771_ == 0)
{
v___x_3773_ = v___x_3770_;
goto v_reusejp_3772_;
}
else
{
lean_object* v_reuseFailAlloc_3774_; 
v_reuseFailAlloc_3774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3774_, 0, v_a_3768_);
v___x_3773_ = v_reuseFailAlloc_3774_;
goto v_reusejp_3772_;
}
v_reusejp_3772_:
{
return v___x_3773_;
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
}
}
}
else
{
lean_object* v_a_3790_; lean_object* v___x_3792_; uint8_t v_isShared_3793_; uint8_t v_isSharedCheck_3797_; 
lean_dec(v_a_3629_);
lean_dec(v_a_3625_);
lean_dec(v___y_3604_);
lean_dec_ref(v___y_3600_);
lean_dec_ref(v___y_3596_);
lean_dec(v___y_3595_);
lean_dec(v___y_3590_);
lean_dec_ref(v___y_3585_);
lean_dec(v_parent_3569_);
v_a_3790_ = lean_ctor_get(v___x_3647_, 0);
v_isSharedCheck_3797_ = !lean_is_exclusive(v___x_3647_);
if (v_isSharedCheck_3797_ == 0)
{
v___x_3792_ = v___x_3647_;
v_isShared_3793_ = v_isSharedCheck_3797_;
goto v_resetjp_3791_;
}
else
{
lean_inc(v_a_3790_);
lean_dec(v___x_3647_);
v___x_3792_ = lean_box(0);
v_isShared_3793_ = v_isSharedCheck_3797_;
goto v_resetjp_3791_;
}
v_resetjp_3791_:
{
lean_object* v___x_3795_; 
if (v_isShared_3793_ == 0)
{
v___x_3795_ = v___x_3792_;
goto v_reusejp_3794_;
}
else
{
lean_object* v_reuseFailAlloc_3796_; 
v_reuseFailAlloc_3796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3796_, 0, v_a_3790_);
v___x_3795_ = v_reuseFailAlloc_3796_;
goto v_reusejp_3794_;
}
v_reusejp_3794_:
{
return v___x_3795_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3801_; lean_object* v___x_3803_; uint8_t v_isShared_3804_; uint8_t v_isSharedCheck_3808_; 
lean_dec(v_a_3625_);
lean_del_object(v___x_3621_);
lean_dec(v___y_3604_);
lean_dec_ref(v___y_3601_);
lean_dec_ref(v___y_3600_);
lean_dec_ref(v___y_3596_);
lean_dec(v___y_3595_);
lean_dec_ref(v___y_3594_);
lean_dec_ref(v___y_3593_);
lean_dec(v___y_3590_);
lean_dec_ref(v___y_3585_);
lean_dec(v_parent_3569_);
v_a_3801_ = lean_ctor_get(v___x_3628_, 0);
v_isSharedCheck_3808_ = !lean_is_exclusive(v___x_3628_);
if (v_isSharedCheck_3808_ == 0)
{
v___x_3803_ = v___x_3628_;
v_isShared_3804_ = v_isSharedCheck_3808_;
goto v_resetjp_3802_;
}
else
{
lean_inc(v_a_3801_);
lean_dec(v___x_3628_);
v___x_3803_ = lean_box(0);
v_isShared_3804_ = v_isSharedCheck_3808_;
goto v_resetjp_3802_;
}
v_resetjp_3802_:
{
lean_object* v___x_3806_; 
if (v_isShared_3804_ == 0)
{
v___x_3806_ = v___x_3803_;
goto v_reusejp_3805_;
}
else
{
lean_object* v_reuseFailAlloc_3807_; 
v_reuseFailAlloc_3807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3807_, 0, v_a_3801_);
v___x_3806_ = v_reuseFailAlloc_3807_;
goto v_reusejp_3805_;
}
v_reusejp_3805_:
{
return v___x_3806_;
}
}
}
}
else
{
lean_object* v_a_3809_; lean_object* v___x_3811_; uint8_t v_isShared_3812_; uint8_t v_isSharedCheck_3816_; 
lean_del_object(v___x_3621_);
lean_dec(v___y_3604_);
lean_dec_ref(v___y_3601_);
lean_dec_ref(v___y_3600_);
lean_dec_ref(v___y_3596_);
lean_dec(v___y_3595_);
lean_dec_ref(v___y_3594_);
lean_dec_ref(v___y_3593_);
lean_dec(v___y_3590_);
lean_dec_ref(v___y_3585_);
lean_dec(v_parent_3569_);
v_a_3809_ = lean_ctor_get(v___x_3624_, 0);
v_isSharedCheck_3816_ = !lean_is_exclusive(v___x_3624_);
if (v_isSharedCheck_3816_ == 0)
{
v___x_3811_ = v___x_3624_;
v_isShared_3812_ = v_isSharedCheck_3816_;
goto v_resetjp_3810_;
}
else
{
lean_inc(v_a_3809_);
lean_dec(v___x_3624_);
v___x_3811_ = lean_box(0);
v_isShared_3812_ = v_isSharedCheck_3816_;
goto v_resetjp_3810_;
}
v_resetjp_3810_:
{
lean_object* v___x_3814_; 
if (v_isShared_3812_ == 0)
{
v___x_3814_ = v___x_3811_;
goto v_reusejp_3813_;
}
else
{
lean_object* v_reuseFailAlloc_3815_; 
v_reuseFailAlloc_3815_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3815_, 0, v_a_3809_);
v___x_3814_ = v_reuseFailAlloc_3815_;
goto v_reusejp_3813_;
}
v_reusejp_3813_:
{
return v___x_3814_;
}
}
}
}
}
else
{
lean_object* v_a_3819_; lean_object* v___x_3821_; uint8_t v_isShared_3822_; uint8_t v_isSharedCheck_3826_; 
lean_dec(v_a_3613_);
lean_dec(v___y_3605_);
lean_dec(v___y_3604_);
lean_dec_ref(v___y_3601_);
lean_dec_ref(v___y_3600_);
lean_dec_ref(v___y_3596_);
lean_dec(v___y_3595_);
lean_dec_ref(v___y_3594_);
lean_dec_ref(v___y_3593_);
lean_dec(v___y_3590_);
lean_dec(v___y_3589_);
lean_dec_ref(v___y_3585_);
lean_dec_ref(v___y_3581_);
lean_dec_ref(v_postState_3575_);
lean_dec_ref(v_appliedRule_3570_);
lean_dec(v_parent_3569_);
lean_dec_ref(v_r_3559_);
v_a_3819_ = lean_ctor_get(v___x_3617_, 0);
v_isSharedCheck_3826_ = !lean_is_exclusive(v___x_3617_);
if (v_isSharedCheck_3826_ == 0)
{
v___x_3821_ = v___x_3617_;
v_isShared_3822_ = v_isSharedCheck_3826_;
goto v_resetjp_3820_;
}
else
{
lean_inc(v_a_3819_);
lean_dec(v___x_3617_);
v___x_3821_ = lean_box(0);
v_isShared_3822_ = v_isSharedCheck_3826_;
goto v_resetjp_3820_;
}
v_resetjp_3820_:
{
lean_object* v___x_3824_; 
if (v_isShared_3822_ == 0)
{
v___x_3824_ = v___x_3821_;
goto v_reusejp_3823_;
}
else
{
lean_object* v_reuseFailAlloc_3825_; 
v_reuseFailAlloc_3825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3825_, 0, v_a_3819_);
v___x_3824_ = v_reuseFailAlloc_3825_;
goto v_reusejp_3823_;
}
v_reusejp_3823_:
{
return v___x_3824_;
}
}
}
}
else
{
lean_object* v_a_3827_; lean_object* v___x_3829_; uint8_t v_isShared_3830_; uint8_t v_isSharedCheck_3834_; 
lean_dec(v___y_3605_);
lean_dec(v___y_3604_);
lean_dec_ref(v___y_3603_);
lean_dec_ref(v___y_3601_);
lean_dec_ref(v___y_3600_);
lean_dec_ref(v___y_3596_);
lean_dec(v___y_3595_);
lean_dec_ref(v___y_3594_);
lean_dec_ref(v___y_3593_);
lean_dec(v___y_3590_);
lean_dec(v___y_3589_);
lean_dec_ref(v___y_3585_);
lean_dec_ref(v___y_3581_);
lean_dec_ref(v_postState_3575_);
lean_dec_ref(v_appliedRule_3570_);
lean_dec(v_parent_3569_);
lean_dec_ref(v_r_3559_);
v_a_3827_ = lean_ctor_get(v___x_3612_, 0);
v_isSharedCheck_3834_ = !lean_is_exclusive(v___x_3612_);
if (v_isSharedCheck_3834_ == 0)
{
v___x_3829_ = v___x_3612_;
v_isShared_3830_ = v_isSharedCheck_3834_;
goto v_resetjp_3828_;
}
else
{
lean_inc(v_a_3827_);
lean_dec(v___x_3612_);
v___x_3829_ = lean_box(0);
v_isShared_3830_ = v_isSharedCheck_3834_;
goto v_resetjp_3828_;
}
v_resetjp_3828_:
{
lean_object* v___x_3832_; 
if (v_isShared_3830_ == 0)
{
v___x_3832_ = v___x_3829_;
goto v_reusejp_3831_;
}
else
{
lean_object* v_reuseFailAlloc_3833_; 
v_reuseFailAlloc_3833_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3833_, 0, v_a_3827_);
v___x_3832_ = v_reuseFailAlloc_3833_;
goto v_reusejp_3831_;
}
v_reusejp_3831_:
{
return v___x_3832_;
}
}
}
}
v___jp_3835_:
{
lean_object* v___x_3844_; 
v___x_3844_ = lp_aesop_Aesop_getAndIncrementNextRappId___redArg(v___y_3838_);
if (lean_obj_tag(v___x_3844_) == 0)
{
lean_object* v_core_3845_; lean_object* v_toState_3846_; lean_object* v_a_3847_; lean_object* v_meta_3848_; lean_object* v_passedHeartbeats_3849_; lean_object* v___x_3851_; uint8_t v_isShared_3852_; uint8_t v_isSharedCheck_3932_; 
v_core_3845_ = lean_ctor_get(v_postState_3575_, 0);
lean_inc_ref(v_core_3845_);
v_toState_3846_ = lean_ctor_get(v_core_3845_, 0);
lean_inc_ref(v_toState_3846_);
v_a_3847_ = lean_ctor_get(v___x_3844_, 0);
lean_inc(v_a_3847_);
lean_dec_ref_known(v___x_3844_, 1);
v_meta_3848_ = lean_ctor_get(v_postState_3575_, 1);
v_passedHeartbeats_3849_ = lean_ctor_get(v_core_3845_, 1);
v_isSharedCheck_3932_ = !lean_is_exclusive(v_core_3845_);
if (v_isSharedCheck_3932_ == 0)
{
lean_object* v_unused_3933_; 
v_unused_3933_ = lean_ctor_get(v_core_3845_, 0);
lean_dec(v_unused_3933_);
v___x_3851_ = v_core_3845_;
v_isShared_3852_ = v_isSharedCheck_3932_;
goto v_resetjp_3850_;
}
else
{
lean_inc(v_passedHeartbeats_3849_);
lean_dec(v_core_3845_);
v___x_3851_ = lean_box(0);
v_isShared_3852_ = v_isSharedCheck_3932_;
goto v_resetjp_3850_;
}
v_resetjp_3850_:
{
lean_object* v_env_3853_; lean_object* v_nextMacroScope_3854_; lean_object* v_ngen_3855_; lean_object* v_traceState_3856_; lean_object* v_cache_3857_; lean_object* v_messages_3858_; lean_object* v_infoState_3859_; lean_object* v_snapshotTasks_3860_; lean_object* v___x_3862_; uint8_t v_isShared_3863_; uint8_t v_isSharedCheck_3930_; 
v_env_3853_ = lean_ctor_get(v_toState_3846_, 0);
v_nextMacroScope_3854_ = lean_ctor_get(v_toState_3846_, 1);
v_ngen_3855_ = lean_ctor_get(v_toState_3846_, 2);
v_traceState_3856_ = lean_ctor_get(v_toState_3846_, 4);
v_cache_3857_ = lean_ctor_get(v_toState_3846_, 5);
v_messages_3858_ = lean_ctor_get(v_toState_3846_, 6);
v_infoState_3859_ = lean_ctor_get(v_toState_3846_, 7);
v_snapshotTasks_3860_ = lean_ctor_get(v_toState_3846_, 8);
v_isSharedCheck_3930_ = !lean_is_exclusive(v_toState_3846_);
if (v_isSharedCheck_3930_ == 0)
{
lean_object* v_unused_3931_; 
v_unused_3931_ = lean_ctor_get(v_toState_3846_, 3);
lean_dec(v_unused_3931_);
v___x_3862_ = v_toState_3846_;
v_isShared_3863_ = v_isSharedCheck_3930_;
goto v_resetjp_3861_;
}
else
{
lean_inc(v_snapshotTasks_3860_);
lean_inc(v_infoState_3859_);
lean_inc(v_messages_3858_);
lean_inc(v_cache_3857_);
lean_inc(v_traceState_3856_);
lean_inc(v_ngen_3855_);
lean_inc(v_nextMacroScope_3854_);
lean_inc(v_env_3853_);
lean_dec(v_toState_3846_);
v___x_3862_ = lean_box(0);
v_isShared_3863_ = v_isSharedCheck_3930_;
goto v_resetjp_3861_;
}
v_resetjp_3861_:
{
lean_object* v___x_3865_; 
if (v_isShared_3863_ == 0)
{
lean_ctor_set(v___x_3862_, 3, v_auxDeclNGen_3836_);
v___x_3865_ = v___x_3862_;
goto v_reusejp_3864_;
}
else
{
lean_object* v_reuseFailAlloc_3929_; 
v_reuseFailAlloc_3929_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3929_, 0, v_env_3853_);
lean_ctor_set(v_reuseFailAlloc_3929_, 1, v_nextMacroScope_3854_);
lean_ctor_set(v_reuseFailAlloc_3929_, 2, v_ngen_3855_);
lean_ctor_set(v_reuseFailAlloc_3929_, 3, v_auxDeclNGen_3836_);
lean_ctor_set(v_reuseFailAlloc_3929_, 4, v_traceState_3856_);
lean_ctor_set(v_reuseFailAlloc_3929_, 5, v_cache_3857_);
lean_ctor_set(v_reuseFailAlloc_3929_, 6, v_messages_3858_);
lean_ctor_set(v_reuseFailAlloc_3929_, 7, v_infoState_3859_);
lean_ctor_set(v_reuseFailAlloc_3929_, 8, v_snapshotTasks_3860_);
v___x_3865_ = v_reuseFailAlloc_3929_;
goto v_reusejp_3864_;
}
v_reusejp_3864_:
{
lean_object* v___x_3867_; 
if (v_isShared_3852_ == 0)
{
lean_ctor_set(v___x_3851_, 0, v___x_3865_);
v___x_3867_ = v___x_3851_;
goto v_reusejp_3866_;
}
else
{
lean_object* v_reuseFailAlloc_3928_; 
v_reuseFailAlloc_3928_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3928_, 0, v___x_3865_);
lean_ctor_set(v_reuseFailAlloc_3928_, 1, v_passedHeartbeats_3849_);
v___x_3867_ = v_reuseFailAlloc_3928_;
goto v_reusejp_3866_;
}
v_reusejp_3866_:
{
lean_object* v___x_3868_; lean_object* v___x_3869_; lean_object* v___x_3870_; uint8_t v___x_3871_; uint8_t v___x_3872_; size_t v_sz_3873_; size_t v___x_3874_; lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v___x_3877_; lean_object* v___x_3878_; lean_object* v_introGoal_3879_; lean_object* v_elimGoal_3880_; lean_object* v_introRapp_3881_; lean_object* v_elimRapp_3882_; lean_object* v___x_3883_; lean_object* v___x_3884_; lean_object* v___x_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___f_3889_; lean_object* v___x_3890_; 
lean_inc_ref(v_meta_3848_);
v___x_3868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3868_, 0, v___x_3867_);
lean_ctor_set(v___x_3868_, 1, v_meta_3848_);
v___x_3869_ = lean_unsigned_to_nat(0u);
v___x_3870_ = ((lean_object*)(lp_aesop_Aesop_findPathForAssignedMVars___closed__0));
v___x_3871_ = 0;
v___x_3872_ = 0;
v_sz_3873_ = lean_array_size(v_goals_3574_);
v___x_3874_ = ((size_t)0ULL);
lean_inc_ref_n(v_goals_3574_, 2);
v___x_3875_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__2(v_sz_3873_, v___x_3874_, v_goals_3574_);
v___x_3876_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_empty___at___00Aesop_addRappUnsafe_spec__3));
lean_inc(v_scriptSteps_x3f_3576_);
lean_inc_ref(v_appliedRule_3570_);
lean_inc(v_parent_3569_);
v___x_3877_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v___x_3877_, 0, v_a_3847_);
lean_ctor_set(v___x_3877_, 1, v_parent_3569_);
lean_ctor_set(v___x_3877_, 2, v___x_3870_);
lean_ctor_set(v___x_3877_, 3, v_appliedRule_3570_);
lean_ctor_set(v___x_3877_, 4, v_scriptSteps_x3f_3576_);
lean_ctor_set(v___x_3877_, 5, v___x_3875_);
lean_ctor_set(v___x_3877_, 6, v___x_3868_);
lean_ctor_set(v___x_3877_, 7, v___x_3876_);
lean_ctor_set(v___x_3877_, 8, v___x_3876_);
lean_ctor_set_uint8(v___x_3877_, sizeof(void*)*9 + 8, v___x_3871_);
lean_ctor_set_uint8(v___x_3877_, sizeof(void*)*9 + 9, v___x_3872_);
lean_ctor_set_float(v___x_3877_, sizeof(void*)*9, v_successProbability_3571_);
v___x_3878_ = lp_aesop_Aesop_treeImpl;
v_introGoal_3879_ = lean_ctor_get(v___x_3878_, 0);
v_elimGoal_3880_ = lean_ctor_get(v___x_3878_, 1);
v_introRapp_3881_ = lean_ctor_get(v___x_3878_, 2);
v_elimRapp_3882_ = lean_ctor_get(v___x_3878_, 3);
lean_inc(v_introRapp_3881_);
v___x_3883_ = lean_apply_1(v_introRapp_3881_, v___x_3877_);
v___x_3884_ = lean_st_mk_ref(v___x_3883_);
v___x_3885_ = lean_st_ref_get(v_parent_3569_);
v___x_3886_ = lean_obj_once(&lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1, &lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1_once, _init_lp_aesop_Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19___closed__1);
v___x_3887_ = lean_array_get_size(v_goals_3574_);
v___x_3888_ = ((lean_object*)(lp_aesop_Aesop_addRappUnsafe___boxed__const__1));
lean_inc(v___x_3885_);
lean_inc_ref(v_elimGoal_3880_);
v___f_3889_ = lean_alloc_closure((void*)(lp_aesop_Aesop_addRappUnsafe___lam__1___boxed), 13, 8);
lean_closure_set(v___f_3889_, 0, v_elimGoal_3880_);
lean_closure_set(v___f_3889_, 1, v___x_3885_);
lean_closure_set(v___f_3889_, 2, v___x_3876_);
lean_closure_set(v___f_3889_, 3, v___x_3888_);
lean_closure_set(v___f_3889_, 4, v___x_3869_);
lean_closure_set(v___f_3889_, 5, v___x_3887_);
lean_closure_set(v___f_3889_, 6, v___x_3886_);
lean_closure_set(v___f_3889_, 7, v_goals_3574_);
lean_inc_ref(v_postState_3575_);
v___x_3890_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_postState_3575_, v___f_3889_, v___y_3840_, v___y_3841_, v___y_3842_, v___y_3843_);
if (lean_obj_tag(v___x_3890_) == 0)
{
lean_object* v_a_3891_; lean_object* v_snd_3892_; lean_object* v_fst_3893_; lean_object* v_fst_3894_; lean_object* v_snd_3895_; lean_object* v___x_3896_; lean_object* v_depth_3897_; lean_object* v_mvars_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; lean_object* v___x_3901_; 
v_a_3891_ = lean_ctor_get(v___x_3890_, 0);
lean_inc(v_a_3891_);
lean_dec_ref_known(v___x_3890_, 1);
v_snd_3892_ = lean_ctor_get(v_a_3891_, 1);
lean_inc(v_snd_3892_);
v_fst_3893_ = lean_ctor_get(v_a_3891_, 0);
lean_inc(v_fst_3893_);
lean_dec(v_a_3891_);
v_fst_3894_ = lean_ctor_get(v_snd_3892_, 0);
lean_inc(v_fst_3894_);
v_snd_3895_ = lean_ctor_get(v_snd_3892_, 1);
lean_inc(v_snd_3895_);
lean_dec(v_snd_3892_);
lean_inc_ref(v_elimGoal_3880_);
lean_inc(v___x_3885_);
v___x_3896_ = lean_apply_1(v_elimGoal_3880_, v___x_3885_);
v_depth_3897_ = lean_ctor_get(v___x_3896_, 4);
lean_inc(v_depth_3897_);
v_mvars_3898_ = lean_ctor_get(v___x_3896_, 7);
lean_inc_ref(v_mvars_3898_);
lean_dec_ref(v___x_3896_);
v___x_3899_ = lean_unsigned_to_nat(1u);
v___x_3900_ = lean_nat_add(v_depth_3897_, v___x_3899_);
lean_dec(v_depth_3897_);
lean_inc(v___x_3900_);
lean_inc(v_parent_3569_);
v___x_3901_ = lp_aesop_Aesop_copyGoals(v_snd_3895_, v_parent_3569_, v_postState_3575_, v_successProbability_3571_, v___x_3900_, v___y_3837_, v___y_3838_, v___y_3839_, v___y_3840_, v___y_3841_, v___y_3842_, v___y_3843_);
if (lean_obj_tag(v___x_3901_) == 0)
{
lean_object* v_a_3902_; size_t v_sz_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; uint8_t v___x_3906_; 
v_a_3902_ = lean_ctor_get(v___x_3901_, 0);
lean_inc_n(v_a_3902_, 2);
lean_dec_ref_known(v___x_3901_, 1);
v_sz_3903_ = lean_array_size(v_a_3902_);
v___x_3904_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__11(v_sz_3903_, v___x_3874_, v_a_3902_);
v___x_3905_ = lean_array_get_size(v_a_3902_);
v___x_3906_ = lean_nat_dec_lt(v___x_3869_, v___x_3905_);
if (v___x_3906_ == 0)
{
lean_inc_ref(v_elimRapp_3882_);
lean_inc(v_introGoal_3879_);
lean_inc(v_introRapp_3881_);
lean_inc_ref(v_elimGoal_3880_);
lean_inc(v___x_3885_);
lean_inc_ref(v_mvars_3898_);
v___y_3580_ = v_mvars_3898_;
v___y_3581_ = v_a_3902_;
v___y_3582_ = v___x_3885_;
v___y_3583_ = v___x_3874_;
v___y_3584_ = v___y_3840_;
v___y_3585_ = v_elimGoal_3880_;
v___y_3586_ = v___y_3839_;
v___y_3587_ = v___x_3899_;
v___y_3588_ = v___y_3841_;
v___y_3589_ = v___x_3885_;
v___y_3590_ = v_introRapp_3881_;
v___y_3591_ = v___x_3869_;
v___y_3592_ = v___x_3874_;
v___y_3593_ = v___x_3876_;
v___y_3594_ = v_mvars_3898_;
v___y_3595_ = v_introGoal_3879_;
v___y_3596_ = v_elimRapp_3882_;
v___y_3597_ = v___y_3842_;
v___y_3598_ = v___y_3837_;
v___y_3599_ = v___y_3843_;
v___y_3600_ = v_fst_3894_;
v___y_3601_ = v___x_3886_;
v___y_3602_ = v___y_3838_;
v___y_3603_ = v___x_3904_;
v___y_3604_ = v___x_3884_;
v___y_3605_ = v___x_3900_;
v___y_3606_ = v_fst_3893_;
goto v___jp_3579_;
}
else
{
uint8_t v___x_3907_; 
v___x_3907_ = lean_nat_dec_le(v___x_3905_, v___x_3905_);
if (v___x_3907_ == 0)
{
if (v___x_3906_ == 0)
{
lean_inc_ref(v_elimRapp_3882_);
lean_inc(v_introGoal_3879_);
lean_inc(v_introRapp_3881_);
lean_inc_ref(v_elimGoal_3880_);
lean_inc(v___x_3885_);
lean_inc_ref(v_mvars_3898_);
v___y_3580_ = v_mvars_3898_;
v___y_3581_ = v_a_3902_;
v___y_3582_ = v___x_3885_;
v___y_3583_ = v___x_3874_;
v___y_3584_ = v___y_3840_;
v___y_3585_ = v_elimGoal_3880_;
v___y_3586_ = v___y_3839_;
v___y_3587_ = v___x_3899_;
v___y_3588_ = v___y_3841_;
v___y_3589_ = v___x_3885_;
v___y_3590_ = v_introRapp_3881_;
v___y_3591_ = v___x_3869_;
v___y_3592_ = v___x_3874_;
v___y_3593_ = v___x_3876_;
v___y_3594_ = v_mvars_3898_;
v___y_3595_ = v_introGoal_3879_;
v___y_3596_ = v_elimRapp_3882_;
v___y_3597_ = v___y_3842_;
v___y_3598_ = v___y_3837_;
v___y_3599_ = v___y_3843_;
v___y_3600_ = v_fst_3894_;
v___y_3601_ = v___x_3886_;
v___y_3602_ = v___y_3838_;
v___y_3603_ = v___x_3904_;
v___y_3604_ = v___x_3884_;
v___y_3605_ = v___x_3900_;
v___y_3606_ = v_fst_3893_;
goto v___jp_3579_;
}
else
{
size_t v___x_3908_; lean_object* v___x_3909_; 
v___x_3908_ = lean_usize_of_nat(v___x_3905_);
v___x_3909_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__22(v_a_3902_, v___x_3874_, v___x_3908_, v_fst_3893_);
lean_inc_ref(v_elimRapp_3882_);
lean_inc(v_introGoal_3879_);
lean_inc(v_introRapp_3881_);
lean_inc_ref(v_elimGoal_3880_);
lean_inc(v___x_3885_);
lean_inc_ref(v_mvars_3898_);
v___y_3580_ = v_mvars_3898_;
v___y_3581_ = v_a_3902_;
v___y_3582_ = v___x_3885_;
v___y_3583_ = v___x_3874_;
v___y_3584_ = v___y_3840_;
v___y_3585_ = v_elimGoal_3880_;
v___y_3586_ = v___y_3839_;
v___y_3587_ = v___x_3899_;
v___y_3588_ = v___y_3841_;
v___y_3589_ = v___x_3885_;
v___y_3590_ = v_introRapp_3881_;
v___y_3591_ = v___x_3869_;
v___y_3592_ = v___x_3874_;
v___y_3593_ = v___x_3876_;
v___y_3594_ = v_mvars_3898_;
v___y_3595_ = v_introGoal_3879_;
v___y_3596_ = v_elimRapp_3882_;
v___y_3597_ = v___y_3842_;
v___y_3598_ = v___y_3837_;
v___y_3599_ = v___y_3843_;
v___y_3600_ = v_fst_3894_;
v___y_3601_ = v___x_3886_;
v___y_3602_ = v___y_3838_;
v___y_3603_ = v___x_3904_;
v___y_3604_ = v___x_3884_;
v___y_3605_ = v___x_3900_;
v___y_3606_ = v___x_3909_;
goto v___jp_3579_;
}
}
else
{
size_t v___x_3910_; lean_object* v___x_3911_; 
v___x_3910_ = lean_usize_of_nat(v___x_3905_);
v___x_3911_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__22(v_a_3902_, v___x_3874_, v___x_3910_, v_fst_3893_);
lean_inc_ref(v_elimRapp_3882_);
lean_inc(v_introGoal_3879_);
lean_inc(v_introRapp_3881_);
lean_inc_ref(v_elimGoal_3880_);
lean_inc(v___x_3885_);
lean_inc_ref(v_mvars_3898_);
v___y_3580_ = v_mvars_3898_;
v___y_3581_ = v_a_3902_;
v___y_3582_ = v___x_3885_;
v___y_3583_ = v___x_3874_;
v___y_3584_ = v___y_3840_;
v___y_3585_ = v_elimGoal_3880_;
v___y_3586_ = v___y_3839_;
v___y_3587_ = v___x_3899_;
v___y_3588_ = v___y_3841_;
v___y_3589_ = v___x_3885_;
v___y_3590_ = v_introRapp_3881_;
v___y_3591_ = v___x_3869_;
v___y_3592_ = v___x_3874_;
v___y_3593_ = v___x_3876_;
v___y_3594_ = v_mvars_3898_;
v___y_3595_ = v_introGoal_3879_;
v___y_3596_ = v_elimRapp_3882_;
v___y_3597_ = v___y_3842_;
v___y_3598_ = v___y_3837_;
v___y_3599_ = v___y_3843_;
v___y_3600_ = v_fst_3894_;
v___y_3601_ = v___x_3886_;
v___y_3602_ = v___y_3838_;
v___y_3603_ = v___x_3904_;
v___y_3604_ = v___x_3884_;
v___y_3605_ = v___x_3900_;
v___y_3606_ = v___x_3911_;
goto v___jp_3579_;
}
}
}
else
{
lean_object* v_a_3912_; lean_object* v___x_3914_; uint8_t v_isShared_3915_; uint8_t v_isSharedCheck_3919_; 
lean_dec(v___x_3900_);
lean_dec_ref(v_mvars_3898_);
lean_dec(v_fst_3894_);
lean_dec(v_fst_3893_);
lean_dec(v___x_3885_);
lean_dec(v___x_3884_);
lean_dec_ref(v_postState_3575_);
lean_dec_ref(v_appliedRule_3570_);
lean_dec(v_parent_3569_);
lean_dec_ref(v_r_3559_);
v_a_3912_ = lean_ctor_get(v___x_3901_, 0);
v_isSharedCheck_3919_ = !lean_is_exclusive(v___x_3901_);
if (v_isSharedCheck_3919_ == 0)
{
v___x_3914_ = v___x_3901_;
v_isShared_3915_ = v_isSharedCheck_3919_;
goto v_resetjp_3913_;
}
else
{
lean_inc(v_a_3912_);
lean_dec(v___x_3901_);
v___x_3914_ = lean_box(0);
v_isShared_3915_ = v_isSharedCheck_3919_;
goto v_resetjp_3913_;
}
v_resetjp_3913_:
{
lean_object* v___x_3917_; 
if (v_isShared_3915_ == 0)
{
v___x_3917_ = v___x_3914_;
goto v_reusejp_3916_;
}
else
{
lean_object* v_reuseFailAlloc_3918_; 
v_reuseFailAlloc_3918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3918_, 0, v_a_3912_);
v___x_3917_ = v_reuseFailAlloc_3918_;
goto v_reusejp_3916_;
}
v_reusejp_3916_:
{
return v___x_3917_;
}
}
}
}
else
{
lean_object* v_a_3920_; lean_object* v___x_3922_; uint8_t v_isShared_3923_; uint8_t v_isSharedCheck_3927_; 
lean_dec(v___x_3885_);
lean_dec(v___x_3884_);
lean_dec_ref(v_postState_3575_);
lean_dec_ref(v_appliedRule_3570_);
lean_dec(v_parent_3569_);
lean_dec_ref(v_r_3559_);
v_a_3920_ = lean_ctor_get(v___x_3890_, 0);
v_isSharedCheck_3927_ = !lean_is_exclusive(v___x_3890_);
if (v_isSharedCheck_3927_ == 0)
{
v___x_3922_ = v___x_3890_;
v_isShared_3923_ = v_isSharedCheck_3927_;
goto v_resetjp_3921_;
}
else
{
lean_inc(v_a_3920_);
lean_dec(v___x_3890_);
v___x_3922_ = lean_box(0);
v_isShared_3923_ = v_isSharedCheck_3927_;
goto v_resetjp_3921_;
}
v_resetjp_3921_:
{
lean_object* v___x_3925_; 
if (v_isShared_3923_ == 0)
{
v___x_3925_ = v___x_3922_;
goto v_reusejp_3924_;
}
else
{
lean_object* v_reuseFailAlloc_3926_; 
v_reuseFailAlloc_3926_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3926_, 0, v_a_3920_);
v___x_3925_ = v_reuseFailAlloc_3926_;
goto v_reusejp_3924_;
}
v_reusejp_3924_:
{
return v___x_3925_;
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
lean_object* v_a_3934_; lean_object* v___x_3936_; uint8_t v_isShared_3937_; uint8_t v_isSharedCheck_3941_; 
lean_dec_ref(v_auxDeclNGen_3836_);
lean_dec_ref(v_postState_3575_);
lean_dec_ref(v_appliedRule_3570_);
lean_dec(v_parent_3569_);
lean_dec_ref(v_r_3559_);
v_a_3934_ = lean_ctor_get(v___x_3844_, 0);
v_isSharedCheck_3941_ = !lean_is_exclusive(v___x_3844_);
if (v_isSharedCheck_3941_ == 0)
{
v___x_3936_ = v___x_3844_;
v_isShared_3937_ = v_isSharedCheck_3941_;
goto v_resetjp_3935_;
}
else
{
lean_inc(v_a_3934_);
lean_dec(v___x_3844_);
v___x_3936_ = lean_box(0);
v_isShared_3937_ = v_isSharedCheck_3941_;
goto v_resetjp_3935_;
}
v_resetjp_3935_:
{
lean_object* v___x_3939_; 
if (v_isShared_3937_ == 0)
{
v___x_3939_ = v___x_3936_;
goto v_reusejp_3938_;
}
else
{
lean_object* v_reuseFailAlloc_3940_; 
v_reuseFailAlloc_3940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3940_, 0, v_a_3934_);
v___x_3939_ = v_reuseFailAlloc_3940_;
goto v_reusejp_3938_;
}
v_reusejp_3938_:
{
return v___x_3939_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_addRappUnsafe___boxed(lean_object* v_r_3967_, lean_object* v_a_3968_, lean_object* v_a_3969_, lean_object* v_a_3970_, lean_object* v_a_3971_, lean_object* v_a_3972_, lean_object* v_a_3973_, lean_object* v_a_3974_, lean_object* v_a_3975_){
_start:
{
lean_object* v_res_3976_; 
v_res_3976_ = lp_aesop_Aesop_addRappUnsafe(v_r_3967_, v_a_3968_, v_a_3969_, v_a_3970_, v_a_3971_, v_a_3972_, v_a_3973_, v_a_3974_);
lean_dec(v_a_3974_);
lean_dec_ref(v_a_3973_);
lean_dec(v_a_3972_);
lean_dec_ref(v_a_3971_);
lean_dec(v_a_3970_);
lean_dec(v_a_3969_);
lean_dec_ref(v_a_3968_);
return v_res_3976_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4(size_t v_sz_3977_, size_t v_i_3978_, lean_object* v_bs_3979_, lean_object* v___y_3980_, lean_object* v___y_3981_, lean_object* v___y_3982_, lean_object* v___y_3983_, lean_object* v___y_3984_, lean_object* v___y_3985_, lean_object* v___y_3986_){
_start:
{
lean_object* v___x_3988_; 
v___x_3988_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___redArg(v_sz_3977_, v_i_3978_, v_bs_3979_);
return v___x_3988_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4___boxed(lean_object* v_sz_3989_, lean_object* v_i_3990_, lean_object* v_bs_3991_, lean_object* v___y_3992_, lean_object* v___y_3993_, lean_object* v___y_3994_, lean_object* v___y_3995_, lean_object* v___y_3996_, lean_object* v___y_3997_, lean_object* v___y_3998_, lean_object* v___y_3999_){
_start:
{
size_t v_sz_boxed_4000_; size_t v_i_boxed_4001_; lean_object* v_res_4002_; 
v_sz_boxed_4000_ = lean_unbox_usize(v_sz_3989_);
lean_dec(v_sz_3989_);
v_i_boxed_4001_ = lean_unbox_usize(v_i_3990_);
lean_dec(v_i_3990_);
v_res_4002_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_addRappUnsafe_spec__4(v_sz_boxed_4000_, v_i_boxed_4001_, v_bs_3991_, v___y_3992_, v___y_3993_, v___y_3994_, v___y_3995_, v___y_3996_, v___y_3997_, v___y_3998_);
lean_dec(v___y_3998_);
lean_dec_ref(v___y_3997_);
lean_dec(v___y_3996_);
lean_dec_ref(v___y_3995_);
lean_dec(v___y_3994_);
lean_dec(v___y_3993_);
lean_dec_ref(v___y_3992_);
return v_res_4002_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5(lean_object* v_val_4003_, lean_object* v_as_4004_, size_t v_i_4005_, size_t v_stop_4006_, lean_object* v_b_4007_, lean_object* v___y_4008_, lean_object* v___y_4009_, lean_object* v___y_4010_, lean_object* v___y_4011_, lean_object* v___y_4012_, lean_object* v___y_4013_, lean_object* v___y_4014_){
_start:
{
lean_object* v___x_4016_; 
v___x_4016_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___redArg(v_val_4003_, v_as_4004_, v_i_4005_, v_stop_4006_, v_b_4007_);
return v___x_4016_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5___boxed(lean_object* v_val_4017_, lean_object* v_as_4018_, lean_object* v_i_4019_, lean_object* v_stop_4020_, lean_object* v_b_4021_, lean_object* v___y_4022_, lean_object* v___y_4023_, lean_object* v___y_4024_, lean_object* v___y_4025_, lean_object* v___y_4026_, lean_object* v___y_4027_, lean_object* v___y_4028_, lean_object* v___y_4029_){
_start:
{
size_t v_i_boxed_4030_; size_t v_stop_boxed_4031_; lean_object* v_res_4032_; 
v_i_boxed_4030_ = lean_unbox_usize(v_i_4019_);
lean_dec(v_i_4019_);
v_stop_boxed_4031_ = lean_unbox_usize(v_stop_4020_);
lean_dec(v_stop_4020_);
v_res_4032_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_addRappUnsafe_spec__5(v_val_4017_, v_as_4018_, v_i_boxed_4030_, v_stop_boxed_4031_, v_b_4021_, v___y_4022_, v___y_4023_, v___y_4024_, v___y_4025_, v___y_4026_, v___y_4027_, v___y_4028_);
lean_dec(v___y_4028_);
lean_dec_ref(v___y_4027_);
lean_dec(v___y_4026_);
lean_dec_ref(v___y_4025_);
lean_dec(v___y_4024_);
lean_dec(v___y_4023_);
lean_dec_ref(v___y_4022_);
lean_dec_ref(v_as_4018_);
return v_res_4032_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6(lean_object* v_mvarId_4033_, lean_object* v___y_4034_, lean_object* v___y_4035_, lean_object* v___y_4036_, lean_object* v___y_4037_){
_start:
{
lean_object* v___x_4039_; 
v___x_4039_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___redArg(v_mvarId_4033_, v___y_4035_);
return v___x_4039_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6___boxed(lean_object* v_mvarId_4040_, lean_object* v___y_4041_, lean_object* v___y_4042_, lean_object* v___y_4043_, lean_object* v___y_4044_, lean_object* v___y_4045_){
_start:
{
lean_object* v_res_4046_; 
v_res_4046_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6(v_mvarId_4040_, v___y_4041_, v___y_4042_, v___y_4043_, v___y_4044_);
lean_dec(v___y_4044_);
lean_dec_ref(v___y_4043_);
lean_dec(v___y_4042_);
lean_dec_ref(v___y_4041_);
lean_dec(v_mvarId_4040_);
return v_res_4046_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7(lean_object* v_00_u03b2_4047_, lean_object* v_m_4048_, lean_object* v_a_4049_){
_start:
{
uint8_t v___x_4050_; 
v___x_4050_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___redArg(v_m_4048_, v_a_4049_);
return v___x_4050_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7___boxed(lean_object* v_00_u03b2_4051_, lean_object* v_m_4052_, lean_object* v_a_4053_){
_start:
{
uint8_t v_res_4054_; lean_object* v_r_4055_; 
v_res_4054_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7(v_00_u03b2_4051_, v_m_4052_, v_a_4053_);
lean_dec(v_a_4053_);
lean_dec_ref(v_m_4052_);
v_r_4055_ = lean_box(v_res_4054_);
return v_r_4055_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12(lean_object* v_mvarId_4056_, lean_object* v___y_4057_, lean_object* v___y_4058_, lean_object* v___y_4059_, lean_object* v___y_4060_, lean_object* v___y_4061_, lean_object* v___y_4062_, lean_object* v___y_4063_){
_start:
{
lean_object* v___x_4065_; 
v___x_4065_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___redArg(v_mvarId_4056_, v___y_4061_);
return v___x_4065_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12___boxed(lean_object* v_mvarId_4066_, lean_object* v___y_4067_, lean_object* v___y_4068_, lean_object* v___y_4069_, lean_object* v___y_4070_, lean_object* v___y_4071_, lean_object* v___y_4072_, lean_object* v___y_4073_, lean_object* v___y_4074_){
_start:
{
lean_object* v_res_4075_; 
v_res_4075_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__12(v_mvarId_4066_, v___y_4067_, v___y_4068_, v___y_4069_, v___y_4070_, v___y_4071_, v___y_4072_, v___y_4073_);
lean_dec(v___y_4073_);
lean_dec_ref(v___y_4072_);
lean_dec(v___y_4071_);
lean_dec_ref(v___y_4070_);
lean_dec(v___y_4069_);
lean_dec(v___y_4068_);
lean_dec_ref(v___y_4067_);
lean_dec(v_mvarId_4066_);
return v_res_4075_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13(lean_object* v_00_u03b2_4076_, lean_object* v_m_4077_, lean_object* v_a_4078_, lean_object* v_b_4079_){
_start:
{
lean_object* v___x_4080_; 
v___x_4080_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13___redArg(v_m_4077_, v_a_4078_, v_b_4079_);
return v___x_4080_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14(lean_object* v___x_4081_, lean_object* v_as_4082_, size_t v_sz_4083_, size_t v_i_4084_, lean_object* v_b_4085_, lean_object* v___y_4086_, lean_object* v___y_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_, lean_object* v___y_4090_, lean_object* v___y_4091_, lean_object* v___y_4092_){
_start:
{
lean_object* v___x_4094_; 
v___x_4094_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___redArg(v___x_4081_, v_as_4082_, v_sz_4083_, v_i_4084_, v_b_4085_);
return v___x_4094_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14___boxed(lean_object* v___x_4095_, lean_object* v_as_4096_, lean_object* v_sz_4097_, lean_object* v_i_4098_, lean_object* v_b_4099_, lean_object* v___y_4100_, lean_object* v___y_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_, lean_object* v___y_4104_, lean_object* v___y_4105_, lean_object* v___y_4106_, lean_object* v___y_4107_){
_start:
{
size_t v_sz_boxed_4108_; size_t v_i_boxed_4109_; lean_object* v_res_4110_; 
v_sz_boxed_4108_ = lean_unbox_usize(v_sz_4097_);
lean_dec(v_sz_4097_);
v_i_boxed_4109_ = lean_unbox_usize(v_i_4098_);
lean_dec(v_i_4098_);
v_res_4110_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_addRappUnsafe_spec__14(v___x_4095_, v_as_4096_, v_sz_boxed_4108_, v_i_boxed_4109_, v_b_4099_, v___y_4100_, v___y_4101_, v___y_4102_, v___y_4103_, v___y_4104_, v___y_4105_, v___y_4106_);
lean_dec(v___y_4106_);
lean_dec_ref(v___y_4105_);
lean_dec(v___y_4104_);
lean_dec_ref(v___y_4103_);
lean_dec(v___y_4102_);
lean_dec(v___y_4101_);
lean_dec_ref(v___y_4100_);
lean_dec_ref(v_as_4096_);
lean_dec_ref(v___x_4095_);
return v_res_4110_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9(lean_object* v_00_u03b2_4111_, lean_object* v_x_4112_, lean_object* v_x_4113_){
_start:
{
uint8_t v___x_4114_; 
v___x_4114_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___redArg(v_x_4112_, v_x_4113_);
return v___x_4114_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9___boxed(lean_object* v_00_u03b2_4115_, lean_object* v_x_4116_, lean_object* v_x_4117_){
_start:
{
uint8_t v_res_4118_; lean_object* v_r_4119_; 
v_res_4118_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9(v_00_u03b2_4115_, v_x_4116_, v_x_4117_);
lean_dec(v_x_4117_);
lean_dec_ref(v_x_4116_);
v_r_4119_ = lean_box(v_res_4118_);
return v_r_4119_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11(lean_object* v_00_u03b2_4120_, lean_object* v_a_4121_, lean_object* v_x_4122_){
_start:
{
uint8_t v___x_4123_; 
v___x_4123_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___redArg(v_a_4121_, v_x_4122_);
return v___x_4123_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11___boxed(lean_object* v_00_u03b2_4124_, lean_object* v_a_4125_, lean_object* v_x_4126_){
_start:
{
uint8_t v_res_4127_; lean_object* v_r_4128_; 
v_res_4127_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_addRappUnsafe_spec__7_spec__11(v_00_u03b2_4124_, v_a_4125_, v_x_4126_);
lean_dec(v_x_4126_);
lean_dec(v_a_4125_);
v_r_4128_ = lean_box(v_res_4127_);
return v_r_4128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18(lean_object* v_00_u03b2_4129_, lean_object* v_data_4130_){
_start:
{
lean_object* v___x_4131_; 
v___x_4131_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18___redArg(v_data_4130_);
return v___x_4131_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26(lean_object* v_00_u03b2_4132_, lean_object* v_m_4133_, lean_object* v_a_4134_){
_start:
{
lean_object* v___x_4135_; 
v___x_4135_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___redArg(v_m_4133_, v_a_4134_);
return v___x_4135_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26___boxed(lean_object* v_00_u03b2_4136_, lean_object* v_m_4137_, lean_object* v_a_4138_){
_start:
{
lean_object* v_res_4139_; 
v_res_4139_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26(v_00_u03b2_4136_, v_m_4137_, v_a_4138_);
lean_dec(v_a_4138_);
lean_dec_ref(v_m_4137_);
return v_res_4139_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27(lean_object* v_00_u03b2_4140_, lean_object* v_m_4141_, lean_object* v_a_4142_, lean_object* v_b_4143_){
_start:
{
lean_object* v___x_4144_; 
v___x_4144_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27___redArg(v_m_4141_, v_a_4142_, v_b_4143_);
return v___x_4144_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10(lean_object* v_00_u03b2_4145_, lean_object* v_x_4146_, size_t v_x_4147_, lean_object* v_x_4148_){
_start:
{
uint8_t v___x_4149_; 
v___x_4149_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___redArg(v_x_4146_, v_x_4147_, v_x_4148_);
return v___x_4149_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10___boxed(lean_object* v_00_u03b2_4150_, lean_object* v_x_4151_, lean_object* v_x_4152_, lean_object* v_x_4153_){
_start:
{
size_t v_x_126072__boxed_4154_; uint8_t v_res_4155_; lean_object* v_r_4156_; 
v_x_126072__boxed_4154_ = lean_unbox_usize(v_x_4152_);
lean_dec(v_x_4152_);
v_res_4155_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10(v_00_u03b2_4150_, v_x_4151_, v_x_126072__boxed_4154_, v_x_4153_);
lean_dec(v_x_4153_);
lean_dec_ref(v_x_4151_);
v_r_4156_ = lean_box(v_res_4155_);
return v_r_4156_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20(lean_object* v_00_u03b2_4157_, lean_object* v_i_4158_, lean_object* v_source_4159_, lean_object* v_target_4160_){
_start:
{
lean_object* v___x_4161_; 
v___x_4161_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20___redArg(v_i_4158_, v_source_4159_, v_target_4160_);
return v___x_4161_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30(lean_object* v_00_u03b2_4162_, lean_object* v_a_4163_, lean_object* v_x_4164_){
_start:
{
lean_object* v___x_4165_; 
v___x_4165_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___redArg(v_a_4163_, v_x_4164_);
return v___x_4165_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30___boxed(lean_object* v_00_u03b2_4166_, lean_object* v_a_4167_, lean_object* v_x_4168_){
_start:
{
lean_object* v_res_4169_; 
v_res_4169_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__26_spec__30(v_00_u03b2_4166_, v_a_4167_, v_x_4168_);
lean_dec(v_x_4168_);
lean_dec(v_a_4167_);
return v_res_4169_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27_spec__32(lean_object* v_00_u03b2_4170_, lean_object* v_a_4171_, lean_object* v_b_4172_, lean_object* v_x_4173_){
_start:
{
lean_object* v___x_4174_; 
v___x_4174_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__27_spec__32___redArg(v_a_4171_, v_b_4172_, v_x_4173_);
return v___x_4174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42(lean_object* v_00_u03b2_4175_, lean_object* v_m_4176_, size_t v_a_4177_){
_start:
{
lean_object* v___x_4178_; 
v___x_4178_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___redArg(v_m_4176_, v_a_4177_);
return v___x_4178_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42___boxed(lean_object* v_00_u03b2_4179_, lean_object* v_m_4180_, lean_object* v_a_4181_){
_start:
{
size_t v_a_boxed_4182_; lean_object* v_res_4183_; 
v_a_boxed_4182_ = lean_unbox_usize(v_a_4181_);
lean_dec(v_a_4181_);
v_res_4183_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42(v_00_u03b2_4179_, v_m_4180_, v_a_boxed_4182_);
lean_dec_ref(v_m_4180_);
return v_res_4183_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43(lean_object* v_00_u03b2_4184_, lean_object* v_m_4185_, size_t v_a_4186_, lean_object* v_b_4187_){
_start:
{
lean_object* v___x_4188_; 
v___x_4188_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___redArg(v_m_4185_, v_a_4186_, v_b_4187_);
return v___x_4188_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43___boxed(lean_object* v_00_u03b2_4189_, lean_object* v_m_4190_, lean_object* v_a_4191_, lean_object* v_b_4192_){
_start:
{
size_t v_a_boxed_4193_; lean_object* v_res_4194_; 
v_a_boxed_4193_ = lean_unbox_usize(v_a_4191_);
lean_dec(v_a_4191_);
v_res_4194_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43(v_00_u03b2_4189_, v_m_4190_, v_a_boxed_4193_, v_b_4192_);
return v_res_4194_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27(lean_object* v_00_u03b2_4195_, lean_object* v_keys_4196_, lean_object* v_vals_4197_, lean_object* v_heq_4198_, lean_object* v_i_4199_, lean_object* v_k_4200_){
_start:
{
uint8_t v___x_4201_; 
v___x_4201_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___redArg(v_keys_4196_, v_i_4199_, v_k_4200_);
return v___x_4201_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27___boxed(lean_object* v_00_u03b2_4202_, lean_object* v_keys_4203_, lean_object* v_vals_4204_, lean_object* v_heq_4205_, lean_object* v_i_4206_, lean_object* v_k_4207_){
_start:
{
uint8_t v_res_4208_; lean_object* v_r_4209_; 
v_res_4208_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Aesop_addRappUnsafe_spec__6_spec__9_spec__10_spec__27(v_00_u03b2_4202_, v_keys_4203_, v_vals_4204_, v_heq_4205_, v_i_4206_, v_k_4207_);
lean_dec(v_k_4207_);
lean_dec_ref(v_vals_4204_);
lean_dec_ref(v_keys_4203_);
v_r_4209_ = lean_box(v_res_4208_);
return v_r_4209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20_spec__31(lean_object* v_00_u03b2_4210_, lean_object* v_x_4211_, lean_object* v_x_4212_){
_start:
{
lean_object* v___x_4213_; 
v___x_4213_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_addRappUnsafe_spec__13_spec__18_spec__20_spec__31___redArg(v_x_4211_, v_x_4212_);
return v___x_4213_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34(lean_object* v_00_u03b2_4214_, lean_object* v_m_4215_, lean_object* v_a_4216_){
_start:
{
lean_object* v___x_4217_; 
v___x_4217_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___redArg(v_m_4215_, v_a_4216_);
return v___x_4217_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34___boxed(lean_object* v_00_u03b2_4218_, lean_object* v_m_4219_, lean_object* v_a_4220_){
_start:
{
lean_object* v_res_4221_; 
v_res_4221_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34(v_00_u03b2_4218_, v_m_4219_, v_a_4220_);
lean_dec_ref(v_m_4219_);
return v_res_4221_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51(lean_object* v_00_u03b2_4222_, size_t v_a_4223_, lean_object* v_x_4224_){
_start:
{
lean_object* v___x_4225_; 
v___x_4225_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___redArg(v_a_4223_, v_x_4224_);
return v___x_4225_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51___boxed(lean_object* v_00_u03b2_4226_, lean_object* v_a_4227_, lean_object* v_x_4228_){
_start:
{
size_t v_a_boxed_4229_; lean_object* v_res_4230_; 
v_a_boxed_4229_ = lean_unbox_usize(v_a_4227_);
lean_dec(v_a_4227_);
v_res_4230_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__42_spec__51(v_00_u03b2_4226_, v_a_boxed_4229_, v_x_4228_);
lean_dec(v_x_4228_);
return v_res_4230_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53(lean_object* v_00_u03b2_4231_, size_t v_a_4232_, lean_object* v_x_4233_){
_start:
{
uint8_t v___x_4234_; 
v___x_4234_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___redArg(v_a_4232_, v_x_4233_);
return v___x_4234_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53___boxed(lean_object* v_00_u03b2_4235_, lean_object* v_a_4236_, lean_object* v_x_4237_){
_start:
{
size_t v_a_boxed_4238_; uint8_t v_res_4239_; lean_object* v_r_4240_; 
v_a_boxed_4238_ = lean_unbox_usize(v_a_4236_);
lean_dec(v_a_4236_);
v_res_4239_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__53(v_00_u03b2_4235_, v_a_boxed_4238_, v_x_4237_);
lean_dec(v_x_4237_);
v_r_4240_ = lean_box(v_res_4239_);
return v_r_4240_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54(lean_object* v_00_u03b2_4241_, lean_object* v_data_4242_){
_start:
{
lean_object* v___x_4243_; 
v___x_4243_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54___redArg(v_data_4242_);
return v___x_4243_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55(lean_object* v_00_u03b2_4244_, size_t v_a_4245_, lean_object* v_b_4246_, lean_object* v_x_4247_){
_start:
{
lean_object* v___x_4248_; 
v___x_4248_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___redArg(v_a_4245_, v_b_4246_, v_x_4247_);
return v___x_4248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55___boxed(lean_object* v_00_u03b2_4249_, lean_object* v_a_4250_, lean_object* v_b_4251_, lean_object* v_x_4252_){
_start:
{
size_t v_a_boxed_4253_; lean_object* v_res_4254_; 
v_a_boxed_4253_ = lean_unbox_usize(v_a_4250_);
lean_dec(v_a_4250_);
v_res_4254_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__55(v_00_u03b2_4249_, v_a_boxed_4253_, v_b_4251_, v_x_4252_);
return v_res_4254_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34_spec__42(lean_object* v_00_u03b2_4255_, lean_object* v_a_4256_, lean_object* v_x_4257_){
_start:
{
lean_object* v___x_4258_; 
v___x_4258_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_UnionFind_find_x3f___at___00__private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__25_spec__28_spec__34_spec__42___redArg(v_a_4256_, v_x_4257_);
return v___x_4258_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47(lean_object* v_00_u03b2_4259_, lean_object* v_m_4260_, lean_object* v_a_4261_){
_start:
{
uint8_t v___x_4262_; 
v___x_4262_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___redArg(v_m_4260_, v_a_4261_);
return v___x_4262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47___boxed(lean_object* v_00_u03b2_4263_, lean_object* v_m_4264_, lean_object* v_a_4265_){
_start:
{
uint8_t v_res_4266_; lean_object* v_r_4267_; 
v_res_4266_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47(v_00_u03b2_4263_, v_m_4264_, v_a_4265_);
lean_dec_ref(v_m_4264_);
v_r_4267_ = lean_box(v_res_4266_);
return v_r_4267_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48(lean_object* v_00_u03b2_4268_, lean_object* v_m_4269_, lean_object* v_a_4270_, lean_object* v_b_4271_){
_start:
{
lean_object* v___x_4272_; 
v___x_4272_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48___redArg(v_m_4269_, v_a_4270_, v_b_4271_);
return v___x_4272_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58(lean_object* v_00_u03b2_4273_, lean_object* v_i_4274_, lean_object* v_source_4275_, lean_object* v_target_4276_){
_start:
{
lean_object* v___x_4277_; 
v___x_4277_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58___redArg(v_i_4274_, v_source_4275_, v_target_4276_);
return v___x_4277_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55(lean_object* v_00_u03b2_4278_, lean_object* v_a_4279_, lean_object* v_x_4280_){
_start:
{
uint8_t v___x_4281_; 
v___x_4281_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___redArg(v_a_4279_, v_x_4280_);
return v___x_4281_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55___boxed(lean_object* v_00_u03b2_4282_, lean_object* v_a_4283_, lean_object* v_x_4284_){
_start:
{
uint8_t v_res_4285_; lean_object* v_r_4286_; 
v_res_4285_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__47_spec__55(v_00_u03b2_4282_, v_a_4283_, v_x_4284_);
v_r_4286_ = lean_box(v_res_4285_);
return v_r_4286_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57(lean_object* v_00_u03b2_4287_, lean_object* v_data_4288_){
_start:
{
lean_object* v___x_4289_; 
v___x_4289_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57___redArg(v_data_4288_);
return v___x_4289_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__58(lean_object* v_00_u03b2_4290_, lean_object* v_a_4291_, lean_object* v_b_4292_, lean_object* v_x_4293_){
_start:
{
lean_object* v___x_4294_; 
v___x_4294_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__58___redArg(v_a_4291_, v_b_4292_, v_x_4293_);
return v___x_4294_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58_spec__64(lean_object* v_00_u03b2_4295_, lean_object* v_x_4296_, lean_object* v_x_4297_){
_start:
{
lean_object* v___x_4298_; 
v___x_4298_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_sets___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__32_spec__43_spec__54_spec__58_spec__64___redArg(v_x_4296_, v_x_4297_);
return v___x_4298_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63(lean_object* v_00_u03b2_4299_, lean_object* v_i_4300_, lean_object* v_source_4301_, lean_object* v_target_4302_){
_start:
{
lean_object* v___x_4303_; 
v___x_4303_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63___redArg(v_i_4300_, v_source_4301_, v_target_4302_);
return v___x_4303_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63_spec__65(lean_object* v_00_u03b2_4304_, lean_object* v_x_4305_, lean_object* v_x_4306_){
_start:
{
lean_object* v___x_4307_; 
v___x_4307_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_UnionFind_add___at___00Aesop_UnionFind_addArray___at___00Aesop_UnionFind_ofArray___at___00Aesop_cluster___at___00Aesop_addRappUnsafe_spec__19_spec__30_spec__36_spec__43_spec__48_spec__57_spec__63_spec__65___redArg(v_x_4305_, v_x_4306_);
return v___x_4307_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_UnionFind(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_AddRapp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_UnionFind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_AddRapp(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_UnionFind(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_AddRapp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_State_ApplyGoalDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_UnionFind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_AddRapp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_AddRapp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_AddRapp(builtin);
}
#ifdef __cplusplus
}
#endif
