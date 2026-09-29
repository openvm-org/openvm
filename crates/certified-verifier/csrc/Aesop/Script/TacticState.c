// Lean compiler output
// Module: Aesop.Script.TacticState
// Imports: public import Init public meta import Init import Batteries.Lean.Meta.Basic public import Aesop.Script.GoalWithMVars import Lean.Meta.CollectMVars
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
uint8_t lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_instBEqMVarId_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_instHashableMVarId_hash___boxed(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_instBEqGoalWithMVars___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Std_DHashMap_Internal_AssocList_length___redArg(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_instInhabitedTacticState_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_instInhabitedTacticState;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_mkInitial(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_mkInitial___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "internal error: "};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = ": unknown goal '\?"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_findIdx_x3f_loop___at___00Aesop_Script_TacticState_getVisibleGoalIndex_x3f_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_findIdx_x3f_loop___at___00Aesop_Script_TacticState_getVisibleGoalIndex_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "getVisibleGoalIndex"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getMainGoal_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getMainGoal_x3f___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Script_TacticState_visibleGoalsHaveMVars_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Script_TacticState_visibleGoalsHaveMVars_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Script_TacticState_visibleGoalsHaveMVars(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_visibleGoalsHaveMVars___boxed(lean_object*);
static const lean_array_object lp_aesop_Aesop_Script_TacticState_solveVisibleGoals___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_solveVisibleGoals___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_solveVisibleGoals___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_solveVisibleGoals(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__1_value;
static const lean_closure_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__2_value;
static const lean_closure_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__3_value;
static const lean_closure_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__4_value;
static const lean_closure_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__5_value;
static const lean_closure_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__6_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__0_value),((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__1_value)}};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__7_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__7_value),((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__2_value),((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__3_value),((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__4_value),((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__5_value)}};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__8_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__8_value),((lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__6_value)}};
static const lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_eraseSolvedGoals_mvarWasSolved(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_eraseSolvedGoals_mvarWasSolved___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_filter_go___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_filter_go___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqGoalWithMVars___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "applyTactic"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableMVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__2(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "focus"};
static const lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__2;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqMVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_3_ = lean_box(0);
v___x_4_ = lean_unsigned_to_nat(16u);
v___x_5_ = lean_mk_array(v___x_4_, v___x_3_);
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_obj_once(&lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__1, &lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__1_once, _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__1);
v___x_7_ = lean_unsigned_to_nat(0u);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__3(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_obj_once(&lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2, &lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2_once, _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2);
v___x_10_ = ((lean_object*)(lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__0));
v___x_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
lean_ctor_set(v___x_11_, 1, v___x_9_);
return v___x_11_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default(void){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_obj_once(&lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__3, &lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__3_once, _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__3);
return v___x_12_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_instInhabitedTacticState(void){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_aesop_Aesop_Script_instInhabitedTacticState_default;
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_mkInitial(lean_object* v_goal_14_, lean_object* v_a_15_, lean_object* v_a_16_, lean_object* v_a_17_, lean_object* v_a_18_){
_start:
{
uint8_t v___x_20_; lean_object* v___x_21_; 
v___x_20_ = 0;
lean_inc(v_goal_14_);
v___x_21_ = l_Lean_MVarId_getMVarDependencies(v_goal_14_, v___x_20_, v_a_15_, v_a_16_, v_a_17_, v_a_18_);
if (lean_obj_tag(v___x_21_) == 0)
{
lean_object* v_a_22_; lean_object* v___x_24_; uint8_t v_isShared_25_; uint8_t v_isSharedCheck_35_; 
v_a_22_ = lean_ctor_get(v___x_21_, 0);
v_isSharedCheck_35_ = !lean_is_exclusive(v___x_21_);
if (v_isSharedCheck_35_ == 0)
{
v___x_24_ = v___x_21_;
v_isShared_25_ = v_isSharedCheck_35_;
goto v_resetjp_23_;
}
else
{
lean_inc(v_a_22_);
lean_dec(v___x_21_);
v___x_24_ = lean_box(0);
v_isShared_25_ = v_isSharedCheck_35_;
goto v_resetjp_23_;
}
v_resetjp_23_:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_33_; 
v___x_26_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_26_, 0, v_goal_14_);
lean_ctor_set(v___x_26_, 1, v_a_22_);
v___x_27_ = lean_unsigned_to_nat(1u);
v___x_28_ = lean_mk_empty_array_with_capacity(v___x_27_);
v___x_29_ = lean_array_push(v___x_28_, v___x_26_);
v___x_30_ = lean_obj_once(&lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2, &lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2_once, _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2);
v___x_31_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_31_, 0, v___x_29_);
lean_ctor_set(v___x_31_, 1, v___x_30_);
if (v_isShared_25_ == 0)
{
lean_ctor_set(v___x_24_, 0, v___x_31_);
v___x_33_ = v___x_24_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v___x_31_);
v___x_33_ = v_reuseFailAlloc_34_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
return v___x_33_;
}
}
}
else
{
lean_object* v_a_36_; lean_object* v___x_38_; uint8_t v_isShared_39_; uint8_t v_isSharedCheck_43_; 
lean_dec(v_goal_14_);
v_a_36_ = lean_ctor_get(v___x_21_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_21_);
if (v_isSharedCheck_43_ == 0)
{
v___x_38_ = v___x_21_;
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
else
{
lean_inc(v_a_36_);
lean_dec(v___x_21_);
v___x_38_ = lean_box(0);
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
v_resetjp_37_:
{
lean_object* v___x_41_; 
if (v_isShared_39_ == 0)
{
v___x_41_ = v___x_38_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_a_36_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_mkInitial___boxed(lean_object* v_goal_44_, lean_object* v_a_45_, lean_object* v_a_46_, lean_object* v_a_47_, lean_object* v_a_48_, lean_object* v_a_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_aesop_Aesop_Script_TacticState_mkInitial(v_goal_44_, v_a_45_, v_a_46_, v_a_47_, v_a_48_);
lean_dec(v_a_48_);
lean_dec_ref(v_a_47_);
lean_dec(v_a_46_);
lean_dec_ref(v_a_45_);
return v_res_50_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__1(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_52_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__0));
v___x_53_ = l_Lean_stringToMessageData(v___x_52_);
return v___x_53_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__3(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__2));
v___x_56_ = l_Lean_stringToMessageData(v___x_55_);
return v___x_56_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__5(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__4));
v___x_59_ = l_Lean_stringToMessageData(v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg(lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_goal_62_, lean_object* v_pre_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_64_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__1, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__1);
v___x_65_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_64_);
lean_ctor_set(v___x_65_, 1, v_pre_63_);
v___x_66_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__3, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__3_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__3);
v___x_67_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_65_);
lean_ctor_set(v___x_67_, 1, v___x_66_);
v___x_68_ = l_Lean_MessageData_ofName(v_goal_62_);
v___x_69_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_67_);
lean_ctor_set(v___x_69_, 1, v___x_68_);
v___x_70_ = lean_obj_once(&lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__5, &lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg___closed__5);
v___x_71_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_69_);
lean_ctor_set(v___x_71_, 1, v___x_70_);
v___x_72_ = l_Lean_throwError___redArg(v_inst_60_, v_inst_61_, v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError(lean_object* v_m_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_00_u03b1_76_, lean_object* v_goal_77_, lean_object* v_pre_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg(v_inst_74_, v_inst_75_, v_goal_77_, v_pre_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_findIdx_x3f_loop___at___00Aesop_Script_TacticState_getVisibleGoalIndex_x3f_spec__0(lean_object* v_goal_80_, lean_object* v_as_81_, lean_object* v_j_82_){
_start:
{
lean_object* v___x_83_; uint8_t v___x_84_; 
v___x_83_ = lean_array_get_size(v_as_81_);
v___x_84_ = lean_nat_dec_lt(v_j_82_, v___x_83_);
if (v___x_84_ == 0)
{
lean_object* v___x_85_; 
lean_dec(v_j_82_);
v___x_85_ = lean_box(0);
return v___x_85_;
}
else
{
lean_object* v___x_86_; lean_object* v_goal_87_; uint8_t v___x_88_; 
v___x_86_ = lean_array_fget_borrowed(v_as_81_, v_j_82_);
v_goal_87_ = lean_ctor_get(v___x_86_, 0);
v___x_88_ = l_Lean_instBEqMVarId_beq(v_goal_87_, v_goal_80_);
if (v___x_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_89_ = lean_unsigned_to_nat(1u);
v___x_90_ = lean_nat_add(v_j_82_, v___x_89_);
lean_dec(v_j_82_);
v_j_82_ = v___x_90_;
goto _start;
}
else
{
lean_object* v___x_92_; 
v___x_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_92_, 0, v_j_82_);
return v___x_92_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_findIdx_x3f_loop___at___00Aesop_Script_TacticState_getVisibleGoalIndex_x3f_spec__0___boxed(lean_object* v_goal_93_, lean_object* v_as_94_, lean_object* v_j_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_aesop_Array_findIdx_x3f_loop___at___00Aesop_Script_TacticState_getVisibleGoalIndex_x3f_spec__0(v_goal_93_, v_as_94_, v_j_95_);
lean_dec_ref(v_as_94_);
lean_dec(v_goal_93_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(lean_object* v_ts_97_, lean_object* v_goal_98_){
_start:
{
lean_object* v_visibleGoals_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v_visibleGoals_99_ = lean_ctor_get(v_ts_97_, 0);
v___x_100_ = lean_unsigned_to_nat(0u);
v___x_101_ = lp_aesop_Array_findIdx_x3f_loop___at___00Aesop_Script_TacticState_getVisibleGoalIndex_x3f_spec__0(v_goal_98_, v_visibleGoals_99_, v___x_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f___boxed(lean_object* v_ts_102_, lean_object* v_goal_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(v_ts_102_, v_goal_103_);
lean_dec(v_goal_103_);
lean_dec_ref(v_ts_102_);
return v_res_104_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__2(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__1));
v___x_109_ = l_Lean_MessageData_ofFormat(v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg(lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_ts_112_, lean_object* v_goal_113_){
_start:
{
lean_object* v_toApplicative_114_; lean_object* v_toPure_115_; lean_object* v___x_116_; 
v_toApplicative_114_ = lean_ctor_get(v_inst_110_, 0);
v_toPure_115_ = lean_ctor_get(v_toApplicative_114_, 1);
v___x_116_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex_x3f(v_ts_112_, v_goal_113_);
if (lean_obj_tag(v___x_116_) == 1)
{
lean_object* v_val_117_; lean_object* v___x_118_; 
lean_inc(v_toPure_115_);
lean_dec(v_goal_113_);
lean_dec_ref(v_inst_111_);
lean_dec_ref(v_inst_110_);
v_val_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_val_117_);
lean_dec_ref_known(v___x_116_, 1);
v___x_118_ = lean_apply_2(v_toPure_115_, lean_box(0), v_val_117_);
return v___x_118_;
}
else
{
lean_object* v___x_119_; lean_object* v___x_120_; 
lean_dec(v___x_116_);
v___x_119_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__2, &lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__2_once, _init_lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___closed__2);
v___x_120_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg(v_inst_110_, v_inst_111_, v_goal_113_, v___x_119_);
return v___x_120_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg___boxed(lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_ts_123_, lean_object* v_goal_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg(v_inst_121_, v_inst_122_, v_ts_123_, v_goal_124_);
lean_dec_ref(v_ts_123_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex(lean_object* v_m_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_ts_129_, lean_object* v_goal_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___redArg(v_inst_127_, v_inst_128_, v_ts_129_, v_goal_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex___boxed(lean_object* v_m_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_ts_135_, lean_object* v_goal_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_aesop_Aesop_Script_TacticState_getVisibleGoalIndex(v_m_132_, v_inst_133_, v_inst_134_, v_ts_135_, v_goal_136_);
lean_dec_ref(v_ts_135_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getMainGoal_x3f(lean_object* v_ts_138_){
_start:
{
lean_object* v_visibleGoals_139_; lean_object* v___x_140_; lean_object* v___x_141_; uint8_t v___x_142_; 
v_visibleGoals_139_ = lean_ctor_get(v_ts_138_, 0);
v___x_140_ = lean_unsigned_to_nat(0u);
v___x_141_ = lean_array_get_size(v_visibleGoals_139_);
v___x_142_ = lean_nat_dec_lt(v___x_140_, v___x_141_);
if (v___x_142_ == 0)
{
lean_object* v___x_143_; 
v___x_143_ = lean_box(0);
return v___x_143_;
}
else
{
lean_object* v___x_144_; lean_object* v_goal_145_; lean_object* v___x_146_; 
v___x_144_ = lean_array_fget_borrowed(v_visibleGoals_139_, v___x_140_);
v_goal_145_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_goal_145_);
v___x_146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_146_, 0, v_goal_145_);
return v___x_146_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_getMainGoal_x3f___boxed(lean_object* v_ts_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_aesop_Aesop_Script_TacticState_getMainGoal_x3f(v_ts_147_);
lean_dec_ref(v_ts_147_);
return v_res_148_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Script_TacticState_visibleGoalsHaveMVars_spec__0(lean_object* v_as_149_, size_t v_i_150_, size_t v_stop_151_){
_start:
{
uint8_t v___x_152_; 
v___x_152_ = lean_usize_dec_eq(v_i_150_, v_stop_151_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; lean_object* v_mvars_154_; lean_object* v_size_155_; uint8_t v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_153_ = lean_array_uget_borrowed(v_as_149_, v_i_150_);
v_mvars_154_ = lean_ctor_get(v___x_153_, 1);
v_size_155_ = lean_ctor_get(v_mvars_154_, 0);
v___x_156_ = 1;
v___x_157_ = lean_unsigned_to_nat(0u);
v___x_158_ = lean_nat_dec_eq(v_size_155_, v___x_157_);
if (v___x_158_ == 0)
{
return v___x_156_;
}
else
{
if (v___x_152_ == 0)
{
size_t v___x_159_; size_t v___x_160_; 
v___x_159_ = ((size_t)1ULL);
v___x_160_ = lean_usize_add(v_i_150_, v___x_159_);
v_i_150_ = v___x_160_;
goto _start;
}
else
{
return v___x_156_;
}
}
}
else
{
uint8_t v___x_162_; 
v___x_162_ = 0;
return v___x_162_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Script_TacticState_visibleGoalsHaveMVars_spec__0___boxed(lean_object* v_as_163_, lean_object* v_i_164_, lean_object* v_stop_165_){
_start:
{
size_t v_i_boxed_166_; size_t v_stop_boxed_167_; uint8_t v_res_168_; lean_object* v_r_169_; 
v_i_boxed_166_ = lean_unbox_usize(v_i_164_);
lean_dec(v_i_164_);
v_stop_boxed_167_ = lean_unbox_usize(v_stop_165_);
lean_dec(v_stop_165_);
v_res_168_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Script_TacticState_visibleGoalsHaveMVars_spec__0(v_as_163_, v_i_boxed_166_, v_stop_boxed_167_);
lean_dec_ref(v_as_163_);
v_r_169_ = lean_box(v_res_168_);
return v_r_169_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Script_TacticState_visibleGoalsHaveMVars(lean_object* v_ts_170_){
_start:
{
lean_object* v_visibleGoals_171_; lean_object* v___x_172_; lean_object* v___x_173_; uint8_t v___x_174_; 
v_visibleGoals_171_ = lean_ctor_get(v_ts_170_, 0);
v___x_172_ = lean_unsigned_to_nat(0u);
v___x_173_ = lean_array_get_size(v_visibleGoals_171_);
v___x_174_ = lean_nat_dec_lt(v___x_172_, v___x_173_);
if (v___x_174_ == 0)
{
return v___x_174_;
}
else
{
if (v___x_174_ == 0)
{
return v___x_174_;
}
else
{
size_t v___x_175_; size_t v___x_176_; uint8_t v___x_177_; 
v___x_175_ = ((size_t)0ULL);
v___x_176_ = lean_usize_of_nat(v___x_173_);
v___x_177_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Script_TacticState_visibleGoalsHaveMVars_spec__0(v_visibleGoals_171_, v___x_175_, v___x_176_);
return v___x_177_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_visibleGoalsHaveMVars___boxed(lean_object* v_ts_178_){
_start:
{
uint8_t v_res_179_; lean_object* v_r_180_; 
v_res_179_ = lp_aesop_Aesop_Script_TacticState_visibleGoalsHaveMVars(v_ts_178_);
lean_dec_ref(v_ts_178_);
v_r_180_ = lean_box(v_res_179_);
return v_r_180_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_solveVisibleGoals(lean_object* v_ts_183_){
_start:
{
lean_object* v_invisibleGoals_184_; lean_object* v___x_186_; uint8_t v_isShared_187_; uint8_t v_isSharedCheck_192_; 
v_invisibleGoals_184_ = lean_ctor_get(v_ts_183_, 1);
v_isSharedCheck_192_ = !lean_is_exclusive(v_ts_183_);
if (v_isSharedCheck_192_ == 0)
{
lean_object* v_unused_193_; 
v_unused_193_ = lean_ctor_get(v_ts_183_, 0);
lean_dec(v_unused_193_);
v___x_186_ = v_ts_183_;
v_isShared_187_ = v_isSharedCheck_192_;
goto v_resetjp_185_;
}
else
{
lean_inc(v_invisibleGoals_184_);
lean_dec(v_ts_183_);
v___x_186_ = lean_box(0);
v_isShared_187_ = v_isSharedCheck_192_;
goto v_resetjp_185_;
}
v_resetjp_185_:
{
lean_object* v___x_188_; lean_object* v___x_190_; 
v___x_188_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_solveVisibleGoals___closed__0));
if (v_isShared_187_ == 0)
{
lean_ctor_set(v___x_186_, 0, v___x_188_);
v___x_190_ = v___x_186_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v___x_188_);
lean_ctor_set(v_reuseFailAlloc_191_, 1, v_invisibleGoals_184_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___lam__0(lean_object* v_inst_194_, lean_object* v_x_195_, lean_object* v_r_196_, lean_object* v_a_197_, lean_object* v_x_198_, lean_object* v___y_199_){
_start:
{
lean_object* v_fst_200_; lean_object* v_snd_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_217_; 
v_fst_200_ = lean_ctor_get(v___y_199_, 0);
v_snd_201_ = lean_ctor_get(v___y_199_, 1);
v_isSharedCheck_217_ = !lean_is_exclusive(v___y_199_);
if (v_isSharedCheck_217_ == 0)
{
v___x_203_ = v___y_199_;
v_isShared_204_ = v_isSharedCheck_217_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_snd_201_);
lean_inc(v_fst_200_);
lean_dec(v___y_199_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_217_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_205_; uint8_t v___x_206_; 
lean_inc(v_a_197_);
v___x_205_ = lean_apply_2(v_inst_194_, v_a_197_, v_x_195_);
v___x_206_ = lean_unbox(v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; lean_object* v___x_209_; 
v___x_207_ = lean_array_push(v_snd_201_, v_a_197_);
if (v_isShared_204_ == 0)
{
lean_ctor_set(v___x_203_, 1, v___x_207_);
v___x_209_ = v___x_203_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_fst_200_);
lean_ctor_set(v_reuseFailAlloc_211_, 1, v___x_207_);
v___x_209_ = v_reuseFailAlloc_211_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
lean_object* v___x_210_; 
v___x_210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_210_, 0, v___x_209_);
return v___x_210_;
}
}
else
{
lean_object* v___x_212_; lean_object* v___x_214_; 
lean_dec(v_fst_200_);
lean_dec(v_a_197_);
v___x_212_ = l_Array_append___redArg(v_snd_201_, v_r_196_);
if (v_isShared_204_ == 0)
{
lean_ctor_set(v___x_203_, 1, v___x_212_);
lean_ctor_set(v___x_203_, 0, v___x_205_);
v___x_214_ = v___x_203_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v___x_205_);
lean_ctor_set(v_reuseFailAlloc_216_, 1, v___x_212_);
v___x_214_ = v_reuseFailAlloc_216_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
lean_object* v___x_215_; 
v___x_215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
return v___x_215_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___lam__0___boxed(lean_object* v_inst_218_, lean_object* v_x_219_, lean_object* v_r_220_, lean_object* v_a_221_, lean_object* v_x_222_, lean_object* v___y_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___lam__0(v_inst_218_, v_x_219_, v_r_220_, v_a_221_, v_x_222_, v___y_223_);
lean_dec_ref(v_r_220_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg(lean_object* v_inst_244_, lean_object* v_xs_245_, lean_object* v_x_246_, lean_object* v_r_247_){
_start:
{
lean_object* v___f_248_; lean_object* v___x_249_; uint8_t v_found_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v_ys_256_; lean_object* v___x_257_; lean_object* v___x_258_; size_t v_sz_259_; size_t v___x_260_; lean_object* v___x_261_; lean_object* v_fst_262_; uint8_t v___x_263_; 
lean_inc_ref(v_r_247_);
v___f_248_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___lam__0___boxed), 6, 3);
lean_closure_set(v___f_248_, 0, v_inst_244_);
lean_closure_set(v___f_248_, 1, v_x_246_);
lean_closure_set(v___f_248_, 2, v_r_247_);
v___x_249_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__9));
v_found_250_ = 0;
v___x_251_ = lean_array_get_size(v_xs_245_);
v___x_252_ = lean_unsigned_to_nat(1u);
v___x_253_ = lean_nat_sub(v___x_251_, v___x_252_);
v___x_254_ = lean_array_get_size(v_r_247_);
lean_dec_ref(v_r_247_);
v___x_255_ = lean_nat_add(v___x_253_, v___x_254_);
lean_dec(v___x_253_);
v_ys_256_ = lean_mk_empty_array_with_capacity(v___x_255_);
lean_dec(v___x_255_);
v___x_257_ = lean_box(v_found_250_);
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_ys_256_);
v_sz_259_ = lean_array_size(v_xs_245_);
v___x_260_ = ((size_t)0ULL);
v___x_261_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_249_, v_xs_245_, v___f_248_, v_sz_259_, v___x_260_, v___x_258_);
v_fst_262_ = lean_ctor_get(v___x_261_, 0);
lean_inc(v_fst_262_);
v___x_263_ = lean_unbox(v_fst_262_);
lean_dec(v_fst_262_);
if (v___x_263_ == 0)
{
lean_object* v___x_264_; 
lean_dec(v___x_261_);
v___x_264_ = lean_box(0);
return v___x_264_;
}
else
{
lean_object* v_snd_265_; lean_object* v___x_266_; 
v_snd_265_ = lean_ctor_get(v___x_261_, 1);
lean_inc(v_snd_265_);
lean_dec(v___x_261_);
v___x_266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_266_, 0, v_snd_265_);
return v___x_266_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray(lean_object* v_00_u03b1_267_, lean_object* v_inst_268_, lean_object* v_xs_269_, lean_object* v_x_270_, lean_object* v_r_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg(v_inst_268_, v_xs_269_, v_x_270_, v_r_271_);
return v___x_272_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_eraseSolvedGoals_mvarWasSolved(lean_object* v_preMCtx_273_, lean_object* v_postMCtx_274_, lean_object* v_mvarId_275_){
_start:
{
uint8_t v___x_276_; 
v___x_276_ = lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned(v_postMCtx_274_, v_mvarId_275_);
if (v___x_276_ == 0)
{
return v___x_276_;
}
else
{
uint8_t v___x_277_; 
v___x_277_ = lp_batteries_Lean_MetavarContext_isExprMVarAssignedOrDelayedAssigned(v_preMCtx_273_, v_mvarId_275_);
if (v___x_277_ == 0)
{
return v___x_276_;
}
else
{
uint8_t v___x_278_; 
v___x_278_ = 0;
return v___x_278_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_eraseSolvedGoals_mvarWasSolved___boxed(lean_object* v_preMCtx_279_, lean_object* v_postMCtx_280_, lean_object* v_mvarId_281_){
_start:
{
uint8_t v_res_282_; lean_object* v_r_283_; 
v_res_282_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_eraseSolvedGoals_mvarWasSolved(v_preMCtx_279_, v_postMCtx_280_, v_mvarId_281_);
lean_dec(v_mvarId_281_);
lean_dec_ref(v_postMCtx_280_);
lean_dec_ref(v_preMCtx_279_);
v_r_283_ = lean_box(v_res_282_);
return v_r_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__1(lean_object* v_preMCtx_284_, lean_object* v_postMCtx_285_, lean_object* v_as_286_, size_t v_i_287_, size_t v_stop_288_, lean_object* v_b_289_){
_start:
{
lean_object* v___y_291_; uint8_t v___x_295_; 
v___x_295_ = lean_usize_dec_eq(v_i_287_, v_stop_288_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; lean_object* v_goal_297_; uint8_t v___x_298_; 
v___x_296_ = lean_array_uget_borrowed(v_as_286_, v_i_287_);
v_goal_297_ = lean_ctor_get(v___x_296_, 0);
v___x_298_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_eraseSolvedGoals_mvarWasSolved(v_preMCtx_284_, v_postMCtx_285_, v_goal_297_);
if (v___x_298_ == 0)
{
lean_object* v___x_299_; 
lean_inc(v___x_296_);
v___x_299_ = lean_array_push(v_b_289_, v___x_296_);
v___y_291_ = v___x_299_;
goto v___jp_290_;
}
else
{
v___y_291_ = v_b_289_;
goto v___jp_290_;
}
}
else
{
return v_b_289_;
}
v___jp_290_:
{
size_t v___x_292_; size_t v___x_293_; 
v___x_292_ = ((size_t)1ULL);
v___x_293_ = lean_usize_add(v_i_287_, v___x_292_);
v_i_287_ = v___x_293_;
v_b_289_ = v___y_291_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__1___boxed(lean_object* v_preMCtx_300_, lean_object* v_postMCtx_301_, lean_object* v_as_302_, lean_object* v_i_303_, lean_object* v_stop_304_, lean_object* v_b_305_){
_start:
{
size_t v_i_boxed_306_; size_t v_stop_boxed_307_; lean_object* v_res_308_; 
v_i_boxed_306_ = lean_unbox_usize(v_i_303_);
lean_dec(v_i_303_);
v_stop_boxed_307_ = lean_unbox_usize(v_stop_304_);
lean_dec(v_stop_304_);
v_res_308_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__1(v_preMCtx_300_, v_postMCtx_301_, v_as_302_, v_i_boxed_306_, v_stop_boxed_307_, v_b_305_);
lean_dec_ref(v_as_302_);
lean_dec_ref(v_postMCtx_301_);
lean_dec_ref(v_preMCtx_300_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__2(lean_object* v_as_309_, size_t v_i_310_, size_t v_stop_311_, lean_object* v_b_312_){
_start:
{
uint8_t v___x_313_; 
v___x_313_ = lean_usize_dec_eq(v_i_310_, v_stop_311_);
if (v___x_313_ == 0)
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; size_t v___x_317_; size_t v___x_318_; 
v___x_314_ = lean_array_uget_borrowed(v_as_309_, v_i_310_);
v___x_315_ = l_Std_DHashMap_Internal_AssocList_length___redArg(v___x_314_);
v___x_316_ = lean_nat_add(v_b_312_, v___x_315_);
lean_dec(v___x_315_);
lean_dec(v_b_312_);
v___x_317_ = ((size_t)1ULL);
v___x_318_ = lean_usize_add(v_i_310_, v___x_317_);
v_i_310_ = v___x_318_;
v_b_312_ = v___x_316_;
goto _start;
}
else
{
return v_b_312_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__2___boxed(lean_object* v_as_320_, lean_object* v_i_321_, lean_object* v_stop_322_, lean_object* v_b_323_){
_start:
{
size_t v_i_boxed_324_; size_t v_stop_boxed_325_; lean_object* v_res_326_; 
v_i_boxed_324_ = lean_unbox_usize(v_i_321_);
lean_dec(v_i_321_);
v_stop_boxed_325_ = lean_unbox_usize(v_stop_322_);
lean_dec(v_stop_322_);
v_res_326_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__2(v_as_320_, v_i_boxed_324_, v_stop_boxed_325_, v_b_323_);
lean_dec_ref(v_as_320_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_filter_go___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__0(lean_object* v_preMCtx_327_, lean_object* v_postMCtx_328_, lean_object* v_acc_329_, lean_object* v_a_330_){
_start:
{
if (lean_obj_tag(v_a_330_) == 0)
{
return v_acc_329_;
}
else
{
lean_object* v_key_331_; lean_object* v_value_332_; lean_object* v_tail_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_343_; 
v_key_331_ = lean_ctor_get(v_a_330_, 0);
v_value_332_ = lean_ctor_get(v_a_330_, 1);
v_tail_333_ = lean_ctor_get(v_a_330_, 2);
v_isSharedCheck_343_ = !lean_is_exclusive(v_a_330_);
if (v_isSharedCheck_343_ == 0)
{
v___x_335_ = v_a_330_;
v_isShared_336_ = v_isSharedCheck_343_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_tail_333_);
lean_inc(v_value_332_);
lean_inc(v_key_331_);
lean_dec(v_a_330_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_343_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
uint8_t v___x_337_; 
v___x_337_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_eraseSolvedGoals_mvarWasSolved(v_preMCtx_327_, v_postMCtx_328_, v_key_331_);
if (v___x_337_ == 0)
{
lean_object* v___x_339_; 
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 2, v_acc_329_);
v___x_339_ = v___x_335_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_key_331_);
lean_ctor_set(v_reuseFailAlloc_341_, 1, v_value_332_);
lean_ctor_set(v_reuseFailAlloc_341_, 2, v_acc_329_);
v___x_339_ = v_reuseFailAlloc_341_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
v_acc_329_ = v___x_339_;
v_a_330_ = v_tail_333_;
goto _start;
}
}
else
{
lean_del_object(v___x_335_);
lean_dec(v_value_332_);
lean_dec(v_key_331_);
v_a_330_ = v_tail_333_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_filter_go___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__0___boxed(lean_object* v_preMCtx_344_, lean_object* v_postMCtx_345_, lean_object* v_acc_346_, lean_object* v_a_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_filter_go___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__0(v_preMCtx_344_, v_postMCtx_345_, v_acc_346_, v_a_347_);
lean_dec_ref(v_postMCtx_345_);
lean_dec_ref(v_preMCtx_344_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__1(lean_object* v_preMCtx_349_, lean_object* v_postMCtx_350_, size_t v_sz_351_, size_t v_i_352_, lean_object* v_bs_353_){
_start:
{
uint8_t v___x_354_; 
v___x_354_ = lean_usize_dec_lt(v_i_352_, v_sz_351_);
if (v___x_354_ == 0)
{
return v_bs_353_;
}
else
{
lean_object* v_v_355_; lean_object* v___x_356_; lean_object* v_bs_x27_357_; lean_object* v___x_358_; lean_object* v___x_359_; size_t v___x_360_; size_t v___x_361_; lean_object* v___x_362_; 
v_v_355_ = lean_array_uget(v_bs_353_, v_i_352_);
v___x_356_ = lean_unsigned_to_nat(0u);
v_bs_x27_357_ = lean_array_uset(v_bs_353_, v_i_352_, v___x_356_);
v___x_358_ = lean_box(0);
v___x_359_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_filter_go___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__0(v_preMCtx_349_, v_postMCtx_350_, v___x_358_, v_v_355_);
v___x_360_ = ((size_t)1ULL);
v___x_361_ = lean_usize_add(v_i_352_, v___x_360_);
v___x_362_ = lean_array_uset(v_bs_x27_357_, v_i_352_, v___x_359_);
v_i_352_ = v___x_361_;
v_bs_353_ = v___x_362_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__1___boxed(lean_object* v_preMCtx_364_, lean_object* v_postMCtx_365_, lean_object* v_sz_366_, lean_object* v_i_367_, lean_object* v_bs_368_){
_start:
{
size_t v_sz_boxed_369_; size_t v_i_boxed_370_; lean_object* v_res_371_; 
v_sz_boxed_369_ = lean_unbox_usize(v_sz_366_);
lean_dec(v_sz_366_);
v_i_boxed_370_ = lean_unbox_usize(v_i_367_);
lean_dec(v_i_367_);
v_res_371_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__1(v_preMCtx_364_, v_postMCtx_365_, v_sz_boxed_369_, v_i_boxed_370_, v_bs_368_);
lean_dec_ref(v_postMCtx_365_);
lean_dec_ref(v_preMCtx_364_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0(lean_object* v_preMCtx_372_, lean_object* v_postMCtx_373_, lean_object* v_m_374_){
_start:
{
lean_object* v_buckets_375_; lean_object* v___x_377_; uint8_t v_isShared_378_; uint8_t v_isSharedCheck_402_; 
v_buckets_375_ = lean_ctor_get(v_m_374_, 1);
v_isSharedCheck_402_ = !lean_is_exclusive(v_m_374_);
if (v_isSharedCheck_402_ == 0)
{
lean_object* v_unused_403_; 
v_unused_403_ = lean_ctor_get(v_m_374_, 0);
lean_dec(v_unused_403_);
v___x_377_ = v_m_374_;
v_isShared_378_ = v_isSharedCheck_402_;
goto v_resetjp_376_;
}
else
{
lean_inc(v_buckets_375_);
lean_dec(v_m_374_);
v___x_377_ = lean_box(0);
v_isShared_378_ = v_isSharedCheck_402_;
goto v_resetjp_376_;
}
v_resetjp_376_:
{
size_t v_sz_379_; size_t v___x_380_; lean_object* v_newBuckets_381_; lean_object* v___x_382_; lean_object* v___x_383_; uint8_t v___x_384_; 
v_sz_379_ = lean_array_size(v_buckets_375_);
v___x_380_ = ((size_t)0ULL);
v_newBuckets_381_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__1(v_preMCtx_372_, v_postMCtx_373_, v_sz_379_, v___x_380_, v_buckets_375_);
v___x_382_ = lean_unsigned_to_nat(0u);
v___x_383_ = lean_array_get_size(v_newBuckets_381_);
v___x_384_ = lean_nat_dec_lt(v___x_382_, v___x_383_);
if (v___x_384_ == 0)
{
lean_object* v___x_386_; 
if (v_isShared_378_ == 0)
{
lean_ctor_set(v___x_377_, 1, v_newBuckets_381_);
lean_ctor_set(v___x_377_, 0, v___x_382_);
v___x_386_ = v___x_377_;
goto v_reusejp_385_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v___x_382_);
lean_ctor_set(v_reuseFailAlloc_387_, 1, v_newBuckets_381_);
v___x_386_ = v_reuseFailAlloc_387_;
goto v_reusejp_385_;
}
v_reusejp_385_:
{
return v___x_386_;
}
}
else
{
uint8_t v___x_388_; 
v___x_388_ = lean_nat_dec_le(v___x_383_, v___x_383_);
if (v___x_388_ == 0)
{
if (v___x_384_ == 0)
{
lean_object* v___x_390_; 
if (v_isShared_378_ == 0)
{
lean_ctor_set(v___x_377_, 1, v_newBuckets_381_);
lean_ctor_set(v___x_377_, 0, v___x_382_);
v___x_390_ = v___x_377_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v___x_382_);
lean_ctor_set(v_reuseFailAlloc_391_, 1, v_newBuckets_381_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
}
}
else
{
size_t v___x_392_; lean_object* v___x_393_; lean_object* v___x_395_; 
v___x_392_ = lean_usize_of_nat(v___x_383_);
v___x_393_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__2(v_newBuckets_381_, v___x_380_, v___x_392_, v___x_382_);
if (v_isShared_378_ == 0)
{
lean_ctor_set(v___x_377_, 1, v_newBuckets_381_);
lean_ctor_set(v___x_377_, 0, v___x_393_);
v___x_395_ = v___x_377_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_396_; 
v_reuseFailAlloc_396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_396_, 0, v___x_393_);
lean_ctor_set(v_reuseFailAlloc_396_, 1, v_newBuckets_381_);
v___x_395_ = v_reuseFailAlloc_396_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
return v___x_395_;
}
}
}
else
{
size_t v___x_397_; lean_object* v___x_398_; lean_object* v___x_400_; 
v___x_397_ = lean_usize_of_nat(v___x_383_);
v___x_398_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0_spec__2(v_newBuckets_381_, v___x_380_, v___x_397_, v___x_382_);
if (v_isShared_378_ == 0)
{
lean_ctor_set(v___x_377_, 1, v_newBuckets_381_);
lean_ctor_set(v___x_377_, 0, v___x_398_);
v___x_400_ = v___x_377_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v___x_398_);
lean_ctor_set(v_reuseFailAlloc_401_, 1, v_newBuckets_381_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0___boxed(lean_object* v_preMCtx_404_, lean_object* v_postMCtx_405_, lean_object* v_m_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0(v_preMCtx_404_, v_postMCtx_405_, v_m_406_);
lean_dec_ref(v_postMCtx_405_);
lean_dec_ref(v_preMCtx_404_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(lean_object* v_ts_408_, lean_object* v_preMCtx_409_, lean_object* v_postMCtx_410_){
_start:
{
lean_object* v_visibleGoals_411_; lean_object* v_invisibleGoals_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_433_; 
v_visibleGoals_411_ = lean_ctor_get(v_ts_408_, 0);
v_invisibleGoals_412_ = lean_ctor_get(v_ts_408_, 1);
v_isSharedCheck_433_ = !lean_is_exclusive(v_ts_408_);
if (v_isSharedCheck_433_ == 0)
{
v___x_414_ = v_ts_408_;
v_isShared_415_ = v_isSharedCheck_433_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_invisibleGoals_412_);
lean_inc(v_visibleGoals_411_);
lean_dec(v_ts_408_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_433_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___y_417_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; uint8_t v___x_425_; 
v___x_422_ = lean_unsigned_to_nat(0u);
v___x_423_ = lean_array_get_size(v_visibleGoals_411_);
v___x_424_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_solveVisibleGoals___closed__0));
v___x_425_ = lean_nat_dec_lt(v___x_422_, v___x_423_);
if (v___x_425_ == 0)
{
lean_dec_ref(v_visibleGoals_411_);
v___y_417_ = v___x_424_;
goto v___jp_416_;
}
else
{
uint8_t v___x_426_; 
v___x_426_ = lean_nat_dec_le(v___x_423_, v___x_423_);
if (v___x_426_ == 0)
{
if (v___x_425_ == 0)
{
lean_dec_ref(v_visibleGoals_411_);
v___y_417_ = v___x_424_;
goto v___jp_416_;
}
else
{
size_t v___x_427_; size_t v___x_428_; lean_object* v___x_429_; 
v___x_427_ = ((size_t)0ULL);
v___x_428_ = lean_usize_of_nat(v___x_423_);
v___x_429_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__1(v_preMCtx_409_, v_postMCtx_410_, v_visibleGoals_411_, v___x_427_, v___x_428_, v___x_424_);
lean_dec_ref(v_visibleGoals_411_);
v___y_417_ = v___x_429_;
goto v___jp_416_;
}
}
else
{
size_t v___x_430_; size_t v___x_431_; lean_object* v___x_432_; 
v___x_430_ = ((size_t)0ULL);
v___x_431_ = lean_usize_of_nat(v___x_423_);
v___x_432_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__1(v_preMCtx_409_, v_postMCtx_410_, v_visibleGoals_411_, v___x_430_, v___x_431_, v___x_424_);
lean_dec_ref(v_visibleGoals_411_);
v___y_417_ = v___x_432_;
goto v___jp_416_;
}
}
v___jp_416_:
{
lean_object* v___x_418_; lean_object* v___x_420_; 
v___x_418_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_filter___at___00Aesop_Script_TacticState_eraseSolvedGoals_spec__0(v_preMCtx_409_, v_postMCtx_410_, v_invisibleGoals_412_);
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 1, v___x_418_);
lean_ctor_set(v___x_414_, 0, v___y_417_);
v___x_420_ = v___x_414_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v___y_417_);
lean_ctor_set(v_reuseFailAlloc_421_, 1, v___x_418_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
return v___x_420_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals___boxed(lean_object* v_ts_434_, lean_object* v_preMCtx_435_, lean_object* v_postMCtx_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(v_ts_434_, v_preMCtx_435_, v_postMCtx_436_);
lean_dec_ref(v_postMCtx_436_);
lean_dec_ref(v_preMCtx_435_);
return v_res_437_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__3(void){
_start:
{
lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_442_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__2));
v___x_443_ = l_Lean_MessageData_ofFormat(v___x_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg(lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_ts_446_, lean_object* v_inGoal_447_, lean_object* v_outGoals_448_, lean_object* v_preMCtx_449_, lean_object* v_postMCtx_450_){
_start:
{
lean_object* v_toApplicative_451_; lean_object* v_toPure_452_; lean_object* v_visibleGoals_453_; lean_object* v_invisibleGoals_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_470_; 
v_toApplicative_451_ = lean_ctor_get(v_inst_444_, 0);
v_toPure_452_ = lean_ctor_get(v_toApplicative_451_, 1);
v_visibleGoals_453_ = lean_ctor_get(v_ts_446_, 0);
v_invisibleGoals_454_ = lean_ctor_get(v_ts_446_, 1);
v_isSharedCheck_470_ = !lean_is_exclusive(v_ts_446_);
if (v_isSharedCheck_470_ == 0)
{
v___x_456_ = v_ts_446_;
v_isShared_457_ = v_isSharedCheck_470_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_invisibleGoals_454_);
lean_inc(v_visibleGoals_453_);
lean_dec(v_ts_446_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_470_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v___f_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
v___f_458_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__0));
v___x_459_ = lean_obj_once(&lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2, &lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2_once, _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2);
lean_inc(v_inGoal_447_);
v___x_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_460_, 0, v_inGoal_447_);
lean_ctor_set(v___x_460_, 1, v___x_459_);
v___x_461_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg(v___f_458_, v_visibleGoals_453_, v___x_460_, v_outGoals_448_);
if (lean_obj_tag(v___x_461_) == 1)
{
lean_object* v_val_462_; lean_object* v_ts_464_; 
lean_inc(v_toPure_452_);
lean_dec(v_inGoal_447_);
lean_dec_ref(v_inst_445_);
lean_dec_ref(v_inst_444_);
v_val_462_ = lean_ctor_get(v___x_461_, 0);
lean_inc(v_val_462_);
lean_dec_ref_known(v___x_461_, 1);
if (v_isShared_457_ == 0)
{
lean_ctor_set(v___x_456_, 0, v_val_462_);
v_ts_464_ = v___x_456_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v_val_462_);
lean_ctor_set(v_reuseFailAlloc_467_, 1, v_invisibleGoals_454_);
v_ts_464_ = v_reuseFailAlloc_467_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_465_ = lp_aesop_Aesop_Script_TacticState_eraseSolvedGoals(v_ts_464_, v_preMCtx_449_, v_postMCtx_450_);
v___x_466_ = lean_apply_2(v_toPure_452_, lean_box(0), v___x_465_);
return v___x_466_;
}
}
else
{
lean_object* v___x_468_; lean_object* v___x_469_; 
lean_dec(v___x_461_);
lean_del_object(v___x_456_);
lean_dec_ref(v_invisibleGoals_454_);
v___x_468_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__3, &lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__3_once, _init_lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___closed__3);
v___x_469_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg(v_inst_444_, v_inst_445_, v_inGoal_447_, v___x_468_);
return v___x_469_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___redArg___boxed(lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_ts_473_, lean_object* v_inGoal_474_, lean_object* v_outGoals_475_, lean_object* v_preMCtx_476_, lean_object* v_postMCtx_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_aesop_Aesop_Script_TacticState_applyTactic___redArg(v_inst_471_, v_inst_472_, v_ts_473_, v_inGoal_474_, v_outGoals_475_, v_preMCtx_476_, v_postMCtx_477_);
lean_dec_ref(v_postMCtx_477_);
lean_dec_ref(v_preMCtx_476_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic(lean_object* v_m_479_, lean_object* v_inst_480_, lean_object* v_inst_481_, lean_object* v_ts_482_, lean_object* v_inGoal_483_, lean_object* v_outGoals_484_, lean_object* v_preMCtx_485_, lean_object* v_postMCtx_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_aesop_Aesop_Script_TacticState_applyTactic___redArg(v_inst_480_, v_inst_481_, v_ts_482_, v_inGoal_483_, v_outGoals_484_, v_preMCtx_485_, v_postMCtx_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_applyTactic___boxed(lean_object* v_m_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_ts_491_, lean_object* v_inGoal_492_, lean_object* v_outGoals_493_, lean_object* v_preMCtx_494_, lean_object* v_postMCtx_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_aesop_Aesop_Script_TacticState_applyTactic(v_m_488_, v_inst_489_, v_inst_490_, v_ts_491_, v_inGoal_492_, v_outGoals_493_, v_preMCtx_494_, v_postMCtx_495_);
lean_dec_ref(v_postMCtx_495_);
lean_dec_ref(v_preMCtx_494_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__0(lean_object* v_goal_497_, lean_object* v___x_498_, lean_object* v___x_499_, lean_object* v_a_500_, lean_object* v_x_501_, lean_object* v___y_502_){
_start:
{
lean_object* v_goal_503_; uint8_t v___x_504_; 
v_goal_503_ = lean_ctor_get(v_a_500_, 0);
v___x_504_ = l_Lean_instBEqMVarId_beq(v_goal_503_, v_goal_497_);
if (v___x_504_ == 0)
{
lean_object* v___x_505_; 
lean_dec_ref(v_a_500_);
v___x_505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_505_, 0, v___x_498_);
return v___x_505_;
}
else
{
lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
lean_dec_ref(v___x_498_);
v___x_506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_506_, 0, v_a_500_);
v___x_507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_507_, 0, v___x_506_);
v___x_508_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_508_, 0, v___x_507_);
lean_ctor_set(v___x_508_, 1, v___x_499_);
v___x_509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_509_, 0, v___x_508_);
return v___x_509_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__0___boxed(lean_object* v_goal_510_, lean_object* v___x_511_, lean_object* v___x_512_, lean_object* v_a_513_, lean_object* v_x_514_, lean_object* v___y_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__0(v_goal_510_, v___x_511_, v___x_512_, v_a_513_, v_x_514_, v___y_515_);
lean_dec_ref(v___y_515_);
lean_dec(v_goal_510_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1(lean_object* v_goal_518_, lean_object* v___x_519_, lean_object* v_toPure_520_, lean_object* v_a_521_, lean_object* v_x_522_, lean_object* v___y_523_){
_start:
{
lean_object* v_goal_524_; uint8_t v___x_525_; 
v_goal_524_ = lean_ctor_get(v_a_521_, 0);
lean_inc(v_goal_524_);
lean_dec_ref(v_a_521_);
v___x_525_ = l_Lean_instBEqMVarId_beq(v_goal_524_, v_goal_518_);
if (v___x_525_ == 0)
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_526_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___closed__0));
v___x_527_ = lean_box(0);
v___x_528_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_519_, v___x_526_, v___y_523_, v_goal_524_, v___x_527_);
v___x_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_529_, 0, v___x_528_);
v___x_530_ = lean_apply_2(v_toPure_520_, lean_box(0), v___x_529_);
return v___x_530_;
}
else
{
lean_object* v___x_531_; lean_object* v___x_532_; 
lean_dec(v_goal_524_);
lean_dec_ref(v___x_519_);
v___x_531_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_531_, 0, v___y_523_);
v___x_532_ = lean_apply_2(v_toPure_520_, lean_box(0), v___x_531_);
return v___x_532_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___boxed(lean_object* v_goal_533_, lean_object* v___x_534_, lean_object* v_toPure_535_, lean_object* v_a_536_, lean_object* v_x_537_, lean_object* v___y_538_){
_start:
{
lean_object* v_res_539_; 
v_res_539_ = lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1(v_goal_533_, v___x_534_, v_toPure_535_, v_a_536_, v_x_537_, v___y_538_);
lean_dec(v_goal_533_);
return v_res_539_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__2(lean_object* v_val_540_, lean_object* v_toPure_541_, lean_object* v_____s_542_){
_start:
{
lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_543_ = lean_unsigned_to_nat(1u);
v___x_544_ = lean_mk_empty_array_with_capacity(v___x_543_);
v___x_545_ = lean_array_push(v___x_544_, v_val_540_);
v___x_546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_546_, 0, v___x_545_);
lean_ctor_set(v___x_546_, 1, v_____s_542_);
v___x_547_ = lean_apply_2(v_toPure_541_, lean_box(0), v___x_546_);
return v___x_547_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__2(void){
_start:
{
lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_551_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__1));
v___x_552_ = l_Lean_MessageData_ofFormat(v___x_551_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus___redArg(lean_object* v_inst_557_, lean_object* v_inst_558_, lean_object* v_ts_559_, lean_object* v_goal_560_){
_start:
{
lean_object* v_toApplicative_564_; lean_object* v_toBind_565_; lean_object* v_toPure_566_; lean_object* v_visibleGoals_567_; lean_object* v_invisibleGoals_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___f_572_; size_t v_sz_573_; size_t v___x_574_; lean_object* v___x_575_; lean_object* v_fst_576_; 
v_toApplicative_564_ = lean_ctor_get(v_inst_557_, 0);
v_toBind_565_ = lean_ctor_get(v_inst_557_, 1);
v_toPure_566_ = lean_ctor_get(v_toApplicative_564_, 1);
v_visibleGoals_567_ = lean_ctor_get(v_ts_559_, 0);
lean_inc_ref_n(v_visibleGoals_567_, 2);
v_invisibleGoals_568_ = lean_ctor_get(v_ts_559_, 1);
lean_inc_ref(v_invisibleGoals_568_);
lean_dec_ref(v_ts_559_);
v___x_569_ = ((lean_object*)(lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_replaceWithArray___redArg___closed__9));
v___x_570_ = lean_box(0);
v___x_571_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__3));
lean_inc(v_goal_560_);
v___f_572_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__0___boxed), 6, 3);
lean_closure_set(v___f_572_, 0, v_goal_560_);
lean_closure_set(v___f_572_, 1, v___x_571_);
lean_closure_set(v___f_572_, 2, v___x_570_);
v_sz_573_ = lean_array_size(v_visibleGoals_567_);
v___x_574_ = ((size_t)0ULL);
v___x_575_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_569_, v_visibleGoals_567_, v___f_572_, v_sz_573_, v___x_574_, v___x_571_);
v_fst_576_ = lean_ctor_get(v___x_575_, 0);
lean_inc(v_fst_576_);
lean_dec(v___x_575_);
if (lean_obj_tag(v_fst_576_) == 0)
{
lean_dec_ref(v_invisibleGoals_568_);
lean_dec_ref(v_visibleGoals_567_);
goto v___jp_561_;
}
else
{
lean_object* v_val_577_; 
v_val_577_ = lean_ctor_get(v_fst_576_, 0);
lean_inc(v_val_577_);
lean_dec_ref_known(v_fst_576_, 1);
if (lean_obj_tag(v_val_577_) == 1)
{
lean_object* v_val_578_; lean_object* v___x_579_; lean_object* v___f_580_; lean_object* v___f_581_; lean_object* v___x_582_; lean_object* v___x_583_; 
lean_inc(v_toBind_565_);
lean_dec_ref(v_inst_558_);
v_val_578_ = lean_ctor_get(v_val_577_, 0);
lean_inc(v_val_578_);
lean_dec_ref_known(v_val_577_, 1);
v___x_579_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__4));
lean_inc_n(v_toPure_566_, 2);
v___f_580_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___boxed), 6, 3);
lean_closure_set(v___f_580_, 0, v_goal_560_);
lean_closure_set(v___f_580_, 1, v___x_579_);
lean_closure_set(v___f_580_, 2, v_toPure_566_);
v___f_581_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__2), 3, 2);
lean_closure_set(v___f_581_, 0, v_val_578_);
lean_closure_set(v___f_581_, 1, v_toPure_566_);
v___x_582_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_557_, v_visibleGoals_567_, v___f_580_, v_sz_573_, v___x_574_, v_invisibleGoals_568_);
v___x_583_ = lean_apply_4(v_toBind_565_, lean_box(0), lean_box(0), v___x_582_, v___f_581_);
return v___x_583_;
}
else
{
lean_dec(v_val_577_);
lean_dec_ref(v_invisibleGoals_568_);
lean_dec_ref(v_visibleGoals_567_);
goto v___jp_561_;
}
}
v___jp_561_:
{
lean_object* v___x_562_; lean_object* v___x_563_; 
v___x_562_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__2, &lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__2_once, _init_lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__2);
v___x_563_ = lp_aesop___private_Aesop_Script_TacticState_0__Aesop_Script_TacticState_throwUnknownGoalError___redArg(v_inst_557_, v_inst_558_, v_goal_560_, v___x_562_);
return v___x_563_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_focus(lean_object* v_m_584_, lean_object* v_inst_585_, lean_object* v_inst_586_, lean_object* v_ts_587_, lean_object* v_goal_588_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lp_aesop_Aesop_Script_TacticState_focus___redArg(v_inst_585_, v_inst_586_, v_ts_587_, v_goal_588_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__0(lean_object* v_g_590_, lean_object* v_snd_591_, lean_object* v___x_592_, lean_object* v___x_593_, lean_object* v_toPure_594_, lean_object* v_a_595_, lean_object* v_x_596_, lean_object* v___y_597_){
_start:
{
lean_object* v_goal_598_; uint8_t v___x_599_; 
v_goal_598_ = lean_ctor_get(v_a_595_, 0);
v___x_599_ = l_Lean_instBEqMVarId_beq(v_goal_598_, v_g_590_);
if (v___x_599_ == 0)
{
lean_object* v_invisibleGoals_600_; uint8_t v___x_601_; 
v_invisibleGoals_600_ = lean_ctor_get(v_snd_591_, 1);
lean_inc(v_goal_598_);
v___x_601_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v___x_592_, v___x_593_, v_invisibleGoals_600_, v_goal_598_);
if (v___x_601_ == 0)
{
lean_object* v___x_602_; lean_object* v___x_603_; 
lean_dec_ref(v_a_595_);
v___x_602_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_602_, 0, v___y_597_);
v___x_603_ = lean_apply_2(v_toPure_594_, lean_box(0), v___x_602_);
return v___x_603_;
}
else
{
lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; 
v___x_604_ = lean_array_push(v___y_597_, v_a_595_);
v___x_605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_605_, 0, v___x_604_);
v___x_606_ = lean_apply_2(v_toPure_594_, lean_box(0), v___x_605_);
return v___x_606_;
}
}
else
{
lean_object* v_visibleGoals_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
lean_dec_ref(v_a_595_);
lean_dec_ref(v___x_593_);
lean_dec_ref(v___x_592_);
v_visibleGoals_607_ = lean_ctor_get(v_snd_591_, 0);
v___x_608_ = l_Array_append___redArg(v___y_597_, v_visibleGoals_607_);
v___x_609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_609_, 0, v___x_608_);
v___x_610_ = lean_apply_2(v_toPure_594_, lean_box(0), v___x_609_);
return v___x_610_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__0___boxed(lean_object* v_g_611_, lean_object* v_snd_612_, lean_object* v___x_613_, lean_object* v___x_614_, lean_object* v_toPure_615_, lean_object* v_a_616_, lean_object* v_x_617_, lean_object* v___y_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__0(v_g_611_, v_snd_612_, v___x_613_, v___x_614_, v_toPure_615_, v_a_616_, v_x_617_, v___y_618_);
lean_dec_ref(v_snd_612_);
lean_dec(v_g_611_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__1(lean_object* v_snd_620_, lean_object* v___x_621_, lean_object* v___x_622_, lean_object* v_toPure_623_, lean_object* v_a_624_, lean_object* v_x_625_, lean_object* v_acc_626_){
_start:
{
lean_object* v_invisibleGoals_627_; uint8_t v___x_628_; 
v_invisibleGoals_627_ = lean_ctor_get(v_snd_620_, 1);
lean_inc(v_a_624_);
lean_inc_ref(v___x_622_);
lean_inc_ref(v___x_621_);
v___x_628_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v___x_621_, v___x_622_, v_invisibleGoals_627_, v_a_624_);
if (v___x_628_ == 0)
{
lean_object* v___x_629_; lean_object* v___x_630_; 
lean_dec(v_a_624_);
lean_dec_ref(v___x_622_);
lean_dec_ref(v___x_621_);
v___x_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_629_, 0, v_acc_626_);
v___x_630_ = lean_apply_2(v_toPure_623_, lean_box(0), v___x_629_);
return v___x_630_;
}
else
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; 
v___x_631_ = lean_box(0);
v___x_632_ = l_Std_DHashMap_Internal_Raw_u2080_insertIfNew___redArg(v___x_621_, v___x_622_, v_acc_626_, v_a_624_, v___x_631_);
v___x_633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_633_, 0, v___x_632_);
v___x_634_ = lean_apply_2(v_toPure_623_, lean_box(0), v___x_633_);
return v___x_634_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__1___boxed(lean_object* v_snd_635_, lean_object* v___x_636_, lean_object* v___x_637_, lean_object* v_toPure_638_, lean_object* v_a_639_, lean_object* v_x_640_, lean_object* v_acc_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__1(v_snd_635_, v___x_636_, v___x_637_, v_toPure_638_, v_a_639_, v_x_640_, v_acc_641_);
lean_dec_ref(v_snd_635_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__2(lean_object* v_inst_643_, lean_object* v___f_644_, lean_object* v_a_645_, lean_object* v_x_646_, lean_object* v___y_647_){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = l___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go(lean_box(0), lean_box(0), lean_box(0), lean_box(0), v_inst_643_, v___f_644_, v_a_645_, v___y_647_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__3(lean_object* v_____s_649_, lean_object* v_fst_650_, lean_object* v_toPure_651_, lean_object* v_____s_652_){
_start:
{
lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v___x_653_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_653_, 0, v_____s_649_);
lean_ctor_set(v___x_653_, 1, v_____s_652_);
v___x_654_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_654_, 0, v_fst_650_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
v___x_655_ = lean_apply_2(v_toPure_651_, lean_box(0), v___x_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__4(lean_object* v_invisibleGoals_656_, lean_object* v_fst_657_, lean_object* v_toPure_658_, lean_object* v_inst_659_, lean_object* v___f_660_, lean_object* v_toBind_661_, lean_object* v_____s_662_){
_start:
{
lean_object* v_buckets_663_; lean_object* v___f_664_; lean_object* v_invisibleGoals_665_; size_t v_sz_666_; size_t v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; 
v_buckets_663_ = lean_ctor_get(v_invisibleGoals_656_, 1);
lean_inc_ref(v_buckets_663_);
lean_dec_ref(v_invisibleGoals_656_);
v___f_664_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__3), 4, 3);
lean_closure_set(v___f_664_, 0, v_____s_662_);
lean_closure_set(v___f_664_, 1, v_fst_657_);
lean_closure_set(v___f_664_, 2, v_toPure_658_);
v_invisibleGoals_665_ = lean_obj_once(&lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2, &lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2_once, _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default___closed__2);
v_sz_666_ = lean_array_size(v_buckets_663_);
v___x_667_ = ((size_t)0ULL);
v___x_668_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_659_, v_buckets_663_, v___f_660_, v_sz_666_, v___x_667_, v_invisibleGoals_665_);
v___x_669_ = lean_apply_4(v_toBind_661_, lean_box(0), lean_box(0), v___x_668_, v___f_664_);
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__5(lean_object* v_ts_670_, lean_object* v_g_671_, lean_object* v___x_672_, lean_object* v___x_673_, lean_object* v_toPure_674_, lean_object* v_inst_675_, lean_object* v_toBind_676_, lean_object* v_____x_677_){
_start:
{
lean_object* v_fst_678_; lean_object* v_snd_679_; lean_object* v_visibleGoals_680_; lean_object* v_invisibleGoals_681_; lean_object* v___f_682_; lean_object* v___f_683_; lean_object* v___f_684_; lean_object* v___f_685_; lean_object* v_visibleGoals_686_; size_t v_sz_687_; size_t v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v_fst_678_ = lean_ctor_get(v_____x_677_, 0);
lean_inc(v_fst_678_);
v_snd_679_ = lean_ctor_get(v_____x_677_, 1);
lean_inc_n(v_snd_679_, 2);
lean_dec_ref(v_____x_677_);
v_visibleGoals_680_ = lean_ctor_get(v_ts_670_, 0);
lean_inc_ref(v_visibleGoals_680_);
v_invisibleGoals_681_ = lean_ctor_get(v_ts_670_, 1);
lean_inc_ref(v_invisibleGoals_681_);
lean_dec_ref(v_ts_670_);
lean_inc_n(v_toPure_674_, 2);
lean_inc_ref(v___x_673_);
lean_inc_ref(v___x_672_);
v___f_682_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__0___boxed), 8, 5);
lean_closure_set(v___f_682_, 0, v_g_671_);
lean_closure_set(v___f_682_, 1, v_snd_679_);
lean_closure_set(v___f_682_, 2, v___x_672_);
lean_closure_set(v___f_682_, 3, v___x_673_);
lean_closure_set(v___f_682_, 4, v_toPure_674_);
v___f_683_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__1___boxed), 7, 4);
lean_closure_set(v___f_683_, 0, v_snd_679_);
lean_closure_set(v___f_683_, 1, v___x_672_);
lean_closure_set(v___f_683_, 2, v___x_673_);
lean_closure_set(v___f_683_, 3, v_toPure_674_);
lean_inc_ref_n(v_inst_675_, 2);
v___f_684_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__2), 5, 2);
lean_closure_set(v___f_684_, 0, v_inst_675_);
lean_closure_set(v___f_684_, 1, v___f_683_);
lean_inc(v_toBind_676_);
v___f_685_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__4), 7, 6);
lean_closure_set(v___f_685_, 0, v_invisibleGoals_681_);
lean_closure_set(v___f_685_, 1, v_fst_678_);
lean_closure_set(v___f_685_, 2, v_toPure_674_);
lean_closure_set(v___f_685_, 3, v_inst_675_);
lean_closure_set(v___f_685_, 4, v___f_684_);
lean_closure_set(v___f_685_, 5, v_toBind_676_);
v_visibleGoals_686_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_solveVisibleGoals___closed__0));
v_sz_687_ = lean_array_size(v_visibleGoals_680_);
v___x_688_ = ((size_t)0ULL);
v___x_689_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_675_, v_visibleGoals_680_, v___f_682_, v_sz_687_, v___x_688_, v_visibleGoals_686_);
v___x_690_ = lean_apply_4(v_toBind_676_, lean_box(0), lean_box(0), v___x_689_, v___f_685_);
return v___x_690_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__6(lean_object* v_f_691_, lean_object* v_toBind_692_, lean_object* v___f_693_, lean_object* v_____do__lift_694_){
_start:
{
lean_object* v___x_695_; lean_object* v___x_696_; 
v___x_695_ = lean_apply_1(v_f_691_, v_____do__lift_694_);
v___x_696_ = lean_apply_4(v_toBind_692_, lean_box(0), lean_box(0), v___x_695_, v___f_693_);
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM___redArg(lean_object* v_inst_697_, lean_object* v_inst_698_, lean_object* v_ts_699_, lean_object* v_g_700_, lean_object* v_f_701_){
_start:
{
lean_object* v_toApplicative_702_; lean_object* v_toBind_703_; lean_object* v_toPure_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___f_708_; lean_object* v___f_709_; lean_object* v___x_710_; 
v_toApplicative_702_ = lean_ctor_get(v_inst_697_, 0);
v_toBind_703_ = lean_ctor_get(v_inst_697_, 1);
lean_inc_n(v_toBind_703_, 3);
v_toPure_704_ = lean_ctor_get(v_toApplicative_702_, 1);
lean_inc(v_toPure_704_);
v___x_705_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__4));
v___x_706_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___closed__0));
lean_inc(v_g_700_);
lean_inc_ref(v_ts_699_);
lean_inc_ref(v_inst_697_);
v___x_707_ = lp_aesop_Aesop_Script_TacticState_focus___redArg(v_inst_697_, v_inst_698_, v_ts_699_, v_g_700_);
v___f_708_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__5), 8, 7);
lean_closure_set(v___f_708_, 0, v_ts_699_);
lean_closure_set(v___f_708_, 1, v_g_700_);
lean_closure_set(v___f_708_, 2, v___x_705_);
lean_closure_set(v___f_708_, 3, v___x_706_);
lean_closure_set(v___f_708_, 4, v_toPure_704_);
lean_closure_set(v___f_708_, 5, v_inst_697_);
lean_closure_set(v___f_708_, 6, v_toBind_703_);
v___f_709_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__6), 4, 3);
lean_closure_set(v___f_709_, 0, v_f_701_);
lean_closure_set(v___f_709_, 1, v_toBind_703_);
lean_closure_set(v___f_709_, 2, v___f_708_);
v___x_710_ = lean_apply_4(v_toBind_703_, lean_box(0), lean_box(0), v___x_707_, v___f_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticState_onGoalM(lean_object* v_m_711_, lean_object* v_inst_712_, lean_object* v_inst_713_, lean_object* v_00_u03b1_714_, lean_object* v_ts_715_, lean_object* v_g_716_, lean_object* v_f_717_){
_start:
{
lean_object* v_toApplicative_718_; lean_object* v_toBind_719_; lean_object* v_toPure_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___f_724_; lean_object* v___f_725_; lean_object* v___x_726_; 
v_toApplicative_718_ = lean_ctor_get(v_inst_712_, 0);
v_toBind_719_ = lean_ctor_get(v_inst_712_, 1);
lean_inc_n(v_toBind_719_, 3);
v_toPure_720_ = lean_ctor_get(v_toApplicative_718_, 1);
lean_inc(v_toPure_720_);
v___x_721_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___closed__4));
v___x_722_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticState_focus___redArg___lam__1___closed__0));
lean_inc(v_g_716_);
lean_inc_ref(v_ts_715_);
lean_inc_ref(v_inst_712_);
v___x_723_ = lp_aesop_Aesop_Script_TacticState_focus___redArg(v_inst_712_, v_inst_713_, v_ts_715_, v_g_716_);
v___f_724_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__5), 8, 7);
lean_closure_set(v___f_724_, 0, v_ts_715_);
lean_closure_set(v___f_724_, 1, v_g_716_);
lean_closure_set(v___f_724_, 2, v___x_721_);
lean_closure_set(v___f_724_, 3, v___x_722_);
lean_closure_set(v___f_724_, 4, v_toPure_720_);
lean_closure_set(v___f_724_, 5, v_inst_712_);
lean_closure_set(v___f_724_, 6, v_toBind_719_);
v___f_725_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticState_onGoalM___redArg___lam__6), 4, 3);
lean_closure_set(v___f_725_, 0, v_f_717_);
lean_closure_set(v___f_725_, 1, v_toBind_719_);
lean_closure_set(v___f_725_, 2, v___f_724_);
v___x_726_ = lean_apply_4(v_toBind_719_, lean_box(0), lean_box(0), v___x_723_, v___f_725_);
return v___x_726_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_GoalWithMVars(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_CollectMVars(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_TacticState(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_GoalWithMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_CollectMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_Script_instInhabitedTacticState_default = _init_lp_aesop_Aesop_Script_instInhabitedTacticState_default();
lean_mark_persistent(lp_aesop_Aesop_Script_instInhabitedTacticState_default);
lp_aesop_Aesop_Script_instInhabitedTacticState = _init_lp_aesop_Aesop_Script_instInhabitedTacticState();
lean_mark_persistent(lp_aesop_Aesop_Script_instInhabitedTacticState);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_TacticState(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_GoalWithMVars(uint8_t builtin);
lean_object* initialize_Lean_Meta_CollectMVars(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_TacticState(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_GoalWithMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_CollectMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_TacticState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_TacticState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_TacticState(builtin);
}
#ifdef __cplusplus
}
#endif
